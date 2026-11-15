"""One approved E11G panel: geometry-selected views, cached incident visibility.

Imports are metadata-only. No new primary/incident-shadow rays or neural models.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments.bell_anisotropy_e11f import (ROOT, GRIDS, LIGHT, inputs as cached_inputs,
                                             strengthen, source_paths as previous_sources)
from experiments.bell_highlight_pilot_e11d import counterfactual, kernel_axes
from experiments.bell_quadrature_e11e import weighted_components, record
from experiments.analyze_bell_highlight_e11d import response_fields, replay_equal
from experiments.analyze_bell_anisotropy_e11f import verify_grid as verify_previous_grid
from experiments.bell_highlight_e11d import write_json
from benchmarks.bell_e11 import convergence
from infrastructure.signatures import require, read_report
from infrastructure.stage_sources import baseline, PINS, digest
from infrastructure.verified_ray_backend import identity

PROTOCOL = 'bell-angular-diversity-e11g-v1'
F_REPORT_SHA = '9f8c8a3e52d0aaea434c11f623a05c6da948d08d266ee38fef4f3e940bcbadae'
CALIBRATION_SHA = '4d8f4e13d7fd5fffec8d258bfc7c5599ef6d81525547d678ba088e76b1656c9c'
CASES = ('truth', 'theta_plus', 'theta_minus', 'major_plus', 'major_minus',
         'minor_plus', 'minor_minus', 'isotropic', 'mean_plus', 'mean_minus', 'ratio_plus', 'ratio_minus')
SELECTION = dict(additional_views=3, minimum_geometric_probes=42, score_decimals=10)
GEOMETRY = dict(origin_offset=1e-5, endpoint_uncertainty=2e-4,
                base_direction_error=2e-6, rotation_error=2e-6)
SCREENS = dict(median_smin_gain=2., median_condition_ratio=.5, width_coverage=.25)
# Tightened to the geometry-only 146-correspondence plan, before any responses.
LIMITS = dict(wall_seconds=900, grace_seconds=30, cpu_threads=4, host_memory_gib=64,
              cuda_allocator_fraction=.25, output_bytes=256*1024**2,
              point_batch=4, direction_chunk=16384,
              new_primary_rays=0, new_incident_shadow_queries=0,
              max_outgoing_visibility_queries=146,
              environment_point_directions=27525120, max_brdf_point_directions=1320550400)


def source_paths():
    return sorted(set(previous_sources()) | {
        'docs/protocols/BELL_ANISOTROPY_E11F_V1.freeze.json',
        'docs/BELL_ANISOTROPY_E11F_RESULTS_V1.md', 'docs/evidence/BELL_ANISOTROPY_E11F_RESULTS_V1.json',
        'experiments/analyze_bell_anisotropy_e11f.py', 'experiments/bell_multiview_e11g.py',
        'experiments/run_bell_multiview_e11g.sh', 'run_bell_multiview_e11g.py',
        'tests/benchmarks/test_bell_multiview_e11g.py', 'docs/protocols/BELL_MULTIVIEW_E11G_V1.md'})


def outgoing_geometry(positions, normals, poses, K):
    require(positions.shape == normals.shape == (84, 3) and poses.shape == (128, 3, 4) and
            K.shape == (128, 3, 3) and all(np.isfinite(x).all() for x in (positions, normals, poses, K)),
            'Finite admitted camera/probe shapes required')
    R = poses[:, :, :3].astype(np.float64); t = poses[:, :, 3].astype(np.float64)
    require(abs(R@R.transpose(0, 2, 1)-np.eye(3)).max() <= GEOMETRY['rotation_error'] and
            abs(np.linalg.det(R)-1).max() <= GEOMETRY['rotation_error'], 'Invalid world-to-camera rotation')
    centers = -np.einsum('nji,nj->ni', R, t)
    delta = centers[:, None]-positions[None]; distance = np.linalg.norm(delta, axis=-1)
    require((distance > .01).all(), 'Camera intersects probe')
    wo = delta/distance[..., None]; facing = (wo*normals[None]).sum(-1)
    camera = np.einsum('vij,pj->vpi', R, positions)+t[:, None]
    h = np.einsum('vij,vpj->vpi', K.astype(float), camera)
    uv = np.divide(h[..., :2], h[..., 2, None], out=np.full(h[..., :2].shape, np.nan), where=h[..., 2, None] != 0)
    in_frame = (camera[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < 128) & (uv[..., 1] >= 0) & (uv[..., 1] < 128)
    return dict(centers=centers, directions=wo, distance=distance, facing=facing,
                uv=uv, in_frame=in_frame, geometric=(facing > 0) & in_frame)


def select_views(g):
    """Greedy angular max-min of median angles over all84 fixed points; no radiance."""
    selected = [58]; counts = g['geometric'].sum(1); steps = []
    for unused in range(SELECTION['additional_views']):
        scores = []
        for camera in range(128):
            if camera in selected or counts[camera] < SELECTION['minimum_geometric_probes']:
                continue
            angles = [np.median(np.arccos(np.clip((g['directions'][camera]*g['directions'][old]).sum(-1), -1, 1)))
                      for old in selected]
            score = round(float(min(angles)), SELECTION['score_decimals'])
            scores.append((score, -camera))
        require(bool(scores), 'Insufficient eligible geometric cameras; no substitution')
        score, negative_id = max(scores); camera = -negative_id; selected.append(camera)
        steps.append(dict(camera_id=camera, min_median_angle_degrees=float(np.degrees(score)),
                          geometric_probes=int(counts[camera])))
    return dict(camera_ids=selected, selection=SELECTION, steps=steps,
                candidate_geometric_counts=counts.tolist(),
                outgoing_visibility_queries=int(sum(counts[c] for c in selected[1:])))


def classify_visibility(depth, distance):
    require(depth.shape == distance.shape and np.isfinite(depth).all() and np.isfinite(distance).all() and
            (depth >= 0).all() and (distance > .01).all(), 'Invalid outgoing segment depth')
    margin = depth-(distance-GEOMETRY['origin_offset'])
    return dict(visible=margin > GEOMETRY['endpoint_uncertainty'],
                occluded=margin < -GEOMETRY['endpoint_uncertainty'],
                unresolved=abs(margin) <= GEOMETRY['endpoint_uncertainty'])


def transformed_material(torch, labels, case):
    if case not in ('mean_plus', 'mean_minus', 'ratio_plus', 'ratio_minus'):
        return counterfactual(torch, labels, case)
    result = counterfactual(torch, labels, 'truth'); w = result['widths'].clone()
    sign = 1 if case.endswith('plus') else -1
    # u=log(sqrt(mx*my)), v=log(mx/my); fixed +/-0.1 in either coordinate.
    if case.startswith('mean_'):
        w *= np.exp(sign*.1)
    else:
        w[:, 0] *= np.exp(sign*.05); w[:, 1] *= np.exp(-sign*.05)
    require(bool(((w >= .04) & (w <= 1)).all()), 'Log-coordinate perturbation outside fixed physical domain')
    result['widths'] = w
    return result


def spectrum(J):
    singular = np.linalg.svd(J, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    return dict(singular_values=singular.tolist(), condition_number=condition,
                numerical_full_rank=bool(singular[-1] > max(1e-6*singular[0], 1e-8)))


def conditioning_summary(rows, coverage):
    result = []
    for arm in range(2):
        matched = [row for row in rows if row['material'] == arm and row['valid_view_count'] >= 2]
        n = len(matched)
        factors = {name: sum(row['factors'][name]['multiview_any_resolved'] for row in coverage
                            if row['material'] == arm and row['valid_view_count'] >= 2)
                   for name in ('orientation', 'major', 'minor', 'logmean', 'logratio')}
        for basis in ('legacy_scaled', 'theta_logmean_logratio'):
            values = [row[basis] for row in matched]
            finite = [v for v in values if v['single']['condition_number'] is not None and
                      v['stacked_RMS']['condition_number'] is not None and v['single']['singular_values'][-1] > 0]
            ratio = float(np.median([v['stacked_RMS']['condition_number']/v['single']['condition_number'] for v in finite])) if finite else None
            gain = float(np.median([v['stacked_RMS']['singular_values'][-1]/v['single']['singular_values'][-1] for v in finite])) if finite else None
            screen = bool(n and len(finite) == n and ratio <= SCREENS['median_condition_ratio'] and
                          gain >= SCREENS['median_smin_gain'] and
                          all(factors[k]/n >= SCREENS['width_coverage'] for k in ('major', 'minor')))
            result.append(dict(material=arm, basis=basis, matched_probes=n, finite_ratio_probes=len(finite),
                median_paired_condition_ratio=ratio, median_paired_smin_gain=gain,
                resolved_counts_on_matched_probes=factors, exploratory_screen_passed=screen,
                grid_proxy_positive_lower_count=sum(v['grid_proxy_smin_lower'] > 0 for v in values),
                full_dataset_admitted=False))
    return result


def compare_views(fine, coarse, valid):
    """RMS-normalized stacks and matched probes; invalid slots never measured zeros."""
    require(fine.shape == coarse.shape == (2, 4, 84, 12, 3, 3) and valid.shape == (4, 84) and
            valid.dtype == np.bool_ and valid[0].all(), 'Complete matched multiview layout required')
    rows = []; convergence_rows = []; coverage = []
    for arm in range(2):
        for view in range(4):
            mask = valid[view]
            if not mask.any():
                convergence_rows.append(dict(material=arm, view_index=view, valid_points=0, status='unresolved'))
                continue
            a, b = fine[arm, view, mask], coarse[arm, view, mask]
            require(np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b >= 0).all(),
                    'Invalid valid-view radiance')
            convergence_rows.append(dict(material=arm, view_index=view, valid_points=int(mask.sum()),
                cases={case: convergence(a[:, k, 2], b[:, k, 2]) for k, case in enumerate(CASES)}))
        for p in range(84):
            views = np.flatnonzero(valid[:, p]); rgb = fine[arm, views, p, :, 2]
            low = coarse[arm, views, p, :, 2]
            legacy = np.stack([(rgb[:, a]-rgb[:, b])/2 for a, b in [(1, 2), (3, 4), (5, 6)]], -1)
            legacy_low = np.stack([(low[:, a]-low[:, b])/2 for a, b in [(1, 2), (3, 4), (5, 6)]], -1)
            # Direct log-coordinate probes, not a relabelled post-hoc width secant.
            physical = np.stack([(rgb[:, 1]-rgb[:, 2])/(np.pi/6),
                                 (rgb[:, 8]-rgb[:, 9])/.2, (rgb[:, 10]-rgb[:, 11])/.2], -1)
            physical_low = np.stack([(low[:, 1]-low[:, 2])/(np.pi/6),
                                    (low[:, 8]-low[:, 9])/.2, (low[:, 10]-low[:, 11])/.2], -1)
            row = dict(material=arm, probe_index=p, valid_view_count=len(views), view_indices=views.tolist())
            for name, J, L in [('legacy_scaled', legacy, legacy_low), ('theta_logmean_logratio', physical, physical_low)]:
                error = float(np.linalg.norm((J-L).reshape(-1, 3)/np.sqrt(len(views)), ord=2))
                row[name] = dict(single=spectrum(J[0]), stacked_RMS=spectrum(J.reshape(-1, 3)/np.sqrt(len(views))),
                                 column_norms_RMS=np.sqrt((J*J).sum((0, 1))/len(views)).tolist(),
                                 stacked_grid_difference_operator_norm=error,
                                 grid_proxy_smin_lower=max(0., spectrum(J.reshape(-1, 3)/np.sqrt(len(views)))['singular_values'][-1]-error))
            rows.append(row)
        # Preserve individual-view frozen gates. Report unions, not chosen best conditioning.
        masks = {name: np.zeros((4, 84), dtype=np.bool_) for name in ('orientation', 'major', 'minor', 'isotropic', 'logmean', 'logratio')}
        code_masks = {name: value.copy() for name, value in masks.items()}
        amplitudes = {name: np.full((4, 84), np.nan) for name in masks}
        for v in range(4):
            ids = np.flatnonzero(valid[v]); a = fine[arm, v, ids]; b = coarse[arm, v, ids]
            if not len(ids):
                continue
            fields = response_fields(a[:, :8], b[:, :8])
            # Reuse exact same response definitions for the two declared new contrasts.
            for name, pair in [('logmean', (8, 9)), ('logratio', (10, 11))]:
                order = [0, pair[0], pair[1], 3, 4, 5, 6, 7]
                fields[name] = response_fields(a[:, order], b[:, order])['orientation']
            for name, values in fields.items():
                masks[name][v, ids] = values['resolved']; code_masks[name][v, ids] = values['code_gate']
                amplitudes[name][v, ids] = values['amplitude']
        for n in range(1, 5):
            matched = valid.sum(0) == n
            coverage.append(dict(material=arm, valid_view_count=n, matched_probes=int(matched.sum()),
                factors={name: dict(single_resolved=int(masks[name][0, matched].sum()),
                                    multiview_any_resolved=int(masks[name][:, matched].any(0).sum()),
                                    single_two_codes=int(code_masks[name][0, matched].sum()),
                                    multiview_any_two_codes=int(code_masks[name][:, matched].any(0).sum())) for name in masks},
                joint_orientation_major_minor=int((masks['orientation'].any(0) & masks['major'].any(0) & masks['minor'].any(0) & matched).sum())))
    converged = all(all(c['passed'] for c in row['cases'].values()) for row in convergence_rows if 'cases' in row)
    summaries = conditioning_summary(rows, coverage)
    for item in summaries:
        item['references_usable'] = bool(converged and valid[1:].any())
        item['exploratory_screen_passed'] &= item['references_usable']
    return dict(per_probe_conditioning=rows, convergence=convergence_rows, matched_coverage=coverage,
                all_measured_cases_converged=converged,
                matched_conditioning_summary=summaries,
                valid_view_counts=valid.sum(0).tolist(),
                invalid_payload_policy='NaN slots are unmeasured; valid mask controls every computation',
                interpretation='RMS stacking removes duplicate-view scaling; numerical rank alone is not robust recovery. No favorable view subset is chosen.')


def inputs(benchmark, pilot, reference, factor):
    labels, wo, pixels, cache, records = cached_inputs(benchmark, pilot, reference)
    require(digest(factor/'anisotropy.json') == F_REPORT_SHA, 'Accepted E11F report changed')
    f = json.loads((factor/'anisotropy.json').read_text())
    require(f['status'] == 'complete' and f['cleanup_verified'] is True, 'Incomplete E11F baseline')
    records.append(record(factor/'anisotropy.json'))
    for r in f['outputs']:
        p = factor/r['path']; require(p.resolve() == p and factor in p.parents and
            digest(p) == r['sha256'] and p.stat().st_size == r['bytes'], 'E11F artifact changed')
        records.append(record(p))
    for s in GRIDS:
        with np.load(factor/('grid-'+str(s)+'.npz'), allow_pickle=False) as z:
            cache[s]['stronger'] = verify_previous_grid({k: z[k].copy() for k in z.files}, cache[s]['original'], pixels)
    path = benchmark/'geometry/calibrations-safe.npz'
    require(digest(path) == CALIBRATION_SHA, 'Safe numerical camera identity changed')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {'pose', 'K', 'source_camera_ids'} and np.array_equal(z['source_camera_ids'], np.arange(128)),
                'Camera ID ordering changed')
        g = outgoing_geometry(labels['position'], labels['normals'], z['pose'], z['K'])
    require(g['geometric'][58].all() and abs(g['directions'][58]-wo).max() <= GEOMETRY['base_direction_error'],
            'Existing camera58 correspondence mismatch')
    # Camera58 keeps its actual saved ray-based direction, not a silently recomputed one.
    g['directions'][58] = wo
    records.append(record(path))
    return labels, wo, pixels, cache, g, records


def authorize(args):
    f, unused = read_report(args.freeze, args.freeze_sha256)
    require(f['protocol'] == PROTOCOL and f['limits'] == LIMITS and f['cases'] == list(CASES) and
            f['geometry'] == GEOMETRY and f['selection'] == SELECTION and f['screens'] == SCREENS and
            f['lighting'] == LIGHT and f['grids'] == list(GRIDS) and f['F_report_sha256'] == F_REPORT_SHA and
            f['calibration_sha256'] == CALIBRATION_SHA and
            set(f['files']) == set(source_paths()), 'Frozen definition mismatch')
    for p, sha in f['files'].items():
        require(digest(ROOT/p) == sha, 'Frozen source changed: '+p)
    a, unused = read_report(args.authorization, args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol') == PROTOCOL and a.get('commit') == args.commit and
            a.get('freeze_sha256') == args.freeze_sha256 and a.get('image_id') == args.image_id == identity()['image_id'] and
            a.get('limits') == LIMITS and a.get('output_directory') == str(args.output), 'Separate exact E11G approval required')
    require(os.geteuid() != 0 and args.output.is_absolute() and args.output.resolve() == args.output and
            args.output.parent == Path('/outputs') and not args.output.exists(), 'Fresh non-root output required')
    for p in (Path('/'), ROOT, args.benchmark, args.pilot, args.reference, args.factor, args.freeze, args.authorization):
        require(os.statvfs(p).f_flag & os.ST_RDONLY, 'Read-only original/source/root required')
    require(not os.statvfs('/outputs').f_flag & os.ST_RDONLY, 'Writable output required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'}, 'No-network runtime required')
    return f


def execute(args):
    f = authorize(args); data = inputs(args.benchmark, args.pilot, args.reference, args.factor)
    plan = select_views(data[4]); replay_equal(plan, f['view_plan'])
    args.output.mkdir(); start = time.monotonic()
    r = dict(schema_version=1, protocol=PROTOCOL, status='incomplete', cleanup_verified=False,
        commit=args.commit, freeze_sha256=args.freeze_sha256, authorization_sha256=args.authorization_sha256,
        source_files=f['files'], inputs=data[-1], limits=LIMITS, view_plan=plan, geometry=GEOMETRY,
        cases=list(CASES), lighting=LIGHT, counters=dict(outgoing_visibility_attempts=0, outgoing_visibility_returns=0,
        brdf_attempts=0, brdf_returns=0, environment_attempts=0, environment_returns=0,
        new_primary_rays=0, new_incident_shadow_queries=0, optimizer_returns=0, model_calls=0), completed_grids=[])
    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        r.update(run_with_cleanup(lambda: render(args, data, plan, r, start, torch), torch)); r['cleanup_verified'] = True
        require(all(digest(ROOT/p) == sha for p, sha in f['files'].items()) and
                all(digest(Path(x['path'])) == x['sha256'] for x in data[-1]), 'Terminal integrity failed')
        r['status'] = 'complete'
    except BaseException as exc:
        r.update(status='failed', failure=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc()); raise
    finally:
        partial = r.pop('_partial', None)
        if partial is not None and r['status'] != 'complete':
            from benchmarks.generate_bell_e11 import write_arrays
            write_arrays(args.output/'incomplete-grid.npz', **partial)
        r['elapsed_seconds'] = time.monotonic()-start; write_json(args.output/'multiview.json', r)


def render(args, data, plan, r, start, torch):
    from benchmarks.generate_bell_e11 import write_arrays
    from benchmarks.bell_reference_e11 import basis, environment
    from benchmarks.bell_geometry_e11 import mesh_arrays
    from infrastructure.verified_ray_backend import load
    require(torch.__version__ == '1.8.1+cu111' and torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'Reviewed runtime required')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.25)
    r['backend'] = load('predicate_fp64', args.image_id)
    r['runtime'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0), cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    science, unused = baseline('material', PINS['material'])
    baseline('geometry', PINS['geometry'])
    spec = importlib.util.spec_from_file_location('_e11g_bvh', science/'raytracing/raytracer.py')
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    vertices, faces = mesh_arrays(args.benchmark/'geometry/mesh.ply')
    tracer = wrapper.RayTracer(vertices, faces.astype(np.int32))
    labels, unused, pixels, cache, geometry, records = data; ids = plan['camera_ids']
    wo = geometry['directions'][ids]; valid = np.zeros((4, 84), dtype=np.bool_); valid[0] = True
    geom = geometry['geometric'][ids]; queried = geom.copy(); queried[0] = False
    distances = geometry['distance'][ids]; depths = np.full((4, 84), np.nan)
    unresolved = np.zeros_like(valid); occluded = np.zeros_like(valid)
    count = int(queried.sum()); require(count == plan['outgoing_visibility_queries'] and
        count <= LIMITS['max_outgoing_visibility_queries'], 'Visibility work limit exceeded')
    with torch.no_grad():
        o = np.broadcast_to(labels['position'], (4, 84, 3))[queried].astype(np.float32)
        d = wo[queried].astype(np.float32); origin = o+GEOMETRY['origin_offset']*d
        r['counters']['outgoing_visibility_attempts'] = count
        unused, unused, depth = tracer.trace(torch.from_numpy(origin).cuda(), torch.from_numpy(d).cuda())
        values = depth.cpu().numpy(); r['counters']['outgoing_visibility_returns'] = count
        depths[queried] = values; visibility = classify_visibility(values, distances[queried])
        for name, target in [('visible', valid), ('unresolved', unresolved), ('occluded', occluded)]:
            target[queried] = visibility[name]
        write_arrays(args.output/'outgoing-geometry.npz', camera_ids=np.array(ids), pixel_indices=pixels,
            camera_centers=geometry['centers'][ids], outgoing_directions=wo, camera_distances=distances,
            facing=geometry['facing'][ids], uv=geometry['uv'][ids], in_frame=geometry['in_frame'][ids],
            geometric=geom, queried=queried, valid=valid, occluded=occluded, unresolved=unresolved, nearest_depth=depths)
        r['view_coverage'] = [dict(camera_id=c, positive_facing=int((geometry['facing'][c] > 0).sum()),
             in_frame=int(geometry['in_frame'][c].sum()), queried=int(queried[v].sum()), valid=int(valid[v].sum()),
             occluded=int(occluded[v].sum()), unresolved=int(unresolved[v].sum())) for v, c in enumerate(ids)]
        # Do not replace a failed/occluded view. An empty shared panel is scientifically unresolved.
        r['any_additional_visible'] = bool(valid[1:].any())
        original = {k: torch.from_numpy(v).cuda() for k, v in labels.items()}
        stronger = strengthen(torch, original)
        from experiments.bell_anisotropy_e11f import admit_isotropic
        r['isotropic_equivalence'] = admit_isotropic(torch, original, stronger)
        material = [original, stronger]; results = {}
        def budget(key, n, cap):
            require(r['counters'][key]+n <= cap and time.monotonic()-start < 840 and
                sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) < LIMITS['output_bytes'], 'Work/resource ceiling exceeded')
            r['counters'][key] += n
        for size in GRIDS:
            c = cache[size]; N = size*size
            array = np.full((2, 4, 84, 12, 3, 3), np.nan, dtype=np.float64)
            array[0, 0, :, :8] = c['original']; array[1, 0, :, :8] = c['stronger']
            completed = np.zeros(84, dtype=np.bool_)
            r['_partial'] = dict(components=array, valid=valid, completed_probes=completed, grid_size=np.array(size))
            for j in range(0, 84, 4):
                end = j+4; n = original['normals'][j:end]; t, b = basis(n)
                acc = torch.zeros((2, 4, 4, 12, 2, 3), dtype=torch.float64, device='cuda')
                for offset in range(0, N, 16384):
                    stop = min(offset+16384, N); loc = torch.from_numpy(c['nodes'][offset:stop]).cuda()
                    wi = loc[None, :, 0, None]*t[:, None]+loc[None, :, 1, None]*b[:, None]+loc[None, :, 2, None]*n[:, None]
                    cosine = (wi*n[:, None]).sum(-1)
                    require(float(abs(torch.linalg.norm(wi, dim=-1)-1).max()) <= 2e-12 and
                        float(abs(cosine-loc[None, :, 2]).max()) <= 2e-12 and bool((cosine > 0).all()), 'Cached incident direction changed')
                    visible = torch.from_numpy(np.unpackbits(c['visibility'][j:end, offset//8:stop//8], axis=1, bitorder='big').astype(np.bool_)).cuda()
                    budget('environment_attempts', 4*(stop-offset), LIMITS['environment_point_directions'])
                    light = environment(wi, LIGHT); r['counters']['environment_returns'] += 4*(stop-offset)
                    for v in range(4):
                        local = np.flatnonzero(valid[v, j:end])
                        if not len(local):
                            continue
                        ix = torch.from_numpy(local).cuda(); view = torch.from_numpy(wo[v, j:end][local]).cuda()
                        for arm in range(2):
                            current = {k: value[j:end][ix] for k, value in material[arm].items()}
                            for k, case in enumerate(CASES):
                                if (v == 0 and k < 8) or (arm == 1 and case == 'isotropic'):
                                    continue
                                budget('brdf_attempts', len(local)*(stop-offset), LIMITS['max_brdf_point_directions'])
                                fd, fs = kernel_axes(torch, wi[ix], view, transformed_material(torch, current, case))
                                acc[arm, v, ix, k] += weighted_components(torch, fd, fs, light[ix], cosine[ix], visible[ix], c['weight'])
                                r['counters']['brdf_returns'] += len(local)*(stop-offset)
                pairs = torch.cat([acc, acc.sum(-2, keepdim=True)], -2).cpu().numpy()
                for arm in range(2):
                    for v in range(4):
                        for p in range(j, end):
                            if valid[v, p]:
                                begin = 8 if v == 0 else 0
                                array[arm, v, p, begin:] = pairs[arm, v, p-j, begin:]
                # Exactly shared isotropic reference for both material controls, for every valid view.
                array[1, :, j:end, 7] = array[0, :, j:end, 7]
                completed[j:end] = True
                tmp = args.output/'progress.next'; tmp.write_text(json.dumps(dict(grid=size, completed_probes=int(completed.sum()), counters=r['counters']))+'\n'); tmp.replace(args.output/'progress.json')
            for arm in range(2):
                expected = c['original'] if arm == 0 else c['stronger']
                require(np.array_equal(array[arm, 0, :, :8], expected), 'Single-view accepted baseline changed')
                for v in range(4):
                    a = array[arm, v, valid[v]]
                    require(np.isfinite(a).all() and (a >= 0).all() and
                            np.array_equal(a[:, :, 0]+a[:, :, 1], a[:, :, 2]), 'Invalid valid-view components')
                    require(np.array_equal(a[:, :, 0], np.repeat(a[:, :1, 0], 12, axis=1)), 'Fixed diffuse changed across cases')
            require(np.array_equal(array[0, :, :, :, 0][valid], array[1, :, :, :, 0][valid]),
                    'Fixed diffuse changed between materials')
            require(np.isnan(array[:, ~valid]).all(), 'Invalid correspondences contain fabricated measurements')
            write_arrays(args.output/('grid-'+str(size)+'.npz'), components=array, valid=valid,
                         camera_ids=np.array(ids), pixel_indices=pixels, completed_probes=completed)
            results[size] = array; r['completed_grids'].append(size)
            print(json.dumps(dict(grid=size, counters=r['counters'])), flush=True)
        expected_brdf = (8*84+23*int(valid[1:].sum()))*sum(s*s for s in GRIDS)
        require(r['counters']['brdf_attempts'] == r['counters']['brdf_returns'] == expected_brdf and
            r['counters']['environment_attempts'] == r['counters']['environment_returns'] == 84*sum(s*s for s in GRIDS), 'Actual work accounting mismatch')
        r['comparison'] = compare_views(results[GRIDS[-1]], results[GRIDS[0]], valid)
        write_json(args.output/'comparison.json', r['comparison'])
        r['outputs'] = [record(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ('progress.json', 'progress.next')]
        for item in r['outputs']:
            item['path'] = str(Path(item['path']).relative_to(args.output))
    import resource
    return dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'pilot', 'reference', 'factor', 'output', 'freeze', 'authorization'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('commit', 'image-id', 'freeze-sha256', 'authorization-sha256'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--execute', action='store_true', required=True)
    execute(p.parse_args())


if __name__ == '__main__':
    main()
