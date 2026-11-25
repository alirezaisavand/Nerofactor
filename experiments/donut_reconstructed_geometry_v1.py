"""Run the authorized Donut primary-ray and material-correspondence audit."""
import argparse
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_reconstructed_geometry_v1 as g
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_heterogeneous_v1 import material_field, geometry_fields, read
from benchmarks.donut_recovery_v1 import camera_rays, json_new, unit
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest
from infrastructure.verified_ray_backend import load


def save(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def trace(tracer, origin, direction, torch):
    with torch.no_grad():
        data = tracer.trace(torch.from_numpy(origin).cuda(), torch.from_numpy(direction).cuda())
    p, n, depth = (v.cpu().numpy() for v in data)
    require(p.shape == n.shape == origin.shape and depth.shape == (len(origin),), 'Unexpected native ray layout')
    hit = np.isfinite(depth) & (depth < 10)
    return dict(position=p, raw=n, depth=depth, hit=hit)


def finite_hits(result):
    ids = result['hit']
    return (np.isfinite(result['depth']).all() and (result['depth'] >= 0).all()
            and np.isfinite(result['position'][ids]).all())


def cpu_witness(surface, result, origin, direction, i):
    hit = surface.independent_hit(origin, direction)
    row = dict(pixel=int(i), native_hit=bool(result['hit'][i]), reference_hit=hit is not None)
    if hit is None or not row['native_hit']:
        return dict(row, passed=row['native_hit'] == row['reference_hit'])
    face, depth, low, tied = hit
    raw = result['raw'][i].astype(float)
    pos = result['position'][i].astype(float)
    row.update(face=face, tied_faces=tied.tolist(), reference_depth=depth, barycentric_minimum=low,
               shared_edge=low <= g.TOL['edge'])
    if not np.isfinite(raw).all() or not np.isfinite(pos).all():
        return dict(row, passed=False, nonfinite=True)
    match = np.max(abs(surface.normal[tied]-raw), axis=1)
    row.update(normal_witness_face=int(tied[match.argmin()]), normal_error=float(match.min()),
               normal_length_error=float(abs(np.linalg.norm(raw)-1)),
               depth_error=float(abs(depth-result['depth'][i])),
               position_error=float(np.max(abs(pos-(origin+depth*direction)))))
    return dict(row, passed=all(row[k+'_error'] <= g.TOL[k] for k in ('depth', 'position', 'normal'))
                and row['normal_length_error'] <= g.TOL['unit_normal'])


def reconstructed_normals(raw, hit):
    normals = np.zeros_like(raw, dtype=float)
    valid = hit & np.isfinite(raw).all(axis=1) & (np.linalg.norm(raw, axis=1) > 1e-12)
    normals[valid] = unit(-raw[valid].astype(float))
    return normals, valid


def near_degenerate(positions, degenerate):
    if not len(degenerate):
        return np.zeros(len(positions), bool)
    edge = np.roll(degenerate, -1, axis=1)-degenerate
    offset = positions[:, None, None]-degenerate
    length2 = np.square(edge).sum(-1)
    frac = np.divide(np.sum(offset*edge, axis=-1), length2,
                     out=np.zeros(offset.shape[:-1]), where=length2>0)
    residual = offset-np.clip(frac, 0, 1)[..., None]*edge
    return np.min(np.linalg.norm(residual, axis=-1), axis=(1, 2)) <= g.TOL['radius']


def primary(camera, raw, surfaces, geometry, view, meta, tracers, charge, torch):
    origin64, direction64, _ = camera_rays(meta, camera)
    origin, direction = origin64.astype('f4'), direction64.astype('f4')
    result = {}
    for name, tracer in tracers.items():
        charge('primary_rays', len(origin))
        result[name] = trace(tracer, origin, direction, torch)
    oracle, recon = result['oracle'], result['reconstruction']
    ids = view['pixel_indices']
    expected = view['hit_mask'].ravel()
    mask_equal = np.array_equal(oracle['hit'], expected)
    both = oracle['hit'] & expected
    common_ids = view['pixel_indices'][both[ids]]
    error_p = float(abs(oracle['position'][common_ids]-view['position'][both[ids]]).max()) if len(common_ids) else None
    error_d = float(abs(oracle['depth'][common_ids]-view['depth'][both[ids]]).max()) if len(common_ids) else None
    smooth = g.smooth_normals(geometry, view['triangle'], view['barycentric'])
    normal_error = float(abs(smooth-view['normals']).max())
    oracle_n = np.zeros_like(origin64)
    oracle_n[ids] = smooth
    rec_n, rec_valid = reconstructed_normals(recon['raw'], recon['hit'])
    degenerate = surfaces['reconstruction']
    near = np.zeros(len(origin), bool)
    finite = recon['hit'] & np.isfinite(recon['position']).all(axis=1)
    near[finite] = near_degenerate(recon['position'][finite], degenerate.tri[~degenerate.valid])
    raw.update(origin=origin, direction=direction, oracle_smooth_normal=oracle_n,
               reconstruction_normal=rec_n, reconstruction_normal_valid=rec_valid, near_degenerate=near,
               archived_hit=expected, archived_pixel_indices=ids, archived_triangle=view['triangle'],
               archived_barycentric=view['barycentric'])
    raw.update({name+'_'+key: val for name, row in result.items() for key, val in row.items()})
    witnesses = {}
    for name, surface in surfaces.items():
        rows = []
        for i in g.sampled_pixels(expected):
            charge('cpu_ray_checks', 1)
            rows.append(cpu_witness(surface, result[name], origin[i].astype(float), direction[i].astype(float), i))
        witnesses[name] = rows
    okay = (mask_equal and error_p is not None and error_p <= g.TOL['position'] and error_d <= g.TOL['depth']
            and normal_error <= g.TOL['unit_normal'] and all(finite_hits(r) for r in result.values())
            and all(w['passed'] for rows in witnesses.values() for w in rows))
    return dict(passed=bool(okay), oracle_mask_equal=mask_equal, oracle_position_error=error_p, oracle_depth_error=error_d,
                oracle_smooth_normal_error=normal_error, witnesses=witnesses)


def correspondence(arrays, raw, surfaces, geometry, view, radius, tracer, charge, torch):
    rec_hit, rec_n = raw['reconstruction_hit'], raw['reconstruction_normal']
    rec_ids = np.flatnonzero(rec_hit)
    pos = raw['reconstruction_position'].astype(float)
    n = len(rec_ids)
    reasons = {key: np.zeros(n, bool) for key in g.REASONS}
    arrays.update(pixel=rec_ids, forward_face=np.full(n, -1, int), reverse_face=np.full(n, -1, int),
                  forward_bary=np.zeros((n, 3)), reverse_bary=np.zeros((n, 3)), oracle_point=np.zeros((n, 3)),
                  reverse_point=np.zeros((n, 3)), projection_distance=np.zeros(n), roundtrip_distance=np.zeros(n),
                  visibility_tested=np.zeros(n, bool), visibility_reused=np.zeros(n, bool),
                  search_completed=np.zeros(n, bool), valid=np.zeros(n, bool), **reasons)
    oracle_row = np.full(len(rec_hit), -1, int)
    oracle_row[view['pixel_indices']] = np.arange(len(view['pixel_indices']))
    for j, i in enumerate(rec_ids):
        forward = surfaces['oracle'].nearest(pos[i], charge)
        reasons['invalid_reconstructed_normal'][j] = not raw['reconstruction_normal_valid'][i]
        if forward is None:
            reasons['no_oracle_candidate'][j] = True
        else:
            face, point, bary, distance, ties = forward
            arrays['forward_face'][j], arrays['oracle_point'][j] = face, point
            arrays['forward_bary'][j], arrays['projection_distance'][j] = bary, distance
            reverse = surfaces['reconstruction'].nearest(point, charge)
            if reverse is not None:
                arrays['reverse_face'][j], arrays['reverse_point'][j], arrays['reverse_bary'][j] = reverse[:3]
                arrays['roundtrip_distance'][j] = np.linalg.norm(pos[i]-reverse[1])
            reasons['reverse_unresolved'][j] = reverse is None or arrays['roundtrip_distance'][j] > g.TOL['radius']
            side = g.material_side(point[None])[0]
            reasons['boundary_side_ambiguity'][j] = ((g.material_side(ties) != side).any() or
                (oracle_row[i] >= 0 and view['material_region'][oracle_row[i]] != side))
        arrays['search_completed'][j] = True
    qids = np.flatnonzero(arrays['forward_face'] >= 0)
    on = g.smooth_normals(geometry, arrays['forward_face'][qids], arrays['forward_bary'][qids])
    arrays['oracle_smooth_normal'] = np.zeros((n, 3))
    arrays['oracle_smooth_normal'][qids] = on
    cos = (on*rec_n[rec_ids[qids]]).sum(1)
    reasons['normal_mismatch'][qids] = (~reasons['invalid_reconstructed_normal'][qids]
                                       & (cos < np.cos(np.deg2rad(g.TOL['normal_degrees']))))
    o = raw['origin'][rec_ids[qids]]
    v = arrays['oracle_point'][qids]-o
    depth = np.linalg.norm(v, axis=1)
    d = unit(v)
    native_d = d.astype('f4')
    coincident = np.max(abs(native_d-raw['direction'][rec_ids[qids]]), axis=1) <= g.TOL['coincident_direction']
    arrays['visibility_reused'][qids[coincident]] = True
    depth_hit = raw['oracle_depth'][rec_ids[qids]].copy()
    use = ~coincident
    if use.any():
        charge('visibility_rays', int(use.sum()))
        witness = trace(tracer, o[use], native_d[use], torch)
        arrays['visibility_tested'][qids[use]] = True
        depth_hit[use] = witness['depth']
    reasons['oracle_hidden'][qids] = ~(np.isfinite(depth_hit) & (depth_hit >= 0) & (depth_hit < 10)
                                       & (abs(depth_hit-depth) <= g.TOL['depth']))
    arrays['visibility_depth_error'] = np.zeros(n)
    arrays['visibility_depth_finite'] = np.zeros(n, bool)
    finite = np.isfinite(depth_hit)
    arrays['visibility_depth_finite'][qids] = finite
    arrays['visibility_depth_error'][qids[finite]] = depth_hit[finite]-depth[finite]
    reason = np.zeros(n, np.int8)
    for index, key in reversed(list(enumerate(g.REASONS, 1))):
        reason[reasons[key]] = index
    keep = np.flatnonzero(reason == 0)
    arrays['reason_code'] = reason
    arrays['valid'][keep] = True
    arrays['label_rows'] = keep
    if not len(keep):
        return
    label_arrays(arrays, keep, raw, view, oracle_row, rec_n, radius)


def label_arrays(arrays, keep, raw, view, oracle_row, rec_n, radius):
    ix = arrays['pixel'][keep]
    n1 = arrays['oracle_smooth_normal'][keep]
    labels = material_field(arrays['oracle_point'][keep], n1, radius)
    rotation = g.minimal_rotation(n1, rec_n[ix])
    Q = g.rotate_tensor(labels['Q_world'], rotation)
    axis = np.einsum('nij,nj->ni', rotation, labels['axis'])
    width = np.sqrt(np.linalg.eigvalsh(Q)[:, [2, 1]])
    require(np.max(abs(width-labels['widths_major_minor'])) <= g.TOL['transport_width'],
            'Normal transport changed the physical widths')
    arrays.update({f'labels_{key}': val for key, val in labels.items()})
    arrays.update(transported_world_Q=Q, transported_axis=axis, transported_widths=width)
    rows = oracle_row[ix]
    common = np.flatnonzero(rows >= 0)
    rows = rows[common]
    arrays['common_label_rows'] = common
    n0 = raw['oracle_smooth_normal'][ix[common]]
    comparable = (n1[common]*n0).sum(1) > -1+1e-6
    arrays['decomposition_eligible'] = comparable
    common, rows, n0 = common[comparable], rows[comparable], n0[comparable]
    sample_q = g.rotate_tensor(labels['Q_world'][common], g.minimal_rotation(n1[common], n0))
    source_q = view['Q_world'][rows]
    sampling = sample_q-source_q
    frame = Q[common]-sample_q
    orient_ok = (labels['anisotropy_gap'][common] >= g.TOL['gap']) & (view['anisotropy_gap'][rows] >= g.TOL['gap'])
    arrays.update(sampling_delta_Q=sampling, frame_delta_Q=frame, total_delta_Q=Q[common]-source_q,
                  source_Q_frobenius=np.linalg.norm(source_q, axis=(1, 2)),
                  sampling_signed_widths=labels['widths_major_minor'][common]-view['widths_major_minor'][rows],
                  orientation_eligible=orient_ok, orientation_shift_degrees=np.degrees(np.arccos(np.clip(
                      abs((axis[common]*view['axis'][rows]).sum(1)), 0, 1))))


def execute(args):
    from benchmarks.generate_bell_e11 import run_with_cleanup

    f = g.freeze(args.freeze_sha256)
    p = g.paths(f, native=True)
    auth = Path('/authorization.json')
    require(digest(auth) == args.authorization_sha256 and read(auth) == g.authorization(f, args.commit, args.freeze_sha256),
            'Exact separate geometry authorization required')
    inputs = (Path('/'), g.ROOT, auth, *p.values())
    require(os.geteuid() != 0 and all(os.statvfs(path).f_flag & os.ST_RDONLY for path in inputs)
            and not os.statvfs('/outputs').f_flag & os.ST_RDONLY
            and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
            'Read-only networkless non-root geometry runtime required')
    out = Path('/outputs/audit')
    out.mkdir()
    started = time.monotonic()
    counts = dict.fromkeys(g.LIMITS, 0)
    report = dict(protocol=g.PROTOCOL, status='incomplete', commit=args.commit, freeze_sha256=args.freeze_sha256,
                  authorization_sha256=args.authorization_sha256, inputs={key: f[key] for key in ('mesh', 'oracle', 'cameras', 'dataset', 'views')},
                  counters=counts, limits=g.LIMITS, cleanup_verified=False, gates={}, view_products=[],
                  historical_geometry_provenance='unresolved', renderer_calls=0, optimizer_updates=0, checkpoint_loads=0)

    def charge(key, amount):
        require(time.monotonic()-started < g.RESOURCES['inner_seconds'], 'Geometry runtime exhausted')
        require(counts[key]+amount <= g.LIMITS[key], 'Geometry work exhausted: '+key)
        counts[key] += amount

    def archive(name, arrays):
        written = sum(path.stat().st_size for path in out.iterdir())
        estimate = sum(value.nbytes for value in arrays.values())+65536
        require(written+estimate <= g.RESOURCES['output_bytes'], 'Geometry output cap')
        save(out/name, arrays)

    try:
        geometry, meta, views = g.admit(f, p)
        rv, rf = mesh_arrays(p['mesh'])
        inputs_mesh = dict(oracle=(geometry['vertices_native'].astype('f4'), geometry['triangles'].astype('i4')),
                           reconstruction=(rv, rf))
        surfaces = {name: g.Surface(*items) for name, items in inputs_mesh.items()}
        require(int((~surfaces['reconstruction'].valid).sum()) == 4 and surfaces['oracle'].valid.all(), 'Mesh face inventory differs')
        report['degenerate_face_ids'] = np.flatnonzero(~surfaces['reconstruction'].valid).tolist()
        radius, _ = geometry_fields(geometry)
        import torch
        require(torch.__version__ == '1.8.1+cu111' and torch.cuda.device_count() == 1
                and 'A100' in torch.cuda.get_device_name(0), 'Reviewed A100 runtime required')
        report['backend'] = load('predicate_fp64', f['image'])
        report['runtime'] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, CUDA=torch.version.cuda,
                                 numpy=np.__version__, material_source=PINS['material'])
        activate('material', PINS['material'])
        from raytracing import RayTracer

        def work():
            torch.set_num_threads(g.RESOURCES['cpu_threads'])
            torch.cuda.set_per_process_memory_fraction(g.RESOURCES['allocator_fraction'])
            tracers = {name: RayTracer(*items) for name, items in inputs_mesh.items()}
            primary_arrays = {}
            for camera, view in views.items():
                arrays = {}
                try:
                    gate = primary(camera, arrays, surfaces, geometry, view, meta, tracers, charge, torch)
                finally:
                    if arrays:
                        archive(f'primary-{camera}.npz', arrays)
                json_new(out/f'witnesses-{camera}.json', gate)
                report['gates'][str(camera)] = gate['passed']
                primary_arrays[camera] = arrays
                print(f'Camera {camera} primary witnesses passed={gate["passed"]}', flush=True)
            require(all(report['gates'].values()), 'Mandatory primary replay or independent witness failure')
            for camera, view in views.items():
                arrays = {}
                try:
                    correspondence(arrays, primary_arrays[camera], surfaces, geometry, view, radius, tracers['oracle'], charge, torch)
                finally:
                    archive(f'correspondence-{camera}.npz', arrays)
                report['view_products'].append(camera)
                print(f'Camera {camera} correspondences validated', flush=True)

        run_with_cleanup(work, torch)
        report['cleanup_verified'] = True
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        require(counts['primary_rays'] == g.LIMITS['primary_rays'] and counts['cpu_ray_checks'] == g.LIMITS['cpu_ray_checks'],
                'Incomplete primary inventory')
        g.freeze(args.freeze_sha256)
        g.admit(f, p)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['outputs'] = {path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in sorted(out.iterdir())}
        json_new(out/'receipt.json', report)


def interrupted(signum, frame):
    raise InterruptedError(f'Geometry execution interrupted by signal {signum}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('freeze-sha256', 'commit', 'authorization-sha256'):
        cli.add_argument('--'+key, required=True)
    cli.add_argument('--execute', action='store_true', required=True)
    signal.signal(signal.SIGTERM, interrupted)
    execute(cli.parse_args())
