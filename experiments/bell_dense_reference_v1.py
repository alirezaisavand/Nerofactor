"""Generate only the 12 appended Bell views, then replay their evidence offline."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from benchmarks import bell_dense_reference_v1 as d
from benchmarks import bell_heterogeneous_v1 as b
from benchmarks import donut_heterogeneous_v2 as refinement
from benchmarks.bell_e11 import linear_to_srgb
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.generate_bell_heterogeneous_v1 import add_highlights
from experiments.generate_donut_heterogeneous_v1 import compare, incident_grid, ordinary_probes, previews
from experiments.generate_donut_heterogeneous_v2 import archive, baseline, capture, extra_gates, require_passing
from experiments.verify_bell_heterogeneous_replay_v2 import highlight_replay
from experiments.verify_bell_heterogeneous_v1 import compare_geometry_metadata
from experiments.verify_donut_heterogeneous_v2 import replay_gates, validate_physical_view, view_identity
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def view_outputs(camera, pixels):
    return {*(f'{stem}-{camera}.npz' for stem in ('view','baseline','quadrature')),
            *(f'view-{camera}-{tag}.png' for tag in ('RGB','diffuse','specular')),
            *(f'patch-{camera}-{pixel}.npz' for pixel in pixels)}


def copy_sparse(source, out, records):
    for name in d.SPARSE_FILES:
        with b.check_artifact(source/name, records[name]).open('rb') as src, (out/name).open('xb') as dest:
            shutil.copyfileobj(src, dest)
        b.check_artifact(out/name, records[name])


def generate_new(out, views, render, charge, report, budget):
    """Use the existing full-view baseline and deterministic Donut v2 refinement."""
    for camera in d.NEW:
        report['phase'] = f'camera_{camera}'
        print(f'Dense Bell camera {camera} started', flush=True)
        old, counts = baseline(views[camera], camera, charge, render, out)
        archive(out/f'baseline-{camera}.npz', dict(old, baseline_coarse_visible_counts=counts))
        require_passing(extra_gates(old), f'Camera {camera} baseline reflectance controls failed')

        def fetch(row):
            require(report['refined_points'] < b.MAX_TOTAL, 'Dense Bell refinement budget exhausted')
            pixel = int(old['pixel_indices'][row])
            patch = capture(old, np.asarray([row]), render)
            archive(out/f'patch-{camera}-{pixel}.npz', patch)
            report['refined_points'] += 1
            return patch

        view, grids = refinement.assemble(old, counts, fetch)
        highlight = add_highlights(view)
        archive(out/f'view-{camera}.npz', view)
        archive(out/f'quadrature-{camera}.npz', grids)
        gates = compare(view)
        row = dict(camera=camera, valid=int(view['shader_valid'].sum()), selected_pixels=grids['refined_pixels'].tolist(),
                   ordinary_probe_pixels=view['pixel_indices'][ordinary_probes(view)].tolist(),
                   highlight_diagnostics=highlight, **gates)
        report['views'].append(row)
        require_passing(gates, f'Camera {camera} final convergence failed')
        row['previews'] = previews(out, camera, view)
        print(f'Dense Bell camera {camera} verified, {len(row["selected_pixels"])} refined', flush=True)
        budget()


def execute(f, sha, commit, auth_sha, paths, data):
    import torch

    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.verified_ray_backend import load

    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed CUDA runtime required')
    geometry, views, _, angles = data
    out = Path('/outputs/reference')
    out.mkdir()
    started = last_progress = time.monotonic()
    report = dict(schema_version=1, protocol=d.PROTOCOL, phase='initializing', status='incomplete', commit=commit,
        freeze_sha256=sha, authorization_sha256=auth_sha, image=b.IMAGE, splits=d.SPLITS, panel=d.PANEL,
        new_cameras=d.NEW, field=b.FIELD, environment=b.ENVIRONMENT, BRDF=b.BRDF, indirect='zero',
        point_light='absent', tolerances=b.TOLERANCES, limits=d.LIMITS, refinement=b.REFINEMENT,
        camera_selection=d.SELECTION, nearest_picked_angles_degrees=angles,
        input_records=f['input_records'], reused_sparse_records=f['sparse']['outputs'],
        counters=dict.fromkeys(b.WORK,0), refined_points=0, views=[], cleanup_verified=False,
        scientific_work_complete=False, training_authorized=False, native_model_calls=0, optimizer_returns=0)

    def budget():
        require(time.monotonic()-started < d.LIMITS['inner_seconds'], 'Dense Bell reference time limit')
        require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) < d.LIMITS['output_bytes'],
                'Dense Bell reference output limit')

    def charge(key, cost):
        nonlocal last_progress
        require(report['counters'][key]+cost <= d.LIMITS[key], 'Dense Bell work cap: '+key)
        report['counters'][key] += cost
        now = time.monotonic()
        if now-last_progress >= 5:
            budget()
            p = out/'progress.next'
            p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            p.replace(out/'progress.json')
            last_progress = now

    def work():
        torch.set_num_threads(d.LIMITS['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(d.LIMITS['cuda_allocator_fraction'])
        report['backend'] = load('predicate_fp64', b.IMAGE)
        activate('material', PINS['material'])
        import raytracing

        report['runtime'] = dict(torch=torch.__version__, CUDA=torch.version.cuda,
                                 GPU=torch.cuda.get_device_name(0), numpy=np.__version__)
        tracer = raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),
                                    geometry['triangles'].astype(np.int32))

        def render(view, grid):
            with torch.no_grad():
                return incident_grid(view, grid, tracer, torch, charge)

        generate_new(out, views, render, charge, report, budget)

    def observed_work():
        try:
            work()
        except Exception as error:
            return error
        return None

    try:
        report['geometry_checks'] = {str(c): b.view_geometry_summary(v) for c,v in views.items()}
        copy_sparse(paths[2], out, f['sparse']['outputs'])
        archive(out/'surface-area.npz', dict(triangle_area_native=geometry['area']))
        error = run_with_cleanup(observed_work, torch)
        report['cleanup_verified'] = True
        if error is not None:
            raise error
        require([v['camera'] for v in report['views']]==d.NEW
                and report['counters']==b.counters(sum(d.NEW_COUNTS), report['refined_points']),
                'Incomplete dense Bell reference work')
        d.freeze(b.ROOT/d.FREEZE, sha)
        d.source(f, *paths)
        budget()
        report.update(status='complete_pending_researcher_admission', phase='complete', scientific_work_complete=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['outputs'] = {p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(out.iterdir())
                             if p.is_file() and p.name not in ('progress.json','progress.next')}
        report['wall_seconds'] = time.monotonic()-started
        report['host_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        json_new(out/'generation.json', report)


def replay_view(out, row, labels, check_budget):
    camera = row['camera']
    old = numeric(out/f'baseline-{camera}.npz')
    coarse_counts = old.pop('baseline_coarse_visible_counts')
    view_identity(old, labels)
    require_passing(extra_gates(old), 'Dense Bell baseline reflectance gate differs')
    selected = []

    def fetch(i):
        check_budget()
        pixel = int(old['pixel_indices'][i])
        require(row['selected_pixels'][len(selected):len(selected)+1]==[pixel],
                'Dense Bell refinement order differs')
        selected.append(pixel)
        return numeric(out/f'patch-{camera}-{pixel}.npz')

    view, grids = refinement.assemble(old, coarse_counts, fetch)
    highlight = add_highlights(view)
    require(selected==row['selected_pixels'], 'Dense Bell refinement inventory differs')
    roundoff = highlight_replay(row['highlight_diagnostics'], highlight)
    for stem, expected in (('view',view),('quadrature',grids)):
        saved = numeric(out/f'{stem}-{camera}.npz')
        require(saved.keys()==expected.keys(), 'Dense Bell archive arrays differ')
        for key, array in expected.items():
            require(saved[key].dtype==array.dtype, 'Dense Bell archive dtype differs')
            np.testing.assert_array_equal(saved[key], array)
    validate_physical_view(view, grids)
    gates = compare(view)
    replay_gates(gates, {k:row[k] for k in gates})
    require_passing(gates, 'Dense Bell final gate differs')
    require(row['valid']==int(view['shader_valid'].sum())==d.COUNTS[camera]
            and row['ordinary_probe_pixels']==view['pixel_indices'][ordinary_probes(view)].tolist(),
            'Dense Bell valid hits or reflectance controls differ')
    for slot, tag in enumerate(('diffuse','specular','RGB')):
        rgb = view['linear_components'][:,slot]
        canvas = np.zeros((b.SIZE*b.SIZE,3))
        canvas[view['pixel_indices']] = np.clip(rgb,0,1)
        with Image.open(out/f'view-{camera}-{tag}.png') as png:
            np.testing.assert_array_equal(np.asarray(png), linear_to_srgb(canvas.reshape(b.SIZE,b.SIZE,3)))
        require(row['previews'][slot]==dict(component=tag, channels_above_one=int((rgb>1).sum())),
                'Dense Bell PNG clipping differs')
    return view, dict(camera=camera, refined_pixels=selected, gates=gates,
                      highlight_summary_roundoff=roundoff, highlights=int(view['highlight_mask'].sum()))


def verify(f, sha, destination):
    batch = Path(f['host_output'])
    require(destination.resolve()==destination and not destination.exists() and destination.parent.is_dir()
            and all(p not in destination.parents for p in (batch,*d.host_paths(f))),
            'Fresh separate dense Bell verification directory required')
    auth = batch/'authorization.json'
    approval = b.read(auth)
    require(approval==d.authorization(f,approval['commit'],sha)
            and digest(auth)==digest(batch.with_name(batch.name+'-authorization.json'))
            and digest(batch/'freeze.json')==sha and (batch/'exit-code.txt').read_text().strip()=='0',
            'Dense Bell execution or authorization evidence differs')
    out = batch/'reference'
    r = b.read(out/'generation.json')
    expected = dict(status='complete_pending_researcher_admission', phase='complete',
        scientific_work_complete=True, cleanup_verified=True, training_authorized=False,
        protocol=d.PROTOCOL, commit=approval['commit'], freeze_sha256=sha, authorization_sha256=digest(auth),
        splits=d.SPLITS, panel=d.PANEL, new_cameras=d.NEW, image=b.IMAGE, backend=b.identity(),
        input_records=f['input_records'], reused_sparse_records=f['sparse']['outputs'],
        field=b.FIELD, environment=b.ENVIRONMENT, BRDF=b.BRDF, indirect='zero', point_light='absent',
        camera_selection=d.SELECTION, limits=d.LIMITS, refinement=b.REFINEMENT, tolerances=b.TOLERANCES,
        native_model_calls=0, optimizer_returns=0)
    require(all(r[k]==v for k,v in expected.items()) and [row['camera'] for row in r['views']]==d.NEW
            and 0<r['wall_seconds']<d.LIMITS['inner_seconds']
            and 0<r['host_peak_RSS_bytes']<d.LIMITS['host_memory_gib']*1024**3
            and sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<d.LIMITS['output_bytes'],
            'Incomplete or foreign dense Bell terminal receipt')
    names = {'surface-area.npz',*d.SPARSE_FILES}
    for row in r['views']:
        names |= view_outputs(row['camera'], row['selected_pixels'])
    require(set(r['outputs'])==names=={p.name for p in out.iterdir()
            if p.is_file() and p.name not in ('progress.json','generation.json')}, 'Dense Bell output inventory differs')
    for name, item in r['outputs'].items():
        b.check_artifact(out/name, item)
    require({name:r['outputs'][name] for name in d.SPARSE_FILES}==f['sparse']['outputs'],
            'Sparse Bell reference bytes changed')

    geometry, views, centers, angles = d.source(f, *d.host_paths(f))
    compare_geometry_metadata(r['geometry_checks'], {str(c):b.view_geometry_summary(v) for c,v in views.items()})
    np.testing.assert_allclose(r['nearest_picked_angles_degrees'], angles, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(numeric(out/'surface-area.npz')['triangle_area_native'], geometry['area'])
    summaries = []
    total = 0

    def check_budget():
        nonlocal total
        require(total<b.MAX_TOTAL, 'Dense Bell replay refinement budget exhausted')
        total += 1

    for row in r['views']:
        camera = row['camera']
        views[camera], summary = replay_view(out, row, views[camera], check_budget)
        summaries.append(summary)
    require(total==r['refined_points'] and r['counters']==b.counters(sum(d.NEW_COUNTS),total),
            'Dense Bell reference work differs')
    coverage, arrays = d.coverage(geometry, views, centers, highlights=True)
    destination.mkdir()
    support = destination/'held-out-support.npz'
    archive(support, arrays)
    json_new(destination/'verification.json', dict(schema_version=1, protocol=d.PROTOCOL,
        status='verified_pending_researcher_admission', researcher_admitted_dataset=False, training_authorized=False,
        package=str(out), generation_sha256=digest(out/'generation.json'), freeze_sha256=sha,
        authorization_sha256=digest(auth), native_commit=r['commit'], outputs=r['outputs'],
        counters=r['counters'], splits=d.SPLITS, new_views=summaries, coverage=coverage,
        support_archive=dict(bytes=support.stat().st_size,sha256=digest(support)), training_projection=d.training_projection()))
    return len(summaries)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--verify', type=Path)
    args = cli.parse_args()
    f = d.freeze(b.ROOT/d.FREEZE,args.freeze_sha256)
    if args.verify is not None:
        print(f'Verified {verify(f,args.freeze_sha256,args.verify)} new dense Bell references')
        return
    paths = d.host_paths(f)
    if args.execute:
        paths = tuple(Path('/inputs')/name for name in
                      ('oracle','admission.json','sparse','sparse-admission.json','sparse-verification.json'))
        auth = Path('/authorization.json')
        require(args.commit and digest(auth)==args.authorization_sha256
                and b.read(auth)==d.authorization(f,args.commit,args.freeze_sha256),
                'Exact separate dense Bell reference authorization required')
        require(os.getuid()!=0 and all(os.statvfs(p).f_flag & os.ST_RDONLY for p in (*paths,auth,b.ROOT,Path('/')))
                and not os.statvfs('/outputs').f_flag & os.ST_RDONLY
                and {row.split(':')[0].strip() for row in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Reviewed non-root, read-only, networkless runtime required')
    data = d.source(f,*paths)
    if args.execute:
        execute(f,args.freeze_sha256,args.commit,args.authorization_sha256,paths,data)
    else:
        g,views,centers,angles = data
        coverage,_ = d.coverage(g,views,centers,highlights=False)
        print(json.dumps(dict(status='metadata_only', counters=dict.fromkeys(b.WORK,0), splits=d.SPLITS,
            nearest_picked_angles_degrees=angles, coverage=coverage, training_projection=d.training_projection()),
            sort_keys=True))


if __name__=='__main__':
    main()
