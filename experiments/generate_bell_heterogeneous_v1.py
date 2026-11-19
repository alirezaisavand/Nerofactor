"""One bounded Bell reference generation using archived primary intersections."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_heterogeneous_v1 as b
from benchmarks import donut_heterogeneous_v2 as refinement
from benchmarks.donut_recovery_v1 import json_new
from benchmarks.generate_donut_recovery_v1 import highlight_record
from experiments.generate_donut_heterogeneous_v1 import compare, incident_grid, ordinary_probes, previews
from experiments.generate_donut_heterogeneous_v2 import archive, baseline, capture, extra_gates, require_passing
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def sheet(out):
    from PIL import Image

    canvas = Image.new('RGB', (3*b.SIZE, len(b.PANEL)*b.SIZE))
    for row, camera in enumerate(b.PANEL):
        for col, tag in enumerate(('RGB', 'diffuse', 'specular')):
            with Image.open(out/f'view-{camera}-{tag}.png') as image:
                canvas.paste(image, (col*b.SIZE, row*b.SIZE))
    return canvas


def add_highlights(view):
    valid = view['shader_valid']
    summary, mask = highlight_record(view['linear_components'][valid])
    view['highlight_mask'] = np.zeros(len(valid), dtype=bool)
    view['highlight_mask'][valid] = mask
    return summary


def execute(f, sha, commit, auth_sha, paths, data):
    import torch

    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.verified_ray_backend import load

    require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count() == 1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed CUDA runtime required')
    geometry, views = data
    output = Path('/outputs/reference')
    output.mkdir()
    started = last_progress = time.monotonic()
    report = dict(schema_version=1, protocol=b.PROTOCOL, phase='initializing', status='incomplete',
                  commit=commit, freeze_sha256=sha, authorization_sha256=auth_sha, image=b.IMAGE,
                  source_generation_sha256=b.SOURCE_GENERATION, source_admission_sha256=b.SOURCE_ADMISSION,
                  input_records=f['input_records'], panel=b.PANEL, splits=b.SPLITS, field=b.FIELD,
                  environment=b.ENVIRONMENT, BRDF=b.BRDF, indirect='zero', point_light='absent',
                  tolerances=b.TOLERANCES, refinement=b.REFINEMENT, limits=b.LIMITS,
                  counters=dict.fromkeys(b.WORK, 0), refined_points=0, views=[], cleanup_verified=False,
                  scientific_work_complete=False, training_authorized=False, native_model_calls=0, optimizer_returns=0)

    def budget():
        require(time.monotonic()-started < b.LIMITS['inner_seconds'], 'Bell reference time limit')
        require(sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file()) < b.LIMITS['output_bytes'],
                'Bell reference output limit')

    def charge(key, cost):
        nonlocal last_progress
        require(report['counters'][key]+cost <= b.LIMITS[key], 'Bell reference work cap: '+key)
        report['counters'][key] += cost
        now = time.monotonic()
        if now-last_progress >= 5:
            budget()
            p = output/'progress.next'
            p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            p.replace(output/'progress.json')
            last_progress = now

    def work():
        torch.set_num_threads(b.LIMITS['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(b.LIMITS['cuda_allocator_fraction'])
        report['backend'] = load('predicate_fp64', b.IMAGE)
        activate('material', PINS['material'])
        import raytracing

        report['runtime'] = dict(torch=torch.__version__, CUDA=torch.version.cuda,
                                 GPU=torch.cuda.get_device_name(0), numpy=np.__version__)
        tracer = raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),
                                    geometry['triangles'].astype(np.int32))
        report['coverage'] = b.coverage(geometry, views)
        archive(output/'surface-area.npz', dict(triangle_area_native=geometry['area']))

        def render(view, grid):
            with torch.no_grad():
                return incident_grid(view, grid, tracer, torch, charge)

        for camera in b.PANEL:
            report['phase'] = f'camera_{camera}'
            print(f'Heterogeneous Bell camera {camera} started', flush=True)
            old, counts = baseline(views[camera], camera, charge, render, output)
            archive(output/f'baseline-{camera}.npz', dict(old, baseline_coarse_visible_counts=counts))
            require_passing(extra_gates(old), f'Camera {camera} baseline reflectance controls failed')

            def fetch(row):
                require(report['refined_points'] < b.MAX_TOTAL, 'Bell panel refinement budget exhausted')
                pixel = int(old['pixel_indices'][row])
                patch = capture(old, np.asarray([row]), render)
                archive(output/f'patch-{camera}-{pixel}.npz', patch)
                report['refined_points'] += 1
                return patch

            view, grids = refinement.assemble(old, counts, fetch)
            highlight = add_highlights(view)
            archive(output/f'view-{camera}.npz', view)
            archive(output/f'quadrature-{camera}.npz', grids)
            gates = compare(view)
            report['views'].append(dict(camera=camera, valid=int(view['shader_valid'].sum()),
                                        selected_pixels=grids['refined_pixels'].tolist(),
                                        ordinary_probe_pixels=view['pixel_indices'][ordinary_probes(view)].tolist(),
                                        highlight_diagnostics=highlight, **gates))
            require_passing(gates, f'Camera {camera} final convergence failed')
            report['views'][-1]['previews'] = previews(output, camera, view)
            print(f'Camera {camera} verified, {len(grids["refined_pixels"])} refined', flush=True)
            budget()

    def observed_work():
        try:
            work()
        except Exception as error:
            return error
        return None

    try:
        report['geometry_checks'] = {str(c): b.view_geometry_summary(v) for c, v in views.items()}
        error = run_with_cleanup(observed_work, torch)
        report['cleanup_verified'] = True
        if error is not None:
            raise error
        require([v['camera'] for v in report['views']] == b.PANEL
                and report['counters'] == b.counters(b.POINTS, report['refined_points']),
                'Incomplete Bell reference work')
        b.freeze(b.ROOT/b.FREEZE, sha)
        b.source(f, *paths)
        sheet(output).save(output/'reference-components.png')
        budget()
        report.update(status='complete_pending_researcher_admission', phase='complete', scientific_work_complete=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['outputs'] = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                             for p in sorted(output.iterdir()) if p.is_file()
                             and p.name not in ('progress.json', 'progress.next')}
        report['wall_seconds'] = time.monotonic()-started
        report['host_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        json_new(output/'generation.json', report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    f = b.freeze(b.ROOT/b.FREEZE, args.freeze_sha256)
    if args.execute:
        paths = (Path('/inputs/oracle'), Path('/inputs/admission.json'))
        auth = Path('/authorization.json')
        require(args.commit and digest(auth) == args.authorization_sha256
                and b.read(auth) == b.authorization(f, args.commit, args.freeze_sha256),
                'Exact separate Bell reference authorization required')
        require(os.getuid() != 0 and all(os.statvfs(p).f_flag & os.ST_RDONLY for p in
                (*paths, auth, b.ROOT, Path('/'))) and not os.statvfs('/outputs').f_flag & os.ST_RDONLY
                and {row.split(':')[0].strip() for row in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
                'Reviewed non-root, read-only, networkless runtime required')
    else:
        paths = tuple(Path(f['host_inputs'][k]) for k in ('oracle', 'admission'))
    geometry, views = b.source(f, *paths)
    if args.execute:
        execute(f, args.freeze_sha256, args.commit, args.authorization_sha256, paths, (geometry, views))
    else:
        checks = {c: b.view_geometry_summary(v) for c, v in views.items()}
        print(json.dumps(dict(status='metadata_only', counters=dict.fromkeys(b.WORK, 0), panel=b.PANEL,
                              geometry_checks=checks, coverage=b.coverage(geometry, views)), sort_keys=True))


if __name__ == '__main__':
    main()
