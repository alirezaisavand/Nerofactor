"""Authorized geometry-sensitivity rendering with the admitted independent integrator."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_oracle_geometry_rgb_v1 as rgb
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_recovery_v1 import json_new
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments.donut_reconstructed_geometry_v1 import save, interrupted
from experiments.generate_donut_heterogeneous_v1 import incident_grid
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest
from infrastructure.verified_ray_backend import load


def capture_name(camera, arm, prefix, grid):
    return f'camera-{camera}-{arm}-{prefix}-{grid}.npz'


def execute(args):
    f = rgb.freeze(args.freeze_sha256)
    p = rgb.paths(f, native=True)
    auth = Path('/authorization.json')
    require(digest(auth) == args.authorization_sha256
            and rgb.geo.h.read(auth) == rgb.authorization(f, args.commit, args.freeze_sha256), 'Exact separate RGB approval required')
    inputs = (Path('/'), rgb.ROOT, auth, *p.values())
    require(os.geteuid() != 0 and all(os.statvfs(v).f_flag & os.ST_RDONLY for v in inputs)
            and not os.statvfs('/outputs').f_flag & os.ST_RDONLY
            and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
            'Read-only networkless non-root RGB runtime required')
    out = Path('/outputs/render')
    out.mkdir()
    started = checked = time.monotonic()
    counters = dict.fromkeys(rgb.LIMITS, 0)
    report = dict(protocol=rgb.PROTOCOL, status='incomplete', commit=args.commit, freeze_sha256=args.freeze_sha256,
                  authorization_sha256=args.authorization_sha256, gate2_receipt_sha256=f['gate2_receipt']['sha256'],
                  cleanup_verified=False, counters=counters, limits=rgb.LIMITS, resources=rgb.RESOURCES, views={},
                  primary_rays=0, native_model_calls=0, checkpoint_loads=0, optimizer_updates=0,
                  historical_geometry_provenance='unresolved')

    def check_storage(extra=0):
        require(sum(p.stat().st_size for p in out.iterdir())+extra <= rgb.RESOURCES['output_bytes'], 'RGB output cap')

    def charge(key, count):
        nonlocal checked
        now = time.monotonic()
        require(now-started < rgb.RESOURCES['inner_seconds'], 'RGB runtime exhausted')
        require(counters[key]+count <= rgb.LIMITS[key], 'RGB work cap: '+key)
        if now-checked >= 5:
            check_storage()
            tmp = out/'progress.next'
            tmp.write_text(json.dumps(report, allow_nan=False)+'\n')
            tmp.replace(out/'progress.json')
            checked = now
        counters[key] += count

    try:
        geometry, data = rgb.admit(f, p)
        import torch
        require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1'
                and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count() == 1
                and 'A100' in torch.cuda.get_device_name(0), 'Reviewed A100 CUDA runtime required')
        report['runtime'] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, CUDA=torch.version.cuda,
                                 numpy=np.__version__, material_source=PINS['material'])
        report['backend'] = load('predicate_fp64', f['backend']['image_id'])
        activate('material', PINS['material'])
        from raytracing import RayTracer

        def work():
            torch.set_num_threads(rgb.RESOURCES['cpu_threads'])
            torch.cuda.set_per_process_memory_fraction(rgb.RESOURCES['allocator_fraction'])
            tracers = dict(oracle=RayTracer(geometry['vertices_native'].astype('f4'), geometry['triangles'].astype('i4')),
                           reconstruction=RayTracer(*mesh_arrays(p['mesh'])))
            total = 0
            for camera, view in data.items():
                report['camera_in_progress'] = camera
                def fetch(arm, selected, grid, prefix):
                    report['grid_in_progress'] = dict(arm=arm, work=prefix, grid=grid)
                    with torch.no_grad():
                        components, _, visibility = incident_grid(selected, grid, tracers[arm], torch, charge)
                    capture = dict(components=components, visibility=visibility, pixels=selected['pixel_indices'], grid=np.array(grid))
                    check_storage(sum(value.nbytes for value in capture.values())+65536)
                    save(out/capture_name(camera, arm, prefix, grid), capture)
                    return capture
                print(f'Camera {camera} rendering fixed material on both meshes', flush=True)
                states, selected = rgb.paired(view['views'], fetch, total)
                total += len(selected)
                _, gates = rgb.raster(view, states)
                report['views'][str(camera)] = dict(refined_pixels=selected, **gates)
                print(f'Camera {camera} passed with {len(selected)} refined pixels', flush=True)
            report['refined_pixels_total'] = total

        run_with_cleanup(work, torch)
        report['cleanup_verified'] = True
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        require(counters['shadow_directions'] == counters['BRDF_point_directions'] >= rgb.BASE_COST, 'Incomplete paired direction totals')
        rgb.freeze(args.freeze_sha256)
        rgb.admit(f, p)
        check_storage()
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['outputs'] = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(out.glob('*.npz'))}
        json_new(out/'receipt.json', report)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('freeze-sha256', 'commit', 'authorization-sha256'):
        cli.add_argument('--'+key, required=True)
    cli.add_argument('--execute', action='store_true', required=True)
    signal.signal(signal.SIGTERM, interrupted)
    execute(cli.parse_args())
