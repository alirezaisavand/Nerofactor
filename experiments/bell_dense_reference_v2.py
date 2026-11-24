"""Bounded continuation and offline replay of the dense Bell reference."""
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
from benchmarks import bell_dense_reference_v2 as v2
from benchmarks.bell_e11 import linear_to_srgb
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.bell_dense_reference_v1 import add_highlights, copy_sparse, view_outputs
from experiments.generate_donut_heterogeneous_v1 import compare, incident_grid, ordinary_probes, previews
from experiments.generate_donut_heterogeneous_v2 import archive, baseline, capture, extra_gates, require_passing
from experiments.verify_bell_heterogeneous_replay_v2 import highlight_replay
from experiments.verify_bell_heterogeneous_v1 import compare_geometry_metadata
from experiments.verify_donut_heterogeneous_v2 import replay_gates, validate_physical_view, view_identity
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest

b = d.b


def identity_fields(f, sha, commit, auth_sha, reused, reused_work):
    return dict(schema_version=1, protocol=v2.PROTOCOL, commit=commit, freeze_sha256=sha, authorization_sha256=auth_sha,
        image=b.IMAGE, splits=d.SPLITS, panel=d.PANEL, new_cameras=d.NEW, field=b.FIELD, environment=b.ENVIRONMENT,
        BRDF=b.BRDF, indirect='zero', point_light='absent', tolerances=b.TOLERANCES, refinement=v2.POLICY,
        limits=v2.LIMITS, reused_camera=v2.CAMERA, reused_reference_records=reused, reused_counters=reused_work,
        failed_generation_sha256=f['failure_files']['reference/generation.json']['sha256'],
        native_model_calls=0, optimizer_returns=0, training_authorized=False)


def copy_prefix(source, out, records):
    for name, item in records.items():
        with b.check_artifact(source/name,item).open('rb') as src, (out/name).open('xb') as target:
            shutil.copyfileobj(src,target)
        b.check_artifact(out/name,item)


def complete_camera(out, data, camera, render, charge, report, prefix):
    name = f'baseline-{camera}.npz'
    if prefix:
        old = numeric(out/name)
        counts = old.pop('baseline_coarse_visible_counts')
        view_identity(old,data)
    else:
        old, counts = baseline(data,camera,charge,render,out)
        archive(out/name,dict(old,baseline_coarse_visible_counts=counts))
    require_passing(extra_gates(old),f'Camera {camera} baseline reflectance controls failed')

    def fetch(row, pixel, index):
        name = out/f'patch-{camera}-{pixel}.npz'
        if index < len(prefix):
            return numeric(name)
        require(report['new_refined_points']<b.MAX_TOTAL-v2.PREFIX, 'Dense panel refinement budget exhausted')
        patch = capture(old,np.asarray([row]),render)
        archive(name,patch)
        report['new_refined_points'] += 1
        return patch

    view, grids = v2.assemble(old,counts,fetch,prefix)
    highlight = add_highlights(view)
    archive(out/f'view-{camera}.npz',view)
    archive(out/f'quadrature-{camera}.npz',grids)
    gates = compare(view)
    row = dict(camera=camera,valid=int(view['shader_valid'].sum()),selected_pixels=grids['refined_pixels'].tolist(),
               ordinary_probe_pixels=view['pixel_indices'][ordinary_probes(view)].tolist(),
               highlight_diagnostics=highlight,**gates)
    report['views'].append(row)
    require_passing(gates,f'Camera {camera} final convergence failed')
    row['previews'] = previews(out,camera,view)


def execute(f, sha, commit, auth_sha, paths, failed, data, admitted):
    import torch

    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.verified_ray_backend import load

    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed CUDA runtime required')
    old = v2.prior_definition()
    geometry, views, _, angles = data
    prefix, records, reused = admitted
    out = Path('/outputs/reference')
    out.mkdir()
    started = last_progress = time.monotonic()
    report = dict(identity_fields(f,sha,commit,auth_sha,records,reused),status='incomplete',phase='initializing',
                  views=[],new_refined_points=0,counters=dict.fromkeys(b.WORK,0),cleanup_verified=False,
                  scientific_work_complete=False,nearest_picked_angles_degrees=angles)

    def budget():
        require(time.monotonic()-started<v2.LIMITS['inner_seconds'], 'Dense Bell v2 reference time limit')
        require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<v2.LIMITS['output_bytes'],
                'Dense Bell v2 reference output limit')

    def charge(key, cost):
        nonlocal last_progress
        require(report['counters'][key]+cost<=v2.LIMITS[key], 'Dense Bell v2 work cap: '+key)
        report['counters'][key] += cost
        now = time.monotonic()
        if now-last_progress>=5:
            budget()
            p = out/'progress.next'
            p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            p.replace(out/'progress.json')
            last_progress = now

    def work():
        torch.set_num_threads(v2.LIMITS['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(v2.LIMITS['cuda_allocator_fraction'])
        report['backend'] = load('predicate_fp64',b.IMAGE)
        activate('material',PINS['material'])
        import raytracing

        tracer = raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),geometry['triangles'].astype(np.int32))
        report['runtime'] = dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
                                 numpy=np.__version__)

        def render(view, grid):
            with torch.no_grad():
                return incident_grid(view,grid,tracer,torch,charge)

        for c in d.NEW:
            report['phase'] = f'camera_{c}'
            print(f'Dense Bell continuation camera {c} started',flush=True)
            complete_camera(out,views[c],c,render,charge,report,prefix if c==v2.CAMERA else ())
            budget()
            print(f'Dense Bell camera {c} verified',flush=True)

    def observed_work():
        try:
            work()
        except Exception as error:
            return error
        return None

    try:
        report['geometry_checks'] = {str(c):b.view_geometry_summary(v) for c,v in views.items()}
        copy_sparse(paths[2],out,old['sparse']['outputs'])
        copy_prefix(failed/'reference',out,records)
        archive(out/'surface-area.npz',dict(triangle_area_native=geometry['area']))
        error = run_with_cleanup(observed_work,torch)
        report['cleanup_verified'] = True
        if error is not None:
            raise error
        require([row['camera'] for row in report['views']]==d.NEW
                and report['counters']==b.counters(v2.NEW_BASELINE_POINTS,report['new_refined_points']),
                'Incomplete dense Bell continuation work')
        d.source(old,*paths)
        require(v2.admit_prefix(f,failed)==admitted,'Archived prefix changed during continuation')
        v2.freeze(b.ROOT/v2.FREEZE,sha)
        budget()
        report.update(status='complete_pending_researcher_admission',phase='complete',scientific_work_complete=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['outputs'] = {p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir())
                             if p.is_file() and p.name not in ('progress.json','progress.next')}
        report['wall_seconds'] = time.monotonic()-started
        report['host_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        json_new(out/'generation.json',report)


def replay_camera(out, row, labels, prefix, charge_patch):
    camera = row['camera']
    old = numeric(out/f'baseline-{camera}.npz')
    counts = old.pop('baseline_coarse_visible_counts')
    view_identity(old,labels)
    require_passing(extra_gates(old),'Dense Bell baseline reflectance gate differs')

    def fetch(_, pixel, index):
        require(row['selected_pixels'][index:index+1]==[pixel], 'Dense Bell refinement order differs')
        if index>=len(prefix):
            charge_patch()
        return numeric(out/f'patch-{camera}-{pixel}.npz')

    view,grids = v2.assemble(old,counts,fetch,prefix)
    highlight = add_highlights(view)
    roundoff = highlight_replay(row['highlight_diagnostics'],highlight)
    require(grids['refined_pixels'].tolist()==row['selected_pixels']
            and row['valid']==int(view['shader_valid'].sum())==d.COUNTS[camera]
            and row['ordinary_probe_pixels']==view['pixel_indices'][ordinary_probes(view)].tolist(),
            'Dense Bell final valid-hit or patch inventory differs')
    for stem, expected in (('view',view),('quadrature',grids)):
        saved = numeric(out/f'{stem}-{camera}.npz')
        require(saved.keys()==expected.keys(), 'Dense Bell saved array fields differ')
        for key,array in expected.items():
            require(saved[key].dtype==array.dtype, 'Dense Bell saved array dtype differs')
            np.testing.assert_array_equal(saved[key],array)
    validate_physical_view(view,grids)
    gates = compare(view)
    replay_gates(gates,{key:row[key] for key in gates})
    require_passing(gates,'Dense Bell final convergence differs')
    for slot,tag in enumerate(('diffuse','specular','RGB')):
        rgb = view['linear_components'][:,slot]
        canvas = np.zeros((b.SIZE*b.SIZE,3))
        canvas[view['pixel_indices']] = np.clip(rgb,0,1)
        with Image.open(out/f'view-{camera}-{tag}.png') as png:
            np.testing.assert_array_equal(np.asarray(png),linear_to_srgb(canvas.reshape(b.SIZE,b.SIZE,3)))
        require(row['previews'][slot]==dict(component=tag,channels_above_one=int((rgb>1).sum())),
                'Dense Bell preview clipping differs')
    return view,dict(camera=camera,refined_pixels=row['selected_pixels'],gates=gates,
                     highlight_summary_roundoff=roundoff,highlights=int(view['highlight_mask'].sum()))


def verify(f, sha, destination, paths, failed, data, admitted):
    batch = Path(f['host_output'])
    require(destination.resolve()==destination and not destination.exists() and destination.parent.is_dir()
            and all(root not in destination.parents for root in (batch,failed,*paths)),
            'Fresh separate dense Bell verification directory required')
    auth = batch/'authorization.json'
    approval = b.read(auth)
    require(approval==v2.authorization(f,approval['commit'],sha)
            and digest(auth)==digest(batch.with_name(batch.name+'-authorization.json'))
            and digest(batch/'freeze.json')==sha and (batch/'exit-code.txt').read_text().strip()=='0',
            'Dense Bell v2 execution or authorization evidence differs')
    prefix, records, reused = admitted
    geometry,views,centers,angles = data
    out = batch/'reference'
    r = b.read(out/'generation.json')
    expected = dict(identity_fields(f,sha,approval['commit'],digest(auth),records,reused),
                    status='complete_pending_researcher_admission',phase='complete',backend=b.identity(),
                    cleanup_verified=True,scientific_work_complete=True)
    require(all(r[key]==value for key,value in expected.items()) and [row['camera'] for row in r['views']]==d.NEW
            and 0<r['wall_seconds']<v2.LIMITS['inner_seconds']
            and 0<r['host_peak_RSS_bytes']<v2.LIMITS['host_memory_gib']*1024**3
            and sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<v2.LIMITS['output_bytes'],
            'Foreign or incomplete dense Bell v2 terminal receipt')
    compare_geometry_metadata(r['geometry_checks'],{str(c):b.view_geometry_summary(v) for c,v in views.items()})
    np.testing.assert_allclose(r['nearest_picked_angles_degrees'],angles,atol=2e-12,rtol=0)
    names = {'surface-area.npz',*d.SPARSE_FILES}
    for row in r['views']:
        names |= view_outputs(row['camera'],row['selected_pixels'])
    require(set(r['outputs'])==names=={p.name for p in out.iterdir() if p.is_file()
            and p.name not in ('progress.json','generation.json')}, 'Unexpected dense Bell v2 output inventory')
    retained = dict(v2.prior_definition()['sparse']['outputs'],**records)
    require(all(r['outputs'][name]==item for name,item in retained.items()), 'Reused dense Bell bytes differ')
    for name,item in r['outputs'].items():
        b.check_artifact(out/name,item)
    np.testing.assert_array_equal(numeric(out/'surface-area.npz')['triangle_area_native'],geometry['area'])
    total = 0

    def charge_patch():
        nonlocal total
        require(total < b.MAX_TOTAL-v2.PREFIX,'Dense Bell replay refinement budget exhausted')
        total += 1

    summaries = []
    for row in r['views']:
        camera = row['camera']
        views[camera],summary = replay_camera(out,row,views[camera],prefix if camera==v2.CAMERA else (),charge_patch)
        summaries.append(summary)
    require(total==r['new_refined_points'] and r['counters']==b.counters(v2.NEW_BASELINE_POINTS,total),
            'Dense Bell v2 work or patch counts differ')
    coverage,arrays = d.coverage(geometry,views,centers,highlights=True)
    destination.mkdir()
    support = destination/'held-out-support.npz'
    archive(support,arrays)
    json_new(destination/'verification.json',dict(schema_version=1,protocol=v2.PROTOCOL,
        status='verified_pending_researcher_admission',researcher_admitted_dataset=False,training_authorized=False,
        package=str(out),generation_sha256=digest(out/'generation.json'),freeze_sha256=sha,
        authorization_sha256=digest(auth),native_commit=r['commit'],outputs=r['outputs'],counters=r['counters'],
        reused_counters=reused,splits=d.SPLITS,new_views=summaries,coverage=coverage,
        support_archive=dict(bytes=support.stat().st_size,sha256=digest(support)),training_projection=d.training_projection()))
    return len(summaries)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--verify',type=Path)
    args = cli.parse_args()
    f = v2.freeze(b.ROOT/v2.FREEZE,args.freeze_sha256)
    old = v2.prior_definition()
    paths = d.host_paths(old)
    failed = Path(f['failed_batch'])
    if args.execute:
        paths = tuple(Path('/inputs')/name for name in
                      ('oracle','admission.json','sparse','sparse-admission.json','sparse-verification.json'))
        failed = Path('/inputs/failed')
        auth = Path('/authorization.json')
        require(args.commit and digest(auth)==args.authorization_sha256
                and b.read(auth)==v2.authorization(f,args.commit,args.freeze_sha256),
                'Exact separate dense Bell v2 authorization required')
        require(os.getuid()!=0 and all(os.statvfs(p).f_flag & os.ST_RDONLY
                for p in (*paths,failed,auth,b.ROOT,Path('/'))) and not os.statvfs('/outputs').f_flag & os.ST_RDONLY
                and {row.split(':')[0].strip() for row in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Reviewed non-root, read-only, networkless runtime required')
    data = d.source(old,*paths)
    admitted = v2.admit_prefix(f,failed)
    if args.execute:
        execute(f,args.freeze_sha256,args.commit,args.authorization_sha256,paths,failed,data,admitted)
    elif args.verify is not None:
        print(f'Verified {verify(f,args.freeze_sha256,args.verify,paths,failed,data,admitted)} dense Bell references')
    else:
        print(json.dumps(dict(status='metadata_only',reused_pixels=admitted[0],new_work_limits=v2.LIMITS),sort_keys=True))


if __name__=='__main__':
    main()
