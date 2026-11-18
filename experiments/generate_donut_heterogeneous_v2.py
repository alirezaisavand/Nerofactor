"""One bounded diagnostic and conditional heterogeneous Donut completion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_heterogeneous_v1 as v1
from benchmarks import donut_heterogeneous_v2 as d
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import json_new
from benchmarks.generate_donut_recovery_v1 import highlight_record
from experiments.generate_donut_heterogeneous_v1 import compare, incident_grid, ordinary_probes, previews, slope_pair
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def archive(path, fields):
    with path.open('xb') as buffer:
        np.savez_compressed(buffer,**fields)


def capture(view, rows, render):
    selected = {k:view[k][rows] for k in ('position','shader_valid','normals','outgoing','axis',
                                         'widths_major_minor','color','metallic')}
    parts = [render(selected,grid) for grid in d.HIGH]
    return d.check_capture(view['pixel_indices'][rows], dict(
        pixels=view['pixel_indices'][rows],grids=np.asarray(d.HIGH),
        **{k:np.stack([p[i] for p in parts]) for i,k in enumerate(('components','incident_reflectance','visible_counts'))}))


def baseline(view, camera, charge, render, out):
    slopes = slope_pair(view,charge)
    low,high = (slopes[order] for order in v1.ORDERS)
    valid = view['shader_valid']
    gate = convergence(high['total'][valid],low['total'][valid])
    fields = dict(view,reflectance=high['total'],coarse_reflectance=low['total'],
                  reflectance_diffuse=high['diffuse'],reflectance_specular=high['specular'])
    if not gate['passed']:
        archive(out/f'slope-failure-{camera}.npz',fields)
        raise ValueError(f'Camera {camera} unoccluded slope reflectance failed: {gate}')
    lo,le,lv = render(view,d.BASE[0])
    hi,he,hv = render(view,d.BASE[1])
    return dict(fields,linear_components=hi,coarse_components=lo,incident_reflectance=he,
                coarse_incident_reflectance=le,visible_counts=hv),lv


def extra_gates(view):
    gates = compare(view)
    return {name:gate for name,gate in gates.items() if name!='radiance_convergence'}


def require_passing(gates, description):
    failed = {name:gate for name,gate in gates.items() if not gate['passed']}
    require(not failed,f'{description}: {failed}')


def finalize_view(camera, view, quadrature, out, prior):
    valid = view['shader_valid']
    summary,highlight = highlight_record(view['linear_components'][valid])
    mask = np.zeros(len(valid),dtype=bool)
    mask[valid] = highlight
    saved = out/f'view-{camera}.npz'
    if camera==0:
        np.testing.assert_array_equal(view['highlight_mask'],mask)
        shutil.copyfile(prior/saved.name,saved)
        for tag in ('RGB','diffuse','specular'):
            name=f'view-0-{tag}.png'
            shutil.copyfile(prior/name,out/name)
        preview = next(row['previews'] for row in v1.read(prior/'generation.json')['views'] if row['camera']==0)
    else:
        view['highlight_mask'] = mask
        archive(saved,view)
        preview = previews(out,camera,view)
    archive(out/f'quadrature-{camera}.npz',quadrature)
    return dict(camera=camera,valid=int(valid.sum()),geometry_hits=len(valid),
                **compare(view), ordinary_probe_pixels=view['pixel_indices'][ordinary_probes(view)].tolist(),
                selected_pixels=quadrature['refined_pixels'].tolist(),highlight_diagnostics=summary,previews=preview)


def execute(f, sha, commit, auth_sha, prior, paths, data):
    import torch
    from PIL import Image
    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.verified_ray_backend import load
    import resource

    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed native CUDA runtime required')
    geometry, labels, radius, originals = data
    out=Path('/outputs/reference')
    out.mkdir()
    started=time.monotonic()
    report=dict(protocol=d.PROTOCOL,status='incomplete',phase='three_point_diagnostic',
                commit=commit,freeze_sha256=sha,authorization_sha256=auth_sha,
                image=v1.IMAGE,retained_failed_generation_sha256=f['failed_records']['reference/generation.json']['sha256'],
                v1_freeze_sha256=d.V1_SHA,limits=d.LIMITS,refinement=d.POLICY,tolerances=v1.TOLERANCES,
                environment=v1.ENVIRONMENT,field=v1.FIELD,BRDF=v1.BRDF,indirect='zero',point_light='absent',
                panel=v1.PANEL,splits=v1.SPLITS,ring_radius_native=radius,
                counters=dict.fromkeys(d.WORK,0),new_high_points=0,refined_points=0,
                views=[],cleanup_verified=False,scientific_work_complete=False,
                training_authorized=False,native_model_calls=0,optimizer_returns=0)
    last_progress=started

    def budget():
        require(time.monotonic()-started < d.LIMITS['inner_seconds'], 'Reference v2 runtime exhausted')
        require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<d.LIMITS['output_bytes'],
                'Reference v2 output limit reached')

    def charge(key,cost):
        nonlocal last_progress
        require(report['counters'][key]+cost<=d.LIMITS[key], 'Reference v2 work cap: '+key)
        report['counters'][key]+=cost
        now=time.monotonic()
        if now-last_progress>=5:
            budget()
            p=out/'progress.next'
            p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            p.replace(out/'progress.json')
            last_progress=now

    def work():
        torch.set_num_threads(d.LIMITS['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(d.LIMITS['cuda_allocator_fraction'])
        report['backend']=load('predicate_fp64',v1.IMAGE)
        activate('material',PINS['material'])
        import raytracing
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,
                               GPU=torch.cuda.get_device_name(0),numpy=np.__version__)
        tracer=raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),
                                    geometry['triangles'].astype(np.int32))

        def render(view,grid):
            with torch.no_grad():
                return incident_grid(view,grid,tracer,torch,charge)

        reference=prior/'reference'
        first=originals[d.PROBE_CAMERA]
        rows=d.pixel_rows(first,d.PROBE_PIXELS)
        initial=capture(first,rows,render)
        archive(out/'camera50-high-grid-diagnostic.npz',initial)
        gates=d.diagnostic_gates(initial)
        report['diagnostic']=gates
        require_passing({'RGB':gates['RGB']}, '1024²/2048² diagnostic did not converge')
        print('Camera 50 three-point RGB diagnostic passed',flush=True)

        shutil.copyfile(reference/'surface-area.npz',out/'surface-area.npz')
        for camera in v1.PANEL:
            report['phase']=f'camera_{camera}'
            print(f'Heterogeneous v2 camera {camera} started',flush=True)
            if camera in d.REUSED:
                old=originals[camera]
                coarse_count=d.reused_coarse_counts(old)
                if camera==50:
                    shutil.copyfile(reference/'view-50.npz',out/'baseline-50.npz')
            else:
                old,coarse_count=baseline(labels[camera],camera,charge,render,out)
                archive(out/f'baseline-{camera}.npz',dict(old,baseline_coarse_visible_counts=coarse_count))
            require_passing(extra_gates(old), f'Camera {camera} baseline reflectance controls failed')
            if camera==0:
                require_passing(compare(old),'Byte-preserved camera 0 failed replay')

            def fetch(row):
                require(camera!=0 and report['refined_points']<d.MAX_TOTAL, 'Panel refinement budget exhausted')
                pixel=int(old['pixel_indices'][row])
                if camera==d.PROBE_CAMERA and pixel in d.PROBE_PIXELS:
                    result=d.probe_slice(initial,pixel)
                else:
                    result=capture(old,np.asarray([row]),render)
                    report['new_high_points']+=1
                archive(out/f'patch-{camera}-{pixel}.npz',result)
                report['refined_points']+=1
                return result

            final,quadrature=d.assemble(old,coarse_count,fetch)
            row=finalize_view(camera,final,quadrature,out,reference)
            report['views'].append(row)
            require_passing(compare(final),f'Camera {camera} assembled view failed')
            print(f'Heterogeneous v2 camera {camera} verified; refined={len(row["selected_pixels"])}',flush=True)
            budget()

    # Preserve the native error and mark cleanup only after the shared
    # synchronisation/cache cleanup returns successfully.
    def observed_work():
        try:
            work()
        except Exception as exc:
            return exc
        return None

    try:
        failure=run_with_cleanup(observed_work,torch)
        report['cleanup_verified']=True
        if failure is not None:
            raise failure
        require([v['camera'] for v in report['views']]==v1.PANEL
                and report['counters']==d.counters(report['new_high_points']),
                'Incomplete reference v2 work')
        d.freeze(d.ROOT/d.FREEZE,sha)
        d.source(f,prior,paths)
        report['phase']='complete'
        sheet=Image.new('RGB',(3*v1.SIZE,len(v1.PANEL)*v1.SIZE))
        for row,c in enumerate(v1.PANEL):
            for column,tag in enumerate(('RGB','diffuse','specular')):
                with Image.open(out/f'view-{c}-{tag}.png') as image:
                    sheet.paste(image,(column*v1.SIZE,row*v1.SIZE))
        sheet.save(out/'reference-components.png')
        budget()
        report.update(status='complete_pending_researcher_admission',scientific_work_complete=True)
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['outputs']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir())
                           if p.is_file() and p.name not in ('progress.json','progress.next')}
        report['wall_seconds']=time.monotonic()-started
        report['host_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['CUDA_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        json_new(out/'generation.json',report)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    cli.add_argument('--execute',action='store_true')
    args=cli.parse_args()
    f=d.freeze(d.ROOT/d.FREEZE,args.freeze_sha256)
    if args.execute:
        paths=tuple(Path(p) for p in ('/inputs/oracle','/inputs/cameras.json','/inputs/reference','/inputs/admission.json'))
        prior=Path('/inputs/failed-v1')
        auth=Path('/authorization.json')
        require(args.commit and digest(auth)==args.authorization_sha256
                and v1.read(auth)==d.authorization(f,args.commit,args.freeze_sha256),'Exact v2 authorization required')
        require(os.getuid()!=0 and all(os.statvfs(p).f_flag&os.ST_RDONLY for p in [*paths,prior,auth,d.ROOT,Path('/')])
                and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
                and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Reviewed non-root, read-only, networkless runtime required')
    else:
        paths=tuple(Path(f['host_inputs'][k]) for k in ('oracle','cameras','reference','admission'))
        prior=Path(f['retained_failed_batch'])
    data=d.source(f,prior,paths)
    if args.execute:
        execute(f,args.freeze_sha256,args.commit,args.authorization_sha256,prior,paths,data)
    else:
        d.pixel_rows(data[3][50],d.PROBE_PIXELS)
        print(json.dumps(dict(status='metadata_only',prior_verified=True,panel=v1.PANEL,
                              reference_calls=0,refinement=d.POLICY)))


if __name__=='__main__':
    main()
