"""Run or verify independent relighting of accepted learned-light Donut fields."""
from argparse import ArgumentParser
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_learned_light_relighting_gate9_v1 as g
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_relighting_gate6_v1 import raster, scores
from experiments.donut_reconstructed_geometry_v1 import interrupted, save
from experiments.donut_relighting_gate6_v1 import integrate
from infrastructure.signatures import require


def gates(grids):
    return {arm:convergence(grids[1,i,:,2],grids[0,i,:,2]) for i,arm in enumerate(g.ARMS)}


def execute(sha,commit,auth_sha):
    f=g.freeze(sha)
    paths=g.paths(f,True)
    authorization=Path('/authorization.json')
    out=Path('/outputs/run')
    require(os.geteuid()!=0 and not out.exists() and out.resolve()==out
            and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
            and all(os.statvfs(path).f_flag&os.ST_RDONLY for path in (Path('/'),g.ROOT,authorization,*paths.values()))
            and {line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Fresh non-root read-only Gate 9 execution required')
    require(g.digest(authorization)==auth_sha and g.read(authorization)==g.authorization(f,commit,sha),
            'Exact separate Gate 9 approval required')
    out.mkdir()
    started=last=time.monotonic()
    receipt=dict(protocol=g.PROTOCOL,status='incomplete',commit=commit,freeze_sha256=sha,authorization_sha256=auth_sha,
                 image=f['image'],counters=dict.fromkeys(g.LIMITS,0),captures={},views={},cleanup_verified=False)

    def budget(extra=0):
        require(time.monotonic()-started<g.RESOURCES['inner_seconds'],'Gate 9 wall limit')
        require(sum(p.stat().st_size for p in Path('/outputs').rglob('*') if p.is_file())+extra
                <=g.RESOURCES['output_bytes'],'Gate 9 owned-output limit')

    def progress():
        p=out/'progress.next'
        if p.exists():
            p.unlink()
        json_new(p,receipt)
        p.replace(out/'progress.json')

    def charge(key,count):
        nonlocal last
        require(receipt['counters'][key]+count<=g.LIMITS[key],'Gate 9 work cap: '+key)
        if time.monotonic()-last>=5:
            budget()
            progress()
            last=time.monotonic()
        receipt['counters'][key]+=count

    def work():
        torch.set_num_threads(g.RESOURCES['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(g.RESOURCES['cuda_allocator_fraction'])
        receipt['backend']=load('predicate_fp64',f['image'])
        require(receipt['backend']==f['backend'],'Corrected backend differs')
        from raytracing import RayTracer

        geom=numeric(g.read_input(f,paths,'oracle'))
        tracer=RayTracer(geom['vertices_native'].astype('f4'),geom['triangles'].astype('i4'))
        # Only the role inventory differs. The kernel, visibility and quadrature are the Gate 6 functions.
        from types import SimpleNamespace
        policy=SimpleNamespace(ROLES=g.ARMS,POINT_BATCH=g.g6.POINT_BATCH,DIRECTION_CHUNK=g.g6.DIRECTION_CHUNK,
                               LIGHT=f['light'])
        render=g.with_roles(integrate,g=policy)
        total=0
        for camera in g.CAMERAS:
            view=views[camera]
            print(f'Independent relighting camera {camera}',flush=True)

            def fetch(rows,grid,prefix):
                receipt['in_progress']=dict(camera=camera,grid=grid,prefix=prefix)
                with torch.no_grad():
                    values=render(view,materials[camera],rows,grid,tracer,torch,charge)
                budget(sum(x.nbytes for x in values.values())+65536)
                name=f'camera-{camera}-{prefix}-{grid}.npz'
                path=out/name
                save(path,values)
                receipt['captures'][name]=dict(bytes=path.stat().st_size,sha256=g.digest(path))
                return values

            rgb,_,_,done=g.refine(view['pixel_indices'],fetch,total)
            total+=len(done)
            receipt['views'][str(camera)]=dict(refined_pixels=done,convergence=gates(rgb))
            progress()
        receipt.pop('in_progress',None)
        receipt['refined_pixels_total']=total

    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        from experiments.bell_quadrature_e11e import controls
        from infrastructure.verified_ray_backend import load

        require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
                and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
                and 'A100' in torch.cuda.get_device_name(0),'Reviewed A100 runtime required')
        receipt['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        views,materials,receipt['endpoint_parity']=g.admit(f,paths)
        receipt['quadrature_controls']=controls()
        run_with_cleanup(work,torch)
        receipt['cleanup_verified']=True
        require(receipt['counters']==g.expected_work(receipt['refined_pixels_total']),'Incomplete Gate 9 work')
        g.freeze(sha)
        g.admit(f,paths)
        budget()
        receipt.update(status='complete',cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started,
                       host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        json_new(out/'receipt.json',receipt)


def completed(f,sha):
    batch=Path(f['host_output'])
    q=batch/'authorization.json'
    auth=g.read(q)
    run=batch/'run'
    r=g.read(run/'receipt.json')
    require((batch/'exit-code.txt').read_text().strip()=='0' and not (batch/'cleanup.txt').exists()
            and auth==g.authorization(f,auth['commit'],sha) and g.digest(batch/'freeze.json')==sha
            and g.digest(q)==g.digest(batch.with_name(batch.name+'-authorization.json'))
            and r['status']=='complete' and r['cleanup_verified'] and r['protocol']==g.PROTOCOL
            and r['commit']==auth['commit'] and r['freeze_sha256']==sha and r['authorization_sha256']==g.digest(q)
            and r['image']==f['image'] and r['backend']==f['backend']
            and r['elapsed_seconds']<g.RESOURCES['inner_seconds']
            and r['host_peak_rss_bytes']<=g.RESOURCES['host_memory_gib']*1024**3
            and sum(item['bytes'] for item in r['captures'].values())<=g.RESOURCES['output_bytes']
            and set(r['views'])==set(map(str,g.CAMERAS))
            and r['counters']==g.expected_work(r['refined_pixels_total'])
            and set(r['captures'])=={p.name for p in run.glob('*.npz')},
            'Incomplete or foreign Gate 9 receipt')
    return run,r


def analyze(sha,output):
    f=g.freeze(sha)
    run,r=completed(f,sha)
    require(output.resolve()==output and not output.exists() and output.parent.is_dir()
            and run.parent not in output.parents,'Fresh offline Gate 9 output required')
    p=g.paths(f)
    views,_,parity=g.admit(f,p)
    require(r['endpoint_parity']==parity,'Saved material-field admission differs')
    total=0
    seen=set()
    panels={}
    diagnostics={}
    known=('K1-Oracle','Final-Chart-Oracle','Ambient-Oracle')
    for camera,v in views.items():
        pixels=v['pixel_indices']
        archived=numeric(g.read_input(f,p,f'reference_{camera}'))
        masks={key[5:]:value for key,value in archived.items() if key.startswith('mask_')}
        require(np.array_equal(archived['pixels'],pixels)
                and np.array_equal(np.flatnonzero(masks['expanded_foreground']),pixels)
                and g.g6.pixels_sha(np.flatnonzero(masks['common']))==f['gate6_common_support'][str(camera)]['matched_sha256'],
                'Archived target support differs')

        def fetch(rows,grid,prefix):
            name=f'camera-{camera}-{prefix}-{grid}.npz'
            require(name not in seen,'Duplicate quadrature capture')
            seen.add(name)
            return numeric(g.check(run/name,r['captures'][name]))

        rgb,_,res,done=g.refine(pixels,fetch,total)
        total+=len(done)
        current=gates(rgb)
        prior=r['views'][str(camera)]
        require(prior['refined_pixels']==done and prior['convergence'].keys()==current.keys(),
                'Frozen refinement decisions differ')
        for arm,row in current.items():
            require(row['passed'] and prior['convergence'][arm]['passed'] is True
                    and prior['convergence'][arm].keys()==row.keys(),'Convergence decision differs')
            for key,value in row.items():
                if isinstance(value,float):
                    np.testing.assert_allclose(value,prior['convergence'][arm][key],atol=1e-15,rtol=1e-12)
                else:
                    require(value==prior['convergence'][arm][key],'Convergence inventory differs')
        predictions={arm:raster(rgb[1,i],pixels) for i,arm in enumerate(g.ARMS)}
        predictions.update({arm:archived[arm+'_independent'] for arm in known})
        panels[camera]=dict(reference=archived['reference'],masks=masks,predictions=predictions)
        diagnostics[str(camera)]=dict(refined_pixels=done,convergence=current,
            grid_counts={str(size):int((res[1]==size).sum()) for size in np.unique(res[1])},
            oracle_RGB_grid_RMS=float(np.sqrt(np.mean((archived['reference'][pixels,2]
                                                     -archived['reference_coarse'][pixels,2])**2))))

    def group(chosen):
        truth=np.concatenate([a['reference'] for a in chosen])
        masks={key:np.concatenate([a['masks'][key] for a in chosen]) for key in chosen[0]['masks']}
        return {arm:{name:scores(np.concatenate([a['predictions'][arm] for a in chosen]),truth,mask)
                     for name,mask in masks.items()} for arm in (*g.ARMS,*known)}

    require(seen==set(r['captures']) and total==r['refined_pixels_total'],'Incomplete quadrature capture replay')
    light={}
    for arm in g.ARMS:
        q=numeric(g.read_input(f,p,f'illumination_{arm}'))
        require(q['directions'].shape==(128,256,3),'Historical illumination grid differs')
        weights=np.sqrt(1-q['directions'][...,2]**2)
        light[arm]=g.illumination_score(q['linear_RGB'],q['known_linear_RGB'],weights)
    output.mkdir()
    json_new(output/'comparison.json',dict(status='verified_complete',protocol=g.PROTOCOL,
        native_receipt_sha256=g.digest(run/'receipt.json'),independent_unseen_by_camera={str(c):group([a]) for c,a in panels.items()},
        independent_unseen_pooled=group(list(panels.values())),integration=diagnostics,
        learned_fitting_light_evaluation=light,known_light_histories_role='context, different training recipe and support',
        scientific_calls_during_analysis=0))
    from benchmarks.bell_e11 import linear_to_srgb
    from PIL import Image
    # Same exposure and display-only clipping across stacked rows.
    for camera,a in panels.items():
        row=np.concatenate([a['reference'][:,2].reshape(g.SIZE,g.SIZE,3)]+
            [a['predictions'][arm][:,2].reshape(g.SIZE,g.SIZE,3) for arm in (*g.ARMS,*known)],axis=1)
        Image.fromarray(linear_to_srgb(np.clip(row,0,1))).save(output/f'camera-{camera}-reference-D2-D8C-D8T-K1-chart-ambient.png')


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument('--execute',action='store_true')
    action.add_argument('--analyze',type=Path)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    args=cli.parse_args()
    if args.execute:
        require(args.commit and args.authorization_sha256,'Explicit Gate 9 authorization identity required')
        signal.signal(signal.SIGTERM,interrupted)
        execute(args.freeze_sha256,args.commit,args.authorization_sha256)
    else:
        analyze(args.freeze_sha256,args.analyze)


if __name__=='__main__':
    main()
