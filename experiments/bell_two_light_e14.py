"""One explicitly authorized E14 second-light integration; cached geometry only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from benchmarks.material_information_e14 import rotated_light,SIGMA
from experiments.bell_multiview_e11g import (CASES,LIGHT,GRIDS,inputs as prior_inputs,
    source_paths as prior_sources,select_views,transformed_material,kernel_axes,weighted_components)
from experiments.analyze_bell_multiview_e11g import verify_geometry,verify_grid
from experiments.bell_anisotropy_e11f import strengthen,admit_isotropic
from experiments.bell_highlight_e11d import write_json
from experiments.bell_two_light_e14_analysis import analyze
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.verified_ray_backend import identity

PROTOCOL='bell-two-light-observability-e14-v1'
G_REPORT='db1c0b3d5ce630501645d07612f940e46df6bfa77e551d839882be8503205e09'
LIGHTS=(LIGHT,rotated_light(LIGHT))
LIMITS=dict(wall_seconds=900,grace_seconds=30,cpu_threads=4,host_memory_gib=64,
            cuda_allocator_fraction=.25,output_bytes=256*1024**2,point_batch=4,direction_chunk=16384,
            new_primary_rays=0,new_shadow_queries=0,new_outgoing_queries=0,new_L2_brdf_queries=0,
            environment_point_directions=27525120,brdf_point_directions=1658060800)


def source_paths():
    return sorted(set(prior_sources())|{'docs/protocols/BELL_MULTIVIEW_E11G_V1.freeze.json',
        'docs/evidence/BELL_MULTIVIEW_E11G_RESULTS_V1.json','docs/BELL_MULTIVIEW_E11G_RESULTS_V1.md',
        'experiments/analyze_bell_multiview_e11g.py','benchmarks/material_information_e14.py',
        'experiments/bell_two_light_e14.py','experiments/bell_two_light_e14_analysis.py',
        'experiments/run_bell_two_light_e14.sh','run_bell_two_light_e14.py',
        'tests/benchmarks/test_material_information_e14.py','tests/benchmarks/test_bell_two_light_e14.py',
        'docs/protocols/BELL_TWO_LIGHT_INFORMATION_E12_V1.md','docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.md'})


def record(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def inputs(benchmark,pilot,reference,factor,multiview):
    data=prior_inputs(benchmark,pilot,reference,factor)
    labels,wo,pixels,cache,geometry,records=data;plan=select_views(geometry)
    require(plan['camera_ids']==[58,44,9,0],'Admitted outgoing view identity changed')
    report=multiview/'multiview.json'
    require(report.resolve()==report and digest(report)==G_REPORT,'Accepted E11G report changed')
    r=json.loads(report.read_text())
    require(r['status']=='complete' and r['cleanup_verified'] is True and r['comparison']['all_measured_cases_converged'] is True and
            r['view_plan']==plan and r['cases']==list(CASES) and r['lighting']==LIGHT and r['backend']==identity(),
            'Accepted E11G scientific identities differ')
    records.append(record(report))
    for item in r['outputs']:
        p=multiview/item['path']
        require(p.resolve()==p and multiview in p.parents and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],
                'Accepted multiview artifact changed')
        records.append(record(p))
    with np.load(multiview/'outgoing-geometry.npz',allow_pickle=False) as z:outgoing={k:z[k].copy() for k in z.files}
    verify_geometry(outgoing,geometry,plan,pixels);valid=outgoing['valid']
    require(int(valid.sum())==220 and int(valid[1:].sum())==136 and int(outgoing['occluded'].sum())==10,
            'Admitted point/view coverage differs; no replacement allowed')
    for size in GRIDS:
        with np.load(multiview/('grid-'+str(size)+'.npz'),allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
        cache[size]['L2']=verify_grid(saved,valid,pixels,plan['camera_ids'],cache[size])
    return labels,pixels,cache,outgoing,records


def logical_inputs(records,args):
    roots={name:getattr(args,name) for name in ('benchmark','pilot','reference','factor','multiview')}
    result=[]
    for row in records:
        p=Path(row['path']);names=[name for name,path in roots.items() if path in p.parents]
        require(len(names)==1,'Ambiguous explicit input root')
        name=names[0];result.append(dict(root=name,path=p.relative_to(roots[name]).as_posix(),bytes=row['bytes'],sha256=row['sha256']))
    return result


def definition(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['limits']==LIMITS and f['sigma_linear_RGB']==SIGMA and
            f['lights']==list(LIGHTS) and f['cases']==list(CASES) and f['grids']==list(GRIDS) and
            f['G_report_sha256']==G_REPORT and set(f['files'])==set(source_paths()),'E14 definition mismatch')
    for path,h in f['files'].items():require(digest(ROOT/path)==h,'Frozen source changed: '+path)
    data=inputs(args.benchmark,args.pilot,args.reference,args.factor,args.multiview)
    require(logical_inputs(data[-1],args)==f['inputs'],'Frozen input identity mismatch')
    return f,data


def authorize(args,f):
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==args.commit and
            a.get('freeze_sha256')==args.freeze_sha256 and a.get('limits')==LIMITS and
            a.get('image_id')==args.image_id==identity()['image_id'] and a.get('output_directory')==str(args.output),
            'Separate exact E14 execution authorization required')
    require(os.geteuid()!=0 and args.output.is_absolute() and args.output.resolve()==args.output and
            args.output.parent==Path('/outputs') and not args.output.exists(),'Fresh non-root output required')
    for p in (Path('/'),ROOT,args.freeze,args.authorization,args.benchmark,args.pilot,args.reference,args.factor,args.multiview):
        require(os.statvfs(p).f_flag & os.ST_RDONLY,'Read-only inputs/source/root required')
    require(not os.statvfs('/outputs').f_flag & os.ST_RDONLY,'Writable outputs required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')


def render(args,data,r,start,torch):
    from benchmarks.bell_reference_e11 import basis,environment
    from benchmarks.generate_bell_e11 import write_arrays
    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and torch.cuda.device_count()==1 and
            'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    for stage in PINS:baseline(stage,PINS[stage])
    r['cached_tracing_identity']=identity()
    require(digest(Path(identity()['binary_path']))==identity()['binary_sha256'],'Image tracing dependency changed')
    r['runtime']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    labels,pixels,cache,outgoing,records=data;valid=outgoing['valid'];wo=outgoing['outgoing_directions']
    original={k:torch.from_numpy(v).cuda() for k,v in labels.items()};stronger=strengthen(torch,original)
    r['isotropic_equivalence']=admit_isotropic(torch,original,stronger)
    r['isotropic_equivalence']['reused_case']='Common canonical isotropic material; L2 data are cached; rotated-light isotropic evaluated once per admitted point/view and shared.'
    material=(original,stronger);results={}
    def budget(kind,n):
        cap=LIMITS['brdf_point_directions'] if kind=='brdf_attempts' else LIMITS['environment_point_directions']
        require(r['counters'][kind]+n<=cap and time.monotonic()-start<840 and
                sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],
                'E14 fixed work/resource limit reached')
        r['counters'][kind]+=n
    with torch.no_grad():
        for size in GRIDS:
            c=cache[size];N=size*size
            array=np.full((2,2,4,84,12,3,3),np.nan,dtype=np.float64);array[:,0]=c['L2']
            completed=np.zeros(84,bool)
            r['_partial']=dict(components=array,valid=valid,completed_probes=completed,pixel_indices=pixels,grid_size=np.array(size))
            for j in range(0,84,4):
                n=original['normals'][j:j+4];t,b=basis(n)
                acc=torch.zeros((2,4,4,12,2,3),dtype=torch.float64,device='cuda')
                for offset in range(0,N,16384):
                    stop=min(offset+16384,N);loc=torch.from_numpy(c['nodes'][offset:stop]).cuda()
                    wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
                    cosine=(wi*n[:,None]).sum(-1)
                    require(float(abs(torch.linalg.norm(wi,dim=-1)-1).max())<=2e-12 and
                            float(abs(cosine-loc[None,:,2]).max())<=2e-12 and bool((cosine>0).all()),'Cached incident direction/horizon invariant failed')
                    visible=torch.from_numpy(np.unpackbits(c['visibility'][j:j+4,offset//8:stop//8],axis=1,bitorder='big').astype(np.bool_)).cuda()
                    budget('environment_attempts',4*(stop-offset));light=environment(wi,r['lighting'][1])
                    r['counters']['environment_returns']+=4*(stop-offset)
                    for view in range(4):
                        local=np.flatnonzero(valid[view,j:j+4])
                        if not len(local):continue
                        ix=torch.from_numpy(local).cuda();v=torch.from_numpy(wo[view,j:j+4][local]).cuda()
                        for arm in range(2):
                            current={k:value[j:j+4][ix] for k,value in material[arm].items()}
                            for k,case in enumerate(CASES):
                                if arm==1 and case=='isotropic':continue
                                count=len(local)*(stop-offset);budget('brdf_attempts',count)
                                fd,fs=kernel_axes(torch,wi[ix],v,transformed_material(torch,current,case))
                                acc[arm,view,ix,k]+=weighted_components(torch,fd,fs,light[ix],cosine[ix],visible[ix],c['weight'])
                                r['counters']['brdf_returns']+=count
                pairs=torch.cat([acc,acc.sum(-2,keepdim=True)],-2).cpu().numpy()
                for arm in range(2):
                    for view in range(4):
                        for point in range(j,j+4):
                            if valid[view,point]:array[arm,1,view,point]=pairs[arm,view,point-j]
                array[1,1,:,j:j+4,7]=array[0,1,:,j:j+4,7];completed[j:j+4]=True
                tmp=args.output/'progress.next';tmp.write_text(json.dumps(dict(grid=size,completed_probes=int(completed.sum()),counters=r['counters']))+'\n');tmp.replace(args.output/'progress.json')
            require(np.array_equal(array[:,0],c['L2'],equal_nan=True),'Cached first-light measurements changed')
            for arm in range(2):
                measured=array[arm,1,valid]
                require(np.isfinite(measured).all() and (measured>=0).all() and
                        np.array_equal(measured[:,:,0]+measured[:,:,1],measured[:,:,2]) and
                        np.array_equal(measured[:,:,0],np.repeat(measured[:,:1,0],12,axis=1)),
                        'Rotated-light finite/component/diffuse invariant failed')
            require(np.array_equal(array[0,1,valid,7],array[1,1,valid,7]),'Shared new isotropic control changed')
            write_arrays(args.output/('grid-'+str(size)+'.npz'),components=array,valid=valid,pixel_indices=pixels,
                         camera_ids=outgoing['camera_ids'],completed_probes=completed)
            results[size]=array;r['completed_grids'].append(size)
            print(json.dumps(dict(grid=size,counters=r['counters'])),flush=True)
    expected=23*int(valid.sum())*sum(s*s for s in GRIDS)
    require(r['counters']['brdf_attempts']==r['counters']['brdf_returns']==expected and
            r['counters']['environment_attempts']==r['counters']['environment_returns']==84*sum(s*s for s in GRIDS),
            'E14 exact work accounting failed')
    require(not any(n.split('.')[0] in ('raytracing','_raytracing','network','train') for n in sys.modules),
            'Forbidden model/tracing import; zero geometry-query boundary violated')
    import resource
    return results,dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def execute(args,f,data):
    authorize(args,f);args.output.mkdir();start=time.monotonic()
    r=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,scientific_work_complete=False,
           commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
           source_files=f['files'],inputs=data[-1],limits=LIMITS,lighting=f['lights'],cases=list(CASES),
           counters=dict(brdf_attempts=0,brdf_returns=0,environment_attempts=0,environment_returns=0,
                         new_primary_rays=0,new_shadow_queries=0,new_outgoing_queries=0,new_L2_brdf_queries=0,
                         model_calls=0,optimizer_returns=0),completed_grids=[])
    try:
        import scipy,torch
        require(scipy.__version__=='1.10.1','Reviewed SciPy1.10.1 required for information diagnostics')
        from benchmarks.generate_bell_e11 import run_with_cleanup
        results,resources=run_with_cleanup(lambda:render(args,data,r,start,torch),torch)
        r.update(resources);r['cleanup_verified']=True
        # CPU secant/information analysis uses only captured arrays, no further rendering.
        r['comparison']=analyze(results[GRIDS[-1]],results[GRIDS[0]],data[3]['valid'])
        write_json(args.output/'comparison.json',r['comparison'])
        require(all(digest(ROOT/p)==h for p,h in f['files'].items()) and
                all(digest(Path(x['path']))==x['sha256'] for x in data[-1]),'Terminal source/input integrity failed')
        require(time.monotonic()-start<840 and sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes']-4096,'Terminal resource limit failed')
        r['outputs']=[record(p) for p in sorted(args.output.iterdir()) if p.is_file() and not p.name.startswith('progress')]
        for row in r['outputs']:row['path']=str(Path(row['path']).relative_to(args.output))
        r.update(status='complete',scientific_work_complete=True)
    except BaseException as exc:
        r.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        partial=r.pop('_partial',None)
        if partial is not None and r['status']!='complete':
            from benchmarks.generate_bell_e11 import write_arrays
            write_arrays(args.output/'incomplete-grid.npz',**partial)
        r['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'two-light.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','pilot','reference','factor','multiview','freeze'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--execute',action='store_true')
    for name in ('output','authorization'):p.add_argument('--'+name,type=Path)
    for name in ('commit','image-id','authorization-sha256'):p.add_argument('--'+name)
    args=p.parse_args();f,data=definition(args)
    if not args.execute:
        print(json.dumps(dict(status='metadata_only',observations=int(data[3]['valid'].sum()),new_BRDF_queries=0,new_geometry_queries=0)));return
    require(all((args.output,args.authorization,args.commit,args.image_id,args.authorization_sha256)),'Separate isolated execution approval required')
    execute(args,f,data)


if __name__=='__main__':main()
