"""E11E: one authorized midpoint-solid-angle check,84 fixed Bell probes only.

Metadata and synthetic controls import no Torch. No native material shader/model.
The physical kernel/counterfactuals are the unchanged independent E11D reference.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.bell_highlight_e11d import ROOT, GENERATION_SHA, CASES, candidates, subset, write_json
from experiments.bell_highlight_pilot_e11d import counterfactual, kernel_axes, sensitivity
from benchmarks.bell_e11 import convergence
from infrastructure.signatures import require, read_report
from infrastructure.stage_sources import PINS, baseline, digest
from infrastructure.verified_ray_backend import identity

PROTOCOL='bell-l2-midpoint-quadrature-e11e-v1'
PILOT_SHA='4ad7c79fb63713c5710aa8407c3229fb8bd71341dc81d39e985cb24608a3f1f1'
MIS_SHA='71d64abbaf312c2c2d24a337639e0458b9e3025678c7b3d92494a89dab889797'
GRIDS=(256,512)
LIGHT=candidates()['L2_six_cardinal']
LIMITS={'wall_seconds':900,'grace_seconds':30,'cpu_threads':4,'host_memory_gib':64,
        'cuda_allocator_fraction':.25,'output_bytes':256*1024**2,'point_batch':4,
        'direction_chunk':16384,'shadow_directions':27525120,'brdf_point_directions':220200960}


def source_paths():
    old='docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.freeze.json'
    return sorted(set(json.loads((ROOT/old).read_text())['files']) | {old,
        'docs/BELL_HIGHLIGHT_PILOT_E11D_RESULTS_V1.md','docs/evidence/BELL_HIGHLIGHT_PILOT_E11D_RESULTS_V1.json',
        'experiments/bell_quadrature_e11e.py','experiments/run_bell_quadrature_e11e.sh',
        'run_bell_quadrature_e11e.py','tests/benchmarks/test_bell_quadrature_e11e.py',
        'docs/protocols/BELL_QUADRATURE_E11E_V1.md'})


def midpoint_grid(size):
    require(type(size) is int and size>=2,'Integer midpoint grid size>=2 required')
    z=(np.arange(size,dtype=np.float64)+.5)/size
    phi=(np.arange(size,dtype=np.float64)+.5)*(2*np.pi/size)
    zz,pp=np.meshgrid(z,phi,indexing='ij');radius=np.sqrt(1-zz*zz)
    local=np.stack([radius*np.cos(pp),radius*np.sin(pp),zz],-1).reshape(-1,3)
    return local,2*np.pi/(size*size)


def controls():
    """Tiny analytic geometry/quadrature controls, no BRDF or visibility queries."""
    rows=[]
    for size in (4,8,32):
        d,w=midpoint_grid(size);z=d[:,2]
        area=abs(len(d)*w-2*np.pi);cosine=abs(float(z.sum()*w)-np.pi)
        z2=float((z*z).sum()*w);expected=2*np.pi*(1/3-1/(12*size**2))
        unit=float(abs(np.linalg.norm(d,axis=1)-1).max())
        require(area<=2e-13 and cosine<=2e-14 and abs(z2-expected)<=2e-14 and unit<=2e-14
                and (z>0).all() and (z<1).all(),'Analytic solid-angle control failed')
        rows.append(dict(size=size,area_error=area,cosine_integral_error=cosine,
                         z2_integral=z2,z2_midpoint_exact=expected,unit_error=unit))
    return dict(status='pass',solid_angle_domain='z=cos(theta) in(0,1),phi in(0,2pi)',
                pdf=1/(2*np.pi),weight='dOmega=2*pi/N; SUM f*Li*cos*dOmega, not mean and no extra sin(theta)',rows=rows)


def query_visibility(torch,tracer,positions,directions):
    """Exactly E11D float32 origin/direction and depth>=10 policy; no redraw."""
    wi32=directions.float();origin=positions.float()[:,None]+1e-5*wi32
    _,_,depth=tracer.trace(origin,wi32)
    require(bool(torch.isfinite(depth).all()),'Nonfinite native shadow depth')
    return depth>=10,depth


def weighted_components(torch,fd,fs,light,cosine,visible,domega):
    require(np.isfinite(domega) and domega>0 and bool((cosine>0).all()),'Invalid hemisphere weight/domain')
    weight=cosine*visible.to(torch.float64)*domega
    values=torch.stack([fd*light*weight[...,None],fs*light*weight[...,None]],2)
    require(bool(torch.isfinite(values).all()) and bool((values>=0).all()),'Invalid quadrature contribution')
    return values.sum(1)  # dOmega already contains1/N; never average again.


def record(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def inputs(benchmark,pilot):
    require(all(p.is_absolute() and p.resolve()==p for p in (benchmark,pilot)),'Explicit unaliased input roots required')
    require(digest(benchmark/'geometry/generation.json')==GENERATION_SHA and digest(pilot/'pilot.json')==PILOT_SHA,
            'Admitted v3/E11D report changed')
    original=json.loads((pilot/'pilot.json').read_text());generation=json.loads((benchmark/'geometry/generation.json').read_text())
    require(original['status']=='complete' and original['cleanup_verified'] is True and
            original['recommendation']['recommended'] is None and original['recommendation']['decision']=='FIX_AND_PROCEED'
            and original['backend']==identity() and original['lighting']['L2_six_cardinal']==LIGHT,'E11D identity/failed admission changed')
    consumed=[record(benchmark/'geometry/generation.json'),record(pilot/'pilot.json')]
    for item in original['outputs']:
        p=pilot/item['path'];require(p.resolve()==p and pilot in p.parents and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'E11D artifact changed')
    wanted=['geometry/mesh.ply','geometry/labels/58.npz','geometry/rays/58.npz','references/58.npz']
    for name in wanted:
        p=benchmark/name;item=next(x for x in generation['outputs'] if x['path']==name)
        require(p.resolve()==p and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'Consumed v3 artifact changed: '+name)
        consumed.append(record(p))
    p=pilot/'58-L2_six_cardinal.npz';require(digest(p)==MIS_SHA,'Pinned MIS capture mismatch');consumed.append(record(p))
    with np.load(p,allow_pickle=False) as z:
        pixels=z['pixel_indices'].copy();mis=z['paired_components'].copy();mis_coarse=z['paired_coarse_components'].copy()
    with np.load(benchmark/'references/58.npz',allow_pickle=False) as z:
        require(np.array_equal(pixels,subset(z['hit'])) and len(pixels)==84,'Original84 probes required')
    with np.load(benchmark/'geometry/labels/58.npz',allow_pickle=False) as z:
        indices=np.searchsorted(z['pixel_indices'],pixels);require(np.array_equal(z['pixel_indices'][indices],pixels),'Probe/label identity mismatch')
        labels={k:z[k][indices].copy() for k in ('position','normals','T','B','theta_major','widths_major_minor','color','metallic')}
    with np.load(benchmark/'geometry/rays/58.npz',allow_pickle=False) as z:
        wo=-z['directions'][pixels].astype(np.float64);wo/=np.linalg.norm(wo,axis=-1,keepdims=True)
    require(mis.shape==mis_coarse.shape==(84,8,3,3) and mis.dtype==mis_coarse.dtype==np.float64 and
            all(np.isfinite(a).all() for a in labels.values()) and ((wo*labels['normals']).sum(1)>0).all(), 'Invalid admitted numerical inputs')
    return labels,wo,pixels,mis,mis_coarse,consumed


def difference(a,b):
    delta=a-b
    return dict(max_absolute=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-15)),
                rgb_rmse=float(np.sqrt(np.mean(delta**2))),signed_mean_rgb=delta.mean(0).tolist())


def compare(coarse,fine,mis,mis_coarse):
    require(all(a.shape==(84,8,3,3) and a.dtype==np.float64 and np.isfinite(a).all() and (a>=0).all()
                for a in (coarse,fine,mis,mis_coarse)),'Complete finite84x8 components required')
    require(all(np.array_equal(a[:,:,0]+a[:,:,1],a[:,:,2]) for a in (coarse,fine,mis,mis_coarse)), 'Component sum mismatch')
    gates={case:convergence(fine[:,k,2],coarse[:,k,2]) for k,case in enumerate(CASES)}
    return dict(original_E11D_decision='FIX_AND_PROCEED',original_E11D_recommended=None,E11D_admission_changed=False,
                two_grid_convergence=gates,all_cases_two_grid_converged=all(x['passed'] for x in gates.values()),
                sensitivity=sensitivity(fine,coarse),preserved_MIS_sensitivity=sensitivity(mis,mis_coarse),
                comparison={case:{'fine_quadrature_minus_coarse':difference(fine[:,k,2],coarse[:,k,2]),
                    'fine_quadrature_minus_MIS_fine':difference(fine[:,k,2],mis[:,k,2]),
                    'coarse_quadrature_minus_MIS_fine':difference(coarse[:,k,2],mis[:,k,2]),
                    'fine_quadrature_minus_MIS_coarse':difference(fine[:,k,2],mis_coarse[:,k,2])} for k,case in enumerate(CASES)},
                component_comparison={case:{name:{'fine_minus_coarse':difference(fine[:,k,c],coarse[:,k,c]),
                    'fine_minus_MIS_fine':difference(fine[:,k,c],mis[:,k,c])}
                    for c,name in enumerate(('diffuse','specular'))} for k,case in enumerate(CASES)},
                interpretation='Two deterministic grids and one MIS tape cannot prove absolute accuracy or statistical estimator bias.',
                full_dataset_or_training_authorized=False)


def authorize(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['grids']==list(GRIDS) and f['lighting']==LIGHT and f['limits']==LIMITS
            and f['cases']==list(CASES) and f['camera_id']=='58' and f['probes']==84 and
            set(f['files'])==set(source_paths()),'Frozen E11E definition changed')
    for p,sha in f['files'].items():require(digest(ROOT/p)==sha,'Frozen source changed: '+p)
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==args.commit
            and a.get('freeze_sha256')==args.freeze_sha256 and a.get('image_id')==args.image_id==identity()['image_id']
            and a.get('output_directory')==str(args.output) and a.get('limits')==LIMITS,'Exact independent E11E authorization required')
    require(args.output.is_absolute() and args.output.resolve()==args.output and args.output.parent==Path('/outputs')
            and not args.output.exists() and os.geteuid()!=0,'Fresh /outputs child and non-root execution required')
    for p in (Path('/'),ROOT,args.benchmark,args.pilot,args.authorization,args.freeze):
        require(os.statvfs(p).f_flag & os.ST_RDONLY,'Read-only source, originals, authorization and root required')
    require(not os.statvfs('/outputs').f_flag & os.ST_RDONLY,'Separate writable output mount required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')
    return f


def execute(args):
    f=authorize(args);data=inputs(args.benchmark,args.pilot);control=controls()
    args.output.mkdir();start=time.monotonic()
    report=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,commit=args.commit,
                freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,source_files=f['files'],
                inputs=data[-1],grids=list(GRIDS),camera_id='58',probes=84,cases=list(CASES),lighting=LIGHT,
                limits=LIMITS,synthetic_controls=control,completed_grids=[],
                counters={k:0 for k in ('shadow_attempts','shadow_returns','brdf_attempts','brdf_returns')},
                native_model_calls=0,native_shader_calls=0,optimizer_returns=0,new_primary_rays=0)
    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        report.update(run_with_cleanup(lambda:render(args,data,report,start,torch),torch))
        report['cleanup_verified']=True
        require(all(digest(ROOT/p)==sha for p,sha in f['files'].items()) and
                all(digest(Path(p['path']))==p['sha256'] for p in data[-1]),'Terminal source/input integrity failed')
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        partial=report.pop('_partial_arrays',None)
        if partial is not None and report['status']!='complete':
            from benchmarks.generate_bell_e11 import write_arrays
            p=args.output/'incomplete-grid.npz';write_arrays(p,**partial)
            report['incomplete_capture']=record(p)
        report['elapsed_seconds']=time.monotonic()-start
        write_json(args.output/'quadrature.json',report)


def render(args,data,report,start,torch):
    from benchmarks.bell_reference_e11 import basis,environment
    from benchmarks.bell_geometry_e11 import mesh_arrays
    from benchmarks.generate_bell_e11 import write_arrays
    from infrastructure.verified_ray_backend import load
    require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0), 'Reviewed GPU runtime required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    report['backend']=load('predicate_fp64',args.image_id)
    report['runtime']={'python':__import__('sys').version,'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(0),'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
    science,_=baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    spec=importlib.util.spec_from_file_location('_e11e_bvh',science/'raytracing/raytracer.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    vertices,faces=mesh_arrays(args.benchmark/'geometry/mesh.ply');tracer=wrapper.RayTracer(vertices,faces)
    labels,wo,pixels,mis,mis_coarse,unused=data;counts=report['counters'];results={}
    def budget(key,increment):
        cap=LIMITS['shadow_directions'] if key=='shadow_attempts' else LIMITS['brdf_point_directions']
        require(counts[key]+increment<=cap and time.monotonic()-start<840,'Scientific work/time cap reached')
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Output cap reached')
        counts[key]+=increment
    with torch.no_grad():
        for size in GRIDS:
            local,domega=midpoint_grid(size);N=len(local)
            packed=np.zeros((84,N//8),dtype=np.uint8);complete=np.zeros(84,dtype=np.bool_)
            array=np.zeros((84,8,3,3),dtype=np.float64)
            report['_partial_arrays']={'components':array,'completed_probes':complete,'pixel_indices':pixels,
                                       'grid_size':np.array(size,dtype=np.int64)}
            direction_hash=hashlib.sha256(local.astype('<f8').tobytes()).hexdigest()
            write_arrays(args.output/('directions-'+str(size)+'.npz'),local_directions=local,
                         solid_angle_weight=np.array(domega),proposal_pdf=np.array(1/(2*np.pi)))
            visible_counts=np.zeros(84,dtype=np.int64);unit_error=horizon_error=0.
            for j in range(0,84,LIMITS['point_batch']):
                end=min(j+4,84);current={k:torch.from_numpy(v[j:end]).cuda() for k,v in labels.items()}
                n=current['normals'];view=torch.from_numpy(wo[j:end]).cuda();t,b=basis(n)
                accumulated=torch.zeros((end-j,8,2,3),dtype=torch.float64,device='cuda')
                for offset in range(0,N,LIMITS['direction_chunk']):
                    stop=min(offset+16384,N);loc=torch.from_numpy(local[offset:stop]).cuda()
                    wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
                    ci=(wi*n[:,None]).sum(-1)
                    ue=float(abs(torch.linalg.norm(wi,dim=-1)-1).max());he=float(abs(ci-loc[None,:,2]).max())
                    require(ue<=2e-12 and he<=2e-12 and bool((ci>0).all()),'World quadrature domain/unit gate failed')
                    unit_error=max(unit_error,ue);horizon_error=max(horizon_error,he)
                    count=(end-j)*(stop-offset);budget('shadow_attempts',count)
                    visible,depth=query_visibility(torch,tracer,current['position'],wi);counts['shadow_returns']+=count
                    raw=visible.cpu().numpy();visible_counts[j:end]+=raw.sum(1)
                    packed[j:end,offset//8:stop//8]=np.packbits(raw,axis=1,bitorder='big')
                    light=environment(wi,LIGHT)
                    for k,case in enumerate(CASES):
                        budget('brdf_attempts',count)
                        fd,fs=kernel_axes(torch,wi,view,counterfactual(torch,current,case))
                        accumulated[:,k]+=weighted_components(torch,fd,fs,light,ci,visible,domega)
                        counts['brdf_returns']+=count
                pair=torch.cat([accumulated,accumulated.sum(2,keepdim=True)],2).cpu().numpy()
                array[j:end]=pair;complete[j:end]=True
                progress=dict(status='incomplete',grid_size=size,completed_probes=int(complete.sum()),counters=counts.copy())
                temp=args.output/'progress.next';temp.write_text(json.dumps(progress,sort_keys=True,indent=2)+'\n');temp.replace(args.output/'progress.json')
            require(complete.all(),'Incomplete fixed probe grid')
            write_arrays(args.output/('grid-'+str(size)+'.npz'),components=array,pixel_indices=pixels,
                         visibility_packed=packed,visible_counts=visible_counts)
            report['completed_grids'].append(dict(size=size,directions_per_point=N,solid_angle_weight=domega,
                pdf=1/(2*np.pi),node_float64_little_endian_sha256=direction_hash,visible_count=int(visible_counts.sum()),
                max_world_unit_error=unit_error,max_world_cosine_error=horizon_error))
            results[size]=array
            print(json.dumps({'grid':size,'completed_probes':84,'counters':counts}),flush=True)
    require(counts['shadow_attempts']==counts['shadow_returns']==LIMITS['shadow_directions'] and
            counts['brdf_attempts']==counts['brdf_returns']==LIMITS['brdf_point_directions'],'Incomplete scientific returns')
    report['comparison']=compare(results[256],results[512],mis,mis_coarse)
    write_json(args.output/'comparison.json',report['comparison'])
    report['outputs']=[record(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ('progress.json','progress.next')]
    for r in report['outputs']:r['path']=str(Path(r['path']).relative_to(args.output))
    require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes']-65536,'Terminal output cap exceeded')
    import resource
    return dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','pilot','output','freeze','authorization'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('commit','image-id','freeze-sha256','authorization-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true',required=True);execute(p.parse_args())


if __name__=='__main__':main()
