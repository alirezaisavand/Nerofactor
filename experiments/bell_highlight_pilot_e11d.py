"""One separately authorized lighting pilot; independent reference, no native shader.

Imports are metadata-only. CUDA/BVH and new BRDF evaluations occur only in execute().
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.bell_highlight_e11d import (ROOT, GENERATION_SHA, PROTOCOL, VIEWS, CASES,
    candidates, subset, measures, linear_to_srgb, write_json)
from infrastructure.signatures import require, read_report
from infrastructure.stage_sources import PINS, baseline, digest
from infrastructure.verified_ray_backend import identity

LIMITS={'wall_seconds':1800,'grace_seconds':30,'cpu_threads':4,'host_memory_gib':64,
        'cuda_allocator_fraction':.25,'output_bytes':512*1024**2,'point_batch':16,
        'shadow_directions':3*128*128*16384,
        'brdf_point_directions':(3*128*128+3*192*7)*16384,
        'light_weighted_point_directions':3*(3*128*128+3*192*7)*16384}


def source_paths():
    old='docs/protocols/BELL_ORACLE_GENERATION_E11_V3.freeze.json'
    return sorted(set(json.loads((ROOT/old).read_text())['files']) | {old,
        'experiments/bell_highlight_e11d.py','experiments/bell_highlight_pilot_e11d.py',
        'experiments/run_bell_highlight_e11d.sh','run_bell_highlight_pilot_e11d.py',
        'tests/benchmarks/test_bell_highlight_e11d.py','docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.md'})


def counterfactual(torch, labels, case):
    n=labels['normals'];theta=labels['theta_major']
    require(bool(torch.isfinite(theta).all()),'Undefined probe orientation: stop, no substitution')
    angle=theta + (np.pi/12 if case=='theta_plus' else -np.pi/12 if case=='theta_minus' else 0)
    axis=torch.cos(angle)[:,None]*labels['T']+torch.sin(angle)[:,None]*labels['B']
    widths=labels['widths_major_minor'].clone()
    if case.startswith('major_'):widths[:,0]*=1.1 if case.endswith('plus') else .9
    if case.startswith('minor_'):widths[:,1]*=1.1 if case.endswith('plus') else .9
    if case=='isotropic':widths=torch.sqrt(widths.prod(-1))[:,None].expand(-1,2)
    require(case in CASES and bool(((widths>=.04)&(widths<=1)).all()),'Invalid fixed counterfactual')
    return dict(n=n,axis=axis,widths=widths,color=labels['color'],metallic=labels['metallic'])


def kernel_axes(torch, wi, wo, material):
    """Cartesian Beckmann with explicit principal axes; additive Kurt a=0, ks=1."""
    n=material['n'];a=material['axis'];b=torch.cross(n,a,dim=-1)
    h=torch.nn.functional.normalize(wi+wo[:,None],dim=-1)
    ci=(wi*n[:,None]).sum(-1);ch=(h*n[:,None]).sum(-1);oh=(h*wo[:,None]).sum(-1)
    valid=(ci>0)&(ch>0)&(oh>0)&((wo*n).sum(-1)[:,None]>0)
    safe_h=torch.where(valid,ch,torch.ones_like(ch));safe_o=torch.where(valid,oh,torch.ones_like(oh))
    mx,my=material['widths'].unbind(-1)
    quad=((h*a[:,None]).sum(-1)/mx[:,None])**2+((h*b[:,None]).sum(-1)/my[:,None])**2
    D=torch.exp(-quad/safe_h**2)/(np.pi*mx[:,None]*my[:,None]*safe_h**4)
    color=material['color'];metal=material['metallic'];f0=.04*(1-metal)+metal*color
    F=f0[:,None]+(1-f0[:,None])*(1-oh.clamp(0,1))[...,None]**5
    fs=F*D[...,None]/(4*safe_o[...,None]);fd=color[:,None]*(1-metal[:,None])/np.pi
    return torch.where(valid[...,None],fd,torch.zeros_like(fs)),torch.where(valid[...,None],fs,torch.zeros_like(fs))


def concentration(values):
    """Per-component weight concentration, not statistical variance of fixed QMC."""
    lum=(values*values.new_tensor([.2126,.7152,.0722])).sum(-1)
    total=lum.sum(1);square=(lum**2).sum(1)
    ess=total**2/square.clamp(min=1e-300)
    maximum=lum.max(1)[0]/total.clamp(min=1e-300)
    return ess.cpu().numpy(),maximum.cpu().numpy()


def sensitivity(fine, coarse):
    """Nx8x3 components xRGB; finite interventions, not global identifiability."""
    require(fine.shape==coarse.shape and fine.ndim==4 and fine.shape[1:]==(8,3,3), 'All eight paired cases required')
    total=fine[:,:,2];low=coarse[:,:,2]
    require(np.isfinite(fine).all() and np.isfinite(coarse).all() and (fine>=0).all(), 'Invalid paired radiance')
    if (total>1).any():return {'png_domain':False,'reason':'No clipping/rescaling; candidate cannot be admitted'}
    codes=linear_to_srgb(total).astype(int);result={'png_domain':True,'probes':len(fine)}
    for name,ids in [('orientation',(1,2)),('major',(3,4)),('minor',(5,6)),('isotropic',(7,))]:
        change=np.abs(total[:,ids]-total[:,0,None]).max((1,2))
        grid=np.maximum(np.abs(total[:,ids]-low[:,ids]).max((1,2)),np.abs(total[:,0]-low[:,0]).max(1))
        code_change=np.abs(codes[:,ids]-codes[:,0,None]).max((1,2))
        resolved=(change>=.003)&(change>=4*grid)&(code_change>=2)
        result[name]={'median_max_channel_response':float(np.median(change)),
                      'two_code_fraction':float((code_change>=2).mean()),
                      'resolved_fraction':float(resolved.mean()),'resolved_count':int(resolved.sum()),
                      'median_grid_difference':float(np.median(grid))}
    J=np.stack([(total[:,a]-total[:,b])/2 for a,b in [(1,2),(3,4),(5,6)]],-1)
    sv=np.linalg.svd(J,compute_uv=False)
    result['conditional_three_column_jacobian']={
        'column_units':'half response of fixed +/-15deg orientation, +/-10percent principal widths',
        'median_singular_values':np.median(sv,axis=0).tolist(),
        'full_numerical_rank_fraction':float((sv[:,-1]>np.maximum(sv[:,0]*1e-6,1e-8)).mean()),
        'meaning':'one-view RGB; color/metal fixed; not seven-parameter or multi-view identifiability'}
    return result


def recommend(rows):
    require(len(rows)==9 and {(r['camera_id'],r['lighting']) for r in rows}==
            {(v,l) for v in VIEWS for l in candidates()},'Complete fixed pilot panel required')
    control={r['camera_id']:r for r in rows if r['lighting']=='L0_v3'}
    if not all(r['eligible'] for r in control.values()):
        return {'decision':'FIX_AND_PROCEED','recommended':None,'reason':'v3 counterfactual control not eligible; preserve failed reference checks'}
    candidates_ranked=[];assessments=[]
    for order,name in enumerate(('L1_three_oblique','L2_six_cardinal')):
        panel=[r for r in rows if r['lighting']==name]
        valid=all(r['eligible'] for r in panel)
        signals=valid and all(r['sensitivity'][k]['resolved_fraction']>=.25 for r in panel
                             for k in ('orientation','major','minor','isotropic'))
        nondecreasing=valid and all(r['sensitivity']['orientation']['resolved_fraction']>=
                                    control[r['camera_id']]['sensitivity']['orientation']['resolved_fraction'] for r in panel)
        weights=[r['sensitivity'].get('probes',0) for r in panel]
        gain=(sum(w*(r['sensitivity']['orientation']['resolved_fraction']-
                    control[r['camera_id']]['sensitivity']['orientation']['resolved_fraction'])
                  for w,r in zip(weights,panel))/max(sum(weights),1)) if valid else None
        coverage=sum(r['metrics']['highlight_coverage']>=.05 for r in panel)>=2
        passed=bool(valid and signals and nondecreasing and gain>=.10 and coverage)
        assessments.append(dict(lighting=name,passed=passed,references_valid=valid,
                                all_sensitivity_floors=signals,nondecreasing_orientation=nondecreasing,
                                weighted_orientation_gain=gain,highlight_coverage_gate=coverage))
        if passed:
            rank=(min(r['sensitivity']['orientation']['resolved_fraction'] for r in panel),
                  np.mean([r['sensitivity'][k]['resolved_fraction'] for r in panel for k in ('major','minor')]),
                  -max([r['convergence']['relative_l2'] for r in panel]+
                       [g['relative_l2'] for r in panel for g in r['paired_convergence'].values()]),-order)
            candidates_ranked.append((rank,name))
    selected=max(candidates_ranked)[1] if candidates_ranked else None
    return dict(decision='PROCEED_TO_SEPARATE_V4_REVIEW' if selected else 'FIX_AND_PROCEED',
                recommended=selected,candidates=assessments,full_dataset_or_training_authorized=False)


def authorize(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['limits']==LIMITS and f['lighting']==candidates()
            and f['views']==list(VIEWS) and f['cases']==list(CASES)
            and f['generation_sha256']==GENERATION_SHA and f['image_id']==identity()['image_id']
            and set(f['files'])==set(source_paths()),'Pilot definition/source closure changed')
    for name,sha in f['files'].items():require(digest(ROOT/name)==sha,'Frozen pilot source changed: '+name)
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL
            and a.get('commit')==args.commit and a.get('freeze_sha256')==args.freeze_sha256
            and a.get('image_id')==args.image_id==identity()['image_id']
            and a.get('output_directory')==str(args.output) and a.get('limits')==LIMITS,
            'Separate exact pilot authorization required')
    require(digest(args.root/'geometry/generation.json')==GENERATION_SHA,'Changed v3 source evidence')
    require(args.output.is_absolute() and args.output.resolve()==args.output and args.output.parent==Path('/outputs')
            and not args.output.exists(),'Fresh /outputs child required')
    require(os.geteuid()!=0,'Non-root runtime required')
    for p in (ROOT,args.root,args.freeze,args.authorization,Path('/')):
        require(os.statvfs(p).f_flag & os.ST_RDONLY,'Source, benchmark, authorization and root must be read-only')
    require(not os.statvfs('/outputs').f_flag & os.ST_RDONLY,'Separate writable output mount required')
    require({v.split(':')[0].strip() for v in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')
    return f


def execute(args):
    frozen=authorize(args)
    from benchmarks.verify_bell_e11 import verify
    verify(args.root,GENERATION_SHA)  # File-only integrity before any new query.
    args.output.mkdir();start=time.monotonic()
    report=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,
                commit=args.commit,freeze_sha256=args.freeze_sha256,source_files=frozen['files'],
                authorization_sha256=args.authorization_sha256,
                generation_sha256=GENERATION_SHA,limits=LIMITS,lighting=candidates(),views=[],
                counters={k:0 for k in ('shadow_directions','valid_shadow_queries','valid_shadow_returns',
                                      'brdf_point_directions','light_weighted_point_directions',
                                      'brdf_point_direction_returns','light_weighted_point_direction_returns')},
                native_model_calls=0,native_shader_calls=0,optimizer_returns=0)
    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        report.update(run_with_cleanup(lambda:render(args,report,start,torch),torch))
        report['cleanup_verified']=True
        report['recommendation']=recommend(report['views'])
        authorize_after={name:digest(ROOT/name) for name in frozen['files']}
        require(authorize_after==frozen['files'],'Source changed during pilot')
        for p,sha in [(args.root/'geometry/generation.json',GENERATION_SHA)]:require(digest(p)==sha,'Input identity changed')
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        partial=report.pop('_partial_arrays',None);camera=report.pop('_partial_camera',None)
        if partial is not None and report['status']!='complete':
            from benchmarks.generate_bell_e11 import write_arrays
            path=args.output/('incomplete-'+camera+'.npz');write_arrays(path,**partial)
            report['incomplete_capture']={'camera_id':camera,'path':path.name,'sha256':digest(path),
                                          'meaning':'Zero slots are uncompleted; use completed_primary_pixels, not pixel values'}
        report['elapsed_seconds']=time.monotonic()-start
        write_json(args.output/'pilot.json',report)


def render(args,report,start,torch):
    from benchmarks.bell_reference_e11 import samples,environment
    from benchmarks.bell_geometry_e11 import mesh_arrays
    from benchmarks.bell_e11 import convergence
    from benchmarks.generate_bell_e11 import write_arrays
    from infrastructure.verified_ray_backend import load
    require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed runtime required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    report['backend']=load('predicate_fp64',args.image_id)
    report['runtime']={'python':__import__('sys').version,'torch':torch.__version__,
                       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                       'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
    science,_=baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    spec=importlib.util.spec_from_file_location('_e11d_bvh',science/'raytracing/raytracer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    v,f=mesh_arrays(args.root/'geometry/mesh.ply');tracer=module.RayTracer(v,f)
    with np.load(ROOT/'benchmarks/inputs/bell_reference_tapes_e11_v1.npz',allow_pickle=False) as z:
        td,th=[torch.from_numpy(z[k]).cuda() for k in ('diffuse','half')]
    count=report['counters'];light=candidates()
    expected_shadow=expected_brdf=0
    def budget(key,increment):
        require(time.monotonic()-start<1740,'Pilot active time cap reached')
        require(count[key]+increment<=LIMITS[key],'Pilot work cap reached: '+key)
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Output cap reached')
        count[key]+=increment
    with torch.no_grad():
        for camera in VIEWS:
            with np.load(args.root/'references'/(camera+'.npz'),allow_pickle=False) as z:
                hit=z['hit'].copy();original=z['linear_components'].copy();original_low=z['coarse_components'].copy()
            chosen=subset(hit);ids=np.flatnonzero(hit);choice={int(v):j for j,v in enumerate(chosen)}
            expected_shadow+=len(ids)*16384;expected_brdf+=(len(ids)+7*len(chosen))*16384
            fine={k:np.zeros((128*128,3,3)) for k in light};low={k:a.copy() for k,a in fine.items()}
            pairs={k:np.zeros((len(chosen),8,3,3)) for k in light};pairs_low={k:a.copy() for k,a in pairs.items()}
            ess={k:np.zeros((128*128,2)) for k in light};max_mass={k:a.copy() for k,a in ess.items()}
            completed=np.zeros(128*128,dtype=np.bool_)
            report['_partial_camera']=camera
            report['_partial_arrays']={'completed_primary_pixels':completed,'probe_pixel_indices':chosen}
            for name in light:
                report['_partial_arrays'].update({name+'_fine':fine[name],name+'_coarse':low[name],
                                                   name+'_paired':pairs[name],name+'_paired_coarse':pairs_low[name]})
            with np.load(args.root/'geometry/labels'/(camera+'.npz'),allow_pickle=False) as z:
                labels={k:z[k].copy() for k in ('normals','theta_major','T','B','widths_major_minor','color','metallic','position')}
            with np.load(args.root/'geometry/rays'/(camera+'.npz'),allow_pickle=False) as z:
                wo=-z['directions'][ids].astype(float);wo/=np.linalg.norm(wo,axis=1,keepdims=True)
            for j in range(0,len(ids),16):
                end=min(j+16,len(ids));data={k:torch.from_numpy(a[j:end]).cuda() for k,a in labels.items()}
                n=data['normals'];view=torch.from_numpy(wo[j:end]).cuda();wi,pdf,valid=samples(n,view,td,th)
                budget('shadow_directions',wi.shape[0]*wi.shape[1]);count['valid_shadow_queries']+=int(valid.sum())
                wi32=wi.float();origin=data['position'].float()[:,None]+1e-5*wi32
                _,_,depth=tracer.trace(origin[valid],wi32[valid]);require(bool(torch.isfinite(depth).all()),'Invalid shadow depth')
                count['valid_shadow_returns']+=int(valid.sum());visible=torch.zeros_like(pdf);visible[valid]=(depth>=10).to(pdf.dtype)
                weight=(wi*n[:,None]).sum(-1).clamp(min=0)/torch.where(pdf>0,pdf,torch.ones_like(pdf))*visible
                local=[k for k,pixel in enumerate(ids[j:end]) if int(pixel) in choice]
                for case_index,case in enumerate(CASES):
                    select=list(range(end-j)) if case=='truth' else local
                    if not select:continue
                    current={k:a[select] for k,a in data.items()}
                    total_dirs=len(select)*16384;budget('brdf_point_directions',total_dirs)
                    fd,fs=kernel_axes(torch,wi[select],view[select],counterfactual(torch,current,case))
                    count['brdf_point_direction_returns']+=total_dirs
                    for name,definition in light.items():
                        budget('light_weighted_point_directions',total_dirs)
                        L=environment(wi[select],definition);values=torch.stack([fd*L*weight[select,...,None],fs*L*weight[select,...,None]],2)
                        coarse_idx=torch.cat([torch.arange(2048,device='cuda'),torch.arange(8192,10240,device='cuda')])
                        hi=values.mean(1);lo=values[:,coarse_idx].mean(1)
                        if case=='truth':effective,mass=concentration(values)
                        hi=torch.cat([hi,hi.sum(1,keepdim=True)],1).cpu().numpy();lo=torch.cat([lo,lo.sum(1,keepdim=True)],1).cpu().numpy()
                        require(np.isfinite(hi).all() and np.isfinite(lo).all() and (hi>=0).all(),'Nonfinite/negative pilot radiance')
                        count['light_weighted_point_direction_returns']+=total_dirs
                        for k,index in enumerate(select):
                            pixel=int(ids[j+index])
                            if case=='truth':
                                fine[name][pixel]=hi[k];low[name][pixel]=lo[k]
                                ess[name][pixel]=effective[k];max_mass[name][pixel]=mass[k]
                            if pixel in choice:pairs[name][choice[pixel],case_index]=hi[k];pairs_low[name][choice[pixel],case_index]=lo[k]
                completed[ids[j:end]]=True
            for name in light:
                image=fine[name].reshape(128,128,3,3);coarse=low[name].reshape(128,128,3,3)
                write_arrays(args.output/(camera+'-'+name+'.npz'),linear_components=image,coarse_components=coarse,
                             hit=hit,pixel_indices=chosen,paired_components=pairs[name],paired_coarse_components=pairs_low[name],
                             component_effective_samples=ess[name].reshape(128,128,2),
                             component_max_sample_fraction=max_mass[name].reshape(128,128,2))
                if name=='L0_v3':require(max(abs(image-original).max(),abs(coarse-original_low).max())<=2e-10,'v3 independent truth replay mismatch')
                stats,_=measures(image[hit]);conv=convergence(image[hit,2],coarse[hit,2])
                paired_gates={case:convergence(pairs[name][:,k,2],pairs_low[name][:,k,2]) for k,case in enumerate(CASES)}
                info=sensitivity(pairs[name],pairs_low[name]);eligible=conv['passed'] and all(g['passed'] for g in paired_gates.values()) and not stats['out_of_png_domain'] and info['png_domain']
                row=dict(camera_id=camera,lighting=name,metrics=stats,convergence=conv,paired_convergence=paired_gates,sensitivity=info,eligible=eligible)
                row['specular_weight_concentration']={'minimum_effective_samples':float(ess[name][ids,1].min()),
                    'median_effective_samples':float(np.median(ess[name][ids,1])),
                    'maximum_sample_fraction':float(max_mass[name][ids,1].max()),
                    'meaning':'Fixed QMC contribution concentration; not an independent variance estimate'}
                report['views'].append(row);write_json(args.output/(camera+'-'+name+'.json'),row)
                if not stats['out_of_png_domain']:
                    from PIL import Image
                    Image.fromarray(linear_to_srgb(image[:,:,2])).save(args.output/(camera+'-'+name+'.png'))
            print(json.dumps({'camera':camera,'completed':True,'counters':count}),flush=True)
    require(count['valid_shadow_queries']==count['valid_shadow_returns'] and len(report['views'])==9
            and count['brdf_point_directions']==count['brdf_point_direction_returns']
            and count['light_weighted_point_directions']==count['light_weighted_point_direction_returns']
            and count['shadow_directions']==expected_shadow and count['brdf_point_directions']==expected_brdf
            and count['light_weighted_point_directions']==3*expected_brdf, 'Incomplete pilot work')
    report['outputs']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(args.output.iterdir()) if p.is_file()]
    import resource
    return dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','freeze','authorization'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('freeze-sha256','authorization-sha256','commit','image-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true',required=True);execute(p.parse_args())


if __name__=='__main__':main()
