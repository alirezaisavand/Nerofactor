"""E11F material-only factor; seven new cases, cached nodes/visibility, no rays.

No native material model, ray-tracing extension or training code is imported.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments.bell_quadrature_e11e import (ROOT,GRIDS,CASES,LIGHT,inputs as base_inputs,
    weighted_components,compare as compare_components,record)
from experiments.bell_highlight_e11d import write_json
from experiments.bell_highlight_pilot_e11d import kernel_axes,counterfactual
from experiments.analyze_bell_quadrature_e11e import verify_grid,verify_nodes
from experiments.analyze_bell_highlight_e11d import response_fields
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import PINS,baseline,digest
from infrastructure.verified_ray_backend import identity

PROTOCOL='bell-stronger-anisotropy-e11f-v1'
REFERENCE_SHA='589f894d653b715a55739aad2e7667ce0105b14eb0119bb5b54e69010f7cc383'
NEW_CASES=CASES[:7]
LIMITS={'wall_seconds':900,'grace_seconds':30,'cpu_threads':4,'host_memory_gib':64,
        'cuda_allocator_fraction':.25,'output_bytes':256*1024**2,'point_batch':4,
        'direction_chunk':16384,'new_shadow_queries':0,'new_primary_rays':0,
        'environment_point_directions':27525120,'brdf_point_directions':192675840}


def source_paths():
    old='docs/protocols/BELL_QUADRATURE_E11E_V1.freeze.json'
    return sorted(set(json.loads((ROOT/old).read_text())['files']) | {old,
        'docs/BELL_QUADRATURE_E11E_RESULTS_V1.md','docs/evidence/BELL_QUADRATURE_E11E_RESULTS_V1.json',
        'experiments/analyze_bell_quadrature_e11e.py','experiments/analyze_bell_highlight_e11d.py',
        'experiments/bell_anisotropy_e11f.py','experiments/run_bell_anisotropy_e11f.sh',
        'run_bell_anisotropy_e11f.py','tests/benchmarks/test_bell_anisotropy_e11f.py',
        'docs/protocols/BELL_ANISOTROPY_E11F_V1.md'})


def strengthen(torch,labels):
    w=labels['widths_major_minor'];require(w.ndim==2 and w.shape[1]==2 and w.dtype==torch.float64 and bool(torch.isfinite(w).all())
        and bool((w[:,0]>=w[:,1]).all()) and bool((w>0).all()),'Finite ordered float64 widths required')
    ratio=w[:,0]/w[:,1];scale=torch.sqrt(ratio)
    changed=torch.stack([w[:,0]*scale,w[:,1]/scale],-1)
    require(bool(((changed>=.04)&(changed<=1)).all()),'Strong widths outside unchanged material domain')
    result=dict(labels);result['widths_major_minor']=changed
    return result


def admit_isotropic(torch,original,strong):
    """Exact algebra, explicit canonical mean, disclosed floating-point residue.

    g=sqrt(original mx*my) is the experiment's invariant canonical mean. The
    naive product of rounded transformed widths can differ by a few ULPs; it
    must not be silently presented as bit-identical or trigger a new rendering.
    """
    w=original['widths_major_minor'];s=strong['widths_major_minor']
    old=counterfactual(torch,original,'isotropic');naive=counterfactual(torch,strong,'isotropic')
    eps=16*np.finfo(np.float64).eps
    require(torch.allclose(w.prod(-1),s.prod(-1),rtol=eps,atol=0),'Geometric-mean invariant failed')
    require(torch.allclose(old['widths'],naive['widths'],rtol=eps,atol=0),'Isotropic mathematical equivalence failed')
    require(all(torch.equal(original[k],strong[k]) for k in original if k!='widths_major_minor') and
            all(torch.equal(old[k],naive[k]) for k in ('n','axis','color','metallic')),'Non-width scientific input changed')
    return dict(algebra='(mx*sqrt(mx/my))*(my/sqrt(mx/my)) = mx*my for positive widths',
        canonical_mean='sqrt(original mx*my); original isotropic reference is reused by this exact definition',
        mathematical_equivalence=True,naive_rounded_product_max_abs=float(abs(w.prod(-1)-s.prod(-1)).max()),
        naive_isotropic_width_max_abs=float(abs(old['widths']-naive['widths']).max()),
        relative_roundoff_gate=eps,naive_recomputed_widths_claimed_bit_identical=False,
        canonical_mean_sha256=__import__('hashlib').sha256(old['widths'].cpu().numpy().astype('<f8').tobytes()).hexdigest(),
        reused_case='E11E components[:,7], exact stored bytes; no new isotropic BRDF calls')


def inputs(benchmark,pilot,reference):
    require(reference.is_absolute() and reference.resolve()==reference,'Explicit reference root required')
    labels,wo,pixels,mis,mis_coarse,records=base_inputs(benchmark,pilot)
    require(digest(reference/'quadrature.json')==REFERENCE_SHA,'Accepted E11E identity mismatch')
    r=json.loads((reference/'quadrature.json').read_text())
    require(r['status']=='complete' and r['cleanup_verified'] is True and
        r['comparison']['all_cases_two_grid_converged'] is True and r['backend']==identity() and
        r['comparison']['E11D_admission_changed'] is False and r['lighting']==LIGHT,'Accepted cache provenance incomplete')
    records.append(record(reference/'quadrature.json'))
    for item in r['outputs']:
        p=reference/item['path'];require(p.resolve()==p and reference in p.parents and
            digest(p)==item['sha256'] and p.stat().st_size==item['bytes'],'E11E artifact changed')
        records.append(record(p))
    cache={}
    for size in GRIDS:
        native=next(v for v in r['completed_grids'] if v['size']==size)
        with np.load(reference/('directions-'+str(size)+'.npz'),allow_pickle=False) as z:
            node={k:z[k].copy() for k in z.files};verify_nodes(size,node,native)
        with np.load(reference/('grid-'+str(size)+'.npz'),allow_pickle=False) as z:
            saved={k:z[k].copy() for k in z.files};verify_grid(size,saved,pixels,native)
        cache[size]=dict(nodes=node['local_directions'],weight=float(node['solid_angle_weight']),
                         visibility=saved['visibility_packed'],original=saved['components'],visible_counts=saved['visible_counts'])
    return labels,wo,pixels,cache,records


def coupling(components,labels,visible_counts,N):
    rgb=components[:,:,2]
    J=np.stack([(rgb[:,a]-rgb[:,b])/2 for a,b in [(1,2),(3,4),(5,6)]],-1)
    singular=np.linalg.svd(J,compute_uv=False);norms=np.linalg.norm(J,axis=1)
    with np.errstate(divide='ignore',over='ignore'):
        condition=np.divide(singular[:,0],singular[:,-1],out=np.full(len(J),np.inf),where=singular[:,-1]>0)
    finite=np.isfinite(condition)
    pairs={}
    for a,b,name in [(0,1,'orientation_major'),(0,2,'orientation_minor'),(1,2,'major_minor')]:
        denom=norms[:,a]*norms[:,b];valid=denom>0
        cosine=(J[:,:,a]*J[:,:,b]).sum(1)[valid]/denom[valid]
        pairs[name]=dict(nonzero_pairs=int(valid.sum()),median_absolute_column_cosine=float(np.median(abs(cosine))) if valid.any() else None)
    diffuse=components[:,0,0];allocation=labels['color']*(1-labels['metallic'])
    require((allocation>0).all(),'Positive known diffuse allocation required for irradiance diagnostic')
    irradiance=np.pi*diffuse/allocation
    return dict(median_singular_values=np.median(singular,axis=0).tolist(),
        median_finite_condition_number=float(np.median(condition[finite])) if finite.any() else None,
        p95_finite_condition_number=float(np.quantile(condition[finite],.95)) if finite.any() else None,
        nonfinite_condition_count=int((~finite).sum()),
        median_scaled_column_norms=np.median(norms,axis=0).tolist(),column_alignment=pairs,
        visible_fraction_range=[float(visible_counts.min()/N),float(visible_counts.max()/N)],
        irradiance_RGB_min_median_max=[irradiance.min(0).tolist(),np.median(irradiance,axis=0).tolist(),irradiance.max(0).tolist()],
        interpretation='Conditional one-view RGB at fixed color/metal. Low irradiance/visibility is compatible with coverage limitations; aligned nonzero columns support coupling. Positive diffuse irradiance does not prove useful angular coverage or full identifiability.')


def spatial_coverage(fine,coarse,pixels):
    require(pixels.shape==(84,) and pixels.dtype==np.int64 and ((pixels>=0)&(pixels<128*128)).all(),'Original84 pixel IDs required')
    fields=response_fields(fine,coarse);cells=(pixels//128//16)*8+(pixels%128//16)
    result={'sampled_cells':int(len(np.unique(cells)))};joint=np.ones(84,dtype=np.bool_)
    for name,v in fields.items():
        ok=v['resolved'];result[name]={'resolved_probes':int(ok.sum()),'resolved_cells':int(len(np.unique(cells[ok]))),
                                     'cell_ids':np.unique(cells[ok]).tolist()}
        if name in ('orientation','major','minor'):joint&=ok
    result['joint_orientation_major_minor']={'resolved_probes':int(joint.sum()),'resolved_cells':int(len(np.unique(cells[joint])))}
    return result


def summarize(coarse,fine,original_coarse,original_fine,labels,visible_counts,pixels):
    result=compare_components(coarse,fine,original_fine,original_coarse)
    result['comparison_baseline']='Accepted E11E quadrature; original materials are reused, never re-evaluated.'
    names={'fine_quadrature_minus_coarse':'stronger512_minus_stronger256',
           'fine_quadrature_minus_MIS_fine':'stronger512_minus_original512',
           'coarse_quadrature_minus_MIS_fine':'stronger256_minus_original512',
           'fine_quadrature_minus_MIS_coarse':'stronger512_minus_original256'}
    result['comparison']={case:{names[k]:v for k,v in values.items()} for case,values in result['comparison'].items()}
    for values in result['component_comparison'].values():
        for component in values.values():component['stronger512_minus_original512']=component.pop('fine_minus_MIS_fine')
    result['interpretation']='Fixed two-grid, one-camera material-only comparison; not absolute accuracy, full-scene coverage or seven-parameter identifiability.'
    result['stronger_anisotropy']=result.pop('sensitivity');result['original_material']=result.pop('preserved_MIS_sensitivity')
    result['coupling_diagnostics']={'original':coupling(original_fine,labels,visible_counts,512**2),
                                    'stronger':coupling(fine,labels,visible_counts,512**2)}
    result['spatial_coverage']={'original':spatial_coverage(original_fine,original_coarse,pixels),
                               'stronger':spatial_coverage(fine,coarse,pixels)}
    q=result['stronger_anisotropy'];base=result['original_material']
    usable=result['all_cases_two_grid_converged'] and q['png_domain']
    floor=bool(usable and all(q[k]['resolved_fraction']>=.25 for k in ('orientation','major','minor','isotropic')))
    gain=q['orientation']['resolved_fraction']-base['orientation']['resolved_fraction'] if usable else None
    result['single_panel_screen']={'references_usable':usable,'all_original_25percent_floors':floor,
        'orientation_gain':gain,'original_10percentagepoint_orientation_gain_marker':bool(usable and gain>=.10),
        'meaning':'Descriptive inherited screen on one camera, not a full-scene or three-view admission.',
        'full_dataset_authorized':False}
    return result


def authorize(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['grids']==list(GRIDS) and f['new_cases']==list(NEW_CASES)
        and f['limits']==LIMITS and f['lighting']==LIGHT and f['reference_sha256']==REFERENCE_SHA
        and set(f['files'])==set(source_paths()),'Frozen material-only definition changed')
    for p,sha in f['files'].items():require(digest(ROOT/p)==sha,'Frozen source changed')
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==args.commit
        and a.get('freeze_sha256')==args.freeze_sha256 and a.get('image_id')==args.image_id==identity()['image_id']
        and a.get('output_directory')==str(args.output) and a.get('limits')==LIMITS,'Separate exact E11F approval required')
    require(os.geteuid()!=0 and args.output.is_absolute() and args.output.resolve()==args.output and
        args.output.parent==Path('/outputs') and not args.output.exists(),'Fresh output child/non-root required')
    for p in (Path('/'),ROOT,args.benchmark,args.pilot,args.reference,args.freeze,args.authorization):
        require(os.statvfs(p).f_flag & os.ST_RDONLY,'Originals/source/root must be read-only')
    require(not os.statvfs('/outputs').f_flag & os.ST_RDONLY,'Writable separate output mount required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')
    return f


def execute(args):
    f=authorize(args);data=inputs(args.benchmark,args.pilot,args.reference);args.output.mkdir();start=time.monotonic()
    report=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,commit=args.commit,
        freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,source_files=f['files'],inputs=data[-1],
        reference_sha256=REFERENCE_SHA,camera_id='58',probes=84,grids=list(GRIDS),new_cases=list(NEW_CASES),
        reused_case='isotropic',lighting=LIGHT,limits=LIMITS,
        counters=dict(brdf_attempts=0,brdf_returns=0,environment_attempts=0,environment_returns=0,
                      new_shadow_queries=0,new_primary_rays=0,original_BRDF_queries=0,new_isotropic_BRDF_queries=0),
        native_model_calls=0,native_shader_calls=0,optimizer_returns=0,completed_grids=[])
    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        report.update(run_with_cleanup(lambda:render(args,data,report,start,torch),torch));report['cleanup_verified']=True
        require(all(digest(ROOT/p)==sha for p,sha in f['files'].items()) and
            all(digest(Path(r['path']))==r['sha256'] for r in data[-1]),'Terminal input/source integrity failed')
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        partial=report.pop('_partial_arrays',None)
        if partial is not None and report['status']!='complete':
            from benchmarks.generate_bell_e11 import write_arrays
            p=args.output/'incomplete-grid.npz';write_arrays(p,**partial);report['incomplete_capture']=record(p)
        report['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'anisotropy.json',report)


def render(args,data,report,start,torch):
    from benchmarks.bell_reference_e11 import basis,environment
    from benchmarks.generate_bell_e11 import write_arrays
    require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed GPU runtime required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    for stage in PINS:baseline(stage,PINS[stage])
    report['runtime']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(0),'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI)}
    report['cached_tracing_identity']=identity();labels,wo,pixels,cache,unused=data
    all_original={k:torch.from_numpy(v).cuda() for k,v in labels.items()};all_strong=strengthen(torch,all_original)
    report['isotropic_equivalence']=admit_isotropic(torch,all_original,all_strong)
    write_arrays(args.output/'material-factor.npz',pixel_indices=pixels,original_widths=labels['widths_major_minor'],
                 stronger_widths=all_strong['widths_major_minor'].cpu().numpy(),canonical_mean=torch.sqrt(all_original['widths_major_minor'].prod(-1)).cpu().numpy())
    counts=report['counters'];results={}
    def budget(kind,n):
        cap=LIMITS['brdf_point_directions'] if kind=='brdf_attempts' else LIMITS['environment_point_directions']
        require(counts[kind]+n<=cap and time.monotonic()-start<840,'Fixed work/time limit reached')
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Output limit exceeded')
        counts[kind]+=n
    with torch.no_grad():
        for size in GRIDS:
            c=cache[size];local=c['nodes'];N=len(local);array=np.zeros((84,8,3,3));array[:,7]=c['original'][:,7]
            complete=np.zeros(84,dtype=np.bool_);report['_partial_arrays']={'components':array,'completed_new_case_probes':complete,
                'pixel_indices':pixels,'grid_size':np.array(size),'reused_case_mask':np.array([False]*7+[True])}
            for j in range(0,84,4):
                end=min(j+4,84);current={k:v[j:end] for k,v in all_strong.items()};n=current['normals'];view=torch.from_numpy(wo[j:end]).cuda()
                t,b=basis(n);accumulated=torch.zeros((end-j,7,2,3),dtype=torch.float64,device='cuda')
                for offset in range(0,N,16384):
                    stop=min(offset+16384,N);loc=torch.from_numpy(local[offset:stop]).cuda()
                    wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
                    ci=(wi*n[:,None]).sum(-1);require(bool((ci>0).all()) and float(abs(torch.linalg.norm(wi,dim=-1)-1).max())<=2e-12
                        and float(abs(ci-loc[None,:,2]).max())<=2e-12,'Cached world direction invariant failed')
                    packed=c['visibility'][j:end,offset//8:stop//8]
                    visible=torch.from_numpy(np.unpackbits(packed,axis=1,bitorder='big').astype(np.bool_)).cuda()
                    count=(end-j)*(stop-offset);budget('environment_attempts',count)
                    light=environment(wi,LIGHT);require(bool(torch.isfinite(light).all()) and bool((light>=0).all()),'Invalid unchanged illumination')
                    counts['environment_returns']+=count
                    for k,case in enumerate(NEW_CASES):
                        budget('brdf_attempts',count);fd,fs=kernel_axes(torch,wi,view,counterfactual(torch,current,case))
                        accumulated[:,k]+=weighted_components(torch,fd,fs,light,ci,visible,c['weight']);counts['brdf_returns']+=count
                pair=torch.cat([accumulated,accumulated.sum(2,keepdim=True)],2).cpu().numpy();array[j:end,:7]=pair;complete[j:end]=True
                temp=args.output/'progress.next';temp.write_text(json.dumps({'status':'incomplete','grid_size':size,'completed_new_case_probes':int(complete.sum()),'counters':counts.copy()},indent=2)+'\n');temp.replace(args.output/'progress.json')
            diffuse_error=float(abs(array[:,:7,0]-c['original'][:,:7,0]).max())
            require(complete.all() and diffuse_error<=2e-10 and np.array_equal(array[:,7],c['original'][:,7]),'Unchanged diffuse/cached isotropic invariant failed')
            write_arrays(args.output/('grid-'+str(size)+'.npz'),components=array,pixel_indices=pixels,reused_case_mask=np.array([False]*7+[True]))
            report['completed_grids'].append(dict(size=size,completed_new_probes=84,diffuse_control_max_abs=diffuse_error,reused_isotropic_exact=True))
            results[size]=array;print(json.dumps({'grid':size,'completed':True,'counters':counts}),flush=True)
    require(counts['brdf_attempts']==counts['brdf_returns']==192675840 and
        counts['environment_attempts']==counts['environment_returns']==27525120 and
        all(counts[k]==0 for k in ('new_shadow_queries','new_primary_rays','original_BRDF_queries','new_isotropic_BRDF_queries')),'Fixed work/zero-query boundary failed')
    require(not any(k.split('.')[0] in ('raytracing','_raytracing','network','train') for k in sys.modules),'Forbidden native model/tracing import')
    report['comparison']=summarize(results[256],results[512],cache[256]['original'],cache[512]['original'],labels,cache[512]['visible_counts'],pixels)
    write_json(args.output/'comparison.json',report['comparison'])
    report['outputs']=[record(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ('progress.json','progress.next')]
    for r in report['outputs']:r['path']=str(Path(r['path']).relative_to(args.output))
    import resource
    return dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','pilot','reference','output','freeze','authorization'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('commit','image-id','freeze-sha256','authorization-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true',required=True);execute(p.parse_args())


if __name__=='__main__':main()
