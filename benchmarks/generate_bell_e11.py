"""One guarded, bounded Bell generation. Import is metadata-only, CUDA is opt-in."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import traceback
import numpy as np
from infrastructure.signatures import require, read_report
from infrastructure.stage_sources import ROOT, digest, baseline, PINS
from infrastructure.stage_workspace import source_records
from audits.bell_calibration_e11 import rays
from benchmarks.bell_e11 import (PROTOCOL,IMAGE,MESH,LAW,ENV,RELIGHT,LIMITS,TOL,INPUT,TAPES,
    admit_inputs,split,record,linear_to_srgb,convergence,geometry_gate,verify_authorization,verify_labels,source_paths)
from benchmarks.bell_geometry_e11 import mesh_arrays,intersect


def write_json(path,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    # New owned files only; terminal manifest must never overwrite an earlier result.
    with Path(path).open('x') as f:f.write(data)


def write_arrays(path,**arrays):
    require(all(a.dtype.kind in 'biuf' for a in arrays.values()),'Primitive numeric arrays only')
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)


def frozen(path,expected):
    f,_=read_report(path,expected)
    require(f.get('protocol')==PROTOCOL and f.get('work_limits')==LIMITS and f.get('tolerances')==TOL,
            'Frozen generation protocol mismatch')
    require(type(f.get('files')) is dict and set(f['files'])==set(source_paths()),'Incomplete source/configuration freeze')
    for name,sha in f['files'].items():
        p=ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==sha,'Changed generation source/input: '+name)
    return f


def runtime(args):
    require(args.image_id==IMAGE and os.geteuid()!=0,'Reviewed image/non-root required')
    for p in [Path('/'),ROOT,args.mesh,args.cameras,args.authorization,args.freeze]:
        require(os.statvfs(str(p)).f_flag & os.ST_RDONLY,'Read-only root, source, inputs and authorization required')
    require({line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'No-network runtime required')
    # This repository is a linked worktree: its .git administration path is not
    # mounted into Docker. The foreground host launcher attests clean HEAD;
    # authorization binds that commit, and frozen bytes are rechecked here.
    require(isinstance(args.commit,str) and len(args.commit)==40 and all(c in '0123456789abcdef' for c in args.commit),
            'Explicit host-attested infrastructure commit required')
    require(args.output.is_absolute() and args.output.resolve()==args.output and not args.output.exists(),
            'Fresh explicit unaliased output directory required')
    require(args.output.parent==Path('/outputs') and not os.statvfs('/outputs').f_flag & os.ST_RDONLY,
            'Output must be a new child of the separately mounted writable /outputs')
    require(shutil.disk_usage('/outputs').free>=8*1024**3,'At least 8 GiB free output space required')


def primary(torch,P,K):
    """Original A CPU float32 pinhole arithmetic; independently checked via NumPy."""
    y,x=torch.meshgrid(torch.arange(128),torch.arange(128))
    pixels=torch.stack([x.float()+.5,y.float()+.5,torch.ones_like(x).float()],-1).reshape(1,-1,3)
    P=torch.from_numpy(P[None]);K=torch.from_numpy(K[None]);R=P[:,:,:3];t=P[:,:,3:]
    d=torch.nn.functional.normalize((pixels@torch.inverse(K).permute(0,2,1))@R,dim=-1)[0]
    o=(-R.permute(0,2,1)@t).permute(0,2,1).repeat(1,len(d),1)[0]
    return o,d


def labels(torch,p,n,bounds):
    from benchmarks.bell_reference_e11 import field
    m=field(p,n,bounds);u=m['u'];norm=torch.linalg.norm(u,dim=-1)
    t=torch.cross(n,n.new_tensor([1.,0.,0.]).expand_as(n),dim=-1)
    t=torch.where((torch.linalg.norm(t,dim=-1)<1e-6)[:,None],torch.cross(n,n.new_tensor([0.,1.,0.]).expand_as(n),dim=-1),t)
    t=torch.nn.functional.normalize(t,dim=-1);b=torch.nn.functional.normalize(torch.cross(n,t,dim=-1),dim=-1)
    theta=torch.atan2((u*b).sum(-1),(u*t).sum(-1));theta=torch.where(norm>1e-12,theta,torch.full_like(theta,float('nan')))
    frame=torch.stack([t,b],-1);Qlocal=frame.transpose(-1,-2)@m['Q_world']@frame
    widths=torch.cat([torch.sqrt(m['a']**2+m['b']**2*norm[:,None]**2),m['a']],-1)
    encoded=torch.stack([torch.cos(2*theta),torch.sin(2*theta)],-1)
    return m,dict(Q_world=m['Q_world'],Q_local=Qlocal,widths_major_minor=widths,theta_major=theta,
                  rotation_double=encoded,orientation_defined=norm>1e-12,
                  orientation_eligible=(widths[:,0]-widths[:,1])>=.1,color=m['color'],metallic=m['metallic'],T=t,B=b)


def cpu_checks(o,d,arrays,vertices,faces):
    results=[]
    for y,x in [(64,64),(48,64),(80,64),(64,48),(64,80),(48,48),(80,80),(0,0)]:
        j=y*128+x;r=intersect(o[j],d[j],vertices,faces);hit=bool(arrays['hit'][j])
        item={'pixel_yx':[y,x],'cpu_hit':r is not None,'bvh_hit':hit}
        require((r is not None)==hit,'Independent CPU/BVH hit disagreement')
        if hit:
            error=abs(r['depth']-float(arrays['depth'][j]));normal=float(abs(r['raw_normal']-arrays['raw_normals'][j]).max())
            item.update(depth_error=error,raw_normal_error=normal,face=r['face'],barycentric_minimum=r['barycentric_minimum'])
            require(error<=TOL['cpu_depth_abs'] and normal<=TOL['cpu_normal_abs'],'Independent CPU/BVH intersection/normal mismatch; no repair allowed')
        results.append(item)
    return results


def render(args,out,report,budget,progress,P,K):
    """All CUDA tensor ownership ends on return/unwind, before outer cleanup."""
    import torch
    from PIL import Image
    from benchmarks.bell_reference_e11 import samples,integrate
    require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1,'Reviewed Torch/CUDA and exactly one selected GPU required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(LIMITS['cuda_allocator_fraction'])
    require('A100' in torch.cuda.get_device_name(0),'Reviewed A100 generation device required')
    science,_=baseline('material',PINS['material'])
    spec=importlib.util.spec_from_file_location('_e11_pinned_bvh',science/'raytracing/raytracer.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    import _raytracing
    report['runtime']={'python':__import__('sys').version,'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(0),'backend':record(Path(_raytracing.__file__)),
                       'wrapper':record(science/'raytracing/raytracer.py'),'pid':os.getpid(),
                       'device_capability':list(torch.cuda.get_device_capability(0)),
                       'device_total_memory':torch.cuda.get_device_properties(0).total_memory,
                       'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                       'driver_version':Path('/proc/driver/nvidia/version').read_text() if Path('/proc/driver/nvidia/version').exists() else 'unavailable'}
    vertices,faces=mesh_arrays(args.mesh);bounds=torch.tensor(np.stack([vertices.min(0),vertices.max(0)]),dtype=torch.float64,device='cuda')
    tracer=wrapper.RayTracer(vertices.astype(np.float32),faces.astype(np.int32))
    scene=out/'datasets/GlossySynthetic/bell';scene.mkdir(parents=True);(scene/'masks_train').mkdir()
    geometry=out/'geometry';geometry.mkdir();(geometry/'labels').mkdir();(geometry/'rays').mkdir()
    references=out/'references';references.mkdir();(references/'relight').mkdir()
    shutil.copyfile(args.mesh,geometry/'mesh.ply');require(digest(geometry/'mesh.ply')==MESH,'Mesh copy changed')
    write_arrays(geometry/'calibrations-safe.npz',pose=P,K=K,source_camera_ids=np.arange(128))
    with np.load(ROOT/TAPES,allow_pickle=False) as tapes:
        require(set(tapes.files)=={'diffuse','half'} and all(tapes[n].shape==(8192,2) and tapes[n].dtype==np.float64 and
                np.isfinite(tapes[n]).all() and ((tapes[n]>=0)&(tapes[n]<1)).all() for n in tapes.files),'Invalid frozen numeric integration tapes')
        td,th=[torch.from_numpy(tapes[n].copy()).cuda() for n in ('diffuse','half')]
    # All geometry/camera gates precede reference radiance generation.
    with torch.no_grad():
        for i in range(128):
            budget();report['phase']='geometry_'+str(i);o,d=primary(torch,P[i],K[i]);on,dn=o.numpy(),d.numpy();ref_o,ref_d=rays(P[i],K[i],128,128)
            ray_error=float(max(abs(on-ref_o).max(),abs(dn-ref_d).max()));require(ray_error<=2e-6,'Independent pinhole ray mismatch')
            pos,raw,depth=tracer.trace(o.cuda(),d.cuda());norm=-torch.nn.functional.normalize(raw,dim=-1)
            dep=depth.cpu().numpy();hit=dep<10
            a=dict(origins=on,directions=dn,positions=pos.cpu().numpy(),raw_normals=raw.cpu().numpy(),normals=norm.cpu().numpy(),depth=dep,hit=hit)
            with Image.open(args.cameras/'masks_train'/(str(i)+'.png')) as source:
                mask=np.asarray(source);require(mask.shape==(800,800),'Original Bell mask dimensions changed')
            index=np.floor((np.arange(128)+.5)*800/128).astype(int);mask=mask[np.ix_(index,index)]>127
            stats=geometry_gate(on,dn,a['positions'],a['raw_normals'],a['normals'],dep,hit,mask,P[i],K[i]);stats['independent_ray_error']=ray_error
            report['views'].append({'camera_id':str(i),'geometry':stats});report['counters']['primary_rays']+=len(dep)
            # Miss payloads of the original BVH are unspecified. Zero only storage slots for misses.
            for name in ('positions','raw_normals','normals'):a[name][~hit]=0
            write_arrays(geometry/'rays'/(str(i)+'.npz'),**a)
            if i in (0,11,17,58,76,105,114,127):
                report['cpu_crosschecks'].append({'camera_id':str(i),'rays':cpu_checks(on,dn,a,vertices,faces)})
            progress();require(stats['passed'],'Camera/mesh/normal/silhouette admission failed')
        require(sum(x['cpu_hit'] for v in report['cpu_crosschecks'] for x in v['rays'])>=16,'Insufficient independent CPU hit checks')
        for i in range(128):
            budget();report['phase']='independent_radiance_'+str(i);progress()
            with np.load(geometry/'rays'/(str(i)+'.npz'),allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
            ids=np.flatnonzero(a['hit']);fine=np.zeros((128*128,3,3));low=fine.copy();relfine=fine.copy();rellow=fine.copy()
            energy=np.zeros((128*128,3));energy_low=energy.copy()
            pt=torch.from_numpy(a['positions'][ids]).cuda().double();nn=torch.from_numpy(a['normals'][ids]).cuda().double()
            nn=torch.nn.functional.normalize(nn,dim=-1);wo=-torch.from_numpy(a['directions'][ids]).cuda().double();wo=torch.nn.functional.normalize(wo,dim=-1)
            _,all_labels=labels(torch,pt,nn,bounds)
            label_arrays={k:v.cpu().numpy() for k,v in all_labels.items()}
            report['views'][i]['labels']=verify_labels(pt.cpu().numpy(),nn.cpu().numpy(),bounds.cpu().numpy(),label_arrays)
            write_arrays(geometry/'labels'/(str(i)+'.npz'),pixel_indices=ids,position=pt.cpu().numpy(),normals=nn.cpu().numpy(),**label_arrays)
            relight=str(i) in split()['validation']+split()['held_view_nvs']
            completed_indices=[]
            try:
                for j in range(0,len(ids),LIMITS['point_batch']):
                    budget();end=min(j+LIMITS['point_batch'],len(ids));p,n,v=pt[j:end],nn[j:end],wo[j:end]
                    material,_=labels(torch,p,n,bounds);wi,pdf,valid=samples(n,v,td,th)
                    count=wi.shape[0]*wi.shape[1];require(report['counters']['shadow_directions']+count<=LIMITS['shadow_directions'],'Shadow work cap exceeded')
                    require(report['counters']['brdf_direction_evaluations']+count*(2 if relight else 1)<=LIMITS['brdf_direction_evaluations'],'Reference evaluation work cap exceeded')
                    report['counters']['shadow_directions']+=count;report['counters']['valid_shadow_queries']+=int(valid.sum())
                    # Original A ray offset + float32 BVH; no bounce illumination, no redraw.
                    wi32=wi.float();origin=p.float()[:,None]+1e-5*wi32;visible=torch.zeros_like(pdf)
                    _,_,distance=tracer.trace(origin[valid],wi32[valid]);require(bool(torch.isfinite(distance).all()),'Nonfinite shadow depths');visible[valid]=(distance>=10).to(pdf.dtype)
                    report['counters']['valid_shadow_query_returns']+=int(valid.sum())
                    report['counters']['brdf_direction_evaluations']+=count
                    f,c,e,ec=integrate(p,n,v,material,wi,pdf,visible,ENV,4096,return_energy=True)
                    report['counters']['brdf_direction_returns']+=count
                    fine[ids[j:end]]=f.cpu().numpy();low[ids[j:end]]=c.cpu().numpy()
                    energy[ids[j:end]]=e.cpu().numpy();energy_low[ids[j:end]]=ec.cpu().numpy()
                    completed_indices.extend(ids[j:end].tolist())
                    if relight:
                        report['counters']['brdf_direction_evaluations']+=count
                        f,c=integrate(p,n,v,material,wi,pdf,visible,RELIGHT,4096);relfine[ids[j:end]]=f.cpu().numpy();rellow[ids[j:end]]=c.cpu().numpy()
                        report['counters']['brdf_direction_returns']+=count
            except BaseException:
                write_arrays(references/('partial-'+str(i)+'.npz'),completed_pixel_indices=np.array(completed_indices,dtype=np.int64),
                             linear_components=fine,coarse_components=low,relight_components=relfine,relight_coarse=rellow)
                raise
            gate=convergence(fine[ids,2],low[ids,2]);report['views'][i]['radiance_convergence']=gate
            require(np.isfinite(energy).all() and (energy>=0).all(),'Nonfinite/negative directional energy')
            report['views'][i]['energy']={'maximum':float(energy[ids].max()),'minimum':float(energy[ids].min()),
                'channels_over_one':int((energy[ids]>1).sum()),'coarse_difference_maximum':float(abs(energy-energy_low).max()),
                'interpretation':'Descriptive additive-model reflectance; not a passive-material guarantee'}
            write_arrays(references/(str(i)+'.npz'),linear_components=fine.reshape(128,128,3,3),coarse_components=low.reshape(128,128,3,3),hit=a['hit'].reshape(128,128),unoccluded_directional_energy=energy.reshape(128,128,3),coarse_energy=energy_low.reshape(128,128,3))
            # Always preserve reference arrays before deciding admission.
            progress();require(gate['passed'],'Independent two-grid radiance convergence failed; no retries/tuning')
            Image.fromarray(linear_to_srgb(fine[:,2].reshape(128,128,3))).save(scene/(str(i)+'.png'))
            depth=np.where(a['hit'],a['depth'],15.);require((depth<=15).all(),'Depth encoding range exceeded')
            Image.fromarray(np.rint(depth.reshape(128,128)/15*65535).astype(np.uint16)).save(scene/(str(i)+'-depth.png'))
            Image.fromarray((a['hit'].reshape(128,128)*255).astype(np.uint8)).save(scene/'masks_train'/(str(i)+'.png'))
            # Newly generated trusted numeric calibration payload; never load original pickle objects.
            with (scene/(str(i)+'-camera.pkl')).open('xb') as f:pickle.dump([P[i],K[i]],f,protocol=4)
            if relight:
                gate=convergence(relfine[ids,2],rellow[ids,2]);report['views'][i]['relight_convergence']=gate
                write_arrays(references/'relight'/(str(i)+'.npz'),linear_components=relfine.reshape(128,128,3,3),coarse_components=rellow.reshape(128,128,3,3))
                require(gate['passed'],'Independent relighting convergence failed')
                Image.fromarray(linear_to_srgb(relfine[:,2].reshape(128,128,3))).save(references/'relight'/(str(i)+'.png'))
            progress()
    return {'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def run_with_cleanup(work,torch):
    """No successful return until owned work and CUDA/thread cleanup return."""
    import gc
    previous=torch.get_num_threads()
    try:
        result=work()
    finally:
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        torch.set_num_threads(previous)
    return result


def execute(args,freeze):
    P,K,inputs=admit_inputs(args.mesh,args.cameras)
    verify_authorization(args.authorization,args.authorization_sha256,args.freeze_sha256,args.commit,args.output)
    runtime(args)
    out=args.output;out.mkdir();start=time.monotonic()
    report=dict(schema_version=1,protocol='paper-benchmark-generation-v1',generation_protocol=PROTOCOL,status='incomplete',
                database_name='syn/bell',mesh_sha256=MESH,BRDF=LAW,independent_reference=True,
                commit=args.commit,commit_verification='host clean Git HEAD + runtime source-byte freeze; linked Git administration not mounted',image_id=IMAGE,freeze_sha256=args.freeze_sha256,source_files=freeze['files'],inputs=inputs,
                known_environment=ENV,relighting_environment=RELIGHT,indirect_mode='zero_bounce_v1',
                splits=split(),work_limits=LIMITS,tolerances=TOL,views=[],cpu_crosschecks=[],
                counters={'primary_rays':0,'shadow_directions':0,'valid_shadow_queries':0,'brdf_direction_evaluations':0,'valid_shadow_query_returns':0,'brdf_direction_returns':0},
                cleanup_verified=False,scientific_work_complete=False,researcher_reviewed_generation=False,
                native_model_calls=0,native_shader_calls=0,optimizer_returns=0,checkpoint_loads=0)
    last_storage_check=[0.]
    def budget():
        require(time.monotonic()-start<LIMITS['wall_seconds'],'Generation wall limit reached')
        if time.monotonic()-last_storage_check[0]>5:
            require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=LIMITS['output_bytes'],'Output byte limit exceeded')
            last_storage_check[0]=time.monotonic()
    last_phase=[None]
    def progress():
        # Atomic rolling progress is intentionally not an admission report.
        phase=report.get('phase')
        if phase!=last_phase[0]:
            print(json.dumps({'phase':phase,'counters':report['counters']}),flush=True);last_phase[0]=phase
        tmp=out/'progress.next';tmp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(out/'progress.json')
    try:
        import torch
        report.update(run_with_cleanup(lambda:render(args,out,report,budget,progress,P,K),torch))
        report['cleanup_verified']=True
        import resource
        report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        # Input/source terminal integrity is required, independently of earlier success flags.
        require(all(record(Path(r['path']))==r for r in inputs),'Original inputs changed during generation')
        frozen(args.freeze,args.freeze_sha256);require(report['counters']['primary_rays']==LIMITS['primary_rays'],'Incomplete primary rays')
        expected=sum(v['geometry']['hits'] for v in report['views'])*16384
        require(report['counters']['shadow_directions']==expected and len(report['views'])==128 and
                all(v.get('radiance_convergence',{}).get('passed') for v in report['views']), 'Incomplete radiance work')
        require(report['counters']['valid_shadow_queries']==report['counters']['valid_shadow_query_returns'] and
                report['counters']['brdf_direction_evaluations']==report['counters']['brdf_direction_returns'], 'Incomplete reference/visibility returns')
        report['scientific_work_complete']=True
        # Drop all owned CUDA resources before any complete/admission metadata.
        scene=out/'datasets/GlossySynthetic/bell';geometry=out/'geometry'
        label_index={'schema_version':1,'definition':PROTOCOL,'files':[record(p) for p in sorted((geometry/'labels').glob('*.npz'))],
                     'normals':'original A normalize(-BVH raw face normal); faceted; no repair',
                     'Q':'world tangent tensor; major/minor widths; orientation undefined where projected grain vanishes',
                     'material_domain':'additive Kurt model, not globally energy conserving'}
        # Store relocatable index paths, never make data depend on the host mount path.
        for r in label_index['files']:r['path']=str(Path(r['path']).relative_to(geometry))
        write_json(geometry/'labels.json',label_index)
        report.update(status='complete',dataset_files=source_records(scene),material_labels_sha256=digest(geometry/'labels.json'),elapsed_seconds=time.monotonic()-start)
        report['outputs']=[dict(record(p),path=p.relative_to(out).as_posix()) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('progress.json','progress.next')]
        write_json(geometry/'generation.json',report)
        proof=dict(schema_version=1,protocol='paper-anisotropic-oracle-dataset-v1',status='complete',database_name='syn/bell',
                   mesh_sha256=MESH,BRDF=LAW,known_material_labels=True,researcher_reviewed_generation=False,
                   coordinates='unchanged source world units (meters unverified), world-to-camera R|t; pixel +.5, K scaled128/800',
                   normals=label_index['normals'],image_formation='independent float64 single-bounce MIS, linear RGB + original sRGB, round-to-nearest uint8, black misses',
                   illumination='known world distant ENV, zero indirect/human; separate +90deg Y relighting',split_policy=split().__repr__(),
                   generation_report={'root':'geometry','path':'generation.json','sha256':digest(geometry/'generation.json')},
                   material_labels={'root':'geometry','path':'labels.json','sha256':digest(geometry/'labels.json')},dataset_files=report['dataset_files'])
        write_json(geometry/'oracle-admission.candidate.json',proof)
        require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<LIMITS['output_bytes']-4096,'Terminal output cap exceeded')
        write_json(out/'completion.json',{'status':'complete_pending_researcher_admission','generation_sha256':digest(geometry/'generation.json'),
                   'candidate_sha256':digest(geometry/'oracle-admission.candidate.json'),'scientific_work_complete':True,'cleanup_verified':True})
    except BaseException as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write_json(out/'failure.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--authorization',type=Path)
    p.add_argument('--authorization-sha256');p.add_argument('--output',type=Path);p.add_argument('--commit');p.add_argument('--image-id')
    a=p.parse_args();f=frozen(a.freeze,a.freeze_sha256)
    if not a.execute:
        P,K,r=admit_inputs(a.mesh,a.cameras)
        print(json.dumps({'status':'input_inspection_only','files':len(r),'calibration_shape':list(P.shape),'K128':K[0].tolist(),
                          'rendering_executed':False,'generation_authorized':False},indent=2));return
    require(all([a.authorization,a.authorization_sha256,a.output,a.commit,a.image_id]),'Explicit generation authorization and isolated runtime required')
    execute(a,f)
