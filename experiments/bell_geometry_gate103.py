"""Authorized-only original G checkpoint evolution, silhouettes and opacity."""
import argparse
import inspect
from contextlib import contextmanager
from pathlib import Path
import os
import resource
import shutil
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate103 as p
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import Surface
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments import bell_geometry_gate102 as prior
from experiments import bell_geometry_gate10_evaluate_v1 as audit
from experiments.bell_geometry_gate10_train_v1 import own_checkpoint,state_sha
from experiments.bell_geometry_gate101 import Queries,save_array
from experiments.donut_reconstructed_geometry_v1 import save,trace,reconstructed_normals,finite_hits,cpu_witness,interrupted
from infrastructure.g_checkpoint import strict_network_load
from infrastructure.mesh_diagnostics import geometry
from infrastructure.signatures import require
from infrastructure.stage_sources import activate,baseline,PINS,imported_sources
from infrastructure.verified_ray_backend import load


@contextmanager
def cuda_factories(torch):
    """Original G validation helpers allocate unqualified CUDA factory tensors."""
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    try:yield
    finally:torch.set_default_tensor_type('torch.FloatTensor')


class Calls:
    """Observe original function entries; no replacement of rendering arithmetic."""
    def __init__(self,torch,field,renderer,record):
        self.phase='idle';self.r=record
        self.codes=dict(sdf=field.SDFNetwork.forward.__code__,gradient=field.SDFNetwork.gradient.__code__,
            render=renderer.NeROShapeRenderer.render.__code__,appearance=field.AppShadingNetwork.forward.__code__,
            intersection=field.get_intersection.__code__,load=torch.load.__code__,
            jit=torch.utils.cpp_extension.load.__code__)
        self.forbidden={inspect.unwrap(fn).__code__ for fn in (torch.save,torch.Tensor.backward,
                        torch.optim.Adam.step,torch.optim.Optimizer.load_state_dict,torch.optim.Optimizer.__init__)}
        self.r.update(grid_SDF=0,consistency_SDF=0,neural_SDF=0,neural_gradients=0,loads=0,
                      renders=0,appearance_calls=0,appearance_points=0,occ_origins=0,validation_origins=0,jit_loads=0)
    def observe(self,frame,event,arg):
        if event!='call':return
        code=frame.f_code
        require(code not in self.forbidden,'Forbidden update/checkpoint writer/backward reached')
        file=code.co_filename
        require('/scientific_sources/geometry_g/train/' not in file and
                '/scientific_sources/geometry_g/dataset/' not in file,'No Trainer or dataset execution authorized')
        if code is self.codes['load']:
            require(self.phase=='load' and frame.f_locals.get('map_location')=='cpu' and self.r['loads']<4,
                    'Unapproved checkpoint deserialization')
            self.r['loads']+=1
        elif code is self.codes['sdf']:
            require(self.phase in ('grid','consistency','neural'),'Unaccounted SDF call')
            key=self.phase+'_SDF';n=frame.f_locals['inputs'].numel()//3
            cap={'grid':p.LIMITS['dense_SDF_points'],'consistency':p.LIMITS['consistency_SDF_points'],
                 'neural':p.LIMITS['neural_SDF_forward_points']}[self.phase]
            require(self.r[key]+n<=cap,'SDF primitive work cap exceeded');self.r[key]+=n
        elif code is self.codes['gradient']:
            require(self.phase=='neural','Unexpected original gradient helper')
            n=frame.f_locals['x'].numel()//3
            require(self.r['neural_gradients']+n<=p.LIMITS['neural_gradient_points'],'Gradient work cap exceeded')
            self.r['neural_gradients']+=n
        elif code is self.codes['render']:
            step=frame.f_locals['step']
            require(self.phase=='neural' and self.r['renders']<p.LIMITS['neural_render_calls'] and
                    frame.f_locals['is_train'] is True and frame.f_locals['perturb_overwrite']==-1 and
                    frame.f_locals['is_nerf'] is False and step in [s-1 for s in p.STEPS] and
                    frame.f_locals['cos_anneal_ratio']==min(1.,step/50000),
                    'Unapproved native render')
            self.r['renders']+=1
        elif code is self.codes['appearance']:
            n=frame.f_locals['points'].numel()//3
            require(self.phase=='neural' and self.r['appearance_calls']<128 and
                    self.r['appearance_points']+n<=p.LIMITS['appearance_point_slots'],'Appearance budget exceeded')
            self.r['appearance_calls']+=1;self.r['appearance_points']+=n
        elif code is self.codes['intersection']:
            require(self.phase=='neural','Unexpected original intersection helper')
            pair=(frame.f_locals['sn0'],frame.f_locals['sn1']);n=len(frame.f_locals['pts'])
            require(pair in ((64,16),(128,9)),'Unexpected original intersection resolution')
            key,cap=('occ_origins',p.LIMITS['occ_probe_origins']) if pair==(64,16) else ('validation_origins',p.LIMITS['validation_probe_origins'])
            require(self.r[key]+n<=cap,'Secondary SDF-probe origin cap exceeded');self.r[key]+=n
        elif code is self.codes['jit']:
            require(self.phase=='plugin' and self.r['jit_loads']==0 and
                    frame.f_locals['name']=='nvdiffrast_plugin','Only the original cached plugin loader is authorized')
            self.r['jit_loads']+=1
    def __enter__(self):
        require(sys.getprofile() is None and sys.gettrace() is None,'Unobserved fresh process required')
        sys.setprofile(self.observe);return self
    def __exit__(self,*args):sys.setprofile(None)


def dense_field(network,out,outside,torch,row,budget,progress):
    from network.field import extract_fields
    raw=np.lib.format.open_memmap(out/'raw-SDF-grid.npy',mode='w+',dtype=np.float32,shape=(512,512,512))
    row['grid_tiles']=0
    def query(points):
        k=row['grid_tiles']
        require(k<512 and points.shape==(64**3,3) and not torch.is_grad_enabled(),'Original extraction tile differs')
        y=network.sdf_network.sdf(points)
        require(y.shape==(64**3,1) and bool(torch.isfinite(y).all().item()),'Invalid original raw SDF')
        xi,yi,zi=k//64,(k//8)%8,k%8
        sl=(slice(xi*64,(xi+1)*64),slice(yi*64,(yi+1)*64),slice(zi*64,(zi+1)*64))
        raw[sl]=y.detach().cpu().numpy().reshape(64,64,64)
        require(np.array_equal((torch.norm(points,dim=-1)>=1).cpu().numpy().reshape(64,64,64),outside[sl]),
                'Original extraction nodes/domain changed')
        row['grid_tiles']+=1
        if k%32==0:budget();progress()
        return y
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    try:
        box=torch.ones(3)
        masked=extract_fields(-box,box,512,query,outside_val=1.)
    finally:torch.set_default_tensor_type('torch.FloatTensor')
    require(row['grid_tiles']==512,'Incomplete raw field')
    raw.flush()
    for start in range(0,512,32):
        require(np.array_equal(masked[start:start+32],np.where(outside[start:start+32],np.float32(1),raw[start:start+32])),
                'Original raw/sentinel relation differs')
    del masked
    return raw


def candidate(raw,valid,any_inside,out,row):
    import mcubes
    import open3d
    import importlib.metadata
    require(importlib.metadata.version('PyMCubes')=='0.1.4','Original MCubes required')
    data=np.array(raw,copy=True,order='C')
    require(row['MC_attempts']==row['MC_returns']==0,'One extraction per earlier checkpoint only')
    row['MC_attempts']+=1
    voxel,faces=mcubes.marching_cubes(data,0);row['MC_returns']+=1
    row['raw_mesh_layout']=dict(vertices=len(voxel),faces=len(faces),vertex_dtype=str(voxel.dtype),face_dtype=str(faces.dtype))
    require(np.array_equal(data,raw) and 8<len(faces)<=4000000 and np.isfinite(voxel).all() and
            voxel.min()>=0 and voxel.max()<=511,
            'Invalid or over-budget raw extraction')
    world=voxel/511*2-1
    save_array(out/'raw-voxel-vertices.npy',voxel);save_array(out/'raw-world-vertices.npy',world)
    save_array(out/'raw-faces.npy',faces)
    keep,reason,possible,good=p.policy.face_domain(voxel,faces,valid,any_inside)
    save(out/'face-domain.npz',dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good))
    before,labels,edges=p.policy.topology(world,faces[keep])
    save(out/'topology-before-cleanup.npz',dict(face_labels=labels,boundary_edges=edges))
    v,f,mapping,retained,removed,cleanup=p.policy.numerical_cleanup(world,faces,keep)
    save(out/'cleanup-mappings.npz',dict(raw_vertex_to_candidate=mapping,candidate_face_to_raw=retained,
                                      removed_exact_zero_raw_faces=removed))
    after,labels,edges=p.policy.topology(v,f)
    save(out/'topology-after-cleanup.npz',dict(face_labels=labels,boundary_edges=edges))
    mesh=out/'candidate.ply';p.policy.previous.write_ply(mesh,v,f)
    readv,readf=mesh_arrays(mesh);native=open3d.io.read_triangle_mesh(str(mesh))
    require(np.array_equal(v,readv) and np.array_equal(f,readf) and
            np.array_equal(v,np.asarray(native.vertices)) and np.array_equal(f,np.asarray(native.triangles)),
            'Native float32 mesh roundtrip differs')
    row.update(cleanup=cleanup,topology_before=before,topology_after=after,geometry=geometry(v,f,np),mesh=p.item(mesh))
    return v,f


def fitting_views(plan,v,f,valid,any_inside,network,torch,guard,out,row,budget,progress):
    from raytracing import RayTracer
    tracer=RayTracer(v,f);surface=Surface(v,f)
    row['fitting_BVH_rays']=row['fitting_CPU_checks']=row['neural_rays']=0
    row['fitting']=[];(out/'fitting').mkdir();(out/'neural').mkdir()
    with np.load('/inputs/oracle/calibrations-safe.npz',allow_pickle=False) as cal:
        for camera in plan['fitting_ids']:
            with np.load('/inputs/oracle/rays/'+camera+'.npz',allow_pickle=False) as z:
                a={k:z[k] for k in ('origins','directions','hit','depth','positions','normals')}
            ref_o,ref_d=audit.rays(cal['pose'][int(camera)],cal['K'][int(camera)],128,128)
            require(a['hit'].shape==(128**2,) and max(np.max(abs(ref_o-a['origins'])),np.max(abs(ref_d-a['directions'])))<=2e-6,
                    'Fitting archive/calibrated ray mismatch')
            row['fitting_BVH_rays']+=128**2
            hit=trace(tracer,a['origins'],a['directions'],torch)
            normals,eligible=reconstructed_normals(hit['raw'],hit['hit'])
            require(finite_hits(hit) and np.array_equal(eligible,hit['hit']) and
                np.max(abs(np.linalg.norm(hit['raw'][eligible],axis=1)-1),initial=0)<=p.policy.original.RAY_TOL['unit_normal'],
                'Invalid fitting closest hit')
            reconstructed=dict(hit=hit['hit'],normal=normals,position=hit['position'],depth=hit['depth'])
            oracle=dict(hit=a['hit'],normal=a['normals'],position=a['positions'],depth=a['depth'])
            metrics,masks=p.policy.original.paired_surface(oracle,reconstructed,a['directions'])
            metrics.update(camera=camera,split='fitting',**p.silhouette(a['hit'],hit['hit']))
            extra=hit['hit']&~a['hit'];radius=np.linalg.norm(hit['position'][extra].astype(float),axis=1)
            metrics['extra_radial_interior']=int((radius<1-plan['domain']['cell_counts']['maximum_boundary_omission_depth']).sum())
            domain=np.zeros_like(a['hit']);domain[a['hit']]=p.policy.point_domain(a['positions'][a['hit']],valid,any_inside)
            metrics.update(oracle_domain_supported=int(domain.sum()),oracle_domain_unsupported=int((a['hit']&~domain).sum()),
                missing_inside_domain=int((a['hit']&~hit['hit']&domain).sum()),missing_outside_domain=int((a['hit']&~hit['hit']&~domain).sum()))
            save(out/'fitting'/f'{camera}.npz',dict(oracle_hit=a['hit'],**hit,normal=normals,**masks))
            if camera in plan['neural_panel']:
                fg=np.flatnonzero(a['hit']);bg=np.flatnonzero(~a['hit'])
                witnesses=np.r_[fg[[0,-1]],bg[[0,-1]]]
                metrics['CPU_witnesses']=[cpu_witness(surface,dict(**hit,normal=normals),a['origins'][i],a['directions'][i],int(i)) for i in witnesses]
                row['fitting_CPU_checks']+=len(witnesses)
                require(all(w['passed'] for w in metrics['CPU_witnesses']),'Fitting independent witness failed')
                indices=np.array(plan['neural_panel'][camera]['indices'],np.int64)
                with torch.no_grad():
                    require(torch.tensor(0.).device.type=='cpu','Original ray-pool construction requires CPU factories')
                    pose=torch.from_numpy(cal['pose'][int(camera)].astype(np.float32))[None]
                    K=torch.from_numpy(cal['K'][int(camera)].astype(np.float32))[None]
                    info=dict(imgs=torch.zeros((1,3,128,128),device='cpu'),Ks=K,poses=pose)
                    batch,poses,_,_,_=network._construct_ray_batch(info,device='cpu',is_train=False)
                    index=torch.from_numpy(indices)
                    batch={k:value[index].cuda() for k,value in batch.items()}
                    poses=poses.cuda()
                with torch.no_grad(),cuda_factories(torch):
                    o,d,near,far,human=network._process_ray_batch(batch,poses)
                    require(max(np.max(abs(o.cpu().numpy()-a['origins'][indices])),
                                np.max(abs(d.cpu().numpy()-a['directions'][indices])))<=2e-6,'Neural/mesh rays differ')
                    opacity=[]
                    for draw in range(2):
                        seed=6033+1000*int(camera)+draw
                        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                        network.train();guard.phase='neural'
                        try:render=network.render(o,d,near,far,human,-1,network.get_anneal_val(row['step']-1),
                                                 is_train=True,step=row['step']-1,is_nerf=False)
                        finally:guard.phase='idle';network.eval()
                        captured={k:render[k].detach().cpu().numpy() for k in ('fg_acc','acc','ray_rgb','fg_rgb','bg_rgb')}
                        require(all(np.isfinite(value).all() for value in captured.values()) and
                                (captured['fg_acc']>=-1e-4).all() and (captured['fg_acc']<=1+1e-4).all(),
                                'Invalid original neural accumulation/output')
                        row['neural_rays']+=len(indices)
                        save(out/'neural'/f'{camera}-{draw}.npz',dict(indices=indices,oracle_hit=a['hit'][indices],
                            mesh_hit=hit['hit'][indices],origins=a['origins'][indices],directions=a['directions'][indices],
                            seed=np.asarray(seed,np.int64),loop_index=np.asarray(row['step']-1,np.int64),**captured))
                        opacity.append(captured['fg_acc'])
                        del render
                    metrics['neural']=p.opacity_summary(plan['neural_panel'][camera],a['hit'][indices],hit['hit'][indices],np.stack(opacity))
            row['fitting'].append(metrics);budget();progress()
    require(row['fitting_BVH_rays']==122*128**2 and row['fitting_CPU_checks']==64 and
            row['neural_rays']==2*sum(len(v['indices']) for v in plan['neural_panel'].values()),'Incomplete fitting inventory')


def native(plan,args):
    require(p.digest(Path('/authorization.json'))==args.authorization_sha256 and
        p.read('/authorization.json')==p.authorization(plan,args.commit,args.freeze_sha256),'Exact execution approval required')
    import torch
    import torch.utils.cpp_extension
    import yaml
    require(os.getuid()!=0 and sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and
        torch.version.cuda=='11.1' and torch.cuda.is_available() and torch.cuda.device_count()==1 and
        torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe','Frozen runtime required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','CPU float32 construction required')
    out=Path('/outputs');require(not any(out.iterdir()),'Fresh native output required')
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.cuda.set_per_process_memory_fraction(.75)
    require(load('predicate_fp64',plan['image'])==plan['backend'],'Corrected BVH identity differs')
    science,manifest=activate('geometry',PINS['geometry'])
    from network import field,renderer
    r=dict(protocol=p.PROTOCOL,commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
        status='incomplete',scientific_work_complete=False,cleanup_verified=False,geometry_admitted=False,
        handoff_created=False,steps=[],observed={},image=plan['image'],backend=plan['backend'],optimizer_returns=0,
        original_geometry_quality=plan['quality'],optimizer_state_restored=False,
        scientific_commit=PINS['geometry'],scientific_files=manifest['files'])
    from infrastructure.geometry_training_smoke import runtime_identity
    r['runtime']=runtime_identity(torch)
    start=time.monotonic()
    def progress():
        r['wall_seconds']=time.monotonic()-start
        temp=out/'receipt.next';temp.unlink(missing_ok=True);p.write(temp,r);temp.replace(out/'receipt.json')
    def budget():
        require(time.monotonic()-start<p.RESOURCES['inner'] and
            sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())<=p.RESOURCES['output_bytes'],
            'Time/output cap exceeded')
    def work():
        for name,identity in plan['eval_files'].items():p.check(Path('/inputs/oracle')/name,identity)
        for name,identity in plan['reuse']['files'].items():p.check(Path('/inputs/accepted')/name,identity)
        outside=np.load(p.check(Path('/inputs/mask.npy'),plan['outside_mask']),allow_pickle=False,mmap_mode='r')
        valid,any_inside=p.policy.domain_cells(outside)
        require(p.policy.cell_counts(valid,any_inside)==plan['domain']['cell_counts'],'Frozen domain differs')
        cache=out/'torch_extensions';cache.mkdir()
        for name,identity in plan['jit_cache'].items():
            source=p.check(Path('/inputs/jit')/name,identity);dest=cache/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
        workspace=out/'workspace';workspace.mkdir();(workspace/'assets').symlink_to(science/'assets',target_is_directory=True)
        os.chdir(workspace)
        with Calls(torch,field,renderer,r['observed']) as guard:
            import nvdiffrast.torch.ops as ops
            guard.phase='plugin';plugin=ops._get_plugin();guard.phase='idle'
            require(p.item(Path(plugin.__file__))==plan['jit_cache']['nvdiffrast_plugin/nvdiffrast_plugin.so'],
                    'Accepted plugin binary must verify before any scientific query/render')
            for step in p.STEPS:
                row=dict(step=step,MC_attempts=0,MC_returns=0,grid_tiles=0,primary_BVH_rays=0,CPU_ray_checks=0,views=[])
                r['steps'].append(row);folder=out/str(step);folder.mkdir();progress()
                require(torch.tensor(0.).device.type=='cpu','Original constructor factory context was not restored')
                network=renderer.NeROShapeRenderer(yaml.safe_load((p.ROOT/p.policy.original.CFG).read_text()),training=False)
                require(not network.is_nerf and network.cfg['n_samples']==64 and network.cfg['n_bg_samples']==32 and
                        network.cfg['n_importance']==64 and network.cfg['up_sample_steps']==4 and
                        network.cfg['perturb']==1 and network.cfg['anneal_end']==50000,
                        'Original rendering/sampling configuration differs')
                ref=plan['checkpoints'][str(step)];checkpoint=p.check(Path('/inputs/checkpoints')/ref['file'],ref['identity'])
                guard.phase='load';identity,state=own_checkpoint(checkpoint,step,network,torch,expected=ref['identity'])
                guard.phase='idle'
                require(all(bool(torch.isfinite(value).all().item()) for value in state['network_state_dict'].values()),
                        'Nonfinite checkpoint network tensor')
                row['finite_checkpoint_tensors']=True
                row['strict_load']=strict_network_load(network,state['network_state_dict'],torch);del state
                network.eval().cuda();before=state_sha(network.state_dict());row.update(checkpoint=identity,network_sha256=before)
                if step!=300000:
                    guard.phase='grid';raw=dense_field(network,folder,outside,torch,row,budget,progress);guard.phase='idle'
                    v,f=candidate(raw,valid,any_inside,folder,row)
                    points,normals=prior.geometry_audit(v,f,valid,any_inside,plan,folder,row,torch,budget,progress)
                    centers,n,fi,vi=audit.surface_witnesses(v,f)
                    x=np.concatenate((centers[fi],v[vi],points)).astype(np.float32)
                    normal=np.concatenate((-n[fi],np.zeros((len(vi),3)),normals))
                    guard.phase='consistency'
                    q=Queries(network.sdf_network.sdf,torch,1536,512,row.setdefault('SDF_work',{}))
                    gate,arrays=audit.sdf_consistency(x,normal,q);guard.phase='idle'
                    save(folder/'sdf-witnesses.npz',dict(**arrays,selected_faces=fi,selected_vertices=vi));row['SDF_gate']=gate
                else:
                    prior_receipt=p.read('/inputs/accepted/run/receipt.json')
                    v,f=mesh_arrays('/inputs/accepted/run/domain-aware-300000.ply')
                    raw=np.load(p.check(Path('/inputs/raw300.npy'),plan['raw300']),allow_pickle=False,mmap_mode='r')
                    row.update(reused_300k=True,views=prior_receipt['views'],SDF_gate=prior_receipt['original_SDF_gate'],
                               topology_after=prior_receipt['topology_after_cleanup'],mesh=prior_receipt['mesh_outputs'])
                    require(before==p.policy.previous.NETWORK_SHA,'Accepted endpoint weights differ')
                row['negative_extension']=p.negative_extension(raw,outside)
                fitting_views(plan,v,f,valid,any_inside,network,torch,guard,folder,row,budget,progress)
                require(state_sha(network.state_dict())==before and all(x.grad is None for x in network.parameters()),
                        'Evaluation mutated model or accumulated parameter gradients')
                row['network_preserved']=True;del network,raw;torch.cuda.empty_cache();budget();progress()
        r['imported_sources']=imported_sources(science)
        plugin=out/'torch_extensions/nvdiffrast_plugin/nvdiffrast_plugin.so'
        require(p.item(plugin)==plan['jit_cache']['nvdiffrast_plugin/nvdiffrast_plugin.so'],
                'Original runtime changed the accepted nvdiffrast binary')
        r['plugin']=p.item(plugin);r['scientific_work_complete']=True
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);progress()
    try:
        try:run_with_cleanup(work,torch)
        finally:torch.set_default_tensor_type('torch.FloatTensor')
        require(sys.getprofile() is None and sys.gettrace() is None and torch.tensor(0.).device.type=='cpu','Cleanup incomplete')
        budget();baseline('geometry',PINS['geometry']);p.freeze(args.freeze_sha256)
        r.update(status='complete',cleanup_verified=True,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        r['artifacts']={x.relative_to(out).as_posix():p.item(x) for x in sorted(out.rglob('*'))
                        if x.is_file() and x.resolve()==x and x.name!='receipt.json'}
    except BaseException as exc:r.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:progress()


def verify(plan,args):
    from experiments.bell_geometry_gate10_v1 import verify_rasters,verify_sdf
    root=Path(plan['host_output']);out=root/'run';r=p.read(out/'receipt.json')
    require((root/'exit-code.txt').read_text().strip()=='0' and (root/'container-absent.txt').read_text().strip()=='absent' and
        (root/'image-id.txt').read_text().strip()==plan['image'],'Host execution/image/cleanup receipts differ')
    require(r['commit']==args.commit and r['freeze_sha256']==args.freeze_sha256 and
        r['authorization_sha256']==args.authorization_sha256 and r['status']=='complete' and
        r['scientific_work_complete'] and r['cleanup_verified'] and not r['geometry_admitted'] and
        not r['handoff_created'] and r['observed']['loads']==4 and r['observed']['renders']==128 and
        r['observed']['grid_SDF']==p.LIMITS['dense_SDF_points'],'Incomplete frozen study')
    require([row['step'] for row in r['steps']]==list(p.STEPS) and not r['optimizer_state_restored'] and
            r['optimizer_returns']==0 and r['scientific_commit']==PINS['geometry'],'Study identity/history inventory differs')
    ceilings=dict(consistency_SDF='consistency_SDF_points',neural_SDF='neural_SDF_forward_points',
        neural_gradients='neural_gradient_points',appearance_calls='appearance_forward_calls',
        appearance_points='appearance_point_slots',occ_origins='occ_probe_origins',validation_origins='validation_probe_origins')
    require(all(r['observed'][key]<=p.LIMITS[limit] for key,limit in ceilings.items()) and r['observed']['jit_loads']<=1,
            'Observed primitive work exceeds freeze')
    require(sum(row['MC_attempts'] for row in r['steps'])==sum(row['MC_returns'] for row in r['steps'])==3 and
        sum(row['grid_tiles'] for row in r['steps'])==1536 and
        sum(row['primary_BVH_rays'] for row in r['steps'])==p.LIMITS['audit_BVH_rays'] and
        sum(row['CPU_ray_checks']+row['fitting_CPU_checks'] for row in r['steps'])==p.LIMITS['CPU_ray_checks'] and
        sum(row['fitting_BVH_rays'] for row in r['steps'])==p.LIMITS['fitting_BVH_rays'],
        'Total extraction/ray inventories differ')
    for name,identity in r['artifacts'].items():p.check(out/name,identity)
    outside=np.load(Path(p.policy.read(p.ROOT/p.policy.FREEZE)['diagnostic'])/'run/stages/outside-sphere.npy',allow_pickle=False,mmap_mode='r')
    valid,any_inside=p.policy.domain_cells(outside)
    for row in r['steps']:
        step=row['step'];folder=out/str(step)
        require(row['network_preserved'] and row['finite_checkpoint_tensors'] and
                row['checkpoint']==plan['checkpoints'][str(step)]['identity'],'Checkpoint identity/state differs')
        if step!=300000:
            voxel=np.load(folder/'raw-voxel-vertices.npy',allow_pickle=False);faces=np.load(folder/'raw-faces.npy',allow_pickle=False)
            world=np.load(folder/'raw-world-vertices.npy',allow_pickle=False)
            require(np.array_equal(world,voxel/511*2-1),'Coordinate conversion differs')
            keep,reason,possible,good=p.policy.face_domain(voxel,faces,valid,any_inside)
            with np.load(folder/'face-domain.npz',allow_pickle=False) as z:
                require(all(np.array_equal(z[k],v) for k,v in dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good).items()),'Domain replay differs')
            v,f,mapping,retained,removed,cleanup=p.policy.numerical_cleanup(world,faces,keep)
            cv,cf=mesh_arrays(folder/'candidate.ply')
            require(np.array_equal(cv,v) and np.array_equal(cf,f) and row['cleanup']==cleanup,'Cleanup replay differs')
            with np.load(folder/'cleanup-mappings.npz',allow_pickle=False) as z:
                require(np.array_equal(z['raw_vertex_to_candidate'],mapping) and np.array_equal(z['candidate_face_to_raw'],retained)
                        and np.array_equal(z['removed_exact_zero_raw_faces'],removed),'Cleanup map replay differs')
            topo,labels,edges=p.policy.topology(v,f)
            require(topo==row['topology_after'],'Topology replay differs')
            with np.load(folder/'topology-after-cleanup.npz',allow_pickle=False) as z:
                require(np.array_equal(z['face_labels'],labels) and np.array_equal(z['boundary_edges'],edges),'Topology arrays differ')
            verify_rasters(folder,row['views']);verify_sdf(folder/'sdf-witnesses.npz',row['SDF_gate'])
        else:
            require(row['MC_attempts']==row['MC_returns']==row['grid_tiles']==row['primary_BVH_rays']==0 and
                    row['views']==p.read(p.ACCEPTED/'run/receipt.json')['views'],'300k audit was changed or repeated')
        require(row['fitting_BVH_rays']==122*128**2 and row['fitting_CPU_checks']==64,'Fitting inventory incomplete')
        require([v['camera'] for v in row['fitting']]==plan['fitting_ids'] and
                row['neural_rays']==2*sum(len(v['indices']) for v in plan['neural_panel'].values()),
                'Fitting camera or neural-ray inventory differs')
        for fit in row['fitting']:
            camera=fit['camera']
            with np.load(folder/'fitting'/f'{camera}.npz',allow_pickle=False) as z:
                expected=p.silhouette(z['oracle_hit'],z['hit'])
                require(all(fit[k]==v for k,v in expected.items()),'Fitting silhouette replay differs')
                with np.load(Path(plan['acquisition'])/'geometry/rays'/f'{camera}.npz',allow_pickle=False) as a:
                    oracle=dict(hit=a['hit'],normal=a['normals'],position=a['positions'],depth=a['depth'])
                    reconstructed=dict(hit=z['hit'],normal=z['normal'],position=z['position'],depth=z['depth'])
                    observed,masks=p.policy.original.paired_surface(oracle,reconstructed,a['directions'])
                    require(all(fit[k]==v for k,v in observed.items()) and all(np.array_equal(z[k],v) for k,v in masks.items()),
                            'Fitting physical/eligibility replay differs')
            if camera in plan['neural_panel']:
                opacity=[]
                for draw in range(2):
                    with np.load(folder/'neural'/f'{camera}-{draw}.npz',allow_pickle=False) as z:
                        require(np.array_equal(z['indices'],plan['neural_panel'][camera]['indices']) and
                                int(z['seed'])==6033+1000*int(camera)+draw and int(z['loop_index'])==step-1 and
                                all(np.isfinite(z[k]).all() for k in ('fg_acc','acc','ray_rgb','fg_rgb','bg_rgb')),
                                'Neural panel/values differ')
                        opacity.append(z['fg_acc'])
                        oracle_hit=z['oracle_hit'];mesh_hit=z['mesh_hit']
                require(fit['neural']==p.opacity_summary(plan['neural_panel'][camera],oracle_hit,mesh_hit,np.stack(opacity)),
                        'Opacity summary replay differs')
    p.write(root/'verification.json',dict(status='complete_verified_diagnostic',receipt=p.item(out/'receipt.json'),
        geometry_admitted=False,handoff_created=False,publication_endpoint=300000,
        comparison_is_development_evidence=True,quality_decisions={str(row['step']):dict(
            original_six_views=[dict(camera=x['camera'],passed=x['passed']) for x in row['views']],SDF=row['SDF_gate']['passed']) for row in r['steps']}))
    print(root/'verification.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('preflight','execute','verify'))
    parser.add_argument('--commit',required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--authorization-sha256',required=True);parser.add_argument('--authorization',type=Path)
    args=parser.parse_args();plan=p.freeze(args.freeze_sha256)
    if args.phase=='execute':native(plan,args);return
    auth=args.authorization or Path(plan['host_output'])/'authorization.json'
    require(p.digest(auth)==args.authorization_sha256 and p.read(auth)==p.authorization(plan,args.commit,args.freeze_sha256),
            'Exact-source approval required')
    p.host_inputs(plan)
    if args.phase=='verify':verify(plan,args);return
    require(not os.path.lexists(plan['host_output']),'Fresh output required')
    print(plan['host_output']);print(plan['image']);print(plan['acquisition']+'/geometry')


if __name__=='__main__':main()
