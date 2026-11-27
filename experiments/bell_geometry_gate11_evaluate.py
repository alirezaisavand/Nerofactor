"""Frozen four-checkpoint O1 evaluation using accepted raw-SDF/camera policies."""
import inspect
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
from benchmarks import bell_geometry_gate11 as p
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import Surface
from experiments import bell_geometry_gate102 as prior
from experiments import bell_geometry_gate10_evaluate_v1 as audit
from experiments.bell_geometry_gate10_train_v1 import own_checkpoint,state_sha
from experiments.bell_geometry_gate101 import Queries,save_array
from experiments.donut_reconstructed_geometry_v1 import save,trace,reconstructed_normals,finite_hits,cpu_witness
from infrastructure.g_checkpoint import strict_network_load
from infrastructure.mesh_diagnostics import geometry
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS


@contextmanager
def cuda_factories(torch):
    """Original G validation helpers allocate unqualified CUDA factory tensors."""
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    try:yield
    finally:torch.set_default_tensor_type('torch.FloatTensor')


def camera_prefix_cpu(network,info,indices,torch):
    """Preserve original camera methods, run their small calibration algebra on CPU."""
    with torch.no_grad():
        require(torch.tensor(0.).device.type=='cpu' and
                all(value.device.type=='cpu' for value in info.values()),'CPU camera prefix required')
        batch,poses,_,_,_=network._construct_ray_batch(info,device='cpu',is_train=False)
        index=torch.from_numpy(indices)
        batch={k:value[index] for k,value in batch.items()}
        values=network._process_ray_batch(batch,poses)
        require(all(value.device.type=='cpu' and bool(torch.isfinite(value).all()) for value in values),
                'Camera prefix left CPU or produced nonfinite values')
    return values,batch,poses


def ray_error(values,origins,directions):
    o,d=values[:2]
    o,d=o.detach().cpu().numpy(),d.detach().cpu().numpy()
    require(o.shape==origins.shape==d.shape==directions.shape and
            all(np.isfinite(v).all() for v in (o,d,origins,directions)),'Invalid camera-ray witness')
    return dict(origin=float(np.max(abs(o-origins))),direction=float(np.max(abs(d-directions))))


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
                    cpu,batch,poses=camera_prefix_cpu(network,info,indices,torch)
                errors=ray_error(cpu,a['origins'][indices],a['directions'][indices])
                metrics['camera_ray_errors']=errors
                save(out/'neural'/f'{camera}-rays.npz',dict(indices=indices,
                    origins=cpu[0].numpy(),directions=cpu[1].numpy(),near=cpu[2].numpy(),far=cpu[3].numpy(),
                    archived_origins=a['origins'][indices],archived_directions=a['directions'][indices]))
                require(max(errors.values())<=2e-6,'CPU neural/mesh rays differ; saved numerical witness')
                with torch.no_grad(),cuda_factories(torch):
                    o,d,near,far,human=(value.cuda() for value in cpu)
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


def run(plan,training_receipt,science,out,r,torch,budget,progress):
    import shutil
    import yaml
    from network import field,renderer
    for name,identity in plan['eval_files'].items():p.check(Path('/inputs/oracle')/name,identity)
    outside=np.load(p.check(Path('/inputs/mask.npy'),plan['outside_mask']),allow_pickle=False,mmap_mode='r')
    valid,any_inside=p.policy.domain_cells(outside)
    require(p.policy.cell_counts(valid,any_inside)==plan['domain']['cell_counts'],'Frozen domain differs')
    workspace=out/'workspace';workspace.mkdir();(workspace/'assets').symlink_to(science/'assets',target_is_directory=True)
    import os
    os.chdir(workspace)
    with Calls(torch,field,renderer,r['observed']) as guard:
        import nvdiffrast.torch.ops as ops
        guard.phase='plugin';plugin=ops._get_plugin();guard.phase='idle'
        require(p.item(Path(plugin.__file__))==plan['jit_cache']['nvdiffrast_plugin/nvdiffrast_plugin.so'],
                'Accepted runtime plugin differs')
        for step in p.STEPS:
            folder=out/str(step);folder.mkdir()
            row=dict(step=step,MC_attempts=0,MC_returns=0,grid_tiles=0,primary_BVH_rays=0,CPU_ray_checks=0,views=[])
            r['steps'].append(row);progress()
            require(torch.tensor(0.).device.type=='cpu','CPU constructor context required')
            network=renderer.NeROShapeRenderer(yaml.safe_load((p.ROOT/p.CFG).read_text()),training=False)
            ref=training_receipt['endpoint_checkpoint'] if step==p.STEP else training_receipt['retained_checkpoints'][str(step)]
            filename='model.pth' if step==p.STEP else f'model-step{step}.pth'
            path=p.check(Path('/inputs/training-result/checkpoints')/filename,ref)
            guard.phase='load';identity,state=own_checkpoint(path,step,network,torch,expected=ref);guard.phase='idle'
            row['strict_load']=strict_network_load(network,state['network_state_dict'],torch);del state
            network.eval().cuda();before=state_sha(network.state_dict())
            row.update(checkpoint=identity,network_sha256=before,finite_checkpoint_tensors=True)
            if step==p.STEP:require(before==training_receipt['endpoint_network_sha256'],'Endpoint network identity differs')
            guard.phase='grid';raw=dense_field(network,folder,outside,torch,row,budget,progress);guard.phase='idle'
            v,f=candidate(raw,valid,any_inside,folder,row)
            points,normals=prior.geometry_audit(v,f,valid,any_inside,plan,folder,row,torch,budget,progress)
            centers,n,fi,vi=audit.surface_witnesses(v,f)
            x=np.concatenate((centers[fi],v[vi],points)).astype(np.float32)
            normals=np.concatenate((-n[fi],np.zeros((len(vi),3)),normals))
            guard.phase='consistency'
            q=Queries(network.sdf_network.sdf,torch,1536,512,row.setdefault('SDF_work',{}))
            gate,arrays=audit.sdf_consistency(x,normals,q);guard.phase='idle'
            save(folder/'sdf-witnesses.npz',dict(**arrays,selected_faces=fi,selected_vertices=vi));row['SDF_gate']=gate
            row['negative_extension']=p.negative_extension(raw,outside)
            fitting_views(plan,v,f,valid,any_inside,network,torch,guard,folder,row,budget,progress)
            require(state_sha(network.state_dict())==before and all(x.grad is None for x in network.parameters()),
                    'Evaluation changed network state or accumulated gradients')
            row['network_preserved']=True;del network,raw;torch.cuda.empty_cache();budget();progress()
    require(r['observed']['loads']==4 and r['observed']['renders']==128 and
            r['observed']['grid_SDF']==p.LIMITS['dense_SDF_points'],'Evaluation inventory incomplete')
