"""One authorized endpoint-only extraction diagnosis and conditional Gate 10 audit."""
import argparse
import ast
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate101 as p
from benchmarks import bell_geometry_gate10_v1 as g
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import Surface
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments import bell_geometry_gate10_evaluate_v1 as e
from experiments.bell_geometry_gate10_train_v1 import state_sha,own_checkpoint
from experiments.donut_reconstructed_geometry_v1 import save,interrupted
from infrastructure.geometry_sdf_mesh_audit import TorchQueries,spaced
from infrastructure.g_checkpoint import strict_network_load
from infrastructure.mesh_diagnostics import geometry
from infrastructure.signatures import require
from infrastructure.stage_sources import activate,baseline,PINS,imported_sources
from infrastructure.verified_ray_backend import load


def save_array(path,a):
    with path.open('xb') as stream:
        np.save(stream,a,allow_pickle=False)


def topology(faces):
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,inv,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    orientation=np.bincount(inv,weights=np.where(edges[:,0]<edges[:,1],1,-1),minlength=len(count))
    return dict(boundary_edges=int((count==1).sum()),nonmanifold_edges=int((count>2).sum()),
                self_edges=int((edges[:,0]==edges[:,1]).sum()),
                inconsistent_two_face_edges=int(((count==2)&(orientation!=0)).sum()))


def capture(network,science,out,report,torch,budget,progress):
    import trimesh
    import open3d
    import importlib.metadata
    from network import field
    require({n:importlib.metadata.version(n) for n in ('PyMCubes','trimesh','open3d')}==
            {'PyMCubes':'0.1.4','trimesh':'4.12.2','open3d':'0.16.0'},'Original mesh runtime differs')
    folder=out/'stages';folder.mkdir()
    shape=(512,512,512)
    raw=np.lib.format.open_memmap(folder/'raw-SDF-grid.npy',mode='w+',dtype=np.float32,shape=shape)
    outside=np.lib.format.open_memmap(folder/'outside-sphere.npy',mode='w+',dtype=bool,shape=shape)
    captured={}
    tree=ast.parse((science/'network/field.py').read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_geometry')
    mc=next(i for i,n in enumerate(fun.body) if isinstance(n,ast.Assign) and
            isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='marching_cubes')
    boundary=fun.body[mc+1].lineno
    def observer(frame,event,arg):
        if frame.f_code is field.extract_fields.__code__:
            if event=='return' and arg is not None:
                require('masked' not in captured and arg.shape==shape and arg.dtype==np.float32,'Unexpected dense field')
                for x in range(0,512,64):
                    expected=np.where(outside[x:x+64],np.float32(1),raw[x:x+64])
                    require(np.array_equal(arg[x:x+64],expected),'Sentinel field differs from captured raw SDF')
                save_array(folder/'extraction-field.npy',arg)
                captured['masked']=True
            return observer
        if frame.f_code is field.extract_geometry.__code__:
            if event=='line' and frame.f_lineno==boundary:
                require('marching_cubes' not in captured,'Repeated marching cubes capture')
                v=frame.f_locals['vertices'].copy();f=frame.f_locals['triangles'].copy()
                require(len(f)<=p.LIMITS['max_faces'],'Extraction face cap exceeded')
                save_array(folder/'marching-vertices.npy',v);save_array(folder/'marching-faces.npy',f)
                captured['marching_cubes']=(v,f)
            return observer
        return None
    def query(points):
        k=report['extraction_tiles']
        require(report['extraction_tile_attempts']<p.LIMITS['extraction_tiles'] and k<p.LIMITS['extraction_tiles'] and points.shape==(64**3,3) and not torch.is_grad_enabled(),
                'Extraction tile budget/layout/context differs')
        report['extraction_tile_attempts']+=1
        y=network.sdf_network.sdf(points)
        require(y.shape==(64**3,1) and bool(torch.isfinite(y).all().item()),'Nonfinite original extraction')
        xi,yi,zi=k//64,(k//8)%8,k%8
        sl=(slice(xi*64,(xi+1)*64),slice(yi*64,(yi+1)*64),slice(zi*64,(zi+1)*64))
        raw[sl]=y.detach().cpu().numpy().reshape(64,64,64)
        outside[sl]=(torch.norm(points,dim=-1)>=1).cpu().numpy().reshape(64,64,64)
        report['extraction_tiles']+=1
        if k%32==0:budget();progress()
        return y
    require(sys.gettrace() is None,'Untraced extraction process required')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    try:
        sys.settrace(observer)
        with torch.no_grad():
            bound=torch.ones(3)
            v,f=field.extract_geometry(-bound,bound,512,0,query,outside_val=1.)
    finally:
        sys.settrace(None)
        torch.set_default_tensor_type('torch.FloatTensor')
        raw.flush();outside.flush()
    require(report['extraction_tiles']==512 and captured.get('masked') and 'marching_cubes' in captured,
            'Incomplete capture of original extraction')
    report['marching_cubes_calls']=1
    rv,rf=captured['marching_cubes']
    require(np.array_equal(f,rf) and np.array_equal(v,rv/511*2-1),'Original voxel-to-world mapping changed')
    save_array(folder/'returned-vertices.npy',v);save_array(folder/'returned-faces.npy',f)
    returned_v,returned_f=v.copy(),f.copy()
    mesh=trimesh.Trimesh(v,f)
    pv,pf=np.asarray(mesh.vertices).copy(),np.asarray(mesh.faces).copy()
    save_array(folder/'processed-vertices.npy',pv);save_array(folder/'processed-faces.npy',pf)
    mesh.export(str(folder/'original-default.ply'))
    require(p.digest(folder/'original-default.ply')==p.MESH_SHA,'Original PLY replay differs from failed artifact')
    sv,sf=mesh_arrays(folder/'original-default.ply')
    require(np.array_equal(sf,pf) and np.array_equal(sv,pv.astype(np.float32)),'PLY coordinate/face mapping differs')
    save_array(folder/'serialized-vertices.npy',sv);save_array(folder/'serialized-faces.npy',sf)
    mapping,change=p.exact_vertex_mapping(returned_v,pv,pf,returned_f)
    save_array(folder/'returned-to-processed.npy',mapping)
    stages={name:p.mesh_stage(a,b)[0] for name,a,b in
            (('marching',rv,rf),('returned',returned_v,returned_f),('processed',pv,pf),('serialized',sv,sf))}
    report['stages']=stages
    report['vertex_processing']=change
    report['PLY_maximum_coordinate_shift']=float(abs(pv-sv.astype(float)).max())
    localization=[]
    for face in p.BAD_FACES:
        first=next((name for name in stages if face in stages[name]['zero_area_face_ids']),None)
        localization.append(dict(serialized_face_id=face,first_zero_area_stage=first))
    report['defect_localization']=localization
    p.write_ply(folder/'processed-float64.ply',pv,pf,dtype='<f8')
    double=open3d.io.read_triangle_mesh(str(folder/'processed-float64.ply'))
    require(np.array_equal(np.asarray(double.vertices),pv) and np.array_equal(np.asarray(double.triangles),pf),
            'Double-precision PLY/Open3D parity failed')
    report['float64_export']=dict(open3d_exact=True,zero_area_faces=stages['processed']['zero_area_count'],
                                 float32_BVH_input_zero_faces=p.mesh_stage(pv.astype(np.float32),pf)[0]['zero_area_count'])
    counts={}
    masked=np.load(folder/'extraction-field.npy',allow_pickle=False,mmap_mode='r')
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        interface=outside[a]!=outside[b]
        unsupported=interface&(raw[a]<-1e-7)&(raw[b]<-1e-7)&(masked[a]*masked[b]<0)
        counts[str(axis)]=int(unsupported.sum())
    report['sentinel_grid_edges_raw_negative_on_both_sides']=counts
    report['topology_original']=topology(sf)
    progress()
    return sv,sf


class Queries:
    def __init__(self,model,torch,points,gradients,record):
        self.record=record
        self.record.update(point_slots_attempted=0,gradient_slots_attempted=0)
        self.query=TorchQueries(model,torch)
        self.maximum=points;self.gradient_maximum=gradients
    def __call__(self,x,gradients=False):
        require(self.record['point_slots_attempted']+len(x)<=self.maximum and
                self.record['gradient_slots_attempted']+(len(x) if gradients else 0)<=self.gradient_maximum,
                'Raw SDF witness budget exhausted')
        self.record['point_slots_attempted']+=len(x)
        self.record['gradient_slots_attempted']+=len(x) if gradients else 0
        try:
            y,grad=self.query(x,gradients)
            self.last_inside=self.query.last_inside
            return y,grad
        finally:
            self.record.update(query_point_slots=self.query.points,completed_gradient_points=self.query.gradient_points)
    @property
    def points(self):return self.query.points
    @property
    def gradient_points(self):return self.query.gradient_points


def support_witness(points,query):
    y,grad=query(points,True)
    length=np.linalg.norm(grad.astype(float),axis=1)
    direction=np.divide(grad,length[:,None],out=np.zeros_like(points,dtype=float),where=length[:,None]>=1e-6)
    offsets=np.concatenate((points-g.QUALITY['sdf_step']*direction,points+g.QUALITY['sdf_step']*direction))
    sides,_=query(offsets,False)
    minus,plus=sides.reshape(2,-1)
    return dict(points=points,raw_sdf=y,gradient=grad,minus=minus,plus=plus,
                raw_support=p.raw_support(y,grad,minus,plus))


def broad_diagnosis(network,torch,v,f,face_mapping,out,report):
    query=Queries(network.sdf_network.sdf,torch,p.LIMITS['diagnostic_SDF_points'],p.LIMITS['diagnostic_gradient_points'],report.setdefault('diagnostic_work',{}))
    direction=p.sphere_directions()
    h=2/511
    radial=(direction[:,None,:]*(1+h*np.array([-1,-.5,0,.5,1]))[None,:,None]).astype(np.float32)
    values,_=query(radial.reshape(-1,3))
    radial_inside=query.last_inside.reshape(384,5).copy()
    require(radial_inside[:,1].all() and not radial_inside[:,3].any(),'Frozen radial probes do not straddle sphere')
    labels=p.radial_labels(values.reshape(384,5))
    sphere=support_witness(radial[:,1],query)
    tri=v[f].astype(float)
    centers=tri.mean(1)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    eligible=(centers[:,1]<-.65)&(abs(np.linalg.norm(centers,axis=1)-1)<=h)&(area>0)
    ids=np.flatnonzero(eligible)
    if len(ids):
        cumulative=np.cumsum(area[ids])
        selected=ids[np.searchsorted(cumulative,cumulative[-1]*(np.arange(256)+.5)/256)]
        surface=support_witness(centers[selected].astype(np.float32),query)
    else:
        selected=np.empty(0,np.int64)
        surface=dict(points=np.empty((0,3),np.float32),raw_support=np.empty(0,bool))
    save(out/'sphere-witnesses.npz',dict(radial_points=radial,raw_values=values.reshape(384,5),
                                       inside_sphere=radial_inside,**labels,**sphere))
    save(out/'lower-surface-witnesses.npz',dict(candidate_face_ids=selected,original_face_ids=face_mapping[selected],**surface))
    summary=dict(radial_counts={k:int(v.sum()) for k,v in labels.items()},lower_region_eligible_faces=len(ids),
                 lower_samples=len(selected),lower_raw_supported=int(surface['raw_support'].sum()),
                 lower_unsupported=int((~surface['raw_support']).sum()),sphere_raw_supported=int(sphere['raw_support'].sum()),
                 point_evaluations=query.points,gradient_points=query.gradient_points)
    summary['substantive_review_required']=bool(labels['sentinel_crossing'].any() or
                                                (~surface['raw_support']).any() or not len(ids))
    # An empty lower band is explicitly missing. It receives researcher review rather
    # than a replacement sample or a claim about an absent extracted region.
    report['broad_SDF_diagnosis']=summary
    return not summary['substantive_review_required']


def remaining_audit(network,torch,v,f,retained,out,plan,report,budget,progress):
    import open3d
    from raytracing import RayTracer
    candidate=out/'derived-exact-zero-filter.ply'
    p.write_ply(candidate,v,f)
    decoded=open3d.io.read_triangle_mesh(str(candidate))
    require(np.array_equal(np.asarray(decoded.vertices),v) and np.array_equal(np.asarray(decoded.triangles),f),
            'Derived candidate Open3D mismatch')
    cv,cf=mesh_arrays(candidate)
    require(np.array_equal(cv,v) and np.array_equal(cf,f),'Derived PLY modified geometry')
    save_array(out/'candidate-to-original-face.npy',retained)
    report['candidate_topology']=geometry(v,f,np)
    report['candidate_mesh']=p.item(candidate)
    report['mesh_outputs']={candidate.name:report['candidate_mesh']['sha256']}
    files={key:p.check(Path('/inputs/oracle')/key,row) for key,row in plan['eval_files'].items()}
    ov,of=mesh_arrays(files['mesh.ply'])
    models={name:RayTracer(a,b) for name,a,b in (('oracle',ov,of),('reconstructed',v,f))}
    surfaces={name:Surface(a,b) for name,a,b in (('oracle',ov,of),('reconstructed',v,f))}
    points=[];normals=[]
    (out/'rasters').mkdir()
    with np.load(files['calibrations-safe.npz'],allow_pickle=False) as cams:
        for camera in g.EVAL_IDS:
            require(report['primary_BVH_rays']+2*128**2<=p.LIMITS['primary_BVH_rays'],'Primary ray cap exhausted')
            require(report['CPU_ray_checks']+64<=384,'CPU ray cap exhausted before audit')
            report['primary_BVH_rays']+=2*128**2
            row,arrays=e.inspect_view(files[f'rays/{camera}.npz'],camera,
                                      (cams['pose'][int(camera)],cams['K'][int(camera)]),models,surfaces,torch)
            witnesses=sum(len(w) for w in row['cpu_closest_hit'].values())
            require(report['CPU_ray_checks']+witnesses<=384,'CPU ray cap exhausted')
            report['CPU_ray_checks']+=witnesses
            ids=np.flatnonzero(arrays['reconstructed_hit'])
            selected=ids[spaced(len(ids),16)]
            points.extend(arrays['reconstructed_position'][selected])
            normals.extend(arrays['reconstructed_normal'][selected])
            row['split']='validation' if camera=='58' else 'held_view_nvs'
            row['SDF_witness_raster_indices']=selected.tolist()
            save(out/'rasters'/f'{camera}.npz',arrays)
            report['views'].append(row);progress();budget()
            require(all(w['passed'] for ws in row['cpu_closest_hit'].values() for w in ws),'Closest-hit witness failure')
    center,n,faces,vertices=e.surface_witnesses(v,f)
    points=np.concatenate((center[faces],v[vertices],np.asarray(points))).astype(np.float32)
    normals=np.concatenate((-n[faces],np.zeros((len(vertices),3)),np.asarray(normals)))
    q=Queries(network.sdf_network.sdf,torch,p.LIMITS['original_SDF_points'],p.LIMITS['original_gradient_points'],report.setdefault('original_gate_work',{}))
    sdf,arrays=e.sdf_consistency(points,normals,q)
    save(out/'sdf-witnesses.npz',dict(**arrays,selected_faces=faces,selected_vertices=vertices))
    report['original_SDF_gate']=sdf
    report['quality_passed']=sdf['passed'] and all(v['passed'] for v in report['views'])
    require(report['primary_BVH_rays']==196608 and report['CPU_ray_checks']==384,'Incomplete raster inventory')


def execute(plan,args):
    import torch
    import yaml
    from infrastructure.geometry_training_smoke import runtime_identity
    out=Path('/outputs')
    require(out.is_dir() and not any(out.iterdir()),'Fresh native diagnosis output required')
    require(p.digest(Path('/authorization.json'))==args.authorization_sha256 and
            p.read('/authorization.json')==p.authorization(plan,args.commit,args.freeze_sha256),'Exact approval required')
    require(os.getuid()!=0 and sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and
            torch.version.cuda=='11.1' and torch.cuda.is_available() and torch.cuda.device_count()==1 and
            torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe','Unexpected native runtime')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu',
            'Original CPU float32 constructor context required')
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75)
    require(load('predicate_fp64',g.IMAGE)==plan['backend'],'Corrected BVH identity differs')
    science,manifest=activate('geometry',PINS['geometry'])
    cfg=yaml.safe_load((p.ROOT/g.CFG).read_text())
    report=dict(protocol=p.PROTOCOL,commit=args.commit,freeze_sha256=args.freeze_sha256,
                authorization_sha256=args.authorization_sha256,image_id=g.IMAGE,scientific_commit=PINS['geometry'],
                source=dict(stage='geometry',algorithm='geometry-G',scientific_commit=PINS['geometry'],
                            scientific_files=manifest['files'],source_manifest_sha256=p.digest(p.ROOT/'scientific_sources/manifest-v1.json')),
                original_configuration=cfg,stage='geometry',run_id=Path(plan['host_output']).name,
                status='incomplete',cleanup_verified=False,quality_passed=False,geometry_admitted=False,
                scientific_work_complete=False,checkpoint_loads=0,extraction_tiles=0,extraction_tile_attempts=0,primary_BVH_rays=0,
                CPU_ray_checks=0,views=[],optimizer_returns=0,shader_calls=0,runtime=runtime_identity(torch))
    start=time.monotonic()
    def progress():
        temp=out/'receipt.next';temp.unlink(missing_ok=True)
        report['wall_seconds']=time.monotonic()-start
        p.write(temp,report);temp.replace(out/'receipt.json')
    def budget():
        require(time.monotonic()-start<1740 and
                sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())<=p.RESOURCES['output_bytes'],
                'Diagnostic runtime/output limit exceeded')
    def work():
        from network.renderer import NeROShapeRenderer
        workdir=out/'workspace';workdir.mkdir()
        (workdir/'assets').symlink_to(science/'assets',target_is_directory=True)
        os.chdir(workdir)
        network=NeROShapeRenderer(cfg,training=False)
        require(report['checkpoint_loads']==0,'One checkpoint load only')
        path=p.check(Path('/inputs/checkpoint.pth'),plan['checkpoint'])
        report['checkpoint_loads']+=1
        identity,saved=own_checkpoint(path,300000,network,torch,expected=plan['checkpoint'])
        report['strict_checkpoint']=strict_network_load(network,saved['network_state_dict'],torch)
        del saved
        network.eval().cuda()
        require(state_sha(network.state_dict())==plan['network_sha256'],'Accepted network state differs')
        report['checkpoint']=identity
        v,f=capture(network,science,out,report,torch,budget,progress)
        cv,cf,retained=p.candidate_arrays(v,f)
        require(np.array_equal(cv,v) and np.array_equal(cf,f[retained]),'Positive-area support changed')
        report['positive_support_preserved_exactly']=True
        report['candidate_face_count']=len(retained)
        budget()
        if broad_diagnosis(network,torch,cv,cf,retained,out,report):
            report['decision']='conditional_numerical_correction_and_original_audit'
            remaining_audit(network,torch,cv,cf,retained,out,plan,report,budget,progress)
            report['decision']='PROCEED_pending_researcher_review' if report['quality_passed'] else 'BLOCKED_original_quality_gate'
        else:
            report['decision']='BLOCKED_substantive_geometry_review_required'
        require(state_sha(network.state_dict())==plan['network_sha256'],'Diagnosis mutated network state')
        report['network_state_preserved']=True
        report['scientific_work_complete']=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    progress()
    try:
        try:
            run_with_cleanup(work,torch)
        finally:
            torch.set_default_tensor_type('torch.FloatTensor')
            sys.settrace(None)
        report['cleanup_verified']=torch.tensor(0.).device.type=='cpu' and sys.gettrace() is None
        require(report['cleanup_verified'],'Cleanup incomplete')
        budget();baseline('geometry',PINS['geometry']);p.freeze(args.freeze_sha256)
        report['artifacts']={x.relative_to(out).as_posix():p.item(x) for x in sorted(out.rglob('*'))
                             if x.is_file() and x.resolve()==x and x.name!='receipt.json'}
        report.update(status='complete',peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      imported_scientific_modules=imported_sources(science))
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        progress()


def verify(plan,args):
    root=Path(plan['host_output']);out=root/'run'
    require(p.read(root/'authorization.json')==p.authorization(plan,args.commit,args.freeze_sha256),'Approval differs')
    require((root/'exit-code.txt').read_text().strip()=='0' and
            (root/'container-absent.txt').read_text().strip()=='absent','Missing host completion/cleanup')
    r=p.read(out/'receipt.json')
    require(r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified'] and
            r['freeze_sha256']==args.freeze_sha256 and r['commit']==args.commit and r['checkpoint_loads']==1 and
            r['checkpoint']==plan['checkpoint'] and r['network_state_preserved'] and r['extraction_tiles']==512
            and r['extraction_tile_attempts']==512 and r['marching_cubes_calls']==1,
            'Missing completed diagnostic evidence')
    for name,row in r['artifacts'].items():p.check(out/name,row)
    require(not (out/'handoff.json').exists(),'Handoff cannot precede offline verification')
    # Recheck all numerical candidate decisions from saved, immutable arrays.
    for name in ('marching','returned','processed','serialized'):
        vertices=np.load(out/'stages'/f'{name}-vertices.npy',allow_pickle=False)
        faces=np.load(out/'stages'/f'{name}-faces.npy',allow_pickle=False)
        require(p.mesh_stage(vertices,faces)[0]==r['stages'][name],'Mesh-stage replay differs')
    with np.load(out/'sphere-witnesses.npz',allow_pickle=False) as z:
        labels=p.radial_labels(z['raw_values'])
        require(all(np.array_equal(z[k],v) for k,v in labels.items()),'Sphere decision replay differs')
    with np.load(out/'sphere-witnesses.npz',allow_pickle=False) as z:
        support=p.raw_support(z['raw_sdf'],z['gradient'],z['minus'],z['plus'])
        require(np.array_equal(support,z['raw_support']),'Sphere raw-support replay differs')
    with np.load(out/'lower-surface-witnesses.npz',allow_pickle=False) as z:
        if len(z['points']):
            supported=p.raw_support(z['raw_sdf'],z['gradient'],z['minus'],z['plus'])
            require(np.array_equal(supported,z['raw_support']),'Surface raw-support replay differs')
        else:
            supported=np.empty(0,bool)
    substantive=bool(labels['sentinel_crossing'].any() or (~supported).any() or not len(supported))
    require(substantive==r['broad_SDF_diagnosis']['substantive_review_required'],
            'Substantive-geometry review decision differs')
    require(not substantive or (not r['quality_passed'] and not r['views'] and
                                r['decision']=='BLOCKED_substantive_geometry_review_required'),
            'Execution continued beyond substantive geometry stop')
    if r['primary_BVH_rays']:
        original_v,original_f=mesh_arrays(out/'stages/original-default.ply')
        cv,cf,indices=p.candidate_arrays(original_v,original_f)
        candidate_v,candidate_f=mesh_arrays(out/'derived-exact-zero-filter.ply')
        require(np.array_equal(cv,candidate_v) and np.array_equal(cf,candidate_f) and
                np.array_equal(indices,np.load(out/'candidate-to-original-face.npy',allow_pickle=False)),
                'Numerical correction changed positive-area support')
    if r['quality_passed']:
        from experiments.bell_geometry_gate10_v1 import verify_rasters,verify_sdf
        require(len(r['views'])==6 and r['primary_BVH_rays']==196608 and r['CPU_ray_checks']==384,
                'Incomplete original audit')
        r['split_summaries']=verify_rasters(out,r['views'])
        verify_sdf(out/'sdf-witnesses.npz',r['original_SDF_gate'])
        require(all(v['passed'] for v in r['views']) and r['original_SDF_gate']['passed'],'Original gates failed')
        p.check(out/'derived-exact-zero-filter.ply',r['candidate_mesh'])
        handoff_record=dict(schema_version=1,geometry_algorithm='geometry-G',scientific_commit=PINS['geometry'],
                            geometry_run_id=r['run_id'],geometry_run_manifest={'root':'geometry','path':'receipt.json'},
                            geometry_run_manifest_sha256=p.digest(out/'receipt.json'),mesh_sha256=r['candidate_mesh']['sha256'],
                            database_name='syn/bell',coordinates='Unchanged admitted Bell XYZ',
                            scale='Original normalized scene units',normals='Pinned A negates BVH face normals; signed gates passed',
                            orientation='Original winding; no normal flips',
                            material_preprocessing='A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write',
                            mesh_derivation=dict(original_failed_mesh_sha256=p.MESH_SHA,
                                                 removed_exact_zero_faces=p.BAD_FACES,
                                                 candidate_face_mapping=p.item(out/'candidate-to-original-face.npy')))
        # The existing validator requires a complete producing receipt binding the mesh.
        require(r.get('mesh_outputs',{}).get('derived-exact-zero-filter.ply')==r['candidate_mesh']['sha256'],
                'Producing receipt does not bind derived mesh')
        from infrastructure.paths import PathResolver
        from infrastructure.stage_workspace import handoff
        p.write(out/'handoff.json',handoff_record)
        resolver=PathResolver(dict(datasets=Path(plan['acquisition'])/'datasets',thesis=p.PRIOR/'train',geometry=out),
                              root/'future-material')
        handoff(out/'handoff.json',out/'derived-exact-zero-filter.ply',resolver,'syn/bell')
    p.write(root/'verification.json',dict(status=r['decision'],receipt=p.item(out/'receipt.json'),
                                         quality_passed=r['quality_passed'],geometry_admitted=False,
                                         handoff_created=(out/'handoff.json').exists()))
    print(root/'verification.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','execute','verify'))
    parser.add_argument('--commit',required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args();plan=p.freeze(args.freeze_sha256)
    if args.phase=='execute':execute(plan,args);return
    auth=args.authorization or Path(plan['host_output'])/'authorization.json'
    require(p.digest(auth)==args.authorization_sha256 and p.read(auth)==p.authorization(plan,args.commit,args.freeze_sha256),
            'Exact diagnostic approval required')
    if args.phase=='verify':verify(plan,args);return
    out=Path(plan['host_output'])
    require(not os.path.lexists(out),'Fresh diagnostic output required')
    p.verify_host_inputs(plan)
    print(out);print(plan['image']);print(plan['acquisition']+'/geometry')


if __name__=='__main__':
    main()
