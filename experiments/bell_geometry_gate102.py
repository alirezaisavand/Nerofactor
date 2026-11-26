"""One authorized saved-extraction continuation and original Gate 10 geometry audit."""
import argparse
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate102 as p
from benchmarks import bell_geometry_gate10_v1 as original
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import Surface
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments import bell_geometry_gate10_evaluate_v1 as audit
from experiments.bell_geometry_gate10_train_v1 import own_checkpoint,state_sha
from experiments.bell_geometry_gate101 import Queries,save_array
from experiments.donut_reconstructed_geometry_v1 import save,interrupted
from infrastructure.g_checkpoint import strict_network_load
from infrastructure.geometry_sdf_mesh_audit import spaced
from infrastructure.mesh_diagnostics import geometry
from infrastructure.signatures import require
from infrastructure.stage_sources import activate,baseline,PINS
from infrastructure.verified_ray_backend import load


def read_completed_extraction(captured,root):
    arrays={name:np.load(p.check(root/name,identity),allow_pickle=False,mmap_mode='r')
            for name,identity in captured['arrays'].items()}
    voxel,world,faces=(arrays[name] for name in ('raw-voxel-vertices.npy','raw-world-vertices.npy','raw-faces.npy'))
    layout=captured['layout']
    require(voxel.shape==world.shape==(layout['vertices'],3) and faces.shape==(layout['faces'],3) and
            str(voxel.dtype)==str(world.dtype)==layout['vertex_dtype'] and str(faces.dtype)==layout['face_dtype'] and
            faces.min()>=0 and faces.max()<len(voxel) and np.array_equal(world,voxel/511*2-1),
            'Captured extraction layout or coordinate transform differs')
    return voxel,world,faces


def extract(plan,out,r,budget,progress):
    raw=np.load(p.check(Path('/inputs/raw.npy'),plan['raw_grid']),allow_pickle=False,mmap_mode='r')
    outside=np.load(p.check(Path('/inputs/outside.npy'),plan['outside_mask']),allow_pickle=False,mmap_mode='r')
    require(raw.shape==outside.shape==(512,512,512) and raw.dtype==np.float32 and outside.dtype==bool
            and np.isfinite(raw).all(),'Captured grid layout/values differ')
    valid,any_inside=p.domain_cells(outside)
    require(p.cell_counts(valid,any_inside)==plan['domain']['cell_counts'],'Supported-cell inventory differs')
    save_array(out/'supported-cells.npy',valid)
    r['domain_cells']=p.cell_counts(valid,any_inside)
    voxel,world,faces=read_completed_extraction(plan['completed_extraction'],Path('/inputs/capture'))
    require(r['MC_attempts']==r['MC_returns']==0,'No new extraction authorized')
    r['reused_completed_extraction']=True
    require(8<len(faces)<=p.LIMITS['max_faces'] and np.isfinite(voxel).all() and
            voxel.min()>=0 and voxel.max()<=511,'Raw extraction empty, invalid or over budget')
    r['raw_mesh_layout']=dict(vertices=len(voxel),faces=len(faces),vertex_dtype=str(voxel.dtype),face_dtype=str(faces.dtype))
    save_array(out/'raw-voxel-vertices.npy',voxel)
    save_array(out/'raw-world-vertices.npy',world)
    save_array(out/'raw-faces.npy',faces)
    keep,reason,possible,good=p.face_domain(voxel,faces,valid,any_inside)
    save(out/'face-domain.npz',dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good))
    r['domain_faces']=dict(global_raw=len(faces),retained=int(keep.sum()),
                           rejected_mixed_generators=int((reason==1).sum()),
                           rejected_boundary_cells=int((reason==2).sum()),
                           rejected_outside_cells=int((reason==3).sum()),
                           ambiguous_generator_faces=int((possible>1).sum()))
    progress();budget()
    require(keep.any(),'No raw zero-level faces satisfy the supported-domain policy')
    before,labels,edges=p.topology(world,faces[keep])
    save(out/'topology-before-cleanup.npz',dict(face_labels=labels,boundary_edges=edges))
    r['topology_before_cleanup']=before
    v,f,mapping,retained,removed,info=p.numerical_cleanup(world,faces,keep)
    save(out/'cleanup-mappings.npz',dict(raw_vertex_to_candidate=mapping,
         candidate_face_to_raw=retained,removed_exact_zero_raw_faces=removed))
    save_array(out/'candidate-vertices.npy',v);save_array(out/'candidate-faces.npy',f)
    r['cleanup']=info
    require(8<len(f)<=p.LIMITS['max_faces'],'Insufficient native-compatible candidate faces')
    r['candidate_geometry']=geometry(v,f,np)
    after,labels,edges=p.topology(v,f)
    save(out/'topology-after-cleanup.npz',dict(face_labels=labels,boundary_edges=edges))
    r['topology_after_cleanup']=after
    candidate=out/'domain-aware-300000.ply'
    p.previous.write_ply(candidate,v,f)
    cv,cf=mesh_arrays(candidate)
    require(np.array_equal(v,cv) and np.array_equal(f,cf),'Float32 PLY roundtrip changed candidate')
    import open3d
    downstream=open3d.io.read_triangle_mesh(str(candidate))
    require(np.array_equal(np.asarray(downstream.vertices),v) and np.array_equal(np.asarray(downstream.triangles),f),
            'Original Open3D/native mesh interface differs')
    r['open3d_exact']=True
    r['mesh_outputs']={candidate.name:p.digest(candidate)}
    original=Path('/inputs/original.ply');p.check(original,plan['original_mesh'])
    ov,of=mesh_arrays(original)
    oa=p.areas(ov,of)
    old_xyz=ov[of].astype(float)
    band=(abs(np.linalg.norm(old_xyz,axis=2)-1)<=2/511).all(axis=1)
    xyz=v[f].astype(float)
    new_band=(abs(np.linalg.norm(xyz,axis=2)-1)<=2/511).all(axis=1)
    area=p.areas(v,f)
    original_topology,old_labels,old_edges=p.topology(ov,of[oa>0])
    save(out/'historical-positive-area-topology.npz',dict(face_labels=old_labels,boundary_edges=old_edges,
         face_to_original=np.flatnonzero(oa>0)))
    r['historical_positive_area_topology']=original_topology
    r['historical_comparison']=dict(original_faces=len(of),candidate_faces=len(f),
        original_zero_area_faces=int((oa==0).sum()),candidate_zero_area_faces=int((area==0).sum()),
        original_area=float(oa.sum()),candidate_area=float(area.sum()),
        original_near_sphere_area=float(oa[band].sum()),candidate_near_sphere_area=float(area[new_band].sum()),
        comparison_scope='Saved positive-area geometry statistics. Native audit uses the valid candidate.')
    progress();budget()
    return v,f,valid,any_inside


def geometry_audit(v,f,valid,any_inside,plan,out,r,torch,budget,progress):
    from raytracing import RayTracer
    files={name:p.check(Path('/inputs/oracle')/name,row) for name,row in plan['eval_files'].items()}
    ov,of=mesh_arrays(files['mesh.ply'])
    models={name:RayTracer(a,b) for name,a,b in (('oracle',ov,of),('reconstructed',v,f))}
    surfaces={name:Surface(a,b) for name,a,b in (('oracle',ov,of),('reconstructed',v,f))}
    points=[];normals=[]
    (out/'rasters').mkdir()
    with np.load(files['calibrations-safe.npz'],allow_pickle=False) as cameras:
        for camera in original.EVAL_IDS:
            require(r['primary_BVH_rays']+2*128**2<=p.LIMITS['primary_BVH_rays'] and
                    r['CPU_ray_checks']+64<=p.LIMITS['CPU_ray_checks'],'Original ray budgets exhausted')
            r['primary_BVH_rays']+=2*128**2
            row,arrays=audit.inspect_view(files[f'rays/{camera}.npz'],camera,
                           (cameras['pose'][int(camera)],cameras['K'][int(camera)]),models,surfaces,torch)
            r['CPU_ray_checks']+=64
            hit=arrays['oracle_hit']
            domain=np.zeros_like(hit)
            domain[hit]=p.point_domain(arrays['oracle_position'][hit],valid,any_inside)
            sphere=np.zeros_like(hit)
            sphere[hit]=np.linalg.norm(arrays['oracle_position'][hit].astype(float),axis=1)<1
            arrays.update(oracle_conservative_domain=domain,oracle_float64_inside_sphere=sphere)
            missing=hit&~arrays['reconstructed_hit']
            row.update(split='validation' if camera=='58' else 'held_view_nvs',
                       oracle_domain_supported=int(domain.sum()),oracle_domain_unsupported=int((hit&~domain).sum()),
                       missing_inside_domain=int((missing&domain).sum()),missing_outside_domain=int((missing&~domain).sum()),
                       oracle_float64_outside_sphere=int((hit&~sphere).sum()))
            ids=np.flatnonzero(arrays['reconstructed_hit'])
            selected=ids[spaced(len(ids),16)] if len(ids) else np.empty(0,np.int64)
            points.extend(arrays['reconstructed_position'][selected])
            normals.extend(arrays['reconstructed_normal'][selected])
            row['SDF_witness_raster_indices']=selected.tolist()
            save(out/'rasters'/f'{camera}.npz',arrays)
            r['views'].append(row);progress();budget()
            require(all(w['passed'] for group in row['cpu_closest_hit'].values() for w in group),
                    'Original independent closest-hit witness failed')
    require(r['primary_BVH_rays']==196608 and r['CPU_ray_checks']==384,'Incomplete six-raster numerical audit')
    return np.asarray(points).reshape(-1,3),np.asarray(normals).reshape(-1,3)


def sdf_audit(v,f,points,normals,plan,science,out,r,torch,progress):
    # This is the sole network/checkpoint phase. Extraction has used only saved arrays.
    import yaml
    from network.renderer import NeROShapeRenderer
    cfg=yaml.safe_load((p.ROOT/original.CFG).read_text())
    work=out/'workspace';work.mkdir()
    (work/'assets').symlink_to(science/'assets',target_is_directory=True)
    os.chdir(work)
    network=NeROShapeRenderer(cfg,training=False)
    checkpoint=p.check(Path('/inputs/checkpoint.pth'),plan['checkpoint'])
    require(r['checkpoint_loads']==0,'Only one accepted endpoint load')
    r['checkpoint_loads']+=1
    identity,state=own_checkpoint(checkpoint,300000,network,torch,expected=plan['checkpoint'])
    r['strict_checkpoint']=strict_network_load(network,state['network_state_dict'],torch)
    del state
    network.eval().cuda()
    require(state_sha(network.state_dict())==plan['network_sha256'],'Accepted neural identity differs')
    r['checkpoint']=identity;progress()
    centers,n,faces,vertices=audit.surface_witnesses(v,f)
    x=np.concatenate((centers[faces],v[vertices],points)).astype(np.float32)
    normal=np.concatenate((-n[faces],np.zeros((len(vertices),3)),normals))
    q=Queries(network.sdf_network.sdf,torch,1536,512,r.setdefault('SDF_work',{}))
    result,arrays=audit.sdf_consistency(x,normal,q)
    save(out/'sdf-witnesses.npz',dict(**arrays,selected_faces=faces,selected_vertices=vertices))
    r['original_SDF_gate']=result
    require(state_sha(network.state_dict())==plan['network_sha256'],'Consistency evaluation changed neural state')
    r['network_state_preserved']=True


def native(plan,args):
    import torch
    import yaml
    from infrastructure.geometry_training_smoke import runtime_identity
    out=Path('/outputs')
    require(out.is_dir() and not any(out.iterdir()),'Fresh execution output required')
    auth=Path('/authorization.json')
    require(p.digest(auth)==args.authorization_sha256 and p.read(auth)==p.authorization(plan,args.commit,args.freeze_sha256),
            'Exact Gate 10.2 approval required')
    require(os.getuid()!=0 and sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and
            torch.version.cuda=='11.1' and torch.cuda.is_available() and torch.cuda.device_count()==1 and
            torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe','Frozen native runtime required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','CPU float32 construction required')
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75)
    require(load('predicate_fp64',plan['image'])==plan['backend'],'Frozen corrected BVH differs')
    science,manifest=activate('geometry',PINS['geometry'])
    r=dict(protocol=p.PROTOCOL,commit=args.commit,freeze_sha256=args.freeze_sha256,
           authorization_sha256=args.authorization_sha256,run_id=Path(plan['host_output']).name,
           image_id=plan['image'],backend=plan['backend'],scientific_commit=PINS['geometry'],stage='geometry',
           source=dict(stage='geometry',algorithm='geometry-G',scientific_commit=PINS['geometry'],
                       scientific_files=manifest['files'],source_manifest_sha256=p.digest(p.ROOT/'scientific_sources/manifest-v1.json')),
           original_configuration=yaml.safe_load((p.ROOT/original.CFG).read_text()),
           status='incomplete',scientific_work_complete=False,cleanup_verified=False,quality_passed=False,
           geometry_admitted=False,checkpoint_loads=0,network_queries_for_extraction=0,
           optimizer_returns=0,shader_calls=0,MC_attempts=0,MC_returns=0,
           primary_BVH_rays=0,CPU_ray_checks=0,views=[],runtime=runtime_identity(torch),
           raw_grid_identity=plan['raw_grid'],domain_policy=plan['cell_policy'],numerical_policy=plan['numerical_policy'])
    r['completed_extraction_input']=plan['completed_extraction']
    start=time.monotonic()
    def progress():
        temp=out/'receipt.next';temp.unlink(missing_ok=True)
        r['wall_seconds']=time.monotonic()-start
        p.write(temp,r);temp.replace(out/'receipt.json')
    def budget():
        require(time.monotonic()-start<p.RESOURCES['inner'] and
                sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())<=p.RESOURCES['output_bytes'],
                'Runtime/output budget exceeded')
    def work():
        v,f,valid,any_inside=extract(plan,out,r,budget,progress)
        points,normals=geometry_audit(v,f,valid,any_inside,plan,out,r,torch,budget,progress)
        sdf_audit(v,f,points,normals,plan,science,out,r,torch,progress)
        r['quality_passed']=r['original_SDF_gate']['passed'] and all(row['passed'] for row in r['views'])
        r['decision']='PROCEED_pending_researcher_admission' if r['quality_passed'] else 'BLOCKED_original_geometry_quality_gate'
        r['scientific_work_complete']=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    progress()
    try:
        try:run_with_cleanup(work,torch)
        finally:torch.set_default_tensor_type('torch.FloatTensor')
        r['cleanup_verified']=torch.tensor(0.).device.type=='cpu' and sys.gettrace() is None
        require(r['cleanup_verified'],'Native cleanup unverified')
        budget();baseline('geometry',PINS['geometry']);p.freeze(args.freeze_sha256)
        p.check(Path('/inputs/raw.npy'),plan['raw_grid']);p.check(Path('/inputs/outside.npy'),plan['outside_mask'])
        r['artifacts']={x.relative_to(out).as_posix():p.item(x) for x in sorted(out.rglob('*'))
                        if x.is_file() and x.resolve()==x and x.name!='receipt.json'}
        r.update(status='complete',peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        r.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:progress()


def verify(plan,args):
    from experiments.bell_geometry_gate10_v1 import verify_rasters,verify_sdf
    root=Path(plan['host_output']);out=root/'run'
    require((root/'exit-code.txt').read_text().strip()=='0' and
            (root/'container-absent.txt').read_text().strip()=='absent' and
            (root/'image-id.txt').read_text().strip()==plan['image'],'Missing host image/success/cleanup receipt')
    r=p.read(out/'receipt.json')
    require(r['commit']==args.commit and r['freeze_sha256']==args.freeze_sha256 and r['status']=='complete' and
            r['authorization_sha256']==args.authorization_sha256 and r['image_id']==plan['image'] and
            r['backend']==plan['backend'] and r['raw_grid_identity']==plan['raw_grid'] and
            r['domain_policy']==plan['cell_policy'] and r['numerical_policy']==plan['numerical_policy'] and
            r['optimizer_returns']==r['shader_calls']==0 and r['open3d_exact'] and
            r['scientific_work_complete'] and r['cleanup_verified'] and r['checkpoint_loads']==1 and
            r['MC_attempts']==r['MC_returns']==0 and r['reused_completed_extraction'] and
            r['completed_extraction_input']==plan['completed_extraction'] and r['network_queries_for_extraction']==0 and
            r['raw_mesh_layout']==plan['completed_extraction']['layout'] and
            r['network_state_preserved'] and r['primary_BVH_rays']==196608 and r['CPU_ray_checks']==384,
            'Incomplete frozen extraction/audit receipt')
    require(r['checkpoint']==plan['checkpoint'] and r['SDF_work']['point_slots_attempted']<=p.LIMITS['SDF_points'] and
            r['SDF_work']['gradient_slots_attempted']<=p.LIMITS['gradient_points'],
            'Checkpoint or consistency work differs')
    for name,row in r['artifacts'].items():p.check(out/name,row)
    valid=np.load(out/'supported-cells.npy',allow_pickle=False,mmap_mode='r')
    outside=np.load(Path(plan['diagnostic'])/'run/stages/outside-sphere.npy',allow_pickle=False,mmap_mode='r')
    expected,any_inside=p.domain_cells(outside)
    require(np.array_equal(expected,valid),'Domain cell replay differs')
    voxel=np.load(out/'raw-voxel-vertices.npy',allow_pickle=False)
    world=np.load(out/'raw-world-vertices.npy',allow_pickle=False)
    faces=np.load(out/'raw-faces.npy',allow_pickle=False)
    for name,value in (('raw-voxel-vertices.npy',voxel),('raw-world-vertices.npy',world),('raw-faces.npy',faces)):
        source=p.check(Path(plan['completed_extraction']['root'])/'run'/name,plan['completed_extraction']['arrays'][name])
        require(np.array_equal(np.load(source,allow_pickle=False,mmap_mode='r'),value),'Captured extraction replay differs')
    require(np.array_equal(world,voxel/511*2-1),'Coordinate transformation changed')
    keep,reason,possible,good=p.face_domain(voxel,faces,valid,any_inside)
    with np.load(out/'face-domain.npz',allow_pickle=False) as z:
        require(all(np.array_equal(z[k],v) for k,v in dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good).items()),
                'Face support decision replay differs')
    v,f,mapping,retained,removed,info=p.numerical_cleanup(world,faces,keep)
    actual_v,actual_f=mesh_arrays(out/'domain-aware-300000.ply')
    require(np.array_equal(v,actual_v) and np.array_equal(f,actual_f) and info==r['cleanup'],
            'Cleanup changed positive-area support')
    with np.load(out/'cleanup-mappings.npz',allow_pickle=False) as z:
        require(np.array_equal(z['raw_vertex_to_candidate'],mapping) and np.array_equal(z['candidate_face_to_raw'],retained)
                and np.array_equal(z['removed_exact_zero_raw_faces'],removed),'Cleanup maps differ')
    topology,labels,boundary=p.topology(v,f)
    require(topology==r['topology_after_cleanup'],'Candidate topology differs')
    with np.load(out/'topology-after-cleanup.npz',allow_pickle=False) as z:
        require(np.array_equal(labels,z['face_labels']) and np.array_equal(boundary,z['boundary_edges']),'Component/boundary maps differ')
    summary=verify_rasters(out,r['views'])
    verify_sdf(out/'sdf-witnesses.npz',r['original_SDF_gate'])
    for row in r['views']:
        with np.load(out/'rasters'/f"{row['camera']}.npz",allow_pickle=False) as z:
            hit=z['oracle_hit'];domain=np.zeros_like(hit)
            domain[hit]=p.point_domain(z['oracle_position'][hit],valid,any_inside)
            require(np.array_equal(domain,z['oracle_conservative_domain']),'Oracle domain cohort differs')
    passed=all(row['passed'] for row in r['views']) and r['original_SDF_gate']['passed']
    require(passed==r['quality_passed'],'Frozen quality decision differs')
    record=dict(status=r['decision'],quality_passed=passed,geometry_admitted=False,
                receipt=p.item(out/'receipt.json'),split_summaries=summary,handoff_created=False)
    if passed:
        from infrastructure.paths import PathResolver
        from infrastructure.stage_workspace import handoff
        mesh=out/'domain-aware-300000.ply'
        value=dict(schema_version=1,geometry_algorithm='geometry-G',scientific_commit=PINS['geometry'],
                   geometry_run_id=r['run_id'],geometry_run_manifest={'root':'geometry','path':'receipt.json'},
                   geometry_run_manifest_sha256=record['receipt']['sha256'],mesh_sha256=p.digest(mesh),
                   database_name='syn/bell',coordinates='Original admitted Bell XYZ',scale='Original normalized scene units',
                   normals='Pinned A negates raw BVH face normals; frozen signed gates passed',
                   orientation='Inherited marching-cubes winding; no normal flips',
                   material_preprocessing='A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write',
                   mesh_derivation=dict(policy=p.PROTOCOL,raw_grid=plan['raw_grid'],cell_policy=plan['cell_policy'],
                                        numerical_policy=plan['numerical_policy'],mapping=p.item(out/'cleanup-mappings.npz'),
                                        completed_extraction=plan['completed_extraction']),
                   researcher_admission_pending=True)
        p.write(out/'handoff.json',value)
        resolver=PathResolver(dict(datasets=Path(plan['acquisition'])/'datasets',thesis=p.previous.PRIOR/'train',geometry=out),
                              root/'future-material')
        handoff(out/'handoff.json',mesh,resolver,'syn/bell')
        record['handoff_created']=True;record['handoff']=p.item(out/'handoff.json')
    p.write(root/'verification.json',record)
    print(root/'verification.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','execute','verify'))
    parser.add_argument('--commit',required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--authorization-sha256',required=True);parser.add_argument('--authorization',type=Path)
    args=parser.parse_args();plan=p.freeze(args.freeze_sha256)
    if args.phase=='execute':native(plan,args);return
    root=Path(plan['host_output']);auth=args.authorization or root/'authorization.json'
    require(p.digest(auth)==args.authorization_sha256 and p.read(auth)==p.authorization(plan,args.commit,args.freeze_sha256),
            'Exact source/freeze/output approval required')
    if args.phase=='verify':p.host_inputs(plan);verify(plan,args);return
    require(not os.path.lexists(root),'Fresh output required')
    p.host_inputs(plan)
    print(root);print(plan['image']);print(plan['acquisition']+'/geometry')


if __name__=='__main__':
    main()
