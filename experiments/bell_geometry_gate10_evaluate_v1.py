"""Extract the fixed native G endpoint and evaluate it on the frozen Bell rays."""
from pathlib import Path
import importlib.metadata

import numpy as np

from audits.bell_calibration_e11 import rays
from benchmarks import bell_geometry_gate10_v1 as g
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import Surface, sampled_pixels
from experiments.bell_geometry_gate10_train_v1 import own_checkpoint, state_sha
from experiments.donut_reconstructed_geometry_v1 import (cpu_witness, finite_hits, reconstructed_normals,
                                                         save, trace)
from infrastructure.g_checkpoint import strict_network_load
from infrastructure.geometry_sdf_mesh_audit import TorchQueries, spaced
from infrastructure.mesh_diagnostics import geometry
from infrastructure.signatures import require


MESH = 'bell_shape-300000.ply'


def extraction(network, cfg, out, torch, report, budget, progress):
    import open3d
    import trimesh
    from network.field import extract_geometry

    packages={n:importlib.metadata.version(n) for n in ('PyMCubes','trimesh','open3d')}
    require(packages['PyMCubes']=='0.1.4','Unexpected PyMCubes version')
    report['mesh_packages']=packages

    def query(points):
        require(not torch.is_grad_enabled() and points.shape==(64**3,3)
                and points.device.type=='cuda' and points.dtype==torch.float32, 'Unexpected original G extraction tile')
        tiles=report.get('extraction_tiles',0)
        require(tiles<512,'Extraction tile cap exceeded')
        y=network.sdf_network.sdf(points)
        require(y.shape==(64**3,1) and bool(torch.isfinite(y).all().item()), 'Nonfinite or invalid G extraction values')
        report['extraction_tiles']=tiles+1
        if tiles%32==0:
            budget()
            progress()
        return y

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    try:
        box=torch.ones(3)
        vertices,faces=extract_geometry(-box,box,g.RESOLUTION,0,query,outside_val=1.)
    finally:
        torch.set_default_tensor_type('torch.FloatTensor')
    require(report.get('extraction_tiles')==512 and 8<len(faces)<=g.LIMITS['max_mesh_faces'],
            'Unexpected original G tile count or mesh size')
    report['mesh_raw_counts']=dict(vertices=len(vertices),faces=len(faces))
    dest=out/MESH
    # Same Trimesh constructor and PLY export defaults as the pinned extraction CLI.
    trimesh.Trimesh(vertices,faces).export(str(dest))
    v,faces=mesh_arrays(dest)
    report['mesh_topology']=geometry(v,faces,np)
    downstream=open3d.io.read_triangle_mesh(str(dest))
    require(np.array_equal(np.asarray(downstream.vertices),v) and np.array_equal(np.asarray(downstream.triangles),faces),
            'Pinned A Open3D mesh ordering/coordinates differ from extracted PLY')
    report.update(open3d_A_mesh_parity=True, mesh_outputs={MESH:g.digest(dest)})
    progress()
    return v,faces


def sdf_consistency(points, normals, query):
    """Raw SDF witnesses. The unit-sphere sentinel never substitutes for learned SDF."""
    quality=g.QUALITY
    require(quality['minimum_sdf_interior_points']<=len(points)<=g.LIMITS['sdf_witness_points_max'],
            'SDF witness inventory outside frozen limits')
    y,gradient=query(points,True)
    inside=query.last_inside.copy()
    length=np.linalg.norm(gradient.astype(float),axis=1)
    good=length>=quality['sdf_min_gradient']
    unit=np.divide(gradient,length[:,None],out=np.zeros_like(points,dtype=float),where=good[:,None])
    sign_steps=quality['sdf_step']*unit
    minus,plus=np.concatenate((points-sign_steps,points+sign_steps)).reshape(2,len(points),3)
    values,_=query(np.concatenate((minus,plus)),False)
    vminus,vplus=values.reshape(2,-1)
    in_minus,in_plus=query.last_inside.reshape(2,-1)
    eligible=inside&in_minus&in_plus
    rim=abs(np.linalg.norm(points.astype(float),axis=1)-1)<=2*quality['sdf_step']
    crosses=(vminus<-quality['sdf_min_sign'])&(vplus>quality['sdf_min_sign'])
    proxy=np.divide(abs(y),length,out=np.full(len(points),np.inf),where=good)
    agrees=good&crosses&(proxy<=quality['sdf_step'])
    alignment=(normals*unit).sum(1)
    oriented=np.linalg.norm(normals,axis=1)>0
    signed_failed=eligible&oriented&(alignment<np.cos(np.radians(quality['normal_degrees'])))
    report=dict(samples=len(points),raw_SDF_evaluations=query.points,gradient_evaluations=query.gradient_points,
                eligible=int(eligible.sum()),outside_or_crosses_sphere=int((~eligible).sum()),
                near_zero_gradients=int((~good).sum()),interior_failures=int((eligible&~agrees).sum()),
                near_sphere_samples=int(rim.sum()),near_sphere_unsupported=int((rim&~agrees).sum()),
                oriented_interior_samples=int((eligible&oriented).sum()),signed_orientation_failures=int(signed_failed.sum()),
                first_order_distance=g.distribution(proxy[good]),signed_alignment=g.distribution(alignment[eligible&oriented]))
    report['passed']=(report['eligible']>=quality['minimum_sdf_interior_points'] and int(query.points)==3*len(points)
                      <=g.LIMITS['sdf_witness_evaluations_max'] and report['oriented_interior_samples']>=32
                      and all(report[k]<=quality['max_sdf_failures'] for k in
                              ('interior_failures','near_sphere_unsupported','signed_orientation_failures')))
    arrays=dict(points=points,raw_sdf=y,gradient=gradient,gradient_length=length,raw_sdf_inward=vminus,
                raw_sdf_outward=vplus,inside_unit_ball=inside,inward_inside_unit_ball=in_minus,outward_inside_unit_ball=in_plus,
                eligible=eligible,near_sphere=rim,
                consistent=agrees,native_normals=normals,signed_alignment=alignment,signed_failed=signed_failed)
    return report,arrays


def inspect_view(source, c, calibration, models, surfaces, torch):
    with np.load(source,allow_pickle=False) as z:
        archived={k:z[k] for k in ('origins','directions','hit','depth','positions','raw_normals','normals')}
    orig,dirs=archived['origins'],archived['directions']
    pose,K=calibration
    ref_o,ref_d=rays(pose,K,g.SIZE,g.SIZE)
    require(orig.shape==dirs.shape==(g.SIZE**2,3) and orig.dtype==dirs.dtype==np.float32 and
            max(np.max(abs(orig-ref_o)),np.max(abs(dirs-ref_d)))<=2e-6, 'Frozen Bell rays differ from calibration')
    native={}
    for name,tracer in models.items():
        hit=trace(tracer,orig,dirs,torch)
        n,valid=reconstructed_normals(hit['raw'],hit['hit'])
        require(finite_hits(hit) and np.array_equal(valid,hit['hit']) and
                np.max(abs(np.linalg.norm(hit['raw'][valid],axis=1)-1),initial=0)<=g.RAY_TOL['unit_normal'],
                'Invalid native Bell hit normal or depth')
        native[name]={**hit,'normal':n}
    a,native_r=native['oracle'],native['reconstructed']
    hit=archived['hit']
    require(hit.dtype==bool and hit.shape==(g.SIZE**2,) and np.array_equal(a['hit'],hit),
            'Corrected BVH oracle replay changed hit classification')
    parity={k:float(np.max(abs(a[key][hit]-archived[saved][hit])))
            for k,key,saved in (('depth','depth','depth'),('position','position','positions'),
                                ('raw_normal','raw','raw_normals'),('normal','normal','normals'))}
    require(all(v<=g.RAY_TOL['normal' if k=='raw_normal' else k] for k,v in parity.items()),
            'Corrected BVH oracle replay exceeds accepted tolerances')
    cpu={name:[cpu_witness(surfaces[name],row,orig[i],dirs[i],i) for i in sampled_pixels(hit)]
         for name,row in native.items()}
    row,masks=g.paired_surface(a,native_r,dirs)
    arrays=dict(origins=orig,directions=dirs,**masks)
    for name,data in native.items():
        arrays.update({name+'_'+k:v for k,v in data.items()})
    return dict(camera=c,oracle_replay_maximum=parity,cpu_closest_hit=cpu,**row),arrays



def surface_witnesses(vertices,triangles):
    require(8<len(triangles)<=g.LIMITS['max_mesh_faces'],'Frozen mesh sampling cap exceeded')
    a,b,c=(vertices[triangles[:,i]].astype(float) for i in range(3))
    cross=np.cross(b-a,c-a)
    length=np.linalg.norm(cross,axis=1)
    require(np.isfinite(cross).all() and (length>0).all(),'Invalid mesh witness faces')
    centers=(a+b+c)/3
    cumulative=np.cumsum(length/2)
    homogeneous=np.searchsorted(cumulative,cumulative[-1]*(np.arange(224)+.5)/224)
    rim=np.flatnonzero(abs(np.linalg.norm(centers,axis=1)-1)<=2*g.QUALITY['sdf_step'])
    selected_rim=rim[spaced(len(rim),64)] if len(rim) else np.empty(0,np.int64)
    faces=np.unique(np.r_[homogeneous,selected_rim])
    return centers,cross/length[:,None],faces,spaced(len(vertices),128)


def run(out, source, f, report, torch, cfg, train_root, train, budget, progress):
    from network.renderer import NeROShapeRenderer
    from raytracing import RayTracer

    import os
    work=out/'workspace'
    work.mkdir()
    (work/'assets').symlink_to(source/'assets',target_is_directory=True)
    os.chdir(work)
    network=NeROShapeRenderer(cfg,training=False)
    checkpoint,_saved=own_checkpoint(train_root/'checkpoints/model.pth',g.STEP,network,torch,
                                     expected=train['endpoint_checkpoint'])
    report['endpoint_checkpoint_loads']=1
    report['strict_checkpoint_comparison']=strict_network_load(network,_saved['network_state_dict'],torch)
    del _saved
    network.eval().cuda()
    before=state_sha(network.state_dict())
    require(before==train['endpoint_network_sha256'],'Native extraction network differs from trained endpoint')
    report.update(checkpoint=checkpoint,network_sha256=before,optimizer_state_restored=False)
    progress()
    vertices,triangles=extraction(network,cfg,out,torch,report,budget,progress)

    oracle=Path('/inputs/oracle')
    files={key:g.check(oracle/key,item) for key,item in f['eval_files'].items()}
    ov,of=mesh_arrays(files['mesh.ply'])
    models={name:RayTracer(v,tri) for name,v,tri in (('oracle',ov,of),('reconstructed',vertices,triangles))}
    surfaces={name:Surface(v,tri) for name,v,tri in (('oracle',ov,of),('reconstructed',vertices,triangles))}
    points=[]
    normals=[]
    rasters=out/'rasters';rasters.mkdir()
    with np.load(files['calibrations-safe.npz'],allow_pickle=False) as cams:
        require(np.array_equal(cams['source_camera_ids'],np.arange(128)),'Frozen evaluation camera IDs differ')
        for camera in g.EVAL_IDS:
            report['primary_BVH_ray_slots']+=2*g.SIZE**2
            require(report['primary_BVH_ray_slots']<=g.LIMITS['primary_BVH_rays'],'Bell primary ray cap exceeded')
            row,arrays=inspect_view(files[f'rays/{camera}.npz'],camera,
                                    (cams['pose'][int(camera)],cams['K'][int(camera)]),models,surfaces,torch)
            ids=np.flatnonzero(arrays['reconstructed_hit'])
            selected=ids[spaced(len(ids),16)]
            points.extend(arrays['reconstructed_position'][selected])
            normals.extend(arrays['reconstructed_normal'][selected])
            row.update(split='validation' if camera in f['splits']['validation'] else 'held_view_nvs',
                       SDF_witness_raster_indices=selected.tolist())
            report['cpu_witnesses']+=sum(len(v) for v in row['cpu_closest_hit'].values())
            save(rasters/f'{camera}.npz',arrays)
            report['views'].append(row)
            progress()
            require(all(r['passed'] for v in row['cpu_closest_hit'].values() for r in v),
                    'Independent closest-hit disagreement at camera '+camera)
            budget()

    centers,raw_normals,faces,vertex_ids=surface_witnesses(vertices,triangles)
    samples=np.concatenate((centers[faces],vertices[vertex_ids],np.asarray(points).reshape(-1,3))).astype(np.float32)
    n=np.concatenate((-raw_normals[faces],np.zeros((len(vertex_ids),3)),np.asarray(normals).reshape(-1,3)))
    sdf,arrays=sdf_consistency(samples,n,TorchQueries(network.sdf_network.sdf,torch))
    save(out/'sdf-witnesses.npz',dict(**arrays,selected_faces=faces,selected_vertices=vertex_ids))
    report['sdf']=sdf
    report['quality_passed']=sdf['passed'] and all(r['passed'] for r in report['views'])
    report['network_state_preserved']=state_sha(network.state_dict())==before
    require(report['network_state_preserved'] and
            report['primary_BVH_ray_slots']==g.LIMITS['primary_BVH_rays'] and
            report['cpu_witnesses']==g.LIMITS['cpu_closest_hit_witnesses'], 'Incomplete or mutated Bell evaluation')
