"""Offline saved-ray diagnosis and predicate arithmetic; no model, GPU or repair I/O."""
import ctypes
import numpy as np
from infrastructure.signatures import require


def ray_hits(origin,direction,triangles,dtype=np.float64):
    """Independent double Moller-Trumbore over explicit existing triangles."""
    a,b,c=[triangles[:,i].astype(dtype) for i in range(3)]
    e1=b-a;e2=c-a;o=np.asarray(origin,dtype=dtype);d=np.asarray(direction,dtype=dtype)
    q=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.einsum('ij,ij->i',e1,q)
    inv=np.divide(dtype(1),det,out=np.zeros_like(det),where=det!=0);s=o-a
    u=np.einsum('ij,ij->i',s,q)*inv;r=np.cross(s,e1);v=(r*d).sum(1)*inv;t=np.einsum('ij,ij->i',e2,r)*inv
    good=(det!=0)&(u>=0)&(v>=0)&(u+v<=1)&(t>0)&np.isfinite(t)
    ids=np.flatnonzero(good);ids=ids[np.argsort(t[ids])]
    return [dict(face=int(i),depth=float(t[i]),barycentric=[float(1-u[i]-v[i]),float(u[i]),float(v[i])]) for i in ids]


def iq_predicate(triangle,origin,direction,dtype=np.float64):
    """Scalar mirror of pinned IQ predicate; float64 is the proposed derivative.

    Promotion occurs BEFORE subtraction, including ray-origin minus vertex.
    Native output depth stays float32; no barycentric tolerance is introduced.
    """
    a,b,c=np.asarray(triangle,dtype=dtype);ro=np.asarray(origin,dtype=dtype);rd=np.asarray(direction,dtype=dtype)
    e1=b-a;e2=c-a;s=ro-a;n=np.cross(e1,e2);q=np.cross(s,rd)
    with np.errstate(divide='ignore',invalid='ignore'):
        inv=dtype(1)/(rd@n);u=inv*-(q@e2);v=inv*(q@e1);t=inv*-(n@s)
    accepted=bool(np.isfinite(t) and u>=0 and u<=1 and v>=0 and u+v<=1 and t>=0)
    return dict(accepted=accepted,u=float(u),v=float(v),barycentric=[float(1-u-v),float(u),float(v)],
                depth=float(t) if np.isfinite(t) else None,output_depth=float(np.float32(t)) if accepted else 1e6)


def fma_edge_witness(triangles,origin,direction):
    """Fixed permitted float32 evaluation order, NOT measured CUDA instructions.

    Ordinary n.cross; q.cross contracts its positive product with fmaf; dots
    reduce as fmaf(a0,b0,fmaf(a1,b1,a2*b2)). This exhibits a shared-edge hole.
    """
    library=ctypes.CDLL('libm.so.6');fn=library.fmaf
    fn.argtypes=[ctypes.c_float]*3;fn.restype=ctypes.c_float
    def fma(a,b,c):return np.float32(fn(float(a),float(b),float(c)))
    def dot(a,b):return fma(a[0],b[0],fma(a[1],b[1],a[2]*b[2]))
    rows=[];o=np.asarray(origin,dtype=np.float32);d=np.asarray(direction,dtype=np.float32)
    for triangle in triangles:
        a,b,c=np.asarray(triangle,dtype=np.float32);e1=b-a;e2=c-a;s=o-a;n=np.cross(e1,e2)
        q=np.array([fma(s[i],d[j],-s[j]*d[i]) for i,j in [(1,2),(2,0),(0,1)]],dtype=np.float32)
        inv=np.float32(1)/dot(d,n);u=inv*-dot(q,e2);v=inv*dot(q,e1);t=inv*-dot(n,s)
        rows.append(dict(accepted=bool(u>=0 and u<=1 and v>=0 and np.float32(u+v)<=1 and t>=0),
                         u=float(u),v=float(v),sum_uv=float(u+v),depth=float(t)))
    return rows


def face_geometry(vertices,faces,index):
    triangle=vertices[faces[index]].astype(float);cross=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);length=np.linalg.norm(cross)
    require(length>0,'Degenerate target face')
    edges=np.roll(triangle,-1,axis=0)-triangle
    return dict(index=int(index),vertex_indices=faces[index].tolist(),vertices=triangle.tolist(),
                raw_normal=(cross/length).tolist(),final_A_normal=(-cross/length).tolist(),area=float(length/2),
                longest_edge_over_altitude=float((edges*edges).sum(1).max()/length))


def edge_neighbors(vertices,faces,index):
    base=faces[index];normal=np.array(face_geometry(vertices,faces,index)['raw_normal']);rows=[]
    for j in range(3):
        edge=[int(base[j]),int(base[(j+1)%3])];members=np.flatnonzero(np.isin(faces,edge).sum(1)==2);neighbors=[]
        for i in members:
            if i==index:continue
            g=face_geometry(vertices,faces,i)
            directed=[(int(faces[i,k]),int(faces[i,(k+1)%3])) for k in range(3)]
            g.update(normal_dot_target=float(normal@np.array(g['raw_normal'])),opposite_shared_edge=(edge[1],edge[0]) in directed)
            neighbors.append(g)
        rows.append(dict(edge=edge,incident_faces=members.tolist(),manifold=len(members)==2,neighbors=neighbors))
    return rows


def saved_ray_local_coverage(arrays,triangles,indices):
    """Intersections only for saved rays against explicit local faces, no new rays."""
    o=arrays['origins'].astype(float);d=arrays['directions'].astype(float);N=len(d)
    closest=np.full(N,np.inf);which=np.full(N,-1,dtype=int);counts={}
    for i in indices:
        a,b,c=triangles[i].astype(float);e1=b-a;e2=c-a;s=o-a;q=np.cross(d,e2);det=q@e1
        inv=np.divide(1,det,out=np.zeros_like(det),where=det!=0);u=(s*q).sum(1)*inv;r=np.cross(s,e1);v=(d*r).sum(1)*inv;t=(r@e2)*inv
        hit=(det!=0)&(u>=0)&(v>=0)&(u+v<=1)&(t>0);counts[str(i)]=int(hit.sum())
        nearer=hit&(t<closest);closest[nearer]=t[nearer];which[nearer]=i
    missed=np.flatnonzero(np.isfinite(closest)&(closest<arrays['depth']-2e-4))
    return dict(face_ray_intersection_counts=counts,closer_than_saved_by_over_2e_4=[dict(ray_index=int(j),
                face=int(which[j]),local_depth=float(closest[j]),saved_depth=float(arrays['depth'][j])) for j in missed])


def diagnose(mesh,evidence):
    from pathlib import Path
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from benchmarks.bell_geometry_e11 import mesh_arrays
    from infrastructure.stage_sources import digest
    from benchmarks.bell_e11 import MESH
    require(digest(mesh)==MESH,'Exact original Bell mesh required')
    evidence=Path(evidence);vertices,faces=mesh_arrays(mesh);triangles=vertices[faces]
    archives=[]
    for camera in (0,1):
        with np.load(evidence/'benchmark/geometry/rays'/(str(camera)+'.npz'),allow_pickle=False) as z:
            archives.append({k:z[k] for k in z.files})
    z=archives[1];j=13621;o=z['origins'][j];d=z['directions'][j];point=z['positions'][j]
    hits=ray_hits(o,d,triangles);require(len(hits)>=2,'Entry/exit evidence missing')
    returned=min(hits,key=lambda h:abs(h['depth']-float(z['depth'][j])));front=hits[0]
    require(abs(returned['depth']-float(z['depth'][j]))<2e-4,'Saved hit cannot be identified')
    rows=[]
    for h in hits:
        i=h['face'];g=face_geometry(vertices,faces,i);a,b,c=triangles[i].astype(float)
        bary=np.linalg.lstsq(np.stack([b-a,c-a],-1),point-a,rcond=None)[0]
        g.update(ray_intersection=h,ray_position=(o.astype(float)+h['depth']*d).tolist(),
                 saved_point_plane_distance=float((point-a)@np.array(g['raw_normal'])),
                 saved_point_projected_barycentric=[float(1-bary.sum()),float(bary[0]),float(bary[1])],
                 final_normal_dot_view=float(np.array(g['final_A_normal'])@-d.astype(float)),
                 edge_neighbors=edge_neighbors(vertices,faces,i))
        matrix=np.stack([b-a,c-a,-d.astype(float)],-1);uvt=np.linalg.solve(matrix,o.astype(float)-a)
        precise=ray_hits(o,d,triangles[i:i+1],np.longdouble)[0]
        g['independent_checks']={'linear_solve_depth':float(uvt[2]),'linear_solve_barycentric':[float(1-uvt[:2].sum()),*map(float,uvt[:2])],
            'longdouble_depth':precise['depth'],'longdouble_barycentric':precise['barycentric'],'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant)}
        rows.append(g)
    neighborhood=[]
    for y in range(104,109):
        for x in range(51,56):
            idx=y*128+x;cpu=ray_hits(z['origins'][idx],z['directions'][idx],triangles)[0]
            neighborhood.append(dict(pixel_xy=[x,y],ray_index=idx,saved_depth=float(z['depth'][idx]),saved_position=z['positions'][idx].tolist(),
                saved_normal=z['normals'][idx].tolist(),saved_dot_view=float(z['normals'][idx]@-z['directions'][idx]),
                cpu_closest=cpu,depth_error=float(z['depth'][idx])-cpu['depth']))
    local=set(h['face'] for h in hits)
    for row in rows:
        for edge in row['edge_neighbors']:local.update(edge['incident_faces'])
    coverage={str(c):saved_ray_local_coverage(archives[c],triangles,sorted(local)) for c in (0,1)}
    edge_list=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);ordered=np.sort(edge_list,axis=1)
    edges,inverse,counts=np.unique(ordered,axis=0,return_inverse=True,return_counts=True)
    directions=np.bincount(inverse,weights=np.where(edge_list[:,0]<edge_list[:,1],1,-1))
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices))).tocsr()
    components,component_ids=connected_components(graph,directed=False)
    topology=dict(vertices=len(vertices),triangles=len(faces),connected_vertex_components=int(components),boundary_edges=int((counts==1).sum()),
        nonmanifold_edges=int((counts>2).sum()),inconsistent_two_face_edge_directions=int(((counts==2)&(directions!=0)).sum()),
        same_component_entry_exit=bool(component_ids[faces[front['face'],0]]==component_ids[faces[returned['face'],0]]))
    topology['components']=[]
    for component in range(components):
        vf=component_ids==component;ff=np.flatnonzero(vf[faces[:,0]]);tt=triangles[ff].astype(float)
        volume=np.einsum('ij,ij->i',tt[:,0],np.cross(tt[:,1],tt[:,2])).sum()/6
        topology['components'].append(dict(vertices=int(vf.sum()),triangles=len(ff),signed_volume=float(volume),
            bbox=[vertices[vf].min(0).tolist(),vertices[vf].max(0).tolist()]))
    projection={}
    with np.load(evidence/'benchmark/geometry/calibrations-safe.npz',allow_pickle=False) as cameras:
        poses=cameras['pose'].astype(float);K=cameras['K'].astype(float);centers=-np.einsum('nji,nj->ni',poses[:,:,:3],poses[:,:,3])
        topology['all_camera_centers_outside_mesh_bbox']=bool(((centers<vertices.min(0))|(centers>vertices.max(0))).any(1).all())
        for row in rows:
            t=triangles[row['index']].astype(float);pc=np.einsum('nkj,ij->nik',poses[:,:,:3],t)+poses[:,None,:,3]
            uv=np.einsum('nkj,nij->nik',K,pc);uv=uv[:,:,:2]/uv[:,:,2:]
            front_facing=(centers-t.mean(0))@np.array(row['final_A_normal'])>0
            inframe=(pc[:,:,2]>0).all(1)&(uv.max(1)>0).all(1)&(uv.min(1)<128).all(1)
            eligible=np.flatnonzero(front_facing&inframe)
            projection[str(row['index'])]={'front_facing_inframe_cameras':eligible.tolist(),
                'count':len(eligible),'interpretation':'Projection and orientation only; occlusion not evaluated, no new camera rays traced'}
    neighbor_face=next(edge['incident_faces'] for row in rows if row['index']==front['face'] for edge in row['edge_neighbors'] if edge['edge']==[int(faces[front['face'],1]),int(faces[front['face'],2])])
    neighbor_face=[i for i in neighbor_face if i!=front['face']][0]
    fma=fma_edge_witness(triangles[[front['face'],neighbor_face]],o,d)
    proof=dict(schema_version=1,protocol='bell-saved-triangle-diagnosis-e11-v1',status='offline_diagnosis_complete_native_repair_unexecuted',
        mesh_sha256=digest(mesh),camera_id=1,pixel_xy=[53,106],ray_index=j,
        saved=dict(origin=o.tolist(),direction=d.tolist(),position=point.tolist(),normal=z['normals'][j].tolist(),depth=float(z['depth'][j])),
        returned_face=returned['face'],correct_closest_face=front['face'],depth_gap=float(z['depth'][j])-front['depth'],
        intersections=rows,topology=topology,neighborhood=neighborhood,saved_local_coverage=coverage,potential_camera_footprints=projection,
        float32_fma_witness={'faces':[front['face'],neighbor_face],'results':fma,'both_rejected':all(not q['accepted'] for q in fma),
            'interpretation':'Reproduces a hole in pinned predicate under a permitted contraction order; actual compiled CUDA instructions/traversal not inspected'},
        proposed_float64_results=[iq_predicate(t,o,d) for t in triangles[[front['face'],neighbor_face]]],
        conclusions={'verified':'Saved BVH hit is farther than an independently established entry hit; returned triangle and both local one-rings have consistent manifold winding.',
            'supported_cause':'Non-watertight float32 triangle predicate has a reproducible shared-edge rounding hole.',
            'unresolved':'Exact native contraction order and whether traversal/AABB contributes; no global self-intersection audit or new native execution.',
            'recommendation':'Keep mesh/normals/front-facing gate. Test separately versioned predicate-only float64 patch on the exact saved ray and shared-edge controls before generation.'})
    return proof,dict(triangles=triangles[[front['face'],neighbor_face,returned['face']]],face_indices=np.array([front['face'],neighbor_face,returned['face']]),
                      origin=o,direction=d,saved_depth=z['depth'][j],saved_position=point,saved_normal=z['normals'][j])


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report,_=diagnose(args.mesh,args.evidence)
    from infrastructure.stage_sources import digest
    report['explicit_inputs']=[{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
        for p in [args.mesh,args.evidence/'benchmark/geometry/rays/0.npz',args.evidence/'benchmark/geometry/rays/1.npz',
                  args.evidence/'benchmark/geometry/calibrations-safe.npz']]
    with args.output.open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
