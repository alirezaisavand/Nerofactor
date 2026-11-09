"""Read-only mesh measurements and original-default Trimesh export diagnostics.

NumPy/Trimesh are injected. No repair, remeshing, scaling, winding or normal fix.
"""
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def geometry(vertices,faces,np):
    v=np.asarray(vertices);f=np.asarray(faces)
    require(v.ndim==f.ndim==2 and v.shape[1:]==f.shape[1:]==(3,) and len(v)>0 and len(f)>0,
            'Empty or malformed vertex/triangle arrays')
    require(v.dtype.kind=='f' and f.dtype.kind in 'iu','Mesh requires floating coordinates and integer indices')
    require(np.isfinite(v).all() and int(f.min())>=0 and int(f.max())<len(v),'Nonfinite geometry or invalid indices')
    xyz=v.astype(np.float64);a,b,c=(xyz[f[:,i]] for i in range(3))
    cross=np.cross(b-a,c-a);twice_area=np.linalg.norm(cross,axis=1)
    require(np.isfinite(twice_area).all() and (twice_area>0).all(),'Degenerate or nonfinite triangles; no repair permitted')
    bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()]
    require((xyz>=-1).all() and (xyz<=1).all(),'Mesh escapes original extraction box')
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=0)
    unique,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    directions=np.where(edges[:,0]<edges[:,1],1,-1)
    sums=np.bincount(inverse,weights=directions,minlength=len(unique))
    inconsistent=int(((counts==2)&(sums!=0)).sum())
    volume=float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6.0)
    require(np.isfinite(volume),'Nonfinite oriented-volume diagnostic')
    return {'vertices':len(v),'faces':len(f),'vertices_dtype':str(v.dtype),'faces_dtype':str(f.dtype),
            'bounds':bounds,'nondegenerate_triangles':len(f),'minimum_triangle_area':float(twice_area.min()/2),
            'maximum_triangle_area':float(twice_area.max()/2),'boundary_edges':int((counts==1).sum()),
            'nonmanifold_edges':int((counts>2).sum()),'inconsistently_wound_two_face_edges':inconsistent,
            'watertight_by_edge_incidence':bool((counts==2).all()),
            'winding_consistent_on_two_face_edges':inconsistent==0,
            'signed_volume_diagnostic':volume,'global_outward_orientation_verified':False,
            'orientation_limit':'Signed volume/edge winding are observations, not SDF/physical outward-normal proof.'}


def normals_summary(values,expected,np):
    values=np.asarray(values)
    require(values.shape==(expected,3) and np.isfinite(values).all(),'Invalid Trimesh diagnostic normals')
    lengths=np.linalg.norm(values.astype(np.float64),axis=1)
    return {'count':len(values),'dtype':str(values.dtype),'zero_length':int((lengths==0).sum()),
            'minimum_length':float(lengths.min()),'maximum_length':float(lengths.max())}


def export_diagnostic(vertices,faces,run,np,trimesh):
    """Export BEFORE lazy normal inspection, preserving original export cache state."""
    names=['raw-vertices.npy','raw-faces.npy','processed-vertices.npy','processed-faces.npy',
           'face-normals.npy','vertex-normals.npy']
    ply=run.output('meshes/bell_shape-best-150000-r128.ply')
    arrays=[run.output('meshes/'+n) for n in names]
    require(not ply.exists() and not ply.is_symlink() and all(not p.exists() and not p.is_symlink() for p in arrays),
            'Mesh/diagnostic output collision')
    raw=geometry(vertices,faces,np)
    raw_v=np.asarray(vertices).copy();raw_f=np.asarray(faces).copy()
    for path,value in zip(arrays[:2],[raw_v,raw_f]):
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    mesh=trimesh.Trimesh(vertices,faces)  # Exact original constructor: retain default processing.
    processed_v=np.asarray(mesh.vertices).copy();processed_f=np.asarray(mesh.faces).copy()
    processed=geometry(processed_v,processed_f,np)
    for path,value in zip(arrays[2:4],[processed_v,processed_f]):
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    mesh.export(str(ply))  # Exact original export API/PLY defaults, no computed-normal access before this.
    checksum=digest(ply)
    # Read our newly generated PLY without processing/repair. This is inspection,
    # never a replacement for the original mesh construction/export operation.
    decoded=trimesh.load_mesh(str(ply),process=False)
    serialized_v=np.asarray(decoded.vertices);serialized_f=np.asarray(decoded.faces)
    serialized=geometry(serialized_v,serialized_f,np)
    require(serialized_v.shape==processed_v.shape and np.array_equal(serialized_f,processed_f),
            'PLY serialization changed vertex count/triangle ordering or winding')
    face_normals=np.asarray(mesh.face_normals).copy();vertex_normals=np.asarray(mesh.vertex_normals).copy()
    normal_info={'face':normals_summary(face_normals,len(processed_f),np),
                 'vertex':normals_summary(vertex_normals,len(processed_v),np),
                 'interpretation':'Trimesh lazy normals from original processed winding, inspected only AFTER export; no orientation repair.'}
    require(np.array_equal(mesh.vertices,processed_v) and np.array_equal(mesh.faces,processed_f) and digest(ply)==checksum,
            'Read-only normal/PLY inspection changed exported geometry')
    with ply.open('rb') as stream:
        header=[]
        for _ in range(128):
            line=stream.readline(1024).decode('ascii').rstrip('\r\n');header.append(line)
            if line=='end_header':break
        require(header[-1]=='end_header','Unrecognized exported PLY header')
    for path,value in zip(arrays[4:],[face_normals,vertex_normals]):
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
    return {'path':ply.relative_to(run.root).as_posix(),'sha256':checksum,'bytes':ply.stat().st_size,
            'format':'PLY','ply_header':header,'raw':raw,'processed':processed,'serialized':serialized,
            'serialization_maximum_coordinate_difference':float(np.abs(serialized_v.astype(np.float64)-processed_v).max()),
            'triangle_order_and_winding_preserved_in_export':True,'normals':normal_info,
            'explicit_repairs_or_transforms':[],
            'processing':'Original trimesh.Trimesh(vertices,triangles) defaults; no extra processing/export options.'}
