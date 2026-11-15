"""Immutable PLY reader and independent float64 Moller-Trumbore primary-ray checks."""
from pathlib import Path
import numpy as np
from infrastructure.signatures import require


def mesh_arrays(path):
    with Path(path).open('rb') as f:
        header=[]
        for _ in range(32):
            line=f.readline(256).decode('ascii').strip();header.append(line)
            if line=='end_header':break
        require(header[:2]==['ply','format binary_little_endian 1.0'] and header[-1]=='end_header','Reviewed binary PLY required')
        nv=int(next(l.split()[-1] for l in header if l.startswith('element vertex ')))
        nf=int(next(l.split()[-1] for l in header if l.startswith('element face ')))
        vertices=np.frombuffer(f.read(12*nv),dtype='<f4').copy().reshape(nv,3)
        faces=np.frombuffer(f.read(),dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]))
    require(len(faces)==nf and (faces['count']==3).all() and faces['indices'].min()>=0 and faces['indices'].max()<nv and np.isfinite(vertices).all(),'Invalid mesh payload')
    return vertices,faces['indices'].copy()


def intersect(origin,direction,vertices,faces):
    a,b,c=[vertices[faces[:,i]].astype(np.float64) for i in range(3)];e1=b-a;e2=c-a
    q=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,q);valid=abs(det)>1e-14
    inv=np.divide(1,det,out=np.zeros_like(det),where=valid);s=origin-a
    u=np.einsum('ij,ij->i',s,q)*inv;r=np.cross(s,e1);v=r@direction*inv;depth=np.einsum('ij,ij->i',e2,r)*inv
    valid&=(u>=0)&(v>=0)&(u+v<=1)&(depth>0)
    if not valid.any():return None
    index=int(np.argmin(np.where(valid,depth,np.inf)));normal=np.cross(e1[index],e2[index]);normal/=np.linalg.norm(normal)
    return dict(depth=float(depth[index]),position=origin+depth[index]*direction,raw_normal=normal,face=index,
                barycentric_minimum=float(min(u[index],v[index],1-u[index]-v[index])))
