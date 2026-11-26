"""Frozen fully-inside-cell extraction policy over Gate 10.1's saved raw grid."""
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from benchmarks import bell_geometry_gate101 as previous
from benchmarks import bell_geometry_gate10_v1 as original
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,PINS,baseline,digest

PROTOCOL='bell-domain-aware-raw-SDF-gate10.2-v1'
FREEZE='docs/protocols/BELL_DOMAIN_AWARE_EXTRACTION_GATE102_V1.freeze.json'
DIAGNOSTIC=previous.read(ROOT/previous.FREEZE)['host_output']
DOMAIN='docs/results/publication_gate102/domain-statistics.json'
LIMITS=dict(MC_calls=1,grid_nodes=512**3,network_queries_for_extraction=0,checkpoint_loads=1,
            SDF_points=1536,gradient_points=512,primary_BVH_rays=196608,CPU_ray_checks=384,
            max_faces=4000000,optimizer_returns=0,shader_calls=0)
RESOURCES=previous.RESOURCES
SOURCES=tuple(dict.fromkeys((*previous.SOURCES,previous.FREEZE,
    'docs/results/publication_gate101/results-v1/summary.json',
    'docs/results/publication_gate101/results-v1/artifact-manifest.json',
    'benchmarks/bell_geometry_gate102.py','experiments/bell_geometry_gate102.py',
    'experiments/run_bell_geometry_gate102.sh','tests/scientific/test_bell_geometry_gate102.py',
    'docs/protocols/BELL_DOMAIN_AWARE_EXTRACTION_GATE102_V1.md',
    'docs/results/publication_gate102/README.md',DOMAIN)))
read,write,item,check=previous.read,previous.write,previous.item,previous.check


def definition(out):
    prior=read(ROOT/previous.FREEZE)
    accepted=read(ROOT/'docs/results/publication_gate101/results-v1/summary.json')
    inventory=read(ROOT/'docs/results/publication_gate101/results-v1/artifact-manifest.json')['files']
    require(accepted['execution_status']=='complete' and accepted['cleanup_verified'] and
            accepted['checkpoint']==prior['checkpoint'],'Accepted Gate 10.1 identity differs')
    return dict(protocol=PROTOCOL,host_output=str(out),diagnostic=DIAGNOSTIC,
                image=prior['image'],backend=prior['backend'],scientific_commit=PINS['geometry'],
                checkpoint=prior['checkpoint'],network_sha256=prior['network_sha256'],
                original_mesh=prior['original_mesh'],acquisition=prior['acquisition'],eval_files=prior['eval_files'],
                raw_grid=inventory['run/stages/raw-SDF-grid.npy'],
                outside_mask=inventory['run/stages/outside-sphere.npy'],
                diagnostic_receipt=inventory['run/receipt.json'],verification=inventory['verification.json'],
                grid_shape=[512,512,512],bounds=[-1,1],level=0,domain=read(ROOT/DOMAIN),
                cell_policy='all-eight-corners-inside-for-every-consistent-generating-cell',
                numerical_policy='float32-exact-coordinate-weld-and-exact-zero-area-removal',
                quality=original.QUALITY,limits=LIMITS,resources=RESOURCES,
                sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 10.2 freeze differs')
    f=read(ROOT/FREEZE);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==previous.PRIOR.parent and
            out.name.startswith('bell-geometry-gate102-') and f==definition(out),'Domain-aware policy differs')
    return f


def host_inputs(f):
    root=Path(f['diagnostic'])
    check(root/'run/stages/raw-SDF-grid.npy',f['raw_grid'])
    check(root/'run/stages/outside-sphere.npy',f['outside_mask'])
    receipt=read(check(root/'run/receipt.json',f['diagnostic_receipt']))
    verified=read(check(root/'verification.json',f['verification']))
    require((root/'exit-code.txt').read_text().strip()=='0' and
            (root/'container-absent.txt').read_text().strip()=='absent','Prior diagnostic process cleanup missing')
    require(receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['network_state_preserved']
            and verified['receipt']==f['diagnostic_receipt'] and receipt['checkpoint']==f['checkpoint'],
            'Expected accepted Gate 10.1 captures')
    check(previous.PRIOR/'train/checkpoints/model.pth',f['checkpoint'])
    check(previous.PRIOR/'evaluate/bell_shape-300000.ply',f['original_mesh'])
    for name,row in f['eval_files'].items():check(Path(f['acquisition'])/'geometry'/name,row)
    baseline('geometry',PINS['geometry'])


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
                image=f['image'],backend=f['backend'],selected_gpu='0',raw_grid=f['raw_grid'],
                checkpoint_sha256=previous.CHECKPOINT_SHA,
                CPU_pickle='at most one accepted endpoint load, only for original consistency evaluation',
                limits=LIMITS,resources=RESOURCES,maximum_execution_launches=1,
                no_training=True,cell_policy=f['cell_policy'],numerical_policy=f['numerical_policy'])


def domain_cells(outside):
    require(outside.dtype==bool and outside.ndim==3 and len(set(outside.shape))==1,'Boolean cubic mask required')
    n=outside.shape[0]-1
    all_inside=np.ones((n,n,n),bool)
    any_inside=np.zeros((n,n,n),bool)
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                inside=~outside[i:i+n,j:j+n,k:k+n]
                all_inside&=inside;any_inside|=inside
    return all_inside,any_inside


def cell_counts(valid,any_inside):
    return dict(total_cells=valid.size,fully_inside=int(valid.sum()),
                boundary_cells=int((any_inside&~valid).sum()),fully_outside=int((~any_inside).sum()),
                grid_spacing=2/valid.shape[0],
                maximum_boundary_omission_depth=float(np.sqrt(3)*2/valid.shape[0]))


def bounds_support(low,high,valid,any_inside):
    """All possible unit cells containing the supplied bounds must be supported."""
    require(low.shape==high.shape and low.ndim==2 and low.shape[1]==3 and
            np.isfinite(low).all() and np.isfinite(high).all(),'Invalid voxel bounds')
    n=valid.shape[0]
    lower=np.maximum(np.ceil(high-1),0).astype(np.int64)
    upper=np.minimum(np.floor(low),n-1).astype(np.int64)
    feasible=(lower<=upper).all(1)
    possible=np.where(feasible,np.prod(upper-lower+1,axis=1),0)
    require(((upper-lower)[feasible]<=1).all(),'Triangle spans more than one marching cell')
    good=np.zeros(len(low),np.uint8);touches=np.zeros(len(low),bool)
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                bits=np.array([i,j,k],bool)
                choice=feasible&np.all(~bits|(upper>lower),axis=1)
                cell=np.where(bits,upper,lower).clip(0,n-1)
                hit=valid[cell[:,0],cell[:,1],cell[:,2]]
                touches|=choice&any_inside[cell[:,0],cell[:,1],cell[:,2]]
                good+=choice&hit
    keep=feasible&(good==possible)
    reason=np.full(len(low),3,np.uint8)
    reason[touches]=2
    reason[(good>0)&~keep]=1
    reason[keep]=0
    return keep,reason,possible,good


def face_domain(voxel_vertices,faces,valid,any_inside):
    triangle=voxel_vertices[faces]
    result=bounds_support(triangle.min(1),triangle.max(1),valid,any_inside)
    require((result[2]>0).all(),'Marching-cubes face has no consistent generating cell')
    return result


def point_domain(points,valid,any_inside):
    voxel=(points.astype(float)+1)*(valid.shape[0]/2)
    return bounds_support(voxel,voxel,valid,any_inside)[0]


def areas(v,f):
    x=v[f].astype(float)
    result=np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)/2
    require(np.isfinite(result).all(),'Nonfinite triangle areas')
    return result


def numerical_cleanup(world,faces,keep):
    selected=np.flatnonzero(keep)
    require(len(selected)>0,'No fully supported raw-SDF faces')
    df=faces[selected].astype(np.int64)
    used=np.unique(df)
    rounded=world[used].astype(np.float32)
    unique,inverse=np.unique(rounded,axis=0,return_inverse=True)
    mapped=inverse[np.searchsorted(used,df)]
    area=areas(unique,mapped)
    nonzero=area>0
    require(nonzero.any(),'All supported faces degenerate at native precision')
    retained_faces=selected[nonzero]
    positive=mapped[nonzero]
    used_unique=np.unique(positive)
    remap=np.full(len(unique),-1,np.int64);remap[used_unique]=np.arange(len(used_unique))
    vertex_map=np.full(len(world),-1,np.int64);vertex_map[used]=remap[inverse]
    v=unique[used_unique]
    require(len(v)<=np.iinfo(np.int32).max,'Native index range exceeded')
    f=remap[positive].astype(np.int32)
    require(np.array_equal(v[f],world[faces[retained_faces]].astype(np.float32)),
            'Cleanup changed positive-area coordinates or winding')
    info=dict(domain_faces=len(selected),retained_faces=len(f),removed_zero_faces=int((~nonzero).sum()),
              exact_coordinate_merges=len(used)-len(unique),unused_canonical_vertices=len(unique)-len(used_unique),
              maximum_float32_coordinate_change=float(np.max(abs(rounded.astype(float)-world[used]))),
              minimum_positive_area=float(area[nonzero].min()),no_positive_area_face_removed=True)
    return v,f,vertex_map,retained_faces,selected[~nonzero],info


def topology(vertices,faces):
    f=faces.astype(np.int64)
    edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    unique,inv,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    winding=np.bincount(inv,weights=np.where(edges[:,0]<edges[:,1],1,-1),minlength=len(count))
    face=np.tile(np.arange(len(f)),3)
    order=np.argsort(inv,kind='stable')
    first=np.r_[0,np.cumsum(count)[:-1]]
    anchor=face[order[first]]
    graph=sparse.coo_matrix((np.ones(len(face),np.uint8),(anchor[inv],face)),shape=(len(f),len(f))).tocsr()
    number,labels=connected_components(graph,directed=False)
    first_face=np.full(number,len(f),np.int64)
    np.minimum.at(first_face,labels,np.arange(len(f)))
    canonical=np.empty(number,np.int64);canonical[np.argsort(first_face)]=np.arange(number)
    labels=canonical[labels]
    component_faces=np.bincount(labels,minlength=number)
    component_area=np.bincount(labels,weights=areas(vertices,f),minlength=number)
    report=dict(edge_connected_components=int(number),open_boundary_edges=int((count==1).sum()),
                nonmanifold_edges=int((count>2).sum()),self_edges=int((unique[:,0]==unique[:,1]).sum()),
                inconsistent_two_face_edges=int(((count==2)&(winding!=0)).sum()),
                component_face_counts=component_faces.tolist(),component_areas=component_area.tolist())
    return report,labels,unique[count==1]
