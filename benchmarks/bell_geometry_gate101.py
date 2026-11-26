"""Gate 10.1 frozen extraction-stage diagnosis and exact-zero correction policy."""
import hashlib
from pathlib import Path

import numpy as np

from benchmarks import bell_geometry_gate10_v1 as old
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,PINS,baseline,digest

PROTOCOL='bell-authenticated-geometry-gate10.1-v1'
FREEZE='docs/protocols/BELL_GEOMETRY_EXTRACTION_GATE101_V1.freeze.json'
PRIOR=old.ARCHIVE.parent/'bell-geometry-gate10-v4-5eba8365e19e4ab9aa2320491429cffb'
CHECKPOINT_SHA='27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd'
NETWORK_SHA='b524c01566b1407b3618705fa622883c3d148426c7109422f8add4ad6f62a670'
MESH_SHA='99c093250840be41dac95b73de38db9e7da92dbbc5235215ccccd27fc581d149'
BAD_FACES=[229038,229053,229055,233117,458754,462320,701575,705281,773844,777213]
LIMITS=dict(extraction_nodes=512**3,extraction_tiles=512,checkpoint_loads=1,MC_calls=1,
            diagnostic_SDF_points=3840,diagnostic_gradient_points=640,
            original_SDF_points=1536,original_gradient_points=512,
            primary_BVH_rays=old.LIMITS['primary_BVH_rays'],CPU_ray_checks=384,
            max_faces=old.LIMITS['max_mesh_faces'],optimizer_returns=0,shader_calls=0)
RESOURCES=dict(watchdog=1800,inner=1740,grace=30,cpu_threads=4,memory_gib=128,pids=128,
               allocator=.75,output_bytes=8*1024**3)
SOURCES=tuple(dict.fromkeys((*old.SOURCES,old.FREEZE,'benchmarks/bell_geometry_gate101.py',
    'experiments/bell_geometry_gate101.py','experiments/run_bell_geometry_gate101.sh',
    'tests/scientific/test_bell_geometry_gate101.py',
    'docs/protocols/BELL_GEOMETRY_EXTRACTION_GATE101_V1.md',
    'docs/results/publication_gate101/README.md',
    'docs/results/publication_gate10/RESULTS_V4.md',
    'docs/results/publication_gate10/results-v4/summary.json')))
read,write,item,check=old.read,old.write,old.item,old.check


def definition(out):
    accepted=read(ROOT/'docs/results/publication_gate10/results-v4/summary.json')
    require(accepted['endpoint_checkpoint']['sha256']==CHECKPOINT_SHA and accepted['mesh']['sha256']==MESH_SHA,
            'Accepted checkpoint/mesh identities differ')
    prior=read(ROOT/old.FREEZE)
    return dict(protocol=PROTOCOL,host_output=str(out),prior=str(PRIOR),scientific_commit=PINS['geometry'],
                image=old.IMAGE,backend=prior['backend'],checkpoint=accepted['endpoint_checkpoint'],
                original_mesh=accepted['mesh'],training_receipt=accepted['training_receipt'],
                failed_receipt=accepted['evaluation_receipt'],network_sha256=NETWORK_SHA,
                eval_files=prior['eval_files'],acquisition=prior['acquisition'],
                original_extraction=old.EXTRACTION,quality=old.QUALITY,
                limits=LIMITS,resources=RESOURCES,bad_faces=BAD_FACES,
                correction='drop-only-the-ten-exact-zero-area-serialized-faces-without-reindexing',
                sphere_samples=dict(global_directions=256,lower_directions=128,lower_y=[-1,-.65],
                                    radial_factors=[-1,-.5,0,.5,1],lower_surface_centroids=256),
                sources={n:digest(ROOT/n) for n in SOURCES})


def verify_host_inputs(f):
    training=check(PRIOR/'train/training.json',f['training_receipt'])
    failed=check(PRIOR/'evaluate/evaluation.json',f['failed_receipt'])
    check(PRIOR/'evaluate/bell_shape-300000.ply',f['original_mesh'])
    check(PRIOR/'train/checkpoints/model.pth',f['checkpoint'])
    t,e=read(training),read(failed)
    require(t['status']=='complete' and t['adam_returns']==300000 and t['cleanup_verified']
            and e['status']=='failed' and e['extraction_tiles']==512 and e['primary_BVH_ray_slots']==0
            and t['endpoint_checkpoint']==f['checkpoint'] and t['endpoint_network_sha256']==e['network_sha256']==NETWORK_SHA,
            'Expected accepted authenticated training and failed evaluation')
    baseline('geometry',PINS['geometry'])
    for name,row in f['eval_files'].items():
        check(Path(f['acquisition'])/'geometry'/name,row)


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 10.1 freeze differs')
    f=read(ROOT/FREEZE)
    out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==PRIOR.parent and out.name.startswith('bell-geometry-gate101-')
            and f==definition(out),'Gate 10.1 inputs, source or policy differ')
    return f


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
                image=f['image'],backend=f['backend'],checkpoint_sha256=CHECKPOINT_SHA,selected_gpu='0',
                trusted_CPU_pickle='only the exact accepted step300000 checkpoint, once',
                limits=LIMITS,resources=RESOURCES,maximum_execution_launches=1,
                conditional_correction=f['correction'],no_geometry_or_material_training=True)


def array_identity(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def mesh_stage(v,f):
    require(v.dtype.kind=='f' and f.dtype.kind in 'iu','Expected floating vertices and integer face indices')
    require(v.ndim==f.ndim==2 and v.shape[1:]==f.shape[1:]==(3,) and np.isfinite(v).all()
            and 0<=f.min() and f.max()<len(v),'Invalid diagnostic mesh layout')
    xyz=v[f].astype(float)
    cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    area=np.linalg.norm(cross,axis=1)/2
    require(np.isfinite(area).all(),'Nonfinite diagnostic face area')
    repeated=(f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2])
    zero=np.flatnonzero(area==0)
    coincident=np.any(np.stack([np.all(xyz[:,0]==xyz[:,1],1),np.all(xyz[:,1]==xyz[:,2],1),
                               np.all(xyz[:,0]==xyz[:,2],1)]),axis=0)
    return dict(vertices=array_identity(v),faces=array_identity(f),zero_area_face_ids=zero.tolist(),
                repeated_index_face_ids=np.flatnonzero(repeated).tolist(),
                coincident_coordinate_face_ids=np.flatnonzero(coincident).tolist(),
                zero_area_count=len(zero),area_sum=float(area.sum())),area


def exact_vertex_mapping(raw,processed,processed_faces,source_faces):
    # Default Trimesh may merge vertices and change indices, but keeps face ordering.
    require(source_faces.shape==processed_faces.shape,'Default processing changed face inventory')
    mapping=np.full(len(raw),-1,np.int64)
    # Every occurrence of one raw vertex must name the same processed vertex.
    require(source_faces.dtype.kind in 'iu' and processed_faces.dtype.kind in 'iu','Integer index maps required')
    pairs=np.stack((source_faces.ravel().astype(np.int64),processed_faces.ravel().astype(np.int64)),axis=1)
    unique=np.unique(pairs,axis=0)
    ids,counts=np.unique(unique[:,0],return_counts=True)
    require((counts==1).all(),'Vertex processing changed face correspondence')
    mapping[ids]=unique[:,1]
    displacement=np.linalg.norm(raw[ids].astype(float)-processed[mapping[ids]].astype(float),axis=1)
    return mapping,dict(merged_vertices=len(raw)-len(processed),
                        unreferenced_raw_vertices=int((mapping<0).sum()),
                        maximum_coordinate_shift=float(displacement.max()),
                        mapping=array_identity(mapping))


def candidate_arrays(v,f,expected=BAD_FACES):
    stage,area=mesh_stage(v,f)
    require(stage['zero_area_face_ids']==expected,'Unexpected zero-face inventory. Researcher review required')
    retained=np.flatnonzero(area>0)
    return v.copy(),f[retained].copy(),retained


def sphere_directions():
    def points(n,low,high):
        y=low+(high-low)*(np.arange(n)+.5)/n
        phi=np.arange(n)*np.pi*(3-np.sqrt(5))
        r=np.sqrt(1-y*y)
        return np.stack((r*np.cos(phi),y,r*np.sin(phi)),1)
    return np.concatenate((points(256,-1,1),points(128,-1,-.65))).astype(np.float32)


def raw_support(y,gradient,minus,plus):
    require(all(np.isfinite(a).all() for a in (y,gradient,minus,plus)),'Nonfinite SDF witness')
    length=np.linalg.norm(gradient.astype(float),axis=1)
    usable=length>=old.QUALITY['sdf_min_gradient']
    proxy=np.divide(abs(y),length,out=np.full(len(y),np.inf),where=usable)
    epsilon=old.QUALITY['sdf_min_sign']
    return usable&(proxy<=old.QUALITY['sdf_step'])&(minus<-epsilon)&(plus>epsilon)


def radial_labels(values):
    require(values.shape==(384,5) and np.isfinite(values).all(),'Invalid frozen sphere radial grid')
    # Samples straddle the fixed sentinel at r=1. Negative learned values on both
    # sides yield an artificial masked-field sign change with no raw sign change.
    epsilon=old.QUALITY['sdf_min_sign']
    artificial=(values<-epsilon).all(axis=1)
    genuine=(values[:,1]<-epsilon)&(values[:,3]>epsilon)
    no_cross=(values>epsilon).all(axis=1)
    return dict(sentinel_crossing=artificial,raw_radial_crossing=genuine,
                no_radial_crossing=no_cross,ambiguous=~(artificial|genuine|no_cross))


def write_ply(path,v,f,dtype='<f4'):
    require(dtype in ('<f4','<f8'),'Explicit PLY precision required')
    header=('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(v))+
            '\nproperty '+('float' if dtype=='<f4' else 'double')+' x\nproperty '+
            ('float' if dtype=='<f4' else 'double')+' y\nproperty '+
            ('float' if dtype=='<f4' else 'double')+' z\nelement face '+str(len(f))+
            '\nproperty list uchar int vertex_indices\nend_header\n')
    faces=np.empty(len(f),dtype=[('count','u1'),('indices','<i4',(3,))])
    faces['count']=3;faces['indices']=f
    with path.open('xb') as stream:
        stream.write(header.encode('ascii'));stream.write(np.asarray(v,dtype=dtype).tobytes())
        stream.write(faces.tobytes())
