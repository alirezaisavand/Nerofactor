"""Independent NumPy/CPU audit of admitted, saved bell geometry/ray evidence.

No scientific model imports, checkpoint/camera pickle, ray tracing, or repairs.
An audit pass applies only to the bounded saved-evidence coordinate checks.
"""
from contextlib import contextmanager
import ast
import datetime
import hashlib
import os
from pathlib import Path
import platform
import random
import re
import stat
import struct
import sys
import zipfile
import zlib

import numpy as np

from infrastructure.signatures import require,read_report,write_report
from infrastructure.stage_sources import ROOT,PINS,baseline,digest

PROTOCOL='bell-geometry-scientific-audit-v1'
BASE='aec29d2063b9c1a88113380018239722dd2682b6'
SOURCE_MANIFEST_SHA='39b9baecce2aab6fc62ec57bd6f45e453d8e9d68215f6af3f25b8ed118c5a3f6'
ACCEPTED={'extraction':'caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f',
          'mesh':'eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5',
          'rays':'88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510'}
CHECKPOINT_SHA='d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc'
VIEWS=(('train','14'),('validation','58'),('nvs','11'))
IMAGE_SIZE=(800,800)
ARRAY_KEYS={'origins','directions','positions','raw_normals','raw_depth','normals','depth','hit','mask','K','pose','rgb'}
TOLERANCES={'rotation_orthogonality':1e-5,'rotation_determinant':1e-5,'intrinsic_structure':1e-6,
    'intrinsic_condition_maximum':1e6,'camera_center_scene_units':1e-5,'ray_direction_component':2e-6,
    'unit_vector_length':2e-6,'pixel_reprojection':.005,'intersection_scene_units':2e-4,
    'normal_negation_component':2e-6,'minimum_facing_fraction':.99,'minimum_silhouette_iou':.90,
    'reported_iou_absolute':1e-12,'mesh_bounds_scene_units':2e-4,'mesh_manifest_absolute':1e-12}
MAX_BYTES=256*1024**2


def reference(value):
    require(type(value) is dict and set(value)=={'path','sha256'},'Explicit path/SHA-256 pair required')
    path=Path(value['path']);sha=value['sha256']
    require(path.is_absolute() and path.resolve()==path and path.is_file(),'Missing/aliased absolute artifact path')
    require(type(sha) is str and re.fullmatch('[a-f0-9]{64}',sha) is not None,'Invalid artifact SHA-256')
    return path,sha


@contextmanager
def verified(value):
    path,sha=reference(value)
    with os.fdopen(os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size<=MAX_BYTES,'Unsupported/oversized artifact')
        def checksum():
            stream.seek(0);h=hashlib.sha256()
            for block in iter(lambda:stream.read(1048576),b''):h.update(block)
            stream.seek(0);return h.hexdigest()
        require(checksum()==sha,'Artifact SHA-256 mismatch: '+str(path))
        yield stream
        after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and checksum()==sha,'Artifact changed during reading')


def array_shapes(resolution):
    h,w=resolution;n=h*w
    return {**{k:(n,3) for k in ['origins','directions','positions','raw_normals','normals']},
            'raw_depth':(n,),'depth':(n,1),'hit':(n,1),'mask':(h,w),'K':(1,3,3),'pose':(1,3,4),'rgb':(1,3,h,w)}


def load_arrays(value):
    with verified(value) as stream:
        require(zipfile.is_zipfile(stream),'Expected numerical NPZ archive, not standalone NPY/pickle')
        with zipfile.ZipFile(stream) as archive:
            members=archive.infolist()
            require(len(members)==len(ARRAY_KEYS) and {v.filename for v in members}=={k+'.npy' for k in ARRAY_KEYS} and
                    sum(v.file_size for v in members)<=MAX_BYTES,'Incomplete/duplicate/extra/oversized NPZ content')
            # Check bounded numerical NPY headers before NumPy can allocate from
            # an invalid shape. No object payload is ever deserialized.
            for item in members:
                require(not item.flag_bits & 1,'Encrypted NPZ member refused')
                require(item.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED),'Unsupported NPZ compression')
                with archive.open(item) as member:
                    magic=member.read(8)
                    require(magic in (b'\x93NUMPY\x01\x00',b'\x93NUMPY\x02\x00'),'Unsupported NPY member header')
                    fmt='<H' if magic[-2:]==b'\x01\x00' else '<I'
                    length_bytes=member.read(struct.calcsize(fmt))
                    require(len(length_bytes)==struct.calcsize(fmt),'Truncated NPY header')
                    size=struct.unpack(fmt,length_bytes)[0];require(size<=4096,'Oversized NPY header')
                    header=ast.literal_eval(member.read(size).decode('ascii'))
                    key=item.filename[:-4];dtype=np.dtype(header['descr']);shape=array_shapes(IMAGE_SIZE)[key]
                    expected=np.dtype(np.bool_ if key in ('hit','mask') else np.float32)
                    require(set(header)=={'shape','descr','fortran_order'} and header['shape']==shape and
                            type(header['fortran_order']) is bool and dtype==expected and not dtype.hasobject,
                            'Invalid numerical NPY shape/dtype header: '+key)
                    count=1
                    for dimension in shape:count*=dimension
                    require(item.file_size==8+len(length_bytes)+size+count*dtype.itemsize,'NPY header/payload size mismatch')
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in ARRAY_KEYS}  # All members checked, including unused RGB/raw-depth payloads.
        for key,array in arrays.items():
            require(array.dtype.kind in 'bfiu' and array.dtype.fields is None,'Non-numerical NPZ member: '+key)
            array.setflags(write=False)
    return arrays


def load_mesh(value):
    """The accepted export's simple PLY layout; never a mesh processing library."""
    with verified(value) as stream:
        header=[]
        for _ in range(32):
            line=stream.readline(256)
            require(line and len(line)<256,'Missing/oversized PLY header')
            text=line.decode('ascii').strip()
            if not text.startswith('comment '):header.append(text)
            if text=='end_header':break
        require(len(header)==9 and header[:2]==['ply','format binary_little_endian 1.0'] and
                header[3:6]==['property float x','property float y','property float z'] and
                header[7:]==['property list uchar int vertex_indices','end_header'],'Unsupported PLY layout')
        nv=int(header[2].split()[-1])
        nf=int(header[6].split()[-1])
        require(header[2]=='element vertex '+str(nv) and header[6]=='element face '+str(nf) and nv>0 and nf>0 and
                12*nv+13*nf<=MAX_BYTES,'Invalid mesh element counts')
        raw=stream.read();require(len(raw)==12*nv+13*nf,'Truncated/extra PLY data')
        vertices=np.frombuffer(raw[:12*nv],dtype='<f4').reshape(nv,3).astype(np.float64)
        records=np.frombuffer(raw[12*nv:],dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]))
        faces=records['indices']
        require(np.isfinite(vertices).all() and (records['count']==3).all() and faces.min()>=0 and faces.max()<nv,
                'Invalid mesh coordinates/triangles')
        a,b,c=(vertices[faces[:,i]] for i in range(3));cross=np.cross(b-a,c-a)
        areas=np.linalg.norm(cross,axis=1)/2
        require(np.isfinite(areas).all() and (areas>0).all(),'Nonfinite/degenerate triangles; no repair')
        bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()]
        require((vertices>=-1).all() and (vertices<=1).all(),'Mesh escapes original G extraction box')
        return {'vertices':nv,'faces':nf,'bounds':bounds,'minimum_area':float(areas.min()),
                'signed_volume':float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6),
                'winding_modified':False,'surface_accuracy_assessed':False}


def add(checks,name,passed,measured=None,tolerance=None,reason=None):
    row={'status':'pass' if bool(passed) else 'fail','measured':measured,'tolerance':tolerance}
    if not passed:row['blocker']=reason or name+' outside the predeclared contract/bounds'
    checks[name]=row


def check_view(arrays,resolution,bounds,claimed):
    """Independent float64 column-vector camera algebra; no original summarize()."""
    h,w=resolution;n=h*w
    shapes=array_shapes(resolution)
    require(set(arrays)==ARRAY_KEYS and all(arrays[k].shape==v for k,v in shapes.items()),'Incompatible NPZ array dimensions')
    require(all(arrays[k].dtype==np.bool_ if k in ('hit','mask') else arrays[k].dtype==np.float32 for k in ARRAY_KEYS),
            'Unexpected numerical precision/mask dtype')
    require(all(np.isfinite(arrays[k]).all() for k in ['origins','directions','depth','raw_depth','K','pose','rgb']),
            'Nonfinite camera/ray/depth/RGB data')
    require((arrays['rgb']>=0).all() and (arrays['rgb']<=1).all(),'Saved source RGB not in original RGB/255 range')
    K=arrays['K'][0].astype(np.float64);pose=arrays['pose'][0].astype(np.float64)
    R=pose[:,:3];t=pose[:,3];o=arrays['origins'].astype(np.float64);d=arrays['directions'].astype(np.float64)
    depth=arrays['depth'][:,0].astype(np.float64);hit=arrays['hit'][:,0];mask=arrays['mask'].reshape(-1)
    checks={}
    orth=float(np.max(np.abs(R@R.T-np.eye(3))));det=float(np.linalg.det(R));cond=float(np.linalg.cond(K))
    intrinsic_error=float(max(np.max(np.abs(K[2]-[0,0,1])),abs(K[1,0])))
    add(checks,'rotation_orthogonality',orth<=TOLERANCES['rotation_orthogonality'],orth,TOLERANCES['rotation_orthogonality'])
    add(checks,'proper_rotation',abs(det-1)<=TOLERANCES['rotation_determinant'],
        {'determinant':det,'absolute_error_from_one':abs(det-1)},{'absolute_error_from_one':TOLERANCES['rotation_determinant']})
    require(np.isfinite(cond),'Singular intrinsic calibration')
    add(checks,'intrinsic_structure',intrinsic_error<=TOLERANCES['intrinsic_structure'] and K[0,0]>0 and K[1,1]>0 and
        cond<=TOLERANCES['intrinsic_condition_maximum'],{'structure_error':intrinsic_error,'condition_number':cond,
        'focal_xy':[float(K[0,0]),float(K[1,1])],'principal_xy':K[:2,2].tolist()},
        {'structure':TOLERANCES['intrinsic_structure'],'maximum_condition':TOLERANCES['intrinsic_condition_maximum'],'positive_focal_lengths':True})
    center=np.linalg.solve(R,-t)  # Independently invert X_camera = R X_world + t.
    center_formula=-R.T@t
    center_error=float(max(np.abs(o-center).max(),np.abs(center-center_formula).max(),np.abs(R@center+t).max()))
    add(checks,'camera_center',center_error<=TOLERANCES['camera_center_scene_units'],
        {'center':center.tolist(),'maximum_error':center_error},TOLERANCES['camera_center_scene_units'])
    yy,xx=np.indices((h,w),dtype=np.float64)
    pixels=np.vstack([(xx+.5).reshape(-1),(yy+.5).reshape(-1),np.ones(n)])
    camera=np.linalg.solve(K,pixels);world=R.T@camera;world/=np.linalg.norm(world,axis=0)
    direction_error=float(np.abs(d-world.T).max());length=np.linalg.norm(d,axis=1)
    require((length>0).all(),'Zero-length saved ray direction')
    add(checks,'ray_unit_lengths',float(np.abs(length-1).max())<=TOLERANCES['unit_vector_length'],
        {'minimum':float(length.min()),'maximum':float(length.max()),'maximum_unit_length_error':float(np.abs(length-1).max())},TOLERANCES['unit_vector_length'])
    add(checks,'camera_ray_directions',direction_error<=TOLERANCES['ray_direction_component'],direction_error,TOLERANCES['ray_direction_component'])
    projected=K@(R@d.T)
    require((np.abs(projected[2])>1e-12).all(),'Ray reprojection has undefined pixel depth')
    pixel_error=float(np.abs(projected[:2]/projected[2]-pixels[:2]).max())
    add(checks,'pixel_center_reprojection',pixel_error<=TOLERANCES['pixel_reprojection'],pixel_error,TOLERANCES['pixel_reprojection'])
    add(checks,'depth_and_hit_contract',np.array_equal(hit,depth<10) and np.array_equal(arrays['raw_depth'],arrays['depth'][:,0]) and
        (depth[hit]>0).all(),{'hits':int(hit.sum()),'misses':int((~hit).sum()),
        'nonfinite_miss_positions':int((~np.isfinite(arrays['positions'][~hit])).sum()),
        'miss_position_normal_contract':'Unspecified payloads excluded from hit-geometry checks'},
        {'hit':'0 < depth < 10','miss':'depth >= 10','raw_depth_equality':'exact'})
    add(checks,'nonempty_intersections',hit.any(),int(hit.sum()),{'minimum':1})
    if not hit.any():
        checks['hit_geometry']={'status':'unresolved','blocker':'No hits: intersection/normal/facing statistics cannot be established'}
    else:
        p=arrays['positions'][hit].astype(np.float64);raw=arrays['raw_normals'][hit].astype(np.float64);normal=arrays['normals'][hit].astype(np.float64)
        require(all(np.isfinite(v).all() for v in [p,raw,normal]),'Nonfinite intersection/normal at a hit')
        unit=d[hit]/length[hit,None]
        error=float(np.abs(p-o[hit]-depth[hit,None]*unit).max())
        calibrated=float(np.abs(p-center-depth[hit,None]*world[:,hit].T).max())
        add(checks,'intersection_saved_unit_ray',error<=TOLERANCES['intersection_scene_units'],error,TOLERANCES['intersection_scene_units'])
        add(checks,'intersection_calibrated_ray',calibrated<=TOLERANCES['intersection_scene_units'],calibrated,TOLERANCES['intersection_scene_units'])
        lo,hi=np.asarray(bounds);escape=float(max(0.,(lo-p).max(),(p-hi).max()))
        add(checks,'hit_points_in_mesh_bounds',escape<=TOLERANCES['mesh_bounds_scene_units'],escape,TOLERANCES['mesh_bounds_scene_units'])
        raw_len=np.linalg.norm(raw,axis=1);normal_len=np.linalg.norm(normal,axis=1)
        require((raw_len>0).all() and (normal_len>0).all(),'Zero-length hit normals')
        checks['raw_normal_lengths']={'status':'pass','measured':{'minimum':float(raw_len.min()),'maximum':float(raw_len.max()),
            'maximum_unit_length_error':float(np.abs(raw_len-1).max())},'tolerance':'Finite and positive only; raw BVH magnitude is not assumed to be one'}
        normal_error=float(np.abs(normal_len-1).max());negation=float(np.abs(normal+raw/raw_len[:,None]).max())
        add(checks,'final_normal_unit_lengths',normal_error<=TOLERANCES['unit_vector_length'],
            {'minimum':float(normal_len.min()),'maximum':float(normal_len.max()),'maximum_unit_length_error':normal_error},TOLERANCES['unit_vector_length'])
        add(checks,'existing_A_normal_negation',negation<=TOLERANCES['normal_negation_component'],negation,TOLERANCES['normal_negation_component'])
        for name,values,positive in [('final_normal_facing',normal/normal_len[:,None],True),('raw_BVH_opposite_view',raw/raw_len[:,None],False)]:
            dot=-np.einsum('ij,ij->i',values,unit);fraction=float(((dot>0) if positive else (dot<0)).mean())
            add(checks,name,fraction>=TOLERANCES['minimum_facing_fraction'],{'minimum_dot':float(dot.min()),'maximum_dot':float(dot.max()),
                'mean_dot':float(dot.mean()),'front_facing_count':int((dot>0).sum()),'back_facing_count':int((dot<0).sum()),
                'grazing_count':int((dot==0).sum()),'required_fraction':fraction},TOLERANCES['minimum_facing_fraction'])
    tp=int(np.count_nonzero(hit & mask));fp=int(np.count_nonzero(hit & ~mask));fn=int(np.count_nonzero(~hit & mask));tn=int(np.count_nonzero(~hit & ~mask))
    union=tp+fp+fn;iou=tp/union if union else 0.
    counts={'rays':n,'hits':int(hit.sum()),'misses':int((~hit).sum()),'mask_pixels':int(mask.sum()),'mask_intersection':tp,'mask_union':union}
    add(checks,'silhouette_iou',union>0 and iou>=TOLERANCES['minimum_silhouette_iou'],
        {'iou':iou,'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn},TOLERANCES['minimum_silhouette_iou'])
    require(all(k in claimed for k in [*counts,'silhouette_iou']),'Incomplete accepted per-view numerical evidence')
    delta=abs(iou-claimed['silhouette_iou'])
    add(checks,'reported_count_iou_consistency',all(claimed[k]==v for k,v in counts.items()) and delta<=TOLERANCES['reported_iou_absolute'],
        {'independent_counts':counts,'iou_absolute_difference':delta},TOLERANCES['reported_iou_absolute'])
    return checks


def admit(selection):
    artifacts=selection['artifacts']
    require(artifacts['source_manifest']['sha256']==SOURCE_MANIFEST_SHA and
            Path(artifacts['source_manifest']['path'])==ROOT/'scientific_sources/manifest-v1.json','Explicit pinned source manifest required')
    with verified(artifacts['source_manifest']):pass
    sources={stage:baseline(stage,commit)[1] for stage,commit in PINS.items()}
    for name in ACCEPTED:
        require(artifacts[name]['sha256']==ACCEPTED[name],'Not the accepted '+name+' SHA-256')
        with verified(artifacts[name]):pass
    extraction,_=read_report(*reference(artifacts['extraction']))
    rays,_=read_report(*reference(artifacts['rays']))
    comparison=extraction.get('network_comparison',{});settings=extraction.get('settings',{})
    require(extraction.get('schema_version')==1 and extraction.get('protocol')=='bell-historical-best-G-mesh-r128-v1' and
            extraction.get('status')==extraction.get('phase')=='complete' and
            extraction.get('run_id')=='bell-g-best-mesh-2f4a25e1828f47109433f42a56d53f4a' and
            extraction.get('scientific_commit')==PINS['geometry'] and extraction.get('source')==sources['geometry'] and
            extraction.get('source_manifest_sha256')==SOURCE_MANIFEST_SHA and
            extraction.get('checkpoint_selection',{}).get('sha256')==CHECKPOINT_SHA and
            extraction.get('checkpoint_metadata',{}).get('stored_step')==150000 and
            extraction.get('strict_network_loaded') is True and extraction.get('network_unchanged') is True and
            comparison.get('result')=='exact_structural_match' and
            all(comparison.get(k)==[] for k in ['missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches']) and
            extraction.get('research_geometry_admitted') is False and extraction.get('research_handoff_admitted') is False,
            'Incomplete/incompatible extraction identity/compatibility evidence')
    require(settings.get('resolution')==128 and settings.get('bbox_min')==[-1,-1,-1] and settings.get('bbox_max')==[1,1,1] and
            settings.get('isovalue')==0 and settings.get('outside_val')==1. and settings.get('explicit_mesh_transforms')==[] and
            extraction.get('mesh',{}).get('sha256')==artifacts['mesh']['sha256'] and
            extraction['mesh'].get('explicit_repairs_or_transforms')==[] and
            extraction['mesh'].get('triangle_order_and_winding_preserved_in_export') is True,
            'Incomplete/changed extraction settings or mesh binding')
    ids=[str(i) for i in range(128)];random.Random(6033).shuffle(ids)
    expected={'train':ids[6:],'validation':ids[5:6],'nvs':ids[:5]}
    require(rays.get('schema_version')==1 and rays.get('protocol')=='historical-G-bell-to-A-rays-v1' and
            rays.get('status')=='complete' and rays.get('phase')=='ray_gates_passed' and
            rays.get('scientific_commit')==PINS['material'] and rays.get('source')==sources['material'] and
            rays.get('mesh_identity',{}).get('sha256')==artifacts['mesh']['sha256'] and
            rays.get('extraction_report',{}).get('sha256')==artifacts['extraction']['sha256'] and
            rays.get('diagnostic_only') is True and rays.get('research_handoff_admitted') is False and
            rays.get('ray_gate_passed') is True and rays.get('optimizer_calls_completed')==0 and rays.get('splits')==expected,
            'Incomplete/incompatible accepted G-to-A ray evidence')
    rows=rays.get('views',[])
    require(len(rows)==3,'Exactly three accepted source-camera views required')
    for chosen,row,pair in zip(selection['views'],rows,VIEWS):
        require((chosen['role'],chosen['source_camera_id'])==(row.get('role'),row.get('source_camera_id'))==pair and row.get('resolution')==list(IMAGE_SIZE) and
                chosen['sha256']==row.get('sha256') and row.get('artifact')=='evaluations/'+pair[0]+'-'+pair[1]+'.npz' and
                rays.get('output_hashes',{}).get(row['artifact'])==chosen['sha256'] and
                row.get('statistics',{}).get('passed') is True and row['statistics'].get('checks') and
                all(v is True for v in row['statistics']['checks'].values()),'Wrong camera/archive binding or incomplete accepted view')
    return extraction,rays,sources


def selection_file(path,sha):
    path,_=reference({'path':str(path),'sha256':sha})
    value,identity=read_report(path,sha)
    require(set(value)=={'schema_version','protocol','artifacts','views'} and type(value['schema_version']) is int and value['schema_version']==1 and
            value['protocol']==PROTOCOL,'Unsupported audit selection structure/version')
    require(type(value['artifacts']) is dict and set(value['artifacts'])=={'source_manifest','extraction','mesh','rays'},'Complete artifact selections required')
    for ref in value['artifacts'].values():reference(ref)
    require(type(value['views']) is list and len(value['views'])==3,'Exactly three explicit NPZ selections required')
    for row,pair in zip(value['views'],VIEWS):
        require(type(row) is dict and set(row)=={'role','source_camera_id','path','sha256'} and
                type(row['source_camera_id']) is str and row['source_camera_id'].isdigit() and
                row['role'] in ('train','validation','nvs'),'Invalid source-camera ID/role selection')
        reference({k:row[k] for k in ['path','sha256']})
    return value,identity


def categories(coordinates=False,extraction=False):
    unresolved=lambda reason:{'status':'unresolved','blocker':reason}
    return {'historical_checkpoint_origin':unresolved('Producing revision/configuration/training split remain unattributed.'),
        'G_architecture_compatibility':{**unresolved('No new checkpoint/reference architecture inspection in 3A-1.'),
            'prior_evidence':'Accepted extraction manifest records a prior exact structural match; that finding is not revoked or re-established by camera algebra.'},
        'extraction_identity':{'status':'pass' if extraction else 'unresolved','scope':'Exact selected manifest/mesh/source identity, not re-extraction.',
            **({} if extraction else {'blocker':'Extraction identity was not admitted; see artifact-admission blocker.'})},
        'saved_evidence_coordinate_consistency':{'status':'pass' if coordinates else 'fail','scope':'Only admitted saved cameras/rays and bounded algebra/mask checks.',
            **({} if coordinates else {'blocker':'Required coordinate/admission checks failed or were not executed; see check blockers.'})},
        'source_camera_calibration_identity':unresolved('Saved K/pose are internally checked; original camera pickles are not opened or independently calibrated.'),
        'native_G_camera_execution':unresolved('The saved ray arrays are A observations, not a new G camera/renderer execution.'),
        'scientific_geometry_accuracy':unresolved('No independent 3D reference, surface-distance or convergence assessment. Silhouettes do not establish 3D accuracy.'),
        '3A2_SDF_orientation':unresolved('No SDF queries/sign/gradient test. Winding and view-facing normals do not verify SDF orientation.'),
        'G_to_A_research_admission':unresolved('Historical r128 diagnostic mesh is not attributable new G training or canonical512 research geometry.')}


def audit(inputs,inputs_sha256,output):
    """No originals opened writable; refuse aliases, overlaps and all collisions."""
    selected,selection_identity=selection_file(inputs,inputs_sha256)
    output=Path(output)
    require(output.is_absolute() and output.resolve()==output and output.parent.is_dir() and not output.exists(),
            'Output must be a new unaliased absolute directory with an existing parent')
    refs=list(selected['artifacts'].values())+[{k:r[k] for k in ['path','sha256']} for r in selected['views']]
    protected=[ROOT,ROOT.parent/'data',ROOT.parent/'final_results']+[Path(r['path']).parent for r in refs]
    require(all(output!=p and p not in output.parents for p in protected) and
            all(output not in Path(r['path']).parents for r in [*refs,selection_identity]),'Output overlaps code/original input/evidence storage')
    output.mkdir(exist_ok=False)
    report={'schema_version':1,'protocol':PROTOCOL,'reviewed_base_commit':BASE,'status':'fail',
        'scope':'bounded saved-evidence coordinate audit only','selection':selection_identity,'artifacts':selected,
        'tolerances':dict(TOLERANCES),'checks':{},'views':[],'categories':categories(),
        'research_handoff_admitted':False,'runtime':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
        'implementation':{name:digest(ROOT/name) for name in ['infrastructure/geometry_scientific_audit.py','audit_geometry_scientific.py',
            'infrastructure/signatures.py','infrastructure/stage_sources.py','infrastructure/paths.py','docs/GEOMETRY_SCIENTIFIC_AUDIT_V1.md']},
        'conventions':{'pose':'3x4 world-to-camera [R|t]: X_camera = R X_world + t; C_world = -R.T t',
            'pixels':'row-major image order; u=column+0.5, v=row+0.5; positive camera Z',
            'G':'Pinhole dirs=K^-1 pixel, then R.T and unit normalization in _process_ray_batch; original quarter validation scales K',
            'A':'Same pinhole convention; full-resolution directions are normalized before BVH; Euclidean ray distance, hit depth<10',
            'units':'Original synthetic scale_factor=1; computational scene/SDF units, physical length units unknown',
            'extraction':'XYZ; bbox [-1,1]^3, voxel/(128-1)*2-1; zero isovalue; outside unit sphere assigned +1; no new transform',
            'normals':'Observe original raw BVH normals and A -normalize(raw); no mesh flip or SDF sign inference'},
        'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        extraction,rays,sources=admit(selected)
        add(report['checks'],'artifact_admission',True,{'exact_accepted_artifacts':True})
        report['source_files']={stage:{str(ROOT/item['root']/name):record['sha256'] for name,record in item['files'].items()}
                                for stage,item in sources.items()}
        mesh=load_mesh(selected['artifacts']['mesh']);reported=extraction['mesh']['serialized']
        require(all(k in reported for k in ['vertices','faces','bounds','signed_volume_diagnostic']),'Incomplete extraction mesh measurements')
        add(report['checks'],'mesh_measurement_identity',mesh['vertices']==reported['vertices'] and mesh['faces']==reported['faces'] and
            np.max(np.abs(np.asarray(mesh['bounds'])-reported['bounds']))<=TOLERANCES['mesh_manifest_absolute'] and
            abs(mesh['signed_volume']-reported['signed_volume_diagnostic'])<=TOLERANCES['mesh_manifest_absolute'],mesh,
            {'bounds_and_volume_absolute':TOLERANCES['mesh_manifest_absolute']})
        for chosen,original in zip(selected['views'],rays['views']):
            entry={'role':chosen['role'],'source_camera_id':chosen['source_camera_id'],
                   'current_G_split_membership':'training (not verified historical producing membership)',
                   'prior_declared_statistics':original['statistics'],'checks':{}}
            report['views'].append(entry)
            try:
                arrays=load_arrays({k:chosen[k] for k in ['path','sha256']})
                entry['checks']=check_view(arrays,IMAGE_SIZE,mesh['bounds'],original['statistics'])
            except (ValueError,OSError,KeyError,TypeError,EOFError,UnicodeError,SyntaxError,OverflowError,zipfile.BadZipFile,zlib.error,np.linalg.LinAlgError) as exc:
                add(entry['checks'],'array_contract',False,reason=type(exc).__name__+': '+str(exc))
        all_checks=[*report['checks'].values(),*[c for v in report['views'] for c in v['checks'].values()]]
        passed=all(c['status']=='pass' for c in all_checks)
        report['status']='pass' if passed else 'fail';report['categories']=categories(passed,True)
    except (ValueError,OSError,KeyError,TypeError,EOFError,UnicodeError) as exc:
        add(report['checks'],'artifact_admission',False,reason=type(exc).__name__+': '+str(exc))
    # Recheck all selected bytes. Metadata reporting never silently admits a file
    # that changed after its NPZ/PLY read, including the explicit selection itself.
    try:
        for ref in [*refs,{'path':selection_identity['path'],'sha256':selection_identity['sha256']}]:
            with verified(ref):pass
    except (ValueError,OSError) as exc:
        add(report['checks'],'final_identity',False,reason=str(exc));report['status']='fail'
        report['categories']['saved_evidence_coordinate_consistency']['status']='fail'
        report['categories']['saved_evidence_coordinate_consistency']['blocker']='Selected evidence changed or failed its final identity check.'
    report['blockers']=[c['blocker'] for c in [*report['checks'].values(),*[c for v in report['views'] for c in v['checks'].values()]]
                        if c['status']!='pass' and 'blocker' in c]
    report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (output/'audit.json').open('x') as stream:write_report(stream,report)
    return report
