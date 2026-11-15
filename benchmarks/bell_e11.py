"""Frozen Bell oracle definition and data-only admission (no native imports)."""
from pathlib import Path
import json
import random
import numpy as np
from infrastructure.signatures import require, read_report
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from audits.bell_calibration_e11 import calibrations

PROTOCOL='bell-oracle-generation-e11-v1'
IMAGE='sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137'
MESH='4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01'
LAW='Kurt-a0-metallic-additive-v1'
ENV={'ambient':[.08,.08,.08],'lobes':[{'axis':[0,0,1],'RGB':[1.,.8,.6],'power':12},
                                  {'axis':[1,0,0],'RGB':[.3,.5,.9],'power':6}]}
# Active +90 degree right-handed Y rotation of original world light directions.
RELIGHT={'ambient':[.08,.08,.08],'lobes':[{'axis':[1,0,0],'RGB':[1.,.8,.6],'power':12},
                                      {'axis':[0,0,-1],'RGB':[.3,.5,.9],'power':6}]}
LIMITS={'wall_seconds':7200,'grace_seconds':30,'output_bytes':4*1024**3,
        'primary_rays':128**3,'shadow_directions':128**3*16384,
        'brdf_direction_evaluations':134*128*128*16384,'point_batch':16,
        'host_memory_gib':64,'cpu_threads':4,'cuda_allocator_fraction':.25}
TOL={'silhouette_iou_min':.90,'rotation_abs':2e-6,'intersection_abs':2e-4,
     'unit_normal_abs':2e-6,'reprojection_pixels':2e-4,'cpu_depth_abs':2e-4,
     'cpu_normal_abs':2e-4,'convergence_rgb_abs':.003,'convergence_relative_l2':.002}
INPUT='docs/evidence/PAPER_BENCHMARK_E10_ASSET_INVENTORY_V1.json'
TAPES='benchmarks/inputs/bell_reference_tapes_e11_v1.npz'
MASKS='benchmarks/inputs/bell_mask_identities_e11_v1.json'


def split():
    ids=[str(i) for i in range(128)];random.Random(6033).shuffle(ids)
    return {'train':ids[6:],'validation':ids[5:6],'held_view_nvs':ids[:5]}


def record(path):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.resolve()==path,'Explicit regular input file required')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}


def admit_inputs(mesh,cameras):
    """Hashes + non-executable numeric decoding; no rays, BRDF or CUDA execution."""
    baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    inventory=json.loads((ROOT/INPUT).read_text());bell=next(x for x in inventory['scenes'] if x['scene']=='bell')
    require(digest(mesh)==MESH,'Bell oracle mesh hash mismatch')
    # Inventory is versioned by the generation freeze; every consumed mask is also hashed.
    rows={Path(r['path']).name:r for r in bell['camera_files']}
    mask_rows={r['path']:r for r in json.loads((ROOT/MASKS).read_text())['files']}
    paths=[];records=[record(mesh)]
    for i in range(128):
        p=Path(cameras)/(str(i)+'-camera.pkl');r=record(p)
        require(p.name in rows and r['sha256']==rows[p.name]['sha256'] and r['bytes']==rows[p.name]['bytes'],
                'Camera differs from accepted E10 identity: '+p.name)
        paths.append(p);records.append(r)
        m=Path(cameras)/'masks_train'/(str(i)+'.png');mr=record(m);expected=mask_rows['masks_train/'+str(i)+'.png']
        require(mr['sha256']==expected['sha256'] and mr['bytes']==expected['bytes'],'Source Bell mask differs from frozen identity')
        records.append(mr)
    P,K=calibrations(paths)
    return P,K,records


def linear_to_srgb(rgb):
    require(np.isfinite(rgb).all() and (rgb>=0).all() and (rgb<=1).all(),
            'Linear reference outside PNG domain; no hidden clipping/exposure/tone mapping allowed')
    # Same constants and epsilon as A utils/raw_utils.py. Quantization is explicit.
    eps=np.finfo(np.float32).eps
    srgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,eps)**(1/2.4)-.055)
    return np.rint(srgb*255).astype(np.uint8)


def convergence(fine,coarse):
    require(fine.shape==coarse.shape and fine.ndim==2 and fine.shape[1]==3 and len(fine)>0,
            'Nonempty matching RGB convergence arrays required')
    require(np.isfinite(fine).all() and np.isfinite(coarse).all() and (fine>=0).all() and (coarse>=0).all(),
            'Nonfinite/negative independent reference')
    absolute=float(abs(fine-coarse).max());relative=float(np.linalg.norm(fine-coarse)/max(np.linalg.norm(fine),1e-15))
    return {'maximum_rgb_difference':absolute,'relative_l2':relative,
            'passed':absolute<=TOL['convergence_rgb_abs'] and relative<=TOL['convergence_relative_l2']}


def mask_for_primary_rays(mask):
    """Row-major image mask: ray y*128+x corresponds to pixel [y,x]."""
    require(mask.shape==(128,128) and mask.dtype==np.bool_,'Expected128x128 Boolean source mask')
    return mask.ravel(order='C')


def geometry_gate(o,d,p,raw,n,depth,hit,mask,P,K):
    require(hit.shape==mask.shape==depth.shape==(128*128,) and
            all(v.shape==(128*128,3) for v in (o,d,p,raw,n)),
            'Matching flattened128x128 ray, depth and mask shapes required')
    require(hit.dtype==np.bool_ and mask.dtype==np.bool_,'Boolean hit and mask arrays required')
    union=int((hit|mask).sum());require(union>0,'Nonempty hit/mask union required')
    require(hit.any() and hit.dtype==np.bool_ and mask.dtype==np.bool_,'Empty/invalid primary hits or masks')
    require(np.isfinite(depth).all() and np.array_equal(hit,depth<10) and (depth[hit]>0).all(),'Invalid native depth policy')
    require(all(np.isfinite(v[hit]).all() for v in (p,raw,n)),'Nonfinite native intersections/normals')
    err=float(abs(p[hit]-(o[hit]+depth[hit,None]*d[hit])).max())
    raw_length=np.linalg.norm(raw[hit],axis=1);require((raw_length>0).all(),'Zero raw face normals')
    negation=float(abs(n[hit]+raw[hit]/raw_length[:,None]).max())
    norm=float(abs(np.linalg.norm(n[hit],axis=1)-1).max());facing=(n[hit]*-d[hit]).sum(1)
    xyz=p[hit]@P[:,:3].T+P[:,3];uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:]
    pixels=np.stack(np.meshgrid(np.arange(128)+.5,np.arange(128)+.5),-1).reshape(-1,2)[hit]
    reproj=float(abs(uv-pixels).max());iou=float((hit&mask).sum()/union)
    passed=negation<=TOL['unit_normal_abs'] and err<=TOL['intersection_abs'] and norm<=TOL['unit_normal_abs'] and reproj<=TOL['reprojection_pixels'] and iou>=TOL['silhouette_iou_min'] and (facing>0).all()
    return {'passed':bool(passed),'hits':int(hit.sum()),'silhouette_iou':iou,'intersection_abs':err,
            'normal_length_abs':norm,'normal_negation_abs':negation,'reprojection_pixels':reproj,'normal_dot_view_min':float(facing.min()),
            'front_facing_fraction':float((facing>0).mean()),'raw_dot_view_max':float((raw[hit]*-d[hit]).sum(1).max())}


def verify_authorization(path,sha,freeze_sha,commit,output):
    a,_=read_report(path,sha)
    require(a.get('protocol')==PROTOCOL and a.get('approved') is True and a.get('freeze_sha256')==freeze_sha and
            a.get('commit')==commit and a.get('output_directory')==str(output) and a.get('image_id')==IMAGE and
            a.get('work_limits')==LIMITS,'Authorization must bind this exact generation, source, image, fresh output and limits')
    return a


def verify_labels(p,n,bounds,data):
    """Independent NumPy reconstruction of labels, not a stored success flag."""
    q=(p-bounds.mean(0))/((bounds[1]-bounds[0]).max()/2)
    a=.23+.03*q[:,0];b=.35+.03*q[:,1];g=np.array([1.,2.,3.])/np.sqrt(14)
    u=g-(n@g)[:,None]*n;P=np.eye(3)[None]-n[:,:,None]*n[:,None,:]
    Q=a[:,None,None]**2*P+b[:,None,None]**2*u[:,:,None]*u[:,None,:]
    color=np.stack([.35+.08*q[:,0],.50+.06*q[:,1],.20+.04*q[:,2]],-1)
    metal=.25+.05*np.sin(np.pi*q[:,0])*np.cos(np.pi*q[:,1])
    widths=np.stack([np.sqrt(a*a+b*b*(u*u).sum(-1)),a],-1)
    for name,expected in [('Q_world',Q),('color',color),('metallic',metal[:,None]),('widths_major_minor',widths)]:
        require(data[name].shape==expected.shape and np.isfinite(data[name]).all() and
                np.max(abs(data[name]-expected))<=2e-12,'Independent label reconstruction failed: '+name)
    require((widths>=.04).all() and (widths<=1).all() and (color>=0).all() and (color<=1).all() and
            (metal>=0).all() and (metal<=1).all(),'Material labels outside native domain')
    defined=np.linalg.norm(u,axis=-1)>1e-12
    require(np.array_equal(data['orientation_defined'],defined) and
            np.isfinite(data['theta_major'][defined]).all() and np.isnan(data['theta_major'][~defined]).all(),
            'Undefined truth orientation must remain explicit')
    t=data['T'];B=data['B'];f=np.stack([t,B],-1)
    require(np.max(abs(data['Q_local']-f.transpose(0,2,1)@Q@f))<=2e-12,'Local/world Q mismatch')
    # Major-axis theta/widths must reconstruct the same physical tensor in this chart.
    theta=np.where(defined,data['theta_major'],0);axis=np.cos(theta)[:,None]*t+np.sin(theta)[:,None]*B
    represented=a[:,None,None]**2*P+(widths[:,0]**2-a*a)[:,None,None]*axis[:,:,None]*axis[:,None,:]
    require(np.max(abs(represented-Q))<=2e-12,'Theta/axis tensor label inconsistency')
    return {'independent_label_maximum_abs_error':float(np.max(abs(data['Q_world']-Q))),
            'orientation_undefined':int((~defined).sum()),'orientation_eligible':int(data['orientation_eligible'].sum())}


def source_paths():
    """Small explicit closure: generator, admission/native interfaces and their tests."""
    paths=[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'benchmarks').glob('*.py'))]
    paths+=['AGENTS.md','audits/bell_calibration_e11.py','generate_bell_benchmark_e11.py',
            'benchmarks/run_bell_e11.sh',TAPES,MASKS,INPUT,'scientific_sources/manifest-v1.json',
            'tests/benchmarks/test_bell_e11.py','tests/benchmarks/test_bell_e11_mask_fix.py',
            'docs/protocols/BELL_ORACLE_GENERATION_E11_V1.md','docs/protocols/BELL_ORACLE_GENERATION_E11_V2.md',
            'docs/protocols/PAPER_ANISOTROPIC_BENCHMARK_E10_V1.md','docs/NATIVE_MATERIAL_A_CORRECTIONS_E10.md',
            'scientific_variants/__init__.py','scientific_variants/material_a_v1.py','run_stage_experiment.py']
    paths+=['infrastructure/'+n+'.py' for n in ('__init__','stage_sources','stage_execution','stage_workspace',
             'paths','signatures','material_variants','corrected_material_smoke','material_smoke','bounded_training','training_probe')]
    for v in ('original','estimator','core'):
        paths+=['configs/material_corrected/bell_'+v+'_known_v1.yaml',
                'configs/experiments/material_a_bell_'+v+'_known_v1.yaml']
    return sorted(paths)
