"""Frozen Bell v3 inputs and geometry gates for a fresh pinned G run."""
import hashlib
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from benchmarks.bell_e11 import split
from benchmarks.donut_reconstructed_geometry_v1 import TOL as RAY_TOL
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, digest
from infrastructure.verified_ray_backend import identity

PROTOCOL = 'bell-v3-authenticated-geometry-gate10-v2'
FREEZE = 'docs/protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V2.freeze.json'
CFG = 'scientific_sources/geometry_g/configs/shape/syn/bell.yaml'
ARCHIVE = Path('/home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240')
ADMISSION = ARCHIVE.parent/'bell-oracle-e12-admission-v1/oracle-admission.json'
GEN_SHA = '5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7'
VERIFY_SHA = 'b7c682e18c28ec5bd81c174734faf59e534c68446c980553fece88804f350c1e'
ADMISSION_SHA = 'f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605'
IMAGE = 'sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331'
DATASET = 'datasets/GlossySynthetic/bell'
SIZE, STEP, CADENCE, SEED = 128, 300000, 1000, 6033
KEEP_STEPS = (1000, 15000, 30000, 50000, 100000, 200000)
RESOLUTION = 512
EXTRACTION = dict(resolution=RESOLUTION,bounds=[-1,1],threshold=0,outside_value=1,chunk_axis=64)
EVAL_IDS = tuple(split()['validation']+split()['held_view_nvs'])
LIMITS = dict(adam_returns=STEP, train_rays=STEP*512, train_sample_slots=STEP*512*160,
              occlusion_probe_ray_slots=(STEP-20000)*2048,
              validation_calls=STEP//CADENCE, validation_rays=STEP//CADENCE*32**2,
              validation_sample_slots=STEP//CADENCE*32**2*160,
              periodic_saves=STEP//CADENCE, best_saves_max=STEP//CADENCE,
              owned_checkpoint_loads=2, extraction_nodes=RESOLUTION**3,
              primary_BVH_rays=2*len(EVAL_IDS)*SIZE**2,
              cpu_closest_hit_witnesses=2*len(EVAL_IDS)*32,
              max_mesh_faces=4000000, sdf_witness_points_max=512, sdf_witness_evaluations_max=1536,
              material_updates=0, external_updates=0)
RESOURCES = dict(train_watchdog=129600, eval_watchdog=7200, inner_margin_seconds=60, kill_grace=30,
                 step_1000_seconds=3600, host_memory_gib=128, cpu_threads=4, pids=256,
                 cuda_allocator_fraction=.75, owned_outputs_bytes=16*1024**3)
QUALITY = dict(min_iou=.90, min_oracle_correspondence=.80, radius=.02,
               normal_degrees=30., tight_radius=RAY_TOL['radius'], grazing_cosine=RAY_TOL['grazing_cosine'],
               sdf_min_gradient=1e-6, sdf_min_sign=1e-7, sdf_step=1/(RESOLUTION-1),
               max_sdf_failures=0, minimum_sdf_interior_points=128)
SOURCES = (
    'benchmarks/bell_geometry_gate10_v1.py', 'experiments/bell_geometry_gate10_v1.py',
    'experiments/bell_geometry_gate10_train_v1.py', 'experiments/bell_geometry_gate10_evaluate_v1.py',
    'experiments/run_bell_geometry_gate10_v1.sh', 'tests/scientific/test_bell_geometry_gate10_v1.py',
    'docs/protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.md',
    'docs/protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V2.md',
    'infrastructure/bell_geometry_workers_v2.py',
    'tests/scientific/test_bell_geometry_workers_v2.py',
    'docs/results/publication_gate10/README.md', CFG,
    'scientific_sources/manifest-v1.json', 'infrastructure/stage_sources.py',
    'infrastructure/g_training.py', 'infrastructure/g_checkpoint.py',
    'infrastructure/mesh_diagnostics.py', 'infrastructure/stage_workspace.py',
    'infrastructure/verified_ray_backend.py', 'infrastructure/ray_backend_e11c.py',
    'docs/evidence/RAY_BACKEND_E11C_ACCEPTANCE_V2.json', 'benchmarks/bell_e11.py',
    'infrastructure/geometry_sdf_mesh_audit.py', 'infrastructure/geometry_training_smoke.py',
    'infrastructure/paths.py', 'infrastructure/signatures.py', 'benchmarks/donut_recovery_v1.py',
    'benchmarks/bell_geometry_e11.py', 'benchmarks/donut_reconstructed_geometry_v1.py',
    'experiments/donut_reconstructed_geometry_v1.py', 'benchmarks/generate_bell_e11.py',
    'audits/bell_calibration_e11.py', 'scientific_sources/material_a/raytracing/raytracer.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def item(path):
    return dict(sha256=digest(path),bytes=path.stat().st_size)


def check(path, identity):
    require(path.is_file() and path.resolve()==path and item(path)==identity, 'Missing or changed file '+str(path))
    return path


def train_names(splits):
    ids=splits['train']+splits['validation']
    return {name for c in ids for name in (f'bell/{c}.png',f'bell/masks_train/{c}.png')} | {
        f'bell/{c}-depth.png' for c in splits['validation']}


def calibration_sha(a):
    h=hashlib.sha256()
    for key in ('source_camera_ids','pose','K'):
        h.update(np.ascontiguousarray(a[key]).tobytes())
    return h.hexdigest()


def training_calibration(full, ids):
    with np.load(full,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['source_camera_ids'],np.arange(128))
        rows=np.array(ids,dtype=int)
        return {key:a[key][rows] for key in ('source_camera_ids','pose','K')}


def definition(out):
    root=ARCHIVE/'benchmark'
    gen=root/'geometry/generation.json'
    verify=ARCHIVE/'offline-verification-v1.json'
    require(digest(gen)==GEN_SHA and digest(verify)==VERIFY_SHA and digest(ADMISSION)==ADMISSION_SHA,
            'Accepted Bell v3 receipts differ')
    generation,verification,admission=map(read,(gen,verify,ADMISSION))
    require(generation['status']=='complete' and generation['cleanup_verified'] and generation['scientific_work_complete']
            and generation['splits']==split() and verification['generation_sha256']==GEN_SHA
            and admission['researcher_reviewed_generation'] and admission['status']=='complete'
            and admission['dataset_files']==generation['dataset_files']
            and admission['mesh_sha256']==generation['mesh_sha256'], 'Expected admitted Bell v3 acquisition')
    records={e['path']:{k:e[k] for k in ('bytes','sha256')} for e in generation['outputs']}
    training={name:records[f'{DATASET}/{name[5:]}'] for name in sorted(train_names(split()))}
    test_names=['calibrations-safe.npz','mesh.ply']+[f'rays/{c}.npz' for c in EVAL_IDS]
    evaluation={name:records[f'geometry/{name}'] for name in test_names}
    for name,row in training.items():
        check(root/'datasets/GlossySynthetic'/name,row)
    for name,row in evaluation.items():
        check(root/'geometry'/name,row)
    subset=training_calibration(root/'geometry/calibrations-safe.npz',split()['train']+split()['validation'])
    return dict(protocol=PROTOCOL,host_output=str(out),scientific_commit=PINS['geometry'],seed=SEED,
                stage='geometry',worker_transport='spawn-eight-original-dummy-workers',retained_checkpoints=list(KEEP_STEPS),image=IMAGE,backend=identity(),splits=split(),acquisition=str(root),train_files=training,
                eval_files=evaluation,calibration_subset_sha256=calibration_sha(subset),
                receipts={'generation':GEN_SHA,'verification':VERIFY_SHA,'admission':ADMISSION_SHA},
                extraction=EXTRACTION,
                limits=LIMITS,resources=RESOURCES,quality=QUALITY,
                sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    p=ROOT/FREEZE
    require(digest(p)==sha,'Gate 10 freeze differs')
    f=read(p)
    o=Path(f['host_output'])
    require(o.resolve()==o and o.parent==ARCHIVE.parent and o.name.startswith('bell-geometry-gate10-')
            and f['protocol']==PROTOCOL and f['scientific_commit']==PINS['geometry'] and f['image']==IMAGE
            and f['worker_transport']=='spawn-eight-original-dummy-workers' and f['retained_checkpoints']==list(KEEP_STEPS) and f['stage']=='geometry' and f['acquisition']==str(ARCHIVE/'benchmark')
            and f['backend']==identity() and f['extraction']==EXTRACTION
            and f['seed']==SEED and f['splits']==split() and f['limits']==LIMITS and f['resources']==RESOURCES
            and f['quality']==QUALITY and set(f['sources'])==set(SOURCES)
            and set(f['train_files'])==train_names(f['splits'])
            and set(f['eval_files'])=={'mesh.ply','calibrations-safe.npz',*(f'rays/{c}.npz' for c in EVAL_IDS)}
            and f['receipts']=={'generation':GEN_SHA,'verification':VERIFY_SHA,'admission':ADMISSION_SHA},
            'Invalid Gate 10 frozen policy')
    for name,want in f['sources'].items():
        require(digest(ROOT/name)==want,'Changed frozen source '+name)
    return f


def authorization(f,commit,sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,
                host_output=f['host_output'],image=IMAGE,backend=f['backend'],selected_gpu='0',scientific_commit=PINS['geometry'],
                fresh_seed=SEED,limits=LIMITS,resources=RESOURCES,maximum_execution_launches=1,
                trusted_cpu_pickle='only this run’s own checkpoints at 1000 and 300000',
                historical_checkpoint_loads_authorized=False,material_or_external_training_authorized=False)


def stage_training(f, destination):
    """Physically omit held-out assets, training depth, mesh, labels and all pickle."""
    scene=Path(f['acquisition'])/'datasets/GlossySynthetic'
    destination.mkdir()
    for name,row in f['train_files'].items():
        src=check(scene/name,row)
        out=destination/name
        out.parent.mkdir(exist_ok=True,parents=True)
        copyfile(src,out)
    source=check(Path(f['acquisition'])/'geometry/calibrations-safe.npz',f['eval_files']['calibrations-safe.npz'])
    a=training_calibration(source,f['splits']['train']+f['splits']['validation'])
    with (destination/'calibrations.npz').open('xb') as stream:
        np.savez(stream,**a)
    read_staged(f,destination)


def read_staged(f, root):
    names={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(names==set(f['train_files'])|{'calibrations.npz'},'Unexpected file in split-scoped training mount')
    for name,row in f['train_files'].items():
        check(root/name,row)
    with np.load(root/'calibrations.npz',allow_pickle=False) as z:
        require(set(z.files)=={'source_camera_ids','pose','K'},'Invalid staged camera keys')
        a={k:z[k] for k in z.files}
    ids=f['splits']['train']+f['splits']['validation']
    require(a['pose'].shape==(123,3,4) and a['K'].shape==(123,3,3)
            and a['K'].dtype==a['pose'].dtype==np.float32 and a['source_camera_ids'].dtype==np.int64
            and np.array_equal(a['source_camera_ids'],np.array(ids,dtype=int))
            and calibration_sha(a)==f['calibration_subset_sha256'],'Training calibration rows differ')
    return {c:(a['pose'][i],a['K'][i]) for i,c in enumerate(ids)}


def distribution(values):
    if len(values)==0:
        return None
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),
                mean=float(np.mean(values)),maximum=float(np.max(values)))


def paired_surface(oracle,reconstruction,direction):
    """Same-ray closest-surface correspondence with signed A-native normals."""
    a,b=(v['hit'] for v in (oracle,reconstruction))
    common=a&b
    require(a.any(),'Empty Bell oracle foreground')
    n0,n1=(v['normal'] for v in (oracle,reconstruction))
    cos0=-(n0*direction).sum(1);cos1=-(n1*direction).sum(1)
    ndot=(n0*n1).sum(1)
    depth=reconstruction['depth'][common].astype(float)-oracle['depth'][common].astype(float)
    dis=np.linalg.norm(reconstruction['position'][common]-oracle['position'][common],axis=1)
    theta=np.degrees(np.arccos(np.clip(ndot[common],-1,1)))
    both_front=common&(cos0>0)&(cos1>0)
    close=np.zeros_like(common)
    tight=np.zeros_like(common)
    close[common]=dis<=QUALITY['radius']
    tight[common]=dis<=QUALITY['tight_radius']
    good_normal=ndot>=np.cos(np.radians(QUALITY['normal_degrees']))
    valid=both_front&close&good_normal
    iou=float(common.sum()/(a|b).sum())
    fraction=float(valid.sum()/max(int(a.sum()),1))
    report=dict(oracle=int(a.sum()),reconstructed=int(b.sum()),common=int(common.sum()),
                silhouette_iou=iou,oracle_only=int((a&~b).sum()),reconstruction_only=int((~a&b).sum()),
                shared_misses=int((~a&~b).sum()),matched_support=int(valid.sum()),
                matched_fraction_oracle=fraction,excluded_common=int((common&~valid).sum()),
                facing_failures=int((common&~both_front).sum()),distance_failures=int((common&~close).sum()),
                normal_failures=int((common&~good_normal).sum()),tight_distance_matches=int((valid&tight).sum()),
                common_grazing=int((common&(np.minimum(cos0,cos1)<QUALITY['grazing_cosine'])).sum()),
                signed_depth_error=distribution(depth),absolute_depth_error=distribution(abs(depth)),
                surface_displacement=distribution(dis),signed_normal_error_degrees=distribution(theta),
                passed=iou>=QUALITY['min_iou'] and fraction>=QUALITY['min_oracle_correspondence'])
    return report,dict(valid=valid,common=common,both_front=both_front,depth_close=close,
                       signed_normal_ok=good_normal,tight_distance=tight)
