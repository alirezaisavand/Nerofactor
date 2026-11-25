"""Exact public NeRO source, common Donut acquisition and one-run Gate 7 plan."""
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np

from benchmarks import donut_relighting_gate6_v2 as gate6
from benchmarks import donut_representation_gate5_v1 as gate5
from benchmarks.donut_recovery_v1 import SIZE, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import PREFIXES, StageFinder

ROOT = gate6.ROOT
read,check,digest = gate6.read,gate6.check,gate6.digest
m=gate5.m
PROTOCOL='donut-nero-known-geometry-light-gate7-v1'
FREEZE='docs/protocols/DONUT_NERO_GATE7_V1.freeze.json'
UPSTREAM='https://github.com/liuyuan-pal/NeRO'
UPSTREAM_COMMIT='89166a96ea81141dfeec5ba0fc541ccaec92aae4'
UPSTREAM_FILES='docs/results/publication_gate7/nero-source.json'
UPSTREAM_HOST=Path('/home/isavand/paper/evidence')/('nero-official-'+UPSTREAM_COMMIT)
GATE6_SHA='0ed87914b995df462c12516333bd7ee2fbb8f3057c5d6bf906fdb2c3503ff6a3'
GATE6_MANIFEST=ROOT/'docs/results/publication_gate6/results-v2/artifact-manifest.json'
GATE5_MANIFEST=ROOT/gate6.v1.MANIFEST
STEPS, BATCH, SEED = 10000, 512, 6033
SHADER=dict(human_lights=False,diffuse_sample_num=512,specular_sample_num=1024,
            outer_light_version='direction',light_exp_max=5.,inner_light_exp_max=5.)
LR=dict(end_warm=1000,end_iter=100000,lr=.0005)
RENDERER=dict(rgb_loss='charbonier',reg_diffuse_light_lambda=.1)
TRAINABLE=('feats_network','metallic_predictor','roughness_predictor','albedo_predictor')
EVAL_POINTS=sum(m.COUNTS.values())+sum(gate6.v1.COUNTS)
EVAL_CALLS=sum((n+BATCH-1)//BATCH for n in (*m.COUNTS.values(),*gate6.v1.COUNTS))
LIMITS=dict(optimizer_returns=STEPS,backwards=STEPS,shader_calls=STEPS+EVAL_CALLS,
            shadow_slots=(BATCH*STEPS+EVAL_POINTS)*1536,primary_rays=0,reference_renders=0,checkpoint_loads=0)
RESOURCES=dict(gpu='0',cpu_threads=4,host_memory_gib=64,pids=128,cuda_allocator_fraction=.25,
               output_bytes=512*1024**2,watchdog_seconds=3600,kill_grace_seconds=30,inner_seconds=3540,
               inner_kill_grace_seconds=20)
SOURCES=(
    'benchmarks/donut_nero_gate7_v1.py','experiments/donut_nero_gate7_v1.py',
    'experiments/analyze_donut_nero_gate7_v1.py','experiments/run_donut_nero_gate7_v1.sh',
    'tests/scientific/test_donut_nero_gate7_v1.py','docs/protocols/DONUT_NERO_GATE7_V1.md',
    'docs/results/publication_gate7/README.md', UPSTREAM_FILES,
    'infrastructure/stage_sources.py','infrastructure/verified_ray_backend.py','infrastructure/sphere_replay.py',
    'infrastructure/material_recovery_e13_evaluation.py','benchmarks/bell_reference_e11.py',
    'benchmarks/generate_bell_e11.py','experiments/donut_reconstructed_geometry_v1.py',
    'experiments/analyze_donut_relighting_gate6_v1.py',
)


def definition(out):
    parent=gate5.parent()
    old=gate6.freeze(GATE6_SHA)
    manifest=read(GATE6_MANIFEST)
    require(manifest['status']=='verified_complete' and manifest['commit']=='d1bf1392ab10c8dbcc5e9ffcbef127c718380a6f'
            and manifest['execution_artifacts']['freeze.json']['sha256']==GATE6_SHA
            and manifest['backend']==old['backend'],'Expected accepted Gate 6 provenance')
    source=read(ROOT/UPSTREAM_FILES)
    require(source['commit']==UPSTREAM_COMMIT and source['official_repository']==UPSTREAM,
            'Expected official synthetic-scene source')
    for name in ('relighting-results.json','offline-replay.json'):
        check(GATE6_MANIFEST.parent/name,manifest['packaged_files'][name])
    prior=read(GATE5_MANIFEST)
    require(digest(GATE5_MANIFEST)==gate6.v1.MANIFEST_SHA and prior['support']==parent['support'],
            'Gate 5 manifest differs')
    check(GATE5_MANIFEST.parent/'representation-results.json',prior['packaged_files']['representation-results.json'])
    return dict(schema_version=1,protocol=PROTOCOL,host_output=str(out),upstream=source,upstream_root=str(UPSTREAM_HOST),
                image=old['image'],backend=old['backend'],gate4_freeze_sha256=gate5.GATE4_SHA,gate6_freeze_sha256=GATE6_SHA,
                support=parent['support'],gate6_result_manifest_sha256=digest(GATE6_MANIFEST),
                panels={str(c):manifest['offline_analysis'][f'camera-{c}-relighting.npz'] for c in gate6.CAMERAS},
                gate5_manifest_sha256=gate6.v1.MANIFEST_SHA,illuminations=dict(L2=old['fitting_environment'],unseen=old['environment']),
                seed=SEED,updates=STEPS,train_batch=BATCH,eval_batch=BATCH,eval_draws_per_condition=1,
                shader_overrides=SHADER,renderer_cfg=RENDERER,native_lr=LR,
                objective_domain='native unclipped sRGB transfer of the same linear L2 RGB observations',
                limits=LIMITS,resources=RESOURCES,sources={n:digest(ROOT/n) for n in SOURCES})


def freeze(sha):
    p=ROOT/FREEZE
    require(digest(p)==sha,'Gate 7 freeze identity differs')
    f=read(p)
    o=Path(f['host_output'])
    require(o.resolve()==o and o.parent==UPSTREAM_HOST.parent and o.name.startswith('donut-nero-gate7-')
            and f==definition(o),'Frozen Gate 7 definition differs')
    return f


def authorization(f,commit,sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
                image=f['image'],upstream_commit=UPSTREAM_COMMIT,selected_gpu='0',maximum_execution_launches=1,
                limits=LIMITS,resources=RESOURCES,source_manifest_sha256=f['sources'][UPSTREAM_FILES],
                historic_checkpoint_loads_authorized=False)


def paths(f,native=False):
    base=m.paths(gate5.parent(),native)
    upstream=Path('/inputs/official-nero') if native else Path(f['upstream_root'])
    panels={f'unseen_{c}':Path(f'/inputs/unseen-{c}.npz' if native else v['path']) for c,v in f['panels'].items()}
    return dict(base,upstream=upstream,**panels)


def source_check(root,f):
    require(root.resolve()==root and root.is_dir(),'Unsafe NeRO source root')
    files=f['upstream']['files']
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files),
            'Official source closure differs')
    for name,item in files.items():
        check(root/name,item)


def activate_nero(root,f):
    source_check(root,f)
    require(not any(isinstance(x,StageFinder) for x in sys.meta_path)
            and not any(n.split('.')[0] in PREFIXES for n in sys.modules),'Fresh official NeRO process required')
    sys.meta_path.insert(0,StageFinder(root))
    sys.path.insert(0,str(root))


def admit(f,p):
    source_check(p['upstream'],f)
    _,views,_,_,meshes=m.admit(gate5.parent(),p)
    oracle=views['oracle']
    unseen={}
    old=gate6.freeze(GATE6_SHA)
    for camera in gate6.CAMERAS:
        a=numeric(check(p[f'unseen_{camera}'],f['panels'][str(camera)]))
        require(a['reference'].shape==(SIZE**2,3,3) and a['mask_common'].dtype==bool
                and a['mask_expanded_foreground'].dtype==bool and a['mask_full_raster'].all()
                and np.array_equal(a['pixels'],np.flatnonzero(a['mask_expanded_foreground']))
                and np.array_equal(np.flatnonzero(a['mask_common']),oracle[camera]['pixel_indices'])
                and not a['reference'][~a['mask_expanded_foreground']].any(),'Gate 6 oracle support differs')
        raw=numeric(check(Path(p['reference'])/f'view-{camera}.npz',old['artifacts'][f'view_{camera}']))
        indices=np.searchsorted(raw['pixel_indices'],a['pixels'])
        require(np.array_equal(raw['pixel_indices'][indices],a['pixels']) and raw['shader_valid'][indices].all(),
                'Unseen oracle foreground differs')
        panel={k:raw[k][indices] for k in ('position','normals','outgoing','color','metallic','pixel_indices')}
        np.testing.assert_array_equal(panel['pixel_indices'],a['pixels'])
        unseen[camera]=dict(view=panel,reference=a['reference'],masks={k[5:]:v for k,v in a.items() if k.startswith('mask_')})
    return oracle,unseen,meshes['oracle']


@contextmanager
def observe_linear(field,tensors):
    """Observe NeRO's native transfer inputs and leave the actual conversion intact."""
    convert=field.linear_to_srgb

    def observed(value):
        tensors.append(value)
        return convert(value)

    try:
        field.linear_to_srgb=observed
        yield
    finally:
        field.linear_to_srgb=convert


def linear_components(tensors):
    require(len(tensors)==6,'Upstream transfer call inventory differs')
    # Upstream order: total, diffuse light, weighted specular light,
    # diffuse color, specular color, approximate light.
    return tensors[3],tensors[4],tensors[0]
