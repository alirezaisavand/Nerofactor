"""Gate 13 saved-data admission and exact K1-style paired acquisition."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import numpy as np
from benchmarks import bell_geometry_rgb_gate12 as g
from benchmarks import donut_heterogeneous_training_v1 as old
from benchmarks import bell_final_tensor_v1 as accepted
from benchmarks.bell_dense_known_l2_v1 import index_hash
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,digest,PINS,baseline

PROTOCOL='bell-authenticated-O1-known-light-material-gate13-v1'
FREEZE='docs/protocols/BELL_MATCHED_MATERIAL_GATE13_V1.freeze.json'
INVENTORY='docs/results/publication_gate13/input-inventory.json'
ARMS=('K1-Oracle','K1-O1')
STEPS=old.STEPS
POLICY=deepcopy(old.POLICY)
SHADER=dict(old.SHADER,known_environment=deepcopy(g.ENV))
TARGET_KEYS,FIT_KEYS=old.TARGET_KEYS,old.FIT_KEYS
GENERATION=g.prior.original.GEN_SHA
ADMISSION=g.prior.original.ADMISSION_SHA
IMAGE=g.prior.original.IMAGE
FOREGROUND_SECONDS=18600
TRAINER_LIMIT_SECONDS=9000
POINT_BATCH=512
SOURCES=(
    'benchmarks/bell_material_geometry_gate13.py','experiments/train_bell_material_geometry_gate13.py',
    'experiments/prepare_bell_material_geometry_gate13.py','experiments/analyze_bell_material_geometry_gate13.py',
    'experiments/run_bell_material_geometry_gate13.sh','infrastructure/bell_material_pair_gate13.py',
    'tests/scientific/test_bell_material_geometry_gate13.py','docs/protocols/BELL_MATCHED_MATERIAL_GATE13_V1.md',
    INVENTORY,'experiments/train_donut_matched_geometry_v1.py','benchmarks/donut_matched_geometry_recovery_v1.py',
    'experiments/bell_final_tensor_trainer_v1.py','experiments/donut_material_recovery_v1.py',
    'benchmarks/bell_dense_known_l2_v1.py','infrastructure/known_l2_audit.py','benchmarks/bell_known_l2_v1.py',
    'scientific_variants/material_a_v1.py','benchmarks/donut_recovery_v1.py',
    'benchmarks/donut_heterogeneous_training_v1.py','infrastructure/surface_q_regularization.py',
    'infrastructure/material_recovery_e13_evaluation.py','infrastructure/sphere_visuals.py','infrastructure/sphere_replay.py',
    'benchmarks/bell_geometry_rgb_gate12.py','docs/results/publication_gate12/O1-controlled-handoff.json',
    'docs/results/publication_gate12/results-v1/finalization-manifest.json',
    'docs/results/publication_gate11/results-v1/artifact-manifest.json',
    'scientific_sources/manifest-v1.json','infrastructure/stage_sources.py','infrastructure/verified_ray_backend.py')
K1_FREEZE='docs/protocols/BELL_DENSE_KNOWN_L2_RECOVERY_V2.freeze.json'
K1_SHA='1c7887d33f8f8ba3fb28cc48d795636ee2955eedbeebc7ead56bdef26736faf7'
SOURCES=tuple(dict.fromkeys((*SOURCES,K1_FREEZE,*g.SOURCES,
    'experiments/train_bell_dense_known_l2_v1.py','infrastructure/sphere_visuals.py',
    'infrastructure/sphere_replay.py','experiments/analyze_donut_heterogeneous_training_v1.py',
    'benchmarks/neural_observation_e16.py','experiments/bell_neural_material_e16.py',
    'docs/protocols/BELL_GATE13_RELIGHTING_PROSPECTIVE.md',
    'docs/results/publication_gate13/SAM_READINESS.md')))
read,write,item,check=g.read,g.write,g.item,g.check
point_errors=old.point_errors


def split():return g.prior.original.split()


def panel():
    s=split();return tuple(map(int,s['train']+s['validation']+s['held_view_nvs']))


def paths(native=False):
    inv=read(ROOT/INVENTORY)
    return {k:Path('/inputs')/k if native else Path(v) for k,v in inv['roots'].items()}


def definition(out):
    from experiments.train_bell_material_geometry_gate13 import trainer_source
    inv=read(ROOT/INVENTORY)
    require(digest(ROOT/K1_FREEZE)==K1_SHA,'Accepted K1 recipe freeze differs')
    base=read(ROOT/K1_FREEZE)['arms']['K1-Dense']
    require(base['representation'] is None and base['spatial_policy'] is None,'Accepted K1 control changed')
    calls=sum((x['common']+511)//512 for x in inv['support']['cameras'])
    points=sum(x['common'] for x in inv['support']['cameras'])
    work=dict(shader_calls=10000+32+48*calls,shadow_rays=(10000*512+32*24+48*points)*1536,component_reverse_sweeps=30)
    limits=dict(base['limits'],**work,arm_seconds=9000,inner_seconds=8940,wall_seconds=18600,
                boundary_predictor_calls=0,boundary_predictor_points=0,smoothness_field_calls=0)
    return dict(protocol=PROTOCOL,host_output=str(out),order=list(ARMS),panel=list(panel()),splits=split(),
        policy=POLICY,shader=SHADER,steps=STEPS,image=IMAGE,backend=inv['backend'],support=inv['support'],
        input_inventory=item(ROOT/INVENTORY),base_initial=base['initial'],
        generation_sha256=GENERATION,dataset_admission_sha256=ADMISSION,
        reference_freeze_sha256=inv['reference_freeze_sha256'],
        O1_handoff=inv['O1_handoff'],foreground_seconds=FOREGROUND_SECONDS,
        arms={arm:dict(base,protocol=PROTOCOL,geometry='oracle' if arm=='K1-Oracle' else 'O1',
                      illumination='known_Bell_v3',work=work,limits=limits,trainer_sha256=sha256(trainer_source().encode()).hexdigest()) for arm in ARMS},
        resources=dict(gpu=0,cpu_threads=4,host_memory_gib=64,pids=128,cuda_allocator_fraction=.25,
                       output_bytes_per_arm=8*1024**3,arm_seconds=9000,inner_seconds=8940,whole_seconds=18600,kill_grace=30),
        native_preparation_executed=False,sources={n:digest(ROOT/n) for n in SOURCES})


def freeze(path,sha):
    require(path==ROOT/FREEZE and digest(path)==sha,'Gate 13 freeze differs')
    f=read(path);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence') and
        out.name.startswith('bell-material-gate13-') and f==definition(out),'Gate 13 definition differs')
    for stage,pin in PINS.items():baseline(stage,pin)
    return f


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
        image=IMAGE,backend=f['backend'],order=list(ARMS),arms=f['arms'],resources=f['resources'],
        researcher_admitted_dataset=True,benchmark_generation_sha256=GENERATION,dataset_admission_sha256=ADMISSION,
        O1_handoff_sha256=f['O1_handoff']['sha256'],oracle_assisted_correspondence=True,maximum_execution_launches=1,
        CPU_pickle='Only five newly produced checkpoints per arm at steps 2000/4000/6000/8000/10000, CPU roundtrip, no optimizer restore',
        historical_checkpoint_loading=False,relighting=False,SAM2=False,SAM3=False,retries=False)


def admit(p):
    inv=read(ROOT/INVENTORY)
    for root,files in inv['files'].items():
        for name,i in files.items():check(p[root]/name,i)
    require(digest(p['oracle']/'geometry/generation.json')==GENERATION,'Bell generation differs')
    require(digest(p['admission'])==ADMISSION and digest(p['verification'])==g.prior.original.VERIFY_SHA,
            'Original researcher dataset admission/verification differs')
    h=read(ROOT/g.HANDOFF)
    require(h['perfect_synthetic_mask_assisted'] and not h['unchanged_original_G'] and
        h['status']=='admitted_for_controlled_downstream_evaluation' and h['mesh']==inv['files']['O1']['candidate.ply'],
        'Explicit O1 derivative admission required')
    return inv


def view(camera,kind,p,bounds):
    """Same row IDs/targets, geometry-specific hits and analytic evaluation labels."""
    with np.load(p['oracle']/'geometry/rays'/f'{camera}.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    group='fitting' if str(camera) in split()['train'] else 'rasters'
    with np.load(p['O1']/group/f'{camera}.npz',allow_pickle=False) as z:r={k:z[k] for k in z.files}
    rr={k:r[k if group=='fitting' else 'reconstructed_'+k] for k in ('hit','position','normal','depth')}
    original=dict(hit=a['hit'],position=a['positions'],normal=a['normals'],depth=a['depth'])
    if group=='fitting':
        _,m=g.prior.original.paired_surface(original,rr,a['directions'])
    else:
        _,m=g.prior.original.paired_surface({k:r['oracle_'+k] for k in ('hit','position','normal','depth')},rr,a['directions'])
    require(all(np.array_equal(r[k],v) for k,v in m.items()),'Accepted O1 correspondence mask changed')
    wo=-a['directions'].astype(float);wo/=np.linalg.norm(wo,axis=1,keepdims=True)
    ids=np.flatnonzero(m['valid'])
    with np.load(p['oracle']/'geometry/labels'/f'{camera}.npz',allow_pickle=False) as z:l={k:z[k] for k in z.files}
    ix=np.searchsorted(l['pixel_indices'],ids);np.testing.assert_array_equal(l['pixel_indices'][ix],ids)
    n0=l['normals'][ix];n1=rr['normal'][ids]
    keep=((n0*wo[ids]).sum(1)>0)&((n1*wo[ids]).sum(1)>0);ids=ids[keep];ix=ix[keep]
    n0=l['normals'][ix];n1=rr['normal'][ids];n=n0 if kind=='oracle' else n1
    pos=l['position'][ix] if kind=='oracle' else rr['position'][ids].astype(float)
    field=g.analytic(pos,n,bounds)
    with np.load(p['oracle']/'references'/f'{camera}.npz',allow_pickle=False) as z:
        full=z['linear_components'].reshape(-1,3,3);coarse=z['coarse_components'].reshape(-1,3,3)
    values=dict(position=pos,normals=n,outgoing=wo[ids],pixel_indices=ids,shader_valid=np.ones(len(ids),bool),
        hit_mask=(a['hit'] if kind=='oracle' else rr['hit']).reshape(128,128),
        axis=field['axis'],Q_world=field['Q'],widths_major_minor=field['widths_major_minor'],color=field['color'],metallic=field['metallic'],
        linear_components=full[ids],coarse_components=coarse[ids],full_linear_components=full,
        oracle_position=l['position'][ix],oracle_normals=n0,oracle_Q=l['Q_world'][ix],
        oracle_widths=l['widths_major_minor'][ix],oracle_axis=(np.cos(l['theta_major'][ix])[:,None]*l['T'][ix]+
            np.sin(l['theta_major'][ix])[:,None]*l['B'][ix]),oracle_color=l['color'][ix],oracle_metallic=l['metallic'][ix])
    # True isotropy does not define an orientation; Q remains well-defined.
    undefined=~l['orientation_defined'][ix]
    if undefined.any():values['oracle_axis'][undefined]=l['T'][ix][undefined]
    coverage=dict(oracle_hit=a['hit'],O1_hit=rr['hit'],common=m['common'],valid=m['valid'],
        positive_pool=np.isin(np.arange(16384),ids),distance_failure=m['common']&~m['depth_close'],
        normal_failure=m['common']&~m['signed_normal_ok'],facing_failure=m['common']&~m['both_front'])
    return values,coverage


def dataset(f,p,kind):
    admit(p);v,_=g.mesh_arrays(p['oracle']/'geometry/mesh.ply');bounds=np.stack([v.min(0),v.max(0)]).astype(float)
    data={c:view(c,kind,p,bounds)[0] for c in panel()}
    for row in f['support']['cameras']:
        require(len(data[row['camera']]['pixel_indices'])==row['common'],'Frozen common support differs')
    return read(p['oracle']/'geometry/generation.json'),data


def fit_pool(data):
    return {k:np.concatenate([data[int(c)][k] for c in split()['train']]) for k in FIT_KEYS}


def support_identity(data):
    h=sha256();rows=[]
    for c in panel():
        v=data[c];ids=v['pixel_indices'];h.update(np.array([c,len(ids)],dtype='<u4').tobytes());h.update(ids.astype('<u4').tobytes())
        h.update(v['linear_components'][:,2].astype('<f8').tobytes());h.update(v['outgoing'].astype('<f8').tobytes())
        rows.append(dict(camera=c,common=len(ids),split='fitting' if str(c) in split()['train'] else 'validation' if c==58 else 'held_out'))
    count=sum(r['common'] for r in rows if r['split']=='fitting')
    return dict(cameras=rows,fitting_observations=count,evaluation_observations=sum(r['common'] for r in rows),
        observations_and_targets_sha256=h.hexdigest(),photometric_index_sha256=index_hash(count),
        training_observation_exposures=10000*512,effective_training_passes=10000*512/count,
        restriction='oracle-assisted same-ray correspondence, displacement<=.02, signed normals<=30deg, both actual normals positive')


def completed(batch,arm,f,commit,sha,auth_sha):
    r=read(batch/arm/'recovery.json');expected=f['arms'][arm]['work']
    require((batch/f'{arm}-exit-code.txt').read_text().strip()=='0' and
        (batch/f'{arm}-container-absent.txt').read_text().strip()=='absent' and r['status']=='complete' and
        r['scientific_work_complete'] and r['cleanup_verified'] and r['optimizer_returns']==10000 and
        r['commit']==commit and r['freeze_sha256']==sha and r['authorization_sha256']==auth_sha and
        r['protocol']==PROTOCOL and r['counters']==expected and [x['step'] for x in r['saved_endpoints']]==STEPS and
        r['initial']==read(Path(f['base_initial']['path'])) and r['known_lighting_audit']['learned_module_calls']==0,
        'Incomplete/foreign Gate 13 history')
    for name,i in r['outputs'].items():check(batch/arm/name,i)
    require(digest(batch/arm/'photometric-indices.u32le')==f['support']['photometric_index_sha256'],'Photometric stream differs')
    if arm=='K1-O1':
        a=read(batch/'K1-Oracle/recovery.json')
        require(a['initial']==r['initial'] and digest(batch/arm/'sampling-tapes.jsonl')==digest(batch/'K1-Oracle/sampling-tapes.jsonl'),
                'Pair weights or actual random-sampling streams differ')
    return r
