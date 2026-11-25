"""Bind a single chart history to the accepted Gate 4 oracle comparator."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from benchmarks import donut_matched_geometry_recovery_v1 as m
from benchmarks.donut_recovery_v1 import numeric
from experiments.bell_final_tensor_admission_v1 import AUDIT
from experiments.bell_final_tensor_trainer_v1 import REGULARIZATION
from infrastructure.signatures import require

ROOT = m.ROOT
PROTOCOL = 'donut-representation-gate5-v1'
FREEZE = 'docs/protocols/DONUT_REPRESENTATION_GATE5_V1.freeze.json'
GATE4_SHA = '3dfcd4b6c9ad5ac4549dd11d93a4f0780a07820b48a03d0f66cae24e0033e517'
GATE4_MANIFEST = ROOT/'docs/results/publication_gate4/artifact-manifest.json'
GATE4_MANIFEST_SHA = 'c17ddfb54c647d2d09fb33898484de11a229654f5acaece89dd3e0b7425ff67c'
ARM = 'Final-Chart-Oracle'
AMBIENT = 'Ambient-Oracle'
CONTROLS = ('K1-Oracle', AMBIENT)
CAPTURES = dict(paired='paired-initial-gate.npz', physical='tensor-physical-admission.npz',
                native='native-reference-gate.npz')
MOUNTS = dict(m.MOUNTS, chart_initial='/inputs/chart-initial.json',
              **{key:f'/inputs/gate4-{name}' for key,name in CAPTURES.items()})
SOURCES = (
    'benchmarks/donut_representation_gate5_v1.py',
    'experiments/train_donut_representation_gate5_v1.py',
    'experiments/analyze_donut_representation_gate5_v1.py',
    'experiments/run_donut_representation_gate5_v1.sh',
    'tests/scientific/test_donut_representation_gate5_v1.py',
    'docs/protocols/DONUT_REPRESENTATION_GATE5_V1.md',
)
RESOURCES = dict(foreground_seconds=3600, kill_grace_seconds=30, inner_seconds=3540,
                 inner_grace_seconds=20, cpu_threads=4, host_memory_gib=64, pids=128,
                 cuda_allocator_fraction=.25, output_bytes=8*1024**3, gpu='0', gpu_model='A100')
PARITY = dict(initial_Q_atol=2e-8, initial_RGB_atol=1e-6, physical_atol=AUDIT['output_atol'],
              physical_rtol=AUDIT['output_rtol'], native_RGB_atol=1e-6)
read = m.rgb.geo.h.read
check = m.rgb.geo.h.check_artifact
digest = m.digest


def parent():
    return m.freeze(ROOT/m.FREEZE,GATE4_SHA)


def manifest():
    require(digest(GATE4_MANIFEST)==GATE4_MANIFEST_SHA, 'Accepted Gate 4 manifest differs')
    return read(GATE4_MANIFEST)


def definition(out, captures):
    from experiments.train_donut_matched_geometry_v1 import trainer_source

    f = parent()
    accepted = manifest()
    require(accepted['freeze']['sha256']==GATE4_SHA and accepted['support']==f['support'],
            'Accepted Gate 4 definition differs')
    bell = m.bell.freeze(ROOT/m.bell.FREEZE,m.BELL_SHA)
    a, b = (bell['arms'][k] for k in ('Final-Ambient','Final-Chart'))
    require(a.keys()==b.keys() and {k for k in a if a[k]!=b[k]}=={'initial','learned_control'},
            'Bell tensor pair differs beyond representation and initial state')
    comparator = f['arms'][AMBIENT]
    require(comparator['learned_control']==a['learned_control']=='D9-T'
            and b['learned_control']=='D9-C' and comparator['initial']==a['initial']
            and comparator['representation']==a['representation'] and comparator['illumination']=='known_L2'
            and comparator['spatial_policy'] is None
            and comparator['trainer_sha256']==sha256(trainer_source(True).encode()).hexdigest(),
            'Oracle comparator recipe differs')
    recipe = dict(comparator, initial=b['initial'], learned_control=b['learned_control'])
    require(set(captures)==set(CAPTURES)
            and all(set(v)=={'bytes','sha256'} for v in captures.values()), 'Unknown capture inventory')
    return dict(schema_version=1,protocol=PROTOCOL,host_output=str(out),arms={ARM:recipe},
                gate4_freeze_sha256=GATE4_SHA,gate4_manifest_sha256=GATE4_MANIFEST_SHA,
                gate4_controls={arm:accepted['arms'][arm]['receipt'] for arm in CONTROLS},
                ambient_captures=captures,support=f['support'],resources=RESOURCES,image_id=m.IMAGE,
                parity=PARITY,seeds=[6033],steps=m.STEPS,historical_checkpoint_loads_authorized=False,
                sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(path, sha):
    require(path==ROOT/FREEZE and path.resolve()==path and digest(path)==sha, 'Gate 5 freeze differs')
    f = read(path)
    out = Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-representation-gate5-')
            and f==definition(out,f['ambient_captures']), 'Gate 5 definition differs')
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
                arms=[ARM],maximum_execution_launches=1,image_id=m.IMAGE,resources=f['resources'],
                researcher_admitted_dataset=True,benchmark_generation_sha256=m.GENERATION,
                dataset_admission_sha256=m.ADMISSION,gate3_receipt_sha256=m.GATE3_RECEIPT,
                ambient_oracle_receipt_sha256=f['gate4_controls'][AMBIENT]['sha256'])


def paths(f, native=False):
    if native:
        return {k:Path(v) for k,v in MOUNTS.items()}
    ambient = Path(f['gate4_controls'][AMBIENT]['path']).parent
    return dict(m.paths(parent()),chart_initial=Path(f['arms'][ARM]['initial']['path']),
                **{key:ambient/name for key,name in CAPTURES.items()})


def control_receipts(f):
    accepted = manifest()
    batch = Path(accepted['batch'])
    auth = read(check(batch/'authorization.json',accepted['authorization']))
    check(batch/'freeze.json',accepted['freeze'])
    require(auth==m.authorization(f,accepted['execution_commit'],GATE4_SHA), 'Gate 4 authorization differs')
    result = {}
    for arm in CONTROLS:
        identity = accepted['arms'][arm]
        check(batch/f'{arm}-exit-code.txt',identity['exit_code'])
        check(batch/arm/'recovery.json',identity['receipt'])
        result[arm] = m.completed(batch,arm,f,auth['commit'],GATE4_SHA,accepted['authorization']['sha256'])
    return batch,auth,result


def check_inputs(f, p):
    inherited = parent()
    base = read(check(p['initial'],inherited['base_initial']))
    ambient = read(check(p['ambient_initial'],inherited['ambient_initial']))
    chart = read(check(p['chart_initial'],f['arms'][ARM]['initial']))
    require(chart['common_base_state_sha256']==ambient['common_base_state_sha256']==base['state_sha256']
            and chart['seed']==ambient['seed']==base['seed']==6033, 'Shared physical initialization anchor differs')
    for name,identity in f['ambient_captures'].items():
        check(p[name],identity)


def compare_capture(kind, saved, new):
    """Verify complete arrays using only the frozen field-specific tolerances."""
    require(kind in CAPTURES and saved.keys()==new.keys() and saved, 'Capture fields differ')
    drift = {}
    for field,a in saved.items():
        b = new[field]
        require(a.shape==b.shape and a.dtype==b.dtype and a.size
                and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid capture field '+field)
        atol = rtol = 0.
        if kind=='physical':
            require(field.split('_',1)[-1] in {'diffuse','specular','half','reflected','ndf','pdf'},
                    'Unknown tensor diagnostic field '+field)
            atol,rtol = PARITY['physical_atol'],PARITY['physical_rtol']
        elif kind=='paired':
            key = field.split('_',1)[-1]
            if key=='world_Q':
                atol = PARITY['initial_Q_atol']
            elif key in {'linear_rgb','linear_diffuse','linear_specular'}:
                atol = PARITY['initial_RGB_atol']
            else:
                require(key in {'kd','metallic'}, 'Unknown initial capture field '+field)
        else:
            require(field in {'native_draws','reference','coarse','pixel_indices'}, 'Unknown native capture field '+field)
            if field=='native_draws':
                atol = PARITY['native_RGB_atol']
        require(np.allclose(b,a,atol=atol,rtol=rtol) if atol or rtol else np.array_equal(b,a),
                'Historical comparator differs: '+kind+'/'+field)
        drift[field] = float(np.max(np.abs(b.astype(float)-a.astype(float))))
    return drift


def replay_captures(f, p, root):
    return {kind:compare_capture(kind,numeric(check(p[kind],f['ambient_captures'][kind])),numeric(root/name))
            for kind,name in CAPTURES.items()}


def completed(batch, f, auth, sha, auth_sha):
    r = read(batch/ARM/'recovery.json')
    require((batch/f'{ARM}-exit-code.txt').read_text().strip()=='0'
            and r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified']
            and r['historical_arm_initial_parity_verified'] and r['protocol']==PROTOCOL and r['arm']==ARM
            and r['commit']==auth['commit'] and r['freeze_sha256']==sha and r['authorization_sha256']==auth_sha
            and r['geometry']=='oracle' and r['matched_support']==f['support']
            and r['reference_Gate3_receipt_sha256']==m.GATE3_RECEIPT and r['optimizer_returns']==10000
            and [step['step'] for step in r['saved_endpoints']]==m.STEPS
            and r['counters']==f['arms'][ARM]['work'] and r['known_lighting_audit']['learned_module_calls']==0
            and r['representation']==f['arms'][ARM]['representation'] and r['effective_regularization']==REGULARIZATION
            and r['paired_initial_gate']['passed'] and r['physical_tensor_admission']['passed'],
            'Incomplete or foreign Final-Chart-Oracle history')
    stream = m.bell.dense.STREAM
    expected = dict(bytes=20480000,sha256=f['support']['photometric_index_sha256'])
    require(r['outputs'][stream]==expected, 'Native fitting stream differs')
    check(batch/ARM/stream,expected)
    replay = replay_captures(f,paths(f),batch/ARM)
    require(r['physical_tensor_admission']['gate4_ambient_replay']==replay,
            'Historical parity receipt differs from saved captures')
    return r
