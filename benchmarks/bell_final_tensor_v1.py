"""Final matched chart/ambient Bell comparison over the admitted dense pool."""
import os
from pathlib import Path

from benchmarks import bell_dense_known_l2_v1 as dense
from benchmarks import donut_tensor_d9 as d9
from experiments.bell_final_tensor_admission_v1 import AUDIT
from experiments.bell_final_tensor_trainer_v1 import transformed_sha256
from infrastructure.signatures import require

b = dense.b
ROOT, IMAGE, STEPS = dense.ROOT, dense.IMAGE, dense.STEPS
PANEL, SPLITS = dense.PANEL, dense.SPLITS
GENERATION, ADMISSION = dense.GENERATION, dense.ADMISSION
MATCH = {'Final-Chart':'D9-C','Final-Ambient':'D9-T'}
ARMS = tuple(MATCH)
FREEZE = 'docs/protocols/BELL_FINAL_TENSOR_PAIR_V1.freeze.json'
PROTOCOL = 'bell-final-chart-ambient-dense-known-L2-v1'
DENSE_SHA = '1c7887d33f8f8ba3fb28cc48d795636ee2955eedbeebc7ead56bdef26736faf7'
DENSE_RESULTS = 'docs/results/bell_dense_known_l2_recovery_v2/artifact-manifest.json'
D9_RESULTS = 'docs/results/donut_paired_tensor_d9_v1/artifact-manifest.json'
FOREGROUND_SECONDS = 7500
INITIALIZATION = 'validated_D9_weak_start_v1'


def __getattr__(name):
    return getattr(dense,name)


def dense_definition():
    return dense.freeze(ROOT/dense.FREEZE,DENSE_SHA)


def recipe(arm, prior, original):
    v = prior['arms']['K1-Dense']
    work = dict(v['work'],shader_calls=v['work']['shader_calls']+2,
                shadow_rays=v['work']['shadow_rays']+2*AUDIT['points']*1536,
                tensor_admission_predictor_calls=2,tensor_admission_reverse_sweeps=2)
    return dict(v,initial=original,learned_control=MATCH[arm],representation=d9.REPRESENTATION,
                limits=dict(v['limits'],**work),work=work)


def d9_initials():
    manifest = b.read(ROOT/D9_RESULTS)
    root = Path(manifest['batch'])
    return {arm:{**manifest['records'][str(root/method/'initial.json')], 'path':str(root/method/'initial.json')}
            for arm,method in MATCH.items()}


def freeze(path, sha):
    require(path.resolve()==path and b.digest(path)==sha,'Final tensor comparison freeze differs')
    f = b.read(path)
    old = dense_definition()
    fixed = {k:v for k,v in old.items() if k not in
             ('sources','protocol','order','arms','host_output','foreground_seconds','failed_launch','controls')}
    expected = d9_initials()
    manifest = b.read(ROOT/DENSE_RESULTS)
    require(all(f[k]==v for k,v in fixed.items()) and f['protocol']==PROTOCOL and f['order']==list(ARMS)
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['host_output']!=old['host_output']
            and f['arms']=={arm:recipe(arm,old,expected[arm]) for arm in ARMS}
            and f['controls']=={'K1-Dense':manifest['arms']['K1-Dense']['receipt']}
            and f['dense_analysis']==manifest['offline_analysis'][0]
            and f['baseline_freeze_sha256']==DENSE_SHA and f['initialization']==INITIALIZATION
            and f['tensor_admission']==AUDIT and f['trainer_overlay_sha256']==transformed_sha256(),
            'Final pair recipe, dense reference or overlay changed')
    for name,want in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and b.digest(p)==want,'Frozen final pair source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(dense.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),
                foreground_seconds=FOREGROUND_SECONDS,tensor_admission=AUDIT,initialization=INITIALIZATION)


def verify_controls(f):
    manifest = b.read(ROOT/DENSE_RESULTS)
    item = manifest['arms']['K1-Dense']
    base = dense.read_artifact(f['controls']['K1-Dense'])
    summary = dense.read_artifact(f['dense_analysis'])
    require(base['status']=='complete' and base['scientific_work_complete'] and base['cleanup_verified']
            and base['optimizer_returns']==10000 and base['arm']=='K1-Dense'
            and base['freeze_sha256']==DENSE_SHA and base['dataset_admission_sha256']==ADMISSION
            and summary['status']=='verified_complete' and summary['arms']['K1-Dense']['receipt_sha256']==item['receipt']['sha256']
            and b.check_artifact(Path(item['exit_code']['path']),item['exit_code']).read_text().strip()=='0',
            'Expected completed verified K1-Dense reference')
    stream = base['outputs'][dense.STREAM]
    require(stream==dict(bytes=20480000,sha256=f['photometric_index_sha256']),
            'K1-Dense fitting indices differ from the planned pair')
    b.check_artifact(Path(f['controls']['K1-Dense']['path']).parent/dense.STREAM,stream)
    control_initial = dense.read_artifact(f['base_initial'])
    for arm in ARMS:
        state = dense.read_artifact(f['arms'][arm]['initial'])
        require(state['common_base_state_sha256']==control_initial['state_sha256'] and state['seed']==6033,
                'Unexpected D9 initialization state')


def runtime(args):
    f = freeze(args.freeze,args.freeze_sha256)
    a = b.read(args.authorization)
    require(args.phase in ARMS and args.image_id==IMAGE and b.digest(args.authorization)==args.authorization_sha256
            and a==authorization(f,args.commit,args.freeze_sha256),'Exact final tensor pair authorization required')
    require(os.geteuid()!=0 and args.output==Path('/outputs')/args.phase and args.output.resolve()==args.output
            and not args.output.exists() and not os.statvfs('/outputs').f_flag&os.ST_RDONLY,
            'Fresh non-root final pair output required')
    for p in (Path('/'),ROOT,args.freeze,args.oracle,args.authorization,args.benchmark,args.admission,
              args.verification,args.initial,args.arm_initial):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only final pair source, inputs and root required')
    require({line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Networkless execution required')
    b.check_artifact(args.initial,f['base_initial'])
    method = f['arms'][args.phase]
    b.check_artifact(args.arm_initial,method['initial'])
    require(args.initial_sha256==f['base_initial']['sha256'] and args.benchmark_sha256==GENERATION,
            'Wrong final pair reference or base initialization')
    plan = dict(method,initial_sha256=args.initial_sha256,benchmark_generation_sha256=GENERATION,protocol=PROTOCOL)
    return dict(f,arm_training=plan,tensor_training=dict(plan,arms=list(d9.ARMS),
                initialization_policy=INITIALIZATION,surface_q_enabled=False)),a
