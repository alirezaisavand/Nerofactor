"""Dense fitting pool for the accepted known-L2 Bell recovery recipes."""
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from benchmarks import bell_dense_reference_v1 as dense
from benchmarks import bell_dense_reference_v2 as reference
from benchmarks import bell_heterogeneous_training_v1 as native
from benchmarks import bell_known_l2_completion_v3 as completion
from benchmarks import bell_known_l2_v1 as known
from benchmarks.donut_surface_q_d6 import fitting_face_counts
from infrastructure.signatures import require

b = dense.b
ROOT, IMAGE, STEPS = known.ROOT, known.IMAGE, known.STEPS
PANEL, SPLITS = dense.PANEL, dense.SPLITS
FIT_KEYS, TARGET_KEYS = native.FIT_KEYS, native.TARGET_KEYS
CANDIDATE = 'docs/evidence/BELL_DENSE_KNOWN_L2_ADMISSION_CANDIDATE_V1.json'
REFERENCE_MANIFEST = 'docs/results/bell_dense_reference_v2/artifact-manifest.json'
CONTROL_MANIFEST = 'docs/results/bell_known_l2_recovery_v1/artifact-manifest.json'
REFERENCE_SHA = 'fa2c6a69e1f7347e9ecdae206a3fe881a28fea22f6ec89d691e499c1b476a6fd'
SPARSE_V3_SHA = 'fc415bff93a2aab011cfe892d1ca5eaa232aef4d2cedbab2fc69823d5b48b6d2'
GENERATION = 'd68db52c2d491f2fbf4870b1021bec0240355ae1e7f7c9c33ef1390622a027b4'
VERIFICATION = 'f0081135df01a7a89f0a640f6fd8e166a9aabfc9bd7fc5737ecaaed8ed9a1cd4'
ADMISSION = '87197a0bc7e5cec3a320ed92193ac25e238151b2cfb408af58a79b0c20e7399c'
MATCH = {arm+'-Dense':arm for arm in known.ARMS}
ARMS = tuple(MATCH)
PROTOCOL = 'bell-kurt-dense-known-L2-recovery-v1'
V1_FREEZE = 'docs/protocols/BELL_DENSE_KNOWN_L2_RECOVERY_V1.freeze.json'
V1_SHA = '49b2235d7ed716e77f8067a63a859cf8fe50cc2dbffc7bf84cbd89b1b1af83c6'
V1_COMMIT = '977402c282691458ce5210573c8d31b0a28dff0a'
V1_AUTH_SHA = '8e4e40ccbd52ffd8c17247cc9a6ca864b48d16cca633528dff0f0fa2d9fcb24d'
FREEZE = 'docs/protocols/BELL_DENSE_KNOWN_L2_RECOVERY_V2.freeze.json'
STREAM = 'photometric-indices.u32le'
FOREGROUND_SECONDS = known.FOREGROUND_SECONDS
INDEX_DTYPE = np.dtype('<u4')


def __getattr__(name):
    return getattr(known,name)


def sparse_definition():
    first = completion.prior_definition()
    last = completion.freeze(ROOT/completion.FREEZE,SPARSE_V3_SHA)
    require(all(first['arms'][arm]==last['arms'][arm] for arm in last['arms']),
            'Accepted sparse recipe definitions diverge')
    return first


def recipe(arm, sparse):
    v = sparse['arms'][MATCH[arm]]
    work = dict(v['work'])
    paired = 2 if v['representation'] is not None else 0
    plan = dense.training_projection()
    work['shader_calls'] = 10000+32+paired+plan['evaluation_shader_calls']
    work['shadow_rays'] = (plan['training_slots']+(32+paired)*24+plan['evaluation_shading_observations'])*1536
    return dict(v, work=work, limits=dict(v['limits'],**work))


def json_identity(value):
    payload = (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    return dict(bytes=len(payload),sha256=sha256(payload).hexdigest())


def read_artifact(item):
    return b.read(b.check_artifact(Path(item['path']),item))


def candidate(manifest):
    return dict(schema_version=1,protocol='bell-dense-dataset-admission-v1',
                status='pending_researcher_admission',researcher_admitted_dataset=False,
                training_authorized=False,package=manifest['package'],scope=list(ARMS),splits=SPLITS,
                generation=manifest['generation'],verification=manifest['verification'],
                replay_policy=manifest['replay_policy'],held_out_support=manifest['held_out_support'],
                observed_fitting_points=dense.training_projection()['fitting_observations'],
                observed_held_out_points=dense.training_projection()['held_observations'])


def admission_record(pending):
    require(pending['status']=='pending_researcher_admission' and pending['researcher_admitted_dataset'] is False
            and pending['training_authorized'] is False, 'Expected the pending admission candidate')
    return dict(pending,status='researcher_admitted',researcher_admitted_dataset=True)


def reference_views(root, verify_path, manifest):
    """Verify the complete accepted reference before exposing the ordered observations."""
    g,v = (b.read(b.check_artifact(path,manifest[key])) for key,path in
           (('generation',root/'generation.json'),('verification',verify_path)))
    require(manifest['generation']['sha256']==GENERATION and manifest['verification']['sha256']==VERIFICATION
            and v['generation_sha256']==GENERATION and g['freeze_sha256']==v['freeze_sha256']==REFERENCE_SHA
            and g['protocol']==v['protocol']==reference.PROTOCOL and g['splits']==v['splits']==SPLITS
            and g['panel']==PANEL and g['field']==b.FIELD and g['environment']==b.ENVIRONMENT and g['BRDF']==b.BRDF
            and g['backend']==b.identity() and g['indirect']=='zero' and g['point_light']=='absent'
            and g['status']=='complete_pending_researcher_admission' and g['scientific_work_complete'] and g['cleanup_verified']
            and v['status']=='verified_pending_researcher_admission' and not v['researcher_admitted_dataset']
            and not v['training_authorized'] and g['outputs']==v['outputs'] and len(v['outputs'])==146,
            'Dense reference receipt differs')
    for name,item in v['outputs'].items():
        require(Path(name).name==name,'Invalid dense reference member')
        b.check_artifact(root/name,item)
    views = {c:b.numeric(root/f'view-{c}.npz') for c in PANEL}
    for camera,view in views.items():
        valid = view['shader_valid']
        require(valid.dtype==bool and len(valid)==dense.COUNTS[camera] and valid.all()
                and np.array_equal(view['pixel_indices'],np.flatnonzero(view['hit_mask'])),
                'Dense reference valid hit inventory differs')
    return g,views


def dataset(f, root, admission, verification):
    pending = b.read(ROOT/CANDIDATE)
    record = b.read(b.check_artifact(admission,f['dataset']['admission']))
    require(record==admission_record(pending), 'Explicit dense researcher admission required')
    return reference_views(root,verification,f['dataset'])


def fit_pool(data):
    return {k:np.concatenate([data[camera][k][data[camera]['shader_valid']] for camera in SPLITS['fitting']])
            for k in FIT_KEYS}


def index_batches(size, seed, updates, batch):
    require(batch<=size<2**32 and updates>0,'Invalid fitting stream extent')
    rng = np.random.RandomState(seed)
    order = rng.permutation(size)
    cursor = 0
    for _ in range(updates):
        if cursor+batch>size:
            order = rng.permutation(size)
            cursor = 0
        yield order[cursor:cursor+batch]
        cursor += batch


def index_hash(size, seed=known.POLICY['seed'], updates=10000, batch=512):
    h = sha256()
    for chosen in index_batches(size,seed,updates,batch):
        h.update(chosen.astype(INDEX_DTYPE,copy=False).tobytes())
    return h.hexdigest()


def original_launch():
    path = ROOT/V1_FREEZE
    require(b.digest(path)==V1_SHA,'Historical dense training freeze differs')
    return b.read(path)


def verify_prior_launch(f):
    old = original_launch()
    root = Path(old['host_output'])
    files = f['failed_launch']['artifacts']
    require(root.resolve()==root and {p.name for p in root.iterdir()}==set(files)=={'authorization.json','freeze.json'},
            'Unexpected evidence in the pre-Docker failure directory')
    for name,item in files.items():
        b.check_artifact(root/name,item)
    auth = root/'authorization.json'
    require(b.digest(auth)==V1_AUTH_SHA and b.digest(root/'freeze.json')==V1_SHA
            and b.read(auth)==authorization(old,V1_COMMIT,V1_SHA)
            and b.digest(root.with_name(root.name+'-authorization.json'))==V1_AUTH_SHA,
            'Pre-Docker failure launch identity differs')


def freeze(path, sha):
    require(path.resolve()==path and b.digest(path)==sha,'Dense training freeze differs')
    f = b.read(path)
    old = original_launch()
    inherited = {k:v for k,v in old.items() if k not in ('host_output','sources')}
    require(all(f[k]==value for k,value in inherited.items())
            and f['host_output']!=old['host_output']
            and f['failed_launch']['host_output']==old['host_output']
            and f['failed_launch']['commit']==V1_COMMIT and f['failed_launch']['freeze_sha256']==V1_SHA
            and f['failed_launch']['authorization_sha256']==V1_AUTH_SHA,
            'Repaired launcher must preserve the approved scientific definition')
    sparse = sparse_definition()
    ref = b.read(ROOT/REFERENCE_MANIFEST)
    pending = b.read(ROOT/CANDIDATE)
    require(pending==candidate(ref) and f['dataset']['generation']==pending['generation']
            and f['dataset']['verification']==pending['verification']
            and f['dataset']['admission']==dict(path=str(Path(ref['package']).parent)+'-admission.json',
                                             **json_identity(admission_record(pending))),
            'Dense admission candidate or expected record differs')
    fixed = ('mesh','base_initial','boundary_policy','boundary_probe_count','integer_probe_hashes','sampler_snapshots',
             'policy','lighting_policy','shader','steps')
    counts = f['integer_domain_hashes']
    require(all(f[key]==sparse[key] for key in fixed) and f['arms']=={arm:recipe(arm,sparse) for arm in ARMS}
            and {k:v for k,v in counts.items() if k!='fitting_camera_count'}==
                {k:v for k,v in sparse['integer_domain_hashes'].items() if k!='fitting_camera_count'}
            and 'fitting_camera_count' in counts and f['protocol']==PROTOCOL and f['reference_freeze_sha256']==REFERENCE_SHA
            and f['panel']==PANEL and f['splits']==SPLITS and f['image']==IMAGE and f['order']==list(ARMS)
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['host_reference']==ref['package']
            and f['held_out_support']==pending['held_out_support']
            and f['dataset']['admission']['sha256']==ADMISSION
            and f['photometric_index_sha256']==index_hash(dense.training_projection()['fitting_observations'])
            and f['training_projection']==dense.training_projection()
            and f['sparse_analysis']==b.read(ROOT/CONTROL_MANIFEST)['offline_analysis'][1],
            'Dense acquisition or accepted recipe changed')
    for name,value in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and b.digest(p)==value,'Frozen Dense Bell source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(native.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),
        foreground_seconds=FOREGROUND_SECONDS,lighting_policy=known.LIGHT_POLICY,
        benchmark_generation_sha256=GENERATION,verification_sha256=VERIFICATION,
        dataset_admission_sha256=f['dataset']['admission']['sha256'],photometric_index_sha256=f['photometric_index_sha256'])


def verify_controls(f):
    manifest = b.read(ROOT/CONTROL_MANIFEST)
    for arm,control in MATCH.items():
        item = manifest['arms'][control]
        saved = read_artifact(item['receipt'])
        require(item['receipt']==f['controls'][arm] and saved['arm']==control and saved['status']=='complete'
                and saved['optimizer_returns']==10000 and saved['scientific_work_complete'] and saved['cleanup_verified']
                and saved['effective_lighting']=='known analytic L2'
                and b.check_artifact(Path(item['exit_code']['path']),item['exit_code']).read_text().strip()=='0'
                and saved['initial']==read_artifact(f['arms'][arm]['initial']), 'Accepted sparse control differs')


def runtime(args):
    f = freeze(args.freeze,args.freeze_sha256)
    a = b.read(args.authorization)
    require(args.phase in ARMS and args.image_id==IMAGE and b.digest(args.authorization)==args.authorization_sha256
            and a==authorization(f,args.commit,args.freeze_sha256),
            'Exact separate dense Bell training authorization required')
    require(os.geteuid()!=0 and args.output==Path('/outputs')/args.phase and args.output.resolve()==args.output
            and not args.output.exists() and not os.statvfs('/outputs').f_flag&os.ST_RDONLY,
            'Fresh non-root dense Bell output required')
    for p in (Path('/'),ROOT,args.freeze,args.oracle,args.authorization,args.benchmark,args.admission,
              args.verification,args.initial,args.arm_initial):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only dense Bell source, inputs and root required')
    require({line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Networkless execution required')
    b.check_artifact(args.initial,f['base_initial'])
    method = f['arms'][args.phase]
    b.check_artifact(args.arm_initial,method['initial'])
    require(args.initial_sha256==f['base_initial']['sha256'] and args.benchmark_sha256==GENERATION,
            'Wrong dense Bell reference or base initialization')
    plan = dict(method,initial_sha256=args.initial_sha256,benchmark_generation_sha256=GENERATION,protocol=PROTOCOL,
                D6_domain_array_sha256=f['integer_domain_hashes'],D6_sampler_snapshots=f['sampler_snapshots'])
    active = dict(f,arm_training=plan)
    if method['representation'] is not None:
        active['tensor_training'] = dict(plan,arms=list(native.d8.ARMS))
    return active,a


def check_domain_archive(f, domain, views, receipt_hashes, captured):
    expected = dict(domain,fitting_camera_count=fitting_face_counts(views,SPLITS['fitting'],len(domain['face_area'])))
    known.check_probe_archive(expected,captured)
    hashes = {k:native.array_hash(v) for k,v in captured.items()}
    require(receipt_hashes==hashes and all(hashes[k]==v for k,v in f['integer_domain_hashes'].items()),
            'Dense Bell neighborhood archive hash differs')
