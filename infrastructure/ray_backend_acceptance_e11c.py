"""File-only E11C replay/admission. Never loads extensions or invokes CUDA."""
import hashlib,math,subprocess
from pathlib import Path
import numpy as np
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import ROOT,digest
from infrastructure.ray_backend_e11c import BASE,ORIGINAL_BINARY,PATCH_SHA,LIMITS,TOL,compare,authorize

# Unit roundoff for binary32 on a unit-scale diagnostic: eps32/2 = 2^-24.
# This tolerance applies ONLY to replayed facing_dot, never scientific decisions.
FACING_DOT_REPLAY_ABS=float(np.finfo(np.float32).eps/2)
REPLAY_PROTOCOL='e11c-offline-replay-v2'
NATIVE_COMMIT='6af0474dc0167a64ad1dfedcc3b1f5f4dc049ca2'
NATIVE_FREEZE='f55d193f5d2cc05287da36c73c2a50575cbaa941d71e4324f8f4733286e62925'


def replay_cases(saved,computed,scene):
    mismatches=[];roundoff=[]
    def walk(a,b,path):
        def reject(reason):mismatches.append(dict(path=list(path),reason=reason,saved=repr(a),replayed=repr(b)))
        if type(a)!=type(b):reject('type mismatch');return
        if isinstance(a,float) and (not math.isfinite(a) or not math.isfinite(b)):
            reject('nonfinite number');return
        if isinstance(a,dict):
            if a.keys()!=b.keys():reject('missing or extra fields');return
            for k in a:walk(a[k],b[k],path+(k,))
        elif isinstance(a,list):
            if len(a)!=len(b):reject('list length mismatch');return
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+(i,))
        elif a!=b:
            facing=(len(path)==3 and path[0] in ('bell','cube') and type(path[1]) is int and path[2]=='facing_dot')
            if facing and type(a) is float and abs(a-b)<=FACING_DOT_REPLAY_ABS:
                roundoff.append(dict(path=list(path),saved=a,replayed=b,absolute_difference=abs(a-b)))
            else:reject('nonmatching value')
    walk(saved,computed,(scene,))
    return dict(passed=not mismatches,mismatches=mismatches,permitted_facing_roundoff=roundoff)


def read(path,expected=None):return read_report(path,expected or digest(path))[0]


def verify_record(path,record):
    p=Path(path);require(p.is_file() and p.resolve()==p and p.stat().st_size==record['bytes'] and digest(p)==record['sha256'],
                        'Artifact identity mismatch: '+str(p))


def analyze(root):
    root=Path(root);require(root.is_absolute() and root.resolve()==root,'Explicit unaliased execution root required')
    inputs=root.parent/'inputs';freeze_path=ROOT/'docs/protocols/RAY_BACKEND_E11C_V1.freeze.json'
    freeze=read(freeze_path,NATIVE_FREEZE)
    auth=root.parent/'authorization.json';authorize(auth,digest(auth),NATIVE_FREEZE,NATIVE_COMMIT,root.parent)
    require(freeze['work_limits']==LIMITS and freeze['tolerances']==TOL,'Original scientific limits/tolerances changed')
    # Verify executed source through its original Git objects; current offline code is a new version.
    for name,sha in freeze['sources'].items():
        original=subprocess.check_output(['git','show',NATIVE_COMMIT+':'+name],cwd=ROOT)
        require(hashlib.sha256(original).hexdigest()==sha,'Executed source identity mismatch: '+name)
    require({p.relative_to(inputs).as_posix() for p in inputs.rglob('*') if p.is_file()}==set(freeze['inputs']),'Staged input inventory mismatch')
    for name,sha in freeze['inputs'].items():require(digest(inputs/name)==sha,'Staged input changed: '+name)
    build_folder=root/'build/out';build_receipt=read(build_folder/'completion.json')
    require(build_receipt['status']=='complete','Build completion receipt failed')
    build=read(build_folder/'build.json',build_receipt['report_sha256'])
    require(build['status']=='complete' and build['commit']==NATIVE_COMMIT and build['freeze_sha256']==NATIVE_FREEZE,'Incomplete/misattributed build')
    require(build['source_manifest']==read(inputs/'source-manifest.json'),'Build/source manifest mismatch')
    require(build['original_binary']['sha256']==ORIGINAL_BINARY,'Historical binary mismatch')
    corrected=build['corrected_binary'];verify_record(build_folder/'corrected'/Path(corrected['path']).name,corrected)
    require(corrected['sha256']!=ORIGINAL_BINARY,'Derivative binary did not change')
    verify_record(build_folder/'compile.log',build['compile_log'])
    for r in build['build_files']:
        relative=Path(r['path']).relative_to('/outputs');require('..' not in relative.parts,'Unsafe build artifact path');verify_record(build_folder/relative,r)
    ctx=root/'image-context/e11c';require(digest(ctx/'build.json')==digest(build_folder/'build.json'),'Image-context build record differs')
    verify_record(ctx/'corrected'/Path(corrected['path']).name,corrected)
    require(digest(ctx/'predicate.patch')==PATCH_SHA and read(ctx/'source-manifest.json')==build['source_manifest'],'Image source/patch mismatch')
    for name,sha in build['source_manifest']['patched'].items():require(digest(ctx/'source'/name)==sha,'Image-context source changed')
    image=read(root/'image.json');base_layers=__import__('json').loads((root/'base-layers.json').read_text());layers=__import__('json').loads((root/'derivative-layers.json').read_text())
    require(image['base_id']==BASE and image['image_id']!=BASE and image['base_layers']==base_layers and image['layers']==layers and layers[:len(base_layers)]==base_layers,'Image ancestry metadata mismatch')
    records={};replays={};refs=read(inputs/'references.json')
    with np.load(inputs/'panels.npz',allow_pickle=False) as panel:
        for stage,backend in [('original','original'),('corrected','predicate_fp64')]:
            folder=root/stage/'out';receipt=read(folder/'completion.json');result=read(folder/'probe.json',receipt['report_sha256']);records[stage]=result
            require(receipt['status']==result['status']=='pass' and result['backend']==backend and result['commit']==NATIVE_COMMIT and
                    result['freeze_sha256']==NATIVE_FREEZE and result['cleanup_verified'] is True,'Native completion/identity failure')
            expected_counts=dict(attempted_calls=27,returned_calls=27,attempted_rays=819244,returned_rays=819244)
            require(type(result['counts']) is dict and result['counts'].keys()==expected_counts.keys() and
                    all(type(result['counts'][k]) is int and result['counts'][k]==v for k,v in expected_counts.items()),'Native work counts mismatch')
            require(result['input_sha256']==freeze['inputs'],'Native input identity mismatch')
            expected=build['original_binary'] if stage=='original' else corrected
            require(result['binary']['sha256']==expected['sha256'] and result['binary']['bytes']==expected['bytes'],'Loaded backend identity mismatch')
            artifacts={Path(x['path']).name:x for x in result['artifacts']}
            require(len(result['artifacts'])==len(artifacts)==3 and set(artifacts)=={'bell.npz','cube.npz','performance-last.npz'},'Native artifact inventory mismatch')
            for name,r in artifacts.items():verify_record(folder/name,r)
            require(set(result['cases'])=={'bell','cube'},'Native case inventory mismatch')
            replays[stage]={}
            for scene,count in [('bell',36),('cube',8)]:
                require(len(result['cases'][scene])==len(refs[scene])==count,'Native case count mismatch')
                with np.load(folder/(scene+'.npz'),allow_pickle=False) as capture:
                    require(set(capture.files)=={'positions','raw_normals','depth'},'Capture field mismatch')
                    computed=compare(panel[scene+'_o'],panel[scene+'_d'],capture['positions'],capture['raw_normals'],capture['depth'],refs[scene])
                replays[stage][scene]=replay_cases(result['cases'][scene],computed,scene)
            perf=result['performance'];require(type(perf['batch_size']) is int and perf['batch_size']==32768 and perf['warmups']==5,'Timing workload mismatch')
            for field in ('wall_ms','cuda_event_ms'):
                require(len(perf[field])==20 and all(type(x) is float and math.isfinite(x) and x>0 for x in perf[field]),'Invalid timing capture')
    original=records['original'];new=records['corrected'];k=refs['known_failure_index']
    require(original['baseline_reproduced'] is True and original['all_cases_match_independent_reference'] is False and
            abs(original['cases']['bell'][k]['depth']-refs['saved_failure_depth'])<=TOL['depth_abs'],'Historical failure reproduction invalid')
    require(new['baseline_reproduced'] is False and new['all_cases_match_independent_reference'] is True and
            all(x['passed'] is True for rows in new['cases'].values() for x in rows),'Corrected native agreement invalid')
    mismatches=[dict(backend=stage,scene=scene,**m) for stage,scenes in replays.items() for scene,r in scenes.items() for m in r['mismatches']]
    wall={s:float(np.median(x['performance']['wall_ms'])) for s,x in records.items()}
    gpu={s:float(np.median(x['performance']['cuda_event_ms'])) for s,x in records.items()}
    return dict(schema_version=2,protocol=REPLAY_PROTOCOL,status='PROCEED' if not mismatches else 'FAIL',
        native_source_commit=NATIVE_COMMIT,native_freeze_sha256=NATIVE_FREEZE,scientific_tolerances_unchanged=TOL,
        facing_dot_replay_absolute_tolerance=FACING_DOT_REPLAY_ABS,replay=replays,remaining_mismatches=mismatches,
        original_failure_reproduced=True,original_reference_matches=sum(x['passed'] for rows in original['cases'].values() for x in rows),
        corrected_reference_matches=sum(x['passed'] for rows in new['cases'].values() for x in rows),known_ray=new['cases']['bell'][k],
        original_binary=build['original_binary'],corrected_binary=corrected,image=image,
        timing=dict(wall_median_ms=wall,cuda_median_ms=gpu,wall_ratio=wall['corrected']/wall['original'],cuda_ratio=gpu['corrected']/gpu['original'],
                    substantial_penalty_over_2x=wall['corrected']/wall['original']>2,interpretation='Sequential same-image small panel, no significance or full-generation extrapolation'),
        original_native_report=read_report(root/'original/out/probe.json',digest(root/'original/out/probe.json'))[1],
        corrected_native_report=read_report(root/'corrected/out/probe.json',digest(root/'corrected/out/probe.json'))[1],
        build_record=read_report(build_folder/'build.json',digest(build_folder/'build.json'))[1],
        actual_native_ray_queries=1638488,new_gpu_queries=0,new_builds=0,globally_watertight=False,
        full_generation_authorized=False,material_training_authorized=False,
        image_identity_scope='Inspected researcher-captured immutable ID, layer ancestry and COPY-context binary/source hashes; no Docker or GPU re-execution')
