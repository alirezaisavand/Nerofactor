"""Read-only, hash-pinned aggregate C/C.1 admission. No shader/model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments import complete_neural_material_validation_v1 as c
from infrastructure.signatures import require, read_report

ROOT=c.ROOT
BASE='d331eda7837c620666e3ea5a9c7b7cfd27ad2dd8'
PROTOCOL='neural-material-validation-aggregate-v1'
COMPLETION_PATH='docs/evidence/NEURAL_MATERIAL_FIELD_VALIDATION_COMPLETION_V1.json'
COMPLETION_SHA='69f69d08b9b3893996dfc9794fa8d89f3c62ac402d6eb1959ac7f222bd4adb3a'
COMPLETION_RAW_SHA='9289801a3f7fc2120525354dcb09ee6760df18b4877fe24c63d03b81f1ac3901'
HARNESS='experiments/neural_material_recovery_v1.py'
SOURCE_PATHS=('experiments/validate_neural_material_recovery_v1.py','experiments/complete_neural_material_validation_v1.py',
              'tests/scientific/test_neural_material_validation_v1.py','tests/scientific/test_neural_material_validation_completion_v1.py')


def old_source_identity(name,sha,transitions):
    blob=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
    require(hashlib.sha256(blob).hexdigest()==sha,'Historical source Git blob mismatch: '+name)
    actual=c.v.h.digest(ROOT/name)
    require(actual==sha or name==HARNESS,'Unapproved change to historical scientific/validation file: '+name)
    if actual!=sha:transitions[name]=dict(admitted_historical_sha256=sha,current_sha256=actual,
        scope='C.2 admission/failure/output guards; unchanged field/shader/optimizer equations')


def verify_old_initial():
    """Reuse immutable C.1 numeric verifier; adapt ONLY its legacy-source checks.

    Its source is admitted from Git before compiling a private function copy.
    No global/module patching and no false digest substitution occurs: historical
    blobs and the current C.2 harness are recorded as distinct identities.
    """
    transitions={}
    for name in SOURCE_PATHS:
        expected=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)).hexdigest()
        require(c.v.h.digest(ROOT/name)==expected,'Original C/C.1 verifier changed: '+name)
    def admit_legacy():
        reference,ids=c.v.h.admit_reference()
        for name,sha in c.v.IMPLEMENTATION.items():old_source_identity(name,sha,transitions);ids[name]=sha
        return reference,ids
    def verify_legacy(name,sha):old_source_identity(name,sha,transitions)
    tree=ast.parse((ROOT/'experiments/complete_neural_material_validation_v1.py').read_text())
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_previous'))
    first=ast.parse('v.admit()',mode='eval').body
    target=next(n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and
        isinstance(n.iter.func,ast.Attribute) and isinstance(n.iter.func.value,ast.Subscript) and
        isinstance(n.iter.func.value.value,ast.Name) and n.iter.func.value.value.id=='old' and
        isinstance(n.iter.func.value.slice,ast.Constant) and n.iter.func.value.slice.value=='accepted_identities')
    require(len(target.body)==1 and isinstance(target.body[0],ast.Expr),'Unexpected legacy-source loop')
    legacy_call=target.body[0].value
    require(isinstance(legacy_call,ast.Call) and isinstance(legacy_call.func,ast.Name) and legacy_call.func.id=='require' and
        ast.dump(legacy_call.args[0])==ast.dump(ast.parse('v.h.digest(ROOT/name)==sha',mode='eval').body),'Legacy-source predicate changed')
    class SourceChecks(ast.NodeTransformer):
        initial=0;legacy=0
        def visit_Call(self,n):
            if ast.dump(n)==ast.dump(first):self.initial+=1;return ast.parse('admit_legacy()',mode='eval').body
            if n is legacy_call:self.legacy+=1;return ast.parse('verify_legacy(name,sha)',mode='eval').body
            return self.generic_visit(n)
    transform=SourceChecks();node=transform.visit(node)
    require((transform.initial,transform.legacy)==(1,1),'Unexpected legacy-admission adaptation')
    ns=dict(c.verify_previous.__globals__,admit_legacy=admit_legacy,verify_legacy=verify_legacy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'admitted-C1-saved-data-verifier','exec'),ns)
    result=ns['verify_previous']()
    return result,transitions


def verify_state(evidence,row,expected_options):
    selected=row['checkpoint'];manifest_path=evidence.path(selected['manifest'])
    manifest,_=read_report(manifest_path,selected['manifest_sha256'])
    require(manifest['purpose']=='state' and manifest['arm']==row['arm'] and manifest['completed_updates']==1 and
        manifest['optimizer_effective']==expected_options,'Wrong synthetic checkpoint state/options')
    archive=evidence.path(str(manifest_path.parent.relative_to(evidence.root)/manifest['archive']))
    require(c.v.h.digest(archive)==manifest['archive_sha256'] and archive.stat().st_size==manifest['archive_bytes'],'Checkpoint hash/size mismatch before loading')
    with np.load(archive,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    require(len(manifest['entries'])==len({e['key'] for e in manifest['entries']}) and
        set(arrays)=={e['key'] for e in manifest['entries']}|{'numpy_rng'},'Checkpoint tensor inventory mismatch')
    params=c.parameter_shapes(row['arm']);seen=set();moments={name:set() for name in params}
    for entry in manifest['entries']:
        pair=(entry['role'],entry['name']);require(pair not in seen,'Duplicate state tensor');seen.add(pair)
        a=arrays[entry['key']];t=torch.from_numpy(a)
        require(a.dtype.kind!='O' and np.isfinite(a).all() and c.v.field.record(t)=={k:entry[k] for k in ('dtype','device','shape','sha256')},'Checkpoint numeric catalogue mismatch')
        if entry['role']=='parameter':require(entry['name'] in params and list(a.shape)==params[entry['name']] and a.dtype==np.float32,'Wrong parameter tensor')
        elif entry['role']=='adam':
            name,key=entry['name'].rsplit('/',1);require(name in params and key in ('step','exp_avg','exp_avg_sq'),'Unknown Adam state')
            moments[name].add(key)
            require(a.dtype==np.float32 and (a.shape==() and float(a)==1 if key=='step' else list(a.shape)==params[name]),'Wrong Adam shape/dtype/counter')
            if key=='exp_avg_sq':require(bool((a>=0).all()),'Negative second moment')
        else:require(entry['role'] in ('buffer','torch_rng'),'Unknown state tensor role')
    require({name for role,name in seen if role=='parameter'}==set(params) and all(v=={'step','exp_avg','exp_avg_sq'} for v in moments.values()),'Incomplete parameter/moment state')
    require(hashlib.sha256(arrays['numpy_rng'].tobytes()).hexdigest()==manifest['numpy_rng']['array_sha256'],'State RNG hash mismatch')
    return dict(arm=row['arm'],manifest_sha256=selected['manifest_sha256'],archive_sha256=manifest['archive_sha256'],
        archive_bytes=manifest['archive_bytes'],parameter_tensors=len(params),verified_numeric_arrays=len(arrays),allow_pickle=False)


def check_matrix(old,completed):
    require(type(old) is dict and type(completed) is dict,'Both original failure and completion evidence are required')
    require(old.get('status')=='failed' and old.get('failure_type')=='TypeError','The original failure must stay failed')
    require(completed.get('status')=='passed' and completed.get('protocol')==c.PROTOCOL,'Valid completion required')
    require(old['work']=={k:dict(attempted=n,completed=n) for k,n in c.PRIOR_COUNTS.items()} and
        completed['work']=={k:dict(attempted=n,completed=n) for k,n in c.CAPS.items()},'Validation counters mismatch')
    require([(r['material_name'],r['seed'],r['arm']) for r in completed['new_replay']]==[(m,101,a) for m,a in c.replay_schedule()],
        'Missing/duplicate/substituted replay')
    require([(r['seed'],r['variant']) for r in completed['new_gauge']]==[(s,k) for s in c.v.field.SEEDS for k in ('periodic','uncoupled_swap','coupled_swap')],
        'Missing/duplicate/substituted gauge')
    require([r['arm'] for r in completed['storage']]==list(c.v.h.ARMS) and
        all(r['synthetic_updates']==1 for r in completed['storage']),'Wrong synthetic state matrix')
    require(old['recovery_updates']==completed['recovery_updates']==0 and completed['repeated_initial_renders']==0,'Validation contained recovery/repeated initial work')
    work={k:{n:old['work'][k][n]+completed['work'][k][n] for n in ('attempted','completed')} for k in c.CAPS}
    require(work==completed['combined_work'] and all(v['attempted']==v['completed']==c.v.CAPS[k] for k,v in work.items()),'Incomplete aggregate work')
    return work


def verify_aggregate():
    (reference,old,old_evidence,targets,tapes,ids,initial),transitions=verify_old_initial()
    summary,summary_id=read_report(ROOT/COMPLETION_PATH,COMPLETION_SHA)
    raw=summary['original_completion_report'];require(raw['sha256']==COMPLETION_RAW_SHA,'Unapproved C.1 owned report')
    path=Path(raw['path']);require(path.resolve()==path and path.stat().st_size==raw['bytes'],'Missing/aliased C.1 report')
    completed,raw_id=read_report(path,COMPLETION_RAW_SHA);evidence=c.Evidence(path.parent)
    check_matrix(old,completed)
    operational=[]
    for filename,key in (('pre-execution.json','pre_execution'),('process-outcome.json','external_execution')):
        selected=evidence.path(filename);data,meta=read_report(selected,c.v.h.digest(selected))
        require(data==summary[key],'C.1 operational manifest differs from the hash-pinned versioned evidence')
        operational.append(meta)
    require(summary['pre_execution']['sha256']==completed['validation_source_sha256'],'C.1 executed source manifest differs')
    for selected in (evidence.path('console.log'),Path(summary['external_execution']['console_path'])):
        require(selected.is_file() and selected.resolve()==selected and c.v.h.digest(selected)==summary['external_execution']['console_sha256'],
            'C.1 console evidence missing/changed')
        operational.append(dict(path=str(selected),sha256=c.v.h.digest(selected)))
    startup=summary['pre_execution']['preallocation_failure'];record,meta=read_report(startup['path'],startup['sha256'])
    require(record==summary['startup_input_rejection'] and record['scientific_operations']==record['model_constructions']==record['allocated_outputs']==0,
        'Startup rejection was not a zero-work preallocation failure')
    operational.append(meta)
    startup_console=Path(record['process_outcome']['console_path'])
    require(startup_console.is_file() and c.v.h.digest(startup_console)==record['process_outcome']['console_sha256'],'Startup console evidence changed')
    operational.append(dict(path=str(startup_console),sha256=c.v.h.digest(startup_console)))
    require(completed['protocol']==c.PROTOCOL and completed['status']=='passed' and completed['work']=={k:dict(attempted=v,completed=v) for k,v in c.CAPS.items()},'Wrong C.1 report/counters')
    require(completed['fixture_plan']==c.plan() and completed['recovery_updates']==completed['repeated_initial_renders']==0 and
        not completed['final_freeze_created'] and not completed['Phase_D_authorized'],'Invalid C.1 scientific scope')
    for name,sha in completed['validation_source_sha256'].items():old_source_identity(name,sha,transitions)
    require(completed['previous_evidence_verification']==initial,'C.1 old-evidence verification differs from independent recomputation')
    # Its metadata transition describes the B->C.1 epoch, not the new C.2 harness.
    entries={(r['material'],r['seed']):r for r in old['initial_admission']}
    expected=[(m,101,a) for m,a in c.replay_schedule()]
    require([(r['material_name'],r['seed'],r['arm']) for r in completed['new_replay']]==expected,'Missing/duplicate/substituted replay')
    replay=[]
    for row in completed['new_replay']:
        base=old_evidence.arrays(entries[(row['material_name'],row['seed'])]['arms'][row['arm']]['raw']);actual=evidence.arrays(row['raw'])
        require(c.v.array_catalog(actual)==c.v.array_catalog(base),'Non-exact saved deterministic replay')
        checks={key:c.v.comparison(actual[key],base[key]) for key in base}
        require(checks==row['checks'] and all(x['passed'] for x in checks.values()),'Replay diagnostics mismatch')
        replay.append(dict(material_name=row['material_name'],seed=row['seed'],arm=row['arm'],sha256=row['raw']['sha256'],passed=True))
    expected_gauges=[(s,k) for s in c.v.field.SEEDS for k in ('periodic','uncoupled_swap','coupled_swap')]
    require([(r['seed'],r['variant']) for r in completed['new_gauge']]==expected_gauges,'Missing/duplicate/substituted gauge')
    gauges=[];fit,external,tape_manifest=tapes
    for row in completed['new_gauge']:
        actual=evidence.arrays(row['raw']);base=old_evidence.arrays(entries[('M1',row['seed'])]['arms']['D0']['raw'])
        p=torch.from_numpy(base['physical'][0].copy());variant=row['variant']
        if variant=='periodic':p[2]+=np.pi
        else:p[:2]=p[[1,0]];p[2]+=np.pi/2
        require(np.array_equal(actual['physical'][0],p.numpy()),'Gauge physical coordinates substituted')
        if variant!='periodic':
            f0=np.stack((base['tangent'][0],base['bitangent'][0]),-1);f1=np.stack((actual['tangent'][0],actual['bitangent'][0]),-1)
            mapping=c.v.h.previous.coupling.frame_sign(f0,f1)
            require(mapping['sign']==row['mapping']['sign'],'Wrong signed frame coupling')
            if variant=='coupled_swap':
                _,record=c.v.h.previous.coupling.copy_coupled_tape(fit,mapping['sign'])
                require(record==row['coupled_tape'],'Wrong coupled tape identity')
        if variant=='uncoupled_swap':
            q0,q1=(c.v.h.previous.coupling.accepted.tensor_q(v) for v in (base['physical'][0],p.numpy()))
            check=c.v.comparison(q1,q0,2e-7,2e-6);require(check==row['physical_Q'] and check['passed'],'Gauge Q gate failed')
        else:
            check=c.v.parity(actual,base,targets['M1'],gauge=variant=='coupled_swap')
            require(check==row['comparison'] and check['passed'],'Gauge numerical gate failed')
        gauges.append(dict(seed=row['seed'],variant=variant,sha256=row['raw']['sha256'],passed=True))
    require([r['arm'] for r in completed['storage']]==list(c.v.h.ARMS) and all(r['synthetic_updates']==1 and r['exact_state_restored'] for r in completed['storage']), 'Synthetic state matrix incomplete')
    options=reference['Adam_runs'][0]['snapshots']['250']['optimizer_effective']
    states=[verify_state(evidence,r,options) for r in completed['storage']]
    feasibility=c.v.feasibility(old['timings']+completed['timings'],completed['storage'])
    require(feasibility==completed['runtime_feasibility'] and feasibility['passed'],'Invalid feasibility evidence')
    work={k:{n:old['work'][k][n]+completed['work'][k][n] for n in ('attempted','completed')} for k in c.CAPS}
    require(work==completed['combined_work'] and all(x['attempted']==x['completed']==c.v.CAPS[k] for k,x in work.items()),'Aggregate counts incomplete')
    for name in ('all_prescribed_fixtures_completed','full_shader_admission_passed','runtime_feasibility_passed',
                 'global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored'):
        require(completed[name] is True,'C.1 runtime/scope completion flag inconsistent')
    return dict(schema_version=1,protocol=PROTOCOL,status='passed',baseline=BASE,
        original_failed_report=initial['original_report'],original_failure_status='failed',original_failure_type='TypeError',
        original_versioned_sha256=c.PRIOR_SHA,completion_versioned=summary_id,completion_owned_report=raw_id,
        source_epoch_transitions=transitions,initial_admission=initial['independent_initial_admission'],
        previous_replay=initial['recovered_replay'],new_replay=replay,gauges=gauges,synthetic_states=states,
        combined_work=work,initial_comparisons=9,initial_shader_gradients=27,replay_total=9,gauge_total=9,
        recovery_updates=0,full_shader_admission_passed=True,runtime_feasibility_passed=True,
        runtime=reference['runtime'],runtime_feasibility=feasibility,artifacts=dict(previous=initial['verified_archives'],completion=list(evidence.verified.values())),
        operational_manifests=operational,
        scientific_calls_in_aggregate_verification=0)


def admit_selected(path,sha):
    stored,_=read_report(path,sha)
    require(stored==verify_aggregate(),'Aggregate report is not independently reproducible from its admitted evidence')
    return stored
