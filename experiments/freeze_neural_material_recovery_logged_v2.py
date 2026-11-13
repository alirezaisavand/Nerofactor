"""D.3 deterministic final freeze; saved-data validation only, never execution."""
import argparse
import copy
import json
from pathlib import Path

from experiments import neural_material_recovery_logged_v2 as d2
from experiments import freeze_neural_material_recovery_v1 as legacy
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.neural_material_resources_v2 import budget,capacity,selected,D2,D2_SHA
from infrastructure.signatures import require

ROOT=Path(__file__).resolve().parents[1]
BASE='7580126b8d30fd7ad6621dc673e473d49aebfe2d'
PROTOCOL='neural-material-recovery-logged-v2'
PROTOCOL_PATH='docs/protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.md'
FREEZE='docs/protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.freeze.json'
READINESS='docs/evidence/NEURAL_MATERIAL_RECOVERY_LOGGED_V2_READINESS.json'
OLD_FREEZE='docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json'
D1='docs/evidence/NEURAL_MATERIAL_REPORTING_AUDIT_V1.json'
D1_SHA='44253370392a110b571ccd67fa6c9ff128af3a916e20a5c33eaba0ac933e37dc'
RAW_SHA='4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49'
PARENT='/home/isavand/paper/evidence'
WATCHDOG=dict(wall_seconds=28800,kill_grace_seconds=10)
PHASE='D.logged-v2'
CODE=(PROTOCOL_PATH,'experiments/freeze_neural_material_recovery_logged_v2.py',
 'experiments/neural_material_recovery_logged_v2_execution.py','experiments/finalize_neural_material_journal_v2.py',
 'infrastructure/neural_material_resources_v2.py','infrastructure/neural_material_journal_runtime_v2.py',
 'tests/scientific/test_neural_material_logged_freeze_v2.py','experiments/audit_neural_material_logged_readiness_v2.py')

EXTRA=('infrastructure/append_only_scientific_log_v1.py','infrastructure/paths.py','infrastructure/runs.py',
 'infrastructure/signatures.py','infrastructure/stage_sources.py','experiments/benchmark_neural_material_reporting_v1.py',
 'tests/scientific/test_append_only_scientific_log_v1.py','docs/protocols/NEURAL_MATERIAL_REPORTING_OPTIMIZATION_V1_DRAFT.md')


def lineage():
    old=d2.source_admission();a=selected(ROOT/D1,D1_SHA);b=selected(ROOT/D2,D2_SHA)
    for name,sha in b['source_sha256'].items():require(log.file_sha(ROOT/name)==sha,'Accepted D2 source changed: '+name)
    for name,value in a['benchmark_artifact_inventories'].items():
        # Historical artifacts are selected through their admitted D.1 manifest;
        # all individual measurements below are admitted independently.
        require(type(value) is dict,'D1 artifact inventory absent')
    raw=b['preservation']['original_report'];require(raw['sha256']==RAW_SHA and log.file_sha(raw['path'])==RAW_SHA,'Original incomplete D report changed')
    for case in b['reporting_benchmarks'].values():
        selected(case['measurement']['path'],case['measurement']['sha256'])
        selected(case['reporting']['final_manifest']['path'],case['reporting']['final_manifest']['sha256'])
    selected(b['field_only']['measurement']['path'],b['field_only']['measurement']['sha256'])
    require(b['tests']['final']['successful'] and b['tests']['final']['tests']==23,'D2 test admission mismatch')
    selected(b['tests']['results']['path'],b['tests']['results']['sha256'])
    return old


def inventory(old):
    b=selected(ROOT/D2,D2_SHA)
    names=set(old['sha256'])|set(CODE)|set(EXTRA)|set(b['source_sha256'])|{D2,D1,OLD_FREEZE,legacy.AGGREGATE}
    require(FREEZE not in names and READINESS not in names,'Circular freeze inventory')
    require(len(names-set(old['sha256']))<=32,'Source provenance allowance needs review')
    return {n:log.file_sha(ROOT/n) for n in sorted(names)}


def contract(old,resources):
    cfg=copy.deepcopy(old['configuration'])
    cfg.update(reporting=dict(protocol=PROTOCOL,ordered_mutations=True,commit='journal fsync then bounded atomic fsynced anchor',
        shutdown='bounded execution seal, no whole-report serialization or chain scan',
        offline_finalization='read-only selected input hashes; fresh separate output; explicit prefix-only diagnostic',
        corruption='reject; no repair, implicit recovery or resume',scientific_completion='all frozen matrix/admission/control/work/endpoint checks and execution seal required',
        journal_cap_bytes=resources['journal_cap_bytes'],report_cap_bytes=resources['materialized_report_cap_bytes']))
    return cfg


def expected_freeze(old,resources,readiness_sha):
    cfg=contract(old,resources)
    evidence=selected(ROOT/D2,D2_SHA);raw=evidence['preservation']['original_report']
    initial=selected(raw['path'],RAW_SHA)['initial_admission'][0]
    schema=dict(checks=sorted(initial['checks']),D0_N0_exact=sorted(initial['D0_N0_exact']))
    return dict(schema_version=2,protocol=PROTOCOL,baseline=BASE,phase='D.3_preparation_only',scientific_sources=old['scientific_sources'],
        settings=old['settings'],configuration=cfg,configuration_sha256=log.digest(cfg),sha256=inventory(old),
        initial_gate_schema=schema,initial_models=old['initial_models'],initial_models_sha256=old['initial_models_sha256'],inputs=old['inputs'],tapes=old['tapes'],
        expected_total_work=old['expected_total_work'],checkpoint_schema=old['checkpoint_schema'],failure_taxonomy=old['failure_taxonomy'],
        lineage=dict(G='79252178e65914ad1acbc14790b1514f276bfea0',A='e1e859779301faced46d6476691e20b546c27ae0',
            accepted_3D1='ccb03d0ec5adc814775538b9cc7427cc57480a98',C2='ffed2b075b997e7b6faf37840fdec7e1c489d8ef',
            original_D='de5277686653033d5655f71bcaa3d70627c12945',D1='40b7b301d02835d1b324ed54db6a6c2649f0e049',D2=BASE,
            original_freeze_sha256=d2.FREEZE_SHA,original_incomplete_report_sha256=RAW_SHA,D1_evidence_sha256=D1_SHA,D2_evidence_sha256=D2_SHA),
        phase_c=old['phase_c'],readiness=dict(path=READINESS,sha256=readiness_sha),resource_budget=resources,
        output_policy=dict(approved_parent=PARENT,required_run_prefix='3d2-recovery-v2-',minimum_free_bytes=resources['minimum_free_bytes'],
            root_must_not_exist=True,owned_unaliased=True,implicit_resume=False,capacity_recheck_before_allocation=True),
        authorization=dict(execution_authorized=False,separate_researcher_instruction_required=True,phase=PHASE,external_watchdog=WATCHDOG,
            required_fields=['phase','scientific_execution_authorized','freeze_path','freeze_sha256','source_inventory_sha256','approved_output_parent','approved_output_root','approved_run_id','external_watchdog']),
        hashing=dict(algorithm='SHA-256',exclude=[FREEZE,READINESS,'docs/NEURAL_MATERIAL_RECOVERY_LOGGED_V2_FREEZE_RESULTS.md'],
            readiness='separately hash-selected DAG input; does not reference freeze hash',freeze='caller supplies exact file hash; no self/commit hash'))


def validate_readiness(old,resources):
    r=log.decode((ROOT/READINESS).read_bytes())
    require(r.get('status')=='READY FOR SEPARATE EXECUTION AUTHORIZATION' and r.get('scientific_execution_authorized') is False and
        r.get('scientific_calls')==0 and r.get('source_inventory_sha256')==log.digest(inventory(old)) and r.get('resource_budget')==resources,
        'D3 readiness gates not complete or source/resource evidence changed')
    require(all(r.get('mandatory_gates',{}).values()) and len(r.get('mandatory_gates',{}))>=8,'Missing mandatory readiness gates')
    require(r['tests']['failed']==r['tests']['errors']==r['tests']['skipped']==0 and r['tests']['passed']>0,'Focused D3 tests did not pass')
    preserved=selected(r['preservation_manifest']['path'],r['preservation_manifest']['sha256'])
    for path,record in preserved['external'].items():
        require(Path(path).resolve()==Path(path) and Path(path).is_file() and Path(path).stat().st_size==record['bytes'] and log.file_sha(path)==record['sha256'],'Historical external evidence changed: '+path)
    for record in r['artifacts'].values():
        require(log.file_sha(record['path'])==record['sha256'],'D3 bounded evidence changed: '+record['path'])
    return r


def admit_aggregate(old):
    """Reuse every original saved-array check; admit ONE explicit path relocation.

    Original admit_selected compares repo-local metadata paths literally and
    fails in a new worktree. Both original/current completion files are hashed;
    no numerical/source/evidence mismatch may be normalized.
    """
    from experiments import neural_material_validation_admission_v1 as aggregate
    stored=selected(ROOT/old['phase_c']['path'],old['phase_c']['sha256'])
    actual=aggregate.verify_aggregate()
    current=ROOT/aggregate.COMPLETION_PATH
    historic=Path(stored['completion_versioned']['path'])
    require(actual['completion_versioned']['path']==str(current) and historic.is_absolute() and historic.resolve()==historic and
        log.file_sha(current)==log.file_sha(historic)==stored['completion_versioned']['sha256'],
        'Completion-report path relocation is not byte-identical')
    normalized=copy.deepcopy(actual);normalized['completion_versioned']['path']=str(historic)
    require(normalized==stored,'C/C.1 aggregate scientific/artifact identity mismatch beyond admitted repo-local path relocation')
    return dict(status='passed',scientific_calls=0,original_aggregate_sha256=old['phase_c']['sha256'],
        permitted_metadata_relocation=dict(field='completion_versioned/path',historical=str(historic),current=str(current),sha256=log.file_sha(current)),
        all_other_fields_exact=True,checks_implementation='unchanged hash-pinned original verify_aggregate')


def validate_freeze(path,sha,verify_aggregate=True):
    path=Path(path);require(path.is_absolute() and path.resolve()==path and path==ROOT/FREEZE and sha!=d2.FREEZE_SHA,'Exact new freeze selection required; v1 freeze cannot authorize v2')
    value=selected(path,sha);old=lineage();resources=budget(old);validate_readiness(old,resources)
    require(resources['memory']['fits_accounting'],'Memory readiness accounting exceeds unchanged cap')
    require(value==expected_freeze(old,resources,log.file_sha(ROOT/READINESS)),'Freeze differs from source/scientific/resource/evidence contract')
    if verify_aggregate:
        admit_aggregate(old)
    return value


def prepare():
    old=lineage();resources=budget(old);validate_readiness(old,resources);capacity(PARENT,resources['minimum_free_bytes'])
    # Actual saved-array aggregate, never scientific forwards or old trajectories.
    admit_aggregate(old)
    value=expected_freeze(old,resources,log.file_sha(ROOT/READINESS))
    with (ROOT/FREEZE).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    sha=log.file_sha(ROOT/FREEZE);validate_freeze((ROOT/FREEZE).resolve(),sha,verify_aggregate=False)
    return dict(status='READY FOR SEPARATE EXECUTION AUTHORIZATION',freeze_path=FREEZE,freeze_sha256=sha,scientific_execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:r=prepare()
    else:validate_freeze(ROOT/FREEZE,a.freeze_sha256);r=dict(status='READY FOR SEPARATE EXECUTION AUTHORIZATION',scientific_execution_authorized=False)
    print(json.dumps(r,sort_keys=True,indent=2))


if __name__=='__main__':main()
