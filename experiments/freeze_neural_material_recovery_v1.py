"""C.2 freeze preparation/verification only. No scientific execution CLI."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from experiments import neural_material_recovery_v1 as h
from experiments import neural_material_validation_admission_v1 as aggregate
from infrastructure.signatures import require,read_report

ROOT=h.ROOT
BASE=aggregate.BASE
AGGREGATE='docs/evidence/NEURAL_MATERIAL_FIELD_VALIDATION_AGGREGATE_V1.json'
PROTOCOL='docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.md'
FREEZE='docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json'
CODE=('experiments/neural_material_field_v1.py','experiments/neural_material_recovery_v1.py',
      'experiments/neural_material_validation_admission_v1.py','experiments/freeze_neural_material_recovery_v1.py',
      'tests/scientific/test_neural_material_execution_freeze_v1.py',PROTOCOL)
WATCHDOG=dict(wall_seconds=28800,kill_grace_seconds=10)


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def contract():
    return dict(settings=asdict(h.Settings().validate()),materials=h.previous.reference.PANEL,seeds=list(h.previous.SEEDS),
        arms=list(h.ARMS),network_weight_seeds=[10101,10103,10105],spatial_anchor=[0.,0.,0.],rho=.5,
        shader=dict(variant='A2',decoder='atan2',precision='float32',diffuse_samples=512,specular_samples=1024,is_train=True),
        optimizer=h.previous.original.SETTINGS['optimizer'],
        optimizer_effective=dict(**h.previous.original.SETTINGS['optimizer'],amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None),
        scheduler=None,clipping=None,regularization=None,LM=False,
        snapshots=list(h.SNAPSHOTS),success=h.previous.original.SETTINGS['success'],numerical_policy=h.POLICY,
        training_seeds=[3000,3001,3002,3003],evaluation_seeds=[1000,1001,1002,1003],fit_observations=24,external_observations=84,
        work_per_history={arm:h.work_limits(arm) for arm in h.ARMS},external_watchdog=WATCHDOG,
        failure_policy=dict(global_stop=['source_evidence_integrity','initialization_admission','renderer_sample_parity','baseline_reproduction'],
            isolated_N1=['numerical_optimization','active_resource_censoring','numerical_evaluation'],
            incomplete_controls_stop_before_neural=True,whole_process_resource='global_censored',unknown_evaluation='global_failed',
            completed_endpoint=['complete_pass','complete_fail'],retries=0,resume=False),
        checkpoint=dict(schema_version=1,format='numeric NPZ plus strict JSON',allow_pickle=False,hash_before_load=True,
            arrays=['parameters','immutable_buffers','Adam_step','Adam_first_moments','Adam_second_moments','CPU_Torch_RNG','NumPy_RNG'],
            failure_only_state_resumable=False,execution_provenance_required=True,large_arrays_in_git=False),
        initialization=dict(six_scalars='exact',N0_latent_gradients_states_trajectory='exact',calibration='fixed pre-sigmoid centering',
            neural_encoding='actual normalized two-channel input; no theta round-trip',global_RNG='must not be consumed'),
        run_order=['all27_initial_gates','all9_D0_with_exact1000_prefix','all9_exact_N0_controls','all9_N1'],
        execution_authorized=False,separate_Phase_D_instruction_required=True)


def inventory_names():
    _,ids=h.admit_reference()
    snapshots=subprocess.check_output(['git','ls-files','-z','scientific_sources'],cwd=ROOT).decode().split('\0')
    names=set(ids)|set(CODE)|set(aggregate.SOURCE_PATHS)|set(aggregate.c.CODE)|set(aggregate.c.v.CODE)|set(h.INPUTS)|{aggregate.c.PRIOR_PATH,aggregate.COMPLETION_PATH}
    names.update(n for n in snapshots if n)
    require(FREEZE not in names and AGGREGATE not in names,'Circular hash inventory')
    return sorted(names)


def source_inventory():return {name:h.digest(ROOT/name) for name in inventory_names()}


def initial_models():
    old_versioned,_=read_report(ROOT/aggregate.c.PRIOR_PATH,aggregate.c.PRIOR_SHA)
    record=old_versioned['original_report'];old,_=read_report(record['path'],record['sha256'])
    values={r['material']+'/'+str(r['seed'])+'/'+a:r['arms'][a]['initial_model'] for r in old['initial_admission'] for a in h.ARMS}
    expected={m+'/'+str(s)+'/'+a for m in h.previous.reference.PANEL for s in h.previous.SEEDS for a in h.ARMS}
    require(set(values)==expected,'Missing/duplicate exact initial model catalogue')
    return values


def reference_inputs():
    reference,_=h.admit_reference();fit,evaluation,tapes=h.inputs(reference)
    actual=dict(python=platform.python_version(),numpy=h.np.__version__,torch=h.torch.__version__,device='cpu',torch_threads=h.torch.get_num_threads())
    require(actual==reference['runtime'] and all(os.environ.get(k)==v for k,v in h.previous.original.REQUIRED_ENV.items()),
        'Exact accepted CPU runtime and hidden-GPU/single-thread environment required')
    require(h.torch.get_default_dtype()==h.torch.float32 and str(h.torch.get_default_device())=='cpu','Original CPU float32 defaults required')
    arithmetic=h.previous.accepted.arithmetic_runtime()
    require(all(row['snapshots'][str(step)]['optimizer_effective']==contract()['optimizer_effective']
        for row in reference['Adam_runs'] for step in (250,1000)),'Historical effective Adam flags differ from the frozen contract')
    targets={}
    for material in h.previous.reference.PANEL:
        stored=next(r for r in reference['materials'] if r['material']==material);fine,target=h.previous.material_target(stored,material)
        targets[material]=dict(truth=h.previous.reference.PANEL[material],fine_sha256=h.previous.oldref.sha_array(fine),
            fit_float32_sha256=h.previous.oldref.sha_array(target.numpy()),source_entry_sha256=identity(stored))
    h.verify_tapes(fit,evaluation,tapes)
    return dict(runtime=actual,arithmetic_runtime=arithmetic,scene=reference['scene'],shader=reference['shader'],targets=targets,tapes=tapes,
        historical_prefixes={r['material']+'/'+str(r['seed']):identity(r) for r in reference['Adam_runs']},
        accepted_3D1_SHA256=h.INPUTS['docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'])


def storage_check(parent,minimum):
    parent=Path(parent)
    require(parent.is_absolute() and parent.resolve()==parent and parent.is_dir(),'Explicit unaliased existing output parent required')
    require(parent.stat().st_uid==os.geteuid() and os.access(parent,os.W_OK|os.X_OK),'Approved output parent is not user-owned/writable')
    for forbidden in (ROOT,ROOT.parent/'data',ROOT.parent/'final_results'):
        require(parent!=forbidden and forbidden not in parent.parents and parent not in forbidden.parents,'Output parent overlaps preserved code/input root')
    require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (parent,*parent.parents)),'Output parent is inside a checkout')
    free=shutil.disk_usage(parent).free;require(free>=minimum,'Insufficient free capacity for minimum frozen checkpoint storage')
    return dict(parent=str(parent),owner_uid=parent.stat().st_uid,available_bytes=free,minimum_checkpoint_bytes=minimum,
        sufficient=True,allocated_Phase_D_outputs=False,scope='minimum numeric checkpoints; logs/failures require additional space')


def check_output_selection(freeze,root,run_id):
    h.validate_run_id(run_id);root=Path(root);policy=freeze['output_policy']
    parent=Path(policy['approved_parent']);require(root.is_absolute() and root==parent/run_id and root.resolve()==root and
        run_id.startswith(policy['required_run_prefix']),'Output root/run ID is outside the explicitly approved experiment selection')
    storage_check(parent,policy['minimum_free_bytes'])
    if root.exists():require(root.is_dir() and root.stat().st_uid==os.geteuid() and not any(root.iterdir()),'Run collision; no overwrite or implicit resume')
    return root


def make_freeze(aggregate_report,storage_parent):
    require(aggregate_report['protocol']==aggregate.PROTOCOL and aggregate_report['status']=='passed','Verified aggregate required')
    minimum=aggregate_report['runtime_feasibility']['minimum_27_history_checkpoint_bytes'];storage_check(storage_parent,minimum)
    inputs=reference_inputs();models=initial_models();cfg=contract()
    return dict(schema_version=1,protocol=h.PROTOCOL,baseline=BASE,phase='C.2_preparation_only',
        settings=asdict(h.Settings()),configuration=cfg,configuration_sha256=identity(cfg),sha256=source_inventory(),
        scientific_sources=h.PINS,lineage=dict(Phase_A='a1f15d1030cb954deddd18fb408de39a452fd441',
            Phase_B='a11efd615d18ac69fa1c26a3692af511f1c78302',Phase_C='42c8f4b08a0db11a47413f1007fb504f6e995378',Phase_C1=BASE),
        phase_c=dict(protocol=aggregate.PROTOCOL,path=AGGREGATE,sha256=h.digest(ROOT/AGGREGATE),status='passed',
            original_failed_status='failed',full_shader_admission_passed=True,runtime_feasibility_passed=True),
        initial_models=models,initial_models_sha256=identity(models),inputs=inputs,tapes=inputs['tapes'],
        output_policy=dict(approved_parent=str(Path(storage_parent)),required_run_prefix='3d2-recovery-',minimum_free_bytes=minimum,
            fresh_owned_unaliased=True,no_overwrite=True,no_implicit_resume=True,recheck_space_before_allocation=True),
        authorization=dict(execution_authorized=False,separate_Phase_D_instruction_required=True,external_watchdog=WATCHDOG),
        failure_taxonomy=cfg['failure_policy'],checkpoint_schema=cfg['checkpoint'],
        hashing=dict(algorithm='SHA-256',inventory='sorted explicit relative paths; bytes of each source/test/protocol',
            exclude=[FREEZE,AGGREGATE,'docs/NEURAL_MATERIAL_FIELD_RECOVERY_FREEZE_V1_RESULTS.md'],
            aggregate_selection='explicit separate hash; recomputed from both immutable reports and referenced artifacts',
            freeze_selection='caller supplies exact file SHA-256; no self hash'),
        expected_total_work={k:9*sum(h.work_limits(a)[k] for a in h.ARMS) for k in h.work_limits('D0')})


def validate_freeze(path,sha):
    path=Path(path);require(path.is_absolute() and path.resolve()==path,'Unaliased absolute freeze selection required')
    frozen,_=read_report(path,sha)
    require(frozen.get('schema_version')==1 and frozen.get('protocol')==h.PROTOCOL and frozen.get('baseline')==BASE and
        frozen.get('configuration')==contract() and frozen.get('configuration_sha256')==identity(contract()) and
        frozen.get('settings')==asdict(h.Settings()),'Incomplete/changed frozen scientific configuration')
    require(frozen.get('authorization')==dict(execution_authorized=False,separate_Phase_D_instruction_required=True,external_watchdog=WATCHDOG),
        'Freeze cannot grant execution permission')
    require(frozen.get('sha256')==source_inventory(),'Source/test/protocol inventory changed or incomplete')
    selection=frozen['phase_c'];require(selection.get('protocol')==aggregate.PROTOCOL and selection.get('path')==AGGREGATE,'Explicit C/C.1 aggregate selection required')
    admitted=aggregate.admit_selected(ROOT/selection['path'],selection['sha256'])
    require(frozen==make_freeze(admitted,frozen['output_policy']['approved_parent']),'Freeze does not exactly match implemented/admitted matrix and state identities')
    return frozen


def prepare(storage_parent):
    # Never allocate a Phase D output, construct a model/shader, create Adam,
    # render, differentiate, restore checkpoints into models or perform updates.
    admitted=aggregate.verify_aggregate()
    with (ROOT/AGGREGATE).open('x') as stream:json.dump(admitted,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    frozen=make_freeze(admitted,storage_parent)
    with (ROOT/FREEZE).open('x') as stream:json.dump(frozen,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    sha=h.digest(ROOT/FREEZE);validate_freeze((ROOT/FREEZE).resolve(),sha)
    return dict(status='READY',freeze_path=FREEZE,freeze_sha256=sha,aggregate_sha256=h.digest(ROOT/AGGREGATE),
        storage=storage_check(storage_parent,frozen['output_policy']['minimum_free_bytes']),scientific_execution_authorized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--verify',action='store_true')
    parser.add_argument('--storage-parent',type=Path);parser.add_argument('--freeze-sha256');args=parser.parse_args()
    if args.prepare:
        require(args.storage_parent is not None,'Explicit researcher-approved storage parent required');result=prepare(args.storage_parent)
    else:
        validate_freeze((ROOT/FREEZE).resolve(),args.freeze_sha256);result=dict(status='READY',execution_authorized=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
