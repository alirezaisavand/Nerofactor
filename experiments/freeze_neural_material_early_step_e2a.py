"""E2a metadata-only freeze. No Torch, model, renderer or optimizer imports."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments import analyze_neural_material_failures_e1 as e1
from experiments.neural_material_early_step_e2a import BASE, CASES, RATES, LIMITS, caps
from infrastructure.signatures import require, read_report

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='neural-material-early-step-lr-e2a-v1'
FREEZE='docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A.freeze.json'
VALIDATION='docs/evidence/NEURAL_MATERIAL_E2A_VALIDATION.json'
E1='docs/evidence/NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.json'
E1_SHA='5cb4011e170ab4312bb3f4f41c09b9440e3d5983e0e90c2deaa10c572b5e0c55'
CODE=('experiments/neural_material_early_step_e2a.py','experiments/freeze_neural_material_early_step_e2a.py',
      'infrastructure/neural_material_observer_e2a.py','tests/scientific/test_neural_material_early_step_e2a.py',
      'docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A.md')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inherited():
    report,_=read_report(ROOT/E1,E1_SHA)
    require(sha(ROOT/'experiments/analyze_neural_material_failures_e1.py')==report['analysis_source_sha256'],
            'E1 source identity differs')
    evidence,rows,reference=e1.admit()  # JSON catalogs and hashes only; never loads tensors/models.
    f=evidence['freeze'];initial={};parity={}
    for m,s in CASES:
        key=m+'/'+str(s);initial[key]=f['initial_models'][key+'/N1']
        row=rows[(m,s,'N1')][0];history=e1.history_records(row)
        parity[key]={str(label):next(r for r in history if r['update']==label) for label in (1,10)}
        require(all(parity[key][str(label)]['loss_at_update']==label-1 for label in (1,10)),
                'Historical label/state association differs')
    ids=dict(f['sha256'])
    for name,value in report['source_sha256'].items():
        require(sha(ROOT/name)==value,'E1 inherited source differs: '+name);ids[name]=value
    ids.update(evidence['complete']['authorization']['continuation_sources'])
    ids.update({E1:E1_SHA,'experiments/analyze_neural_material_failures_e1.py':report['analysis_source_sha256']})
    for name in ('docs/NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.md','docs/protocols/NEURAL_MATERIAL_ABLATION_E2_DRAFT.md'):
        # Identity is pinned again by the final freeze and clean reviewed commit.
        ids[name]=sha(ROOT/name)
    return evidence,initial,parity,ids


def expected():
    evidence,initial,parity,ids=inherited();f=evidence['freeze']
    storage=caps(next(iter(initial.values())))
    require(all(caps(m)==storage for m in initial.values()),'Model capture sizes differ')
    new={n:sha(ROOT/n) for n in CODE}
    previous_peak=evidence['complete']['process']['peak_RSS_bytes']
    require(previous_peak==1229254656,'Accepted peak-memory evidence differs')
    # Conservative residency allowances in addition to twice the observed native RSS.
    memory=dict(prior_observed_peak_RSS_bytes=previous_peak,prior_peak_is_new_observer_benchmark=False,
        doubled_prior_peak_bytes=2*previous_peak,paired_model_and_checkpoint_temporary_bytes=16*storage['parameters'],
        retained_intermediates_bytes=storage['retained_signal_and_gradient_bytes'],
        bounded_JSON_and_serialization_bytes=128*1024**2,
        estimated_planning_peak_bytes=2*previous_peak+16*storage['parameters']+storage['retained_signal_and_gradient_bytes']+128*1024**2,
        caveat='Planning bound, not measured native hook/serialization peak; runtime caps remain mandatory')
    require(memory['estimated_planning_peak_bytes']<4*1024**3,'Memory planning exceeds RSS cap')
    return dict(schema_version=1,protocol=PROTOCOL,baseline_commit=BASE,
        execution_authorized=False,required_next_phase='separately reviewed E.2b',
        branch='feat/neural-material-early-step-e2a',output_parent='/home/isavand/paper/evidence',
        cases=[dict(material=m,seed=s,rates=list(RATES)) for m,s in CASES],updates_per_arm=10,
        work_per_arm=LIMITS,total_work={k:10*v for k,v in LIMITS.items()},
        native_work_per_arm=dict(field_attempts=12,field_completed=12,calibration_forwards=1,shader_attempts=12,shader_completed=12,
            reverse_attempts=11,reverse_completed=11,optimizer_attempts=10,optimizer_returns=10,fit_forwards=11,evaluation_forwards=1,
            initial_forwards=0,prefix_forwards=0,terminal_forwards=1,attempted_points=1392,completed_points=1392,
            diffuse_directions=712704,specular_directions=1425408),
        initial_models=initial,historical_parity=parity,optimizer=f['configuration']['optimizer_effective'],
        inputs=f['inputs'],tapes=f['tapes'],runtime=f['inputs']['runtime'],arithmetic_runtime=f['inputs']['arithmetic_runtime'],
        inherited_configuration_context=f['configuration'],only_scientific_intervention='Adam learning rate .01 -> .001',
        inherited_sha256=ids,source_sha256=new,evidence_pins={k:dict(path=p,sha256=s) for k,(p,s) in e1.PINS.items()},
        E1_report=dict(path=E1,sha256=E1_SHA),storage=storage,memory_planning=memory,
        time_planning=dict(prior_continuation_wall_seconds=evidence['complete']['process']['wall_seconds'],
            prior_continuation_optimizer_returns=evidence['complete']['new_physical_work']['optimizer_returns'],
            ten_return_linear_compute_projection_seconds=10*evidence['complete']['process']['wall_seconds']/evidence['complete']['new_physical_work']['optimizer_returns'],
            measured_new_instrumentation_overhead=False,interpretation='Compute-only scale context, not a native arm benchmark; 120s/arm remains a censored bound'),
        resources=dict(CPU_threads=1,hidden_GPUs=True,RSS_bytes=4*1024**3,address_bytes=8*1024**3,
            active_seconds_per_arm=120,whole_seconds=1800,pre_update_reserve_seconds=10,retries=0,
            external_watchdog=dict(wall_seconds=1800,kill_grace_seconds=10)),
        observability=dict(actual_graph=True,extra_scientific_queries=0,ReLU_modules=22,
            sigmoid='actual calibrated pre-Sigmoid input and output, with retained gradients',
            orientation='actual two-channel shader encoding; diagnostic theta/radius require no gradient',
            parameter_changes='actual float32 pre/post values, uint32 XOR archives, float64 summaries',
            first_boundary_state='every recorded state; no interpolation; never infer a 5000-step outcome'),
        failures=dict(global_stop=['identity','parity','missing/nonfinite observation','work/resource/storage cap','interruption'],
            arm_states=['not_started','running','complete','failed','censored'],retry=False,resume=False,
            incomplete='preserve attempted/returned counters, record chain and one failure-state capture if possible'),
        authorization_contract='Hash-selected separate positive E2b review binds freeze, exact clean branch/commit, fresh output root and watchdog; no CLI execution route',
        validation_path=VALIDATION,freeze_hash='external SHA-256 selection; no circular self/commit identity')


def validate_freeze(selected_sha):
    value,_=read_report(ROOT/FREEZE,selected_sha)
    validation,_=read_report(ROOT/VALIDATION,value['validation_sha256'])
    require(validation['scientific_model_calls']==validation['scientific_renderer_calls']==validation['scientific_optimizer_updates']==0,
            'Preparation evidence includes scientific execution')
    require(validation['tests']['failed']==validation['tests']['errors']==validation['tests']['skipped']==0 and validation['tests']['passed']>0,
            'Focused engineering validation incomplete')
    target=expected();target['validation_sha256']=value['validation_sha256']
    require(value==target and validation['source_sha256']==value['source_sha256'],'Freeze/source/validation contract changed')
    return value


def prepare():
    value=expected();validation,_=read_report(ROOT/VALIDATION,sha(ROOT/VALIDATION))
    require(validation['source_sha256']==value['source_sha256'] and
        validation['scientific_model_calls']==validation['scientific_renderer_calls']==validation['scientific_optimizer_updates']==0 and
        validation['tests']['failed']==validation['tests']['errors']==validation['tests']['skipped']==0 and validation['tests']['passed']>0,
        'Validation source inventory or preparation gates differ')
    value['validation_sha256']=sha(ROOT/VALIDATION)
    with (ROOT/FREEZE).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return dict(path=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--verify-sha256');a=p.parse_args()
    print(json.dumps(prepare() if a.prepare else dict(verified=bool(validate_freeze(a.verify_sha256)),execution_authorized=False),sort_keys=True))


if __name__=='__main__':main()
