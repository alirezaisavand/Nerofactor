"""3C-10: conditional LM replication with one preregistered independent tape set.

The original Adam-pretrained starting vectors are fixed. This is not an
independent end-to-end optimization replication or a population estimate.
"""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import torch

from experiments import material_lm_gauge_trajectories_v2 as accepted
from infrastructure.signatures import require, SignatureError
from infrastructure.stage_sources import baseline, PINS

v1, lm, original = accepted.v1, accepted.lm, accepted.original
reference, coupling, roundoff = accepted.reference, accepted.coupling, accepted.roundoff
ROOT = Path(__file__).resolve().parents[1]
BASE = '1e7e3ad8c15b85c569e241beb037cfadb29502e4'
NAME = 'MATERIAL_INDEPENDENT_TAPE_LM_V1'
PROTOCOL = ROOT/('docs/protocols/'+NAME+'.md')
OUTPUT = ROOT/('docs/evidence/'+NAME+'.json')
CODE = ('experiments/material_independent_tape_lm_v1.py','tests/scientific/test_material_independent_tape_lm_v1.py')
TRAIN_SEEDS, OLD_SEEDS, EVAL_SEEDS = (2000,2001,2002,2003),(0,1,2,3),(1000,1001,1002,1003)
SCOPE = 'Conditional LM sensitivity to one new training realization; fixed Adam-pretrained update1000 vectors; not independent end-to-end replication; two tape realizations are not a population estimate'
INPUTS = {
    'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json':'b40f548f07a66a545936cf295fa7b6bb3c84cb104bc6a2698dbed81705460ec3',
    'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2_ANALYSIS.json':'a92476bb6e5856b49509d1e8e59e74c1a89e555bc079db34dcf087baf798063a',
    'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.freeze.json':'6f7333424789bbefbf0aceb5f7c8d4a564be188ccc803ed22386f4fbc327e20e',
    'docs/MATERIAL_LM_GAUGE_TRAJECTORIES_V2_RESULTS.md':'636bd6fafdafd65cfbce8bf304c56fe32c532abe8dea720a5d1689c0d72d2a05',
}
GROUPS = ('historical_more_accurate','historical_less_accurate')


def admit():
    old,_,continuation,identities = accepted.admit()  # Read-only chain, never old execute().
    for name,sha in INPUTS.items():
        require(accepted.digest(ROOT/name) == sha,'Accepted 3C-9c artifact changed: '+name)
    history = json.loads((ROOT/'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json').read_text())
    analysis = json.loads((ROOT/'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2_ANALYSIS.json').read_text())
    freeze = json.loads((ROOT/'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.freeze.json').read_text())
    require(history['status'] == 'complete' and history['optimization_began'] and history['initialization_gate'] == 'passed' and
        history['frozen_execution'] == freeze and len(history['outcomes']) == 15 and
        all(r['status'] == 'complete' for r in history['outcomes']), 'Complete accepted 15-arm 3C-9c evidence required')
    require(accepted.identity(accepted.POLICY) == freeze['numerical_policy_sha256'] and
        accepted.identity(accepted.CAPS) == freeze['arm_work_caps_sha256'], 'Accepted gate/work policy changed')
    require(analysis['source_report_sha256'] == INPUTS['docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json'] and
        analysis['additional_shader_calls'] == analysis['additional_derivative_calls'] == analysis['additional_solves'] == 0,
        'Accepted offline-analysis provenance changed')
    for flag in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored','fixed_tapes_unchanged'):
        require(history[flag] is True,'Prior runtime integrity failure: '+flag)
    for group in (history['accepted_identities'],freeze['sha256'],analysis['figures']):
        for name,sha in group.items():
            require(ROOT in (ROOT/name).resolve().parents and accepted.digest(ROOT/name) == sha,'Prior source/evidence changed: '+name)
            identities[name] = sha
    identities.update(INPUTS)
    return old,history,continuation,identities


def make_independent_tapes(expected=None):
    require(TRAIN_SEEDS == (2000,2001,2002,2003) and len(set(TRAIN_SEEDS)) == 4 and
        not set(TRAIN_SEEDS)&(set(OLD_SEEDS)|set(EVAL_SEEDS)), 'Only preregistered disjoint training seeds permitted')
    rng,nrng = torch.get_rng_state().clone(),np.random.get_state()
    tape,ids = original.make_tapes(list(TRAIN_SEEDS),24)
    after = np.random.get_state()
    require(torch.equal(rng,torch.get_rng_state()) and nrng[0] == after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:] == after[2:],
        'Tape generation consumed global RNG')
    tape.validate(96,1024)
    arrays = (tape.u,tape.v,tape.azimuth)
    require(len({t.data_ptr() for t in arrays}) == 3,'Independent owned tape arrays required')
    manifest = dict(training_seeds=list(TRAIN_SEEDS),original_training_seeds=list(OLD_SEEDS),evaluation_seeds=list(EVAL_SEEDS),
        points_per_seed=24,specular_samples=1024,ordering='seed-major, original make_tapes private NumPy RandomState',
        per_tape=ids,arrays={name:roundoff.array_record(t.numpy()) for name,t in zip(('u','v','azimuth'),arrays)},
        global_Torch_NumPy_RNG_unchanged=True)
    if expected is not None:
        require(manifest == expected,'Frozen independent training-tape identity mismatch; no replacement')
    return tape,manifest


def group_references(history):
    rows = sorted((r for r in history['outcomes'] if r['arm'] == 'P0'),key=lambda r:r['seed'])
    require([r['seed'] for r in rows] == list(range(101,106)), 'Exactly historical P0 101--105 required')
    out = dict(construction='Ordered float64 arithmetic mean of physical Q matrices, grouped by original historical simultaneous recovery success',
        source_report_sha256=INPUTS['docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json'],representatives={},
        metric='Frobenius absolute distance; also report distance/norm(representative)',
        admission_radius='half the absolute separation between historical group means, same radius for both balls',
        tie_rule='64*eps64*max(norm(query),norm(representatives),separation,1e-30) for distance ties and boundary roundoff; outside-both takes precedence')
    for name,success in zip(GROUPS,(True,False)):
        selected = [r for r in rows if r['success']['all_passed'] is success]
        require(len(selected) == (3 if success else 2),'Reviewed historical success groups required')
        matrices = []
        for row in selected:
            q = coupling.accepted.tensor_q(row['final'])
            require(np.array_equal(q,row['trajectory_analysis']['final_Q']),'Historical Q reconstruction mismatch')
            matrices.append(q)
        mean = np.mean(np.stack(matrices).astype(np.float64),axis=0)
        out['representatives'][name] = dict(seeds=[r['seed'] for r in selected],Q=roundoff.array_record(mean,True))
    means = [np.asarray(out['representatives'][name]['Q']['values'],float) for name in GROUPS]
    separation = float(np.linalg.norm(means[0]-means[1]))
    require(math.isfinite(separation) and separation > 0,'Separated historical representatives required')
    out.update(separation_absolute=separation,radius_absolute=separation/2)
    for i,name in enumerate(GROUPS):
        out['representatives'][name]['radius_relative'] = (separation/2)/float(np.linalg.norm(means[i]))
    return out


def proximity(q,refs):
    q = np.asarray(q,np.float64)
    require(q.shape == (2,2) and np.isfinite(q).all(),'Finite 2x2 physical Q required')
    matrices = [roundoff.read_array(refs['representatives'][name]['Q'],(2,2),'float64') for name in GROUPS]
    distances = [float(np.linalg.norm(q-r)) for r in matrices]
    radius = refs['radius_absolute']
    tolerance = float(64*np.finfo(float).eps*max(np.linalg.norm(q),*(np.linalg.norm(r) for r in matrices),refs['separation_absolute'],1e-30))
    inside = [d <= radius+tolerance for d in distances]
    tied = abs(distances[0]-distances[1]) <= tolerance
    nearest = None if tied else GROUPS[int(distances[1] < distances[0])]
    label = ('outside_historical_group_balls' if not any(inside) else
             'ambiguous_historical_proximity' if tied else nearest)
    return dict(label=label,nearest_representative=nearest,
        absolute_distances=dict(zip(GROUPS,distances)),
        relative_distances={name:distances[i]/float(np.linalg.norm(matrices[i])) for i,name in enumerate(GROUPS)},
        within_balls=dict(zip(GROUPS,inside)),tie_tolerance=tolerance,tied=tied,
        interpretation='Descriptive historical proximity, not recovery success or an optimizer decision')


def initial_gates(p0,pu,pc,historical,target,obs,construction,expected_state,manifest):
    # Reuse unchanged B--E and preserve the REAL old/new A mismatch as a diagnostic.
    result = accepted.initialization_gates(p0,pu,pc,historical,target,obs,construction)
    old_A = result['gates']['A']
    checks = dict(exact_continuation_latent=p0.row['initial_state']['unconstrained'] == expected_state['unconstrained'],
        exact_continuation_physical=p0.row['initial_state']['physical'] == expected_state['physical'],
        exact_source_target_observation_runtime=True)
    expected_original = [manifest['arrays'][name]['sha256'] for name in ('u','v','azimuth')]
    expected_coupled = [r['sha256'] for r in construction['coupled_tape']['coupled_arrays']]
    replays = []
    for arm,expected_tape in zip((p0,pu,pc),(expected_original,expected_original,expected_coupled)):
        d,row = arm.initial_capture,arm.row
        per,mean = d['per_tape'],d['mean']
        require(per.shape == (4,24,3,3) and mean.shape == (24,3,3) and per.dtype == mean.dtype == np.float32,
            'Original captured float32 component layout required')
        replay = np.stack([torch.from_numpy(per[:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
        saved = roundoff.saved_mean(row)
        passed = (replay.tobytes() == mean.tobytes() == saved.tobytes() and
            roundoff.mse32(mean[:,2],target)[0] == row['initial_fit_MSE'] and d['tape_arrays'] == expected_tape)
        replays.append(dict(arm=row['arm'],passed=passed,exact_component_mean=replay.tobytes() == mean.tobytes(),
            exact_saved_mean=mean.tobytes() == saved.tobytes(),exact_MSE=roundoff.mse32(mean[:,2],target)[0] == row['initial_fit_MSE'],
            correct_frozen_active_tape=d['tape_arrays'] == expected_tape))
    checks['all_three_internal_forward_replays'] = all(v['passed'] for v in replays)
    result['gates']['A'] = dict(passed=all(checks.values()),checks=checks,internal_replays=replays,
        historical_RGB_MSE_parity_required=False)
    result['passed'] = all(v['passed'] for v in result['gates'].values())
    result['historical_initial_comparison'] = dict(old_tape_identity_diagnostic=old_A,
        original_training_MSE=historical['initial_fit_MSE'],new_training_MSE=p0.row['initial_fit_MSE'],
        mean_RGB_difference=roundoff.compare(p0.initial_capture['mean'],roundoff.saved_mean(historical)),
        identical_finite_sample_objective=False,interpretation='Expected sample-realization difference, not source identity failure')
    return result


def cross_tape_analysis(report,history,refs):
    comparisons = []
    for row in report['outcomes']:
        old = next(r for r in history['outcomes'] if r['seed'] == row['seed'] and r['arm'] == row['arm'])
        item = dict(seed=row['seed'],arm=row['arm'],new_status=row['status'],new_censored=row['censored'],
            original_termination=old['termination'],new_termination=row['termination'],
            original_accepted=old['work']['accepted_parameter_steps'],new_accepted=row['work']['accepted_parameter_steps'],
            original_work=old['work'],new_work=row['work'],original_active_wall_seconds=old['active_wall_seconds'],
            new_active_wall_seconds=row.get('active_wall_seconds'),
            training_objectives=dict(original=old['final_fit_MSE'],new=row.get('final_fit_MSE'),
                same_finite_sample_objective=False,raw_loss_difference_is_not_optimization_improvement=True))
        if 'success' in row and 'final' in row:
            qo,qn = coupling.accepted.tensor_q(old['final']),coupling.accepted.tensor_q(row['final'])
            po,pn = proximity(qo,refs),proximity(qn,refs)
            item.update(original_recovery=old['success'],new_recovery=row['success'],original_Q=qo.tolist(),new_Q=qn.tolist(),
                original_Q_error=old['recovery_errors']['Q_relative_frobenius'],new_Q_error=row['recovery_errors']['Q_relative_frobenius'],
                physical_change=v1.state_difference(row['final'],old['final']),
                original_parameters=old['evaluation_gauge']['physical'],new_parameters=row['evaluation_gauge']['physical'],
                historical_group_original=po,historical_group_new=pn,
                group_transition=po['label']+' -> '+pn['label'],
                same_eval_tapes=True,same_per_seed_P0_gauge=True,
                original_radiance_errors=old['radiance_errors'],new_radiance_errors=row['radiance_errors'],
                radiance_error_differences={split:{component:{key:row['radiance_errors'][split][component][key]-value
                    for key,value in metrics.items()} for component,metrics in components.items()}
                    for split,components in old['radiance_errors'].items()},
                original_compensation=old['component_compensation'],new_compensation=row['component_compensation'])
        else:
            item['comparison_blocker'] = 'No valid evaluated new terminal state; do not invent recovery/group outcomes'
        comparisons.append(item)
    return dict(scope=SCOPE,original_training_seeds=list(OLD_SEEDS),independent_training_seeds=list(TRAIN_SEEDS),
        evaluation_seeds=list(EVAL_SEEDS),group_reference=refs,comparisons=comparisons,
        limitation='Two fixed tape realizations, fixed original-tape Adam states, no population significance or end-to-end replication')


def make_figures(report,history,destination):
    """Offline figures; never overlay different-tape training losses as one objective."""
    plotted = copy.deepcopy(report)
    for row in plotted['outcomes']:
        if 'trajectory_analysis' not in row:
            row['trajectory_analysis'] = v1.trajectory_analysis(row)
    outputs = accepted.make_figures(plotted,destination)
    import matplotlib.pyplot as plt
    destination = Path(destination)
    colors = dict(zip(v1.ARMS,('#222222','#c65102','#0868ac')))
    fig,axes = plt.subplots(1,5,figsize=(16,3.3),squeeze=False)
    for ax,seed in zip(axes[0],range(101,106)):
        for row in plotted['outcomes']:
            if row['seed'] != seed:
                continue
            points = row['trajectory_analysis']['points']
            if points:
                ax.semilogy([p['work']['linear_solves'] for p in points],[p['training_MSE'] for p in points],
                    color=colors[row['arm']],marker='.',markersize=3,label=row['arm'])
        ax.set_title('Seed '+str(seed))
        ax.set_xlabel('LM solve attempts')
        ax.grid(alpha=.25)
    axes[0,0].set_ylabel('New-tape training MSE')
    axes[0,-1].legend(fontsize=8)
    fig.tight_layout()
    path = destination/'loss-solves.svg'
    require(not path.exists(),'Fresh figure required')
    fig.savefig(path,metadata={'Date':None})
    plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    outputs[str(path.relative_to(ROOT))] = accepted.digest(path)
    evaluated = [r for r in report['outcomes'] if 'recovery_errors' in r]
    if evaluated:
        fig,ax = plt.subplots(figsize=(6.4,5.2))
        values = []
        for arm in v1.ARMS:
            rows = [r for r in evaluated if r['arm'] == arm]
            before,after = [],[]
            for row in rows:
                prior_row = next(r for r in history['outcomes'] if r['seed'] == row['seed'] and r['arm'] == arm)
                before.append(prior_row['recovery_errors']['Q_relative_frobenius'])
                after.append(row['recovery_errors']['Q_relative_frobenius'])
            values.extend(before+after)
            ax.scatter(before,after,color=colors[arm],label=arm,s=32,alpha=.8)
            for row,a,b in zip(rows,before,after):
                ax.annotate(str(row['seed']),(a,b),fontsize=7,xytext=(3,3),textcoords='offset points')
        lo,hi = 0.,max(values)*1.2
        ax.plot([lo,hi],[lo,hi],color='#999999',linestyle='--',label='Equal Q error')
        ax.axhline(.05,color='#777777',linestyle=':',label='Q-only recovery bound')
        ax.axvline(.05,color='#777777',linestyle=':')
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.set_xlabel('Original-tape terminal Q relative error')
        ax.set_ylabel('Independent-tape terminal Q relative error')
        ax.set_title('Conditional LM replication: fixed Adam starts')
        ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
        path = destination/'cross-tape-Q-error.svg'
        require(not path.exists(),'Fresh figure required')
        fig.savefig(path,metadata={'Date':None});plt.close(fig)
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs[str(path.relative_to(ROOT))] = accepted.digest(path)
    return outputs


def execute(old,history,continuation,report,persist,whole,freeze):
    runtime = dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
    require(runtime == history['runtime'],'Exact accepted numeric runtime required')
    report.update(runtime=runtime,arithmetic_runtime=accepted.arithmetic_runtime())
    for stage,pin in PINS.items():
        baseline(stage,pin)
    require(old['scene'] == reference.design() and old['targets']['passed'] and old['identifiability']['passed'],'Accepted scene/targets/gates required')
    targets = np.asarray(old['targets']['fine'],np.float64)
    target = torch.tensor(targets[:24,2],dtype=torch.float32)
    require(reference.sha_array(targets) == old['fine_targets_sha256'] and reference.sha_array(target.numpy()) == old['fit_target_float32_sha256'],'Original targets changed')
    fit_tape,manifest = make_independent_tapes(freeze['training_tapes'])
    eval_tape,eval_ids = original.make_tapes(list(EVAL_SEEDS),84)
    require(eval_ids == old['tapes']['evaluation'] == history['tapes']['evaluation'],'Original external evaluation tapes changed')
    refs = group_references(history)
    require(refs == freeze['historical_group_references'],'Frozen historical group construction changed')
    obs = old['scene']['observations']
    require(obs[:24] == [o for o in obs if o['split'] == 'fit'],'Original observation ordering required')
    shader = original.MaterialShaderV1('A2','atan2')
    require(shader.identity == old['shader'] == history['shader'],'Accepted shader identity changed')
    evaluation = original.ObservationBatch(old['scene'],obs,eval_tape,4)
    report.update(scene=old['scene'],shader=shader.identity,reused_gates=old['gates'],fine_targets_sha256=old['fine_targets_sha256'],
        fit_target_float32_sha256=old['fit_target_float32_sha256'],observation_sha256=accepted.identity(obs),
        tapes=dict(fit=manifest['per_tape'],evaluation=eval_ids),training_tape_manifest=manifest,
        historical_group_references=refs,outcomes=[],construction=[],initialization=[])
    arms,all_tapes,expected_states = [],[fit_tape,eval_tape],[]
    for saved in continuation['initializations']:
        z = v1.restore(saved)
        z1 = v1.gauge(z)
        p0,p1 = original.physical(z),original.physical(z1)
        mapping = coupling.frame_sign(coupling.decoded_frame(shader,p0,96),coupling.decoded_frame(shader,p1,96))
        copied,record = coupling.copy_coupled_tape(fit_tape,mapping['sign'])
        report['construction'].append(dict(seed=saved['seed'],mapping=mapping,coupled_tape=record))
        expected_states.append(dict(unconstrained=saved['latest_state']['unconstrained'],physical=saved['latest_state']['physical']))
        all_tapes.append(copied)
        for name in v1.ARMS:
            selected = copied if name == 'P1-C' else fit_tape
            fit = original.ObservationBatch(old['scene'],obs[:24],selected,4)
            row = dict(seed=saved['seed'],arm=name,status='restored',budget_limited=False,censored=False,termination=None,
                completed_outer_iterations=0,work=accepted.work(),trajectory=[],iterations=[])
            arm = accepted.Arm(row,lm.ArmClock(whole),shader,fit,evaluation,target,(z if name == 'P0' else z1).clone().requires_grad_(),sign=mapping['sign'])
            arm.save_state()
            row['initial_state'] = copy.deepcopy(row['final_state'])
            arms.append(arm)
            report['outcomes'].append(row)
    hashes = [[reference.sha_array(t.numpy()) for t in (tape.u,tape.v,tape.azimuth)] for tape in all_tapes]
    report.update(restoration_gate='passed',phase='initial_forwards',optimization_began=False)
    persist()
    for arm in arms:
        report.update(active_seed=arm.row['seed'],active_arm=arm.row['arm'])
        with arm.clock.active():
            arm.current = lm.detached(arm.forward(arm.z))
            arm.row.update(initial_fit_MSE=float(arm.current['loss']),initial_fit_predictions=arm.current['values'].tolist(),
                initial_fit_prediction_sha256=reference.sha_array(arm.current['values'].numpy()),initial_mask=arm.current['mask_identity'])
            arm.row['initial_capture_identities'] = {key:roundoff.array_record(value,key == 'per_tape')
                for key,value in arm.initial_capture.items() if isinstance(value,np.ndarray)}
            arm.log(phase='initial',training_MSE=float(arm.current['loss']),unconstrained=arm.z.detach().tolist(),physical=original.physical(arm.z).detach().tolist())
        persist()
    report['phase'] = 'initial_gates'
    for i,seed in enumerate(range(101,106)):
        previous = next(r for r in history['outcomes'] if r['seed'] == seed and r['arm'] == 'P0')
        try:
            result = initial_gates(*arms[3*i:3*i+3],previous,target.numpy(),obs[:24],report['construction'][i],expected_states[i],manifest)
        except SignatureError as exc:
            result = dict(seed=seed,passed=False,integrity_failure=str(exc))
        report['initialization'].append(result)
        persist()
    report['initialization_gate'] = 'passed' if all(r['passed'] for r in report['initialization']) else 'failed'
    persist()
    require(report['initialization_gate'] == 'passed','Mandatory independent-tape initialization failure; no optimization or retry')
    require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes],'Initial tape mutation')
    for arm in arms:
        arm.initial_capture = None
    for arm in arms:
        report.update(phase='optimization',active_seed=arm.row['seed'],active_arm=arm.row['arm'],optimization_began=True)
        persist()
        try:
            v1.run_arm(arm,targets,obs,persist)
        except SignatureError as exc:
            if str(exc) not in accepted.SAFE_NUMERICAL_FAILURES or not bool(torch.isfinite(arm.z).all()):
                raise
            arm.row.update(status='failed_numerical',termination='nonfinite_numerical_failure',failure=str(exc))
            require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes],'Tape mutation after numerical failure')
        arm.row['trajectory_analysis'] = v1.trajectory_analysis(arm.row)
        arm.row['initial_gradients'] = accepted.initial_gradient_diagnostics(arm.row)
        print('{} {}: {}, {}, accepted={}, MSE={}'.format(arm.row['seed'],arm.row['arm'],arm.row['status'],arm.row['termination'],
            arm.row['work']['accepted_parameter_steps'],arm.row.get('final_fit_MSE')),flush=True)
        whole.check()
    require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes],'Fixed tapes mutated')
    for arm in arms:
        accepted.check_caps(arm.row['work'])
    totals = {key:sum(arm.row['work'][key] for arm in arms) for key in accepted.work()}
    for key,cap in accepted.CAPS.items():
        require(totals[key] <= 15*cap,'Whole work cap exceeded: '+key)
    report.update(total_work=totals,fixed_tapes_unchanged=True,comparisons=accepted.analyze(report['outcomes']),
        cross_tape=cross_tape_analysis(report,history,refs),status='complete',phase='finished')
    persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(all(os.environ.get(k) == v for k,v in original.REQUIRED_ENV.items()),'Explicit CPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT,'Fresh explicit versioned output required')
    freeze = json.loads((ROOT/('docs/protocols/'+NAME+'.freeze.json')).read_text())
    for name,sha in freeze['sha256'].items():
        require(accepted.digest(ROOT/name) == sha,'Pre-execution freeze mismatch: '+name)
    require(accepted.identity(accepted.POLICY) == freeze['numerical_policy_sha256'] and
        accepted.identity(accepted.CAPS) == freeze['arm_work_caps_sha256'] and
        accepted.identity(freeze['training_tapes']) == freeze['training_tapes_sha256'] and
        accepted.identity(freeze['historical_group_references']) == freeze['historical_group_references_sha256'],'Policy/tape/reference freeze mismatch')
    report = dict(schema_version=1,protocol='material-independent-tape-lm-v1',milestone='3C-10',base_commit=BASE,
        scope=SCOPE,status='running',phase='admission',optimization_began=False,frozen_execution=freeze,
        protocol_sha256=accepted.digest(PROTOCOL),implementation_sha256={n:accepted.digest(ROOT/n) for n in CODE},
        numerical_policy=accepted.POLICY,settings=dict(legacy_reference_settings=original.SETTINGS,training_seeds=list(TRAIN_SEEDS),
            evaluation_seeds=list(EVAL_SEEDS),initial_lambda=.01,minimum_lambda=1e-8,maximum_lambda=1e8,
            outer_cap=80,trial_cap=4,arm_wall_seconds=120,endpoint_reserve_seconds=10,whole_wall_seconds=2400,
            peak_RSS_bytes=v1.RSS_BYTES,address_bytes=v1.ADDRESS_BYTES,arm_work_caps=accepted.CAPS,
            whole_work_caps={key:15*cap for key,cap in accepted.CAPS.items()},cross_arm_similarity=dict(Q_relative=.005,albedo_max=.002,metallic=.002)),
        GPU_execution=False,neural_training=False,new_targets=False,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=[sys.executable,*sys.argv])
    threads,rng,nrng = torch.get_num_threads(),torch.get_rng_state().clone(),np.random.get_state()
    handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    address = resource.getrlimit(resource.RLIMIT_AS)
    whole = v1.WholeClock()
    begin,cpu = time.perf_counter(),time.process_time()
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream,report)
        persist()
        try:
            for s in handlers:
                signal.signal(s,whole.alarm)
            signal.setitimer(signal.ITIMER_REAL,2400)
            cap = min(v1.ADDRESS_BYTES,address[0]) if address[0] >= 0 else v1.ADDRESS_BYTES
            if address[1] >= 0:
                cap = min(cap,address[1])
            resource.setrlimit(resource.RLIMIT_AS,(cap,address[1]))
            torch.set_num_threads(1)
            old,history,continuation,identities = admit()
            report['accepted_identities'] = identities
            execute(old,history,continuation,report,persist,whole,freeze)
            for name,sha in identities.items():
                require(accepted.digest(ROOT/name) == sha,'Accepted artifact changed during experiment: '+name)
            whole.check()
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
            raise
        finally:
            after = np.random.get_state()
            report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng,torch.get_rng_state())
            report['global_numpy_RNG_unchanged'] = nrng[0] == after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:] == after[2:]
            torch.set_num_threads(threads)
            resource.setrlimit(resource.RLIMIT_AS,address)
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in handlers.items():
                signal.signal(s,handler)
            report.update(threads_restored=torch.get_num_threads() == threads,memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS) == address,
                wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if not all(report[k] for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')):
                report.update(status='failed',failure='Runtime state integrity failed')
            persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
