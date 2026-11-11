"""One controlled CPU Adam versus damped Gauss-Newton comparison; no overrides."""
import argparse
from contextlib import contextmanager
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np
import torch

from experiments import material_recovery_continuation_v1 as accepted
from experiments import run_known_material_recovery_v1 as original
from experiments import known_material_reference_v1 as reference
from infrastructure.signatures import require, SignatureError, write_report
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a64e7e77503fb3ed1a9fb26969d2bf2ad330fcea'
PROTOCOL = ROOT / 'docs/protocols/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.md'
INPUT_SHA = '224cb9a0cb27bbaacc61b8d575efbcd2c37854b716fc51bc78b439020239f24f'
ARM_SECONDS, RESERVE_SECONDS, WHOLE_SECONDS = 120., 10., 1800.
ADAM_UPDATES, LM_OUTERS, LM_TRIALS = 750, 80, 4
LAMBDA_INITIAL, LAMBDA_MIN, LAMBDA_MAX = 1e-2, 1e-8, 1e8
MSE_ATOL, MSE_RTOL = 1e-12, 5e-7


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit():
    old, diagnostic, identities = accepted.admit(
        ROOT/'docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json',
        ROOT/'docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json')
    path = ROOT/'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json'
    require(digest(path) == INPUT_SHA, 'Continuation evidence SHA-256 mismatch')
    evidence = json.loads(path.read_text())
    require(evidence['status'] == 'complete' and evidence['restoration_gate'] == 'passed' and
            evidence['budget']['optimizer_returns'] == 3750 and
            [r['seed'] for r in evidence['initializations']] == list(range(101, 106)) and
            all(r['status'] == 'complete' and r['completed_updates'] == 1000 and r['additional_updates'] == 750
                for r in evidence['initializations']), 'Only five completed update1000 states admitted')
    require(evidence['input_sha256'] == accepted.admission.EVIDENCE_SHA and
            evidence['diagnostic_sha256'] == accepted.DIAGNOSTIC_SHA and
            evidence['tapes'] == {k: old['tapes'][k] for k in ('fit', 'evaluation')}, 'Continuation input/tape identity changed')
    require(evidence['fixed_tapes_unchanged'] and evidence['global_torch_CPU_RNG_unchanged'] and
            evidence['global_numpy_RNG_unchanged'], 'Continuation randomness not preserved')
    settings = copy.deepcopy(original.SETTINGS); settings['max_updates_per_start'] = 1000
    require(evidence['settings'] == settings and evidence['scene'] == old['scene'], 'Continuation settings changed')
    for name, sha in evidence['implementation_sha256'].items():
        require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name) == sha, 'Continuation source changed: '+name)
        identities[name] = sha
    name = 'docs/protocols/MATERIAL_RECOVERY_CONTINUATION_V1.md'
    require(digest(ROOT/name) == evidence['protocol_sha256'], 'Continuation protocol changed')
    identities[name] = evidence['protocol_sha256']
    identities.update({'docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json': accepted.admission.EVIDENCE_SHA,
                       'docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json': accepted.DIAGNOSTIC_SHA,
                       'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json': INPUT_SHA})
    return old, evidence, identities


def restore(row):
    """Restore typed, hash-verified update1000 state without invoking old runners."""
    require(torch.get_default_dtype() == torch.float32 and row['completed_updates'] == 1000 and
            row['status'] == 'complete', 'Expected complete CPU float32 update1000 state')
    saved = row['latest_state']
    def tensor(record, shape, name):
        value = accepted.float32_exact(record['values'], shape, name)
        require(accepted.describe(value) == record, 'State tensor dtype/device/shape/hash mismatch: '+name)
        return value
    z = tensor(saved['unconstrained'], (7,), 'coordinates').requires_grad_()
    p = tensor(saved['physical'], (7,), 'physical')
    require(torch.equal(original.physical(z).detach(), p) and p.tolist() == row['final'], 'Physical/latent identity mismatch')
    options = dict(lr=.01, betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False,
                   maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    require(saved['optimizer_effective'] == options and set(saved['adam']) == {'step','exp_avg','exp_avg_sq'},
            'Original Adam options/state required')
    tensors = {k: tensor(v, () if k == 'step' else (7,), k) for k,v in saved['adam'].items()}
    require(float(tensors['step']) == 1000 and bool((tensors['exp_avg_sq'] >= 0).all()), 'Invalid Adam counter/moments')
    optimizer = torch.optim.Adam([z], **{**options, 'betas': tuple(options['betas'])})
    state = optimizer.state_dict(); state['state'] = {state['param_groups'][0]['params'][0]: tensors}
    optimizer.load_state_dict(state)
    require(accepted.snapshot(z, optimizer) == saved, 'Exact update1000 Adam restoration failed')
    return z, optimizer


class ArmLimit(RuntimeError): pass
class WholeLimit(RuntimeError): pass


class WholeClock:
    def __init__(self): self.start = time.perf_counter(); self.active = None
    def remaining(self): return WHOLE_SECONDS - (time.perf_counter() - self.start)
    def check(self):
        if self.remaining() <= 0: raise WholeLimit('Single1800-second process budget reached')
    def alarm(self, signum, frame):
        if signum == signal.SIGTERM: raise WholeLimit('External process termination; no retry')
        self.check()
        if self.active is not None: raise ArmLimit('Arm120-second active wall budget reached')
        raise WholeLimit('Single1800-second process alarm')


class ArmClock:
    """Accumulate active segments, excluding the other arm's execution/wait."""
    def __init__(self, whole):
        self.whole = whole; self.wall = self.cpu = 0.; self.started = None
    def elapsed(self): return self.wall + (time.perf_counter()-self.started[0] if self.started else 0.)
    def check(self, reserve=0.):
        self.whole.check()
        if self.elapsed() >= ARM_SECONDS-reserve: raise ArmLimit('Arm active wall allowance reached; no extension')
    @contextmanager
    def active(self):
        require(self.started is None and self.whole.active is None, 'Nested arm clocks forbidden')
        self.check(); self.started = (time.perf_counter(), time.process_time()); self.whole.active = self
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, min(self.whole.remaining(), ARM_SECONDS-self.wall)))
        try: yield
        finally:
            self.wall += time.perf_counter()-self.started[0]; self.cpu += time.process_time()-self.started[1]
            self.started = None; self.whole.active = None
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, self.whole.remaining()))


def work():
    return dict(forward_attempts=0, full_batch_forward_evaluations=0, fit_forward_evaluations=0,
        evaluation_forward_evaluations=0, reverse_mode_gradient_evaluations=0, reverse_completed=0,
        jacobian_vector_evaluations=0, JVP_completed=0, jacobian_constructions=0, jacobians_completed=0,
        linear_solves=0, linear_solves_completed=0, accepted_parameter_steps=0, rejected_parameter_steps=0,
        fit_points=0, evaluation_points=0, diffuse_directions=0, specular_directions=0,
        attempted_points=0)


def directional_jvp(function, z, vector, counts):
    """One explicit double-reverse JVP: one primal, two reverse sweeps; no fallback."""
    counts['jacobian_vector_evaluations'] += 1
    x = z.detach().requires_grad_(); y = function(x)
    w = torch.zeros_like(y, requires_grad=True)
    counts['reverse_mode_gradient_evaluations'] += 1
    vjp, = torch.autograd.grad(y, x, grad_outputs=w, create_graph=True)
    counts['reverse_completed'] += 1; counts['reverse_mode_gradient_evaluations'] += 1
    jvp, = torch.autograd.grad(vjp, w, grad_outputs=vector)
    counts['reverse_completed'] += 1; counts['JVP_completed'] += 1
    require(bool(torch.isfinite(jvp).all()), 'Nonfinite directional derivative; no alternative derivative method')
    return y.detach(), jvp.detach()


def residual_jacobian(function, z, expected, counts):
    require(z.shape == (7,) and expected.shape == (72,), 'Declared72x7 residual Jacobian required')
    counts['jacobian_constructions'] += 1; columns = []
    for i in range(7):
        v = torch.zeros_like(z); v[i] = 1
        y, jv = directional_jvp(function, z, v, counts)
        require(torch.equal(y, expected), 'JVP primal differs from accepted current residual')
        columns.append(jv)
    result = torch.stack(columns, dim=1).double()
    require(result.shape == (72,7) and bool(torch.isfinite(result).all()), 'Invalid residual Jacobian')
    counts['jacobians_completed'] += 1
    return result


def bounded_lambda(value): return min(LAMBDA_MAX, max(LAMBDA_MIN, value))


def acceptance(old, new, predicted):
    if not all(math.isfinite(v) for v in (old, new, predicted)) or predicted <= 0:
        return False, None
    ratio = (old-new)/predicted
    if not math.isfinite(ratio): return False, None
    return bool(new < old and ratio > .1), ratio


def accepted_lambda(value, ratio):
    return bounded_lambda(value/3 if ratio > .75 else value*2 if ratio < .25 else value)


def proposal(z, jacobian, residual, damping, counts=None):
    h = jacobian.T@jacobian; g = jacobian.T@residual.double()
    d = h.diag()+1e-8
    if counts is not None: counts['linear_solves'] += 1
    delta = torch.linalg.solve(h+damping*torch.diag(d), -g)
    if counts is not None: counts['linear_solves_completed'] += 1
    candidate = (z.detach().double()+delta).float()
    represented_delta = candidate.double()-z.detach().double()
    predicted = float(-2*g.dot(represented_delta)-represented_delta.dot(h@represented_delta))
    return candidate, predicted, delta, represented_delta


def finite_list(value):
    """Preserve attempted invalid coordinates without emitting non-JSON NaN/Inf."""
    return [float(x) if math.isfinite(float(x)) else None for x in value.detach().reshape(-1)]


def verify_start(saved, pair, states):
    require(torch.equal(pair[0].z.detach(),pair[1].z.detach()) and
            torch.equal(original.physical(pair[0].z),original.physical(pair[1].z)) and
            torch.equal(states[0]['values'],states[1]['values']) and
            torch.equal(states[0]['residual'],states[1]['residual']) and
            float(states[0]['loss'])==float(states[1]['loss'])==saved['final_fit_MSE'],
            'Starting state/predictions/objective differ from common admitted update1000 state')


class ShaderObserver:
    def __init__(self, shader): self.shader = shader; self.mask = None
    def shade(self, *args, **kwargs):
        result = self.shader.shade(*args, **kwargs)
        self.mask = (result['specular_directions'][...,2] > 0).detach().numpy().copy()
        return result


class Arm:
    def __init__(self, row, clock, shader, fit, evaluation, target, z, optimizer=None):
        self.row, self.clock = row, clock
        self.shader = ShaderObserver(shader); self.fit, self.evaluation = fit, evaluation
        self.target, self.z, self.optimizer = target, z, optimizer
        self.stage = 'initial'; self.current = None
    def check(self): self.clock.check(RESERVE_SECONDS if self.stage == 'optimization' else 0.)
    def forward(self, z, evaluation=False, physical=False):
        self.check(); batch = self.evaluation if evaluation else self.fit; counts = self.row['work']
        # Includes initial checks, each JVP primal, each proposal and endpoint measurements.
        require(counts['forward_attempts'] < (753 if self.row['arm']=='Adam' else 883), 'Frozen arm forward cap reached')
        counts['forward_attempts'] += 1; counts['attempted_points'] += batch.n*batch.seeds
        values = batch(self.shader, z if physical else original.physical(z))
        counts['full_batch_forward_evaluations'] += 1
        counts['evaluation_forward_evaluations' if evaluation else 'fit_forward_evaluations'] += 1
        counts['evaluation_points' if evaluation else 'fit_points'] += batch.n*batch.seeds
        counts['diffuse_directions'] += batch.n*batch.seeds*512; counts['specular_directions'] += batch.n*batch.seeds*1024
        require(bool(torch.isfinite(values).all()), 'Nonfinite original shader predictions')
        if evaluation: return values
        error = values[:,2]-self.target; loss = torch.mean(error**2)
        residual = error.reshape(-1)/math.sqrt(72)
        require(residual.shape == (72,) and math.isclose(float(loss.detach()), float(residual.detach().double().dot(residual.detach().double())),
            rel_tol=MSE_RTOL, abs_tol=MSE_ATOL), 'Residual norm and original MSE disagree')
        return dict(values=values, residual=residual, loss=loss,
                    mask=self.shader.mask, mask_identity=mask_identity(self.shader.mask))
    def residual(self, z): return self.forward(z)['residual']
    def gradient(self, state, z):
        c = self.row['work']; c['reverse_mode_gradient_evaluations'] += 1
        state['loss'].backward(); c['reverse_completed'] += 1
        require(z.grad is not None and bool(torch.isfinite(z.grad).all()), 'Nonfinite/missing original latent gradient')
        return z.grad.detach().clone()
    def save_state(self):
        self.row['final_state'] = (accepted.snapshot(self.z, self.optimizer) if self.optimizer is not None else
            dict(unconstrained=accepted.describe(self.z), physical=accepted.describe(original.physical(self.z))))
    def log(self, **extra):
        value = dict(active_wall_seconds=self.clock.elapsed(), work=copy.deepcopy(self.row['work']), **extra)
        self.row['trajectory'].append(value)


def mask_identity(mask):
    require(mask is not None and mask.shape == (96,1024), 'Original fit mask shape required')
    return dict(sha256_packbits=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                valid_count=int(mask.sum()), total=int(mask.size))


def detached(state):
    return {k: v.detach() if isinstance(v,torch.Tensor) else v for k,v in state.items()}


def run_adam(arm, persist):
    row = arm.row
    for i in range(1,ADAM_UPDATES+1):
        arm.check(); arm.optimizer.zero_grad()
        state = arm.forward(arm.z); gradient = arm.gradient(state,arm.z)
        arm.check(); arm.optimizer.step()
        row['work']['accepted_parameter_steps'] += 1; row['additional_updates'] = i
        require(float(arm.optimizer.state[arm.z]['step']) == 1000+i and bool(torch.isfinite(arm.z).all()) and
                all(bool(torch.isfinite(v).all()) for v in arm.optimizer.state[arm.z].values()), 'Adam state nonfinite/counter mismatch')
        if i==1 or i%10==0:
            arm.log(additional_updates=i, objective_at_additional_updates=i-1, training_MSE=float(state['loss'].detach()),
                gradient_norm=float(torch.norm(gradient)), physical_after_update=original.physical(arm.z).detach().tolist(),
                unconstrained_after_update=arm.z.detach().tolist())
            arm.save_state(); persist()
    row['termination'] = 'update_ceiling'


def run_lm(arm, persist):
    row = arm.row; damping = LAMBDA_INITIAL; current = arm.current
    row['damping'] = damping
    for outer in range(1,LM_OUTERS+1):
        arm.check(); record = dict(outer_iteration=outer, z=arm.z.detach().tolist(),
            physical=original.physical(arm.z).detach().tolist(), training_MSE=float(current['loss']),
            lambda_before=damping, mask=current['mask_identity'], trials=[], status='linearizing',
            active_wall_seconds=arm.clock.elapsed(),work_before=copy.deepcopy(row['work']))
        row['iterations'].append(record)
        jac = residual_jacobian(arm.residual,arm.z,current['residual'],row['work'])
        # Report sensitivity at this precise linearization point, never silently at its successor.
        _, singular, vh = torch.linalg.svd(jac,full_matrices=False)
        record.update(jacobian_sha256=reference.sha_array(jac.numpy()), singular_values=singular.tolist(),
            right_singular_vectors=vh.tolist(), gradient=(2*jac.T@current['residual'].double()).tolist(),
            normal_matrix=(jac.T@jac).tolist(), status='trials')
        for trial in range(1,LM_TRIALS+1):
            arm.check(); row['damping']=damping
            attempt = dict(trial=trial,damping=damping,status='attempting',active_wall_seconds=arm.clock.elapsed())
            record['trials'].append(attempt)
            try:
                candidate,predicted,delta,represented = proposal(arm.z,jac,current['residual'],damping,row['work'])
                attempt.update(proposed_unconstrained=finite_list(candidate), delta_float64=finite_list(delta),
                    represented_delta=finite_list(represented), predicted_reduction=predicted if math.isfinite(predicted) else None,
                    nonfinite_coordinate_indices=torch.nonzero(~torch.isfinite(candidate)).reshape(-1).tolist())
                require(bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(delta).all()), 'Nonfinite proposal coordinates')
                require(math.isfinite(predicted) and predicted>0,'Zero/negative/nonfinite predicted reduction')
                value = detached(arm.forward(candidate))
                old,new = float(current['loss']),float(value['loss'])
                passed,ratio = acceptance(old,new,predicted)
                attempt.update(training_MSE=new,actual_reduction=old-new,ratio=ratio,
                    proposed_physical=original.physical(candidate).tolist(), mask=value['mask_identity'],
                    mask_flips=int(np.count_nonzero(current['mask']!=value['mask'])),
                    angle_branch_crossings=abs(math.floor((float(candidate[2])+math.pi/2)/math.pi)-
                                              math.floor((float(arm.z[2])+math.pi/2)/math.pi)))
                arm.check()  # Never accept a proposal after the reserved optimization window.
                if passed:
                    arm.z = candidate.detach().requires_grad_(); current = value; damping = accepted_lambda(damping,ratio)
                    row['work']['accepted_parameter_steps'] += 1; attempt['status']='accepted'
                    attempt.update(active_wall_seconds_after=arm.clock.elapsed(),work_after=copy.deepcopy(row['work']))
                    break
                attempt.update(status='rejected',reason='Objective/ratio acceptance test failed')
            except (SignatureError,torch.linalg.LinAlgError) as exc:
                attempt.update(status='rejected',reason=str(exc),failure_type=type(exc).__name__)
            row['work']['rejected_parameter_steps'] += 1
            attempt.update(active_wall_seconds_after=arm.clock.elapsed(),work_after=copy.deepcopy(row['work']))
            if damping==LAMBDA_MAX:
                record['status']='damping_saturated'; row['termination']='damping_saturated'; break
            damping=bounded_lambda(damping*10)
            row['damping']=damping
        record.update(lambda_after=damping,status=record['status'] if record['status']=='damping_saturated' else 'complete',
                      training_MSE_after=float(current['loss']), z_after=arm.z.detach().tolist())
        row['completed_outer_iterations']=outer; row['damping']=damping; arm.current=current
        arm.log(outer_iteration=outer,training_MSE=float(current['loss']),damping=damping,
                unconstrained=arm.z.detach().tolist(),physical=original.physical(arm.z).detach().tolist())
        arm.save_state(); persist()
        if row.get('termination')=='damping_saturated': return
    row['termination']='outer_iteration_ceiling'


def evaluate(arm, targets, observations):
    # Keep the original metric/split/success helper; only intercept/account its one forward.
    class Batch:
        n,seeds=84,4
        def __call__(self,shader,p): return arm.forward(p,evaluation=True,physical=True)
    arm.row['final']=original.physical(arm.z).detach().tolist()
    old_budget={}
    original.evaluate_start(arm.row,original.physical(arm.z).detach(),None,Batch(),targets,observations,old_budget,arm.check)
    arm.row['original_evaluation_accounting']=old_budget


def run_arm(arm, targets, observations, persist):
    row=arm.row
    try:
        with arm.clock.active():
            arm.stage='optimization'; row['status']='optimizing'
            try:
                (run_adam if row['arm']=='Adam' else run_lm)(arm,persist)
            except ArmLimit as exc:
                row.update(termination='time_budget',budget_limited=True,budget_reason=str(exc))
                if row['iterations'] and row['iterations'][-1]['status'] in ('linearizing','trials'):
                    record=row['iterations'][-1];record['status']='interrupted_budget'
                    if record['trials'] and record['trials'][-1]['status']=='attempting':
                        record['trials'][-1]['status']='interrupted_budget'
            finally:
                arm.save_state(); persist()
            if arm.optimizer is not None:
                require(float(arm.optimizer.state[arm.z]['step'])==1000+row['additional_updates'],
                        'Interrupted Adam return boundary cannot be verified; no endpoint execution')
            arm.stage='endpoint'; arm.check(); arm.z.grad=None
            state=arm.forward(arm.z); gradient=arm.gradient(state,arm.z)
            row.update(final_fit_MSE=float(state['loss'].detach()), final_gradient=gradient.tolist(),
                       final_gradient_norm=float(torch.norm(gradient)),final_fit_predictions=state['values'].detach().tolist(),
                       final_fit_prediction_sha256=reference.sha_array(state['values'].detach().numpy()),
                       final_fit_mask=state['mask_identity'])
            arm.log(phase='endpoint',training_MSE=row['final_fit_MSE'],gradient_norm=row['final_gradient_norm'])
            evaluate(arm,targets,observations)
            arm.check()
            if row['budget_limited']:row['status']='complete_budget_limited'
    except ArmLimit as exc:
        row.update(status='incomplete_budget_limited',termination='time_budget',budget_limited=True,budget_reason=str(exc))
    except BaseException as exc:
        row.update(status='failed',failure_type=type(exc).__name__,failure=str(exc)); raise
    finally:
        arm.save_state(); row['active_wall_seconds']=arm.clock.wall; row['CPU_seconds']=arm.clock.cpu; persist()


def execute(old, evidence, report, persist, whole):
    begin=time.perf_counter(); cpu=time.process_time(); threads=torch.get_num_threads()
    rng=torch.get_rng_state().clone(); nrng=np.random.get_state(); torch.set_num_threads(1)
    try:
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
        report['runtime']=runtime
        require(all(runtime[k]==evidence['runtime'][k] for k in runtime),'Original numerical runtime required')
        for stage,pin in PINS.items():baseline(stage,pin)
        require(evidence['scene']==reference.design(),'Original scene changed')
        targets=np.asarray(old['targets']['fine'],np.float64)
        require(reference.sha_array(targets)==old['fine_targets_sha256'] and old['targets']['passed'] and old['identifiability']['passed'],
                'Original targets/gates invalid')
        target=torch.tensor(targets[:24,2],dtype=torch.float32)
        require(reference.sha_array(target.numpy())==old['fit_target_float32_sha256'],'Original fit target changed')
        tapes=[];report['tapes']={}
        for seeds,points,key in [(list(range(4)),24,'fit'),(list(range(1000,1004)),84,'evaluation')]:
            tape,ids=original.make_tapes(seeds,points);require(ids==old['tapes'][key],'Original fixed tapes changed')
            tapes.append(tape);report['tapes'][key]=ids
        hashes=[reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u,tape.v,tape.azimuth)]
        scene=old['scene'];obs=scene['observations'];fit_obs=[o for o in obs if o['split']=='fit']
        require(fit_obs==obs[:24],'Original fitting observation order changed')
        fit=original.ObservationBatch(scene,fit_obs,tapes[0],4); evaluation=original.ObservationBatch(scene,obs,tapes[1],4)
        shader=original.MaterialShaderV1('A2','atan2');require(shader.identity==old['shader'],'Accepted shader identity changed')
        report.update(scene=scene,shader=shader.identity,fine_targets_sha256=old['fine_targets_sha256'],
            fit_target_float32_sha256=old['fit_target_float32_sha256'],reused_gates=old['gates'],outcomes=[],starting_checks=[])
        arms=[]
        for saved in evidence['initializations']:
            z,opt=restore(saved);pair=[]
            for name,value,optimizer in [('Adam',z,opt),('LM',z.detach().clone().requires_grad_(),None)]:
                row=dict(seed=saved['seed'],arm=name,status='restored',budget_limited=False,termination=None,
                    additional_updates=0,completed_outer_iterations=0,work=work(),trajectory=[],iterations=[])
                arm=Arm(row,ArmClock(whole),shader,fit,evaluation,target,value,optimizer)
                arm.save_state();row['initial_state']=copy.deepcopy(row['final_state'])
                report['outcomes'].append(row);pair.append(arm)
            arms.append(pair)
        report['restoration_gate']='passed';persist()
        # Admit all ten initial forward identities before either algorithm updates anything.
        for saved,pair in zip(evidence['initializations'],arms):
            states=[]
            for arm in pair:
                with arm.clock.active():
                    arm.current=detached(arm.forward(arm.z));states.append(arm.current)
                    arm.row['initial_fit_MSE']=float(arm.current['loss'])
                    arm.row['initial_fit_predictions']=arm.current['values'].tolist()
                    arm.row['initial_fit_prediction_sha256']=reference.sha_array(arm.current['values'].numpy())
                    arm.row['initial_mask']=arm.current['mask_identity']
                    arm.log(phase='initial',training_MSE=float(arm.current['loss']))
            verify_start(saved,pair,states)
            report['starting_checks'].append(dict(seed=saved['seed'],exact_coordinates=True,exact_physical=True,
                bitwise_predictions=True,bitwise_residuals=True,exact_saved_MSE=True,fit_MSE=saved['final_fit_MSE']))
        report['initial_objective_gate']='passed';persist()
        for pair in arms:
            for arm in pair:
                report.update(phase='comparison',active_seed=arm.row['seed'],active_arm=arm.row['arm']);persist()
                run_arm(arm,targets,obs,persist)
                print('seed {} {}: {}, {}, MSE={}, steps={}'.format(arm.row['seed'],arm.row['arm'],
                    arm.row['status'],arm.row['termination'],arm.row.get('final_fit_MSE'),arm.row['work']['accepted_parameter_steps']),flush=True)
        require(hashes==[reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u,tape.v,tape.azimuth)],'Fixed tapes mutated')
        report.update(status='complete',phase='finished',fixed_tapes_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        after=np.random.get_state()
        report['global_torch_CPU_RNG_unchanged']=torch.equal(rng,torch.get_rng_state())
        report['global_numpy_RNG_unchanged']=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed',failure='Unexpected global CPU RNG consumption')
        torch.set_num_threads(threads);report['threads_restored']=torch.get_num_threads()==threads
        report['wall_seconds']=time.perf_counter()-begin;report['CPU_seconds']=time.process_time()-cpu;persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(all(os.environ.get(k)==v for k,v in original.REQUIRED_ENV.items()),'Explicit hidden-GPU/single-thread environment required')
    require(args.output.is_absolute() and args.output.resolve()==args.output and
            args.output==ROOT/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json','Explicit fresh versioned output required')
    report=dict(schema_version=1,protocol='material-recovery-optimizer-comparison-v1',milestone='3C-4',base_commit=BASE,
        status='running',phase='admission',restoration_gate='not_started',initial_objective_gate='not_started',
        input_sha256=INPUT_SHA,protocol_sha256=digest(PROTOCOL),implementation_sha256={
            'experiments/material_recovery_optimizer_comparison_v1.py':digest(__file__)},
        settings=dict(original_settings=copy.deepcopy(original.SETTINGS),starting_update=1000,
            Adam_additional_update_cap=ADAM_UPDATES,LM_outer_cap=LM_OUTERS,LM_trial_cap=LM_TRIALS,
            initial_lambda=LAMBDA_INITIAL,minimum_lambda=LAMBDA_MIN,maximum_lambda=LAMBDA_MAX,
            arm_wall_seconds=ARM_SECONDS,endpoint_reserve_seconds=RESERVE_SECONDS,whole_wall_seconds=WHOLE_SECONDS,
            residual_MSE_atol=MSE_ATOL,residual_MSE_rtol=MSE_RTOL),
        GPU_execution=False,neural_training=False,new_reference_evaluations=False,
        argv=[sys.executable,*sys.argv],started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)};whole=WholeClock()
    with args.output.open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            for s in handlers:signal.signal(s,whole.alarm)
            signal.setitimer(signal.ITIMER_REAL,WHOLE_SECONDS)
            old,evidence,identities=admit();report['accepted_identities']=identities
            execute(old,evidence,report,persist,whole)
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in handlers.items():signal.signal(s,handler)
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
