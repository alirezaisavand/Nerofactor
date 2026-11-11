"""One frozen, CPU-only, branch-local 15-arm LM trajectory experiment (3C-9)."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
from types import FunctionType

import numpy as np
import torch

from experiments import material_recovery_optimizer_comparison_v1 as lm
from experiments import material_gradient_equivariance_v1 as derivatives
from experiments import material_lm_proposal_gauge_v1 as proposals
from experiments import material_sample_coupling_v1 as coupling
from experiments import run_known_material_recovery_v1 as original
from experiments import known_material_reference_v1 as reference
from infrastructure.signatures import require, SignatureError
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = 'ebdfa2cdbf4bbd5d7b6ad32c1e558349c2c16153'
NAME = 'MATERIAL_LM_GAUGE_TRAJECTORIES_V1'
PROTOCOL = ROOT / ('docs/protocols/' + NAME + '.md')
OUTPUT = ROOT / ('docs/evidence/' + NAME + '.json')
CODE = ('experiments/material_lm_gauge_trajectories_v1.py',
        'tests/scientific/test_material_lm_gauge_trajectories_v1.py')
PERM = [1, 0, 2, 3, 4, 5, 6]
ARMS = ('P0', 'P1-U', 'P1-C')
WHOLE_SECONDS, RSS_BYTES, ADDRESS_BYTES = 2400., 4 * 1024**3, 8 * 1024**3
CAPS = dict(forward_attempts=883, jacobian_constructions=80, jacobian_vector_evaluations=560,
            reverse_mode_gradient_evaluations=1121, linear_solves=320,
            initial_forwards=1, candidate_forwards=320, terminal_fit_forwards=1,
            terminal_gradients=1, evaluation_forward_evaluations=1, branch_blocked_proposals=1,
            fit_points=84672, evaluation_points=336, diffuse_directions=43524096,
            specular_directions=87048192)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IntegrityFailure(RuntimeError):
    pass


class BranchBlocked(RuntimeError):
    pass


def invariant(passed, message):
    if not passed:
        raise IntegrityFailure(message)


def admit():
    # Admission only: no prior runner, renderer, quadrature or optimizer is executed.
    old, history, _, _, _, identities = derivatives.admit()
    _, _, extra, _ = proposals.admit()
    identities.update(extra)
    path = ROOT / 'docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json'
    require(digest(path) == '54920234b87edf17eeba688e0ffab83cdbd71f9246a4ca7654fb68f12bd5663e',
            'Accepted 3C-8 report changed')
    prior = json.loads(path.read_text())
    require(prior['status'] == 'complete' and prior['work']['completed_solves'] == 15,
            'Completed accepted proposal audit required')
    for key in ('accepted_identities', 'implementation_sha256'):
        for name, sha in prior[key].items():
            require(ROOT in (ROOT / name).resolve().parents and digest(ROOT / name) == sha,
                    'Prior source/evidence changed: ' + name)
            identities[name] = sha
    require(digest(PROTOCOL.parent / 'MATERIAL_LM_PROPOSAL_GAUGE_V1.md') == prior['protocol_sha256'],
            '3C-8 protocol changed')
    identities[str(path.relative_to(ROOT))] = digest(path)
    identities['docs/protocols/MATERIAL_LM_PROPOSAL_GAUGE_V1.md'] = prior['protocol_sha256']
    _, continuation, more = lm.admit()
    identities.update(more)
    return old, history, continuation, identities


def restore(saved):
    require(saved['status'] == 'complete' and saved['completed_updates'] == 1000, 'Update1000 required')
    record = saved['latest_state']['unconstrained']
    z = lm.accepted.float32_exact(record['values'], (7,), 'update1000')
    require(lm.accepted.describe(z) == record and
            lm.accepted.describe(original.physical(z)) == saved['latest_state']['physical'],
            'Exact latent/physical restoration required')
    return z


def gauge(z, inverse=False):
    """One original float32 translation, no wrap/canonicalization/inverse-logit."""
    require(z.shape == (7,) and z.dtype == torch.float32 and z.device.type == 'cpu', 'CPU float32 seven-vector required')
    out = z.detach()[PERM].clone()
    out[2] = z.detach()[2] + (-math.pi / 2 if inverse else math.pi / 2)
    return out


def branch(theta):
    """Cell inferred from the ACTUAL float32 double-angle atan2 discontinuity."""
    t = torch.tensor(float(theta), dtype=torch.float32)
    require(bool(torch.isfinite(t)), 'Finite represented theta required')
    enc = torch.stack((torch.cos(2*t), torch.sin(2*t)))
    unit = enc / torch.norm(enc)
    principal = .5 * torch.atan2(unit[1], unit[0])
    cell = round((float(t) - float(principal)) / math.pi)
    reconstruction_error = abs(float(t) - (float(principal) + cell * math.pi))
    require(reconstruction_error <= 8*np.finfo(np.float32).eps*max(1., abs(float(t))),
            'Decoder cell reconstruction failed')
    margin = max(0., math.pi/2 - abs(float(principal)))
    nominal = math.floor((float(t) + math.pi/2) / math.pi)
    require(cell == nominal or margin <= 2e-6, 'Actual/analytical branch classification disagreement away from cut')
    return dict(theta_float32=float(t), decoded_angle_float32=float(principal), cell=cell,
                margin_radians=margin, encoding=enc.tolist(), reconstruction_error=reconstruction_error,
                independent_analytic_cell=nominal)


def chart(z, arm):
    # Both actual decoder frames define the frozen local signed correspondence.
    other = gauge(z, inverse=arm != 'P0')
    return dict(own=branch(z[2]), counterpart=branch(other[2]))


def crossed(initial, candidate):
    return [key for key in ('own', 'counterpart') if initial[key]['cell'] != candidate[key]['cell']]


def initial_objective_gate(base, coupled):
    require(abs(coupled-base) <= 1e-12+5e-7*abs(base), 'Mandatory coupled initial objective gate failed')


def memory_check():
    invariant(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RSS_BYTES, '4 GiB peak RSS exceeded')


class WholeClock(lm.WholeClock):
    def remaining(self):
        return WHOLE_SECONDS - (time.perf_counter() - self.start)

    def check(self):
        memory_check()
        if self.remaining() <= 0:
            raise lm.WholeLimit('Single 2400-second whole-run budget reached; no retry')


def work():
    return dict(lm.work(), initial_forwards=0, candidate_forwards=0, terminal_fit_forwards=0,
                terminal_gradients=0, branch_blocked_proposals=0)


def check_caps(counts):
    for name, cap in CAPS.items():
        invariant(counts[name] <= cap, 'Arm work cap exceeded: ' + name)


class Arm(lm.Arm):
    def __init__(self, *args, sign=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign = sign
        self.in_jacobian = False
        self.row['initial_chart'] = chart(self.z, self.row['arm'])

    def forward(self, z, evaluation=False, physical=False):
        self.check()
        counts = self.row['work']
        invariant(counts['forward_attempts'] < CAPS['forward_attempts'], 'Forward budget exhausted')
        key = ('initial_forwards' if self.stage == 'initial' else
               'terminal_fit_forwards' if self.stage == 'endpoint' and not evaluation else
               'candidate_forwards' if self.stage == 'optimization' and not self.in_jacobian else None)
        if key:
            invariant(counts[key] < CAPS[key], 'Phase forward budget exhausted: ' + key)
            counts[key] += 1
        result = super().forward(z, evaluation, physical)
        check_caps(counts)
        return result

    def save_state(self):
        super().save_state()
        if self.sign is not None and self.row['arm'] == 'P1-C':
            p1 = original.physical(self.z).detach()
            p0 = original.physical(gauge(self.z, inverse=True))
            actual = coupling.frame_sign(coupling.decoded_frame(self.shader.shader, p0, 96),
                                         coupling.decoded_frame(self.shader.shader, p1, 96))
            invariant(actual['sign'] == self.sign and not crossed(self.row['initial_chart'], chart(self.z, 'P1-C')),
                      'Frozen coupled chart/sign failed at accepted state')
            self.row['last_verified_frame_mapping'] = actual


def guarded_loop(arm, persist):
    """Private namespace; original LM bytecode, proposal and AD operations unchanged."""
    def jacobian(function, z, expected, counts):
        invariant(counts['jacobian_constructions'] < CAPS['jacobian_constructions'], 'Jacobian budget exhausted')
        arm.in_jacobian = True
        try:
            value = lm.residual_jacobian(function, z, expected, counts)
        finally:
            arm.in_jacobian = False
        if len(arm.row['iterations']) == 1:
            arm.row['initial_J_normalized_float64'] = value.tolist()
            arm.row['initial_residual_float32'] = expected.tolist()
        check_caps(counts)
        return value

    def proposal(z, jac, residual, damping, counts):
        invariant(counts['linear_solves'] < CAPS['linear_solves'], 'Linear-solve budget exhausted')
        candidate, predicted, delta, represented = lm.proposal(z, jac, residual, damping, counts)
        rec = arm.row['iterations'][-1]
        trial = rec['trials'][-1]
        # Preserve invalid numerical proposals for the ORIGINAL rejection controller.
        if bool(torch.isfinite(candidate).all()):
            try:
                candidate_chart = chart(candidate, arm.row['arm'])
            except SignatureError as exc:
                raise IntegrityFailure('Candidate decoder-chart construction failed') from exc
            changes = crossed(arm.row['initial_chart'], candidate_chart)
            trial.update(candidate_chart=candidate_chart, branch_transition=changes)
            if changes:
                trial.update(status='branch_crossing_blocked', proposed_unconstrained=candidate.tolist(),
                             delta_float64=lm.finite_list(delta), represented_delta=lm.finite_list(represented),
                             predicted_reduction=predicted if math.isfinite(predicted) else None,
                             candidate_rendered=False, state_before=z.detach().tolist(),
                             active_wall_seconds_after=arm.clock.elapsed())
                arm.row['work']['branch_blocked_proposals'] += 1
                trial['work_after'] = copy.deepcopy(counts)
                rec.update(status='branch_crossing_blocked', lambda_after=damping,
                           z_after=z.detach().tolist(), training_MSE_after=float(arm.current['loss']))
                arm.row.update(termination='branch_crossing_blocked', damping=damping)
                arm.log(phase='branch_crossing_blocked', outer_iteration=rec['outer_iteration'],
                        unconstrained=z.detach().tolist(), physical=original.physical(z).detach().tolist(),
                        training_MSE=float(arm.current['loss']), damping=damping)
                arm.save_state()
                persist()
                raise BranchBlocked('First proposed departure from frozen decoder chart; no rendering/retry')
        check_caps(counts)
        return candidate, predicted, delta, represented

    namespace = dict(lm.run_lm.__globals__, residual_jacobian=jacobian, proposal=proposal)
    function = FunctionType(lm.run_lm.__code__, namespace, lm.run_lm.__name__, lm.run_lm.__defaults__, lm.run_lm.__closure__)
    function(arm, persist)


def evaluate(arm, targets, observations):
    p = original.physical(arm.z).detach()
    mapped = p.clone()
    if arm.row['arm'] != 'P0':
        mapped = p[PERM].clone()
        mapped[2] = p[2] - math.pi/2
    ideal = p.double()[PERM].numpy() if arm.row['arm'] != 'P0' else p.double().numpy()
    if arm.row['arm'] != 'P0':
        ideal[2] -= math.pi/2
    arm.row.update(final=p.tolist(), evaluation_gauge=dict(physical=mapped.tolist(),
        ideal_float64=ideal.tolist(), rounding_delta=(mapped.double().numpy()-ideal).tolist(),
        tapes='original seeds1000-1003', Q_relative_mapping_error=q_difference(mapped.tolist(), p.tolist())))

    class Batch:
        n, seeds = 84, 4

        def __call__(self, shader, physical):
            return arm.forward(physical, evaluation=True, physical=True)

    accounting = {}
    original.evaluate_start(arm.row, mapped, None, Batch(), targets, observations, accounting, arm.check)
    arm.row['original_evaluation_accounting'] = accounting
    # Physical-state recovery and rounded common-gauge recovery are both explicit.
    arm.row['unmapped_recovery_errors'] = original.recovery_errors(p.numpy())
    predictions = np.asarray(arm.row['final_predictions'])
    delta = predictions - targets
    compensation = {}
    for split in ('fit', 'held_out', 'relight'):
        mask = [o['split'] == split for o in observations]
        d, s, t = (delta[mask, i] for i in range(3))
        compensation[split] = dict(diffuse_specular_inner_product=float(np.sum(d*s)),
                                  residual_correlation=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s), 1e-30)),
                                  total_squared_error=float(np.sum(t*t)),
                                  component_squared_error=float(np.sum(d*d)+np.sum(s*s)))
    arm.row['component_compensation'] = compensation


def run_arm(arm, targets, observations, persist):
    row = arm.row
    try:
        with arm.clock.active():
            arm.stage = 'optimization'
            row['status'] = 'optimizing'
            try:
                guarded_loop(arm, persist)
            except BranchBlocked:
                row['censored'] = True
            except lm.ArmLimit as exc:
                row.update(termination='time_budget', budget_limited=True, budget_reason=str(exc), censored=True)
                if row['iterations']:
                    row['iterations'][-1]['status'] = 'interrupted_budget'
            finally:
                arm.save_state()
                persist()
            arm.stage = 'endpoint'
            arm.check()
            arm.z.grad = None
            state = arm.forward(arm.z)
            row['work']['terminal_gradients'] += 1
            gradient = arm.gradient(state, arm.z)
            row.update(final_fit_MSE=float(state['loss'].detach()), final_gradient=gradient.tolist(),
                       final_gradient_norm=float(torch.norm(gradient)), final_fit_predictions=state['values'].detach().tolist(),
                       final_fit_prediction_sha256=reference.sha_array(state['values'].detach().numpy()),
                       final_fit_mask=state['mask_identity'])
            arm.log(phase='endpoint', training_MSE=row['final_fit_MSE'], gradient_norm=row['final_gradient_norm'],
                    unconstrained=arm.z.detach().tolist(), physical=original.physical(arm.z).detach().tolist())
            evaluate(arm, targets, observations)
            arm.check()
            row['status'] = 'complete_censored' if row['censored'] else 'complete'
    except lm.ArmLimit as exc:
        row.update(status='incomplete_budget_limited', budget_limited=True, censored=True, budget_reason=str(exc))
    except BaseException as exc:
        row.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        arm.save_state()
        check_caps(row['work'])
        row.update(active_wall_seconds=arm.clock.wall, CPU_seconds=arm.clock.cpu)
        persist()


def q_difference(a, b):
    qa, qb = coupling.accepted.tensor_q(a), coupling.accepted.tensor_q(b)
    return float(np.linalg.norm(qa-qb)/np.linalg.norm(qb))


def state_difference(a, b):
    # Compare physical axes, not raw theta (which is gauge dependent).
    from experiments.material_recovery_metrics_v1 import gauge_metrics
    g = gauge_metrics(np.asarray(a)[:3], np.asarray(b)[:3])
    q = q_difference(a, b)
    k = float(np.max(abs(np.asarray(a)[3:6]-np.asarray(b)[3:6])))
    metallic = abs(a[6]-b[6])
    return dict(Q_relative=q, max_albedo_absolute=k, metallic_absolute=metallic,
                orientation_degrees=float(np.rad2deg(g['major_axis_error_radians'])),
                practical_similarity=bool(q <= .005 and k <= .002 and metallic <= .002))


def historical_control(row, archived):
    """Bitwise/exact numerical history; times, instrumentation and censoring excluded."""
    mismatches = []
    compared = 0
    for i, rec in enumerate(row['iterations']):
        if i >= len(archived['iterations']):
            mismatches.append('extra outer ' + str(i+1))
            break
        old = archived['iterations'][i]
        for key in ('z', 'training_MSE', 'lambda_before', 'jacobian_sha256', 'normal_matrix', 'gradient'):
            if key in rec and rec[key] != old.get(key):
                mismatches.append('outer{} {}'.format(i+1, key))
        for j, trial in enumerate(rec['trials']):
            if j >= len(old['trials']):
                mismatches.append('extra trial {}:{}'.format(i+1, j+1))
                break
            prior = old['trials'][j]
            for key in ('damping', 'proposed_unconstrained', 'delta_float64', 'represented_delta', 'predicted_reduction'):
                if key in trial and trial[key] != prior.get(key):
                    mismatches.append('trial{}:{} {}'.format(i+1, j+1, key))
            if trial['status'] not in ('branch_crossing_blocked', 'attempting', 'interrupted_budget'):
                for key in ('status', 'training_MSE', 'ratio', 'reason', 'mask', 'mask_flips'):
                    if trial.get(key) != prior.get(key):
                        mismatches.append('trial{}:{} {}'.format(i+1, j+1, key))
            compared += 1
    if not row['censored']:
        for key in ('termination', 'final_fit_MSE', 'final_fit_prediction_sha256', 'final_state'):
            if row.get(key) != archived.get(key):
                mismatches.append('final ' + key)
    return dict(passed=not mismatches, exact_numeric_mismatches=mismatches, compared_trials=compared,
                scope='shared prefix only' if row['censored'] else 'full trajectory and endpoint',
                archived_termination=archived['termination'])


def comparisons(rows):
    result = []
    for seed in range(101, 106):
        p0 = next(r for r in rows if r['seed'] == seed and r['arm'] == 'P0')
        for name in ('P1-U', 'P1-C'):
            p1 = next(r for r in rows if r['seed'] == seed and r['arm'] == name)
            record = dict(seed=seed, pair=['P0', name],
                          completed_pair=all(r['status'] == 'complete' and not r['censored'] for r in (p0, p1)))
            if all('initial_J_normalized_float64' in r and r['iterations'][0]['trials'] for r in (p0, p1)):
                a = np.asarray(p1['initial_J_normalized_float64'])[:, PERM]
                b = np.asarray(p0['initial_J_normalized_float64'])
                record['initial_J'] = derivatives.errors(a, b, derivatives.J_ATOL, derivatives.J_RTOL, derivatives.J_FLOOR)
                record['initial_H'] = derivatives.errors(a.T@a, b.T@b, 2e-9, 2e-4, 1e-12)
                t0, t1 = (r['iterations'][0]['trials'][0] for r in (p0, p1))
                for key in ('delta_float64', 'represented_delta'):
                    if key in t0 and key in t1 and None not in t0[key]+t1[key]:
                        record['initial_'+key] = derivatives.errors(np.asarray(t1[key])[PERM], t0[key],
                            proposals.REP_ATOL if key == 'represented_delta' else proposals.STEP_ATOL,
                            proposals.REP_RTOL if key == 'represented_delta' else proposals.STEP_RTOL, proposals.STEP_FLOOR)
                record['initial_predicted_reductions'] = [t.get('predicted_reduction') for t in (t0, t1)]
                if all('proposed_unconstrained' in t and None not in t['proposed_unconstrained'] for t in (t0, t1)):
                    record['initial_proposed_Q_relative'] = q_difference(
                        original.physical(torch.tensor(t1['proposed_unconstrained'])).tolist(),
                        original.physical(torch.tensor(t0['proposed_unconstrained'])).tolist())
            if all('final' in r for r in (p0, p1)):
                record['final_physical_difference'] = state_difference(p1['final'], p0['final'])
                record['final_similarity_verdict'] = ('censored_or_incomplete' if not record['completed_pair'] else
                    'pass' if record['final_physical_difference']['practical_similarity'] else 'fail')
                record['final_fit_MSE_difference'] = p1['final_fit_MSE']-p0['final_fit_MSE']
            aligned = []
            for a, b in zip(p0['iterations'], p1['iterations']):
                invariant(a['outer_iteration'] == b['outer_iteration'], 'Outer alignment error')
                aligned.append(dict(outer_iteration=a['outer_iteration'],
                    start_state_difference=state_difference(b['physical'], a['physical']),
                    same_exact_mapped_latent=bool(np.array_equal(gauge(torch.tensor(b['z']), inverse=True).numpy(), a['z']))))
            record['outer_start_alignment'] = aligned
            accepted_alignment = []
            if all('trajectory_analysis' in r for r in (p0, p1)):
                indexed = []
                for row in (p0, p1):
                    states = {}
                    for point in row['trajectory_analysis']['points']:
                        states.setdefault(point['accepted_updates'], point)
                    indexed.append(states)
                for count in sorted(indexed[0].keys() & indexed[1].keys()):
                    a, b = (states[count] for states in indexed)
                    accepted_alignment.append(dict(accepted_update_count=count,
                        physical_difference=state_difference(b['physical'], a['physical']),
                        mapped_latent_L2=float(np.linalg.norm(np.asarray(b['mapped_latent'])-a['mapped_latent']))))
            record['accepted_update_alignment'] = accepted_alignment
            record['first_practical_divergence_outer'] = next((r['outer_iteration'] for r in aligned
                if not r['start_state_difference']['practical_similarity']), None)
            record['comparison_limit'] = 'Initial proposals share physical state; later aligned indices are distinct optimization states, not proposal gauge proofs'
            result.append(record)
    return result


def execute(old, history, continuation, report, persist, whole):
    threads = torch.get_num_threads()
    rng, nrng = torch.get_rng_state().clone(), np.random.get_state()
    torch.set_num_threads(1)
    try:
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
        require(runtime == history['runtime'], 'Historical numerical runtime required')
        report['runtime'] = runtime
        for stage, pin in PINS.items():
            baseline(stage, pin)
        require(old['scene'] == reference.design() and old['targets']['passed'] and old['identifiability']['passed'],
                'Accepted original scene/reference/gates required')
        targets = np.asarray(old['targets']['fine'], np.float64)
        target = torch.tensor(targets[:24, 2], dtype=torch.float32)
        require(reference.sha_array(targets) == old['fine_targets_sha256'] and
                reference.sha_array(target.numpy()) == old['fit_target_float32_sha256'], 'Original targets changed')
        fit_tape, fit_ids = original.make_tapes(list(range(4)), 24)
        eval_tape, eval_ids = original.make_tapes(list(range(1000, 1004)), 84)
        require(fit_ids == old['tapes']['fit'] and eval_ids == old['tapes']['evaluation'], 'Original tapes changed')
        report.update(tapes=dict(fit=fit_ids, evaluation=eval_ids), scene=old['scene'], shader=old['shader'],
                      fine_targets_sha256=old['fine_targets_sha256'], reused_gates=old['gates'], outcomes=[], starting_checks=[])
        obs = old['scene']['observations']
        evaluation = original.ObservationBatch(old['scene'], obs, eval_tape, 4)
        shader = original.MaterialShaderV1('A2', 'atan2')
        require(shader.identity == old['shader'], 'Shader identity changed')
        all_tapes = [fit_tape, eval_tape]
        arms = []
        for saved in continuation['initializations']:
            z = restore(saved)
            archived = next(r for r in history['outcomes'] if r['arm'] == 'LM' and r['seed'] == saved['seed'])
            require(lm.accepted.describe(z) == archived['initial_state']['unconstrained'], 'P0 historical initial coordinates changed')
            swapped = gauge(z)
            p0, p1 = original.physical(z), original.physical(swapped)
            q0, q1 = coupling.accepted.tensor_q(p0.tolist()), coupling.accepted.tensor_q(p1.tolist())
            require(np.all(abs(q0-q1) <= 2e-7+2e-6*abs(q0)) and torch.equal(p0[3:], p1[3:]), 'Initial rounded physical gauge mismatch')
            mapping = coupling.frame_sign(coupling.decoded_frame(shader, p0, 96), coupling.decoded_frame(shader, p1, 96))
            copied, tape_record = coupling.copy_coupled_tape(fit_tape, mapping['sign'])
            all_tapes.append(copied)
            report['starting_checks'].append(dict(seed=saved['seed'], restored_exactly=True, Q_relative=q_difference(p1.tolist(), p0.tolist()),
                albedo_metallic_identical=True, frame_mapping=mapping, coupled_tape=tape_record))
            for name in ARMS:
                selected = copied if name == 'P1-C' else fit_tape
                fit = original.ObservationBatch(old['scene'], obs[:24], selected, 4)
                row = dict(seed=saved['seed'], arm=name, status='restored', budget_limited=False, censored=False,
                           termination=None, completed_outer_iterations=0, work=work(), trajectory=[], iterations=[])
                arm = Arm(row, lm.ArmClock(whole), shader, fit, evaluation, target,
                          (z if name == 'P0' else swapped).clone().requires_grad_(), sign=mapping['sign'])
                arm.save_state()
                row['initial_state'] = copy.deepcopy(row['final_state'])
                report['outcomes'].append(row)
                arms.append(arm)
        identities = [[reference.sha_array(t.numpy()) for t in (tape.u, tape.v, tape.azimuth)] for tape in all_tapes]
        report['restoration_gate'] = 'passed'
        persist()
        for i, arm in enumerate(arms):
            with arm.clock.active():
                arm.current = lm.detached(arm.forward(arm.z))
                arm.row.update(initial_fit_MSE=float(arm.current['loss']), initial_fit_predictions=arm.current['values'].tolist(),
                    initial_fit_prediction_sha256=reference.sha_array(arm.current['values'].numpy()), initial_mask=arm.current['mask_identity'])
                arm.log(phase='initial', training_MSE=float(arm.current['loss']), unconstrained=arm.z.detach().tolist(),
                        physical=original.physical(arm.z).detach().tolist())
            if arm.row['arm'] == 'P0':
                saved = next(r for r in history['outcomes'] if r['arm'] == 'LM' and r['seed'] == arm.row['seed'])
                require(all(arm.row[key] == saved[key] for key in ('initial_fit_MSE', 'initial_fit_predictions', 'initial_fit_prediction_sha256', 'initial_mask')),
                        'Mandatory P0 initial forward identity failed')
            elif arm.row['arm'] == 'P1-C':
                p0 = arms[i-2].row['initial_fit_MSE']
                error = abs(arm.row['initial_fit_MSE']-p0)
                initial_objective_gate(p0, arm.row['initial_fit_MSE'])
                report['starting_checks'][i//3].update(initial_MSEs=[a.row['initial_fit_MSE'] for a in arms[i-2:i+1]],
                    coupled_objective_absolute_error=error, coupled_objective_gate=True,
                    uncoupled_objective_absolute_error=abs(arms[i-1].row['initial_fit_MSE']-p0))
        invariant(identities == [[reference.sha_array(t.numpy()) for t in (t.u, t.v, t.azimuth)] for t in all_tapes], 'Tape mutation during construction')
        report['initial_objective_gate'] = 'passed'
        persist()
        for arm in arms:
            report.update(phase='optimization', active_seed=arm.row['seed'], active_arm=arm.row['arm'])
            persist()
            run_arm(arm, targets, obs, persist)
            arm.row['trajectory_analysis'] = trajectory_analysis(arm.row)
            if arm.row['arm'] == 'P0':
                saved = next(r for r in history['outcomes'] if r['arm'] == 'LM' and r['seed'] == arm.row['seed'])
                arm.row['historical_control'] = historical_control(arm.row, saved)
            print('seed {} {}: {}, {}, accepted={}, MSE={}'.format(arm.row['seed'], arm.row['arm'], arm.row['status'],
                arm.row['termination'], arm.row['work']['accepted_parameter_steps'], arm.row.get('final_fit_MSE')), flush=True)
        invariant(identities == [[reference.sha_array(t.numpy()) for t in (t.u, t.v, t.azimuth)] for t in all_tapes], 'Fixed tapes mutated')
        totals = {k: sum(a.row['work'][k] for a in arms) for k in work()}
        for key, cap in CAPS.items():
            invariant(totals[key] <= 15*cap, 'Whole-run work cap exceeded: '+key)
        report.update(status='complete', phase='finished', fixed_tapes_unchanged=True,
                      total_work=totals, comparisons=comparisons(report['outcomes']))
    finally:
        after = np.random.get_state()
        report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng, torch.get_rng_state())
        report['global_numpy_RNG_unchanged'] = nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:]
        torch.set_num_threads(threads)
        report['threads_restored'] = torch.get_num_threads() == threads
        persist()
        invariant(report['global_torch_CPU_RNG_unchanged'] and report['global_numpy_RNG_unchanged'], 'Unexpected global RNG consumption')


def trajectory_analysis(row):
    for rec in row['iterations']:
        for i, trial in enumerate(rec['trials']):
            trial['damping_after'] = (rec['trials'][i+1]['damping'] if i+1 < len(rec['trials']) else
                                      rec.get('lambda_after', row.get('damping')))
    points = []
    # Identified outer-end or terminal states; initial state always included.
    for entry in row['trajectory']:
        if 'physical' not in entry:
            continue
        p = entry['physical']
        points.append(dict(phase=entry.get('phase', 'outer_end'), outer_iteration=entry.get('outer_iteration'),
            accepted_updates=entry['work']['accepted_parameter_steps'], training_MSE=entry['training_MSE'],
            active_wall_seconds=entry['active_wall_seconds'], work=entry['work'],
            Q_error=original.recovery_errors(p)['Q_relative_frobenius'],
            Q=coupling.accepted.tensor_q(p).tolist(), physical=p,
            mapped_latent=(gauge(torch.tensor(entry['unconstrained']), inverse=True).tolist()
                           if row['arm'] != 'P0' else entry['unconstrained'])))
    return dict(points=points, final_Q=coupling.accepted.tensor_q(row['final']).tolist() if 'final' in row else None,
        zero_represented_proposals=sum(all(v == 0 for v in t.get('represented_delta', [None]))
            for rec in row['iterations'] for t in rec['trials']),
        proposal_lengths=[dict(outer=rec['outer_iteration'], trial=t['trial'], status=t['status'],
            mathematical_L2=float(np.linalg.norm(t['delta_float64'])), represented_L2=float(np.linalg.norm(t['represented_delta'])))
            for rec in row['iterations'] for t in rec['trials']
            if 'delta_float64' in t and None not in t['delta_float64']+t['represented_delta']])


def make_plots(report, destination):
    """Offline figures only, from saved evidence; no shader/solver calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    colors = dict(zip(ARMS, ('#222222', '#c65102', '#0868ac')))
    figures = {}
    for name, xkey, ykey, xlabel, ylabel in (
        ('training-loss', 'outer_iteration', 'training_MSE', 'Outer iteration (initial=0)', 'Training MSE'),
        ('training-loss-wall', 'active_wall_seconds', 'training_MSE', 'Active wall seconds', 'Training MSE'),
        ('training-loss-work', 'work', 'training_MSE', 'Full-batch shader forwards (includes JVP primals)', 'Training MSE'),
        ('material-Q', 'accepted_updates', 'Q_error', 'Accepted updates', 'Relative Q error to truth')):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), squeeze=False)
        for ax, seed in zip(axes[0], range(101, 106)):
            for row in report['outcomes']:
                if row['seed'] != seed or 'trajectory_analysis' not in row:
                    continue
                points = row['trajectory_analysis']['points']
                xs = [(p['work']['full_batch_forward_evaluations'] if xkey == 'work' else
                       p['outer_iteration'] or 0 if xkey == 'outer_iteration' else p[xkey]) for p in points]
                ys = [p[ykey] for p in points]
                ax.semilogy(xs, ys, color=colors[row['arm']], label=row['arm'], marker='.', markersize=3)
            ax.set_title('Seed '+str(seed))
            ax.set_xlabel(xlabel, fontsize=8)
            ax.grid(alpha=.25)
        axes[0, 0].set_ylabel(ylabel)
        axes[0, -1].legend(fontsize=8)
        fig.tight_layout()
        path = destination/(name+'.svg')
        fig.savefig(path, metadata={'Date': None})
        plt.close(fig)
        figures[str(path.relative_to(ROOT))] = digest(path)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in original.REQUIRED_ENV.items()), 'Explicit CPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT, 'Fresh explicit versioned output required')
    freeze = json.loads((ROOT/'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.freeze.json').read_text())
    for name, sha in freeze['sha256'].items():
        require(digest(ROOT/name) == sha, 'Pre-execution freeze mismatch: ' + name)
    report = dict(schema_version=1, protocol='material-lm-gauge-trajectories-v1', milestone='3C-9', base_commit=BASE,
        status='running', phase='admission', frozen_execution=freeze, protocol_sha256=digest(PROTOCOL),
        implementation_sha256={n: digest(ROOT/n) for n in CODE}, GPU_execution=False, neural_training=False,
        new_reference_evaluations=False, settings=dict(original=original.SETTINGS, arms=list(ARMS), seeds=list(range(101, 106)),
        lambda_initial=.01, lambda_min=1e-8, lambda_max=1e8, outer_cap=80, trial_cap=4, per_arm_caps=CAPS,
        whole_caps={k:15*v for k, v in CAPS.items()}, active_arm_wall_seconds=120, endpoint_reserve_seconds=10,
        whole_wall_seconds=WHOLE_SECONDS, RSS_bytes=RSS_BYTES, address_bytes=ADDRESS_BYTES,
        physical_Q_atol=2e-7, physical_Q_rtol=2e-6, initial_objective_atol=1e-12, initial_objective_rtol=5e-7,
        final_similarity=dict(Q_relative=.005, albedo_max=.002, metallic=.002)),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), argv=[sys.executable, *sys.argv])
    whole = WholeClock()
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    address = resource.getrlimit(resource.RLIMIT_AS)
    begin, cpu = time.perf_counter(), time.process_time()
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream, report)
        persist()
        try:
            for sig in handlers:
                signal.signal(sig, whole.alarm)
            signal.setitimer(signal.ITIMER_REAL, WHOLE_SECONDS)
            cap = min(ADDRESS_BYTES, address[0]) if address[0] >= 0 else ADDRESS_BYTES
            if address[1] >= 0:
                cap = min(cap, address[1])
            resource.setrlimit(resource.RLIMIT_AS, (cap, address[1]))
            memory_check()
            old, history, continuation, identities = admit()
            report['accepted_identities'] = identities
            execute(old, history, continuation, report, persist, whole)
        except BaseException as exc:
            report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
            raise
        finally:
            report.update(wall_seconds=time.perf_counter()-begin, CPU_seconds=time.process_time()-cpu,
                          peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            resource.setrlimit(resource.RLIMIT_AS, address)
            report['memory_limit_restored'] = resource.getrlimit(resource.RLIMIT_AS) == address
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            persist()
    print('Final status:', report['status'], flush=True)


if __name__ == '__main__':
    main()
