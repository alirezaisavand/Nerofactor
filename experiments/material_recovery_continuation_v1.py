"""One CPU continuation: restore five admitted Adam states, updates 251..1000 only."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np
import torch

from experiments import run_known_material_recovery_v1 as original
from experiments import material_recovery_trajectories_v1 as admission
from experiments import known_material_reference_v1 as reference
from infrastructure.signatures import require, write_report
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = '15d0a07e65e5d3f18f72d68819b62e386886c492'
PROTOCOL = ROOT / 'docs/protocols/MATERIAL_RECOVERY_CONTINUATION_V1.md'
DIAGNOSTIC_SHA = '6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f'
SEEDS = list(range(101, 106))
START, FINISH = 250, 1000
MAX_RETURNS = 3750
MAX_FIT_POINTS = 360480  # (3750 update forwards + five endpoint gradients) * 24 * 4


def float32_exact(value, shape, name):
    """JSON numbers must already be exactly representable, never silently rounded."""
    require(isinstance(value, (list, int, float)) and not isinstance(value, bool), 'Invalid ' + name)
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'Nonfinite/wrong shape: ' + name)
    b = a.astype(np.float32)
    require(np.array_equal(b.astype(np.float64), a), 'Non-exact float32 JSON values: ' + name)
    return torch.tensor(b, dtype=torch.float32, device='cpu')


def describe(tensor):
    a = tensor.detach().cpu().numpy()
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'device': str(tensor.device),
            'values': a.tolist(), 'sha256': reference.sha_array(a)}


def optimizer_options(optimizer):
    return json.loads(json.dumps({k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}))


def snapshot(z, optimizer):
    return {'unconstrained': describe(z), 'physical': describe(original.physical(z)),
            'adam': {k: describe(v) for k, v in optimizer.state[z].items()},
            'optimizer_effective': optimizer_options(optimizer)}


def restore(row):
    """Source/runtime-established CPU float32 Adam representation; no update/replay."""
    require(torch.get_default_dtype() == torch.float32, 'Original default float32 dtype required')
    require(row['seed'] in SEEDS and row['status'] == 'complete' and row['completed_updates'] == START,
            'Expected admitted completed update-250 state')
    z = float32_exact(row['last_unconstrained'], (7,), 'unconstrained').requires_grad_()
    p = float32_exact(row['final'], (7,), 'physical')
    require(torch.equal(p, original.physical(z).detach()) and row['final'] == row['last_physical'],
            'Saved physical/latent state mismatch')
    saved = row['optimizer_final_state']
    require(set(saved) == {'step', 'exp_avg', 'exp_avg_sq'} and saved['step'] == START,
            'Missing/foreign Adam state or step counter')
    tensors = {k: float32_exact(v, () if k == 'step' else (7,), 'Adam ' + k) for k, v in saved.items()}
    require(bool((tensors['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    config = original.SETTINGS['optimizer']
    expected = dict(lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'],
                    amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    require(row['optimizer_effective'] == expected, 'Optimizer flags/defaults differ from original')
    optimizer = torch.optim.Adam([z], **{**expected, 'betas': tuple(expected['betas'])})
    state = optimizer.state_dict()
    require(len(state['param_groups']) == 1 and len(state['param_groups'][0]['params']) == 1,
            'One original seven-vector parameter required')
    state['state'] = {state['param_groups'][0]['params'][0]: tensors}
    optimizer.load_state_dict(state)
    require(optimizer_options(optimizer) == expected, 'Optimizer configuration restoration mismatch')
    actual = optimizer.state[z]
    require(set(actual) == set(tensors) and all(
        v.dtype == torch.float32 and v.device.type == 'cpu' and v.shape == tensors[k].shape and
        torch.equal(v, tensors[k]) for k, v in actual.items()), 'Adam state restoration mismatch')
    return z, optimizer


def admit(input_path, diagnostic_path):
    evidence, identities = admission.admit(input_path, admission.EVIDENCE_SHA)
    require(evidence['settings'] == original.SETTINGS, 'Original settings changed')
    require(evidence['fixed_tapes_unchanged'] and evidence['global_torch_CPU_RNG_unchanged'] and
            evidence['global_numpy_RNG_unchanged'], 'Original fixed randomness not established')
    require(admission.digest(diagnostic_path) == DIAGNOSTIC_SHA, '3C-2 diagnostic SHA-256 mismatch')
    diagnostic = json.loads(diagnostic_path.read_text())
    require(diagnostic['status'] == 'complete' and diagnostic['input_sha256'] == admission.EVIDENCE_SHA,
            'Incomplete/foreign diagnostic baseline')
    for name, sha in diagnostic['implementation_sha256'].items():
        require(ROOT in (ROOT / name).resolve().parents and admission.digest(ROOT / name) == sha,
                'Accepted diagnostic source changed: ' + name)
    require(admission.digest(ROOT / 'docs/protocols/MATERIAL_RECOVERY_DIAGNOSTICS_V1.md') ==
            diagnostic['protocol_sha256'], 'Accepted diagnostic protocol changed')
    states = {r['id']: r for r in diagnostic['states']}
    for row in evidence['initializations']:
        d = states['seed-{}'.format(row['seed'])]
        require(d['effective_physical_float32'] == row['final'] and d['unconstrained_gradient_kind'] == 'autograd',
                'Diagnostic terminal-state identity mismatch')
    return evidence, diagnostic, identities


def count_fit(budget, points, endpoint=False):
    require(points == 96 and budget['fit_shader_points'] + points <= MAX_FIT_POINTS,
            'Fixed fit shape/point budget exceeded')
    key = 'endpoint_gradient_calls' if endpoint else 'update_forward_calls'
    require(budget[key] < (5 if endpoint else MAX_RETURNS), 'Fixed forward budget exceeded')
    budget[key] += 1
    budget['fit_shader_points'] += points
    budget['fit_specular_directions'] += points * 1024
    budget['fit_diffuse_directions'] += points * 512


def objective(z, shader, batch, target, budget, deadline, endpoint=False):
    deadline(); count_fit(budget, batch.n * batch.seeds, endpoint)
    predictions = batch(shader, original.physical(z))
    require(bool(torch.isfinite(predictions).all()), 'Nonfinite fit radiance')
    loss = torch.mean((predictions[:, 2] - target) ** 2)
    require(bool(torch.isfinite(loss)), 'Nonfinite objective')
    loss.backward()
    require(z.grad is not None and bool(torch.isfinite(z.grad).all()), 'Nonfinite/missing latent gradients')
    return loss, float(torch.norm(z.grad))


def continue_start(row, z, optimizer, shader, batch, target, budget, deadline, persist):
    require(row['status'] == 'restored' and row['completed_updates'] == START and
            float(optimizer.state[z]['step']) == START, 'Continuation must start at exact update250')
    row['status'] = 'optimizing'
    try:
        for update in range(START + 1, FINISH + 1):
            require(budget['optimizer_returns'] < MAX_RETURNS, 'Update budget exhausted')
            optimizer.zero_grad()
            loss, gradient = objective(z, shader, batch, target, budget, deadline)
            deadline(); optimizer.step()
            # Count successful original Adam returns, including a subsequently failed finite check.
            budget['optimizer_returns'] += 1; row['additional_updates'] += 1; row['completed_updates'] = update
            require(float(optimizer.state[z]['step']) == update and bool(torch.isfinite(z).all()) and
                    all(bool(torch.isfinite(v).all()) for v in optimizer.state[z].values()),
                    'Nonfinite state or Adam counter diverged')
            if update == START + 1 or update % 10 == 0:
                row['history'].append({'completed_updates': update, 'loss_at_update_count': update - 1,
                    'fit_MSE_before_update': float(loss.detach()), 'gradient_norm_before_update': gradient,
                    'physical_after_update': original.physical(z).detach().tolist(),
                    'unconstrained_after_update': z.detach().tolist()})
                row['latest_state'] = snapshot(z, optimizer); persist()
            if update % 100 == 0:
                print('seed {}: {}/1000 total updates'.format(row['seed'], update), flush=True)
        optimizer.zero_grad()
        loss, gradient = objective(z, shader, batch, target, budget, deadline, endpoint=True)
        row.update(status='optimized', final=original.physical(z).detach().tolist(),
                   final_fit_MSE=float(loss.detach()), final_gradient_norm=gradient,
                   final_gradient=z.grad.detach().tolist())
        return original.physical(z).detach()
    except BaseException as exc:
        row.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        row['latest_state'] = snapshot(z, optimizer); persist()


def comparison(old, new, diagnostic):
    before = {'update': START, 'physical': old['final'], 'recovery_errors': old['recovery_errors'],
              'radiance_errors': old['radiance_errors'], 'success': old['success'],
              'fit_MSE': diagnostic['objective_float32'],
              'gradient_norm': float(np.linalg.norm(diagnostic['g_unconstrained']))}
    after = {'update': FINISH, 'physical': new['final'], 'recovery_errors': new['recovery_errors'],
             'radiance_errors': new['radiance_errors'], 'success': new['success'],
             'fit_MSE': new['final_fit_MSE'], 'gradient_norm': new['final_gradient_norm']}
    h = new['history']; late = h[-3:]
    return {'seed': new['seed'], 'update250': before, 'update1000': after,
            'fit_MSE_ratio_1000_to_250': after['fit_MSE'] / before['fit_MSE'],
            'last20_preupdate_loss_reduction': 1 - late[-1]['fit_MSE_before_update'] / late[0]['fit_MSE_before_update'],
            'last10_physical_delta': (np.asarray(h[-1]['physical_after_update']) - np.asarray(h[-2]['physical_after_update'])).tolist(),
            'stationarity': 'Not established by a small loss or finite budget; inspect remaining gradients and trajectory'}


def execute(evidence, diagnostic, report, persist):
    begin, cpu = time.perf_counter(), time.process_time()
    deadline = original.Deadline(); old_threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone(); numpy_rng = np.random.get_state()
    torch.set_num_threads(1)
    budget = dict(optimizer_returns=0, update_forward_calls=0, endpoint_gradient_calls=0,
                  fit_shader_points=0, fit_specular_directions=0, fit_diffuse_directions=0,
                  evaluation_shader_points=0, new_reference_direction_evaluations=0)
    report['budget'] = budget
    try:
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                       device='cpu', torch_threads=torch.get_num_threads())
        report['runtime'] = runtime
        require(all(runtime[k] == evidence['runtime'][k] for k in runtime), 'Original numerical runtime required')
        for stage, pin in PINS.items(): baseline(stage, pin)
        require(torch.get_default_dtype() == torch.float32, 'Original float32 defaults required')
        scene = evidence['scene']; require(scene == reference.design(), 'Observation/scene design changed')
        targets = np.asarray(evidence['targets']['fine'], dtype=np.float64)
        require(targets.shape == (84, 3, 3) and np.isfinite(targets).all() and
                reference.sha_array(targets) == evidence['fine_targets_sha256'] and
                evidence['targets']['passed'] and evidence['identifiability']['passed'], 'Reused scientific gates/targets invalid')
        target = torch.tensor(targets[:24, 2], dtype=torch.float32)
        require(reference.sha_array(target.numpy()) == evidence['fit_target_float32_sha256'], 'Fit targets changed')
        tapes = []; identities = []
        for seeds, points, key in [(list(range(4)), 24, 'fit'), (list(range(1000, 1004)), 84, 'evaluation')]:
            tape, ids = original.make_tapes(seeds, points)
            require(ids == evidence['tapes'][key], 'Original tape identity mismatch')
            tapes.append(tape); identities.append(ids)
        tape_hashes = [reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u, tape.v, tape.azimuth)]
        report['tapes'] = dict(zip(('fit', 'evaluation'), identities))
        report['scene'] = scene
        report['fine_targets_sha256'] = evidence['fine_targets_sha256']
        report['fit_target_float32_sha256'] = evidence['fit_target_float32_sha256']
        fit_obs = [o for o in scene['observations'] if o['split'] == 'fit']
        require(fit_obs == scene['observations'][:24], 'Original ordered fitting observations changed')
        fit = original.ObservationBatch(scene, fit_obs, tapes[0], 4)
        evaluation = original.ObservationBatch(scene, scene['observations'], tapes[1], 4)
        # Restore ALL FIVE before any forward/update; there is no partial admission.
        restored = []
        report['initializations'] = []
        for old in evidence['initializations']:
            z, opt = restore(old); restored.append((z, opt))
            report['initializations'].append(dict(seed=old['seed'], status='restored', completed_updates=START,
                additional_updates=0, history=[], restoration=snapshot(z, opt), final=None))
        report['restoration_gate'] = 'passed'; persist(); deadline()
        shader = original.MaterialShaderV1('A2', 'atan2')
        require(shader.identity == evidence['shader'], 'Accepted shader identity mismatch')
        report['shader'] = shader.identity
        states = {r['id']: r for r in diagnostic['states']}; report['comparisons'] = []
        for old, row, (z, opt) in zip(evidence['initializations'], report['initializations'], restored):
            try:
                report.update(phase='continuation', active_seed=row['seed']); persist()
                p = continue_start(row, z, opt, shader, fit, target, budget, deadline, persist)
                report['phase'] = 'evaluation'; persist()
                original.evaluate_start(row, p, shader, evaluation, targets, scene['observations'], budget, deadline)
                require(np.isfinite(np.asarray(row['final_gradient'])).all(), 'Final gradient nonfinite')
                report['comparisons'].append(comparison(old, row, states['seed-{}'.format(row['seed'])])); persist()
            except BaseException as exc:
                row.update(status='failed', failure_phase=report['phase'], failure_type=type(exc).__name__, failure=str(exc))
                raise
        require(budget['optimizer_returns'] == MAX_RETURNS and budget['update_forward_calls'] == MAX_RETURNS and
                budget['endpoint_gradient_calls'] == 5 and budget['fit_shader_points'] == MAX_FIT_POINTS and
                budget['evaluation_shader_points'] == 1680 and all(
                    r['additional_updates'] == 750 and r['completed_updates'] == FINISH and r['status'] == 'complete'
                    for r in report['initializations']), 'Exact update/evaluation budget not completed')
        require(tape_hashes == [reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u, tape.v, tape.azimuth)],
                'Original fixed tapes mutated')
        report.update(status='complete', phase='finished', fixed_tapes_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        report['stability'] = original.stability(report.get('initializations', []))
        after = np.random.get_state()
        report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng, torch.get_rng_state())
        report['global_numpy_RNG_unchanged'] = numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed', failure='Unexpected global RNG consumption')
        torch.set_num_threads(old_threads); report['threads_restored'] = torch.get_num_threads() == old_threads
        report['wall_seconds'] = time.perf_counter() - begin; report['CPU_seconds'] = time.process_time() - cpu
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in original.REQUIRED_ENV.items()), 'Hidden GPU/single CPU thread required')
    require(args.output.is_absolute() and args.output.resolve() == args.output and ROOT in args.output.parents and
            args.output == ROOT / 'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json', 'Explicit fresh versioned output required')
    report = dict(schema_version=1, protocol='material-recovery-continuation-v1', milestone='3C-3', base_commit=BASE,
        status='running', phase='admission', restoration_gate='not_started',
        settings={**copy.deepcopy(original.SETTINGS), 'max_updates_per_start': FINISH},
        additional_updates_per_start=750, input_path=str(args.input), input_sha256=admission.EVIDENCE_SHA,
        diagnostic_path=str(args.diagnostic), diagnostic_sha256=DIAGNOSTIC_SHA,
        protocol_sha256=admission.digest(PROTOCOL), implementation_sha256={
            str(Path(__file__).relative_to(ROOT)): admission.digest(__file__)},
        argv=[sys.executable, *sys.argv], started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        neural_network_training=False, GPU_execution=False, new_reference_evaluations=False)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame): raise original.RuntimeLimit('Single approved 1800-second boundary; no retry')
    with args.output.open('x') as stream:
        def persist(): write_report(stream, report)
        persist()
        try:
            for s in handlers: signal.signal(s, stop)
            signal.alarm(1800)
            evidence, diagnostic, identities = admit(args.input, args.diagnostic)
            report['accepted_source_identities'] = identities
            report['truth'] = evidence['truth']; report['parameter_order'] = evidence['parameter_order']
            report['reused_gates'] = evidence['gates']
            execute(evidence, diagnostic, report, persist)
        except BaseException as exc:
            report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc)); raise
        finally:
            signal.alarm(0)
            for s, handler in handlers.items(): signal.signal(s, handler)
            report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); persist()
    print('Final status:', report['status'], flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__': raise SystemExit(main())
