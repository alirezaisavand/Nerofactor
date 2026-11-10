"""ONE approved CPU-only flat-material v1 experiment; fixed protocol, no overrides."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np
import torch

from experiments import known_material_reference_v1 as reference
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs
from experiments.material_recovery_metrics_v1 import gauge_metrics, roughness_tensor
from infrastructure.signatures import require, write_report
from infrastructure.stage_sources import ROOT, digest, baseline, PINS

BASE = '179e36eb01e976d216a45fd323cc764308b49bd0'
PROTOCOL_PATH = ROOT / 'docs/protocols/KNOWN_MATERIAL_RECOVERY_V1.md'
EXECUTION_PATH = ROOT / 'docs/KNOWN_MATERIAL_RECOVERY_V1_EXECUTION.md'
REQUIRED_ENV = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
SETTINGS = {'variant': 'A2', 'decoder': 'atan2', 'diffuse_samples': 512, 'specular_samples': 1024,
            'initialization_seeds': list(range(101, 106)), 'fit_seeds': list(range(4)),
            'evaluation_seeds': list(range(1000, 1004)), 'max_updates_per_start': 250,
            'optimizer': {'lr': .01, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
            'scheduler': None, 'clipping': None, 'regularizers': None, 'noise': None,
            'reference_grids': [list(g) for g in reference.GRIDS], 'reference_atol': reference.ATOL,
            'reference_rtol': reference.RTOL, 'rank_ratio': reference.RANK_RATIO,
            'fd_steps': list(reference.FD_STEPS), 'wall_seconds': 1800, 'torch_threads': 1,
            'success': {'Q_relative': .05, 'orientation_degrees': 5., 'max_albedo_absolute': .02,
                        'metallic_absolute': .03, 'held_out_relative_L2': .01}}


class RuntimeLimit(RuntimeError): pass


class Deadline:
    def __init__(self, seconds=1800):
        self.start = time.perf_counter(); self.seconds = seconds

    def __call__(self):
        if time.perf_counter() - self.start >= self.seconds: raise RuntimeLimit('Approved CPU wall cap reached')


def initializations():
    rows = []
    for seed in SETTINGS['initialization_seeds']:
        rng = np.random.RandomState(seed)
        widths = rng.uniform(.15, .75, 2); k = rng.uniform(.20, .80, 3)
        m = rng.uniform(.10, .90); theta = rng.uniform(-np.pi / 2, np.pi / 2)
        p = np.r_[widths, theta, k, m]
        normalized = np.r_[(widths - .04) / .96, k, m]
        logits = np.log(normalized / (1 - normalized))
        z = np.r_[logits[:2], theta, logits[2:]]
        rows.append({'seed': seed, 'initial_physical_float64': p.tolist(),
                     'initial_unconstrained_float32': z.astype(np.float32).tolist(),
                     'status': 'not_started', 'completed_updates': 0, 'history': [], 'final': None})
    return rows


def physical(z):
    return torch.cat([.04 + .96 * torch.sigmoid(z[:2]), z[2:3], torch.sigmoid(z[3:])])


def make_tapes(seeds, points):
    u, v, azimuth, identities = [], [], [], []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        arrays = [rng.uniform(size=(points, 1024)).astype(np.float32),
                  rng.uniform(size=(points, 1024)).astype(np.float32),
                  rng.uniform(size=(points, 1, 1)).astype(np.float32)]
        identities.append({'seed': seed, 'points': points, 'specular_samples': 1024,
                           'sha256': hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest(),
                           'array_sha256': [reference.sha_array(a) for a in arrays]})
        for output, value in zip((u, v, azimuth), arrays): output.append(value)
    return RandomInputs(*(torch.from_numpy(np.concatenate(x)) for x in (u, v, azimuth))), identities


class ObservationBatch:
    """Seed-major point batching, with mean over tapes before MSE."""
    def __init__(self, scene, observations, tapes, seeds):
        self.observations = observations; self.n = len(observations); self.seeds = seeds
        n = self.n * seeds
        self.positions = torch.zeros((n, 3), dtype=torch.float32, device='cpu')
        self.normals = torch.tensor([[0., 0., 1.]], dtype=torch.float32).repeat(n, 1)
        self.views = torch.tensor(np.tile([scene['views'][o['view']]['direction'] for o in observations], (seeds, 1)), dtype=torch.float32)
        selected = [scene['lights'][o['light']] for o in observations] * seeds
        self.axes = torch.tensor([l['axis'] if l['axis'] is not None else [0., 0., 1.] for l in selected], dtype=torch.float32)
        self.powers = torch.tensor([l['power'] or 1 for l in selected], dtype=torch.float32)
        self.constant = torch.tensor([l['axis'] is None for l in selected], dtype=torch.bool)
        self.tapes = tapes
        tapes.validate(n, 1024)

    def illumination(self, positions, directions):
        dot = (directions * self.axes[:, None]).sum(-1).clamp(min=0)
        lobe = .1 + .9 * dot ** self.powers[:, None]
        value = torch.where(self.constant[:, None], torch.ones_like(lobe), lobe)
        return value[..., None].expand_as(directions)

    def __call__(self, shader, p):
        n = len(self.positions)
        result = shader.shade(self.positions, self.normals, self.views,
                             p[0:1].reshape(1, 1).expand(n, 1), p[1:2].reshape(1, 1).expand(n, 1),
                             p[3:6].reshape(1, 3).expand(n, 3), p[6:7].reshape(1, 1).expand(n, 1),
                             torch.stack([torch.cos(2 * p[2]), torch.sin(2 * p[2])]).reshape(1, 2).expand(n, 2),
                             self.illumination, self.tapes, is_train=True)
        require(bool(result['frame_orthonormal'].all()), 'Corrected frame unexpectedly invalid')
        return torch.stack([result[k].reshape(self.seeds, self.n, 3).mean(0)
                            for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], dim=1)


def radiance_errors(estimate, target):
    a, b = np.asarray(estimate, np.float64), np.asarray(target, np.float64)
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite matching radiance arrays required')
    delta = a - b
    return {'RMSE': float(np.sqrt(np.mean(delta ** 2))), 'MAE': float(np.mean(abs(delta))),
            'relative_L2': float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-4)),
            'max_absolute': float(np.max(abs(delta)))}


def recovery_errors(p):
    p = np.asarray(p); truth = reference.TRUTH
    gauges = gauge_metrics(p[:3], truth[:3]); dk = p[3:6] - truth[3:6]
    return {**gauges, 'sorted_axis_relative_errors': (abs(np.sort(p[:2]) - np.sort(truth[:2])) / np.sort(truth[:2])).tolist(),
            'orientation_degrees': float(np.rad2deg(gauges['major_axis_error_radians'])) if gauges['orientation_identifiable'] else None,
            'albedo_absolute_errors': abs(dk).tolist(), 'albedo_RMSE': float(np.sqrt(np.mean(dk ** 2))),
            'albedo_relative_L2': float(np.linalg.norm(dk) / max(np.linalg.norm(truth[3:6]), 1e-4)),
            'metallic_absolute_error': float(abs(p[6] - truth[6])),
            'metallic_near_boundary': bool(p[6] <= .01 or p[6] >= .99)}


def assess_success(errors, split_metrics):
    c = SETTINGS['success']
    checks = {'Q': errors['Q_relative_frobenius'] <= c['Q_relative'],
              'orientation': errors['orientation_degrees'] is not None and errors['orientation_degrees'] <= c['orientation_degrees'],
              'albedo': max(errors['albedo_absolute_errors']) <= c['max_albedo_absolute'],
              'metallic': errors['metallic_absolute_error'] <= c['metallic_absolute'],
              'held_out_radiance': split_metrics['held_out']['total']['relative_L2'] <= c['held_out_relative_L2']}
    return {'checks': checks, 'all_passed': bool(all(checks.values()))}


def use_points(budget, phase, points):
    key = phase + '_shader_points'; limit = 120000 if phase == 'fit' else 1680
    require(budget.get(key, 0) + points <= limit, 'Declared shader-point budget exceeded')
    budget[key] = budget.get(key, 0) + points
    budget[phase + '_specular_directions'] = budget.get(phase + '_specular_directions', 0) + points * 1024
    budget[phase + '_diffuse_directions'] = budget.get(phase + '_diffuse_directions', 0) + points * 512
    budget[phase + '_shader_calls'] = budget.get(phase + '_shader_calls', 0) + 1


def fit_start(row, shader, batch, target, budget, deadline, persist):
    """Exactly the predeclared bounded optimizer; no priors/scheduler/clipping."""
    require(row['status'] == 'not_started', 'Implicit continuation is forbidden')
    z = torch.tensor(row['initial_unconstrained_float32'], dtype=torch.float32, device='cpu', requires_grad=True)
    row['initial_effective_physical_float32'] = physical(z).detach().tolist()
    optimizer = torch.optim.Adam([z], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    target = torch.tensor(target, dtype=torch.float32, device='cpu')
    row['status'] = 'optimizing'; persist()
    begin = time.perf_counter()
    try:
        for update in range(1, 251):
            deadline(); optimizer.zero_grad()
            use_points(budget, 'fit', batch.n * batch.seeds)
            prediction = batch(shader, physical(z))
            loss = torch.mean((prediction[:, 2] - target) ** 2)
            require(bool(torch.isfinite(loss)), 'Nonfinite fit objective')
            loss.backward()
            require(z.grad is not None and bool(torch.isfinite(z.grad).all()), 'Missing/nonfinite parameter gradients')
            gradient = float(torch.norm(z.grad)); before = float(torch.norm(z.detach()))
            deadline(); optimizer.step()
            row['completed_updates'] += 1; budget['optimizer_returns'] = budget.get('optimizer_returns', 0) + 1
            require(bool(torch.isfinite(z).all()), 'Nonfinite post-update parameters')
            row['last_physical'] = physical(z).detach().tolist(); row['last_unconstrained'] = z.detach().tolist()
            if update == 1 or update % 10 == 0:
                row['history'].append({'completed_updates': update, 'loss_at_update_count': update - 1,
                                       'fit_MSE_before_update': float(loss.detach()), 'gradient_norm_before_update': gradient,
                                       'unconstrained_norm_before_update': before, 'unconstrained_norm_after_update': float(torch.norm(z.detach())),
                                       'physical_after_update': row['last_physical'], 'elapsed_seconds': time.perf_counter() - begin})
                persist(); print('seed {}: returned {}/250 updates, pre-update MSE={:.8g}'.format(row['seed'], update, float(loss.detach())), flush=True)
        row['final'] = row['last_physical']; row['status'] = 'optimized'
        state = optimizer.state[z]
        row['optimizer_final_state'] = {name: value.detach().tolist() if isinstance(value, torch.Tensor) else value for name, value in state.items()}
        row['optimizer_effective'] = {name: value for name, value in optimizer.param_groups[0].items() if name != 'params'}
        return physical(z).detach()
    except BaseException as exc:
        row.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        row['last_physical'] = physical(z).detach().tolist() if bool(torch.isfinite(z).all()) else None
        row['last_unconstrained'] = z.detach().tolist() if bool(torch.isfinite(z).all()) else None
        raise
    finally:
        row['fit_wall_seconds'] = time.perf_counter() - begin; persist()


def evaluate_start(row, p, shader, batch, target, observations, budget, deadline):
    deadline(); use_points(budget, 'evaluation', batch.n * batch.seeds)
    with torch.no_grad(): predictions = batch(shader, p).numpy().astype(np.float64)
    require(np.isfinite(predictions).all(), 'Nonfinite final evaluation')
    splits = {}
    for split in ('fit', 'held_out', 'relight', 'relight_seen_views', 'relight_unseen_views'):
        mask = [o['split'] == split if split in ('fit', 'held_out', 'relight') else
                o['split'] == 'relight' and (o['view'] < 6 if split == 'relight_seen_views' else o['view'] >= 6) for o in observations]
        splits[split] = {component: radiance_errors(predictions[mask, i], target[mask, i])
                         for i, component in enumerate(('diffuse', 'specular', 'total'))}
    errors = recovery_errors(p.numpy())
    row.update(status='complete', final_predictions=predictions.tolist(),
               prediction_sha256=reference.sha_array(predictions), recovery_errors=errors,
               radiance_errors=splits, success=assess_success(errors, splits))


def stability(rows):
    complete = [r for r in rows if r['status'] == 'complete']
    if not complete: return {'complete_starts': 0, 'parameter_median': None, 'parameter_range': None, 'pairwise_Q_distances': []}
    x = np.array([r['final'] for r in complete]); pairs = []
    for i, a in enumerate(complete):
        for b in complete[i + 1:]:
            pairs.append({'seeds': [a['seed'], b['seed']], 'Q_frobenius': float(np.linalg.norm(roughness_tensor(*a['final'][:3]) - roughness_tensor(*b['final'][:3])))})
    return {'complete_starts': len(complete), 'successful_starts': sum(r['success']['all_passed'] for r in complete),
            'parameter_median': np.median(x, axis=0).tolist(), 'parameter_range': np.ptp(x, axis=0).tolist(),
            'raw_parameter_statistics_are_gauge_dependent': True, 'pairwise_Q_distances': pairs}


def run(report, persist):
    start = time.perf_counter(); cpu_start = time.process_time(); deadline = Deadline()
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    torch_rng = torch.get_rng_state().clone(); numpy_rng = np.random.get_state()
    budget = {'optimizer_returns': 0, 'fit_shader_points': 0, 'evaluation_shader_points': 0,
              'reference_direction_evaluations': 0}; report['budget'] = budget
    report['initializations'] = initializations(); report['gates'] = {'reference': 'not_started', 'identifiability': 'not_started'}
    try:
        persist(); scene = reference.design(); report['scene'] = scene
        for stage in ('geometry', 'material'): baseline(stage, PINS[stage])
        report['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                             'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'selected_environment': {k: os.environ.get(k) for k in REQUIRED_ENV}}
        report['phase'] = 'reference'; persist()
        grids = reference.make_grids()
        budget['reference_grid_directions'] = sum(len(g['directions']) for g in grids)
        budget['reference_precomputed_scalar_lights'] = 7 * budget['reference_grid_directions']
        report['reference_grid_identities'] = [{k: g[k] for k in ('shape', 'directions_sha256', 'weights_sha256', 'lighting_sha256')} for g in grids]
        deadline(); targets = reference.render(reference.TRUTH, scene['observations'], grids, budget, deadline)
        report['targets'] = reference.serial_reference(targets); budget['target_direction_evaluations'] = budget['reference_direction_evaluations']
        report['gates']['reference'] = 'passed' if targets['passed'] else 'failed'; persist()
        if not targets['passed']:
            report.update(status='stopped_reference_gate', reason='Mandatory target convergence failed; no optimization'); return
        report['phase'] = 'identifiability'; persist()
        def progress(step, column):
            print('Jacobian step {} column {}/7 complete'.format(step + 1, column + 1), flush=True)
        report['identifiability'] = reference.identifiability(grids, scene['observations'], budget, deadline, progress)
        budget['jacobian_direction_evaluations'] = budget['reference_direction_evaluations'] - budget['target_direction_evaluations']
        gate = report['identifiability']['passed']; report['gates']['identifiability'] = 'passed' if gate else 'failed'; persist()
        if not gate:
            report.update(status='stopped_identifiability_gate', reason='Mandatory independent Jacobian gate failed; no optimization'); return
        # No optimizer or shader construction before both mandatory gates.
        del grids
        deadline(); shader = MaterialShaderV1('A2', 'atan2'); report['shader'] = shader.identity
        fit_obs = [o for o in scene['observations'] if o['split'] == 'fit']
        fit_tapes, fit_ids = make_tapes(SETTINGS['fit_seeds'], len(fit_obs))
        eval_tapes, eval_ids = make_tapes(SETTINGS['evaluation_seeds'], len(scene['observations']))
        report['tapes'] = {'fit': fit_ids, 'evaluation': eval_ids, 'order': 'seed-major; observations view-major/light-minor'}
        report['fit_target_float32_sha256'] = reference.sha_array(targets['fine'][:24, 2].astype(np.float32))
        report['fine_targets_sha256'] = reference.sha_array(targets['fine'])
        tape_hashes = [reference.sha_array(t.numpy()) for d in (fit_tapes, eval_tapes) for t in (d.u, d.v, d.azimuth)]
        fit_batch = ObservationBatch(scene, fit_obs, fit_tapes, 4)
        eval_batch = ObservationBatch(scene, scene['observations'], eval_tapes, 4)
        for row in report['initializations']:
            report['phase'] = 'optimization'; report['active_seed'] = row['seed']; deadline()
            try:
                persist()
                p = fit_start(row, shader, fit_batch, targets['fine'][:24, 2], budget, deadline, persist)
                report['phase'] = 'evaluation'; persist()
                evaluate_start(row, p, shader, eval_batch, targets['fine'], scene['observations'], budget, deadline); persist()
            except BaseException as exc:
                row.update(status='failed', failure_phase=report['phase'], failure_type=type(exc).__name__, failure=str(exc))
                raise
        require(tape_hashes == [reference.sha_array(t.numpy()) for d in (fit_tapes, eval_tapes) for t in (d.u, d.v, d.azimuth)], 'Fixed sample tapes mutated')
        report.update(status='complete', fixed_tapes_unchanged=True, stability=stability(report['initializations']))
    except BaseException as exc:
        report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc)); raise
    finally:
        report['stability'] = stability(report['initializations'])
        report['global_torch_CPU_RNG_unchanged'] = torch.equal(torch_rng, torch.get_rng_state())
        after = np.random.get_state()
        report['global_numpy_RNG_unchanged'] = numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]
        torch.set_num_threads(old_threads); report['threads_restored'] = torch.get_num_threads() == old_threads
        report['wall_seconds'] = time.perf_counter() - start; report['process_cpu_seconds'] = time.process_time() - cpu_start
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed', failure='Unexpected global RNG consumption')
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in REQUIRED_ENV.items()), 'Explicit hidden-GPU/single-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output, 'Explicit unaliased absolute new output required')
    report = {'schema_version': 1, 'protocol': reference.PROTOCOL, 'milestone': '3C-1', 'base_commit': BASE,
              'status': 'running', 'phase': 'setup', 'settings': SETTINGS, 'truth': reference.TRUTH.tolist(),
              'parameter_order': reference.PARAMETERS, 'protocol_sha256': digest(PROTOCOL_PATH),
              'argv': [sys.executable, *sys.argv],
              'execution_notes_sha256': digest(EXECUTION_PATH),
              'implementation_sha256': {p: digest(ROOT / p) for p in ('experiments/known_material_reference_v1.py',
                  'experiments/run_known_material_recovery_v1.py', 'experiments/material_shader_v1.py',
                  'experiments/material_recovery_metrics_v1.py', 'experiments/anisotropic_estimator_v1.py',
                  'audits/anisotropic_brdf.py')},
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'neural_network_training': False, 'GPU_execution': False, 'historical_artifacts_used': False}
    previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame): raise RuntimeLimit('1800-second external/internal execution boundary')
    with args.output.open('x') as stream:
        def persist(): write_report(stream, report)
        persist()
        try:
            for s in previous: signal.signal(s, stop)
            signal.alarm(1800)
            run(report, persist)
        finally:
            signal.alarm(0)
            for s, handler in previous.items(): signal.signal(s, handler)
            report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); persist()
    print('Final status:', report['status'], flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__': raise SystemExit(main())
