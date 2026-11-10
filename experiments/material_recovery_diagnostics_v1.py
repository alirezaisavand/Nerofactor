"""ONE bounded CPU diagnostic: no optimizer construction or parameter updates."""
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

from experiments import material_recovery_trajectories_v1 as offline
from experiments import known_material_reference_v1 as ref
from experiments import run_known_material_recovery_v1 as previous
from experiments.material_shader_v1 import MaterialShaderV1
from experiments.material_recovery_metrics_v1 import roughness_tensor
from infrastructure.signatures import require, write_report
from infrastructure.stage_sources import baseline, PINS

ROOT = offline.ROOT
PROTOCOL = ROOT / 'docs/protocols/MATERIAL_RECOVERY_DIAGNOSTICS_V1.md'
OFFLINE_SHA = '4f69828f70d0f8d99063d2ffd10169aa991c312b8e930dbb558d7366d2c2693e'
NEAR_ZERO = 1e-12
MAX_REFERENCE = 570163200
MAX_FIT_POINTS = 23040
MAX_EVAL_POINTS = 336


class ReferenceFailure(RuntimeError): pass


def alignment(a, b):
    a, b = np.asarray(a), np.asarray(b); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {'absolute_difference': float(np.linalg.norm(a - b)), 'norm_a': float(na), 'norm_b': float(nb),
            'cosine': float(np.clip(a @ b / (na * nb), -1, 1)) if min(na, nb) > NEAR_ZERO else None,
            'relative_difference_to_b': float(np.linalg.norm(a - b) / nb) if nb > NEAR_ZERO else None,
            'near_zero_threshold': NEAR_ZERO}


def chain_diagonal(p):
    p = np.asarray(p); q = (p[:2] - .04) / .96
    return np.r_[.96 * q * (1 - q), 1., p[3:] * (1 - p[3:])]


def inverse_coordinates(p):
    p = np.asarray(p); v = np.r_[(p[:2] - .04) / .96, p[3:]]
    logits = np.log(v / (1 - v))
    return np.r_[logits[:2], p[2], logits[2:]].astype(np.float32)


def mask_summary(mask):
    mask = np.asarray(mask, bool)
    return {'valid_counts': mask.sum(-1).reshape(4, -1).tolist(),
            'sha256_packbits': hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
            'total_valid': int(mask.sum()), 'total_rejected': int(mask.size - mask.sum())}


def transitions(a, b):
    require(a.shape == b.shape, 'Common-tape mask identities differ')
    return {'flipped_total': int(np.count_nonzero(a != b)),
            'valid_to_rejected': int(np.count_nonzero(a & ~b)),
            'rejected_to_valid': int(np.count_nonzero(~a & b)),
            'flips_per_tape_observation': np.count_nonzero(a != b, axis=-1).reshape(4, -1).tolist()}


def objective_gradient(prediction, target, normalized_jacobian):
    residual = (np.asarray(prediction) - np.asarray(target)).reshape(-1)
    j = np.asarray(normalized_jacobian) / ref.SCALES
    return 2 * j.T @ residual / residual.size


def residual_report(per_tape, targets, observations, source_mean=None):
    x = np.asarray(per_tape, np.float64); target = np.asarray(targets)
    require(x.shape == (4, len(observations), 3, 3) and target.shape == x.shape[1:], 'Four tapes and matching component/RGB targets required')
    mean = x.mean(0) if source_mean is None else np.asarray(source_mean, np.float64)
    require(mean.shape == target.shape and np.isfinite(mean).all(), 'Matching finite source mean required')
    error = x - target; groups = {}
    for label, mask in [('all', [True] * len(observations)), ('constant', [o['light'] == 0 for o in observations]),
                        ('directional', [o['light'] != 0 for o in observations])]:
        if not any(mask): continue
        groups[label] = {c: previous.radiance_errors(mean[mask, i], target[mask, i]) for i, c in enumerate(('diffuse', 'specular', 'total'))}
    return {'per_tape_predictions': x.tolist(), 'mean_predictions': mean.tolist(),
            'per_observation_mean_residual': (mean - target).tolist(),
            'per_observation_tape_RMSE': np.sqrt(np.mean(error * error, axis=(0, 3))).tolist(),
            'per_observation_mean_RMSE_RGB': np.sqrt(np.mean((mean - target) ** 2, axis=-1)).tolist(),
            'per_observation_standard_error_RGB': (x.std(axis=0, ddof=1) / 2).tolist(),
            'groups': groups, 'individual_tape_MSE': [float(np.mean((v[:, 2] - target[:, 2]) ** 2)) for v in x],
            'mean_precision': 'source Torch float32' if source_mean is not None else 'NumPy float64 fixture mean',
            'scope': 'Fixed four-tape mean error; not a population-bias estimate or proof of unbiasedness'}


class Diagnostic:
    def __init__(self, evidence, report, persist):
        self.evidence, self.report, self.persist = evidence, report, persist
        self.deadline = previous.Deadline(); self.budget = report['budget']
        scene = evidence['scene']; self.fit_obs = [o for o in scene['observations'] if o['split'] == 'fit']
        fit_tapes, fit_ids = previous.make_tapes(list(range(4)), 24)
        eval_tapes, eval_ids = previous.make_tapes(list(range(1000, 1004)), 84)
        require(fit_ids == evidence['tapes']['fit'] and eval_ids == evidence['tapes']['evaluation'], 'Frozen 3C-1 tape hash mismatch')
        self.tapes = [fit_tapes, eval_tapes]
        self.tape_hashes = [ref.sha_array(t.numpy()) for d in self.tapes for t in (d.u, d.v, d.azimuth)]
        report['tapes'] = {'fit': fit_ids, 'evaluation': eval_ids}
        self.fit_batch = previous.ObservationBatch(scene, self.fit_obs, fit_tapes, 4)
        self.eval_batch = previous.ObservationBatch(scene, scene['observations'], eval_tapes, 4)
        self.targets = np.asarray(evidence['targets']['fine']); self.target = torch.tensor(self.targets[:24, 2], dtype=torch.float32)
        self.shader = MaterialShaderV1('A2', 'atan2'); report['shader'] = self.shader.identity
        self.cache = {}

    def forward(self, p, evaluation=False):
        self.deadline(); batch = self.eval_batch if evaluation else self.fit_batch
        n = len(batch.positions); key = 'evaluation_points' if evaluation else 'fit_points'
        limit = MAX_EVAL_POINTS if evaluation else MAX_FIT_POINTS
        require(self.budget[key] + n <= limit, 'Predeclared shader point cap exceeded')
        self.budget[key] += n; self.budget['shader_calls'] += 1
        self.budget['specular_directions'] += n * 1024; self.budget['diffuse_directions'] += n * 512
        r = self.shader.shade(batch.positions, batch.normals, batch.views,
                              p[:1].reshape(1, 1).expand(n, 1), p[1:2].reshape(1, 1).expand(n, 1),
                              p[3:6].reshape(1, 3).expand(n, 3), p[6:7].reshape(1, 1).expand(n, 1),
                              torch.stack([torch.cos(2*p[2]), torch.sin(2*p[2])]).reshape(1, 2).expand(n, 2),
                              batch.illumination, batch.tapes, is_train=True)
        require(bool(r['frame_orthonormal'].all()), 'Accepted corrected frame unexpectedly invalid')
        predictions = torch.stack([r[k].reshape(4, batch.n, 3) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], dim=2)
        mask = (r['specular_directions'][:, :, 2] > 0).detach().numpy()
        return predictions, mask

    def state(self, identifier, physical, unconstrained=None):
        z = None if unconstrained is None else torch.tensor(unconstrained, dtype=torch.float32, requires_grad=True)
        p = torch.tensor(physical, dtype=torch.float32, requires_grad=True) if z is None else previous.physical(z)
        if identifier != 'truth_latent_roundtrip':
            require(np.array_equal(p.detach().numpy(), np.asarray(physical, np.float32)), 'Saved terminal coordinate/physical mismatch')
        values, mask = self.forward(p); mean = values.mean(0)
        loss = torch.mean((mean[:, 2] - self.target) ** 2)
        gp, = torch.autograd.grad(loss, p, retain_graph=z is not None)
        chain = chain_diagonal(p.detach().numpy())
        gz = torch.autograd.grad(loss, z)[0].detach().numpy() if z is not None else gp.detach().numpy() * chain
        require(np.isfinite(gz).all() and bool(torch.isfinite(gp).all()), 'Nonfinite diagnostic gradient')
        row = {'id': identifier, 'nominal_physical': list(physical), 'effective_physical_float32': p.detach().tolist(),
               'objective_float32': float(loss.detach()), 'g_physical_autograd': gp.detach().tolist(),
               'g_unconstrained': gz.tolist(), 'unconstrained_gradient_kind': 'autograd' if z is not None else 'chain_converted_at_direct_physical_point',
               'dp_dz': chain.tolist(), 'gradient_conversion': alignment(gz / chain, gp.detach().numpy()) if np.all(abs(chain) > 1e-8) else None,
               'masks': mask_summary(mask), 'forward': residual_report(values.detach().numpy(), self.targets[:len(self.fit_obs)], self.fit_obs, mean.detach().numpy())}
        return row, mask

    def local_finite_differences(self, row, mask):
        p = np.asarray(row['nominal_physical']); jacobians = np.empty((2, self.target.numel(), 7)); gradients = []
        probes = []
        row['finite_difference_probes'] = probes; row['finite_objective_FD'] = gradients
        for i, step in enumerate(ref.FD_STEPS):
            objective = np.empty(7)
            for j, scale in enumerate(ref.SCALES):
                endpoints = []
                for sign in (-1, 1):
                    value = p.copy(); value[j] += sign * step * scale
                    with torch.no_grad(): x, m = self.forward(torch.tensor(value, dtype=torch.float32))
                    mean = x.mean(0); loss = torch.mean((mean[:, 2] - self.target) ** 2)
                    endpoints.append((mean[:, 2].numpy().astype(np.float64), float(loss)))
                    probes.append({'step': step, 'parameter': ref.PARAMETERS[j], 'sign': sign,
                                   'requested_physical': value.tolist(), 'represented_float32': value.astype(np.float32).tolist(),
                                   'objective': float(loss), 'mask_transition_from_center': transitions(mask, m), 'masks': mask_summary(m)})
                jacobians[i, :, j] = ((endpoints[1][0] - endpoints[0][0]) / (2 * step)).reshape(-1)
                objective[j] = (endpoints[1][1] - endpoints[0][1]) / (2 * step * scale)
            gradients.append({'step': step, 'g_physical_FD': objective.tolist(),
                              'autograd_vs_FD': alignment(row['g_physical_autograd'], objective)})
            self.persist()
        row['finite_difference_probes'] = probes; row['finite_objective_FD'] = gradients
        row['finite_sensitivity_normalized_jacobians'] = jacobians.tolist()
        row['finite_sensitivity_spectra'] = [np.linalg.svd(j, compute_uv=False).tolist() for j in jacobians]

    def independent(self, p, grids, store):
        expected = 6 * sum(len(g['directions']) for g in grids)
        require(self.budget['new_reference_directions'] + expected <= MAX_REFERENCE, 'New independent reference cap exceeded')
        self.deadline(); local = {}
        try:
            value = ref.render(p, self.fit_obs, grids, local, self.deadline)
        finally:
            self.budget['new_reference_directions'] += local.get('reference_direction_evaluations', 0)
        serial = ref.serial_reference(value); store.append({'physical_parameters': np.asarray(p).tolist(), **serial})
        if not value['passed']:
            self.persist(); raise ReferenceFailure('New terminal reference/perturbation failed unchanged convergence criteria')
        return value

    def continuous_terminal(self, p, grids, row):
        row['independent_references'] = []
        values = row['independent_references']; center = self.independent(p, grids, values)
        matrices = np.empty((2, 2, self.target.numel(), 7))
        for i, step in enumerate(ref.FD_STEPS):
            for j, scale in enumerate(ref.SCALES):
                endpoints = []
                for sign in (-1, 1):
                    changed = np.asarray(p).copy(); changed[j] += sign * step * scale
                    endpoints.append(self.independent(changed, grids, values))
                for grid, name in enumerate(('coarse', 'fine')):
                    matrices[i, grid, :, j] = ((endpoints[1][name][:, 2] - endpoints[0][name][:, 2]) / (2 * step)).reshape(-1)
                self.persist()
                print(row['id'], 'independent FD', i + 1, j + 1, flush=True)
        return center, matrices


def compare_continuous(row, center, matrices, target):
    descriptions = []
    for i, step in enumerate(ref.FD_STEPS):
        for grid, name in enumerate(('coarse', 'fine')):
            gradient = objective_gradient(center[name][:, 2], target[:, 2], matrices[i, grid])
            gz = gradient * row['dp_dz']
            descriptions.append({'step': step, 'grid': name, 'g_physical': gradient.tolist(), 'g_unconstrained': gz.tolist(),
                                 'finite_vs_continuous_physical': alignment(row['g_physical_autograd'], gradient),
                                 'finite_vs_continuous_unconstrained': alignment(row['g_unconstrained'], gz),
                                 'continuous_directional_derivative_along_negative_finite_gz': float(-np.asarray(row['g_unconstrained']) @ gz)})
    row['continuous_gradients'] = descriptions
    row['continuous_objectives'] = {name: float(np.mean((center[name][:, 2] - target[:, 2]) ** 2)) for name in ('coarse', 'fine')}
    row['continuous_reference_conditioning'] = ref.jacobian_gate(matrices)
    row['continuous_sensitivity_normalized_jacobians'] = matrices.tolist()
    row['continuous_center_components'] = {name: center[name].tolist() for name in ('coarse', 'fine')}


def branch_cases(truth, terminal103):
    cases = []
    for label, background in [('truth_material', truth), ('seed103_material', terminal103)]:
        for center in (-np.pi/2, np.pi/2):
            for offset in (-ref.FD_STEPS[0]*np.pi, -ref.FD_STEPS[1]*np.pi, 0., ref.FD_STEPS[1]*np.pi, ref.FD_STEPS[0]*np.pi):
                p = np.asarray(background).copy(); p[2] = center + offset
                cases.append({'id': '{}:center={}:offset={}'.format(label, center, offset), 'background': label,
                              'center': center, 'offset': offset, 'physical': p.tolist()})
    return cases


def execute(evidence, trajectory, report, persist):
    start = time.perf_counter(); cpu = time.process_time(); old_threads = torch.get_num_threads()
    torch.set_num_threads(1); trng = torch.get_rng_state().clone(); nrng = np.random.get_state()
    report['budget'] = {'optimizer_calls': 0, 'shader_calls': 0, 'fit_points': 0, 'evaluation_points': 0,
                        'specular_directions': 0, 'diffuse_directions': 0, 'new_reference_directions': 0}
    report['states'] = []; report['trajectory_replays'] = []; report['branch_probes'] = []
    try:
        for stage in ('geometry', 'material'): baseline(stage, PINS[stage])
        report['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                             'device': 'cpu', 'threads': torch.get_num_threads(), 'environment': {k: os.environ.get(k) for k in previous.REQUIRED_ENV}}
        d = Diagnostic(evidence, report, persist); report['phase'] = 'finite_baselines_and_derivatives'; persist()
        selected = [('truth', evidence['truth'], None)] + [('seed-{}'.format(s['seed']), s['final'], s['last_unconstrained']) for s in evidence['initializations']]
        masks = {}
        for label, p, z in selected:
            row, mask = d.state(label, p, z); report['states'].append(row); masks[label] = mask
            if label.startswith('seed-'): d.cache[label + '/u-250'] = (row, mask)
            d.local_finite_differences(row, mask); persist(); print('Finite baseline/FD complete:', label, flush=True)
        row, _ = d.state('truth_latent_roundtrip', evidence['truth'], inverse_coordinates(evidence['truth']))
        row['roundtrip_physical_difference'] = (np.array(row['effective_physical_float32']) - evidence['truth']).tolist()
        report['truth_latent_roundtrip'] = row
        with torch.no_grad(): x, _ = d.forward(torch.tensor(evidence['truth'], dtype=torch.float32), evaluation=True)
        report['truth_evaluation_tapes'] = residual_report(x.numpy(), d.targets, evidence['scene']['observations'], x.mean(0).numpy()); persist()
        report['phase'] = 'new_terminal_references'; grids = ref.make_grids()
        report['grid_identities'] = [{k: g[k] for k in ('shape', 'directions_sha256', 'weights_sha256', 'lighting_sha256')} for g in grids]
        require(report['grid_identities'] == evidence['reference_grid_identities'], 'Original quadrature grid/light hashes differ')
        report['budget']['new_grid_directions'] = sum(len(g['directions']) for g in grids)
        report['budget']['precomputed_scalar_lights'] = 7 * report['budget']['new_grid_directions']
        truth = report['states'][0]
        center = {name: np.asarray(evidence['targets'][name])[:24] for name in ('coarse', 'fine')}
        compare_continuous(truth, center, np.asarray(evidence['identifiability']['jacobians']), d.targets[:24])
        truth['independent_source'] = 'Reused admitted 3C-1 targets and Jacobians; no truth reference regenerated'
        for row in report['states'][1:]:
            center, matrices = d.continuous_terminal(row['nominal_physical'], grids, row)
            compare_continuous(row, center, matrices, d.targets[:24]); persist()
        del grids
        report['phase'] = 'saved_trajectory_mask_replays'; last = {}
        for state in trajectory['selected_replay_states']:
            key = state['id']; seed = key.split('/')[0]
            if key in d.cache:
                row, mask = d.cache[key]; reused = True
                prediction = np.asarray(row['forward']['mean_predictions']); objective = row['objective_float32']
            else:
                with torch.no_grad(): x, mask = d.forward(torch.tensor(state['physical'], dtype=torch.float32))
                prediction = x.mean(0).numpy().astype(np.float64)
                objective = float(torch.mean((x.mean(0)[:, 2] - d.target) ** 2)); reused = False
            item = {**state, 'objective': objective, 'masks': mask_summary(mask), 'reused_terminal_baseline': reused}
            if seed in last:
                old_item, old_mask, old_prediction = last[seed]
                item['transition'] = {'from_id': old_item['id'], **transitions(old_mask, mask),
                                      'mean_radiance_delta_RMSE': float(np.sqrt(np.mean((prediction[:, 2] - old_prediction[:, 2]) ** 2)))}
            report['trajectory_replays'].append(item); last[seed] = (item, mask, prediction)
        persist(); report['phase'] = 'branch_probes'; branch_data = []
        for case in branch_cases(evidence['truth'], evidence['initializations'][2]['final']):
            represented = torch.tensor(case['physical'], dtype=torch.float32)
            with torch.no_grad(): x, mask = d.forward(represented)
            prediction = x.mean(0).numpy().astype(np.float64)
            item = {**case, 'masks': mask_summary(mask), 'mean_predictions': prediction.tolist(),
                    'represented_float32': represented.tolist(),
                    'encoding_float32': [float(torch.cos(2*represented[2])), float(torch.sin(2*represented[2]))],
                    'Q': roughness_tensor(*case['physical'][:3]).tolist()}
            report['branch_probes'].append(item); branch_data.append((item, mask, prediction))
        comparisons = []
        wi = ref.reference.unit(np.array([[.2, .4, 1.], [-.4, .1, 1.], [.6, -.3, 1.]])); wo = ref.reference.unit([.3, -.1, 1.])
        for background in ('truth_material', 'seed103_material'):
            for center in (-np.pi/2, np.pi/2):
                for step in ref.FD_STEPS:
                    left, right = [next(v for v in branch_data if v[0]['background'] == background and v[0]['center'] == center and v[0]['offset'] == sign*step*np.pi) for sign in (-1, 1)]
                    a, ma, pa = left; b, mb, pb = right
                    p = a['physical']; q = b['physical']
                    k1 = ref.reference.kernel(wi, wo, p[0], p[1], p[2], .1); k2 = ref.reference.kernel(wi, wo, q[0], q[1], q[2], .1)
                    kp = ref.reference.kernel(wi, wo, p[0], p[1], p[2]+np.pi, .1)
                    comparisons.append({'background': background, 'center': center, 'step': step,
                                        'radiance_jump_RMSE': float(np.sqrt(np.mean((pb[:, 2]-pa[:, 2])**2))),
                                        'Q_jump_frobenius': float(np.linalg.norm(np.array(b['Q'])-a['Q'])),
                                        'pointwise_kernel_jump_max': float(np.max(abs(k2-k1))),
                                        'pointwise_pi_periodicity_error': float(np.max(abs(kp-k1))),
                                        'mask_transition': transitions(ma, mb),
                                        'scope': 'Analytical pointwise physical law, not a new hemisphere reference integration'})
        report['branch_comparisons'] = comparisons
        require(d.tape_hashes == [ref.sha_array(t.numpy()) for tape in d.tapes for t in (tape.u, tape.v, tape.azimuth)], 'Frozen tapes mutated')
        report.update(status='complete', tapes_unchanged=True)
    except ReferenceFailure as exc:
        report.update(status='stopped_reference_gate', error=str(exc))
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        after = np.random.get_state()
        report['global_RNG_unchanged'] = bool(torch.equal(trng, torch.get_rng_state()) and nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:])
        if not report['global_RNG_unchanged']: report.update(status='failed', error='Global RNG changed')
        torch.set_num_threads(old_threads); report['threads_restored'] = torch.get_num_threads() == old_threads
        report['wall_seconds'] = time.perf_counter() - start; report['CPU_seconds'] = time.process_time() - cpu; persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--offline', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in previous.REQUIRED_ENV.items()), 'Explicit hidden-GPU/single-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output, 'Explicit absolute fresh output required')
    evidence, identities = offline.admit(args.input, args.sha256)
    require(offline.digest(args.offline) == OFFLINE_SHA, 'Offline analysis identity mismatch')
    trajectory = json.loads(args.offline.read_text()); require(trajectory['input_sha256'] == args.sha256, 'Offline/input mismatch')
    require(trajectory['implementation_sha256'] == offline.digest(ROOT/'experiments/material_recovery_trajectories_v1.py') and
            trajectory['source_identities'] == identities, 'Offline source identities differ')
    report = {'schema_version': 1, 'protocol': 'material-recovery-diagnostics-v1', 'base_commit': offline.BASE,
              'status': 'running', 'input_path': str(args.input), 'input_sha256': args.sha256, 'accepted_source_identities': identities,
              'offline_sha256': OFFLINE_SHA, 'protocol_sha256': offline.digest(PROTOCOL),
              'implementation_sha256': {p: offline.digest(ROOT/p) for p in ('experiments/material_recovery_diagnostics_v1.py', 'experiments/material_recovery_trajectories_v1.py')},
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': [sys.executable, *sys.argv],
              'optimization': False, 'GPU_execution': False, 'new_initialization_search': False}
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame): raise previous.RuntimeLimit('1800-second diagnostic boundary')
    with args.output.open('x') as stream:
        def persist(): write_report(stream, report)
        persist()
        try:
            for s in previous_handlers: signal.signal(s, stop)
            signal.alarm(1800); execute(evidence, trajectory, report, persist)
        finally:
            signal.alarm(0)
            for s, handler in previous_handlers.items(): signal.signal(s, handler)
            report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); persist()
    print('Diagnostic status:', report['status'])
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__': raise SystemExit(main())
