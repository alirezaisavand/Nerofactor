"""One offline archived-Jacobian LM proposal per case; no rendering or updates."""
import argparse
import ast
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

import numpy as np
import torch

from infrastructure.signatures import require
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = '31dd7979256795abd8e0043099c649518aabad35'
PROTOCOL = ROOT/'docs/protocols/MATERIAL_LM_PROPOSAL_GAUGE_V1.md'
OUTPUT = ROOT/'docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json'
SEEDS, ARMS = (101, 102, 103, 104, 105), ('P0', 'P1-U', 'P1-C')
PERM = [1, 0, 2, 3, 4, 5, 6]
LAMBDA_INITIAL, LAMBDA_MIN, LAMBDA_MAX = 1e-2, 1e-8, 1e8
MAX_SOLVES, WALL_SECONDS, MEMORY_BYTES = 15, 300, 2*1024**3
STEP_ATOL, STEP_RTOL, STEP_FLOOR = 1e-16, 2e-4, 1e-16
REP_ATOL, REP_RTOL = 2e-7, 2e-5
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-7, 2e-6
PRED_ATOL, PRED_RTOL, PRED_FLOOR = 1e-30, 2e-4, 1e-30
SOLVE_RTOL, RANK_RTOL = 1e-11, 1e-12
REQUIRED_ENV = {'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
INPUTS = {
    'docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json': 'f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451',
    'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json': '66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f',
    'docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json': '58a47eca9978e32385d94c1b3c1f087b2be0dff15fb095fe95b3e5601bc34070',
    'docs/MATERIAL_GRADIENT_EQUIVARIANCE_V1_RESULTS.md': 'f92e5bfadc1dd38e25dc05c26585cc5787550079b5554963026be0ec8f2d24c8',
    'docs/protocols/MATERIAL_GRADIENT_EQUIVARIANCE_V1.md': '352452eb1699a8203efa6a77a5e9c42101614b1a8104447844eb9dd5960204c3',
    'experiments/material_gradient_equivariance_v1.py': 'b53bd46bf766d0e2203bb74cfc09bddac1680bc7b75a9c8c23972bccb556495c',
    'experiments/material_recovery_optimizer_comparison_v1.py': '4abdf70b1a8c3b309f32561e31bcdb605392d0858bbe2015bf98c1a8f065cb6e',
    'experiments/run_known_material_recovery_v1.py': 'b168dc1a0432a246dd6f2346d7b6150fa6d88b2f816ec062fb5c9674aaab2231',
    'experiments/material_recovery_metrics_v1.py': '316e79b5e06615c9fa46e714f35f515b204c1f175c6c56636a6c4d9235ccabc4',
}


class RuntimeLimit(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, 'Duplicate JSON key')
        result[k] = v
    return result


def read_json(name):
    path = ROOT/name
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024**2, 'Explicit small regular evidence required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUTS[name], 'Admitted evidence SHA-256 mismatch: '+name)
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=strict_pairs, parse_constant=invalid)


def describe(a):
    a = np.asarray(a)
    require(a.dtype.kind in 'fiu' and np.isfinite(a).all(), 'Finite numeric arrays only')
    return dict(values=a.tolist(), dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def read_array(record, shape, dtype):
    require(record['shape'] == list(shape) and record['dtype'] in (dtype, 'torch.'+dtype), 'Admitted array dtype/shape mismatch')
    a = np.asarray(record['values'], dtype=dtype)
    require(a.shape == shape and np.isfinite(a).all() and np.array_equal(a.astype(float), np.asarray(record['values'])) and
            describe(a)['sha256'] == record['sha256'], 'Admitted array values/hash mismatch')
    a.flags.writeable = False
    return a


def terminal_damping(row, outcome_index):
    """Replay only logged controller decisions; never propose, render or optimize."""
    require({'arm','status','budget_limited','termination','iterations','completed_outer_iterations','initial_state',
             'initial_fit_MSE','damping','final_state','final','final_fit_MSE','trajectory'} <= row.keys(), 'Missing damping provenance fields')
    require(row['arm'] == 'LM' and row['status'] == 'complete' and row['budget_limited'] is False and
            row['termination'] == 'damping_saturated', 'Only complete admitted saturation endpoints are authorized')
    records = row['iterations']
    require(0 < len(records) == row['completed_outer_iterations'] <= 80, 'Incomplete terminal iteration sequence')
    z = read_array(row['initial_state']['unconstrained'], (7,), 'float32').tolist()
    physical = read_array(row['initial_state']['physical'], (7,), 'float32').tolist()
    mse, damping, last_accepted = row['initial_fit_MSE'], LAMBDA_INITIAL, None
    require(math.isfinite(mse) and mse >= 0, 'Invalid initial objective')
    ledger = []
    for i, rec in enumerate(records):
        require(rec['outer_iteration'] == i+1 and rec['z'] == z and rec['physical'] == physical and
                rec['training_MSE'] == mse and rec['lambda_before'] == damping, 'Contradictory state/damping linearization history')
        trials = rec['trials']
        require(0 < len(trials) <= 4, 'Missing/incomplete LM trials')
        saturated, accepted = False, False
        for k, t in enumerate(trials):
            require(t['trial'] == k+1 and t['damping'] == damping and math.isfinite(damping) and
                    LAMBDA_MIN <= damping <= LAMBDA_MAX, 'Trial damping order/range contradiction')
            if t['status'] == 'accepted':
                require(k == len(trials)-1 and math.isfinite(t['ratio']) and t['ratio'] > .1 and
                        0 < t['predicted_reduction'] and t['training_MSE'] < mse and
                        math.isclose(t['actual_reduction'], mse-t['training_MSE'], rel_tol=1e-14, abs_tol=0.) and
                        math.isclose(t['ratio'], t['actual_reduction']/t['predicted_reduction'], rel_tol=1e-14), 'Contradictory accepted trial')
                require(len(t['proposed_unconstrained']) == len(t['proposed_physical']) == 7 and
                        np.isfinite(t['proposed_unconstrained']+t['proposed_physical']).all(), 'Invalid accepted saved coordinates')
                z, physical, mse = t['proposed_unconstrained'], t['proposed_physical'], t['training_MSE']
                last_accepted = dict(outer_iteration=i+1, trial=k+1, damping_used=damping,
                    pointer='/outcomes/{}/iterations/{}/trials/{}'.format(outcome_index, i, k))
                damping = min(LAMBDA_MAX, max(LAMBDA_MIN, damping/3 if t['ratio'] > .75 else damping*2 if t['ratio'] < .25 else damping))
                accepted = True
            else:
                require(t['status'] == 'rejected' and isinstance(t.get('reason'), str), 'Unfinished/unattributed trial')
                if 'ratio' in t:
                    require(math.isfinite(t['ratio']) and not (t['training_MSE'] < mse and t['ratio'] > .1), 'Rejected trial satisfies acceptance rule')
                if damping == LAMBDA_MAX:
                    saturated = True
                    require(k == len(trials)-1 and i == len(records)-1, 'History continued after damping saturation')
                else:
                    damping = min(LAMBDA_MAX, max(LAMBDA_MIN, damping*10))
        require(accepted or saturated or len(trials) == 4, 'Outer iteration ended before its declared trial boundary')
        require(rec['lambda_after'] == damping and rec['z_after'] == z and rec['training_MSE_after'] == mse and
                rec['status'] == ('damping_saturated' if saturated else 'complete'), 'Recorded post-iteration state/damping contradicts policy')
        ledger.append(dict(outer_iteration=i+1, lambda_before=rec['lambda_before'], lambda_after=rec['lambda_after'],
            status=rec['status'], trials=[{key: t[key] for key in ('trial','damping','status','reason','ratio','predicted_reduction') if key in t} for t in trials]))
    final_z = read_array(row['final_state']['unconstrained'], (7,), 'float32')
    final_p = read_array(row['final_state']['physical'], (7,), 'float32')
    require(saturated and row['damping'] == damping == LAMBDA_MAX and final_z.tolist() == z and
            final_p.tolist() == physical == row['final'] and mse == row['final_fit_MSE'], 'Terminal damping/state association is ambiguous')
    trajectory = [t for t in row['trajectory'] if 'outer_iteration' in t]
    endpoint = row['trajectory'][-1]
    require(len(trajectory) == len(records) and trajectory[-1]['damping'] == damping and
            trajectory[-1]['unconstrained'] == z and trajectory[-1]['physical'] == physical and
            endpoint['phase'] == 'endpoint' and endpoint['training_MSE'] == mse, 'Terminal trajectory/endpoint contradiction')
    terminal = records[-1]
    return dict(value=damping, outcome_index=outcome_index, terminal_outer_iteration=len(records),
        terminal_trial=terminal['trials'][-1]['trial'], termination=row['termination'],
        terminal_iteration_pointer='/outcomes/{}/iterations/{}'.format(outcome_index, len(records)-1),
        terminal_trial_pointer='/outcomes/{}/iterations/{}/trials/{}'.format(outcome_index, len(records)-1, len(terminal['trials'])-1),
        terminal_state_sha256=row['final_state']['unconstrained']['sha256'], last_accepted=last_accepted,
        controller_replay=ledger, counterfactual=True,
        interpretation='Hypothetical next proposal only: original run already terminated at damping saturation')


def extract_function(name, source, namespace):
    """Compile just one hash-admitted pure function, never import its module."""
    require(digest(ROOT/source) == INPUTS[source], 'Original function source changed')
    tree = ast.parse((ROOT/source).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    require(not node.decorator_list and not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)), 'Unexpected function imports/decorators')
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), source, 'exec'), scope)
    return scope[name]


def physical_function():
    original = extract_function('physical', 'experiments/run_known_material_recovery_v1.py', {'torch': torch})
    def evaluate(z):
        require(np.asarray(z).shape == (7,) and np.asarray(z).dtype == np.float32 and np.isfinite(z).all(), 'Finite CPU float32 proposed coordinates required')
        with torch.no_grad():
            result = original(torch.tensor(np.asarray(z), dtype=torch.float32, device='cpu'))
        require(result.device.type == 'cpu' and result.dtype == torch.float32 and not result.requires_grad and
                bool(torch.isfinite(result).all()), 'Original parameter transform output invalid')
        p = result.numpy()
        require(((p[:2] >= .04) & (p[:2] <= 1)).all() and ((p[3:] >= 0) & (p[3:] <= 1)).all(), 'Invalid original material parameterization')
        return p
    return evaluate


def admit():
    identities = dict(INPUTS)
    for name, sha in identities.items():
        require(digest(ROOT/name) == sha, 'Accepted source/evidence SHA-256 mismatch: '+name)
    history = read_json('docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json')
    gradients = read_json('docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json')
    original = read_json('docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json')
    require(history['status'] == gradients['status'] == original['status'] == 'complete' and
            history['restoration_gate'] == history['initial_objective_gate'] == gradients['mapping_gate'] == 'passed', 'Complete accepted input gates required')
    require(history['settings']['initial_lambda'] == LAMBDA_INITIAL and history['settings']['minimum_lambda'] == LAMBDA_MIN and
            history['settings']['maximum_lambda'] == LAMBDA_MAX and history['settings']['LM_trial_cap'] == 4, 'Original damping policy changed')
    for source in (history, gradients):
        for name, sha in {**source['accepted_identities'], **source['implementation_sha256']}.items():
            require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name) == sha, 'Prior source/evidence changed: '+name)
            identities[name] = sha
    for stage, pin in PINS.items():
        baseline(stage, pin)
    identities['scientific_sources/manifest-v1.json'] = digest(ROOT/'scientific_sources/manifest-v1.json')
    selected = [(i, r) for i, r in enumerate(history['outcomes']) if r['arm'] == 'LM']
    require([r['seed'] for _, r in selected] == list(SEEDS), 'Exactly five original LM endpoints required')
    terminal_ids = {101: (14,1), 102: (17,3), 103: (11,4), 104: (9,2), 105: (14,4)}
    rows = []
    for index, row in selected:
        provenance = terminal_damping(row, index)
        require((provenance['terminal_outer_iteration'], provenance['terminal_trial']) == terminal_ids[row['seed']], 'Reviewed terminal identifiers changed')
        saved = next(s for s in gradients['states'] if s['id'] == 'LM-'+str(row['seed']))
        coordinate = saved['coordinate_reconstruction']['representations']['P0']
        require(saved['status'] == 'complete' and coordinate['valid'] and coordinate['exact_physical_reconstruction'] and
                coordinate['unconstrained']['values'] == row['final_state']['unconstrained']['values'] and
                coordinate['unconstrained']['sha256'] == row['final_state']['unconstrained']['sha256'] and
                saved['fit']['cases']['P0']['forward']['actual_physical_float32']['values'] == row['final'], 'Damping endpoint differs from 3C-7 endpoint')
        rows.append((row, saved, provenance))
    require(original['scene'] == gradients['scene'] == history['scene'] and gradients['shader'] == original['shader'], 'Original target/scene/shader identities differ')
    targets = np.asarray(original['targets']['fine'], np.float64)
    require(describe(targets)['sha256'] == original['fine_targets_sha256'] == gradients['fine_targets_sha256'], 'Original target hash changed')
    target = targets[:24, 2].astype(np.float32)
    require(describe(target)['sha256'] == original['fit_target_float32_sha256'], 'Original fitting target conversion changed')
    return rows, target, identities, gradients['runtime']


def case_inputs(case, coordinate, target):
    require(case['status'] == 'complete' and case['exact_accepted_forward'] and coordinate['valid'] and
            coordinate['exact_physical_reconstruction'] and case['optimizer_coordinates']['valid'], 'Valid exact latent coordinates required')
    z = read_array(coordinate['unconstrained'], (7,), 'float32')
    p = read_array(case['forward']['actual_physical_float32'], (7,), 'float32')
    require(np.array_equal(p, read_array(coordinate['reconstructed_physical'], (7,), 'float32')), 'Physical reconstruction record mismatch')
    d = read_array(coordinate['dp_dz_diagonal_float32'], (7,), 'float32')
    require((d > 0).all() and d[2] == 1., 'Invalid original parameter-transform derivative')
    for component in ('diffuse', 'specular', 'total'):
        jp = read_array(case['physical_jacobians'][component]['array'], (72,7), 'float32')
        jz = read_array(case['optimizer_coordinates']['jacobians'][component]['array'], (72,7), 'float64')
        require(np.array_equal(jz, jp.astype(float)*d.astype(float)), 'Archived physical-to-latent Jacobian conversion mismatch')
    values = read_array(case['forward']['mean_predictions'], (24,3,3), 'float32')
    error = (values[:, 2]-target).astype(np.float32).reshape(72)
    # Match the original float32 residual values. The Jacobian remains the
    # explicitly admitted float64 chain-derived approximation, not historical AD.
    residual32 = error/np.float32(math.sqrt(72))
    residual = residual32.astype(float)
    j = jz.astype(float)/math.sqrt(72)
    mse = case['gradient']['objective_float32']
    require(math.isclose(float(residual@residual), mse, rel_tol=5e-7, abs_tol=1e-12), 'Original normalized residual/MSE convention failed')
    return dict(z=z, physical=p, D_transform=d, J_physical=jp, J_latent=jz,
                J_normalized=j, residual=residual, residual_float32=residual32, mse=mse,
                archived_full_MSE_gradient=read_array(case['optimizer_coordinates']['gradient'], (7,), 'float64'))


def spectra(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(float(s[0])*RANK_RTOL, 1e-300)
    return dict(singular_values=s.tolist(), numerical_rank=int((s > threshold).sum()), rank_relative_threshold=RANK_RTOL,
                condition_number=float(s[0]/s[-1]) if s[-1] > 0 else None,
                condition_reason=None if s[-1] > 0 else 'Singular matrix; no finite condition number')


def system(j, r, damping):
    require(j.shape == (72,7) and r.shape == (72,) and j.dtype == r.dtype == np.float64 and
            np.isfinite(j).all() and np.isfinite(r).all() and math.isfinite(damping) and
            LAMBDA_MIN <= damping <= LAMBDA_MAX, 'Invalid normalized system inputs/damping')
    h, g = j.T@j, j.T@r
    diagonal = h.diagonal()+1e-8
    a = h+damping*np.diag(diagonal)
    require(np.isfinite(h).all() and np.isfinite(g).all() and np.isfinite(a).all(), 'Nonfinite normal system')
    return h, g, diagonal, a


class Budget:
    def __init__(self, report):
        self.start, self.report = time.perf_counter(), report
        report['work'] = dict(normal_equation_solves=0, completed_solves=0, proposed_coordinate_arrays=0,
            proposed_physical_transforms=0, input_physical_reconstruction_checks=0,
            shader_forwards=0, derivative_evaluations=0, optimizer_updates=0, candidate_objective_evaluations=0)

    def check(self):
        if time.perf_counter()-self.start >= WALL_SECONDS:
            raise RuntimeLimit('Single 300-second offline proposal limit; no retry')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= MEMORY_BYTES, '2 GiB peak RSS cap exceeded')

    def solve(self, a, g):
        self.check()
        require(self.report['work']['normal_equation_solves'] < MAX_SOLVES, '15-solve budget exceeded')
        self.report['work']['normal_equation_solves'] += 1
        delta = np.linalg.solve(a, -g)  # The only production normal-equation solve call site.
        self.report['work']['completed_solves'] += 1
        require(delta.shape == (7,) and np.isfinite(delta).all(), 'Invalid solve output; no fallback')
        numerator = float(np.linalg.norm(a@delta+g))
        denominator = float(np.linalg.norm(a, ord='fro')*np.linalg.norm(delta)+np.linalg.norm(g))
        relative = numerator/max(denominator, 1e-300)
        require(relative <= SOLVE_RTOL, 'Linear-solve residual verification failed; no second solve')
        self.check()
        return delta, dict(absolute_L2=numerator, relative_backward_error=relative, denominator=denominator,
                           denominator_floor=1e-300, tolerance=SOLVE_RTOL, passed=True)


def round_candidate(z, delta):
    candidate = (z.astype(float)+delta).astype(np.float32)
    require(np.isfinite(candidate).all(), 'Nonfinite proposed float32 coordinates')
    represented = candidate.astype(float)-z.astype(float)
    return candidate, represented


def predicted(g, h, delta, mse):
    value = float(-2*float(g@delta)-float(delta@(h@delta)))
    finite = math.isfinite(value)
    interpretation_floor = 1e-12+5e-7*abs(mse)
    return dict(value=value if finite else None, finite=finite,
        sign='nonfinite' if not finite else ('positive' if value > 0 else 'negative' if value < 0 else 'zero'),
        negative_zero=bool(finite and value == 0 and np.signbit(value)),
        interpretation_floor=interpretation_floor, too_small_to_interpret=not finite or abs(value) <= interpretation_floor,
        interpretation='Model prediction only; no candidate rendering/acceptance or actual reduction')


def vector_difference(current, base, atol, rtol, floor):
    a, b = np.asarray(current, float), np.asarray(base, float)
    require(a.shape == b.shape and np.isfinite([a,b]).all(), 'Finite matching comparison vectors required')
    d, na, nb = a-b, float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(difference=d.tolist(), absolute_L2=float(np.linalg.norm(d)), relative_L2=float(np.linalg.norm(d)/max(nb, floor)),
        relative_denominator_floor=floor, reference_norm=nb, current_norm=na, reference_below_floor=nb <= floor,
        max_absolute=float(np.max(abs(d))), per_parameter_absolute=abs(d).tolist(),
        cosine=float(np.clip(a@b/(na*nb), -1, 1)) if min(na,nb) > floor else None,
        cosine_reason=None if min(na,nb) > floor else 'One or both norms at/below declared floor',
        atol=atol, rtol=rtol, passed=bool(np.all(abs(d) <= atol+rtol*abs(b))))


def reduction_comparison(current, base):
    if current['value'] is None or base['value'] is None:
        return dict(passed=False, reason='Nonfinite predicted reduction retained; no numerical equivalence claim')
    a, b = current['value'], base['value']
    return dict(signed_difference=a-b, absolute_difference=abs(a-b), relative_difference=(a-b)/max(abs(b),PRED_FLOOR),
        denominator_floor=PRED_FLOOR, reference_below_floor=abs(b) <= PRED_FLOOR,
        passed=abs(a-b) <= PRED_ATOL+PRED_RTOL*abs(b), atol=PRED_ATOL, rtol=PRED_RTOL,
        meaningful=not (current['too_small_to_interpret'] or base['too_small_to_interpret']))


def comparisons(current, base, q_function, offset):
    p1 = read_array(current['proposed_physical'], (7,), 'float32').astype(float)
    p0 = read_array(base['proposed_physical'], (7,), 'float32').astype(float)
    mapped = p1[PERM].copy()
    mapped[2] -= math.pi/2  # Ideal physical gauge, retaining original float32 phase error.
    q1, q0 = q_function(*p1[:3]), q_function(*p0[:3])
    result = dict(mathematical_step=vector_difference(np.asarray(current['delta_float64']['values'])[PERM], base['delta_float64']['values'], STEP_ATOL, STEP_RTOL, STEP_FLOOR),
        represented_step=vector_difference(np.asarray(current['delta_represented']['values'])[PERM], base['delta_represented']['values'], REP_ATOL, REP_RTOL, STEP_FLOOR),
        proposed_physical_mapped=describe(mapped), physical=vector_difference(mapped, p0, PHYSICAL_ATOL, PHYSICAL_RTOL, 1e-12),
        physical_Q=vector_difference(q1.reshape(-1), q0.reshape(-1), PHYSICAL_ATOL, PHYSICAL_RTOL, 1e-12),
        predicted_reduction=reduction_comparison(current['predicted_reduction_represented'],base['predicted_reduction_represented']),
        unrounded_predicted_reduction=reduction_comparison(current['predicted_reduction_float64'],base['predicted_reduction_float64']))
    mapped_z = np.asarray(current['candidate_float32']['values'], float)[PERM].copy()
    mapped_z[2] -= offset
    result['candidate_mapped_by_actual_affine_offset'] = describe(mapped_z)
    result['affine_candidate_difference'] = vector_difference(mapped_z, base['candidate_float32']['values'], REP_ATOL, REP_RTOL, STEP_FLOOR)
    return result


def execute(rows, target, identities, expected_runtime, report, persist):
    start, cpu, threads = time.perf_counter(), time.process_time(), torch.get_num_threads()
    trng, nrng = torch.get_rng_state().clone(), np.random.get_state()
    torch.set_num_threads(1)
    budget = Budget(report)
    try:
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
        require(runtime == expected_runtime and torch.get_default_dtype() == torch.float32, 'Accepted CPU runtime/default dtype changed')
        report['runtime'] = runtime
        require('experiments.material_shader_v1' not in sys.modules and not any(n.startswith(('network.', 'scientific_sources.')) for n in sys.modules), 'Scientific shader modules must never be imported')
        physical = physical_function()
        q_function = extract_function('roughness_tensor', 'experiments/material_recovery_metrics_v1.py', {'np':np, 'require':require})
        report.update(states=[], phase='admission', targets_float32=describe(target), admission_complete=False,
            numerical_conventions=dict(residual='Original float32 RGB subtraction/division, promoted to float64',
                Jacobian='Archived float64 physical-to-latent chain, divided by sqrt(72) in float64 analysis',
                limitation='Not a bitwise historical latent-AD Jacobian or 3C-4 proposal replay', solver='NumPy float64 LAPACK solve, one call per case'))
        inputs = {}
        for old, saved, provenance in rows:
            row = dict(seed=old['seed'], status='admitted', damping_provenance=provenance, cases={})
            report['states'].append(row)
            coordinate = saved['coordinate_reconstruction']['representations']
            p0z, p1z = [read_array(coordinate[label]['unconstrained'], (7,), 'float32') for label in ('P0','P1')]
            require(np.array_equal(p1z[[0,1,3,4,5,6]],p0z[[1,0,3,4,5,6]]), 'Latent permutation differs')
            offset = float(p1z[2])-float(p0z[2])
            require(abs(offset-math.pi/2) <= 2e-7, 'Admitted angle translation differs from float32 pi/2 gauge')
            row.update(actual_orientation_offset=offset, ideal_orientation_offset=math.pi/2,
                       archived_mapping=saved['mapping'], original_termination=old['termination'])
            for label in ARMS:
                coord = coordinate['P0' if label=='P0' else 'P1']
                data = case_inputs(saved['fit']['cases'][label], coord, target)
                if label != 'P1-C':
                    actual = physical(data['z'])
                    budget.report['work']['input_physical_reconstruction_checks'] += 1
                    require(np.array_equal(actual, data['physical']), 'Original physical transform reconstruction failed')
                else:
                    require(np.array_equal(data['z'], inputs[(old['seed'],'P1-U')]['z']) and
                            np.array_equal(data['physical'], inputs[(old['seed'],'P1-U')]['physical']), 'P1-U/P1-C material differs')
                inputs[(old['seed'],label)] = data
            require(np.array_equal(inputs[(old['seed'],'P1-C')]['D_transform'][PERM], inputs[(old['seed'],'P0')]['D_transform']), 'Original transform diagonal does not permute')
            persist()
            budget.check()
        report.update(admission_complete=True, phase='proposals')
        input_hashes={key: {k:describe(v)['sha256'] for k,v in value.items() if isinstance(v,np.ndarray)} for key,value in inputs.items()}
        for row in report['states']:
            report['active_seed'] = row['seed']
            for label in ARMS:
                report['active_arm'] = label
                data = inputs[(row['seed'],label)]
                h,g,d,a = system(data['J_normalized'],data['residual'],row['damping_provenance']['value'])
                case = row['cases'][label] = dict(status='system_ready', original_unconstrained=describe(data['z']), original_physical=describe(data['physical']),
                    dp_dz=describe(data['D_transform']), physical_total_RGB_J=describe(data['J_physical']),
                    chain_derived_latent_RGB_J=describe(data['J_latent']), normalized_residual_J=describe(data['J_normalized']),
                    normalized_residual_float32=describe(data['residual_float32']), normalized_residual=describe(data['residual']),
                    residual_norm=float(np.linalg.norm(data['residual'])), archived_objective_float32=data['mse'],
                    H=describe(h), g=describe(g), twice_g=describe(2*g), archived_full_MSE_latent_gradient=describe(data['archived_full_MSE_gradient']),
                    damping_diagonal=describe(d), damped_normal=describe(a), normal_spectrum=spectra(h), damped_spectrum=spectra(a),
                    residual_J_spectrum=spectra(data['J_normalized']), matrices_finite=True, historical_damping=row['damping_provenance']['value'])
                persist()
                delta, residual_check = budget.solve(a,g)
                candidate, represented = round_candidate(data['z'],delta)
                budget.report['work']['proposed_coordinate_arrays'] += 1
                case.update(status='candidate_ready',delta_float64=describe(delta),candidate_float32=describe(candidate),
                            delta_represented=describe(represented),linear_solve_residual=residual_check)
                persist()
                proposed = physical(candidate)
                budget.report['work']['proposed_physical_transforms'] += 1
                case.update(status='complete', delta_float64=describe(delta), candidate_float32=describe(candidate), delta_represented=describe(represented),
                    proposed_physical=describe(proposed), original_Q=describe(q_function(*data['physical'][:3].astype(float))), proposed_Q=describe(q_function(*proposed[:3].astype(float))),
                    predicted_reduction_float64=predicted(g,h,delta,data['mse']), predicted_reduction_represented=predicted(g,h,represented,data['mse']),
                    linear_solve_residual=residual_check, parameterization_finite_valid=True,
                    represented_zero=bool(np.all(represented==0)), unchanged_coordinate_count=int(np.count_nonzero(represented==0)),
                    rounding_error=describe(represented-delta), max_coordinate_rounding_error=float(np.max(abs(represented-delta))),
                    candidate_applied=False, actual_reduction=None, accepted_or_rejected=None)
                persist()
            row['comparisons'] = {label+'_to_P0': comparisons(row['cases'][label],row['cases']['P0'],q_function,row['actual_orientation_offset']) for label in ARMS[1:]}
            row['status']='complete'
            persist()
            print('Offline proposal comparison complete:',row['seed'],flush=True)
        require(report['work']['normal_equation_solves']==report['work']['completed_solves']==15 and
                report['work']['proposed_coordinate_arrays']==report['work']['proposed_physical_transforms']==15 and
                report['work']['input_physical_reconstruction_checks']==10, 'Exact offline work accounting failed')
        require(input_hashes=={key:{k:describe(v)['sha256'] for k,v in value.items() if isinstance(v,np.ndarray)} for key,value in inputs.items()}, 'Archived input arrays mutated')
        require('experiments.material_shader_v1' not in sys.modules, 'Unexpected shader import')
        report.update(status='complete',phase='finished',archived_input_arrays_unchanged=True,shader_modules_imported=False)
    except BaseException as exc:
        report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
        raise
    finally:
        after=np.random.get_state()
        report['global_torch_CPU_RNG_unchanged']=torch.equal(trng,torch.get_rng_state())
        report['global_numpy_RNG_unchanged']=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed',failure='Unexpected global RNG use')
        torch.set_num_threads(threads)
        report.update(threads_restored=torch.get_num_threads()==threads,wall_seconds=time.perf_counter()-start,
            CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        persist()


def persist_report(stream, report):
    payload=json.dumps(report,indent=2,allow_nan=False)+'\n'
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(all(os.environ.get(k)==v for k,v in REQUIRED_ENV.items()),'Hidden-GPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve()==args.output==OUTPUT,'Fresh explicit versioned output required')
    report=dict(schema_version=1,protocol='material-lm-proposal-gauge-v1',milestone='3C-8',base_commit=BASE,status='running',
        protocol_sha256=digest(PROTOCOL),implementation_sha256={name:digest(ROOT/name) for name in
            ('experiments/material_lm_proposal_gauge_v1.py','tests/scientific/test_material_lm_proposal_gauge_v1.py')},
        settings=dict(max_solves=MAX_SOLVES,wall_seconds=WALL_SECONDS,peak_RSS_bytes=MEMORY_BYTES,data_allocation_soft_cap=MEMORY_BYTES,
            step_atol=STEP_ATOL,step_rtol=STEP_RTOL,step_relative_floor=STEP_FLOOR,represented_atol=REP_ATOL,represented_rtol=REP_RTOL,
            physical_Q_atol=PHYSICAL_ATOL,physical_Q_rtol=PHYSICAL_RTOL,prediction_atol=PRED_ATOL,prediction_rtol=PRED_RTOL,
            prediction_relative_floor=PRED_FLOOR,solve_backward_tolerance=SOLVE_RTOL,rank_relative_threshold=RANK_RTOL),
        optimizer_execution=False,parameter_updates=False,shader_execution=False,derivative_execution=False,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=[sys.executable,*sys.argv])
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    old_limit=resource.getrlimit(resource.RLIMIT_DATA)
    def stop(signum,frame):
        raise RuntimeLimit('Single 300-second proposal audit terminated; no retry')
    with args.output.open('x') as stream:
        def persist():persist_report(stream,report)
        persist()
        try:
            for s in handlers:signal.signal(s,stop)
            signal.alarm(WALL_SECONDS)
            cap=min(MEMORY_BYTES,old_limit[0]) if old_limit[0]>=0 else MEMORY_BYTES
            if old_limit[1]>=0:cap=min(cap,old_limit[1])
            resource.setrlimit(resource.RLIMIT_DATA,(cap,old_limit[1]))
            report['effective_data_allocation_soft_limit']=cap
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= MEMORY_BYTES,'2 GiB RSS cap before admission')
            rows,target,identities,runtime=admit()
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= MEMORY_BYTES,'2 GiB RSS cap after admission')
            report['accepted_identities']=identities
            execute(rows,target,identities,runtime,report,persist)
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
            raise
        finally:
            resource.setrlimit(resource.RLIMIT_DATA,old_limit)
            report['memory_limit_restored']=resource.getrlimit(resource.RLIMIT_DATA)==old_limit
            signal.alarm(0)
            for s,handler in handlers.items():signal.signal(s,handler)
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
