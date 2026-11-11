"""One bounded CPU physical/optimizer-coordinate derivative audit; no updates."""
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

from experiments import material_sample_coupling_v1 as coupling
from experiments import material_recovery_optimizer_comparison_v1 as derivative
from experiments import material_recovery_diagnostics_v1 as diagnostics
from experiments import run_known_material_recovery_v1 as original
from experiments import known_material_reference_v1 as reference
from experiments.material_shader_v1 import MaterialShaderV1
from infrastructure.signatures import require

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f5a3d714fd57bfef02baf6ca47b15ab69927e7ea'
PROTOCOL = ROOT/'docs/protocols/MATERIAL_GRADIENT_EQUIVARIANCE_V1.md'
OUTPUT = ROOT/'docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json'
MAX_CALLS, MAX_POINTS, MAX_JVP, WALL_SECONDS = 330, 54720, 210, 1800
RSS_BYTES, ADDRESS_BYTES = 4*1024**3, 8*1024**3
LIMITS = {'baseline': 30, 'jvp': 210, 'direct_gradient': 18, 'finite_difference': 72}
PERMUTATION = [1, 0, 2, 3, 4, 5, 6]
PARAMETERS = ['mx', 'my', 'theta', 'kR', 'kG', 'kB', 'metallic']
COMPONENTS = ['diffuse', 'specular', 'total']
J_ATOL, J_RTOL, J_FLOOR = 2e-6, 2e-4, 1e-8
G_ATOL, G_RTOL, G_FLOOR = 2e-9, 2e-4, 1e-10
CONSTRUCTION_ATOL, CONSTRUCTION_RTOL = 2e-9, 5e-5
FD_ATOL, FD_RTOL, FD_FLOOR = 1e-3, 2e-2, 1e-8
SVD_ATOL, SVD_RTOL = 1e-8, 1e-4
FD_STEPS = (1e-4, 5e-5)
FD_STATES = ('truth', 'LM-101', 'LM-104')
INPUTS = {
    'docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json': 'c8e9ddc1d0ab3c5d028bf9925062b6e2d27b9b6bc017800e7a2b00d92aebd87d',
    'docs/MATERIAL_SAMPLE_COUPLING_V1_RESULTS.md': '9b9fd3a20be59e295606323d2c91f534a8bab631eebb55c0ebffcc805ce24dde',
    'docs/protocols/MATERIAL_SAMPLE_COUPLING_V1.md': 'ca968222f7f67c2164b238b9da35145ad9ad8a4d785ea675282971680ad4611f',
    'experiments/material_sample_coupling_v1.py': '598bc1f55b40d83c89f7768a1d0a2008d70e8f5fdce08b7a58757028a015a49c',
    'experiments/material_recovery_diagnostics_v1.py': '687a84cf141a4d005e30be9b0d8bd20599bca1fc93f812c54679fe92c79d5d2a',
    'experiments/material_recovery_optimizer_comparison_v1.py': '4abdf70b1a8c3b309f32561e31bcdb605392d0858bbe2015bf98c1a8f065cb6e',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def permutation():
    return np.eye(7, dtype=np.float64)[PERMUTATION]


def array(value):
    return coupling.accepted.array_identity(np.asarray(value), True)


def decode_array(record):
    a = np.asarray(record['values'], dtype=record['dtype'])
    require(list(a.shape) == record['shape'] and np.isfinite(a).all() and
            reference.sha_array(a) == record['sha256'], 'Saved array shape/finite/hash failure')
    return a


def admit():
    for name, sha in INPUTS.items():
        require(digest(ROOT/name) == sha, 'Accepted coupling identity changed: '+name)
    old, comparison, audit, states, identities = coupling.admit()  # Read-only admission; no old execution.
    evidence = json.loads((ROOT/'docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json').read_text())
    require(evidence['status'] == 'complete' and evidence['mapping_gate'] == 'passed' and
            evidence['work'] == dict(shader_calls=24, completed_shader_calls=24, points=5184,
                diffuse_directions=2654208, specular_directions=5308416) and
            evidence['scene'] == old['scene'] and evidence['shader'] == old['shader'] and
            evidence['tapes'] == {k: old['tapes'][k] for k in ('fit', 'evaluation')}, 'Complete accepted coupling evidence required')
    for flag in ('fixed_tapes_unchanged', 'global_torch_CPU_RNG_unchanged', 'global_numpy_RNG_unchanged',
                 'threads_restored', 'memory_limit_restored', 'accepted_shader_unchanged'):
        require(evidence[flag] is True, 'Coupling integrity failure: '+flag)
    for name, sha in {**evidence['accepted_identities'], **evidence['implementation_sha256']}.items():
        require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name) == sha, 'Prior source/protocol changed: '+name)
        identities[name] = sha
    require(evidence['protocol_sha256'] == INPUTS['docs/protocols/MATERIAL_SAMPLE_COUPLING_V1.md'], 'Coupling protocol mismatch')
    require([s['id'] for s in evidence['states']] == [s['id'] for s in states], 'Exactly six ordered accepted states required')
    for state, saved in zip(states, evidence['states']):
        require(state['nominal_physical'] == saved['nominal_physical'], 'Accepted physical state changed')
        for phase in ('fit', 'evaluation'):
            require(saved['batches'][phase]['status'] == 'complete' and saved['batches'][phase]['P0_identity']['exact_match'] and
                    saved['batches'][phase]['original_and_copied_tapes_preserved'], 'Incomplete coupling state/batch')
    identities.update(INPUTS)
    return old, comparison, audit, evidence, states, identities


def pdf_dependency():
    """Inspect active original AST, not comments/docstrings or auxiliary diagnostics."""
    source = ROOT/'scientific_sources/material_a/network/field.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MCShadingNetwork')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_radiance')
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == 'pdf']
    require(not reads, 'Reviewed active source RGB/PDF dependency has changed')
    return dict(source_sha256=digest(source), compute_radiance_pdf_active_reads=0,
        conclusion='Sampled PDF is passed but never read in active compute_radiance; auxiliary returned PDF is not part of original RGB MSE',
        unchanged_A2_specular_denominator='M_spec; approved AST change, not valid_counts',
        limits='Does not prove PDF symmetry; preserve 43 grazing exceptions independently of RGB derivative results')


def branch_status(theta):
    theta = float(theta)
    require(math.isfinite(theta), 'Finite angle required')
    cell = math.floor((theta+math.pi/2)/math.pi)
    principal = theta-cell*math.pi
    return dict(theta=theta, ideal_principal_angle=principal, branch_cell=cell,
                distance_to_branch_radians=max(0., math.pi/2-abs(principal)),
                boundaries='theta=(k+1/2)*pi; actual float32 encoding/frame separately admitted')


def errors(current, base, atol, rtol, floor):
    a, b = np.asarray(current, np.float64), np.asarray(base, np.float64)
    require(a.shape == b.shape and a.size and np.isfinite([a, b]).all(), 'Finite matching derivative arrays required')
    d = a-b
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(absolute_frobenius=float(np.linalg.norm(d)), relative_frobenius=float(np.linalg.norm(d)/max(nb, floor)),
        denominator_norm=nb, denominator_floor=floor, denominator_below_floor=nb < floor,
        max_absolute=float(np.max(abs(d))), RMSE=float(np.sqrt(np.mean(d*d))),
        current_norm=na, reference_norm=nb, atol=atol, rtol=rtol,
        exceeding_elements=int(np.count_nonzero(abs(d) > atol+rtol*abs(b))),
        passed=bool(np.all(abs(d) <= atol+rtol*abs(b))))


def gradient_comparison(current, base):
    result = errors(current, base, G_ATOL, G_RTOL, G_FLOOR)
    na, nb = result['current_norm'], result['reference_norm']
    meaningful = min(na, nb) > G_FLOOR
    result.update(cosine_similarity=float(np.clip(np.asarray(current)@np.asarray(base)/(na*nb), -1, 1)) if meaningful else None,
                  cosine_reason=None if meaningful else 'One or both gradient norms at/below the declared floor')
    return result


def spectrum(jacobian):
    j = np.asarray(jacobian, np.float64)
    _, s, vh = np.linalg.svd(j, full_matrices=False)
    threshold = max(SVD_ATOL, SVD_RTOL*float(s[0]))
    rank = int((s > threshold).sum())
    condition = float(s[0]/s[-1]) if s[-1] > SVD_ATOL else None
    return dict(singular_values=s.tolist(), condition_number=condition,
        condition_reason=None if condition is not None else 'Smallest singular value at/below absolute floor; no finite condition claimed',
        relative_near_null_threshold=SVD_RTOL, absolute_near_null_floor=SVD_ATOL,
        numerical_threshold=threshold, numerical_rank=rank, near_null_right_vectors=vh[s <= threshold].tolist(),
        condition_on_retained_subspace=float(s[0]/s[rank-1]) if rank else None,
        metric='Euclidean coordinates as labeled; physical widths/angle/albedo/metallic are not unit-rescaled')


def components(j):
    return {name: j[:, i].reshape(-1, 7) for i, name in enumerate(COMPONENTS)}


def describe_jacobian(j):
    return {name: dict(array=array(value), spectrum=spectrum(value)) for name, value in components(j).items()}


def jacobian_comparison(current, base):
    mapped = np.asarray(current, np.float64)[..., PERMUTATION]
    return {name: dict(**errors(components(mapped)[name], value, J_ATOL, J_RTOL, J_FLOOR),
        columns={p: errors(components(mapped)[name][:, k], value[:, k], J_ATOL, J_RTOL, J_FLOOR) for k, p in enumerate(PARAMETERS)})
        for name, value in components(np.asarray(base, np.float64)).items()}


class Budget:
    def __init__(self, report):
        self.start, self.report = time.perf_counter(), report
        report['work'] = dict(full_batch_forward_evaluations=0, completed_forwards=0, shader_points=0,
            diffuse_directions=0, specular_directions=0, fit_points=0, evaluation_points=0,
            categories={k: 0 for k in LIMITS}, jacobian_constructions=0, jacobians_completed=0,
            jacobian_vector_evaluations=0, JVP_completed=0, reverse_mode_gradient_evaluations=0, reverse_completed=0,
            direct_gradient_reverse_sweeps=0, direct_gradient_reverse_completed=0,
            coordinate_transform_reverse_sweeps=0, coordinate_transform_reverse_completed=0)

    def check(self):
        if time.perf_counter()-self.start >= WALL_SECONDS:
            raise original.RuntimeLimit('Single 1800-second derivative diagnostic; no retry')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= RSS_BYTES, '4 GiB peak RSS cap exceeded')

    def before(self, points, phase, kind):
        self.check()
        w = self.report['work']
        require(kind in LIMITS and phase in ('fit', 'evaluation') and type(points) is int and points > 0 and
                w['categories'][kind] < LIMITS[kind] and w['full_batch_forward_evaluations'] < MAX_CALLS and
                w['shader_points']+points <= MAX_POINTS, 'Declared forward/category/point budget exceeded')
        w['categories'][kind] += 1
        w['full_batch_forward_evaluations'] += 1
        w['shader_points'] += points
        w[phase+'_points'] += points


class Measurement:
    def __init__(self, shader, batch, ids, phase, budget):
        self.shader, self.batch, self.ids, self.phase, self.budget = shader, batch, ids, phase, budget
        self.latest = None

    def __call__(self, p, kind):
        self.budget.before(self.batch.n*self.batch.seeds, self.phase, kind)
        observer = coupling.accepted.Capture(self.shader)
        mean = self.batch(observer, p)
        r = observer.result
        require(mean.shape == (self.batch.n, 3, 3) and mean.dtype == torch.float32 and mean.device.type == 'cpu' and
                bool(torch.isfinite(mean).all()), 'Finite original float32 component layout required')
        with torch.no_grad():
            per = torch.stack([r[k].reshape(4, self.batch.n, 3) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], 2)
            mask = ((r['specular_directions']*r['normal'][:, None]).sum(-1) > 0).numpy().reshape(4, self.batch.n, -1)
            require(np.array_equal(mask.sum(-1), r['valid_specular_counts'].numpy().reshape(4, self.batch.n)), 'Source mask/count disagreement')
            ni = (r['specular_directions']*r['normal'][:, None]).sum(-1).numpy().reshape(mask.shape)
            oh = (self.batch.views[:, None]*r['half_vectors']).sum(-1).numpy().reshape(mask.shape)
            require(torch.allclose(per[:, :, 0]+per[:, :, 1], per[:, :, 2], atol=2e-7, rtol=2e-6), 'Component addition mismatch')
            self.latest = dict(mean=mean.detach(), per_tape=per.detach(), mask=mask.copy(),
                physical=p.detach().clone(), encoding=observer.encoding.numpy(),
                near_reflection_boundary=int((abs(ni) <= coupling.BOUNDARY).sum()),
                grazing_wo_dot_h=int((abs(oh) <= coupling.GRAZING).sum()))
        w = self.budget.report['work']
        w['completed_forwards'] += 1
        w['diffuse_directions'] += self.batch.n*4*self.shader.diffuse_samples
        w['specular_directions'] += self.batch.n*4*self.shader.specular_samples
        self.budget.check()
        return mean


def snapshot(data, ids, values=True):
    mask = coupling.accepted.mask_summary(data['mask'])
    if values:
        mask['packed_hex'] = np.packbits(data['mask']).tobytes().hex()
    return dict(mean_predictions=coupling.accepted.array_identity(data['mean'].numpy(), values),
        per_tape_predictions=coupling.accepted.array_identity(data['per_tape'].numpy(), values), masks=mask,
        actual_physical_float32=coupling.accepted.array_identity(data['physical'].numpy(), True),
        actual_encoding_float32=data['encoding'].tolist(), tape_identities=ids,
        near_reflection_boundary=data['near_reflection_boundary'], grazing_wo_dot_h=data['grazing_wo_dot_h'])


def verify_forward(data, archived, ids):
    value = snapshot(data, ids)
    for name in ('mean_predictions', 'per_tape_predictions', 'actual_physical_float32', 'actual_encoding_float32', 'tape_identities'):
        require(value[name] == archived[name], 'Accepted forward identity mismatch: '+name)
    require({k: v for k, v in value['masks'].items() if k != 'packed_hex'} == archived['masks'], 'Accepted mask identity mismatch')
    return value


def verify_primal(data, baseline):
    require(torch.equal(data['mean'], baseline['mean']) and torch.equal(data['per_tape'], baseline['per_tape']) and
            np.array_equal(data['mask'], baseline['mask']), 'AD primal/accepted baseline identity mismatch')


def build_jacobian(measure, p, base, row, persist):
    w = measure.budget.report['work']
    require(w['jacobian_vector_evaluations']+7 <= MAX_JVP, '210-JVP budget exceeded')
    w['jacobian_constructions'] += 1
    row['partial_columns'] = []
    expected = base['mean'].reshape(-1)
    for k in range(7):
        measure.budget.check()
        v = torch.zeros_like(p)
        v[k] = 1
        y, column = derivative.directional_jvp(lambda x: measure(x, 'jvp').reshape(-1), p, v, w)
        require(torch.equal(y, expected) and column.shape == expected.shape and column.dtype == torch.float32,
                'Original seven-direction AD shape/dtype/primal failure; no finite-difference fallback')
        verify_primal(measure.latest, base)
        row['partial_columns'].append(array(column.numpy()))
        persist()
    columns = [decode_array(r) for r in row['partial_columns']]
    j = np.stack(columns, -1).reshape(measure.batch.n, 3, 3, 7)
    require(j.dtype == np.float32 and np.isfinite(j).all(), 'Nonfinite/wrong dtype AD Jacobian')
    w['jacobians_completed'] += 1
    row['physical_jacobians'] = describe_jacobian(j)
    del row['partial_columns']
    return j


def direct_gradient(measure, p, base, j, target):
    x = p.detach().clone().requires_grad_()
    mean = measure(x, 'direct_gradient')
    verify_primal(measure.latest, base)
    loss = torch.mean((mean[:, 2]-target)**2)
    w = measure.budget.report['work']
    require(w['direct_gradient_reverse_sweeps'] < 18, 'Direct gradient budget exceeded')
    w['direct_gradient_reverse_sweeps'] += 1
    g, = torch.autograd.grad(loss, x)
    w['direct_gradient_reverse_completed'] += 1
    require(g.shape == (7,) and g.dtype == torch.float32 and bool(torch.isfinite(g).all()), 'Invalid direct autograd gradient')
    residual = (base['mean'][:, 2]-target).numpy().reshape(-1).astype(float)
    reconstructed = 2*components(j)['total'].astype(float).T@residual/residual.size
    check = errors(g.detach().numpy(), reconstructed, CONSTRUCTION_ATOL, CONSTRUCTION_RTOL, G_FLOOR)
    require(check['passed'], 'Mandatory direct MSE gradient/Jacobian reconstruction disagreement')
    return dict(objective_float32=float(loss.detach()), direct_autograd=array(g.detach().numpy()),
                residual_J_reconstruction=array(reconstructed), construction_check=check)


def latent_coordinates(p0, p1, saved, budget):
    if saved is None:
        z0 = torch.tensor(diagnostics.inverse_coordinates(reference.TRUTH), dtype=torch.float32)
        origin = 'Inverse-sigmoid candidate at original nominal truth; never substituted for direct physical truth'
    else:
        record = saved['final_state']['unconstrained']
        require(record['dtype'] == 'torch.float32' and record['device'] == 'cpu' and record['shape'] == [7], 'Original LM coordinates required')
        a = np.asarray(record['values'], np.float32)
        require(reference.sha_array(a) == record['sha256'] and np.array_equal(a.astype(float), np.asarray(record['values'])), 'LM coordinate hash/roundtrip mismatch')
        z0 = torch.tensor(a, dtype=torch.float32)
        origin = 'Accepted 3C-4 saved final LM coordinates; no optimizer restoration'
    z1 = z0[PERMUTATION].clone()
    z1[2] = p1[2]  # Same once-rounded nominal theta+pi/2 as admitted physical P1.
    result = dict(origin=origin, representations={})
    for label, z, p in (('P0', z0, p0), ('P1', z1, p1)):
        x = z.detach().clone().requires_grad_()
        physical = original.physical(x)
        exact = torch.equal(physical.detach(), p)
        item = dict(unconstrained=array(z.numpy()), reconstructed_physical=array(physical.detach().numpy()), exact_physical_reconstruction=exact,
                    max_physical_difference=float(abs(physical.detach()-p).max()),
                    valid=exact, reason=None if exact else 'Inverse float32 coordinates do not reproduce the admitted physical point; no latent derivative claim')
        if saved is not None:
            require(exact, 'Saved LM coordinate reconstruction is not exact')
        if exact:
            w = budget.report['work']
            require(w['coordinate_transform_reverse_sweeps'] < 12, 'Coordinate-chain reverse budget exceeded')
            w['coordinate_transform_reverse_sweeps'] += 1
            d, = torch.autograd.grad(physical, x, grad_outputs=torch.ones_like(physical))
            w['coordinate_transform_reverse_completed'] += 1
            require(bool(torch.isfinite(d).all()) and bool((d > 0).all()), 'Invalid original sigmoid chain derivative')
            item['dp_dz_diagonal_float32'] = array(d.detach().numpy())
        result['representations'][label] = item
    return result


def latent_result(j, gradient, coordinate):
    if not coordinate['valid']:
        return dict(valid=False, reason=coordinate['reason'])
    d = decode_array(coordinate['dp_dz_diagonal_float32']).astype(float)
    result = dict(valid=True, method='Original float32 transform AD diagonal applied in float64 analysis; no extra renderer call',
                  jacobians=describe_jacobian(j.astype(float)*d))
    if gradient is not None:
        result['gradient'] = array(decode_array(gradient['direct_autograd']).astype(float)*d)
    return result


def get_jacobian(record, coordinate='physical'):
    values = record['physical_jacobians'] if coordinate == 'physical' else record['optimizer_coordinates']['jacobians']
    return np.stack([decode_array(values[name]['array']).reshape(-1, 3, 7) for name in COMPONENTS], 1)


def compare_cases(current, base, fit):
    result = dict(physical_jacobians=jacobian_comparison(get_jacobian(current), get_jacobian(base)))
    if fit:
        result['physical_gradient'] = gradient_comparison(decode_array(current['gradient']['direct_autograd'])[PERMUTATION],
                                                         decode_array(base['gradient']['direct_autograd']))
    if current['optimizer_coordinates']['valid'] and base['optimizer_coordinates']['valid']:
        result['optimizer_coordinates'] = dict(valid=True, jacobians=jacobian_comparison(
            get_jacobian(current, 'latent'), get_jacobian(base, 'latent')))
        if fit:
            result['optimizer_coordinates']['gradient'] = gradient_comparison(
                decode_array(current['optimizer_coordinates']['gradient'])[PERMUTATION],
                decode_array(base['optimizer_coordinates']['gradient']))
    else:
        result['optimizer_coordinates'] = dict(valid=False, reason='Exact physical reconstruction unavailable; no latent gauge comparison')
    return result


def masks_from(record):
    m = record['masks']
    packed = bytes.fromhex(m['packed_hex'])
    require(hashlib.sha256(packed).hexdigest() == m['sha256_packbits'], 'Packed mask identity failed')
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:np.prod(m['shape'])].astype(bool).reshape(m['shape'])


def fd_record(plus, minus, base_record, baseline_branch, coordinate, h, ad_column):
    pm, mm = masks_from(plus), masks_from(minus)
    base_mask = masks_from(base_record)
    pb, mb = [branch_status(decode_array(r['actual_physical_float32'])[2]) for r in (plus, minus)]
    crossed = pb['branch_cell'] != baseline_branch['branch_cell'] or mb['branch_cell'] != baseline_branch['branch_cell']
    mask_changes = dict(plus_to_base=coupling.accepted.mask_difference(pm, base_mask),
                        minus_to_base=coupling.accepted.mask_difference(mm, base_mask), plus_to_minus=coupling.accepted.mask_difference(pm, mm))
    changes = any(r['flipped_total'] for r in mask_changes.values())
    p, m = decode_array(plus['mean_predictions']), decode_array(minus['mean_predictions'])
    central = (p.astype(float)-m.astype(float))/(2*h)
    pc, mc, bc = [decode_array(r['actual_physical_float32'])[coordinate] for r in (plus, minus, base_record)]
    return dict(nominal_step=h, physical_coordinate=coordinate, derivative=array(central),
        agreement_to_AD={name: errors(central[:, i], ad_column[:, i], FD_ATOL, FD_RTOL, FD_FLOOR) for i, name in enumerate(COMPONENTS)},
        represented_plus_step=float(pc)-float(bc), represented_minus_step=float(bc)-float(mc),
        represented_span=float(pc)-float(mc), denominator_used=2*h, branch_crossed=crossed,
        plus_branch=pb, minus_branch=mb, mask_changes=mask_changes, masks_changed=changes,
        smooth_interpretation='branch_crossed' if crossed else ('mask_changed; secant crosses pieces, not certified smooth derivative' if changes else 'No observed mask/decoder branch change'))


def finite_differences(shader, batches, row, report, budget, persist):
    row['finite_differences'] = {}
    for label in ('P0', 'P1-C'):
        batch, ids = batches[label]
        measure = Measurement(shader, batch, ids, 'fit', budget)
        base = row['fit']['cases'][label]
        p = decode_array(base['forward']['actual_physical_float32'])
        j = get_jacobian(base)
        arm = row['finite_differences'][label] = {}
        for original_coordinate in (0, 1, 2):
            coordinate = original_coordinate if label == 'P0' else PERMUTATION[original_coordinate]
            item = arm[PARAMETERS[original_coordinate]] = dict(original_coordinate=original_coordinate,
                transformed_coordinate=coordinate, steps=[])
            for h in FD_STEPS:
                pending = dict(nominal_step=h, status='running', forwards={})
                item['steps'].append(pending)
                pair = pending['forwards']
                for direction, sign in (('plus', 1), ('minus', -1)):
                    report.update(active_phase='finite_difference', active_representation=label,
                        active_coordinate=original_coordinate, active_h=h, active_sign=sign)
                    changed = p.astype(float).copy()
                    changed[coordinate] += sign*h
                    point = torch.tensor(changed, dtype=torch.float32)
                    with torch.no_grad():
                        measure(point, 'finite_difference')
                    pair[direction] = snapshot(measure.latest, ids)
                    persist()
                record = fd_record(pair['plus'], pair['minus'], base['forward'], base['branch'], coordinate, h, j[..., coordinate])
                pending.update(record, status='complete')
                persist()
            a, b = [decode_array(r['derivative']) for r in item['steps']]
            item['step_stability'] = {name: errors(b[:, i], a[:, i], FD_ATOL, FD_RTOL, FD_FLOOR) for i, name in enumerate(COMPONENTS)}
    row['finite_difference_gauge_comparisons'] = {}
    for name in PARAMETERS[:3]:
        row['finite_difference_gauge_comparisons'][name] = [
            {component: errors(decode_array(b['derivative'])[:, i], decode_array(a['derivative'])[:, i], FD_ATOL, FD_RTOL, FD_FLOOR)
                for i, component in enumerate(COMPONENTS)} for a, b in zip(
                    row['finite_differences']['P0'][name]['steps'], row['finite_differences']['P1-C'][name]['steps'])]


def execute(old, comparison, audit, evidence, states, report, persist):
    start, cpu, threads = time.perf_counter(), time.process_time(), torch.get_num_threads()
    rng, nrng = torch.get_rng_state().clone(), np.random.get_state()
    torch.set_num_threads(1)
    budget = Budget(report)
    try:
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
        report['runtime'] = runtime
        require(runtime == evidence['runtime'] == audit['runtime'] == comparison['runtime'] and torch.get_default_dtype() == torch.float32,
                'Accepted numerical CPU runtime/default dtype required')
        report['PDF_dependency'] = pdf_dependency()
        require(old['scene'] == reference.design(), 'Original scene changed')
        target64 = np.asarray(old['targets']['fine'], np.float64)
        require(old['targets']['passed'] and target64.shape == (84, 3, 3) and reference.sha_array(target64) == old['fine_targets_sha256'],
                'Original independent target identity failed')
        target = torch.tensor(target64[:24, 2], dtype=torch.float32)
        require(reference.sha_array(target.numpy()) == old['fit_target_float32_sha256'], 'Fitting target identity failed')
        shader = MaterialShaderV1('A2', 'atan2')
        require(shader.identity == old['shader'] == evidence['shader'], 'Original shader identity failed')
        report.update(states=[], shader=shader.identity, scene=old['scene'], fine_targets_sha256=old['fine_targets_sha256'],
            tapes={}, phase='state_and_tape_admission', mapping_gate='not_started')
        originals, phase_batches, state_batches = {}, {}, {}
        for phase, seeds, obs in [('fit', list(range(4)), old['scene']['observations'][:24]),
                                  ('evaluation', list(range(1000, 1004)), old['scene']['observations'])]:
            tape, ids = original.make_tapes(seeds, len(obs))
            require(ids == old['tapes'][phase], 'Original tape hash changed')
            phase_batches[phase] = (original.ObservationBatch(old['scene'], obs, tape, 4), seeds, ids)
            originals[phase] = [reference.sha_array(t.numpy()) for t in (tape.u, tape.v, tape.azimuth)]
            report['tapes'][phase] = ids
        for state, saved in zip(states, evidence['states']):
            p0, p1 = [torch.tensor(saved[name], dtype=torch.float32) for name in ('nominal_P0', 'nominal_P1')]
            require(np.allclose(coupling.accepted.tensor_q(saved['nominal_P0']), coupling.accepted.tensor_q(saved['nominal_P1']),
                atol=coupling.accepted.Q_ATOL, rtol=coupling.accepted.Q_RTOL), 'Q symmetry gate failed')
            row = {**state, 'status': 'not_started', 'mapping': saved['mapping'], 'fit': {'cases': {}}, 'evaluation': {'cases': {}}}
            report['states'].append(row)
            lm = next((r for r in comparison['outcomes'] if r['arm'] == 'LM' and r['seed'] == state.get('seed')), None)
            row['coordinate_reconstruction'] = latent_coordinates(p0, p1, lm, budget)
            state_batches[state['id']] = {}
            for phase, (batch, seeds, ids) in phase_batches.items():
                frames = [coupling.decoded_frame(shader, p, batch.n*4) for p in (p0, p1)]
                mapping = coupling.frame_sign(*frames)
                require(mapping == saved['mapping']['frames_by_batch'][phase], 'Independent actual signed frame differs from accepted mapping')
                copied, copied_record = coupling.copy_coupled_tape(batch.tapes, mapping['sign'])
                require(copied_record == saved['batches'][phase]['diagnostic_tape'], 'Diagnostic copied tape/hash differs from 3C-6')
                copied_ids = coupling.tape_ids(copied, seeds, batch.n)
                require(copied_ids == saved['batches'][phase]['P1_C']['tape_identities'], 'Coupled per-tape identity differs')
                state_batches[state['id']][phase] = {
                    'P0': (batch, ids), 'P1-U': (batch, ids),
                    'P1-C': (original.ObservationBatch(old['scene'], batch.observations, copied, 4), copied_ids)}
                row[phase]['coupled_tape'] = copied_record
            persist()
        report.update(mapping_gate='passed', phase='derivatives')
        for index, row in enumerate(report['states']):
            report['active_state'] = row['id']
            saved = evidence['states'][index]
            row['status'] = 'running'
            for phase in ('fit', 'evaluation'):
                labels = ('P0', 'P1-U', 'P1-C') if phase == 'fit' else ('P0', 'P1-C')
                for label in labels:
                    report.update(active_phase=phase, active_representation=label)
                    batch, ids = state_batches[row['id']][phase][label]
                    p = torch.tensor(saved['nominal_P0' if label == 'P0' else 'nominal_P1'], dtype=torch.float32)
                    archived = audit['states'][index]['representations'][1]['forwards']['fit'] if label == 'P1-U' else saved['batches'][phase][label.replace('-', '_')]
                    measure = Measurement(shader, batch, ids, phase, budget)
                    with torch.no_grad():
                        measure(p, 'baseline')
                    base = measure.latest
                    item = row[phase]['cases'][label] = dict(status='running', forward=snapshot(base, ids),
                        exact_accepted_forward=False, branch=branch_status(p[2]), locally_fixed_tapes=True)
                    persist()
                    verify_forward(base, archived, ids)
                    item['exact_accepted_forward'] = True
                    persist()
                    j = build_jacobian(measure, p, base, item, persist)
                    g = direct_gradient(measure, p, base, j, target) if phase == 'fit' else None
                    if g is not None:
                        require(g['objective_float32'] == archived['objective_float32'], 'Original objective identity mismatch')
                        item['gradient'] = g
                    item['optimizer_coordinates'] = latent_result(j, g, row['coordinate_reconstruction']['representations']['P0' if label == 'P0' else 'P1'])
                    item['status'] = 'complete'
                    persist()
                    del j, g, base, measure
                cases = row[phase]['cases']
                row[phase]['comparisons'] = {label+'_to_P0': compare_cases(cases[label], cases['P0'], phase == 'fit') for label in labels[1:]}
            row['status'] = 'derivatives_complete'
            persist()
            print('Derivative audit complete:', row['id'], flush=True)
        report['phase'] = 'finite_differences'
        for row in report['states']:
            if row['id'] in FD_STATES:
                report['active_state'] = row['id']
                finite_differences(shader, state_batches[row['id']]['fit'], row, report, budget, persist)
                print('Finite-difference checks complete:', row['id'], flush=True)
            row['status'] = 'complete'
        w = report['work']
        require(w['full_batch_forward_evaluations'] == w['completed_forwards'] == 330 and w['shader_points'] == 54720 and
            w['categories'] == LIMITS and w['jacobian_vector_evaluations'] == w['JVP_completed'] == 210 and
            w['reverse_mode_gradient_evaluations'] == w['reverse_completed'] == 420 and
            w['direct_gradient_reverse_sweeps'] == w['direct_gradient_reverse_completed'] == 18 and
            w['jacobian_constructions'] == w['jacobians_completed'] == 30 and
            w['coordinate_transform_reverse_sweeps'] == w['coordinate_transform_reverse_completed'] <= 12 and
            w['diffuse_directions'] == 54720*512 and w['specular_directions'] == 54720*1024, 'Exact derivative work accounting failed')
        require(all(originals[phase] == [reference.sha_array(t.numpy()) for t in (b.tapes.u, b.tapes.v, b.tapes.azimuth)]
            for phase, (b, _, _) in phase_batches.items()), 'Original tapes mutated')
        require(all(ids == coupling.tape_ids(b.tapes, phase_batches[phase][1], b.n)
            for batches in state_batches.values() for phase, arms in batches.items() for b, ids in [arms['P1-C']]), 'Coupled copied tapes mutated')
        report.update(status='complete', phase='finished', original_and_coupled_tapes_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        after = np.random.get_state()
        report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng, torch.get_rng_state())
        report['global_numpy_RNG_unchanged'] = nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:]
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed', failure='Unexpected global RNG consumption')
        torch.set_num_threads(threads)
        report['threads_restored'] = torch.get_num_threads() == threads
        report.update(wall_seconds=time.perf_counter()-start, CPU_seconds=time.process_time()-cpu,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in original.REQUIRED_ENV.items()), 'Explicit hidden-GPU one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT, 'Explicit fresh versioned output required')
    report = dict(schema_version=1, protocol='material-gradient-equivariance-v1', milestone='3C-7', base_commit=BASE, status='running',
        protocol_sha256=digest(PROTOCOL), implementation_sha256={name: digest(ROOT/name) for name in
            ('experiments/material_gradient_equivariance_v1.py', 'tests/scientific/test_material_gradient_equivariance_v1.py')},
        optimizer_creation=False, parameter_updates=False, new_targets=False, GPU_execution=False,
        jacobian_layout='[observation, component(diffuse,specular,total), RGB, physical_parameter]; each component flattened observation/RGB x7',
        parameters=PARAMETERS, permutation_matrix=permutation().tolist(),
        settings=dict(max_forwards=MAX_CALLS, max_points=MAX_POINTS, max_JVP=MAX_JVP, category_limits=LIMITS,
            wall_seconds=WALL_SECONDS, peak_RSS_budget_bytes=RSS_BYTES, address_space_budget_bytes=ADDRESS_BYTES,
            J_atol=J_ATOL, J_rtol=J_RTOL, J_relative_floor=J_FLOOR, g_atol=G_ATOL, g_rtol=G_RTOL, g_relative_floor=G_FLOOR,
            gradient_construction_atol=CONSTRUCTION_ATOL, gradient_construction_rtol=CONSTRUCTION_RTOL,
            FD_steps=list(FD_STEPS), FD_states=list(FD_STATES), FD_atol=FD_ATOL, FD_rtol=FD_RTOL, FD_relative_floor=FD_FLOOR,
            SVD_absolute_floor=SVD_ATOL, SVD_relative_near_null=SVD_RTOL),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), argv=[sys.executable, *sys.argv])
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    def stop(signum, frame):
        raise original.RuntimeLimit('Single 1800-second derivative audit terminated; no retry')
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream, report)
        persist()
        try:
            for s in handlers:
                signal.signal(s, stop)
            signal.alarm(WALL_SECONDS)
            cap = min(ADDRESS_BYTES, old_limit[0]) if old_limit[0] >= 0 else ADDRESS_BYTES
            if old_limit[1] >= 0:
                cap = min(cap, old_limit[1])
            resource.setrlimit(resource.RLIMIT_AS, (cap, old_limit[1]))
            report['effective_address_space_soft_limit'] = cap
            old, comparison, audit, evidence, states, identities = admit()
            report['accepted_identities'] = identities
            execute(old, comparison, audit, evidence, states, report, persist)
        except BaseException as exc:
            report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
            raise
        finally:
            resource.setrlimit(resource.RLIMIT_AS, old_limit)
            report['memory_limit_restored'] = resource.getrlimit(resource.RLIMIT_AS) == old_limit
            signal.alarm(0)
            for s, handler in handlers.items():
                signal.signal(s, handler)
            report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            persist()
    print('Final status:', report['status'], flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
