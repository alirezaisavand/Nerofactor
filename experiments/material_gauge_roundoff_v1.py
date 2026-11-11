"""3C-9b: six CPU forwards and offline gauge-roundoff analysis; no optimization."""
import argparse
from contextlib import contextmanager
import datetime
from fractions import Fraction
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

from experiments import material_lm_gauge_trajectories_v1 as prior
from experiments import material_shader_v1 as shader_module
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline, PINS

original, coupling, reference = prior.original, prior.coupling, prior.reference
ROOT = Path(__file__).resolve().parents[1]
BASE = 'f9043aaecb709cebde597eab970458c72ad0f2c5'
NAME = 'MATERIAL_GAUGE_ROUNDOFF_V1'
PROTOCOL = ROOT / ('docs/protocols/' + NAME + '.md')
OUTPUT = ROOT / ('docs/evidence/' + NAME + '.json')
CODE = ('experiments/material_gauge_roundoff_v1.py', 'tests/scientific/test_material_gauge_roundoff_v1.py')
SEEDS = (101, 102)
CALLS = ((101, 'P0'), (101, 'P1-C'), (102, 'P0'), (102, 'P1-C'), (101, 'P1-C-E'), (102, 'P1-C-E'))
WALL_SECONDS, RSS_BYTES, ADDRESS_BYTES = 600, 4*1024**3, 8*1024**3
INPUTS = {
    'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json': '88b7e83ec7f79b3e9f476c9aa5ccd199e5394342ca83205394188f800cfb8f9a',
    'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1_FAILURE_ANALYSIS.json': '218ebd3e03f8e14accf8286789409dc93d1e704a434ea7856cd219c4dfc3c503',
    'docs/MATERIAL_LM_GAUGE_TRAJECTORIES_V1_RESULTS.md': 'c07e24dbc6cc35e1c5960714003d48e6503bbbb814210dc53747c99345bf8f0b',
    'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.md': '47e083d21f66f8ffe75e01072e512219264d3161974ea9397819231ab8455ba7',
    'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.freeze.json': '9d2e6845cd14d5991c6a5c4c655c14bf406f97c49b9a15572ffceb235d6f4db7',
    'experiments/material_lm_gauge_trajectories_v1.py': '27383436180896391f78e2bab3c3404ee029c6bd614506884b59a4ad0610c0d3',
    'tests/scientific/test_material_lm_gauge_trajectories_v1.py': 'b4e4d6e20d24fbcffad04b9f15c377fc109a9e2a1ca0d2ce825636fb27863662',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit():
    for name, sha in INPUTS.items():
        require(digest(ROOT/name) == sha, 'Accepted 3C-9 identity changed: '+name)
    freeze = json.loads((ROOT/'docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.freeze.json').read_text())
    for name, sha in freeze['sha256'].items():
        require(digest(ROOT/name) == sha, 'Frozen 3C-9 execution identity changed: '+name)
    old, _, continuation, identities = prior.admit()  # Read-only, never prior.execute().
    failed = json.loads((ROOT/'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json').read_text())
    require(failed['status'] == 'failed' and failed['restoration_gate'] == 'passed' and
            failed['failure'] == 'Mandatory coupled initial objective gate failed' and
            failed['frozen_execution'] == freeze, 'Exact failed attempt required')
    require(sum(r['work']['full_batch_forward_evaluations'] for r in failed['outcomes']) == 6 and
            all(r['work']['linear_solves'] == r['work']['reverse_mode_gradient_evaluations'] ==
                r['work']['accepted_parameter_steps'] == 0 for r in failed['outcomes']), 'Prior no-optimization boundary changed')
    for name, sha in failed['accepted_identities'].items():
        require(digest(ROOT/name) == sha, 'Failed experiment input changed: '+name)
        identities[name] = sha
    identities.update(INPUTS)
    return old, continuation, failed, identities


def array_record(a, values=False):
    a = np.asarray(a)
    require(a.dtype in (np.float32, np.float64, np.bool_) and a.size and np.isfinite(a).all(), 'Finite typed numerical array required')
    out = dict(dtype=str(a.dtype), shape=list(a.shape), sha256=reference.sha_array(a))
    if values:
        out['values'] = a.tolist()
    return out


def read_array(record, shape, dtype='float32'):
    require(record['dtype'] == dtype and record['shape'] == list(shape), 'Saved array layout mismatch')
    raw = np.asarray(record['values'])
    a = np.asarray(record['values'], dtype=dtype)
    require(a.shape == shape and np.isfinite(a).all() and np.array_equal(a.astype(float), raw.astype(float)) and
            reference.sha_array(a) == record['sha256'], 'Saved numerical values/hash mismatch')
    return a


def saved_mean(row):
    a = np.asarray(row['initial_fit_predictions'], np.float32)
    require(a.shape == (24, 3, 3) and np.isfinite(a).all() and
            np.array_equal(a.astype(float), row['initial_fit_predictions']) and
            reference.sha_array(a) == row['initial_fit_prediction_sha256'], 'Exact saved float32 predictions required')
    return a


def mse32(prediction, target):
    # Same original Torch subtraction, square and mean ordering; no normalized residual surrogate.
    p, t = torch.from_numpy(np.asarray(prediction, np.float32).copy()), torch.from_numpy(np.asarray(target, np.float32).copy())
    residual = p-t
    square = residual**2
    return float(torch.mean(square)), residual.numpy(), square.numpy()


def mse_variants(prediction, target):
    loss, residual, square = mse32(prediction, target)
    p, t = np.asarray(prediction, np.float32).astype(float), np.asarray(target, np.float32).astype(float)
    r64 = p-t
    return dict(original_float32=loss, square32_mean64=float(np.mean(square.astype(float))),
                residual32_square64_mean64=float(np.mean(residual.astype(float)**2)),
                prediction32_residual64_square64_mean64=float(np.mean(r64**2)))


def compare(a, b, per_tape=False, vector=False):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'Matching finite captured arrays required')
    af, bf = a.astype(float), b.astype(float)
    d = af-bf
    result = dict(bitwise_equal=bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
        max_absolute=float(np.max(abs(d))), RMSE=float(np.sqrt(np.mean(d*d))),
        relative_L2=float(np.linalg.norm(d)/max(np.linalg.norm(bf), 1e-30)),
        relative_denominator_floor=1e-30, differing_elements=int(np.count_nonzero(a != b)), elements=int(a.size))
    if a.dtype == b.dtype == np.float32:
        # Ordered integer bit distance, monotonic across signs; collapse +/-zero.
        def ordered(x):
            bits = x.view(np.uint32).astype(np.int64)
            return np.where(bits & 0x80000000, 0x80000000-(bits & 0x7fffffff), 0x80000000+bits)
        result['max_float32_ULP_distance'] = int(np.max(abs(ordered(a)-ordered(b))))
        spacing = np.maximum(abs(np.spacing(b)), np.float32(np.nextafter(np.float32(0), np.float32(1))))
        result['max_error_in_reference_spacing'] = float(np.max(abs(d)/spacing.astype(float)))
        result['ULP_limit'] = 'Near zero, sign changes and differing gauges can yield large ULP counts; no pass criterion'
    if vector:
        angle = np.arctan2(np.linalg.norm(np.cross(af, bf), axis=-1), np.sum(af*bf, axis=-1))
        result.update(max_angle_radians=float(angle.max()), mean_angle_radians=float(angle.mean()))
    if per_tape:
        require(a.shape[0] == 4, 'Four seed-major tapes required')
        result['per_tape'] = [compare(a[i], b[i], vector=vector) for i in range(4)]
    return result


def gate(base, candidate):
    bound = 1e-12+5e-7*abs(base)
    error = abs(candidate-base)
    return dict(base=base, candidate=candidate, signed_difference=candidate-base, absolute_error=error,
                frozen_bound=bound, signed_distance_from_bound=error-bound, passed=error <= bound)


def mse_decomposition(base, candidate, target, observations):
    a, b = np.asarray(base, np.float32), np.asarray(candidate, np.float32)
    require(a.shape == b.shape == target.shape == (24, 3), 'Exactly 24 fitting RGB observations required')
    va, vb = mse_variants(a, target), mse_variants(b, target)
    _, r0, sq0 = mse32(a, target)
    _, r1, sq1 = mse32(b, target)
    x, y = r0.astype(float), r1.astype(float)
    dr = y-x
    first, second = 2*x*dr, dr*dr
    left = float(np.mean(y*y)-np.mean(x*x))
    right = float(np.mean(first+second))
    error_bound = float(64*np.finfo(float).eps*(np.mean(y*y)+np.mean(x*x)+np.mean(abs(first+second))))
    require(abs(left-right) <= error_bound, 'Float64 residual difference identity failed')
    # Exact dyadic arithmetic verifies the algebra; do not claim rounded float64 sides are bitwise equal.
    pairs = [(Fraction(float(v)), Fraction(float(w))) for v, w in zip(x.flat, y.flat)]
    exact_left = sum(w*w-v*v for v, w in pairs)/72
    exact_right = sum(2*v*(w-v)+(w-v)**2 for v, w in pairs)/72
    require(exact_left == exact_right, 'Exact stored-residual algebra failed')
    differences = {key: vb[key]-va[key] for key in va}
    before = differences['prediction32_residual64_square64_mean64']
    subtract = differences['residual32_square64_mean64']-before
    square = differences['square32_mean64']-differences['residual32_square64_mean64']
    reduction = differences['original_float32']-differences['square32_mean64']
    contributions = (first+second)/72
    largest = []
    for index in np.argsort(abs(contributions).reshape(-1), kind='stable')[::-1][:10]:
        i, channel = divmod(int(index), 3)
        largest.append(dict(observation=i, view=observations[i]['view'], light=observations[i]['light'],
            channel=('R', 'G', 'B')[channel], contribution=float(contributions[i, channel]),
            RGB_difference=float(b[i, channel])-float(a[i, channel]),
            residual_difference=float(dr[i, channel])))
    return dict(base_MSEs=va, candidate_MSEs=vb, signed_MSE_differences=differences,
        original_gate=gate(va['original_float32'], vb['original_float32']),
        hypothetical_gates={key:gate(va[key], vb[key]) for key in va if key != 'original_float32'},
        fixed_historical_bound_hypothetical_gates={key:bool(abs(d) <= 1e-12+5e-7*abs(va['original_float32']))
            for key, d in differences.items()},
        signed_stage_contributions=dict(stored_RGB_before_objective=before, residual_subtraction_rounding=subtract,
            square_rounding=square, final_mean_rounding=reduction,
            total=before+subtract+square+reduction),
        residuals_P0=array_record(r0, True), residuals_candidate=array_record(r1, True),
        residual_difference=array_record(dr, True), per_observation_channel_MSE_contributions=contributions.tolist(),
        per_observation_MSE_contributions=contributions.sum(-1).tolist(),
        first_order_contribution=float(np.mean(first)), quadratic_contribution=float(np.mean(second)),
        algebra=dict(float64_left=left, float64_right=right, absolute_difference=abs(left-right),
            float64_roundoff_bound=error_bound, exact_dyadic_identity=True, exact_dyadic_difference_as_float=float(exact_left)),
        largest_observation_channel_contributors=largest,
        interpretation='Offline arithmetic on stored float32 RGB; no objective/renderer change')


def offline(failed, old):
    targets = np.asarray(old['targets']['fine'], np.float64)
    require(reference.sha_array(targets) == old['fine_targets_sha256'], 'Accepted reference array changed')
    target = targets[:24, 2].astype(np.float32)
    require(reference.sha_array(target) == old['fit_target_float32_sha256'], 'Original float32 targets changed')
    results = []
    for seed in SEEDS:
        rows = {r['arm']:r for r in failed['outcomes'] if r['seed'] == seed}
        p0, pc = saved_mean(rows['P0']), saved_mean(rows['P1-C'])
        for name, values in (('P0', p0), ('P1-C', pc)):
            require(mse32(values[:, 2], target)[0] == rows[name]['initial_fit_MSE'], 'Offline saved MSE replay mismatch')
        result = mse_decomposition(p0[:, 2], pc[:, 2], target, old['scene']['observations'][:24])
        result.update(seed=seed, saved_MSEs_exactly_reconstructed=True,
            archived_prediction_sha256={name: rows[name]['initial_fit_prediction_sha256'] for name in ('P0', 'P1-C')},
            components={name:compare(pc[:, i], p0[:, i]) for i, name in enumerate(('diffuse', 'specular', 'total'))},
            P1_U_archived_reference_MSE=rows['P1-U']['initial_fit_MSE'])
        require(result['original_gate']['passed'] == (seed == 101), 'Exact original gate outcomes required')
        results.append(result)
    return target, results


class RuntimeLimit(RuntimeError):
    pass


class Budget:
    def __init__(self, report):
        self.report = report
        self.start = time.perf_counter()
        self.work = report.setdefault('work', dict(attempted_shader_calls=0, completed_shader_calls=0,
            attempted_points=0, completed_points=0, attempted_diffuse_directions=0, attempted_specular_directions=0,
            diffuse_directions=0, specular_directions=0, optimizer_updates=0, derivative_calls=0))

    def check(self):
        if time.perf_counter()-self.start >= WALL_SECONDS:
            raise RuntimeLimit('Single 600-second diagnostic limit; no retry')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= RSS_BYTES, '4 GiB RSS cap exceeded')

    def before(self, seed, arm):
        self.check()
        require(not torch.is_grad_enabled(), 'Full forwards must run with gradients disabled')
        index = self.work['attempted_shader_calls']
        require(index < 6 and (seed, arm) == CALLS[index], 'Unapproved extra or reordered shader call')
        self.work['attempted_shader_calls'] += 1
        self.work['attempted_points'] += 96
        self.work['attempted_diffuse_directions'] += 96*512
        self.work['attempted_specular_directions'] += 96*1024

    def completed(self):
        self.work['completed_shader_calls'] += 1
        self.work['completed_points'] += 96
        self.work['diffuse_directions'] += 96*512
        self.work['specular_directions'] += 96*1024
        self.check()


def snapshot(value):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and not value.requires_grad,
            'Read-only CPU capture without autograd required')
    out = value.detach().numpy().copy()
    require(np.isfinite(out).all(), 'Nonfinite active intermediate')
    return out


class Capture:
    """Observe exact registered Python code objects at return; no method replacement."""
    def __init__(self, shader):
        self.data, self.counts = {}, {}
        self.codes = {
            shader_module.decode_atan2.__code__: ('decoder', ('encoding', 'r', 'theta')),
            shader_module.MaterialShaderV1.shade.__code__: ('shade', ('mx', 'my', 'albedo', 'metallic', 'encoding', 'rotation', 'lights', 'fresnel', 'fd', 'f0')),
            shader.template.sample_aniso_ggx_directions.__func__.__code__: ('sampler', ('xi1', 'xi2', 'two_pi_xi2', 'phi_h', 'theta_h', 'coeff_x', 'coeff_y', 'coeff_z', 'dot')),
            shader.template.compute_radiance.__func__.__code__: ('radiance', ('diff_weighted', 'spec_weighted', 'f_s', 'cos_in', 'cos_on', 'denom', 'mask')),
            shader.template.compute_pdf_aniso_ggx.__func__.__code__: ('pdf', ('cos_th', 'cos_th2', 'exp_term', 'dot_wo_h', 'denom')),
        }

    def profile(self, frame, event, result):
        if event != 'return' or frame.f_code not in self.codes:
            return
        name, fields = self.codes[frame.f_code]
        self.counts[name] = self.counts.get(name, 0)+1
        for field in fields:
            if field in frame.f_locals:
                self.data[name+'.'+field] = snapshot(frame.f_locals[field])
        if name == 'decoder' and isinstance(result, tuple):
            self.data['decoder.cos'], self.data['decoder.sin'] = map(snapshot, result)

    @contextmanager
    def active(self):
        old = sys.getprofile()
        require(old is None, 'Existing profiler must not be replaced')
        sys.setprofile(self.profile)
        try:
            yield self
        finally:
            sys.setprofile(old)
            require(sys.getprofile() is old, 'Profile hook cleanup failed')


class Invocation:
    def __init__(self, shader, encoding=None):
        self.shader, self.encoding, self.result = shader, encoding, None
        self.calls = 0
        self.actual_encoding = None

    def shade(self, *args, **kwargs):
        require(self.calls == 0, 'Exactly one shader invocation per original batch')
        self.calls += 1
        if self.encoding is not None:
            args = (*args[:7], self.encoding, *args[8:])
        self.actual_encoding = args[7]
        self.result = self.shader.shade(*args, **kwargs)
        return self.result


def validate_data(data):
    expected = {
        'shade.mx':(96, 1), 'shade.my':(96, 1), 'shade.albedo':(96, 3), 'shade.metallic':(96, 1),
        'shade.encoding':(96, 2), 'shade.rotation':(96, 2), 'decoder.r':(96, 2),
        'decoder.theta':(96, 1), 'decoder.cos':(96, 1), 'decoder.sin':(96, 1),
        'half_vectors':(96, 1024, 3), 'specular_directions':(96, 1024, 3),
        'diffuse_directions':(96, 512, 3), 'pdf':(96, 1024, 1),
        'tangent':(96, 3), 'bitangent':(96, 3), 'normal':(96, 3),
        'shade.lights':(96, 1536, 3), 'shade.fresnel':(96, 1024, 3),
        'radiance.diff_weighted':(96, 512, 3), 'radiance.spec_weighted':(96, 1024, 3),
        'radiance.mask':(96, 1024), 'per_tape':(4, 24, 3, 3), 'mean':(24, 3, 3),
    }
    for name, shape in expected.items():
        require(name in data and data[name].shape == shape and np.isfinite(data[name]).all(), 'Missing/invalid active capture: '+name)
        require(data[name].dtype == (np.bool_ if name == 'radiance.mask' else np.float32), 'Unexpected capture dtype: '+name)


def capture(shader, batch, p, budget, seed, arm, encoding=None):
    require(batch.n == 24 and batch.seeds == 4 and not p.requires_grad, 'Original fixed fitting batch required')
    invocation, observer = Invocation(shader, encoding), Capture(shader)
    budget.before(seed, arm)
    with observer.active():
        mean = batch(invocation, p)
    require(observer.counts == dict(decoder=1, sampler=1, pdf=1, radiance=1, shade=1), 'Incomplete/extra active capture calls')
    result = invocation.result
    data = observer.data
    for name in ('half_vectors', 'specular_directions', 'diffuse_directions', 'pdf', 'tangent', 'bitangent', 'normal'):
        data[name] = snapshot(result[name])
    data['per_tape'] = np.stack([snapshot(result[k]).reshape(4, 24, 3) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], 2)
    data['mean'] = snapshot(mean)
    data['random.u'], data['random.v'], data['random.azimuth'] = (snapshot(t) for t in (batch.tapes.u, batch.tapes.v, batch.tapes.azimuth))
    validate_data(data)
    require(np.array_equal(data['radiance.mask'], data['specular_directions'][..., 2] > 0), 'Actual and archived flat-mask conventions disagree')
    budget.completed()
    return data


def verify_baseline(data, row, target):
    expected = saved_mean(row)
    require(data['mean'].tobytes() == expected.tobytes() and
            mse32(data['mean'][:, 2], target)[0] == row['initial_fit_MSE'], 'Mandatory exact baseline RGB/MSE replay failed')
    mask = prior.lm.mask_identity(data['radiance.mask'])
    require(mask == row['initial_mask'], 'Mandatory exact baseline mask replay failed')
    return dict(exact_RGB=True, exact_MSE=True, exact_mask=True,
                mean_sha256=reference.sha_array(data['mean']), mask=mask)


def encoding(theta):
    t = torch.as_tensor(theta, dtype=torch.float32)
    return torch.stack((torch.cos(2*t), torch.sin(2*t)))


def encoding_equivalence(p0, p1, e0, e1):
    require(np.array_equal(e1, -e0), 'The only approved encoding intervention is exact negation')
    require(np.array_equal(p0[:2], p1[:2][::-1]) and np.array_equal(p0[3:], p1[3:]), 'Intervention must preserve physical gauge/other parameters')
    # Independent ideal interpretation of the actual float32 encodings.
    q = []
    for p, e in ((p0, e0), (p1, e1)):
        angle = .5*math.atan2(float(e[1]), float(e[0]))
        q.append(coupling.accepted.tensor_q([p[0], p[1], angle]))
    err = float(np.max(abs(q[1]-q[0])))
    require(np.all(abs(q[1]-q[0]) <= 5e-14+5e-12*abs(q[0])), 'Controlled encoding ideal physical Q mismatch')
    return dict(ideal_Q_max_absolute=err, ideal_physical_equivalence=True,
                comparison='Negation preserves ideal orientation gauge of the original float32 encoding, not exact nominal theta')


def frame_q(data):
    t, b = data['tangent'].astype(float), data['bitangent'].astype(float)
    mx, my = data['shade.mx'].astype(float), data['shade.my'].astype(float)
    return mx[:, :, None]**2*t[:, :, None]*t[:, None, :]+my[:, :, None]**2*b[:, :, None]*b[:, None, :]


def stage_comparisons(candidate, base, sign):
    pairs = {
        'orientation_encoding_vs_negative_P0':(candidate['shade.encoding'], -base['shade.encoding']),
        'normalized_encoding_vs_negative_P0':(candidate['decoder.r'], -base['decoder.r']),
        'head_normalized_encoding_vs_negative_P0':(candidate['shade.rotation'], -base['shade.rotation']),
        'decoder_cos_vs_signed_P0_sin':(candidate['decoder.cos'], -sign*base['decoder.sin']),
        'decoder_sin_vs_signed_P0_cos':(candidate['decoder.sin'], sign*base['decoder.cos']),
        'tangent_vs_signed_P0_bitangent':(candidate['tangent'], sign*base['bitangent']),
        'bitangent_vs_signed_P0_tangent':(candidate['bitangent'], -sign*base['tangent']),
        'local_half_coeff_x_vs_signed_P0_y':(candidate['sampler.coeff_x'], sign*base['sampler.coeff_y']),
        'local_half_coeff_y_vs_signed_P0_x':(candidate['sampler.coeff_y'], -sign*base['sampler.coeff_x']),
    }
    out = {name:compare(a.reshape(4, 24, *a.shape[1:]), b.reshape(4, 24, *b.shape[1:]), True) for name, (a, b) in pairs.items()}
    for name in ('half_vectors', 'specular_directions', 'diffuse_directions', 'pdf', 'shade.lights', 'shade.fresnel',
                 'radiance.diff_weighted', 'radiance.spec_weighted', 'radiance.f_s', 'radiance.cos_in', 'radiance.denom', 'sampler.theta_h'):
        if name not in candidate or name not in base:
            out[name] = dict(omitted='Not accessible at registered function return')
            continue
        a, b = candidate[name], base[name]
        out[name] = compare(a.reshape(4, 24, *a.shape[1:]), b.reshape(4, 24, *b.shape[1:]), True,
                            vector=name in ('half_vectors', 'specular_directions', 'diffuse_directions'))
    out['mask_XOR_count'] = int(np.count_nonzero(candidate['radiance.mask'] != base['radiance.mask']))
    for i, name in enumerate(('diffuse', 'specular', 'total')):
        out['per_tape_'+name] = compare(candidate['per_tape'][:, :, i], base['per_tape'][:, :, i], True)
        out['mean_'+name] = compare(candidate['mean'][:, i], base['mean'][:, i])
    return out


def raw_comparisons(candidate, base):
    require(candidate.keys() == base.keys(), 'Same captured fields required')
    out = {}
    for key in candidate:
        a, b = candidate[key], base[key]
        if a.shape[0] == 96:
            a, b = (v.reshape(4, 24, *v.shape[1:]) for v in (a, b))
        out[key] = compare(a, b, per_tape=a.shape[0] == 4)
    return dict(fields=out, limitation='Raw width/theta/frame/local-coordinate differences include intended gauge changes; use gauge-aligned stage comparisons for numerical correspondence')


def accumulation(data, target):
    per = data['per_tape']
    # Recreate the original independent component-wise means with contiguous input storage.
    mean32 = np.stack([torch.from_numpy(per[:, :, i].copy()).mean(0).numpy() for i in range(3)], 1)
    require(mean32.tobytes() == data['mean'].tobytes(), 'Original four-tape mean replay differs')
    mean64 = per.astype(float).mean(0)
    components64 = []
    sample_stats = {}
    for key, count in (('radiance.diff_weighted', 512), ('radiance.spec_weighted', 1024)):
        array = data[key]
        native = (torch.from_numpy(array).sum(1)/count).numpy()
        i = len(components64)
        require(native.reshape(4, 24, 3).tobytes() == per[:, :, i].copy().tobytes(), 'Native sample accumulation replay differs')
        double = array.astype(float).sum(1)/count
        sample_stats[key] = compare(native, double)
        components64.append(double)
    d64, s64 = components64
    sample64_per = np.stack((d64, s64, d64+s64), 1).reshape(4, 24, 3, 3)
    sample64_mean = sample64_per.mean(0)
    return dict(original_four_tape_mean_exact=True, original_sample_sums_exact=True,
        native_mean_MSEs=mse_variants(data['mean'][:, 2], target),
        mean64_of_stored_per_tape_RGB_MSE=float(np.mean((mean64[:, 2]-target.astype(float))**2)),
        sample64_sum_and_tape64_mean_MSE=float(np.mean((sample64_mean[:, 2]-target.astype(float))**2)),
        four_tape_mean_rounding=compare(data['mean'], mean64),
        sample_accumulation_rounding=sample_stats,
        sample64_mean_RGB=array_record(sample64_mean, True),
        mean64_stored_per_tape_RGB=array_record(mean64, True),
        limits='Offline alternatives on stored float32 contributions, not higher-precision shader executions')


def serialize(data, p, target):
    small = ('shade.encoding', 'shade.rotation', 'decoder.r', 'decoder.theta', 'decoder.cos', 'decoder.sin',
             'tangent', 'bitangent', 'normal', 'shade.mx', 'shade.my', 'shade.albedo', 'shade.metallic')
    records = {name:array_record(a, name in small or name in ('mean', 'per_tape')) for name, a in data.items()}
    mask = data['radiance.mask']
    dot = data['half_vectors'][..., 2]
    # Explicit indicators; no dependency claim from auxiliary PDF differences.
    return dict(physical_float32=array_record(p.numpy(), True), intermediates=records,
        accumulation=accumulation(data, target), fitting_MSE=mse32(data['mean'][:, 2], target)[0],
        mask=prior.lm.mask_identity(mask), grazing=dict(threshold=1e-5,
            half_vector_cosine_near_zero=int((abs(dot) <= 1e-5).sum()),
            source_clamped_wo_dot_h_near_zero=int((abs(data['pdf.dot_wo_h']) <= 1e-5).sum()) if 'pdf.dot_wo_h' in data else None,
            reflection_boundary_near_zero=int((abs(data['specular_directions'][..., 2]) <= 1e-5).sum()),
            sampled_PDF_min=float(data['pdf'].min()), sampled_PDF_max=float(data['pdf'].max())),
        capture_limit='Full sample tensors are held for paired offline analysis; JSON preserves their shapes/hashes, complete per-tape/mean RGB and stage error summaries, not millions of raw sample values. Return-local PDF exponent and wo-dot-h are already clamped; overwritten pre-clamp values are not captured.')


def prepare(old, continuation, failed, report):
    targets, offline_results = offline(failed, old)
    report['offline_saved_predictions'] = offline_results
    tape, ids = original.make_tapes(list(range(4)), 24)
    require(ids == old['tapes']['fit'], 'Original fitting tapes changed')
    states = {}
    shader = original.MaterialShaderV1('A2', 'atan2')
    require(shader.identity == old['shader'], 'Accepted shader identity changed')
    require(old['scene']['observations'][:24] == [o for o in old['scene']['observations'] if o['split'] == 'fit'], 'Fitting observation order changed')
    report['initial_states'] = []
    for seed in SEEDS:
        saved = next(s for s in continuation['initializations'] if s['seed'] == seed)
        z = prior.restore(saved)
        z1 = prior.gauge(z)
        p0, p1 = original.physical(z), original.physical(z1)
        rows = {r['arm']:r for r in failed['outcomes'] if r['seed'] == seed}
        for name, value, physical in (('P0', z, p0), ('P1-C', z1, p1)):
            require(prior.lm.accepted.describe(value) == rows[name]['initial_state']['unconstrained'] and
                    prior.lm.accepted.describe(physical) == rows[name]['initial_state']['physical'], 'Exact 3C-9 representation mismatch')
        require(np.all(abs(coupling.accepted.tensor_q(p1.numpy())-coupling.accepted.tensor_q(p0.numpy())) <=
                       2e-7+2e-6*abs(coupling.accepted.tensor_q(p0.numpy()))) and torch.equal(p0[3:], p1[3:]), 'Original rounded physical equivalence failed')
        mapping = coupling.frame_sign(coupling.decoded_frame(shader, p0, 96), coupling.decoded_frame(shader, p1, 96))
        copied, record = coupling.copy_coupled_tape(tape, mapping['sign'])
        prior_check = next(s for s in failed['starting_checks'] if s['seed'] == seed)
        require(mapping == prior_check['frame_mapping'] and record == prior_check['coupled_tape'], 'Exact coupled mapping/tape identity mismatch')
        e0 = encoding(p0[2])
        e_native = encoding(p1[2])
        e_override = -e0
        eq = encoding_equivalence(p0.numpy(), p1.numpy(), e0.numpy(), e_override.numpy())
        report['initial_states'].append(dict(seed=seed, exact_3C9_coordinates=True, original_tapes=ids, coupled_tape=record,
            physical_Q_relative=prior.q_difference(p1.numpy(), p0.numpy()), mapping=mapping,
            orientation=dict(theta0=float(p0[2]), theta1_float32=float(p1[2]),
                theta1_ideal_float64=float(p0[2])+math.pi/2,
                translation_rounding=float(p1[2])-(float(p0[2])+math.pi/2),
                E0=array_record(e0.numpy(), True), E1_native=array_record(e_native.numpy(), True),
                E1_override=array_record(e_override.numpy(), True),
                native_vs_ideal_negation=compare(e_native.numpy(), e_override.numpy())),
            controlled_encoding_equivalence=eq))
        states[seed] = dict(p0=p0, p1=p1, rows=rows, mapping=mapping, original_tape=tape, coupled_tape=copied,
            original_batch=original.ObservationBatch(old['scene'], old['scene']['observations'][:24], tape, 4),
            coupled_batch=original.ObservationBatch(old['scene'], old['scene']['observations'][:24], copied, 4),
            encoding_override=e_override.reshape(1, 2).expand(96, 2))
    return shader, targets, states


def execute(old, continuation, failed, report, persist, budget):
    runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
    require(runtime == failed['runtime'], 'Accepted original CPU numerical runtime required')
    report['runtime'] = runtime
    with torch.no_grad():
        shader, target, states = prepare(old, continuation, failed, report)
        report.update(phase='offline_and_construction_complete', shader=shader.identity,
            fitting_observations=old['scene']['observations'][:24], target_float32=array_record(target, True),
            sampled_PDF_dependency=prior.derivatives.pdf_dependency(), calls=[], comparisons=[])
        persist()
        captured = {}
        for seed, arm in CALLS:
            state = states[seed]
            report.update(phase='forward', active_seed=seed, active_arm=arm)
            row = dict(seed=seed, arm=arm, status='attempting')
            report['calls'].append(row)
            persist()
            p = state['p0'] if arm == 'P0' else state['p1']
            batch = state['original_batch'] if arm == 'P0' else state['coupled_batch']
            override = state['encoding_override'] if arm == 'P1-C-E' else None
            if override is not None:
                require(snapshot(override).tobytes() == (-captured[(seed, 'P0')]['shade.encoding']).tobytes(),
                        'Intervention must negate the actual captured P0 encoding')
            data = capture(shader, batch, p, budget, seed, arm, override)
            captured[(seed, arm)] = data
            # Preserve obtained arrays/identities before mandatory identity decisions.
            row['forward'] = serialize(data, p, target)
            persist()
            if arm != 'P1-C-E':
                row['baseline_identity'] = verify_baseline(data, state['rows'][arm], target)
            else:
                require(data['shade.encoding'].tobytes() == snapshot(override).tobytes(), 'Controlled encoding substitution mismatch')
                require(all(np.array_equal(data['shade.'+name], captured[(seed, 'P1-C')]['shade.'+name])
                            for name in ('mx', 'my', 'albedo', 'metallic')), 'Intervention changed material parameters')
                q0, q1 = frame_q(captured[(seed, 'P0')]), frame_q(data)
                row['controlled_actual_frame_Q'] = compare(q1, q0)
                require(np.all(abs(q1-q0) <= 2e-7+2e-6*abs(q0)), 'Controlled encoding rounded frame Q mismatch')
            require(all(np.array_equal(data['random.'+name], snapshot(value)) for name, value in
                zip(('u', 'v', 'azimuth'), (batch.tapes.u, batch.tapes.v, batch.tapes.azimuth))), 'Active tape mismatch')
            row['status'] = 'complete'
            persist()
            print('{} {}: MSE={:.17g}, baseline={}'.format(seed, arm, row['forward']['fitting_MSE'], arm != 'P1-C-E'), flush=True)
        for seed in SEEDS:
            state, comparisons = states[seed], {}
            base, native, intervention = (captured[(seed, a)] for a in ('P0', 'P1-C', 'P1-C-E'))
            for name, current in (('native_vs_P0', native), ('encoding_intervention_vs_P0', intervention)):
                comparisons[name] = dict(stages=stage_comparisons(current, base, state['mapping']['sign']),
                    raw_intermediates=raw_comparisons(current, base),
                    objective=mse_decomposition(base['mean'][:, 2], current['mean'][:, 2], target, old['scene']['observations'][:24]))
            comparisons['intervention_vs_native'] = dict(
                raw_intermediates=raw_comparisons(intervention, native),
                mean=compare(intervention['mean'], native['mean']),
                per_tape=compare(intervention['per_tape'], native['per_tape'], True),
                half_vectors=compare(intervention['half_vectors'].reshape(4,24,1024,3), native['half_vectors'].reshape(4,24,1024,3), True, True),
                tangent=compare(intervention['tangent'], native['tangent']),
                mask_XOR=int(np.count_nonzero(intervention['radiance.mask'] != native['radiance.mask'])))
            report['comparisons'].append(dict(seed=seed, **comparisons))
            budget.check()
        # Re-admit unchanged inputs/coupled tapes after observation; never reconstruct to repair a mismatch.
        for seed, state in states.items():
            record = next(s for s in report['initial_states'] if s['seed'] == seed)['coupled_tape']
            require([coupling.accepted.array_identity(t.numpy()) for t in (state['coupled_tape'].u, state['coupled_tape'].v, state['coupled_tape'].azimuth)] == record['coupled_arrays'], 'Coupled tape mutated')
            require(coupling.tape_ids(state['original_tape'], list(range(4)), 24) == old['tapes']['fit'], 'Original tape mutated')
        require(budget.work['completed_shader_calls'] == 6 and budget.work['completed_points'] == 576, 'Exact full-forward plan incomplete')
        report.update(status='complete', phase='finished', original_and_coupled_tapes_unchanged=True,
            original_gate_unchanged=True, no_optimizer_or_derivatives=True)
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(all(os.environ.get(k) == v for k, v in original.REQUIRED_ENV.items()), 'Explicit hidden-GPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT, 'Fresh explicit versioned output required')
    freeze = json.loads((ROOT/('docs/protocols/'+NAME+'.freeze.json')).read_text())
    for name, sha in freeze['sha256'].items():
        require(digest(ROOT/name) == sha, 'Pre-execution freeze mismatch: '+name)
    report = dict(schema_version=1, protocol='material-gauge-roundoff-v1', milestone='3C-9b', base_commit=BASE,
        status='running', phase='admission', frozen_execution=freeze, implementation_sha256={n:digest(ROOT/n) for n in CODE},
        protocol_sha256=digest(PROTOCOL), optimizer_execution=False, derivative_execution=False, GPU_execution=False,
        new_targets=False, started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=[sys.executable, *sys.argv], settings=dict(calls=list(CALLS), maximum_calls=6, maximum_points=576,
            diffuse_directions=294912, specular_directions=589824, wall_seconds=WALL_SECONDS, peak_RSS_bytes=RSS_BYTES,
            address_bytes=ADDRESS_BYTES, gate_atol=1e-12, gate_rtol=5e-7, physical_Q_atol=2e-7, physical_Q_rtol=2e-6,
            ideal_encoding_Q_atol=5e-14, ideal_encoding_Q_rtol=5e-12, float64_algebra_error_factor=64))
    threads, rng, nrng = torch.get_num_threads(), torch.get_rng_state().clone(), np.random.get_state()
    handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    address = resource.getrlimit(resource.RLIMIT_AS)
    profile = sys.getprofile()
    begin, cpu = time.perf_counter(), time.process_time()
    def stop(signum, frame):
        raise RuntimeLimit('Single bounded diagnostic interrupted; no retry')
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream, report)
        persist()
        try:
            for sig in handlers:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
            cap = min(ADDRESS_BYTES, address[0]) if address[0] >= 0 else ADDRESS_BYTES
            if address[1] >= 0:
                cap = min(cap, address[1])
            resource.setrlimit(resource.RLIMIT_AS, (cap, address[1]))
            torch.set_num_threads(1)
            budget = Budget(report)
            old, continuation, failed, identities = admit()
            report['accepted_identities'] = identities
            execute(old, continuation, failed, report, persist, budget)
            budget.check()
            for name, sha in identities.items():
                require(digest(ROOT/name) == sha, 'Accepted input changed during diagnostic: '+name)
        except BaseException as exc:
            report.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
            if report.get('calls') and report['calls'][-1]['status'] == 'attempting':
                report['calls'][-1].update(status='failed', failure=str(exc))
            raise
        finally:
            after = np.random.get_state()
            report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng, torch.get_rng_state())
            report['global_numpy_RNG_unchanged'] = nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:]
            torch.set_num_threads(threads)
            resource.setrlimit(resource.RLIMIT_AS, address)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            report.update(threads_restored=torch.get_num_threads() == threads,
                memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS) == address,
                profile_hook_restored=sys.getprofile() is profile,
                wall_seconds=time.perf_counter()-begin, CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if not all(report[k] for k in ('global_torch_CPU_RNG_unchanged', 'global_numpy_RNG_unchanged',
                                           'threads_restored', 'memory_limit_restored', 'profile_hook_restored')):
                report.update(status='failed', failure='Runtime state integrity failed')
            persist()
    print('Final status:', report['status'], flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
