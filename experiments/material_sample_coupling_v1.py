"""One CPU forward-only common-world-sample diagnostic; no estimator changes."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import torch

from experiments import material_symmetry_audit_v1 as accepted
from experiments import run_known_material_recovery_v1 as original
from experiments import known_material_reference_v1 as reference
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs, decode_atan2
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = '1d16e9e147f6cdf270fbc0ed19e7add36ad17281'
PROTOCOL = ROOT/'docs/protocols/MATERIAL_SAMPLE_COUPLING_V1.md'
OUTPUT = ROOT/'docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json'
MAX_CALLS, MAX_POINTS, WALL_SECONDS = 24, 5184, 1800
RSS_BYTES, ADDRESS_BYTES = 4*1024**3, 8*1024**3
FRAME_ATOL, ANALYTIC_ATOL = 2e-6, 5e-14
H_ATOL, WI_ATOL, VECTOR_RTOL, ANGLE_ATOL = 2e-6, 4e-6, 2e-5, 2e-5
PDF_ATOL, PDF_RTOL, PDF_FLOOR = 1e-5, 2e-4, 1e-12
BOUNDARY, GRAZING = 2e-5, 1e-3
INPUTS = {
    'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json': 'a7ab47837718eccb69e935c676796ada39aac367b021ce08d79533cb729299a1',
    'docs/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2_RESULTS.md': 'ab785097f3a2f7e648bc51f7330dbda99853e54aa0c1dda375759b370864c6a8',
    'docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1.md': '8e510d566d443ed44f3e0428c429837ccd70b6bcfba0951efd5ce6bcf1d4572a',
    'docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.md': '0fe9f41b299f848c661529a6c36d5b2d061ef1fb5f028873f42960021807dbac',
    'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json': '055aa0a2c13457dc198bb888c45ff15062fec90856de31220e09ada196568a55',
    'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json': '47e879568455b593117304a89707a4216620fe6ca0e0a8985e4c98686f579b46',
    'experiments/material_symmetry_audit_v1.py': '53dce44200eb3278cf99561da0eb1d099e8c9da5bd15b7cc341149c86628fde9',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit():
    for name, sha in INPUTS.items():
        require(digest(ROOT/name) == sha, 'Accepted artifact SHA-256 mismatch: '+name)
    old, _, comparison, states, identities = accepted.admit()  # Read-only chain, never old runners.
    audit = json.loads((ROOT/'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json').read_text())
    require(audit['status'] == 'complete' and audit['physical_gate'] == 'passed' and
            audit['work']['shader_calls'] == audit['work']['completed_shader_calls'] == 36 and
            audit['scene'] == old['scene'] and audit['shader'] == old['shader'] and
            audit['tapes'] == {k: old['tapes'][k] for k in ('fit', 'evaluation')},
            'Complete accepted 3C-5 evidence with identical inputs required')
    for flag in ('fixed_tapes_unchanged', 'global_torch_CPU_RNG_unchanged', 'global_numpy_RNG_unchanged',
                 'threads_restored', 'memory_limit_restored'):
        require(audit[flag] is True, 'Accepted audit integrity failure: '+flag)
    for name, sha in {**audit['accepted_identities'], **audit['implementation_sha256']}.items():
        require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name) == sha,
                'Accepted source/protocol identity changed: '+name)
        identities[name] = sha
    require(audit['protocol_sha256'] == INPUTS['docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.md'],
            'Accepted protocol identity mismatch')
    require([s['id'] for s in audit['states']] == [s['id'] for s in states], 'Fixed six-state order changed')
    for state, saved in zip(states, audit['states']):
        require(state['nominal_physical'] == saved['nominal_physical'], 'Saved physical state changed')
        for label in ('P0', 'P1'):
            rep = next(r for r in saved['representations'] if r['id'] == label)
            p = accepted.transform(state['nominal_physical'], label)
            require(rep['nominal_float64'] == p.tolist() and rep['rounded_float32'] == p.astype(np.float32).astype(float).tolist()
                    and rep['physical_equivalence']['passed'] and rep['rounded_equivalence']['passed'],
                    'Admitted representation/physical gate changed')
            for phase, n in (('fit', 24), ('evaluation', 84)):
                f = rep['forwards'][phase]
                for key, shape in (('mean_predictions', (n, 3, 3)), ('per_tape_predictions', (4, n, 3, 3))):
                    read_array(f[key], shape)
                require(f['actual_physical_float32'] == accepted.array_identity(p.astype(np.float32), True),
                        'Archived shader parameter identity mismatch')
    identities.update(INPUTS)
    for stage, pin in PINS.items():
        baseline(stage, pin)
    identities['scientific_sources/manifest-v1.json'] = digest(ROOT/'scientific_sources/manifest-v1.json')
    return old, comparison, audit, states, identities


def read_array(record, shape):
    require(record['dtype'] == 'float32' and record['shape'] == list(shape), 'Expected saved float32 array shape')
    a = np.asarray(record['values'], np.float32)
    require(a.shape == shape and np.isfinite(a).all() and
            np.array_equal(a.astype(float), np.asarray(record['values'])) and
            reference.sha_array(a) == record['sha256'], 'Saved array values/hash mismatch')
    return a


def decoded_frame(shader, p, points=1):
    """Original flat basis and actual atan2 coefficients; no sampling/forward call."""
    with torch.no_grad():
        t, b = shader.template.compute_tangent_bitangent_flat(torch.tensor([[0., 0., 1.]]).repeat(points, 1), None)
        encoding = torch.stack([torch.cos(2*p[2]), torch.sin(2*p[2])]).reshape(1, 2).expand(points, 2)
        c, s = decode_atan2(encoding)
        frames = torch.stack([c*t+s*b, -s*t+c*b], -1).numpy()
        require(np.array_equal(frames, np.broadcast_to(frames[0], frames.shape)), 'Flat constant-material frame varied across points')
    return frames[0]


def frame_sign(f0, f1):
    f0, f1 = np.asarray(f0, np.float64), np.asarray(f1, np.float64)
    require(f0.shape == f1.shape == (3, 2) and np.isfinite([f0, f1]).all(), 'Finite world T/B frames required')
    for f in (f0, f1):
        require(np.allclose(f.T@f, np.eye(2), atol=FRAME_ATOL, rtol=0), 'Orthonormal actual T/B required')
    mapping = f1.T@f0
    sign = 1 if mapping[0, 1] > 0 else -1
    expected = np.array([[0., sign], [-sign, 0.]])
    require(np.allclose(mapping, expected, atol=FRAME_ATOL, rtol=0) and
            np.allclose(f1@expected, f0, atol=FRAME_ATOL, rtol=0), 'Decoded frames are not the declared signed quarter turn')
    return dict(sign=sign, F0=f0.tolist(), F1=f1.tolist(), F1_transpose_F0=mapping.tolist(),
                max_quarter_turn_error=float(np.max(abs(mapping-expected))),
                mapping='u_c=u; v_c=(v-s/4) mod 1', signed_azimuth_shift_turns=-sign/4)


def analytic_half_vectors(mx, my, u, v, frame):
    """Float64 closed form derived from source inversion, not a replacement sampler."""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    require(u.shape == v.shape and np.isfinite([u, v]).all() and ((u >= 0) & (u < 1)).all()
            and ((v >= 0) & (v < 1)).all(), 'Declared finite analytical uniforms required')
    alpha = 2*np.pi*v
    slope = np.sqrt(-np.log(np.maximum(u, 1e-6)))[..., None]*np.stack([mx*np.cos(alpha), my*np.sin(alpha)], -1)
    world = slope@np.asarray(frame, np.float64).T + np.array([0., 0., 1.])
    return world/np.linalg.norm(world, axis=-1, keepdims=True)


def tape_ids(tape, seeds, n):
    arrays = [t.numpy().reshape(4, n, *t.shape[1:]) for t in (tape.u, tape.v, tape.azimuth)]
    return [dict(seed=seed, points=n, specular_samples=tape.u.shape[1],
                 sha256=hashlib.sha256(b''.join(a[j].tobytes() for a in arrays)).hexdigest(),
                 array_sha256=[reference.sha_array(a[j]) for a in arrays]) for j, seed in enumerate(seeds)]


def copy_coupled_tape(tape, sign):
    require(type(sign) is int and sign in (-1, 1), 'Explicit signed quarter turn required')
    exact = np.remainder(tape.v.numpy().astype(np.float64)-sign/4, 1.)
    rounded = exact.astype(np.float32)
    # Float32 rounding can reach the periodic endpoint 1; reduce modulo 1 again,
    # rather than clipping, jittering or accepting an out-of-domain draw.
    endpoint = rounded == 1.
    coupled_v = np.remainder(rounded, np.float32(1.))
    result = RandomInputs(tape.u.clone(), torch.from_numpy(coupled_v.copy()), tape.azimuth.clone())
    result.validate(tape.u.shape[0], tape.u.shape[1])
    require(all(a.data_ptr() != b.data_ptr() for a, b in zip(
        (tape.u, tape.v, tape.azimuth), (result.u, result.v, result.azimuth))), 'Diagnostic tapes must own disjoint storage')
    require(torch.equal(tape.u, result.u) and torch.equal(tape.azimuth, result.azimuth), 'Radial/diffuse tape changed')
    record = dict(sign=sign, shift_turns=-sign/4, original_arrays=[accepted.array_identity(t.numpy()) for t in (tape.u, tape.v, tape.azimuth)],
                  coupled_arrays=[accepted.array_identity(t.numpy()) for t in (result.u, result.v, result.azimuth)],
                  radial_bitwise_unchanged=True, diffuse_azimuth_bitwise_unchanged=True, independent_storage=True,
                  radial_uniform_minimum=float(tape.u.min()), radial_uniform_maximum=float(tape.u.max()),
                  radial_draws_below_original_source_epsilon=int((tape.u < 1e-6).sum()), source_radial_epsilon=1e-6,
                  float32_periodic_endpoint_reductions=int(endpoint.sum()),
                  maximum_periodic_rounding_error=float(np.max(np.minimum(abs(coupled_v.astype(float)-exact),
                      1-abs(coupled_v.astype(float)-exact)))),
                  measure_preservation='Exact translation on R/Z preserves Lebesgue-uniform azimuth; finite float32 rounding recorded separately')
    return result, record


@contextmanager
def observe_pdf(shader):
    """Observe the owned dynamic class call; invoke the exact source method once."""
    cls = type(shader.template)
    require('compute_pdf_aniso_ggx' not in cls.__dict__, 'Unexpected owned shader PDF override')
    method = cls.compute_pdf_aniso_ggx
    trace = []
    def observed(model, mx, my, wo, h, theta, phi, eps=1e-6):
        value = method(model, mx, my, wo, h, theta, phi, eps)
        trace.append((mx, my, wo, h, theta, phi, eps, value))
        return value
    cls.compute_pdf_aniso_ggx = observed
    try:
        yield trace
    finally:
        delattr(cls, 'compute_pdf_aniso_ggx')
        require(cls.compute_pdf_aniso_ggx is method, 'Owned PDF observation hook not restored')


def pdf_fields(trace, n):
    require(len(trace) == 1, 'Exactly one original source PDF call required per shader call')
    mx, my, wo, h, theta, phi, eps, value = trace[0]
    require(eps == 1e-6, 'Original source PDF epsilon changed')
    with torch.no_grad():
        cp, sp, ch, sh = (f(a).unsqueeze(-1) for f, a in ((torch.cos, phi), (torch.sin, phi), (torch.cos, theta), (torch.sin, theta)))
        cos2 = ch*ch
        exp_raw = (-(sh*sh)/cos2.clamp_min(eps))*((sp*sp)/(my.view(-1, 1, 1)**2)+(cp*cp)/(mx.view(-1, 1, 1)**2))
        dot_raw = (wo.unsqueeze(1)*h).sum(-1, keepdim=True)
        denominator = (4.*np.pi)*mx.view(-1, 1, 1)*my.view(-1, 1, 1)*(ch**3)*dot_raw.clamp(min=0., max=1.)
        rebuilt = torch.exp(exp_raw.clamp(min=-40., max=40.))/denominator.clamp(min=eps)
        require(torch.equal(rebuilt, value), 'Read-only source-PDF operation reconstruction differs')
    arrays = {k: a.detach().numpy().reshape(4, n, -1) for k, a in
              [('cos_squared_raw', cos2), ('exponent_raw', exp_raw), ('wo_dot_h_raw', dot_raw), ('denominator_raw', denominator)]}
    arrays['cos_squared_clamped'] = arrays['cos_squared_raw'] < eps
    arrays['exponent_clamped'] = (arrays['exponent_raw'] < -40) | (arrays['exponent_raw'] > 40)
    arrays['dot_clamped'] = (arrays['wo_dot_h_raw'] < 0) | (arrays['wo_dot_h_raw'] > 1)
    arrays['denominator_clamped'] = arrays['denominator_raw'] < eps
    require(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite observed source PDF intermediates')
    return arrays


def capture(shader, batch, p, budget, ids):
    with observe_pdf(shader) as trace:
        data = accepted.capture_forward(shader, batch, p, budget, ids)
    data['pdf_fields'] = pdf_fields(trace, batch.n)
    require(np.array_equal(trace[0][3].detach().numpy().reshape(data['samples']['half_vectors'].shape),
                           data['samples']['half_vectors']), 'Observed PDF and returned half-vectors differ')
    return data


def pdf_summary(data):
    return {k: {**accepted.array_identity(v), 'minimum': float(v.min()), 'maximum': float(v.max()),
                **({'active_count': int(v.sum())} if v.dtype == np.bool_ else {})} for k, v in data['pdf_fields'].items()}


def error_record(a, b, atol, rtol, vector=False):
    a, b = np.asarray(a), np.asarray(b)
    delta = abs(a.astype(float)-b)
    beyond = delta > atol+rtol*abs(b)
    samples = beyond.any(-1) if vector else beyond
    result = dict(metrics=accepted.difference(a, b), atol=atol, rtol=rtol,
                  bitwise_identical=bool(np.array_equal(a, b)), samples=int(samples.size),
                  samples_exceeding_bound=int(samples.sum()),
                  exceeding_each_tape=[int(v.sum()) for v in samples],
                  individual_tapes=[accepted.difference(a[j], b[j]) for j in range(4)])
    if vector:
        af, bf = a.astype(float), b.astype(float)
        angles = np.arctan2(np.linalg.norm(np.cross(af, bf), axis=-1), np.sum(af*bf, axis=-1))
        result.update(max_angle_radians=float(angles.max()), mean_angle_radians=float(angles.mean()),
                      angular_bound_radians=ANGLE_ATOL, angles_exceeding_bound=int((angles > ANGLE_ATOL).sum()),
                      angle_max_each_tape=[float(v.max()) for v in angles])
    else:
        rel = delta/np.maximum(abs(b.astype(float)), PDF_FLOOR)
        result.update(max_elementwise_relative=float(rel.max()), relative_denominator_floor=PDF_FLOOR,
                      relative_RMSE=float(np.sqrt(np.mean(rel**2))))
    return result


def correspondence(current, base):
    require(current['observation_identity'] == base['observation_identity'], 'Common observation identity required')
    fields = {name: error_record(current['samples'][name], base['samples'][name], atol, rtol, vector)
              for name, atol, rtol, vector in [('half_vectors', H_ATOL, VECTOR_RTOL, True),
                  ('specular_directions', WI_ATOL, VECTOR_RTOL, True), ('pdf', PDF_ATOL, PDF_RTOL, False)]}
    mask = accepted.mask_difference(current['mask'], base['mask'])
    ni = [np.sum(d['samples']['specular_directions']*d['frames']['normal'][:, :, None], -1) for d in (current, base)]
    near = (abs(ni[0]) <= BOUNDARY) | (abs(ni[1]) <= BOUNDARY)
    flipped = current['mask'] != base['mask']
    mask.update(near_boundary_threshold=BOUNDARY, near_boundary_samples=int(near.sum()),
                flips_near_boundary=int((flipped & near).sum()), flips_outside_boundary=int((flipped & ~near).sum()),
                minimum_absolute_ni_current=float(abs(ni[0]).min()), minimum_absolute_ni_P0=float(abs(ni[1]).min()),
                current_valid=int(current['mask'].sum()), P0_valid=int(base['mask'].sum()), samples=int(near.size))
    a, b = current['pdf_fields'], base['pdf_fields']
    regions = {key: a[key] | b[key] for key in ('cos_squared_clamped', 'exponent_clamped', 'dot_clamped', 'denominator_clamped')}
    regions['any_clamp'] = np.logical_or.reduce(list(regions.values()))
    regions['grazing_wo_dot_h'] = (abs(a['wo_dot_h_raw']) <= GRAZING) | (abs(b['wo_dot_h_raw']) <= GRAZING)
    regions['near_reflection_boundary'] = near
    regions['outside_clamps_grazing_boundary'] = ~(regions['any_clamp'] | regions['grazing_wo_dot_h'] | near)
    pdf_a, pdf_b = (d['samples']['pdf'][..., 0] for d in (current, base))
    pdf_delta = abs(pdf_a.astype(float)-pdf_b)
    pdf_bad = pdf_delta > PDF_ATOL+PDF_RTOL*abs(pdf_b)
    breakdown = {}
    for key, selected in regions.items():
        count = int(selected.sum())
        breakdown[key] = dict(samples=count, exceeding_PDF_bound=int((pdf_bad & selected).sum()),
            errors=accepted.difference(pdf_a[selected], pdf_b[selected]) if count else None,
            max_elementwise_relative=float((pdf_delta[selected]/np.maximum(abs(pdf_b[selected]), PDF_FLOOR)).max()) if count else None,
            reason=None if count else 'No observed samples in this category')
    diffuse = bool(np.array_equal(current['samples']['diffuse_directions'], base['samples']['diffuse_directions']) and
                   np.array_equal(current['per_tape'][:, :, 0], base['per_tape'][:, :, 0]))
    require(diffuse, 'Specular-only copied tape unexpectedly changed diffuse directions/RGB')
    return dict(fields=fields, masks=mask, PDF_regions=breakdown, grazing_threshold=GRAZING,
                PDF_clamp_status_flips={key: int((a[key] != b[key]).sum()) for key in
                    ('cos_squared_clamped', 'exponent_clamped', 'dot_clamped', 'denominator_clamped')},
                diffuse_bitwise_unchanged=diffuse,
                within_vector_bounds=all(fields[k]['samples_exceeding_bound'] == fields[k]['angles_exceeding_bound'] == 0
                    for k in ('half_vectors', 'specular_directions')),
                masks_identical=mask['flipped_total'] == 0, within_PDF_bound=fields['pdf']['samples_exceeding_bound'] == 0,
                meaning='Original source PDFs evaluated at coupled approximately corresponding world half-vectors; no ideal-PDF substitution')


def verify_baseline(actual, archived):
    for key, value in actual.items():
        require(key in archived and value == archived[key], 'Exact P0 evidence identity failed: '+key)
    return dict(exact_match=True, mean_sha256=actual['mean_predictions']['sha256'],
                per_tape_sha256=actual['per_tape_predictions']['sha256'],
                sample_hashes={k: v['sha256'] for k, v in actual['samples'].items()},
                source='Admitted 3C-5 attempt2 P0; all original serialized fields identical')


class Budget:
    def __init__(self, report):
        self.start, self.report = time.perf_counter(), report
        report['work'] = dict(shader_calls=0, completed_shader_calls=0, points=0, diffuse_directions=0, specular_directions=0)

    def check(self):
        if time.perf_counter()-self.start >= WALL_SECONDS:
            raise original.RuntimeLimit('Single 1800-second coupling diagnostic limit')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= RSS_BYTES, '4 GiB peak RSS budget exceeded')

    def before_forward(self, points):
        self.check()
        w = self.report['work']
        require(w['shader_calls'] < MAX_CALLS and w['points']+points <= MAX_POINTS, '24-call/5184-point budget exceeded')
        w['shader_calls'] += 1
        w['points'] += points


def execute(old, comparison, audit, states, report, persist):
    start, cpu, threads = time.perf_counter(), time.process_time(), torch.get_num_threads()
    rng, nrng = torch.get_rng_state().clone(), np.random.get_state()
    torch.set_num_threads(1)
    budget = Budget(report)
    try:
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
        report['runtime'] = runtime
        require(runtime == audit['runtime'] == comparison['runtime'] and torch.get_default_dtype() == torch.float32,
                'Accepted CPU numerical runtime/default dtype required')
        require(old['scene'] == reference.design(), 'Original observation/lighting design changed')
        targets = np.asarray(old['targets']['fine'], np.float64)
        require(targets.shape == (84, 3, 3) and reference.sha_array(targets) == old['fine_targets_sha256'] and old['targets']['passed'],
                'Accepted converged targets required; never regenerate')
        target = torch.tensor(targets[:24, 2], dtype=torch.float32)
        require(reference.sha_array(target.numpy()) == old['fit_target_float32_sha256'], 'Original fitting targets changed')
        shader = MaterialShaderV1('A2', 'atan2')
        require(shader.identity == old['shader'] == audit['shader'], 'Accepted shader identity changed')
        report.update(states=[], shader=shader.identity, scene=old['scene'], fine_targets_sha256=old['fine_targets_sha256'], tapes={},
                      phase='mapping_admission', archived_P1_U='Reused exclusively from admitted 3C-5 attempt2; no P1-U or P2 forwards')
        # Admit every physical gauge and actual signed frame before any forward.
        for state, saved in zip(states, audit['states']):
            p0, p1 = [accepted.transform(state['nominal_physical'], label) for label in ('P0', 'P1')]
            require(np.allclose(accepted.tensor_q(p1), accepted.tensor_q(p0), atol=accepted.Q_ATOL, rtol=accepted.Q_RTOL),
                    'Nominal Q symmetry gate failed')
            batch_frames = {}
            for phase, points in (('fit', 96), ('evaluation', 336)):
                f0, f1 = [decoded_frame(shader, torch.tensor(p, dtype=torch.float32), points) for p in (p0, p1)]
                batch_frames[phase] = frame_sign(f0, f1)
                for rep, f in zip(saved['representations'][:2], (f0, f1)):
                    archived = rep['forwards'][phase]['frames']
                    require(np.array_equal(f, np.stack([archived[k]['first'] for k in ('tangent', 'bitangent')], -1)),
                            'Actual pre-execution decoder differs from accepted frames')
            mapping = batch_frames['fit'].copy()
            require(mapping['sign'] == batch_frames['evaluation']['sign'], 'Frame sign differs by batch')
            mapping['frames_by_batch'] = batch_frames
            report['states'].append({**state, 'nominal_P0': p0.tolist(), 'nominal_P1': p1.tolist(), 'mapping': mapping,
                'Q_difference': accepted.matrix_difference(accepted.tensor_q(p1), accepted.tensor_q(p0)), 'batches': {}})
        report['mapping_gate'] = 'passed'
        persist()
        batches = {}
        for phase, seeds, observations in [('fit', list(range(4)), old['scene']['observations'][:24]),
                                           ('evaluation', list(range(1000, 1004)), old['scene']['observations'])]:
            tape, ids = original.make_tapes(seeds, len(observations))
            require(ids == old['tapes'][phase], 'Original accepted tape hashes changed')
            report['tapes'][phase] = ids
            batches[phase] = (original.ObservationBatch(old['scene'], observations, tape, 4), seeds)
        originals = {phase: [reference.sha_array(t.numpy()) for t in (b.tapes.u, b.tapes.v, b.tapes.azimuth)] for phase, (b, _) in batches.items()}
        report['phase'] = 'coupled_forward_comparisons'
        for row, saved in zip(report['states'], audit['states']):
            report['active_state'] = row['id']
            for phase, (batch, seeds) in batches.items():
                report.update(active_phase=phase, active_representation='P0')
                p0, p1 = [torch.tensor(row[k], dtype=torch.float32) for k in ('nominal_P0', 'nominal_P1')]
                archived0, archived1 = [r['forwards'][phase] for r in saved['representations'][:2]]
                base = capture(shader, batch, p0, budget, report['tapes'][phase])
                serial0 = accepted.serialize_forward(base, p0, targets[:batch.n], batch.observations, target if phase == 'fit' else None)
                gate = verify_baseline(serial0, archived0)
                result = dict(P0_identity=gate, P0=serial0, P0_PDF_intermediates=pdf_summary(base), status='P0_verified')
                row['batches'][phase] = result
                persist()
                copied, mapping = copy_coupled_tape(batch.tapes, row['mapping']['sign'])
                ids = tape_ids(copied, seeds, batch.n)
                coupled_batch = original.ObservationBatch(old['scene'], batch.observations, copied, 4)
                report['active_representation'] = 'P1-C'
                current = capture(shader, coupled_batch, p1, budget, ids)
                frame = np.broadcast_to(row['mapping']['frames_by_batch'][phase]['F1'], (4, batch.n, 3, 2))
                require(np.array_equal(current['frames']['tangent'], frame[..., 0]) and
                        np.array_equal(current['frames']['bitangent'], frame[..., 1]),
                        'Forward frame differs from admitted signed frame')
                serial1 = accepted.serialize_forward(current, p1, targets[:batch.n], batch.observations, target if phase == 'fit' else None)
                archived_record = {k: archived1[k] for k in ('mean_predictions', 'per_tape_predictions', 'target_metrics')}
                compare = dict(radiance=accepted.radiance_comparison(current, base, batch.observations), sampling=correspondence(current, base))
                if phase == 'fit':
                    archived_record['objective_float32'] = archived1['objective_float32']
                    compare['objectives'] = dict(P0=serial0['objective_float32'], P1_U=archived1['objective_float32'], P1_C=serial1['objective_float32'],
                        P1_U_minus_P0=accepted.objective_difference(archived1['objective_float32'], serial0['objective_float32']),
                        P1_C_minus_P0=accepted.objective_difference(serial1['objective_float32'], serial0['objective_float32']))
                result.update(P1_C=serial1, P1_U_archived=archived_record,
                    P1_U_to_P0_archived=saved['representations'][1]['comparisons_to_P0'][phase],
                    P1_C_PDF_intermediates=pdf_summary(current), diagnostic_tape=mapping, comparison=compare, status='complete')
                require(ids == tape_ids(copied, seeds, batch.n), 'Diagnostic copied tape mutated during evaluation')
                require(originals[phase] == [reference.sha_array(t.numpy()) for t in (batch.tapes.u, batch.tapes.v, batch.tapes.azimuth)],
                        'Accepted original tape mutated')
                result['original_and_copied_tapes_preserved'] = True
                persist()
                # Do not retain the last phase's raw millions of sample vectors.
                del base, current, copied, coupled_batch
                budget.check()
            print('Coupled forward diagnostic complete:', row['id'], flush=True)
        require(report['work'] == dict(shader_calls=24, completed_shader_calls=24, points=5184,
                diffuse_directions=2654208, specular_directions=5308416), 'Exact declared diagnostic work accounting failed')
        report.update(status='complete', phase='finished', fixed_tapes_unchanged=True,
                      accepted_shader_unchanged=digest(ROOT/'experiments/material_shader_v1.py') == report['accepted_identities']['experiments/material_shader_v1.py'])
        require(report['accepted_shader_unchanged'], 'Accepted shader file changed')
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
    require(all(os.environ.get(k) == v for k, v in original.REQUIRED_ENV.items()), 'Explicit hidden-GPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT, 'Explicit fresh versioned output required')
    report = dict(schema_version=1, protocol='material-sample-coupling-v1', milestone='3C-6', base_commit=BASE, status='running',
        protocol_sha256=digest(PROTOCOL), implementation_sha256={name: digest(ROOT/name) for name in
            ('experiments/material_sample_coupling_v1.py', 'tests/scientific/test_material_sample_coupling_v1.py')},
        mapping_gate='not_started', optimizer_creation=False, parameter_updates=False, backward_calls=0, new_targets=False, GPU_execution=False,
        settings=dict(max_shader_calls=MAX_CALLS, max_points=MAX_POINTS, wall_seconds=WALL_SECONDS, peak_RSS_budget_bytes=RSS_BYTES,
            address_space_budget_bytes=ADDRESS_BYTES, frame_atol=FRAME_ATOL, analytical_half_vector_atol=ANALYTIC_ATOL,
            half_vector_atol=H_ATOL, incident_direction_atol=WI_ATOL, vector_rtol=VECTOR_RTOL, angular_atol_radians=ANGLE_ATOL,
            PDF_atol=PDF_ATOL, PDF_rtol=PDF_RTOL, PDF_relative_floor=PDF_FLOOR, reflection_boundary=BOUNDARY, grazing=GRAZING,
            RGB_atol=accepted.RGB_ATOL, RGB_rtol=accepted.RGB_RTOL, objective_atol=accepted.OBJECTIVE_ATOL, objective_rtol=accepted.OBJECTIVE_RTOL),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), argv=[sys.executable, *sys.argv])
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    def stop(signum, frame):
        raise original.RuntimeLimit('Single 1800-second coupling diagnostic stopped; no retry')
    with args.output.open('x') as stream:
        def persist():
            accepted.persist_report(stream, report)  # Strict serialization before touching the prior snapshot.
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
            old, comparison, audit, states, identities = admit()
            report['accepted_identities'] = identities
            execute(old, comparison, audit, states, report, persist)
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
