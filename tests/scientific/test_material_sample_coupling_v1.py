"""New tiny fixtures only: coupled correspondence, admission and owned cleanup."""
import ast
import copy
import io
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_sample_coupling_v1 as diagnostic
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs
from infrastructure.signatures import SignatureError


def tiny_tape():
    u = torch.linspace(.01, .99, 64).reshape(4, 16)
    v = torch.linspace(0., .999, 64).reshape(4, 16)
    return RandomInputs(u, v, torch.tensor([.1, .3, .5, .7]).reshape(4, 1, 1))


class TinyBatch:
    n, seeds = 1, 4
    observations = [dict(view=0, light=0, split='fit')]

    def __init__(self, tape):
        self.tapes = tape

    def __call__(self, shader, p):
        result = shader.shade(torch.zeros(4, 3), torch.tensor([[0., 0., 1.]]).repeat(4, 1),
            torch.tensor([[.2, -.1, np.sqrt(.95)]], dtype=torch.float32).repeat(4, 1),
            p[:1].reshape(1, 1).expand(4, 1), p[1:2].reshape(1, 1).expand(4, 1),
            p[3:6].reshape(1, 3).expand(4, 3), p[6:].reshape(1, 1).expand(4, 1),
            torch.stack([torch.cos(2*p[2]), torch.sin(2*p[2])]).reshape(1, 2).expand(4, 2),
            lambda pts, dirs: (.1+.9*dirs[..., 2:3].clamp(min=0)**8).expand_as(dirs), self.tapes, is_train=True)
        return torch.stack([result[k].reshape(4, 1, 3).mean(0) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], 1)


def tiny_pair(theta=.31):
    shader = MaterialShaderV1('A2', 'atan2', 8, 16)
    p = diagnostic.reference.TRUTH.copy()
    p[2] = theta
    q = diagnostic.accepted.transform(p, 'P1')
    a, b = [torch.tensor(v, dtype=torch.float32) for v in (p, q)]
    frames = [diagnostic.decoded_frame(shader, v) for v in (a, b)]
    mapping = diagnostic.frame_sign(*frames)
    tape = tiny_tape()
    copied, record = diagnostic.copy_coupled_tape(tape, mapping['sign'])
    report = {}
    budget = diagnostic.Budget(report)
    base = diagnostic.capture(shader, TinyBatch(tape), a, budget, [dict(fixture=True)]*4)
    current = diagnostic.capture(shader, TinyBatch(copied), b, budget, [dict(coupled_fixture=True)]*4)
    return shader, a, b, base, current, record, mapping, report


class MaterialSampleCouplingTests(unittest.TestCase):
    def test_Q_and_actual_decoder_signed_quarter_turn_both_branches(self):
        shader = MaterialShaderV1('A2', 'atan2', 8, 16)
        signs = set()
        for theta in (.31, -.8):
            p = diagnostic.reference.TRUTH.copy()
            p[2] = theta
            q = diagnostic.accepted.transform(p, 'P1')
            np.testing.assert_allclose(diagnostic.accepted.tensor_q(p), diagnostic.accepted.tensor_q(q), atol=5e-14, rtol=5e-12)
            f0, f1 = [diagnostic.decoded_frame(shader, torch.tensor(v, dtype=torch.float32)) for v in (p, q)]
            result = diagnostic.frame_sign(f0, f1)
            signs.add(result['sign'])
            self.assertLess(result['max_quarter_turn_error'], diagnostic.FRAME_ATOL)
        self.assertEqual(signs, {-1, 1})

    def test_wrong_nonquarter_or_nonorthogonal_frames_fail_closed(self):
        f = np.array([[0., -1.], [1., 0.], [0., 0.]])
        for bad in (f, f*2, f*np.nan):
            with self.assertRaises(SignatureError):
                diagnostic.frame_sign(f, bad)

    def test_archived_full_batch_frames_match_without_any_shader_forward(self):
        audit = json.loads((diagnostic.ROOT/'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json').read_text())
        shader = MaterialShaderV1('A2', 'atan2', 8, 16)
        with patch.object(shader, 'shade', side_effect=AssertionError('No forward in frame admission')):
            signs = []
            for state in audit['states']:
                state_signs = []
                for phase, points in (('fit', 96), ('evaluation', 336)):
                    frames = []
                    for rep in state['representations'][:2]:
                        p = torch.tensor(rep['rounded_float32'], dtype=torch.float32)
                        f = diagnostic.decoded_frame(shader, p, points)
                        saved = rep['forwards'][phase]['frames']
                        np.testing.assert_array_equal(f, np.stack([saved[k]['first'] for k in ('tangent', 'bitangent')], -1))
                        frames.append(f)
                    state_signs.append(diagnostic.frame_sign(*frames)['sign'])
                self.assertEqual(state_signs[0], state_signs[1])
                signs.append(state_signs[0])
            self.assertEqual(signs, [-1, 1, 1, 1, -1, -1])

    def test_float64_source_derived_half_vector_equality_both_signs_and_wraps(self):
        u = np.array([0., 1e-8, .1, .5, np.nextafter(1., 0.)])[:, None]
        v = np.array([0., np.nextafter(.25, 0.), .25, np.nextafter(.25, 1.), .5, .75, np.nextafter(1., 0.)])[None]
        u, v = np.broadcast_arrays(u, v)
        f0 = np.array([[0., -1.], [1., 0.], [0., 0.]])
        for sign in (-1, 1):
            m = np.array([[0., sign], [-sign, 0.]])
            f1 = f0@m.T
            h0 = diagnostic.analytic_half_vectors(.2, .55, u, v, f0)
            # A rounded remainder of exactly 1 is the periodic endpoint 0,
            # including the analogous float64 endpoint case.
            coupled = np.remainder(np.remainder(v-sign/4, 1), 1)
            h1 = diagnostic.analytic_half_vectors(.55, .2, u, coupled, f1)
            np.testing.assert_allclose(h0, h1, atol=diagnostic.ANALYTIC_ATOL, rtol=0)

    def test_uniform_circle_translation_preserves_interval_measure(self):
        # Deterministic exact dyadic grid: quarter-turn translations permute it.
        v = np.arange(4096, dtype=np.float64)/4096
        for sign in (-1, 1):
            mapped = (v-sign/4) % 1
            np.testing.assert_array_equal(np.sort(mapped), v)
            for lo, hi in ((0., .25), (.25, .75), (.75, 1.)):
                self.assertEqual(np.count_nonzero((mapped >= lo) & (mapped < hi)), int((hi-lo)*len(v)))

    def test_copy_preserves_radial_diffuse_index_order_and_disjoint_storage(self):
        tape = tiny_tape()
        original = [x.clone() for x in (tape.u, tape.v, tape.azimuth)]
        for sign in (-1, 1):
            copied, record = diagnostic.copy_coupled_tape(tape, sign)
            self.assertTrue(torch.equal(copied.u, tape.u))
            self.assertTrue(torch.equal(copied.azimuth, tape.azimuth))
            self.assertTrue(bool(((copied.v >= 0) & (copied.v < 1)).all()))
            self.assertTrue(record['independent_storage'])
            copied.u.fill_(.4)
            copied.v.fill_(.4)
            copied.azimuth.fill_(.4)
            for actual, saved in zip((tape.u, tape.v, tape.azimuth), original):
                self.assertTrue(torch.equal(actual, saved))
        for sign in (0, 2, True):
            with self.assertRaises(SignatureError):
                diagnostic.copy_coupled_tape(tape, sign)

    def test_float32_periodic_endpoint_and_wrap_neighbours(self):
        values = np.array([0., np.nextafter(np.float32(.25), np.float32(0)), .25,
            np.nextafter(np.float32(.25), np.float32(1)), .75, np.nextafter(np.float32(1), np.float32(0))], np.float32)
        tape = RandomInputs(torch.full((1, len(values)), .5), torch.from_numpy(values[None].copy()), torch.zeros(1, 1, 1))
        for sign in (-1, 1):
            coupled, record = diagnostic.copy_coupled_tape(tape, sign)
            self.assertTrue(bool(((coupled.v >= 0) & (coupled.v < 1)).all()))
            exact = (values.astype(float)-sign/4) % 1
            delta = abs(coupled.v.numpy()[0]-exact)
            self.assertLessEqual(float(np.minimum(delta, 1-delta).max()), 2**-24)
            if sign == 1:
                self.assertEqual(record['float32_periodic_endpoint_reductions'], 1)
                self.assertEqual(float(coupled.v[0, 1]), 0.)

    def test_tiny_real_A2_correspondence_both_frame_signs(self):
        signs = set()
        for theta in (.31, -.8):
            _, _, _, base, current, _, mapping, report = tiny_pair(theta)
            result = diagnostic.correspondence(current, base)
            signs.add(mapping['sign'])
            self.assertTrue(result['within_vector_bounds'])
            self.assertTrue(result['masks_identical'])
            self.assertTrue(result['diffuse_bitwise_unchanged'])
            self.assertTrue(result['within_PDF_bound'])
            self.assertEqual(report['work']['shader_calls'], 2)
            self.assertEqual(report['work']['completed_shader_calls'], 2)
        self.assertEqual(signs, {-1, 1})

    def test_source_PDF_intermediates_and_clamps_are_observed_not_substituted(self):
        shader, _, _, base, _, _, _, _ = tiny_pair()
        self.assertNotIn('compute_pdf_aniso_ggx', type(shader.template).__dict__)
        self.assertEqual(set(base['pdf_fields']), {'cos_squared_raw', 'exponent_raw', 'wo_dot_h_raw', 'denominator_raw',
            'cos_squared_clamped', 'exponent_clamped', 'dot_clamped', 'denominator_clamped'})
        summary = diagnostic.pdf_summary(base)
        self.assertEqual(summary['denominator_raw']['shape'], [4, 1, 16])
        self.assertEqual(summary['denominator_clamped']['active_count'], int(base['pdf_fields']['denominator_clamped'].sum()))

    def test_source_observation_hook_restored_on_failure(self):
        shader = MaterialShaderV1('A2', 'atan2', 8, 16)
        method = type(shader.template).compute_pdf_aniso_ggx
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with diagnostic.observe_pdf(shader):
                raise RuntimeError('fixture failure')
        self.assertIs(type(shader.template).compute_pdf_aniso_ggx, method)
        self.assertNotIn('compute_pdf_aniso_ggx', type(shader.template).__dict__)

    def test_complete_success_record_strict_JSON_roundtrip(self):
        _, p0, p1, base, current, tape, mapping, _ = tiny_pair()
        target = np.zeros((1, 3, 3))
        obs = TinyBatch.observations
        a = diagnostic.accepted.serialize_forward(base, p0, target, obs, torch.zeros(1, 3))
        c = diagnostic.accepted.serialize_forward(current, p1, target, obs, torch.zeros(1, 3))
        record = dict(status='complete', mapping=mapping, diagnostic_tape=tape, P0=a, P1_C=c,
            P0_identity=diagnostic.verify_baseline(a, a), P0_PDF=diagnostic.pdf_summary(base),
            P1_C_PDF=diagnostic.pdf_summary(current), sampling=diagnostic.correspondence(current, base),
            radiance=diagnostic.accepted.radiance_comparison(current, base, obs),
            objective=diagnostic.accepted.objective_difference(c['objective_float32'], a['objective_float32']))
        stream = io.StringIO()
        diagnostic.accepted.persist_report(stream, record)
        self.assertEqual(json.loads(stream.getvalue()), record)

    def test_P0_identity_includes_per_tape_sample_and_mask_hashes(self):
        _, p, _, base, _, _, _, _ = tiny_pair()
        serial = diagnostic.accepted.serialize_forward(base, p, np.zeros((1, 3, 3)), TinyBatch.observations, torch.zeros(1, 3))
        self.assertTrue(diagnostic.verify_baseline(serial, serial)['exact_match'])
        for key in ('per_tape_predictions', 'mean_predictions', 'samples', 'masks', 'tape_identities'):
            bad = copy.deepcopy(serial)
            bad[key] = 'wrong'
            with self.assertRaisesRegex(SignatureError, 'Exact P0 evidence identity'):
                diagnostic.verify_baseline(serial, bad)

    def test_saved_array_hash_dtype_shape_and_value_admission(self):
        a = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        record = diagnostic.accepted.array_identity(a, True)
        np.testing.assert_array_equal(diagnostic.read_array(record, a.shape), a)
        for field, bad_value in (('dtype', 'float64'), ('shape', [9]), ('sha256', 'wrong'), ('values', [[[.1]*3]*3])):
            bad = copy.deepcopy(record)
            bad[field] = bad_value
            with self.assertRaises(SignatureError):
                diagnostic.read_array(bad, a.shape)

    def test_corrupt_input_hash_rejected_before_old_admission_or_any_shader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = next(iter(diagnostic.INPUTS))
            path = root/name
            path.parent.mkdir(parents=True)
            path.write_text('not JSON')
            with patch.object(diagnostic, 'ROOT', root), patch.object(diagnostic.accepted, 'admit', side_effect=AssertionError('No old runner')), \
                 patch.object(diagnostic, 'MaterialShaderV1', side_effect=AssertionError('No shader')):
                with self.assertRaisesRegex(SignatureError, 'SHA-256'):
                    diagnostic.admit()

    def test_budget_rejects_before_25th_call_or_excess_points(self):
        report = {}
        budget = diagnostic.Budget(report)
        report['work']['shader_calls'] = 24
        with self.assertRaises(SignatureError):
            budget.before_forward(96)
        report['work']['shader_calls'] = 0
        report['work']['points'] = diagnostic.MAX_POINTS
        with self.assertRaises(SignatureError):
            budget.before_forward(96)
        with patch.object(diagnostic.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=diagnostic.RSS_BYTES//1024+1)):
            with self.assertRaisesRegex(SignatureError, 'RSS'):
                budget.check()

    def test_no_optimizer_backward_or_global_RNG_use(self):
        rng, nrng = torch.get_rng_state().clone(), np.random.get_state()
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer')), \
             patch.object(torch.optim, 'LBFGS', side_effect=AssertionError('No optimizer')), \
             patch.object(torch.Tensor, 'backward', side_effect=AssertionError('No backward')):
            _, _, _, base, current, _, _, _ = tiny_pair()
        self.assertFalse(base['mean_tensor'].requires_grad or current['mean_tensor'].requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertTrue(nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:])
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'Adam', 'SGD', 'LBFGS', 'step', 'backward', 'grad', 'requires_grad_', 'restore',
            'run_adam', 'run_lm', 'execute_previous', 'make_grids', 'render', 'hemisphere', 'initializations', 'manual_seed', 'cuda', 'load'})

    def test_runtime_failure_before_first_forward_restores_threads_and_RNG(self):
        threads = torch.get_num_threads()
        report = {}
        with patch.object(diagnostic, 'MaterialShaderV1', side_effect=AssertionError('No shader')):
            with self.assertRaisesRegex(SignatureError, 'runtime'):
                diagnostic.execute({}, dict(runtime={}), dict(runtime={}), [], report, lambda: None)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['work']['shader_calls'], 0)
        self.assertTrue(report['threads_restored'])
        self.assertTrue(report['global_numpy_RNG_unchanged'])
        self.assertTrue(report['global_torch_CPU_RNG_unchanged'])
        self.assertEqual(torch.get_num_threads(), threads)

    def test_failure_evidence_collision_memory_and_signal_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, protocol = root/'output.json', root/'protocol.md'
            target.write_text('preserve')
            protocol.write_text('fixture protocol')
            handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
            with patch.object(diagnostic, 'OUTPUT', target), patch.object(diagnostic, 'PROTOCOL', protocol), \
                 patch.dict(diagnostic.os.environ, diagnostic.original.REQUIRED_ENV), \
                 patch.object(diagnostic, 'admit', side_effect=SignatureError('fixture admission')), \
                 patch.object(diagnostic.resource, 'setrlimit') as limits, patch.object(sys, 'argv', ['fixture', '--output', str(target)]):
                with self.assertRaises(FileExistsError):
                    diagnostic.main()
                self.assertEqual(target.read_text(), 'preserve')
                self.assertEqual(limits.call_count, 0)
                target.unlink()  # Disposable fixture, never original evidence.
                with self.assertRaisesRegex(SignatureError, 'fixture admission'):
                    diagnostic.main()
                record = json.loads(target.read_text())
                self.assertEqual(record['status'], 'failed')
                self.assertTrue(record['memory_limit_restored'])
                self.assertEqual(limits.call_count, 2)
            self.assertEqual(handlers, {s: signal.getsignal(s) for s in handlers})

    def test_accepted_inputs_and_frozen_scientific_sources_unchanged(self):
        for name, sha in diagnostic.INPUTS.items():
            self.assertEqual(diagnostic.digest(diagnostic.ROOT/name), sha)
        for stage, pin in diagnostic.PINS.items():
            diagnostic.baseline(stage, pin)
        for name in ('experiments/material_shader_v1.py', 'experiments/material_symmetry_audit_v1.py',
                     'experiments/anisotropic_estimator_v1.py', 'experiments/run_known_material_recovery_v1.py'):
            self.assertEqual((diagnostic.ROOT/name).read_bytes(),
                subprocess.check_output(['git', 'show', diagnostic.BASE+':'+name], cwd=diagnostic.ROOT))


if __name__ == '__main__':
    unittest.main()
