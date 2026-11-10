"""NEW tiny 3C-0 fixtures only: no recovery, old suites or full experiments."""
import json
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from audits import anisotropic_brdf as independent
from experiments import anisotropic_estimator_v1 as old
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs, decode_atan2, encoding_status
from experiments import material_recovery_metrics_v1 as metrics
from infrastructure.signatures import SignatureError
from infrastructure.stage_sources import ROOT, baseline, PINS

BASE = '1067d3d5370eaaeabac7eedd650d4d913f87f820'


def tensor(x, grad=False):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device='cpu', requires_grad=grad)


def assert_close(a, b, rtol, atol):
    # NumPy assertion keeps the fixture independent of newer Torch test APIs.
    np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy(), rtol=rtol, atol=atol)


def fixture(count=32, points=1):
    rng = np.random.RandomState(17)
    draws = RandomInputs(tensor(rng.uniform(.4, .95, (points, count))),
                         tensor(rng.uniform(size=(points, count))), tensor(rng.uniform(size=(points, 1, 1))))
    return {'positions': tensor([[.1, -.2, .3]] * points), 'normals': tensor([[0, 0, 1]] * points),
            'views': tensor([[.3, .1, np.sqrt(.9)]] * points), 'mx': tensor([[.27]] * points),
            'my': tensor([[.53]] * points), 'albedo': tensor([[.34, .48, .63]] * points),
            'metallic': tensor([[.22]] * points), 'encoding': tensor([[.8, .6]] * points),
            'illumination': light, 'random_inputs': draws}


def light(points, directions):
    scale = .4 + .2 * directions[..., :1] + .1 * directions[..., 1:2] + .3 * directions[..., 2:3] ** 2
    return scale * tensor([1., .7, .3]) * (points[..., :1] + 1)


def white(points, directions):
    return torch.ones_like(directions)


class MaterialRecoveryPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        cls.models = {(v, d): MaterialShaderV1(v, d, 32, 32)
                      for v in ('A0', 'A1', 'A2') for d in ('original', 'atan2')}

    def test_A0_matches_original_A_forward_and_diagnostics_bitwise(self):
        f = fixture(); original, _ = old.shaders(32); source = original['B0']
        inputs = f['random_inputs']; tape = old.RandomTape([x.numpy() for x in (inputs.u, inputs.v, inputs.azimuth)])
        with patch.object(torch, 'rand', tape):
            spec = source.sample_aniso_ggx_directions(f['encoding'], f['positions'], f['mx'], f['my'],
                                                     f['views'], 32, f['normals'], None, device='cpu')
            diffuse = source.sample_diffuse_directions(f['normals'], True)
        directions = torch.cat([diffuse, spec[1]], 1)
        lights = light(f['positions'].unsqueeze(1).repeat(1, 64, 1), directions)
        f0 = .04 * (1 - f['metallic']) + f['metallic'] * f['albedo']
        fresnel, _, _ = source.fresnel_schlick_directions(f0.unsqueeze(1), f['views'].unsqueeze(1), spec[1])
        expected = source.compute_radiance(source.diffuse_term(f['albedo'], f['metallic']),
            lights[:, :32], lights[:, 32:], fresnel, diffuse, spec[1], f['normals'], spec[2], f['views'],
            spec[3], spec[7], spec[8], f['mx'], f['my'])
        actual = self.models['A0', 'original'].shade(**f)
        for key, value in zip(('linear_rgb', 'diffuse_rgb', 'specular_rgb', 'original_f_s_sum', 'original_L_spec'), expected):
            self.assertTrue(torch.equal(actual[key], value), key)
        self.assertTrue(torch.equal(actual['half_vectors'], spec[0]))
        self.assertTrue(torch.equal(actual['diffuse_directions'], diffuse))
        self.assertTrue(torch.equal(actual['srgb'], self.models['A0', 'original'].to_srgb(expected[0])))

    def test_A1_matches_approved_B3_algebra_bitwise(self):
        objects, _ = old.shaders(32); source = objects['B3']; f = fixture()
        wo = np.array([.8, 0, .6]); stream = old.sample_stream(3, 32)
        spec, diffuse, _ = old.draw(source, .6, .9, wo, stream)
        expected = old.estimate(source, spec, diffuse, wo, .6, .9)
        f.update(positions=tensor([[0, 0, 0]]), views=tensor([wo]), mx=tensor([[.6]]), my=tensor([[.9]]),
                 albedo=tensor([[.999] * 3]), metallic=tensor([[.001]]),
                 encoding=tensor([[np.cos(.62), np.sin(.62)]]),
                 random_inputs=RandomInputs(*(tensor(x) for x in stream)))
        for i in range(3):
            # For exact parity, inject the same fixture light values used by
            # 3B-2; differentiability is verified separately with Torch lights.
            def illumination(points, directions):
                value = old.lighting(directions.detach().numpy())[..., i:i+1]
                return tensor(np.repeat(value, 3, axis=-1))
            f['illumination'] = illumination
            result = self.models['A1', 'original'].shade(**f)
            np.testing.assert_array_equal([float(result[k][0, 0]) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], expected[i])
            self.assertTrue(torch.equal(result['specular_directions'], spec[1]))
            self.assertTrue(torch.equal(result['diffuse_directions'], diffuse))

    def test_midpoint_is_only_additional_A2_change(self):
        f = fixture()
        for decoder in ('original', 'atan2'):
            a = self.models['A1', decoder].shade(**f); b = self.models['A2', decoder].shade(**f)
            for key in ('specular_rgb', 'original_f_s_sum', 'original_L_spec', 'half_vectors', 'tangent', 'pdf'):
                self.assertTrue(torch.equal(a[key], b[key]), key)
            np.testing.assert_allclose(b['diffuse_directions'][0, :, 2].numpy() ** 2,
                                       (np.arange(32) + .5) / 32, atol=1e-7)
            self.assertEqual(b['random_calls'], a['random_calls'])

    def test_midpoint_removes_first_moment_error_and_has_declared_second_order_error(self):
        # Tiny N=32 fixture; exact midpoint integration of z and O(N^-2) of z^2.
        f = fixture(); n = 32; d = f['albedo'] * (1 - f['metallic'])
        for power in (2, 4):
            f['illumination'] = lambda points, directions: directions[..., 2:3].clamp(min=0).pow(power).expand_as(directions)
            left = self.models['A1', 'original'].shade(**f)['diffuse_rgb']
            mid = self.models['A2', 'original'].shade(**f)['diffuse_rgb']
            target = d / (1 + power / 2)
            expected_error = torch.zeros_like(d) if power == 2 else -d / (12 * n * n)
            assert_close(mid - target, expected_error, rtol=1e-3, atol=1e-7)
            self.assertTrue(bool((abs(mid - target) < abs(left - target)).all()))

    def test_original_and_corrected_decoder_are_independent_factors(self):
        f = fixture(); f['encoding'] = tensor([[-1, 0]])
        for variant in ('A0', 'A1', 'A2'):
            a = self.models[variant, 'original'].shade(**f)
            b = self.models[variant, 'atan2'].shade(**f)
            self.assertAlmostEqual(float(torch.norm(a['tangent'])), .001, places=7)
            self.assertFalse(bool(a['frame_orthonormal'].all()))
            self.assertTrue(bool(b['frame_orthonormal'].all()))
            self.assertAlmostEqual(float(torch.norm(b['tangent'])), 1., places=6)
            self.assertAlmostEqual(float(torch.norm(b['bitangent'])), 1., places=6)
            self.assertLess(float(abs((b['tangent'] * b['bitangent']).sum())), 1e-6)
            self.assertEqual(a['identity']['variant'], b['identity']['variant'])

    def test_zero_and_near_zero_encodings_are_explicitly_invalid(self):
        values = tensor([[0, 0], [1e-8, 0], [1e-6, 0], [-1, 0]])
        status = encoding_status(values)
        self.assertEqual(status['zero'].flatten().tolist(), [True, False, False, False])
        self.assertEqual(status['near_zero'].flatten().tolist(), [False, True, False, False])
        self.assertTrue(bool(encoding_status(tensor([[1e-30, 0]]))['near_zero'].all()))
        with self.assertRaisesRegex(SignatureError, 'overflow'): decode_atan2(tensor([[3e38, 3e38]]))
        for value in ([0, 0], [1e-8, 0]):
            with self.assertRaisesRegex(SignatureError, 'no fallback'): decode_atan2(tensor([value]))
            f = fixture(); f['encoding'] = tensor([value])
            with self.assertRaisesRegex(SignatureError, 'no fallback'): self.models['A2', 'atan2'].shade(**f)
        # Immutable original mode still exposes its original behavior, marked invalid.
        f = fixture(); f['encoding'] = tensor([[0, 0]])
        self.assertFalse(bool(self.models['A0', 'original'].shade(**f)['encoding_status']['valid'].all()))

    def test_corrected_frames_on_curved_known_normals(self):
        f = fixture(points=3)
        f['normals'] = tensor([[0, 0, 1], [.25, 0, np.sqrt(.9375)], [1, 0, 0]])
        f['views'] = f['normals'].clone()
        f['encoding'] = tensor([[-1, 0], [.8, .6], [1, 0]])
        result = self.models['A2', 'atan2'].shade(**f)
        basis = torch.stack([result[k] for k in ('tangent', 'bitangent', 'normal')], -1)
        assert_close(basis.transpose(-1, -2) @ basis, torch.eye(3).expand(3, -1, -1), atol=3e-7, rtol=1e-6)
        assert_close(torch.det(basis), torch.ones(3), atol=3e-7, rtol=1e-6)

    def test_pathwise_gradients_match_two_bounded_finite_difference_steps(self):
        for mode in self.models:
            model = self.models[mode]
            def evaluate(x):
                f = fixture(); f.update(mx=x[0:1].reshape(1, 1), my=x[1:2].reshape(1, 1),
                    albedo=x[2:5].reshape(1, 3), metallic=x[5:6].reshape(1, 1), encoding=x[6:8].reshape(1, 2))
                result = model.shade(**f)
                self.assertEqual(int(result['valid_specular_counts'][0]), 32)
                return (result['linear_rgb'] * tensor([[.7, 1.1, 1.4]])).sum()
            x = tensor([.27, .53, .34, .48, .63, .22, .8, .6], grad=True)
            gradient, = torch.autograd.grad(evaluate(x), x)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertTrue(bool((abs(gradient) > 1e-6).all()), (mode, gradient))
            for h in (1e-3, 5e-4):
                numerical = []
                for i in range(len(x)):
                    delta = torch.zeros_like(x); delta[i] = h
                    numerical.append(float((evaluate(x.detach() + delta) - evaluate(x.detach() - delta)) / (2 * h)))
                np.testing.assert_allclose(gradient.numpy(), numerical, rtol=.03, atol=2e-4)
            # The encoding norm is a gauge: its radial derivative should vanish.
            self.assertAlmostEqual(float((gradient[6:] * x.detach()[6:]).sum()), 0., delta=2e-6)

    def test_corrected_decoder_realizes_the_physical_tensor_and_gauges(self):
        tensors = []
        for mx, my, theta in [(.27, .53, .31), (.27, .53, .31 + np.pi), (.53, .27, .31 + np.pi / 2)]:
            f = fixture(); f.update(mx=tensor([[mx]]), my=tensor([[my]]),
                encoding=tensor([[np.cos(2 * theta), np.sin(2 * theta)]]))
            result = self.models['A2', 'atan2'].shade(**f)
            base = np.array([[0, -1, 0], [1, 0, 0.]])
            t = base @ result['tangent'][0].numpy(); b = base @ result['bitangent'][0].numpy()
            q = mx ** 2 * np.outer(t, t) + my ** 2 * np.outer(b, b)
            np.testing.assert_allclose(q, metrics.roughness_tensor(mx, my, theta), rtol=8e-7, atol=3e-8)
            tensors.append(q)
        for q in tensors[1:]: np.testing.assert_allclose(q, tensors[0], rtol=8e-7, atol=3e-8)

    def test_regularizer_detach_semantics_preserved(self):
        f = fixture(); f.update(mx=tensor([[.27]], True), my=tensor([[.53]], True),
                                albedo=tensor([[.34, .48, .63]], True))
        r = self.models['A2', 'atan2'].shade(**f)
        gx, gy, gk = torch.autograd.grad(r['original_L_spec'].sum(), (f['mx'], f['my'], f['albedo']), allow_unused=True)
        self.assertIsNone(gx); self.assertIsNone(gy); self.assertTrue(bool((gk > 0).all()))
        # Full acceptance: softmax mass is 1, so RGB L_spec = d/(3*M).
        assert_close(r['original_L_spec'], r['f_d'] / 96, rtol=3e-7, atol=1e-8)

    def test_conditional_and_unconditional_energy_proxies_are_distinct(self):
        f = fixture(); f.update(mx=tensor([[.9]]), my=tensor([[.9]]), illumination=white,
                                random_inputs=RandomInputs(*(tensor(v) for v in old.sample_stream(8, 32))))
        r = self.models['A1', 'original'].shade(**f)
        self.assertLess(int(r['valid_specular_counts'][0]), 32)
        assert_close(r['unconditional_reflectance_proxy'],
            r['original_f_s_sum'] * r['valid_specular_counts'] / 32, rtol=3e-7, atol=1e-8)
        self.assertTrue(torch.equal(r['specular_rgb'], r['unconditional_reflectance_proxy']))

    def test_independent_energy_diagnostic_is_read_only_and_convergence_gated(self):
        f = fixture(); f.update(mx=tensor([[.6]]), my=tensor([[.9]]), views=tensor([[0, 0, 1]]),
                                encoding=tensor([[np.cos(.62), np.sin(.62)]]), illumination=white)
        r = self.models['A1', 'original'].shade(**f)
        kwargs = dict(mx=.6, my=.9, theta=.31, albedo=[.34, .48, .63], metallic=.22, view=[0, 0, 1])
        a = metrics.integrate_unit_energy(**kwargs, quadrature=independent.hemisphere(16, 32))
        b = metrics.integrate_unit_energy(**kwargs, quadrature=independent.hemisphere(32, 64))
        before = r['original_f_s_sum'].clone()
        report = metrics.energy_diagnostics(r, a, b)
        self.assertEqual(report['mode'], 'read_only_no_loss'); self.assertEqual(report['reference_status'], 'converged')
        self.assertTrue(torch.equal(before, r['original_f_s_sum']))
        json.dumps(report, allow_nan=False)
        bad = dict(a, specular=a['specular'] + .1)
        report = metrics.energy_diagnostics(r, bad, b)
        self.assertEqual(report['reference_status'], 'unresolved'); self.assertIsNone(report['unconditional_minus_reference'])

    def test_physical_gauge_equivalences_not_finite_sample_identity(self):
        wi = independent.unit(np.array([[.2, .4, 1], [-.4, .1, 1], [.6, -.3, 1]]))
        wo = independent.unit([.3, -.1, 1])
        original = independent.kernel(wi, wo, .27, .53, .31, .1)
        for mx, my, theta in [(.27, .53, .31 + np.pi), (.53, .27, .31 + np.pi / 2)]:
            np.testing.assert_allclose(original, independent.kernel(wi, wo, mx, my, theta, .1), rtol=2e-14, atol=1e-16)
            g = metrics.gauge_metrics((mx, my, theta), (.27, .53, .31))
            self.assertLess(g['Q_relative_frobenius'], 1e-14); self.assertLess(g['major_axis_error_radians'], 1e-14)
        a = independent.kernel(wi, wo, .4, .4, 0, .1)
        b = independent.kernel(wi, wo, .4, .4, 1.1, .1)
        np.testing.assert_allclose(a, b, rtol=2e-14)

    def test_isotropic_and_near_isotropic_orientation_is_not_scored(self):
        for widths in [(.4, .4), (.4, .41)]:
            g = metrics.gauge_metrics((*widths, 1.2), (*widths, .1))
            self.assertFalse(g['orientation_identifiable']); self.assertIsNone(g['major_axis_error_radians'])
            json.dumps(g, allow_nan=False)
        g = metrics.gauge_metrics((.4, .4, .1), (.2, .7, .1))
        self.assertGreater(g['Q_relative_frobenius'], .1)
        self.assertIsNotNone(g['orientation_reason'])

    def test_defaults_and_invalid_inputs_fail_clearly(self):
        m = MaterialShaderV1('A2', 'atan2')
        self.assertEqual((m.diffuse_samples, m.specular_samples), (512, 1024))
        for args in [('A9', 'atan2'), ('A1', 'repair_zero')]:
            with self.assertRaises(SignatureError): MaterialShaderV1(*args)
        for key, value in [('mx', [[0.01]]), ('metallic', [[1.1]]), ('encoding', [[float('nan'), 1]]),
                           ('normals', [[0, 0, 2]]), ('views', [[0, 0, -1]])]:
            f = fixture(); f[key] = tensor(value)
            with self.assertRaises(SignatureError): self.models['A2', 'atan2'].shade(**f)
        f = fixture(); f['random_inputs'] = None
        with self.assertRaises(SignatureError): self.models['A2', 'atan2'].shade(**f)
        f = fixture(); d = f['random_inputs']
        f['random_inputs'] = RandomInputs(-torch.ones_like(d.u), d.v, d.azimuth)
        with self.assertRaises(SignatureError): self.models['A2', 'atan2'].shade(**f)
        f = fixture(); f['illumination'] = lambda p, d: torch.full_like(d, -1)
        with self.assertRaises(SignatureError): self.models['A2', 'atan2'].shade(**f)

    def test_no_implicit_RNG_training_GPU_or_checkpoint_use(self):
        f = fixture(); state = torch.get_rng_state().clone(); nstate = np.random.get_state()
        with patch.object(torch, 'rand', side_effect=AssertionError('Implicit RNG')), \
             patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU query')), \
             patch.object(torch.nn, 'Linear', side_effect=AssertionError('Network')), \
             patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimization')), \
             patch.object(torch, 'load', side_effect=AssertionError('Checkpoint')):
            result = self.models['A2', 'atan2'].shade(**f)
            self.assertEqual(result['random_calls'], 3)
            result = self.models['A2', 'atan2'].shade(**f, is_train=False)
            self.assertEqual(result['random_calls'], 2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        now = np.random.get_state(); np.testing.assert_array_equal(nstate[1], now[1]); self.assertEqual(nstate[2:], now[2:])

    def test_frozen_sources_prior_operators_and_evidence_are_unchanged(self):
        for stage in ('geometry', 'material'): baseline(stage, PINS[stage])
        protected = ['scientific_sources', 'audits', 'docs/evidence', 'experiments/anisotropic_estimator_v1.py',
                     'experiments/run_anisotropic_ablation_v1.py', 'docs/ANISOTROPIC_ESTIMATOR_ABLATION_V1_PROTOCOL.md']
        subprocess.run(['git', 'diff', '--exit-code', BASE, '--', *protected], cwd=ROOT, check=True, stdout=subprocess.PIPE)


if __name__ == '__main__': unittest.main()
