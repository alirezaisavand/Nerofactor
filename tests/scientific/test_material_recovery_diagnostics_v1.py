"""NEW tiny diagnostic fixtures; no real shader/reference run or optimization."""
import ast
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_recovery_trajectories_v1 as offline
from experiments import material_recovery_diagnostics_v1 as diag
from infrastructure.signatures import SignatureError


class ToyShader:
    def shade(self, positions, normals, views, mx, my, albedo, metallic, encoding, illumination, draws, is_train):
        # Diagnostic plumbing fixture, deliberately not the scientific shader.
        color = albedo*(1-metallic) + .1*metallic + .2*mx + .3*my + .01*encoding[:, 1:2]
        return {'diffuse_rgb': .4*color, 'specular_rgb': .6*color, 'linear_rgb': color,
                'frame_orthonormal': torch.ones(len(color), dtype=torch.bool),
                'specular_directions': torch.ones((len(color), 1, 3))}


def tiny_diagnostic():
    d = diag.Diagnostic.__new__(diag.Diagnostic)
    scene = diag.ref.design(); obs = scene['observations'][:1]
    tapes, _ = diag.previous.make_tapes([0, 1, 2, 3], 1)
    d.fit_batch = diag.previous.ObservationBatch(scene, obs, tapes, 4)
    d.fit_obs = obs; d.targets = np.zeros((1, 3, 3)); d.target = torch.zeros((1, 3))
    d.shader = ToyShader(); d.deadline = lambda: None; d.persist = lambda: None
    d.budget = dict(shader_calls=0, fit_points=0, evaluation_points=0,
                    specular_directions=0, diffuse_directions=0, new_reference_directions=0)
    return d


class RecoveryDiagnosticsTests(unittest.TestCase):
    def test_offline_module_never_imports_torch_or_shader(self):
        tree = ast.parse(Path(offline.__file__).read_text())
        names = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('torch' in n or 'material_shader' in n for n in names))

    def test_correlations_do_not_imply_causality_and_constants_are_null(self):
        self.assertIsNone(offline.correlation([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(offline.correlation([1, 2, 3], [3, 2, 1]), -1.)

    def test_branch_brackets_respect_periodic_boundaries(self):
        points = [{'id': str(i), 'update': i*10, 'physical': [.2, .5, t, .3, .4, .5, .2]}
                  for i, t in enumerate([1.5, 1.6, 1.5])]
        result = offline.branch_brackets(points)
        self.assertEqual(len(result), 2); self.assertAlmostEqual(result[0]['boundary_theta'], np.pi/2)
        self.assertEqual(result[1]['left_update'], 10)

    def test_component_compensation_is_measured_not_recovery(self):
        x = np.zeros((2, 3, 3)); x[:, 0] = -.2; x[:, 1] = .2
        r = offline.compensation(x, np.zeros_like(x))
        self.assertEqual(r['cancellation_fraction'], 1.)
        self.assertAlmostEqual(r['diffuse_specular_residual_cosine'], -1.)
        self.assertEqual(r['component_RMSE'][2], 0)

    def test_hash_mismatch_precedes_parse_or_numerics(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'wrong.json'; p.write_text('not json')
            with self.assertRaisesRegex(SignatureError, 'SHA-256'): offline.admit(p, offline.EVIDENCE_SHA)

    def test_alignment_near_zero_has_no_fake_cosine_or_relative_error(self):
        x = diag.alignment([1, 0], [0, 0])
        self.assertIsNone(x['cosine']); self.assertIsNone(x['relative_difference_to_b'])
        self.assertEqual(x['absolute_difference'], 1.)
        x = diag.alignment([1, 0], [2, 0]); self.assertEqual(x['cosine'], 1.)

    def test_physical_chain_matches_autograd_without_update(self):
        p = diag.ref.TRUTH; z = torch.tensor(diag.inverse_coordinates(p), requires_grad=True)
        y = diag.previous.physical(z)
        derivative = torch.autograd.functional.jacobian(diag.previous.physical, z)
        np.testing.assert_allclose(derivative.numpy(), np.diag(diag.chain_diagonal(y.detach().numpy())), rtol=2e-7, atol=2e-8)

    def test_continuous_objective_gradient_uses_scaled_columns(self):
        target = np.zeros((3, 3)); prediction = np.arange(9).reshape(3, 3)*.01
        j = np.eye(9, 7)
        actual = diag.objective_gradient(prediction, target, j*diag.ref.SCALES)
        np.testing.assert_allclose(actual, 2*j.T@prediction.reshape(-1)/9, rtol=1e-14)

    def test_masks_record_both_transition_directions_and_hash(self):
        a = np.zeros((4, 8), bool); b = a.copy(); a[0, 0] = True; b[1, 2] = True
        r = diag.transitions(a, b)
        self.assertEqual((r['flipped_total'], r['valid_to_rejected'], r['rejected_to_valid']), (2, 1, 1))
        self.assertEqual(diag.mask_summary(a)['sha256_packbits'], hashlib.sha256(np.packbits(a).tobytes()).hexdigest())
        with self.assertRaises(SignatureError): diag.transitions(a, b[:, :4])

    def test_forward_error_keeps_source_mean_and_four_tape_variation(self):
        values = np.arange(4*2*3*3).reshape(4, 2, 3, 3)*.001
        mean = values.astype(np.float32).mean(0).astype(np.float64)
        result = diag.residual_report(values, np.zeros((2, 3, 3)), [{'light': 0}, {'light': 1}], mean)
        np.testing.assert_array_equal(result['mean_predictions'], mean)
        self.assertEqual(result['mean_precision'], 'source Torch float32')
        self.assertEqual(len(result['individual_tape_MSE']), 4)
        self.assertIn('constant', result['groups']); json.dumps(result, allow_nan=False)

    def test_point_cap_fails_before_any_shader_call(self):
        d = tiny_diagnostic(); d.budget['fit_points'] = diag.MAX_FIT_POINTS
        with patch.object(d.shader, 'shade', side_effect=AssertionError('Beyond cap')):
            with self.assertRaises(SignatureError): d.forward(torch.tensor(diag.ref.TRUTH, dtype=torch.float32))

    def test_reference_cap_and_failed_reference_preserve_evidence(self):
        d = tiny_diagnostic(); grids = [{'directions': [1]}, {'directions': [1]}]; store = []
        d.budget['new_reference_directions'] = diag.MAX_REFERENCE
        with patch.object(diag.ref, 'render', side_effect=AssertionError('Beyond cap')):
            with self.assertRaises(SignatureError): d.independent(diag.ref.TRUTH, grids, store)
        d.budget['new_reference_directions'] = 0
        with patch.object(diag.ref, 'render', return_value={'passed': False}):
            with self.assertRaises(diag.ReferenceFailure): d.independent(diag.ref.TRUTH, grids, store)
        self.assertEqual(len(store), 1); self.assertFalse(store[0]['passed'])

    def test_diagnostic_gradients_and_fd_never_construct_optimizer_or_network(self):
        d = tiny_diagnostic()
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer')), \
             patch.object(torch.nn, 'Linear', side_effect=AssertionError('Network')), \
             patch.object(torch, 'load', side_effect=AssertionError('Checkpoint')):
            row, mask = d.state('toy', diag.ref.TRUTH)
            d.local_finite_differences(row, mask)
        self.assertEqual(len(row['finite_difference_probes']), 28)
        self.assertEqual(d.budget['shader_calls'], 29)
        for gradient in row['finite_objective_FD']:
            self.assertGreater(gradient['autograd_vs_FD']['cosine'], .999)
        self.assertTrue(all(r['mask_transition_from_center']['flipped_total'] == 0 for r in row['finite_difference_probes']))

    def test_latent_gradient_conversion_and_saved_coordinate_identity(self):
        d = tiny_diagnostic(); z = diag.inverse_coordinates(diag.ref.TRUTH)
        physical = diag.previous.physical(torch.tensor(z)).tolist()
        row, _ = d.state('toy_latent', physical, z)
        self.assertEqual(row['unconstrained_gradient_kind'], 'autograd')
        self.assertLess(row['gradient_conversion']['relative_difference_to_b'], 1e-6)
        wrong = physical.copy(); wrong[0] += .01
        with self.assertRaises(SignatureError): d.state('bad', wrong, z)

    def test_branch_probe_grid_changes_only_declared_theta(self):
        cases = diag.branch_cases(diag.ref.TRUTH, diag.ref.TRUTH)
        self.assertEqual(len(cases), 20)
        for case in cases:
            self.assertIn(case['center'], (-np.pi/2, np.pi/2))
            self.assertIn(abs(case['offset']), (0., 1e-4*np.pi, 5e-5*np.pi))
            np.testing.assert_array_equal(np.array(case['physical'])[[0, 1, 3, 4, 5, 6]], diag.ref.TRUTH[[0, 1, 3, 4, 5, 6]])

    def test_no_optimizer_update_calls_in_diagnostic_source(self):
        tree = ast.parse(Path(diag.__file__).read_text())
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        forbidden = [f.attr for f in calls if isinstance(f, ast.Attribute) and f.attr in ('Adam', 'SGD', 'step', 'fit_start', 'backward')]
        self.assertEqual(forbidden, [])

    def test_output_collision_never_executes_diagnostic(self):
        root = offline.ROOT
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'existing.json'; target.write_text('preserve')
            args = ['diagnostic', '--input', str(root/'docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json'), '--sha256', offline.EVIDENCE_SHA,
                    '--offline', str(root/'docs/evidence/MATERIAL_RECOVERY_TRAJECTORIES_V1.json'), '--output', str(target)]
            handler = signal.getsignal(signal.SIGTERM)
            with patch.dict(diag.os.environ, diag.previous.REQUIRED_ENV), patch.object(sys, 'argv', args), \
                 patch.object(diag, 'execute', side_effect=AssertionError('Collision')):
                with self.assertRaises(FileExistsError): diag.main()
            self.assertEqual(target.read_text(), 'preserve'); self.assertEqual(handler, signal.getsignal(signal.SIGTERM))


if __name__ == '__main__': unittest.main()
