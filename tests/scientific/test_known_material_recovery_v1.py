"""NEW v1 runner fixtures; never execute the real 84-observation experiment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import known_material_reference_v1 as ref
from experiments import run_known_material_recovery_v1 as run
from infrastructure.signatures import SignatureError


class KnownMaterialRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads, threads); torch.set_num_threads(1)

    def test_exact_scene_and_disjoint_split_ids(self):
        d = ref.design(); obs = d['observations']
        self.assertEqual([sum(o['split'] == name for o in obs) for name in ('fit', 'held_out', 'relight')], [24, 24, 36])
        self.assertEqual(len({o['id'] for o in obs}), 84)
        self.assertEqual(obs[0]['id'], 'v_train_0/l0'); self.assertEqual(obs[23]['id'], 'v_train_5/l3')
        self.assertEqual(obs[24]['id'], 'v_hold_0/l0'); self.assertEqual(obs[-1]['id'], 'v_hold_5/l6')
        self.assertEqual(d['base_tangent'], [0, 1, 0]); self.assertEqual(d['base_bitangent'], [-1, 0, 0])
        np.testing.assert_allclose(np.linalg.norm([v['direction'] for v in d['views']], axis=1), 1., atol=1e-15)

    def test_private_initialization_order_roundtrip_and_all_starts_preserved(self):
        rows = run.initializations(); self.assertEqual([r['seed'] for r in rows], list(range(101, 106)))
        for row in rows:
            p = np.array(row['initial_physical_float64']); z = torch.tensor(row['initial_unconstrained_float32'])
            np.testing.assert_allclose(run.physical(z).numpy(), p, rtol=2e-7, atol=1e-7)
            self.assertTrue(np.all((p[:2] >= .15) & (p[:2] <= .75)))
            self.assertEqual(row['status'], 'not_started'); self.assertIsNone(row['final'])
        self.assertEqual(rows, run.initializations())

    def test_fixed_tapes_no_implicit_RNG(self):
        a = torch.get_rng_state().clone(); b = np.random.get_state()
        tape, identities = run.make_tapes([0, 1], 2); again, ids = run.make_tapes([0, 1], 2)
        self.assertEqual(identities, ids)
        for x, y in zip((tape.u, tape.v, tape.azimuth), (again.u, again.v, again.azimuth)):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a, torch.get_rng_state())); np.testing.assert_array_equal(b[1], np.random.get_state()[1])
        self.assertEqual(tape.u.shape, (4, 1024))

    def test_numpy_and_torch_world_lighting_agree(self):
        d = ref.design(); selected = [d['observations'][i] for i in (0, 1, 48, 51)]
        tapes, _ = run.make_tapes([0], len(selected)); batch = run.ObservationBatch(d, selected, tapes, 1)
        wi = ref.reference.unit(np.array([[1., 2, 3], [-1., .2, 1], [.5, -.8, 1]]))
        directions = torch.tensor(np.tile(wi, (len(selected), 1, 1)), dtype=torch.float32)
        torch_lights = batch.illumination(torch.zeros_like(directions), directions).numpy()
        numpy_lights = ref.lighting(wi, d['lights'])
        for i, obs in enumerate(selected):
            np.testing.assert_allclose(torch_lights[i], np.repeat(numpy_lights[:, obs['light']:obs['light']+1], 3, -1), rtol=3e-6, atol=1e-7)

    def test_seed_major_batch_averages_before_residual(self):
        class ToyShader:
            def shade(self, positions, normals, views, mx, my, albedo, metallic, encoding, illumination, draws, is_train):
                d = draws.u[:, :1].expand(-1, 3); s = draws.v[:, :1].expand(-1, 3)
                return {'frame_orthonormal': torch.ones(len(d), dtype=torch.bool),
                        'diffuse_rgb': d, 'specular_rgb': s, 'linear_rgb': d+s}
        scene = ref.design(); tapes, _ = run.make_tapes([0, 1], 2)
        batch = run.ObservationBatch(scene, scene['observations'][:2], tapes, 2)
        values = batch(ToyShader(), torch.tensor(ref.TRUTH, dtype=torch.float32))
        expected = (tapes.u[:, 0] + tapes.v[:, 0]).reshape(2, 2).mean(0)
        np.testing.assert_array_equal(values[:, 2].numpy(), expected[:, None].expand(-1, 3).numpy())

    def test_factored_reference_matches_independent_kernel_in_each_RGB_channel(self):
        grids = ref.make_grids(((8, 16), (16, 32))); observations = [ref.design()['observations'][i] for i in (0, 1, 4)]
        budget = {}; result = ref.render(ref.TRUTH, observations, grids, budget)
        p = ref.TRUTH; d = p[3:6] * (1-p[6]); f0 = .04*(1-p[6])+p[6]*p[3:6]
        for level, name in zip(grids, ('coarse', 'fine')):
            for i, o in enumerate(observations):
                wi, weights = level['directions'], level['weights']; light = level['lighting'][:, o['light']]
                wo = np.array(ref.design()['views'][o['view']]['direction'])
                expected = [np.sum(weights * wi[:, 2] * light * ref.reference.kernel(wi, wo, p[0], p[1], p[2], f)) for f in f0]
                np.testing.assert_allclose(result[name][i, 1], expected, rtol=2e-14, atol=1e-16)
                np.testing.assert_allclose(result[name][i, 0], d * level['diffuse_integral'][o['light']], rtol=1e-14)
        self.assertEqual(budget['reference_direction_evaluations'], 2*(8*16+16*32))

    def test_reference_convergence_never_silently_promotes_total(self):
        grids = ref.make_grids(((2, 4), (4, 8))); observations = ref.design()['observations'][:1]
        r = ref.render(ref.TRUTH, observations, grids, {})
        self.assertFalse(r['passed']); self.assertFalse(r['converged'].all())
        np.testing.assert_array_equal(r['limit'], ref.ATOL + ref.RTOL * abs(r['fine']))
        json.dumps(ref.serial_reference(r), allow_nan=False)

    def test_jacobian_rank_threshold_and_stability_fail_closed(self):
        m = np.tile(np.diag([1., .9, .8, .7, .6, .5, .4]), (2, 2, 1, 1))
        self.assertTrue(ref.jacobian_gate(m)['passed'])
        low = m.copy(); low[:, :, -1, -1] = 1e-5
        self.assertFalse(ref.jacobian_gate(low)['passed'])
        unstable = m.copy(); unstable[0, 0, 0, 0] += 1
        self.assertFalse(ref.jacobian_gate(unstable)['passed'])
        boundary = m.copy(); boundary[:, :, -1, -1] = 1e-4
        self.assertTrue(ref.jacobian_gate(boundary)['passed'])
        self.assertFalse(ref.jacobian_gate(np.zeros_like(m))['passed'])

    def test_independent_finite_difference_parameter_scales_and_order(self):
        observations = [dict(split='fit', view=0, light=0)] * 3
        a = np.eye(9, 7) + .01
        def render(p, obs, grids, budget, check):
            value = np.zeros((3, 3, 3)); value[:, 2] = (a @ p).reshape(3, 3)
            return {'passed': True, 'coarse': value, 'fine': value}
        with patch.object(ref, 'render', side_effect=render) as called:
            result = ref.identifiability([], observations, {}, lambda: None, lambda i, j: None)
        self.assertEqual(called.call_count, 28); self.assertTrue(result['passed'])
        np.testing.assert_allclose(np.array(result['jacobians'])[0, 1], a * ref.SCALES, rtol=1e-10, atol=1e-10)

    def test_failed_perturbation_stops_without_next_column(self):
        with patch.object(ref, 'render', return_value={'passed': False}) as called:
            r = ref.identifiability([], [dict(split='fit')], {}, lambda: None, lambda i, j: None)
        self.assertFalse(r['passed']); self.assertEqual(called.call_count, 1)

    def test_no_optimizer_or_shader_when_either_mandatory_gate_fails(self):
        for reference_passed in (False, True):
            fake = {'passed': reference_passed, 'fine': np.zeros((84, 3, 3))}
            report = {'status': 'running'}
            with patch.object(ref, 'make_grids', return_value=[]), patch.object(ref, 'render', return_value=fake), \
                 patch.object(ref, 'identifiability', return_value={'passed': False}), \
                 patch.object(run, 'MaterialShaderV1', side_effect=AssertionError('Premature shader')), \
                 patch.object(torch.optim, 'Adam', side_effect=AssertionError('Premature optimizer')):
                run.run(report, lambda: None)
            self.assertTrue(report['status'].startswith('stopped_'))
            self.assertEqual(report['budget']['optimizer_returns'], 0)
            self.assertTrue(all(r['status'] == 'not_started' for r in report['initializations']))

    def test_toy_optimizer_stops_at_exact_250_returns_without_extra_forward(self):
        class ToyBatch:
            n, seeds = 1, 1
            def __call__(self, shader, p):
                self.calls = getattr(self, 'calls', 0) + 1
                value = p[:3].reshape(1, 3)
                return torch.stack([value * 0, value * 0, value], 1)
        row = run.initializations()[0]; budget = {}; batch = ToyBatch()
        result = run.fit_start(row, None, batch, np.zeros((1, 3)), budget, lambda: None, lambda: None)
        self.assertEqual(row['completed_updates'], 250); self.assertEqual(budget['optimizer_returns'], 250)
        self.assertEqual(batch.calls, 250); self.assertEqual(row['optimizer_final_state']['step'], 250)
        self.assertEqual(row['history'][-1]['loss_at_update_count'], 249)
        self.assertTrue(torch.isfinite(result).all()); self.assertEqual(row['optimizer_effective']['lr'], .01)

    def test_nonfinite_failure_preserves_start_and_stops_optimizer(self):
        class BadBatch:
            n, seeds = 1, 1
            def __call__(self, shader, p): return torch.full((1, 3, 3), float('nan'))
        row = run.initializations()[0]
        with self.assertRaises(SignatureError): run.fit_start(row, None, BadBatch(), np.zeros((1, 3)), {}, lambda: None, lambda: None)
        self.assertEqual(row['status'], 'failed'); self.assertEqual(row['completed_updates'], 0)
        self.assertIsNotNone(row['last_physical'])

    def test_criteria_and_gauge_metrics_are_exactly_declared(self):
        p = ref.TRUTH.copy(); p[0], p[1], p[2] = p[1], p[0], p[2] + np.pi/2
        errors = run.recovery_errors(p)
        m = {'held_out': {'total': {'relative_L2': .01}}}
        self.assertTrue(run.assess_success(errors, m)['all_passed'])
        m['held_out']['total']['relative_L2'] = .010001
        self.assertFalse(run.assess_success(errors, m)['all_passed'])
        expected = {'Q_relative': .05, 'orientation_degrees': 5., 'max_albedo_absolute': .02,
                    'metallic_absolute': .03, 'held_out_relative_L2': .01}
        self.assertEqual(run.SETTINGS['success'], expected)
        self.assertEqual(run.radiance_errors(np.zeros((1, 3)), np.zeros((1, 3)))['relative_L2'], 0)

    def test_deadline_and_budgets_fail_without_retry(self):
        with patch.object(run.time, 'perf_counter', side_effect=[0., 1800.]):
            deadline = run.Deadline()
            with self.assertRaises(run.RuntimeLimit): deadline()
        with self.assertRaises(SignatureError): run.use_points({'fit_shader_points': 120000}, 'fit', 1)
        with self.assertRaises(SignatureError): run.use_points({'evaluation_shader_points': 1680}, 'evaluation', 1)

    def test_fresh_output_collision_never_runs_an_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'; path.write_text('preserved')
            with patch.dict(run.os.environ, run.REQUIRED_ENV), patch.object(sys, 'argv', ['fixture', '--output', str(path)]), \
                 patch.object(run, 'run', side_effect=AssertionError('Collision must precede run')):
                with self.assertRaises(FileExistsError): run.main()
            self.assertEqual(path.read_text(), 'preserved')


if __name__ == '__main__': unittest.main()
