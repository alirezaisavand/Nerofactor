"""New small CPU fixtures; zero full scientific shader calls in tests."""
import ast
import copy
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_gauge_roundoff_v1 as m
from infrastructure.signatures import SignatureError


def observations():
    return [dict(view=i % 12, light=i // 12, split='fit') for i in range(24)]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.continuation, cls.failed, cls.identities = m.admit()

    def test_full_accepted_chain_and_failed_freeze(self):
        self.assertEqual(self.failed['status'], 'failed')
        self.assertIn('docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json', self.identities)
        self.assertIn('docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json', self.identities)
        self.assertEqual(len(self.identities), 48)
        for n, sha in m.INPUTS.items():
            self.assertEqual(m.digest(m.ROOT/n), sha)

    def test_exact_two_state_restoration_no_optimizer(self):
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer')):
            for seed in m.SEEDS:
                saved = next(r for r in self.continuation['initializations'] if r['seed'] == seed)
                z = m.prior.restore(saved)
                for arm, coords in [('P0', z), ('P1-C', m.prior.gauge(z))]:
                    archived = next(r for r in self.failed['outcomes'] if r['seed'] == seed and r['arm'] == arm)
                    self.assertEqual(m.prior.lm.accepted.describe(coords), archived['initial_state']['unconstrained'])
                    self.assertEqual(m.prior.lm.accepted.describe(m.original.physical(coords)), archived['initial_state']['physical'])
                bad = copy.deepcopy(saved)
                bad['latest_state']['unconstrained']['sha256'] = '0'*64
                with self.assertRaises(SignatureError):
                    m.prior.restore(bad)

    def test_exact_saved_array_values_and_hash(self):
        rows = [r for r in self.failed['outcomes'] if r['seed'] in m.SEEDS and r['arm'] in ('P0', 'P1-C')]
        for row in rows:
            a = m.saved_mean(row)
            self.assertEqual(m.reference.sha_array(a), row['initial_fit_prediction_sha256'])
        bad = copy.deepcopy(rows[0])
        bad['initial_fit_predictions'][0][0][0] += 1e-7
        with self.assertRaises(SignatureError):
            m.saved_mean(bad)

    def test_original_and_coupled_tape_identity(self):
        tape, ids = m.original.make_tapes(list(range(4)), 24)
        self.assertEqual(ids, self.old['tapes']['fit'])
        for seed in m.SEEDS:
            saved = next(r for r in self.failed['starting_checks'] if r['seed'] == seed)
            copied, record = m.coupling.copy_coupled_tape(tape, saved['frame_mapping']['sign'])
            self.assertEqual(record, saved['coupled_tape'])
            self.assertTrue(torch.equal(copied.u, tape.u))
            self.assertTrue(torch.equal(copied.azimuth, tape.azimuth))
            self.assertNotEqual(copied.v.data_ptr(), tape.v.data_ptr())

    def test_original_encoding_and_ideal_negation(self):
        for theta in (-1.27, -.3, .2, 1.3):
            t = torch.tensor(theta, dtype=torch.float32)
            e = m.encoding(t)
            self.assertTrue(torch.equal(e, torch.stack((torch.cos(2*t), torch.sin(2*t)))))
            p0 = np.array([.2, .6, theta, .3, .4, .5, .4], dtype=np.float32)
            p1 = p0[m.prior.PERM].copy()
            p1[2] += np.float32(math.pi/2)
            result = m.encoding_equivalence(p0, p1, e.numpy(), -e.numpy())
            self.assertTrue(result['ideal_physical_equivalence'])
            self.assertLess(result['ideal_Q_max_absolute'], 1e-14)

    def test_physical_parameter_change_rejected(self):
        p0 = np.array([.2, .6, .3, .3, .4, .5, .4], np.float32)
        p1 = p0[m.prior.PERM].copy()
        p1[6] += .01
        e = m.encoding(.3).numpy()
        with self.assertRaises(SignatureError):
            m.encoding_equivalence(p0, p1, e, -e)

    def test_actual_world_frame_Q_gauge_invariance(self):
        base = {'tangent':np.tile(np.array([[0., 1., 0.]], np.float32), (96, 1)),
                'bitangent':np.tile(np.array([[-1., 0., 0.]], np.float32), (96, 1)),
                'shade.mx':np.full((96, 1), .2, np.float32),
                'shade.my':np.full((96, 1), .6, np.float32)}
        for sign in (-1, 1):
            candidate = {'tangent':sign*base['bitangent'], 'bitangent':-sign*base['tangent'],
                         'shade.mx':base['shade.my'], 'shade.my':base['shade.mx']}
            self.assertTrue(np.array_equal(m.frame_q(base), m.frame_q(candidate)))

    def test_intervention_changes_only_encoding_and_single_call(self):
        class Shader:
            def shade(self, *args, **kwargs):
                self.seen, self.options = args, kwargs
                return dict(marker='same object')
        shader = Shader()
        args = tuple(object() for _ in range(10))
        e = torch.tensor([[-1., 0.]])
        runner = m.Invocation(shader, e)
        result = runner.shade(*args, is_train=True)
        self.assertIs(result, runner.result)
        self.assertIs(shader.seen[7], e)
        for i in range(10):
            if i != 7:
                self.assertIs(shader.seen[i], args[i])
        self.assertEqual(shader.options, dict(is_train=True))
        with self.assertRaises(SignatureError):
            runner.shade(*args)

    def test_native_invocation_preserves_encoding_object(self):
        class Shader:
            def shade(self, *args):
                self.args = args
                return {}
        shader, args = Shader(), tuple(object() for _ in range(10))
        m.Invocation(shader).shade(*args)
        for a, b in zip(shader.args, args):
            self.assertIs(a, b)

    def test_exact_baseline_replay_and_mismatch_fail_closed(self):
        mean = np.full((24, 3, 3), np.float32(.3))
        mask = np.zeros((96, 1024), bool)
        row = dict(initial_fit_predictions=mean.tolist(), initial_fit_prediction_sha256=m.reference.sha_array(mean),
                   initial_fit_MSE=m.mse32(mean[:, 2], np.zeros((24, 3), np.float32))[0],
                   initial_mask=m.prior.lm.mask_identity(mask))
        data = dict(mean=mean.copy(), **{'radiance.mask':mask})
        self.assertTrue(m.verify_baseline(data, row, np.zeros((24, 3), np.float32))['exact_RGB'])
        data['mean'][0, 0, 0] = np.nextafter(data['mean'][0, 0, 0], np.float32(1))
        with self.assertRaises(SignatureError):
            m.verify_baseline(data, row, np.zeros((24, 3), np.float32))

    def test_four_tape_order_and_component_means(self):
        # Independent small fixture for the exact production mean-before-MSE convention.
        per = np.arange(4*24*3*3, dtype=np.float32).reshape(4, 24, 3, 3)*np.float32(1e-4)
        mean = np.stack([torch.from_numpy(per[:, :, i].copy()).mean(0).numpy() for i in range(3)], 1)
        self.assertTrue(np.array_equal(mean[:, 2], np.stack([torch.tensor(per[:, i, 2].copy()).mean(0).numpy() for i in range(24)])))
        target = np.zeros((24, 3), np.float32)
        mean_mse = m.mse32(mean[:, 2], target)[0]
        mse_mean = np.mean([m.mse32(per[j, :, 2], target)[0] for j in range(4)])
        self.assertNotEqual(mean_mse, mse_mean)

    def test_float32_residual_subtraction_vs_float64(self):
        p = np.full((24, 3), np.float32(.3))
        t = np.full((24, 3), np.float32(.1))
        variants = m.mse_variants(p, t)
        self.assertEqual(variants['original_float32'], float(torch.mean((torch.from_numpy(p)-torch.from_numpy(t))**2)))
        self.assertNotEqual(variants['residual32_square64_mean64'], variants['prediction32_residual64_square64_mean64'])

    def test_residual_identity_and_contribution_accounting(self):
        p0 = np.linspace(.1, .8, 72, dtype=np.float32).reshape(24, 3)
        p1 = np.nextafter(p0, np.float32(1))
        target = np.full((24, 3), np.float32(.21))
        d = m.mse_decomposition(p0, p1, target, observations())
        self.assertTrue(d['algebra']['exact_dyadic_identity'])
        contributions = np.asarray(d['per_observation_channel_MSE_contributions'])
        self.assertAlmostEqual(float(contributions.sum()), d['algebra']['exact_dyadic_difference_as_float'], places=22)
        self.assertAlmostEqual(d['first_order_contribution']+d['quadratic_contribution'],
                               d['algebra']['exact_dyadic_difference_as_float'], places=22)
        c = d['signed_stage_contributions']
        self.assertEqual(c['total'], d['signed_MSE_differences']['original_float32'])
        self.assertEqual(len(d['largest_observation_channel_contributors']), 10)

    def test_strict_array_shape_dtype_and_hash(self):
        a = np.array([[1., .25]], np.float32)
        r = m.array_record(a, True)
        self.assertTrue(np.array_equal(m.read_array(r, (1, 2)), a))
        for key, value in [('shape', [2, 1]), ('dtype', 'float64'), ('sha256', '0'*64)]:
            bad = dict(r, **{key:value})
            with self.assertRaises(SignatureError):
                m.read_array(bad, (1, 2))

    def test_ULP_angles_and_tape_error_accounting(self):
        a = np.full((4, 2, 3), np.float32(.2))
        b = np.nextafter(a, np.float32(1))
        out = m.compare(b, a, True)
        self.assertEqual(out['max_float32_ULP_distance'], 1)
        self.assertEqual(out['differing_elements'], 24)
        self.assertEqual(len(out['per_tape']), 4)
        self.assertEqual(m.compare(np.array([0.], np.float32), np.array([-0.], np.float32))['max_float32_ULP_distance'], 0)
        dirs = np.array([[[0., 0., 1.]]]*4, np.float32)
        self.assertEqual(m.compare(dirs, dirs, True, True)['max_angle_radians'], 0.)

    def test_no_grad_and_exact_six_call_budget(self):
        report = {}
        budget = m.Budget(report)
        with self.assertRaises(SignatureError):
            budget.before(*m.CALLS[0])
        with torch.no_grad():
            for call in m.CALLS:
                budget.before(*call)
                budget.completed()
            with self.assertRaises(SignatureError):
                budget.before(101, 'P0')
        self.assertEqual(report['work']['attempted_shader_calls'], 6)
        self.assertEqual(report['work']['completed_points'], 576)
        self.assertEqual(report['work']['diffuse_directions'], 294912)
        self.assertEqual(report['work']['specular_directions'], 589824)

    def test_attempt_and_completion_are_separate(self):
        report = {}
        budget = m.Budget(report)
        with torch.no_grad():
            budget.before(*m.CALLS[0])
            with self.assertRaises(SignatureError):
                budget.before(*m.CALLS[0])
        self.assertEqual(report['work']['attempted_shader_calls'], 1)
        self.assertEqual(report['work']['completed_shader_calls'], 0)

    def test_resource_limits_fail_closed(self):
        budget = m.Budget({})
        with patch.object(m.time, 'perf_counter', return_value=budget.start+601):
            with self.assertRaises(m.RuntimeLimit):
                budget.check()
        with patch.object(m.resource, 'getrusage') as get:
            get.return_value.ru_maxrss = m.RSS_BYTES//1024+1
            with self.assertRaises(SignatureError):
                budget.check()

    def test_profile_observation_copies_without_altering_operation(self):
        def toy(x):
            y = x+torch.tensor(1.)
            return y
        observer = object.__new__(m.Capture)
        observer.data, observer.counts = {}, {}
        observer.codes = {toy.__code__:('toy', ('x', 'y'))}
        x = torch.tensor([1., 2.])
        old = sys.getprofile()
        with torch.no_grad(), observer.active():
            result = toy(x)
        self.assertIs(sys.getprofile(), old)
        self.assertTrue(torch.equal(result, torch.tensor([2., 3.])))
        self.assertEqual(observer.counts, dict(toy=1))
        x[0] = 5.
        self.assertEqual(observer.data['toy.x'][0], 1.)

    def test_capture_targets_original_code_and_accessible_locals(self):
        # Construction/inspection only: never call the scientific shade function.
        shader = m.original.MaterialShaderV1('A2', 'atan2')
        observer = m.Capture(shader)
        self.assertIn(m.shader_module.MaterialShaderV1.shade.__code__, observer.codes)
        self.assertIn(m.shader_module.decode_atan2.__code__, observer.codes)
        self.assertEqual(len(observer.codes), 5)
        for code, (_, fields) in observer.codes.items():
            self.assertTrue(set(fields) <= set(code.co_varnames))

    def test_missing_or_wrong_intermediate_shape_rejected(self):
        with self.assertRaises(SignatureError):
            m.validate_data({})
        with self.assertRaises(SignatureError):
            m.validate_data({'shade.mx':np.zeros((95, 1), np.float32)})

    def test_captured_sample_and_tape_accumulation_replay(self):
        # Full shape fixtures contain constants; no shader or sampling routine executes.
        d = np.full((96, 512, 3), np.float32(.0003))
        s = np.full((96, 1024, 3), np.float32(.0007))
        diffuse = (torch.from_numpy(d).sum(1)/512).numpy()
        specular = (torch.from_numpy(s).sum(1)/1024).numpy()
        per = np.stack((diffuse, specular, diffuse+specular), 1).reshape(4, 24, 3, 3)
        mean = np.stack([torch.from_numpy(per[:, :, i].copy()).mean(0).numpy() for i in range(3)], 1)
        data = {'radiance.diff_weighted':d, 'radiance.spec_weighted':s, 'mean':mean, 'per_tape':per}
        out = m.accumulation(data, np.zeros((24, 3), np.float32))
        self.assertTrue(out['original_sample_sums_exact'])
        self.assertTrue(out['original_four_tape_mean_exact'])
        data['mean'] = mean.copy()
        data['mean'][0, 2, 0] = np.nextafter(mean[0, 2, 0], np.float32(1))
        with self.assertRaises(SignatureError):
            m.accumulation(data, np.zeros((24, 3), np.float32))

    def test_profile_cleanup_on_exception(self):
        observer = object.__new__(m.Capture)
        observer.data, observer.counts, observer.codes = {}, {}, {}
        old = sys.getprofile()
        with self.assertRaises(RuntimeError):
            with observer.active():
                raise RuntimeError('synthetic failure')
        self.assertIs(sys.getprofile(), old)

    def test_strict_JSON_preserves_last_partial_snapshot(self):
        stream = io.StringIO()
        m.coupling.accepted.persist_report(stream, dict(status='partial', calls=[]))
        previous = stream.getvalue()
        with self.assertRaises(ValueError):
            m.coupling.accepted.persist_report(stream, dict(status='failed', value=float('nan')))
        self.assertEqual(stream.getvalue(), previous)

    def test_no_derivative_optimizer_or_source_replacement_code(self):
        source = Path(m.__file__).read_text()
        tree = ast.parse(source)
        forbidden = {'backward', 'grad', 'jacobian', 'jacrev', 'jvp', 'step', 'Adam', 'run_lm'}
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in calls))
        self.assertNotIn('FunctionType(', source)
        self.assertNotIn('setattr(', source)
        self.assertNotIn('patch.object(', source)
        self.assertIn('with torch.no_grad():', source)
        self.assertEqual(source.count('data = capture('), 1)


if __name__ == '__main__':
    unittest.main()
