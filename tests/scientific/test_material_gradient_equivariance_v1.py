"""New small AD/chain-rule/FD fixtures, never a prior scientific run."""
import ast
import copy
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_gradient_equivariance_v1 as audit
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs
from infrastructure.signatures import SignatureError


class TinyBatch:
    n, seeds = 3, 4
    observations = [dict(view=i, light=i, split='fit') for i in range(3)]

    def __init__(self, tape=None):
        self.tapes = tape or RandomInputs(torch.linspace(.01, .99, 192).reshape(12, 16),
            torch.linspace(.003, .997, 192).reshape(12, 16), torch.linspace(.01, .99, 12).reshape(12, 1, 1))
        self.views = torch.tensor([[.2, -.1, np.sqrt(.95)], [-.3, .4, np.sqrt(.75)],
                                  [.5, .3, np.sqrt(.66)]], dtype=torch.float32).repeat(4, 1)

    def __call__(self, shader, p):
        r = shader.shade(torch.zeros(12, 3), torch.tensor([[0., 0., 1.]]).repeat(12, 1), self.views,
            p[:1].reshape(1, 1).expand(12, 1), p[1:2].reshape(1, 1).expand(12, 1),
            p[3:6].reshape(1, 3).expand(12, 3), p[6:].reshape(1, 1).expand(12, 1),
            torch.stack([torch.cos(2*p[2]), torch.sin(2*p[2])]).reshape(1, 2).expand(12, 2),
            lambda pts, dirs: (.1+.9*dirs[..., 2:3].clamp(min=0)**4).expand_as(dirs), self.tapes, is_train=True)
        return torch.stack([r[k].reshape(4, 3, 3).mean(0) for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], 1)


def tiny_case(theta=.31, coupled=False):
    shader = MaterialShaderV1('A2', 'atan2', 8, 16)
    p0 = audit.reference.TRUTH.copy()
    p0[2] = theta
    p1 = audit.coupling.accepted.transform(p0, 'P1')
    a, b = [torch.tensor(p, dtype=torch.float32) for p in (p0, p1)]
    batch = TinyBatch()
    mapping = audit.coupling.frame_sign(*[audit.coupling.decoded_frame(shader, p) for p in (a, b)])
    if coupled:
        tape, _ = audit.coupling.copy_coupled_tape(batch.tapes, mapping['sign'])
        batch = TinyBatch(tape)
    report = {}
    budget = audit.Budget(report)
    measure = audit.Measurement(shader, batch, [dict(fixture=True)]*4, 'fit', budget)
    p = b if coupled else a
    with torch.no_grad():
        measure(p, 'baseline')
    base = measure.latest
    row = {}
    j = audit.build_jacobian(measure, p, base, row, lambda: None)
    return measure, p, base, row, j, mapping


def analytic(p):
    c, s = torch.cos(p[2]), torch.sin(p[2])
    a, b = p[0]**2, p[1]**2
    return torch.stack([a*c*c+b*s*s, (a-b)*s*c, a*s*s+b*c*c,
        p[3]*(1-p[6]), p[4]*(1-p[6]), p[5]*(1-p[6]), p[3]*p[6], p[4]*p[6], p[5]*p[6]])


def fd_snapshot(p, mask=None):
    value = np.full((1, 3, 3), float(p[0])**2, np.float32)
    mask = np.ones((4, 1, 16), bool) if mask is None else mask
    summary = audit.coupling.accepted.mask_summary(mask)
    summary['packed_hex'] = np.packbits(mask).tobytes().hex()
    return dict(mean_predictions=audit.array(value), actual_physical_float32=audit.array(np.asarray(p, np.float32)), masks=summary)


class MaterialGradientEquivarianceTests(unittest.TestCase):
    def test_permutation_and_Q_tensor_invariance(self):
        p = audit.reference.TRUTH
        q = audit.coupling.accepted.transform(p, 'P1')
        P = audit.permutation()
        self.assertEqual(P.shape, (7, 7))
        np.testing.assert_array_equal(P@P, np.eye(7))
        np.testing.assert_array_equal(P.T, P)
        np.testing.assert_allclose(audit.coupling.accepted.tensor_q(p), audit.coupling.accepted.tensor_q(q), atol=5e-14, rtol=5e-12)
        delta = np.arange(7, dtype=float)
        np.testing.assert_array_equal(P@delta, delta[audit.PERMUTATION])

    def test_independent_analytical_J_and_gradient_chain_rule_orientations(self):
        p = torch.tensor(audit.reference.TRUTH, dtype=torch.float64)
        q = p.clone()
        q[:2] = p[:2].flip(0)
        q[2] += math.pi/2
        js = []
        for point in (p, q):
            report = {}
            budget = audit.Budget(report)
            columns = []
            for k in range(7):
                v = torch.zeros_like(point)
                v[k] = 1
                _, col = audit.derivative.directional_jvp(analytic, point, v, report['work'])
                columns.append(col.numpy())
            js.append(np.stack(columns, -1))
        np.testing.assert_allclose(js[1]@audit.permutation(), js[0], atol=1e-14, rtol=1e-12)
        target = np.linspace(.01, .09, 9)
        grads = [2*j.T@(analytic(point).numpy()-target)/9 for j, point in zip(js, (p, q))]
        np.testing.assert_allclose(audit.permutation().T@grads[1], grads[0], atol=1e-14, rtol=1e-12)

    def test_tiny_real_A2_physical_Jacobians_both_signed_frames(self):
        signs = []
        for theta in (.31, -.8):
            m0, _, _, r0, j0, mapping = tiny_case(theta)
            m1, _, _, r1, j1, _ = tiny_case(theta, True)
            comparison = audit.jacobian_comparison(j1, j0)
            signs.append(mapping['sign'])
            self.assertEqual(j0.shape, (3, 3, 3, 7))
            self.assertEqual(j0.dtype, np.float32)
            self.assertTrue(all(v['passed'] for v in comparison.values()))
            self.assertEqual(m0.budget.report['work']['JVP_completed'], 7)
            self.assertEqual(m1.budget.report['work']['reverse_completed'], 14)
            self.assertEqual(r0['physical_jacobians']['total']['array']['shape'], [9, 7])
            self.assertEqual(r1['physical_jacobians']['diffuse']['array']['shape'], [9, 7])
        self.assertEqual(set(signs), {-1, 1})

    def test_original_mean_MSE_direct_gradient_matches_residual_J(self):
        measure, p, base, _, j, _ = tiny_case()
        target = torch.full((3, 3), .1)
        record = audit.direct_gradient(measure, p, base, j, target)
        self.assertTrue(record['construction_check']['passed'])
        self.assertEqual(record['objective_float32'], float(torch.mean((base['mean'][:, 2]-target)**2)))
        self.assertEqual(measure.budget.report['work']['direct_gradient_reverse_completed'], 1)

    def test_wrong_J_is_mandatory_gradient_construction_failure(self):
        measure, p, base, _, j, _ = tiny_case()
        with self.assertRaisesRegex(SignatureError, 'Mandatory direct MSE'):
            audit.direct_gradient(measure, p, base, np.zeros_like(j), torch.zeros(3, 3))

    def test_saved_optimizer_coordinate_reconstruction_and_chain(self):
        z = torch.tensor([-.8, .3, .31, -.4, .1, .8, -.6], dtype=torch.float32)
        p0 = audit.original.physical(z)
        p1 = p0[audit.PERMUTATION].clone()
        p1[2] = float(p0[2])+math.pi/2
        saved = {'final_state': {'unconstrained': {**audit.array(z.numpy()), 'dtype': 'torch.float32', 'device': 'cpu'}}}
        report = {}
        budget = audit.Budget(report)
        r = audit.latent_coordinates(p0, p1, saved, budget)
        for item in r['representations'].values():
            self.assertTrue(item['exact_physical_reconstruction'])
        d0, d1 = [audit.decode_array(r['representations'][k]['dp_dz_diagonal_float32']) for k in ('P0', 'P1')]
        np.testing.assert_array_equal(d1[audit.PERMUTATION], d0)
        np.testing.assert_allclose(d0[:2], .96*torch.sigmoid(z[:2]).numpy()*(1-torch.sigmoid(z[:2]).numpy()), atol=3e-8)
        self.assertEqual(d0[2], 1.)
        self.assertEqual(report['work']['coordinate_transform_reverse_completed'], 2)

    def test_truth_inverse_roundtrip_cannot_replace_direct_physical_state(self):
        p0 = torch.tensor(audit.reference.TRUTH, dtype=torch.float32)
        p1 = torch.tensor(audit.coupling.accepted.transform(audit.reference.TRUTH, 'P1'), dtype=torch.float32)
        before = p0.clone()
        report = {}
        with patch.object(audit.diagnostics, 'inverse_coordinates', return_value=np.zeros(7, np.float32)):
            r = audit.latent_coordinates(p0, p1, None, audit.Budget(report))
        for item in r['representations'].values():
            self.assertFalse(item['valid'])
            self.assertGreater(item['max_physical_difference'], 0)
        self.assertTrue(torch.equal(p0, before))
        self.assertFalse(audit.latent_result(np.zeros((3, 3, 3, 7)), None, r['representations']['P0'])['valid'])
        self.assertEqual(report['work']['coordinate_transform_reverse_sweeps'], 0)

    def test_optimizer_coordinate_chain_matches_direct_composed_renderer_AD(self):
        shader, batch = MaterialShaderV1('A2', 'atan2', 8, 16), TinyBatch()
        z = torch.tensor([-.8, .3, .31, -.4, .1, .8, -.6], dtype=torch.float32, requires_grad=True)
        value = batch(shader, audit.original.physical(z))
        gz, = torch.autograd.grad(torch.mean(value[:, 2]**2), z)
        p = audit.original.physical(z).detach().requires_grad_()
        value_p = batch(shader, p)
        gp, = torch.autograd.grad(torch.mean(value_p[:, 2]**2), p)
        z2 = z.detach().clone().requires_grad_()
        dp, = torch.autograd.grad(audit.original.physical(z2), z2, grad_outputs=torch.ones(7))
        np.testing.assert_allclose(gz.numpy(), (gp*dp).numpy(), atol=audit.CONSTRUCTION_ATOL, rtol=audit.CONSTRUCTION_RTOL)

    def test_full_tiny_FD_path_counts_24_calls_with_fixed_coupled_tapes(self):
        m0, _, b0, r0, _, _ = tiny_case()
        m1, _, b1, r1, _, _ = tiny_case(coupled=True)
        cases = {}
        for label, m, b, r in [('P0', m0, b0, r0), ('P1-C', m1, b1, r1)]:
            cases[label] = dict(physical_jacobians=r['physical_jacobians'], forward=audit.snapshot(b, m.ids),
                               branch=audit.branch_status(b['physical'][2]))
        row, report = dict(fit=dict(cases=cases)), {}
        budget = audit.Budget(report)
        with patch.object(audit.coupling, 'frame_sign', side_effect=AssertionError('No adaptive sign')):
            audit.finite_differences(m0.shader, {'P0': (m0.batch, m0.ids), 'P1-C': (m1.batch, m1.ids)},
                                    row, report, budget, lambda: None)
        self.assertEqual(report['work']['categories']['finite_difference'], 24)
        self.assertEqual(report['work']['completed_forwards'], 24)
        self.assertEqual(report['work']['shader_points'], 24*12)
        for arm in row['finite_differences'].values():
            for item in arm.values():
                self.assertEqual([r['nominal_step'] for r in item['steps']], list(audit.FD_STEPS))
                self.assertTrue(all(r['status'] == 'complete' for r in item['steps']))
        json.dumps(row, allow_nan=False)

    def test_corrupt_saved_coordinate_rejected_without_optimizer(self):
        report = {}
        p = torch.tensor(audit.reference.TRUTH, dtype=torch.float32)
        r = {'final_state': {'unconstrained': {'dtype': 'torch.float32', 'device': 'cpu', 'shape': [7], 'values': [0.]*7, 'sha256': 'wrong'}}}
        with self.assertRaisesRegex(SignatureError, 'coordinate hash'):
            audit.latent_coordinates(p, p, r, audit.Budget(report))

    def test_fixed_coupled_tapes_and_originals_stay_constant_during_AD(self):
        measure, p, _, _, _, _ = tiny_case(coupled=True)
        before = [t.clone() for t in (measure.batch.tapes.u, measure.batch.tapes.v, measure.batch.tapes.azimuth)]
        with patch.object(audit.coupling, 'frame_sign', side_effect=AssertionError('No sign decision inside derivative')), \
             patch.object(audit.coupling, 'copy_coupled_tape', side_effect=AssertionError('No tape recreation inside derivative')):
            x = p.detach().clone().requires_grad_()
            value = measure(x, 'direct_gradient')
            torch.autograd.grad(value[:, 2].sum(), x)
        for t, saved in zip((measure.batch.tapes.u, measure.batch.tapes.v, measure.batch.tapes.azimuth), before):
            self.assertTrue(torch.equal(t, saved))
            self.assertFalse(t.requires_grad)
            self.assertIsNone(t.grad)

    def test_AD_failure_has_no_FD_fallback_and_keeps_completed_columns(self):
        shader = MaterialShaderV1('A2', 'atan2', 8, 16)
        report = {}
        measure = audit.Measurement(shader, TinyBatch(), [], 'fit', audit.Budget(report))
        p = torch.tensor(audit.reference.TRUTH, dtype=torch.float32)
        with torch.no_grad():
            measure(p, 'baseline')
        base = measure.latest
        old = audit.derivative.directional_jvp
        counter = [0]
        def fail_second(*args):
            counter[0] += 1
            if counter[0] == 2:
                raise RuntimeError('fixture unsupported AD')
            return old(*args)
        row, persisted = {}, []
        with patch.object(audit.derivative, 'directional_jvp', side_effect=fail_second), \
             patch.object(audit, 'finite_differences', side_effect=AssertionError('No alternative derivative')):
            with self.assertRaisesRegex(RuntimeError, 'unsupported AD'):
                audit.build_jacobian(measure, p, base, row, lambda: persisted.append(json.dumps(row, allow_nan=False)))
        self.assertEqual(len(row['partial_columns']), 1)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(report['work']['jacobians_completed'], 0)

    def test_accepted_forward_identity_and_AD_primal_mismatch_fail_closed(self):
        measure, _, base, _, _, _ = tiny_case()
        archived = audit.snapshot(base, measure.ids)
        archived['masks'].pop('packed_hex')
        self.assertEqual(audit.verify_forward(base, archived, measure.ids)['mean_predictions'], archived['mean_predictions'])
        for key in ('mean_predictions', 'per_tape_predictions', 'actual_physical_float32', 'masks', 'tape_identities'):
            wrong = copy.deepcopy(archived)
            wrong[key] = {}
            with self.assertRaises(SignatureError):
                audit.verify_forward(base, wrong, measure.ids)
        wrong = dict(base, mean=base['mean']+1)
        with self.assertRaisesRegex(SignatureError, 'AD primal'):
            audit.verify_primal(wrong, base)

    def test_physical_FD_steps_stability_and_correct_transformed_coordinate(self):
        p = np.asarray(audit.reference.TRUTH, np.float32)
        base = fd_snapshot(p)
        records = []
        for h in audit.FD_STEPS:
            a, b = p.astype(float).copy(), p.astype(float).copy()
            a[0] += h
            b[0] -= h
            r = audit.fd_record(fd_snapshot(a), fd_snapshot(b), base, audit.branch_status(p[2]), 0, h,
                np.full((1, 3, 3), 2*float(p[0])))
            self.assertFalse(r['branch_crossed'] or r['masks_changed'])
            self.assertTrue(all(v['passed'] for v in r['agreement_to_AD'].values()))
            self.assertEqual(r['denominator_used'], 2*h)
            records.append(audit.decode_array(r['derivative']))
        self.assertTrue(audit.errors(*records, audit.FD_ATOL, audit.FD_RTOL, audit.FD_FLOOR)['passed'])
        self.assertEqual([audit.PERMUTATION[k] for k in (0, 1, 2)], [1, 0, 2])

    def test_FD_classifies_branch_crossing_and_mask_changes_without_repair(self):
        p = np.asarray(audit.reference.TRUTH, np.float32)
        p[2] = math.pi/2
        a, b = p.astype(float).copy(), p.astype(float).copy()
        a[2] += 1e-4
        b[2] -= 1e-4
        mask = np.ones((4, 1, 16), bool)
        mask[0, 0, 0] = False
        r = audit.fd_record(fd_snapshot(a, mask), fd_snapshot(b), fd_snapshot(p), audit.branch_status(p[2]), 2, 1e-4,
            np.zeros((1, 3, 3)))
        self.assertTrue(r['branch_crossed'])
        self.assertTrue(r['masks_changed'])
        self.assertEqual(r['smooth_interpretation'], 'branch_crossed')
        self.assertEqual(r['mask_changes']['plus_to_minus']['flipped_total'], 1)

    def test_branch_distances_are_local_not_angle_canonicalization(self):
        for theta in (.31, -.8, .31+math.pi, -math.pi/2+1e-5):
            r = audit.branch_status(theta)
            self.assertEqual(r['theta'], theta)
            self.assertGreaterEqual(r['distance_to_branch_radians'], 0)
        self.assertAlmostEqual(audit.branch_status(math.pi/2)['distance_to_branch_radians'], 0)
        self.assertNotEqual(audit.branch_status(math.pi/2-1e-4)['branch_cell'], audit.branch_status(math.pi/2+1e-4)['branch_cell'])

    def test_zero_gradient_and_rank_deficiency_are_explicit_null_not_NaN(self):
        r = audit.gradient_comparison(np.zeros(7), np.zeros(7))
        self.assertIsNone(r['cosine_similarity'])
        self.assertTrue(r['denominator_below_floor'])
        spec = audit.spectrum(np.zeros((9, 7)))
        self.assertIsNone(spec['condition_number'])
        self.assertEqual(spec['numerical_rank'], 0)
        self.assertEqual(len(spec['near_null_right_vectors']), 7)
        json.dumps([r, spec], allow_nan=False)

    def test_source_PDF_is_auxiliary_to_active_RGB_graph(self):
        self.assertEqual(audit.pdf_dependency()['compute_radiance_pdf_active_reads'], 0)
        shader = MaterialShaderV1('A2', 'atan2', 8, 16)
        capture = audit.coupling.accepted.Capture(shader)
        p = torch.tensor(audit.reference.TRUTH, dtype=torch.float32, requires_grad=True)
        mean = TinyBatch()(capture, p)
        g, = torch.autograd.grad(mean[:, 2].sum(), capture.result['pdf'], allow_unused=True)
        self.assertIsNone(g)

    def test_complete_derivative_and_FD_records_strict_JSON_before_snapshot_change(self):
        measure, p, base, row, j, mapping = tiny_case()
        gradient = audit.direct_gradient(measure, p, base, j, torch.zeros(3, 3))
        record = dict(status='complete', physical_jacobians=row['physical_jacobians'], gradient=gradient,
            forward=audit.snapshot(base, measure.ids), mapping=mapping, comparison=audit.jacobian_comparison(j, j),
            branch=audit.branch_status(p[2]))
        q = np.asarray(audit.reference.TRUTH, np.float32)
        plus, minus = q.astype(float).copy(), q.astype(float).copy()
        plus[0] += 1e-4
        minus[0] -= 1e-4
        record['finite_difference'] = audit.fd_record(fd_snapshot(plus), fd_snapshot(minus), fd_snapshot(q),
            audit.branch_status(q[2]), 0, 1e-4, np.full((1, 3, 3), 2*float(q[0])))
        stream = io.StringIO()
        audit.coupling.accepted.persist_report(stream, record)
        before = stream.getvalue()
        self.assertEqual(json.loads(before), record)
        with self.assertRaises(TypeError):
            audit.coupling.accepted.persist_report(stream, {'bad': np.bool_(True)})
        self.assertEqual(stream.getvalue(), before)
        with self.assertRaises(ValueError):
            audit.coupling.accepted.persist_report(stream, {'bad': float('nan')})
        self.assertEqual(stream.getvalue(), before)

    def test_budget_accounts_primal_calls_points_categories_and_rejects_extra_work(self):
        report = {}
        budget = audit.Budget(report)
        expected_points = 18*(7+1+1)*96+12*(7+1)*336+72*96
        self.assertEqual(expected_points, audit.MAX_POINTS)
        self.assertEqual(sum(audit.LIMITS.values()), audit.MAX_CALLS)
        report['work']['categories']['jvp'] = 210
        with self.assertRaises(SignatureError):
            budget.before(96, 'fit', 'jvp')
        report['work']['categories']['jvp'] = 0
        report['work']['shader_points'] = audit.MAX_POINTS
        with self.assertRaises(SignatureError):
            budget.before(96, 'fit', 'baseline')
        report['work']['shader_points'] = 0
        report['work']['full_batch_forward_evaluations'] = 330
        with self.assertRaises(SignatureError):
            budget.before(96, 'fit', 'baseline')
        with patch.object(audit.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=audit.RSS_BYTES//1024+1)):
            with self.assertRaisesRegex(SignatureError, 'RSS'):
                budget.check()

    def test_corrupt_evidence_stops_before_old_admission_or_shader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = next(iter(audit.INPUTS))
            p = root/name
            p.parent.mkdir(parents=True)
            p.write_text('corrupt')
            with patch.object(audit, 'ROOT', root), patch.object(audit.coupling, 'admit', side_effect=AssertionError('No old admission')), \
                 patch.object(audit, 'MaterialShaderV1', side_effect=AssertionError('No renderer')):
                with self.assertRaisesRegex(SignatureError, 'identity changed'):
                    audit.admit()

    def test_runtime_failure_before_any_forward_preserves_partial_and_restores_state(self):
        threads = torch.get_num_threads()
        report, saves = {}, []
        with patch.object(audit, 'MaterialShaderV1', side_effect=AssertionError('No shader')):
            with self.assertRaisesRegex(SignatureError, 'runtime'):
                audit.execute({}, {}, {}, {'runtime': {}}, [], report, lambda: saves.append(json.dumps(report, allow_nan=False)))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['work']['full_batch_forward_evaluations'], 0)
        self.assertTrue(report['global_torch_CPU_RNG_unchanged'] and report['global_numpy_RNG_unchanged'])
        self.assertTrue(report['threads_restored'])
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertTrue(saves)

    def test_collision_failure_JSON_and_signal_memory_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, protocol = root/'output.json', root/'protocol.md'
            out.write_text('keep')
            protocol.write_text('fixture')
            handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
            with patch.object(audit, 'OUTPUT', out), patch.object(audit, 'PROTOCOL', protocol), \
                 patch.dict(audit.os.environ, audit.original.REQUIRED_ENV), patch.object(sys, 'argv', ['fixture', '--output', str(out)]), \
                 patch.object(audit, 'admit', side_effect=SignatureError('fixture gate')), patch.object(audit.resource, 'setrlimit') as limits:
                with self.assertRaises(FileExistsError):
                    audit.main()
                self.assertEqual(out.read_text(), 'keep')
                self.assertEqual(limits.call_count, 0)
                out.unlink()  # Disposable fixture only.
                with self.assertRaisesRegex(SignatureError, 'fixture gate'):
                    audit.main()
                record = json.loads(out.read_text())
                self.assertEqual(record['status'], 'failed')
                self.assertTrue(record['memory_limit_restored'])
                self.assertEqual(limits.call_count, 2)
            self.assertEqual(handlers, {s: signal.getsignal(s) for s in handlers})

    def test_no_optimizer_parameter_updates_or_global_RNG_consumption(self):
        rng, nrng = torch.get_rng_state().clone(), np.random.get_state()
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer')), \
             patch.object(torch.optim, 'LBFGS', side_effect=AssertionError('No optimizer')):
            tiny_case()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertTrue(nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:])
        tree = ast.parse(Path(audit.__file__).read_text())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'Adam', 'SGD', 'LBFGS', 'step', 'run_lm', 'run_adam', 'restore', 'cuda', 'load',
            'manual_seed', 'make_grids', 'hemisphere', 'initializations'})

    def test_frozen_sources_helpers_evidence_and_pins_preserved(self):
        for name, sha in audit.INPUTS.items():
            self.assertEqual(audit.digest(audit.ROOT/name), sha)
        for stage, pin in audit.coupling.PINS.items():
            audit.coupling.baseline(stage, pin)
        for name in ('experiments/material_sample_coupling_v1.py', 'experiments/material_shader_v1.py',
                     'experiments/material_recovery_optimizer_comparison_v1.py', 'experiments/run_known_material_recovery_v1.py'):
            self.assertEqual((audit.ROOT/name).read_bytes(), subprocess.check_output(['git', 'show', audit.BASE+':'+name], cwd=audit.ROOT))


if __name__ == '__main__':
    unittest.main()
