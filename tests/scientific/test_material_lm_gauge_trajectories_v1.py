"""New synthetic/read-only fixtures only; never replay a scientific optimizer run."""
import copy
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_lm_gauge_trajectories_v1 as m
from infrastructure.signatures import SignatureError


class Clock:
    wall = cpu = 0.

    def check(self, *args):
        pass

    def elapsed(self):
        return 0.


class SyntheticArm(m.Arm):
    def __init__(self, z=None):
        row = dict(arm='P0', work=m.work(), iterations=[], trajectory=[], completed_outer_iterations=0,
                   termination=None, censored=False, status='restored')
        if z is None:
            z = torch.tensor([.1, .2, .3, -.2, .1, -.3, .2])
        super().__init__(row, Clock(), object(), None, None, None, z.requires_grad_())
        self.calls = 0
        self.current = m.lm.detached(self.forward(self.z))

    def forward(self, z, **kwargs):
        self.calls += 1
        residual = (z.repeat(11)[:72] - .1) / math.sqrt(72)
        return dict(residual=residual, loss=torch.mean((z.repeat(11)[:72]-.1)**2),
                    values=residual, mask=np.zeros((96, 1024), dtype=bool),
                    mask_identity=dict(fixture=True))


class Tests(unittest.TestCase):
    def test_admission_complete_chain(self):
        old, history, continuation, identities = m.admit()
        self.assertEqual(history['status'], 'complete')
        self.assertEqual(len(continuation['initializations']), 5)
        self.assertIn('docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json', identities)
        self.assertIn('docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json', identities)
        self.assertEqual(old['shader']['variant'], 'A2')

    def test_exact_restore_all_five_no_adam_construction(self):
        evidence = json.loads((m.ROOT/'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json').read_text())
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No Adam')):
            for row in evidence['initializations']:
                z = m.restore(row)
                self.assertEqual(m.lm.accepted.describe(z), row['latest_state']['unconstrained'])
                bad = copy.deepcopy(row)
                bad['latest_state']['unconstrained']['values'][0] += 1e-3
                with self.assertRaises(SignatureError):
                    m.restore(bad)

    def test_gauge_permutation_float32_translation_and_Q(self):
        z = torch.tensor([.3, -.6, .21, .4, -.2, .5, -.7])
        out = m.gauge(z)
        self.assertTrue(torch.equal(out[:2], z[:2].flip(0)))
        self.assertEqual(out[2], z[2]+math.pi/2)
        self.assertTrue(torch.equal(out[3:], z[3:]))
        self.assertLess(m.q_difference(m.original.physical(out).tolist(), m.original.physical(z).tolist()), 2e-6)
        self.assertTrue(torch.equal(m.gauge(out, inverse=True)[[0, 1, 3, 4, 5, 6]], z[[0, 1, 3, 4, 5, 6]]))

    def test_signed_frames_original_update1000(self):
        shader = m.original.MaterialShaderV1('A2', 'atan2')
        evidence = json.loads((m.ROOT/'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json').read_text())
        for row in evidence['initializations']:
            z = m.restore(row)
            p0, p1 = m.original.physical(z), m.original.physical(m.gauge(z))
            mapping = m.coupling.frame_sign(m.coupling.decoded_frame(shader, p0, 96),
                                           m.coupling.decoded_frame(shader, p1, 96))
            self.assertIn(mapping['sign'], (-1, 1))
            self.assertLess(mapping['max_quarter_turn_error'], 2e-6)

    def test_copied_tape_independent_storage_and_original_draws(self):
        from experiments.material_shader_v1 import RandomInputs
        tape = RandomInputs(torch.full((2, 4), .3), torch.tensor([[0., .25, .5, .75]]).repeat(2, 1), torch.full((2, 1, 1), .2))
        for sign in (-1, 1):
            copied, record = m.coupling.copy_coupled_tape(tape, sign)
            for a, b in zip((tape.u, tape.v, tape.azimuth), (copied.u, copied.v, copied.azimuth)):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertTrue(torch.equal(copied.u, tape.u))
            self.assertTrue(torch.equal(copied.azimuth, tape.azimuth))
            self.assertTrue(torch.equal(copied.v, (tape.v-sign/4) % 1))
            self.assertEqual(record['sign'], sign)

    def test_actual_branch_cell_discontinuity(self):
        for cut in (-3*math.pi/2, -math.pi/2, math.pi/2, 3*math.pi/2):
            left, right = m.branch(cut-1e-5), m.branch(cut+1e-5)
            self.assertEqual(right['cell']-left['cell'], 1)
            self.assertGreater(left['decoded_angle_float32'], 1.5)
            self.assertLess(right['decoded_angle_float32'], -1.5)
        self.assertEqual(m.branch(.1)['cell'], 0)
        self.assertEqual(m.branch(.1+math.pi)['cell'], 1)

    def test_counterpart_cut_is_relevant_without_own_cut(self):
        z = torch.tensor([.1, .2, -.01, 0., 0., 0., 0.])
        initial = m.chart(z, 'P0')
        z[2] = .01
        self.assertEqual(m.crossed(initial, m.chart(z, 'P0')), ['counterpart'])

    def test_original_jvp_jacobian_normalization(self):
        z = torch.arange(7, dtype=torch.float32)/10
        residual = lambda x: x.repeat(11)[:72]/math.sqrt(72)
        counts = m.work()
        jac = m.lm.residual_jacobian(residual, z, residual(z), counts)
        expected = (torch.eye(7).repeat(11, 1)[:72]/math.sqrt(72)).double()
        self.assertTrue(torch.equal(jac, expected))
        self.assertEqual(counts['jacobian_vector_evaluations'], 7)
        self.assertEqual(counts['reverse_completed'], 14)

    def test_original_solver_rounding_and_predicted_reduction(self):
        z = torch.tensor([.3]*7)
        jac = torch.eye(72, 7, dtype=torch.float64)
        residual = torch.arange(72, dtype=torch.float32)*.001
        candidate, predicted, delta, represented = m.lm.proposal(z, jac, residual, .01)
        h, g = jac.T@jac, jac.T@residual.double()
        expected = torch.linalg.solve(h+.01*torch.diag(h.diag()+1e-8), -g)
        self.assertTrue(torch.equal(delta, expected))
        self.assertTrue(torch.equal(candidate, (z.double()+expected).float()))
        self.assertEqual(predicted, float(-2*g.dot(represented)-represented.dot(h@represented)))

    def test_historical_initial_solve_parity_without_rendering(self):
        history = json.loads((m.ROOT/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json').read_text())
        for row in history['outcomes']:
            if row['arm'] != 'LM':
                continue
            rec = row['iterations'][0]
            h = torch.tensor(rec['normal_matrix'], dtype=torch.float64)
            g = torch.tensor(rec['gradient'], dtype=torch.float64)/2
            delta = torch.linalg.solve(h+.01*torch.diag(h.diag()+1e-8), -g)
            self.assertEqual(delta.tolist(), rec['trials'][0]['delta_float64'])
            self.assertEqual((torch.tensor(rec['z']).double()+delta).float().tolist(),
                             rec['trials'][0]['proposed_unconstrained'])

    def test_initial_objective_frozen_tolerance(self):
        m.initial_objective_gate(1e-5, 1e-5+1e-12)
        with self.assertRaises(SignatureError):
            m.initial_objective_gate(1e-5, 1e-5+1e-9)

    def test_uncoupled_control_uses_original_tape(self):
        source = Path(m.__file__).read_text()
        self.assertIn("selected = copied if name == 'P1-C' else fit_tape", source)
        self.assertEqual(source.count('coupling.copy_coupled_tape('), 1)
        self.assertIn("actual['sign'] == self.sign", source)

    def test_original_acceptance_damping(self):
        self.assertEqual(m.lm.acceptance(1., .8, .5), (True, .3999999999999999))
        self.assertEqual(m.lm.acceptance(1., 1., .1), (False, 0.))
        self.assertEqual(m.lm.accepted_lambda(.01, .8), .01/3)
        self.assertEqual(m.lm.accepted_lambda(.01, .2), .02)
        self.assertEqual(m.lm.accepted_lambda(.01, .5), .01)

    def test_crossing_stops_before_forward_without_retry(self):
        arm = SyntheticArm()
        candidate = arm.z.detach().clone()
        candidate[2] = 1.8
        def proposal(z, j, r, d, c):
            c['linear_solves'] += 1
            c['linear_solves_completed'] += 1
            delta = candidate.double()-z.detach().double()
            return candidate, 1., delta, delta
        with patch.object(m.lm, 'proposal', side_effect=proposal):
            with self.assertRaises(m.BranchBlocked):
                m.guarded_loop(arm, lambda: None)
        self.assertEqual(arm.calls, 8)  # initial + seven JVP primals; no candidate
        self.assertEqual(arm.row['work']['linear_solves'], 1)
        self.assertEqual(arm.row['work']['accepted_parameter_steps'], 0)
        self.assertEqual(arm.row['work']['rejected_parameter_steps'], 0)
        self.assertEqual(arm.row['work']['branch_blocked_proposals'], 1)
        self.assertEqual(arm.row['termination'], 'branch_crossing_blocked')
        self.assertFalse(arm.row['iterations'][0]['trials'][0]['candidate_rendered'])

    def test_original_zero_step_rejection_and_saturation(self):
        arm = SyntheticArm()
        def proposal(z, j, r, d, c):
            c['linear_solves'] += 1
            c['linear_solves_completed'] += 1
            return z.detach().clone(), 0., torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)
        with patch.object(m.lm, 'proposal', side_effect=proposal), patch.object(m.lm, 'LAMBDA_INITIAL', 1e8):
            m.guarded_loop(arm, lambda: None)
        self.assertEqual(arm.row['termination'], 'damping_saturated')
        self.assertEqual(arm.row['work']['rejected_parameter_steps'], 1)
        self.assertEqual(arm.calls, 8)

    def test_private_loop_preserves_original_module_globals(self):
        arm = SyntheticArm()
        before = dict(m.lm.run_lm.__globals__)
        with patch.object(m.lm, 'LM_OUTERS', 1):
            m.guarded_loop(arm, lambda: None)
        self.assertTrue(all(v is m.lm.run_lm.__globals__[k] for k, v in before.items()))
        self.assertEqual(arm.row['termination'], 'outer_iteration_ceiling')
        self.assertEqual(arm.row['work']['accepted_parameter_steps'], 1)

    def test_full_synthetic_shared_prefix_matches_original_loop(self):
        control, candidate = SyntheticArm(), SyntheticArm()
        with patch.object(m.lm, 'LM_OUTERS', 2):
            m.lm.run_lm(control, lambda: None)
            m.guarded_loop(candidate, lambda: None)
        self.assertTrue(m.historical_control(candidate.row, control.row)['passed'])
        self.assertTrue(torch.equal(control.z, candidate.z))

    def test_complete_forward_and_reverse_accounting(self):
        class Batch:
            seeds = 4

            def __init__(self, n):
                self.n = n

            def __call__(self, shader, p):
                shader.mask = np.zeros((96, 1024), dtype=bool)
                return p[3:6].reshape(1, 1, 3).expand(self.n, 3, 3)
        row = dict(arm='P0', work=m.work())
        arm = m.Arm(row, Clock(), object(), Batch(24), Batch(84), torch.zeros((24, 3)),
                    torch.tensor([.1, .2, .3, -.2, .1, -.3, .2], requires_grad=True))
        arm.forward(arm.z)
        arm.stage = 'optimization'
        arm.in_jacobian = True
        expected = arm.fit(arm.shader, m.original.physical(arm.z))[:, 2].reshape(-1)/math.sqrt(72)
        m.lm.residual_jacobian(arm.residual, arm.z, expected.detach(), row['work'])
        arm.in_jacobian = False
        arm.forward(arm.z)
        arm.stage = 'endpoint'
        state = arm.forward(arm.z)
        row['work']['terminal_gradients'] += 1
        arm.gradient(state, arm.z)
        arm.forward(m.original.physical(arm.z).detach(), evaluation=True, physical=True)
        c = row['work']
        self.assertEqual(c['forward_attempts'], 11)
        self.assertEqual(c['initial_forwards'], 1)
        self.assertEqual(c['candidate_forwards'], 1)
        self.assertEqual(c['terminal_fit_forwards'], 1)
        self.assertEqual(c['evaluation_forward_evaluations'], 1)
        self.assertEqual(c['reverse_completed'], 15)
        self.assertEqual(c['fit_points'], 960)
        self.assertEqual(c['evaluation_points'], 336)
        self.assertEqual(c['specular_directions'], 1296*1024)
        self.assertEqual(c['diffuse_directions'], 1296*512)

    def test_common_gauge_evaluation_uses_original_tapes_and_rounding(self):
        arm = SyntheticArm()
        arm.row['arm'] = 'P1-C'
        z0 = arm.z.detach().clone()
        arm.z = m.gauge(z0).requires_grad_()
        observed = []
        targets = np.ones((84, 3, 3))
        obs = [dict(split=s, view=i % 12) for i, s in enumerate(['fit']*24+['held_out']*24+['relight']*36)]
        def forward(p, **kwargs):
            observed.append(p.clone())
            self.assertTrue(kwargs['evaluation'] and kwargs['physical'])
            return torch.ones((84, 3, 3))
        arm.forward = forward
        m.evaluate(arm, targets, obs)
        expected = m.original.physical(arm.z).detach()[m.PERM].clone()
        expected[2] -= math.pi/2
        self.assertTrue(torch.equal(observed[0], expected))
        self.assertEqual(len(observed), 1)
        self.assertIn('rounding_delta', arm.row['evaluation_gauge'])

    def test_original_recovery_gates_unchanged(self):
        self.assertEqual(m.original.SETTINGS['success'], dict(Q_relative=.05, orientation_degrees=5.,
            max_albedo_absolute=.02, metallic_absolute=.03, held_out_relative_L2=.01))

    def test_work_caps_and_memory_fail_closed(self):
        counts = m.work()
        m.check_caps(counts)
        for name, cap in m.CAPS.items():
            counts = m.work()
            counts[name] = cap+1
            with self.assertRaises(m.IntegrityFailure):
                m.check_caps(counts)
        with patch.object(m.resource, 'getrusage') as get:
            get.return_value.ru_maxrss = m.RSS_BYTES//1024+1
            with self.assertRaises(m.IntegrityFailure):
                m.memory_check()

    def test_arm_time_reserve(self):
        class Whole:
            def check(self):
                pass
        clock = m.lm.ArmClock(Whole())
        clock.wall = 110.
        with self.assertRaises(m.lm.ArmLimit):
            clock.check(10.)
        clock.check(0.)
        clock.wall = 120.
        with self.assertRaises(m.lm.ArmLimit):
            clock.check(0.)

    def test_strict_serialization_preserves_partial_report(self):
        stream = io.StringIO()
        m.coupling.accepted.persist_report(stream, dict(status='partial', rows=[1]))
        before = stream.getvalue()
        with self.assertRaises(ValueError):
            m.coupling.accepted.persist_report(stream, dict(status='bad', x=float('nan')))
        self.assertEqual(stream.getvalue(), before)

    def test_historical_control_detects_mismatch_and_censors_prefix(self):
        row = dict(iterations=[dict(z=[1], training_MSE=1., lambda_before=.01,
                                   trials=[dict(status='branch_crossing_blocked', damping=.01)])], censored=True)
        archived = dict(iterations=[dict(z=[1], training_MSE=1., lambda_before=.01,
                                        trials=[dict(status='accepted', damping=.01)])], termination='damping_saturated')
        self.assertTrue(m.historical_control(row, archived)['passed'])
        row['iterations'][0]['training_MSE'] = .9
        self.assertFalse(m.historical_control(row, archived)['passed'])


if __name__ == '__main__':
    unittest.main()
