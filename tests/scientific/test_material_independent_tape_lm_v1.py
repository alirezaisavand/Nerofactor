"""New conditional-replication fixtures; no full scientific shader calls."""
import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_independent_tape_lm_v1 as m
from infrastructure.signatures import SignatureError
# Reuse fixture definitions only; never load/run the old TestCase suite.
from tests.scientific.test_material_lm_gauge_trajectories_v2 import gate_fixture,update_means,SyntheticArm,FixtureStop


EXPECTED_TAPE_SHA = (
    '713ea6c411b4208f53331ecc8be6fbaa1b6d352c6f0ec8d4dc8e4dd14b5339f5',
    '880003f75dd7fbffbc004fc16c788ccec15c3f8f7e437ccafc0b97fc3c032f93',
    '35598b4a57d229eaf41e5e508d789f569337cbd191dd8985744ca179273efc0a',
    '94b812b45a2c2da1f737a97e7d9be8925602271888063c7d165f2deb11c81e10',
)


def new_gate_fixture():
    arms,historical,target,obs,construction = gate_fixture()
    tape,manifest = m.make_independent_tapes()
    copied,record = m.coupling.copy_coupled_tape(tape,1)
    construction['coupled_tape'] = record
    for arm,t in zip(arms,(tape,tape,copied)):
        arm.fit.tapes = t
        arm.initial_capture['tape_arrays'] = [m.reference.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)]
    expected = copy.deepcopy(arms[0].row['initial_state'])
    # A true old/new scalar/prediction mismatch, not an artificial matching old record.
    older_mean = np.asarray(historical['initial_fit_predictions'],np.float32)
    older_mean[:,1] += np.float32(1e-4)
    older_mean[:,2] = older_mean[:,0]+older_mean[:,1]
    historical['initial_fit_predictions'] = older_mean.tolist()
    historical['initial_fit_prediction_sha256'] = m.reference.sha_array(older_mean)
    historical['initial_fit_MSE'] = m.roundoff.mse32(older_mean[:,2],target)[0]
    return arms,historical,target,obs,construction,expected,manifest


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old,cls.history,cls.continuation,cls.identities = m.admit()
        cls.refs = m.group_references(cls.history)

    def test_complete_chain_admission_and_preserved_failed_attempts(self):
        self.assertEqual(len(self.identities),68)
        for name in ('docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json',
                     'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json',
                     'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2_ANALYSIS.json'):
            self.assertIn(name,self.identities)
        prior = json.loads((m.ROOT/'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json').read_text())
        self.assertEqual(prior['status'],'failed')

    def test_preregistered_seed_nonoverlap_and_deterministic_hashes(self):
        a,manifest = m.make_independent_tapes()
        b,again = m.make_independent_tapes(manifest)
        self.assertEqual(m.TRAIN_SEEDS,(2000,2001,2002,2003))
        self.assertFalse(set(m.TRAIN_SEEDS)&(set(m.OLD_SEEDS)|set(m.EVAL_SEEDS)))
        self.assertEqual(tuple(r['sha256'] for r in manifest['per_tape']),EXPECTED_TAPE_SHA)
        self.assertEqual(manifest,again)
        for x,y in zip((a.u,a.v,a.azimuth),(b.u,b.v,b.azimuth)):
            self.assertTrue(torch.equal(x,y))
            self.assertNotEqual(x.data_ptr(),y.data_ptr())
        self.assertEqual(a.u.shape,(96,1024))
        self.assertEqual(a.azimuth.shape,(96,1,1))

    def test_bad_frozen_tape_hash_is_fatal_without_replacement(self):
        _,manifest = m.make_independent_tapes()
        bad = copy.deepcopy(manifest)
        bad['per_tape'][0]['array_sha256'][0] = '0'*64
        with self.assertRaises(SignatureError):
            m.make_independent_tapes(bad)

    def test_global_RNG_and_original_eval_tapes_preserved(self):
        state,nstate = torch.get_rng_state().clone(),np.random.get_state()
        m.make_independent_tapes()
        after = np.random.get_state()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(np.array_equal(nstate[1],after[1]))
        self.assertEqual(nstate[0],after[0]);self.assertEqual(nstate[2:],after[2:])
        _,ids = m.original.make_tapes(list(m.EVAL_SEEDS),84)
        self.assertEqual(ids,self.old['tapes']['evaluation'])
        self.assertEqual(ids,self.history['tapes']['evaluation'])

    def test_new_signed_coupling_preserves_radial_diffuse_and_storage(self):
        tape,_ = m.make_independent_tapes()
        originals = [v.clone() for v in (tape.u,tape.v,tape.azimuth)]
        for sign in (-1,1):
            copied,record = m.coupling.copy_coupled_tape(tape,sign)
            ideal = np.remainder(tape.v.numpy().astype(float)-sign/4,1.).astype(np.float32)
            ideal = np.remainder(ideal,np.float32(1.))
            self.assertTrue(np.array_equal(copied.v.numpy(),ideal))
            self.assertTrue(torch.equal(copied.u,tape.u));self.assertTrue(torch.equal(copied.azimuth,tape.azimuth))
            self.assertEqual(record['sign'],sign)
            for x,y in zip((copied.u,copied.v,copied.azimuth),(tape.u,tape.v,tape.azimuth)):
                self.assertNotEqual(x.data_ptr(),y.data_ptr())
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(originals,(tape.u,tape.v,tape.azimuth))))

    def test_exact_update1000_gauge_and_frame_signs_no_optimizer(self):
        shader = m.original.MaterialShaderV1('A2','atan2')
        with patch.object(torch.optim,'Adam',side_effect=AssertionError('No Adam')):
            for saved in self.continuation['initializations']:
                z = m.v1.restore(saved);p0 = m.original.physical(z);z1 = m.v1.gauge(z);p1 = m.original.physical(z1)
                self.assertEqual(m.lm.accepted.describe(z),saved['latest_state']['unconstrained'])
                self.assertTrue(torch.equal(z1[:2],z[:2].flip(0)))
                self.assertEqual(z1[2],z[2]+math.pi/2)
                self.assertTrue(torch.equal(p0[3:],p1[3:]))
                q0,q1 = (m.coupling.accepted.tensor_q(p.numpy()) for p in (p0,p1))
                self.assertTrue(np.all(abs(q1-q0) <= 2e-7+2e-6*abs(q0)))
                frame = m.coupling.frame_sign(m.coupling.decoded_frame(shader,p0,96),m.coupling.decoded_frame(shader,p1,96))
                self.assertEqual(frame['sign'],1 if saved['seed'] <= 103 else -1)

    def test_gate_A_accepts_expected_historical_RGB_difference(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertTrue(r['passed'])
        self.assertTrue(r['gates']['A']['passed'])
        self.assertFalse(r['gates']['A']['historical_RGB_MSE_parity_required'])
        self.assertFalse(r['historical_initial_comparison']['old_tape_identity_diagnostic']['passed'])
        self.assertFalse(r['historical_initial_comparison']['identical_finite_sample_objective'])

    def test_gate_A_rejects_wrong_state_or_active_new_tape(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        bad = copy.deepcopy(e);bad['unconstrained']['sha256'] = '0'*64
        self.assertFalse(m.initial_gates(*arms,old,t,o,c,bad,manifest)['gates']['A']['passed'])
        arms[1].initial_capture['tape_arrays'][0] = '0'*64
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertFalse(r['gates']['A']['passed'])

    def test_internal_mean_MSE_replay_is_mandatory(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        arms[1].row['initial_fit_MSE'] += 1e-5
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertFalse(r['gates']['A']['passed'])
        self.assertFalse(r['gates']['A']['internal_replays'][1]['exact_MSE'])

    def test_sample_mask_failure_cannot_be_overridden_by_loss(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        arms[2].initial_capture['mask'][0,0] = False
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertFalse(r['gates']['C']['passed'])
        self.assertTrue(r['gates']['E']['passed'])
        self.assertFalse(r['passed'])

    def test_new_gate_A_does_not_override_frame_failure(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        arms[2].initial_capture['tangent'][10,0] += np.float32(1e-4)
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertTrue(r['gates']['A']['passed'])
        self.assertFalse(r['gates']['B']['passed'])
        self.assertFalse(r['passed'])

    def test_component_compensation_and_diffuse_identity(self):
        arms,old,t,o,c,e,manifest = new_gate_fixture()
        d = arms[2].initial_capture
        d['per_tape'][:,:,0] -= np.float32(.001)
        d['per_tape'][:,:,1] += np.float32(.001)
        update_means(arms[2],t)
        arms[2].row['initial_fit_predictions'] = d['mean'].tolist()
        arms[2].row['initial_fit_prediction_sha256'] = m.reference.sha_array(d['mean'])
        r = m.initial_gates(*arms,old,t,o,c,e,manifest)
        self.assertTrue(r['gates']['A']['passed'])
        self.assertFalse(r['gates']['D']['passed'])
        self.assertTrue(r['gates']['E']['passed'])

    def test_original_arithmetic_envelope_reused_without_retuning(self):
        self.assertIs(m.accepted.objective_envelope,m.accepted.initialization_gates.__globals__['objective_envelope'])
        p0 = np.full((24,3),np.float32(.2));p1 = np.nextafter(p0,np.float32(1));t = np.full((24,3),np.float32(.197))
        losses = [m.roundoff.mse32(p,t)[0] for p in (p0,p1)]
        envelope = m.accepted.objective_envelope(p0,p1,t,*losses)
        self.assertTrue(envelope['passed'])
        self.assertFalse(envelope['bound_is_independent_equivalence_evidence'])
        self.assertFalse(m.roundoff.gate(*losses)['passed'])

    def test_frozen_historical_group_construction(self):
        r = self.refs
        self.assertEqual(r['representatives'][m.GROUPS[0]]['seeds'],[101,102,103])
        self.assertEqual(r['representatives'][m.GROUPS[1]]['seeds'],[104,105])
        self.assertEqual(r['representatives'][m.GROUPS[0]]['Q']['sha256'],'a111a5483abc6c56ebebd2193089941a165b34103327b2174ec0aa1c844609e2')
        self.assertEqual(r['representatives'][m.GROUPS[1]]['Q']['sha256'],'6b712d9f5b98886f0620208b5df737984fcd215ce00a852caceb2200dcea8f0c')
        self.assertEqual(r['radius_absolute'],r['separation_absolute']/2)

    def test_descriptive_group_centers_midpoint_and_distant_Q(self):
        mats = [np.asarray(self.refs['representatives'][g]['Q']['values']) for g in m.GROUPS]
        for g,q in zip(m.GROUPS,mats):
            p = m.proximity(q,self.refs)
            self.assertEqual(p['label'],g)
            self.assertEqual(p['absolute_distances'][g],0.)
        self.assertEqual(m.proximity((mats[0]+mats[1])/2,self.refs)['label'],'ambiguous_historical_proximity')
        distant = m.proximity(np.eye(2)*.9,self.refs)
        self.assertEqual(distant['label'],'outside_historical_group_balls')
        self.assertEqual(len(distant['relative_distances']),2)

    def test_recovery_and_historical_proximity_are_separate(self):
        self.assertEqual(m.original.SETTINGS['success'],dict(Q_relative=.05,orientation_degrees=5.,max_albedo_absolute=.02,
            metallic_absolute=.03,held_out_relative_L2=.01))
        self.assertIn('not recovery success',m.proximity(np.eye(2)*.9,self.refs)['interpretation'])

    def test_cross_tape_raw_training_loss_not_same_objective(self):
        old = self.history['outcomes'][0]
        row = copy.deepcopy(old)
        row['final_fit_MSE'] /= 2  # Fake changed objective, no optimization.
        row['radiance_errors']['held_out']['total']['relative_L2'] += .001
        result = m.cross_tape_analysis(dict(outcomes=[row]),self.history,self.refs)
        c = result['comparisons'][0]
        self.assertFalse(c['training_objectives']['same_finite_sample_objective'])
        self.assertTrue(c['training_objectives']['raw_loss_difference_is_not_optimization_improvement'])
        self.assertAlmostEqual(c['radiance_error_differences']['held_out']['total']['relative_L2'],.001)
        self.assertEqual(c['physical_change']['Q_relative'],0.)
        self.assertIn('fixed original-tape Adam',result['limitation'])

    def test_missing_endpoint_is_not_invented_recovery(self):
        old = self.history['outcomes'][0]
        row = dict(seed=old['seed'],arm=old['arm'],status='restored',censored=False,termination=None,work=m.accepted.work())
        result = m.cross_tape_analysis(dict(outcomes=[row]),self.history,self.refs)['comparisons'][0]
        self.assertIn('comparison_blocker',result)
        self.assertNotIn('new_recovery',result)

    def test_original_LM_policy_synthetic_prefix(self):
        a,b = SyntheticArm(),SyntheticArm()
        for fn,arm in ((m.lm.run_lm,a),(m.v1.guarded_loop,b)):
            with self.assertRaises(FixtureStop):
                fn(arm,lambda:None)
        self.assertTrue(m.v1.historical_control(b.row,a.row)['passed'])
        self.assertEqual(b.row['work']['accepted_parameter_steps'],2)

    def test_frozen_branch_and_saturation_semantics(self):
        arm = SyntheticArm(crossing=True)
        with self.assertRaises(m.v1.BranchBlocked):
            m.v1.guarded_loop(arm,lambda:None)
        self.assertEqual(arm.calls,8)
        self.assertEqual(arm.row['work']['rejected_parameter_steps'],0)
        arm = SyntheticArm(constant=True)
        m.v1.guarded_loop(arm,lambda:None)
        self.assertEqual(arm.row['termination'],'damping_saturated')
        self.assertEqual(arm.row['damping'],1e8)

    def test_original_solver_and_float32_candidate(self):
        z = torch.arange(7,dtype=torch.float32)/10
        j = torch.eye(72,7,dtype=torch.float64);r = torch.arange(72,dtype=torch.float32)*.001
        candidate,predicted,delta,represented = m.lm.proposal(z,j,r,.01)
        h,g = j.T@j,j.T@r.double()
        self.assertTrue(torch.equal(delta,torch.linalg.solve(h+.01*torch.diag(h.diag()+1e-8),-g)))
        self.assertTrue(torch.equal(candidate,(z.double()+delta).float()))
        self.assertEqual(predicted,float(-2*g.dot(represented)-represented.dot(h@represented)))

    def test_work_caps_and_reserve_remain_exactly_accepted(self):
        self.assertEqual(sum(m.accepted.CAPS[n] for n in ('fit_points','evaluation_points'))*15,1275120)
        self.assertEqual(15*m.accepted.CAPS['forward_attempts'],13245)
        self.assertEqual(15*m.accepted.CAPS['jacobian_vector_evaluations'],8400)
        for name,cap in m.accepted.CAPS.items():
            counts = m.accepted.work();counts[name] = cap+1
            with self.assertRaises(m.v1.IntegrityFailure):
                m.accepted.check_caps(counts)
        clock = m.lm.ArmClock(SimpleNamespace(check=lambda:None));clock.wall = 110.
        with self.assertRaises(m.lm.ArmLimit):
            clock.check(10.)
        clock.check(0.)

    def test_common_gauge_original_evaluation_single_call(self):
        z = torch.tensor([.2,-.3,-.7,.1,.2,.3,-.1])
        arm = SimpleNamespace(z=m.v1.gauge(z),row=dict(arm='P1-C'),check=lambda:None)
        calls = []
        def forward(p,**kw):
            self.assertEqual(kw,dict(evaluation=True,physical=True))
            calls.append(p)
            return torch.ones((84,3,3))
        arm.forward = forward
        obs = [dict(split=s,view=i%12)for i,s in enumerate(['fit']*24+['held_out']*24+['relight']*36)]
        m.v1.evaluate(arm,np.ones((84,3,3)),obs)
        expected = m.original.physical(arm.z)[m.v1.PERM].clone();expected[2] -= math.pi/2
        self.assertEqual(len(calls),1)
        self.assertTrue(torch.equal(calls[0],expected))

    def test_strict_partial_report_persistence(self):
        stream = io.StringIO()
        m.coupling.accepted.persist_report(stream,dict(status='partial',initialization=[]))
        saved = stream.getvalue()
        with self.assertRaises(ValueError):
            m.coupling.accepted.persist_report(stream,dict(value=float('inf')))
        self.assertEqual(stream.getvalue(),saved)

    def test_no_optimization_before_all_gates_no_old_parity_enforcement(self):
        source = Path(m.__file__).read_text()
        self.assertLess(source.index("phase='initial_forwards'"),source.index("report['phase'] = 'initial_gates'"))
        self.assertLess(source.index("require(report['initialization_gate'] == 'passed'"),source.index('v1.run_arm(arm,targets,obs,persist)'))
        self.assertNotIn('v1.historical_control(',source)
        self.assertNotIn('encoding_override',source)
        self.assertIn('accepted.Arm(',source)
        for name,sha in self.identities.items():
            self.assertEqual(m.accepted.digest(m.ROOT/name),sha)


if __name__ == '__main__':
    unittest.main()
