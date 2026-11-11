"""New synthetic/read-only v2 fixtures; no full shader or historical experiment replay."""
import copy
from fractions import Fraction
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_lm_gauge_trajectories_v2 as m
from infrastructure.signatures import SignatureError


class Clock:
    wall = cpu = 0.

    def check(self, *args):
        pass

    def elapsed(self):
        return 0.


class FixtureStop(RuntimeError):
    pass


class SyntheticArm(m.v1.Arm):
    def __init__(self, crossing=False, constant=False):
        z = torch.tensor([.1,.2,-.3,-.2,.1,-.3,.2],requires_grad=True)
        row = dict(arm='P0',work=m.work(),iterations=[],trajectory=[],completed_outer_iterations=0,
            termination=None,censored=False,status='restored')
        super().__init__(row,Clock(),object(),None,None,None,z)
        self.calls,self.constant = 0,constant
        self.target = torch.full((7,),-.2)
        if crossing:
            self.target[2] = 1.8
        self.current = m.lm.detached(self.forward(z))

    def check(self):
        if self.row['work']['accepted_parameter_steps'] >= 2:
            raise FixtureStop('Synthetic prefix complete')

    def forward(self,z):
        self.calls += 1
        error = (z*0+1 if self.constant else z-self.target).repeat(11)[:72]
        return dict(residual=error/math.sqrt(72),loss=torch.mean(error**2),values=error,
                    mask=np.zeros((96,1024),bool),mask_identity=dict(fixture=True))


def gate_fixture():
    tape,ids = m.original.make_tapes(list(range(4)),24)
    copied,record = m.coupling.copy_coupled_tape(tape,1)
    z = torch.tensor([.2,-.3,-.7,.1,.2,.3,-.1])
    z1 = m.v1.gauge(z)
    p0,p1 = m.original.physical(z),m.original.physical(z1)
    target = np.full((24,3),np.float32(.197))
    obs = [dict(view=i%6,light=i%4,split='fit') for i in range(24)]
    rows,arms = [],[]
    for name,p,coords,t in (('P0',p0,z,tape),('P1-U',p1,z1,tape),('P1-C',p1,z1,copied)):
        d = np.full((4,24,3),np.float32(.15))
        s = np.full((4,24,3),np.float32(.05))
        per = np.stack((d,s,d+s),2)
        mean = np.stack([torch.from_numpy(per[:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
        n = np.tile(np.array([[0.,0.,1.]],np.float32),(96,1))
        tangent = np.tile(np.array([[0.,1.,0.]],np.float32),(96,1))
        bitangent = np.tile(np.array([[-1.,0.,0.]],np.float32),(96,1))
        if name != 'P0':
            tangent,bitangent = bitangent,-tangent
        wi = np.broadcast_to(np.array([0.,0.,1.],np.float32),(96,1024,3)).copy()
        data = dict(half_vectors=wi.copy(),specular_directions=wi,diffuse_directions=wi[:,:512].copy(),
            pdf=np.ones((96,1024,1),np.float32),tangent=tangent,bitangent=bitangent,normal=n,
            views=n,mask=np.ones((96,1024),bool),per_tape=per,mean=mean,
            tape_arrays=[m.reference.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)])
        row = dict(seed=101,arm=name,initial_state=dict(unconstrained=m.lm.accepted.describe(coords),physical=m.lm.accepted.describe(p)),
            initial_fit_predictions=mean.tolist(),initial_fit_prediction_sha256=m.reference.sha_array(mean),
            initial_fit_MSE=m.roundoff.mse32(mean[:,2],target)[0],initial_mask=m.lm.mask_identity(data['mask']))
        arms.append(SimpleNamespace(row=row,initial_capture=data,fit=SimpleNamespace(tapes=t)))
        rows.append(row)
    return arms,copy.deepcopy(rows[0]),target,obs,dict(mapping=dict(sign=1),coupled_tape=record)


def update_means(arm,target):
    d = arm.initial_capture
    d['mean'] = np.stack([torch.from_numpy(d['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
    arm.row['initial_fit_MSE'] = m.roundoff.mse32(d['mean'][:,2],target)[0]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old,cls.history,cls.continuation,cls.identities = m.admit()

    def test_complete_admission_preserves_failures(self):
        self.assertEqual(len(self.identities),55)
        self.assertIn('docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json',self.identities)
        self.assertIn('docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1_OFFLINE_SUMMARY.json',self.identities)
        failed = json.loads((m.ROOT/'docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json').read_text())
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(failed['failure'],'Mandatory coupled initial objective gate failed')

    def test_exact_restore_gauge_and_no_adam(self):
        with patch.object(torch.optim,'Adam',side_effect=AssertionError('No Adam')):
            for saved in self.continuation['initializations']:
                z = m.v1.restore(saved)
                self.assertEqual(m.lm.accepted.describe(z),saved['latest_state']['unconstrained'])
                z1 = m.v1.gauge(z)
                self.assertTrue(torch.equal(z1[:2],z[:2].flip(0)))
                self.assertTrue(torch.equal(z1[3:],z[3:]))
                self.assertEqual(z1[2],z[2]+math.pi/2)

    def test_actual_signed_frames_and_copied_tape_identity(self):
        # No shader forward; decode only, on all five admitted initial states.
        shader = m.original.MaterialShaderV1('A2','atan2')
        tape,ids = m.original.make_tapes(list(range(4)),24)
        self.assertEqual(ids,self.old['tapes']['fit'])
        for saved in self.continuation['initializations']:
            z = m.v1.restore(saved)
            p0,p1 = m.original.physical(z),m.original.physical(m.v1.gauge(z))
            mapping = m.coupling.frame_sign(m.coupling.decoded_frame(shader,p0,96),m.coupling.decoded_frame(shader,p1,96))
            copied,record = m.coupling.copy_coupled_tape(tape,mapping['sign'])
            self.assertEqual(record['sign'],1 if saved['seed'] <= 103 else -1)
            self.assertTrue(torch.equal(copied.u,tape.u))
            self.assertTrue(torch.equal(copied.azimuth,tape.azimuth))
            self.assertNotEqual(copied.v.data_ptr(),tape.v.data_ptr())

    def test_all_synthetic_initial_gates_pass(self):
        arms,archived,target,obs,construction = gate_fixture()
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertTrue(result['passed'])
        self.assertTrue(all(g['passed'] for g in result['gates'].values()))

    def test_scalar_warning_failure_does_not_override_independent_gates(self):
        arms,archived,target,obs,construction = gate_fixture()
        d = arms[2].initial_capture
        d['per_tape'][:,:,1] += np.float32(1e-7)
        d['per_tape'][:,:,2] = d['per_tape'][:,:,0]+d['per_tape'][:,:,1]
        update_means(arms[2],target)
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertTrue(result['passed'])
        self.assertFalse(result['gates']['E']['old_scalar_gate_passed'])
        self.assertTrue(result['gates']['E']['old_scalar_gate_is_warning_only'])

    def test_specular_diffuse_compensation_cannot_hide_failure(self):
        arms,archived,target,obs,construction = gate_fixture()
        original_total = arms[2].initial_capture['per_tape'][:,:,2].copy()
        arms[2].initial_capture['per_tape'][:,:,0] -= np.float32(.001)
        arms[2].initial_capture['per_tape'][:,:,1] += np.float32(.001)
        arms[2].initial_capture['per_tape'][:,:,2] = original_total
        update_means(arms[2],target)
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertFalse(result['gates']['D']['passed'])
        self.assertTrue(result['gates']['E']['passed'])
        self.assertFalse(result['passed'])

    def test_mask_failure_even_with_identical_loss(self):
        arms,archived,target,obs,construction = gate_fixture()
        arms[2].initial_capture['mask'][0,0] = False
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertEqual(result['gates']['C']['reflection_mask_XOR_count'],1)
        self.assertFalse(result['gates']['C']['passed'])
        self.assertTrue(result['gates']['E']['old_scalar_gate_passed'])

    def test_nonuniform_or_wrong_signed_frames_fail_gate_B(self):
        arms,archived,target,obs,construction = gate_fixture()
        construction['mapping']['sign'] = -1
        self.assertFalse(m.initialization_gates(*arms,archived,target,obs,construction)['gates']['B']['passed'])
        construction['mapping']['sign'] = 1
        arms[2].initial_capture['tangent'][10,0] += np.float32(1e-4)
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertFalse(result['gates']['B']['batch_frames_uniform'])
        self.assertFalse(result['gates']['B']['passed'])

    def test_incident_and_half_vector_bounds_are_independent(self):
        base = np.broadcast_to(np.array([0.,0.,1.],np.float32),(4,2,3))
        candidate = base.copy()
        candidate[...,0] = np.float32(3e-6)
        self.assertFalse(m.bounded(candidate,base,2e-6,2e-5,True)['passed'])
        self.assertTrue(m.bounded(candidate,base,4e-6,2e-5,True)['passed'])
        candidate[...,0] = np.float32(3e-5)
        # Even very loose component bound cannot override angular limit.
        self.assertFalse(m.bounded(candidate,base,1.,1.,True)['passed'])

    def test_diffuse_and_each_tape_checks_are_independent(self):
        arms,archived,target,obs,construction = gate_fixture()
        arms[2].initial_capture['per_tape'][0,:,1] += np.float32(.001)
        arms[2].initial_capture['per_tape'][1,:,1] -= np.float32(.001)
        update_means(arms[2],target)
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertFalse(result['gates']['D']['fields']['per_tape_specular']['passed'])
        self.assertFalse(result['gates']['D']['passed'])

    def test_original_mean_ordering_and_exact_identity_rejection(self):
        arms,archived,target,obs,construction = gate_fixture()
        arms[0].row['initial_fit_prediction_sha256'] = '0'*64
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertFalse(result['gates']['A']['passed'])
        arms,archived,target,obs,construction = gate_fixture()
        arms[2].initial_capture['mean'][0,0,0] = np.nextafter(arms[2].initial_capture['mean'][0,0,0],np.float32(1))
        result = m.initialization_gates(*arms,archived,target,obs,construction)
        self.assertFalse(result['gates']['D']['passed'])

    def test_envelope_against_independent_exact_reduction_oracle(self):
        u = Fraction(1,2**24)
        gamma = 75*u/(1-75*u)
        for order in ('ascending','descending','interleaved'):
            values = np.ldexp(np.ones(72,np.float32),np.arange(72)%21-10)
            if order == 'descending':
                values = values[::-1].copy()
            elif order == 'interleaved':
                values = np.r_[values[::2],values[1::2]]
            p0 = values.reshape(24,3)
            p1 = np.nextafter(p0,np.float32(np.inf))
            target = np.zeros((24,3),np.float32)
            losses = [m.roundoff.mse32(v,target)[0] for v in (p0,p1)]
            for v,loss in zip((p0,p1),losses):
                exact = sum(Fraction(float(x))**2 for x in v.flat)/72
                self.assertLessEqual(abs(Fraction(loss)-exact),gamma*exact)
            result = m.objective_envelope(p0,p1,target,*losses)
            self.assertTrue(result['passed'])
            self.assertTrue(result['exact_rational_comparison'])

    def test_envelope_assumptions_fail_closed(self):
        p0 = np.full((24,3),np.float32(1e-20))
        target = np.zeros_like(p0)
        losses = [m.roundoff.mse32(p0,target)[0]]*2
        with self.assertRaises(SignatureError):
            m.objective_envelope(p0,p0,target,*losses)
        p0.fill(np.float32(.2))
        with self.assertRaises(SignatureError):
            m.objective_envelope(p0,p0,target,0.,0.)

    def test_arithmetic_runtime_rounding_probe(self):
        info = m.arithmetic_runtime()
        self.assertEqual(info['maximum_nontrivial_sum_depth'],71)
        self.assertEqual(info['conservative_gamma_k'],75)
        self.assertEqual(info['fegetround'],0)

    def test_original_residual_J_and_proposal_on_fixture(self):
        z = torch.arange(7,dtype=torch.float32)/10
        residual = lambda x:x.repeat(11)[:72]/math.sqrt(72)
        counts = m.work()
        jac = m.lm.residual_jacobian(residual,z,residual(z),counts)
        expected = (torch.eye(7).repeat(11,1)[:72]/math.sqrt(72)).double()
        self.assertTrue(torch.equal(jac,expected))
        self.assertEqual(counts['JVP_completed'],7)
        self.assertEqual(counts['reverse_completed'],14)
        candidate,predicted,delta,represented = m.lm.proposal(z,jac,residual(z),.01)
        h,g = jac.T@jac,jac.T@residual(z).double()
        self.assertTrue(torch.equal(delta,torch.linalg.solve(h+.01*torch.diag(h.diag()+1e-8),-g)))
        self.assertTrue(torch.equal(candidate,(z.double()+delta).float()))
        self.assertEqual(predicted,float(-2*g.dot(represented)-represented.dot(h@represented)))

    def test_historical_first_proposal_tiny_solve_parity(self):
        for row in self.history['outcomes']:
            if row['arm'] != 'LM':
                continue
            rec = row['iterations'][0]
            h = torch.tensor(rec['normal_matrix'],dtype=torch.float64)
            g = torch.tensor(rec['gradient'],dtype=torch.float64)/2
            delta = torch.linalg.solve(h+.01*torch.diag(h.diag()+1e-8),-g)
            self.assertEqual(delta.tolist(),rec['trials'][0]['delta_float64'])

    def test_synthetic_LM_prefix_matches_original_without_policy_patch(self):
        a,b = SyntheticArm(),SyntheticArm()
        for fn,arm in ((m.lm.run_lm,a),(m.v1.guarded_loop,b)):
            with self.assertRaises(FixtureStop):
                fn(arm,lambda:None)
        self.assertTrue(m.v1.historical_control(b.row,a.row)['passed'])
        self.assertTrue(torch.equal(a.z,b.z))
        self.assertEqual(b.row['work']['accepted_parameter_steps'],2)

    def test_branch_stops_before_candidate_without_retry(self):
        arm = SyntheticArm(crossing=True)
        before = arm.z.detach().clone()
        with self.assertRaises(m.v1.BranchBlocked):
            m.v1.guarded_loop(arm,lambda:None)
        self.assertEqual(arm.calls,8)  # initial + seven JVP primals, no candidate
        self.assertEqual(arm.row['work']['linear_solves'],1)
        self.assertEqual(arm.row['work']['accepted_parameter_steps'],0)
        self.assertEqual(arm.row['work']['rejected_parameter_steps'],0)
        self.assertTrue(torch.equal(arm.z,before))
        self.assertEqual(arm.row['termination'],'branch_crossing_blocked')

    def test_original_damping_saturation_without_mutation(self):
        arm = SyntheticArm(constant=True)
        m.v1.guarded_loop(arm,lambda:None)
        self.assertEqual(arm.row['termination'],'damping_saturated')
        self.assertEqual(arm.row['damping'],1e8)
        self.assertEqual(arm.row['work']['accepted_parameter_steps'],0)
        self.assertEqual(arm.row['work']['rejected_parameter_steps'],11)

    def test_original_success_criteria(self):
        self.assertEqual(m.original.SETTINGS['success'],dict(Q_relative=.05,orientation_degrees=5.,
            max_albedo_absolute=.02,metallic_absolute=.03,held_out_relative_L2=.01))

    def test_common_gauge_evaluation_uses_one_original_batch(self):
        z = torch.tensor([.2,-.3,-.7,.1,.2,.3,-.1])
        arm = SimpleNamespace(z=m.v1.gauge(z),row=dict(arm='P1-C'),check=lambda:None)
        seen = []
        def forward(p,**kw):
            self.assertEqual(kw,dict(evaluation=True,physical=True))
            seen.append(p)
            return torch.ones((84,3,3))
        arm.forward = forward
        observations = [dict(split=s,view=i%12) for i,s in enumerate(['fit']*24+['held_out']*24+['relight']*36)]
        m.v1.evaluate(arm,np.ones((84,3,3)),observations)
        p = m.original.physical(arm.z)[m.v1.PERM].clone()
        p[2] -= math.pi/2
        self.assertEqual(len(seen),1)
        self.assertTrue(torch.equal(seen[0],p))

    def test_complete_work_caps_and_time_reserve(self):
        for key,value in m.v1.CAPS.items():
            self.assertEqual(m.CAPS[key],value)
        self.assertEqual(15*m.CAPS['forward_attempts'],13245)
        self.assertEqual(15*m.CAPS['jacobian_vector_evaluations'],8400)
        self.assertEqual(15*m.CAPS['reverse_mode_gradient_evaluations'],16815)
        self.assertEqual(15*(m.CAPS['fit_points']+m.CAPS['evaluation_points']),1275120)
        for name,cap in m.CAPS.items():
            c = m.work()
            c[name] = cap+1
            with self.assertRaises(m.v1.IntegrityFailure):
                m.check_caps(c)
        clock = m.lm.ArmClock(SimpleNamespace(check=lambda:None))
        clock.wall = 110.
        with self.assertRaises(m.lm.ArmLimit):
            clock.check(10.)
        clock.check(0.)
        c = m.work()
        c['initial_forward_completions'] = 1
        with self.assertRaises(m.v1.IntegrityFailure):
            m.check_caps(c)

    def test_actual_phase_accounting_with_non_scientific_fake_shader(self):
        class Shader:
            def shade(self,positions,normals,views,mx,my,albedo,metallic,encoding,light,tape,is_train):
                n = len(positions)
                wi = torch.tensor([0.,0.,1.]).reshape(1,1,3).expand(n,1024,3)
                diffuse,specular = albedo,mx.expand(n,3)
                return dict(diffuse_rgb=diffuse,specular_rgb=specular,linear_rgb=diffuse+specular,
                    frame_orthonormal=torch.ones(n,dtype=torch.bool),half_vectors=wi,specular_directions=wi,
                    diffuse_directions=wi[:,:512],pdf=torch.ones((n,1024,1)),tangent=torch.tensor([0.,1.,0.]).expand(n,3),
                    bitangent=torch.tensor([-1.,0.,0.]).expand(n,3),normal=normals)
        scene = self.old['scene']
        tape,_ = m.original.make_tapes(list(range(4)),24)
        ev,_ = m.original.make_tapes(list(range(1000,1004)),84)
        fit = m.original.ObservationBatch(scene,scene['observations'][:24],tape,4)
        evaluation = m.original.ObservationBatch(scene,scene['observations'],ev,4)
        row = dict(arm='P0',work=m.work())
        arm = m.Arm(row,Clock(),Shader(),fit,evaluation,torch.zeros((24,3)),
            torch.tensor([.1,.2,-.3,-.2,.1,-.3,.2],requires_grad=True))
        state = arm.forward(arm.z)
        self.assertIsNotNone(arm.initial_capture)
        arm.stage = 'optimization'
        arm.in_jacobian = True
        m.lm.residual_jacobian(arm.residual,arm.z,state['residual'].detach(),row['work'])
        arm.in_jacobian = False
        arm.forward(arm.z)
        arm.stage = 'endpoint'
        state = arm.forward(arm.z)
        row['work']['terminal_gradients'] += 1
        arm.gradient(state,arm.z)
        arm.forward(m.original.physical(arm.z).detach(),evaluation=True,physical=True)
        counts = row['work']
        for key in ('initial_forwards','initial_forward_completions','candidate_forwards','candidate_forward_completions',
                    'terminal_fit_forwards','terminal_fit_forward_completions','terminal_gradients','terminal_gradient_completions',
                    'evaluation_forward_attempts','evaluation_forward_evaluations'):
            self.assertEqual(counts[key],1,key)
        self.assertEqual(counts['full_batch_forward_evaluations'],11)
        self.assertEqual(counts['reverse_completed'],15)
        self.assertEqual(counts['fit_points'],960)
        self.assertEqual(counts['evaluation_points'],336)

    def test_partial_strict_JSON_preserved(self):
        stream = io.StringIO()
        m.coupling.accepted.persist_report(stream,dict(status='partial',initialization=[]))
        saved = stream.getvalue()
        with self.assertRaises(ValueError):
            m.coupling.accepted.persist_report(stream,dict(status='failed',loss=float('nan')))
        self.assertEqual(stream.getvalue(),saved)

    def test_all_initial_forwards_and_gates_precede_optimization(self):
        source = Path(m.__file__).read_text()
        self.assertLess(source.index("phase='initial_forwards'"),source.index("report['phase'] = 'initial_gates'"))
        self.assertLess(source.index("require(report['initialization_gate'] == 'passed'"),source.index('v1.run_arm(arm,targets,obs,persist)'))
        self.assertNotIn('roundoff.capture(',source)
        self.assertNotIn('encoding_override',source)
        self.assertNotIn('initial_objective_gate(',source)
        for n,h in self.identities.items():
            self.assertEqual(m.digest(m.ROOT/n),h)


if __name__ == '__main__':
    unittest.main()
