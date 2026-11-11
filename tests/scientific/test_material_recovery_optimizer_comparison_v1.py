"""NEW tiny optimizer/AD fixtures; never run the full scientific comparison."""
import ast
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_recovery_optimizer_comparison_v1 as run
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs
from infrastructure.signatures import SignatureError


class FakeClock:
    wall,cpu=0.,0.
    def check(self,*args): pass
    def elapsed(self): return 0.
    @contextmanager
    def active(self): yield


class ToyBatch:
    n,seeds=24,4
    def __call__(self,shader,p):
        shader.mask=np.ones((96,1024),bool)
        a=torch.arange(72*7,dtype=torch.float32).reshape(72,7)
        color=(torch.sin(a*.13)@p).reshape(24,3)
        return torch.stack([color*.4,color*.6,color],dim=1)


def toy_row():
    z=torch.tensor([-.2,.3,.4,-.5,.6,.7,-.8],requires_grad=True)
    opt=torch.optim.Adam([z],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    for _ in range(1000):
        opt.zero_grad(); ((z-.2)**2).sum().backward();opt.step()
    saved=dict(status='complete',completed_updates=1000,final=run.original.physical(z).detach().tolist(),
               latest_state=run.accepted.snapshot(z,opt))
    return json.loads(json.dumps(saved)),z,opt


def toy_arm(name='LM'):
    row=dict(arm=name,work=run.work(),trajectory=[],iterations=[],additional_updates=0,completed_outer_iterations=0)
    arm=run.Arm(row,FakeClock(),None,ToyBatch(),ToyBatch(),torch.zeros(24,3),
                torch.tensor([-.2,.3,.4,-.5,.6,.7,-.8],requires_grad=True))
    arm.current=run.detached(arm.forward(arm.z));return arm


class OptimizerComparisonTests(unittest.TestCase):
    def test_json_restore_matches_uninterrupted_adam_at_update1000(self):
        saved,z,opt=toy_row();other,restored=run.restore(saved)
        before=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
        self.assertEqual(run.accepted.snapshot(z,opt),run.accepted.snapshot(other,restored))
        for _ in range(3):
            for p,o in ((z,opt),(other,restored)):
                o.zero_grad();((p-.2)**2).sum().backward();o.step()
            self.assertEqual(run.accepted.snapshot(z,opt),run.accepted.snapshot(other,restored))
        self.assertEqual(float(restored.state[other]['step']),1003.)
        self.assertEqual(saved,before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_restore_rejects_hash_dtype_device_physical_counter_and_options(self):
        saved,_,_=toy_row()
        for key,bad in [('sha256','wrong'),('dtype','torch.float64'),('device','cuda:0'),('shape',[1,7])]:
            row=copy.deepcopy(saved);row['latest_state']['unconstrained'][key]=bad
            with self.assertRaises(SignatureError):run.restore(row)
        for kind in ('counter','physical','options','moment'):
            row=copy.deepcopy(saved)
            if kind=='counter':row['completed_updates']=999
            if kind=='physical':row['final'][0]=.125
            if kind=='options':row['latest_state']['optimizer_effective']['lr']=.02
            if kind=='moment':row['latest_state']['adam']['exp_avg']['values'][0]=float('nan')
            with self.assertRaises(SignatureError):run.restore(row)

    def test_residual_norm_matches_original_mse_and_arms_start_identically(self):
        a,b=toy_arm('Adam'),toy_arm('LM')
        self.assertTrue(torch.equal(a.z,b.z));self.assertTrue(torch.equal(a.current['values'],b.current['values']))
        self.assertEqual(float(a.current['loss']),float(b.current['loss']))
        self.assertTrue(math.isclose(float(a.current['residual'].double().square().sum()),
            float(a.current['loss']),rel_tol=run.MSE_RTOL,abs_tol=run.MSE_ATOL))

    def test_initial_identity_mismatch_fails_before_any_update(self):
        for field in ('coordinates','predictions','residual','loss','saved_loss'):
            a,b=toy_arm('Adam'),toy_arm('LM');saved=dict(final_fit_MSE=float(a.current['loss']))
            states=[a.current,b.current]
            if field=='coordinates':b.z=(b.z.detach()+.01).requires_grad_()
            if field=='predictions':states[1]['values']=states[1]['values']+.01
            if field=='residual':states[1]['residual']=states[1]['residual']+.01
            if field=='loss':states[1]['loss']=states[1]['loss']+.01
            if field=='saved_loss':saved['final_fit_MSE']+=.01
            with self.assertRaises(SignatureError):run.verify_start(saved,[a,b],states)
            self.assertEqual(a.row['work']['accepted_parameter_steps'],0)

    def test_seven_jvps_dimensions_analytic_fd_and_reverse_chain(self):
        matrix=torch.sin(torch.arange(504,dtype=torch.float32).reshape(72,7)*.13)
        z=torch.linspace(-.3,.4,7,requires_grad=True)
        def f(x):return torch.sin(matrix@x)/math.sqrt(72)
        expected=f(z).detach();counts=run.work()
        jac=run.residual_jacobian(f,z,expected,counts)
        exact=(torch.cos(matrix@z)[:,None]*matrix/math.sqrt(72)).double()
        np.testing.assert_allclose(jac.numpy(),exact.detach().numpy(),rtol=2e-6,atol=1e-8)
        for h in (1e-4,5e-5):
            columns=[]
            for i in range(7):
                d=torch.zeros_like(z);d[i]=h;columns.append((f(z.detach()+d)-f(z.detach()-d))/(2*h))
            np.testing.assert_allclose(jac.numpy(),torch.stack(columns,1).numpy(),rtol=.01,atol=3e-4)
        g,=torch.autograd.grad(f(z).square().sum(),z)
        np.testing.assert_allclose((2*jac.T@expected.double()).numpy(),g.numpy(),rtol=2e-6,atol=2e-7)
        self.assertEqual((counts['jacobian_vector_evaluations'],counts['reverse_mode_gradient_evaluations'],counts['jacobians_completed']),(7,14,1))

    def test_real_a2_tiny_fixture_supports_double_reverse_and_fd_contract(self):
        # One synthetic point,8/16 samples: a derivative fixture, not benchmark observations.
        shader=MaterialShaderV1('A2','atan2',8,16);rng=np.random.RandomState(7)
        draws=RandomInputs(*(torch.tensor(rng.uniform(size=s).astype(np.float32)) for s in ((1,16),(1,16),(1,1,1))))
        z=torch.tensor([-1.4,.2,.31,-.8,-.2,.4,-1.],requires_grad=True)
        def f(x):
            p=run.original.physical(x)
            result=shader.shade(torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),torch.tensor([[.2,-.1,math.sqrt(.95)]]),
                p[:1].reshape(1,1),p[1:2].reshape(1,1),p[3:6].reshape(1,3),p[6:].reshape(1,1),
                torch.stack([torch.cos(2*p[2]),torch.sin(2*p[2])]).reshape(1,2),
                lambda positions,directions:(.1+.9*directions[...,2:3].clamp(min=0)**8).expand_as(directions),draws)
            return result['linear_rgb'].expand(24,3).reshape(-1)/math.sqrt(72)
        state=torch.get_rng_state().clone();j=run.residual_jacobian(f,z,f(z).detach(),run.work())
        for h in (1e-4,5e-5):
            columns=[]
            for i in range(7):
                d=torch.zeros_like(z);d[i]=h;columns.append((f(z.detach()+d)-f(z.detach()-d))/(2*h))
            np.testing.assert_allclose(j.numpy(),torch.stack(columns,1).numpy(),rtol=.03,atol=2e-4)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_wrong_dimensions_or_changed_primal_fail_closed(self):
        z=torch.ones(7);f=lambda x:x.repeat(11)[:72]
        with self.assertRaises(SignatureError):run.residual_jacobian(f,z,torch.zeros(71),run.work())
        with self.assertRaises(SignatureError):run.residual_jacobian(f,z,torch.zeros(72),run.work())

    def test_float64_solve_uses_damped_diagonal_and_rounded_actual_delta(self):
        z=torch.linspace(-.1,.2,7);j=torch.eye(72,7,dtype=torch.float64);r=torch.linspace(-.2,.2,72)
        candidate,pred,delta,represented=run.proposal(z,j,r,.01)
        expected=torch.linalg.solve(j.T@j+.01*torch.diag((j.T@j).diag()+1e-8),-j.T@r.double())
        self.assertEqual(delta.dtype,torch.float64);self.assertEqual(candidate.dtype,torch.float32)
        self.assertTrue(torch.equal(delta,expected))
        self.assertTrue(torch.equal(represented,candidate.double()-z.double()))
        self.assertEqual(pred,float(-2*(j.T@r.double()).dot(represented)-represented.dot((j.T@j)@represented)))

    def test_acceptance_strict_ratio_decrease_and_invalid_predicted_reduction(self):
        self.assertEqual(run.acceptance(1.,.5,1.),(True,.5))
        self.assertFalse(run.acceptance(1.,1.,1.)[0])
        self.assertFalse(run.acceptance(1.,.9,2.)[0])
        self.assertEqual(run.acceptance(1.,0.,1e-320),(False,None))
        for value in (0.,-1.,float('nan'),float('inf')):self.assertEqual(run.acceptance(1.,.5,value),(False,None))

    def test_damping_thresholds_clamps_and_saturation_policy(self):
        self.assertEqual(run.accepted_lambda(.03,.8),.01)
        self.assertEqual(run.accepted_lambda(.03,.2),.06)
        self.assertEqual(run.accepted_lambda(.03,.75),.03)
        self.assertEqual(run.accepted_lambda(.03,.25),.03)
        self.assertEqual(run.accepted_lambda(1e-8,.8),1e-8)
        self.assertEqual(run.accepted_lambda(1e8,.2),1e8)

    def test_lm_accepts_bounded_steps_and_accounts_seven_jvps_per_iteration(self):
        arm=toy_arm()
        with patch.object(run,'LM_OUTERS',2):run.run_lm(arm,lambda:None)
        self.assertEqual(arm.row['work']['jacobians_completed'],2)
        self.assertEqual(arm.row['work']['JVP_completed'],14)
        self.assertEqual(arm.row['completed_outer_iterations'],2)
        self.assertGreater(arm.row['work']['accepted_parameter_steps'],0)
        self.assertLessEqual(arm.row['work']['linear_solves'],8)
        self.assertLess(float(arm.current['loss']),float(arm.row['iterations'][0]['training_MSE']))

    def test_rejection_at_most_four_trials_no_parameter_replacement(self):
        arm=toy_arm();before=arm.z.detach().clone();counter=0
        def nonpositive(z,j,r,d,c):
            nonlocal counter;counter+=1;return z.detach(),0.,torch.zeros(7,dtype=torch.float64),torch.zeros(7,dtype=torch.float64)
        with patch.object(run,'LM_OUTERS',2),patch.object(run,'proposal',side_effect=nonpositive):run.run_lm(arm,lambda:None)
        self.assertEqual(counter,8);self.assertTrue(torch.equal(before,arm.z))
        self.assertEqual([len(x['trials']) for x in arm.row['iterations']],[4,4])
        self.assertEqual(arm.row['work']['rejected_parameter_steps'],8)
        self.assertEqual(arm.row['termination'],'outer_iteration_ceiling')

    def test_saturation_stops_without_new_optimizer_or_retry(self):
        arm=toy_arm();zeros=torch.zeros(7,dtype=torch.float64)
        with patch.object(run,'LAMBDA_INITIAL',run.LAMBDA_MAX),patch.object(run,'proposal',return_value=(arm.z.detach(),0.,zeros,zeros)):
            run.run_lm(arm,lambda:None)
        self.assertEqual(arm.row['termination'],'damping_saturated')
        self.assertEqual(len(arm.row['iterations']),1);self.assertEqual(len(arm.row['iterations'][0]['trials']),1)
        self.assertEqual(arm.row['work']['accepted_parameter_steps'],0)

    def test_nonfinite_candidate_and_singular_solve_are_rejections(self):
        arm=toy_arm()
        for error in (SignatureError('Nonfinite proposal'),torch.linalg.LinAlgError('singular fixture')):
            with patch.object(run,'LM_OUTERS',1),patch.object(run,'proposal',side_effect=error):run.run_lm(arm,lambda:None)
        self.assertEqual(arm.row['work']['rejected_parameter_steps'],8)
        self.assertTrue(all(t['status']=='rejected' for i in arm.row['iterations'] for t in i['trials']))

    def test_nonfinite_proposal_coordinates_preserved_as_explicit_nulls(self):
        arm=toy_arm();bad=arm.z.detach().clone();bad[0]=float('inf');zero=torch.zeros(7,dtype=torch.float64)
        with patch.object(run,'LM_OUTERS',1),patch.object(run,'proposal',return_value=(bad,1.,zero,zero)):
            run.run_lm(arm,lambda:None)
        attempt=arm.row['iterations'][0]['trials'][0]
        self.assertEqual(attempt['nonfinite_coordinate_indices'],[0]);self.assertIsNone(attempt['proposed_unconstrained'][0])
        self.assertEqual(attempt['status'],'rejected');json.dumps(arm.row,allow_nan=False)

    def test_adam_original_step_counter_and_new_update_ceiling(self):
        saved,_,_=toy_row();z,opt=run.restore(saved);arm=toy_arm('Adam');arm.z=z;arm.optimizer=opt
        with patch.object(run,'ADAM_UPDATES',3):run.run_adam(arm,lambda:None)
        self.assertEqual((arm.row['additional_updates'],arm.row['work']['accepted_parameter_steps']),(3,3))
        self.assertEqual(float(opt.state[z]['step']),1003.)
        self.assertEqual(arm.row['work']['reverse_mode_gradient_evaluations'],3)
        self.assertEqual(arm.row['termination'],'update_ceiling')

    def test_budget_limited_arm_preserves_state_and_measures_endpoint_without_updates(self):
        arm=toy_arm();arm.row['budget_limited']=False;before=arm.z.detach().clone()
        def evaluated(a,*args):a.row.update(status='complete',final=run.original.physical(a.z).detach().tolist())
        with patch.object(run,'run_lm',side_effect=run.ArmLimit('fixture reserve')),patch.object(run,'evaluate',side_effect=evaluated):
            run.run_arm(arm,None,None,lambda:None)
        self.assertEqual(arm.row['status'],'complete_budget_limited');self.assertTrue(arm.row['budget_limited'])
        self.assertEqual(arm.row['termination'],'time_budget');self.assertTrue(torch.equal(arm.z,before))
        self.assertIn('final_gradient_norm',arm.row);self.assertEqual(arm.row['work']['accepted_parameter_steps'],0)

    def test_evaluation_reuses_original_metrics_and_accounts_one_forward(self):
        arm=toy_arm()
        class Eval:
            n,seeds=84,4
            def __call__(self,shader,p):return p.mean().expand(84,3,3)
        arm.evaluation=Eval();scene=run.reference.design();target=np.zeros((84,3,3))
        actual=arm.row;expected={}
        p=run.original.physical(arm.z).detach()
        run.original.evaluate_start(expected,p,None,Eval(),target,scene['observations'],{},lambda:None)
        run.evaluate(arm,target,scene['observations'])
        for key in ('final_predictions','prediction_sha256','recovery_errors','radiance_errors','success'):
            self.assertEqual(actual[key],expected[key])
        self.assertEqual(arm.row['work']['evaluation_forward_evaluations'],1)
        self.assertEqual(arm.row['work']['evaluation_points'],336)

    def test_runtime_failure_restores_threads_and_cpu_rng(self):
        report={};threads=torch.get_num_threads();rng=torch.get_rng_state().clone()
        with patch.object(run.platform,'python_version',return_value='different'):
            with self.assertRaises(SignatureError):
                run.execute({},dict(runtime=dict(python='original')),report,lambda:None,run.WholeClock())
        self.assertEqual(report['status'],'failed');self.assertTrue(report['threads_restored'])
        self.assertEqual(threads,torch.get_num_threads());self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_arm_clocks_exclude_idle_enforce_reserve_and_restore_global_alarm(self):
        whole=run.WholeClock();clock=run.ArmClock(whole)
        with patch.object(run.signal,'setitimer') as timer:
            with clock.active():
                self.assertIs(whole.active,clock);clock.check()
            self.assertIsNone(whole.active);self.assertIsNone(clock.started);self.assertEqual(timer.call_count,2)
        clock.wall=110.
        with self.assertRaises(run.ArmLimit):clock.check(10.)
        clock.check();clock.wall=120.
        with self.assertRaises(run.ArmLimit):clock.check()
        with self.assertRaises(run.WholeLimit):whole.alarm(signal.SIGTERM,None)

    def test_forward_caps_and_masks_fail_before_extra_work(self):
        arm=toy_arm();arm.row['work']['forward_attempts']=883
        with self.assertRaises(SignatureError):arm.forward(arm.z)
        with self.assertRaises(SignatureError):run.mask_identity(np.ones((1,16),bool))

    def test_no_prior_experiment_or_reference_generation_or_hidden_settings(self):
        tree=ast.parse(Path(run.__file__).read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'fit_start','continue_start','initializations','render','identifiability','make_grids','load','manual_seed','cuda','clip_grad_norm_'})
        self.assertEqual((run.ADAM_UPDATES,run.LM_OUTERS,run.LM_TRIALS),(750,80,4))
        self.assertEqual((run.ARM_SECONDS,run.WHOLE_SECONDS),(120.,1800.))
        self.assertEqual(run.original.SETTINGS['success'],dict(Q_relative=.05,orientation_degrees=5.,max_albedo_absolute=.02,metallic_absolute=.03,held_out_relative_L2=.01))

    def test_collision_preserves_existing_output_before_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json';target.parent.mkdir(parents=True);target.write_text('preserve')
            protocol=root/'p.md';protocol.write_text('fixture');source=root/'f.py';source.write_text('fixture')
            with patch.object(run,'ROOT',root),patch.object(run,'PROTOCOL',protocol),patch.object(run,'__file__',str(source)), \
                 patch.dict(run.os.environ,run.original.REQUIRED_ENV),patch.object(run,'admit',side_effect=AssertionError('No admission')), \
                 patch.object(sys,'argv',['comparison','--output',str(target)]):
                with self.assertRaises(FileExistsError):run.main()
            self.assertEqual(target.read_text(),'preserve')

    def test_admission_failure_preserves_partial_and_signal_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json';target.parent.mkdir(parents=True)
            protocol=root/'p.md';protocol.write_text('fixture');source=root/'f.py';source.write_text('fixture')
            handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
            with patch.object(run,'ROOT',root),patch.object(run,'PROTOCOL',protocol),patch.object(run,'__file__',str(source)), \
                 patch.dict(run.os.environ,run.original.REQUIRED_ENV),patch.object(run,'admit',side_effect=SignatureError('fixture gate')), \
                 patch.object(sys,'argv',['comparison','--output',str(target)]):
                with self.assertRaises(SignatureError):run.main()
            self.assertEqual(json.loads(target.read_text())['status'],'failed')
            self.assertEqual(handlers,{s:signal.getsignal(s) for s in handlers})


if __name__=='__main__':unittest.main()
