"""New synthetic/read-only tests; no scientific shader or quadrature experiment."""
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

from experiments import multi_material_recovery_v1 as m
from infrastructure.signatures import SignatureError
from tests.scientific.test_material_lm_gauge_trajectories_v2 import gate_fixture, update_means


class FixtureClock:
    wall=cpu=0.
    def check(self):pass
    def elapsed(self):return 0.
    def active(self):
        from contextlib import nullcontext
        return nullcontext()


def linear_reference(p, observations, grids, budget, check=lambda:None):
    """A full-rank synthetic 72x7 map, deliberately dependent on supplied truth."""
    x=np.tile(np.asarray(p),(11,1)).reshape(-1)[:72].reshape(24,3)
    values=np.stack((x*.3,x*.7,x),axis=1)
    return dict(coarse=values.copy(),fine=values.copy(),absolute_delta=np.zeros_like(values),
        limit=np.ones_like(values),passed=True,physical_parameters=np.asarray(p).tolist())


def new_gates():
    arms,_,target,obs,construction=gate_fixture()
    fit,_,manifest=m.tapes()
    copied,record=m.coupling.copy_coupled_tape(fit,1)
    construction['coupled_tape']=record
    for arm,t in zip(arms,(fit,fit,copied)):
        arm.fit.tapes=t
        arm.initial_capture['tape_arrays']=[m.oldref.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)]
        arm.row['fit_target_float32_sha256']=m.oldref.sha_array(target.astype(np.float32))
    saved=dict(material='M1',seed=101,status='complete',completed_updates=1000,latest_state=copy.deepcopy(arms[0].row['initial_state']))
    return arms,saved,target,obs,construction,manifest


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archival,cls.history,cls.ids=m.admit()

    def test_complete_historical_chain_and_failed_attempts_preserved(self):
        self.assertEqual(len(self.ids),83)
        for name in ('docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json',
                     'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json',
                     'docs/MATERIAL_INDEPENDENT_TAPE_LM_V1_RESULTS.md'):
            self.assertIn(name,self.ids)
        self.assertEqual(self.history['status'],'complete')

    def test_fixed_panel_exact_float64_and_domains(self):
        expected={'M1':[.28,.49,-.62,.36,.58,.24,.10],'M2':[.17,.62,.91,.56,.31,.18,.65],
                  'M3':[.37,.47,.19,.24,.43,.70,.35]}
        self.assertEqual(m.reference.PANEL,expected)
        for name,p in expected.items():
            self.assertEqual(m.reference.truth(name).dtype,np.float64)
            np.testing.assert_array_equal(m.reference.truth(name),p)
        for bad in ([.01,.5,0,.3,.4,.5,.2],[.2,.5,0,.3,.4,.5,1.1],[.2]*6,[float('nan')]*7):
            with self.assertRaises(SignatureError):m.reference.validate(bad)

    def test_truth_is_explicit_no_historical_global_mutation(self):
        before=m.oldref.TRUTH.copy()
        with patch.object(m.reference,'render',side_effect=linear_reference) as calls:
            result=m.reference.identifiability(m.reference.truth('M2'),[],m.oldref.design()['observations'],{},lambda:None,lambda *a:None)
        self.assertEqual(len(result['evaluations']),28)
        for call in calls.call_args_list:
            delta=np.asarray(call.args[0])-m.reference.truth('M2')
            self.assertEqual(np.count_nonzero(delta),1)
        np.testing.assert_array_equal(m.oldref.TRUTH,before)
        self.assertEqual(result['explicit_truth'],m.reference.PANEL['M2'])
        self.assertTrue(result['passed'])

    def test_material_specific_normalized_reference_Jacobians(self):
        with patch.object(m.reference,'render',side_effect=linear_reference):
            r=m.reference.identifiability(m.reference.truth('M3'),[],m.oldref.design()['observations'],{},lambda:None,lambda *a:None)
        j=np.asarray(r['jacobians'])
        self.assertEqual(j.shape,(2,2,72,7))
        for col,scale in enumerate(m.oldref.SCALES):
            expected=np.zeros(72);expected[col::7]=scale
            np.testing.assert_allclose(j[0,0,:,col],expected,rtol=0,atol=1e-11)
        self.assertEqual(r['steps'][0]['rank_at_declared_ratio'],7)
        self.assertGreater(r['smallest_singular_value_stability_margin'],0)
        self.assertEqual(np.asarray(r['condition_numbers']).shape,(2,2))

    def test_perturbation_convergence_failure_retains_all_28(self):
        def failed(*args,**kwargs):
            r=linear_reference(*args,**kwargs);r['passed']=False;return r
        with patch.object(m.reference,'render',side_effect=failed):
            r=m.reference.identifiability(m.reference.truth('M1'),[],m.oldref.design()['observations'],{},lambda:None,lambda *a:None)
        self.assertFalse(r['passed']);self.assertEqual(len(r['evaluations']),28)
        self.assertTrue(r['jacobian_gate_passed'])

    def test_original_rank_policy_failure_retained(self):
        def deficient(*args,**kwargs):
            r=linear_reference(*args,**kwargs);r['coarse'][:]=1;r['fine'][:]=1;return r
        with patch.object(m.reference,'render',side_effect=deficient):
            r=m.reference.identifiability(m.reference.truth('M1'),[],m.oldref.design()['observations'],{},lambda:None,lambda *a:None)
        self.assertFalse(r['passed']);self.assertFalse(r['jacobian_gate_passed'])
        self.assertTrue(all(v is None for row in r['condition_numbers'] for v in row))
        json.dumps(r,allow_nan=False)

    def test_reference_wrapper_passes_explicit_material_and_counts(self):
        grids=[dict(directions=np.zeros((2,3))),dict(directions=np.zeros((3,3)))]
        observations=[dict(view=0),dict(view=1),dict(view=0)]
        def fake(p,obs,grids,budget,check):
            budget['reference_direction_evaluations']=budget.get('reference_direction_evaluations',0)+10
            x=np.full((3,3,3),p[0])
            return dict(coarse=x.copy(),fine=x.copy(),absolute_delta=np.zeros_like(x),limit=np.ones_like(x),passed=True)
        with patch.object(m.oldref,'render',side_effect=fake) as call:
            r=m.reference.render(m.reference.truth('M2'),observations,grids,{})
        np.testing.assert_array_equal(call.call_args.args[0],m.reference.truth('M2'))
        self.assertEqual(r['direction_evaluations'],10);self.assertEqual(r['maximum_bound_ratio'],0)

    def test_reference_preflight_cap_stops_before_render(self):
        with patch.object(m.oldref,'render') as render:
            with self.assertRaises(SignatureError):
                m.reference.render(m.reference.truth('M1'),[dict(view=0)],[dict(directions=np.zeros((2,3)))],
                    {'reference_direction_evaluations':m.reference.REFERENCE_CAP-1})
            render.assert_not_called()

    def test_new_and_original_evaluation_tape_hashes(self):
        fit,evaluation,manifest=m.tapes()
        self.assertEqual(tuple(r['sha256'] for r in manifest['fit']['per_tape']),(
            '161fbbff92dc5db501ec3bd1e032fbe449d0f635d2cba85244481b7d1f869f34',
            '56fb2f218e48d452511ee72911e63e16daef938f868f4fbcdadcf2f0b7f1c037',
            'b0c6f92624381ea6de65aae4436d7275897ec46f2c0662c1fac683953a67a8b5',
            '0b31d2144bb7148b0ade28ebeffcc2c8080eb2b067487865224260af22668583'))
        self.assertEqual(manifest['evaluation']['per_tape'],self.archival['tapes']['evaluation'])
        self.assertFalse(set(m.TRAIN_SEEDS)&set((*m.EVAL_SEEDS,0,1,2,3,2000,2001,2002,2003)))
        bad=copy.deepcopy(manifest);bad['fit']['arrays']['v']['sha256']='0'*64
        with self.assertRaises(SignatureError):m.tapes(bad)
        fit2,_,again=m.tapes(manifest)
        self.assertTrue(torch.equal(fit.u,fit2.u));self.assertNotEqual(fit.u.data_ptr(),fit2.u.data_ptr())

    def test_global_RNG_and_fresh_original_initializations(self):
        rng,nrng=torch.get_rng_state().clone(),np.random.get_state()
        rows=m.fresh_initializations();m.tapes()
        self.assertEqual([r['seed'] for r in rows],[101,103,105])
        for r in rows:
            old=next(x for x in m.original.initializations() if x['seed']==r['seed'])
            self.assertEqual(r['initial_unconstrained_float32'],old['initial_unconstrained_float32'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])

    def test_continuous_synthetic_Adam_and_both_snapshots(self):
        row=m.fresh_initializations()[0];row['material']='M1'
        def batch(shader,p):
            value=p[:3].repeat(24,1)
            return torch.stack((value*.3,value*.7,value),dim=1)
        target=torch.zeros((24,3));clock=FixtureClock()
        with patch.object(m.torch.optim,'Adam',wraps=torch.optim.Adam) as factory:
            m.fit_adam(row,None,batch,target,clock,lambda:None)
        self.assertEqual(factory.call_count,1)
        self.assertEqual(row['completed_updates'],1000)
        self.assertEqual(row['work']['optimizer_returns'],1000)
        self.assertEqual(row['work']['forward_attempts'],1001)
        self.assertEqual(row['work']['reverse_completed'],1000)
        for update in (250,1000):
            saved=row['snapshots'][str(update)]
            self.assertEqual(saved['adam']['step']['values'],update)
            self.assertEqual(saved['adam']['exp_avg']['shape'],[7])
            self.assertEqual(saved['objective_at_update'],update)
            z=torch.tensor(saved['unconstrained']['values'])
            objective=float(torch.mean(batch(None,m.original.physical(z))[:,2]**2))
            self.assertEqual(objective,saved['objective'])
        self.assertEqual(row['snapshots']['1000']['unconstrained'],row['latest_state']['unconstrained'])
        with self.assertRaises(SignatureError):m.fit_adam(row,None,batch,target,clock,lambda:None)

    def test_Adam_nonfinite_failure_preserves_state_no_retry(self):
        row=m.fresh_initializations()[0]
        with self.assertRaises(SignatureError):
            m.fit_adam(row,None,lambda s,p:torch.full((24,3,3),float('nan')),torch.zeros((24,3)),FixtureClock(),lambda:None)
        self.assertEqual(row['completed_updates'],0);self.assertEqual(row['status'],'failed')
        self.assertIn('latest_state',row)

    def test_material_target_association_rejects_foreign_truth(self):
        x=np.ones((84,3,3),np.float64)
        row=dict(material='M1',truth=m.reference.PANEL['M1'],targets=dict(passed=True,fine=x.tolist(),fine_sha256=m.oldref.sha_array(x)),
            fit_target_float32_sha256=m.oldref.sha_array(x[:24,2].astype(np.float32)))
        self.assertEqual(m.material_target(row,'M1')[0].shape,(84,3,3))
        with self.assertRaises(SignatureError):m.material_target(row,'M2')
        row['fit_target_float32_sha256']='0'*64
        with self.assertRaises(SignatureError):m.material_target(row,'M1')

    def test_truth_aware_recovery_never_reuses_archival_truth(self):
        for name in m.reference.PANEL:
            p=m.reference.truth(name)
            r=m.recovery_errors(p,p)
            self.assertEqual(r['Q_relative_frobenius'],0);self.assertEqual(r['metallic_absolute_error'],0)
            self.assertGreater(m.original.recovery_errors(p)['Q_relative_frobenius'],0)
            gauge=p.copy();gauge[:2]=p[1::-1];gauge[2]+=math.pi/2
            self.assertLess(m.recovery_errors(gauge,p)['Q_relative_frobenius'],1e-14)

    def test_nonidentifiable_orientation_does_not_pass(self):
        p=m.reference.truth('M3');p[1]=p[0]
        r=m.recovery_errors(p,p)
        self.assertIsNone(r['orientation_degrees'])
        self.assertFalse(m.original.assess_success(r,{'held_out':{'total':{'relative_L2':0}}})['checks']['orientation'])
        self.assertIsNone(m.state_difference(p,p)['orientation_degrees'])

    def test_private_truth_binding_does_not_modify_historical_functions(self):
        row=dict(arm='P0',iterations=[],trajectory=[dict(physical=m.reference.PANEL['M1'],phase='initial',
            unconstrained=[0.]*7,training_MSE=.1,active_wall_seconds=0,work={'accepted_parameter_steps':0})])
        historical=m.v1.trajectory_analysis.__globals__['original']
        result=m.trajectory_analysis(row,m.reference.truth('M1'))
        self.assertEqual(result['points'][0]['Q_error'],0)
        self.assertIs(m.v1.trajectory_analysis.__globals__['original'],historical)

    def test_accumulated_Adam_and_whole_wall_caps(self):
        whole=SimpleNamespace(check=lambda:None)
        clock=m.AdamClock(whole);clock.wall=180
        with self.assertRaises(m.lm.ArmLimit):clock.check()
        clock.wall=179.9;clock.check()
        whole=m.WholeClock();whole.start-=7201
        with self.assertRaises(m.lm.WholeLimit):whole.check()

    def test_gates_reuse_components_and_exact_new_Adam_state(self):
        arms,saved,target,obs,c,manifest=new_gates()
        result=m.initial_gates(arms,saved,target,obs,c,manifest)
        self.assertTrue(result['passed']);self.assertEqual(result['material'],'M1')
        saved['latest_state']['unconstrained']['values'][0]+=1
        self.assertFalse(m.initial_gates(arms,saved,target,obs,c,manifest)['gates']['A']['passed'])

    def test_wrong_target_or_tapes_fail_gate_A(self):
        arms,saved,target,obs,c,manifest=new_gates()
        arms[1].row['fit_target_float32_sha256']='0'*64
        self.assertFalse(m.initial_gates(arms,saved,target,obs,c,manifest)['gates']['A']['passed'])
        arms[1].row['fit_target_float32_sha256']=m.oldref.sha_array(target.astype(np.float32))
        arms[1].initial_capture['tape_arrays'][0]='0'*64
        self.assertFalse(m.initial_gates(arms,saved,target,obs,c,manifest)['gates']['A']['passed'])

    def test_new_gate_A_cannot_override_sample_or_component_failure(self):
        arms,saved,target,obs,c,manifest=new_gates()
        arms[2].initial_capture['mask'][0,0]^=True
        self.assertFalse(m.initial_gates(arms,saved,target,obs,c,manifest)['passed'])
        arms,saved,target,obs,c,manifest=new_gates()
        arms[2].initial_capture['per_tape'][0,0,1,0]+=.01;update_means(arms[2],target)
        self.assertFalse(m.initial_gates(arms,saved,target,obs,c,manifest)['gates']['D']['passed'])

    def test_signed_coupling_and_gauge_physical_invariants(self):
        fit,_,_=m.tapes()
        for sign in (-1,1):
            copied,_=m.coupling.copy_coupled_tape(fit,sign)
            self.assertTrue(torch.equal(copied.u,fit.u));self.assertTrue(torch.equal(copied.azimuth,fit.azimuth))
            self.assertNotEqual(copied.v.data_ptr(),fit.v.data_ptr())
        z=torch.tensor(m.fresh_initializations()[0]['initial_unconstrained_float32'])
        self.assertTrue(torch.equal(m.v1.gauge(z)[3:],z[3:]))
        self.assertLess(m.v1.q_difference(m.original.physical(z).tolist(),m.original.physical(m.v1.gauge(z)).tolist()),2e-6)

    def test_frozen_branch_policy_is_before_render(self):
        self.assertIs(m.v1.guarded_loop.__globals__['lm'].run_lm,m.lm.run_lm)
        z=torch.tensor([0.,0.,-.2,0.,0.,0.,0.])
        initial=m.v1.chart(z,'P0');z[2]=.2
        self.assertTrue(m.v1.crossed(initial,m.v1.chart(z,'P0')))

    def test_LM_uses_original_controller_private_evaluator_only(self):
        z=torch.zeros(7);j=torch.eye(7).repeat(11,1)[:72].double();r=torch.ones(72)
        candidate,predicted,delta,represented=m.lm.proposal(z,j,r,.01)
        self.assertEqual(candidate.dtype,torch.float32)
        self.assertTrue(torch.equal(candidate,(z.double()+delta).float()))
        self.assertIs(m.v1.run_arm.__globals__['guarded_loop'],m.v1.guarded_loop)
        self.assertTrue(m.lm.acceptance(2.,1.,1.)[0])
        self.assertFalse(m.lm.acceptance(1.,1.,1.)[0])

    def test_explicit_truth_common_gauge_evaluation_and_compensation(self):
        truth=m.reference.truth('M2');p=truth.copy();p[:2]=p[1::-1];p[2]+=math.pi/2
        # Construct physical values through exactly the original latent transform.
        norm=np.r_[(p[:2]-.04)/.96,p[3:]];logits=np.log(norm/(1-norm));z=torch.tensor(np.r_[logits[:2],p[2],logits[2:]],dtype=torch.float32)
        values=np.ones((84,3,3),np.float32);target=values.astype(float);target[:,0]+=.01;target[:,1]-=.01
        calls=[]
        def forward(selected,**kwargs):calls.append((selected.clone(),kwargs));return torch.tensor(values)
        arm=SimpleNamespace(z=z,row={'arm':'P1-C'},check=lambda:None,forward=forward)
        m.evaluate(arm,target,m.oldref.design()['observations'],truth)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],{'evaluation':True,'physical':True})
        self.assertLess(arm.row['recovery_errors']['Q_relative_frobenius'],1e-6)
        self.assertLess(arm.row['component_compensation']['held_out']['residual_correlation'],-.99)
        self.assertEqual(arm.row['radiance_errors']['held_out']['total']['relative_L2'],0)

    def test_resource_and_work_caps_exact(self):
        self.assertEqual(m.BUDGETS['total_forward_attempts'],32850)
        self.assertEqual(m.BUDGETS['total_reverse_sweeps'],39267)
        self.assertEqual(m.BUDGETS['reference_direction_evaluations'],353894400)
        self.assertEqual(m.BUDGETS['LM_per_arm'],m.accepted.CAPS)
        counts={k:0 for k in m.ADAM_CAPS};counts['optimizer_returns']=1001
        with self.assertRaises(SignatureError):m.check_adam(counts)
        with self.assertRaises(SignatureError):m.check_total_work({'Adam_runs':[{}]*10})

    def test_global_reference_failure_stops_before_Adam_or_shader(self):
        _,_,manifest=m.tapes()
        freeze=dict(material_panel=m.reference.PANEL,initializations=m.fresh_initializations(),tapes=manifest,reference_grid_identities=[])
        report={};calls=[]
        def failed(p,obs,grids,budget,check):
            x=np.ones((84,3,3),np.float64)
            return dict(passed=False,fine=x,coarse=x,maximum_bound_ratio=2.)
        archival=copy.deepcopy(self.archival);archival['reference_grid_identities']=[]
        with patch.object(m,'admit',return_value=(archival,self.history,self.ids)),patch.object(m.oldref,'make_grids',return_value=[]),patch.object(m.reference,'render',side_effect=failed) as render,\
             patch.object(m.original,'MaterialShaderV1') as shader,patch.object(m,'fit_adam') as fit:
            m.execute(report,lambda:calls.append(report.get('phase')),SimpleNamespace(check=lambda:None),freeze)
        self.assertEqual(render.call_count,3);self.assertEqual(len(report['materials']),3)
        self.assertEqual(report['status'],'stopped_reference_gate');shader.assert_not_called();fit.assert_not_called()

    def test_all_27_initial_forwards_and_nine_gates_before_LM(self):
        import inspect
        source=inspect.getsource(m.execute)
        self.assertLess(source.index("report['phase']='LM_initial_gates'"),source.index('run_arm(arm'))
        self.assertLess(source.index("for arm in arms:\n        with arm.clock.active()"),source.index("report['phase']='LM_initial_gates'"))
        self.assertIn("if report['initialization_gate'] != 'passed':",source)
        self.assertEqual(m.SEEDS,(101,103,105));self.assertEqual(len(m.reference.PANEL)*len(m.SEEDS)*len(m.v1.ARMS),27)

    def test_strict_partial_JSON_persistence(self):
        stream=io.StringIO();m.coupling.accepted.persist_report(stream,{'phase':'reference','status':'running'})
        before=stream.getvalue()
        with self.assertRaises((ValueError,SignatureError)):
            m.coupling.accepted.persist_report(stream,{'bad':float('nan')})
        self.assertEqual(stream.getvalue(),before)

    def test_frozen_scientific_sources_match_original_identities(self):
        for name,sha in self.ids.items():
            self.assertEqual(m.accepted.digest(m.ROOT/name),sha,name)
        self.assertEqual(m.oldref.TRUTH.tolist(),[.2,.55,.31,.3,.45,.6,.25])


if __name__=='__main__':
    unittest.main()
