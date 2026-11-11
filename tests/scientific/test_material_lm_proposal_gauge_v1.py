"""New deterministic synthetic systems and read-only provenance fixtures only."""
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

from experiments import material_lm_proposal_gauge_v1 as audit
from infrastructure.signatures import SignatureError


def synthetic_history():
    z=np.zeros(7,np.float32)
    p=np.array([.2,.5,.1,.3,.4,.6,.2],np.float32)
    def typed(a):return {**audit.describe(a),'dtype':'torch.float32','device':'cpu'}
    row=dict(arm='LM',status='complete',budget_limited=False,termination='damping_saturated',
        initial_state=dict(unconstrained=typed(z),physical=typed(p)),initial_fit_MSE=1.,iterations=[],trajectory=[])
    damping=.01
    while True:
        rec=dict(outer_iteration=len(row['iterations'])+1,z=z.tolist(),physical=p.tolist(),training_MSE=1.,lambda_before=damping,trials=[])
        for k in range(4):
            rec['trials'].append(dict(trial=k+1,damping=damping,status='rejected',reason='Zero/negative/nonfinite predicted reduction',predicted_reduction=-0.))
            if damping==1e8:break
            damping=min(1e8,damping*10)
        rec.update(status='damping_saturated' if rec['trials'][-1]['damping']==1e8 else 'complete',
            lambda_after=damping,z_after=z.tolist(),training_MSE_after=1.)
        row['iterations'].append(rec)
        row['trajectory'].append(dict(outer_iteration=rec['outer_iteration'],damping=damping,unconstrained=z.tolist(),physical=p.tolist()))
        if rec['status']=='damping_saturated':break
    row.update(damping=damping,completed_outer_iterations=len(row['iterations']),
        final_state=dict(unconstrained=typed(z),physical=typed(p)),final=p.tolist(),final_fit_MSE=1.)
    row['trajectory'].append(dict(phase='endpoint',training_MSE=1.))
    return row


def fixture():
    i=np.arange(72*7,dtype=float).reshape(72,7)
    j=.01*np.sin(i*.7)+np.eye(72,7)*np.linspace(.1,2.,7)
    r=np.linspace(-.1,.2,72)
    return j.astype(np.float64),r.astype(np.float64)


class LMProposalGaugeTests(unittest.TestCase):
    def test_historical_terminal_damping_and_exact_state_pointers(self):
        history=audit.read_json('docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json')
        expected={101:(14,1),102:(17,3),103:(11,4),104:(9,2),105:(14,4)}
        for i,row in enumerate(history['outcomes']):
            if row['arm']!='LM':continue
            result=audit.terminal_damping(row,i)
            self.assertEqual(result['value'],1e8)
            self.assertEqual((result['terminal_outer_iteration'],result['terminal_trial']),expected[row['seed']])
            self.assertTrue(result['counterfactual'])
            self.assertEqual(result['termination'],'damping_saturated')
            self.assertEqual(result['terminal_state_sha256'],row['final_state']['unconstrained']['sha256'])
            self.assertIsNotNone(result['last_accepted'])

    def test_missing_contradictory_or_nonfinite_damping_is_rejected(self):
        row=synthetic_history()
        self.assertEqual(audit.terminal_damping(row,0)['value'],1e8)
        for field,value in [('damping',1.),('damping',float('nan')),('completed_outer_iterations',1),('termination','converged')]:
            bad=copy.deepcopy(row);bad[field]=value
            with self.assertRaises(SignatureError):audit.terminal_damping(bad,0)
        bad=copy.deepcopy(row);del bad['damping']
        with self.assertRaisesRegex(SignatureError,'Missing damping'):audit.terminal_damping(bad,0)
        for mutation in ('lambda','trial','state'):
            bad=copy.deepcopy(row)
            if mutation=='lambda':bad['iterations'][-1]['lambda_after']=1.
            elif mutation=='trial':bad['iterations'][0]['trials'][0]['damping']=.02
            else:bad['iterations'][-1]['z_after'][0]=1.
            with self.assertRaises(SignatureError):audit.terminal_damping(bad,0)

    def test_original_normal_equation_and_diagonal_epsilon(self):
        j,r=fixture();h,g,d,a=audit.system(j,r,.01)
        np.testing.assert_array_equal(h,j.T@j)
        np.testing.assert_array_equal(g,j.T@r)
        np.testing.assert_array_equal(d,np.diag(h)+1e-8)
        np.testing.assert_array_equal(a,h+.01*np.diag(np.diag(h)+1e-8))
        self.assertFalse(np.array_equal(a,h+.01*np.eye(7)))

    def test_normalization_matches_original_float32_residual_source(self):
        values=torch.from_numpy(np.linspace(.001,.95,72,dtype=np.float32)).reshape(24,3)
        target=torch.from_numpy(np.linspace(.12,.63,72,dtype=np.float32)).reshape(24,3)
        original=((values-target).reshape(-1)/math.sqrt(72)).numpy()
        translated=(values.numpy()-target.numpy()).reshape(-1)/np.float32(math.sqrt(72))
        np.testing.assert_array_equal(translated,original)
        self.assertAlmostEqual(float(original.astype(float)@original.astype(float)),float(torch.mean((values-target)**2)),delta=1e-8)

    def test_case_input_chain_conversion_and_array_hashes(self):
        gradients=audit.read_json('docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json')
        original=audit.read_json('docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json')
        s=next(s for s in gradients['states'] if s['id']=='LM-101')
        case=s['fit']['cases']['P0'];coordinate=s['coordinate_reconstruction']['representations']['P0']
        target=np.asarray(original['targets']['fine'],float)[:24,2].astype(np.float32)
        data=audit.case_inputs(case,coordinate,target)
        np.testing.assert_array_equal(data['J_latent'],data['J_physical'].astype(float)*data['D_transform'])
        np.testing.assert_array_equal(data['J_normalized'],data['J_latent']/math.sqrt(72))
        self.assertFalse(data['J_latent'].flags.writeable)
        bad=copy.deepcopy(case);bad['optimizer_coordinates']['jacobians']['total']['array']['sha256']='wrong'
        with self.assertRaisesRegex(SignatureError,'hash'):audit.case_inputs(bad,coordinate,target)

    def test_permutation_proof_general_system_small_large_damping(self):
        j,r=fixture();p=np.eye(7)[audit.PERM]
        np.testing.assert_array_equal(p@p,np.eye(7))
        for damping in (1e-8,.01,1e8):
            h0,g0,d0,a0=audit.system(j,r,damping)
            h1,g1,d1,a1=audit.system(j@p.T,r,damping)
            np.testing.assert_allclose(h1,p@h0@p.T,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(g1,p@g0,rtol=1e-14,atol=1e-14)
            np.testing.assert_array_equal(d1,d0[audit.PERM])
            report={};budget=audit.Budget(report)
            delta0,_=budget.solve(a0,g0);delta1,_=budget.solve(a1,g1)
            np.testing.assert_allclose(delta1,p@delta0,rtol=1e-12,atol=1e-15)
            self.assertEqual(report['work']['completed_solves'],2)

    def test_independent_diagonal_solution_anisotropic_sensitivity_and_near_stationarity(self):
        for scale in (1.,1e-12):
            j=np.zeros((72,7),float);widths=np.geomspace(.001,10.,7);j[:7]=np.diag(widths)
            r=np.zeros(72,float);r[:7]=scale*np.linspace(-.03,.04,7)
            for damping in (1e-8,1e8):
                h,g,d,a=audit.system(j,r,damping)
                report={};delta,check=audit.Budget(report).solve(a,g)
                expected=-widths*r[:7]/(widths**2+damping*(widths**2+1e-8))
                np.testing.assert_allclose(delta,expected,rtol=2e-15,atol=1e-30)
                self.assertTrue(check['passed'])

    def test_numpy_translation_matches_original_proposal_on_synthetic_inputs(self):
        # Compile only the reviewed pure proposal function for this tiny fixture.
        source=(audit.ROOT/'experiments/material_recovery_optimizer_comparison_v1.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='proposal')
        scope={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'original-proposal-fixture','exec'),scope)
        j,r=fixture();z=np.linspace(-.6,.7,7).astype(np.float32)
        h,g,_,a=audit.system(j,r,.01);delta,_=audit.Budget({}).solve(a,g)
        candidate,represented=audit.round_candidate(z,delta)
        tc,tp,td,tr=scope['proposal'](torch.tensor(z),torch.tensor(j),torch.tensor(r),.01)
        np.testing.assert_allclose(delta,td.numpy(),rtol=1e-12,atol=1e-14)
        np.testing.assert_array_equal(candidate,tc.numpy());np.testing.assert_array_equal(represented,tr.numpy())
        self.assertAlmostEqual(audit.predicted(g,h,represented,.01)['value'],tp,delta=1e-14)

    def test_float32_rounding_and_representation_quantization_are_distinct(self):
        z=np.array([.1,-.3,.31,.2,.4,.6,.8],np.float32);delta=np.full(7,3e-8)
        c,r=audit.round_candidate(z,delta)
        np.testing.assert_array_equal(c,(z.astype(float)+delta).astype(np.float32))
        np.testing.assert_array_equal(r,c.astype(float)-z.astype(float))
        z1=z[audit.PERM].copy();z1[2]=float(z[2])+math.pi/2
        _,r1=audit.round_candidate(z1,delta[audit.PERM])
        self.assertNotEqual(r[2],r1[2])  # Different theta grids, despite equal mathematical steps.

    def test_zero_represented_step_is_not_called_convergence(self):
        j=np.eye(72,7,dtype=float);r=np.full(72,1e-12)
        h,g,_,a=audit.system(j,r,1e8);delta,_=audit.Budget({}).solve(a,g)
        _,represented=audit.round_candidate(np.full(7,.3,np.float32),delta)
        self.assertTrue(np.all(represented==0))
        self.assertGreater(audit.predicted(g,h,delta,1e-7)['value'],0)
        pred=audit.predicted(g,h,represented,1e-7)
        self.assertEqual(pred['sign'],'zero');self.assertTrue(pred['too_small_to_interpret'])
        self.assertTrue(pred['negative_zero'])

    def test_predicted_reduction_uses_represented_step_and_all_signs_are_retained(self):
        g=np.arange(1.,8.);h=np.eye(7);step=np.linspace(-.4,.1,7)
        self.assertEqual(audit.predicted(g,h,step,.01)['value'],float(-2*g@step-step@(h@step)))
        self.assertEqual(audit.predicted(g,h,g,.01)['sign'],'negative')
        self.assertEqual(audit.predicted(g,h,-.1*g,.01)['sign'],'positive')
        with np.errstate(over='ignore',invalid='ignore'):
            nonfinite=audit.predicted(g,h,np.full(7,1e300),.01)
        self.assertEqual(nonfinite['sign'],'nonfinite');self.assertIsNone(nonfinite['value'])
        json.dumps(nonfinite,allow_nan=False)

    def test_singular_or_ill_conditioned_solve_no_fallback_or_extra_solve(self):
        report={};budget=audit.Budget(report)
        with self.assertRaises(np.linalg.LinAlgError):budget.solve(np.zeros((7,7)),np.ones(7))
        self.assertEqual(report['work']['normal_equation_solves'],1);self.assertEqual(report['work']['completed_solves'],0)
        a=np.diag(np.geomspace(1e-20,1.,7));g=np.diag(a).copy()
        delta,residual=budget.solve(a,g)
        np.testing.assert_array_equal(delta,-np.ones(7));self.assertTrue(residual['passed'])
        self.assertLess(audit.spectra(a)['numerical_rank'],7)

    def test_fifteen_solve_cap_and_memory_gate(self):
        report={};budget=audit.Budget(report);report['work']['normal_equation_solves']=15
        with patch.object(np.linalg,'solve',side_effect=AssertionError('No extra solve')):
            with self.assertRaisesRegex(SignatureError,'15-solve'):budget.solve(np.eye(7),np.ones(7))
        with patch.object(audit.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=audit.MEMORY_BYTES//1024+1)):
            with self.assertRaisesRegex(SignatureError,'RSS'):budget.check()

    def test_strict_complete_report_and_invalid_snapshot_preservation(self):
        j,r=fixture();h,g,d,a=audit.system(j,r,1e8);delta,residual=audit.Budget({}).solve(a,g)
        c,represented=audit.round_candidate(np.full(7,.3,np.float32),delta)
        record=dict(status='complete',H=audit.describe(h),g=audit.describe(g),D=audit.describe(d),A=audit.describe(a),
            delta=audit.describe(delta),candidate=audit.describe(c),represented=audit.describe(represented),solve=residual,
            predicted=audit.predicted(g,h,represented,.001),comparison=audit.vector_difference(delta,delta,audit.STEP_ATOL,audit.STEP_RTOL,audit.STEP_FLOOR))
        stream=io.StringIO();audit.persist_report(stream,record);before=stream.getvalue()
        self.assertEqual(json.loads(before),record)
        for bad in (np.bool_(True),float('nan')):
            with self.assertRaises((TypeError,ValueError)):audit.persist_report(stream,dict(bad=bad))
            self.assertEqual(stream.getvalue(),before)

    def test_physical_function_is_exact_original_and_Q_is_analysis_only(self):
        physical=audit.physical_function();z=np.linspace(-1.,1.,7,dtype=np.float32)
        expected=torch.cat([.04+.96*torch.sigmoid(torch.tensor(z[:2])),torch.tensor(z[2:3]),torch.sigmoid(torch.tensor(z[3:]))]).numpy()
        np.testing.assert_array_equal(physical(z),expected)
        q=audit.extract_function('roughness_tensor','experiments/material_recovery_metrics_v1.py',{'np':np,'require':audit.require})
        p=physical(z).astype(float);p1=p[audit.PERM].copy();p1[2]+=math.pi/2
        np.testing.assert_allclose(q(*p[:3]),q(*p1[:3]),atol=1e-14,rtol=1e-12)

    def test_invalid_array_hash_and_residual_inputs_fail(self):
        rec=audit.describe(np.ones(7,np.float32));rec['sha256']='wrong'
        with self.assertRaisesRegex(SignatureError,'hash'):audit.read_array(rec,(7,),'float32')
        j,r=fixture()
        for bad in (np.nan,0.,1e9):
            with self.assertRaises(SignatureError):audit.system(j,r,bad)
        with self.assertRaises(SignatureError):audit.system(j.astype(np.float32),r,.01)

    def test_failure_and_collision_preserve_evidence_memory_and_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            target,protocol=Path(directory)/'output.json',Path(directory)/'protocol.md'
            target.write_text('keep');protocol.write_text('fixture')
            handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
            with patch.object(audit,'OUTPUT',target),patch.object(audit,'PROTOCOL',protocol),patch.dict(audit.os.environ,audit.REQUIRED_ENV), \
                 patch.object(sys,'argv',['fixture','--output',str(target)]),patch.object(audit,'admit',side_effect=SignatureError('fixture integrity')), \
                 patch.object(audit.resource,'setrlimit') as limits:
                with self.assertRaises(FileExistsError):audit.main()
                self.assertEqual(target.read_text(),'keep');self.assertEqual(limits.call_count,0)
                target.unlink()  # Disposable fixture only.
                with self.assertRaisesRegex(SignatureError,'fixture integrity'):audit.main()
                report=json.loads(target.read_text());self.assertEqual(report['status'],'failed');self.assertTrue(report['memory_limit_restored'])
                self.assertEqual(limits.call_count,2)
            self.assertEqual(handlers,{s:signal.getsignal(s) for s in handlers})

    def test_no_shader_AD_optimizer_update_or_rng_calls(self):
        trng,nrng=torch.get_rng_state().clone(),np.random.get_state()
        with patch.object(torch.autograd,'grad',side_effect=AssertionError('No AD')),patch.object(torch.optim,'Adam',side_effect=AssertionError('No optimizer')):
            audit.physical_function()(np.zeros(7,np.float32))
            j,r=fixture();h,g,_,a=audit.system(j,r,.01);audit.Budget({}).solve(a,g)
        self.assertTrue(torch.equal(trng,torch.get_rng_state()));self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]))
        tree=ast.parse(Path(audit.__file__).read_text())
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'shade','backward','grad','requires_grad_','Adam','LBFGS','step','proposal','run_lm','restore','cuda','load','manual_seed','make_tapes','pinv','inv','lstsq'})
        self.assertNotIn('experiments.material_shader_v1',sys.modules)
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='solve' and
            isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='linalg' for n in ast.walk(tree)),1)

    def test_source_evidence_hashes_and_frozen_snapshots_preserved(self):
        for name,sha in audit.INPUTS.items():self.assertEqual(audit.digest(audit.ROOT/name),sha)
        for stage,pin in audit.PINS.items():audit.baseline(stage,pin)
        for name in ['experiments/material_recovery_optimizer_comparison_v1.py','experiments/run_known_material_recovery_v1.py','experiments/material_shader_v1.py']:
            self.assertEqual((audit.ROOT/name).read_bytes(),subprocess.check_output(['git','show',audit.BASE+':'+name],cwd=audit.ROOT))


if __name__=='__main__':
    unittest.main()
