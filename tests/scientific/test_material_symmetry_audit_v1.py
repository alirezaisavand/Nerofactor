"""NEW tiny symmetry/common-tape fixtures; no previous experiment or optimization."""
import ast
import copy
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_symmetry_audit_v1 as audit
from experiments.material_shader_v1 import MaterialShaderV1, RandomInputs
from infrastructure.signatures import SignatureError


class TinyBatch:
    n,seeds=1,4
    observations=[dict(view=0,light=0,split='fit')]
    def __init__(self):
        rng=np.random.RandomState(7)
        self.draws=RandomInputs(*(torch.tensor(rng.uniform(size=s).astype(np.float32)) for s in ((4,16),(4,16),(4,1,1))))
    def __call__(self,shader,p):
        r=shader.shade(torch.zeros(4,3),torch.tensor([[0.,0.,1.]]).repeat(4,1),torch.tensor([[.2,-.1,np.sqrt(.95)]]).float().repeat(4,1),
            p[:1].reshape(1,1).expand(4,1),p[1:2].reshape(1,1).expand(4,1),p[3:6].reshape(1,3).expand(4,3),
            p[6:].reshape(1,1).expand(4,1),torch.stack([torch.cos(2*p[2]),torch.sin(2*p[2])]).reshape(1,2).expand(4,2),
            lambda positions,directions:(.1+.9*directions[...,2:3].clamp(min=0)**8).expand_as(directions),self.draws,is_train=True)
        return torch.stack([r[k].reshape(4,1,3).mean(0) for k in ('diffuse_rgb','specular_rgb','linear_rgb')],1)


def capture(p=None):
    report={};budget=audit.Budget(report)
    p=torch.tensor(audit.reference.TRUTH if p is None else p,dtype=torch.float32)
    data=audit.capture_forward(MaterialShaderV1('A2','atan2',8,16),TinyBatch(),p,budget,[dict(seed=i,fixture=True) for i in range(4)])
    return data,p,report


class MaterialSymmetryAuditTests(unittest.TestCase):
    def test_transformations_preserve_other_parameters_and_input(self):
        p=audit.reference.TRUTH.copy();before=p.copy()
        for label,theta in [('P0',p[2]),('P1',p[2]+np.pi/2),('P2',p[2]+np.pi)]:
            q=audit.transform(p,label);self.assertEqual(q.dtype,np.float64);self.assertEqual(q[2],theta)
            np.testing.assert_array_equal(q[3:],p[3:]);np.testing.assert_array_equal(q[:2],p[1::-1] if label=='P1' else p[:2])
        np.testing.assert_array_equal(p,before)
        with self.assertRaises(SignatureError):audit.transform(p,'P3')

    def test_anisotropic_tensor_eigenorientation_and_independent_kernel_symmetry(self):
        p=audit.reference.TRUTH;base=audit.physics(p)
        for label in ('P0','P1','P2'):
            row=audit.physical_equivalence(audit.physics(audit.transform(p,label)),base)
            self.assertTrue(row['passed']);self.assertTrue(row['orientation_identifiable'])
            self.assertLess(row['Q']['relative_frobenius'],1e-13)
            self.assertLess(row['kernel']['relative_frobenius'],1e-12)

    def test_isotropic_analytical_fixture_orientation_is_null(self):
        p=audit.reference.TRUTH.copy();p[:2]=.4;base=audit.physics(p)
        for label in ('P1','P2'):
            row=audit.physical_equivalence(audit.physics(audit.transform(p,label)),base)
            self.assertTrue(row['passed']);self.assertFalse(row['orientation_identifiable'])
            self.assertIsNone(row['major_axis_error_radians'])

    def test_non_equivalent_physics_fails_instead_of_becoming_sampling_effect(self):
        p=audit.reference.TRUTH.copy();q=p.copy();q[0]+=.1
        row=audit.physical_equivalence(audit.physics(q),audit.physics(p))
        self.assertFalse(row['passed']);self.assertFalse(row['checks']['Q']);self.assertFalse(row['checks']['kernel'])

    def test_nominal_float64_and_rounded_float32_are_separate_records(self):
        p=audit.reference.TRUTH.copy();base=audit.physics(p.astype(np.float32).astype(float))
        for label in ('P1','P2'):
            q=audit.transform(p,label);rounded=audit.physics(q.astype(np.float32).astype(float))
            check=audit.physical_equivalence(rounded,base,rounded=True)
            self.assertTrue(check['passed']);self.assertGreater(check['Q']['relative_frobenius'],0)
            self.assertNotIn('kernel',check['checks'])

    def test_float32_encoding_roundoff_and_actual_frames_preserved(self):
        original,_,_=capture();periodic,_,_=capture(audit.transform(audit.reference.TRUTH,'P2'))
        self.assertFalse(np.array_equal(original['encoding'],periodic['encoding']))
        self.assertLess(np.max(abs(original['encoding']-periodic['encoding'])),1e-6)
        np.testing.assert_allclose(original['frame_Q'],periodic['frame_Q'],atol=audit.Q32_ATOL,rtol=audit.Q32_RTOL)
        self.assertEqual(len(original['decoder_coefficients']),2)

    def test_capture_per_tape_shape_components_sample_masks_and_no_grad(self):
        data,p,report=capture()
        self.assertEqual(data['per_tape'].shape,(4,1,3,3));self.assertEqual(data['mean'].shape,(1,3,3))
        self.assertFalse(data['mean_tensor'].requires_grad);self.assertFalse(p.requires_grad)
        np.testing.assert_allclose(data['per_tape'][:,:,0]+data['per_tape'][:,:,1],data['per_tape'][:,:,2],rtol=2e-6,atol=2e-7)
        self.assertEqual(data['samples']['specular_directions'].shape,(4,1,16,3))
        self.assertEqual(data['mask'].shape,(4,1,16))
        summary=audit.mask_summary(data['mask']);self.assertEqual(summary['valid_total']+summary['rejected_total'],64)
        self.assertEqual(report['work']['shader_calls'],1);self.assertEqual(report['work']['completed_shader_calls'],1)
        self.assertEqual(report['work']['specular_directions'],64)

    def test_common_tape_and_observation_mismatches_fail_closed(self):
        a,_,_=capture()
        b=dict(a,tape_identities=[dict(seed=99)])
        with self.assertRaises(SignatureError):audit.sampling_comparison(a,b)
        with self.assertRaises(SignatureError):audit.sampling_comparison(a,dict(a,observation_identity='different'))

    def test_sample_comparison_identity_and_fixed_correspondence(self):
        data,_,_=capture();result=audit.sampling_comparison(data,data)
        for value in result['fields'].values():self.assertEqual(value['indexwise_all_tapes']['max_absolute'],0.)
        self.assertEqual(result['masks']['flipped_total'],0)
        self.assertIn('different world samples',result['fields']['pdf']['meaning'])

    def test_masks_both_transition_directions_and_per_tape_counts(self):
        a=np.zeros((4,2,8),bool);b=a.copy();a[0,0,0]=True;b[1,1,2]=True
        r=audit.mask_difference(a,b)
        self.assertEqual((r['flipped_total'],r['valid_to_rejected'],r['rejected_to_valid']),(2,1,1))
        self.assertEqual(sum(map(sum,r['per_tape_observation_flips'])),2)
        with self.assertRaises(SignatureError):audit.mask_summary(np.zeros((4,8),bool))

    def test_individual_tape_differences_can_cancel_in_four_tape_mean(self):
        base=np.zeros((4,2,3,3),np.float32);other=base.copy()
        other[0,:,0]=other[0,:,2]=.01;other[1,:,0]=other[1,:,2]=-.01
        a=dict(per_tape=other,mean=other.mean(0));b=dict(per_tape=base,mean=base.mean(0))
        result=audit.radiance_comparison(a,b,[dict(split='fit'),dict(split='held_out')])
        self.assertEqual(set(result),{'fit','held_out'})
        self.assertEqual(result['fit']['total']['four_tape_mean']['RMSE'],0.)
        self.assertGreater(result['fit']['total']['individual_tapes'][0]['RMSE'],0.)

    def test_original_objective_is_mse_after_mean_not_mean_of_tape_mses(self):
        data,p,_=capture();target=torch.zeros(1,3)
        row=audit.serialize_forward(data,p,np.zeros((1,3,3)),TinyBatch.observations,target)
        self.assertEqual(row['objective_float32'],float(torch.mean(data['mean_tensor'][:,2]**2)))
        self.assertGreater(np.mean(row['individual_tape_objectives_float32']),row['objective_float32'])

    def test_exact_saved_mean_reuse_requires_hash_dtype_shape_and_values(self):
        a=np.arange(9,dtype=np.float32).reshape(1,3,3)
        reused,record=audit.reuse_saved(a,a.astype(float).tolist(),True,audit.reference.sha_array(a.astype(float)),np.float64)
        np.testing.assert_array_equal(reused,a);self.assertTrue(record['reused'])
        with self.assertRaisesRegex(SignatureError,'SHA-256'):audit.reuse_saved(a,a.tolist(),True,'wrong',np.float32)
        b=a.copy();b[0,0,0]=1
        with self.assertRaisesRegex(SignatureError,'identity mismatch'):
            audit.reuse_saved(a,b.tolist(),True,audit.reference.sha_array(b),np.float32)

    def test_ineligible_cache_uses_one_existing_baseline_not_unmatched_values(self):
        a=np.ones((1,3,3),np.float32)
        out,record=audit.reuse_saved(a,'not even array data',False,reason='Arithmetic order differs')
        self.assertIs(out,a);self.assertFalse(record['reused'])

    def test_objective_differences_report_sign_and_frozen_roundoff_bound(self):
        r=audit.objective_difference(2e-7,1e-7)
        self.assertEqual(r['signed_difference'],1e-7);self.assertEqual(r['relative_signed_difference'],1.)
        self.assertTrue(r['beyond_roundoff_bound'])
        self.assertFalse(audit.objective_difference(1e-7+1e-14,1e-7)['beyond_roundoff_bound'])

    def test_no_optimizer_parameter_updates_or_global_rng_in_forward_pipeline(self):
        rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        with patch.object(torch.optim,'Adam',side_effect=AssertionError('No optimizer')), \
             patch.object(torch.optim,'LBFGS',side_effect=AssertionError('No optimizer')), \
             patch.object(torch.Tensor,'backward',side_effect=AssertionError('No gradients')):
            data,_,_=capture();audit.physics(audit.reference.TRUTH)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]))
        self.assertFalse(data['mean_tensor'].requires_grad)
        tree=ast.parse(Path(audit.__file__).read_text())
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'Adam','SGD','LBFGS','step','backward','grad','requires_grad_','restore','run_adam','run_lm','make_grids','render','hemisphere','initializations','manual_seed','cuda','load'})

    def test_shader_point_and_memory_budgets_reject_before_extra_forward(self):
        report={};budget=audit.Budget(report);report['work']['shader_calls']=36
        with self.assertRaises(SignatureError):budget.before_forward(4)
        report['work']['shader_calls']=0;report['work']['points']=audit.MAX_POINTS
        with self.assertRaises(SignatureError):budget.before_forward(4)
        with patch.object(audit.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=audit.RSS_BYTES//1024+1)):
            with self.assertRaisesRegex(SignatureError,'RSS'):budget.check()

    def test_corrupt_evidence_hash_fails_before_parse_or_shader(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json';p.parent.mkdir(parents=True);p.write_text('not json')
            with patch.object(audit,'ROOT',root),patch.object(audit.previous,'admit',return_value=({},None,{})), \
                 patch.object(audit,'MaterialShaderV1',side_effect=AssertionError('No shader')):
                with self.assertRaisesRegex(SignatureError,'SHA-256'):audit.admit()

    def test_bad_physical_gate_stops_before_shader_and_cleans_threads(self):
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
        report={};threads=torch.get_num_threads()
        with patch.object(audit,'MaterialShaderV1',side_effect=AssertionError('No shader')):
            with self.assertRaises(SignatureError):
                audit.execute({},None,dict(runtime=runtime),[dict(id='bad',nominal_physical=[float('nan')]*7)],report,lambda:None)
        self.assertEqual(report['status'],'failed');self.assertEqual(report['work']['shader_calls'],0)
        self.assertTrue(report['threads_restored']);self.assertEqual(torch.get_num_threads(),threads)

    def test_pinned_and_accepted_scientific_sources_remain_unchanged(self):
        for stage,pin in audit.PINS.items():audit.baseline(stage,pin)
        for name in ('experiments/material_shader_v1.py','experiments/material_recovery_metrics_v1.py','audits/anisotropic_brdf.py',
                     'experiments/run_known_material_recovery_v1.py','experiments/material_recovery_optimizer_comparison_v1.py'):
            self.assertEqual((audit.ROOT/name).read_bytes(),subprocess.check_output(['git','show',audit.BASE+':'+name],cwd=audit.ROOT))

    def test_collision_and_failed_admission_preserve_output_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json';target.parent.mkdir(parents=True)
            protocol=root/'p.md';protocol.write_text('fixture');target.write_text('preserve')
            handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
            with patch.object(audit,'ROOT',root),patch.object(audit,'PROTOCOL',protocol),patch.dict(audit.os.environ,audit.original.REQUIRED_ENV), \
                 patch.object(audit,'admit',side_effect=SignatureError('fixture gate')),patch.object(audit.resource,'setrlimit') as limits, \
                 patch.object(sys,'argv',['audit','--output',str(target)]):
                with self.assertRaises(FileExistsError):audit.main()
                self.assertEqual(target.read_text(),'preserve');self.assertEqual(limits.call_count,0)
                target.unlink()  # Disposable fixture only, never research evidence.
                with self.assertRaises(SignatureError):audit.main()
                failed=json.loads(target.read_text());self.assertEqual(failed['status'],'failed');self.assertTrue(failed['memory_limit_restored'])
                self.assertEqual(limits.call_count,2)
            self.assertEqual(handlers,{s:signal.getsignal(s) for s in handlers})


if __name__=='__main__':unittest.main()
