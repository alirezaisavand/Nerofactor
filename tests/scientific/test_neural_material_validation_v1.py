"""New synthetic/read-only Phase C preflight tests; zero actual shader calls."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import validate_neural_material_recovery_v1 as v
from tests.scientific.test_neural_material_recovery_v1 import FakeShader as AlgebraFixture
from infrastructure.signatures import SignatureError


class FakeShader(AlgebraFixture):
    def shade(self,*args,**kwargs):
        result=super().shade(*args,**kwargs)
        result['valid_specular_counts']=torch.full((len(args[0]),1),1024,dtype=torch.int64)
        return result


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference,cls.ids=v.admit();cls.fit,cls.eval_tape,cls.tapes=v.h.inputs(cls.reference)

    def test_exact_matrix_and_hard_ceiling(self):
        p=v.validate_plan(v.plan())
        self.assertEqual(p['caps']['shader_forward'],45);self.assertEqual(p['caps']['shader_backward'],36)
        self.assertEqual(p['caps']['field_vjp'],24);self.assertEqual(p['caps']['synthetic_Adam'],3)
        p['caps']['shader_forward']=64
        with self.assertRaises(SignatureError):v.validate_plan(p)

    def test_original_and_Phase_B_source_identity_gate(self):
        self.assertEqual(v.BASE,'a11efd615d18ac69fa1c26a3692af511f1c78302')
        for name,sha in v.IMPLEMENTATION.items():self.assertEqual(v.h.digest(v.ROOT/name),sha)
        with patch.dict(v.IMPLEMENTATION,{'experiments/neural_material_field_v1.py':'0'*64}):
            with self.assertRaises(SignatureError):v.admit()

    def test_failed_attempt_and_retry_accounting(self):
        b=v.Budget()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with b.operation('shader_forward','failed'):raise RuntimeError('fixture')
        self.assertEqual(b.counts['shader_forward'],dict(attempted=1,completed=0))
        with self.assertRaises(SignatureError):
            with b.operation('shader_forward','failed'):pass
        b.counts['shader_forward']['attempted']=45
        with self.assertRaises(SignatureError):
            with b.operation('shader_forward','extra'):pass
        self.assertFalse(b.finished())

    def test_comparison_reports_actual_values_indices_and_original_bounds(self):
        b=np.array([[1.,2.],[0.,3.]],np.float32);a=b.copy();a[1,0]=1e-6
        c=v.comparison(a,b,atol=2e-6,rtol=2e-5)
        self.assertTrue(c['passed']);self.assertEqual(c['worst_absolute_index'],[1,0]);self.assertFalse(c['exact'])
        self.assertEqual(c['candidate_at_worst'],float(a[1,0]))
        self.assertFalse(v.comparison(a,b)['passed'])
        a[0,0]=float('nan');self.assertFalse(v.comparison(a,b)['passed'])

    def test_angular_gate_is_not_replaced_by_component_gate(self):
        a=np.array([[1e-7,0.,0.]],np.float32);b=np.array([[0.,1e-7,0.]],np.float32)
        self.assertTrue(v.comparison(a,b,atol=2e-6)['passed'])
        self.assertFalse(v.comparison(a,b,atol=2e-6,vector=True)['passed'])

    def test_pdf_diagnostic_failure_cannot_silently_become_gate(self):
        c=v.comparison(np.array([2.],np.float32),np.array([1.],np.float32),mandatory=False)
        self.assertFalse(c['passed']);self.assertFalse(c['mandatory'])

    def test_numeric_evidence_hash_gate_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as root:
            with v.h.OwnedOutput(Path(root),'fixture') as output:
                output.path('raw').mkdir();a={'x':np.arange(5,dtype=np.float32),'mask':np.array([True,False])}
                m=v.save_arrays(output,'fixture',a)
                with patch.object(np,'load',wraps=np.load) as load:r=v.read_arrays(output,m)
                self.assertIs(load.call_args.kwargs['allow_pickle'],False);self.assertTrue(np.array_equal(r['x'],a['x']))
                m['sha256']='0'*64
                with patch.object(np,'load') as load:
                    with self.assertRaises(SignatureError):v.read_arrays(output,m)
                    load.assert_not_called()
                with self.assertRaises(SignatureError):v.save_arrays(output,'object',{'bad':np.array([object()],dtype=object)})

    def test_capture_is_observational_and_checks_actual_encoding_gradients(self):
        b=v.Budget();model=v.field.make_material('N0',v.h.fresh_latent(101),101)
        batch=v.h.MaterialBatch(self.reference['scene'],self.reference['scene']['observations'][:24],self.fit)
        arrays,row=v.capture(model,batch,FakeShader(),torch.full((24,3),.9),b,'fake',[])
        self.assertEqual(b.counts['shader_forward'],dict(attempted=1,completed=1))
        self.assertEqual(b.counts['shader_backward'],dict(attempted=1,completed=1))
        self.assertEqual(b.counts['field_query'],dict(attempted=1,completed=1))
        self.assertTrue(row['original_arithmetic_replay']);self.assertTrue(np.count_nonzero(arrays['encoding_gradient']))
        self.assertEqual(arrays['per_tape'].shape,(4,24,3,3))
        self.assertTrue(v.parity(arrays,arrays,torch.full((24,3),.9),exact=True)['passed'])

    def test_exact_parity_checks_per_tape_samples_and_gradients(self):
        model=v.field.make_material('N0',v.h.fresh_latent(101),101)
        batch=v.h.MaterialBatch(self.reference['scene'],self.reference['scene']['observations'][:24],self.fit)
        a,_=v.capture(model,batch,FakeShader(),torch.full((24,3),.9),v.Budget(),'fake',[])
        for key in ('per_tape','pdf','gradient::latent','encoding_gradient'):
            altered={k:x.copy() for k,x in a.items()};altered[key].flat[0]+=.01
            self.assertFalse(v.parity(altered,a,torch.full((24,3),.9),exact=True)['passed'],key)

    def test_invalid_orientation_rejection_uses_existing_decoder(self):
        self.assertTrue(all(r['rejected'] for r in v.zero_encoding_checks()))

    def test_failed_actual_frame_gate_preserves_raw_evidence_without_retry(self):
        class BrokenFixture(FakeShader):
            def shade(self,*args,**kwargs):
                r=super().shade(*args,**kwargs);r['frame_orthonormal'].zero_();return r
        model=v.field.make_material('N0',v.h.fresh_latent(101),101)
        batch=v.h.MaterialBatch(self.reference['scene'],self.reference['scene']['observations'][:24],self.fit)
        saved=[];budget=v.Budget()
        with self.assertRaises(SignatureError):
            v.capture(model,batch,BrokenFixture(),torch.full((24,3),.9),budget,'failed',[],failure_sink=lambda label,a:saved.append((label,a)))
        self.assertEqual(len(saved),1);self.assertIn('parameter::latent',saved[0][1])
        self.assertFalse(saved[0][1]['frame_orthonormal'].any())
        self.assertEqual(budget.counts['shader_forward']['attempted'],1)

    def test_predeclared_positions_order(self):
        self.assertTrue(torch.equal(v.points(1),torch.zeros((1,3))))
        self.assertEqual(v.points(9).tolist(),[[float(x),float(y),0.] for x in (-1,0,1) for y in (-1,0,1)])
        self.assertTrue(torch.equal(v.points(96),v.points(9).repeat(11,1)[:96]))

    def test_runtime_cleanup_and_no_budget_extension(self):
        b=v.Budget();report={};rng=torch.get_rng_state().clone();handler=signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with v.runtime(b,report):raise RuntimeError('fixture')
        self.assertEqual(signal.getsignal(signal.SIGALRM),handler);self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(report['memory_limit_restored'])
        b.start-=301
        with self.assertRaises(v.Stopped):b.check()

    def test_feasibility_rejects_provisional_budget_without_changing_it(self):
        timing=[dict(kind='real_forward_backward',arm=a,forward_seconds=.1,backward_seconds=.2) for a in v.h.ARMS]
        states=[dict(arm=a,Adam_seconds=.01,serialization_seconds=.1,archive_bytes=100) for a in v.h.ARMS]
        r=v.feasibility(timing,states);self.assertFalse(r['passed']);self.assertFalse(r['budget_changed'])
        self.assertTrue(all(row['provisional_active_cap_seconds']==900 for row in r['arms']))

    def test_check_plan_has_no_model_shader_or_output_allocation(self):
        with patch('sys.argv',['validation','--check-plan']),patch.object(v.field,'make_material') as make,\
             patch.object(v.h.previous.original,'MaterialShaderV1') as shader,patch.object(v.h,'OwnedOutput') as output,\
             contextlib.redirect_stdout(io.StringIO()) as stream:
            v.main()
        self.assertEqual(json.loads(stream.getvalue())['fixture_plan']['recovery_updates'],0)
        make.assert_not_called();shader.assert_not_called();output.assert_not_called()


if __name__=='__main__':unittest.main()
