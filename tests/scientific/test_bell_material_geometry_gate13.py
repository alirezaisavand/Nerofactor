"""Focused synthetic CPU controls; no native model, checkpoint or ray tracing."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from benchmarks import bell_material_geometry_gate13 as m
from experiments.train_bell_material_geometry_gate13 import trainer_source,Acquisition
from experiments.analyze_bell_material_geometry_gate13 import physical,paired_mc,known_support
from infrastructure.bell_material_pair_gate13 import SamplingAudit,KnownBellV3Audit,same_record
from infrastructure.known_l2_audit import KnownL2Audit,LIGHT_POLICY


class Value:
    def __init__(self,a):self.a=np.asarray(a);self.shape=self.a.shape;self.dtype='torch.float32';self.device='cuda:0'
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class RandomFixture:
    def __init__(self):
        self.counter=0;self.cuda=SimpleNamespace(get_rng_state=lambda device:Value(np.array([self.counter],np.int64)))
    def rand(self,*shape,**kw):
        self.counter+=1;shape=shape[0] if len(shape)==1 and isinstance(shape[0],tuple) else shape
        return Value(np.full(shape,self.counter,dtype=np.float32))
    def normal(self,*args,**kw):
        self.counter+=1
        return Value(np.full(kw['size'],self.counter,dtype=np.float32))


def requests(torch):
    torch.rand((512,1024));torch.rand((512,1024));torch.rand(512,1,1);torch.rand(512,1)
    torch.normal(0.,.02,size=(512,1))


class Gate13(unittest.TestCase):
    def test_overlay_compiles_and_keeps_original_scientific_sites(self):
        code=trainer_source();ast.parse(code)
        self.assertIn('known_Bell_v3',code);self.assertNotIn("bound.get('illumination')=='known_L2'",code)
        for text in ('optimizer.step()','loss.backward()','loss=training_objective(',
                     "acquisition.sampling_step(step,torch)","ENVIRONMENT=acquisition.illumination"):
            self.assertEqual(code.count(text),1)
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='WarmUpCosLR'
                             for n in ast.walk(ast.parse(code))),1)
        self.assertNotIn('reference_Gate3_receipt_sha256',code)
        self.assertNotIn("budget('boundary_predictor_calls'",code)
        self.assertIn("first=data[58]",code)

    def test_illumination_audit_has_private_environment(self):
        self.assertIsNot(KnownBellV3Audit.attached.__wrapped__.__globals__,KnownL2Audit.attached.__wrapped__.__globals__)
        self.assertEqual(KnownBellV3Audit.attached.__wrapped__.__globals__['LIGHT_POLICY']['environment'],m.g.ENV)
        self.assertEqual(KnownL2Audit.attached.__wrapped__.__globals__['LIGHT_POLICY'],LIGHT_POLICY)
        self.assertNotEqual(LIGHT_POLICY['environment'],m.g.ENV)

    def test_random_observer_preserves_original_results_and_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit=SamplingAudit(tmp);torch=RandomFixture();before=torch.rand
            with audit.step(0,torch):requests(torch)
            self.assertEqual(torch.rand,before);self.assertEqual(torch.counter,5)
            audit.stream.close();row=json.loads(Path(audit.path).read_text())
            self.assertEqual(len(row['requests']),5);self.assertTrue(all('values_sha256' in r for r in row['requests']))

    def test_matched_random_stream_accepted_and_different_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').mkdir();(root/'b').mkdir();(root/'c').mkdir()
            a=SamplingAudit(root/'a');t=RandomFixture()
            with a.step(0,t):requests(t)
            a.stream.close();b=SamplingAudit(root/'b',a.path);t=RandomFixture()
            with b.step(0,t):requests(t)
            b.stream.close();b.reference.close()
            c=SamplingAudit(root/'c',a.path);t=RandomFixture();t.counter=1
            with self.assertRaises(ValueError):
                with c.step(0,t):requests(t)
            c.stream.close();c.reference.close()

    def test_exception_restores_random_methods_without_accepting_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SamplingAudit(tmp);t=RandomFixture();before=t.rand
            with self.assertRaises(RuntimeError):
                with a.step(0,t):raise RuntimeError('fixture failure')
            self.assertEqual(t.rand,before);self.assertEqual(a.steps,0);a.stream.close()

    def test_missing_random_request_and_early_close_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SamplingAudit(tmp);t=RandomFixture()
            with self.assertRaises(ValueError):
                with a.step(0,t):t.rand((512,1024))
            with self.assertRaises(ValueError):a.close()
            a.stream.close()

    def test_unobserved_cpu_rng_is_not_admitted(self):
        class CPUFixture(RandomFixture):
            def rand(self,*shape,**kw):
                v=super().rand(*shape,**kw);v.device='cpu';return v
        with tempfile.TemporaryDirectory() as tmp:
            a=SamplingAudit(tmp);t=CPUFixture()
            with self.assertRaises(ValueError):
                with a.step(0,t):requests(t)
            a.stream.close()

    def test_training_pool_does_not_include_material_labels(self):
        sample={k:np.zeros((1,3,3)) if k=='linear_components' else np.zeros((1,3)) for k in m.FIT_KEYS}
        sample['Q_world']=object();sample['color']=object()
        result=m.fit_pool({int(c):sample for c in m.split()['train']})
        self.assertEqual(set(result),set(m.FIT_KEYS));self.assertEqual(len(result['position']),122)

    def test_physical_transport_axis_swap_and_sign(self):
        # Simple exact 90-degree normal transport. Evaluator-only tensor tests.
        a=dict(hit_mask=np.array([True]),mx=np.array([[.4]]),my=np.array([[.2]]),
            tangents=np.array([[1.,.5,.5]]),bitangents=np.array([[.5,1.,.5]]),kd=np.array([[.3,.4,.5]]),metallic=np.array([[.2]]))
        d=dict(pixel_indices=np.array([0]),normals=np.array([[0.,0,1.]]),oracle_normals=np.array([[0.,1.,0.]]),
            Q_world=np.diag([.16,.04,0])[None],oracle_Q=np.diag([.16,0,.04])[None],axis=np.array([[1.,0,0.]]),
            oracle_axis=np.array([[-1.,0,0.]]),widths_major_minor=np.array([[.4,.2]]),oracle_widths=np.array([[.4,.2]]),
            color=a['kd'],oracle_color=a['kd'],metallic=a['metallic'],oracle_metallic=a['metallic'])
        val,q=physical(a,d);self.assertLess(val['Q_source'][0],1e-14);self.assertEqual(val['orientation_source'][0],0.)
        swapped=dict(a,mx=a['my'],my=a['mx'],tangents=a['bitangents'],bitangents=a['tangents'])
        alt,q2=physical(swapped,d);np.testing.assert_array_equal(q,q2)
        signed=dict(a,tangents=1-a['tangents'],bitangents=1-a['bitangents'])
        np.testing.assert_array_equal(physical(signed,d)[1],q)

    def test_common_draw_uncertainty_uses_paired_covariance(self):
        a=np.arange(8,dtype=float)[:,None,None,None]*np.ones((8,2,3,3))
        np.testing.assert_array_equal(paired_mc(a,a+1),0.)
        self.assertTrue((paired_mc(a,np.zeros_like(a))>0).all())

    def test_undefined_support_is_not_silently_zero_filled(self):
        d=dict(hit_mask=np.array([[True,True,False]]),full_linear_components=np.ones((3,3,3)))
        a=dict(hit_mask=np.array([[True,False,False]]),linear_rgb=np.array([[[1.,1.,1.],[0,0,0],[0,0,0]]]))
        r=known_support(d,a);self.assertEqual(r['undefined'],1);self.assertFalse(r['full_raster_score_defined'])

    def test_draw_capture_assembles_chunks_without_new_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=Acquisition.__new__(Acquisition);obj.draw_seconds=0.;obj.draw_buffer=None
            obj.f={'support':{'cameras':[dict(camera=58,common=3)]}};obj.args=SimpleNamespace(output=Path(tmp))
            a=np.ones((8,2,3,3),np.float32);b=np.full((8,1,3,3),2.,np.float32)
            obj.evaluation_draws(0,58,np.array([0,1]),a);obj.evaluation_draws(0,58,np.array([2]),b)
            from benchmarks.donut_recovery_v1 import numeric
            z=numeric(Path(tmp)/'native-draws/step-00000-camera-58.npz')
            np.testing.assert_array_equal(z['draws'],np.concatenate([a,b],axis=1))
            np.testing.assert_array_equal(z['component_reference_indices'],[2,0,1])
            self.assertIsNone(obj.draw_buffer)

    def test_pair_ledger_comparison_is_exact(self):
        same_record(dict(seed=6033),dict(seed=6033))
        with self.assertRaises(ValueError):same_record(dict(seed=6033),dict(seed=6034))


if __name__=='__main__':unittest.main()
