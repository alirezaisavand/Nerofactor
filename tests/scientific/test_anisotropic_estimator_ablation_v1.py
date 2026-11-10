"""NEW 3B-2 unit fixtures only; never execute the full experiment or old suites."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments import anisotropic_estimator_v1 as core
from experiments import run_anisotropic_ablation_v1 as runner
from infrastructure.signatures import SignatureError
from infrastructure.stage_sources import ROOT,digest


def independent_original(count):
    """Separate extraction of the original pure methods for B0 bitwise fidelity."""
    p=ROOT/'scientific_sources/material_a/network/field.py';tree=ast.parse(p.read_text())
    cls=deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork'))
    cls.bases=[];cls.keywords=[];cls.decorator_list=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in core.METHODS]
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));namespace={'torch':torch,'np':np,'F':F}
    exec(compile(module,str(p),'exec'),namespace);obj=namespace['MCShadingNetwork']();obj.cfg={'random_azimuth':True}
    n=np.arange(count,2*count);gold=(np.sqrt(5)-1)/2
    az=(2*np.pi*n*gold)%(2*np.pi);el=np.arcsin(2*n/(2*count)-1)
    obj.diffuse_direction_samples=core.tensor(np.stack([az*.5/np.pi,1-2*el/np.pi],-1))
    return obj


class EstimatorAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous=torch.get_num_threads();cls.addClassCleanup(torch.set_num_threads,previous);torch.set_num_threads(1)
        cls.models,cls.identity=core.shaders(32);cls.stream=core.sample_stream(7,64);cls.wo=np.array([.8,0.,.6])
    def test_B0_bitwise_fidelity_to_independently_extracted_original(self):
        original=independent_original(32)
        source,sd,sc=core.draw(original,.6,.9,self.wo,self.stream)
        baseline,bd,bc=core.draw(self.models['B0'],.6,.9,self.wo,self.stream)
        self.assertEqual(sc,bc)
        for a,b in zip(source,baseline):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(sd,bd))
        np.testing.assert_array_equal(core.estimate(original,source,sd,self.wo,.6,.9),core.estimate(self.models['B0'],baseline,bd,self.wo,.6,.9))
        for name in core.METHODS:
            self.assertEqual(getattr(type(original),name).__code__.co_code,getattr(type(self.models['B0']),name).__code__.co_code)
    def test_B1_changes_only_specular_normalization(self):
        spec,diffuse,count=core.draw(self.models['B0'],.6,.9,self.wo,self.stream)
        self.assertGreater(count['rejected'],0);self.assertGreater(count['valid'],0)
        a=core.estimate(self.models['B0'],spec,diffuse,self.wo,.6,.9)
        b=core.estimate(self.models['B1'],spec,diffuse,self.wo,.6,.9)
        np.testing.assert_array_equal(a[:,0],b[:,0]);np.testing.assert_allclose(b[:,1],a[:,1]*count['valid']/64,rtol=2e-7,atol=1e-8)
        for name in core.METHODS-{'compute_radiance'}:
            self.assertEqual(getattr(type(self.models['B0']),name).__code__.co_code,getattr(type(self.models['B1']),name).__code__.co_code)
    def test_exact_M_division_and_retained_legacy_diagnostics(self):
        spec,diffuse,count=core.draw(self.models['B0'],.6,.9,self.wo,self.stream);h,wi,ch,pdf,t,b,n,theta,phi=spec
        args=[core.tensor([[.5]*3]),torch.ones((1,32,3)),torch.ones((1,64,3)),torch.ones((1,64,3)),diffuse,wi,
            core.tensor([[0,0,1]]),ch,core.tensor([self.wo]),pdf,theta,phi,core.tensor([[.6]]),core.tensor([[.9]])]
        old=self.models['B0'].compute_radiance(*args);new=self.models['B1'].compute_radiance(*args)
        # Preserve the original three-channel reduction layout: reducing one
        # scalar channel first can round differently in float32 CPU kernels.
        weights=(torch.clamp(wi[:,:,2:3],min=0)/torch.clamp(ch,min=1e-6))*(wi[:,:,2:3]>0)
        expected=weights.expand(-1,-1,3).contiguous().sum(dim=1)/64
        self.assertTrue(torch.equal(new[2],expected))
        for i in [1,3,4,5]:self.assertTrue(torch.equal(old[i],new[i]))
    def test_empty_valid_set_contributes_zero_for_all_variants(self):
        spec,diffuse,_=core.draw(self.models['B0'],.6,.9,self.wo,self.stream)
        spec=list(spec);spec[1]=core.tensor(np.tile([0,0,-1.],(1,64,1)))
        for name in core.VARIANTS:
            value=core.estimate(self.models[name],spec,diffuse,self.wo,.6,.9)
            np.testing.assert_array_equal(value[:,1],0.);self.assertTrue(np.isfinite(value).all())
    def test_B2_exact_height_recovery_and_original_azimuth(self):
        objects,identity=core.shaders(512);stream=core.sample_stream(2,16)
        a,old,_=core.draw(objects['B0'],.3,.7,self.wo,stream);b,new,_=core.draw(objects['B2'],.3,.7,self.wo,stream)
        np.testing.assert_array_equal(objects['B2'].underlying_heights.numpy(),np.arange(512,dtype=np.float32)/512)
        np.testing.assert_allclose(new.numpy()[0,:,2]**2,np.arange(512)/512,atol=2e-7)
        np.testing.assert_allclose(torch.norm(new,dim=-1).numpy(),1.,atol=2e-7)
        xy_old=F.normalize(old[:,:,:2],dim=-1);xy_new=F.normalize(new[:,:,:2],dim=-1)
        np.testing.assert_allclose(xy_old.numpy(),xy_new.numpy(),atol=2e-7)
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assertLess(identity['height_recovery']['maximum_unrounded_sine_error'],1e-12)
        self.assertGreater(identity['height_recovery']['maximum_inverse_rounded_storage_error'],0.)
    def test_correct_cosine_measure_retains_declared_left_endpoint_bias(self):
        objects,_=core.shaders(512);spec,diffuse,_=core.draw(objects['B2'],.3,.7,self.wo,core.sample_stream(0,16))
        values=core.estimate(objects['B2'],spec,diffuse,self.wo,.3,.7)
        d=.999*(1-.001)
        self.assertAlmostEqual(values[1,0],d*511/1024,delta=2e-7)
        self.assertGreater(abs(values[1,0]-d/2),2e-5)
    def test_four_arms_are_factorial_with_paired_counts_and_no_extra_randomness(self):
        drawn={name:core.draw(self.models[name],.6,.9,self.wo,self.stream) for name in core.VARIANTS}
        for name,(s,d,c) in drawn.items():
            self.assertEqual(c,drawn['B0'][2]);self.assertEqual(c['calls'],3);self.assertEqual(c['floats'],129)
            self.assertTrue(torch.equal(s[1],drawn['B0'][0][1]))
        self.assertTrue(torch.equal(drawn['B0'][1],drawn['B1'][1]));self.assertTrue(torch.equal(drawn['B2'][1],drawn['B3'][1]))
        values={name:core.estimate(self.models[name],s,d,self.wo,.6,.9) for name,(s,d,c) in drawn.items()}
        np.testing.assert_array_equal(values['B0'][:,0],values['B1'][:,0]);np.testing.assert_array_equal(values['B2'][:,0],values['B3'][:,0])
        np.testing.assert_array_equal(values['B0'][:,1],values['B2'][:,1]);np.testing.assert_array_equal(values['B1'][:,1],values['B3'][:,1])
    def test_original_decoder_is_unchanged_in_every_arm(self):
        for name in core.VARIANTS:
            s,_,_=core.draw(self.models[name],.3,.7,np.array([0,0,1.]),self.stream,rotation=[-1.,0.])
            self.assertAlmostEqual(float(torch.norm(s[4])),.001,places=7)
            self.assertGreater(abs(float(torch.norm(s[0],dim=-1).mean())-1),.01)
    def test_no_global_RNG_consumption_or_source_stream_mutation(self):
        t=torch.get_rng_state().clone();n=np.random.get_state();before=[x.copy() for x in self.stream]
        core.draw(self.models['B3'],.6,.9,self.wo,self.stream)
        self.assertTrue(torch.equal(t,torch.get_rng_state()));after=np.random.get_state()
        np.testing.assert_array_equal(n[1],after[1]);self.assertEqual(n[2:],after[2:])
        for a,b in zip(before,self.stream):np.testing.assert_array_equal(a,b)
    def test_invalid_random_tape_source_or_GPU_requests_fail_closed(self):
        with self.assertRaises(SignatureError):core.RandomTape([])(1)
        with self.assertRaises(SignatureError):core.RandomTape([np.ones((1,1))])(2)
        with self.assertRaises(SignatureError):core.RandomTape([np.ones((1,1))])((1,1),device='cuda')
        with patch.object(core,'SOURCE_SHA','0'*64):
            with self.assertRaises(SignatureError):core.source_classes()
    def test_no_GPU_model_or_checkpoint_execution(self):
        with patch.object(torch.cuda,'is_available',side_effect=AssertionError('No GPU query')), \
             patch.object(torch,'load',side_effect=AssertionError('No checkpoint')), \
             patch.object(torch.nn,'Linear',side_effect=AssertionError('No network')):
            objects,_=core.shaders(8);spec,diffuse,_=core.draw(objects['B3'],.6,.9,self.wo,core.sample_stream(0,8))
            core.estimate(objects['B3'],spec,diffuse,self.wo,.6,.9)
    def test_independent_quadrature_failure_is_unresolved_without_retry(self):
        directions=np.array([[0.,0.,1.]]);lights=core.lighting(directions)
        grids=[(directions,np.array([1.]),lights),(directions,np.array([2.]),lights)]
        result=runner.reference_integrals(.6,.9,self.wo,grids)
        self.assertEqual(result['status'],'partly_unresolved');self.assertFalse(np.asarray(result['converged']).any())
        raw=np.ones((16,4,3,3));stats,_=runner.summarize_case(raw,result)
        self.assertEqual(stats['B3'][0][0]['agreement'],'unresolved_reference');self.assertIsNone(stats['B3'][0][0]['absolute_bias'])
    def test_variance_and_paired_uncertainty_use_seed_differences(self):
        row=runner.summary(np.arange(16,dtype=float),7.5)
        self.assertEqual(row['mean'],7.5);self.assertAlmostEqual(row['sample_variance'],np.arange(16).var(ddof=1))
        raw=np.ones((16,4,3,3));raw[:,0]=2
        refs={'fine':np.ones((3,3)).tolist(),'absolute_grid_difference':np.zeros((3,3)).tolist(),'converged':np.ones((3,3),bool).tolist()}
        _,paired=runner.summarize_case(raw,refs)
        row=paired['B1_minus_B0'][0][0]
        self.assertEqual(row['absolute_error_difference']['mean'],-1.);self.assertEqual(row['absolute_error_difference']['standard_error'],0.)
        self.assertEqual(runner.summary(np.ones(16),0.)['relative_bias'],None)
    def test_effective_kernel_formula_changes_only_declared_components(self):
        row=runner.reciprocity(.6,.9,self.wo);v=row['variants']
        self.assertEqual(v['B1']['specular_absolute_difference'],0.)
        self.assertEqual(v['B3']['total_absolute_difference'],0.)
        self.assertEqual(v['B0']['specular_forward'],v['B2']['specular_forward'])
        self.assertEqual(v['B0']['diffuse_forward'],v['B1']['diffuse_forward'])
        self.assertNotEqual(v['B2']['specular_forward'],v['B2']['specular_reverse'])
    def test_failure_before_grid_restores_owned_runtime_without_full_experiment(self):
        threads=torch.get_num_threads();state=torch.get_rng_state().clone();report={'status':'running'}
        with patch.object(runner,'make_reference_grids',side_effect=RuntimeError('fixture stop before any experiment case')):
            with self.assertRaises(RuntimeError):runner.run(report,lambda:None)
        self.assertEqual(report['status'],'failed');self.assertEqual(report['budget']['cases'],0)
        self.assertTrue(report['threads_restored']);self.assertEqual(torch.get_num_threads(),threads);self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_output_collision_and_no_experiment_override_flags(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'fixture.json'
            def fake(report,persist):report['status']='test_fixture_only'
            with patch.object(sys,'argv',['runner','--output',str(target)]),patch.object(runner,'run',side_effect=fake) as run:
                runner.main();self.assertEqual(run.call_count,1)
                with self.assertRaises(FileExistsError):runner.main()
                self.assertEqual(run.call_count,1)
            self.assertEqual(json.loads(target.read_text())['status'],'test_fixture_only')
        self.assertEqual(runner.SETTINGS['quadrature_grids'],[[256,512],[512,1024]])
        self.assertEqual(runner.SETTINGS['seeds'],list(range(16)))
        self.assertEqual(digest(ROOT/'scientific_sources/material_a/network/field.py'),core.SOURCE_SHA)


if __name__=='__main__':unittest.main()
