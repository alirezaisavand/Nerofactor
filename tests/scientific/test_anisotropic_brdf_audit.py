"""Independent CPU references versus hash-pinned PURE method bodies, no models."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from audits import anisotropic_brdf as ref
from infrastructure.stage_sources import baseline,PINS,ROOT


def frozen_methods():
    """Compile only inspected algebra, not imports/constructor/renderer/network.

    The independent oracle above uses Cartesian slopes and hemisphere quadrature;
    executing copied pure CPU bodies makes tests about the actual frozen source.
    No source rewriting, CUDA call, model construction or training occurs.
    """
    root,identity=baseline('material',PINS['material'])
    tree=ast.parse((root/'network/field.py').read_text())
    source=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='MCShadingNetwork')
    names={'get_orthogonal_directions','compute_tangent_bitangent_flat','sample_diffuse_directions',
        'compute_Dh','compute_pdf_aniso_ggx','compute_spec_loss','rotate_tangent_bitangent','sample_aniso_ggx_directions',
        'compute_radiance','nan_inf_check','fresnel_schlick','fresnel_schlick_directions','diffuse_term','predict_anisotropic_components'}
    methods=[deepcopy(n) for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods}==names
    for method in methods:
        assert not method.decorator_list
        assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(method))
        assert not any(isinstance(n,ast.Attribute) and n.attr=='cuda' for n in ast.walk(method))
    cls=ast.ClassDef(name='FrozenPureMethods',bases=[],keywords=[],body=methods,decorator_list=[])
    sphere_tree=ast.parse((root/'utils/base_utils.py').read_text())
    sphere=deepcopy(next(n for n in sphere_tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_sphere'))
    module=ast.fix_missing_locations(ast.Module(body=[cls,sphere],type_ignores=[]))
    namespace={'torch':torch,'F':F,'np':np};exec(compile(module,str(root/'network/field.py'),'exec'),namespace)
    obj=namespace['FrozenPureMethods']();obj.cfg={'random_azimuth':True}
    return obj,namespace['sample_sphere'],identity


def tensor(value):return torch.tensor(np.asarray(value),dtype=torch.float32,device='cpu')


def samples(obj,mx=.6,my=.9,wo=(0.,0.,1.),theta=0.,count=16384):
    # Private NumPy RNG; exact same reviewed inputs can be supplied independently
    # to the spherical source sampler and Cartesian-slope oracle.
    rng=np.random.RandomState(891);u=rng.uniform(2e-6,1.,size=(1,count));v=rng.uniform(0.,1.,size=(1,count))
    with patch.object(torch,'rand',side_effect=[tensor(u),tensor(v)]):
        result=obj.sample_aniso_ggx_directions(tensor([[np.cos(2*theta),np.sin(2*theta)]]),tensor([[0,0,0]]),
            tensor([[mx]]),tensor([[my]]),tensor([wo]),count,tensor([[0,0,1]]),None,device='cpu')
    return result,u,v


def radiance(obj,s,wo=(0.,0.,1.),F0=1.,diffuse=0.):
    h,wi,c,p,t,b,n,th,ph=s;M=wi.shape[1]
    fresnel,_,_=obj.fresnel_schlick_directions(tensor([[[F0]*3]]),tensor([[wo]]),wi)
    return obj.compute_radiance(tensor([[diffuse]*3]),torch.ones((1,8,3)),torch.ones((1,M,3)),fresnel,
        torch.zeros((1,8,3)),wi,tensor([[0,0,1]]),c,tensor([wo]),p,th,ph,tensor([[.6]]),tensor([[.9]]))


class BRDFAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();cls.addClassCleanup(torch.set_num_threads,cls.threads);torch.set_num_threads(1)
        cls.shader,sphere,cls.source=frozen_methods();cls.sphere=staticmethod(sphere)
    def test_source_identity_and_roughness_configuration_scope(self):
        self.assertEqual(self.source['commit'],'e1e859779301faced46d6476691e20b546c27ae0')
        tree=ast.parse((ROOT/self.source['root']/'network/field.py').read_text())
        shader=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork')
        defaults=ast.literal_eval(next(n.value for n in shader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_cfg' for t in n.targets)))
        import yaml
        raw=yaml.safe_load((ROOT/self.source['root']/'configs/material/syn/bell.yaml').read_text())
        effective={**defaults,**raw['shader_cfg']}
        self.assertEqual(raw['reg_energy_loss_lambda'],.01);self.assertEqual(effective['reg_energy_loss_lambda'],.05)
        self.assertEqual(effective['reg_spec_loss_lambda'],.05);self.assertEqual(effective['specular_sample_num'],1024)
        method=next(n for n in shader.body if isinstance(n,ast.FunctionDef) and n.name=='predict_anisotropic_components')
        text=ast.get_source_segment((ROOT/self.source['root']/'network/field.py').read_text(),method)
        self.assertIn('0.04, 1.0',text);self.assertNotIn('mx ** 2',text)
    def test_projected_NDF_normalization_and_source_values_match_Beckmann_not_GGX(self):
        h,w=ref.hemisphere(256,512)
        for mx,my in [(.04,.04),(.2,.7),(1.,1.)]:
            D=ref.ndf(h,mx,my);self.assertAlmostEqual(float(np.sum(D*h[:,2]*w)),1.,delta=2e-5)
        h=ref.unit([[.1,.2,1],[.4,-.2,1],[0,0,1]])
        t,b=ref.frame();theta=np.arccos(h[:,2]);phi=np.arctan2(h@b,h@t)
        actual=self.shader.compute_Dh(tensor([[.3]]),tensor([[.7]]),tensor([theta]),tensor([phi])).numpy().reshape(-1)
        np.testing.assert_allclose(actual,ref.ndf(h,.3,.7),rtol=2e-5)
        # Beckmann slope tails are exponential, not GGX's rational tails.
        self.assertLess(float(ref.ndf(ref.unit([1.,0,1.]),.1,.1)),1e-30)
    def test_specular_kernel_reciprocity_and_nonnegative_finite_upper_domain(self):
        rng=np.random.RandomState(45);wi=ref.unit(rng.uniform(-1,1,(200,3)));wi[:,2]=abs(wi[:,2]);wo=ref.unit([.7,.2,.5])
        a=ref.kernel(wi,wo,.2,.7,f0=.04)
        reverse=np.array([ref.kernel(wo,d,.2,.7,f0=.04) for d in wi])
        np.testing.assert_allclose(a,reverse,rtol=2e-12,atol=1e-12)
        self.assertTrue(np.isfinite(a).all());self.assertTrue((a>=0).all())
    def test_thesis_background_2_11_is_not_the_reciprocal_3_8_kernel(self):
        # Visually checked PDF26/printed16: denominator=4*ci*co**alpha,
        # unlike thesis3.8/Kurt10. Record the discrepancy, never amend it.
        wi=ref.unit([.8,0,.6]);wo=np.array([0.,0.,1.]);h=ref.unit(wi+wo)
        numerator=float(ref.fresnel(wo@h,.04)*ref.ndf(h,.3,.7))
        forward=numerator/(4*wi[2]);reverse=numerator/(4*wo[2])
        self.assertGreater(abs(forward-reverse),1e-5)
        self.assertAlmostEqual(float(ref.kernel(wi,wo,.3,.7,f0=.04)),float(ref.kernel(wo,wi,.3,.7,f0=.04)),places=12)
    def test_original_predictor_ranges_and_near_zero_normalization_differ_from_sum_epsilon(self):
        obj=type(self.shader)();obj.feats_network=lambda p:torch.zeros((len(p),256))
        for name,value in [('mx',.25),('my',.75),('metallic',.3),('kd',.8)]:
            setattr(obj,name+'_predictor',lambda x,v=value:torch.full((len(x),3 if v==.8 else 1),v))
        raw=tensor([[.50000006,.5]]);obj.rotation_predictor=lambda x:raw
        mx,my,metal,kd,rotation,_=obj.predict_anisotropic_components(tensor([[0,0,0]]))
        self.assertAlmostEqual(float(mx[0,0]),.28,places=6);self.assertAlmostEqual(float(my[0,0]),.76,places=6)
        mapped=2*raw-1;thesis=mapped/(torch.norm(mapped,dim=-1,keepdim=True)+1e-6)
        self.assertGreater(float(torch.norm(rotation-thesis)),.01)
        self.assertTrue(torch.equal(self.shader.diffuse_term(kd,metal),kd*(1-metal)))
    def test_actual_sampler_matches_independent_cartesian_slopes_and_PDF(self):
        s,u,v=samples(self.shader,mx=.3,my=.7,wo=(.6,0,.8),theta=.3)
        # Float32 transcendental/angle reconstruction roundoff is bounded by
        # 16 machine epsilons; this is not a native acceptance-tolerance change.
        h=s[0].numpy()[0];np.testing.assert_allclose(h,ref.half_samples(u[0].astype(np.float32),v[0].astype(np.float32),.3,.7,.3),atol=16*np.finfo(np.float32).eps)
        wi=s[1].numpy()[0];valid=(wi[:,2]>0)&(h@np.array([.6,0,.8])>1e-3)
        np.testing.assert_allclose(s[3].numpy()[0,valid,0],ref.directional_pdf(wi[valid],np.array([.6,0,.8]),.3,.7,.3),rtol=1e-4,atol=5e-6)
        self.assertAlmostEqual(float((h[:,0]**2+h[:,1]**2+h[:,2]**2).mean()),1.,places=6)
    def test_reflected_PDF_is_subdensity_and_matches_independent_acceptance_mass(self):
        wi,w=ref.hemisphere()
        for wo in [np.array([0,0,1.]),ref.unit([1,0,.25])]:
            mass=float(np.sum(ref.directional_pdf(wi,wo,.6,.9)*w));expected=ref.acceptance(wo,.6,.9)
            self.assertAlmostEqual(mass,expected,delta=2e-5);self.assertLess(mass,1.)
        self.assertAlmostEqual(ref.acceptance([0,0,1.],.7,.7),1-np.exp(-1/.7**2),places=12)
    def test_MC_unconditional_estimator_matches_quadrature_but_source_is_conditional(self):
        s,u,v=samples(self.shader,count=131072);wi=s[1].numpy()[0];h=s[0].numpy()[0];valid=wi[:,2]>0
        weights=np.maximum(wi[:,2],0)/s[2].numpy()[0,:,0]
        integral=float(np.mean(weights));actual=float(radiance(self.shader,s)[2][0,0])
        directions,w=ref.hemisphere();exact=float(np.sum(ref.kernel(directions,np.array([0,0,1.]),.6,.9)*directions[:,2]*w))
        self.assertAlmostEqual(integral,exact,delta=.004)
        self.assertAlmostEqual(actual,integral/valid.mean(),delta=2e-6)
        self.assertGreater(actual-integral,.08)  # Defect witness, not an approved correction.
    def test_accepted_count_limit_breaks_reciprocity(self):
        a=np.array([0,0,1.]);b=ref.unit([1,0,.25]);f=float(ref.kernel(b,a,.6,.9))
        self.assertAlmostEqual(f,float(ref.kernel(a,b,.6,.9)),places=12)
        self.assertGreater(abs(f/ref.acceptance(a,.6,.9)-f/ref.acceptance(b,.6,.9)),.001)
    def test_directional_energy_counterexample_for_allowed_metallic_diffuse_ranges(self):
        result=ref.observations();rows=result['directional_energy']
        self.assertTrue(all(np.isfinite(r['actual_constant_light_total_limit']) and r['actual_constant_light_total_limit']>=0 for r in rows))
        self.assertGreater(rows[0]['actual_constant_light_total_limit'],1.01)
        self.assertTrue(all(r['unconditioned_specular_reflectance']<=1 for r in rows))
        s,_,_=samples(self.shader,count=65536)
        source=radiance(self.shader,s,F0=.04*(1-.001)+.001*.999,diffuse=.999*(1-.001))
        self.assertGreater(float(source[0][0,0]),1.015)
    def test_diffuse_generating_density_normalizes_but_is_not_cosine_sampling(self):
        wi,w=ref.hemisphere();q=ref.diffuse_density(wi[:,2]);self.assertAlmostEqual(float(q@w),1.,places=12)
        self.assertAlmostEqual(float((q*wi[:,2]**2)@w),1-2/np.pi,places=12)
        az,el=self.sphere(512,0)
        self.shader.diffuse_direction_samples=tensor(np.stack([az*.5/np.pi,1-2*el/np.pi],-1))
        directions=self.shader.sample_diffuse_directions(tensor([[0,0,1]]),False).numpy()[0]
        expected=2*np.arcsin(np.arange(512)/512)/np.pi+1e-7
        np.testing.assert_allclose(directions[:,2]**2,expected,rtol=1e-4,atol=2e-7)
        self.assertLess(abs(np.mean(directions[:,2]**2)-(1-2/np.pi)),.003)
        self.assertGreater(abs(np.mean(directions[:,2]**2)-.5),.1)
    def test_diffuse_continuum_kernel_is_nonreciprocal(self):
        a=.3;b=.9
        self.assertGreater(abs(ref.diffuse_density(a)/a-ref.diffuse_density(b)/b),.1)
    def test_theta_periodicity_swapped_axes_and_isotropic_limit(self):
        wi,_=ref.hemisphere(16,32);theta=.31
        np.testing.assert_allclose(ref.ndf(wi,.2,.7,theta),ref.ndf(wi,.2,.7,theta+np.pi),atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(ref.ndf(wi,.2,.7,theta),ref.ndf(wi,.7,.2,theta+np.pi/2),atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(ref.ndf(wi,.4,.4,theta),ref.ndf(wi,.4,.4,1.4),atol=1e-13,rtol=1e-12)
        a,_,_=samples(self.shader,theta=theta,count=128);b,_,_=samples(self.shader,theta=theta+np.pi,count=128)
        np.testing.assert_allclose(a[0].numpy(),b[0].numpy(),atol=2e-6)
        h=ref.unit([[.2,.3,1.],[.5,-.1,1.]])
        def source_D(mx,my,angle):
            result,_,_=samples(self.shader,mx=mx,my=my,theta=angle,count=1)
            t,b=result[4].numpy()[0],result[5].numpy()[0]
            return self.shader.compute_Dh(tensor([[mx]]),tensor([[my]]),tensor([np.arccos(h[:,2])]),tensor([np.arctan2(h@b,h@t)])).numpy().reshape(-1)
        np.testing.assert_allclose(source_D(.2,.7,theta),ref.ndf(h,.2,.7,theta),rtol=2e-5)
        np.testing.assert_allclose(source_D(.2,.7,theta),source_D(.7,.2,theta+np.pi/2),rtol=2e-5)
        np.testing.assert_allclose(source_D(.4,.4,theta),source_D(.4,.4,1.4),rtol=2e-5)
    def test_half_angle_exact_branch_and_zero_encoding_destroy_frame_orthonormality(self):
        # Pure source sampler with a selected encoding; no material network.
        for rotation,expected_length in [([-1.,0.],.001),([0.,0.],np.sqrt(.5))]:
            with patch.object(torch,'rand',side_effect=[tensor([[.5]]),tensor([[.2]])]):
                out=self.shader.sample_aniso_ggx_directions(tensor([rotation]),tensor([[0,0,0]]),tensor([[.2]]),tensor([[.7]]),
                    tensor([[0,0,1]]),1,tensor([[0,0,1]]),None,device='cpu')
            self.assertAlmostEqual(float(torch.norm(out[4])),expected_length,places=6)
            self.assertGreater(abs(float(torch.norm(out[0]))-1.),.01)
            self.assertTrue(all(torch.isfinite(x).all().item() for x in out))
    def test_tangent_switch_is_discontinuous_but_regular_region_is_continuous(self):
        def normal(z):return tensor([[np.sqrt(1-z*z),0.,z]])
        t0,_=self.shader.compute_tangent_bitangent_flat(normal(.999e-6),None)
        t1,_=self.shader.compute_tangent_bitangent_flat(normal(1.001e-6),None)
        self.assertLess(abs(float((t0*t1).sum())),.001)
        a,_=self.shader.compute_tangent_bitangent_flat(normal(.1),None);b,_=self.shader.compute_tangent_bitangent_flat(normal(.100001),None)
        self.assertGreater(float((a*b).sum()),.999999)
    def test_parallel_supplied_source_has_zero_tangents_and_does_not_repair(self):
        t,b=self.shader.compute_tangent_bitangent_flat(tensor([[0,0,1]]),tensor([[0,0,1]]))
        self.assertEqual(float(torch.norm(t)),0.);self.assertEqual(float(torch.norm(b)),0.)
    def test_antipodal_and_zero_normal_cases_are_finite_but_not_valid_frames(self):
        Fvalue,h,_=self.shader.fresnel_schlick_directions(tensor([[[.04]*3]]),tensor([[[0,0,1]]]),tensor([[[0,0,-1]]]))
        self.assertEqual(float(torch.norm(h)),0.);np.testing.assert_array_equal(Fvalue.numpy(),1.)
        t,b=self.shader.compute_tangent_bitangent_flat(tensor([[0,0,0]]),None)
        self.assertEqual(float(torch.norm(t)),0.);self.assertEqual(float(torch.norm(b)),0.)
        self.assertEqual(float(ref.kernel([0,0,-1.],np.array([0,0,1.]),.3,.7)),0.)
    def test_grazing_view_weights_and_empty_accepted_set_remain_finite(self):
        s,_,_=samples(self.shader,wo=(1.,0.,0.),count=1024)
        result=radiance(self.shader,s,wo=(1.,0.,0.))
        self.assertTrue(all(torch.isfinite(x).all().item() for x in result))
        M=16;below=(torch.zeros((1,M,3)),tensor(np.tile([0,0,-1.],(1,M,1))),torch.ones((1,M,1)),torch.ones((1,M,1)),
            torch.zeros((1,3)),torch.zeros((1,3)),torch.zeros((1,3)),torch.zeros((1,M)),torch.zeros((1,M)))
        value=radiance(self.shader,below)
        self.assertTrue(torch.isfinite(value[0]).all().item());np.testing.assert_array_equal(value[2].numpy(),0.)
    def test_grazing_regularized_D_and_pdf_are_finite_but_not_exact_density(self):
        angles=tensor([[0.,np.pi/2-1e-6,np.pi/2]])
        D=self.shader.compute_Dh(tensor([[.04]]),tensor([[1.]]),angles,tensor([[0.,0.,0.]]))
        self.assertTrue(torch.isfinite(D).all().item());self.assertTrue((D>=0).all().item())
        self.assertGreater(float(D[0,-1,0]),0.)  # -40 floor yields a nonzero tail.
        normal=tensor([[0,0,1]]);h=tensor([[[1,0,0]]])
        p=self.shader.compute_pdf_aniso_ggx(tensor([[.04]]),tensor([[1.]]),normal,h,tensor([[np.pi/2]]),tensor([[0.]]))
        self.assertTrue(torch.isfinite(p).all().item());self.assertGreater(float(p[0,0,0]),0.)
    def test_specular_weight_ignores_pdf_argument_and_regularizer_has_inverse_sample_scale(self):
        s,_,_=samples(self.shader,count=512);a=radiance(self.shader,s,diffuse=.5)
        changed=list(s);changed[3]=s[3]*7;b=radiance(self.shader,changed,diffuse=.5)
        self.assertTrue(torch.equal(a[0],b[0]))
        # For all accepted directions, softmax(D).sum=1; subsequent /Nvalid
        # scales the "spec" penalty as diffuse/(3*Nvalid), not BRDF energy.
        M=16;fake=(torch.zeros((1,M,3)),tensor([[[0,0,1]]]*M).reshape(1,M,3),torch.ones((1,M,1)),torch.ones((1,M,1)),
            torch.zeros((1,3)),torch.zeros((1,3)),torch.zeros((1,3)),torch.zeros((1,M)),torch.zeros((1,M)))
        val=radiance(self.shader,fake,diffuse=.6)[4]
        np.testing.assert_allclose(val.numpy(),.6/(3*M),rtol=1e-6)


if __name__=='__main__':unittest.main()
