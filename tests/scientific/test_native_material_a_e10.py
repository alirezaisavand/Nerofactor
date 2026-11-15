"""New bounded CPU fixtures: production derivative methods, synthetic lighting/BVH.

No original N1/native constructor, GPU, historical experiment, optimizer or reference
production. Hash-admitted pure A methods supply the native interface stand-in.
"""
import ast
from copy import deepcopy
import importlib.abc
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from audits import anisotropic_brdf as oracle
from experiments.anisotropic_estimator_v1 import source_classes,RandomTape
from infrastructure.material_variants import CORE,ESTIMATOR,ORIGINAL,ENERGY,configuration,install
from infrastructure.stage_sources import ROOT,PINS,baseline
from scientific_variants.material_a_v1 import native_classes,orientation,uniform32


@pytest.fixture(autouse=True)
def cpu_fixture_only():
    forbidden=('network','dataset','train','nvdiffrast','open3d','experiments.neural_material_field_v1')
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,*args):
            if name.split('.')[0] in forbidden or name.startswith(forbidden[-1]):raise AssertionError('Native scientific import forbidden in synthetic test: '+name)
    gate=Block();sys.meta_path.insert(0,gate);state=np.random.get_state();threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:
        torch.set_num_threads(threads);sys.meta_path.remove(gate)
        after=np.random.get_state();assert after[0]==state[0] and np.array_equal(after[1],state[1]) and after[2:]==state[2:]


def source_method(path,cls,name,ns):
    tree=ast.parse(path.read_text());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    f=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),ns);return ns[name]


def settings(mode='disabled_v1',N=32,M=32):
    return dict(anisotropy=True,diffuse_sample_num=N,specular_sample_num=M,human_lights=False,
        random_azimuth=True,regularization_version=mode,reg_change=False,change_eps=.02,change_type='gaussian',
        reg_lambda1=.05,reg_energy_loss=False,reg_energy_loss_lambda=.017,reg_spec_loss=False,reg_spec_loss_lambda=.023)


def fixture(variant=CORE,cfg=None):
    classes,sphere,_=source_classes();a=classes['B0'];science,_=baseline('material',PINS['material'])
    field=science/'network/field.py';renderer=science/'network/renderer.py'
    class Base(torch.nn.Module):
        def __init__(self,cfg,trace):
            super().__init__();self.cfg=deepcopy(cfg);self.ray_trace_fun=trace;self.trace_calls=0
            self.material=torch.nn.Parameter(torch.tensor([.2,.35,.3,.4,.5,.2,.8,.6]))  # Tiny fixture, NOT N1.
            self.outer_scale=torch.nn.Parameter(torch.tensor(.7))
            az,el=sphere(cfg['diffuse_sample_num'],0)
            self.diffuse_direction_samples=torch.tensor(np.stack([az*.5/np.pi,1-2*el/np.pi],-1),dtype=torch.float32)
        def predict_anisotropic_components(self,p):
            q=self.material;raw=q[6:8].reshape(1,2).expand(len(p),2);rotation=F.normalize(2*raw-1,dim=-1,eps=1e-6)
            return q[0].expand(len(p),1),q[1].expand(len(p),1),q[2].expand(len(p),1),q[3:6].expand(len(p),3),rotation,raw
        def predict_outer_lights(self,p,d):return self.outer_scale*(.3+.7*d[:,2:3].square()).expand_as(d)
        def get_inner_lights(self,p,d,n):return torch.ones_like(p)*.2
    for name,value in a.__dict__.items():
        if not name.startswith('__'):setattr(Base,name,value)
    for name in ('forward','anisotropic_forward','get_lights','shade_anisotropic_mixed','anisotropic_regularization'):
        setattr(Base,name,source_method(field,'MCShadingNetwork',name,dict(torch=torch,np=np,F=F)))
    rawns=dict(torch=torch,np=np);tree=ast.parse((science/'utils/raw_utils.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='linear_to_srgb')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_A_raw_utils','exec'),rawns)
    class Renderer(torch.nn.Module):
        def __init__(self,cfg):
            super().__init__();self.cfg=cfg;self.mesh=SimpleNamespace(vertices=np.array([[0,0,0],[.1,0,0],[0,.1,0]],np.float32));self._init_shader()
        def trace(self,o,d):
            self.shader_network.trace_calls+=1
            return o+2*d,torch.zeros_like(o),torch.full((len(o),1),2.),torch.zeros(len(o),dtype=torch.bool)
    Renderer.shade=source_method(renderer,'NeROMaterialRenderer','shade',{})
    Renderer.compute_rgb_loss=source_method(renderer,'NeROMaterialRenderer','compute_rgb_loss',dict(torch=torch,F=F))
    shader,model=native_classes(Base,Renderer,variant,rawns['linear_to_srgb']);cfg=cfg or settings('legacy_v1' if variant==ESTIMATOR else 'disabled_v1')
    native=model(dict(network='material',shader_cfg=cfg,reg_mat=cfg['regularization_version']!='disabled_v1',database_name='syn/bell',rgb_loss='charbonier'))
    return native,Base,Renderer,shader


def tapes(M=32):
    rng=np.random.RandomState(91)
    return [rng.uniform(.2,.99,(1,M)).astype(np.float32),rng.uniform(0,1,(1,M)).astype(np.float32),np.full((1,1,1),.13,np.float32)]


def shade(model,draws=None):
    draws=draws or tapes(model.shader_network.cfg['specular_sample_num'])
    with patch.object(torch,'rand',RandomTape(draws)):
        return model.shade(torch.zeros(1,3),torch.tensor([[.1,0,np.sqrt(.99)]],dtype=torch.float32),torch.tensor([[0.,0.,1.]]),None,True,0)


def test_native_shader_path_trace_lighting_rgb_and_backward():
    model,_,_,_=fixture();out=shade(model)
    assert model.shader_network.trace_calls==1 and out['linear_rgb'].requires_grad
    loss=model.compute_rgb_loss(out['rgb_pr'],torch.full((1,3),.2)).mean();loss.backward()
    for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
    assert out['normals'].tolist()==[[.5,.5,1.]]  # No normal flip.
    assert model.shader_network.material.grad.shape==(8,)


def test_shader_full_path_autograd_matches_local_finite_differences(record_property):
    model,_,_,_=fixture();p=model.shader_network.material;out=shade(model)['linear_rgb'].sum();out.backward();analytic=p.grad.detach().numpy().copy()
    numerical=[]
    for i in range(8):
        old=float(p[i]);values=[]
        for sign in (-1,1):
            with torch.no_grad():p[i]=old+sign*.0005
            values.append(float(shade(model)['linear_rgb'].sum()))
        with torch.no_grad():p[i]=old
        numerical.append((values[1]-values[0])/.001)
    np.testing.assert_allclose(analytic,numerical,rtol=.04,atol=3e-4)
    record_property('maximum_absolute_local_gradient_error',float(np.max(np.abs(analytic-numerical))))


@pytest.mark.parametrize('raw',[[.5,.5],[.5000001,.5]])
def test_invalid_raw_orientation_fails_before_light_queries_and_without_update(raw):
    model,_,_,_=fixture()
    with torch.no_grad():model.shader_network.material[6:8]=torch.tensor(raw)
    with pytest.raises(ValueError,match='orientation'):shade(model)
    assert model.shader_network.trace_calls==0


@pytest.mark.parametrize('encoding',[[-1.,0.],[-1.,1e-7],[-1.,-1e-7],[1.,0.],[0.,1.]])
def test_frames_are_orthonormal_at_negative_axis_and_regular_points(encoding):
    model,_,_,_=fixture();s=model.shader_network;r=torch.tensor([encoding],requires_grad=True);n=torch.tensor([[0.,0.,1.]])
    with patch.object(torch,'rand',RandomTape(tapes()[:2])):
        out=s.sample_aniso_ggx_directions(r,torch.zeros(1,3),torch.tensor([[.2]]),torch.tensor([[.6]]),n,32,n,None,'cpu')
    frame=torch.stack(out[4:7],-1)
    torch.testing.assert_close(frame.transpose(-1,-2)@frame,torch.eye(3)[None],atol=3e-7,rtol=0)
    torch.testing.assert_close(torch.linalg.norm(out[0],dim=-1),torch.ones(1,32),atol=2e-7,rtol=0)
    out[1][...,0].sum().backward();assert torch.isfinite(r.grad).all()


def test_periodicity_axis_swap_and_original_reference_seam_are_not_hidden():
    theta=.31;mx,my=.2,.6;c,s=orientation(torch.tensor([[np.cos(2*theta),np.sin(2*theta)]],dtype=torch.float64));c1,s1=orientation(torch.tensor([[-np.cos(2*theta),-np.sin(2*theta)]],dtype=torch.float64))
    R=np.array([[c.item(),-s.item()],[s.item(),c.item()]]);R1=np.array([[c1.item(),-s1.item()],[s1.item(),c1.item()]])
    np.testing.assert_allclose(R@np.diag([mx*mx,my*my])@R.T,R1@np.diag([my*my,mx*mx])@R1.T,atol=1e-15)
    np.testing.assert_allclose(oracle.kernel([.3,0,np.sqrt(.91)],[0,0,1],mx,my,theta),oracle.kernel([.3,0,np.sqrt(.91)],[0,0,1],my,mx,theta+np.pi/2),atol=1e-15)
    model,_,_,_=fixture();n=F.normalize(torch.tensor([[1.,0.,1e-7],[1.,0.,2e-6]]),dim=-1);t,b=model.shader_network.compute_tangent_bitangent_flat(n,None)
    torch.testing.assert_close(torch.linalg.norm(t,dim=-1),torch.ones(2),atol=1e-7,rtol=0)
    assert abs(float(t[0]@t[1]))<1e-5  # Documented chart seam, NOT global continuity.
    with pytest.raises(ValueError):model.shader_network.compute_tangent_bitangent_flat(n,torch.zeros_like(n))


def test_diffuse_midpoints_constant_and_directional_moments():
    model,_,_,_=fixture();s=model.shader_network;d=s.sample_diffuse_directions(torch.tensor([[0.,0.,1.]]),False)[0]
    np.testing.assert_allclose(d[:,2].numpy()**2,(np.arange(32)+.5)/32,atol=1e-7)
    assert abs(float(d[:,2].square().mean())-.5)<1e-7
    assert abs(float(d[:,2].pow(4).mean())-(1/3-1/(12*32**2)))<1e-7
    assert torch.equal(torch.ones_like(d).mean(0),torch.ones(3))


@pytest.mark.parametrize('widths,angle',[((.3,.7),0),((.6,.9),80)])
def test_total_count_specular_expectation_against_independent_directional_integration(widths,angle,record_property):
    M=32768;model,_,_,_=fixture(cfg=settings(M=M));s=model.shader_network;mx,my=widths
    wo=np.array([np.sin(np.deg2rad(angle)),0,np.cos(np.deg2rad(angle))]);n=torch.tensor([[0.,0.,1.]]);rng=np.random.RandomState(19)
    draws=[rng.uniform(size=(1,M)).astype(np.float32) for _ in range(2)]
    with patch.object(torch,'rand',RandomTape(draws)):
        h,wi,ch,pdf,t,b,z,theta,phi=s.sample_aniso_ggx_directions(torch.tensor([[np.cos(.62),np.sin(.62)]],dtype=torch.float32),torch.zeros(1,3),torch.tensor([[mx]]),torch.tensor([[my]]),torch.tensor(np.asarray([wo]),dtype=torch.float32),M,n,None)
    f=torch.full((1,M,3),.04);d=torch.zeros(1,3)
    result=s.compute_radiance(d,torch.ones(1,32,3),torch.ones(1,M,3),f,torch.zeros(1,32,3),wi,n,ch,torch.tensor(np.asarray([wo])),pdf,theta,phi,torch.tensor([[mx]]),torch.tensor([[my]]))
    references=[]
    for count in (64,128):
        directions,weights=oracle.hemisphere(count,2*count)
        # F constant .04 makes an independent kernel with no source-sampler cancellation.
        H=oracle.unit(directions+wo);reference=oracle.ndf(H,mx,my,.31)/(4*(H@wo))*.04
        references.append(float(np.sum(reference*directions[:,2]*weights)))
    assert abs(references[1]-references[0])<2e-5
    assert abs(float(result[2][0,0])-references[1])<.004
    record_property('specular_estimate',float(result[2][0,0]));record_property('independent_integral',references[1]);record_property('two_grid_difference',abs(references[1]-references[0]))
    valid=(wi[0,:,2]>0).numpy();assert np.isfinite(pdf.numpy()).all()
    H=h[0].numpy();expected=oracle.ndf(H,mx,my,.31)*H[:,2]/(4*np.abs(H@wo))
    np.testing.assert_allclose(pdf[0,:,0].detach().numpy()[valid],expected[valid],rtol=2e-4,atol=5e-6)


def test_uniform_endpoint_and_zero_draw_grazing_are_finite_not_clipped_to_eps_ring():
    a=np.array([0.,.5,.9999999982641566],np.float64);b=uniform32(a)
    assert b[-1]==np.nextafter(np.float32(1),np.float32(0)) and np.array_equal(b[:2],a[:2].astype(np.float32))
    with pytest.raises(ValueError):uniform32(np.array([1.],np.float64))
    model,_,_,_=fixture();s=model.shader_network
    draws=[np.array([[0,np.nextafter(np.float32(1),np.float32(0)),.2,.8]],np.float32),np.full((1,4),.25,np.float32)]
    n=torch.tensor([[0.,0.,1.]]);wo=F.normalize(torch.tensor([[1.,0.,1e-5]]),dim=-1)
    with patch.object(torch,'rand',RandomTape(draws)):
        out=s.sample_aniso_ggx_directions(torch.tensor([[-1.,0.]]),torch.zeros(1,3),torch.tensor([[.04]]),torch.tensor([[1.]]),wo,4,n,None)
    assert all(torch.isfinite(x).all() for x in out)
    torch.testing.assert_close(torch.linalg.norm(out[1],dim=-1),torch.ones(1,4),atol=3e-7,rtol=0)


def test_unconditional_priors_repeat_invariant_and_direct_gradient_scope():
    cfg=settings('unconditional_v1');cfg.update(reg_energy_loss=True,reg_spec_loss=True)
    model,_,_,_=fixture(cfg=cfg);s=model.shader_network;results=[]
    for copies in (1,3):
        M=4*copies;fd=torch.tensor([[.8,.7,.6]],requires_grad=True);Fresnel=torch.full((1,M,3),.4,requires_grad=True)
        wi=torch.tensor([[[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.]]]).repeat(1,copies,1);ch=torch.full((1,M,1),.5)
        radiance=s.compute_radiance(fd,torch.ones(1,8,3),torch.ones(1,M,3),Fresnel,torch.zeros(1,8,3),wi,torch.tensor([[0.,0.,1.]]),ch,torch.tensor([[0.,0.,1.]]),torch.ones(1,M,1),torch.zeros(1,M),torch.zeros(1,M),torch.ones(1,1),torch.ones(1,1))
        assert torch.equal(radiance[3],torch.full((1,3),.4))
        result=s.anisotropic_regularization(torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),None,None,None,None,None,None,fd,radiance[3],radiance[4],None,None,0)
        expected=.017*torch.relu(fd+.4-1).sum(-1)+.023*(fd*.4/3).sum(-1)
        torch.testing.assert_close(result,expected,atol=1e-8,rtol=0);results.append(result.detach())
        grad=torch.autograd.grad(radiance[4].sum(),(fd,Fresnel),allow_unused=True)
        assert grad[0] is not None and grad[1] is None  # New allocation prior detached specular scope is explicit.
        grad_energy=torch.autograd.grad(torch.relu(fd+radiance[3]-1).sum(),Fresnel)
        assert grad_energy[0].abs().sum()>0  # Energy uses unconditional proxy, independent reg_change gate.
    torch.testing.assert_close(results[0],results[1],atol=1e-8,rtol=0)


def test_configuration_single_owner_and_explicit_prior_versions():
    cfg=dict(network='material',reg_mat=False,shader_cfg=settings())
    assert configuration(cfg,CORE)['shader_cfg']['reg_spec_loss_lambda']==.023
    for changes in ({'material_variant':ORIGINAL},{'reg_energy_loss_lambda':.01},{'shader_cfg':{**settings(),'typo':True}},{'shader_cfg':{**settings(),'geometry_type':'ggx_smith'}},{'shader_cfg':{**settings(),'regularization_version':'invented'}}):
        with pytest.raises(ValueError):configuration({**cfg,**changes},CORE)
    original={**cfg,'reg_energy_loss_lambda':.01};assert configuration(original,ORIGINAL)==original
    with pytest.raises(ValueError):configuration(cfg,ESTIMATOR)


def test_variant_checkpoint_tags_strict_loading_and_registry_geometry_unchanged():
    model,Base,Renderer,Shader=fixture();state=model.state_dict();model.load_state_dict(state)
    with pytest.raises(ValueError):model.load_state_dict(state,strict=False)
    del state['shader_network._a_variant_signature']
    with pytest.raises(ValueError,match='signature'):model.load_state_dict(state)
    changed=settings();changed['reg_energy_loss_lambda']=.1;other,_,_,_=fixture(cfg=changed)
    with pytest.raises(ValueError,match='signature'):model.load_state_dict(other.state_dict())
    registry=SimpleNamespace(name2renderer={'material':Renderer,'shape':object()},NeROMaterialRenderer=Renderer)
    field=SimpleNamespace(MCShadingNetwork=Base,linear_to_srgb=lambda x:x)
    cfg=dict(network='material',reg_mat=False,shader_cfg=settings())
    original_shape=registry.name2renderer['shape'];selected=install(registry,field,CORE,cfg)
    assert registry.name2renderer['material'] is selected and registry.name2renderer['shape'] is original_shape and field.MCShadingNetwork is Base
    with pytest.raises(ValueError):install(registry,field,CORE,cfg)


def test_anisotropic_export_Q_and_relighting_use_native_visibility_and_cleanup():
    model,_,_,_=fixture();result=model.predict_materials(batch_size=2)
    assert set(result)=={'mx','my','metallic','albedo','rotation','Q'} and result['Q'].shape==(3,2,2)
    np.testing.assert_allclose(np.linalg.eigvalsh(result['Q'])[0],[.2**2,.35**2],atol=2e-8)
    with patch.object(torch,'rand',RandomTape(tapes())):
        out=model.relight(torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,0.,1.]]),None,lambda p,d:torch.ones_like(d)*.8,'synthetic-constant-v1',True)
    assert model.shader_network.trace_calls==1 and out['rgb_pr'].requires_grad and model.shader_network._environment is None
    with pytest.raises(RuntimeError):
        with model.shader_network.environment(lambda p,d:torch.ones_like(d),'failure-fixture'):raise RuntimeError('synthetic failure')
    assert model.shader_network._environment is None


def test_actual_shader_axis_swap_with_explicit_world_sample_coupling():
    model,_,_,_=fixture();p=model.shader_network.material;theta=-.31
    with torch.no_grad():p[6:8]=torch.tensor([(np.cos(2*theta)+1)/2,(np.sin(2*theta)+1)/2])
    draws=tapes();a=shade(model,draws)['linear_rgb'].detach()
    with torch.no_grad():
        widths=p[:2].clone();p[0]=widths[1];p[1]=widths[0];p[6:8]=1-p[6:8]
    coupled=[draws[0],np.remainder(draws[1]-.25,1).astype(np.float32),draws[2]]
    b=shade(model,coupled)['linear_rgb'].detach()
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)  # Physical equivalence; uncoupled finite tapes are NOT required to match.


def test_energy_prior_autograd_reaches_roughness_through_actual_native_shader():
    cfg=settings('unconditional_v1');cfg['reg_energy_loss']=True
    model,_,_,_=fixture(cfg=cfg);s=model.shader_network
    with torch.no_grad():s.material[2]=.001;s.material[3:6]=.999
    out=shade(model);energy=torch.relu(out['f_d']+out['f_s_sum']-1).sum()
    assert energy>0  # Soft prior does not claim this material is physically energy conserving.
    energy.backward();assert torch.isfinite(s.material.grad).all() and bool((s.material.grad[:2].abs()>0).all())


def test_estimator_only_preserves_legacy_decoder_and_prior_diagnostics():
    model,_,_,_=fixture(ESTIMATOR);s=model.shader_network;n=torch.tensor([[0.,0.,1.]])
    with patch.object(torch,'rand',RandomTape(tapes()[:2])):
        out=s.sample_aniso_ggx_directions(torch.tensor([[-1.,0.]]),torch.zeros(1,3),torch.tensor([[.2]]),torch.tensor([[.6]]),n,32,n,None,'cpu')
    assert float(torch.linalg.norm(out[4]))<.002  # The isolated estimator arm does NOT silently repair decoding.
    fd=torch.full((1,3),.6);wi=torch.tensor([[[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.]]]);M=4
    r=s.compute_radiance(fd,torch.ones(1,8,3),torch.ones(1,M,3),torch.full((1,M,3),.4),torch.zeros(1,8,3),wi,n,torch.full((1,M,1),.5),n,torch.ones(1,M,1),torch.zeros(1,M),torch.zeros(1,M),torch.ones(1,1),torch.ones(1,1))
    torch.testing.assert_close(r[2],torch.full((1,3),.4),rtol=0,atol=0)
    torch.testing.assert_close(r[3],torch.full((1,3),.8),rtol=0,atol=0)  # Conditional legacy proxy deliberately distinct from corrected radiance.


@pytest.mark.parametrize('width,color,metal',[(.04,0.,0.),(1.,1.,1.)])
def test_actual_shader_material_boundaries_have_finite_gradients(width,color,metal):
    model,_,_,_=fixture();p=model.shader_network.material
    with torch.no_grad():p[:2]=width;p[2]=metal;p[3:6]=color
    out=shade(model);out['linear_rgb'].sum().backward()
    assert torch.isfinite(out['rgb_pr']).all() and torch.isfinite(p.grad).all()
    with torch.no_grad():p[0]=.039
    with pytest.raises(ValueError,match='width'):shade(model)


def test_orientation_overflow_and_renderer_failure_do_not_leave_runtime_mutations():
    with pytest.raises(ValueError,match='orientation norm'):orientation(torch.tensor([[3e38,3e38]]))
    model,Base,Renderer,_=fixture()
    def failing(self,data):
        torch.set_default_tensor_type(torch.DoubleTensor)
        raise RuntimeError('synthetic renderer failure')
    Renderer.forward=failing
    before=torch.tensor([]).type()
    with pytest.raises(RuntimeError):model({})
    assert torch.tensor([]).type()==before


def known_settings():
    cfg=settings();cfg.update(illumination_mode='known_env_v1',indirect_mode='zero_bounce_v1',known_environment=dict(ambient=[.1,.2,.3],lobes=[dict(axis=[0,0,1],RGB=[.5,.5,.5],power=2)]))
    return cfg


def test_known_illumination_preserves_native_BVH_but_has_no_learned_light_gradients():
    model,_,_,_=fixture(cfg=known_settings());s=model.shader_network;out=shade(model);out['linear_rgb'].sum().backward()
    assert s.trace_calls==1 and s.material.grad is not None and s.outer_scale.grad is None
    directions=torch.tensor([[0.,0.,1.],[1.,0.,0.]])
    torch.testing.assert_close(s.predict_outer_lights(torch.zeros(2,3),directions),torch.tensor([[.6,.7,.8],[.1,.2,.3]]))
    assert torch.equal(s.get_inner_lights(directions,directions,directions),torch.zeros_like(directions))


def test_original_known_light_comparator_retains_exact_original_math_method_objects():
    model,Base,_,Shader=fixture(ORIGINAL,cfg=known_settings())
    for name in ('predict_anisotropic_components','compute_tangent_bitangent_flat','sample_diffuse_directions',
                 'sample_aniso_ggx_directions','compute_Dh','compute_pdf_aniso_ggx','compute_radiance','shade_anisotropic_mixed','anisotropic_regularization'):
        assert getattr(Shader,name) is getattr(Base,name),name
    # Original counts/decoder remain original; lighting itself is the identical explicit benchmark factor.
    assert model.shader_network.predict_outer_lights(torch.zeros(1,3),torch.tensor([[0.,0.,1.]])).shape==(1,3)


def test_nonfinite_backward_is_rejected_before_any_optimizer_or_weight_change():
    model,_,_,_=fixture();before={n:p.detach().clone() for n,p in model.named_parameters()}
    with pytest.raises(ValueError,match='parameter gradient'):(shade(model)['linear_rgb'].sum()*float('nan')).backward()
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_near_parallel_explicit_tangent_source_rejected_without_nonunit_frame():
    model,_,_,_=fixture();n=torch.tensor([[0.,0.,1.]])
    with pytest.raises(ValueError,match='tangent reference'):model.shader_network.compute_tangent_bitangent_flat(n,torch.tensor([[1e-9,0.,1.]]))


def test_energy_budget_only_changes_specular_allocation_with_finite_native_gradients():
    core,_,_,_=fixture(CORE);energy,_,_,_=fixture(ENERGY)
    a=shade(core);b=shade(energy);d=a['f_d']
    torch.testing.assert_close(a['linear_diffuse'],b['linear_diffuse'],rtol=0,atol=0)
    torch.testing.assert_close(b['linear_specular'],a['linear_specular']*(1-d),rtol=2e-6,atol=1e-7)
    b['linear_rgb'].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in energy.parameters())
    exported=energy.predict_materials();np.testing.assert_allclose(exported['diffuse_budget']+exported['specular_budget'],1,atol=1e-7)
    with pytest.raises(ValueError):energy.load_state_dict(core.state_dict())


def test_energy_bound_pairing_inequality_and_independent_directional_integration(record_property):
    # Independent algebraic witness for the proof, not a production shader/reference run.
    for t in np.linspace(.001,1,101):
        for co in np.linspace(0,1,101):
            A=co*(2*t-1/t);B=2*np.sqrt(1-co*co)*np.sqrt(1-t*t)
            assert .5*(max(A+B,0)+max(A-B,0))<=1+1e-14
    energies=[]
    for mx,my in ((.6,.9),(1.,1.)):
        for degrees in (0,60,80):
            wo=np.array([np.sin(np.deg2rad(degrees)),0,np.cos(np.deg2rad(degrees))]);rho=[]
            for count in (64,128):
                wi,w=oracle.hemisphere(count,2*count)
                for metal,kd in ((.001,.999),(.5,.7),(1.,1.)):
                    d=kd*(1-metal);f0=.04*(1-metal)+metal*kd
                    spec=float(np.sum(oracle.kernel(wi,wo,mx,my,.31,f0)*wi[:,2]*w))
                    rho.append(d+(1-d)*spec)
            np.testing.assert_allclose(rho[:3],rho[3:],atol=2e-5,rtol=0)
            assert min(rho)>=0 and max(rho)<=1+2e-5;energies+=rho[3:]
            # Reciprocal convex mixture: coefficients depend on material, never on wo/wi.
            wi=np.array([.3,0,np.sqrt(.91)]);d=.7*(1-.5);f0=.04*.5+.7*.5
            f=lambda x,y:d/np.pi+(1-d)*oracle.kernel(x,y,mx,my,.31,f0)
            assert abs(float(f(wi,wo)-f(wo,wi)))<1e-12
    record_property('largest_independently_integrated_budget_reflectance',max(energies))
    record_property('directional_integrals',len(energies))


@pytest.mark.parametrize('corrupt',['dtype','shape','missing','nan'])
def test_incompatible_tensor_catalog_is_rejected_before_weight_copy(corrupt):
    model,_,_,_=fixture();state={n:v.clone() for n,v in model.state_dict().items()};key='shader_network.material'
    before=model.shader_network.material.detach().clone()
    if corrupt=='dtype':state[key]=state[key].double()
    if corrupt=='shape':state[key]=state[key][:-1]
    if corrupt=='missing':del state[key]
    if corrupt=='nan':state[key][0]=float('nan')
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert torch.equal(model.shader_network.material,before)


@pytest.mark.parametrize('vertical',[0.,1e-4,-1e-4])
def test_actual_shader_negative_axis_has_finite_forward_and_backward(vertical):
    model,_,_,_=fixture();s=model.shader_network
    with torch.no_grad():s.material[6:8]=torch.tensor([0.,.5+vertical])
    out=shade(model);out['linear_rgb'].sum().backward()
    assert torch.isfinite(out['rgb_pr']).all() and torch.isfinite(s.material.grad).all() and s.material.grad.abs().sum()>0
