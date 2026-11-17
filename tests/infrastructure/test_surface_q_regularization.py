import math
from pathlib import Path

import numpy as np
import pytest
torch=pytest.importorskip('torch')
from infrastructure.surface_q_regularization import (world_q,tensor_from_axes,normal_transport,
    transported_q_loss,surface_pairs,material_boundary_weights,candidate_pair_loss,gradient_diagnostics)
from infrastructure.signatures import SignatureError


def material(theta=.3,mx=.4,my=.2,n=None):
    dtype=torch.float64
    n=torch.tensor([[0.,0.,1.]],dtype=dtype) if n is None else n
    return torch.tensor([[mx]],dtype=dtype),torch.tensor([[my]],dtype=dtype),torch.tensor([[math.cos(2*theta),math.sin(2*theta)]],dtype=dtype),n


def test_pi_and_axis_swap_equivalence():
    mx,my,r,n=material();q=world_q(mx,my,r,n,torch)
    assert torch.allclose(q,world_q(*material(theta=.3+math.pi),torch),atol=1e-14)
    assert torch.allclose(q,world_q(my,mx,-r,n,torch),atol=1e-14)
    loss,_=transported_q_loss(q,world_q(my,mx,-r,n,torch),n,n,torch.ones(1),torch)
    assert loss<1e-28


def test_reference_frame_change_keeps_world_tensor():
    mx,my,_,_=material();t=torch.tensor([[0.,1.,0.]],dtype=torch.float64);b=torch.tensor([[-1.,0.,0.]],dtype=torch.float64)
    a=.7;theta=.3
    old=tensor_from_axes(mx,my,math.cos(theta)*t+math.sin(theta)*b,-math.sin(theta)*t+math.cos(theta)*b)
    t1=math.cos(a)*t+math.sin(a)*b;b1=-math.sin(a)*t+math.cos(a)*b
    new=tensor_from_axes(mx,my,math.cos(theta-a)*t1+math.sin(theta-a)*b1,-math.sin(theta-a)*t1+math.cos(theta-a)*b1)
    assert torch.allclose(old,new,atol=1e-14)


def test_transport_preserves_curved_surface_tensor():
    mx,my,r,n=material();m=torch.tensor([[.3,0.,math.sqrt(.91)]],dtype=torch.float64)
    R=normal_transport(n,m,torch);q=world_q(mx,my,r,n,torch);q1=R@q@R.transpose(-1,-2)
    assert torch.allclose((R@n[...,None])[...,0],m,atol=1e-14)
    loss,_=transported_q_loss(q,q1,n,m,torch.ones(1),torch)
    assert loss<1e-28 and torch.allclose(torch.linalg.det(R),torch.ones(1,dtype=R.dtype))
    with pytest.raises(SignatureError,match='Antipodal'):normal_transport(n,-n,torch)


def test_gradients_widths_orientation_and_isotropic_limit():
    mx,my,r,n=material();mx.requires_grad_();my.requires_grad_();r.requires_grad_()
    q=world_q(mx,my,r,n,torch);loss=(q-torch.diag_embed(torch.tensor([[.07,.09,0.]],dtype=q.dtype))) .square().sum()
    gs=torch.autograd.grad(loss,(mx,my,r))
    assert all(torch.isfinite(g).all() and torch.linalg.norm(g)>0 for g in gs)
    r2=r.detach().clone().requires_grad_();iso=world_q(mx.detach(),mx.detach(),r2,n,torch)
    grad=torch.autograd.grad(iso.square().sum(),r2)[0]
    assert torch.linalg.norm(grad)<1e-14
    assert torch.autograd.gradcheck(lambda x,y,z:world_q(x,y,z,n,torch),(mx,my,r),eps=1e-6,atol=1e-5)


def test_surface_pairs_do_not_connect_close_disjoint_sheets():
    v=np.array([[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0]],float)
    v=np.r_[v,v+[0,0,.001]];f=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    p=surface_pairs(v,f);assert {tuple(sorted(pair)) for pair in p['face_pairs']}=={(0,1),(2,3)}
    assert np.all(p['distance']+1e-15>=np.linalg.norm(p['points0']-p['points1'],axis=-1))
    assert p['excluded_non_two_face_edges']==8


def test_degenerate_faces_excluded_from_loss_only_without_mesh_repair():
    v=np.array([[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0],[2,2,2]],float)
    f=np.array([[0,1,2],[0,2,3],[4,4,4]]);before=v.copy(),f.copy()
    p=surface_pairs(v,f)
    assert p['excluded_degenerate_faces']==1 and len(p['face_pairs'])==1
    assert np.array_equal(v,before[0]) and np.array_equal(f,before[1])


def test_boundary_weights_detached_and_sharp_color_edge_downweighted():
    weights=material_boundary_weights(np.array([[0,0,0],[0,0,0]]),np.array([[0,0,0],[1,1,1]]),np.zeros((2,1)),np.zeros((2,1)))
    assert weights[0]==1 and weights[1]<1e-7
    mx,my,r,n=material();q=world_q(mx,my,r,n,torch)
    w=torch.ones(1,requires_grad=True);loss,_=transported_q_loss(q,2*q,n,n,w,torch)
    assert not loss.requires_grad and w.grad is None


def test_gradient_observer_does_not_accumulate_or_consume_rng():
    p=torch.tensor([.2,.3],requires_grad=True);p.grad=torch.ones_like(p)*9;before=torch.get_rng_state().clone()
    result=gradient_diagnostics(dict(a=p.sum(),b=-p.sum(),constant=p.new_tensor(0.)),[p],torch)
    assert result['gradient_cosines']['a__b']==pytest.approx(-1)
    assert result['gradient_cosines']['a__constant'] is None
    assert torch.equal(p.grad,torch.ones_like(p)*9) and torch.equal(torch.get_rng_state(),before)


def test_world_export_is_gauge_invariant_and_raw_data_preserved(tmp_path):
    from test_sphere_replay import tiny_arrays
    from infrastructure.sphere_visuals import write_view
    a=tiny_arrays();p=tmp_path/'view';write_view(p,a,dict(camera_id='fixture'),np)
    with np.load(p/'materials.npz',allow_pickle=False) as data:
        q=data['Q_world'];t=2*a['tangents']-1;b=2*a['bitangents']-1
        swapped=a['my'][...,None]**2*b[..., :,None]*b[...,None,:]+a['mx'][...,None]**2*t[..., :,None]*t[...,None,:]
        assert np.array_equal(q,swapped) and np.array_equal(data['mx'],a['mx'])
        assert 'major_axis_projector_world' in data.files and 'raw_angle_radians' in data.files
    assert (p/'orientation_raw.png').is_file() and (p/'orientation_world_projector.png').is_file()


def test_analytical_local_specular_partials_and_quadrature_measure():
    from experiments.analyze_sphere_regularization import local_specular_gradient
    from audits.anisotropic_brdf import hemisphere
    wi,w=hemisphere(32,64);assert w.sum()==pytest.approx(2*np.pi,abs=1e-14)
    assert np.sum(wi[:,2]*w)==pytest.approx(np.pi,abs=1e-14)
    kd=np.array([[.3,.4,.5]]);metal=np.array([[.2]]);coefficient=np.array([[.001,.002,.003]])
    L=coefficient*kd*(1-metal);gkd,gm=local_specular_gradient(kd,metal,L)
    assert (gkd>0).all() and (gm<0).all()
    h=1e-6
    def loss(c,m):return .05*(coefficient*c*(1-m)).sum()
    assert np.isclose((loss(kd+h,metal)-loss(kd-h,metal))/(2*h),gkd.sum())


def test_frozen_variants_and_sources_unchanged():
    from infrastructure.stage_sources import ROOT,PINS,baseline,digest
    import subprocess,hashlib
    for stage in PINS:baseline(stage,PINS[stage])
    for path in ('scientific_variants/material_a_v1.py','infrastructure/material_variants.py'):
        original=subprocess.check_output(['git','show','92d433dd809bab5fe8667a2920dc54ce94071f35:'+path],cwd=str(ROOT))
        assert digest(ROOT/path)==hashlib.sha256(original).hexdigest()


def test_selectable_pair_regularizer_and_finite_field_gradients():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.width=torch.nn.Parameter(torch.tensor(.3));self.angle=torch.nn.Parameter(torch.tensor(.2))
        def predict_anisotropic_components(self,p):
            mx=self.width+.03*p[:,:1];my=self.width*.7+.04*p[:,:1]
            theta=self.angle*p[:,:1];r=torch.cat([torch.cos(2*theta),torch.sin(2*theta)],-1)
            kd=mx.expand(-1,3);metal=my
            return mx,my,metal,kd,r,(r+1)/2
    model=Field();pairs=dict(points0=torch.zeros(2,3),points1=torch.ones(2,3)*.01,
        normals0=torch.tensor([[0.,0.,1.]]).repeat(2,1),normals1=torch.tensor([[0.,0.,1.]]).repeat(2,1),
        weights=torch.ones(2),distance=torch.ones(2)*.01)
    for mode in ('surface_chart_control_v1','surface_world_q_v1'):
        model.zero_grad();parts=candidate_pair_loss(model,pairs,torch,mode,step=100)
        sum(parts.values()).backward()
        assert model.width.grad is not None and torch.isfinite(model.width.grad) and model.width.grad.abs()>0
        assert model.angle.grad is not None and torch.isfinite(model.angle.grad) and model.angle.grad.abs()>0
    with pytest.raises(SignatureError):candidate_pair_loss(model,pairs,torch,'typo')


def test_external_hook_preserves_energy_and_specular_exactly():
    from infrastructure.sphere_surface_experiment import install_spatial
    from types import SimpleNamespace
    shader=SimpleNamespace(cfg=dict(regularization_version='legacy_v1',reg_change=True,
        reg_energy_loss_lambda=.05,reg_spec_loss_lambda=.05,change_type='gaussian',change_eps=.02))
    import infrastructure.sphere_surface_experiment as experiment
    from unittest.mock import patch
    fd=torch.tensor([[.8,.3,.2],[.9,.5,.1]],requires_grad=True)
    proxy=torch.ones_like(fd)*.4;L=torch.ones_like(fd)*.01
    with patch.object(experiment,'candidate_pair_loss',return_value={'loss_surface_q':fd.sum()*0+.2,'loss_surface_scalar':fd.sum()*0+.1}):
        install_spatial(shader,{},'surface_world_q_v1',torch)
        values=shader.anisotropic_regularization(torch.zeros(2,3),None,None,None,None,None,None,None,fd,proxy,L,None,None,0)
    expected=.3+.05*torch.relu(fd+proxy-1).sum(-1).mean()+.05*L.sum(-1).mean()
    assert torch.allclose(values,torch.ones(2)*expected)
    assert shader.cfg['regularization_version']=='legacy_v1'


def test_endpoint_stop_occurs_after_owned_checkpoint_only(tmp_path,monkeypatch):
    import infrastructure.sphere_replay as replay
    from infrastructure.sphere_surface_experiment import trainer_class,ControlledEndpoint
    import json
    events=[]
    class Base:
        def _save_model(self,step,best,save_fn=None):
            events.append(step);(tmp_path/'checkpoints'/('step-%06d.json'%step)).write_text('{}')
    monkeypatch.setattr(replay,'trainer_adapter',lambda base,*args:base)
    (tmp_path/'checkpoints').mkdir();r={'optimizer_returns':8000,'new_optimizer_returns':4000}
    cls=trainer_class(Base,tmp_path,r,lambda:None,None,torch,'surface_world_q_v1',None)
    with pytest.raises(ControlledEndpoint):cls()._save_model(8000,0)
    assert events==[8000]
    assert json.loads((tmp_path/'checkpoints/step-008000.json').read_text())['external_spatial_regularizer']=='surface_world_q_v1'


def test_unconditional_energy_candidate_keeps_rejected_proposals_in_denominator():
    from infrastructure.surface_q_regularization import energy_proxy_candidate
    valid=torch.tensor([[[True],[True],[True],[False]]]);weights=valid.expand(1,4,3).float()
    d=torch.ones(1,3)*.1
    legacy=energy_proxy_candidate(d,weights,valid,torch,'legacy_accepted_count_v1')
    corrected=energy_proxy_candidate(d,weights,valid,torch)
    assert torch.allclose(legacy,torch.ones_like(d)*1.1)
    assert torch.allclose(corrected,torch.ones_like(d)*.85)
    assert torch.relu(legacy-1).sum()>0 and torch.relu(corrected-1).sum()==0


def test_surface_modes_preserve_original_neighborhood_rng_consumption():
    from infrastructure.surface_q_regularization import preserve_legacy_spatial_draws
    state=torch.get_rng_state().clone()
    try:
        torch.manual_seed(6033);torch.rand(512,1);torch.normal(mean=0.,std=.02,size=(512,1));expected=torch.get_rng_state().clone()
        torch.manual_seed(6033)
        preserve_legacy_spatial_draws(torch.zeros(512,3),dict(change_type='gaussian',change_eps=.02),torch)
        assert torch.equal(torch.get_rng_state(),expected)
    finally:torch.set_rng_state(state)


def test_uniform_hemisphere_energy_reference_matches_half_sample_estimator():
    from audits import anisotropic_brdf as ref
    wi,omega=ref.hemisphere(64,128);wo=np.array([0.,0.,1.])
    independent=np.sum(ref.kernel(wi,wo,1.,1.,f0=1.)*wi[:,2]*omega)
    u=(np.arange(4096)+.5)/4096;h=ref.half_samples(u,np.zeros_like(u),1.,1.)
    rays=2*(h@wo)[:,None]*h-wo;ci=np.maximum(rays[:,2],0)
    weights=ci/h[:,2];unconditional=weights.mean();conditional=weights.sum()/(ci>0).sum()
    assert np.isclose(unconditional,independent,atol=2e-5)
    assert np.isclose((ci>0).mean(),1-np.exp(-1),atol=1/4096)
    assert conditional>unconditional


def test_detached_ndf_weights_cannot_by_themselves_localize_constant_diffuse():
    D=torch.tensor([[[1.],[2.],[3.]]],requires_grad=True);d=torch.tensor([[[.2,.3,.4]]],requires_grad=True)
    loss=(torch.softmax(D.detach(),dim=1)*d).sum()/3/3
    expected=d.sum()/9
    assert torch.allclose(loss,expected)
    gd,gD=torch.autograd.grad(loss,(d,D),allow_unused=True)
    assert (gd>0).all() and gD is None


def test_execution_entry_is_cold_and_requires_explicit_authorization(tmp_path):
    import subprocess,sys
    from infrastructure.stage_sources import ROOT
    code="import sys;import run_sphere_surface_experiment;assert not any(n.split('.')[0] in ('torch','train','network','dataset','_raytracing') for n in sys.modules)"
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=str(ROOT),capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    from run_sphere_surface_experiment import authorize
    with pytest.raises(SignatureError,match='authorization'):authorize(None,None,{},'f'*64,'a'*40)


def test_clipped_light_chromaticity_has_no_scale_anchor():
    light=torch.tensor([[.45,2.7,2.85]],dtype=torch.float64,requires_grad=True)
    srgb=torch.where(light<=.0031308,12.92*light,1.055*light.pow(1/2.4)-.055).clamp(0,1)
    loss=.01*(srgb-srgb.mean(-1,keepdim=True)).abs().sum()
    g=torch.autograd.grad(loss,light)[0]
    assert g[0,0]<0 and g[0,1]==g[0,2]==0
    for value in (.2,2.,20.):
        gray=torch.ones(1,3,dtype=torch.float64)*value
        s=torch.where(gray<=.0031308,12.92*gray,1.055*gray.pow(1/2.4)-.055).clamp(0,1)
        assert .01*(s-s.mean(-1,keepdim=True)).abs().sum()==0


def test_exact_checkpoint_adam_fork_restoration_uses_synthetic_fixture_only(tmp_path,monkeypatch):
    import infrastructure.sphere_replay as replay
    import infrastructure.sphere_surface_experiment as experiment
    from infrastructure.stage_sources import digest
    class Shader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.4,.6]))
            self.cfg=dict(regularization_version='legacy_v1',reg_change=True,change_type='gaussian',change_eps=.02,
                          reg_energy_loss_lambda=.05,reg_spec_loss_lambda=.05)
            self.diffuse_direction_samples=torch.arange(4.);self.specular_direction_samples=torch.arange(4.)
    class Network(torch.nn.Module):
        def __init__(self):super().__init__();self.shader_network=Shader()
    def optimizer(net):return torch.optim.Adam(net.parameters(),eps=1e-4,amsgrad=True)
    original=Network();opt=optimizer(original)
    original.shader_network.weight.sum().backward();opt.step()
    for state in opt.state.values():
        if isinstance(state['step'],torch.Tensor):state['step'].fill_(4000)
        else:state['step']=4000  # PyTorch1.8 stores Adam counters as integers.
    checkpoint=tmp_path/'input.pth';torch.save(dict(step=4000,best_para=.4,
        network_state_dict=original.state_dict(),optimizer_state_dict=opt.state_dict()),checkpoint)
    report={'optimizer_returns':0};out=tmp_path/'run';out.mkdir();(out/'cache').mkdir()
    class Loss:
        def __call__(self,*args,**kwargs):return {}
        def cosine_weight_schedule(self,step):return 1.
    class Base:
        def _init_network(self):
            self.network=Network();self.optimizer=optimizer(self.network);self.train_losses=[Loss(),Loss()]
            self.pth_fn=str(out/'model.pth');self.best_pth_fn=str(out/'best.pth')
            step=self.optimizer.step
            def observed(*args,**kwargs):
                v=step(*args,**kwargs);report['optimizer_returns']+=1;return v
            self.optimizer.step=observed
            report['split_ids']={'train':['1'],'validation':['36'],'nvs':[]}
    monkeypatch.setattr(replay,'trainer_adapter',lambda base,*args:base)
    monkeypatch.setattr(experiment,'prepare_pairs',lambda model,t:({'points0':torch.zeros(1,3)},dict(fixed=True)))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    cls=experiment.trainer_class(Base,out,report,lambda:None,None,torch,'surface_world_q_v1',None,
        (checkpoint,dict(sha256=digest(checkpoint))))
    trainer=cls();trainer._init_network()
    assert torch.equal(trainer.network.shader_network.weight,original.shader_network.weight)
    assert trainer._load_model()==(.4,4000) and report['checkpoint_restore']['optimizer_restored']
    p=trainer.network.shader_network.weight
    assert int(trainer.optimizer.state[p]['step'])==4000
    assert torch.equal(trainer.optimizer.state[p]['exp_avg'],opt.state[original.shader_network.weight]['exp_avg'])
    trainer.optimizer.zero_grad();p.sum().backward();trainer.optimizer.step()
    assert report['optimizer_returns']==4001 and report['new_optimizer_returns']==1
    assert report['optimizer_events'][0]['active_state_counters_advanced']
