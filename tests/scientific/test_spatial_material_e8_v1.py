"""New E8 synthetic fixtures only; no N1, shader, references or neural training."""
import ast
import importlib.abc
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import spatial_material_e8_v1 as method
from experiments import spatial_pilot_e8_v1 as pilot

FORBIDDEN=('experiments.neural_material_field_v1','experiments.material_shader_v1',
 'experiments.neural_material_recovery_v1','experiments.run_known_material_recovery_v1',
 'experiments.anisotropic_estimator_v1','experiments.known_material_reference_v1','experiments.multi_material_reference_v1')


@pytest.fixture(autouse=True)
def block_science():
    assert not any(n.startswith(FORBIDDEN) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,*args):
            if name.startswith(FORBIDDEN):raise AssertionError('Scientific import blocked: '+name)
    block=Block();sys.meta_path.insert(0,block);old=torch.get_rng_state().clone();nrng=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:
        sys.meta_path.remove(block);assert torch.equal(old,torch.get_rng_state())
        after=np.random.get_state();assert nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        assert not any(n.startswith(FORBIDDEN) for n in sys.modules)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name,width in method.HEADS:
            head=torch.nn.Module();head.predictor=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Tanh(),
                torch.nn.utils.weight_norm(torch.nn.Linear(8,width)),torch.nn.Sigmoid());setattr(self,name,head)
        self.rotation_predictor.predictor[-2].bias.data.copy_(torch.tensor([1.,-.6]))
    def forward(self,x):
        q=[getattr(self,n).predictor(x) for n,_ in method.HEADS];e=torch.nn.functional.normalize(2*q[3]-1,dim=-1)
        p=torch.cat((.04+.96*q[0],.04+.96*q[1],.5*torch.atan2(e[:,1:2],e[:,:1]),q[4],q[2]),-1)
        return SimpleNamespace(physical=p,encoding=e)


def system():
    p=np.array([.35,.6,.2,.3,.4,.5,.25]);q=np.r_[(p[:2]-.04)/.96,p[6],(1+.5*np.array([np.cos(2*p[2]),np.sin(2*p[2])]))/2,p[3:6]]
    z=np.tile(np.log(q/(1-q)),(4,1));features={n:np.c_[np.arange(4),np.ones(4)] for n,_ in method.HEADS}
    teacher=p.copy();teacher[0]+=.02
    return features,z,np.tile(p,(4,1)),teacher


def test_truth_is_smooth_nonconstant_physical_and_spatial_splits_are_disjoint():
    train=np.asarray(method.TRAIN);held=np.asarray(method.HELD);p=method.truth(np.r_[train,held])
    assert p.shape==(6,7) and (p[:,:2]>.04).all() and (p[:,:2]<1).all() and (p[:,3:]>0).all() and (p[:,3:]<1).all()
    assert all(np.ptp(p,axis=0)>0) and not set(method.TRAIN)&set(method.HELD)
    assert not np.allclose(p[:4],p[0])
    midpoint=method.truth([[0.,0.,0.]])[0]
    np.testing.assert_allclose(p[:4].mean(0),midpoint,atol=1e-15)
    with pytest.raises(Exception):method.truth([[0.,0.,1.]])


def test_genuine_coordinate_dependence_matched_initialization_and_gradients():
    a=Toy();b=Toy();b.load_state_dict(a.state_dict());x=torch.tensor(method.TRAIN,dtype=torch.float32,requires_grad=True)
    ma=a(x);mb=b(x);assert torch.equal(ma.physical,mb.physical)
    assert float(ma.physical[:,0].max()-ma.physical[:,0].min())>1e-5
    g,=torch.autograd.grad(ma.physical[:,0].sum(),x);h=1e-3
    xp=x.detach().clone();xm=x.detach().clone();xp[:,0]+=h;xm[:,0]-=h
    fd=(a(xp).physical[:,0]-a(xm).physical[:,0])/(2*h)
    torch.testing.assert_close(g[:,0],fd,atol=5e-5,rtol=.01)


def test_kernel_retention_reduces_other_location_drift_and_counts_eight_solves():
    f,z,p,t=system();events=[]
    _,retained=method.proposed_updates(f,z,p,0,t,1.,events.append)
    _,free=method.proposed_updates(f,z,p,0,t,0.)
    assert events==['attempt','complete']*8
    assert np.linalg.norm(np.array(retained['predicted_logit_changes'])[1:])<np.linalg.norm(np.array(free['predicted_logit_changes'])[1:])
    assert all(s['solve_residual_L2']<1e-12 for s in retained['systems'])


def test_serial_location_gradients_equal_one_shared_joint_mean():
    a=Toy();b=Toy();b.load_state_dict(a.state_dict());x=torch.tensor(method.TRAIN,dtype=torch.float32)
    target=torch.linspace(.1,.7,28).reshape(4,7)
    for i in range(4):((a(x[i:i+1]).physical-target[i:i+1])**2).mean().div(4).backward()
    ((b(x).physical-target)**2).mean().backward()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-8)


def test_rejected_solve_cap_precedes_any_linear_solve(monkeypatch):
    f,z,p,t=system();calls=[]
    def forbidden(*args):calls.append('solve');raise AssertionError('Must not solve')
    monkeypatch.setattr(np.linalg,'cholesky',forbidden)
    def cap(stage):
        assert stage=='attempt';raise RuntimeError('synthetic solve cap')
    with pytest.raises(RuntimeError,match='cap'):method.proposed_updates(f,z,p,0,t,1.,cap)
    assert calls==[]


def test_native_axis_gauge_covariance_and_isotropic_no_angle_prior():
    f,z,p,t=system();u,_=method.proposed_updates(f,z,p,0,t,1.)
    swapped=p[:,[1,0,2,3,4,5,6]].copy();swapped[:,2]+=np.pi/2;other=t[[1,0,2,3,4,5,6]].copy();other[2]+=np.pi/2
    zz=z.copy();zz[:,:2]=z[:,[1,0]];zz[:,3:5]=-z[:,3:5]
    v,_=method.proposed_updates(f,zz,swapped,0,other,1.)
    for name,target,sign in [('mx_predictor','my_predictor',1),('my_predictor','mx_predictor',1),('rotation_predictor','rotation_predictor',-1)]:
        for a,b in zip(u[name],v[target]):np.testing.assert_allclose(a,sign*b,atol=1e-12)
    np.testing.assert_allclose(method.embedding(swapped),method.embedding(p),atol=1e-15)
    p[:,1]=p[:,0];t=p[0].copy();t[2]+=.8
    v,_=method.proposed_updates(f,z,p,0,t,1.)
    assert all(np.count_nonzero(a)==0 for a in v['rotation_predictor'])


def test_weight_norm_projection_realizes_predicted_logits_and_preserves_hidden_state():
    model=Toy();x=torch.tensor(method.TRAIN,dtype=torch.float32);before={n:p.detach().clone() for n,p in model.named_parameters()}
    mat,f,z=method.capture(model,x);teacher=mat.physical[0].numpy().copy();teacher[0]+=.01
    updates,detail=method.proposed_updates(f,z,mat.physical.numpy(),0,teacher,1.)
    with method.transfer(model,updates) as commit:
        _,_,actual=method.capture(model,x);np.testing.assert_allclose(actual,z+np.array(detail['predicted_logit_changes']),atol=3e-7,rtol=0);commit()
    for name,p in model.named_parameters():
        if '.predictor.2.' not in name:assert torch.equal(before[name],p)
    assert all(not m._forward_hooks and len(m._forward_pre_hooks)<=1 for m in model.modules())


@pytest.mark.parametrize('failure',['reject','exception','exception_after_commit'])
def test_atomic_retention_rejection_and_interruption_restore_exact_weights(failure):
    model=Toy();before={n:p.detach().clone() for n,p in model.named_parameters()}
    mat,f,z=method.capture(model,torch.tensor(method.TRAIN,dtype=torch.float32));t=mat.physical[0].numpy().copy();t[0]+=.01
    updates,_=method.proposed_updates(f,z,mat.physical.numpy(),0,t,1.)
    try:
        with method.transfer(model,updates) as commit:
            if failure=='exception_after_commit':commit()
            if failure!='reject':raise RuntimeError('synthetic interruption')
    except RuntimeError:pass
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())


@pytest.mark.parametrize('mode',['good','other_loss','other_physical','wrong_direction','boundary','joint_loss'])
def test_acceptance_is_joint_training_only_with_spatial_guard(mode):
    _,_,p,t=system();after=p.copy();after[0]=(p[0]+t)/2;old=np.ones(4);new=np.array([.5,1.,1.,1.]);q=np.ones((4,8))*.5
    if mode=='other_loss':new[1]=1.11
    if mode=='other_physical':after[1,3]+=.2
    if mode=='wrong_direction':after[0]=p[0]-(t-p[0])
    if mode=='boundary':q[0,0]=0
    if mode=='joint_loss':new[0]=1.01
    d=method.accept(p,after,t,0,old,new,q);assert d['accepted']==(mode=='good')


def test_head_trust_radius_and_invalid_teacher_domain():
    f,z,p,t=system();t[0]=.999
    _,d=method.proposed_updates(f,z,p,0,t,1.);assert abs(np.array(d['predicted_logit_changes'])).max()<=.5+1e-14
    t[6]=1.
    with pytest.raises(Exception,match='Interior'):method.proposed_updates(f,z,p,0,t,1.)


def test_spatial_shader_adapter_passes_positions_material_and_seed_means():
    point=torch.tensor([.1,-.1,0.]);n=8
    batch=SimpleNamespace(positions=point.repeat(n,1),normals=torch.tensor([0.,0.,1.]).repeat(n,1),views=torch.ones(n,3),
        illumination=None,tapes=object(),seeds=4,n=2)
    material=SimpleNamespace(physical=torch.tensor([[.3,.5,.2,.2,.3,.4,.6]]),encoding=torch.tensor([[1.,0.]]))
    class FakeShader:
        def shade(self,positions,normals,views,mx,my,kd,metal,encoding,illumination,tapes,is_train):
            assert torch.equal(positions,point.repeat(n,1)) and is_train and tapes is batch.tapes
            rgb=torch.arange(24,dtype=torch.float32).reshape(8,3)
            assert torch.equal(kd,material.physical[:,3:6].expand(n,3))
            return dict(frame_orthonormal=torch.ones(n,dtype=torch.bool),diffuse_rgb=rgb,specular_rgb=2*rgb,linear_rgb=3*rgb)
    values=method.shade_material(FakeShader(),batch,material)
    torch.testing.assert_close(values[:,2],3*torch.arange(24,dtype=torch.float32).reshape(4,2,3).mean(0))


def test_work_caps_atomic_rejection_and_cost_matching_includes_extra_evaluation():
    w=pilot.Work();w.attempts['physical']=w.done['physical']=384
    before=dict(w.attempts)
    with pytest.raises(Exception,match='cap'):w.attempt('physical')
    assert w.attempts==before
    w.attempt('projections')
    with pytest.raises(Exception,match='Incomplete'):w.verify()
    for a in range(528,625,8):
        n=pilot.matching_updates([a,624]);assert 16*n+100<=a+74<16*(n+1)+100
    assert pilot.matching_updates([624,624])==37 and pilot.matching_updates([528,624])==31
    assert sum(pilot.CAPS[n] for n in ('neural_fit','neural_eval','physical'))==960
    assert pilot.CAPS['neural_reverse']+pilot.CAPS['physical_reverse']==1008


def test_execution_gate_no_scientific_imports_or_hidden_anchor_substitution(tmp_path):
    path=tmp_path/'review.json';path.write_text('{}\n')
    with pytest.raises(Exception,match='approval'):pilot.execute_reviewed(path,pilot.sha(path),'0'*64)
    tree=ast.parse(Path(method.__file__).read_text());factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_spatial')
    assert 'self._predict(self, positions)' in ast.unparse(factory) and 'self._predict(self, self.anchor)' not in ast.unparse(factory)
    for name in ('proposed_updates','accept'):
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert not any(isinstance(n,ast.Name) and n.id in ('truth','held_out','success') for n in ast.walk(func))
    assert '--execute' not in Path(pilot.__file__).read_text()
