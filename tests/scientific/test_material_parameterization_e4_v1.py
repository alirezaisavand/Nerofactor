"""E4 analytical and tiny synthetic tests only; real N1/shader imports forbidden."""
import importlib.abc
import math
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments import material_chart_e4_v1 as chart
from experiments import material_parameterization_e4_v1 as pilot


@pytest.fixture(autouse=True)
def no_scientific_execution():
    forbidden=('experiments.neural_material_field_v1','experiments.material_shader_v1','experiments.neural_material_recovery_v1')
    assert not any(m.startswith(forbidden) for m in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,*args):
            if name.startswith(forbidden):raise AssertionError('Scientific module forbidden: '+name)
    block=Block();sys.meta_path.insert(0,block);t=torch.get_rng_state().clone();n=np.random.get_state()
    try:yield
    finally:
        sys.meta_path.remove(block);torch.set_rng_state(t);np.random.set_state(n)
        assert not any(m.startswith(forbidden) for m in sys.modules)


def original_block(delta,target):
    q=torch.sigmoid(target[:,:2]+delta[:,:2]);rot=torch.sigmoid(target[:,2:]+delta[:,2:])
    return torch.cat((.04+.96*q,F.normalize(2*rot-1,dim=-1,eps=1e-6)),-1)


def new_block(delta,target):
    q=chart.compressed_probability(delta[:,:2],target[:,:2]);e,_=chart.transport(delta[:,2:],target[:,2:])
    return torch.cat((.04+.96*q,e),-1)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_origin_physical_outputs_and_local_jacobian_match(dtype):
    target=torch.tensor([[.3,-.8,.7,-.4]],dtype=dtype);delta=torch.zeros_like(target,requires_grad=True)
    assert torch.equal(original_block(delta,target),new_block(delta,target))
    a=torch.autograd.functional.jacobian(lambda v:original_block(v,target),delta)
    b=torch.autograd.functional.jacobian(lambda v:new_block(v,target),delta)
    assert torch.allclose(a,b,atol=1e-7 if dtype==torch.float32 else 1e-14,rtol=2e-6 if dtype==torch.float32 else 1e-12)


@pytest.mark.parametrize('displacement',[-1000.,-20.,0.,20.,1000.])
def test_width_domain_correct_derivatives_and_finite_gradients(displacement):
    d=torch.tensor([displacement],dtype=torch.float64,requires_grad=True);target=torch.tensor([.2],dtype=torch.float64)
    q=chart.compressed_probability(d,target);g=torch.autograd.grad(q.sum(),d)[0]
    expected=q.detach()*(1-q.detach())/torch.sqrt(1+d.detach()**2)
    assert 0<float(q)<1 and float(g)>0 and torch.allclose(g,expected,rtol=1e-12,atol=1e-17)
    assert .04<float(.04+.96*q)<1
    assert torch.autograd.gradcheck(lambda v:chart.compressed_probability(v,target),(d,),eps=1e-5,atol=1e-8,rtol=1e-4)


def test_far_from_anchor_attenuation_is_reduced_without_initial_gain_change():
    for value in (-20.,20.):
        a=torch.tensor([value],dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
        ga=torch.autograd.grad(torch.sigmoid(a).sum(),a)[0]
        gb=torch.autograd.grad(chart.compressed_probability(b,torch.zeros_like(b)).sum(),b)[0]
        assert float(gb)>10000*float(ga)


def test_axis_swap_orientation_periodicity_and_beckmann_tensor_equivalence():
    delta=torch.tensor([[.7,-1.1,.4,-.9]],dtype=torch.float64);target=torch.tensor([[.2,-.5,.8,-.3]],dtype=torch.float64)
    a=new_block(delta,target)
    swapped_target=torch.cat((target[:,:2].flip(-1),-target[:,2:]),-1)
    swapped_delta=torch.cat((delta[:,:2].flip(-1),-delta[:,2:]),-1)
    b=new_block(swapped_delta,swapped_target)
    qa=chart.tensor_q(a[:,:2],a[:,2:]);qb=chart.tensor_q(b[:,:2],b[:,2:])
    assert torch.allclose(qa,qb,atol=1e-14,rtol=1e-13)
    # Independent analytic Beckmann D for explicit half-vectors, not shader queries.
    widths=a[0,:2].detach().numpy();encoding=a[0,2:].detach().numpy();theta=.5*np.arctan2(encoding[1],encoding[0])
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    q=rot@np.diag(widths**2)@rot.T
    assert np.allclose(q,qa[0].detach().numpy(),rtol=1e-13,atol=1e-14)
    for hx,hy in ((0.,0.),(.2,.3),(-.6,.1)):
        hz=np.sqrt(1-hx*hx-hy*hy);v=np.array([hx,hy])/hz;local=rot.T@v
        axis=np.exp(-np.sum((local/widths)**2))/(np.pi*np.prod(widths)*hz**4)
        tensor=np.exp(-v@np.linalg.solve(q,v))/(np.pi*np.sqrt(np.linalg.det(q))*hz**4)
        assert np.isclose(axis,tensor,rtol=1e-12,atol=1e-14)
    e,j=chart.orientation_constants(target[:,2:]);shift=math.pi*j/(j*j).sum()
    p,_=chart.transport(delta[:,2:]+shift,target[:,2:])
    assert torch.allclose(p,a[:,2:],atol=1e-14,rtol=1e-13)


def test_isotropic_orientation_is_unconstrained_with_finite_zero_Q_derivative():
    d=torch.tensor([[.4,-.7]],dtype=torch.float64,requires_grad=True);target=torch.tensor([[.8,-.3]],dtype=torch.float64)
    e,_=chart.transport(d,target);widths=torch.tensor([[.4,.4]],dtype=torch.float64)
    q=chart.tensor_q(widths,e);grad=torch.autograd.grad(q.sum(),d)[0]
    assert torch.allclose(q,torch.eye(2,dtype=torch.float64)[None]*.16,atol=1e-15)
    assert torch.equal(grad,torch.zeros_like(grad))
    with pytest.raises(Exception,match='anchor'):chart.orientation_constants(torch.zeros_like(target))


class Head(nn.Module):
    def __init__(self,width,features,target):
        super().__init__();self.predictor=nn.Sequential(nn.Linear(7,5),nn.ReLU(),nn.Linear(5,width),nn.Sigmoid())
        self.register_buffer('center',self.pre_sigmoid(features).detach().clone());self.register_buffer('target',torch.tensor(target,dtype=torch.float32).reshape(1,width))
    def pre_sigmoid(self,x):
        for i in range(len(self.predictor)-1):x=self.predictor[i](x)
        return x
    def forward(self,x):return self.predictor[-1](self.target+(self.pre_sigmoid(x)-self.center))


class Material:
    def __init__(self,physical,encoding):self.physical=physical;self.encoding=encoding
    def validate(self):
        assert torch.isfinite(self.physical).all() and ((self.physical[:,:2]>=.04)&(self.physical[:,:2]<=1)).all();return self
    def expand(self,n):return Material(self.physical.expand(n,7),self.encoding.expand(n,2))


class Tiny(nn.Module):
    """Small fake backbone: no real scientific model definitions or queries."""
    def __init__(self):
        super().__init__();self.seed=101;self.source_identity='synthetic'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3);self.feats_network=nn.Sequential(nn.Linear(3,4),nn.ReLU());anchor=torch.zeros(1,3)
            f=torch.cat((self.feats_network(anchor),anchor),-1)
            for name,width,target in (('mx_predictor',1,[.2]),('my_predictor',1,[-.3]),('metallic_predictor',1,[.1]),('rotation_predictor',2,[.8,-.4]),('kd_predictor',3,[.1,-.2,.3])):
                setattr(self,name,Head(width,f,target))
        self.register_buffer('anchor',anchor);self.register_buffer('initial_latent',torch.zeros(7))
    def forward(self,p):
        f=torch.cat((self.feats_network(self.anchor),self.anchor),-1)
        mx=.04+.96*self.mx_predictor(f);my=.04+.96*self.my_predictor(f);metal=self.metallic_predictor(f);kd=self.kd_predictor(f)
        e=F.normalize(2*self.rotation_predictor(f)-1,dim=-1,eps=1e-6);unit=e/torch.norm(e,dim=-1,keepdim=True)
        theta=.5*torch.atan2(unit[:,1:2],unit[:,:1])
        return Material(torch.cat((mx,my,theta,kd,metal),-1),e).expand(len(p))


def test_tiny_capacity_initialization_gradient_flow_and_cleanup():
    base=Tiny();before={n:p.detach().clone() for n,p in base.named_parameters()};candidate=chart.ChartMaterial(base,Material);points=torch.zeros(1,3)
    assert set(before)==set(dict(candidate.named_parameters())) and all(torch.equal(p,before[n]) for n,p in candidate.named_parameters())
    a=base(points);b=candidate(points);assert torch.equal(a.physical,b.physical) and torch.equal(a.encoding,b.encoding)
    with candidate.observe():
        v=candidate(points);loss=v.physical[:,:2].sum()+v.physical[:,3:].sum()+v.encoding.sum();loss.backward()
        assert candidate.nodes and all(t.grad is not None and torch.isfinite(t.grad).all() for t in candidate.nodes.values())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    assert not candidate.nodes and not candidate._active and all(not m._forward_hooks and not m._forward_pre_hooks for m in candidate.modules())
    with pytest.raises(RuntimeError):
        with candidate.observe():candidate(points);raise RuntimeError('synthetic interrupt')
    assert not candidate.nodes and all(not m._forward_hooks and not m._forward_pre_hooks for m in candidate.modules())
    with pytest.raises(Exception,match='positions'):candidate(points.double())


@pytest.mark.parametrize('material',['M1','M2','M3'])
def test_matched_history_bookkeeping_without_real_optimization(material):
    class Core:
        scientific=False
        def __init__(self):self.material=material;self.work=pilot.Work(material);self.state=0;self.endpoints={};self.cached=self.measure(0)
        def measure(self,state):
            assert state==self.state;self.work.attempt('forwards');self.work.complete('forwards');self.work.attempt('backwards');self.work.complete('backwards');return dict(state=state)
        def step(self):self.state+=1
        def evaluate(self,state):assert state==self.state;return dict(state=state)
    c=Core();rows=[];pilot.run_schedule(c,rows.append)
    assert c.state==100 and c.work.done==c.work.limit and set(c.endpoints)=={'10','100'}
    assert sum(r.get('update',False) for r in rows)==100 and len(rows)==caps(material)['forwards']
    assert sum(r['kind']=='fit' and r['state']==10 for r in rows)==2+int(material=='M3')


def caps(material):return pilot.caps(material)


def test_atomic_caps_exact_total_budget_and_execution_gate():
    w=pilot.Work('M1');w.attempts['forwards']=104;before=dict(w.attempts)
    with pytest.raises(Exception):w.attempt('evaluations')
    assert w.attempts==before
    assert {k:sum(pilot.caps(m)[k] for m,_ in pilot.CASES) for k in pilot.caps('M1')}==dict(updates=500,forwards=522,backwards=512,evaluations=10)
    with pytest.raises(Exception,match='approval'):pilot.validate_review(dict(phase='E.4',scientific_execution_authorized=False),'0'*64)
