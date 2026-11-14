"""E9 new synthetic freeze/restoration/gradient fixtures; no real model or shader."""
import ast
import importlib.abc
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import head_only_adam_e9_v1 as heads
from experiments import spatial_head_only_e9_v1 as pilot
from experiments import neural_material_continuation_e3 as runtime

FORBIDDEN=('experiments.neural_material_field_v1','experiments.material_shader_v1','experiments.neural_material_recovery_v1',
 'experiments.run_known_material_recovery_v1','experiments.anisotropic_estimator_v1','experiments.known_material_reference_v1','experiments.multi_material_reference_v1')


@pytest.fixture(autouse=True)
def synthetic_only():
    assert not any(n.startswith(FORBIDDEN) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,n,*args):
            if n.startswith(FORBIDDEN):raise AssertionError('Scientific import blocked: '+n)
    b=Block();sys.meta_path.insert(0,b);old=torch.get_rng_state().clone();nrng=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):yield
    finally:
        sys.meta_path.remove(b);assert torch.equal(old,torch.get_rng_state())
        after=np.random.get_state();assert nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.feats_network=torch.nn.Linear(3,3)
        self.register_buffer('calibration',torch.ones(1))
        for name,width in heads.HEADS:
            h=torch.nn.Module();h.predictor=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),
                torch.nn.Linear(4,4),torch.nn.Tanh(),torch.nn.Linear(4,4),torch.nn.Tanh(),
                torch.nn.utils.weight_norm(torch.nn.Linear(4,width)),torch.nn.Sigmoid());setattr(self,name,h)
        with torch.no_grad():self.rotation_predictor.predictor[-2].bias.copy_(torch.tensor([1.,-.6]))
    def forward(self,x):
        f=self.feats_network(x);q=[getattr(self,n).predictor(f) for n,_ in heads.HEADS]
        e=torch.nn.functional.normalize(2*q[3]-1,dim=-1)
        p=torch.cat((.04+.96*q[0],.04+.96*q[1],.5*torch.atan2(e[:,1:2],e[:,:1]),q[4],q[2]),-1)
        return SimpleNamespace(physical=p,encoding=e)


def snapshot(model,opt):return runtime.prior.state_arrays(model,opt)


def test_exact_trainable_set_registration_order_and_unchanged_forward():
    m=Toy();x=torch.tensor(pilot.method.TRAIN,dtype=torch.float32);before=m(x).physical.detach().clone()
    values={n:p.detach().clone() for n,p in m.named_parameters()};order=[n for n,_ in m.named_parameters() if n in heads.final_names()]
    selected=heads.freeze(m);view=heads.TrainableView(m)
    assert len(selected)==15 and list(selected)==order and list(view.selected)==order
    assert {n for n,p in m.named_parameters() if p.requires_grad}==heads.final_names()
    assert all(torch.equal(p,values[n]) for n,p in m.named_parameters())
    assert torch.equal(before,m(x).physical.detach()) and not m.calibration.requires_grad


def test_head_only_gradients_equal_unrestricted_model_partial_derivatives():
    a=Toy();b=Toy();b.load_state_dict(a.state_dict());heads.freeze(a);x=torch.tensor(pilot.method.TRAIN,dtype=torch.float32)
    a(x).physical.square().mean().backward();b(x).physical.square().mean().backward()
    for name,p in a.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p.grad,dict(b.named_parameters())[name].grad,rtol=0,atol=0)
        else:assert p.grad is None


def test_joint_four_location_gradient_equals_shared_mean_without_frozen_gradients():
    a=Toy();b=Toy();b.load_state_dict(a.state_dict());heads.freeze(a);heads.freeze(b);x=torch.tensor(pilot.method.TRAIN,dtype=torch.float32)
    target=torch.linspace(.1,.7,28).reshape(4,7)
    for i in range(4):((a(x[i:i+1]).physical-target[i:i+1])**2).mean().div(4).backward()
    ((b(x).physical-target)**2).mean().backward()
    for name,p in a.named_parameters():
        q=dict(b.named_parameters())[name]
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-8)
        else:assert p.grad is q.grad is None


def test_frozen_values_and_state_stay_exact_after_toy_Adam():
    m=Toy();selected=heads.freeze(m);opt=torch.optim.Adam(list(selected.values()),lr=.001)
    initial,_=snapshot(m,opt);m(torch.ones(1,3)).physical.square().sum().backward();opt.step()  # Tiny synthetic fixture only.
    heads.verify_frozen(m,initial,opt)
    assert set(opt.state)==set(selected.values()) and len(opt.state)==15
    assert any(not np.array_equal(p.detach().numpy(),initial['parameter/'+n]) for n,p in selected.items())


@pytest.mark.parametrize('violation',['value','gradient','buffer','optimizer'])
def test_frozen_state_violations_fail_closed(violation):
    m=Toy();selected=heads.freeze(m);opt=torch.optim.Adam(list(selected.values()),lr=.001);initial,_=snapshot(m,opt)
    frozen=next(p for p in m.parameters() if not p.requires_grad)
    if violation=='value':
        with torch.no_grad():frozen.add_(1.)
    if violation=='gradient':frozen.grad=torch.zeros_like(frozen)
    if violation=='buffer':m.calibration.add_(1.)
    if violation=='optimizer':opt=torch.optim.Adam(m.parameters(),lr=.001)
    with pytest.raises(Exception):heads.verify_frozen(m,initial,opt)


def test_frozen_buffer_signed_zero_is_also_bitwise_protected():
    m=Toy();selected=heads.freeze(m);opt=torch.optim.Adam(list(selected.values()),lr=.001)
    m.calibration.zero_();initial,_=snapshot(m,opt);m.calibration.fill_(-0.)
    with pytest.raises(Exception,match='buffer'):heads.verify_frozen(m,initial,opt)


@pytest.mark.parametrize('bad',[None,'shape','identity','optimizer_state'])
def test_strict_saved_initial_restoration_and_fresh_Adam(tmp_path,bad):
    source=Toy();heads.freeze(source);oldopt=torch.optim.Adam(heads.TrainableView(source).parameters(),lr=.001);arrays,rng=snapshot(source,oldopt)
    expected=pilot.obs.parameter_identity({n[10:]:v for n,v in arrays.items() if n.startswith('parameter/')})
    target=Toy();selected=heads.freeze(target);opt=torch.optim.Adam(list(selected.values()),lr=.001)
    if bad=='shape':arrays['parameter/'+next(iter(dict(target.named_parameters())))]=np.zeros(99,np.float32)
    if bad=='identity':expected='wrong'
    if bad=='optimizer_state':opt.state[next(iter(selected.values()))]={'step':torch.tensor(1.)}
    if bad:
        with pytest.raises(Exception):heads.restore_initial(target,opt,arrays,rng,expected)
    else:
        report=heads.restore_initial(target,opt,arrays,rng,expected);actual,metadata=snapshot(target,opt)
        assert report['exact'] and report['Adam_step']==0 and not opt.state and metadata==rng
        assert all(actual[n].tobytes()==v.tobytes() for n,v in arrays.items())
        x=torch.ones(1,3);assert torch.equal(source(x).physical,target(x).physical)


def test_observer_handles_expected_absent_trunk_gradients_and_connected_heads():
    m=Toy();heads.freeze(m);x=torch.ones(1,3)
    class FakeShader:
        def shade(self,pos,norm,view,mx,my,kd,metal,encoding):return mx+my+kd+metal+encoding.sum(-1,keepdim=True)
    with heads.HeadObserver(m) as observer:
        observer.begin();material=m(x);p=material.physical
        y=observer.wrap_shader(FakeShader()).shade(x,x,x,p[:,:1],p[:,1:2],p[:,3:6],p[:,6:7],material.encoding)
        y.square().mean().backward();arrays,summary=observer.collect()
        assert summary['frozen_activation_gradients']=='expected_absent' and len(summary['nodes'])==15
        assert all(np.isfinite(a).all() for a in arrays.values())
    assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
    assert all(not module._forward_hooks and len(module._forward_pre_hooks)<=1 for module in m.modules())


def test_initial_observation_and_probe_do_not_change_cached_gradients():
    m=Toy();heads.freeze(m);m(torch.ones(1,3)).physical.square().mean().backward()
    saved={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    pilot.method.capture(m,torch.tensor(pilot.method.TRAIN,dtype=torch.float32))
    assert all(torch.equal(p.grad,saved[n]) for n,p in m.named_parameters() if p.requires_grad)


def test_sampling_and_evaluation_adapters_are_exact_E8R_and_work_is_fixed():
    def core_funcs(module):
        tree=ast.parse(Path(module.__file__).read_text());c=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Core')
        return {n.name:ast.dump(n) for n in c.body if isinstance(n,ast.FunctionDef)}
    old=core_funcs(pilot.legacy);new=core_funcs(pilot)
    assert old['evaluate']==new['evaluate'] and old['probe']==new['probe'] and old['cost']==new['cost']
    assert pilot.tape_policy is pilot.legacy.tape_policy
    assert pilot.CAPS['neural_fit']+pilot.CAPS['neural_eval']==328 and pilot.CAPS['neural_reverse']==304 and pilot.CAPS['Adam_returns']==74
    assert (16*37+100)==692 and (16*12+42)==234
    w=pilot.Work();w.attempts['Adam_returns']=w.done['Adam_returns']=74;before=dict(w.attempts)
    with pytest.raises(Exception,match='cap'):w.attempt('Adam_returns')
    assert w.attempts==before
    with pytest.raises(Exception,match='cap'):w.attempt('physical')


def test_execution_is_gated_before_any_real_model_import(tmp_path):
    p=tmp_path/'review.json';p.write_text('{}\n')
    with pytest.raises(Exception,match='approval'):pilot.execute_reviewed(p,pilot.sha(p),'0'*64)
    assert '--execute' not in Path(pilot.__file__).read_text()
