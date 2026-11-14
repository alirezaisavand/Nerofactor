"""Only tiny engineering networks/Adam; original N1 and shader imports are forbidden."""
import copy
from contextlib import contextmanager
import importlib.abc
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

from experiments import neural_material_continuation_e3 as c
from infrastructure import neural_material_observer_e2a as o


@pytest.fixture(autouse=True)
def no_science_and_preserved_process_rng():
    forbidden=('experiments.neural_material_field_v1','experiments.material_shader_v1','experiments.neural_material_recovery_v1')
    assert not any(m.startswith(forbidden) for m in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*args):
            if fullname.startswith(forbidden):raise AssertionError('Scientific module forbidden: '+fullname)
    block=Block();sys.meta_path.insert(0,block);t=torch.get_rng_state().clone();n=np.random.get_state()
    try:yield
    finally:
        torch.set_rng_state(t);np.random.set_state(n);sys.meta_path.remove(block)
        assert not any(m.startswith(forbidden) for m in sys.modules)


class Head(nn.Module):
    def __init__(self,width):super().__init__();self.predictor=nn.Sequential(nn.Linear(3,width),nn.Sigmoid())
    def forward(self,x):return self.predictor(x)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7);self.feats_network=nn.Sequential(nn.Linear(2,3),nn.ReLU())
            for n,width in zip(o.HEADS,(1,1,1,2,3)):setattr(self,n,Head(width))
        self.register_buffer('calibration',torch.tensor([.1,.2],dtype=torch.float32))
    def forward(self,shader):
        x=self.feats_network(self.calibration[None]);q={n:getattr(self,n)(x) for n in o.HEADS}
        enc=q['rotation_predictor']*2-1
        physical=torch.cat((q['mx_predictor'],q['my_predictor'],enc[:,:1],q['kd_predictor'],q['metallic_predictor']),1)
        value=shader.shade(None,None,None,q['mx_predictor'],q['my_predictor'],q['kd_predictor'],q['metallic_predictor'],enc)
        return value,physical


class Shader:
    def shade(self,p,n,v,mx,my,kd,metal,encoding):return kd+mx+my+metal+.2*encoding[:,:1]+.3*encoding[:,1:]


class Core:
    scientific=False
    def __init__(self):
        self.model=Tiny();self.opt=torch.optim.Adam(self.model.parameters(),lr=.001);self.shader=Shader()
        self.calls=dict(forwards=0,backwards=0,steps=0,evaluation=0);self.counter=10
    def forward(self,shader,terminal):self.calls['forwards']+=1;return self.model(shader)
    def loss(self,v):return (v*v).mean()
    def backward(self,loss):
        self.calls['backwards']+=1;loss.backward();groups={}
        for name,p in self.model.named_parameters():
            g=groups.setdefault(name.split('.')[0],dict(squared_norm=0.,nonzero=0,parameters=0,finite=True,missing=0))
            g['squared_norm']+=float(torch.sum(p.grad.double()**2));g['nonzero']+=int(torch.count_nonzero(p.grad));g['parameters']+=p.numel()
        return groups
    def physical(self,m):return dict(physical=m.detach()[0].tolist())
    def step(self):self.opt.step();self.counter+=1;self.calls['steps']+=1
    def native_work(self):return dict(optimizer_returns=self.counter)
    def evaluate(self):
        self.calls['evaluation']+=1
        with torch.no_grad():return dict(prediction=self.model(self.shader)[0].tolist())


def checkpoint():
    # Ten TOY engineering steps establish a reference state; no scientific replay.
    core=Core()
    for _ in range(10):
        core.opt.zero_grad();v,_=core.model(core.shader);core.loss(v).backward();core.opt.step()
    arrays,rng=c.prior.state_arrays(core.model,core.opt)
    core.opt.zero_grad();v,m=core.model(core.shader);loss=core.loss(v);g=core.backward(loss)
    expected=dict(loss=float(loss.detach()),original_gradient_norm=sum(x['squared_norm'] for x in g.values())**.5,physical=m.detach()[0].tolist())
    return core,arrays,rng,expected


def restored():
    reference,arrays,rng,expected=checkpoint();core=Core()
    core.restoration=c.restore_state(core.model,core.opt,arrays,rng,c.options(reference.opt),o.parameter_identity({n[10:]:a for n,a in arrays.items() if n.startswith('parameter/')}))
    return reference,core,arrays,rng,expected


def test_exact_state_and_next_original_adam_update():
    reference,core,arrays,rng,_=restored()
    actual,meta=c.prior.state_arrays(core.model,core.opt)
    assert meta==rng and all(actual[k].tobytes()==v.tobytes() for k,v in arrays.items())
    core.opt.zero_grad();v,_=core.model(core.shader);core.loss(v).backward()
    reference.opt.step();core.opt.step()
    a,ar=c.prior.state_arrays(reference.model,reference.opt);b,br=c.prior.state_arrays(core.model,core.opt)
    assert ar==br and all(a[k].tobytes()==b[k].tobytes() for k in a)
    assert all(float(v['step'])==11 for v in core.opt.state.values())


@pytest.mark.parametrize('bad',['counter','shape','dtype','negative_second','missing','RNG','options','identity'])
def test_bad_restoration_is_rejected_before_parameter_mutation(bad):
    ref,arrays,rng,_=checkpoint();arrays={k:v.copy() for k,v in arrays.items()};core=Core();before=o.parameter_identity(o.parameter_arrays(core.model));expected=c.options(ref.opt);identity=None
    key=next(k for k in arrays if k.startswith('parameter/'));moment=next(k for k in arrays if k.endswith('/exp_avg_sq'))
    if bad=='counter':arrays[next(k for k in arrays if k.endswith('/step'))][...]=9
    if bad=='shape':arrays[key]=arrays[key].ravel()[:1]
    if bad=='dtype':arrays[key]=arrays[key].astype(np.float64)
    if bad=='negative_second':arrays[moment].flat[0]=-1
    if bad=='missing':arrays.pop(key)
    if bad=='RNG':arrays['torch_CPU_RNG']=np.zeros(1,np.uint8)
    if bad=='options':expected=dict(expected,lr=.01)
    if bad=='identity':identity='0'*64
    with pytest.raises(Exception):c.restore_state(core.model,core.opt,arrays,rng,expected,identity)
    assert before==o.parameter_identity(o.parameter_arrays(core.model)) and not core.opt.state


def test_saved_rng_context_restores_caller_after_interruption():
    _,arrays,rng,_=checkpoint();torch.manual_seed(42);np.random.seed(42)
    before_t=torch.get_rng_state().clone();before_n=np.random.get_state()
    with pytest.raises(ValueError):
        with c.saved_rng_context(arrays,rng):
            assert torch.get_rng_state().numpy().tobytes()==arrays['torch_CPU_RNG'].tobytes()
            raise ValueError('engineering interruption')
    assert torch.equal(before_t,torch.get_rng_state()) and np.array_equal(before_n[1],np.random.get_state()[1]) and before_n[2:]==np.random.get_state()[2:]


def test_bounded90_updates_use_cached_gate_and_correct_states(tmp_path):
    _,core,_,_,expected=restored();store=o.Store(tmp_path/'engineering',tmp_path)
    c.prepare_arm(core,store,'toy',expected)
    out=c.continue_arm(core,store,'toy',{})
    assert out['status']=='complete' and out['completed_additional_updates']==90 and out['work']==c.CAPS
    assert core.calls==dict(forwards=91,backwards=91,steps=90,evaluation=1)
    assert core.counter==100 and set(out['checkpoints'])=={'20','50','100'}
    lines=(store.root/'toy/trajectory.jsonl').read_bytes().splitlines(keepends=True)
    assert len(lines)==91;previous=None
    for state,line in enumerate(lines,10):
        row=json.loads(line);assert row['state']==state and row['post_state']==min(state+1,100)
        assert row['previous_sha256']==previous and row['Adam_counter']==min(state+1,100)
        previous=c.hashlib.sha256(line).hexdigest()
    assert core.calls['forwards']+core.calls['evaluation']==92


def test_failed_gate_allows_no_updates_and_keeps_actual_measurement(tmp_path):
    _,core,_,_,expected=restored();store=o.Store(tmp_path/'engineering',tmp_path)
    with pytest.raises(Exception,match='parity'):c.prepare_arm(core,store,'toy',dict(expected,loss=-1))
    assert core.calls['steps']==0 and core.counter==10
    report=json.loads((store.root/'toy/state10-parity.json').read_text())
    assert not report['passed'] and report['actual']['loss']==expected['loss']


def test_rejected_work_attempts_do_not_increment():
    w=c.Work();w.value['training_forwards']=90;before=dict(w.value)
    with pytest.raises(Exception):w.attempt('forward','training')
    assert w.value==before
    w=c.Work()
    for _ in range(90):
        w.attempt('forward','training');w.done('forward');w.attempt('backward');w.done('backward');w.attempt('optimizer');w.done('optimizer')
    with pytest.raises(Exception):w.attempt('optimizer')
    w.attempt('forward','terminal');w.done('forward');w.attempt('backward');w.done('backward');w.attempt('forward','external');w.done('forward')
    assert w.value==c.CAPS


def test_interruption_keeps_state13_and_removes_observation_hooks(tmp_path):
    _,core,_,_,expected=restored();store=o.Store(tmp_path/'engineering',tmp_path);c.prepare_arm(core,store,'toy',expected)
    def check(reserve=0):
        if core.counter==13:raise c.prior.Censored('engineering interruption')
    with pytest.raises(c.prior.Censored):c.continue_arm(core,store,'toy',{},check)
    record=json.loads((store.root/'toy/arm-result.json').read_text())
    assert record['status']=='censored' and record['completed_additional_updates']==3 and record['native_work']['optimizer_returns']==13
    arrays=o.read_numeric(store.root,record['failure_state']['selection'],allow_nonfinite=True)
    assert all(float(v)==13 for n,v in arrays.items() if n.endswith('/step'))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.model.modules())


def test_execution_requires_separate_e3_approval_before_scientific_imports():
    with pytest.raises(Exception,match='approval'):c.validate_review(dict(phase='E.3',scientific_execution_authorized=False),'0'*64)
    assert 'experiments.neural_material_field_v1' not in sys.modules


def test_accepted_numeric_inputs_have_complete_state_and_known_provenance():
    old,arms,ids=c.accepted_inputs()
    assert len(arms)==10 and len(ids)==35 and [(x['material'],x['seed'],x['lr']) for x in arms]==[(m,s,lr) for m,s in c.CASES for lr in c.RATES]
    assert all(x['parameter_identity'] for x in arms) and old['runtime']['torch']=='2.4.0+cu121'
