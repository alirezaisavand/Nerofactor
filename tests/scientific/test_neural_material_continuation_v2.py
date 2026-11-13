"""New continuation fixtures; no scientific shader or recovery execution."""
import copy
import inspect
import contextlib
from types import SimpleNamespace

import pytest
import torch

from experiments import continue_neural_material_logged_v2 as c
from experiments import neural_material_recovery_v1 as h


def test_resume_changes_only_four_declared_setup_and_range_sites():
    factory=lambda model: None
    loop=c.resume_loop(h,factory)
    assert loop.__globals__['optimizer_for'] is factory
    assert h.train_history.__globals__['optimizer_for'] is h.optimizer_for
    assert loop.__globals__['step'] is h.step
    assert loop.__globals__['backward'] is h.backward
    assert loop.__globals__['loss_value'] is h.loss_value
    assert 'range(1,UPDATES+1)' in inspect.getsource(h.train_history)
    assert len(c.REPLACEMENTS)==4


def test_numeric_checkpoint_restores_adam_exactly_and_next_update_matches(tmp_path):
    model=h.field.make_material('N0',h.fresh_latent(105),105);opt=h.optimizer_for(model)
    for _ in range(3):
        opt.zero_grad();model.latent.square().sum().backward();opt.step()
    (tmp_path/'fixture').mkdir()
    with h.OwnedOutput(tmp_path/'fixture','fixture') as output:
        sel=h.save_checkpoint(output,'state',model,opt,3)
        before=h.compact_state(model,opt)
        other=h.field.make_material('N0',h.fresh_latent(105),105);restored=h.optimizer_for(other)
        h.restore_checkpoint(c.ReadOnlyInput(output.root),sel,other,restored)
        assert h.compact_state(other,restored)==before
        for m,o in ((model,opt),(other,restored)):
            o.zero_grad();m.latent.square().sum().backward();o.step()
        assert h.compact_state(other,restored)==h.compact_state(model,opt)


def test_checkpoint_counter_carry_is_cross_checked_without_repeating_prefix():
    base=h.counts('N1')
    base.update(field_attempts=1003,field_completed=1003,shader_attempts=1001,shader_completed=1001,
        fit_forwards=1001,initial_forwards=1,attempted_points=96096,completed_points=96096,
        diffuse_directions=96096*512,specular_directions=96096*1024,
        reverse_attempts=1001,reverse_completed=1001,optimizer_attempts=1000,optimizer_returns=1000)
    after=copy.deepcopy(base)
    for k in ('field_attempts','field_completed','shader_attempts','shader_completed','fit_forwards'):after[k]+=11
    after['prefix_forwards']=1
    for k in ('reverse_attempts','reverse_completed','optimizer_attempts','optimizer_returns'):after[k]+=10
    for k,n in (('attempted_points',1056),('completed_points',1056),('diffuse_directions',1056*512),('specular_directions',1056*1024)):after[k]+=n
    old=dict(material='M1',seed=105,arm='N1',completed_updates=4270,status='optimizing',
        work=after,history=[dict(update=1000,work=base,active_wall_seconds=112.),
                           dict(update=1010,work=after,active_wall_seconds=114.)],snapshots={'250':{},'1000':{}})
    before=copy.deepcopy(old);row=c.resumed_row(old)
    assert old==before and row['completed_updates']==1000
    assert row['work']['prefix_forwards']==1 and row['work']['optimizer_returns']==1000
    assert [r['update'] for r in row['history']]==[1000]
    assert row['continuation']['historical_checkpoint_clock_upper_seconds']==114.
    old['history'][1]['work']['fit_forwards']+=1
    with pytest.raises(Exception,match='counter reconstruction'):c.resumed_row(old)


def test_remaining_execution_requires_separate_explicit_replay_authorization():
    with pytest.raises(Exception,match='authorization'):
        c.admit(dict(phase='wrong',authorized=False))


def test_resume_loop_executes_only_the_selected_suffix_with_original_adam():
    model=h.field.make_material('N0',h.fresh_latent(105),105);model.arm='N1'
    opt=h.optimizer_for(model);model.latent.square().sum().backward();opt.step()
    opt.state[model.latent]['step'].fill_(1000)  # Disposable synthetic counter fixture.
    loop=c.resume_loop(h,lambda m:opt);ns=loop.__globals__
    ns.update(UPDATES=1002,SNAPSHOTS=(),material_diagnostics=lambda m:{})
    def backward(m,loss,counts):
        loss.backward();return {'toy':{'squared_norm':float(m.latent.grad.square().sum())}}
    ns['backward']=backward
    calls=[]
    class Batch:
        def render(self,m,*args,**kwargs):
            calls.append(kwargs.get('phase','update'))
            return m.latent.view(1,7,1).expand(1,7,3),SimpleNamespace(physical=m.latent[None]),None
    work=h.counts('N1');work.update(optimizer_attempts=1000,optimizer_returns=1000)
    row=dict(status='optimizing',completed_updates=1000,work=work,history=[],snapshots={'5000':{}})
    clock=SimpleNamespace(wall=0.,cpu=0.,check=lambda *a:None,elapsed=lambda:0.,active=contextlib.nullcontext)
    loop(model,Batch(),None,torch.zeros((1,3)),row,clock,lambda:None,lambda *a:pytest.fail('unexpected snapshot'))
    assert row['completed_updates']==1002 and float(opt.state[model.latent]['step'])==1002
    assert calls==['update','update','terminal'] and row['status']=='optimized'
