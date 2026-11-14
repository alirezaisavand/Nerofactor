"""Only numerical diagnostics and tiny synthetic restoration/continuation fixtures."""
import ast
import json
from types import FunctionType,SimpleNamespace

import numpy as np
import pytest
import torch

import test_neural_material_continuation_e3 as fixtures
from experiments import neural_material_recovery_e3r_v1 as c
from infrastructure import material_diagnostics_e3r_v1 as d

no_science_and_preserved_process_rng=fixtures.no_science_and_preserved_process_rng


def legacy_functions():
    ns=dict(np=np,require=c.require,PROTOCOL='known-material-recovery-metrics-v1',ANISOTROPY_GAP=.1)
    nodes=[d.source_node('experiments/multi_material_reference_v1.py','validate'),
        d.source_node('experiments/material_recovery_metrics_v1.py','roughness_tensor'),
        d.source_node('experiments/material_recovery_metrics_v1.py','gauge_metrics'),
        d.source_node('experiments/multi_material_recovery_v1.py','recovery_errors')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__,'exec'),ns)
    ns['reference']=SimpleNamespace(validate=ns['validate'])
    return ns


def material(dtype=np.float32):return np.array([.25,.5,.3,.2,.4,.6,.3],dtype)


def test_exact_float32_floor_reproduces_both_old_failures_without_sigmoid_endpoint():
    p=material();q=np.float32(1e-9);p[0]=np.float32(.96)*q+np.float32(.04)
    before=p.tobytes();assert q>0 and p[0]==np.float32(.04) and float(p[0])<.04
    legacy=legacy_functions()
    with pytest.raises(Exception,match='domain'):legacy['validate'](p)
    with pytest.raises(Exception,match='widths'):legacy['roughness_tensor'](*p[:3].astype(np.float64))
    result=d.recovery_errors(p,material(np.float64))
    assert result['Q_relative_frobenius']>0 and p.tobytes()==before
    assert d.validate_material(p)[0]==float(np.float32(.04))


@pytest.mark.parametrize('kind',['below_floor32','below_floor64','above_one32','albedo_negative','metal_above_one','nan','inf','shape','integer'])
def test_invalid_domain_still_fails_without_changing_values(kind):
    p=material()
    if kind=='below_floor32':p[0]=np.nextafter(np.float32(.04),np.float32(-np.inf))
    if kind=='below_floor64':p=p.astype(np.float64);p[0]=np.nextafter(.04,-np.inf)
    if kind=='above_one32':p[1]=np.nextafter(np.float32(1),np.float32(np.inf))
    if kind=='albedo_negative':p[3]=-np.finfo(np.float32).tiny
    if kind=='metal_above_one':p[6]=1.01
    if kind=='nan':p[0]=np.nan
    if kind=='inf':p[2]=np.inf
    if kind=='shape':p=p[:6]
    if kind=='integer':p=np.ones(7,dtype=np.int32)
    before=p.tobytes()
    with pytest.raises(Exception):d.recovery_errors(p,material(np.float64))
    assert p.tobytes()==before


def test_float64_origin_is_not_silently_reinterpreted_as_float32():
    p=material(np.float64);p[0]=d.FLOAT32_FLOOR
    with pytest.raises(Exception,match='representation'):d.validate_material(p)


def test_interior_metric_formulas_and_success_inputs_match_exactly():
    old=legacy_functions()
    for dtype in (np.float32,np.float64):
        for widths in ((.25,.5),(.5,.25),(.3,.3),(.04,1.)):
            p=material(dtype);p[:2]=widths
            if dtype==np.float32 and widths[0]==.04:continue  # Explicit newly admitted case, not legacy parity.
            assert d.recovery_errors(p,material(np.float64))==old['recovery_errors'](p,material(np.float64))


def test_evaluation_adapter_keeps_code_and_original_namespace():
    old=legacy_functions();previous=SimpleNamespace(recovery_errors=old['recovery_errors'])
    ns=dict(previous=previous)
    exec('def evaluate(p,t): return previous.recovery_errors(p,t)',ns)
    fn=ns['evaluate'];new=d.evaluation_adapter(fn)
    p=material();p[0]=np.float32(.04)
    assert new.__code__ is fn.__code__ and fn.__globals__['previous'] is previous
    assert new(p,material(np.float64))==d.recovery_errors(p,material(np.float64))
    with pytest.raises(Exception):fn(p,material(np.float64))


class Core(fixtures.Core):
    """Tiny engineering network; synthetic metadata for the state-21 floor gate."""
    def raw_physical(self,m):return self.physical(m)['physical']
    def physical(self,m):
        p=super().physical(m)
        if self.start==21:p['physical'][0]=float(np.float32(.04))
        return p


def restored(start):
    reference,_,_,_=fixtures.checkpoint()
    for _ in range(start-10):
        reference.opt.zero_grad();v,_=reference.model(reference.shader);reference.loss(v).backward();reference.opt.step()
    arrays,rng=c.old.prior.state_arrays(reference.model,reference.opt)
    core=Core();core.start=start;core.counter=start;core.work=c.Work(start)
    identity=c.obs.parameter_identity({n[10:]:v for n,v in arrays.items() if n.startswith('parameter/')})
    core.restoration=c.restore_state(core.model,core.opt,arrays,rng,c.old.options(reference.opt),identity,start)
    return reference,core,arrays,rng


@pytest.mark.parametrize('start',[10,21])
def test_exact_restore_and_next_adam_counter_without_restart(start):
    reference,core,arrays,rng=restored(start)
    actual,meta=c.old.prior.state_arrays(core.model,core.opt)
    assert meta==rng and all(c.obs.array_record(actual[n])==c.obs.array_record(v) for n,v in arrays.items())
    for obj in (reference,core):obj.opt.zero_grad();v,_=obj.model(obj.shader);obj.loss(v).backward();obj.opt.step()
    a,ar=c.old.prior.state_arrays(reference.model,reference.opt);b,br=c.old.prior.state_arrays(core.model,core.opt)
    assert ar==br and all(a[n].tobytes()==b[n].tobytes() for n in a)
    assert all(float(v)==start+1 for n,v in b.items() if n.endswith('/step')) and c.old.START==10


def test_wrong_counter_is_rejected_before_mutating_model():
    _,core,arrays,rng=restored(21);fresh=Core();before=c.obs.parameter_identity(c.obs.parameter_arrays(fresh.model))
    with pytest.raises(Exception):c.restore_state(fresh.model,fresh.opt,arrays,rng,c.old.options(core.opt),None,10)
    assert c.obs.parameter_identity(c.obs.parameter_arrays(fresh.model))==before and not fresh.opt.state


@pytest.mark.parametrize('start',[10,21])
def test_only_remaining_updates_and_cached_measurement_are_used(start,tmp_path):
    _,core,_,_=restored(start);store=c.obs.Store(tmp_path/'toy',tmp_path)
    # Synthetic state-10 expected measurement, never a real model/shader query.
    core.opt.zero_grad();v,m=core.model(core.shader);loss=core.loss(v);groups=core.backward(loss)
    expected=dict(loss=float(loss.detach()),original_gradient_norm=sum(g['squared_norm'] for g in groups.values())**.5,physical=core.physical(m)['physical'])
    core.calls={k:0 for k in core.calls}
    c.prepare_arm(core,store,'arm',expected if start==10 else None)
    result=c.continue_arm(core,store,'arm',{})
    assert result['status']=='complete' and result['completed_additional_updates']==100-start and result['work']==c.caps(start)
    assert core.calls==dict(forwards=101-start,backwards=101-start,steps=100-start,evaluation=1)
    assert core.counter==100 and set(result['checkpoints'])=={'50','100'}
    rows=[json.loads(x) for x in (store.root/'arm/trajectory.jsonl').read_text().splitlines()]
    assert [r['state'] for r in rows]==list(range(start,101)) and rows[0]['Adam_counter']==start+1
    assert rows[-1]['Adam_counter']==100 and rows[-1]['post_state']==100


def test_rejected_phase_attempt_is_atomic_and_work_budget_exact():
    w=c.Work(21);w.value['training_forwards']=79;before=dict(w.value)
    with pytest.raises(Exception):w.attempt('forward','training')
    assert w.value==before
    assert sum(c.caps(s)['successful_updates'] for _,_,_,s in c.MATRIX)==349
    assert sum(c.caps(s)['forwards'] for _,_,_,s in c.MATRIX)==357
    assert sum(c.caps(s)['backwards'] for _,_,_,s in c.MATRIX)==353


def test_interruption_keeps_partial_state_and_cleans_hooks_and_rng(tmp_path):
    _,core,_,_=restored(21);store=c.obs.Store(tmp_path/'toy',tmp_path);c.prepare_arm(core,store,'arm',None)
    t=torch.get_rng_state().clone();n=np.random.get_state()
    def check(reserve=0):
        if core.counter==23:raise c.old.prior.Censored('Synthetic interruption')
    with pytest.raises(c.old.prior.Censored):c.continue_arm(core,store,'arm',{},check)
    report=json.loads((store.root/'arm/arm-result.json').read_text());a=c.obs.read_numeric(store.root,report['failure_state']['selection'])
    assert report['completed_additional_updates']==2 and all(float(v)==23 for k,v in a.items() if k.endswith('/step'))
    assert torch.equal(t,torch.get_rng_state()) and np.array_equal(n[1],np.random.get_state()[1])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.model.modules())


def test_recovery_execution_is_not_authorized_by_preparation():
    with pytest.raises(Exception,match='approval'):c.validate_review(dict(phase='E.3R',scientific_execution_authorized=False),'0'*64)
