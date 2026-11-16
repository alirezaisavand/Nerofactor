"""New E17-only CPU stand-ins; no real native model, shader, BVH or GPU."""
import copy
import json
import subprocess
import numpy as np
import pytest
import torch
from benchmarks.neural_retention_e17 import COEFFICIENT,POLICY,LIMITS,scale_evidence,objective,strict_restore
from experiments.bell_neural_retention_e17 import Retention
from experiments.analyze_bell_neural_material_e16 import state_hash


@pytest.fixture(autouse=True)
def CPU_only(monkeypatch):
    def forbidden():raise AssertionError('CUDA initialization forbidden in E17 synthetic tests')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])


def counts(obj):
    obj.check=lambda:None
    obj.r=dict(attempted={k:0 for k in LIMITS if k not in ('wall_seconds','inner_seconds','grace_seconds','cpu_threads','host_memory_gib','cuda_allocator_fraction','output_bytes')},
        restoration_admitted=True,snapshots=[],gradient_probes=[],timing={})
    obj.r['completed']=dict(obj.r['attempted'])


def test_coefficient_uses_only_fitting_and_teacher_drift():
    target=np.tile(np.eye(3),(84,1,1));captures={};logs={}
    class Guard:
        def __init__(self,Q):self.Q=Q
        def __getitem__(self,k):
            assert k in ('Q','teacher_Q')
            return self.Q if k=='Q' else target
    for material in ('original','stronger'):
        captures[material,'post_distillation']=Guard(target*1.01)
        captures[material,'photo1000']=Guard(target*(1+np.sqrt(.0101)))
        logs[material]=[dict(phase='photo',loss=1e-6) for _ in range(1000)]
    r=scale_evidence(captures,logs);assert r['nearest_decade_coefficient']==COEFFICIENT and not r['uses_GT_or_withheld']
    logs['original']=logs['original'][:999]
    with pytest.raises(ValueError):scale_evidence(captures,logs)


def test_uniform_retention_has_exact_declared_value_and_gradient():
    Q=torch.ones((84,3,3),requires_grad=True);teacher=torch.ones_like(Q)*2
    rgb=torch.ones((137,2,3),requires_grad=True);target=torch.zeros_like(rgb)
    loss,photo,retention=objective(torch,rgb,target,Q,teacher)
    assert float(retention)==pytest.approx(.25) and float(loss)==pytest.approx(1+COEFFICIENT*.25)
    loss.backward();assert torch.isfinite(Q.grad).all() and torch.isfinite(rgb.grad).all()
    np.testing.assert_allclose(Q.grad[0],Q.grad[-1])  # No point filtering or confidence weights.


@pytest.mark.parametrize('change',['keys','shape','dtype','hash'])
def test_strict_restoration_rejects_incompatibility(change):
    model=torch.nn.Linear(3,2);arrays={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    expected=state_hash(arrays)
    if change=='keys':arrays['extra']=np.array(0,np.float32)
    if change=='shape':arrays['weight']=arrays['weight'][:1]
    if change=='dtype':arrays['weight']=arrays['weight'].astype(np.float64)
    if change=='hash':expected='0'*64
    with pytest.raises(ValueError):strict_restore(torch,model,arrays,expected if change=='hash' else state_hash(arrays))


def test_exact_restoration_preserves_all_weights_and_keeps_Adam_fresh():
    model=torch.nn.Linear(3,2);source=copy.deepcopy(model);arrays={k:v.detach().numpy().copy() for k,v in source.state_dict().items()}
    with torch.no_grad():model.weight.add_(1)
    strict_restore(torch,model,arrays,state_hash(arrays));assert torch.equal(source.weight,model.weight)
    obj=Retention.__new__(Retention);obj.t=torch
    opt=obj.optimizer(model);assert not opt.state and opt.param_groups[0]['lr']==.001 and opt.param_groups[0]['eps']==1e-8


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.feats_network=torch.nn.Linear(3,4)
        for name,n in [('mx_predictor',1),('my_predictor',1),('rotation_predictor',1),('kd_predictor',3)]:setattr(self,name,torch.nn.Linear(4,n))
    def forward(self):
        f=torch.tanh(self.feats_network(torch.ones((84,3))))
        diagonal=torch.cat([getattr(self,k)(f) for k in ('mx_predictor','my_predictor','rotation_predictor')],1)+2
        return torch.diag_embed(diagonal)


def test_component_probes_do_not_change_combined_gradient_weights_or_RNG():
    model=Tiny();control=copy.deepcopy(model);obj=Retention.__new__(Retention);obj.t=torch;counts(obj)
    Q=model();teacher=torch.eye(3).expand(84,-1,-1);rgb=Q.sum((1,2))[:,None,None].expand(-1,2,3)
    loss,photo,retention=objective(torch,rgb,torch.zeros_like(rgb),Q,teacher);rng=torch.random.get_rng_state().clone()
    p=obj.component_probes(model,photo,retention);assert set(p['groups'])=={'feats_network','mx_predictor','my_predictor','rotation_predictor'}
    assert torch.equal(rng,torch.random.get_rng_state()) and all(p.grad is None for p in model.parameters())
    loss.backward();q=control();objective(torch,q.sum((1,2))[:,None,None].expand(-1,2,3),torch.zeros_like(rgb),q,teacher)[0].backward()
    for a,b in zip(model.parameters(),control.parameters()):
        if a.grad is None:assert b.grad is None
        else:assert torch.equal(a.grad,b.grad)
    assert obj.r['completed']['retention_reverse_sweeps']==2


def test_retention_lifecycle_only_trains_new_C_and_preserves_two_starts(tmp_path,monkeypatch):
    import experiments.bell_neural_retention_e17 as run
    obj=Retention.__new__(Retention);obj.t=torch;obj.out=tmp_path;counts(obj)
    obj.models={(m,'C'):Tiny() for m in ('original','stronger')};obj.inactive={m:{} for m in ('original','stronger')}
    obj.known={'positions':torch.ones((84,3))};obj.views=np.zeros(84,np.int64);obj.points=np.arange(84);obj.fit=obj.all=np.arange(84)
    obj.data=dict(known=dict(position=np.ones((84,3))),teachers={m:dict(Q=np.tile(np.eye(3),(84,1,1))) for m in ('original','stronger')},
        restoration={m:dict(initial=dict(Q=model().detach().numpy())) for (m,_),model in obj.models.items()},legacy=[None,None,None,None,None,{512:None}])
    def field(model):obj.bump('field_calls');q=model();obj.done('field_calls');return dict(Q=q)
    obj.field=field
    def render(model,q,obs,seed,phase):
        obj.bump('photo_shader_calls',2);obj.done('photo_shader_calls',2)
        assert seed in (800000,800001,800002)
        return q['Q'].sum((1,2))[:,None,None,None].expand(-1,2,3,3)
    obj.render=render
    def snapshot(material,arm,stage,model,opt):
        assert arm=='C' and {int(v['step']) for v in opt.state.values()}=={3}
        obj.r['snapshots'].append(dict(material=material,arm=arm))
    obj.snapshot=snapshot
    monkeypatch.setitem(POLICY,'photometric_updates',3);monkeypatch.setitem(POLICY,'gradient_probe_steps',[0,2])
    monkeypatch.setattr(run.prior.previous,'captured_target',lambda *a:np.zeros((84,2,3)))
    actual=torch.as_tensor
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:actual(*a,**{k:('cpu' if k=='device' and v=='cuda' else v) for k,v in kw.items()}))
    obj.run();assert obj.r['completed']['photometric_Adam_returns']==6 and obj.r['completed']['retention_reverse_sweeps']==8
    assert len(obj.r['snapshots'])==2 and obj.r['known_inputs_verified_unchanged']
    for material in ('original','stronger'):
        rows=json.loads((tmp_path/(material+'-C-training.json')).read_text());assert len(rows)==3 and rows[0]['initial_Q_drift']==0
    obj.r['attempted']['photo_shader_calls']=LIMITS['photo_shader_calls'];before=dict(obj.r['attempted'])
    with pytest.raises(ValueError):obj.bump('photo_shader_calls')
    assert obj.r['attempted']==before


def test_cold_import_and_work_limits():
    code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','network','train','dataset','raytracing','_raytracing','open3d','nvdiffrast'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.bell_neural_retention_e17
'''
    subprocess.run(['python3','-B','-c',code],check=True)
    assert LIMITS['shadow_rays']==(4000*137+32*220)*1536
    assert LIMITS['independent_point_directions']==2*220*(512**2+256**2)
    assert LIMITS['field_calls']==2000+2+2 and LIMITS['retention_reverse_sweeps']==2*2*2
