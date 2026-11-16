"""E16 focused synthetic checks. No full native field, CUDA, BVH or real fitting."""
import contextlib
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from benchmarks.neural_observation_e16 import (teacher,world_Q,Q_loss,photo_loss,compatibility,compute_match,POLICY,LIMITS)
from experiments.bell_neural_material_e16 import Native
from infrastructure.stage_sources import ROOT


@pytest.fixture(autouse=True)
def CPU_only(monkeypatch):
    def forbidden():raise AssertionError('CUDA initialization forbidden in E16 synthetic tests')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)


def test_Q_equivalences_world_frame_and_differentiable_uniform_distillation():
    p=np.array([[.3,np.log(.3),np.log(2.)]])
    T=np.array([[1.,0.,0.]]);B=np.array([[0.,1.,0.]])
    Q=world_Q(p,T,B);period=p.copy();period[:,0]+=np.pi
    swap=p.copy();swap[:,0]+=np.pi/2;swap[:,2]*=-1
    np.testing.assert_allclose(Q,world_Q(period,T,B),atol=1e-16)
    np.testing.assert_allclose(Q,world_Q(swap,T,B),atol=1e-16)
    R=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    np.testing.assert_allclose(world_Q(p,T@R.T,B@R.T),R@Q@R.T,atol=1e-16)
    pred=torch.tensor(Q*1.1,requires_grad=True);target=torch.tensor(Q)
    loss=Q_loss(torch,pred,target);loss.backward();assert loss>0 and torch.isfinite(pred.grad).all()


def test_teacher_reads_no_truth_or_withheld_fields_and_keeps_all_bad_estimates():
    p=np.tile([0.,np.log(.5),np.log(2.)],(84,3,1));p[:,1,0]=1.;p[:,2,0]=-1.
    class Archive:
        def __getitem__(self,key):
            assert key in {'parameters','training_objective','selected_by_training'}
            return dict(parameters=p,training_objective=np.zeros((84,3)),selected_by_training=np.zeros(84,np.int64))[key]
    history=dict(history=[dict(normal_eigenvalues=np.ones((84,3,3)).tolist())])
    selected,Q,c=teacher(Archive(),history,np.tile([1.,0.,0.],(84,1)),np.tile([0.,1.,0.],(84,1)))
    np.testing.assert_array_equal(selected,p[:,0]);assert Q.shape==(84,3,3) and not c['weighting_used']
    assert (c['near_fit_starts']==3).all() and (c['relative_Q_disagreement']>0).all()


def test_known_fitting_RGB_objective_and_mechanical_compute_match():
    y=torch.ones((3,2,3),requires_grad=True);known=torch.zeros_like(y)
    photo_loss(torch,y,known).backward();assert y.grad.shape==y.shape
    with pytest.raises(ValueError):photo_loss(torch,y,torch.zeros((4,2,3)))
    assert compute_match(100,100,20,72,0)['continue_baseline']
    assert compute_match(200,100,20,72,0)['reached']
    assert compute_match(100,100,20,72,1000)['capped']


def test_compatibility_separates_bias_variance_precision_and_incomplete_panels():
    ref=np.ones((2,3))*.1;grid=np.zeros_like(ref);constant=np.broadcast_to(ref,(32,2,3)).copy()
    assert compatibility(constant,ref,grid,.001,.02)['passed']
    bad=compatibility(constant+.01,ref,grid,.001,.02)
    assert not bad['passed'] and bad['maximum_empirical_SE']<1e-16
    noisy=constant.copy();noisy[:16]+=1;noisy[16:]-=1
    assert not compatibility(noisy,ref,grid,.001,.02)['precision_adequate']
    with pytest.raises(ValueError):compatibility(constant[:31],ref,grid,.001,.02)


def counters(obj):
    obj.check=lambda:None
    obj.r=dict(attempted={k:0 for k in LIMITS if k not in ('wall_seconds','inner_seconds','grace_seconds',
        'cpu_threads','host_memory_gib','cuda_allocator_fraction','output_bytes')},compatibility={})
    obj.r['completed']=dict(obj.r['attempted'])


def test_real_corrected_method_adapter_with_synthetic_native_dependencies(monkeypatch):
    # Reuse the already reviewed Tiny fixture, not its prior test suite or N1.
    path=ROOT/'tests/scientific/test_native_material_a_e10.py'
    spec=importlib.util.spec_from_file_location('e16_tiny_native_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,_,_,_=module.fixture();shader=model.shader_network
    obj=Native.__new__(Native);obj.t=torch;counters(obj)
    obj.points=np.array([0,1]);obj.wo=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    obj.known=dict(positions=torch.zeros((2,3)),normals=torch.tensor([[0.,0.,1.]]*2),T=torch.tensor([[0.,1.,0.]]*2),
        B=torch.tensor([[-1.,0.,0.]]*2),color=torch.full((2,3),.3),metallic=torch.full((2,1),.2))
    actual_as_tensor=torch.as_tensor
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:actual_as_tensor(*a,**{k:('cpu' if k=='device' and v=='cuda' else v) for k,v in kw.items()}))
    real_fork=torch.random.fork_rng;monkeypatch.setattr(torch.random,'fork_rng',lambda **kw:real_fork(devices=[]))
    monkeypatch.setattr(torch.cuda,'manual_seed',lambda n:None);monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda n:None)
    p=torch.tensor([[.2,np.log(.3),np.log(2.)],[.4,np.log(.4),np.log(1.5)]],requires_grad=True)
    q=obj.explicit(shader,p,np.array([0,1]));out=obj.render(shader,q,np.array([0,1]),701,'gate',per_observation=True)
    assert out.shape==(2,2,3,3) and torch.isfinite(out).all()
    grad=torch.autograd.grad(out[:,:,0].sum(),p)[0];assert torch.isfinite(grad).all() and grad.abs().max()>0
    assert obj.r['completed']['gate_shader_calls']==2


@pytest.mark.parametrize('passed',[True,False])
def test_compatibility_reverse_counts_and_fail_closed_before_updates(tmp_path,monkeypatch,passed):
    import experiments.bell_neural_material_e16 as run
    obj=Native.__new__(Native);obj.t=torch;obj.out=tmp_path;counters(obj)
    valid=np.zeros((4,84),bool)
    for view,n in enumerate((84,36,40,60)):valid[view,:n]=True
    obj.views,obj.points=np.nonzero(valid);obj.fit=np.arange(137)
    obj.known=dict(normals=torch.tensor([[0.,0.,1.]]*84),positions=torch.zeros((84,3)),T=torch.tensor([[1.,0.,0.]]*84),B=torch.tensor([[0.,1.,0.]]*84))
    model=SimpleNamespace(compute_tangent_bitangent_flat=lambda n,_:(torch.tensor([[1.,0.,0.]]*len(n)),torch.tensor([[0.,1.,0.]]*len(n))))
    obj.models={(m,'A'):model for m in run.FIELDS}
    data=np.full((2,2,4,84,12,3,3),.01)
    obj.data=dict(teachers={m:dict(parameters=np.tile([.2,np.log(.3),np.log(2.)],(84,1))) for m in run.FIELDS},
        legacy=[None,None,None,None,None,{512:data,256:data}],compat_reference={m:dict(predicted_RGB=np.full((220,2,3),.03),coarse_RGB=np.full((220,2,3),.03)) for m in run.FIELDS})
    class Reference:
        def evaluate(self,p,jacobian,observations):return None,np.ones((6,1,2,3,3))
    obj.references={512:Reference(),256:Reference()}
    def fake_render(model,q,obs,seed,phase,per_observation=False):
        obj.bump('gate_shader_calls',2);obj.done('gate_shader_calls',2)
        scalar=q['mx']+q['my']+q['rotation'][:,:1];return scalar[:,None,None].expand(-1,2,3,3)
    obj.render=fake_render
    monkeypatch.setattr(run,'compatibility',lambda *a,**kw:dict(passed=passed,synthetic_only=True))
    actual_tensor=torch.tensor;monkeypatch.setattr(torch,'tensor',lambda *a,**kw:actual_tensor(*a,**{k:('cpu' if k=='device' and v=='cuda' else v) for k,v in kw.items()}))
    actual_as_tensor=torch.as_tensor;monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:actual_as_tensor(*a,**{k:('cpu' if k=='device' and v=='cuda' else v) for k,v in kw.items()}))
    if passed:obj.compatibility()
    else:
        with pytest.raises(ValueError,match='compatibility failed'):obj.compatibility()
        assert (tmp_path/'original-compatibility.npz').is_file() and not obj.r['compatibility']['original']['forward']['passed']
    expected=384 if passed else 192
    assert obj.r['attempted']['gate_reverse_sweeps']==obj.r['completed']['gate_reverse_sweeps']==expected
    assert obj.r['completed']['photometric_Adam_returns']==obj.r['completed']['distillation_Adam_returns']==0


def test_full_toy_field_update_uses_all_parameters_without_head_freezing(monkeypatch):
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.feats_network=torch.nn.Linear(3,4)
            for name,n in [('mx_predictor',1),('my_predictor',1),('rotation_predictor',2),('kd_predictor',3),('metallic_predictor',1),('inner_light',3),('outer_light',3)]:
                setattr(self,name,torch.nn.Linear(4,n))
        def values(self,x):
            f=torch.tanh(self.feats_network(x));return torch.cat([getattr(self,k)(f) for k in ('mx_predictor','my_predictor','rotation_predictor')],1)
    model=Field();obj=Native.__new__(Native);obj.t=torch;counters(obj)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    opt=obj.optimizer(model);assert len(opt.param_groups[0]['params'])==len(list(model.parameters()))
    opt.zero_grad(set_to_none=True);loss=(model.values(torch.ones((2,3)))-.2).square().mean()
    result=obj.update(model,opt,loss,'distillation')
    assert all(p.requires_grad for p in model.parameters()) and result['preclip_gradient_L2']>0
    assert model.feats_network.weight.grad is not None and model.kd_predictor.weight.grad is None
    assert obj.r['completed']['distillation_Adam_returns']==1


def test_cold_metadata_import_and_work_ceilings():
    code='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','network','train','raytracing','_raytracing','open3d','nvdiffrast'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.bell_neural_material_e16
'''
    subprocess.run([sys.executable,'-B','-c',code],check=True)
    assert LIMITS['shadow_rays']==(12000*137+128*137+192*220)*1536
    assert LIMITS['field_calls']==6000+400+12
    assert LIMITS['independent_point_directions']==(12*220+2*6)*(512**2+256**2)


def test_paired_training_lifecycle_resets_guided_Adam_and_bounds_extension(tmp_path,monkeypatch):
    import experiments.bell_neural_material_e16 as run
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.feats_network=torch.nn.Linear(3,4)
            for name,size in [('mx_predictor',1),('my_predictor',1),('rotation_predictor',2)]:setattr(self,name,torch.nn.Linear(4,size))
        def forward(self,x):
            f=torch.tanh(self.feats_network(x));z=torch.cat([getattr(self,k)(f) for k in ('mx_predictor','my_predictor','rotation_predictor')],1)
            return torch.stack((z[:,0],z[:,2],z[:,3],z[:,1]),1).reshape(-1,2,2)
    obj=Native.__new__(Native);obj.t=torch;obj.out=tmp_path;counters(obj);obj.r.update(snapshots=[],compute_comparisons={})
    obj.models={}
    for material in run.FIELDS:
        for arm in ('A','B'):
            torch.manual_seed(2);obj.models[material,arm]=Tiny()
    obj.inactive={k:{} for k in obj.models}
    obj.points=np.arange(84);obj.views=np.zeros(84,np.int64);obj.fit=obj.all=np.arange(84)
    obj.data=dict(known=dict(position=np.ones((84,3))),confidence={},teachers={m:dict(Q=np.tile(np.eye(2),(84,1,1))) for m in run.FIELDS},
        teacher_wall_charge=0.,legacy=[None,np.arange(84),None,None,None,{512:None}])
    obj.known=dict(positions=torch.ones((84,3)))
    def field(model):
        obj.bump('field_calls');q=model(torch.ones((84,3)));obj.done('field_calls');return dict(Q=q)
    obj.field=field
    def render(model,q,observations,seed,phase):return q['Q'].sum((1,2))[:,None,None,None].expand(-1,2,3,3)
    obj.render=render;obj.compatibility=lambda:None
    def snapshot(material,arm,stage,model,opt=None):
        q=field(model)['Q'].detach().numpy();mean=np.zeros((84,2,3))
        steps=sorted({int(v['step']) for v in opt.state.values()}) if opt is not None else []
        obj.r['snapshots'].append(dict(material=material,arm=arm,stage=stage,steps=steps));return dict(Q=q),mean
    obj.snapshot=snapshot
    monkeypatch.setitem(POLICY,'distillation_updates',2);monkeypatch.setitem(POLICY,'photometric_updates',3)
    monkeypatch.setattr(run.previous,'captured_target',lambda *a:np.zeros((84,2,3)))
    monkeypatch.setattr(run,'compute_match',lambda a,b,d,t,e:dict(continue_baseline=e<2,reached=e>=2,goal_seconds=b+d+t,baseline_active_seconds=a,capped=False))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    actual_as_tensor=torch.as_tensor
    monkeypatch.setattr(torch,'as_tensor',lambda *a,**kw:actual_as_tensor(*a,**{k:('cpu' if k=='device' and v=='cuda' else v) for k,v in kw.items()}))
    obj.run()
    assert obj.r['completed']['photometric_Adam_returns']==16 and obj.r['completed']['distillation_Adam_returns']==4
    assert len(obj.r['snapshots'])==12 and obj.r['inactive_parameters_verified_unchanged']
    assert obj.r['known_inputs_verified_unchanged']
    for row in obj.r['snapshots']:
        if row['stage']=='post_distillation':assert row['steps']==[2]
        if row['stage']=='photo1000':assert row['steps']==[3]
        if row['stage']=='compute_endpoint':assert row['steps']==[5]
