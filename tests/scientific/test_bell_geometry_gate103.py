"""CPU-only new selection, gradient, lifecycle and work-bound checks."""
import ast
import functools
import inspect
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from benchmarks import bell_geometry_gate103 as p
from experiments.bell_geometry_gate103 import Calls,cuda_factories
from experiments.donut_reconstructed_geometry_v1 import save
from infrastructure.signatures import SignatureError


def original_class():
    tree=ast.parse((p.ROOT/'scientific_sources/geometry_g/network/renderer.py').read_text())
    return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeROShapeRenderer')


def test_actual_original_mask_gradient_and_additive_candidate():
    fun=next(n for n in original_class().body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
    def target_name(n):
        if not isinstance(n,ast.Assign) or not isinstance(n.targets[0],ast.Subscript):return None
        value=n.targets[0].slice
        if isinstance(value,ast.Index):value=value.value
        return value.value if isinstance(value,ast.Constant) else None
    statement=next(n for n in fun.body if target_name(n)=='loss_mask')
    m=torch.tensor([1.,1.,0.,0.],dtype=torch.float64)
    a=torch.tensor([.2,.8,.2,.8],dtype=torch.float64,requires_grad=True)
    scope=dict(F=torch.nn.functional,train_ray_batch={'masks':m},outputs={'fg_acc':a})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[statement],type_ignores=[])),'original-mask-expression','exec'),scope)
    old=scope['outputs']['loss_mask'];go=torch.autograd.grad(old,a,retain_graph=True)[0]
    new=old+((1-m)*a).mean();gn=torch.autograd.grad(new,a)[0]
    assert torch.equal(go,torch.tensor([-.25,-.25,0.,0.],dtype=torch.float64))
    assert torch.equal(gn,torch.tensor([-.25,-.25,.25,.25],dtype=torch.float64))
    assert new.item()==pytest.approx(p.mask_terms(m.numpy(),a.detach().numpy())['candidate'])


def test_candidate_finite_differences_and_foreground_equivalence():
    m=np.array([1.,0.,1.,0.]);a=np.array([.3,.7,.8,.2]);eps=1e-6
    numeric=[]
    for i in range(4):
        offset=np.eye(4)[i]*eps
        numeric.append((p.mask_terms(m,a+offset)['candidate']-p.mask_terms(m,a-offset)['candidate'])/(2*eps))
    assert np.allclose(numeric,[-.25,.25,-.25,.25],atol=1e-9)
    for a in (np.array([0.,.5,1.,.99]),np.array([1.00001,.5,.99,.2])):
        assert p.mask_terms(np.ones(4),a)['candidate']==p.mask_terms(np.ones(4),a)['original']
    assert p.mask_terms(m,np.array([.3,.7,.8,.2]))['candidate']==pytest.approx(np.mean(m*(1-np.array([.3,.7,.8,.2]))+(1-m)*np.array([.3,.7,.8,.2])))


def test_actual_opacity_entropy_gradient_differentiates_both_arguments():
    fun=next(n for n in original_class().body if isinstance(n,ast.FunctionDef) and n.name=='binary_cross_entropy')
    scope=dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),'original-opacity','exec'),scope)
    a=torch.tensor([.2,.5,.8],dtype=torch.float64,requires_grad=True)
    value=scope['binary_cross_entropy'](None,a,a)
    grad=torch.autograd.grad(value,a)[0]
    assert torch.allclose(grad,torch.log((1-a.detach())/a.detach())/3,atol=1e-12)
    assert grad[0]>0 and grad[1]==0 and grad[2]<0
    a=torch.tensor([.00001,.99999],dtype=torch.float64,requires_grad=True);b=a.clamp(.001,.999)
    assert torch.equal(torch.autograd.grad(scope['binary_cross_entropy'](None,b,b),a)[0],torch.zeros_like(a))


def test_angular_panel_is_geometry_only_with_numeric_ties_and_no_rng_consumption():
    ids=['3','2','1','0'];poses=np.zeros((4,3,4))
    centers=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
    for i,c in enumerate(centers):poses[i,:3,:3]=np.eye(3);poses[i,:3,3]=-c
    before=np.random.get_state()
    assert p.angular_panel(ids,poses,count=3)==['3','2','0']
    after=np.random.get_state();assert all(np.array_equal(a,b) for a,b in zip(before,after))


def test_pixel_panel_has_fixed_c_order_strata_and_deduplicated_indices():
    hit=np.zeros((128,128),bool);hit[32:96,32:96]=True
    panel=p.pixel_panel(hit.ravel(order='C'))
    assert all(len(v)==64 for v in panel['strata'].values())
    assert panel['indices']==sorted(set(sum(panel['strata'].values(),[])))
    assert all(hit.ravel()[i] for i in panel['strata']['inner_band'])
    assert all(not hit.ravel()[i] for i in panel['strata']['outer_band'])
    empty=p.pixel_panel(np.zeros(128**2,bool))
    assert empty['strata']['foreground']==empty['strata']['inner_band']==empty['strata']['outer_band']==[]
    assert len(empty['strata']['background'])==64  # No silent replacements.


def test_matching_inventory_retains_original_split_and_all_four_steps():
    inv=p.read(p.ROOT/p.INVENTORY)
    assert len(inv['fitting_ids'])==122 and not set(inv['fitting_ids'])&set(p.policy.original.EVAL_IDS)
    assert len(inv['neural_panel'])==16 and set(inv['neural_panel'])<=set(inv['fitting_ids'])
    assert sum(len(v['indices']) for v in inv['neural_panel'].values())==4055
    assert set(inv['checkpoints'])=={'15000','30000','50000','300000'}


def test_precision_recall_and_missing_support_are_separate():
    r=p.silhouette(np.array([True,True,False,False]),np.array([True,False,True,True]))
    assert r==dict(tp=1,fp=2,fn=1,tn=0,iou=.25,precision=1/3,recall=.5)


def test_two_draw_opacity_summary_keeps_stable_errors_and_sampling_flips():
    panel=dict(indices=[10,11,12,13],strata=dict(foreground=[10,11],background=[12,13],inner_band=[11],outer_band=[12]))
    result=p.opacity_summary(panel,np.array([True,True,False,False]),np.array([True,False,True,False]),
        np.array([[.9,.1,.8,.2],[.8,.6,.7,.1]]))
    a=result['unique_panel']
    assert a['stable_false_positive']==1 and a['stable_false_negative']==0 and a['threshold_flips']==1
    assert a['mesh_false_positive']==a['mesh_false_negative']==1
    assert result['outer_band']['threshold_diagnostics']['0.5']['mesh_disagreements']==[0,0]


def test_actual_original_camera_prefix_matches_independent_calibrated_rays():
    names={'_construct_ray_batch','_process_ray_batch','near_far_from_sphere','get_human_coordinate_poses'}
    methods=[n for n in original_class().body if isinstance(n,ast.FunctionDef) and n.name in names]
    cls=ast.ClassDef(name='CameraPrefix',bases=[],keywords=[],body=methods,decorator_list=[])
    scope=dict(torch=torch,F=torch.nn.functional)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original-camera-prefix','exec'),scope)
    model=scope['CameraPrefix']();model.cfg={'fixed_camera':False}
    angle=.5;R=np.array([[1.,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],np.float32)
    center=np.array([.2,.3,3.],np.float32);pose=np.c_[R,-R@center]
    K=np.array([[100.,0,64],[0,100,64],[0,0,1]],np.float32)
    info=dict(imgs=torch.zeros(1,3,128,128),Ks=torch.from_numpy(K)[None],poses=torch.from_numpy(pose)[None])
    batch,poses,_,_,_=model._construct_ray_batch(info,is_train=False)
    o,d,near,far,human=model._process_ray_batch(batch,poses)
    from audits.bell_calibration_e11 import rays
    ro,rd=rays(pose,K,128,128)
    assert np.max(abs(o.numpy()-ro))<=2e-6 and np.max(abs(d.numpy()-rd))<=2e-6
    assert near.shape==far.shape==(128**2,1) and human.shape==(128**2,3,4)


def test_cuda_factory_context_restores_cpu_on_failure():
    events=[]
    fake=NS(set_default_tensor_type=events.append)
    with pytest.raises(RuntimeError):
        with cuda_factories(fake):
            assert events==['torch.cuda.FloatTensor']
            raise RuntimeError('fixture')
    assert events==['torch.cuda.FloatTensor','torch.FloatTensor']


def dummy_tools():
    def forward(self,inputs):pass
    def gradient(self,x):pass
    def render(self,*args):pass
    def appearance(self,*args):pass
    def intersect(*args):pass
    def load(*args):pass
    def save_fn(*args):pass
    def backward(*args):pass
    def step(*args):pass
    def restore(*args):pass
    def initialize(*args):pass
    def jit(*args):pass
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args,**kwargs):return fn(*args,**kwargs)
        return wrapper
    optimizer=NS(__init__=initialize,load_state_dict=restore)
    fake=NS(load=load,save=save_fn,Tensor=NS(backward=backward),
            optim=NS(Adam=NS(step=decorate(step)),Optimizer=optimizer),
            utils=NS(cpp_extension=NS(load=jit)))
    field=NS(SDFNetwork=NS(forward=forward,gradient=gradient),AppShadingNetwork=NS(forward=appearance),get_intersection=intersect)
    renderer=NS(NeROShapeRenderer=NS(render=render))
    return fake,field,renderer,decorate


def test_call_guard_rejects_load_and_sdf_outside_phase_and_enforces_caps():
    fake,field,renderer,_=dummy_tools();record={};guard=Calls(fake,field,renderer,record)
    frame=NS(f_code=guard.codes['load'],f_locals={'map_location':'cpu'})
    with pytest.raises(SignatureError):guard.observe(frame,'call',None)
    guard.phase='load';guard.observe(frame,'call',None);assert record['loads']==1
    guard.phase='grid';record['grid_SDF']=p.LIMITS['dense_SDF_points']
    frame=NS(f_code=guard.codes['sdf'],f_locals={'inputs':NS(numel=lambda:3)})
    with pytest.raises(SignatureError):guard.observe(frame,'call',None)


def test_optimizer_guard_uses_unwrapped_code_not_shared_no_grad_wrapper():
    fake,field,renderer,decorate=dummy_tools();guard=Calls(fake,field,renderer,{})
    @decorate
    def unrelated_initializer():pass
    wrapper=unrelated_initializer.__code__
    assert wrapper==fake.optim.Adam.step.__code__ and wrapper not in guard.forbidden
    guard.observe(NS(f_code=wrapper,f_locals={}),'call',None)
    with pytest.raises(SignatureError):
        guard.observe(NS(f_code=inspect.unwrap(fake.optim.Adam.step).__code__,f_locals={}),'call',None)


def test_new_npz_calls_match_real_existing_save_interface():
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate103.py').read_text())
    found=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save']
    assert len(found)==7
    for call in found:inspect.signature(save).bind(*call.args,**{kw.arg:kw.value for kw in call.keywords})


def test_work_ceiling_is_derived_from_original_sampling_and_no_updates():
    limits=p.LIMITS
    assert limits['audit_BVH_rays']+limits['fitting_BVH_rays']==8585216
    assert limits['CPU_ray_checks']==1408 and limits['dense_SDF_points']==402653184
    assert limits['neural_SDF_forward_points']==32768*(64+3*16+3*160)+3*16*2*2048*80
    assert limits['neural_gradient_points']==32768*2*160
    assert limits['optimizer_returns']==limits['checkpoint_writes']==limits['material_calls']==0


def test_launcher_uses_exact_freeze_and_only_the_four_checkpoint_mounts():
    text=(p.ROOT/'experiments/run_bell_geometry_gate103.sh').read_text()
    assert re.search(r'^freeze=(.+)$',text,re.M).group(1)==p.FREEZE
    assert 'src=$prior/train/checkpoints,dst=' not in text
    for ref in p.read(p.ROOT/p.INVENTORY)['checkpoints'].values():
        assert f"src=$prior/train/checkpoints/{ref['file']},dst=/inputs/checkpoints/{ref['file']},readonly" in text


def test_render_guard_accepts_only_original_training_forward_policy():
    fake,field,renderer,_=dummy_tools();r={};guard=Calls(fake,field,renderer,r);guard.phase='neural'
    values=dict(step=14999,is_train=True,is_nerf=False,perturb_overwrite=-1,cos_anneal_ratio=14999/50000)
    frame=NS(f_code=guard.codes['render'],f_locals=values)
    guard.observe(frame,'call',None);assert r['renders']==1
    values['is_train']=False
    with pytest.raises(SignatureError):guard.observe(frame,'call',None)


def test_neural_adapter_preserves_original_cpu_ray_pool_construction():
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate103.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fitting_views')
    call=next(n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and n.func.attr=='_construct_ray_batch')
    assert next(k.value.value for k in call.keywords if k.arg=='device')=='cpu'
    assert next(k.value.value for k in call.keywords if k.arg=='is_train') is False


def test_module_import_requires_no_scientific_runtime():
    code="""import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','network','dataset','train','raytracing','open3d','mcubes','trimesh','_raytracing'):
   raise RuntimeError('Scientific import '+name)
sys.meta_path.insert(0,Block())
import experiments.bell_geometry_gate103
"""
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=p.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_freeze_is_source_only_inside_container(tmp_path,monkeypatch):
    out=p.TRAIN.parent.parent/'bell-geometry-gate103-fixture'
    f=p.definition(out);path=tmp_path/'freeze.json';p.write(path,f)
    monkeypatch.setattr(p,'FREEZE',path)
    def no_host(*args):raise AssertionError('Host evidence access in native freeze validation')
    monkeypatch.setattr(p,'host_inputs',no_host);monkeypatch.setattr(p,'check',no_host)
    assert p.freeze(p.digest(path))==f
