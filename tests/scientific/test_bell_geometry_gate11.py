"""Focused O1 algebra/dispatch and image-only trimap checks; no native models."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import inspect
import subprocess
import sys
import numpy as np
import pytest
import torch

from benchmarks import bell_geometry_gate11 as g
from benchmarks.bell_mask_trimap_gate11 import trimap
from scientific_variants.geometry_background_opacity_v1 import renderer_class,select,VARIANT
from experiments.donut_reconstructed_geometry_v1 import save
from infrastructure.signatures import SignatureError


def parent_class():
    tree=ast.parse((g.ROOT/'scientific_sources/geometry_g/network/renderer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeROShapeRenderer')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('train_step','_shuffle_train_batch')]
    shell=ast.ClassDef(name='OriginalMethods',bases=[],keywords=[],decorator_list=[],body=selected)
    scope=dict(torch=torch,F=torch.nn.functional)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[shell],type_ignores=[])),'original-G-methods','exec'),scope)
    class Parent(scope['OriginalMethods'],torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.cfg={'train_ray_num':4,'use_refscores':False};self.is_nerf=False
            self.acc=torch.nn.Parameter(torch.tensor([.2,.8,.3,.7],dtype=torch.float64))
            self.train_batch_i=4;self.tbn=8;self.train_poses=torch.zeros(1,3,4)
            self.train_batch=dict(masks=torch.tensor([0.,0.,1.,1.,1.,0.,1.,0.],dtype=torch.float64),
                rgbs=torch.zeros(8,3,dtype=torch.float64))
        def _process_ray_batch(self,*args):return (None,)*5
        def get_anneal_val(self,step):return min(1.,step/50000)
        def render(self,*args,**kwargs):
            return dict(fg_acc=self.acc,ray_rgb=self.acc[:,None].expand(4,3),
                        fg_rgb=torch.ones(4,3),bg_rgb=torch.zeros(4,3))
        def compute_rgb_loss(self,a,b,unused=None):return (a-b).square().sum(-1)
    return Parent


def mask_loss_class():
    tree=ast.parse((g.ROOT/'scientific_sources/geometry_g/network/loss.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MaskLoss')
    scope=dict(Loss=object,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original-MaskLoss','exec'),scope)
    return scope['MaskLoss']


@pytest.mark.parametrize('step,weight',[(0,.3),(15000,.65),(30000,1.),(299999,1.)])
def test_original_train_step_and_schedule_plus_only_background(step,weight,monkeypatch):
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    parent=parent_class();events=[];derived=renderer_class(parent,lambda *values:events.append(values))
    a,b=parent(),derived();initial={k:v.clone() for k,v in a.state_dict().items()}
    torch.manual_seed(90);old=a.train_step(step);rng=torch.get_rng_state().clone()
    torch.manual_seed(90);new=b.train_step(step)
    assert torch.equal(rng,torch.get_rng_state()) and a.train_batch_i==b.train_batch_i==0
    assert all(torch.equal(a.train_batch[k],b.train_batch[k]) for k in a.train_batch)
    assert set(old)==set(new) and all(torch.equal(old[k],new[k]) for k in old if k!='loss_mask')
    # The pre-shuffle second half was [1,0,1,0]; use the same full-ray denominator.
    mask=torch.tensor([1.,0.,1.,0.],dtype=torch.float64)
    assert torch.equal(new['loss_mask'],old['loss_mask']+((1-mask)*b.acc).mean())
    loss=mask_loss_class()({})
    assert loss.get_mask_weight(step)==pytest.approx(weight)
    old_loss=loss(old,{},step)['loss_mask'];new_loss=loss(new,{},step)['loss_mask']
    go=torch.autograd.grad(old_loss,a.acc)[0];gn=torch.autograd.grad(new_loss,b.acc)[0]
    assert torch.allclose(gn-go,(1-mask)*weight/4,atol=1e-15)
    assert len(events)==1 and events[0][0]==step and all(not v.requires_grad for v in events[0][1:])
    assert set(initial)==set(b.state_dict()) and all(torch.equal(initial[k],v) for k,v in b.state_dict().items())
    assert b.geometry_variant==VARIANT


@pytest.mark.parametrize('failure',[False,True])
def test_dispatch_selection_is_explicit_and_restored(failure):
    parent=parent_class();module=NS(NeROShapeRenderer=parent,name2renderer={'shape':parent})
    def exercise():
        with select(module) as derivative:
            assert module.name2renderer['shape'] is derivative and issubclass(derivative,parent)
            if failure:raise RuntimeError('fixture')
    if failure:
        with pytest.raises(RuntimeError):exercise()
    else:exercise()
    assert module.name2renderer['shape'] is parent


def test_exact_foreground_subgradient_at_opacity_one_is_retained(monkeypatch):
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    parent=parent_class();a,b=parent(),renderer_class(parent)()
    with torch.no_grad():a.acc.fill_(1.);b.acc.fill_(1.)
    torch.manual_seed(3);old=a.train_step(0);torch.manual_seed(3);new=b.train_step(0)
    go=torch.autograd.grad(old['loss_mask'],a.acc)[0];gn=torch.autograd.grad(new['loss_mask'],b.acc)[0]
    assert torch.equal(go,torch.zeros_like(go)) and torch.equal(gn,torch.tensor([0.,.25,0.,.25],dtype=torch.float64))


def test_trimap_preserves_fg_and_does_not_label_disagreement_background():
    scores=np.full((4,32,32),-2.,np.float32);scores[:,12:20,12:20]=2.
    scores[2,5,5]=2.;scores[0,0,31]=.5
    original=scores.copy();labels=trimap(scores,np.ones(4))
    assert np.array_equal(labels['foreground'],scores[0]>1) and np.array_equal(scores,original)
    assert labels['background'][31,0] and labels['uncertain'][5,5]
    assert labels['uncertain'][9,12] and not labels['background'][9,12]
    assert not (labels['foreground']&labels['background']).any()
    assert np.all(labels['foreground']|labels['background']|labels['uncertain'])


def test_missing_object_abstains_and_nonfinite_scores_fail():
    scores=np.full((4,16,16),-2.,np.float32);scores[:,7:9,7:9]=2.
    labels=trimap(scores,np.array([1.,1.,-1.,1.]))
    assert not labels['background'].any() and labels['foreground'].sum()==4
    scores[1,0,0]=np.nan
    with pytest.raises(SignatureError):trimap(scores,np.ones(4))


def test_evaluation_preserves_accepted_camera_extraction_and_call_guard_methods():
    old=ast.parse((g.ROOT/'experiments/bell_geometry_gate103_v2.py').read_text())
    new=ast.parse((g.ROOT/'experiments/bell_geometry_gate11_evaluate.py').read_text())
    for name in ('cuda_factories','camera_prefix_cpu','ray_error','Calls','dense_field','candidate'):
        before=next(n for n in old.body if getattr(n,'name',None)==name)
        after=next(n for n in new.body if getattr(n,'name',None)==name)
        assert ast.dump(before)==ast.dump(after),name
    for node in ast.walk(new):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='save':
            inspect.signature(save).bind(*node.args,**{k.arg:k.value for k in node.keywords})
    assert 'camera_precision_witness(' not in (g.ROOT/'experiments/bell_geometry_gate11_evaluate.py').read_text()


def test_envelope_and_camera_split_inherit_accepted_definitions():
    f=g.read(g.ROOT/g.INVENTORY)['original_freeze']
    assert g.STEP==300000 and g.SEED==6033 and g.TRAIN_LIMITS['train_sample_slots']==24576000000
    assert g.TRAIN_RESOURCES==f['resources'] and g.LIMITS['MC_calls']==4
    assert g.LIMITS['dense_SDF_points']==536870912
    assert g.LIMITS['audit_BVH_rays']+g.LIMITS['fitting_BVH_rays']==8781824
    assert g.LIMITS['CPU_ray_checks']==1792 and g.LIMITS['checkpoint_loads']==4
    definition=g.read(g.ROOT/'docs/results/publication_gate11/realistic-mask-definition.json')
    assert set(definition['frame_to_camera'])==set(f['splits']['train'])
    assert not set(definition['frame_to_camera'])&set(definition['excluded_validation_and_held'])
    assert definition['logits']==dict(foreground_gt=1.,background_lt=-1.,object_presence_gt=0.,calibrated_probabilities=False)
    assert set(definition['prompts'])=={'P1','P2'}


def test_entry_import_constructs_no_model_or_native_runtime():
    code="""import sys,importlib.abc
class Stop(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','network','dataset','train','open3d','raytracing','mcubes','_raytracing'):
   raise RuntimeError(name)
sys.meta_path.insert(0,Stop())
import experiments.bell_geometry_gate11
"""
    done=subprocess.run([sys.executable,'-B','-c',code],cwd=g.ROOT,text=True,capture_output=True)
    assert done.returncode==0,done.stderr


def test_launcher_uses_reviewed_runtime_mounts_and_foreground_phase_limits():
    text=(g.ROOT/'experiments/run_bell_geometry_gate11.sh').read_text()
    assert 'freeze='+g.FREEZE in text and 'entry=experiments/bell_geometry_gate11.py' in text
    assert 'launch train 129600 129540' in text and 'launch evaluate 7200 7140' in text
    assert 'dst=/inputs/jit,readonly' in text and 'dst=/inputs/mask.npy,readonly' in text
    assert '--env MAX_JOBS=4' in text and '--shm-size 4g' in text
    assert 'dst=/inputs/training-result,readonly' in text and 'dst=/inputs/oracle,readonly' in text


def test_freeze_definition_is_metadata_only_inside_container(tmp_path,monkeypatch):
    # All acquisition/checkpoint identities come from committed JSON, not host I/O.
    output=g.evolution.TRAIN.parent.parent/'bell-background-opacity-gate11-fixture'
    expected=g.definition(output);path=tmp_path/'freeze.json';g.write(path,expected)
    monkeypatch.setattr(g,'FREEZE',path)
    def denied(*args):raise AssertionError('Native freeze accessed host evidence')
    monkeypatch.setattr(g,'host_inputs',denied);monkeypatch.setattr(g,'check',denied)
    assert g.freeze(g.digest(path))==expected


@pytest.mark.parametrize('defect',[None,'legacy','initial','mask_count'])
def test_training_receipt_keeps_derivative_identity_and_owned_checkpoint_binding(tmp_path,defect):
    from experiments.bell_geometry_gate11 import context,completed_train
    import json
    f={'host_output':str(tmp_path),'image':'fixture','splits':{'train':['14'],'validation':['58']},
       'O0':{'initial_state_sha256':'i'*64}}
    args=NS(commit='c'*40,freeze_sha256='d'*64,authorization_sha256='e'*64)
    root=tmp_path/'train';(root/'checkpoints').mkdir(parents=True)
    retained={}
    for step in g.KEEP_STEPS:
        path=root/f'checkpoints/model-step{step}.pth';path.write_bytes(b'identity-only fixture')
        retained[str(step)]=g.item(path)
    path=root/'checkpoints/model.pth';path.write_bytes(b'owned endpoint identity')
    report=dict(**context(f,args),status='complete',cleanup_verified=True,cleanup_errors=[],phase='train',
        adam_returns=300000,validation_calls=300,periodic_saves=300,initial_state_sha256='i'*64,
        initial_parity_passed=True,held_out_seen_by_trainer=False,
        step_1000_gate={'passed':True,'own_cpu_roundtrip':True,'checkpoint':retained['1000'],
                        'validation':{'view':'58','checkpoint_step':1000}},
        last_validation={'view':'58','checkpoint_step':300000},
        loader_transport=[{'workers':8,'start_method':'spawn','dataset':'DummyDataset'}]*2,
        training_witnesses=[{'parameters_changed':False},{'parameters_changed':True}],
        mask_forward_calls=300000,mask_trace_rows=15006,retained_checkpoints=retained,
        endpoint_checkpoint=g.item(path),artifacts={})
    if defect=='legacy':report['variant']='geometry-G'
    if defect=='initial':report['initial_state_sha256']='wrong'
    if defect=='mask_count':report['mask_forward_calls']-=1
    (root/'training.json').write_text(json.dumps(report))
    if defect:
        with pytest.raises(SignatureError):completed_train(root,f,args)
    else:assert completed_train(root,f,args)['variant']==VARIANT


def test_plugin_identity_admission_precedes_original_trainer_and_preserves_loss_trace_count():
    from experiments.bell_geometry_gate11 import MASK_PROBES
    assert len(set(range(20,300001,20))|MASK_PROBES)==15006
    tree=ast.parse((g.ROOT/'experiments/bell_geometry_gate11.py').read_text())
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native')
    work=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='work')
    plugin=next(n for n in ast.walk(work) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_get_plugin')
    train=next(n for n in ast.walk(work) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run' and
               isinstance(n.func.value,ast.Name) and n.func.value.id=='training')
    assert plugin.lineno<train.lineno
