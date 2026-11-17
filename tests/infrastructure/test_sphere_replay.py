"""New sphere adapter checks only; no native field, shader, CUDA or dataset load."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from infrastructure.signatures import SignatureError
from infrastructure.sphere_replay import (ROOT, ORIGINAL_CFG, LIMITS, CORE, CAPTURE_KEYS,
    effective_configuration, split_ids, Budget, BudgetStop, evaluation_scope,
    renderer_adapter, trainer_adapter, bind_optimizer, workspace)
from infrastructure.sphere_visuals import orientation_maps, write_view


def test_effective_legacy_sphere_semantics_and_no_rng_consumption():
    raw=yaml.safe_load((ROOT/ORIGINAL_CFG).read_text());before=deepcopy(raw)
    state=random.getstate();numpy=np.random.get_state()
    cfg,diag=effective_configuration(raw,'/inputs/datasets/nerf_synthetic','/inputs/thesis/meshes/sphere_shape-300000.ply','sphere_corrected_core_v1')
    assert raw==before and random.getstate()==state
    assert np.array_equal(np.random.get_state()[1],numpy[1])
    assert cfg['is_nerf'] and cfg['total_step']==100000 and cfg['worker_num']==8 and cfg['random_seed']==6033
    assert cfg['lr_cfg']==raw['lr_cfg'] and cfg['loss']==raw['loss'] and cfg['reg_mat']
    assert cfg['reg_diffuse_light_lambda']==.01
    shader=cfg['shader_cfg']
    assert shader['illumination_mode']=='learned_native' and shader['indirect_mode']=='native_learned'
    assert shader['regularization_version']=='legacy_v1' and shader['reg_change'] and shader['random_azimuth']
    assert shader['reg_energy_loss_lambda']==shader['reg_spec_loss_lambda']==shader['reg_lambda1']==.05
    assert diag['ignored_top_level']['reg_energy_loss_lambda']==.01
    assert diag['unused_shader_fields']==dict(shading_feature_num=256,num_viewdir_freqs=0)
    assert shader['diffuse_sample_num']==512 and shader['specular_sample_num']==1024 and not shader['human_lights']
    assert cfg['train_dataset_cfg']['dataset_dir']==cfg['val_set_list'][0]['cfg']['dataset_dir']==cfg['dataset_dir']
    assert yaml.safe_load((ROOT/'configs/replays/sphere_corrected_core_v1.yaml').read_text())==cfg


def test_split_matches_exact_pinned_function_without_global_rng_change():
    tree=ast.parse((ROOT/'scientific_sources/material_a/dataset/database.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_database_split')
    env=dict(random=random,BaseDatabase=object,NeRFSyntheticDatabase=type('Nerf',(),{}))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-split>','exec'),env)
    db=env['NeRFSyntheticDatabase']();db.get_img_ids=lambda:[str(i) for i in range(100)]
    state=random.getstate()
    try:
        actual=env['get_database_split'](db,'validation')
        assert dict(zip(('train','validation','nvs'),actual))==split_ids()
    finally:random.setstate(state)
    expected=split_ids();assert expected['validation']==['36'] and expected['nvs']==['39','2','94','62','23']
    assert len(expected['train'])==94 and random.getstate()==state


def test_original_adam_scheduler_loss_and_frozen_sources():
    from infrastructure.stage_sources import baseline,PINS
    baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    lr_source=ROOT/'scientific_sources/material_a/train/lr_common_manager.py'
    env={};exec(compile(lr_source.read_text(),str(lr_source),'exec'),env)
    lr=env['WarmUpCosLR'](dict(end_warm=1000,end_iter=100000))
    opt=SimpleNamespace(param_groups=[{}]);assert lr(opt,0)==0 and lr(opt,1)==5e-7
    assert lr(opt,1000)==.0005 and abs(lr(opt,100000)-.0001)<1e-15
    assert trainer_adapter.__name__=='trainer_adapter'  # Adapter must inherit, never replace run.
    source=(ROOT/'infrastructure/sphere_replay.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in ast.walk(tree))


def test_orientation_physical_axis_equivalence_and_exact_isotropy():
    mask=np.ones((2,3),bool);r=np.zeros((2,3,2));r[...,0]=1
    mx=np.ones((2,3,1))*.2;my=mx*2
    theta,rgb=orientation_maps(r,mx,my,mask,np)
    theta_swapped,_=orientation_maps(-r,my,mx,mask,np)
    assert np.allclose(theta,theta_swapped) and np.allclose(theta,np.pi/2)
    _,iso=orientation_maps(r,mx,mx,mask,np);assert np.all(iso==.5)
    assert rgb.shape==(2,3,3) and np.isfinite(rgb).all()


def tiny_arrays():
    mask=np.array([[1,0],[0,1]],bool)
    a={k:np.zeros((2,2,3),np.float32) for k in ('normals','rgb_gt','rgb_pr','kd','diffuse_color','specular_color')}
    a.update(hit_mask=mask,mx=np.ones((2,2,1),np.float32)*.2,my=np.ones((2,2,1),np.float32)*.4,
             metallic=np.ones((2,2,1),np.float32)*.6,rotation=np.broadcast_to([1.,0.],(2,2,2)).copy())
    a['normals'][mask]=[.5,.5,1.]
    a['tangents']=np.zeros((2,2,3),np.float32);a['tangents'][mask]=[.5,1.,.5]
    a['bitangents']=np.zeros((2,2,3),np.float32);a['bitangents'][mask]=[0.,.5,.5]
    return a


def test_lossless_maps_fixed_scales_no_overwrite(tmp_path):
    from PIL import Image
    a=tiny_arrays();p=tmp_path/'view'
    write_view(p,a,dict(camera_id='36',completed_updates=2000),np)
    with np.load(p/'materials.npz',allow_pickle=False) as data:
        assert np.array_equal(data['mx'],a['mx']) and np.array_equal(data['hit_mask'],a['hit_mask'])
        assert np.array_equal(data['major_width'],a['my'])
    assert np.array(Image.open(p/'mx.png'))[0,0]==51
    assert json.loads((p/'view.json').read_text())['width_scale']==[0,1]
    with pytest.raises(FileExistsError):write_view(p,a,{},np)


def test_capture_observes_same_shade_calls_and_pixel_order(tmp_path):
    torch=pytest.importorskip('torch')
    a=tiny_arrays();calls=[]
    class Base:
        test_ids=['36'];nvs_ids=split_ids()['nvs']
        def shade(self,pts,view_dirs,normals,human_poses,is_train,step=None):
            calls.append(is_train)
            return {k:torch.ones((len(pts),2 if k.startswith('rotation') else 3))*i for i,k in enumerate(CAPTURE_KEYS,1)}
        def test_step(self,index,is_nvs=False):
            pts=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
            self.shade(pts,-pts,pts,None,False)
            return {k:torch.from_numpy(v) for k,v in a.items() if k!='hit_mask'}
    report={'optimizer_returns':2000}
    cls=renderer_adapter(Base,tmp_path,report,Budget(tmp_path),torch,np);obj=cls()
    result=obj.test_step(0)
    assert calls==[False] and obj._captures is None
    p=tmp_path/'renders/step-002000/validation/camera-36/materials.npz'
    with np.load(p,allow_pickle=False) as data:
        assert np.array_equal(data['position'][a['hit_mask']],[[1,2,3],[4,5,6]])
        assert np.array_equal(result['mx'].numpy(),data['mx'])
    assert report['saved_views']==[dict(completed_updates=2000,split='validation',camera_id='36')]


def test_extra_evaluation_restores_all_rng_modes_even_failure(monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda s:None)
    model=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity());model[1].eval()
    py,ns,ts=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with evaluation_scope(model,torch,np):
            random.random();np.random.rand();torch.rand(2);model.eval();raise RuntimeError('fake NVS failure')
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1]) and torch.equal(torch.get_rng_state(),ts)
    assert model.training and model[0].training and not model[1].training


def test_lazy_trainer_binds_only_original_once_and_refuses_resume(tmp_path):
    torch=pytest.importorskip('torch');initialized=[]
    class Base:
        def __init__(self,cfg):self.cfg=cfg
        def _init_network(self):
            initialized.append(1)
            self.network=torch.nn.Linear(1,1);self.network.is_nerf=True
            ids=split_ids();self.network.train_ids=ids['train'];self.network.test_ids=ids['validation'];self.network.nvs_ids=ids['nvs']
            self.network.cfg={};self.network.shader_network=SimpleNamespace(cfg=dict(illumination_mode='learned_native',regularization_version='legacy_v1'))
            self.optimizer=torch.optim.Adam(self.network.parameters(),lr=0,eps=1e-4,amsgrad=True)
            self.lr_manager=SimpleNamespace()
        def _load_model(self):return 0,0
        def run(self):pass
    report={'optimizer_returns':0}
    cls=trainer_adapter(Base,tmp_path,report,lambda:None,Budget(tmp_path),torch,np)
    trainer=cls({});assert not hasattr(trainer,'network') and cls.run is Base.run
    trainer._init_network();assert initialized==[1]
    assert trainer._load_model()==(0,0)
    with pytest.raises(SignatureError):trainer._init_network()
    for i in range(2):
        trainer.optimizer.zero_grad();trainer.network(torch.ones(1,1)).sum().backward()
        trainer.optimizer.param_groups[0]['lr']=0 if i==0 else 5e-7
        trainer.optimizer.step()
    assert report['optimizer_returns']==2
    assert report['initial_updates'][0]['absolute_parameter_displacement']==0
    assert report['initial_updates'][1]['absolute_parameter_displacement']>0
    (tmp_path/'checkpoints').mkdir();Path(trainer.pth_fn).write_bytes(b'fake')
    with pytest.raises(SignatureError,match='resume'):trainer._load_model()


def test_nvs_10k_cadence_no_final_duplicate(tmp_path):
    torch=pytest.importorskip('torch');events=[]
    class Base:
        def _save_model(self,step,best_para,save_fn=None):Path(self.pth_fn).write_bytes(b'fake checkpoint')
    report={};cls=trainer_adapter(Base,tmp_path,report,lambda:None,Budget(tmp_path),torch,np)
    obj=object.__new__(cls);obj.pth_fn=str(tmp_path/'checkpoints/model.pth')
    obj.network=SimpleNamespace(nvs=lambda *args:events.append('nvs'))
    (tmp_path/'checkpoints').mkdir();(tmp_path/'config.effective.yaml').write_text('fixture')
    for step in (2000,10000,90000,100000):obj._save_model(step,0)
    assert events==['nvs','nvs']  # Final call stays in ORIGINAL Trainer.run.
    assert sorted(p.name for p in (tmp_path/'checkpoints').glob('step-*.pth'))==[
        'step-002000.pth','step-010000.pth','step-090000.pth','step-100000.pth']


def test_budget_fails_without_removing_partial_evidence(tmp_path):
    (tmp_path/'partial.pth').write_bytes(b'keep')
    with pytest.raises(BudgetStop):Budget(tmp_path,seconds=-1).check()
    assert (tmp_path/'partial.pth').read_bytes()==b'keep'


def test_workspace_only_output_owned_links(tmp_path):
    science=tmp_path/'source';science.mkdir();(science/'assets').mkdir()
    output=tmp_path/'output';output.mkdir();(output/'renders').mkdir()
    work=workspace(output,science)
    assert (work/'data/train_vis').resolve()==output/'renders/legacy_validation'
    assert (work/'assets').resolve()==science/'assets'


def test_authorization_is_explicit_and_binds_budget_output_commit(tmp_path):
    from run_sphere_replay import verify_authorization
    from infrastructure.sphere_replay import PROTOCOL, IMAGE
    from infrastructure.stage_sources import digest
    freeze={'host_output_directory':'/approved/fresh'}
    with pytest.raises(SignatureError,match='Authorization'):
        verify_authorization(None,None,freeze,'f'*64,'a'*40)
    a=dict(approved=True,protocol=PROTOCOL,commit='a'*40,freeze_sha256='f'*64,image_id=IMAGE,
           limits=LIMITS,host_output_directory='/approved/fresh',output_directory='/outputs/sphere',selected_gpu='0')
    p=tmp_path/'auth.json';p.write_text(json.dumps(a))
    assert verify_authorization(p,digest(p),freeze,'f'*64,'a'*40)==a
    for key,val in (('commit','b'*40),('approved',False),('host_output_directory','/other'),('limits',{})):
        bad={**a,key:val};p.write_text(json.dumps(bad))
        with pytest.raises(SignatureError):verify_authorization(p,digest(p),freeze,'f'*64,'a'*40)


def test_unsafe_missing_inputs_fail_before_scientific_import(tmp_path):
    from infrastructure.sphere_replay import input_records
    with pytest.raises(SignatureError,match='absolute'):
        input_records('relative',tmp_path/'mesh.ply',tmp_path)
    data=tmp_path/'data';data.mkdir()
    with pytest.raises(SignatureError,match='approved historical'):
        input_records(data,tmp_path/'made-up.ply',tmp_path)


def test_duplicate_view_rejected_before_any_new_shader_call(tmp_path):
    torch=pytest.importorskip('torch')
    class Base:
        test_ids=['36']
        def test_step(self,*args,**kwargs):raise AssertionError('No additional native query allowed')
    report=dict(optimizer_returns=2000,saved_views=[dict(completed_updates=2000,split='validation',camera_id='36')])
    obj=renderer_adapter(Base,tmp_path,report,Budget(tmp_path),torch,np)()
    with pytest.raises(SignatureError,match='duplicate'):obj.test_step(0)


def test_native_dependency_imports_remain_cold_until_explicit_execution():
    import subprocess,sys
    code='''
import sys
import run_sphere_replay
import infrastructure.sphere_replay
import infrastructure.sphere_visuals
assert not any(n.split('.')[0] in ('torch','network','dataset','train','raytracing','_raytracing','scientific_variants') for n in sys.modules)
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=str(ROOT),capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('is_train',[True,False])
def test_original_nerf_nonunit_views_adapt_at_actual_shade_boundary(tmp_path,is_train):
    torch=pytest.importorskip('torch')
    from infrastructure.sphere_replay import unit_outgoing_views,VIEW_INTERFACE
    # Use the actual corrected unit-domain guard, without importing/constructing
    # native shader classes or executing any BRDF/sampling/model calculation.
    source=ast.parse((ROOT/'scientific_variants/material_a_v1.py').read_text())
    helpers=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('finite','unit')]
    env=dict(torch=torch,require=lambda condition,message: (_ for _ in ()).throw(SignatureError(message)) if not condition else None)
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<corrected-unit-guard>','exec'),env)
    views=torch.tensor([[.36,-.36,-1.],[0.,0.,-1.]],dtype=torch.float32)
    raw=views.clone();normals=torch.tensor([[0.,0.,-1.],[0.,0.,-1.]])
    pts=torch.zeros_like(views);poses=torch.eye(4).repeat(2,1,1);calls=[];token=object()
    with pytest.raises(SignatureError,match='Unit views'):env['unit'](views,'views')
    class Base:
        def shade(self,p,v,n,h,train,step=None):
            env['unit'](v,'views')
            calls.append((p,v,n,h,train,step));return token
    report={};obj=renderer_adapter(Base,tmp_path,report,Budget(tmp_path),torch,np)()
    py,ns,ts=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    assert obj.shade(pts,views,normals,poses,is_train,step=0) is token
    assert len(calls)==1 and calls[0][0] is pts and calls[0][2] is normals and calls[0][3] is poses
    assert calls[0][4:]==(is_train,0) and torch.equal(views,raw)
    actual=calls[0][1]
    assert actual.dtype==views.dtype and actual.device==views.device
    assert torch.allclose(actual,raw/torch.linalg.norm(raw,dim=-1,keepdim=True))
    assert torch.allclose(torch.cross(actual,raw,dim=-1),torch.zeros_like(raw),atol=1e-7)
    assert torch.all((actual*raw).sum(-1)>0)  # Same outgoing half-line, no sign flip.
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1]) and torch.equal(torch.get_rng_state(),ts)
    assert report['view_direction_adapter']['identity']==VIEW_INTERFACE
    assert torch.equal(unit_outgoing_views(torch.tensor([[0.,0.,-1.]]),torch),torch.tensor([[0.,0.,-1.]]))


@pytest.mark.parametrize('bad',[[[0.,0.,0.]],[[float('nan'),0.,1.]],[[float('inf'),0.,1.]],[[1.,2.]], [1.,2.,3.]])
def test_invalid_views_fail_before_shader_no_fallback(tmp_path,bad):
    torch=pytest.importorskip('torch')
    class Base:
        def shade(self,*args,**kwargs):raise AssertionError('Invalid views reached shader')
    obj=renderer_adapter(Base,tmp_path,{},Budget(tmp_path),torch,np)()
    with pytest.raises(SignatureError,match='Sphere outgoing views'):
        obj.shade(None,torch.tensor(bad,dtype=torch.float32),None,None,True)
