"""New E.13 CPU stand-ins only. No native renderer, BVH, reference or training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from infrastructure import material_recovery_e13 as m
from infrastructure.material_recovery_e13_evaluation import numeric, image_metrics, capture_native, native_tensor_default, render_evaluation
from infrastructure.signatures import SignatureError
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.training_probe import parameter_copies, snapshot


def materials():
    n = 4
    t = np.tile([1.,0.,0.], (n,1)); b = np.tile([0.,1.,0.], (n,1))
    normal = np.tile([0.,0.,1.], (n,1))
    Q = np.tile(np.diag([.36,.09,0]), (n,1,1))
    truth = dict(T=t,B=b,normals=normal,Q_world=Q,Q_local=Q[:,:2,:2],
                 widths_major_minor=np.tile([.6,.3],(n,1)),theta_major=np.zeros(n),
                 orientation_defined=np.ones(n,bool),orientation_eligible=np.ones(n,bool),
                 color=np.tile([.4,.5,.6],(n,1)),metallic=np.full((n,1),.2))
    pred = dict(mx=np.full((n,1),.6),my=np.full((n,1),.3),rotation=np.tile([1.,0.],(n,1)),
                color=truth['color'].copy(),metallic=truth['metallic'].copy(),
                T=t,B=b,actual_T=t.copy(),actual_B=b.copy())
    return pred, truth


@pytest.mark.parametrize('swapped',[False,True])
def test_gauge_equivalent_Q_orientation_and_recovery(swapped):
    p,t = materials()
    if swapped:
        p['mx'],p['my'] = p['my'],p['mx'];p['rotation'] *= -1
    e = m.material_errors(p,t,np);a = m.summarize_material([e],np)
    assert np.max(e['Q_world_relative']) < 1e-14
    assert np.max(e['orientation_degrees']) < 1e-6
    assert a['invalid_frames'] == a['invalid_materials'] == 0
    assert all(v['fraction'] == 1 for v in a['descriptive_pass_fractions'].values())


def test_frame_invalid_is_retained_and_no_repair():
    p,t = materials();p['actual_B'][0] *= -2;before = deepcopy(p)
    a = m.summarize_material([m.material_errors(p,t,np)],np)
    assert a['observations'] == 4 and a['invalid_frames'] == 1
    assert a['descriptive_pass_fractions']['Q_world_relative']['fraction'] == .75
    for k in p:assert np.array_equal(p[k],before[k])


def test_reversed_actual_handedness_is_invalid():
    p,t=materials();p['actual_B'][0]*=-1
    e=m.material_errors(p,t,np)
    assert e['invalid_frame'].tolist()==[True,False,False,False]


def test_orientation_eligibility_and_undefined_prediction():
    p,t = materials();t['orientation_eligible'][0] = False;p['mx'][1] = p['my'][1]
    a = m.summarize_material([m.material_errors(p,t,np)],np)
    assert a['gt_orientation_eligible'] == 3
    assert a['predicted_orientation_undefined_on_gt_eligible'] == 1
    assert a['orientation_degrees']['count'] == 2
    assert a['descriptive_pass_fractions']['orientation_degrees']['denominator'] == 3


def test_world_local_Q_in_rotated_world_frame():
    p,t = materials();R = np.array([[0,0,1],[1,0,0],[0,1,0.]])
    for k in ('T','B','actual_T','actual_B'):p[k] = p[k]@R.T
    for k in ('T','B','normals'):t[k] = t[k]@R.T
    t['Q_world'] = np.einsum('ij,njk,lk->nil',R,t['Q_world'],R)
    e = m.material_errors(p,t,np)
    assert np.max(e['Q_world_relative']) == np.max(e['Q_local_relative']) == 0


def test_initialization_parity_exact_no_substitution():
    packet = dict(parameters={'w':{'sha256':'a','shape':[3],'dtype':'float32'}},splits={'train_ids':['1']})
    m.initialization_gate(packet,deepcopy(packet))
    for key in ('parameters','splits'):
        other = deepcopy(packet);other[key] = {}
        with pytest.raises(SignatureError):m.initialization_gate(packet,other)


def test_numeric_archive_rejects_pickle_and_nonfinite(tmp_path):
    p = tmp_path/'data.npz'
    np.savez(p,a=np.arange(3));assert numeric(p,np)['a'].tolist() == [0,1,2]
    np.savez(p,a=np.array([{'unsafe':'payload'}],object))
    with pytest.raises(ValueError):numeric(p,np)
    np.savez(p,a=np.array([float('nan')]))
    with pytest.raises(SignatureError):numeric(p,np)


def test_component_and_native_image_metrics():
    gt = np.full((128,128,3,3),.1);gt[:,:,2] = .2
    pred = gt[:,:,[2,0,1]].astype(np.float32)
    hit = np.ones((128,128),bool)
    srgb = 1.055*np.maximum(pred[:,:,0],np.finfo(np.float32).eps)**(1/2.4)-.055
    r = image_metrics(pred,gt,hit,srgb,np,native_pred=srgb)
    assert r['native_psnr_infinite'] and r['native_ssim'] == 1.
    assert r['total']['relative_l2'] < 2e-8
    assert r['diffuse']['relative_l2'] < 2e-8
    pred[:,:,2] *= 2
    r = image_metrics(pred,gt,hit,srgb,np,native_pred=srgb)
    assert r['specular']['relative_l2'] == pytest.approx(1,abs=1e-7)


def test_actual_native_linear_capture_no_extra_calls_and_cleanup():
    calls=[]
    class Shader:
        def compute_radiance(self):
            calls.append('radiance');return tuple(torch.ones(2,3)*i for i in range(6))
    class Model:
        shader_network=Shader()
        def _construct_ray_batch(self):
            calls.append('rays');return {'hit_mask':torch.tensor([True,True])}
    model=Model()
    with capture_native(model,torch,np) as (chunks,batches):
        model._construct_ray_batch();model.shader_network.compute_radiance()
    assert calls==['rays','radiance'] and len(chunks)==len(batches)==1
    assert chunks[0].shape==(2,3,3) and sys.getprofile() is None
    with pytest.raises(RuntimeError):
        with capture_native(model,torch,np):raise RuntimeError('synthetic evaluation failure')
    assert sys.getprofile() is None


def test_native_tensor_default_restores_on_failure():
    calls=[]
    fake=SimpleNamespace(tensor=lambda v:SimpleNamespace(type=lambda:'CPU-before'),set_default_tensor_type=calls.append)
    with pytest.raises(RuntimeError):
        with native_tensor_default(fake):raise RuntimeError('synthetic failure')
    assert calls==['torch.cuda.FloatTensor','CPU-before']


def test_complete_twelve_image_adapter_with_cpu_standins(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from infrastructure.runs import RunPaths
    import imageio
    geometry=tmp_path/'geometry';(geometry/'rays').mkdir(parents=True)
    refs=tmp_path/'refs';(refs/'relight').mkdir(parents=True)
    out=tmp_path/'out';out.mkdir();(out/'evaluations').mkdir();(out/'renders').mkdir()
    ids=['58','11','105','114','17','76'];hit=np.zeros(16384,bool);hit[[0,100]]=True
    calls=[];types=[]
    monkeypatch.setattr(torch,'set_default_tensor_type',types.append)
    def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,np.finfo(np.float32).eps)**(1/2.4)-.055)
    class Shader:
        def __init__(self):self.relighting=False
        @contextmanager
        def environment(self,callback,identity):
            assert identity=='e11-v3-unseen-plus90Y';self.relighting=True
            try:yield
            finally:self.relighting=False
        def compute_radiance(self):
            calls.append('shader');x=.15 if self.relighting else .1
            return (torch.full((2,3),2*x),torch.full((2,3),x),torch.full((2,3),x),None,None,None)
    class Model:
        test_ids=['58'];nvs_ids=ids[1:];shader_network=Shader()
        def eval(self):return self
        def _construct_ray_batch(self):
            return {k:torch.from_numpy(v) for k,v in batch.items()}
        def test_step(self,index,is_nvs):
            self._construct_ray_batch();total=self.shader_network.compute_radiance()[0].numpy()
            rgb=np.zeros((16384,3),np.float32);rgb[hit]=srgb(total)
            gt=np.zeros((16384,3),np.float32);gt[hit]=srgb(np.full((2,3),.2,np.float32))
            calls.append(('view',self.nvs_ids[index] if is_nvs else self.test_ids[index]))
            return dict(rgb_pr=torch.from_numpy(rgb.reshape(128,128,3)),rgb_gt=torch.from_numpy(gt.reshape(128,128,3)))
    directions=np.tile([0,0,1.],(16384,1)).astype(np.float32)
    batch=dict(hit_mask=hit,inters=directions.copy(),normals=directions.copy(),rays_d=directions.copy(),
               rays_o=np.zeros_like(directions),depth=np.ones((16384,1),np.float32))
    for i in ids:
        np.savez(geometry/'rays'/(i+'.npz'),hit=hit,positions=batch['inters'],normals=batch['normals'],directions=batch['rays_d'],
                 origins=batch['rays_o'],depth=batch['depth'].reshape(-1))
        for mode,x in (('',.1),('relight/',.15)):
            linear=np.zeros((16384,3,3));linear[hit,0]=x;linear[hit,1]=x;linear[hit,2]=2*x
            np.savez(refs/(mode+i+'.npz'),linear_components=linear.reshape(128,128,3,3))
            if mode:imageio.imwrite(refs/(mode+i+'.png'),np.rint(srgb(linear[:,2].reshape(128,128,3))*255).astype(np.uint8))
    generation={'splits':{'validation':['58'],'held_view_nvs':ids[1:]},'known_environment':{},'relighting_environment':{}}
    result=render_evaluation(Model(),geometry,refs,generation,RunPaths('fixture',out),torch,np)
    assert len(result['per_view'])==12 and result['shader_chunks']==12
    assert result['primary_rays']==196608 and result['direction_slots']==12*2*1536
    assert all(v['total']['relative_l2']<1e-6 for v in result['per_view'].values())
    assert [v[1] for v in calls if isinstance(v,tuple)]==ids+ids
    assert len(types)==24 and types[1::2]==['torch.FloatTensor']*12
    assert sys.getprofile() is None


def test_strict_checkpoint_roundtrip_and_foreign_tag(tmp_path):
    class Model(torch.nn.Module):
        material_variant=m.ARMS[0]
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.ones(3))
            self.shader_network=torch.nn.Module();self.shader_network.register_buffer('_a_variant_signature',torch.tensor([1],dtype=torch.uint8))
        def load_state_dict(self,state_dict,strict=True):
            assert strict is True
            if not torch.equal(state_dict['shader_network._a_variant_signature'],self.shader_network._a_variant_signature):
                raise SignatureError('foreign tag')
            return super().load_state_dict(state_dict,strict=True)
    model=Model();opt=torch.optim.Adam(model.parameters(),amsgrad=True)
    before=snapshot(parameter_copies(model,np),np);rng=torch.get_rng_state().clone()
    r=m.strict_roundtrip(model,opt,tmp_path/'endpoint.pth',m.ARMS[0],torch,np,{'fixture':True})
    assert r['strict'] and not r['optimizer_restored'] and r['step']==2000
    assert torch.equal(rng,torch.get_rng_state()) and snapshot(parameter_copies(model,np),np)==before
    assert not opt.state
    with pytest.raises(FileExistsError):m.strict_roundtrip(model,opt,tmp_path/'endpoint.pth',m.ARMS[0],torch,np,{})
    with pytest.raises(SignatureError):m.strict_roundtrip(model,opt,tmp_path/'foreign.pth',m.ARMS[1],torch,np,{})
    with pytest.raises(SignatureError):model.load_state_dict({'w':model.w,'shader_network._a_variant_signature':torch.tensor([2],dtype=torch.uint8)})


@pytest.mark.parametrize('fail_at',[None,3])
def test_endpoint_stops_exactly_2000_original_returns_and_cleanup(tmp_path,monkeypatch,fail_at):
    # Existing tiny CPU Trainer fixture; real scalar CPU Adam, no native science.
    from training_smoke_fixtures import trainer_fixture
    from infrastructure.stage_execution import trainer_io
    f=trainer_fixture();monkeypatch.chdir(tmp_path)
    (tmp_path/'data/model').mkdir(parents=True);logs=tmp_path/'logs';logs.mkdir();cp=tmp_path/'cp';cp.mkdir()
    cfg=dict(name='e13-toy',network='shape',train_dataset_type='dummy',train_dataset_cfg={},
             val_set_list=[dict(name='val',type='dummy',cfg={})],dataset_dir='unused',loss=['nerf_render'],
             val_metric=['shape_render'],key_metric_name='psnr',total_step=2000,val_interval=2000,save_interval=2000,
             lr_type='warm_up_cos',lr_cfg={'end_warm':1000,'end_iter':100000},train_log_step=10)
    states=random.getstate(),np.random.get_state(),torch.get_rng_state()
    trainer=trainer_io(f.Trainer,SimpleNamespace(logs=logs,checkpoints=cp))(cfg)
    events=[];clean=[]
    class Observer:
        def event(self,boundary,values):
            if boundary=='before_forward':
                assert values['step']==len(events)
                if fail_at==values['step']:raise RuntimeError('synthetic boundary failure')
        def after_optimizer(self,values,completed):events.append((values['step'],completed))
        def cleanup(self,completed,errors):clean.append((completed,errors))
    try:
        trace=m.EndpointTrace(trainer,Observer(),ROOT/'train/trainer.py')
        if fail_at is None:
            trace.run();assert events==[(i,i+1) for i in range(2000)]
            assert not list(cp.iterdir())
        else:
            with pytest.raises(RuntimeError):trace.run()
            assert len(events)==fail_at
        assert clean==[(len(events),[])] and f.loaders[0].iterator.closed and f.writers[0].closed
        assert sys.gettrace() is None
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])


def test_original_scheduler_arithmetic_and_native_shuffle_cursor():
    # No scheduler is invoked by the observer, including at/after warmup.
    for step in (2,999,1000,1001,1999):
        lr=5e-4*(step/1000) if step<1000 else 5e-4*((np.cos(np.pi*((step-1000)/99000))+1)*.5*(1-.2)+.2)
        network=SimpleNamespace(training=True,train_batch_i=0,tbn=1025)
        t=SimpleNamespace(network=network,optimizer=SimpleNamespace(param_groups=[{'lr':lr}]))
        p=m.RecoveryProbe(t,torch,np,{},lambda:None,rng=lambda:{},memory=lambda:{})
        p.current={'step':step};p.before_forward({'step':step,'train_data':{'step':step},'lr':lr})
        assert p.current['sampling']['cursor_before']==0


def test_no_frozen_scientific_or_existing_evidence_edits():
    baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    changed=subprocess.run(['git','diff','--diff-filter=M','--name-only','0d82199af5d6fb5e8654b8781a4f9f55f028bb2c'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
    assert not any(p.startswith(('scientific_sources/','scientific_variants/','configs/','docs/evidence/')) for p in changed)


def test_checkpoint_and_quantization_conventions_statically_preserved():
    import ast
    tree=ast.parse((ROOT/'scientific_sources/material_a/network/renderer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeROMaterialRenderer')
    test=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='test_step')
    keys=next(n.value for n in test.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='output_keys' for t in n.targets))
    assert 'rgb_pr' in ast.literal_eval(keys)  # Correct earlier mistaken interface suspicion.
    assert 'imgs_dir=nvs_imgs_dir' in (ROOT/'scientific_sources/material_a/train/trainer.py').read_text()


def test_metadata_preflight_has_no_scientific_imports_or_output(tmp_path):
    freeze=ROOT/'docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V1.freeze.json'
    value=json.loads(freeze.read_text())
    script='''
import builtins,runpy,sys
original=builtins.__import__
def blocked(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','network','dataset','train','raytracing','_raytracing','open3d'},name
 return original(name,*a,**kw)
builtins.__import__=blocked
runpy.run_path(sys.argv.pop(1),run_name='__main__')
'''
    for variant in m.ARMS:
        out=tmp_path/variant
        r=subprocess.run([sys.executable,'-I','-B','-c',script,str(ROOT/'experiments/bell_material_recovery_e13.py'),
            '--variant',variant,'--inputs',value['host_benchmark'],'--thesis','/home/isavand/paper/final_results',
            '--admission',value['admission']['path'],'--output',str(out),'--freeze-sha256',digest(freeze)],
            cwd=ROOT,text=True,capture_output=True)
        assert r.returncode==0,r.stderr
        assert json.loads(r.stdout)['native_calls']==0 and not out.exists()
