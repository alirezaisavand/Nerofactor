"""Original G/A class bodies with disposable CPU models, loaders and shaders."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import yaml
try:import torch
except (ImportError,OSError):torch=None
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.paths import PathResolver
from infrastructure.runs import RunStore
from infrastructure.signatures import SignatureError
from infrastructure import lifecycle_execution as execution
from infrastructure.lifecycle_state import original_save,inspect_state,restore_original,state_record,canonical
from infrastructure.lifecycle_training import ReloadTrace,GResumeProbe,AResumeProbe,ForbidCalls
from infrastructure.lifecycle_evaluation import admit_view,metric_comparison
from infrastructure.g_training import GTrainingProbe,GTwoUpdateTrace
from infrastructure.bell_material_integration import BellProbe
from infrastructure.bounded_training import TwoUpdateTrace
from infrastructure.bell_inference import runtime_environment


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=self.root/'data';(self.data/'GlossySynthetic/bell').mkdir(parents=True)
        (self.data/'GlossySynthetic/bell/fixture').write_bytes(b'immutable input')
        self.thesis=self.root/'empty';self.thesis.mkdir();self.output=self.root/'out'
    def plan(self,**overrides):
        values=dict(stage='geometry',action='save',commit=PINS['geometry'],datasets=self.data,thesis=self.thesis,output=self.output)
        values.update(overrides);return execution.plan(**values)
    def producer(self):
        p=self.plan();root=self.root/'producer';(root/'checkpoints').mkdir(parents=True)
        checkpoint=root/'checkpoints/model.pth';checkpoint.write_bytes(b'opaque CPU fixture; never deserialized')
        proof={'manifest_version':1,'protocol':execution.PROTOCOL,'action':'save','status':'complete','stage':'geometry',
               'scientific_commit':PINS['geometry'],'scientific_source':p['source'],'infrastructure_files':execution.identity(),
               'original_configuration':p['configuration'],'splits':execution.splits('geometry'),'mesh_identity':None,
               'research_handoff_admitted':False,'checkpoint':{'sha256':digest(checkpoint),'step':2},'optimizer_calls_completed':2}
        proof.update(run_id='fixture-producer',phase='complete',infrastructure_commit_declared='0'*40,image_id_declared=execution.IMAGE,
                     runtime_configuration={},input_files=[],prerequisites={},checkpoint_input=None,producer_reference=None,
                     natural_schedule_tested=False,uninterrupted_equivalence_proven=False,RNG_loader_cursor_state_in_original_checkpoint=False,
                     started_at_utc='CPU-fixture',finished_at_utc='CPU-fixture',runtime={},effective_configuration={},
                     runtime_cleanup={},output_hashes={},imported_scientific_modules={},updates=[],save_sampling_cursor=1024)
        proof['checkpoint'].update(path='checkpoints/model.pth',bytes=checkpoint.stat().st_size,best_para=0,invocation='CPU fixture',
            scheduler={},state={'network':{},'network_sha256':'0'*64,'optimizer':{},'optimizer_sha256':'0'*64,
                               'optimizer_counter_types':{},'optimizer_group_types':[]})
        report=root/'lifecycle.json';report.write_text(json.dumps(proof));return root,checkpoint,report,proof
    def test_planning_is_metadata_only_and_refuses_implicit_checkpoint_or_wrong_source(self):
        before=random.getstate();p=self.plan();self.assertEqual(before,random.getstate());self.assertFalse(self.output.exists())
        self.assertEqual(p['configuration']['name'],'bell_shape')
        for override in [{'commit':PINS['material']},{'action':'restore'},{'checkpoint_sha':'0'*64},{'output':self.data/'escape'},
                         {'geometry':self.root},{'stage':'material','commit':PINS['material']}]:
            with self.subTest(override=override),self.assertRaises((ValueError,OSError)):self.plan(**override)
    def test_explicit_checkpoint_report_hash_source_and_split_admission(self):
        root,ck,report,proof=self.producer()
        kwargs=dict(action='restore',checkpoint_root=root,checkpoint=ck,checkpoint_sha=digest(ck),producer=report,producer_sha=digest(report))
        self.plan(**kwargs);self.assertFalse(self.output.exists())
        for key,bad in [('stage','material'),('status','failed'),('splits',execution.splits('material')),('infrastructure_files',{}),('action','restore')]:
            rejected={**proof,key:bad};report.write_text(json.dumps(rejected));kwargs['producer_sha']=digest(report)
            with self.subTest(key=key),self.assertRaises(ValueError):self.plan(**kwargs)
        report.write_text(json.dumps(proof));kwargs['producer_sha']=digest(report);ck.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):self.plan(**kwargs)
    def test_trust_and_runtime_guards_precede_allocation(self):
        p=self.plan()
        with patch.object(execution,'runtime_guards') as guard,patch.object(execution,'native') as native:
            with self.assertRaises(ValueError):execution.execute(p,'fixture',execution.IMAGE,'0'*40)
            guard.assert_not_called();native.assert_not_called()
        with patch.object(execution,'runtime_guards',side_effect=ValueError('blocked runtime')):
            with self.assertRaises(ValueError):execution.execute(p,'fixture',execution.IMAGE,'0'*40,trusted=True)
        self.assertFalse(self.output.exists())
    def test_validate_only_does_not_import_native_or_deserialize(self):
        script='''
import builtins,sys
original=builtins.__import__
def guarded(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','numpy','network','dataset','train','open3d','_raytracing'},name
 return original(name,*a,**kw)
builtins.__import__=guarded
from run_lifecycle_diagnostic import main
raise SystemExit(main(sys.argv[1:]))
'''
        result=subprocess.run([sys.executable,'-B','-c',script,'--stage','geometry','--action','save',
            '--scientific-commit',PINS['geometry'],'--datasets-root',str(self.data),'--thesis-root',str(self.thesis),
            '--output-root',str(self.output),'--run-id','fixture','--validate-only'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(self.output.exists())
    def test_read_only_evaluation_leakage_and_cross_stage_splits_rejected(self):
        for stage,role,view in [('geometry','validation','0'),('material','validation','58'),('material','nvs','11')]:
            actual=execution.splits(stage);admit_view(stage,role,view,actual)
            bad=deepcopy(actual);bad['train_ids'].append(view)
            with self.assertRaises(ValueError):admit_view(stage,role,view,bad)
            with self.assertRaises(ValueError):admit_view(stage,role,actual['train_ids'][0],actual)
        with self.assertRaises(ValueError):admit_view('geometry','validation','58',execution.splits('material'))
    def test_schema_validates_failures_and_rejects_false_completion(self):
        import jsonschema
        schema=json.loads((ROOT/execution.SCHEMA).read_text());jsonschema.Draft7Validator.check_schema(schema)
        with self.assertRaises(jsonschema.ValidationError):jsonschema.validate({'status':'complete'},schema)
        # Real failure logging is tested in a cold process, with native backend replaced.
        script='''
import json,os,sys
from pathlib import Path
from unittest.mock import patch
from infrastructure import lifecycle_execution as e
from infrastructure.stage_sources import PINS,ROOT
data,thesis,out=map(Path,sys.argv[1:]);p=e.plan('geometry','save',PINS['geometry'],data,thesis,out)
before=os.getcwd(),os.environ.get('HOME')
def failed(*a):print('explicit CPU backend failure');raise RuntimeError('fixture')
with patch.object(e,'runtime_guards'),patch.object(e,'native',failed):
 try:e.execute(p,'fixture',e.IMAGE,'0'*40,True)
 except RuntimeError:pass
 else:raise AssertionError('failure swallowed')
assert before==(os.getcwd(),os.environ.get('HOME'))
r=json.loads((out/'lifecycle.json').read_text());assert r['status']=='failed'
import jsonschema
jsonschema.validate(r,json.loads((ROOT/e.SCHEMA).read_text()))
assert 'explicit CPU backend failure' in (out/'logs/console.log').read_text()
assert not list((out/'checkpoints').iterdir())
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.data),str(self.thesis),str(self.output)],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipIf(torch is None,'CPU Torch unavailable')
class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def simple(self,stage):
        from test_g_lifecycle import compile_nodes
        science,_=baseline(stage,PINS[stage]);scope={'os':os,'torch':torch}
        compile_nodes(science/'train/trainer.py',{'_save_model','_load_model'},scope,cls='Trainer')
        base=type('OriginalMethods',(),{k:scope[k] for k in ['_save_model','_load_model']})
        lr={'np':np};exec(compile((science/'train/lr_common_manager.py').read_text(),str(science/'train/lr_common_manager.py'),'exec'),lr)
        schedule=lr['WarmUpCosLR']({} if stage=='geometry' else {'end_warm':1000,'end_iter':100000})
        model=torch.nn.Linear(2,1);optimizer=schedule.construct_optimizer(torch.optim.Adam,model)
        return SimpleNamespace(network=model,optimizer=optimizer,lr_manager=schedule,pth_fn=str(self.root/(stage+'.pth')),
                               best_pth_fn=str(self.root/(stage+'-best.pth'))),base
    def saved(self,stage):
        t,base=self.simple(stage)
        for i in [0,1]:
            t.lr_manager(t.optimizer,i);t.optimizer.zero_grad();t.network(torch.ones((2,2))).sum().backward();t.optimizer.step()
        proof=original_save(t,base,t.pth_fn,torch,np);return t,base,proof
    def test_both_original_formats_roundtrip_all_model_moments_counters_groups_and_LR(self):
        for stage in PINS:
            with self.subTest(stage=stage):
                donor,base,proof=self.saved(stage);receiver,_=self.simple(stage);receiver.pth_fn=donor.pth_fn
                best,step,report=restore_original(receiver,base,Path(donor.pth_fn),proof['sha256'],proof,torch,np,True)
                self.assertEqual((best,step),(0.,2));self.assertTrue(report['exact_restoration'])
                self.assertEqual(state_record(donor,np),state_record(receiver,np));self.assertEqual(digest(donor.pth_fn),proof['sha256'])
                expected=5e-4*(2/5000) if stage=='geometry' else 1e-6
                self.assertEqual(receiver.lr_manager(receiver.optimizer,2),expected)
    def test_wrong_hash_and_untrusted_input_rejected_before_deserialization(self):
        donor,base,proof=self.saved('geometry');receiver,_=self.simple('geometry');receiver.pth_fn=donor.pth_fn
        for sha,trusted in [('0'*64,True),(proof['sha256'],False)]:
            with patch.object(torch,'load',side_effect=AssertionError('Must not deserialize')):
                with self.assertRaises(ValueError):restore_original(receiver,base,Path(donor.pth_fn),sha,proof,torch,np,trusted)
        self.assertFalse(receiver.optimizer.state)
    def test_invalid_or_incompatible_payload_rejected_before_live_restore(self):
        donor,base,proof=self.saved('material');path=Path(donor.pth_fn)
        good=torch.load(path,map_location='cpu');bad_cases=[]
        x=deepcopy(good);del x['optimizer_state_dict'];bad_cases.append(x)
        x=deepcopy(good);x['step']=3;bad_cases.append(x)
        x=deepcopy(good);x['network_state_dict']['weight']=torch.zeros((2,2));bad_cases.append(x)
        x=deepcopy(good);x['optimizer_state_dict']['state'][0]['exp_avg'].fill_(float('nan'));bad_cases.append(x)
        x=deepcopy(good);x['optimizer_state_dict']['param_groups'][0]['eps']=1e-8;bad_cases.append(x)
        x=deepcopy(good);x['optimizer_state_dict']['state'][0]['step']+=1;bad_cases.append(x)
        for x in bad_cases:
            receiver,_=self.simple('material');receiver.pth_fn=str(path);torch.save(x,path);before=state_record(receiver,np)
            with patch.object(receiver.network,'load_state_dict',side_effect=AssertionError('No partial weights')),patch.object(receiver.optimizer,'load_state_dict',side_effect=AssertionError('No unsafe optimizer restore')):
                with self.assertRaises(ValueError):restore_original(receiver,base,path,digest(path),proof,torch,np,True)
            self.assertEqual(before,state_record(receiver,np))
    def test_shared_decorator_wrapper_is_allowed_but_genuine_forbidden_body_is_not(self):
        class Forbidden:
            @torch.no_grad()
            def __call__(self):raise AssertionError('Should be blocked before body')
        parameter=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.Adam([parameter]);parameter.grad=torch.ones(1)
        guard=ForbidCalls([Forbidden.__call__]);step=torch.no_grad()(optimizer.step)
        self.assertIs(step.__code__,Forbidden.__call__.__code__)
        with guard:step()
        with self.assertRaises(SignatureError):
            with ForbidCalls([Forbidden.__call__]):Forbidden()()
        self.assertIsNone(sys.getprofile())


@unittest.skipIf(torch is None,'CPU Torch unavailable')
class OriginalLoopTests(unittest.TestCase):
    def run_cycle(self,stage,fail=False):
        from training_smoke_fixtures import cpu_rng
        from test_dual_baselines import GBoundedTests
        from test_material_smoke import material_fixture
        if stage=='geometry':f=GBoundedTests().fixture()
        else:
            f=material_fixture();old=f.Shader.__init__
            def init(self,*a,**kw):
                old(self,*a,**kw)
                if not self.cfg['human_lights']:del self.human_light
            f.Shader.__init__=init
        science,_=baseline(stage,PINS[stage]);cfg=yaml.safe_load((science/('configs/shape/syn/bell.yaml' if stage=='geometry' else 'configs/material/syn/bell.yaml')).read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';thesis=root/'empty';data.mkdir();thesis.mkdir()
            resolver=PathResolver({'datasets':data,'thesis':thesis},root/'out')
            run=RunStore(resolver).create('donor');p={'stage':stage,'action':'save','checkpoint':None}
            report={'mesh_identity':{'vertices':3,'triangles':1},'ray_splits':{'train':['1','2'],'validation':['0'],'nvs':['3']}}
            def bind(probe):
                if stage=='material':probe.bind_calls(f.Renderer,f.Shader,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],f.Trainer,f.Evaluator)
            with runtime_environment(run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
                donor=execution.io_trainer(f.Trainer,p,run,report,torch,np)(cfg)
                probe=(GTrainingProbe if stage=='geometry' else BellProbe)(donor,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True});bind(probe)
                control=(GTwoUpdateTrace if stage=='geometry' else TwoUpdateTrace)(donor,probe,science/'train/trainer.py')
                if stage=='material':
                    with probe.observe_calls():control.run()
                else:control.run()
                proof=original_save(donor,f.Trainer,run.checkpoints/'model.pth',torch,np)
            resumed_run=RunStore(resolver).create('receiver')
            restored={'mesh_identity':report['mesh_identity'],'ray_splits':report['ray_splits'],'producer_sampling':report['updates'][-1]['sampling']}
            p.update(action='restore',checkpoint=run.checkpoints/'model.pth',checkpoint_sha=proof['sha256'],producer={'checkpoint':proof})
            with runtime_environment(resumed_run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
                receiver=execution.io_trainer(f.Trainer,p,resumed_run,restored,torch,np)(cfg)
                if fail:
                    init=receiver._init_network
                    def failing():
                        init()
                        def forward(data):raise RuntimeError('explicit CPU resumed forward failure')
                        receiver.network.forward=forward
                    receiver._init_network=failing
                probe=(GResumeProbe if stage=='geometry' else AResumeProbe)(receiver,torch,np,restored,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True});bind(probe)
                control=ReloadTrace(receiver,probe,science/'train/trainer.py',stage)
                try:
                    if stage=='material':
                        with probe.observe_calls():control.run()
                    else:control.run()
                except RuntimeError:
                    self.assertTrue(fail);self.assertEqual(control.completed,0)
                    self.assertTrue(f.writers[-1].closed);self.assertTrue(f.loaders[-2].iterator.closed)
                    self.assertFalse(list(resumed_run.checkpoints.iterdir()))
                    self.assertIsNone(sys.gettrace());self.assertIsNone(sys.getprofile());return
            self.assertEqual(control.completed,1);self.assertTrue(restored['restoration']['exact_restoration'])
            self.assertEqual(restored['updates'][0]['step'],2);self.assertGreater(restored['updates'][0]['changed_elements'],0)
            self.assertEqual(restored['updates'][0]['lr'],5e-4*(2/5000) if stage=='geometry' else 1e-6)
            self.assertFalse(restored['continuation']['uninterrupted_bitwise_equivalence_proven'])
            self.assertFalse(list(resumed_run.checkpoints.iterdir()))
            self.assertTrue(f.writers[-1].closed);self.assertTrue(f.loaders[-2].iterator.closed)
            self.assertIsNone(sys.gettrace());self.assertIsNone(sys.getprofile())
    def setUp(self):
        states=random.getstate(),np.random.get_state(),torch.get_rng_state()
        self.addCleanup(lambda:(random.setstate(states[0]),np.random.set_state(states[1]),torch.set_rng_state(states[2])))
    def test_original_G_prefix_save_reload_and_one_post_restore_update(self):self.run_cycle('geometry')
    def test_original_A_prefix_save_reload_and_one_post_restore_update(self):self.run_cycle('material')
    def test_post_restore_failure_stops_and_closes_owned_resources(self):
        for stage in PINS:
            with self.subTest(stage=stage):self.run_cycle(stage,fail=True)


class MetricTests(unittest.TestCase):
    def test_original_float32_PSNR_and_independent_scaling_comparison(self):
        from test_g_lifecycle import compile_nodes
        scope={'np':np};compile_nodes(ROOT/'scientific_sources/material_a/network/metrics.py',{'compute_psnr'},scope)
        pred=np.linspace(-.1,1.1,12*12*3,dtype=np.float32).reshape(12,12,3);gt=pred[::-1].copy()
        encode=lambda v:np.clip(v*255,0,255).astype(np.uint8)
        original={'psnr':scope['compute_psnr'](encode(gt),encode(pred)),'ssim':.75}
        def ssim(a,b,**kwargs):
            self.assertEqual(kwargs,{'win_size':11,'channel_axis':2,'data_range':255});return .75
        report,rgb,truth=metric_comparison(pred,gt,original,np,ssim)
        self.assertTrue(report['passed']);np.testing.assert_array_equal(rgb,encode(pred));np.testing.assert_array_equal(truth,encode(gt))
        bad={**original,'psnr':original['psnr']+1}
        self.assertFalse(metric_comparison(pred,gt,bad,np,ssim)[0]['passed'])
        equal={'psnr':float('inf'),'ssim':.75}
        self.assertTrue(metric_comparison(gt,gt,equal,np,ssim)[0]['passed'])

    @unittest.skipIf(torch is None,'CPU Torch unavailable')
    def test_original_A_test_step_black_background_channels_and_metric_body(self):
        from test_material_smoke import compile_members
        from test_g_lifecycle import compile_nodes
        source=ROOT/'scientific_sources/material_a';h=w=12;calls=[]
        info={'imgs':torch.ones((1,3,h,w))*.25}
        scope={'torch':torch,'np':np,'imgs_info_slice':lambda table,ids:table}
        method=compile_members(source/'network/renderer.py','NeROMaterialRenderer',{'test_step'},scope)['test_step']
        hit=torch.ones(h*w,dtype=torch.bool);hit[0]=False
        def rays(table,*args):
            calls.append('nvs' if table is nvs else 'validation')
            return {'rays_o':torch.ones((h*w,3)),'rays_d':torch.ones((h*w,3)),
                    'inters':torch.ones((h*w,3)),'normals':torch.ones((h*w,3)),
                    'human_poses':torch.ones((h*w,3,4)),'rgb':torch.ones((h*w,3))*.25,'hit_mask':hit}
        def shade(pts,*args):
            return {name:torch.ones((len(pts),width))*.5 for name,width in [('rgb_pr',3),('specular_light',3),('diffuse_light',3),
                ('kd',3),('diffuse_color',3),('specular_color',3),('f_d',3),('f_s_sum',3),('L_spec',3),('tangents',3),
                ('bitangents',3),('normals',3),('metallic',1),('mx',1),('my',1)]}
        nvs=deepcopy(info)
        owner=SimpleNamespace(test_imgs_info=info,nvs_imgs_info=nvs,is_nerf=False,_construct_ray_batch=rays,
                              cfg={'test_ray_num':256,'n_lobes':1},shade=shade)
        with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
            output=method(owner,0);method(owner,0,is_nvs=True)
        self.assertEqual(calls,['validation','nvs']);self.assertEqual(tuple(output['rgb_pr'].shape),(12,12,3))
        self.assertTrue((output['rgb_pr'][0,0]==0).all());self.assertTrue((output['rgb_gt'][0,0]==0).all())
        self.assertEqual(float(info['imgs'][0,0,0,0]),.25);self.assertFalse(output['alpha'].any());self.assertNotIn('rotation',output)
        scope={'np':np};compile_nodes(source/'network/metrics.py',{'compute_psnr'},scope)
        scope.update(color_map_backward=lambda v:np.clip(v*255,0,255).astype(np.uint8),
                     structural_similarity=lambda *args,**kw:.8,Path=Path,imsave=lambda *args:None,
                     concat_images_list=lambda *args,**kw:args[0])
        metric=compile_members(source/'network/metrics.py','MaterialRenderMetrics',{'__call__'},scope)['__call__']
        with tempfile.TemporaryDirectory() as directory:
            before=os.getcwd()
            try:
                os.chdir(directory);values=metric(None,output,{},1,data_index=0,model_name='fixture')
            finally:os.chdir(before)
        measured={k:float(v[0]) for k,v in values.items()}
        comparison,_,_=metric_comparison(output['rgb_pr'].numpy(),output['rgb_gt'].numpy(),measured,np,lambda *a,**k:.8)
        self.assertTrue(comparison['passed'])


if __name__=='__main__':unittest.main()
