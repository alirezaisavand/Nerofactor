"""CPU source/namespace/contracts and real toy autograd; no native models or inputs."""
import ast
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
except ImportError:torch=None

from infrastructure.stage_sources import ROOT,PINS,baseline,digest,admit_checkpoint_identity
from infrastructure.stage_execution import plan,trainer_io,execute
from infrastructure.stage_workspace import prepare_view,source_records,handoff
from infrastructure.g_training import boundaries,GTrainingProbe,GTwoUpdateTrace
from infrastructure.runs import RunStore
from infrastructure.bell_inference import runtime_environment
from infrastructure.evidence_policy import matrix,require_target_acceptance
from infrastructure.signatures import SignatureError
from infrastructure.g_comparison import validate,sequence


def node(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
def module_scope(path):
 scope={};exec(compile(path.read_text(),str(path),'exec'),scope);return scope


class SourceTests(unittest.TestCase):
    def test_all_snapshot_bytes_match_pinned_git_objects(self):
        for stage,commit in PINS.items():
            root,source=baseline(stage,commit)
            for name,record in source['files'].items():
                self.assertEqual((root/name).read_bytes(),subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT),name)
    def test_foreign_selection_and_missing_module_cannot_fall_back(self):
        for stage in ['geometry','material']:
            with self.assertRaises(SignatureError):baseline(stage,PINS['material' if stage=='geometry' else 'geometry'])
        script='''
import sys
from infrastructure.stage_sources import activate,PINS,ROOT
stage=sys.argv[1]
other='material' if stage=='geometry' else 'geometry'
root,_=activate(stage,PINS[stage])
from train.lr_common_manager import WarmUpCosLR
assert WarmUpCosLR({}).learning_rate_alpha==(.05 if stage=='geometry' else .2)
try: import train.module_that_only_could_fall_back
except ModuleNotFoundError: pass
else: raise AssertionError('fallback')
try:activate(other,PINS[other])
except ValueError:pass
else:raise AssertionError('mixed stages')
assert __import__('pathlib').Path(__import__('train.lr_common_manager',fromlist=['']).__file__).resolve()==root/'train/lr_common_manager.py'
'''
        for stage in PINS:
            r=subprocess.run([sys.executable,'-B','-c',script,stage],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_preloaded_scientific_namespace_rejected(self):
        script="from infrastructure.stage_sources import activate,PINS\nimport utils\nactivate('geometry',PINS['geometry'])"
        r=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('already imported',r.stderr)
    def test_G_original_split_and_rng(self):
        root,_=baseline('geometry',PINS['geometry']);f=node(root/'dataset/database.py','get_database_split')
        scope={'random':random,'BaseDatabase':object,'read_pickle':lambda p:(['test-fixture'],['train-fixture'])}
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(root/'dataset/database.py'),'exec'),scope)
        state=random.getstate();self.addCleanup(lambda:random.setstate(state))
        ids=[str(i) for i in range(128)];database=SimpleNamespace(get_img_ids=lambda:ids)
        train,val=scope[f.name](database)
        expected=random.Random(6033)
        self.assertEqual(random.getstate(),expected.getstate());self.assertEqual(ids,[str(i) for i in range(128)])
        self.assertEqual(train,ids[1:]);self.assertEqual(val,['0'])
        before=random.getstate();self.assertEqual(scope[f.name](database,'test'),(['train-fixture'],['test-fixture']))
        self.assertEqual(random.getstate(),before)
    def test_stage_optimizer_and_scheduler_are_actual_distinct_sources(self):
        for stage,eps,ams,alpha in [('geometry',1e-8,False,.05),('material',1e-4,True,.2)]:
            root,_=baseline(stage,PINS[stage]);s=module_scope(root/'train/lr_common_manager.py');m=s['WarmUpCosLR']({})
            captured={}
            def opt(params,**kwargs):captured.update(kwargs);return SimpleNamespace(param_groups=[kwargs])
            out=m.construct_optimizer(opt,SimpleNamespace(parameters=lambda:[]))
            self.assertEqual(captured.get('eps',1e-8),eps);self.assertEqual(captured.get('amsgrad',False),ams)
            self.assertEqual(m.learning_rate_alpha,alpha)
            self.assertEqual(m(out,0),0);self.assertEqual(m(out,1),5e-4*(1/5000))
            if stage=='geometry':self.assertAlmostEqual(m(out,300000),5e-4*.05)
            else:self.assertEqual(m(out,100000),m(out,300000))
    def test_G_loss_schedule_and_absence_of_clipping(self):
        g,_=baseline('geometry',PINS['geometry']);_,points=boundaries(g/'train/trainer.py')
        self.assertNotIn('before_clip',points)
        scope=module_scope(g/'network/loss.py')
        loss=scope['OpacityLoss']({});self.assertEqual(loss.get_opacity_weight(0),.001)
        self.assertEqual(loss.get_opacity_weight(20000),.001);self.assertAlmostEqual(loss.get_opacity_weight(50000),.01)
        self.assertEqual(scope['name2loss']['opacity'],scope['OpacityLoss'])
    def test_A_shader_math_and_G_geometry_source_are_exact(self):
        for stage in PINS:
            root,source=baseline(stage,PINS[stage])
            for path in ['network/field.py','network/renderer.py','network/loss.py','utils/base_utils.py','utils/raw_utils.py','train/train_valid.py']:
                self.assertEqual(digest(root/path),source['files'][path]['sha256'])
        a,_=baseline('material',PINS['material']);text=(a/'network/field.py').read_text()
        self.assertIn('class MCShadingNetwork',text);self.assertIn('sample_aniso_ggx_directions',text)
        render=node(a/'network/renderer.py','NeROMaterialRenderer')
        shader=next(n for n in render.body if isinstance(n,ast.FunctionDef) and n.name=='_init_shader')
        self.assertIn("self.cfg['shader_cfg']",ast.get_source_segment((a/'network/renderer.py').read_text(),shader))
    def test_evidence_supersession_and_checkpoint_gates(self):
        self.assertEqual(len(matrix()['entries']),12)
        require_target_acceptance('material-infrastructure-CPU-A','material','legacy-A-infrastructure-contracts')
        for item in ['two-update-R-A','image58-inference-A','camera0-historical-diagnostic-A','checkpoint-architecture-A']:
            with self.assertRaises(SignatureError):require_target_acceptance(item,'geometry','native-training')
        for record in [None,{'stage':'geometry','scientific_commit':PINS['material']},{}]:
            with self.assertRaises(SignatureError):admit_checkpoint_identity('geometry',PINS['geometry'],record)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name)
        self.data=self.root/'datasets';self.thesis=self.root/'thesis';self.out=self.root/'out'
        for p in [self.data/'GlossySynthetic/bell',self.data/'GlossyReal/bear',self.thesis]:p.mkdir(parents=True)
        (self.data/'GlossySynthetic/bell/0.png').write_bytes(b'fixture-not-decoded')
    def planning(self,arm='R',**kw):
        return plan('geometry',PINS['geometry'],arm,self.data,self.thesis,self.out,**kw)
    def test_independent_config_and_no_rng_allocation(self):
        before=random.getstate()
        with patch('infrastructure.config.load_config',side_effect=AssertionError('reference cannot use schema')):r=self.planning()
        c=self.planning('C',cfg_path=ROOT/'configs/experiments/geometry_g_bell_v1.yaml',config_root=ROOT)
        self.assertEqual(r['configuration'],c['configuration']);self.assertEqual(r['raw'],c['raw'])
        self.assertEqual(before,random.getstate());self.assertFalse(self.out.exists())
    def test_unsafe_and_wrong_stage_config_before_execution(self):
        with self.assertRaises(ValueError):plan('geometry',PINS['material'],'R',self.data,self.thesis,self.out)
        with self.assertRaises(ValueError):plan('geometry',PINS['geometry'],'R',self.data,self.thesis,self.data/'out')
        with self.assertRaises(ValueError):self.planning('C',cfg_path=ROOT/'configs/experiments/geometry.example.yaml',config_root=ROOT)
        with self.assertRaises(ValueError):self.planning('C',cfg_path=ROOT/'configs/experiments/geometry_g_bell_v1.yaml',config_root=ROOT/'configs')
    def test_cache_view_isolated_and_originals_unchanged(self):
        p=self.planning();run=RunStore(p['resolver']).create('fixture');before=source_records(p['selected'].dataset/'bell')
        view,family,records=prepare_view(run,p['selected'].dataset,'bell',p['science'],p['configuration'],{'scientific_commit':PINS['geometry']})
        self.assertEqual((family/'bell/0.png').read_bytes(),b'fixture-not-decoded')
        (family/'bell/cache.pkl').write_bytes(b'derived-fixture')
        self.assertFalse((self.data/'GlossySynthetic/bell/cache.pkl').exists());self.assertEqual(before,source_records(p['selected'].dataset/'bell'))
        self.assertIn(run.root,(family/'bell/cache.pkl').resolve().parents)
        with self.assertRaises(ValueError):RunStore(p['resolver']).create('fixture')
        (self.data/'GlossySynthetic/bell/unsafe').symlink_to(self.thesis,target_is_directory=True)
        with self.assertRaises(SignatureError):source_records(p['selected'].dataset/'bell')
    def test_runtime_failure_precedes_allocation_and_native_imports(self):
        p=self.planning()
        with patch('infrastructure.stage_execution.native_guards',side_effect=SignatureError('blocked')),patch('infrastructure.stage_execution.native_run') as native:
            with self.assertRaises(SignatureError):execute(p,'fixture','image','0'*40)
            native.assert_not_called();self.assertFalse(self.out.exists())
    def test_material_selects_pinned_A_and_explicit_mesh_without_execution(self):
        mesh=self.thesis/'mesh.ply';mesh.write_bytes(b'CPU fixture mesh')
        p=plan('material',PINS['material'],'C',self.data,self.thesis,self.out,
               ROOT/'configs/experiments/material_a_bear_v1.yaml',ROOT,{'geometry':self.thesis})
        self.assertEqual(p['selected'].mesh,mesh)
        self.assertEqual(p['source']['algorithm'],'anisotropic-material-A')
        self.assertEqual(p['configuration']['network'],'material');self.assertFalse(self.out.exists())
    def test_whole_orchestration_with_explicit_cpu_backend_and_failure_logging(self):
        # A new process owns namespace activation; backend is explicitly not a model.
        script='''
import json,os,sys
from pathlib import Path
from infrastructure import stage_execution as execution
from infrastructure.stage_sources import PINS
data,thesis,out,mode=map(str,sys.argv[1:])
p=execution.plan('geometry',PINS['geometry'],'R',Path(data),Path(thesis),Path(out))
execution.native_guards=lambda *a:None # Disposable CPU fixtures, never native execution.
def fake(p,run,report,persist):
 report['CPU_fixture_backend']=True
 Path(report['runtime_configuration']['dataset_dir'],'bell','derived.fixture').write_text('owned derived output')
 print('CPU fixture backend')
 if mode=='failure':raise RuntimeError('explicit CPU fixture failure')
execution.native_run=fake
before=os.getcwd(),os.environ.get('HOME')
try:execution.execute(p,'cpu-fixture','CPU-fixture-image','0'*40)
except RuntimeError:
 assert mode=='failure'
else:assert mode=='success'
assert (os.getcwd(),os.environ.get('HOME'))==before
r=json.loads(Path(out,'stage-run.json').read_text())
assert r['status']==('failed' if mode=='failure' else 'complete')
assert r['CPU_fixture_backend'] is True and r['checkpoint_input'] is None
assert r['source']['scientific_commit']==PINS['geometry'] and r['source']['declared_infrastructure_commit']=='0'*40
assert not Path(data,'GlossySynthetic/bell/derived.fixture').exists()
assert 'CPU fixture backend' in Path(out,'logs/console.log').read_text()
if mode=='failure':assert 'explicit CPU fixture failure' in Path(out,'logs/console.log').read_text()
assert not list(Path(out,'checkpoints').iterdir())
'''
        for mode in ['success','failure']:
            out=self.root/mode;out.mkdir()
            r=subprocess.run([sys.executable,'-B','-c',script,str(self.data),str(self.thesis),str(out),mode],
                              cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_old_execution_CLIs_fail_before_allocation_and_new_preflight_is_cpu_only(self):
        for filename in ['run_experiment.py','run_material_experiment.py']:
            result=subprocess.run([sys.executable,'-B',str(ROOT/filename),'--cfg','absent','--config-root',str(ROOT),
                '--datasets-root',str(self.data),'--thesis-root',str(self.thesis),'--output-root',str(self.out),
                '--run-id','blocked','--execute'],cwd=ROOT,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('1D.5-RESET',result.stderr);self.assertFalse(self.out.exists())
        script='''
import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 assert name.split('.')[0] not in {'torch','numpy','network','train','dataset','open3d','nvdiffrast'},name
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from run_stage_experiment import main
raise SystemExit(main(sys.argv[1:]))
'''
        result=subprocess.run([sys.executable,'-B','-c',script,'--stage','geometry','--scientific-commit',PINS['geometry'],
           '--arm','C','--cfg',str(ROOT/'configs/experiments/geometry_g_bell_v1.yaml'),'--config-root',str(ROOT),
           '--datasets-root',str(self.data),'--thesis-root',str(self.thesis),'--output-root',str(self.out),
           '--run-id','preflight','--validate-only'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(self.out.exists())
    def test_mesh_handoff_rejects_historical_attribution_and_hash_changes(self):
        p=self.planning();mesh=self.thesis/'mesh.ply';mesh.write_bytes(b'fixture-mesh')
        proof=self.thesis/'run.json';proof.write_text(json.dumps({'status':'complete','stage':'geometry','scientific_commit':PINS['geometry'],'run_id':'g-run',
             'source':{'stage':'geometry','algorithm':'geometry-G','scientific_commit':PINS['geometry'],'scientific_files':p['source']['files'],
                       'source_manifest_sha256':digest(ROOT/'scientific_sources/manifest-v1.json')},
             'original_configuration':{'database_name':'syn/bell'},'mesh_outputs':{'mesh.ply':digest(mesh)}}))
        value={'schema_version':1,'geometry_algorithm':'geometry-G','scientific_commit':PINS['geometry'],
               'geometry_run_id':'g-run','geometry_run_manifest_sha256':digest(proof),'geometry_run_manifest':{'root':'thesis','path':'run.json'},
               'mesh_sha256':digest(mesh),'database_name':'syn/bell','coordinates':'fixture-scene-world','scale':1.0,'normals':'mesh winding',
               'orientation':'right-handed fixture','material_preprocessing':'A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write'}
        path=self.thesis/'handoff.json';path.write_text(json.dumps(value));self.assertEqual(handoff(path,mesh,p['resolver'],'syn/bell'),value)
        with self.assertRaises(SignatureError):handoff(path,mesh,p['resolver'],'real/bear/black_800')
        original_proof=proof.read_bytes()
        wrong=json.loads(original_proof);wrong['source']['algorithm']='anisotropic-material-A'
        proof.write_text(json.dumps(wrong))
        wrong_handoff={**value,'geometry_run_manifest_sha256':digest(proof)}
        path.write_text(json.dumps(wrong_handoff))
        with self.assertRaisesRegex(SignatureError,'verified G source context'):handoff(path,mesh,p['resolver'],'syn/bell')
        proof.write_bytes(original_proof);path.write_text(json.dumps(value))
        bad={**value,'scale':float('nan')};path.write_text(json.dumps(bad))
        with self.assertRaises(SignatureError):handoff(path,mesh,p['resolver'],'syn/bell')
        path.write_text(json.dumps(value))
        mesh.write_bytes(b'changed')
        with self.assertRaises(SignatureError):handoff(path,mesh,p['resolver'],'syn/bell')
        value['scientific_commit']='unknown';path.write_text(json.dumps(value))
        with self.assertRaises(SignatureError):handoff(path,mesh,p['resolver'],'syn/bell')


@unittest.skipIf(torch is None,'CPU Torch unavailable; no native fallback')
class GBoundedTests(unittest.TestCase):
    setUp=PathTests.setUp
    planning=PathTests.planning
    def fixture(self):
        from training_smoke_fixtures import trainer_fixture
        f=trainer_fixture();scope=dict(f.Trainer.__init__.__globals__)
        g,_=baseline('geometry',PINS['geometry'])
        original=scope['name2renderer']['shape']
        class ToyG(original):
            def __init__(self,cfg):
                super().__init__(cfg);del self.isolated_geometry;del self.nvs_ids
                self.train_ids=np.asarray([str(i) for i in range(1,128)]);self.test_ids=['0']
                self.cfg={**next(ast.literal_eval(n.value) for n in node(g/'network/renderer.py','NeROShapeRenderer').body
                      if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_cfg' for t in n.targets)),**cfg}
            def forward(self,data):return {k:v*100 for k,v in super().forward(data).items()}
        scope['name2renderer']={'shape':ToyG}
        scope['name2lr_manager']=module_scope(g/'train/lr_common_manager.py')['name2lr_manager']
        exec(compile(ast.Module(body=[node(g/'train/trainer.py','Trainer')],type_ignores=[]),str(g/'train/trainer.py'),'exec'),scope)
        f.Trainer=scope['Trainer'];return f
    def run_G(self,arm,fail=False):
        from training_smoke_fixtures import cpu_rng
        from infrastructure.stage_sources import infrastructure_identity
        from infrastructure.g_training import PROTOCOL
        p=self.planning(arm,cfg_path=ROOT/'configs/experiments/geometry_g_bell_v1.yaml' if arm=='C' else None,
                        config_root=ROOT if arm=='C' else None)
        run=RunStore(p['resolver']).create(arm.lower());f=self.fixture()
        report={'schema_version':1,'protocol':PROTOCOL,'status':'running','stage':'geometry','scientific_commit':PINS['geometry'],
                'run_id':run.run_id,'arm':arm,'route':p['route'],'checkpoint_input':None,'optimizer_calls_completed':0,
                'source':{'stage':'geometry','scientific_commit':PINS['geometry'],'algorithm':'geometry-G','scientific_files':p['source']['files'],
                          'source_manifest_sha256':digest(ROOT/'scientific_sources/manifest-v1.json'),'infrastructure_files':infrastructure_identity()},
                'declared_image_id':'CPU-fixture','legacy_sha256':__import__('hashlib').sha256(p['raw']).hexdigest(),
                'original_configuration':p['configuration'],'input_files':[{'synthetic':'CPU only'}],
                'runtime':{'CPU_only':True},'imported_scientific_modules':{'CPU_standins':True}}
        view,family,_=prepare_view(run,p['selected'].dataset,'bell',p['science'],p['configuration'],report['source'])
        cfg=deepcopy(p['configuration']);cfg['dataset_dir']='/equivalent-CPU-fixture-path'
        report['runtime_configuration']=cfg
        with runtime_environment(run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            os.chdir(view);trainer=trainer_io(f.Trainer,run)(cfg)
            probe=GTrainingProbe(trainer,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_only':True})
            if fail:
                original=trainer._init_network
                def initialize():original();trainer.network.fail_step=1
                trainer._init_network=initialize
            controller=GTwoUpdateTrace(trainer,probe,p['science']/'train/trainer.py')
            if fail:
                with self.assertRaisesRegex(RuntimeError,'fixture forward failure'):controller.run()
            else:
                controller.run();report.update(status='complete',phase='bounded_stop_after_second_optimizer_return')
            self.assertIsNone(sys.gettrace())
        self.assertTrue(f.loaders[0].iterator.closed);self.assertTrue(f.writers[0].closed)
        self.assertFalse(list(run.checkpoints.glob('*.pth')))
        return report,f
    def test_two_actual_G_loop_updates_no_clip_and_independent_repeat_candidate(self):
        states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        self.addCleanup(lambda:(random.setstate(states[0]),np.random.set_state(states[1]),torch.set_rng_state(states[2])))
        reports=[self.run_G(arm)[0] for arm in ['R','R-repeat','C']]
        for report in reports:
            validate(report,report['arm'])
            self.assertGreater(report['updates'][0]['gradient_l2_before_optimizer_cpu64'],.7)
            self.assertFalse(report['updates'][0]['clipping_applied'])
            self.assertEqual(report['updates'][0]['changed_elements'],0)
            self.assertGreater(report['updates'][1]['changed_elements'],0)
        self.assertEqual(sequence(*reports)['result'],'G_regression_passed')
        changed=deepcopy(reports[2]);changed['updates'][1]['losses']['loss_rgb']+=1
        self.assertEqual(sequence(reports[0],reports[1],changed)['result'],'G_candidate_mismatch')
        changed=deepcopy(reports[1]);changed['updates'][0]['rng_after']['python']='different'
        result=sequence(reports[0],changed,reports[2]);self.assertEqual(result['result'],'G_repeatability_failed');self.assertIsNone(result['candidate'])
        changed=deepcopy(reports[0]);changed['scientific_commit']=PINS['material']
        with self.assertRaises(SignatureError):validate(changed,'R')
    def test_failure_shutdown_and_no_implicit_restore(self):
        report,f=self.run_G('R',fail=True)
        self.assertEqual(report['optimizer_calls_completed'],1)
        self.assertNotIn('unexpected_natural_log',f.events)
        self.assertFalse(any(x=='torch_load' for x in f.events))
    def test_pristine_A_material_loop_resource_capture_on_fixture_failure(self):
        # Original A Trainer.run with an explicit CPU model failure, not material rendering.
        from training_smoke_fixtures import trainer_fixture
        from infrastructure.stage_execution import run_material_owned
        f=trainer_fixture();scope=dict(f.Trainer.__init__.__globals__)
        a,_=baseline('material',PINS['material'])
        exec(compile(ast.Module(body=[node(a/'train/trainer.py','Trainer')],type_ignores=[]),str(a/'train/trainer.py'),'exec'),scope)
        p=self.planning();run=RunStore(p['resolver']).create('a-cleanup')
        view,_,_=prepare_view(run,p['selected'].dataset,'bell',p['science'],p['configuration'],{})
        report={}
        with runtime_environment(run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            os.chdir(view);trainer=trainer_io(scope['Trainer'],run)(p['configuration'])
            original=trainer._init_network
            def initialize():original();trainer.network.fail_step=0
            trainer._init_network=initialize
            with self.assertRaisesRegex(RuntimeError,'fixture forward failure'):run_material_owned(trainer,report,lambda:None)
        self.assertIsNone(sys.gettrace());self.assertEqual(report['cleanup_errors'],[])
        self.assertTrue(f.loaders[0].iterator.closed);self.assertTrue(f.writers[0].closed)
