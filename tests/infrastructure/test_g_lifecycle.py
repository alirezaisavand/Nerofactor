"""CPU fixtures only: no original checkpoint/camera pickle or native model loads."""
import ast
from collections import OrderedDict
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
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
try:import torch
except ImportError:torch=None
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.signatures import SignatureError,CHECKPOINT_SHA256
from infrastructure import g_checkpoint as ck
from infrastructure import g_lifecycle as life
from infrastructure.evidence_policy import matrix,require_target_acceptance

G=ROOT/'scientific_sources/geometry_g'


def compile_nodes(path,names,scope,cls=None):
    tree=ast.parse(path.read_text())
    if cls:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


class AdmissionTests(unittest.TestCase):
    def test_admitted_prefix_and_v2_have_only_narrow_scope(self):
        value=life.admitted_prefix()
        self.assertEqual(value['accepted_scopes'],['fresh-two-update-prefix'])
        row=require_target_acceptance('G-two-update-R-R-repeat-C','geometry','fresh-two-update-prefix')
        self.assertEqual(row['accepted_scopes'],value['accepted_scopes'])
        for scope in ['validation','serialization-roundtrip','mesh-extraction','full-training-reproducibility','resume']:
            with self.assertRaises(SignatureError):require_target_acceptance(row['id'],'geometry',scope)
        v1=ROOT/matrix()['supersedes']['path']
        self.assertEqual(digest(v1),matrix()['supersedes']['sha256'])
    def test_archived_reports_and_declared_outputs_still_match(self):
        # JSON and opaque bytes only; no binary/pickle deserialization.
        a=life.admitted_prefix();root=Path('/home/isavand/paper/evidence')
        if not (root/a['reports']['R']['path']).exists():self.skipTest('External archived evidence not present')
        for spec in [*a['reports'].values(),a['final_comparison'],a['repeatability_report']]:
            p=root/spec['path'];self.assertEqual(digest(p),spec['sha256']);self.assertEqual(p.stat().st_size,spec['bytes'])
        for label,spec in a['reports'].items():
            p=root/spec['path'];report=json.loads(p.read_text())
            for name,sha in report['source']['infrastructure_files'].items():
                self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',a['review_baseline']+':'+name],cwd=ROOT)).hexdigest(),sha)
            for name,sha in report['output_hashes'].items():self.assertEqual(digest(p.parent/name),sha)
    def test_source_audit_exact_hashes_and_ast_members(self):
        audit=json.loads((ROOT/'docs/evidence/g-validation-checkpoint-source-audit-v1.json').read_text())
        self.assertEqual(audit['scientific_commit'],PINS['geometry'])
        for name,row in audit['files'].items():
            path=G/name;self.assertEqual(digest(path),row['sha256'])
            self.assertEqual(path.stat().st_size,row['bytes'])
            tree=ast.parse(path.read_text())
            for member,info in row['members'].items():
                node=tree
                for part in member.split('.'):node=next(n for n in node.body if getattr(n,'name',None)==part)
                self.assertEqual(node.lineno,info['first_line']);self.assertEqual(node.end_lineno,info['last_line'])
                self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),info['ast_sha256'])
    def test_original_extraction_selection_and_operations_are_frozen(self):
        text=(G/'extract_mesh.py').read_text()
        self.assertIn('/model.pth',text);self.assertNotIn('model_best.pth',text)
        self.assertIn('default=512',text);self.assertIn('trimesh.Trimesh(vertices, triangles)',text)
        field=(G/'network/field.py').read_text()
        self.assertIn('torch.norm(pts, dim=-1) >= 1.0',field)
        self.assertIn('vertices / (resolution - 1.0)',field)


@unittest.skipIf(torch is None,'CPU Torch unavailable; no native fallback')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        state=torch.get_rng_state();self.addCleanup(lambda:torch.set_rng_state(state))
    def model(self):return torch.nn.Linear(2,2).cpu()
    def saved(self,model):return {'step':ck.STORED_STEP,'best_para':ck.STORED_BEST,'network_state_dict':deepcopy(model.state_dict()),'optimizer_state_dict':{'state':{},'param_groups':[]}}
    def test_trust_hash_and_file_alias_fail_before_torch_load(self):
        path=self.root/'synthetic.pth';torch.save(self.saved(self.model()),path)
        original=torch.load
        with patch.object(torch,'load',wraps=original) as load:
            with self.assertRaises(SignatureError):ck.load_checked(path,digest(path),torch)
            with self.assertRaises(SignatureError):ck.load_checked(path,'0'*64,torch,trusted=True)
            alias=self.root/'alias';alias.symlink_to(path)
            with self.assertRaises(SignatureError):ck.load_checked(alias,digest(path),torch,trusted=True)
            load.assert_not_called()
            result=ck.load_checked(path,digest(path),torch,trusted=True)
            self.assertEqual(result['step'],150000);self.assertEqual(load.call_args.kwargs,{'map_location':'cpu'})
    def test_historical_schema_primitive_provenance(self):
        from inspect_bell_checkpoint import summarize_checkpoint
        state=self.saved(self.model());state['git_commit']='unknown'
        self.assertEqual(ck.historical_schema(state)['stored_step'],150000)
        self.assertEqual(summarize_checkpoint(state,torch)['tensor_count'],2)
        for key,value in [('step',149999),('best_para',float('nan')),('optimizer_state_dict',None)]:
            bad=deepcopy(state);bad[key]=value
            with self.assertRaises(SignatureError):ck.historical_schema(bad)
    def test_strict_exact_load_and_no_partial_load_on_any_mismatch(self):
        model=self.model();saved=deepcopy(model.state_dict());receiver=self.model()
        report=ck.strict_network_load(receiver,saved,torch)
        self.assertEqual(report['result'],'exact_structural_match');self.assertEqual(ck.identities(receiver.state_dict()),ck.identities(saved))
        variants=[]
        bad=deepcopy(saved);bad.pop('bias');variants.append(bad)
        bad=deepcopy(saved);bad['unexpected']=torch.zeros(1);variants.append(bad)
        bad=deepcopy(saved);bad['weight']=torch.zeros(3,2);variants.append(bad)
        bad=deepcopy(saved);bad['weight']=bad['weight'].double();variants.append(bad)
        bad=deepcopy(saved);bad['weight'][0,0]=float('nan');variants.append(bad)
        original=ck.identities(receiver.state_dict())
        for bad in variants:
            with patch.object(receiver,'load_state_dict',wraps=receiver.load_state_dict) as load:
                with self.assertRaises(SignatureError):ck.strict_network_load(receiver,bad,torch)
                load.assert_not_called()
            self.assertEqual(ck.identities(receiver.state_dict()),original)
    def populated(self,eps=1e-8,amsgrad=False):
        model=self.model();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=eps,amsgrad=amsgrad)
        for _ in range(2):optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
        return model,optimizer
    def test_optional_optimizer_load_reports_overrides_without_any_update(self):
        source,opt=self.populated(eps=1e-4,amsgrad=True);target=self.model()
        fresh=torch.optim.Adam(target.parameters(),lr=.001)
        saved=opt.state_dict();saved['param_groups'][0]['lr']=np.float64(1e-7)
        before=ck.identities(target.state_dict())
        with patch.object(fresh,'step',side_effect=AssertionError('No diagnostic updates')):
            report=ck.inspect_optimizer_load(fresh,saved,target,torch)
        self.assertEqual(report['optimizer_updates'],0);self.assertFalse(report['G_continuation_admitted'])
        self.assertEqual(report['scientific_hyperparameter_changes'][0]['eps'],{'G_fresh':1e-8,'loaded':1e-4})
        self.assertEqual(report['scientific_hyperparameter_changes'][0]['amsgrad'],{'G_fresh':False,'loaded':True})
        self.assertEqual(ck.identities(target.state_dict()),before)
        json.dumps(report,allow_nan=False)
    def test_optimizer_preflight_rejects_mismatches_before_load(self):
        model,opt=self.populated();target=self.model();fresh=torch.optim.Adam(target.parameters())
        saved=opt.state_dict();variants=[]
        bad=deepcopy(saved);bad['param_groups'][0]['params'].pop();variants.append(bad)
        bad=deepcopy(saved);bad['state'][0]['exp_avg']=torch.zeros(1);variants.append(bad)
        bad=deepcopy(saved);bad['state'][0]['exp_avg']=bad['state'][0]['exp_avg'].double();variants.append(bad)
        bad=deepcopy(saved);bad['state'][0]['exp_avg'][0,0]=float('nan');variants.append(bad)
        bad=deepcopy(saved);bad['state'][0]['step']=-1;variants.append(bad)
        bad=deepcopy(saved);bad['param_groups'][0]['eps']=-1;variants.append(bad)
        bad=deepcopy(saved);bad['state'][999]={};variants.append(bad)
        bad=deepcopy(saved);bad['state'][0]['extra']=1;variants.append(bad)
        for bad in variants:
            with patch.object(fresh,'load_state_dict',wraps=fresh.load_state_dict) as load:
                with self.assertRaises((SignatureError,ValueError)):ck.inspect_optimizer_load(fresh,bad,target,torch)
                load.assert_not_called()
    def test_original_G_manual_roundtrip_with_populated_CPU_adam(self):
        scope={'torch':torch,'os':os}
        compile_nodes(G/'train/trainer.py',{'_save_model','_load_model'},scope,cls='Trainer')
        Original=type('OriginalGMethods',(),{k:scope[k] for k in ['_save_model','_load_model']})
        model,opt=self.populated();other=self.model();fresh=torch.optim.Adam(other.parameters(),lr=.001)
        donor=SimpleNamespace(network=model,optimizer=opt)
        receiver=SimpleNamespace(network=other,optimizer=fresh)
        path=self.root/'owned.pth'
        with patch.object(fresh,'step',side_effect=AssertionError('No post-load updates')):
            report=ck.manual_roundtrip(donor,receiver,Original,path,2,0.0,torch,np)
        self.assertTrue(report['optimizer_roundtrip']);self.assertGreater(report['optimizer_state_entries'],0)
        self.assertFalse(report['natural_schedule_tested']);self.assertEqual(report['stored_step'],2)
        with self.assertRaises(SignatureError):ck.manual_roundtrip(donor,receiver,Original,path,2,0.0,torch,np)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.data=self.root/'datasets';(self.data/'GlossySynthetic/bell').mkdir(parents=True)
        (self.data/'GlossySynthetic/bell/input.fixture').write_text('untouched')
        self.thesis=self.root/'thesis';self.thesis.mkdir();self.out=self.root/'new-output'
        self.checkpoint=self.thesis/'model/bell_shape/model_best.pth';self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'Opaque fixture, never deserialize')
        self.fixture_sha=digest(self.checkpoint)
    def prepare(self,action='compatibility',**kwargs):
        with patch.object(life,'CHECKPOINT_SHA256',self.fixture_sha):
            return life.prepare(action,PINS['geometry'],self.data,self.thesis,self.out,**kwargs)
    def prior(self):
        _,source=baseline('geometry',PINS['geometry'])
        prior={'protocol':life.PROTOCOLS['compatibility'],'action':'compatibility','status':'complete',
               'scientific_commit':PINS['geometry'],'checkpoint_sha256':self.fixture_sha,'strict_network_loaded':True,
               'network_comparison':{'result':'exact_structural_match'},'source':{'scientific_files':source['files']},
               'infrastructure_files':life.identity()}
        folder=self.root/'prior';folder.mkdir()
        path=folder/'prior.json';path.write_text(json.dumps(prior));return path,prior
    def test_preflight_no_allocation_no_scientific_imports_or_rng_consumption(self):
        rng=random.getstate();p=self.prepare()
        self.assertEqual(p['cfg']['database_name'],'syn/bell');self.assertEqual(random.getstate(),rng)
        self.assertFalse(self.out.exists());self.assertEqual(digest(self.checkpoint),self.fixture_sha)
    def test_hash_source_paths_and_explicit_prior_fail_closed(self):
        with self.assertRaises(SignatureError):life.prepare('compatibility',PINS['material'],self.data,self.thesis,self.out)
        with self.assertRaises(SignatureError):life.prepare('compatibility',PINS['geometry'],self.data,self.thesis,self.out)
        for action in ['optimizer','validation']:
            with self.assertRaises(SignatureError):self.prepare(action)
        prior,value=self.prior();self.prepare('validation',compatibility=prior,compatibility_sha=digest(prior))
        for key,bad in [('status','running'),('strict_network_loaded',False),('scientific_commit',PINS['material']),('infrastructure_files',{})]:
            changed=deepcopy(value);changed[key]=bad;prior.write_text(json.dumps(changed))
            with self.assertRaises(SignatureError):self.prepare('validation',compatibility=prior,compatibility_sha=digest(prior))
        with self.assertRaises(SignatureError):self.prepare(compatibility=prior,compatibility_sha=digest(prior))
    def test_explicit_trust_and_runtime_gates_precede_allocation(self):
        p=self.prepare()
        with patch.object(life,'runtime_guards') as guard,patch.object(life,'native') as native:
            with self.assertRaises(SignatureError):life.execute(p,'fresh',life.IMAGE,'0'*40)
            guard.assert_not_called();native.assert_not_called()
        with patch.object(life,'runtime_guards',side_effect=SignatureError('blocked runtime')),patch.object(life,'native') as native:
            with self.assertRaises(SignatureError):life.execute(p,'fresh',life.IMAGE,'0'*40,trusted=True)
            native.assert_not_called();self.assertFalse(self.out.exists())
    def test_optional_optimizer_acknowledgement_is_required_by_execution_API(self):
        prior,_=self.prior()
        p=self.prepare('optimizer',compatibility=prior,compatibility_sha=digest(prior))
        with patch.object(life,'runtime_guards') as guard,patch.object(life,'native') as native:
            with self.assertRaisesRegex(SignatureError,'optimizer diagnostic acknowledgement'):
                life.execute(p,'fresh',life.IMAGE,'0'*40,trusted=True)
            guard.assert_not_called();native.assert_not_called();self.assertFalse(self.out.exists())
    def test_roundtrip_requires_empty_thesis_and_never_selects_checkpoint(self):
        with self.assertRaises(SignatureError):self.prepare('roundtrip')
        empty=self.root/'empty';empty.mkdir()
        p=life.prepare('roundtrip',PINS['geometry'],self.data,empty,self.out)
        self.assertIsNone(p['checkpoint']);self.assertIsNone(p['prior']);self.assertFalse(self.out.exists())
    def test_output_overlap_and_unused_checkpoint_references_rejected(self):
        from infrastructure.paths import PathError
        for output in [ROOT,ROOT/'owned-looking',self.thesis/'output',self.data]:
            with self.assertRaises((SignatureError,PathError)):
                with patch.object(life,'CHECKPOINT_SHA256',self.fixture_sha):life.prepare('compatibility',PINS['geometry'],self.data,self.thesis,output)
    def test_validate_only_CLI_import_guard(self):
        script='''
import builtins,sys
from infrastructure import g_lifecycle
from infrastructure.stage_sources import digest
g_lifecycle.CHECKPOINT_SHA256=digest(__import__('pathlib').Path(sys.argv[1]))
sys.argv.pop(1)
original=builtins.__import__
def guarded(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','numpy','network','train','dataset','nvdiffrast','open3d'},name
 return original(name,*a,**kw)
builtins.__import__=guarded
from run_g_geometry_diagnostic import main
raise SystemExit(main(sys.argv[1:]))
'''
        args=['--action','compatibility','--scientific-commit',PINS['geometry'],'--run-id','preflight','--validate-only',
              '--datasets-root',str(self.data),'--thesis-root',str(self.thesis),'--output-root',str(self.out)]
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.checkpoint),*args],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('no allocation',result.stdout);self.assertFalse(self.out.exists())
    def test_whole_execution_with_explicit_CPU_fake_and_failure_cleanup(self):
        # New processes isolate scientific namespace activation. No native backend runs.
        script='''
import json,os,sys
from pathlib import Path
from infrastructure import g_lifecycle as life
from infrastructure.stage_sources import PINS,digest
checkpoint,data,thesis,out,mode=map(Path,sys.argv[1:])
life.CHECKPOINT_SHA256=digest(checkpoint)
p=life.prepare('compatibility',PINS['geometry'],data,thesis,out)
life.runtime_guards=lambda *a:None  # Disposable CPU test only.
def fake(p,run,cfg,report,mark):
 mark('CPU_fixture_backend')
 print('CPU fixture backend; no model')
 report['CPU_fixture_backend']=True
 if str(mode)=='failure':raise RuntimeError('explicit CPU fixture failure')
life.native=fake
before=os.getcwd(),os.environ.get('HOME')
try:life.execute(p,'cpu-fixture',life.IMAGE,'0'*40,trusted=True)
except RuntimeError:assert str(mode)=='failure'
else:assert str(mode)=='success'
assert (os.getcwd(),os.environ.get('HOME'))==before
r=json.loads((out/'G-diagnostic.json').read_text())
assert r['status']==('failed' if str(mode)=='failure' else 'complete')
assert r['CPU_fixture_backend'] and not r['natural_schedule_tested']
assert not list((out/'checkpoints').iterdir())
assert 'CPU fixture backend' in (out/'logs/console.log').read_text()
assert digest(checkpoint)==life.CHECKPOINT_SHA256
before_report=(out/'G-diagnostic.json').read_bytes()
# The actual mounted allocator rejects the occupied output. Namespace selection
# intentionally cannot be repeated in one process, so test the allocator directly.
from infrastructure.runs import RunStore
try:RunStore(p['resolver'],layout='mounted').create('cpu-fixture')
except ValueError:pass
else:raise AssertionError('Run collision was not rejected')
assert (out/'G-diagnostic.json').read_bytes()==before_report
'''
        for mode in ['success','failure']:
            output=self.root/mode;output.mkdir()
            result=subprocess.run([sys.executable,'-B','-c',script,str(self.checkpoint),str(self.data),str(self.thesis),str(output),mode],
                                  cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipIf(torch is None,'CPU Torch unavailable; no native fallback')
class OriginalValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        self.addCleanup(lambda:(random.setstate(states[0]),np.random.set_state(states[1]),torch.set_rng_state(states[2])))
    def evaluate(self,observed=True,fail=False):
        from camera_diagnostic_fixtures import fixture_renderer
        from infrastructure.g_validation_observer import ReadAudit
        from skimage.metrics import structural_similarity
        from PIL import Image
        with fixture_renderer() as f:
            # Only disposable fixture classes are adapted. Production G methods
            # are compiled unchanged; native operators/I/O are explicit CPU fakes.
            scope=dict(f.Renderer.forward.__globals__)
            compile_nodes(G/'network/renderer.py',{'build_imgs_info','imgs_info_slice','imgs_info_to_torch'},scope)
            compile_nodes(G/'network/renderer.py',{'forward','test_step'},scope,cls='NeROShapeRenderer')
            f.Renderer.forward=scope['forward'];f.Renderer.test_step=scope['test_step']
            for cell in f.Renderer.render.__closure__:
                if isinstance(cell.cell_contents,type) and cell.cell_contents.__name__=='SDF':
                    cell.cell_contents.gradient=compile_nodes(G/'network/field.py',{'gradient'},{'torch':torch},cls='SDFNetwork')['gradient']
            class DB:
                def get_img_ids(self):return [str(i) for i in range(128)]
                def get_image(self,img_id):return np.full((48,48,3),64 if img_id=='0' else 192,np.uint8)
                def get_pose(self,img_id):return np.zeros((3,4),np.float32)
                def get_K(self,img_id):return np.eye(3,dtype=np.float32)
                def get_mask(self,img_id):return np.ones((48,48),bool)
                def get_depth(self,img_id):return np.ones((48,48),np.float32),np.ones((48,48),bool)
            def no_pickle(path):raise AssertionError('No fixture pickle loads')
            splits=compile_nodes(G/'dataset/database.py',{'get_database_split'},
                {'random':random,'BaseDatabase':object,'read_pickle':no_pickle})['get_database_split']
            dummy_scope=compile_nodes(G/'dataset/train_dataset.py',{'DummyDataset'},
                {'Dataset':object,'parse_database_name':lambda *a:DB(),'get_database_split':splits})
            model=f.model;model.database=DB();del model.nvs_ids;model.train_batch_i=0
            class Iterator:
                def __init__(self,dataset):self.dataset=dataset;self.i=0;self.closed=False
                def __next__(self):
                    if self.i>=len(self.dataset):raise StopIteration
                    item=self.dataset[self.i];self.i+=1;return item
                def _shutdown_workers(self):self.closed=True
            class Loader:
                def __iter__(self):self.iterator=Iterator(self.dataset);return self.iterator
            class Progress:
                bars=[]
                def __init__(self,items):self.items=items;self.closed=False;self.bars.append(self)
                def __iter__(self):return iter(self.items)
                def close(self):self.closed=True
            loader=Loader()
            class TorchCPU:
                cuda=SimpleNamespace(empty_cache=lambda:None)
                def __getattr__(self,name):return getattr(torch,name)
            metric_scope={'np':np,'Path':Path,'Loss':object,'structural_similarity':structural_similarity,
                          'imsave':lambda p,image:Image.fromarray(image).save(str(p))}
            # Original concatenation; only cv2's border operator has a CPU stand-in.
            metric_scope['cv2']=SimpleNamespace(BORDER_CONSTANT=0,copyMakeBorder=lambda a,t,b,l,r,**kw:
                                                np.pad(a,((t,b),(l,r),(0,0)),constant_values=0))
            compile_nodes(G/'utils/base_utils.py',{'color_map_backward'},metric_scope)
            compile_nodes(G/'utils/draw_utils.py',{'concat_images','concat_images_list'},metric_scope)
            compile_nodes(G/'network/metrics.py',{'compute_psnr','process_key_img','get_key_images','draw_materials','ShapeRenderMetrics','psnr'},metric_scope)
            tools=compile_nodes(G/'train/train_tools.py',{'to_cuda'},{})
            valid_scope={'torch':TorchCPU(),'np':np,'time':__import__('time'),'tqdm':Progress,
                         'to_cuda':tools['to_cuda'],'name2key_metrics':{'psnr':metric_scope['psnr']}}
            compile_nodes(G/'train/train_valid.py',{'ValidationEvaluator'},valid_scope)
            evaluator=valid_scope['ValidationEvaluator']({'key_metric_name':'psnr'})
            metric=metric_scope['ShapeRenderMetrics']({})
            def metric_call(*a,**kw):
                if fail:raise RuntimeError('CPU metric fixture failure')
                return metric(*a,**kw)
            output_dir=self.root/('observed-fail' if fail else ('observed' if observed else 'plain'));output_dir.mkdir()
            old=os.getcwd();os.chdir(output_dir)
            try:
                audit=ReadAudit(DB,f.Renderer,scope['build_imgs_info'],splits,no_pickle,
                    forbidden=[torch.optim.Adam.step,torch.Tensor.backward],loader=loader,progress_class=Progress)
                from contextlib import nullcontext
                with audit if observed else nullcontext():
                    loader.dataset=dummy_scope['DummyDataset']({},False)
                    model.train_ids,model.test_ids=splits(model.database)
                    model.test_imgs_info=scope['imgs_info_to_torch'](scope['build_imgs_info'](model.database,['0'],False))
                    before=ck.identities(model.state_dict());rng=random.getstate()
                    audit.phase='evaluation'
                    values,key=evaluator(model,[metric_call],loader,149999,'bell_shape',val_set_name='val')
                    self.assertEqual(random.getstate(),rng)
                self.assertEqual(ck.identities(model.state_dict()),before)
                self.assertTrue(all(event[-1] for event in f.events if isinstance(event,tuple) and event[0]=='sdf_gradient'))
                if observed:
                    record=audit.record(model)
                    self.assertEqual(record['database_source_id'],'0');self.assertEqual(record['original_G_splits']['test_ids'],['0'])
                    self.assertTrue(loader.iterator.closed);self.assertTrue(all(bar.closed for bar in Progress.bars))
                self.assertIsNone(sys.getprofile());self.assertEqual(model.train_batch_i,0)
                self.assertTrue((output_dir/'data/train_vis/bell_shape-val/149999-index-0.jpg').is_file())
                expected_gt=np.full((12,12,3),64,np.uint8);expected_pr=np.full((12,12,3),76,np.uint8)
                self.assertEqual(key,metric_scope['compute_psnr'](expected_gt,expected_pr))
                self.assertEqual(values['ssim'][0],structural_similarity(expected_gt,expected_pr,win_size=11,channel_axis=2,data_range=255))
                return values,key,audit
            finally:
                os.chdir(old)
                if observed and hasattr(loader,'iterator'):
                    self.assertTrue(loader.iterator.closed);self.assertIsNone(sys.getprofile())
    def test_original_G_forward_sdf_grad_metrics_evaluator_and_observer_parity(self):
        with redirect_stdout(io.StringIO()):
            plain=self.evaluate(False);observed=self.evaluate(True)
        for key in plain[0]:np.testing.assert_array_equal(plain[0][key],observed[0][key])
        self.assertEqual(plain[1],observed[1]);self.assertEqual(len(observed[2].forwards),1)
    def test_original_evaluator_failure_closes_worker_and_restores_profile(self):
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'CPU metric fixture failure'):self.evaluate(True,True)
        self.assertIsNone(sys.getprofile())
    def test_observer_blocks_update_before_mutation_and_draws_no_rng(self):
        from infrastructure.g_validation_observer import ReadAudit
        class DB:
            def get_img_ids(self):return ['0']
            def get_image(self,img_id):return None
            get_pose=get_K=get_mask=get_depth=get_image
        class Renderer:
            def forward(self,data):return {}
        def build(database,img_ids):return {}
        def split(database,split_type='validation'):return [],['0']
        def read(path):return None
        def forbidden():raise AssertionError('The forbidden body must not run')
        state=random.getstate();torch_state=torch.get_rng_state().clone()
        with self.assertRaisesRegex(SignatureError,'Forbidden'):
            with ReadAudit(DB,Renderer,build,split,read,forbidden=[forbidden]):forbidden()
        self.assertIsNone(sys.getprofile());self.assertEqual(state,random.getstate());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
    def test_camera_id_guard_rejects_wrong_depth_and_three_way_split(self):
        from infrastructure.g_validation_observer import ReadAudit
        # Construct minimal metadata; record must reject ID58 even if iterator is0.
        class DB:
            def get_img_ids(self):return ['0']
            def get_image(self,img_id):return None
            get_pose=get_K=get_mask=get_depth=get_image
        class Renderer:
            def forward(self,data):return {}
        def build(database,img_ids):return {}
        def split(database):return [],['0']
        def read(path):return None
        audit=ReadAudit(DB,Renderer,build,split,read);db=DB();key=audit.database_id(db)
        audit.builds=[{'database':key,'source_ids':['0']}];audit.forwards=[{}];audit.outputs=[{}]
        audit.reads=[{'phase':'evaluation','method':'get_depth','database':key,'source_id':'58'}]
        model=SimpleNamespace(database=db,train_ids=[str(i) for i in range(1,128)],test_ids=['0'])
        with self.assertRaisesRegex(SignatureError,'source camera 0'):audit.record(model)
