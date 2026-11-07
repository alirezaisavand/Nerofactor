"""Disposable CPU fixtures; no historical checkpoint load, native renderer or GPU execution."""

from contextlib import redirect_stdout,redirect_stderr
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from unittest import TestCase,skipIf
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image
try:
    import torch
except (ImportError,OSError):torch=None

from infrastructure import bell_camera_diagnostic as diagnostic
from infrastructure import bell_camera_comparison as comparison
from infrastructure.bell_comparison import rgb_metrics
from infrastructure.bell_inference import proposed_split,THRESHOLDS
from infrastructure.signatures import SignatureError
import run_bell_camera_diagnostic as cli


class CameraComparisonTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def fixture_report(self,step,offset=0):
        root=self.root/str(step);root.mkdir();(root/'arrays').mkdir();(root/'renders').mkdir()
        p=np.full((200,200,3),.4+offset,np.float32);g=np.full_like(p,.5);m=np.ones((200,200,1),np.int32)
        for name,value in [('prediction',p),('ground_truth',g),('mask',m)]:np.save(str(root/'arrays'/(name+'.npy')),value,allow_pickle=False)
        for name,value in [('prediction',p),('ground_truth',g)]:Image.fromarray(np.clip(value*255,0,255).astype(np.uint8)).save(root/'renders'/(name+'.png'))
        metrics=rgb_metrics(p,g);(root/'metrics.json').write_text(json.dumps(metrics))
        panel=np.full((200,200,3),127,np.uint8)
        Image.fromarray(panel).save(root/'renders/historical-GT-panel-decoded.png')
        difference=np.clip(g*255,0,255).astype(np.uint8).astype(np.int16)-panel.astype(np.int16)
        np.save(str(root/'arrays/current-GT-minus-JPEG-panel.npy'),difference,allow_pickle=False)
        approved,_=diagnostic.admission()
        historical={**comparison.score_differences(metrics,approved['historical_metrics']),
                    'historical_montage_sha256':approved['historical_montage']['sha256'],
                    'panel_box_xyxy':[0,0,200,200],
                    'jpeg':{**comparison.jpeg_statistics(difference),'acceptance_tolerance':None,'resize_align_recompress_denoise':False}}
        selection={'database_source_id':'0','diagnostic_iterator_index':0,'diagnostic_index_to_source_id':{'0':'0'},
                   'getter_reads':[{'method':n,'source_id':'0'} for n in ['get_image','get_pose','get_K','get_mask','get_depth']],
                   'original_A_validation_ids':['58'],'original_A_tables_unchanged':True}
        r={'schema_version':1,'protocol':diagnostic.PROTOCOL,'status':'complete','phase':'complete','run_id':'fixture-'+str(step),
           'source_id':'0','evaluation_step':step,'evaluation_calls':1,'checkpoint_stored_step':150000,
           'checkpoint_sha256':diagnostic.CHECKPOINT_SHA256,'diagnostic_admission_sha256':diagnostic.ADMISSION_SHA256,
           'implementation':diagnostic.identity(),'selection':selection,'weights_and_buffers_unchanged':True,
           'selection_rng_unchanged':True,'original_A_splits':proposed_split(),'metrics':metrics,
           'runtime':{'CPU_fixture':True,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scikit-image')}},
           'historical_comparison':historical,'evidence':{'CPU_fixture':True},
           'declared_image_id':'CPU_fixture_no_GPU','declared_source_commit':'0'*40,'legacy_sha256':'0'*64,
           'used_inputs':[{'CPU_fixture':True}],'route':'direct-legacy-FullLoader',
           'effective_trainer':{'random_seed':6033},'effective_renderer':{'downsample_ratio':.25},'effective_shader':{},
           'network_state_fingerprints':{'CPU_fixture':'0'*64},'preprocessing':{'CPU_fixture':True},
           'rng_before':{'CPU_fixture':'same'},'rng_after':{'CPU_fixture':'same-after'},'auxiliary_outputs':{'CPU_fixture':True}}
        r['output_hashes']={x.relative_to(root).as_posix():diagnostic.file_hash(x) for x in root.rglob('*') if x.is_file()}
        path=root/'camera-diagnostic.json';path.write_text(json.dumps(r));return path,diagnostic.file_hash(path),r

    def test_step_comparison_exact_and_difference_arrays(self):
        left=self.fixture_report(149999);right=self.fixture_report(150000);out=self.root/'comparison';out.mkdir()
        result=comparison.compare_steps(comparison.load_diagnostic(*left[:2],149999),comparison.load_diagnostic(*right[:2],150000),out)
        self.assertEqual(result['result'],'within_frozen_step_comparison_bounds');self.assertTrue(result['rgb_bitwise_equal'])
        self.assertEqual(result['thresholds'],THRESHOLDS)
        self.assertFalse(np.load(out/'step-control-minus-primary.npy',allow_pickle=False).any())

    def test_numeric_and_rng_mismatch_remain_failures(self):
        left=self.fixture_report(149999);right=self.fixture_report(150000,.01);out=self.root/'comparison';out.mkdir()
        loaded=comparison.load_diagnostic(*left[:2],149999),comparison.load_diagnostic(*right[:2],150000)
        result=comparison.compare_steps(*loaded,out)
        self.assertEqual(result['result'],'step_pair_mismatch');self.assertGreater(result['float_violation_count'],0)
        loaded[1][0]['rng_before']={'different':True}
        other=self.root/'other';other.mkdir();result=comparison.compare_steps(*loaded,other)
        self.assertIn('rng_before',result['identity_mismatches'])

    def test_wrong_camera_iterator_step_and_checkpoint_rejected(self):
        path,_,report=self.fixture_report(149999)
        cases=[('source_id','58'),('evaluation_step',150000),('checkpoint_sha256','0'*64),('checkpoint_stored_step',149999),('status','running')]
        for field,value in cases:
            bad=deepcopy(report);bad[field]=value;path.write_text(json.dumps(bad))
            with self.subTest(field=field),self.assertRaises(SignatureError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)
        bad=deepcopy(report);bad['selection']['diagnostic_index_to_source_id']={'0':'58'};path.write_text(json.dumps(bad))
        with self.assertRaises(SignatureError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)
        bad=deepcopy(report);bad['historical_comparison']['psnr_minus_logged']=0.0;path.write_text(json.dumps(bad))
        with self.assertRaises(SignatureError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)
        bad=deepcopy(report);bad['historical_comparison']['jpeg']['max_absolute_uint8_difference']=1;path.write_text(json.dumps(bad))
        with self.assertRaises(SignatureError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)

    def test_output_hash_and_path_escape_rejection(self):
        path,_,r=self.fixture_report(149999)
        r['output_hashes']['../escape.npy']='0'*64;path.write_text(json.dumps(r))
        with self.assertRaises(ValueError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)
        del r['output_hashes']['../escape.npy'];path.write_text(json.dumps(r));(path.parent/'arrays/prediction.npy').write_bytes(b'bad')
        with self.assertRaises(SignatureError):comparison.load_diagnostic(path,diagnostic.file_hash(path),149999)

    def test_historical_comparison_accounts_for_jpeg_without_tuning(self):
        montage=self.root/'montage.jpg'
        yy,xx=np.indices((200,200));gt=np.stack([(xx*7)%256,(yy*11)%256,((xx+yy)*13)%256],-1).astype(np.uint8)
        image=np.zeros((1000,800,3),np.uint8);image[:200,:200]=gt;Image.fromarray(image).save(montage,quality=60)
        original=montage.read_bytes();out=self.root/'out';(out/'arrays').mkdir(parents=True);(out/'renders').mkdir()
        admitted,_=diagnostic.admission();admitted=deepcopy(admitted);admitted['historical_montage']['sha256']=diagnostic.file_hash(montage)
        run=SimpleNamespace(output=lambda name:out/name)
        measured={'psnr':{'status':'finite','value':30.0},'ssim':.95}
        with patch.object(comparison,'admission',return_value=(admitted,{})):
            result=comparison.historical_comparison(gt,measured,montage,run)
        self.assertGreater(result['jpeg']['max_absolute_uint8_difference'],0)
        self.assertIsNone(result['jpeg']['acceptance_tolerance']);self.assertIsNone(result['historical_metric_acceptance_tolerance'])
        self.assertAlmostEqual(result['psnr_minus_stored_best'],30-23.787122565451323)
        self.assertFalse(result['same_five_decimal_psnr_display'])
        self.assertEqual(montage.read_bytes(),original)
        self.assertEqual(np.load(out/'arrays/current-GT-minus-JPEG-panel.npy',allow_pickle=False).dtype,np.int16)

    def test_cli_explicit_flags_and_report_collision(self):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),patch.object(diagnostic,'execute_diagnostic') as native:
            with self.assertRaises(SystemExit):cli.main(['evaluate','--source-id','58'])
            with self.assertRaises(SystemExit):cli.main(['evaluate','--source-id','0','--eval-step','149999'])
            native.assert_not_called()
        p=self.fixture_report(149999);c=self.fixture_report(150000);out=self.root/'out';out.mkdir()
        args=['compare','--primary',str(p[0]),'--primary-sha256',p[1],'--control',str(c[0]),'--control-sha256',c[1],'--output-root',str(out)]
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args),0);before=(out/'step-comparison.json').read_bytes()
            self.assertEqual(cli.main(args),2);self.assertEqual((out/'step-comparison.json').read_bytes(),before)

    def test_metadata_only_admission_and_immutable_original_runner(self):
        admitted,_=diagnostic.admission()
        self.assertEqual(admitted['selected_source']['source_id'],'0')
        self.assertFalse(admitted['researcher_evidence']['independent_historical_producing_run_attribution'])
        with patch.object(diagnostic,'file_hash',return_value='0'*64),self.assertRaises(SignatureError):diagnostic.admission()

    def test_invalid_selection_fails_before_allocation_or_model_work(self):
        with patch.object(diagnostic,'native_evaluate',side_effect=AssertionError('must not execute')) as native:
            for source,step in [(0,149999),('58',149999),('0',0),('0',True)]:
                with self.assertRaises(SignatureError):
                    diagnostic.execute_diagnostic('fixture',source,step,*([self.root]*6),'image','commit')
            native.assert_not_called()
        self.assertEqual(list(self.root.iterdir()),[])

    def test_reviewed_active_step_predicates_are_identical(self):
        tree=ast.parse((diagnostic.ROOT/'network/renderer.py').read_text())
        renderer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeROShapeRenderer')
        owner=SimpleNamespace(cfg={'freeze_inv_s_step':15000,'occ_loss_step':20000})
        for method in ['compute_sdf_alpha','compute_occ_loss','render_core']:
            body=next(n for n in renderer.body if isinstance(n,ast.FunctionDef) and n.name==method)
            tests=[n.test for n in ast.walk(body) if isinstance(n,ast.If) and
                   any(isinstance(x,ast.Name) and x.id=='step' for x in ast.walk(n.test))]
            self.assertEqual(len(tests),1)
            predicate=compile(ast.Expression(body=tests[0]),'<reviewed step predicate>','eval')
            self.assertEqual(eval(predicate,{},dict(self=owner,step=149999)),eval(predicate,{},dict(self=owner,step=150000)))
        fields=ast.parse((diagnostic.ROOT/'network/field.py').read_text())
        shader=next(n for n in fields.body if isinstance(n,ast.ClassDef) and n.name=='AppShadingNetwork')
        lights=next(n for n in shader.body if isinstance(n,ast.FunctionDef) and n.name=='predict_specular_lights')
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='step' and isinstance(n.ctx,ast.Load) for n in ast.walk(lights)))

    def test_failure_logging_runtime_cleanup_and_checkpoint_hash_gate(self):
        data=self.root/'datasets';thesis=self.root/'thesis';output=self.root/'outputs'
        for p in [data/'GlossySynthetic/bell',thesis/'model/bell_shape',thesis/'train_vis/bell_shape-val',output]:p.mkdir(parents=True)
        for i in range(128):(data/'GlossySynthetic/bell'/(str(i)+'-camera.pkl')).write_bytes(b'opaque CPU fixture')
        checkpoint=thesis/'model/bell_shape/model_best.pth';checkpoint.write_bytes(b'CPU fixture: not a serialized checkpoint')
        historical=thesis/'train_vis/bell_shape-val/149999-index-0.jpg';historical.write_bytes(b'CPU fixture not decoded')
        approved,baseline=diagnostic.admission();approved=deepcopy(approved)
        approved['historical_montage']['sha256']=diagnostic.file_hash(historical)
        approved['selected_source']['files']=[]
        packet={'identities':{'CPU_fixture':True}}
        old_hash=diagnostic.file_hash;read=Path.read_text
        def fake_read(path,*args,**kwargs):return 'header\nheader\n lo: 0\n' if str(path)=='/proc/net/dev' else read(path,*args,**kwargs)
        def fail(cfg,cache,run,checkpoint,packet,step,source,historical,inputs,mark):
            mark('CPU_fixture_evaluation_failure');print('CPU fixture diagnostic');raise RuntimeError('fixture failure')
        with patch.object(diagnostic,'admission',return_value=(approved,baseline)),patch.object(diagnostic,'evidence',return_value=packet), \
                patch.object(diagnostic,'diagnostic_inputs',return_value=[]),patch.object(diagnostic,'native_evaluate',side_effect=fail) as native, \
                patch.object(os,'statvfs',return_value=SimpleNamespace(f_flag=os.ST_RDONLY)),patch.object(os,'geteuid',return_value=1109), \
                patch.object(Path,'read_text',fake_read),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(SignatureError,'admitted artifact'):
                diagnostic.execute_diagnostic('bad-hash','0',149999,data,thesis,output,*([self.root]*3),baseline['runtime']['declared_image_id'],'0'*40)
            native.assert_not_called();self.assertEqual(list(output.iterdir()),[])
            cwd,home=Path.cwd(),os.environ.get('HOME')
            with patch.object(diagnostic,'file_hash',side_effect=lambda p:diagnostic.CHECKPOINT_SHA256 if Path(p)==checkpoint else old_hash(p)):
                with self.assertRaises(RuntimeError):
                    diagnostic.execute_diagnostic('failed-run','0',149999,data,thesis,output,*([self.root]*3),baseline['runtime']['declared_image_id'],'0'*40)
            self.assertEqual(Path.cwd(),cwd);self.assertEqual(os.environ.get('HOME'),home)
        report=json.loads((output/'camera-diagnostic.json').read_text())
        self.assertEqual(report['status'],'failed');self.assertEqual(report['phase'],'CPU_fixture_evaluation_failure')
        self.assertIn('CPU fixture diagnostic',(output/'logs/console.log').read_text())
        self.assertEqual(checkpoint.read_bytes(),b'CPU fixture: not a serialized checkpoint')
        self.assertEqual(list((output/'checkpoints').iterdir()),[])


@skipIf(torch is None,'CPU Torch unavailable; no GPU fallback')
class CameraAdapterTests(TestCase):
    def setUp(self):
        state=torch.get_rng_state();self.addCleanup(lambda:torch.set_rng_state(state))

    def test_original_forward_test_step_explicit_source0_and_gradients(self):
        from camera_diagnostic_fixtures import fixture_renderer
        with fixture_renderer() as f,redirect_stdout(io.StringIO()):
            model=f.model;original=model.test_imgs_info;before=diagnostic.fingerprints(model);rng=torch.get_rng_state().clone()
            adapter=diagnostic.select_camera(model,'0',f.Renderer,f.build,f.to_torch,torch)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            with torch.no_grad():output=adapter({'index':0,'eval':True,'step':149999})
            self.assertTrue(torch.allclose(output['gt_rgb'],torch.full((12,12,3),64/255)))
            self.assertEqual(list(model.test_ids),['58']);self.assertIs(model.test_imgs_info,original)
            self.assertEqual(diagnostic.fingerprints(model),before);self.assertEqual(model.zero_calls,1)
            record=adapter.selection_record();self.assertEqual(record['getter_reads'][-1],{'method':'get_depth','source_id':'0'})
            self.assertIn(('sdf_gradient',True),f.events);self.assertIn('child_hook',f.events)
            self.assertEqual([e for e in f.events if isinstance(e,tuple) and e[0]=='default_tensor_type'],
                             [('default_tensor_type','torch.cuda.FloatTensor'),('default_tensor_type','torch.FloatTensor')])
            self.assertEqual(list(adapter.parameters()),[])

    def test_both_steps_preserve_inputs_flags_and_primary_output(self):
        from camera_diagnostic_fixtures import fixture_renderer
        values=[]
        for step in [149999,150000]:
            with fixture_renderer() as f,redirect_stdout(io.StringIO()):
                view=diagnostic.select_camera(f.model,'0',f.Renderer,f.build,f.to_torch,torch)
                with torch.no_grad():values.append(view({'index':0,'eval':True,'step':step})['ray_rgb'])
                self.assertIn(('render',step,0,0,False),f.events)
                self.assertIn(('downsample',.25),f.events)
        self.assertTrue(torch.equal(*values))

    def test_invalid_camera_hooks_and_training_dispatch_rejected(self):
        from camera_diagnostic_fixtures import fixture_renderer
        with fixture_renderer() as f,redirect_stdout(io.StringIO()):
            for source in [0,'58','missing']:
                with self.assertRaises(SignatureError):diagnostic.select_camera(f.model,source,f.Renderer,f.build,f.to_torch,torch)
            hook=f.model.register_forward_hook(lambda *args:None)
            with self.assertRaises(SignatureError):diagnostic.select_camera(f.model,'0',f.Renderer,f.build,f.to_torch,torch)
            hook.remove();view=diagnostic.select_camera(f.model,'0',f.Renderer,f.build,f.to_torch,torch)
            for data in [{'step':149999},{'index':0,'eval':True,'step':0},{'index':False,'eval':True,'step':149999}]:
                with self.assertRaises(SignatureError):view(data)

    def test_getter_guard_and_original_split_mutation_detected(self):
        from camera_diagnostic_fixtures import fixture_renderer
        with fixture_renderer() as f,redirect_stdout(io.StringIO()):
            view=diagnostic.select_camera(f.model,'0',f.Renderer,f.build,f.to_torch,torch)
            with self.assertRaises(SignatureError):view.database.get_depth('58')
            f.model.test_ids=['0']
            with self.assertRaises(SignatureError):view.selection_record()
