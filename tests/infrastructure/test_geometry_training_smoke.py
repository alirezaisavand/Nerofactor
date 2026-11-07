"""CPU-only fixtures; no scientific renderer, native dataset decoding or GPU calls."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest import TestCase, skipIf
from unittest.mock import Mock, patch

import numpy as np
try:
    import torch
except (ImportError,OSError):
    torch=None

from infrastructure import geometry_training_smoke as smoke
from infrastructure.bounded_training import TwoUpdateTrace, run_boundaries
from infrastructure.bell_inference import runtime_environment
from infrastructure.geometry import prepare_geometry
from infrastructure.signatures import SignatureError
from infrastructure.training_probe import TrainingProbe, parameter_changes
from infrastructure.training_comparison import (compare_training_pair, compare_training_sequence,
                                               load_training_arm, validate_training_report)
import compare_geometry_training_smoke as cli


class TrainingPlanningTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.datasets=self.root/'datasets';self.thesis=self.root/'empty-thesis'
        self.output=self.root/'outputs'
        for p in [self.datasets/'GlossySynthetic/bell/masks_train',self.thesis,self.output]:p.mkdir(parents=True)
        for name in [str(i)+'-camera.pkl' for i in range(128)]+['0.png','0-depth.png']:
            (self.datasets/'GlossySynthetic/bell'/name).write_bytes(b'fixture')

    def test_independent_reference_candidate_planning_and_no_rng(self):
        before=random.getstate()
        with patch('infrastructure.config.load_config',side_effect=AssertionError('R cannot call C loader')):
            r,_=smoke.resolve_plan('R',self.datasets,self.thesis,self.output)
        c,_=smoke.resolve_plan('C',self.datasets,self.thesis,self.output)
        self.assertEqual(r.pipeline_config,c.pipeline_config);self.assertEqual(r.legacy_bytes,c.legacy_bytes)
        self.assertEqual(r.configuration.preprocessing,c.configuration.preprocessing)
        self.assertEqual(random.getstate(),before);self.assertEqual(list(self.output.iterdir()),[])
        self.assertEqual(r.pipeline_config['total_step'],300000);self.assertEqual(r.pipeline_config['lr_cfg'],{})

    def test_invalid_dataset_and_overlapping_outputs_fail(self):
        for arm in ['R','C']:
            with self.subTest(arm=arm),self.assertRaises(ValueError):
                smoke.resolve_plan(arm,self.datasets,self.thesis,self.datasets/'out')
        with self.assertRaises(ValueError):smoke.resolve_plan('C',self.root/'missing',self.thesis,self.output)

    def test_runtime_guard_rejects_mounted_historical_weights(self):
        plan,_=smoke.resolve_plan('R',self.datasets,self.thesis,self.output)
        (self.thesis/'model.pth').write_bytes(b'never loaded')
        with patch.object(os,'geteuid',return_value=1109),patch.object(os,'statvfs',return_value=type('Stat',(),{'f_flag':os.ST_RDONLY})()):
            with self.assertRaisesRegex(SignatureError,'empty read-only thesis'):smoke.runtime_guard(plan,'image',{})

    def test_source_pins_and_boundary_locations(self):
        identity=smoke.source_identity();self.assertIn('train/trainer.py',identity['protected'])
        first,points=run_boundaries(smoke.ROOT/'train/trainer.py')
        self.assertLess(first,points['before_forward']);self.assertLess(points['before_backward'],points['before_clip'])
        self.assertLess(points['before_optimizer'],points['after_optimizer'])
        original=smoke.file_hash
        with patch.object(smoke,'file_hash',side_effect=lambda p:'0'*64 if str(p).endswith('train/trainer.py') else original(p)):
            with self.assertRaises(SignatureError):smoke.source_identity()

    def test_isolation_failure_precedes_native_work_and_allocation(self):
        with patch.object(smoke,'runtime_guard',side_effect=SignatureError('fixture guard blocked')), \
                patch.object(smoke,'native_smoke',side_effect=AssertionError('no native work')) as backend:
            with self.assertRaisesRegex(SignatureError,'guard blocked'):
                smoke.execute_smoke('R','fixture-run',self.datasets,self.thesis,self.output,'image',smoke.BASELINE)
        backend.assert_not_called();self.assertEqual(list(self.output.iterdir()),[])


@skipIf(torch is None,'CPU Torch unavailable; native GPU execution is never a fallback')
class TrainingObservationTests(TestCase):
    def setUp(self):
        TrainingPlanningTests.setUp(self)
        from training_smoke_fixtures import trainer_fixture, cpu_rng
        self.factory,self.cpu_rng=trainer_fixture,cpu_rng
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        self.addCleanup(lambda:(random.setstate(state[0]),np.random.set_state(state[1]),torch.set_rng_state(state[2])))

    def run_fixture(self,arm='R',name=None,configure=None):
        from infrastructure.geometry_training_smoke import PROTOCOL
        out=self.root/(name or arm);out.mkdir()
        plan,route=smoke.resolve_plan(arm,self.datasets,self.thesis,out)
        experiment=prepare_geometry(plan,(name or arm).lower().replace('_','-'),layout='mounted')
        fixture=self.factory()
        report={'schema_version':1,'protocol':PROTOCOL,'status':'running','arm':arm,'run_id':experiment.run.run_id,
                'route':route,'requested_optimizer_updates':2,'legacy_total_step':300000,'checkpoint_input':None,
                'execution_mode':'fresh-initialization-two-update-bound','used_inputs':[{'CPU_fixture':True}],
                'source':smoke.source_identity(),'declared_source_commit':smoke.BASELINE,'declared_image_id':'CPU_fixture_no_GPU',
                'legacy_sha256':hashlib.sha256(plan.legacy_bytes).hexdigest(),'runtime':{'CPU_fixture':torch.__version__}}
        def persist():
            (out/'training-smoke.json').write_text(json.dumps(report))
        controller=probe=None
        try:
            with runtime_environment(experiment.run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                trainer=fixture.Trainer(plan.pipeline_config,experiment=experiment)
                # Configure the CPU renderer after the unchanged _init_network returns.
                if configure is not None:
                    initialize=trainer._init_network
                    def initialize_with_fixture():initialize();configure(trainer,fixture)
                    trainer._init_network=initialize_with_fixture
                probe=TrainingProbe(trainer,torch,np,report,persist,rng=self.cpu_rng,memory=lambda:{'CPU_fixture_bytes':0})
                controller=TwoUpdateTrace(trainer,probe,smoke.ROOT/'train/trainer.py')
                controller.run()
            report.update(status='complete',phase='bounded_stop_after_second_optimizer_return',preprocessing={'CPU_fixture':True})
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc));persist()
            self.last=fixture,report,controller,probe,out,experiment
            raise
        self.last=fixture,report,controller,probe,out,experiment
        report['output_hashes']={p.relative_to(out).as_posix():smoke.file_hash(p) for p in out.rglob('*') if p.is_file() and
                                p.name not in {'training-smoke.json','.experiment.json'}}
        persist()
        return report,{'path':str(out/'training-smoke.json'),'sha256':smoke.file_hash(out/'training-smoke.json')}

    def test_exact_two_real_cpu_backward_adam_calls_and_cleanup(self):
        value=self.run_fixture()[0];fixture,_,controller,_,out,exp=self.last
        self.assertEqual(controller.completed,2);self.assertIsNone(sys.gettrace())
        self.assertEqual(fixture.networks[0].backward_calls,2)
        self.assertEqual(fixture.events.count('next_batch'),2)
        self.assertEqual([u['step'] for u in value['updates']],[0,1])
        self.assertEqual([u['lr'] for u in value['updates']],[0,5e-4*(1/5000)])
        self.assertEqual(value['updates'][0]['changed_elements'],0)
        self.assertGreater(value['updates'][1]['changed_elements'],0)
        self.assertTrue(fixture.loaders[0].iterator.closed);self.assertTrue(fixture.writers[0].closed)
        self.assertEqual(list(exp.run.checkpoints.iterdir()),[])
        self.assertNotIn('unexpected_natural_log',fixture.events)
        self.assertEqual(validate_training_report(value,'R'),value)
        with patch.object(torch,'load',side_effect=AssertionError('no checkpoint loading')):
            self.assertEqual(exp.restore(None,None,torch.load,300000),(0,0))

    def test_real_gradient_clipping_and_adam_state_counts(self):
        def larger_gradient(trainer,fixture):
            original=trainer.network.forward
            def forward(data):return {k:v*100 for k,v in original(data).items()}
            trainer.network.forward=forward
        value=self.run_fixture(configure=larger_gradient)[0]
        for i,update in enumerate(value['updates']):
            self.assertGreater(update['gradient_l2_before_clip_cpu64'],.7)
            self.assertLessEqual(update['gradient_l2_after_clip_cpu64'],.70001)
            self.assertEqual(update['clip_max_norm'],.7)
            self.assertTrue(all(v['step']==i+1 for v in update['optimizer']['state'].values()))
            self.assertIsNone(update['gradients_before_clip']['inactive'])

    def test_failure_before_first_forward_stops_and_closes_workers(self):
        def configure(trainer,fixture):trainer.network.fail_step=0
        with self.assertRaisesRegex(RuntimeError,'fixture forward'):
            self.run_fixture(configure=configure)
        fixture,value,controller,_,_,_=self.last
        self.assertEqual(controller.completed,0);self.assertEqual(value['phase'],'before_forward')
        self.assertTrue(fixture.loaders[0].iterator.closed);self.assertTrue(fixture.writers[0].closed)
        self.assertIsNone(sys.gettrace());self.assertEqual(value['optimizer_calls_completed'],0)

    def test_nonfinite_loss_blocks_backward_and_optimizer(self):
        def configure(trainer,fixture):trainer.network.bad_loss=True
        with self.assertRaises(SignatureError):self.run_fixture(configure=configure)
        fixture,value,controller,_,_,_=self.last
        self.assertEqual(controller.completed,0);self.assertEqual(fixture.networks[0].backward_calls,0)
        self.assertEqual(value['phase'],'before_backward');self.assertTrue(fixture.loaders[0].iterator.closed)

    def test_nonfinite_gradient_blocks_clip_and_optimizer(self):
        def configure(trainer,fixture):trainer.network.bad_gradient=True
        with self.assertRaises(SignatureError):self.run_fixture(configure=configure)
        fixture,value,controller,_,_,_=self.last
        self.assertEqual(controller.completed,0);self.assertEqual(fixture.networks[0].backward_calls,1)
        self.assertEqual(value['phase'],'before_clip')

    def test_failure_on_second_update_retains_first_update(self):
        def configure(trainer,fixture):trainer.network.fail_step=1
        with self.assertRaises(RuntimeError):self.run_fixture(configure=configure)
        fixture,value,controller,_,_,_=self.last
        self.assertEqual(controller.completed,1);self.assertEqual(len(value['updates']),1)
        self.assertEqual(value['partial_update']['step'],1)

    def test_parameter_change_detector_and_nonfinite_parameter_abort(self):
        before={'x':np.array([1,2],np.float32)};after={'x':np.array([1,2.5],np.float32)}
        changes=parameter_changes(before,after,np)
        self.assertEqual(changes['x'],{'changed_elements':1,'l2':.5,'max_abs':.5})
        def configure(trainer,fixture):
            with torch.no_grad():trainer.network.weight[0]=float('inf')
        with self.assertRaises(SignatureError):self.run_fixture(configure=configure)
        self.assertEqual(self.last[2].completed,0)

    def test_probe_reporting_consumes_no_rng_or_tensor_mutation(self):
        self.run_fixture();_,report,_,probe,_,_=self.last
        before=self.cpu_rng();parameters={n:p.detach().clone() for n,p in probe.trainer.network.named_parameters()}
        probe.gradients();probe.memory();probe.rng()
        self.assertEqual(self.cpu_rng(),before)
        self.assertTrue(all(torch.equal(p,parameters[n]) for n,p in probe.trainer.network.named_parameters()))
        self.assertNotEqual(report['updates'][0]['rng_before_forward'],report['updates'][0]['rng_after'])
        self.assertEqual(report['updates'][0]['sampling']['cursor_before'],0)
        self.assertEqual(report['updates'][1]['cursor_after'],1024)

    def test_unchanged_scheduler_and_no_shortened_config(self):
        report=self.run_fixture()[0]
        self.assertEqual(report['effective_trainer']['total_step'],300000)
        self.assertEqual(report['effective_trainer']['val_interval'],1000)
        self.assertEqual(report['scheduler']['hard_coded_cosine_endpoint'],100000)
        self.assertEqual(report['scheduler']['attributes']['warm_up_end'],5000)
        self.assertEqual(report['scheduler']['attributes']['end_iter'],300000)
        self.assertEqual(report['initial_optimizer']['groups'][0]['eps'],1e-4)

    def test_repeatability_then_candidate_and_rng_mismatch_gating(self):
        r=self.run_fixture('R');repeat=self.run_fixture('R-repeat');candidate=self.run_fixture('C')
        result=compare_training_sequence(r,repeat,candidate)
        self.assertEqual(result['result'],'bounded_training_regression_passed')
        self.assertTrue(result['candidate_interpreted'])
        bad=deepcopy(repeat);bad[0]['updates'][0]['rng_after']={'changed':True}
        deferred=Mock(side_effect=AssertionError('candidate must not load'))
        result=compare_training_sequence(r,bad,deferred)
        self.assertEqual(result['result'],'repeatability_failed');deferred.assert_not_called()

    def test_exact_gradient_hash_policy_and_scalar_tolerance_diagnostics(self):
        r=self.run_fixture('R')[0];other=deepcopy(r);other['run_id']='other'
        other['updates'][0]['aggregate_loss']+=1e-9
        self.assertTrue(compare_training_pair(r,other)['passed'])
        name=next(n for n,g in other['updates'][0]['gradients_before_clip'].items() if g is not None)
        other['updates'][0]['gradients_before_clip'][name]['sha256']='0'*64
        result=compare_training_pair(r,other)
        self.assertFalse(result['passed']);self.assertTrue(result['exact_state_mismatches'])
        other=deepcopy(r);other['updates'][0]['aggregate_loss']+=.01
        self.assertGreater(compare_training_pair(r,other)['scalar_violation_count'],0)

    def test_report_validation_and_output_hash_containment(self):
        report,identity=self.run_fixture('R');out=self.last[4]
        self.assertEqual(load_training_arm(out/'training-smoke.json',identity['sha256'],'R')[0],json.loads((out/'training-smoke.json').read_text()))
        for key,val in [('status','running'),('optimizer_calls_completed',True),('checkpoint_input','weights.pth')]:
            bad=deepcopy(report);bad[key]=val
            with self.assertRaises(SignatureError):validate_training_report(bad,'R')
        (out/'logs/console.log').write_text('tampered')
        with self.assertRaises(SignatureError):load_training_arm(out/'training-smoke.json',identity['sha256'],'R')

    def test_comparison_cli_and_collision_refusal(self):
        reports=[self.run_fixture(a) for a in ['R','R-repeat','C']];out=self.root/'comparison';out.mkdir()
        args=[]
        for name,(_,identity) in zip(['reference','repeat','candidate'],reports):
            args+=['--'+name,identity['path'],'--'+name+'-sha256',identity['sha256']]
        args+=['--output-root',str(out)]
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args),0)
            before=(out/'training-comparison.json').read_bytes();self.assertEqual(cli.main(args),2)
            self.assertEqual((out/'training-comparison.json').read_bytes(),before)

    def test_existing_trace_is_not_overridden(self):
        self.run_fixture();_,_,controller,_,_,_=self.last
        previous=lambda *args:None
        with patch.object(sys,'gettrace',return_value=previous),patch.object(sys,'settrace') as setter:
            with self.assertRaises(SignatureError):controller.run()
            setter.assert_not_called()

    def test_late_active_parameter_uses_its_own_adam_step(self):
        def configure(trainer,fixture):
            original=trainer.network.forward
            def forward(data):
                result=original(data)
                if data['step']==1:result['CPU_fixture_loss']=result['CPU_fixture_loss']+trainer.network.inactive.square().sum()
                return result
            trainer.network.forward=forward
        report=self.run_fixture(configure=configure)[0]
        self.assertNotIn('inactive',report['updates'][0]['optimizer']['state'])
        self.assertEqual(report['updates'][1]['optimizer']['state']['inactive']['step'],1)
        self.assertEqual(report['updates'][1]['optimizer']['state']['weight']['step'],2)
        validate_training_report(report,'R')

    def test_cleanup_failure_is_not_a_success_or_a_third_update(self):
        def configure(trainer,fixture):
            original=trainer._init_logger
            def initialize():
                original();trainer.logger.writer.close=Mock(side_effect=RuntimeError('fixture close failure'))
            trainer._init_logger=initialize
        with self.assertRaisesRegex(SignatureError,'cleanup failed'):self.run_fixture(configure=configure)
        fixture,report,controller,_,_,_=self.last
        self.assertEqual(controller.completed,2);self.assertEqual(len(report['updates']),2)
        self.assertEqual(fixture.events.count('next_batch'),2)
        self.assertEqual(report['cleanup_errors'][0]['resource'],'writer')
        self.assertIsNone(sys.gettrace())

    def test_scheduler_mismatch_blocks_second_update_without_correction(self):
        def configure(trainer,fixture):trainer.lr_manager.learning_rate=1e-3
        with self.assertRaisesRegex(SignatureError,'warm-up schedule changed'):self.run_fixture(configure=configure)
        self.assertEqual(self.last[2].completed,1)
        self.assertEqual(len(self.last[1]['updates']),1)

    def test_inputs_unchanged_and_writes_confined_to_owned_output(self):
        from infrastructure.bell_inference import file_hash
        before={p:(p.stat().st_mtime_ns,file_hash(p)) for base in (self.datasets,self.thesis) for p in base.rglob('*') if p.is_file()}
        original=Path.open
        def guarded(path,mode='r',*args,**kwargs):
            if any(c in mode for c in 'wax+'):
                for root in (self.datasets,self.thesis,smoke.ROOT):
                    if path.resolve()==root or root in path.resolve().parents:raise AssertionError('write into input/source')
            return original(path,mode,*args,**kwargs)
        with patch.object(Path,'open',guarded):self.run_fixture()
        self.assertTrue(all((p.stat().st_mtime_ns,file_hash(p))==value for p,value in before.items()))

    def test_original_scheduler_endpoint_not_envelope_total_step(self):
        from types import SimpleNamespace
        from train.lr_common_manager import WarmUpCosLR
        manager=WarmUpCosLR({});optimizer=SimpleNamespace(param_groups=[{}])
        self.assertEqual(manager(optimizer,0),0)
        self.assertEqual(manager(optimizer,1),5e-4*(1/5000))
        self.assertEqual(manager(optimizer,5000),5e-4)
        self.assertEqual(manager(optimizer,100000),1e-4)
        self.assertEqual(manager(optimizer,300000),1e-4)

    def test_fresh_run_collision_refuses_reuse(self):
        self.run_fixture();_,_,_,_,out,_=self.last
        plan,_=smoke.resolve_plan('R',self.datasets,self.thesis,out)
        before=(out/'training-smoke.json').read_bytes()
        with self.assertRaises(ValueError):prepare_geometry(plan,'another',layout='mounted')
        self.assertEqual((out/'training-smoke.json').read_bytes(),before)

    def test_execution_orchestrator_outputs_and_failure_log_cleanup(self):
        inputs=[{'path':'CPU_fixture_only','bytes':0,'sha256':'0'*64}]
        canonical=json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()
        real_read=smoke.read_report
        def fixture_read(path,digest):
            value,identity=real_read(path,digest)
            if str(path).endswith(smoke.ADMISSION):
                value=deepcopy(value);value['inputs']['used_manifest_canonical_sha256']=hashlib.sha256(canonical).hexdigest()
            return value,identity
        fixtures=[]
        def backend(experiment,report,persist):
            fixture=self.factory();fixtures.append(fixture)
            trainer=fixture.Trainer(experiment.plan.pipeline_config,experiment=experiment)
            report['runtime']={'CPU_fixture':torch.__version__}
            probe=TrainingProbe(trainer,torch,np,report,persist,rng=self.cpu_rng,memory=lambda:{'CPU_fixture_bytes':0})
            TwoUpdateTrace(trainer,probe,smoke.ROOT/'train/trainer.py').run()
            report['preprocessing']={'CPU_fixture':True}
            print('CPU fixture completed')
        with patch.object(smoke,'read_report',side_effect=fixture_read),patch.object(smoke,'runtime_guard'), \
                patch.object(smoke,'used_inputs',return_value=inputs),patch.object(smoke,'native_smoke',side_effect=backend), \
                redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            report=smoke.execute_smoke('R','fixture-run',self.datasets,self.thesis,self.output,'CPU_fixture',smoke.BASELINE)
        self.assertEqual(report['status'],'complete');self.assertEqual(report['optimizer_calls_completed'],2)
        self.assertTrue(all(w.closed for w in fixtures[0].writers))
        path=self.output/'training-smoke.json'
        self.assertEqual(load_training_arm(path,smoke.file_hash(path),'R')[0],json.loads(path.read_text()))
        failed=self.root/'failed-outputs';failed.mkdir();cwd,home=Path.cwd(),os.environ.get('HOME')
        def fail(experiment,report,persist):
            report['phase']='CPU_fixture_failure';persist();print('CPU fixture diagnostic');raise RuntimeError('CPU fixture failed')
        with patch.object(smoke,'read_report',side_effect=fixture_read),patch.object(smoke,'runtime_guard'), \
                patch.object(smoke,'used_inputs',return_value=inputs),patch.object(smoke,'native_smoke',side_effect=fail), \
                redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                smoke.execute_smoke('C','fixture-failed',self.datasets,self.thesis,failed,'CPU_fixture',smoke.BASELINE)
        self.assertEqual(Path.cwd(),cwd);self.assertEqual(os.environ.get('HOME'),home)
        value=json.loads((failed/'training-smoke.json').read_text())
        self.assertEqual(value['status'],'failed');self.assertEqual(value['phase'],'CPU_fixture_failure')
        self.assertIn('CPU fixture diagnostic',(failed/'logs/console.log').read_text())

    def test_unavailable_trace_never_enters_training(self):
        self.run_fixture();trainer=self.last[3].trainer
        observer=Mock();controller=TwoUpdateTrace(trainer,observer,smoke.ROOT/'train/trainer.py')
        with patch.object(sys,'gettrace',return_value=None),patch.object(sys,'settrace'), \
                patch.object(trainer,'run') as run:
            with self.assertRaisesRegex(SignatureError,'Observer unavailable'):controller.run()
            run.assert_not_called()

    def test_blocked_candidate_keeps_completed_repeatability_diagnostics(self):
        reference=self.run_fixture('R');repeat=self.run_fixture('R-repeat')
        out=self.root/'blocked-comparison';out.mkdir()
        args=['--reference',reference[1]['path'],'--reference-sha256',reference[1]['sha256'],
              '--repeat',repeat[1]['path'],'--repeat-sha256',repeat[1]['sha256'],
              '--candidate',str(self.root/'missing/training-smoke.json'),'--candidate-sha256','0'*64,
              '--output-root',str(out)]
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):self.assertEqual(cli.main(args),2)
        report=json.loads((out/'training-comparison.json').read_text())
        self.assertEqual(report['result'],'blocked');self.assertTrue(report['repeatability']['passed'])
        self.assertFalse(report['candidate_interpreted']);self.assertTrue(report['error'])
