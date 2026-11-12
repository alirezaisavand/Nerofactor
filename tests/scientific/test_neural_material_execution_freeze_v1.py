"""C.2 saved-data/synthetic contract tests. No scientific shader or Adam step."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import freeze_neural_material_recovery_v1 as f
from experiments import neural_material_recovery_v1 as h
from experiments import neural_material_validation_admission_v1 as a
from tests.scientific.test_neural_material_validation_v1 import FakeShader
from infrastructure.signatures import SignatureError


class FakeN1(h.field.DirectMaterial):arm='N1'


class FakeOptimizer:
    def __init__(self,model,state=False):
        self.param_groups=[dict(params=list(model.parameters()),**copy.deepcopy(h.previous.original.SETTINGS['optimizer']))]
        self.state={}
        if state:
            self.state[model.latent]=dict(step=torch.tensor(1.),exp_avg=torch.zeros(7),exp_avg_sq=torch.zeros(7))
    def zero_grad(self):
        for p in self.param_groups[0]['params']:p.grad=None
    def step(self):raise AssertionError('No optimizer work in C.2 tests')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference,_=h.admit_reference();cls.fit,_,_=h.inputs(cls.reference)
        v,_=a.read_report(h.ROOT/a.c.PRIOR_PATH,a.c.PRIOR_SHA);cls.old,_=a.read_report(v['original_report']['path'],v['original_report']['sha256'])
        v,_=a.read_report(h.ROOT/a.COMPLETION_PATH,a.COMPLETION_SHA);cls.completed,_=a.read_report(v['original_completion_report']['path'],v['original_completion_report']['sha256'])

    def models(self):
        z=h.fresh_latent(101);return [h.field.DirectMaterial(z),h.field.ConstantMaterial(z),FakeN1(z)]
    def batch(self):return h.MaterialBatch(self.reference['scene'],self.reference['scene']['observations'][:24],self.fit)

    def test_real_aggregate_reverifies_without_shader_model_or_optimizer(self):
        with patch.object(h.previous.original,'MaterialShaderV1',side_effect=AssertionError('shader')),\
             patch.object(h.field,'make_material',side_effect=AssertionError('model')),\
             patch.object(torch.optim,'Adam',side_effect=AssertionError('optimizer')):
            result=a.verify_aggregate()
        self.assertEqual(result['status'],'passed');self.assertEqual(result['original_failure_status'],'failed')
        self.assertEqual(result['combined_work']['shader_forward'],{'attempted':45,'completed':45})
        self.assertEqual(result['scientific_calls_in_aggregate_verification'],0)
        self.assertEqual(result['replay_total'],9)

    def test_missing_completion_or_rewritten_original_failure_rejected(self):
        with self.assertRaises(SignatureError):a.check_matrix(self.old,None)
        old=copy.deepcopy(self.old);old['status']='passed'
        with self.assertRaises(SignatureError):a.check_matrix(old,self.completed)

    def test_duplicate_missing_substituted_fixture_and_work_rejected(self):
        for key in ('new_replay','new_gauge','storage'):
            bad=copy.deepcopy(self.completed);bad[key][-1]=copy.deepcopy(bad[key][0])
            with self.assertRaises(SignatureError):a.check_matrix(self.old,bad)
        bad=copy.deepcopy(self.completed);bad['work']['shader_forward']['completed']=16
        with self.assertRaises(SignatureError):a.check_matrix(self.old,bad)
        bad=copy.deepcopy(self.completed);bad['repeated_initial_renders']=1
        with self.assertRaises(SignatureError):a.check_matrix(self.old,bad)

    def test_fabricated_success_flags_are_not_aggregate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fake.json';p.write_text('{"status":"passed"}')
            with patch.object(a,'verify_aggregate',return_value={'status':'passed','verified_artifacts':['real']}):
                with self.assertRaises(SignatureError):a.admit_selected(p,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_missing_or_corrupt_archive_fails_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data.npz').write_bytes(b'not a numeric archive')
            e=a.c.Evidence(root);manifest=dict(path='data.npz',bytes=20,sha256='0'*64,arrays={})
            with patch.object(np,'load') as load:
                with self.assertRaises(SignatureError):e.arrays(manifest)
                manifest['path']='missing.npz'
                with self.assertRaises(SignatureError):e.arrays(manifest)
                load.assert_not_called()

    def test_exact_initial_admission_and_observation_preserve_gradients(self):
        models=self.models();result=h.initial_admission(models,self.batch(),FakeShader(),torch.full((24,3),.9),[h.counts(m.arm) for m in models])
        self.assertTrue(result['passed']);self.assertTrue(all(result['D0_N0_exact'].values()))
        self.assertTrue(torch.equal(models[0].latent.grad,models[1].latent.grad))

    def test_per_tape_cancellation_cannot_hide_an_exact_control_failure(self):
        class Fixture(FakeShader):
            executions=0
            def shade(self,*args,**kwargs):
                self.executions+=1;r=super().shade(*args,**kwargs)
                r['diffuse_rgb']=r['diffuse_rgb']*0+.25
                if self.executions==2:
                    r['diffuse_rgb']=r['diffuse_rgb'].clone();r['diffuse_rgb'][0,0]+=.0625;r['diffuse_rgb'][24,0]-=.0625
                return r
        models=self.models();shader=Fixture()
        with self.assertRaisesRegex(SignatureError,'per_tape'):
            h.initial_admission(models,self.batch(),shader,torch.full((24,3),.9),[h.counts(m.arm) for m in models])
        self.assertEqual(shader.executions,3)

    def test_pdf_control_mismatch_saved_without_rerender(self):
        class Fixture(FakeShader):
            executions=0
            def shade(self,*args,**kwargs):
                self.executions+=1;r=super().shade(*args,**kwargs)
                if self.executions==2:r['pdf']=r['pdf']+1
                return r
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                saved=[];models=self.models();shader=Fixture()
                def sink(audits,exc):saved.append(h.persist_failure_capture(output,'admission',audits,exc))
                with self.assertRaisesRegex(SignatureError,'pdf'):
                    h.initial_admission(models,self.batch(),shader,torch.full((24,3),.9),[h.counts(m.arm) for m in models],evidence_sink=sink)
                self.assertEqual(shader.executions,3);self.assertEqual(len(saved),1)
                with np.load(output.path(saved[0]['archive']),allow_pickle=False) as z:
                    self.assertEqual(float(z['1::return::pdf'].mean()),2.)
                manifest=json.loads(output.path(saved[0]['manifest']).read_text());self.assertEqual(manifest['additional_shader_calls'],0)

    def test_observational_gradient_retention_does_not_change_values_or_gradients(self):
        first,second=self.models()[:2];batch=self.batch();shader=FakeShader()
        x,_,_=batch.render(first,shader,h.counts('D0'));h.loss_value(x,torch.ones((24,3))).backward()
        audit={};y,_,_=batch.render(second,shader,h.counts('N0'),audit=audit);h.loss_value(y,torch.ones((24,3))).backward()
        self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(first.latent.grad,second.latent.grad))
        self.assertIsNotNone(audit['observer'].inputs[7].grad)

    def test_prefix_rejects_missing_history_loss_gradient_or_state(self):
        model=self.models()[0];model.latent.grad=torch.ones(7);opt=FakeOptimizer(model,state=True)
        old=dict(history=[dict(update=1,loss=.125,gradient_norm=1.,unconstrained=model.latent.detach().tolist(),physical=h.previous.original.physical(model.latent).detach().tolist())])
        h.prefix_check({},model,opt,old,0,.125,1.)
        for key in ('loss','gradient_norm','unconstrained'):
            bad=copy.deepcopy(old)
            if key=='unconstrained':bad['history'][0][key][0]+=.01
            else:bad['history'][0][key]+=.01
            with self.assertRaises(SignatureError):h.prefix_check({},model,opt,bad,0,.125,1.)
        with self.assertRaises(SignatureError):h.prefix_check({},model,opt,{'history':[]},0,.125,1.)

    def test_rolling_hash_detects_unlogged_divergence_even_after_reconvergence(self):
        models=self.models()[:2];opts=[FakeOptimizer(m,state=True) for m in models];digests=['0'*64,'0'*64]
        for update in range(1,4):
            for i,(model,opt) in enumerate(zip(models,opts)):
                model.latent.grad=torch.ones(7)*(2 if i==1 and update==2 else 1)
                opt.state[model.latent]['step'].fill_(update)
                digests[i]=h.vector_transition_digest(digests[i],model,opt,torch.tensor(.125),update)
        self.assertEqual(h.compact_state(models[0],opts[0]),h.compact_state(models[1],opts[1]))
        self.assertNotEqual(*digests)

    def test_optimizer_and_numeric_functions_unchanged(self):
        import ast,subprocess
        old=ast.parse(subprocess.check_output(['git','show',f.BASE+':experiments/neural_material_recovery_v1.py'],cwd=h.ROOT))
        new=ast.parse((h.ROOT/'experiments/neural_material_recovery_v1.py').read_text())
        for name in ('optimizer_for','step','loss_value','vector_transition_digest','fresh_latent','inputs'):
            x=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            y=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(x),ast.dump(y),name)

    def test_frozen_work_resources_and_continuous_optimizer_construction(self):
        self.assertEqual(h.Settings().active_seconds,900);self.assertEqual(h.Settings().whole_seconds,28800)
        self.assertEqual(h.Settings().endpoint_reserve,10);self.assertEqual(h.work_limits('N1')['optimizer_returns'],5000)
        self.assertEqual(h.work_limits('N1')['shader_attempts'],5004)
        self.assertEqual(h.work_limits('N1')['reverse_attempts'],5002)
        for setting in (h.Settings(updates=1000),h.Settings(active_seconds=901)):
            with self.assertRaises(SignatureError):setting.validate()
        import ast,inspect
        tree=ast.parse(inspect.getsource(h.train_history))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='optimizer_for']
        self.assertEqual(len(calls),1)

    def test_prefix_snapshot_moment_or_option_mismatch_is_not_ignored(self):
        model=self.models()[0];opt=FakeOptimizer(model,state=True);opt.state[model.latent]['step'].fill_(250)
        snapshot=h.previous.lm.accepted.snapshot(model.latent,opt)
        old=dict(history=[dict(update=250,loss=.125,gradient_norm=1.,unconstrained=model.latent.detach().tolist(),
            physical=h.previous.original.physical(model.latent).detach().tolist())],snapshots={'250':snapshot})
        h.prefix_check({},model,opt,old,249,.125,1.)
        bad=copy.deepcopy(old);bad['snapshots']['250']['optimizer_effective']['lr']=.02
        with self.assertRaises(SignatureError):h.prefix_check({},model,opt,bad,249,.125,1.)

    def test_censor_before_snapshot_retains_partial_state_without_optimizer_step(self):
        model=self.models()[1];opt=FakeOptimizer(model);row=dict(status='admitted',completed_updates=0,work=h.counts('N0'))
        class Clock:
            wall=cpu=0.
            def active(self):return contextlib.nullcontext()
            def check(self,*args):raise h.HistoryLimit('fixture cap')
        saved=[]
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output,patch.object(h,'optimizer_for',return_value=opt):
                def preserve(m,o,r):saved.append(h.save_checkpoint(output,'censored',m,o,r['completed_updates'],diagnostic=True))
                h.train_history(model,None,None,None,row,Clock(),lambda:None,lambda *args:None,failure_state=preserve)
                self.assertEqual(row['status'],'censored');self.assertEqual(row['completed_updates'],0)
                self.assertEqual(len(saved),1);self.assertTrue(output.path(saved[0]['manifest']).is_file())
                self.assertFalse(opt.state)

    def test_nonfinite_failure_artifact_stays_numeric_and_json_finite(self):
        class Fixture(FakeShader):
            def shade(self,*args,**kwargs):
                r=super().shade(*args,**kwargs);r['linear_rgb']=r['linear_rgb']*float('nan');return r
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                saved=[];shader=Fixture();models=self.models()
                def sink(audits,exc):saved.append(h.persist_failure_capture(output,'nonfinite',audits,exc))
                with self.assertRaises(h.NumericalFailure):h.initial_admission(models,self.batch(),shader,torch.ones((24,3)),[h.counts(m.arm) for m in models],evidence_sink=sink)
                self.assertEqual(len(shader.calls),1)
                manifest=json.loads(output.path(saved[0]['manifest']).read_text())
                self.assertFalse(manifest['arrays']['0::return::linear_rgb']['finite'])
                with np.load(output.path(saved[0]['archive']),allow_pickle=False) as z:self.assertTrue(np.isnan(z['0::return::linear_rgb']).all())

    def test_checkpoint_provenance_rejected_before_numeric_load(self):
        model=self.models()[1];opt=FakeOptimizer(model,state=True)
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                selected=h.save_checkpoint(output,'state',model,opt,1,provenance={'freeze_sha256':'a'*64})
                with patch.object(np,'load') as load:
                    with self.assertRaises(SignatureError):h.restore_checkpoint(output,selected,model,opt,expected_provenance={'freeze_sha256':'b'*64})
                    load.assert_not_called()

    def test_output_selection_and_capacity_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);root=parent/'3d2-recovery-unit';freeze={'output_policy':dict(approved_parent=directory,required_run_prefix='3d2-recovery-',minimum_free_bytes=1)}
            self.assertEqual(f.check_output_selection(freeze,root,root.name),root);self.assertFalse(root.exists())
            for unsafe in (parent/'other',parent/'../escape'):
                with self.assertRaises((SignatureError,ValueError)):f.check_output_selection(freeze,unsafe,root.name)
            root.mkdir();(root/'old').write_text('preserve')
            with self.assertRaises(SignatureError):f.check_output_selection(freeze,root,root.name)
            with patch.object(f.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaises(SignatureError):f.storage_check(parent,1)

    def test_authorization_remains_closed_before_allocation_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'3d2-recovery-denied'
            with patch.object(h,'OwnedOutput') as output,patch.object(h.field,'make_material') as model,patch.object(torch.optim,'Adam') as opt:
                for review in ({},{'phase':'D'},{'phase':'D','scientific_execution_authorized':False}):
                    with self.assertRaises(SignatureError):h.execute_reviewed(review,target,target.name)
                output.assert_not_called();model.assert_not_called();opt.assert_not_called();self.assertFalse(target.exists())

    def test_explicit_review_must_bind_watchdog_and_exact_output_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);path=parent/'freeze.json';path.write_text('{}');root=parent/'3d2-recovery-unit'
            frozen={'output_policy':dict(approved_parent=directory,required_run_prefix='3d2-recovery-',minimum_free_bytes=1)}
            review=dict(phase='D',scientific_execution_authorized=True,external_watchdog=f.WATCHDOG,freeze_path=str(path),
                freeze_sha256=h.digest(path),approved_output_root=str(root),approved_run_id=root.name)
            with patch.object(f,'validate_freeze',return_value=frozen):
                self.assertEqual(h.validate_execution_review(review,root,root.name),frozen)
                for key,value in (('external_watchdog',{}),('approved_run_id','other'),('approved_output_root',str(parent/'other'))):
                    bad=dict(review);bad[key]=value
                    with self.assertRaises(SignatureError):h.validate_execution_review(bad,root,root.name)
            self.assertFalse(root.exists())

    def test_metadata_CLIs_cannot_start_scientific_execution(self):
        import io
        for main in (h.main,f.main):
            with patch('sys.argv',['metadata','--execute']),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main()

    def test_effective_optimizer_flags_checked_before_synthetic_history(self):
        model=self.models()[1];opt=FakeOptimizer(model);row=dict(status='admitted',completed_updates=0,work=h.counts('N0'))
        clock=SimpleNamespace(wall=0.,cpu=0.,active=lambda:contextlib.nullcontext())
        with patch.object(h,'optimizer_for',return_value=opt),patch.object(h,'step') as step:
            with self.assertRaises(SignatureError):h.train_history(model,None,None,None,row,clock,lambda:None,lambda *args:None,
                expected_optimizer={'lr':.02})
            step.assert_not_called()
        self.assertEqual(row['status'],'failed_integrity')

    def test_freeze_rejects_configuration_source_initial_identity_or_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);path=parent/'freeze.json'
            aggregate=dict(protocol=a.PROTOCOL,status='passed',runtime_feasibility={'minimum_27_history_checkpoint_bytes':1})
            real_digest=h.digest
            def digest(p):return 'a'*64 if Path(p)==h.ROOT/f.AGGREGATE else real_digest(p)
            with patch.object(f,'source_inventory',return_value={'source':'b'*64}),patch.object(f,'reference_inputs',return_value={'tapes':{'fixed':True}}),\
                 patch.object(f,'initial_models',return_value={'M1/101/N0':{'identity':'exact'}}),patch.object(h,'digest',side_effect=digest),\
                 patch.object(f.aggregate,'admit_selected',return_value=aggregate):
                freeze=f.make_freeze(aggregate,parent)
                for field in ('configuration','sha256','initial_models'):
                    bad=copy.deepcopy(freeze);bad[field]['changed']=True;path.write_text(json.dumps(bad))
                    with self.assertRaises(SignatureError):f.validate_freeze(path,real_digest(path))
                path.write_text(json.dumps(freeze))
                with self.assertRaises(SignatureError):f.validate_freeze(path,'0'*64)
                self.assertEqual(f.validate_freeze(path,real_digest(path)),freeze)
            self.assertFalse(any(p.name.startswith('3d2-recovery-') for p in parent.iterdir()))


if __name__=='__main__':unittest.main()
