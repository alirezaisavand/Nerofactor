"""Phase B synthetic harness fixtures. No real shader, quadrature or recovery run."""
import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import neural_material_recovery_v1 as h
from infrastructure.signatures import SignatureError


class FakeShader:
    """Differentiable algebraic fixture; NOT the scientific BRDF/sampler."""
    def __init__(self):self.calls=[]
    def shade(self,positions,normals,views,mx,my,albedo,metallic,encoding,illumination,tapes,is_train):
        self.calls.append(dict(encoding=encoding,physical=(mx,my,albedo,metallic),is_train=is_train,tapes=tapes))
        n=len(positions);d=.2*albedo*(1-metallic)
        s=.1*mx*my*(1+encoding[:,:1]/10);s=s.expand(n,3)
        unit=positions.new_tensor([0.,0.,1.]).expand(n,1024,3)
        return dict(diffuse_rgb=d,specular_rgb=s,linear_rgb=d+s,frame_orthonormal=torch.ones(n,dtype=torch.bool),
            tangent=positions.new_tensor([0.,1.,0.]).expand(n,3),bitangent=positions.new_tensor([-1.,0.,0.]).expand(n,3),
            normal=normals,half_vectors=unit,specular_directions=unit,diffuse_directions=unit[:,:512],
            pdf=torch.ones((n,1024,1)))


class FixtureClock:
    wall=cpu=0.
    def check(self,*args):pass
    def elapsed(self):return 0.
    def active(self):return contextlib.nullcontext()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference,cls.identities=h.admit_reference()
        cls.fit,cls.eval_tape,cls.manifest=h.inputs(cls.reference)

    def model(self,arm='N0'):return h.field.make_material(arm,h.fresh_latent(101),101)
    def batch(self,evaluation=False):
        scene=self.reference['scene'];return h.MaterialBatch(scene,scene['observations'] if evaluation else scene['observations'][:24],self.eval_tape if evaluation else self.fit)

    def test_reference_design_source_chain_and_no_old_runner(self):
        self.assertGreaterEqual(len(self.identities),100)
        self.assertEqual(self.reference['status'],'complete')
        self.assertEqual(h.BASE,'a1f15d1030cb954deddd18fb408de39a452fd441')
        for name,sha in self.identities.items():self.assertEqual(h.digest(h.ROOT/name),sha,name)

    def test_panel_and_fixed_train_eval_tapes_exact(self):
        self.assertEqual(list(h.previous.reference.PANEL),['M1','M2','M3'])
        self.assertEqual(self.manifest,self.reference['tapes'])
        self.assertEqual([r['seed'] for r in self.manifest['fit']['per_tape']],[3000,3001,3002,3003])
        self.assertEqual([r['seed'] for r in self.manifest['evaluation']['per_tape']],[1000,1001,1002,1003])
        fit,ev,_=h.inputs(self.reference);fit.u[0,0]=.125
        with self.assertRaises(SignatureError):h.verify_tapes(fit,ev,self.manifest)

    def test_settings_reject_incompatible_numerical_or_resource_choices(self):
        h.Settings().validate()
        for settings in (h.Settings(decoder='original'),h.Settings(variant='A0'),h.Settings(updates=1000),h.Settings(specular_samples=32),h.Settings(active_seconds=901)):
            with self.assertRaises(SignatureError):settings.validate()

    def test_D0_N0_render_adapter_reuses_original_means_and_gradients(self):
        a,b=self.model('D0'),self.model('N0');shader=FakeShader();batch=self.batch()
        va,_,_=batch.render(a,shader,h.counts('D0'));vb,_,_=batch.render(b,shader,h.counts('N0'))
        self.assertTrue(torch.equal(va,vb))
        va.square().mean().backward();vb.square().mean().backward()
        self.assertTrue(torch.equal(a.latent.grad,b.latent.grad))
        self.assertEqual(va.shape,(24,3,3))
        self.assertTrue(all(c['is_train'] for c in shader.calls))

    def test_N1_passes_actual_encoding_without_angle_reconstruction(self):
        model=self.model('N1');shader=FakeShader();expected=model(torch.zeros((1,3))).encoding
        with patch.object(torch,'cos',side_effect=AssertionError('Unexpected angle reencoding')):
            values,material,_=self.batch().render(model,shader,h.counts('N1'))
        self.assertTrue(torch.equal(shader.calls[-1]['encoding'],expected.expand(96,2)))
        self.assertTrue(torch.equal(material.encoding,expected))
        self.assertTrue(values.requires_grad)

    def test_component_order_is_diffuse_specular_total(self):
        model=self.model();shader=FakeShader();values,_,_=self.batch().render(model,shader,h.counts('N0'))
        self.assertTrue(torch.equal(values[:,2],values[:,0]+values[:,1]))
        call=shader.calls[-1];mx,my,k,metal=call['physical']
        diffuse=.2*k*(1-metal);specular=(.1*mx*my*(1+call['encoding'][:,:1]/10)).expand(96,3)
        self.assertTrue(torch.equal(values[:,0],diffuse.reshape(4,24,3).mean(0)))
        self.assertTrue(torch.equal(values[:,1],specular.reshape(4,24,3).mean(0)))
        self.assertFalse(torch.equal(values[:,0],values[:,1]))

    def test_fake_shader_initial_admission_and_work_separation(self):
        models=[self.model(a) for a in h.ARMS];counts=[h.counts(a) for a in h.ARMS]
        result=h.initial_admission(models,self.batch(),FakeShader(),torch.full((24,3),.9),counts)
        self.assertTrue(result['passed'])
        for c in counts:
            self.assertEqual(c['field_completed'],3)  # two constant probes + one integrated call
            self.assertEqual(c['shader_completed'],1)
            self.assertEqual(c['reverse_completed'],1)
            self.assertEqual(c['optimizer_returns'],0)
        self.assertEqual(counts[2]['calibration_forwards'],1)

    def test_detached_neural_path_is_not_a_structural_zero(self):
        model=self.model('N1');hook=model.feats_network.register_forward_hook(lambda module,args,out:out.detach())
        values,_,_=self.batch().render(model,FakeShader(),h.counts('N1'))
        with self.assertRaises(h.NumericalFailure):h.backward(model,values.sum(),h.counts('N1'))
        hook.remove()

    def test_three_synthetic_Adam_updates_have_exact_D0_N0_states(self):
        models=[self.model('D0'),self.model('N0')];opts=[h.optimizer_for(m) for m in models];cs=[h.counts(m.arm) for m in models]
        identities=['0'*64,'0'*64]
        for _ in range(3):
            for i,(model,opt,c) in enumerate(zip(models,opts,cs)):
                opt.zero_grad();loss=(h.previous.original.physical(model.latent)-.5).square().sum()
                h.backward(model,loss,c);h.step(model,opt,c)
                identities[i]=h.vector_transition_digest(identities[i],model,opt,loss,c['optimizer_returns'])
            self.assertEqual(h.compact_state(models[0],opts[0]),h.compact_state(models[1],opts[1]))
            self.assertEqual(*identities)
        self.assertEqual(cs[0]['optimizer_returns'],3)
        self.assertNotEqual(identities[0],h.vector_transition_digest('f'*64,models[1],opts[1],loss,3))

    def test_snapshot_250_reuses_251_without_renderer(self):
        row={'snapshots':{'250':{'manifest':'fixture'}}};loss=torch.tensor(.125)
        h.reused_snapshot_objective(row,250,loss);self.assertNotIn('objective',row['snapshots']['250'])
        h.reused_snapshot_objective(row,251,loss)
        self.assertEqual(row['snapshots']['250']['objective_at_update'],250)
        self.assertEqual(row['snapshots']['250']['objective'],.125)
        with self.assertRaises(SignatureError):h.reused_snapshot_objective({'snapshots':{}},251,loss)

    def test_exact_prefix_fixture_detects_wrong_state_or_loss(self):
        model=self.model('D0');opt=h.optimizer_for(model);c=h.counts('D0')
        loss=h.previous.original.physical(model.latent).sum();grad=h.backward(model,loss,c)
        norm=float(model.latent.grad.norm());h.step(model,opt,c)
        old={'history':[dict(update=1,loss=float(loss),gradient_norm=norm,
            physical=h.previous.original.physical(model.latent).detach().tolist(),unconstrained=model.latent.detach().tolist())]}
        row={};h.prefix_check(row,model,opt,old,0,loss,norm)
        self.assertEqual(row['prefix_verified_through'],1)
        old['history'][0]['loss']+=.1
        with self.assertRaises(SignatureError):h.prefix_check(row,model,opt,old,0,loss,norm)

    def test_numerical_failure_first_forward_stops_without_update_or_retry(self):
        row=dict(status='admitted',completed_updates=0,work=h.counts('N0'))
        calls=[]
        class FailingBatch:
            def render(self,*args,**kwargs):calls.append(1);raise h.NumericalFailure('fixture nonfinite')
        h.train_history(self.model(),FailingBatch(),None,torch.zeros((24,3)),row,FixtureClock(),lambda:None,lambda *a:None)
        self.assertEqual(len(calls),1);self.assertEqual(row['completed_updates'],0)
        self.assertEqual(row['status'],'failed_numerical')

    def test_censoring_is_not_a_success_or_a_retry(self):
        class Clock(FixtureClock):
            def check(self,*args):raise h.HistoryLimit('fixture cap')
        row=dict(status='admitted',completed_updates=0,work=h.counts('N0'))
        h.train_history(self.model(),None,None,None,row,Clock(),lambda:None,lambda *a:None)
        self.assertEqual(row['status'],'censored');self.assertEqual(row['completed_updates'],0)

    def test_returned_but_nonfinite_update_is_counted_before_failure(self):
        model=self.model();row=dict(status='admitted',completed_updates=0,work=h.counts('N0'))
        class Batch:
            def render(self,m,*args):return m.latent.square().reshape(1,7,1).expand(1,7,3),None,None
        def invalid_step(m,opt,c):
            c['optimizer_attempts']+=1;c['optimizer_returns']+=1
            raise h.NumericalFailure('post-return nonfinite fixture')
        with patch.object(h,'step',side_effect=invalid_step):
            h.train_history(model,Batch(),None,torch.zeros((1,3)),row,FixtureClock(),lambda:None,lambda *a:None)
        self.assertEqual(row['completed_updates'],1)
        self.assertEqual(row['status'],'failed_numerical')

    def test_work_caps_and_attempt_completion_consistency(self):
        for key,value in (('optimizer_returns',5001),('shader_attempts',5005),('reverse_attempts',5003),('field_attempts',5007)):
            c=h.counts('N0');c[key]=value
            with self.assertRaises(SignatureError):h.check_counts(c,'N0')
        c=h.counts('N0');c['shader_completed']=1
        with self.assertRaises(SignatureError):h.check_counts(c,'N0')

    def test_explicit_own_pickle_free_state_roundtrip_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                model=self.model();opt=h.optimizer_for(model);c=h.counts('N0')
                for _ in range(3):
                    opt.zero_grad();loss=model(torch.zeros((1,3))).physical.square().sum();h.backward(model,loss,c);h.step(model,opt,c)
                selection=h.save_checkpoint(output,'state3',model,opt,3)
                restored=self.model();other=h.optimizer_for(restored)
                with patch.object(np,'load',wraps=np.load) as load:h.restore_checkpoint(output,selection,restored,other)
                self.assertIs(load.call_args.kwargs['allow_pickle'],False)
                self.assertEqual(h.compact_state(model,opt),h.compact_state(restored,other))
                for m,o in ((model,opt),(restored,other)):
                    o.zero_grad();m(torch.zeros((1,3))).physical.sum().backward();o.step()
                self.assertEqual(h.compact_state(model,opt),h.compact_state(restored,other))
                with self.assertRaises(SignatureError):h.save_checkpoint(output,'state3',model,opt,4)

    def test_checkpoint_hash_gate_precedes_numeric_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                with patch.object(np,'load') as load:
                    with self.assertRaises((OSError,SignatureError)):
                        h.restore_checkpoint(output,{'manifest':'checkpoints/missing.json','manifest_sha256':'0'*64},self.model(),h.optimizer_for(self.model()))
                    load.assert_not_called()

    def test_checkpoint_rejects_wrong_Adam_dtype_before_changing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                model=self.model();opt=h.optimizer_for(model);c=h.counts('N0')
                model(torch.zeros((1,3))).physical.sum().backward();h.step(model,opt,c)
                selection=h.save_checkpoint(output,'state',model,opt,1)
                value=json.loads(output.path(selection['manifest']).read_text())
                archive=output.path('checkpoints/'+value['archive'])
                with np.load(archive,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
                entry=next(e for e in value['entries'] if e['role']=='adam' and e['name'].endswith('/exp_avg'))
                arrays[entry['key']]=arrays[entry['key']].astype(np.float64)
                entry.update(h.field.record(torch.from_numpy(arrays[entry['key']])))
                altered=output.path('checkpoints/altered.npz')
                with altered.open('xb') as stream:np.savez(stream,**arrays)
                value.update(archive=altered.name,archive_sha256=h.digest(altered),archive_bytes=altered.stat().st_size)
                output.write_json('checkpoints/altered.json',value)
                other=self.model();before=h.field.identity(other)
                selection=dict(manifest='checkpoints/altered.json',manifest_sha256=h.digest(output.path('checkpoints/altered.json')))
                with self.assertRaisesRegex(SignatureError,'Adam counter/moments'):h.restore_checkpoint(output,selection,other,h.optimizer_for(other))
                self.assertEqual(before,h.field.identity(other))

    def test_diagnostic_nonfinite_checkpoint_cannot_be_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                model=self.model();opt=h.optimizer_for(model)
                with torch.no_grad():model.latent[0]=float('nan')
                selection=h.save_checkpoint(output,'failure',model,opt,0,diagnostic=True)
                clean=self.model()
                with self.assertRaises(SignatureError):h.restore_checkpoint(output,selection,clean,h.optimizer_for(clean))

    def test_output_paths_collisions_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            output=h.OwnedOutput(Path(directory),'fixture')
            for unsafe in ('../escape','/tmp/escape','a/../../escape'):
                with self.assertRaises(ValueError):output.path(unsafe)
            output.path('alias').symlink_to('/tmp',target_is_directory=True)
            with self.assertRaises(ValueError):output.path('alias/file')
            output.close();self.assertTrue(output.stream.closed)
            with self.assertRaises(SignatureError):h.OwnedOutput(Path(directory),'fixture')
        with self.assertRaises((SignatureError,OSError)):h.OwnedOutput(h.ROOT,'fixture')

    def test_strict_failure_log_keeps_previous_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with h.OwnedOutput(Path(directory),'fixture') as output:
                output.persist({'status':'failed','failure':'fixture'})
                before=output.path('report.json').read_bytes()
                with self.assertRaises(ValueError):output.persist({'loss':float('nan')})
                self.assertEqual(output.path('report.json').read_bytes(),before)

    def test_evaluation_uses_own_truth_external_tapes_and_no_angle_repair(self):
        model=self.model('N1');shader=FakeShader();before=h.field.identity(model)
        targets=np.ones((84,3,3),np.float64)*.2
        result=h.evaluate(model,self.batch(True),shader,h.counts('N1'),h.previous.reference.truth('M2'),targets,self.reference['scene']['observations'])
        self.assertEqual(h.field.identity(model),before)
        self.assertIs(shader.calls[-1]['tapes'],self.eval_tape)
        self.assertEqual(len(result['predictions']),84)
        self.assertEqual(result['recovery_errors'],h.previous.recovery_errors(result['physical'],h.previous.reference.truth('M2')))
        self.assertIn('relight_unseen_views',result['radiance_errors'])
        self.assertIn('held_out',result['component_compensation'])

    def test_runtime_cleanup_on_fixture_exception(self):
        report={};threads=torch.get_num_threads();rng=torch.get_rng_state().clone();handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with h.controlled_runtime(report):raise RuntimeError('fixture')
        self.assertEqual(torch.get_num_threads(),threads);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(report['memory_limit_restored'] and report['threads_restored'])
        self.assertEqual({s:signal.getsignal(s) for s in handlers},handlers)
        clock=h.HistoryClock(SimpleNamespace(check=lambda:None));clock.wall=900
        with self.assertRaises(h.HistoryLimit):clock.check()

    def test_runtime_RNG_mutation_fails_closed_and_restores_caller(self):
        report={};rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        with self.assertRaisesRegex(SignatureError,'Runtime/RNG integrity'):
            with h.controlled_runtime(report):
                torch.rand(1);np.random.random()
        self.assertFalse(report['global_torch_CPU_RNG_unchanged'])
        self.assertFalse(report['global_numpy_RNG_unchanged'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]))

    def test_phase_B_cannot_authorize_execution_or_allocate_outputs(self):
        with patch.object(h,'OwnedOutput') as allocation,patch.object(h.field,'make_material') as make:
            with self.assertRaises(SignatureError):h.execute_reviewed({},Path('/tmp/no-output'),'fixture')
            allocation.assert_not_called();make.assert_not_called()

    def test_CLI_describe_never_constructs_model_or_shader(self):
        with patch('sys.argv',['neural-material','--describe']),patch.object(h.field,'make_material') as make,\
             patch.object(h.previous.original,'MaterialShaderV1') as shader,contextlib.redirect_stdout(io.StringIO()) as stream:
            h.main()
        value=json.loads(stream.getvalue());self.assertFalse(value['execution_enabled'])
        make.assert_not_called();shader.assert_not_called()
        with patch('sys.argv',['neural-material','--execute']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):h.main()


if __name__=='__main__':unittest.main()
