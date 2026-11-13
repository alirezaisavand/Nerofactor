"""New D.3 synthetic execution, signal, capacity and offline evidence tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from experiments import neural_material_recovery_logged_v2_execution as execution
from experiments import freeze_neural_material_recovery_logged_v2 as freeze
from experiments import finalize_neural_material_journal_v2 as offline
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.neural_material_journal_runtime_v2 import RuntimeOutput,ANCHOR,END
from infrastructure.scientific_reporting_adapter_v2 import Tracker,plain
from infrastructure.neural_material_resources_v2 import budget,capacity
from infrastructure.signatures import SignatureError


def fixture_freeze():
    old=execution.d2.source_admission();resources=budget(old)
    return dict(configuration_sha256='a'*64,sha256={},resource_budget=resources,settings=old['settings'],configuration=old['configuration'],
        initial_models=old['initial_models'],inputs=old['inputs'],tapes=old['tapes'])


def initial(status='running'):
    return Tracker(dict(status=status,synthetic_fixture=True,run_id='synthetic-unit',reviewed_freeze_sha256='b'*64,
        outcomes=[dict(material='M1',seed=101,arm='D0',status='admitted',completed_updates=0,history=[],snapshots={},work={})])).root


def bound(f):
    class Output(RuntimeOutput):pass
    Output.freeze=f
    return Output


def worker(root,kind):
    root=Path(root);root.mkdir();f=fixture_freeze();report=initial();out=bound(f)(root,'synthetic-unit');out.persist(report)
    marker=root.parent/(root.name+'.ready')
    if kind.startswith('offline-'):
        report['status']='failed';out.persist(report);out.close()
        def pause(value,stream,*args,**kwargs):
            stream.write('{\"partial\":');stream.flush();marker.write_text('materialization');signal.pause()
        with mock.patch.object(json,'dump',side_effect=pause):
            offline.finalize_selected(root,log.file_sha(root/ANCHOR),log.file_sha(root/'journal/events.jsonl'),f,'b'*64,root.parent/(root.name+'-post'),log.file_sha(root/END))
    elif kind in ('seal-term','seal-kill'):
        report['status']='failed';out.persist(report)
        from infrastructure import neural_material_journal_runtime_v2 as module
        original=module.atomic_small
        def pause(parent,name,value):
            if name==END:marker.write_text('seal');signal.pause()
            return original(parent,name,value)
        with mock.patch.object(module,'atomic_small',side_effect=pause):out.close()
    elif kind=='write-term':
        class Stop(RuntimeError):pass
        signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(Stop('synthetic TERM')))
        real=os.write
        def partial(fd,data):
            if fd==out.writer.fd:
                real(fd,data[:len(data)//2]);marker.write_text('partial');signal.pause()
            return real(fd,data)
        report['outcomes'][0]['completed_updates']=1
        try:
            with mock.patch.object(os,'write',side_effect=partial):out.persist(report)
        except Stop:out.close(propagate=False)
    else:
        class Stop(RuntimeError):pass
        signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(Stop('synthetic TERM')))
        try:marker.write_text('optimizing');signal.pause()
        except Stop:report['status']='censored';out.persist(report);out.close()


class Lifecycle(unittest.TestCase):
    def test_actual_original_and_D3_runtime_lifecycles(self):
        # Reuse the admitted D2 fixture: no original test suite is rerun. This
        # test substitutes ONLY the newly changed output class. The explicit
        # test subclass materializes outside its measured close for v1 equality;
        # production RuntimeOutput never does this and emits no reporting-result.
        from experiments import benchmark_neural_material_integration_v2 as bench
        f=fixture_freeze()
        class TestOutput(RuntimeOutput):
            def close(self,propagate=True):
                was_closed=self.closed;super().close(propagate)
                if not was_closed and self.report is not None:
                    self.write_json('report.json',plain(self.report))
                    self.write_json('reporting-result.json',dict(verification_status='TEST_ONLY_runtime_sealed_no_production_materialization'))
        TestOutput.freeze=f
        with tempfile.TemporaryDirectory() as t,mock.patch.object(bench.integrated,'JournalOutput',TestOutput):
            for scenario in ('complete','D0_prefix','N0_trajectory','N0_timeout','N1_numerical','N1_timeout','whole_timeout',
                'initial_failure','before_step','checkpoint_failure','evaluation_failure'):
                with self.subTest(scenario=scenario):
                    pair=bench.synthetic_pair(Path(t)/scenario,scenario,capture_boundaries=True)
                    self.assertTrue(pair['semantic_equality']);a,b=pair['results'];self.assertEqual(a['boundaries'],b['boundaries'])
                    root=Path(b['output']);anchor=log.decode((root/ANCHOR).read_bytes())
                    report,meta=offline.scan(root,anchor,log.file_sha(root/'journal/events.jsonl'),False)
                    self.assertEqual(log.digest(report),log.digest(b['report']));self.assertTrue((root/END).is_file())
                    if scenario in ('D0_prefix','N0_trajectory','N0_timeout'):
                        self.assertTrue(all(r['completed_updates']==0 for r in report['outcomes'] if r['arm']=='N1'))


class Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.freeze=fixture_freeze()
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.parent=Path(self.tmp.name)
    def create(self,name='run',status='failed',close=True):
        root=self.parent/name;root.mkdir();report=initial();out=bound(self.freeze)(root,'synthetic-unit');out.persist(report)
        report['status']=status;out.persist(report)
        if close:out.close()
        return out,report
    def finalize(self,out,end=True,prefix=False,name='post'):
        return offline.finalize_selected(out.root,log.file_sha(out.root/ANCHOR),log.file_sha(out.root/'journal/events.jsonl'),
            self.freeze,'b'*64,self.parent/name,log.file_sha(out.root/END) if end else None,prefix)
    def test_first_commit_fsyncs_identity_and_journal_directory_before_anchor(self):
        root=self.parent/'durable';root.mkdir();out=bound(self.freeze)(root,'synthetic-unit');report=initial();calls=[];sync=os.fsync
        def observe(fd):
            calls.append(os.readlink('/proc/self/fd/'+str(fd)));return sync(fd)
        with mock.patch.object(os,'fsync',side_effect=observe):out.persist(report)
        anchor=next(i for i,p in enumerate(calls) if p.endswith('/.commit.json.tmp'))
        for name in ('journal/identity.json','journal','journal/events.jsonl'):
            self.assertLess(calls.index(str(root/name)),anchor)
        report['status']='failed';out.persist(report);out.close()
    def test_checkpoint_catalog_requires_all_Adam_moments_and_exact_counter(self):
        import struct
        root=self.parent/'catalog';root.mkdir();(root/'checkpoints').mkdir()
        model=self.freeze['initial_models']['M1/101/D0']
        parameter={k:model['parameters']['latent'][k] for k in ('dtype','device','shape','sha256')}
        scalar=dict(dtype='torch.float32',device='cpu',shape=[],sha256=hashlib.sha256(struct.pack('<f',250)).hexdigest())
        moment=dict(parameter,sha256='c'*64)
        compact=dict(parameters={'latent':parameter},buffers={},optimizer_effective=self.freeze['configuration']['optimizer_effective'],
            adam={'latent':dict(step=scalar,exp_avg=moment,exp_avg_sq=moment)})
        entries=[dict(role='parameter',name='latent',**parameter),dict(role='torch_rng',name='CPU',dtype='torch.uint8',device='cpu',shape=[5056],sha256='d'*64)]
        entries.extend(dict(role='adam',name='latent/'+k,**v) for k,v in compact['adam']['latent'].items())
        archive=root/'checkpoints/synthetic.bytes';archive.write_bytes(b'inert globally hashed fixture; no tensor decoding')
        report=plain(initial());selection=dict(manifest='checkpoints/state.json',manifest_sha256='',completed_updates=250,compact_state=compact)
        report['outcomes'][0]['snapshots']['250']=selection
        meta=dict(protocol='neural-material-recovery-v1',purpose='state',arm='D0',completed_updates=250,
            source=model['source'],model_configuration=model['configuration'],optimizer_effective=compact['optimizer_effective'],entries=entries,
            execution_provenance=dict(freeze_sha256='b'*64,configuration_sha256=self.freeze['configuration_sha256'],source_sha256=self.freeze['sha256'],run_id='synthetic-unit'),
            archive=archive.name,archive_sha256=log.file_sha(archive),archive_bytes=archive.stat().st_size)
        path=root/selection['manifest'];path.write_text(json.dumps(meta));selection['manifest_sha256']=log.file_sha(path)
        self.assertEqual(len(offline.checkpoint_evidence(root,report,self.freeze,'b'*64)),2)
        for change in ('missing_moment','wrong_counter','missing_catalog'):
            candidate=copy.deepcopy(report);state=candidate['outcomes'][0]['snapshots']['250']['compact_state']['adam']['latent']
            value=copy.deepcopy(meta)
            if change=='missing_moment':del state['exp_avg_sq']
            elif change=='wrong_counter':state['step']['sha256']='0'*64
            else:value['entries']=value['entries'][:-1]
            path.write_text(json.dumps(value));candidate['outcomes'][0]['snapshots']['250']['manifest_sha256']=log.file_sha(path)
            with self.assertRaises(SignatureError):offline.checkpoint_evidence(root,candidate,self.freeze,'b'*64)
    def test_shutdown_never_full_serializes_hashes_or_scans(self):
        out,report=self.create(close=False)
        with mock.patch.object(log,'digest',side_effect=AssertionError('No full-state digest')),\
             mock.patch.object(out,'write_json',side_effect=AssertionError('No report materialization')):
            out.close()
        self.assertLess(out.closing_seconds,1);self.assertFalse((out.root/'report.json').exists());self.assertTrue((out.root/END).is_file())
    def test_failed_censored_and_unstarted_status_preserved(self):
        for status in ('failed','censored','incomplete'):
            out,report=self.create(status,status);result=self.finalize(out,name='post-'+status)
            self.assertEqual(result['original_scientific_status'],status);self.assertFalse(result['scientifically_complete'])
            self.assertEqual(log.decode((self.parent/('post-'+status)/'report.json').read_bytes()),plain(report))
            self.assertEqual(plain(report['outcomes'][0])['status'],'admitted')
    def test_complete_claim_requires_complete_matrix_and_controls(self):
        out,_=self.create(status='complete')
        with self.assertRaisesRegex(SignatureError,'mandatory scientific evidence'):self.finalize(out)
        result=self.finalize(out,prefix=True,name='prefix')
        self.assertFalse(result['scientifically_complete'])
    def test_prefix_only_never_promotes_even_a_complete_looking_seal(self):
        out,_=self.create(status='complete')
        with mock.patch.object(offline,'completion_checks',return_value={'synthetic_all_checks':True}):
            result=self.finalize(out,prefix=True)
        self.assertFalse(result['scientifically_complete'])
        import ast,inspect
        tree=ast.parse(inspect.getsource(offline.finalize_selected))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scientifically_complete' for t in n.targets))
        self.assertTrue(any(isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not) and isinstance(n.operand,ast.Name) and n.operand.id=='allow_prefix' for n in ast.walk(assignment.value)))
    def test_missing_seal_explicit_prefix_only_never_complete(self):
        out,_=self.create(close=False);out.writer.close()
        with self.assertRaises(SignatureError):self.finalize(out,end=False)
        result=self.finalize(out,end=False,prefix=True)
        self.assertEqual(result['classification'],'valid_committed_prefix_only');self.assertFalse(result['scientifically_complete'])
    def test_journal_corruption_duplicate_reorder_and_truncated_record(self):
        for kind in ('duplicate','reorder','changed','truncated'):
            out,_=self.create(kind);path=out.root/'journal/events.jsonl';lines=path.read_bytes().splitlines(keepends=True)
            if kind=='duplicate':lines.insert(2,lines[1])
            elif kind=='reorder':lines[1],lines[2]=lines[2],lines[1]
            elif kind=='changed':lines[0]=lines[0].replace(b'"running"',b'"failed"')
            else:lines[-1]=lines[-1][:-1]
            path.write_bytes(b''.join(lines))
            with self.assertRaises((log.LogError,SignatureError)):self.finalize(out,name='post-'+kind)
    def test_changed_anchor_source_and_checkpoint_hashes_rejected(self):
        out,_=self.create()
        with self.assertRaises(SignatureError):offline.finalize_selected(out.root,'0'*64,log.file_sha(out.root/'journal/events.jsonl'),self.freeze,'b'*64,self.parent/'post')
        changed=copy.deepcopy(self.freeze);changed['sha256']={'wrong':'0'*64}
        with self.assertRaises(SignatureError):offline.finalize_selected(out.root,log.file_sha(out.root/ANCHOR),log.file_sha(out.root/'journal/events.jsonl'),changed,'b'*64,self.parent/'post',log.file_sha(out.root/END))
    def test_missing_and_corrupted_checkpoint_metadata(self):
        for case in ('missing','corrupted'):
            root=self.parent/case;root.mkdir();r=initial();out=bound(self.freeze)(root,'synthetic-unit');out.persist(r)
            r['failure_artifacts'].append({}) if 'failure_artifacts' in r else r.update(failure_artifacts=[dict(manifest='checkpoints/negative.json',manifest_sha256='c'*64)])
            if case=='corrupted':(root/'checkpoints/negative.json').write_text('{}')
            r['status']='failed';out.persist(r);out.close()
            with self.assertRaises((SignatureError,FileNotFoundError)):self.finalize(out,name='post-'+case)
    def test_disk_full_partial_write_no_repair_or_success_seal(self):
        out,_=self.create(close=False);out.report['extra']='changed';real=os.write
        def fail(fd,data):
            if fd==out.writer.fd:real(fd,data[:len(data)//2]);raise OSError(28,'synthetic disk full')
            return real(fd,data)
        with mock.patch.object(os,'write',side_effect=fail),self.assertRaises(OSError):out.persist(out.report)
        self.assertTrue(out.writer.poisoned)
        with self.assertRaises(SignatureError):out.close()
        self.assertFalse((out.root/END).exists())
        with self.assertRaises(SignatureError):self.finalize(out,end=False)
        result=self.finalize(out,end=False,prefix=True)
        self.assertFalse(result['scientifically_complete']);self.assertGreater(result['scan']['uncommitted_tail_bytes'],0)
    def test_missing_or_interrupted_finalizer_cannot_be_success(self):
        out,_=self.create()
        real=json.dump
        def interrupt(value,stream,*args,**kw):
            stream.write('{"partial":');raise OSError(28,'synthetic output disk full')
        with mock.patch.object(json,'dump',side_effect=interrupt),self.assertRaises(OSError):self.finalize(out)
        self.assertFalse((self.parent/'post/final.json').exists());self.assertTrue((out.root/END).exists())
        with self.assertRaises(SignatureError):self.finalize(out)
    def test_output_capacity_id_and_path_rejection(self):
        out,_=self.create()
        with self.assertRaises(SignatureError):bound(self.freeze)(out.root,'synthetic-unit')
        with mock.patch('shutil.disk_usage',return_value=type('Space',(),{'free':0})()),self.assertRaises(SignatureError):capacity(self.parent,1)
        alias=self.parent/'alias';alias.symlink_to(out.root,target_is_directory=True)
        with self.assertRaises(SignatureError):bound(self.freeze)(alias,'synthetic-unit')
        with self.assertRaises(Exception):out.path('../outside')
    def test_atomic_anchor_failure_keeps_prior_committed_prefix(self):
        out,_=self.create(close=False);original=(out.root/ANCHOR).read_bytes();out.report['extra']='new'
        with mock.patch('infrastructure.neural_material_journal_runtime_v2.atomic_small',side_effect=OSError(28,'disk full')),self.assertRaises(OSError):out.persist(out.report)
        self.assertEqual((out.root/ANCHOR).read_bytes(),original)
        out.close(propagate=False);self.assertFalse((out.root/END).exists())
        self.assertFalse(self.finalize(out,end=False,prefix=True)['scientifically_complete'])
    def test_synthetic_SIGTERM_and_SIGKILL_boundaries(self):
        for kind in ('optimization-term','write-term','seal-term','seal-kill','offline-term','offline-kill'):
            root=self.parent/kind;marker=self.parent/(kind+'.ready')
            process=subprocess.Popen(['python3','-B','-m','tests.scientific.test_neural_material_logged_freeze_v2','--signal-worker',str(root),kind],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            self.addCleanup(lambda p=process:p.kill() if p.poll() is None else None)
            deadline=time.monotonic()+15
            while not marker.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
            if not marker.exists():
                process.kill();_,error=process.communicate();self.fail('Worker admission/marker failed: '+error.decode())
            process.send_signal(signal.SIGKILL if kind.endswith('-kill') else signal.SIGTERM)
            _,error=process.communicate(timeout=5)
            self.assertIsNotNone(process.returncode)
            if kind=='optimization-term':self.assertEqual(process.returncode,0,error.decode());self.assertTrue((root/END).exists())
            elif kind.startswith('offline-'):
                self.assertTrue((root/END).exists());self.assertFalse((self.parent/(kind+'-post')/'final.json').exists())
            else:self.assertFalse((root/END).exists())
            anchor=log.decode((root/ANCHOR).read_bytes());report,stats=offline.scan(root,anchor,log.file_sha(root/'journal/events.jsonl'),kind=='write-term')
            self.assertEqual(report['status'],'censored' if kind=='optimization-term' else 'failed' if kind.startswith(('seal','offline')) else 'running')
            if kind=='write-term':self.assertGreater(stats['uncommitted_tail_bytes'],0)


class Admission(unittest.TestCase):
    def test_original_v1_missing_and_unapproved_authorizations_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'never-created'
            for review in (None,{},dict(phase='D',scientific_execution_authorized=True,freeze_sha256=execution.d2.FREEZE_SHA)):
                with self.assertRaises(SignatureError):execution.execute_reviewed(review,p,'3d2-recovery-v2-never')
            self.assertFalse(p.exists())
    def test_correct_review_schema_runtime_source_output_and_watchdog_checks(self):
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);old=execution.d2.source_admission();r=fixture_freeze()
            r.update(output_policy=dict(approved_parent=t,required_run_prefix='3d2-recovery-v2-'))
            rid='3d2-recovery-v2-synthetic';path=parent/rid
            review=dict(phase=freeze.PHASE,scientific_execution_authorized=True,freeze_path=str(parent/'fake-freeze.json'),freeze_sha256='b'*64,
                source_inventory_sha256=log.digest(r['sha256']),approved_output_parent=t,approved_output_root=str(path),approved_run_id=rid,external_watchdog=freeze.WATCHDOG)
            with mock.patch.object(freeze,'validate_freeze',return_value=r):
                self.assertEqual(execution.validate_execution_review(review,path,rid),r)
                for key,value in (('phase','D'),('scientific_execution_authorized',False),('freeze_sha256',execution.d2.FREEZE_SHA),('source_inventory_sha256','0'*64),('approved_output_root',str(parent/'elsewhere')),('external_watchdog',dict(wall_seconds=28801,kill_grace_seconds=10))):
                    wrong=dict(review);wrong[key]=value
                    with self.assertRaises(SignatureError):execution.validate_execution_review(wrong,path,rid)
                path.mkdir()
                with self.assertRaises(SignatureError):execution.validate_execution_review(review,path,rid)
                self.assertFalse(any(path.iterdir()))
    def test_invalid_freeze_source_or_hash_blocks_before_allocation(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'3d2-recovery-v2-test'
            review=dict(phase=freeze.PHASE,scientific_execution_authorized=True,freeze_path=str(Path(t)/'missing'),freeze_sha256='a'*64,
                source_inventory_sha256='b'*64,approved_output_parent=t,approved_output_root=str(path),approved_run_id=path.name,external_watchdog=freeze.WATCHDOG)
            with self.assertRaises(SignatureError):execution.execute_reviewed(review,path,path.name)
            self.assertFalse(path.exists())
    def test_aggregate_relocation_admits_only_identical_repo_local_metadata(self):
        from experiments import neural_material_validation_admission_v1 as aggregate
        old=execution.d2.source_admission();stored=freeze.selected(freeze.ROOT/old['phase_c']['path'],old['phase_c']['sha256'])
        actual=copy.deepcopy(stored);actual['completion_versioned']['path']=str(freeze.ROOT/aggregate.COMPLETION_PATH)
        with mock.patch.object(aggregate,'verify_aggregate',return_value=actual):
            result=freeze.admit_aggregate(old)
        self.assertTrue(result['all_other_fields_exact']);self.assertEqual(result['permitted_metadata_relocation']['field'],'completion_versioned/path')
        actual['initial_shader_gradients']-=1
        with mock.patch.object(aggregate,'verify_aggregate',return_value=actual),self.assertRaisesRegex(SignatureError,'beyond admitted'):
            freeze.admit_aggregate(old)
    def test_freeze_factory_deterministic_false_authorization_no_hash_cycle(self):
        old=freeze.lineage();resources=budget(old)
        a=freeze.expected_freeze(old,resources,'f'*64);b=freeze.expected_freeze(old,resources,'f'*64)
        self.assertEqual(a,b);self.assertFalse(a['authorization']['execution_authorized'])
        self.assertEqual(a['configuration']['external_watchdog'],{'wall_seconds':28800,'kill_grace_seconds':10})
        self.assertNotIn(freeze.FREEZE,a['sha256']);self.assertNotIn(freeze.READINESS,a['sha256'])
        self.assertEqual(a['settings'],old['settings']);self.assertEqual(a['expected_total_work'],old['expected_total_work'])
        self.assertEqual(a['initial_models'],old['initial_models']);self.assertTrue(a['initial_gate_schema']['checks'])
    def test_scientific_graph_settings_and_work_unchanged(self):
        a=execution.source_equivalence();self.assertTrue(a['scientific_bytecode_identical']);self.assertTrue(a['unchanged_scientific_globals'])
        old=execution.d2.source_admission();resources=budget(old);cfg=freeze.contract(old,resources)
        for key,value in old['configuration'].items():self.assertEqual(cfg[key],value)
        self.assertEqual(cfg['settings']['updates'],5000);self.assertEqual(cfg['snapshots'],[250,1000,5000])
        self.assertEqual(sum(cfg['work_per_history'][a]['optimizer_returns']*9 for a in ('D0','N0','N1')),135000)
        self.assertFalse(cfg['execution_authorized']);self.assertGreater(resources['minimum_free_bytes'],resources['subtotal_bytes'])
        self.assertTrue(resources['memory']['fits_accounting']);self.assertFalse(resources['memory']['guarantee'])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--signal-worker':worker(sys.argv[2],sys.argv[3])
    else:unittest.main()
