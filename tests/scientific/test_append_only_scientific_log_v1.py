"""Synthetic report equivalence/corruption tests; no scientific imports."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infrastructure import append_only_scientific_log_v1 as log

IDENTITY=dict(protocol='synthetic-report-fixture-v1',run_id='synthetic-unit',source_sha256='a'*64,configuration_sha256='b'*64)


def initial(arm='D0'):
    return dict(schema_version=1,synthetic=True,status='running',source={'commit':'c'*40},settings={'unchanged':True},
        outcomes=[dict(material='M1',seed=101,arm=arm,status='admitted',completed_updates=0,history=[],snapshots={},work={'optimizer_returns':0})])


class Tests(unittest.TestCase):
    def writer(self,parent,name='run',arm='D0',durability='flush'):
        return log.ScientificLog(Path(parent)/name,IDENTITY,initial(arm),approved_parent=Path(parent),
            preserved_roots=[Path(parent)/'preserved'],durability=durability)

    def history(self,w,report,update):
        row=report['outcomes'][0]
        observation=dict(update=update,loss_at_update=update-1,loss=.125,gradient_norm=.25,
            physical_before_update=[.2,.5,-.3,.1,.2,.3,.4],gradient_groups={'predictor':{'finite':True,'nonzero':7}},
            trajectory_sha256=('%064x'%update),work={'optimizer_returns':update})
        row['history'].append(copy.deepcopy(observation));row['completed_updates']=update;row['work']['optimizer_returns']=update
        w.append('history',[log.change('append',['outcomes',0,'history'],observation),
            log.change('set',['outcomes',0,'completed_updates'],update),log.change('set',['outcomes',0,'work','optimizer_returns'],update)],log.association(row))
        if update in (250,1000,5000):
            snapshot=dict(completed_updates=update,manifest='checkpoints/synthetic-'+str(update)+'.json',manifest_sha256='d'*64,
                compact_state={'parameter_sha256':'e'*64,'adam_sha256':'f'*64})
            row['snapshots'][str(update)]=copy.deepcopy(snapshot)
            w.append('checkpoint',[log.change('set',['outcomes',0,'snapshots',str(update)],snapshot)],log.association(row))

    def finished_fixture(self,parent,name='run',arm='D0',status='complete',full=True,durability='flush'):
        report=initial(arm);w=self.writer(parent,name,arm,durability)
        updates=[1,*range(10,5001,10)] if full else [1,10,250,1000,4860]
        if status=='admitted':updates=[]
        for update in updates:self.history(w,report,update)
        row=report['outcomes'][0];kind='censor' if status=='censored' else 'failure' if status.startswith('failed') else 'status'
        row['status']=status;w.append(kind,[log.change('set',['outcomes',0,'status'],status)],log.association(row))
        if status=='complete':
            row['evaluation']={'physical_pass':False,'predictions':[[.1,.2,.3]],'component_errors':{'diffuse':.02,'specular':.03}}
            w.append('metadata',[log.change('set',['outcomes',0,'evaluation'],copy.deepcopy(row['evaluation']))],log.association(row))
        report['status']='complete' if status=='complete' else 'failed' if status.startswith('failed') else 'incomplete'
        kind='failure' if report['status']=='failed' else 'status'
        w.append(kind,[log.change('set',['status'],report['status'])]);selected=w.finalize();w.close()
        return report,selected

    def test_exact_full_D0_N0_N1_and_negative_report_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            cases=[('D0','complete',True),('N0','complete',True),('N0','censored',False),('N1','complete',True),
                ('N1','failed_numerical',False),('N0','admitted',False),('D0','failed_integrity',False)]
            for i,(arm,status,full) in enumerate(cases):
                with self.subTest(arm=arm,status=status):
                    report,selected=self.finished_fixture(directory,'run'+str(i),arm,status,full)
                    rebuilt=log.verify(Path(directory)/('run'+str(i)),IDENTITY,expected_final_sha256=selected['sha256'])
                    self.assertEqual(log.canonical(report),log.canonical(rebuilt['report']))
                    self.assertTrue(rebuilt['finalized']);self.assertEqual(rebuilt['summary_lag_records'],0)
                    self.assertLessEqual((Path(directory)/('run'+str(i))/'summary.json').stat().st_size,log.MAX_SUMMARY_BYTES)

    def test_global_integrity_failure_and_multiple_associations(self):
        with tempfile.TemporaryDirectory() as directory:
            report=initial();w=self.writer(directory)
            row=dict(material='M3',seed=105,arm='N1',status='admitted',completed_updates=0,history=[],snapshots={},work={})
            w.append('metadata',[log.change('append',['outcomes'],row)],log.association(row));report['outcomes'].append(copy.deepcopy(row))
            w.append('failure',[log.change('set',['status'],'failed'),log.change('set',['failure'],'synthetic source mismatch')])
            report.update(status='failed',failure='synthetic source mismatch');selected=w.finalize();w.close()
            self.assertEqual(log.canonical(log.verify(Path(directory)/'run',IDENTITY,expected_final_sha256=selected['sha256'])['report']),log.canonical(report))

    def test_duplicate_missing_reordered_changed_truncated_and_extra_records(self):
        with tempfile.TemporaryDirectory() as directory:
            _,selected=self.finished_fixture(directory,full=False);root=Path(directory)/'run';path=root/'events.jsonl';original=path.read_bytes();lines=original.splitlines(keepends=True)
            variants=[b''.join(lines[:2]+lines[3:]),b''.join(lines[:2]+[lines[1]]+lines[2:]),
                b''.join([lines[1],lines[0],*lines[2:]]),original.replace(b'"loss":0.125',b'"loss":0.126',1),
                original[:-1],b''.join(lines[:-1]),original+lines[-1],b'{malformed}\n'+b''.join(lines[1:])]
            for value in variants:
                with self.subTest(bytes=len(value)):
                    path.write_bytes(value)
                    with self.assertRaises(log.LogError):log.verify(root,IDENTITY,expected_final_sha256=selected['sha256'])
            path.write_bytes(original)

    def test_changed_identity_and_external_final_anchor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _,selected=self.finished_fixture(directory,full=False);root=Path(directory)/'run'
            with self.assertRaises(log.LogError):log.verify(root,dict(IDENTITY,configuration_sha256='c'*64),expected_final_sha256=selected['sha256'])
            with self.assertRaises(log.LogError):log.verify(root,IDENTITY,expected_final_sha256='0'*64)
            with self.assertRaises(log.LogError):log.verify(root,IDENTITY)

    def test_final_manifest_binds_the_durability_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            _,selected=self.finished_fixture(directory,full=False);root=Path(directory)/'run';path=root/'identity.json'
            value=json.loads(path.read_text());value['durability']='fsync';path.write_bytes(log.canonical(value)+b'\n')
            with self.assertRaisesRegex(log.LogError,'manifest/journal'):log.verify(root,IDENTITY,expected_final_sha256=selected['sha256'])

    def test_bool_is_not_a_seed_or_pre_update_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);scope=dict(material='M1',seed=True,arm='D0')
            with self.assertRaises(log.LogError):w.append('metadata',[log.change('set',['outcomes',0,'note'],'wrong type')],scope)
            with self.assertRaises(log.LogError):w.append('history',[log.change('append',['outcomes',0,'history'],
                dict(update=2,loss_at_update=True))],dict(material='M1',seed=101,arm='D0'))
            w.close()

    def test_independent_verification_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            _,selected=self.finished_fixture(directory,full=False);root=Path(directory)/'run'
            before={p.name:log.file_sha(p) for p in root.iterdir() if p.is_file()}
            log.verify(root,IDENTITY,expected_final_sha256=selected['sha256'])
            self.assertEqual(before,{p.name:log.file_sha(p) for p in root.iterdir() if p.is_file()})

    def test_unfinalized_complete_prefix_and_summary_lag_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);w.append('metadata',[log.change('set',['note'],'last complete event')]);w.close()
            with self.assertRaises(log.LogError):log.verify(Path(directory)/'run',IDENTITY)
            value=log.verify(Path(directory)/'run',IDENTITY,allow_unfinalized=True)
            self.assertFalse(value['finalized']);self.assertEqual(value['summary_lag_records'],1)
            self.assertEqual(value['report']['note'],'last complete event')
            with (Path(directory)/'run/events.jsonl').open('ab') as stream:stream.write(b'{"partial":')
            with self.assertRaisesRegex(log.LogError,'Truncated'):log.verify(Path(directory)/'run',IDENTITY,allow_unfinalized=True)

    def test_summary_corruption_detected_even_without_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);w.close();path=Path(directory)/'run/summary.json'
            summary=json.loads(path.read_text());summary['outcome_status_counts']={'complete':27};path.write_bytes(log.canonical(summary)+b'\n')
            with self.assertRaisesRegex(log.LogError,'anchor'):log.verify(Path(directory)/'run',IDENTITY,allow_unfinalized=True)

    def test_bad_mutation_rolls_back_without_event_or_state_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);before=log.canonical(w.report);seq=w.seq;head=w.head;size=w.bytes
            with self.assertRaises(log.LogError):w.append('metadata',[log.change('set',['new'],'must roll back'),log.change('set',['absent','key'],3)])
            self.assertEqual(log.canonical(w.report),before);self.assertEqual((w.seq,w.head,w.bytes),(seq,head,size));w.close()

    def test_schema_nonfinite_unknown_kind_association_and_duplicate_update_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);report=initial();self.history(w,report,1)
            for kind,changes,scope in [('unknown',[log.change('set',['note'],1)],None),
                ('metadata',[log.change('set',['note'],float('nan'))],None),
                ('metadata',[log.change('set',['outcomes',0,'seed'],103)],log.association(report['outcomes'][0])),
                ('metadata',[log.change('set',['outcomes',0,'note'],'wrong seed')],dict(material='M1',seed=103,arm='D0')),
                ('metadata',[log.change('set',['status'],'censored')],None),
                ('history',[log.change('append',['outcomes',0,'history'],report['outcomes'][0]['history'][0])],log.association(report['outcomes'][0]))]:
                with self.subTest(kind=kind),self.assertRaises(log.LogError):w.append(kind,changes,scope)
            w.close()

    def test_owner_path_collision_symlink_and_preserved_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);w=self.writer(parent);w.close()
            with self.assertRaises(FileExistsError):self.writer(parent)
            for path,preserved in ((parent/'../escape',[parent/'preserved']),(parent/'inside',[parent]),(parent/'preserved'/'run',[parent/'preserved'])):
                with self.assertRaises((log.LogError,FileNotFoundError)):log.ScientificLog(path,IDENTITY,initial(),approved_parent=parent,preserved_roots=preserved,durability='flush')
            (parent/'alias').symlink_to(parent/'run',target_is_directory=True)
            with self.assertRaises(log.LogError):log.ScientificLog(parent/'alias',IDENTITY,initial(),approved_parent=parent,preserved_roots=[parent/'preserved'],durability='flush')

    def test_git_placeholder_is_not_a_checkout_but_real_git_markers_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);(parent/'.git').mkdir()
            w=self.writer(parent);w.close()
            (parent/'.git/HEAD').write_text('ref: refs/heads/synthetic\n')
            with self.assertRaisesRegex(log.LogError,'checkout'):self.writer(parent,'second')

    def test_short_write_poisoning_no_implicit_retry_or_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);before=log.canonical(w.report)
            with mock.patch.object(os,'write',return_value=1):
                with self.assertRaisesRegex(log.LogError,'Short'):w.append('metadata',[log.change('set',['note'],'short write')])
            self.assertTrue(w.poisoned);self.assertEqual(log.canonical(w.report),before)
            with self.assertRaises(log.LogError):w.append('metadata',[log.change('set',['note'],'no retry')])
            with self.assertRaises(log.LogError):w.update_summary()
            w.close()

    def test_caller_mutation_cannot_change_written_state(self):
        with tempfile.TemporaryDirectory() as directory:
            w=self.writer(directory);data={'nested':[1,2]};w.append('metadata',[log.change('set',['note'],data)]);data['nested'].append(3)
            self.assertEqual(w.report['note'],{'nested':[1,2]});w.close()

    def test_fsync_policy_and_finalization_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(os,'fsync',wraps=os.fsync) as synced:
                w=self.writer(directory,durability='fsync')
                with self.assertRaises(log.LogError):w.finalize()
                w.append('censor',[log.change('set',['status'],'censored')]);selected=w.finalize()
                with self.assertRaises(log.LogError):w.finalize()
                self.assertGreater(synced.call_count,0);w.close()
            self.assertTrue(log.verify(Path(directory)/'run',IDENTITY,expected_final_sha256=selected['sha256'])['finalized'])

    def test_realistic_benchmark_workers_are_synthetic_and_semantically_paired(self):
        from experiments import benchmark_neural_material_reporting_v1 as bench
        raw=initial();raw['outcomes']=[]
        meta=dict(dtype='torch.float32',device='cpu',shape=[7],sha256='a'*64)
        for arm in ('D0','N0','N1'):
            for material in ('M1','M2','M3'):
                for seed in (101,103,105):
                    row=initial(arm)['outcomes'][0];row.update(material=material,seed=seed)
                    row['initial_model']=dict(parameters={'latent':meta},buffers={})
                    row['history']=[dict(update=1,loss_at_update=0,loss=.1,gradient_norm=.2,work={},physical_before_update=[.2]*7)]
                    row['snapshots']={'250':{'compact_state':{'optimizer_effective':{'lr':.01}}}}
                    raw['outcomes'].append(row)
        raw['initial_admission']=[{'gradients':[{}, {}, {'synthetic_head':dict(parameters=7,finite=True,nonzero=7,missing=0,squared_norm=.01)}]}]
        before=log.canonical(raw)
        item=bench.fixture(raw,'early');self.assertEqual(log.canonical(raw),before)
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);path=parent/'fixture.json';path.write_text(json.dumps(item))
            results=[bench.reporting_worker(path,log.file_sha(path),parent/arm,arm) for arm in ('R0','R1')]
            self.assertEqual(results[0]['final_canonical_report_sha256'],results[1]['final_canonical_report_sha256'])
            self.assertTrue(results[1]['semantic_equality']);self.assertEqual(results[0]['window_events'],3)
        for name in ('experiments.material_shader_v1','experiments.neural_material_recovery_v1'):
            with self.assertRaises(AssertionError):bench.NoScience().find_spec(name)


if __name__=='__main__':unittest.main()
