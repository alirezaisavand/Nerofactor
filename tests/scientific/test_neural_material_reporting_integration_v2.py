"""Actual-harness synthetic lifecycle, corruption and closed-authorization tests."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from experiments import neural_material_recovery_logged_v2 as v2
from experiments.benchmark_neural_material_integration_v2 import synthetic_pair
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.scientific_reporting_adapter_v2 import Tracker,plain,journal_implementation
from infrastructure.signatures import SignatureError


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.scenarios={}
        for s in ('complete','D0_prefix','N0_trajectory','N0_timeout','N1_numerical','N1_timeout','whole_timeout',
                  'initial_failure','before_step','checkpoint_failure','evaluation_failure'):
            cls.scenarios[s]=synthetic_pair(cls.root/s,s,capture_boundaries=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def report(self,case):return self.scenarios[case]['results'][1]['report']
    def test_actual_harness_exact_lifecycle_reports_and_every_boundary(self):
        for s,pair in self.scenarios.items():
            with self.subTest(s=s):
                a,b=pair['results'];self.assertEqual(a['report'],b['report']);self.assertEqual(a['boundaries'],b['boundaries'])
                self.assertEqual(a['failure'],b['failure']);self.assertEqual(b['reporting']['verification_status'],'pending_independent_scan')
        r=self.report('complete');self.assertEqual(r['status'],'complete');self.assertEqual(len(r['outcomes']),27)
        self.assertTrue(all(x['status']=='complete' and x['completed_updates']==3 for x in r['outcomes']))
        self.assertEqual(len(r['initial_admission']),9)
    def test_controls_block_N1_on_prefix_trajectory_or_timeout(self):
        for s in ('D0_prefix','N0_trajectory','N0_timeout'):
            r=self.report(s);self.assertEqual(r['status'],'failed')
            self.assertTrue(all(x['completed_updates']==0 for x in r['outcomes'] if x['arm']=='N1'))
            self.assertEqual(r['failure_category'],'baseline_reproduction')
        self.assertTrue(any(x['status']=='admitted' for x in self.report('N0_timeout')['outcomes']))
    def test_N1_failures_preserved_and_remaining_schedule_continues(self):
        for s,status in (('N1_numerical','failed_numerical'),('N1_timeout','censored')):
            r=self.report(s);self.assertEqual(r['status'],'incomplete')
            self.assertEqual(sum(x['status']==status for x in r['outcomes']),9)
            self.assertTrue(all(x['status']=='complete' for x in r['outcomes'] if x['arm']!='N1'))
    def test_whole_initial_before_step_checkpoint_and_evaluation_failures(self):
        expected={'whole_timeout':('censored','whole_process_resource_censoring'),'initial_failure':('failed','initialization_admission'),
            'before_step':('failed','baseline_reproduction'),'checkpoint_failure':('failed','source_evidence_integrity'),
            'evaluation_failure':('failed','evaluation')}
        for s,(status,category) in expected.items():
            r=self.report(s);self.assertEqual((r['status'],r['failure_category']),(status,category))
        self.assertTrue(all(x['completed_updates']==0 for x in self.report('before_step')['outcomes']))
    def test_preserved_trajectory_snapshots_counters(self):
        r=self.report('complete');rows={(x['material'],x['seed'],x['arm']):x for x in r['outcomes']}
        for m in ('M1','M2','M3'):
            for seed in (101,103,105):
                a,b=(rows[m,seed,arm] for arm in ('D0','N0'))
                self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256'])
                self.assertEqual(a['final_state'],b['final_state']);self.assertTrue(a['prefix_reproduction_passed'])
                self.assertEqual(a['work'],b['work']);self.assertEqual(set(a['snapshots']),{'1','2','3'})
                self.assertEqual(a['snapshots']['2']['compact_state'],b['snapshots']['2']['compact_state'])
    def test_source_graph_same_objects_and_normalized_AST(self):
        audit=v2.source_equivalence_audit();self.assertTrue(audit['training_code_object_identical'])
        self.assertTrue(audit['execute_AST_normalized_exact']);self.assertEqual(audit['optimizer_update_ceiling'],5000)
        self.assertEqual(audit['snapshots'],[250,1000,5000]);self.assertFalse(audit['scientific_execution_enabled'])
    def test_all_recovery_authorizations_closed_before_output(self):
        for review in ({},{'scientific_execution_authorized':True},{'freeze_sha256':v2.FREEZE_SHA},None):
            with self.assertRaises(SignatureError):v2.execute_reviewed(review,self.root/'never-allocated','no-run')
        self.assertFalse((self.root/'never-allocated').exists())
        with mock.patch.object(v2,'V1_SHA','0'*64),self.assertRaises(SignatureError):v2.validate_execution_review({})
    def test_synthetic_or_incomplete_not_scientifically_admitted(self):
        for case in ('complete','N0_timeout'):
            data=self.scenarios[case]['results'][1];r=data['reporting']
            with self.assertRaises(SignatureError):v2.verify_materialized(data['output'],r['identity'],r['final_manifest']['sha256'],True)
    def copied(self,name):
        data=self.scenarios['complete']['results'][1];destination=self.root/name
        shutil.copytree(data['output'],destination)
        return destination,data['reporting']
    def test_missing_reordered_duplicate_truncated_and_changed_records(self):
        for i,mutation in enumerate(('missing','reorder','duplicate','truncate','change')):
            root,r=self.copied('corrupt-'+mutation);path=root/'journal/events.jsonl';lines=path.read_bytes().splitlines(keepends=True)
            if mutation=='missing':del lines[2]
            elif mutation=='reorder':lines[2],lines[3]=lines[3],lines[2]
            elif mutation=='duplicate':lines.insert(3,lines[2])
            elif mutation=='truncate':lines[-1]=lines[-1][:-1]
            else:lines[2]=lines[2].replace(b'"seq":3',b'"seq":4')
            path.write_bytes(b''.join(lines))
            with self.assertRaises((SignatureError,log.LogError)):v2.verify_materialized(root,r['identity'],r['final_manifest']['sha256'])
    def test_missing_corrupt_checkpoint_metadata(self):
        for name in ('missing-checkpoint','corrupted-checkpoint'):
            root,r=self.copied(name);report=json.loads((root/'report.json').read_text())
            p=root/report['outcomes'][0]['snapshots']['1']['manifest']
            if name.startswith('missing'):p.unlink()
            else:p.write_text('{}')
            with self.assertRaises(SignatureError):v2.verify_materialized(root,r['identity'],r['final_manifest']['sha256'])
    def test_changed_identity_or_invalid_final_hash(self):
        root,r=self.copied('identity')
        for identity,sha in ((dict(r['identity'],configuration_sha256='d'*64),r['final_manifest']['sha256']),(r['identity'],'0'*64)):
            with self.assertRaises(log.LogError):v2.verify_materialized(root,identity,sha)
    def test_missing_final_manifest_not_completed(self):
        root,r=self.copied('unfinalized');(root/'journal/final.json').unlink()
        with self.assertRaises(log.LogError):v2.verify_materialized(root,r['identity'],r['final_manifest']['sha256'])


class Adapter(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def allocate(self,name='out'):
        p=self.root/name;p.mkdir();return v2.JournalOutput(p,'synthetic-unit')
    def initial(self):return Tracker(dict(status='running',outcomes=[],synthetic_fixture=True)).root
    def test_aliases_capture_empty_history_and_reject_erasure(self):
        report=self.initial();writer=self.allocate();writer.persist(report)
        row=report.new_row(dict(material='M1',seed=101,arm='D0',status='admitted',completed_updates=0,work={}))
        report['outcomes'].append(row);writer.persist(report)
        row['history']=[];writer.persist(report)
        row['history'].append(dict(update=1,loss_at_update=0,loss=.125));writer.persist(report)
        self.assertIs(report['outcomes'][0],row)
        row['history']=[]
        with self.assertRaises(log.LogError):writer.persist(report)
        writer.close(propagate=False)
    def test_mutations_not_coalesced_deepcopy_not_owner(self):
        report=self.initial();writer=self.allocate();writer.persist(report)
        report['counter']=1;report['counter']=2;report['counter']=3
        self.assertEqual([x[2]['value'] for x in report._owner.pending],[1,2,3])
        self.assertEqual(copy.deepcopy(report),plain(report));self.assertIs(type(copy.deepcopy(report)),dict)
        report['status']='failed';writer.persist(report);writer.close()
        raw=(writer.root/'journal/events.jsonl').read_text();self.assertIn('"value":1',raw);self.assertIn('"value":2',raw)
    def test_duplicate_allocation_capacity_paths_symlinks(self):
        w=self.allocate();w.close()
        with self.assertRaises(SignatureError):v2.JournalOutput(w.root,'new-id')
        with mock.patch.object(v2.shutil,'disk_usage',return_value=__import__('types').SimpleNamespace(free=0)):
            with self.assertRaises(SignatureError):self.allocate('capacity')
        alias=self.root/'alias';alias.symlink_to(self.root/'out',target_is_directory=True)
        with self.assertRaises(SignatureError):v2.JournalOutput(alias,'new-id')
        other=self.allocate('paths')
        with self.assertRaises(Exception):other.path('../escape')
        other.close()
    def test_partial_write_poison_no_resume_or_successful_finalization(self):
        report=self.initial();writer=self.allocate();writer.persist(report);report['extra']=1
        original=os.write
        def short(fd,data):return original(fd,data[:max(1,len(data)//2)])
        with mock.patch.object(os,'write',side_effect=short),self.assertRaises(log.LogError):writer.persist(report)
        self.assertTrue(writer.writer.poisoned)
        with self.assertRaises(log.LogError):writer.close()
        self.assertFalse((writer.root/'journal/final.json').exists())
        self.assertFalse((writer.root/'report.json').exists());self.assertTrue((writer.root/'reporting-failure-report.json').is_file())
    def test_finalization_failure_propagates_and_preserves_primary_exception(self):
        for mode in ('normal','exception'):
            report=self.initial();writer=self.allocate(mode);writer.persist(report);report['status']='failed';writer.persist(report)
            with mock.patch.object(writer.writer,'finalize',side_effect=OSError('synthetic interrupted finalization')):
                if mode=='normal':
                    with self.assertRaises(log.LogError):writer.close()
                else:
                    with self.assertRaisesRegex(ValueError,'primary'):
                        with writer:raise ValueError('primary')
            self.assertFalse((writer.root/'report.json').exists())
            self.assertEqual(json.loads((writer.root/'reporting-result.json').read_text())['verification_status'],'unfinalized')
    def test_duplicate_persist_ID_rejected_even_with_valid_chain(self):
        report=self.initial();writer=self.allocate();writer.persist(report)
        writer.writer._emit('boundary',None,dict(persist_id=1,changes=0,previous_seq=writer.writer.seq))
        report['status']='failed';writer.persist(report);writer.close()
        result=json.loads((writer.root/'reporting-result.json').read_text())
        with self.assertRaises(log.LogError):v2.verify_materialized(writer.root,result['identity'],result['final_manifest']['sha256'])
    def test_missing_commit_boundary_rejected(self):
        report=self.initial();writer=self.allocate();writer.persist(report)
        # Valid chain but a mutation not committed by its persistence boundary.
        writer.writer.append('failure',[log.change('set',['status'],'failed')])
        dict.__setitem__(report,'status','failed')
        writer.close()
        result=json.loads((writer.root/'reporting-result.json').read_text())
        with self.assertRaises(log.LogError):v2.verify_materialized(writer.root,result['identity'],result['final_manifest']['sha256'])
    def test_field_fixture_is_bounded_and_contains_no_shader_or_recovery_calls(self):
        import ast,inspect
        from experiments import benchmark_neural_material_integration_v2 as bench
        self.assertEqual(bench.FIELD_STEPS,10)
        tree=ast.parse(inspect.getsource(bench.field_fixture))
        names=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))]
        self.assertEqual(names.count('make_material'),1);self.assertEqual(names.count('optimizer_for'),1)
        self.assertFalse(set(names)&{'execute_reviewed','train_history','MaterialBatch','MaterialShaderV1','evaluate','restore_checkpoint'})
        self.assertIn('setitimer',names);self.assertIn('gradient_summary',names)
    def test_complete_looking_matrix_cannot_admit_scientific_recovery_in_D2(self):
        report=self.initial();report['synthetic_fixture']=False
        for m in ('M1','M2','M3'):
            for seed in (101,103,105):
                for arm in ('D0','N0','N1'):
                    report['outcomes'].append(dict(material=m,seed=seed,arm=arm,status='complete',completed_updates=5000))
        report['status']='complete';writer=self.allocate();writer.persist(report);writer.close()
        result=json.loads((writer.root/'reporting-result.json').read_text())
        with self.assertRaisesRegex(SignatureError,'admission remains closed'):
            v2.verify_materialized(writer.root,result['identity'],result['final_manifest']['sha256'],True)
    def test_unmapped_mutations_fail_closed(self):
        report=self.initial()
        for op in (lambda:report.clear(),lambda:report.pop('status'),lambda:report['outcomes'].reverse(),lambda:report.__ior__({'x':1})):
            with self.assertRaises(log.LogError):op()
    def test_reporting_introduces_no_RNG_consumption(self):
        import torch,numpy as np
        tr=torch.get_rng_state().clone();nr=np.random.get_state()
        report=self.initial();writer=self.allocate();writer.persist(report);report['status']='failed';writer.persist(report);writer.close()
        self.assertTrue(torch.equal(tr,torch.get_rng_state()));after=np.random.get_state()
        self.assertEqual(nr[0],after[0]);self.assertTrue(np.array_equal(nr[1],after[1]));self.assertEqual(nr[2:],after[2:])


if __name__=='__main__':unittest.main()
