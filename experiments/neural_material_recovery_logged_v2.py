"""Reporting-integrated execution graph. Scientific recovery remains CLOSED.

Only two dict allocations are instrumented. Numerical functions/loop bytecode
are reused from the hash-pinned v1 implementation in a private namespace.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import FunctionType

from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.scientific_reporting_adapter_v2 import Tracker,TrackedDict,journal_implementation,plain
from infrastructure.signatures import SignatureError,require

ROOT=Path(__file__).resolve().parents[1]
V1_PATH=ROOT/'experiments/neural_material_recovery_v1.py'
V1_SHA='afab3e27d7fcddd54830a3b76fa20d8b97ef49a2da6998e7375ba0d3ce41e18c'
FREEZE_SHA='730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad'
PROTOCOL='neural-material-recovery-reporting-v2'
MINIMUM_FREE_BYTES=1024**3


def source_admission():
    require(log.file_sha(V1_PATH)==V1_SHA,'Original v1 source identity changed')
    path=ROOT/'docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json'
    require(log.file_sha(path)==FREEZE_SHA,'Original scientific freeze changed')
    frozen=json.loads(path.read_text())
    for name,sha in frozen['sha256'].items():require(log.file_sha(ROOT/name)==sha,'Original scientific source changed: '+name)
    return frozen


def validate_execution_review(review,root=None,run_id=None):
    source_admission()
    if type(review) is dict and review.get('freeze_sha256')==FREEZE_SHA:
        raise SignatureError('The original v1 freeze cannot authorize reporting v2')
    raise SignatureError('Phase D.2 recovery execution is closed: no reviewed v2 final freeze or execution authorization exists')


def _scientific_v1():
    source_admission()
    from experiments import neural_material_recovery_v1
    return neural_material_recovery_v1


class JournalOutput:
    """V1 output protocol plus bounded live reporting, one final materialization."""
    def __init__(self,root,run_id):
        self.root=Path(root);self.run_id=run_id;self.writer=None;self.tracker=None;self.report=None;self.closed=False
        self.persist_seconds=0.;self.materialization_seconds=0.;self.persist_calls=0
        require(self.root.is_absolute() and self.root.resolve()==self.root and self.root.is_dir() and
            self.root.stat().st_uid==os.geteuid() and not any(self.root.iterdir()),'Explicit fresh owned output required')
        require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (self.root,*self.root.parents)),
            'Output cannot be inside any checkout')
        for p in (ROOT.parent/'data',ROOT.parent/'final_results'):
            require(self.root!=p and p not in self.root.parents and self.root not in p.parents,'Output overlaps preserved inputs')
        require(shutil.disk_usage(self.root).free>=MINIMUM_FREE_BYTES,'Insufficient reporting/checkpoint capacity')
        from infrastructure.runs import validate_run_id
        validate_run_id(run_id)
        self.path('checkpoints').mkdir()
        self.write_json('identity.json',dict(schema_version=1,protocol='neural-material-recovery-v1',run_id=run_id))

    def path(self,name):
        from infrastructure.paths import contained_path
        require(self.root.resolve()==self.root and self.root.stat().st_uid==os.geteuid(),'Owned output was redirected')
        return contained_path(self.root,name,writable=True)

    def write_json(self,name,value):
        with self.path(name).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')

    def persist(self,report):
        start=time.perf_counter()
        require(isinstance(report,TrackedDict) and report._path==(),'V2 requires the live tracked v1 report')
        if self.writer is None:
            self.report=report;self.tracker=report._owner
            self.identity=dict(protocol=PROTOCOL,run_id=self.run_id,
                source_sha256=log.digest(dict(v1=V1_SHA,integration=log.file_sha(Path(__file__).resolve()),
                    adapter=log.file_sha(ROOT/'infrastructure/scientific_reporting_adapter_v2.py'))),
                configuration_sha256=log.digest(dict(settings=plain(report.get('settings',{})),
                    reviewed_freeze_sha256=report.get('reviewed_freeze_sha256'))))
            Writer,_=journal_implementation()
            self.writer=Writer(self.root/'journal',self.identity,plain(report),approved_parent=self.root,
                preserved_roots=[ROOT,ROOT.parent/'data',ROOT.parent/'final_results'],durability='flush')
            self.tracker.start(self.writer)
        else:
            require(report is self.report,'Cannot mix reports or run identities')
            self.tracker.flush(self.writer)
        self.persist_calls+=1;self.persist_seconds+=time.perf_counter()-start

    def close(self,propagate=True):
        if self.closed:return
        self.closed=True
        if self.writer is None:return
        failure=None
        start=time.perf_counter();result=dict(protocol=PROTOCOL,identity=self.identity,persist_calls=self.persist_calls,
            persistence_seconds=self.persist_seconds,reported_changes=self.tracker.total_changes,max_pending_changes=self.tracker.max_pending,
            scientific_admission=False,verification_status='unfinalized')
        try:
            require(not self.tracker.pending,'Uncommitted report mutations at close')
            require(not self.writer.poisoned,'Journal write failed; no implicit repair')
            require(self.report.get('status') in log.TERMINAL,'Nonterminal scientific report remains unfinalized')
            require(log.digest(plain(self.report))==log.digest(self.writer.report),'Lost or changed report mutations')
            selected=self.writer.finalize()
            self.write_json('report.json',self.writer.report)
            result.update(final_manifest=selected,materialized_report_sha256=log.digest(self.writer.report),
                report_file_sha256=log.file_sha(self.root/'report.json'),verification_status='pending_independent_scan')
        except BaseException as exc:
            failure=exc;result.update(failure_type=type(exc).__name__,failure=str(exc))
            # One owned failure-only materialization, never a successful journal.
            try:self.write_json('reporting-failure-report.json',plain(self.report))
            except BaseException as secondary:result['failure_materialization_error']=str(secondary)
        finally:
            self.writer.close();self.materialization_seconds=time.perf_counter()-start
            result['materialization_seconds']=self.materialization_seconds
            result.update(append_seconds=self.tracker.append_seconds,summary_seconds=self.tracker.summary_seconds)
            try:self.write_json('reporting-result.json',result)
            except BaseException:
                if result.get('verification_status')=='pending_independent_scan':raise

        if failure is not None and propagate:raise log.LogError('Reporting finalization failed: '+str(failure)) from failure

    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):self.close(propagate=exc_type is None)


def _execution_graph():
    """Private DI surface for synthetic tests; public scientific gate stays closed."""
    v1=_scientific_v1();namespace=dict(vars(v1),OwnedOutput=JournalOutput,validate_execution_review=validate_execution_review,
        tracked_report=lambda value:Tracker(value).root)
    namespace['train_history']=FunctionType(v1.train_history.__code__,namespace,v1.train_history.__name__,v1.train_history.__defaults__)
    tree=ast.parse(V1_PATH.read_text());original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_reviewed')
    node=copy.deepcopy(original);changes=[]
    for statement in ast.walk(node):
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            if name in ('report','row') and isinstance(statement.value,ast.Call):
                require(isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id=='dict',
                    'Pinned report/row allocation changed')
                statement.value=ast.Call(func=ast.Name(id='tracked_report',ctx=ast.Load()) if name=='report' else
                    ast.Attribute(value=ast.Name(id='report',ctx=ast.Load()),attr='new_row',ctx=ast.Load()),args=[statement.value],keywords=[])
                changes.append(name)
    require(sorted(changes)==['report','row'],'Unexpected report allocation count')
    normalized=copy.deepcopy(node)
    for statement in ast.walk(normalized):
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id in ('report','row') and isinstance(statement.value,ast.Call):
            statement.value=statement.value.args[0]
    require(ast.dump(normalized)==ast.dump(original),'Unexpected scientific/control-flow AST difference')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'pinned-v1-execute-reporting-v2','exec'),namespace)
    return namespace


def source_equivalence_audit():
    namespace=_execution_graph();v1=_scientific_v1()
    names=('MaterialBatch','initial_admission','query','loss_value','backward','optimizer_for','step','prefix_check',
        'vector_transition_digest','compact_state','evaluate','save_checkpoint','persist_failure_capture','controlled_runtime','HistoryClock',
        'field','previous','Settings','work_limits','fresh_latent','inputs','verify_tapes')
    require(all(namespace[n] is getattr(v1,n) for n in names),'Scientific binding changed')
    allowed={'OwnedOutput','validate_execution_review','train_history','execute_reviewed'}
    changed=[n for n,value in vars(v1).items() if namespace[n] is not value]
    require(set(changed)==allowed,'Unexpected scientific data/control-flow global replacement')
    require(namespace['train_history'].__code__ is v1.train_history.__code__,'Scientific loop bytecode changed')
    return dict(v1_sha256=V1_SHA,unchanged_object_bindings=list(names),training_code_object_identical=True,
        execute_AST_normalized_exact=True,changed_original_global_bindings=changed,unchanged_original_global_bindings=len(vars(v1))-len(changed),intentional_changes=['wrap report allocation','wrap row allocation','private output persistence binding','closed v2 authorization binding'],
        optimizer_update_ceiling=v1.UPDATES,snapshots=list(v1.SNAPSHOTS),settings=vars(v1.Settings()),scientific_execution_enabled=False)


def verify_materialized(root,identity,final_sha256,require_scientific_complete=False,verify_checkpoints=True):
    root=Path(root);_,verify=journal_implementation()
    value=verify(root/'journal',identity,expected_final_sha256=final_sha256)
    require(root.is_absolute() and root.resolve()==root and (root/'report.json').resolve()==root/'report.json', 'Aliased report input')
    report=log.decode((root/'report.json').read_bytes())
    require(log.digest(report)==value['materialized_report_sha256'],'Materialized report differs from journal')
    if verify_checkpoints:
        references=list(report.get('failure_artifacts',[]))
        for row in report.get('outcomes',[]):
            references.extend(row.get('snapshots',{}).values())
            if row.get('failure_checkpoint'):references.append(row['failure_checkpoint'])
        from infrastructure.paths import contained_path
        for selection in references:
            require(type(selection) is dict and 'manifest' in selection and 'manifest_sha256' in selection,'Missing checkpoint selection metadata')
            path=contained_path(root,selection['manifest'],writable=False)
            require(path.is_file() and path.resolve()==path and log.file_sha(path)==selection['manifest_sha256'],'Missing/corrupted checkpoint metadata')
            meta=log.decode(path.read_bytes())
            if 'completed_updates' in selection:require(meta.get('completed_updates')==selection['completed_updates'],'Checkpoint counter metadata mismatch')
            if 'archive' in meta:
                archive=contained_path(path.parent,meta['archive'],writable=False)
                require(archive.is_file() and archive.resolve()==archive and log.file_sha(archive)==meta.get('archive_sha256',meta.get('sha256')),'Missing/corrupted numeric checkpoint artifact')
    if require_scientific_complete:
        require(not report.get('synthetic_fixture'), 'Synthetic fixtures cannot establish scientific completion')
        require(report.get('status')=='complete' and len(report.get('outcomes',[]))==27 and
            all(r['status']=='complete' and r['completed_updates']==5000 for r in report['outcomes']),
            'Incomplete/unstarted matrix is not a scientifically completed experiment')
        raise SignatureError('Scientific admission remains closed throughout Phase D.2; no v2 freeze exists')
    return value


def execute_reviewed(review,root,run_id):
    validate_execution_review(review,root,run_id)  # Always closed during D.2, before allocation/model work.
    raise AssertionError('Unreachable without a separately reviewed implementation/freeze revision')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe',required=True,action='store_true');parser.parse_args()
    print(json.dumps(source_equivalence_audit(),sort_keys=True,indent=2))


if __name__=='__main__':main()
