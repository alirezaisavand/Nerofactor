"""D.3 durable commit anchors and constant-size shutdown. No scientific code."""
import json
import os
from pathlib import Path
import time

from experiments import neural_material_recovery_logged_v2 as d2
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.scientific_reporting_adapter_v2 import TrackedDict,journal_implementation,plain
from infrastructure.neural_material_resources_v2 import capacity
from infrastructure.signatures import require

PROTOCOL='neural-material-recovery-logged-v2'
END='execution-end.json'
ANCHOR='commit.json'


def atomic_small(root,name,value):
    data=log.canonical(value)+b'\n';require(len(data)<=log.MAX_SUMMARY_BYTES,'Bounded control metadata exceeded')
    root=Path(root);temp=root/('.'+name+'.tmp')
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        require(os.write(fd,data)==len(data),'Short control write')
        os.fsync(fd)
    finally:os.close(fd)
    os.replace(temp,root/name)
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


class RuntimeOutput(d2.JournalOutput):
    """Original persist locations; no whole-report/hash/scan during shutdown.

    Subclass is bound to an admitted freeze by the future execution graph.
    A SIGKILL/power failure can prevent close. Only an explicitly selected
    durable commit is authoritative; no automatic repair or resume exists.
    """
    freeze=None
    review=None
    def __init__(self,root,run_id):
        require(type(self.freeze) is dict,'Admitted v2 freeze required before output allocation')
        capacity(Path(root).parent,self.freeze['resource_budget']['minimum_free_bytes'])
        super().__init__(root,run_id)
        self.anchor=None;self.anchor_sha=None;self.close_exception=None
        self.commit_seconds=0.;self.closing_seconds=0.;self.authorization_sha256=None
        if self.review is not None:
            self.write_json('execution-review.json',self.review)
            self.authorization_sha256=log.file_sha(self.root/'execution-review.json')
    def persist(self,report):
        start=time.perf_counter();first=self.writer is None
        require(isinstance(report,TrackedDict) and report._path==(),'Tracked report required')
        if self.writer is None:
            self.report=report;self.tracker=report._owner
            self.identity=dict(protocol=PROTOCOL,run_id=self.run_id,source_sha256=log.digest(self.freeze['sha256']),
                configuration_sha256=self.freeze['configuration_sha256'])
            Parent,_=journal_implementation();budget=self.freeze['resource_budget']
            class Guarded(Parent):
                def _emit(self,kind,scope,payload):
                    # Reserve one full maximum event before writing; no overwrite
                    # or partial event to fit capacity. All records remain bounded.
                    if self.bytes+log.MAX_RECORD_BYTES>budget['journal_cap_bytes']:raise OSError(28,'Frozen journal storage cap reached')
                    return super()._emit(kind,scope,payload)
            self.writer=Guarded(self.root/'journal',self.identity,plain(report),approved_parent=self.root,
                preserved_roots=[d2.ROOT,d2.ROOT.parent/'data',d2.ROOT.parent/'final_results'],durability='flush')
            self.tracker.start(self.writer)
        else:
            require(report is self.report,'Mixed report identity');self.tracker.flush(self.writer)
        self.persist_calls+=1;self.persist_seconds+=time.perf_counter()-start
        before=time.perf_counter()
        # Commit order: journal bytes -> filesystem -> bounded atomic anchor.
        # Existing summary.json is advisory and can advance past the last anchor.
        if first:
            identity_fd=os.open(self.root/'journal/identity.json',os.O_RDONLY|os.O_NOFOLLOW)
            try:os.fsync(identity_fd)
            finally:os.close(identity_fd)
            directory=os.open(self.root/'journal',os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        os.fsync(self.writer.fd)
        anchor=dict(schema_version=1,protocol=PROTOCOL,identity=self.identity,identity_manifest_sha256=log.file_sha(self.root/'journal/identity.json'),
            seq=self.writer.seq,head_sha256=self.writer.head,journal_bytes=self.writer.bytes,
            persist_id=self.tracker.persist_id,status=self.writer.report.get('status'),authorization_sha256=self.authorization_sha256)
        atomic_small(self.root,ANCHOR,anchor)
        self.anchor=anchor;self.anchor_sha=log.file_sha(self.root/ANCHOR);self.commit_seconds+=time.perf_counter()-before
    def write_json(self,name,value):
        # Existing checkpoint/diagnostic scientific serialization is reused.
        super().write_json(name,value)
        fd=os.open(self.path(name),os.O_RDONLY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
        if name.startswith(('checkpoints/','diagnostics/')):
            meta=log.decode(self.path(name).read_bytes())
            if 'archive' in meta:
                p=self.path(str(Path(name).parent/meta['archive']))
                cap=self.freeze['resource_budget']['diagnostic_archive_cap_bytes'] if name.startswith('diagnostics/') else self.freeze['resource_budget']['state_file_caps_bytes'][meta['arm']]
                if p.stat().st_size+self.path(name).stat().st_size>cap:raise OSError(28,'Frozen derived-artifact storage cap exceeded')
                fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.fsync(fd)
                finally:os.close(fd)
        directory=os.open(self.path(name).parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    def close(self,propagate=True):
        if self.closed:return
        self.closed=True;begin=time.perf_counter();failure=None
        try:
            if self.writer is None:return
            require(not self.writer.poisoned and not self.tracker.pending and self.anchor is not None,'No clean committed terminal boundary; preserve partial evidence')
            require(self.anchor['seq']==self.writer.seq and self.anchor['head_sha256']==self.writer.head and
                self.anchor['journal_bytes']==self.writer.bytes and self.anchor['status'] in log.TERMINAL,'No terminal committed execution seal')
            # The seal is a bounded claim about the committed prefix. It is NOT
            # a finalized report or successful-science certificate.
            atomic_small(self.root,END,dict(schema_version=1,protocol=PROTOCOL,identity=self.identity,
                anchor_sha256=self.anchor_sha,scientific_status=self.anchor['status'],
                exception_type=self.close_exception,verification_required=True,
                source_inventory_sha256=self.identity['source_sha256'],
                persist_calls=self.persist_calls,commit_seconds=self.commit_seconds,
                shutdown='bounded seal only; offline materialization required'))
        except BaseException as exc:failure=exc
        finally:
            if self.writer is not None:self.writer.close()
            self.closing_seconds=time.perf_counter()-begin
        if failure is not None and propagate:raise failure
    def __exit__(self,exc_type,exc,tb):
        self.close_exception=exc_type.__name__ if exc_type else None
        self.close(propagate=exc_type is None)
