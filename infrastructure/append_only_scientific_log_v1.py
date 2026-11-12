"""Standalone reporting prototype; never imports or executes scientific code.

JSON semantics (including number types and list order) are preserved. Hash
chains detect accidental corruption; a separately trusted final-manifest hash
is required against deliberate rewriting or deletion of a whole valid suffix.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

VERSION=1
MAX_RECORD_BYTES=8*1024**2
MAX_SUMMARY_BYTES=4096
ZERO='0'*64
KINDS={'initialize','metadata','history','checkpoint','status','failure','censor','finalize'}
TERMINAL={'complete','failed','censored','incomplete'}


class LogError(ValueError):
    pass


def require(condition,message):
    if not condition:raise LogError(message)


def validate_json(value,depth=0):
    require(depth<=64,'JSON nesting limit')
    if type(value) is dict:
        require(all(type(k) is str for k in value),'JSON keys must be strings')
        for v in value.values():validate_json(v,depth+1)
    elif type(value) is list:
        for v in value:validate_json(v,depth+1)
    else:require(value is None or type(value) in (str,bool,int) or type(value) is float and math.isfinite(value),'Finite primitive JSON required')


def canonical(value):
    validate_json(value)
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):result.update(part)
    return result.hexdigest()


def _pairs(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'Duplicate JSON key');out[k]=v
    return out


def decode(raw):
    try:value=json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda x:(_ for _ in ()).throw(LogError('Nonfinite JSON')))
    except (UnicodeError,json.JSONDecodeError,RecursionError) as exc:raise LogError('Malformed JSON') from exc
    validate_json(value);return value


def validate_identity(identity):
    require(type(identity) is dict and set(identity)=={'protocol','run_id','source_sha256','configuration_sha256'},'Exact identity schema required')
    require(type(identity['protocol']) is str and 0<len(identity['protocol'])<=128,'Protocol identity required')
    require(type(identity['run_id']) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',identity['run_id']) is not None,'Invalid run ID')
    for k in ('source_sha256','configuration_sha256'):require(type(identity[k]) is str and re.fullmatch('[a-f0-9]{64}',identity[k]) is not None,'Explicit source/config SHA required')


def association(row):
    require(type(row) is dict and all(k in row for k in ('material','seed','arm')),'History identity missing')
    value={k:row[k] for k in ('material','seed','arm')}
    require(type(value['material']) is str and 0<len(value['material'])<=64 and type(value['seed']) is int and
            value['arm'] in ('D0','N0','N1'),'Invalid history association')
    return value


def _validate_rows(report):
    require(type(report) is dict and type(report.get('outcomes')) is list and len(report['outcomes'])<=27,'Report/outcome schema required')
    keys=[tuple(association(row).values()) for row in report['outcomes']]
    require(len(keys)==len(set(keys)),'Duplicate history association')


def _target(report,path):
    node=report
    for k in path:
        require(type(k) is str and type(node) is dict and k in node or
                type(k) is int and type(node) is list and 0<=k<len(node),'Absent/invalid typed JSON path')
        node=node[k]
    return node


def _rollback(undo):
    for parent,key,present,old in reversed(undo):
        if type(parent) is list:
            if present:parent[key]=old
            else:parent.pop()
        elif present:parent[key]=old
        else:del parent[key]


def apply_event(report,kind,scope,payload):
    """Transactional delta application; never copy/serialize the whole history."""
    require(kind in KINDS-{'initialize','finalize'} and type(payload) is dict and set(payload)=={'changes'} and
            type(payload['changes']) is list and 0<len(payload['changes'])<=128,'Exact mutation schema required')
    undo=[]
    try:
        if scope is not None:require(scope==association(scope),'Exact associated history identity required')
        for change in payload['changes']:
            require(type(change) is dict and set(change)=={'op','path','value'} and change['op'] in ('set','append'),'Invalid change schema')
            path=change['path'];require(type(path) is list and 0<len(path)<=32 and all(type(k) in (str,int) for k in path),'Typed nonempty path required')
            value=change['value'];op=change['op'];validate_json(value)
            if path[0]=='outcomes':
                if path==['outcomes']:
                    require(op=='append' and kind=='metadata' and scope==association(value),'Only explicit associated outcome append allowed')
                else:
                    require(len(path)>=3 and type(path[1]) is int and scope==association(_target(report,path[:2])),'Association/index mismatch')
                    require(path[2] not in ('material','seed','arm'),'History identity is immutable')
                    if path[2]=='history':
                        require(kind=='history' and path[2:]==['history'] and op=='append' and type(value) is dict and
                                type(value.get('update')) is int and value['update']>0 and type(value.get('loss_at_update')) is int and
                                value['loss_at_update']==value['update']-1,
                                'History must be ordered append with explicit pre-update label')
                        history=_target(report,path)
                        require(not history or history[-1]['update']<value['update'],'Duplicate/reordered optimizer log')
                    if path[2]=='snapshots' and len(path)==4:
                        require(kind=='checkpoint' and op=='set' and type(path[3]) is str and path[3].isdigit() and
                                type(value) is dict and type(value.get('completed_updates')) is int and value['completed_updates']==int(path[3]),
                                'Checkpoint association/update mismatch')
            else:require(scope is None,'Global changes require null association')
            if path[-1]=='status':
                require(type(value) is str and value in {'constructed','admitted','optimizing','optimized','complete','incomplete',
                        'failed','failed_integrity','failed_numerical','failed_evaluation','censored','running'},'Unknown status')
                expected='censor' if value=='censored' else 'failure' if type(value) is str and value.startswith('failed') else 'status'
                require(kind==expected,'Explicit status/failure/censor event required')
            parent=_target(report,path[:-1]);key=path[-1]
            if op=='append':
                require(type(parent) is dict and key in parent and type(parent[key]) is list,'Append requires an existing list')
                target=parent[key];undo.append((target,len(target),False,None));target.append(value)
            elif type(parent) is dict:
                require(type(key) is str,'Dictionary key must be a string')
                undo.append((parent,key,key in parent,parent.get(key)));parent[key]=value
            else:
                require(type(parent) is list and type(key) is int and 0<=key<len(parent),'Set requires existing list index')
                undo.append((parent,key,True,parent[key]));parent[key]=value
        _validate_rows(report)
        paths=[x['path'] for x in payload['changes']]
        if kind=='history':require(sum(p[-1]=='history' for p in paths)==1,'History event requires exactly one append')
        if kind=='checkpoint':require(any(len(p)==4 and p[0]=='outcomes' and p[2]=='snapshots' for p in paths),'Checkpoint event missing snapshot')
        if kind in ('status','failure','censor'):require(any(p[-1]=='status' for p in paths),'Status event missing classification')
    except BaseException:
        _rollback(undo);raise
    return undo


def change(op,path,value):return dict(op=op,path=path,value=value)


class ScientificLog:
    """Fresh owned allocation, no reopen/resume API. Durability is explicit."""
    def __init__(self,root,identity,initial_report,*,approved_parent,preserved_roots,durability):
        root,parent=Path(root),Path(approved_parent);validate_identity(identity)
        require(durability in ('flush','fsync'),'Explicit flush/fsync policy required')
        require(root.is_absolute() and root.resolve()==root and root.parent==parent and parent.resolve()==parent and
                parent.is_dir() and parent.stat().st_uid==os.geteuid() and os.access(parent,os.W_OK|os.X_OK),'Explicit owned unaliased parent/child required')
        require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (root,*root.parents)),
                'No log allocation inside a checkout')
        require(type(preserved_roots) in (tuple,list) and preserved_roots,'Explicit preserved roots required')
        for p in map(lambda x:Path(x).resolve(),preserved_roots):require(root!=p and p not in root.parents and root not in p.parents,'Output overlaps preserved input/source')
        report=decode(canonical(initial_report));_validate_rows(report)
        require(len(canonical(dict(report=report)))<MAX_RECORD_BYTES-2048,'Initial event exceeds bounded record size')
        self.root=root;self.identity=decode(canonical(identity));self.identity_sha=digest(identity);self.durability=durability
        self.seq=0;self.head=ZERO;self.bytes=0;self.report=None;self.closed=False;self.poisoned=False;self.finalized=False
        root.mkdir(mode=0o700,exist_ok=False);self.owner=(root.stat().st_dev,root.stat().st_ino,os.geteuid())
        self.fd=None
        try:
            self._exclusive('identity.json',canonical(dict(schema_version=VERSION,identity=identity,durability=durability))+b'\n')
            self.fd=os.open(root/'events.jsonl',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW,0o600)
            self.file_identity=(os.fstat(self.fd).st_dev,os.fstat(self.fd).st_ino)
            self._emit('initialize',None,dict(report=report));self.update_summary()
        except BaseException:
            self.close();raise

    def _owned(self):
        stat=self.root.stat()
        require(self.root.resolve()==self.root and (stat.st_dev,stat.st_ino,stat.st_uid)==self.owner,'Owned log root was redirected')

    def _exclusive(self,name,data):
        self._owned();fd=os.open(self.root/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        try:
            require(os.write(fd,data)==len(data),'Short metadata write')
            if self.durability=='fsync':os.fsync(fd)
        finally:os.close(fd)
        if self.durability=='fsync':
            directory=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)

    def _emit(self,kind,scope,payload):
        require(not self.closed and not self.poisoned and not self.finalized,'Writer closed, poisoned or finalized')
        self._owned();stat=os.stat(self.root/'events.jsonl',follow_symlinks=False)
        require((stat.st_dev,stat.st_ino)==self.file_identity and stat.st_size==self.bytes,'Journal changed outside its writer')
        payload=decode(canonical(payload));scope=decode(canonical(scope))
        body=dict(schema_version=VERSION,seq=self.seq+1,previous_sha256=self.head,identity_sha256=self.identity_sha,
                  kind=kind,association=scope,payload=payload)
        encoded=canonical(body);head=hashlib.sha256(encoded).hexdigest();line=canonical(dict(**body,sha256=head))+b'\n'
        require(len(line)<=MAX_RECORD_BYTES,'Event exceeds record bound')
        undo=[]
        if kind=='initialize':require(self.seq==0 and scope is None,'Initialize only once');candidate=payload['report']
        elif kind=='finalize':candidate=self.report
        else:undo=apply_event(self.report,kind,scope,payload);candidate=self.report
        try:
            require(os.write(self.fd,line)==len(line),'Short journal write; preserve incomplete tail')
            if self.durability=='fsync':os.fsync(self.fd)
        except BaseException:
            _rollback(undo);self.poisoned=True;raise
        self.seq+=1;self.head=head;self.bytes+=len(line);self.report=candidate
        return self.head

    def append(self,kind,changes,association=None):
        require(kind in KINDS-{'initialize','finalize'},'Public append requires mutation kind')
        return self._emit(kind,association,dict(changes=changes))

    def update_summary(self):
        require(not self.closed and not self.poisoned,'Summary cannot hide a failed write')
        summary=dict(schema_version=VERSION,identity_sha256=self.identity_sha,last_seq=self.seq,head_sha256=self.head,journal_bytes=self.bytes,
            status=self.report.get('status'),outcome_status_counts=dict(Counter(x.get('status','unknown') for x in self.report['outcomes'])),
            finalized=self.finalized)
        data=canonical(summary)+b'\n';require(len(data)<=MAX_SUMMARY_BYTES,'Summary exceeds fixed bound')
        name='.summary-'+str(self.seq)+'.tmp';self._exclusive(name,data)
        os.replace(self.root/name,self.root/'summary.json')
        if self.durability=='fsync':
            fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        return summary

    def finalize(self):
        require(self.report.get('status') in TERMINAL,'Explicit terminal global status required')
        state_sha=digest(self.report)
        self._emit('finalize',None,dict(report_sha256=state_sha,status=self.report['status']))
        self.finalized=True;self.update_summary()
        value=dict(schema_version=VERSION,identity_sha256=self.identity_sha,last_seq=self.seq,head_sha256=self.head,
            journal_bytes=self.bytes,journal_sha256=file_sha(self.root/'events.jsonl'),report_sha256=state_sha,
            summary_sha256=file_sha(self.root/'summary.json'),identity_manifest_sha256=file_sha(self.root/'identity.json'))
        self._exclusive('final.json',canonical(value)+b'\n')
        return dict(path=str(self.root/'final.json'),sha256=file_sha(self.root/'final.json'))

    def close(self):
        if getattr(self,'fd',None) is not None:os.close(self.fd);self.fd=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def verify(root,identity,*,expected_final_sha256=None,allow_unfinalized=False):
    """Read-only verification/materialization; never repair, resume or truncate."""
    root=Path(root);validate_identity(identity)
    require(root.is_absolute() and root.resolve()==root,'Unaliased absolute log root required')
    for name in ('identity.json','events.jsonl','summary.json'):
        p=root/name;require(p.is_file() and p.resolve()==p and not p.is_symlink(),'Missing/aliased log artifact')
    manifest=decode((root/'identity.json').read_bytes())
    require(type(manifest) is dict and type(manifest.get('schema_version')) is int and
            manifest==dict(schema_version=VERSION,identity=identity,durability=manifest.get('durability')) and
            manifest['durability'] in ('flush','fsync'),'Identity manifest mismatch')
    summary=decode((root/'summary.json').read_bytes());at=summary.get('last_seq')
    require(set(summary)=={'schema_version','identity_sha256','last_seq','head_sha256','journal_bytes','status','outcome_status_counts','finalized'} and
            type(summary['schema_version']) is int and summary['schema_version']==VERSION and type(at) is int and at>=1 and type(summary['finalized']) is bool and
            type(summary['outcome_status_counts']) is dict and all(type(v) is int and v>=0 for v in summary['outcome_status_counts'].values()) and
            (root/'summary.json').stat().st_size<=MAX_SUMMARY_BYTES,'Invalid bounded summary schema')
    seq=0;head=ZERO;size=0;report=None;finalized=False;summary_anchor=None
    with (root/'events.jsonl').open('rb') as stream:
        while True:
            line=stream.readline(MAX_RECORD_BYTES+1)
            if not line:break
            require(len(line)<=MAX_RECORD_BYTES and line.endswith(b'\n'),'Truncated/oversized journal record; no automatic repair')
            record=decode(line)
            require(type(record) is dict and set(record)=={'schema_version','seq','previous_sha256','identity_sha256','kind','association','payload','sha256'},'Exact event envelope required')
            require(canonical(record)+b'\n'==line,'Noncanonical event encoding')
            body={k:v for k,v in record.items() if k!='sha256'}
            require(not finalized and type(record['seq']) is int and record['seq']==seq+1 and type(record['schema_version']) is int and record['schema_version']==VERSION and
                    record['previous_sha256']==head and record['identity_sha256']==digest(identity) and record['kind'] in KINDS and
                    digest(body)==record['sha256'],'Sequence/hash chain/identity mismatch')
            kind,payload,scope=record['kind'],record['payload'],record['association']
            if kind=='initialize':
                require(seq==0 and scope is None and type(payload) is dict and set(payload)=={'report'},'Invalid initialization')
                report=payload['report'];_validate_rows(report)
            elif kind=='finalize':
                require(report is not None and scope is None and payload==dict(report_sha256=digest(report),status=report.get('status')) and
                        report.get('status') in TERMINAL,'Final report binding mismatch');finalized=True
            else:require(report is not None,'Mutation before initialization');apply_event(report,kind,scope,payload)
            seq+=1;head=record['sha256'];size+=len(line)
            if seq==at:summary_anchor=(head,size,report.get('status'),dict(Counter(x.get('status','unknown') for x in report['outcomes'])),finalized)
    require(seq>0,'Empty journal')
    require(summary_anchor is not None and
            (summary['head_sha256'],summary['journal_bytes'],summary['status'],summary['outcome_status_counts'],summary['finalized'])==summary_anchor and
            summary.get('identity_sha256')==digest(identity),'Summary anchor mismatch')
    if expected_final_sha256 is not None:
        path=root/'final.json';require(path.is_file() and path.resolve()==path and file_sha(path)==expected_final_sha256,'Trusted final manifest hash mismatch')
        final=decode(path.read_bytes())
        require(finalized and summary['last_seq']==seq and final==dict(schema_version=VERSION,identity_sha256=digest(identity),last_seq=seq,
            head_sha256=head,journal_bytes=size,journal_sha256=file_sha(root/'events.jsonl'),report_sha256=digest(report),
            summary_sha256=file_sha(root/'summary.json'),identity_manifest_sha256=file_sha(root/'identity.json')),'Final manifest/journal mismatch')
    else:require(allow_unfinalized and not finalized,'A finalized log needs its explicit trusted manifest hash; incomplete prefixes require explicit selection')
    return dict(report=report,last_seq=seq,head_sha256=head,journal_bytes=size,finalized=finalized,summary_lag_records=seq-at,
        materialized_report_sha256=digest(report),identity=identity)
