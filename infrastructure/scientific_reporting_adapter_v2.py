"""Tracked report mutations and private v2 journal compatibility; no science."""
import ast
import copy
import time
from pathlib import Path

from infrastructure import append_only_scientific_log_v1 as legacy

PROTOTYPE_SHA='ef1237696db6735667a6d963df940cdbbc7d5a4160dded581f68b44f3eeb5625'


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,list):return [plain(v) for v in value]
    return value


def _apply_v2(report,kind,scope,payload):
    if kind=='boundary':
        legacy.require(scope is None and type(payload) is dict and set(payload)=={'persist_id','changes','previous_seq'} and
            all(type(v) is int and v>=0 for v in payload.values()),'Invalid persistence boundary schema')
        return []
    # V1 training creates an empty history AFTER its admitted row was persisted.
    # This adds no observations, permits no clearing, and preserves absence vs [].
    legacy.require(type(payload) is dict and set(payload)=={'changes'} and type(payload['changes']) is list and
        0<len(payload['changes'])<=128,'Exact mutation schema required')
    undo=[];ordinary=[]
    def flush():
        if ordinary:
            undo.extend(legacy.apply_event(report,kind,scope,dict(changes=list(ordinary))));ordinary.clear()
    try:
        for c in payload['changes']:
            p=c.get('path') if type(c) is dict else None
            special=kind=='metadata' and type(p) is list and len(p)==3 and p[0]=='outcomes' and p[2]=='history'
            if special:
                flush()
                legacy.require(set(c)=={'op','path','value'} and c['op']=='set' and type(c['value']) is list and not c['value'],
                    'Only empty history initialization is permitted')
                row=legacy._target(report,p[:2]);legacy.require(scope==legacy.association(row) and scope==legacy.association(scope),
                    'History initialization association mismatch')
                legacy.require('history' not in row or row['history']==[],'Cannot erase recorded history')
                undo.append((row,'history','history' in row,row.get('history')));row['history']=[]
            else:ordinary.append(c)
        flush()
    except BaseException:
        legacy._rollback(undo);raise
    return undo


def journal_implementation():
    """Private compiled reporting definitions; never patch the v1 module."""
    source=Path(legacy.__file__).resolve()
    legacy.require(legacy.file_sha(source)==PROTOTYPE_SHA,'Pinned D.1 reporter changed')
    tree=ast.parse(source.read_text())
    nodes=copy.deepcopy([n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('ScientificLog','verify')])
    legacy.require(len(nodes)==2,'Pinned reporting definitions missing')
    verifier=next(n for n in nodes if n.name=='verify')
    start=next(i for i,n in enumerate(verifier.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='seq' for t in n.targets))
    verifier.body[start:start]=ast.parse('expected_persist=1\npending_changes=0').body
    loop=next(n for n in ast.walk(verifier) if isinstance(n,ast.While))
    index=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and
        [x.id for x in t.elts if isinstance(x,ast.Name)]==['kind','payload','scope'] for t in n.targets))
    checks=ast.parse('''
if kind=='boundary':
    require(type(payload) is dict and set(payload)=={'persist_id','changes','previous_seq'} and
        all(type(v) is int for v in payload.values()) and payload['persist_id']==expected_persist and
        payload['changes']==pending_changes and payload['previous_seq']==seq,'Missing, duplicate or reordered persistence boundary')
    expected_persist+=1
    pending_changes=0
elif kind=='finalize':
    require(expected_persist>1 and pending_changes==0,'Finalization before a committed persistence boundary')
elif kind!='initialize':
    require(type(payload) is dict and type(payload.get('changes')) is list,'Invalid mutation batch')
    pending_changes+=len(payload['changes'])
''').body
    loop.body[index+1:index+1]=checks
    returned=next(n for n in reversed(verifier.body) if isinstance(n,ast.Return))
    returned.value.keywords.extend([ast.keyword(arg='persist_boundaries',value=ast.parse('expected_persist-1',mode='eval').body),
        ast.keyword(arg='uncommitted_changes',value=ast.Name(id='pending_changes',ctx=ast.Load()))])
    namespace=dict(vars(legacy),VERSION=2,KINDS=legacy.KINDS|{'boundary'},apply_event=_apply_v2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'pinned-reporting-v2-private','exec'),namespace)
    return namespace['ScientificLog'],namespace['verify']


class Tracker:
    """O(new mutations) queues. No dirty scans, coalescing, or lost digests."""
    def __init__(self,value):
        self.pending=[];self.persist_id=0;self.total_changes=0;self.max_pending=0;self.append_seconds=0.;self.summary_seconds=0.
        self.root=TrackedDict(value,self,())

    def emit(self,path,op,value):
        if path is None:return
        kind='metadata';scope=None
        if path and path[0]=='outcomes':
            if len(path)==1:scope=legacy.association({k:value[k] for k in ('material','seed','arm')})
            else:scope=legacy.association({k:self.root['outcomes'][path[1]][k] for k in ('material','seed','arm')})
            if len(path)==3 and path[2]=='history' and op=='append':kind='history'
            if len(path)==4 and path[2]=='snapshots':kind='checkpoint'
        if path[-1]=='status':kind='censor' if value=='censored' else 'failure' if str(value).startswith('failed') else 'status'
        self.pending.append((kind,scope,legacy.change(op,list(path),plain(value))))
        self.total_changes+=1;self.max_pending=max(self.max_pending,len(self.pending))

    def flush(self,writer):
        started=time.perf_counter();count=len(self.pending);index=0
        while index<count:
            kind,scope,_=self.pending[index];end=index+1
            while end<count and end-index<128 and kind not in ('history','checkpoint') and self.pending[end][:2]==(kind,scope):end+=1
            writer.append(kind,[v[2] for v in self.pending[index:end]],scope);index=end
        self.persist_id+=1
        writer._emit('boundary',None,dict(persist_id=self.persist_id,changes=count,previous_seq=writer.seq))
        appended=time.perf_counter();writer.update_summary();finished=time.perf_counter()
        self.append_seconds+=appended-started;self.summary_seconds+=finished-appended;self.pending.clear()

    def start(self,writer):
        # Initial state is already in the initialization record, not replayed.
        self.pending.clear();self.flush(writer)


def wrap(value,owner,path):
    if isinstance(value,(TrackedDict,TrackedList)):
        legacy.require(value._owner is owner and value._path is None,'Tracked value was reused or attached twice')
        value._bind(path);return value
    if isinstance(value,dict):return TrackedDict(value,owner,path)
    if isinstance(value,list):return TrackedList(value,owner,path)
    return value


class TrackedDict(dict):
    def __init__(self,value,owner,path):
        self._owner=owner;self._path=path
        for k,v in value.items():dict.__setitem__(self,k,wrap(v,owner,None if path is None else path+(k,)))
    def _bind(self,path):
        self._path=path
        for k,v in self.items():
            if isinstance(v,(TrackedDict,TrackedList)):v._bind(None if path is None else path+(k,))
    def new_row(self,value):return TrackedDict(value,self._owner,None)
    def __setitem__(self,key,value):
        path=None if self._path is None else self._path+(key,)
        converted=wrap(value,self._owner,path);dict.__setitem__(self,key,converted)
        self._owner.emit(path,'set',converted)
    def update(self,*args,**kwargs):
        for k,v in dict(*args,**kwargs).items():self[k]=v
    def __deepcopy__(self,memo):return copy.deepcopy(plain(self),memo)
    def setdefault(self,key,default=None):
        if key not in self:self[key]=default
        return self[key]
    def __delitem__(self,*args):raise legacy.LogError('Deletion is not a mapped v1 report mutation')
    def pop(self,*args):raise legacy.LogError('Pop is not a mapped v1 report mutation')
    def popitem(self,*args):raise legacy.LogError('Popitem is not a mapped v1 report mutation')
    def clear(self):raise legacy.LogError('Clear is not a mapped v1 report mutation')
    def __ior__(self,other):raise legacy.LogError('In-place union is not a mapped v1 mutation')


class TrackedList(list):
    def __init__(self,value,owner,path):
        self._owner=owner;self._path=path
        for i,v in enumerate(value):list.append(self,wrap(v,owner,None if path is None else path+(i,)))
    def _bind(self,path):
        self._path=path
        for i,v in enumerate(self):
            if isinstance(v,(TrackedDict,TrackedList)):v._bind(None if path is None else path+(i,))
    def append(self,value):
        child=None if self._path is None else self._path+(len(self),)
        value=wrap(value,self._owner,child);list.append(self,value)
        self._owner.emit(self._path,'append',value)
    def __deepcopy__(self,memo):return copy.deepcopy(plain(self),memo)
    def __setitem__(self,key,value):
        legacy.require(type(key) is int and 0<=key<len(self),'Only existing typed list index set is supported')
        path=None if self._path is None else self._path+(key,);value=wrap(value,self._owner,path)
        list.__setitem__(self,key,value);self._owner.emit(path,'set',value)
    def extend(self,values):
        for value in values:self.append(value)
    def __delitem__(self,*args):raise legacy.LogError('List deletion is not mapped')
    def pop(self,*args):raise legacy.LogError('List pop is not mapped')
    def clear(self):raise legacy.LogError('List clear is not mapped')
    def insert(self,*args):raise legacy.LogError('List insertion is not mapped')
    def sort(self,*args,**kwargs):raise legacy.LogError('List sorting is not mapped')
    def reverse(self):raise legacy.LogError('List reversal is not mapped')
    def __iadd__(self,*args):raise legacy.LogError('List concatenation is not mapped')
    def __imul__(self,*args):raise legacy.LogError('List repetition is not mapped')
