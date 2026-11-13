"""Passive E2a graph observations and bit-exact audit evidence; no scientific imports."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from infrastructure.paths import contained_path
from infrastructure.signatures import require, read_report

GROUPS=('feats_network','mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor')
HEADS=GROUPS[1:]


def array_record(a):
    a=np.asarray(a);require(a.dtype.kind in 'fiu' and np.isfinite(a).all(),'Finite numeric array required')
    d=a.astype(np.float64);return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
        sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),finite=True,zero=int(np.count_nonzero(a==0)),
        nonzero=int(np.count_nonzero(a)),L2=float(np.linalg.norm(d)),max_absolute=float(abs(d).max()) if a.size else 0.)


def tensor_array(t):
    import torch
    require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32,
            'Actual CPU float32 tensor required')
    return t.detach().numpy().copy()


class GraphObserver:
    """Hooks return None; retain actual nodes, never substitute outputs/gradients."""
    def __init__(self,model,enabled=True,expected_relus=None):
        self.model=model;self.enabled=enabled;self.expected_relus=expected_relus
        self.handles=[];self.nodes={};self.relu_names=[];self.capture=False
    def _keep(self,name,t):
        import torch
        if not self.enabled or not self.capture:return
        require(name not in self.nodes,'Repeated observed node: '+name)
        require(isinstance(t,torch.Tensor) and t.requires_grad and t.device.type=='cpu' and t.dtype==torch.float32,
                'Missing/invalid connected observed tensor: '+name)
        t.retain_grad();self.nodes[name]=t
    def __enter__(self):
        import torch
        if not self.enabled:return self
        try:
            for head in HEADS:
                last=getattr(self.model,head).predictor[-1]
                require(isinstance(last,torch.nn.Sigmoid),'Original sigmoid head required')
                self.handles.append(last.register_forward_pre_hook(lambda m,args,n=head:self._keep(n+'.logits',args[0])))
                self.handles.append(last.register_forward_hook(lambda m,args,out,n=head:self._keep(n+'.sigmoid',out)))
            self.handles.append(self.model.feats_network.register_forward_hook(lambda m,args,out:self._keep('shared_features',out)))
            for name,module in self.model.named_modules():
                if isinstance(module,torch.nn.ReLU):
                    self.relu_names.append(name)
                    self.handles.append(module.register_forward_pre_hook(lambda m,args,n=name:self._keep('relu.'+n+'.pre',args[0])))
                    self.handles.append(module.register_forward_hook(lambda m,args,out,n=name:self._keep('relu.'+n+'.post',out)))
            require(self.expected_relus is None or len(self.relu_names)==self.expected_relus,'ReLU inventory changed')
            return self
        except BaseException:
            self.close();raise
    def begin(self):
        require(not self.nodes,'Previous observation not released');self.capture=True
    def wrap_shader(self,shader):
        observer=self
        class View:
            def shade(self,*args,**kwargs):
                if observer.enabled and observer.capture:
                    require(len(args)>=8,'Actual shader positional inputs required')
                    for name,t in zip(('mx','my','albedo','metallic','encoding'),args[3:8]):
                        observer._keep('shader.'+name,t)
                return shader.shade(*args,**kwargs)
        return View()
    def collect(self):
        if not self.enabled:return {},dict(enabled=False)
        expected={h+'.'+k for h in HEADS for k in ('logits','sigmoid')}|{'shared_features'}|{
            'shader.'+n for n in ('mx','my','albedo','metallic','encoding')}|{
            'relu.'+n+'.'+k for n in self.relu_names for k in ('pre','post')}
        require(set(self.nodes)==expected,'Missing observed graph nodes')
        arrays={};summary={}
        for name,t in self.nodes.items():
            require(t.grad is not None,'Missing retained loss gradient: '+name)
            a,g=tensor_array(t),tensor_array(t.grad)
            require(np.isfinite(a).all() and np.isfinite(g).all(),'Nonfinite observed value/gradient: '+name)
            arrays[name]=a;arrays[name+'.loss_gradient']=g
            summary[name]=dict(value=array_record(a),loss_gradient=array_record(g),phase='pre_update')
            if a.size<=8:summary[name]['values']=a.tolist()
            if name.endswith('.sigmoid'):
                require(((a>=0)&(a<=1)).all(),'Sigmoid output outside bounds')
                arrays[name+'.boundary_distance']=np.minimum(a,1-a)
                arrays[name+'.analytic_derivative']=a.astype(np.float64)*(1-a.astype(np.float64))
                summary[name]['analytic_derivative']=array_record(arrays[name+'.analytic_derivative'])
            if name.startswith('relu.') and name.endswith('.post'):
                summary[name]['activity']=dict(positive=int(np.count_nonzero(a>0)),zero=int(np.count_nonzero(a==0)))
        return arrays,dict(enabled=True,nodes=summary,
            orientation_gradient='actual shader.encoding and rotation sigmoid graph nodes; diagnostic theta not required',
            centered_radius='value diagnostic only; not required to have a loss gradient')
    def release(self):self.nodes.clear();self.capture=False
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.release()
    def __exit__(self,*args):self.close()


def parameter_arrays(model):return {n:tensor_array(p) for n,p in model.named_parameters()}


def parameter_identity(arrays):
    value={n:array_record(a) for n,a in sorted(arrays.items())}
    # State identity excludes derived norm reductions.
    catalog={n:{k:v[k] for k in ('shape','dtype','sha256')} for n,v in value.items()}
    return hashlib.sha256(json.dumps(catalog,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def displacement(before,after):
    require(set(before)==set(after),'Parameter inventory changed')
    tensors={};xor={};groups={}
    for n,a in before.items():
        b=after[n];require(a.dtype==b.dtype==np.float32 and a.shape==b.shape,'Float32 parameter layout changed')
        delta=b.astype(np.float64)-a.astype(np.float64);rec=array_record(delta)
        bitdiff=np.ascontiguousarray(a).view(np.uint32)^np.ascontiguousarray(b).view(np.uint32)
        rec.update(changed_elements=int(np.count_nonzero(a!=b)),bitwise_changed_elements=int(np.count_nonzero(bitdiff)),
            before_sha256=array_record(a)['sha256'],after_sha256=array_record(b)['sha256'])
        tensors[n]=rec;xor[n]=bitdiff
        group=groups.setdefault(n.split('.')[0],dict(changed_elements=0,bitwise_changed_elements=0,squared_L2=0.,max_absolute=0.))
        for k in ('changed_elements','bitwise_changed_elements'):group[k]+=rec[k]
        group['squared_L2']+=rec['L2']**2;group['max_absolute']=max(group['max_absolute'],rec['max_absolute'])
    for g in groups.values():g['L2']=float(np.sqrt(g.pop('squared_L2')))
    return xor,dict(per_tensor=tensors,per_group=groups,before_identity=parameter_identity(before),
        after_identity=parameter_identity(after),method='actual pre/post float32 values; no moment-predicted displacement')


def reconstruct(before,xor):
    require(set(before)==set(xor),'XOR inventory mismatch')
    result={}
    for n,a in before.items():
        x=xor[n];require(a.dtype==np.float32 and x.dtype==np.uint32 and x.shape==a.shape,'Invalid XOR dtype/shape')
        result[n]=(np.ascontiguousarray(a).view(np.uint32)^x).view(np.float32)
        require(np.isfinite(result[n]).all(),'Nonfinite reconstructed parameter')
    return result


def gradient_records(model):
    result={}
    for n,p in model.named_parameters():
        require(p.grad is not None,'Missing parameter gradient: '+n)
        g=tensor_array(p.grad);require(np.isfinite(g).all(),'Nonfinite parameter gradient: '+n)
        result[n]=array_record(g)
    require({n.split('.')[0] for n in result}==set(GROUPS),'All six original groups required')
    return result


def adam_records(model,opt,expected_step):
    result={}
    for n,p in model.named_parameters():
        state=opt.state.get(p,{})
        require(set(state)=={'step','exp_avg','exp_avg_sq'},'Incomplete Adam state')
        require(float(state['step'])==expected_step,'Adam counter mismatch')
        result[n]={k:array_record(tensor_array(v)) for k,v in state.items()}
    return result


class Store:
    """Exclusive numeric/JSON records; bounds checked before and during writes."""
    def __init__(self,root,approved_parent,json_cap=2*1024**2,total_cap=None):
        self.root=Path(root);parent=Path(approved_parent);self.json_cap=json_cap
        self.total_cap=total_cap;self.written=0;self.json_counts={}
        require(self.root.is_absolute() and self.root.resolve()==self.root and self.root.parent==parent and
            parent.resolve()==parent and parent.stat().st_uid==os.geteuid() and not self.root.exists(), 'Fresh owned output required')
        require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (self.root,*self.root.parents)), 'No output inside Git')
        self.root.mkdir(exist_ok=False)
    def account(self,size):
        require(self.total_cap is None or self.written+size<=self.total_cap,'Total capture cap exceeded')
        self.written+=size
    def path(self,name):return contained_path(self.root,name,writable=True)
    def json(self,name,value):
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        require(len(data)<=self.json_cap,'JSON capture cap exceeded');p=self.path(name)
        group=Path(name).parts[0] if len(Path(name).parts)>1 else '.'
        require(self.json_counts.get(group,0)<(2 if group=='.' else 64),'JSON record inventory cap exceeded')
        self.account(len(data));self.json_counts[group]=self.json_counts.get(group,0)+1
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        return dict(path=str(p.relative_to(self.root)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    def numeric(self,name,arrays,cap,allow_nonfinite=False):
        entries=[];packed={}
        for n,a in sorted(arrays.items()):
            a=np.asarray(a);require(a.dtype.kind in 'fiu','Object/non-numeric capture forbidden')
            require(allow_nonfinite or np.isfinite(a).all(),'Nonfinite numeric capture')
            key='a'+str(len(entries)).zfill(4);packed[key]=a
            entries.append(dict(key=key,name=n,shape=list(a.shape),dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),finite=bool(np.isfinite(a).all())))
        require(sum(a.nbytes for a in packed.values())+1024*len(entries)<=cap,'Numeric catalog exceeds frozen cap')
        p=self.path(name);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:
            class Bounded:
                def write(inner,b):
                    require(f.tell()+len(b)<=cap,'Numeric archive cap exceeded');self.account(len(b));return f.write(b)
                def __getattr__(self,k):return getattr(f,k)
            np.savez(Bounded(),**packed);f.flush();os.fsync(f.fileno())
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest=dict(schema_version=1,archive=str(p.relative_to(self.root)),archive_sha256=sha,bytes=p.stat().st_size,
            entries=entries,allow_pickle=False,purpose='E2a numeric audit; no implicit restore/resume')
        return self.json(name+'.json',manifest)


def read_numeric(root,selection,allow_nonfinite=False):
    root=Path(root);p=contained_path(root,selection['path'],writable=False)
    manifest,_=read_report(p,selection['sha256']);archive=contained_path(root,manifest['archive'],writable=False)
    require(archive.stat().st_size<=18*1024**2,'Numeric archive exceeds maximum role cap')
    require(hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256'],'Numeric archive hash mismatch')
    values={};entries=manifest['entries']
    with zipfile.ZipFile(archive) as z:
        items=z.infolist()
        require(len(items)<=350 and sum(i.file_size for i in items)<=18*1024**2 and
            all(i.compress_type==zipfile.ZIP_STORED for i in items),'Unbounded/compressed numeric evidence')
    require(len({e['key'] for e in entries})==len(entries) and len({e['name'] for e in entries})==len(entries),'Duplicate numeric catalog')
    with np.load(archive,allow_pickle=False) as arrays:
        require(set(arrays.files)=={e['key'] for e in entries},'Numeric catalog differs')
        for e in entries:
            a=arrays[e['key']]
            require(a.dtype.kind in 'fiu' and a.dtype==np.dtype(e['dtype']) and list(a.shape)==e['shape'] and
                hashlib.sha256(a.tobytes(order='C')).hexdigest()==e['sha256'] and
                bool(np.isfinite(a).all())==e['finite'] and (e['finite'] or allow_nonfinite),'Malformed/nonfinite numeric evidence')
            values[e['name']]=a
    return values
