"""ONE bounded 3D-2 Phase C validation; never a recovery/training command."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import neural_material_field_v1 as field
from experiments import neural_material_recovery_v1 as h
from experiments.material_shader_v1 import decode_atan2, encoding_status
from infrastructure.signatures import require, SignatureError

ROOT=h.ROOT
BASE='a11efd615d18ac69fa1c26a3692af511f1c78302'
PROTOCOL='neural-material-recovery-v1-phase-c'
CODE=('experiments/validate_neural_material_recovery_v1.py','tests/scientific/test_neural_material_validation_v1.py',
      'docs/protocols/NEURAL_MATERIAL_FIELD_VALIDATION_V1.md')
IMPLEMENTATION={
 'experiments/neural_material_field_v1.py':'9b1f57efb72c2ac1a5e6ea5dc79ba1f546205fea8bc7916b7a1307b60b38f38d',
 'experiments/neural_material_recovery_v1.py':'972384bbc5c384267fa1a1e831594381bdd6766678e33d5011d89c53adf75b7c',
 'tests/scientific/test_neural_material_field_v1.py':'c1192a5d8694bee65b97a7b2ff8f6401da47c5da80f4b0188093dc3daddcae3c',
 'tests/scientific/test_neural_material_recovery_v1.py':'664bd26281a262278ba6c334de7930087df7ced9c6362858dec61cdb6138c268',
}
CAPS=dict(shader_forward=45,shader_backward=36,field_vjp=24,synthetic_Adam=3,field_query=63,
          construct_D0=25,construct_N0=13,construct_N1=16,N1_calibration=16,
          original_components=3,original_component_forward=3)
RETURN_KEYS=('tangent','bitangent','normal','half_vectors','specular_directions','diffuse_directions','pdf',
             'valid_specular_counts','frame_orthonormal')


def plan():
    return dict(schema_version=1,base_commit=BASE,materials=list(h.previous.reference.PANEL),seeds=list(field.SEEDS),
        arms=list(h.ARMS),initial_fixtures=27,replay_fixtures=9,gauge_fixtures=9,
        caps=CAPS.copy(),wall_seconds=300,RSS_bytes=4*1024**3,address_bytes=8*1024**3,
        threads=1,GPU_hidden=True,fit_seeds=[3000,3001,3002,3003],samples=[512,1024],
        recovery_updates=0,reference_calls=0,LM_calls=0)


def validate_plan(value):
    require(value==plan(),'Fixture plan differs from the predeclared validation matrix')
    require(CAPS['shader_forward']==27+9+9<=64 and CAPS['shader_backward']==27+9<=64 and
        CAPS['field_vjp']==3*7+3<=200,'Validation exceeds authorized ceilings')
    return value


class Stopped(RuntimeError):pass


class Budget:
    def __init__(self):
        self.start=time.perf_counter();self.counts={k:dict(attempted=0,completed=0) for k in CAPS};self.labels=set()
    def check(self):
        if time.perf_counter()-self.start>=300:raise Stopped('300-second validation ceiling; no retry')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'4GiB RSS ceiling exceeded')
    @contextmanager
    def operation(self,kind,label):
        self.check();key=(kind,label);require(key not in self.labels,'Operation retry forbidden: '+str(key))
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'Planned work exhausted: '+kind)
        self.labels.add(key);c['attempted']+=1
        yield
        c['completed']+=1;self.check()
    def finished(self):
        return all(c['attempted']==c['completed']==CAPS[k] for k,c in self.counts.items())


@contextmanager
def runtime(budget,report):
    require(all(os.environ.get(k)==v for k,v in h.previous.original.REQUIRED_ENV.items()),'Explicit CPU/thread environment required')
    threads=torch.get_num_threads();rng=torch.get_rng_state().clone();nrng=np.random.get_state()
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)};limits=resource.getrlimit(resource.RLIMIT_AS)
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'Do not replace another timer')
    def stop(*args):raise Stopped('Bounded validation timer/termination; no retry')
    try:
        torch.set_num_threads(1)
        cap=min(8*1024**3,limits[0]) if limits[0]>=0 else 8*1024**3
        if limits[1]>=0:cap=min(cap,limits[1])
        resource.setrlimit(resource.RLIMIT_AS,(cap,limits[1]))
        for s in handlers:signal.signal(s,stop)
        budget.check();signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.perf_counter()-budget.start)))
        yield
    finally:
        after=np.random.get_state()
        report.update(global_torch_CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),
            global_numpy_RNG_unchanged=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:])
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in handlers.items():signal.signal(s,handler)
        resource.setrlimit(resource.RLIMIT_AS,limits);torch.set_num_threads(threads)
        torch.set_rng_state(rng);np.random.set_state(nrng)
        report.update(threads_restored=torch.get_num_threads()==threads,memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS)==limits,
            validation_wall_seconds=time.perf_counter()-budget.start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(all(report[k] for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')),
        'Validation runtime/RNG integrity failure')


def admit():
    reference,ids=h.admit_reference()
    for name,sha in IMPLEMENTATION.items():
        original=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        require(hashlib.sha256(original).hexdigest()==sha==h.digest(ROOT/name),'Phase B implementation changed: '+name)
        ids[name]=sha
    return reference,ids


def comparison(candidate,base,atol=0.,rtol=0.,vector=False,mandatory=True):
    a,b=np.asarray(candidate),np.asarray(base)
    require(a.shape==b.shape and a.dtype==b.dtype and a.size>0,'Matching nonempty tensor layout required')
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:return dict(passed=False,finite=False,mandatory=mandatory,reason='Nonfinite comparison input')
    delta=abs(a.astype(np.float64)-b.astype(np.float64));rel=delta/np.maximum(abs(b.astype(np.float64)),np.finfo(np.float32).tiny)
    wi=tuple(int(i) for i in np.unravel_index(np.argmax(delta),a.shape))
    ri=tuple(int(i) for i in np.unravel_index(np.argmax(rel),a.shape))
    out=dict(passed=bool(np.all(delta<=atol+rtol*abs(b.astype(float)))),finite=True,mandatory=mandatory,
        exact=bool(np.array_equal(a,b)),atol=atol,rtol=rtol,maximum_absolute=float(delta.max()),maximum_relative=float(rel.max()),
        relative_denominator_floor=float(np.finfo(np.float32).tiny),worst_absolute_index=list(wi),worst_relative_index=list(ri),
        candidate_at_worst=float(a[wi]),base_at_worst=float(b[wi]),exceeding_elements=int(np.sum(delta>atol+rtol*abs(b.astype(float)))))
    if vector:
        aa,bb=a.astype(float),b.astype(float)
        angle=np.arctan2(np.linalg.norm(np.cross(aa,bb),axis=-1),np.sum(aa*bb,axis=-1))
        out.update(maximum_angle_radians=float(angle.max()),angle_limit=2e-5)
        out['passed'] &= bool(angle.max()<=2e-5)
    return out


def array_catalog(arrays):
    return {k:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest()) for k,a in arrays.items()}


def save_arrays(output,label,arrays):
    require(all(a.dtype.kind!='O' for a in arrays.values()),'No object arrays/pickle allowed')
    path=output.path('raw/'+label+'.npz')
    with path.open('xb') as stream:np.savez(stream,**arrays)
    return dict(path=str(path.relative_to(output.root)),sha256=h.digest(path),bytes=path.stat().st_size,arrays=array_catalog(arrays))


def read_arrays(output,manifest):
    path=output.path(manifest['path']);require(h.digest(path)==manifest['sha256'],'Raw validation archive hash mismatch')
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    require(array_catalog(arrays)==manifest['arrays'],'Raw validation catalogue mismatch')
    return arrays


def make_model(arm,seed,budget,label,timings):
    start=time.perf_counter()
    with budget.operation('construct_'+arm,label):
        if arm=='N1':
            with budget.operation('N1_calibration',label):model=field.make_material(arm,h.fresh_latent(seed),seed)
        else:model=field.make_material(arm,h.fresh_latent(seed),seed)
    timings.append(dict(kind='construction',arm=arm,label=label,seconds=time.perf_counter()-start))
    return model


def query(model,points,budget,label):
    with budget.operation('field_query',label):material=model(points).validate()
    return material


def points(n):
    p=torch.tensor([[x,y,0.] for x in (-1.,0.,1.) for y in (-1.,0.,1.)])
    return torch.zeros((1,3)) if n==1 else p.repeat((n+8)//9,1)[:n]


def source_field_audit(seed,budget,timings):
    n=make_model('N1',seed,budget,'audit-'+str(seed),timings)
    # Independently select original source AST, rather than calling the extractor twice.
    root,_=h.baseline('material',h.PINS['material']);tree=ast.parse((root/'network/field.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in field.DEFINITIONS]
    ns=dict(torch=torch,nn=nn,F=F,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'independent-original-A','exec'),ns)
    independent_AST={x.name:hashlib.sha256(ast.dump(x).encode()).hexdigest() for x in nodes}
    require(all(n.source_identity['definition_AST_sha256'][name]==sha for name,sha in independent_AST.items()),
        'Extracted definitions differ from independently parsed source AST')
    with budget.operation('original_components',str(seed)),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(10000+seed)
        original=nn.Module();original.feats_network=ns['MaterialFeatsNetwork']()
        for name,width in field.HEADS:setattr(original,name,ns['make_predictor'](259,width))
    actual={name.replace('.predictor.','.'):p for name,p in n.named_parameters()}
    require(set(actual)==set(dict(original.named_parameters())) and all(torch.equal(actual[k],v) for k,v in original.named_parameters()),
        'Original component construction/parameter order mismatch')
    require(sum(p.numel() for p in n.parameters())==1485584 and len(actual)==84,'Original architecture/count mismatch')
    require(all(p.requires_grad for p in n.parameters()) and n.feats_network.module0[0].weight_v.shape==(256,51) and
        n.feats_network.module1[0].weight_v.shape==(256,307),'Original trunk/skip/trainability mismatch')
    require(len([m for m in n.modules() if isinstance(m,nn.Linear)])==28 and
        all(hasattr(m,'weight_g') and hasattr(m,'weight_v') for m in n.modules() if isinstance(m,nn.Linear)),
        'Original weight normalization/topology mismatch')
    enc=original.feats_network.pos_enc(points(1));expected=[points(1)]
    for f in 2.**torch.linspace(0.,7.,8):expected.extend((torch.sin(points(1)*f),torch.cos(points(1)*f)))
    require(enc.shape==(1,51) and torch.equal(enc,torch.cat(expected,-1)),'Original positional encoding mismatch')
    with budget.operation('original_component_forward',str(seed)):
        feats=original.feats_network(points(1));features=torch.cat((feats,points(1)),-1)
        require(torch.equal(feats,n.feats_network(n.anchor)),'Original trunk output mismatch')
        for name,_ in field.HEADS:
            head=getattr(n,name);v=features
            for layer in list(getattr(original,name).children())[:-1]:v=layer(v)
            require(torch.equal(v,head.center) and not head.center.requires_grad and not head.target.requires_grad,
                'Fixed original-logit calibration buffer mismatch')
            require(isinstance(head.predictor[-1],nn.Sigmoid) and head.predictor[0].weight_v.shape==(256,259),
                'Original head activation/input dimensions changed')
    z=h.fresh_latent(seed);e=torch.stack((torch.cos(2*z[2]),torch.sin(2*z[2])));q=(1+.5*e)/2
    targets=(z[:1],z[1:2],z[6:7],torch.log(q/(1-q)),z[3:6])
    require(all(torch.equal(getattr(n,name).target,target.reshape(1,-1)) for (name,_),target in zip(field.HEADS,targets)),
        'Declared fixed scalar/rho=.5 calibration arithmetic changed')
    anchor_inputs=[];head_inputs={name:[] for name,_ in field.HEADS}
    hooks=[n.feats_network.register_forward_pre_hook(lambda module,args:anchor_inputs.append(args[0].detach().clone()))]
    for name,_ in field.HEADS:
        def watch(module,args,name=name):head_inputs[name].append(args[0][:,-3:].detach().clone())
        hooks.append(getattr(n,name).register_forward_pre_hook(watch))
    try:outputs=[query(n,points(count),budget,'audit-'+str(seed)+'-'+str(i)) for i,count in enumerate((1,9,96,9))]
    finally:
        for hook in hooks:hook.remove()
    require(len(anchor_inputs)==4 and all(torch.equal(x,n.anchor) for x in anchor_inputs) and
        all(len(xs)==4 and all(torch.equal(x,n.anchor) for x in xs) for xs in head_inputs.values()),
        'Both trunk and every raw-coordinate head input must use the constant anchor')
    a=outputs[0]
    require(all(torch.equal(m.physical,a.physical.expand_as(m.physical)) and torch.equal(m.encoding,a.encoding.expand_as(m.encoding))
        for m in outputs),'Constant position/batch/replay gate failed')
    # Observe BOTH anchor paths; no input replacement or global monkeypatch.
    require(torch.equal(n.anchor,points(1)) and not n.anchor.requires_grad,'Invalid fixed anchor')
    biases=[getattr(n,name).predictor[-2].bias for name,_ in field.HEADS];rows=[]
    for i,value in enumerate(a.physical[0]):
        with budget.operation('field_vjp','reach-'+str(seed)+'-'+str(i)):
            g=torch.autograd.grad(value,biases,retain_graph=True)
        rows.append(torch.cat(g).detach().double().numpy())
    jacobian=np.stack(rows);rank=int(np.linalg.matrix_rank(jacobian))
    require(np.isfinite(jacobian).all() and rank==7,'Seven physical directions not locally reachable')
    d=make_model('D0',seed,budget,'audit-direct-'+str(seed),timings)
    direct=query(d,points(1),budget,'audit-direct-'+str(seed))
    require(torch.equal(a.physical[:,[0,1,3,4,5,6]],direct.physical[:,[0,1,3,4,5,6]]),'Initial six-scalar matching failed')
    return n,dict(seed=seed,passed=True,model=field.identity(n),source_definition_AST_sha256=n.source_identity['definition_AST_sha256'],
        independent_AST_sha256=independent_AST,both_spatial_input_paths_observed_constant=True,
        original_parameter_count=sum(p.numel() for p in original.parameters()),physical_bias_jacobian=jacobian.tolist(),local_rank=rank,
        constant_batches=[1,9,96,9],initial=h.material_diagnostics(a),
        scalar_theta_difference=float(a.physical[0,2]-direct.physical[0,2]))


class FieldObserver:
    def __init__(self,model,budget,label):self.model=model;self.budget=budget;self.label=label;self.arm=model.arm;self.material=None
    def __call__(self,p):
        self.material=query(self.model,p,self.budget,self.label)
        return self.material


class ShaderObserver:
    def __init__(self,shader,budget,label):self.shader=shader;self.budget=budget;self.label=label;self.result=None;self.inputs=None
    def shade(self,*args,**kwargs):
        self.inputs=args
        if args[7].requires_grad:args[7].retain_grad()
        with self.budget.operation('shader_forward',self.label):self.result=self.shader.shade(*args,**kwargs)
        return self.result


def capture(model,batch,shader,target,budget,label,timings,backward=True,failure_sink=None):
    observed=FieldObserver(model,budget,label);watched=ShaderObserver(shader,budget,label)
    try:return _capture(model,batch,target,budget,label,timings,backward,observed,watched)
    except BaseException:
        if failure_sink is not None:
            partial={k:t.detach().numpy().copy() for k,t in (watched.result or {}).items() if isinstance(t,torch.Tensor)}
            for name,p in getattr(model,'named_parameters',lambda:())():
                partial['parameter::'+name]=p.detach().numpy().copy()
                if p.grad is not None:partial['gradient::'+name]=p.grad.detach().numpy().copy()
            if partial:failure_sink(label,partial)
        raise


def _capture(model,batch,target,budget,label,timings,backward,observed,watched):
    for p in getattr(model,'parameters',lambda:())():p.grad=None
    start=time.perf_counter()
    values,material,_=batch.render(observed,watched,h.counts(model.arm))
    loss=h.loss_value(values,target);forward=time.perf_counter()-start
    gradient=None;elapsed=None
    if backward:
        start=time.perf_counter()
        with budget.operation('shader_backward',label):gradient=h.backward(model,loss,h.counts(model.arm))
        elapsed=time.perf_counter()-start
        require(all(g['finite'] and g['missing']==0 and g['nonzero']>0 for g in gradient.values()),'Disconnected or invalid whole head/trunk')
        require(watched.inputs[7].grad is not None and bool(torch.isfinite(watched.inputs[7].grad).all()) and
            bool(torch.count_nonzero(watched.inputs[7].grad)),'Intended shader encoding lost its gradient')
    r=watched.result
    arrays={k:r[k].detach().numpy().copy() for k in RETURN_KEYS}
    arrays.update(physical=material.physical.detach().numpy().copy(),encoding=watched.inputs[7].detach().numpy().copy(),
        material_encoding=material.encoding.detach().numpy().copy(),mean=values.detach().numpy().copy(),loss=loss.detach().numpy().copy(),
        per_tape=np.stack([r[k].detach().numpy().reshape(4,24,3).copy() for k in ('diffuse_rgb','specular_rgb','linear_rgb')],2),
        mask=((r['specular_directions']*watched.inputs[1].unsqueeze(1)).sum(-1)>0).detach().numpy(),
        views=watched.inputs[2].detach().numpy().copy())
    if material.rotation_raw is not None:
        arrays['rotation_raw']=material.rotation_raw.detach().numpy().copy();arrays['centered_radius']=material.centered_radius.detach().numpy().copy()
    if backward:
        arrays['encoding_gradient']=watched.inputs[7].grad.detach().numpy().copy()
        for name,p in model.named_parameters():arrays['gradient::'+name]=p.grad.detach().numpy().copy()
    require(all(np.isfinite(a).all() for a in arrays.values()),'Nonfinite actual captured output/gradient')
    replay=np.stack([torch.from_numpy(arrays['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
    require(np.array_equal(replay,arrays['mean']) and h.previous.roundoff.mse32(arrays['mean'][:,2],target.numpy())[0]==float(loss),
        'Original four-tape mean/MSE replay failed')
    require(bool(r['frame_orthonormal'].all()),'Invalid actual tangent frames')
    timings.append(dict(kind='real_forward_backward',arm=model.arm,label=label,forward_seconds=forward,backward_seconds=elapsed))
    return arrays,dict(loss=float(loss.detach()),gradient_groups=gradient,original_arithmetic_replay=True,
        forward_seconds=forward,backward_seconds=elapsed,material=h.material_diagnostics(material))


def parity(candidate,base,target,exact=False,gauge=False):
    checks={}
    for key in ('normal','tangent','bitangent'):
        if gauge and key!='normal':continue
        checks[key]=comparison(candidate[key],base[key],0. if exact else 2e-6)
    for key in ('half_vectors','specular_directions'):
        checks[key]=comparison(candidate[key],base[key],0. if exact else (2e-6 if key=='half_vectors' else 4e-6),
            0. if exact else 2e-5,True)
    for key in ('mask','valid_specular_counts','diffuse_directions','frame_orthonormal'):checks[key]=comparison(candidate[key],base[key])
    checks['pdf']=comparison(candidate['pdf'],base['pdf'],mandatory=exact)
    for key in ('per_tape','mean'):
        for i,name in enumerate(('diffuse','specular','total')):
            checks[key+'_'+name]=comparison(candidate[key][...,i,:],base[key][...,i,:],
                0. if exact or i==0 else 2e-6,0. if exact or i==0 else 2e-5)
    q0,q1=(h.previous.coupling.accepted.tensor_q(a['physical'][0]) for a in (base,candidate))
    checks['Q']=comparison(q1,q0,2e-7,2e-6)
    checks['six_scalars']=comparison(candidate['physical'][:,[0,1,3,4,5,6]],base['physical'][:,[0,1,3,4,5,6]]) if not gauge else comparison(candidate['physical'][:,3:],base['physical'][:,3:])
    checks['encoding']=comparison(candidate['encoding'],base['encoding'],mandatory=exact)
    if exact:
        for key in base:
            if key.startswith('gradient::') or key in ('encoding_gradient','loss'):checks[key]=comparison(candidate[key],base[key])
    arithmetic=h.previous.accepted.objective_envelope(base['mean'][:,2],candidate['mean'][:,2],target.numpy(),float(base['loss']),float(candidate['loss']))
    return dict(passed=all(c['passed'] for c in checks.values() if c['mandatory']) and arithmetic['passed'],
        checks=checks,arithmetic=arithmetic,old_scalar_warning_passed=abs(float(candidate['loss'])-float(base['loss']))<=1e-12+5e-7*abs(float(base['loss'])))


class PhysicalFixture:
    arm='D0'
    def __init__(self,physical):self.physical=physical.detach().clone()
    def __call__(self,p):
        v=self.physical;e=torch.stack((torch.cos(2*v[2]),torch.sin(2*v[2]))).reshape(1,2)
        return field.Material(v.reshape(1,7),e).validate().expand(len(p))


def zero_encoding_checks():
    rows=[]
    for label,e in (('zero',torch.zeros((1,2))),('near_zero',torch.tensor([[1e-8,0.]]))):
        try:decode_atan2(e)
        except SignatureError:rows.append(dict(case=label,rejected=True))
        else:raise SignatureError('Existing invalid encoding rejection failed: '+label)
    return rows


def benchmark(output,budget,timings):
    rows=[]
    for arm in h.ARMS:
        label='synthetic-'+arm;model=make_model(arm,101,budget,label,timings);opt=h.optimizer_for(model)
        material=query(model,points(1),budget,label);loss=material.physical.square().sum()
        start=time.perf_counter()
        with budget.operation('field_vjp',label):loss.backward()
        field_backward=time.perf_counter()-start
        start=time.perf_counter()
        with budget.operation('synthetic_Adam',label):h.step(model,opt,h.counts(arm))
        adam=time.perf_counter()-start
        before=h.compact_state(model,opt);start=time.perf_counter()
        selected=h.save_checkpoint(output,'synthetic-'+arm.lower(),model,opt,1)
        serialized=time.perf_counter()-start
        manifest,_=h.read_report(output.path(selected['manifest']),selected['manifest_sha256'])
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        opt.state.clear();start=time.perf_counter();h.restore_checkpoint(output,selected,model,opt)
        restored=time.perf_counter()-start
        require(before==h.compact_state(model,opt),'Own numeric checkpoint strict restoration mismatch')
        rows.append(dict(arm=arm,synthetic_updates=1,synthetic_loss='sum of squared physical outputs; no targets/shader',
            field_backward_seconds=field_backward,Adam_seconds=adam,serialization_seconds=serialized,hash_and_restore_seconds=restored,
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            optimizer_state_bytes=sum(t.numel()*t.element_size() for s in opt.state.values() for t in s.values()),
            archive_bytes=manifest['archive_bytes'],manifest_bytes=output.path(selected['manifest']).stat().st_size,
            checkpoint=selected,exact_state_restored=True))
    return rows


def feasibility(timings,storage):
    rows=[]
    for arm in h.ARMS:
        fit=[r for r in timings if r['kind']=='real_forward_backward' and r['arm']==arm and r['backward_seconds'] is not None]
        measured=next(r for r in storage if r['arm']==arm)
        fw=[r['forward_seconds'] for r in fit];bw=[r['backward_seconds'] for r in fit]
        step=max(fw)+max(bw)+measured['Adam_seconds']
        estimate=5000*step+3*measured['serialization_seconds']+max(fw)+max(bw)
        rows.append(dict(arm=arm,observations=len(fit),forward_median_seconds=float(np.median(fw)),forward_max_seconds=max(fw),
            backward_median_seconds=float(np.median(bw)),backward_max_seconds=max(bw),
            maximum_observed_step_estimate_seconds=step,candidate_5000_active_seconds_estimate=estimate,
            provisional_active_cap_seconds=900,supported=estimate<=900,
            uncertainty='Not a full-run guarantee; logging, prefix/trajectory gates, allocation and OS variability need margin'))
    return dict(passed=all(r['supported'] for r in rows),arms=rows,
        minimum_27_history_checkpoint_bytes=9*3*sum(r['archive_bytes'] for r in storage),budget_changed=False)


def replay_row(material_name,seed,arm,raw,checks,info):
    """Keep material identity separate from captured material diagnostics."""
    metadata=dict(material_name=material_name,seed=seed,arm=arm,raw=raw,
        passed=all(c['passed'] for c in checks.values()),checks=checks)
    require(not metadata.keys() & info.keys(),'Replay metadata/diagnostic key collision')
    return dict(metadata,**info)


def execute(output,budget,report,persist):
    reference,ids=admit();fit_tape,eval_tape,tapes=h.inputs(reference)
    require(reference['runtime']==dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1),
        'Accepted CPU runtime required')
    report.update(accepted_identities=ids,implementation_sha256={k:v for k,v in IMPLEMENTATION.items() if k.startswith('experiments/')},
        validation_source_sha256={n:h.digest(ROOT/n) for n in CODE},tapes=tapes,scene=reference['scene'],
        runtime=reference['runtime'],arithmetic_runtime=h.previous.accepted.arithmetic_runtime())
    report['repository']=dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        working_tree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines())
    output.path('raw').mkdir();timings=report['timings'];audits={}
    def failure_sink(label,arrays):
        report['failure_artifacts'].append(save_arrays(output,label+'-failed',arrays));persist()
    for seed in field.SEEDS:
        _,row=source_field_audit(seed,budget,timings);audits[seed]=row;report['field_audits'].append(row);persist()
    report['invalid_encoding_checks']=zero_encoding_checks()
    shader=h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==reference['shader'],'Original shader identity changed')
    report['shader']=shader.identity
    scene=reference['scene'];batch=h.MaterialBatch(scene,scene['observations'][:24],fit_tape)
    targets={m:h.previous.material_target(next(r for r in reference['materials'] if r['material']==m),m)[1] for m in h.previous.reference.PANEL}
    report['input_archive']=save_arrays(output,'inputs',dict(u=fit_tape.u.numpy(),v=fit_tape.v.numpy(),azimuth=fit_tape.azimuth.numpy(),
        positions=batch.base.positions.numpy(),normals=batch.base.normals.numpy(),views=batch.base.views.numpy(),
        **{'target_'+m:v.numpy() for m,v in targets.items()}))
    admitted={}
    for m in h.previous.reference.PANEL:
        for seed in field.SEEDS:
            triple={};entry=dict(material=m,seed=seed,status='running',arms={},comparisons={});report['initial_admission'].append(entry)
            for arm in h.ARMS:
                label=m+'-'+str(seed)+'-'+arm;model=make_model(arm,seed,budget,label,timings);identity=field.identity(model)
                if arm=='N1':require(identity==audits[seed]['model'],'Material truth changed neural initialization')
                arrays,info=capture(model,batch,shader,targets[m],budget,label,timings,failure_sink=failure_sink)
                raw=save_arrays(output,label,arrays);entry['arms'][arm]=dict(initial_model=identity,raw=raw,**info)
                triple[arm]=arrays;persist()
            entry['comparisons']['D0_N0']=parity(triple['N0'],triple['D0'],targets[m],exact=True)
            require(entry['arms']['D0']['initial_model']['parameters']==entry['arms']['N0']['initial_model']['parameters'],
                'D0/N0 initial latent parameters differ')
            entry['comparisons']['D0_N1']=parity(triple['N1'],triple['D0'],targets[m])
            entry['status']='passed' if all(c['passed'] for c in entry['comparisons'].values()) else 'failed';persist()
            require(entry['status']=='passed','Mandatory initial shader gate failed: '+m+'/'+str(seed))
            admitted[(m,seed)]=entry
    report['full_initial_shader_admission_passed']=True
    for m in h.previous.reference.PANEL:
        original=admitted[(m,101)]
        for arm in h.ARMS:
            label='replay-'+m+'-'+arm;model=make_model(arm,101,budget,label,timings)
            require(field.identity(model)==original['arms'][arm]['initial_model'],'Deterministic network replay changed')
            arrays,info=capture(model,batch,shader,targets[m],budget,label,timings,failure_sink=failure_sink)
            base=read_arrays(output,original['arms'][arm]['raw']);checks={k:comparison(arrays[k],base[k]) for k in base}
            row=replay_row(m,101,arm,save_arrays(output,label,arrays),checks,info)
            report['replay'].append(row);persist();require(row['passed'],'Exact initial rendering/gradient replay failed')
    for seed in field.SEEDS:
        base=read_arrays(output,admitted[('M1',seed)]['arms']['D0']['raw']);p0=torch.from_numpy(base['physical'][0].copy())
        for variant in ('periodic','uncoupled_swap','coupled_swap'):
            label='gauge-'+str(seed)+'-'+variant
            with budget.operation('construct_D0',label):
                p=p0.clone()
                if variant=='periodic':p[2]=p[2]+np.pi
                else:p[:2]=p0[[1,0]];p[2]=p[2]+np.pi/2
                model=PhysicalFixture(p)
            selected=batch;mapping=None;tape_record=None
            if variant!='periodic':
                mapping=h.previous.coupling.frame_sign(h.previous.coupling.decoded_frame(shader,p0,96),h.previous.coupling.decoded_frame(shader,p,96))
            if variant=='coupled_swap':
                tape,tape_record=h.previous.coupling.copy_coupled_tape(fit_tape,mapping['sign'])
                selected=h.MaterialBatch(scene,scene['observations'][:24],tape)
            arrays,info=capture(model,selected,shader,targets['M1'],budget,label,timings,backward=False,failure_sink=failure_sink)
            row=dict(seed=seed,variant=variant,raw=save_arrays(output,label,arrays),mapping=mapping,coupled_tape=tape_record,**info)
            if variant=='uncoupled_swap':
                row['diagnostic_difference']={k:comparison(arrays[k],base[k],mandatory=False) for k in ('half_vectors','specular_directions','pdf','mean')}
                q0,q1=(h.previous.coupling.accepted.tensor_q(v) for v in (p0.numpy(),p.numpy()))
                row['physical_Q']=comparison(q1,q0,2e-7,2e-6);row['passed']=row['physical_Q']['passed']
            else:
                row['comparison']=parity(arrays,base,targets['M1'],gauge=variant=='coupled_swap');row['passed']=row['comparison']['passed']
            report['gauge'].append(row);persist();require(row['passed'],'Mandatory gauge fixture failed: '+label)
    report['storage']=benchmark(output,budget,timings);report['runtime_feasibility']=feasibility(timings,report['storage'])
    h.verify_tapes(fit_tape,eval_tape,tapes)
    for name,sha in {**ids,**report['validation_source_sha256']}.items():require(h.digest(ROOT/name)==sha,'Source/evidence changed during validation: '+name)
    require(budget.finished(),'Incomplete validation work matrix')
    report.update(all_prescribed_fixtures_completed=True,full_shader_admission_passed=True,
        runtime_feasibility_passed=report['runtime_feasibility']['passed'])
    if not report['runtime_feasibility_passed']:raise Stopped('Provisional900-second history budget appears infeasible; review before freezing')
    report['status']='passed'


def run(root,run_id):
    validate_plan(plan());budget=Budget()
    report=dict(schema_version=1,protocol=PROTOCOL,base_commit=BASE,status='running',run_id=run_id,fixture_plan=plan(),
        work=budget.counts,field_audits=[],initial_admission=[],replay=[],gauge=[],storage=[],timings=[],failure_artifacts=[],
        full_shader_admission_passed=False,runtime_feasibility_passed=False,all_prescribed_fixtures_completed=False,
        recovery_updates=0,final_freeze_created=False,Phase_D_authorized=False)
    with h.OwnedOutput(root,run_id) as output:
        output.persist(report)
        def persist():output.persist(report)
        try:
            with runtime(budget,report):execute(output,budget,report,persist)
        except BaseException as exc:
            report.update(status='blocked' if isinstance(exc,Stopped) else 'failed',failure_type=type(exc).__name__,failure=str(exc))
        finally:
            report.update(shader_points={k:96*v for k,v in budget.counts['shader_forward'].items()},
                diffuse_directions={k:96*512*v for k,v in budget.counts['shader_forward'].items()},
                specular_directions={k:96*1024*v for k,v in budget.counts['shader_forward'].items()})
            output.persist(report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan',action='store_true');mode.add_argument('--execute-once',action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--run-id');args=parser.parse_args()
    if args.check_plan:
        ref,ids=admit();print(json.dumps(dict(fixture_plan=validate_plan(plan()),admitted_identities=len(ids),
            accepted_reference_status=ref['status'],validation_source_sha256={n:h.digest(ROOT/n) for n in CODE}),indent=2));return
    require(args.output_root is not None and args.run_id is not None,'Explicit owned output root and unique run ID required')
    result=run(args.output_root,args.run_id);print(json.dumps(dict(status=result['status'],work=result['work'],failure=result.get('failure')),indent=2))
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
