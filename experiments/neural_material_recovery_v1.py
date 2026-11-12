"""3D-2 controlled recovery harness. CLI inspection only; execution needs Phase D review.

Reusable training/evaluation APIs are implemented, but no scientific execution
command or final freeze is provided in Phase B. Tests use synthetic shaders/losses.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

import numpy as np
import torch

from experiments import neural_material_field_v1 as field
from experiments import multi_material_recovery_v1 as previous
from infrastructure.paths import contained_path
from infrastructure.runs import validate_run_id
from infrastructure.signatures import require, reserve_output, read_report
from infrastructure.stage_sources import ROOT, digest, baseline, PINS

BASE='a1f15d1030cb954deddd18fb408de39a452fd441'
PROTOCOL='neural-material-recovery-v1'
ARMS=('D0','N0','N1')
UPDATES=5000
SNAPSHOTS=(250,1000,5000)
FIT_CAP=5003
FORWARD_CAP=5004
REVERSE_CAP=5002
POLICY=previous.accepted.POLICY
INPUTS={
 'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json':'3abc274191d0e84c456355a7c351ac1aa94a2c976c1f789979ddb550a4383e89',
 'docs/evidence/MULTI_MATERIAL_RECOVERY_V1_ANALYSIS.json':'eb0c7d5f5f0d803c1a01dbf31739efc1cd9dbb280dfab2dd772401b5bf8688ee',
 'docs/protocols/MULTI_MATERIAL_RECOVERY_V1.freeze.json':'f8c8de54cff14d614165dfe2fc46a1ec6e88ff3e685a6ebb9e34a2f02bc2e6db',
 'docs/MULTI_MATERIAL_RECOVERY_V1_RESULTS.md':'465270c9bcae9eacbf91dec079134b7f75fe716765f31af65ee3c1a8a6981ead',
 'docs/A_NEURAL_MATERIAL_FIELD_AUDIT_V1.md':'ee9f75f91eae3748c2e166c7bbfab322d00ef4683e4fd3e06c0fcdf1b770a89d',
 'docs/NEURAL_MATERIAL_FIELD_IMPLEMENTATION_PLAN_V1.md':'be9eb0a430321e64548faec34f45f42a5b2275a3c22e2aba0bbb1e96614175bb',
 'docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1_DRAFT.md':'427c43d5c6b28263a2244243ae2cd35faaa353e1bfbe4f56a90c88cdecedc65b',
}


@dataclass(frozen=True)
class Settings:
    updates:int=5000
    variant:str='A2'
    decoder:str='atan2'
    diffuse_samples:int=512
    specular_samples:int=1024
    whole_seconds:int=28800
    active_seconds:int=900
    endpoint_reserve:int=10
    rss_bytes:int=4*1024**3
    address_bytes:int=8*1024**3

    def validate(self):
        require(asdict(self)==asdict(Settings()),'Incompatible numerical/resource configuration; no implicit override')
        return self


def admit_reference():
    """Read-only source/evidence admission, never an old run or reference render."""
    _,_,ids=previous.admit()
    for name,sha in INPUTS.items():
        require(digest(ROOT/name)==sha,'Accepted source/design/evidence changed: '+name)
    p=json.loads((ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json').read_text())
    a=json.loads((ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1_ANALYSIS.json').read_text())
    f=json.loads((ROOT/'docs/protocols/MULTI_MATERIAL_RECOVERY_V1.freeze.json').read_text())
    require(p['status']=='complete' and p['frozen_execution']==f and len(p['Adam_runs'])==9 and
        len(p['outcomes'])==27 and p['initialization_gate']=='passed','Complete accepted 3D-1 required')
    for group in (p['accepted_identities'],f['sha256'],a['figures'],INPUTS):
        for name,sha in group.items():
            require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name)==sha,'Inherited identity mismatch: '+name)
            ids[name]=sha
    require(a['source_report_sha256']==INPUTS['docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'] and
        a['additional_shader_calls']==a['additional_derivative_calls']==a['additional_reference_calls']==a['additional_optimizer_calls']==0,
        'Offline evidence provenance changed')
    ids['experiments/analyze_multi_material_recovery_v1.py']=a['offline_implementation_sha256']
    require(digest(ROOT/'experiments/analyze_multi_material_recovery_v1.py')==a['offline_implementation_sha256'],'Analysis source changed')
    for stage,pin in PINS.items():baseline(stage,pin)
    return p,ids


def inputs(reference):
    fit,evaluation,manifest=previous.tapes(reference['tapes'])
    require(reference['material_panel']==previous.reference.PANEL,'Material panel changed')
    return fit,evaluation,manifest


def verify_tapes(fit,evaluation,manifest):
    for name,tape in (('fit',fit),('evaluation',evaluation)):
        actual={k:previous.roundoff.array_record(v.numpy()) for k,v in zip(('u','v','azimuth'),(tape.u,tape.v,tape.azimuth))}
        require(actual==manifest[name]['arrays'],'Runtime fixed tape mutated: '+name)


def fresh_latent(seed):
    require(type(seed) is int and seed in previous.SEEDS,'Unapproved initialization seed')
    r=next(r for r in previous.fresh_initializations() if r['seed']==seed)
    return previous.lm.accepted.float32_exact(r['initial_unconstrained_float32'],(7,),'fresh coordinates')


def counts(arm):
    require(arm in ARMS,'Unknown representation for work accounting')
    return dict(field_attempts=0,field_completed=0,calibration_forwards=int(arm=='N1'),
        shader_attempts=0,shader_completed=0,fit_forwards=0,evaluation_forwards=0,
        attempted_points=0,completed_points=0,diffuse_directions=0,specular_directions=0,
        reverse_attempts=0,reverse_completed=0,optimizer_attempts=0,optimizer_returns=0,
        initial_forwards=0,prefix_forwards=0,terminal_forwards=0)


def work_limits(arm):
    require(arm in ARMS,'Unknown representation for work accounting')
    return dict(field_attempts=5006,field_completed=5006,calibration_forwards=int(arm=='N1'),
        shader_attempts=FORWARD_CAP,shader_completed=FORWARD_CAP,fit_forwards=FIT_CAP,evaluation_forwards=1,
        attempted_points=480624,completed_points=480624,diffuse_directions=246079488,specular_directions=492158976,
        reverse_attempts=REVERSE_CAP,reverse_completed=REVERSE_CAP,optimizer_attempts=UPDATES,optimizer_returns=UPDATES,
        initial_forwards=1,prefix_forwards=1,terminal_forwards=1)


def check_counts(c,arm):
    limits=work_limits(arm)
    require(set(c)==set(limits),'Incomplete work accounting')
    for key,limit in limits.items():require(type(c[key]) is int and 0<=c[key]<=limit,'Work cap exceeded: '+key)
    for attempt,done in (('field_attempts','field_completed'),('shader_attempts','shader_completed'),
        ('reverse_attempts','reverse_completed'),('optimizer_attempts','optimizer_returns')):
        require(c[done]<=c[attempt],'Completion count exceeds attempts')


def query(model,positions,c):
    check_counts(c,model.arm);require(c['field_attempts']<5006,'Field call cap exhausted')
    c['field_attempts']+=1
    material=model(positions).validate();c['field_completed']+=1
    return material


def material_diagnostics(material):
    return dict(physical=field.record(material.physical,True),encoding=field.record(material.encoding,True),
        rotation_raw=field.record(material.rotation_raw,True) if material.rotation_raw is not None else None,
        centered_radius=field.record(material.centered_radius,True) if material.centered_radius is not None else None,
        shader_input_norm=field.record(torch.norm(material.encoding,dim=-1,keepdim=True),True))


def constant_probes(model,c):
    points=torch.tensor([[x,y,0.] for x in (-1.,0.,1.) for y in (-1.,0.,1.)],dtype=torch.float32)
    a,b=(query(model,points,c) for _ in range(2))
    require(torch.equal(a.physical,b.physical) and torch.equal(a.encoding,b.encoding) and
        torch.equal(a.physical,a.physical[:1].expand_as(a.physical)) and
        torch.equal(a.encoding,a.encoding[:1].expand_as(a.encoding)),'Field is not constant/deterministic')
    return dict(physical=field.record(a.physical,True),encoding=field.record(a.encoding,True))


class _Capture:
    def __init__(self,shader,observe=False):self.shader=shader;self.result=None;self.inputs=None;self.observe=observe
    def shade(self,*args,**kwargs):
        if self.observe:
            self.inputs=args
            if args[7].requires_grad:args[7].retain_grad()  # Observation only; returned values/parameter gradients unchanged.
        result=self.shader.shade(*args,**kwargs);self.result=result
        return result


class MaterialBatch:
    """Compose original data/illumination and means; N1 retains its real encoding."""
    def __init__(self,scene,observations,tape):
        self.base=previous.original.ObservationBatch(scene,observations,tape,4)

    def render(self,model,shader,c,evaluation=False,phase='update',capture=False,audit=None):
        check_counts(c,model.arm)
        require(c['shader_attempts']<FORWARD_CAP,'Renderer forward cap exhausted')
        material=query(model,torch.zeros((1,3),dtype=torch.float32),c)
        if audit is not None:audit['material']=material
        base=self.base;n=base.n*base.seeds
        c['shader_attempts']+=1;c['attempted_points']+=n
        wrapped=_Capture(shader,observe=audit is not None)
        if audit is not None:audit['observer']=wrapped
        if model.arm in ('D0','N0'):
            values=base(wrapped,material.physical[0])  # Exact original operation order.
        else:
            p=material.physical.expand(n,7)
            result=wrapped.shade(base.positions,base.normals,base.views,p[:,:1],p[:,1:2],p[:,3:6],p[:,6:7],
                material.encoding.expand(n,2),base.illumination,base.tapes,is_train=True)
            require(bool(result['frame_orthonormal'].all()),'Invalid original tangent frame')
            values=torch.stack([result[k].reshape(4,base.n,3).mean(0) for k in ('diffuse_rgb','specular_rgb','linear_rgb')],1)
        c['shader_completed']+=1;c['completed_points']+=n
        if audit is not None:audit['values']=values
        c['evaluation_forwards' if evaluation else 'fit_forwards']+=1
        c['diffuse_directions']+=n*512;c['specular_directions']+=n*1024
        if phase in ('initial','prefix','terminal'):c[phase+'_forwards']+=1
        check_counts(c,model.arm)
        if not bool(torch.isfinite(values).all()):raise NumericalFailure('Nonfinite renderer predictions')
        data=None
        if capture:
            r=wrapped.result
            data={k:r[k].detach().numpy().copy() for k in
                ('tangent','bitangent','normal','half_vectors','specular_directions','diffuse_directions','pdf')}
            data['per_tape']=np.stack([r[k].detach().numpy().reshape(4,base.n,3) for k in ('diffuse_rgb','specular_rgb','linear_rgb')],2)
            data['mean']=values.detach().numpy().copy()
            data['mask']=(r['specular_directions'][:,:,2]>0).detach().numpy()
        return values,material,data


def loss_value(values,target):
    loss=torch.mean((values[:,2]-target)**2)
    if not bool(torch.isfinite(loss)):raise NumericalFailure('Nonfinite original MSE')
    return loss


class NumericalFailure(RuntimeError):pass
class HistoryLimit(RuntimeError):pass


def backward(model,loss,c):
    check_counts(c,model.arm);require(c['reverse_attempts']<REVERSE_CAP,'Reverse cap exhausted')
    c['reverse_attempts']+=1;loss.backward();c['reverse_completed']+=1
    summary=field.gradient_summary(model)
    if any(s['missing'] or not s['finite'] for s in summary.values()):raise NumericalFailure('Missing/nonfinite parameter gradient')
    return summary


def optimizer_for(model):
    cfg=previous.original.SETTINGS['optimizer']
    return torch.optim.Adam(list(model.parameters()),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])


def step(model,optimizer,c):
    check_counts(c,model.arm);require(c['optimizer_attempts']<UPDATES,'Adam step cap exhausted')
    c['optimizer_attempts']+=1;optimizer.step();c['optimizer_returns']+=1
    for p in model.parameters():
        state=optimizer.state[p]
        if not bool(torch.isfinite(p).all()) or not all(bool(torch.isfinite(v).all()) for v in state.values()):
            raise NumericalFailure('Nonfinite post-update parameter/Adam state')
        require(float(state['step'])==c['optimizer_returns'],'Original continuous Adam counter mismatch')


def compact_state(model,optimizer):
    return dict(parameters={n:field.record(p,model.arm!='N1') for n,p in model.named_parameters()},
        buffers={n:field.record(b) for n,b in model.named_buffers()},
        optimizer_effective=previous.lm.accepted.optimizer_options(optimizer),
        adam={n:{k:field.record(v,model.arm!='N1') for k,v in optimizer.state[p].items()} for n,p in model.named_parameters()})


def vector_transition_digest(previous_hash,model,optimizer,loss,update):
    """Rolling identity of EVERY D0/N0 transition, without extra model queries."""
    require(model.arm in ('D0','N0') and len(previous_hash)==64,'Vector trajectory identity required')
    state=dict(previous_sha256=previous_hash,update=update,loss=field.record(loss),
        gradient=field.record(model.latent.grad),state=compact_state(model,optimizer))
    return hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()


def prefix_check(row,model,optimizer,archived,update,loss,gradnorm=None):
    """Exact recorded 3D-1 prefix values; no old optimizer execution."""
    require(model.arm=='D0','Archived prefix gate belongs to D0 only')
    if update in (0,) or update%10==9:
        old=next((h for h in archived['history'] if h['update']==update+1),None)
        require(old is not None,'Required D0 archived prefix record is missing')
        require(float(loss)==old['loss'] and gradnorm==old['gradient_norm'],'D0 archived objective/gradient prefix mismatch')
        require(model.latent.detach().tolist()==old['unconstrained'] and
            previous.original.physical(model.latent).detach().tolist()==old['physical'],'D0 archived recorded state mismatch')
    if update+1 in (250,1000):
        actual=previous.lm.accepted.snapshot(model.latent,optimizer)
        expected=archived['snapshots'][str(update+1)]
        for key in ('unconstrained','physical','adam','optimizer_effective'):
            require(actual[key]==expected[key],'D0 archived optimizer snapshot mismatch: '+key)
    row['prefix_verified_through']=min(update+1,1000)


def reused_snapshot_objective(row,forward_update,loss,material=None):
    """Update251's ordinary forward measures the saved post250 coordinates."""
    if forward_update==251:
        require('250' in row['snapshots'],'Missing post250 snapshot before update251')
        row['snapshots']['250'].update(objective=float(loss.detach()),objective_at_update=250,
            objective_source='ordinary update251 forward; no extra renderer call')
        if material is not None:row['snapshots']['250']['material']=material_diagnostics(material)


class OwnedOutput:
    """Fresh owned output root; reuse reviewed containment/allocation guards."""
    def __init__(self,root,run_id):
        self.root=Path(root);self.run_id=validate_run_id(run_id)
        require(self.root.stat().st_uid==os.geteuid(),'Output root is not owned by the current user')
        require(not any((a/'.git').is_file() or (a/'.git/HEAD').is_file() for a in (self.root,*self.root.parents)),
            'Output root is inside a Git checkout')
        for forbidden in (ROOT.parent/'data',ROOT.parent/'final_results'):
            require(self.root!=forbidden and forbidden not in self.root.parents and self.root not in forbidden.parents,'Output overlaps preserved research inputs')
        self.stream=reserve_output(self.root,'report.json',inputs=(ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json',))
        try:
            self.path('checkpoints').mkdir()
            self.write_json('identity.json',dict(schema_version=1,protocol=PROTOCOL,run_id=run_id))
        except BaseException:
            self.stream.close();raise  # Preserve the partial allocation; never reuse/clean it up.
    def path(self,name):
        require(self.root.resolve()==self.root and self.root.stat().st_uid==os.geteuid(),'Owned output root was redirected')
        return contained_path(self.root,name,writable=True)
    def write_json(self,name,value):
        data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
        with self.path(name).open('x') as f:f.write(data)
    def persist(self,value):previous.coupling.accepted.persist_report(self.stream,value)
    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def save_checkpoint(output,stem,model,optimizer,completed_updates,diagnostic=False,provenance=None):
    """Own numeric state only; no pickle, full tensor values in JSON, or overwrite."""
    require(type(completed_updates) is int and 0<=completed_updates<=UPDATES,'Invalid checkpoint update count')
    if not diagnostic:
        require(completed_updates>=1 and all(set(optimizer.state.get(p,{}))=={'step','exp_avg','exp_avg_sq'} and
            float(optimizer.state[p]['step'])==completed_updates for p in model.parameters()),'State checkpoint requires exact initialized Adam counters')
    validate_run_id(stem)
    entries=[];arrays={}
    def add(role,name,tensor):
        a=tensor.detach().cpu().numpy().copy();require(a.dtype.kind!='O','Object array forbidden')
        finite=bool(np.isfinite(a).all());require(diagnostic or finite,'Nonfinite resumable state')
        key='a'+str(len(arrays)).zfill(4);arrays[key]=a
        entries.append(dict(key=key,role=role,name=name,finite=finite,**field.record(tensor)))
    for name,p in model.named_parameters():
        add('parameter',name,p)
        for key,value in optimizer.state.get(p,{}).items():add('adam',name+'/'+key,value)
    for name,b in model.named_buffers():add('buffer',name,b)
    add('torch_rng','CPU',torch.get_rng_state())
    numpy_rng=np.random.get_state();arrays['numpy_rng']=numpy_rng[1].copy()
    archive=output.path('checkpoints/'+stem+'.npz');manifest=output.path('checkpoints/'+stem+'.json')
    require(not archive.exists() and not manifest.exists(),'Checkpoint collision; no implicit overwrite/resume')
    with archive.open('xb') as f:np.savez(f,**arrays)
    value=dict(schema_version=1,protocol=PROTOCOL,purpose='failure_diagnostic' if diagnostic else 'state',
        arm=model.arm,model_configuration=field.identity(model)['configuration'],source=getattr(model,'source_identity',None),
        completed_updates=completed_updates,optimizer_effective=previous.lm.accepted.optimizer_options(optimizer),
        archive=archive.name,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,entries=entries,
        execution_provenance=provenance,
        numpy_rng=dict(kind=numpy_rng[0],position=numpy_rng[2],has_gauss=numpy_rng[3],cached_gaussian=numpy_rng[4],
            array_sha256=hashlib.sha256(arrays['numpy_rng'].tobytes()).hexdigest()))
    output.write_json('checkpoints/'+stem+'.json',value)
    return dict(manifest=str(manifest.relative_to(output.root)),manifest_sha256=digest(manifest),completed_updates=completed_updates)


def persist_failure_capture(output,stem,audits,exc):
    """Save already-produced tensors/gradients; never rerender for diagnostics."""
    validate_run_id(stem);folder=output.path('diagnostics');folder.mkdir(exist_ok=True)
    arrays={}
    def add(name,value):
        if isinstance(value,torch.Tensor):arrays[name]=value.detach().cpu().numpy().copy()
    for i,audit in enumerate(audits):
        prefix=str(i)+'::';model=audit.get('model')
        if model is not None:
            for name,p in model.named_parameters():add(prefix+'parameter::'+name,p);add(prefix+'gradient::'+name,p.grad)
            for name,b in model.named_buffers():add(prefix+'buffer::'+name,b)
        observer=audit.get('observer')
        if observer is not None:
            for name,t in (observer.result or {}).items():add(prefix+'return::'+name,t)
            if observer.inputs is not None:
                for j,t in enumerate(observer.inputs):add(prefix+'input::'+str(j),t)
                add(prefix+'encoding_gradient',observer.inputs[7].grad)
                if len(observer.inputs)>9:
                    for name in ('u','v','azimuth'):add(prefix+'tape::'+name,getattr(observer.inputs[9],name,None))
        add(prefix+'values',audit.get('values'))
    archive=output.path('diagnostics/'+stem+'.npz')
    require(all(a.dtype.kind!='O' for a in arrays.values()),'No pickle/object diagnostics')
    with archive.open('xb') as stream:np.savez(stream,**arrays)
    value=dict(schema_version=1,purpose='failure_only_no_resume',failure_type=type(exc).__name__,failure=str(exc),
        archive=archive.name,sha256=digest(archive),arrays={name:dict(shape=list(a.shape),dtype=str(a.dtype),
            sha256=hashlib.sha256(a.tobytes()).hexdigest(),finite=bool(np.isfinite(a).all())) for name,a in arrays.items()},additional_shader_calls=0)
    output.write_json('diagnostics/'+stem+'.json',value)
    return dict(manifest='diagnostics/'+stem+'.json',manifest_sha256=digest(output.path('diagnostics/'+stem+'.json')),
        archive='diagnostics/'+stem+'.npz',archive_sha256=value['sha256'])


def restore_checkpoint(output,selection,model,optimizer,expected_provenance=None):
    """Explicit own checkpoint selection; not an experiment resume feature."""
    path=output.path(selection['manifest'])
    require(digest(path)==selection['manifest_sha256'],'Checkpoint manifest hash mismatch before numeric loading')
    value,_=read_report(path,selection['manifest_sha256'])
    require(type(value.get('schema_version')) is int and value['schema_version']==1,'Unsupported own checkpoint schema')
    if 'completed_updates' in selection:require(selection['completed_updates']==value['completed_updates'],'Explicit checkpoint counter selection mismatch')
    if expected_provenance is not None:require(value.get('execution_provenance')==expected_provenance,'Checkpoint execution-provenance mismatch')
    require(value['protocol']==PROTOCOL and value['purpose']=='state' and value['arm']==model.arm and
        value['source']==getattr(model,'source_identity',None) and value['model_configuration']==field.identity(model)['configuration'],
        'Foreign/incompatible or failure-only checkpoint')
    archive=contained_path(path.parent,value['archive'],writable=True)
    require(digest(archive)==value['archive_sha256'] and archive.stat().st_size==value['archive_bytes'],'Numeric checkpoint hash/size mismatch')
    expected={('parameter',n):p for n,p in model.named_parameters()}
    expected.update({('buffer',n):b for n,b in model.named_buffers()})
    loaded={};states={n:{} for n,_ in model.named_parameters()}
    with np.load(archive,allow_pickle=False) as arrays:
        require(set(arrays.files)=={e['key'] for e in value['entries']}|{'numpy_rng'},'Unknown/missing numeric arrays')
        require(len(value['entries'])==len({e['key'] for e in value['entries']}),'Duplicate tensor catalog key')
        for entry in value['entries']:
            a=arrays[entry['key']];require(a.dtype.kind!='O' and np.isfinite(a).all(),'Invalid numeric checkpoint array')
            t=torch.from_numpy(a.copy());require(field.record(t)=={k:entry[k] for k in ('dtype','device','shape','sha256')},'Tensor catalog mismatch')
            pair=(entry['role'],entry['name']);require(pair not in loaded,'Duplicate checkpoint tensor');loaded[pair]=t
            if entry['role'] in ('parameter','buffer'):
                require(pair in expected and t.shape==expected[pair].shape and t.dtype==expected[pair].dtype,'Non-strict model state')
                if entry['role']=='buffer':require(torch.equal(t,expected[pair]),'Immutable calibration/anchor buffer changed')
            elif entry['role']=='adam':
                name,key=entry['name'].rsplit('/',1);require(name in states and key in ('step','exp_avg','exp_avg_sq'),'Unknown optimizer state')
                states[name][key]=t
            else:require(pair==('torch_rng','CPU'),'Unknown tensor role')
        require(arrays['numpy_rng'].shape==(624,) and arrays['numpy_rng'].dtype==np.uint32 and value['numpy_rng']['kind']=='MT19937' and
            hashlib.sha256(arrays['numpy_rng'].tobytes()).hexdigest()==value['numpy_rng']['array_sha256'],'NumPy RNG identity mismatch')
    require(('torch_rng','CPU') in loaded and loaded[('torch_rng','CPU')].shape==torch.get_rng_state().shape and
        loaded[('torch_rng','CPU')].dtype==torch.uint8,'Missing/invalid CPU RNG provenance')
    require({p for p in loaded if p[0] in ('parameter','buffer')}==set(expected),'Incomplete strict model state')
    require(value['optimizer_effective']==previous.lm.accepted.optimizer_options(optimizer),'Optimizer configuration mismatch')
    for name,p in model.named_parameters():
        state=states[name]
        require(set(state)=={'step','exp_avg','exp_avg_sq'} and state['step'].shape==() and
            state['step'].dtype==torch.float32 and float(state['step'])==value['completed_updates'] and
            type(value['completed_updates']) is int and 1<=value['completed_updates']<=UPDATES and
            state['exp_avg'].shape==p.shape and state['exp_avg'].dtype==p.dtype and
            state['exp_avg_sq'].shape==p.shape and state['exp_avg_sq'].dtype==p.dtype and
            bool((state['exp_avg_sq']>=0).all()),'Invalid Adam counter/moments')
    with torch.no_grad():
        for pair,t in expected.items():t.copy_(loaded[pair])
    for name,p in model.named_parameters():optimizer.state[p]={k:v.clone() for k,v in states[name].items()}
    # Never mutate caller RNG; its archived numeric state is provenance only.
    return value


class WholeClock(previous.WholeClock):
    def remaining(self):return Settings().whole_seconds-(time.perf_counter()-self.start)
    def check(self):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>Settings().rss_bytes:raise previous.lm.WholeLimit('RSS cap exceeded')
        if self.remaining()<=0:raise previous.lm.WholeLimit('Whole-process provisional wall cap reached')
    def alarm(self,signum,frame):
        if signum==signal.SIGTERM:raise previous.lm.WholeLimit('External termination; no retry')
        self.check()
        if self.active is not None:raise HistoryLimit('Accumulated active history timer reached')
        raise previous.lm.WholeLimit('Whole-process timer reached')


class HistoryClock(previous.lm.ArmClock):
    def check(self,reserve=0.):
        self.whole.check()
        if self.elapsed()>=Settings().active_seconds-reserve:raise HistoryLimit('Accumulated active history cap reached')
    @contextmanager
    def active(self):
        require(self.started is None and self.whole.active is None,'Nested active clocks forbidden')
        self.check();self.started=(time.perf_counter(),time.process_time());self.whole.active=self
        signal.setitimer(signal.ITIMER_REAL,max(1e-6,min(self.whole.remaining(),Settings().active_seconds-self.wall)))
        try:yield
        finally:
            self.wall+=time.perf_counter()-self.started[0];self.cpu+=time.process_time()-self.started[1]
            self.started=None;self.whole.active=None
            remaining=self.whole.remaining();signal.setitimer(signal.ITIMER_REAL,remaining if remaining>0 else 0.)


def evaluate(model,batch,shader,c,truth,targets,observations,audit=None):
    with torch.no_grad():values,material,_=batch.render(model,shader,c,evaluation=True,audit=audit)
    predictions=values.numpy().astype(np.float64)
    metrics={}
    for split in ('fit','held_out','relight','relight_seen_views','relight_unseen_views'):
        mask=[o['split']==split if split in ('fit','held_out','relight') else o['split']=='relight' and
              (o['view']<6 if split=='relight_seen_views' else o['view']>=6) for o in observations]
        metrics[split]={name:previous.original.radiance_errors(predictions[mask,i],targets[mask,i]) for i,name in enumerate(('diffuse','specular','total'))}
    physical=material.physical[0].numpy();errors=previous.recovery_errors(physical,truth)
    delta=predictions-targets;compensation={}
    for split in ('fit','held_out','relight'):
        mask=[o['split']==split for o in observations];d,s,t=(delta[mask,i] for i in range(3))
        compensation[split]=dict(diffuse_specular_inner_product=float(np.sum(d*s)),
            residual_correlation=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s),1e-30)),
            total_squared_error=float(np.sum(t*t)),component_squared_error=float(np.sum(d*d)+np.sum(s*s)))
    return dict(physical=physical.tolist(),encoding=field.record(material.encoding,True),predictions=predictions.tolist(),
        prediction_sha256=previous.oldref.sha_array(predictions),recovery_errors=errors,radiance_errors=metrics,
        success=previous.original.assess_success(errors,metrics),component_compensation=compensation)


def initial_admission(models,batch,shader,target,counters,evidence_sink=None):
    """Complete exact/bounded contract; preserve same-call negative evidence."""
    audits=[]
    try:return _initial_admission(models,batch,shader,target,counters,audits)
    except BaseException as exc:
        if evidence_sink is not None:evidence_sink(audits,exc)
        raise


def _initial_admission(models,batch,shader,target,counters,audits):
    """Future Phase C/D numerical admission, never called by Phase B unit tests with a real shader."""
    rows=[]
    require([m.arm for m in models]==list(ARMS),'Original representation order required')
    require(torch.equal(models[0].latent,models[1].latent),'D0/N0 initial latent identity failed')
    for model,c in zip(models,counters):
        audit=dict(arm=model.arm,model=model);audits.append(audit)
        probes=constant_probes(model,c)
        for p in model.parameters():p.grad=None
        values,material,data=batch.render(model,shader,c,phase='initial',capture=True,audit=audit)
        loss=loss_value(values,target);grad=backward(model,loss,c)
        require(all(g['nonzero'] for g in grad.values()),'Unexpected inactive whole material group at initialization')
        actual_encoding=audit['observer'].inputs[7]
        require(actual_encoding.grad is not None and bool(torch.isfinite(actual_encoding.grad).all()) and
            bool(torch.count_nonzero(actual_encoding.grad)),'Disconnected/nonfinite actual shader orientation gradient')
        data['encoding']=actual_encoding.detach().numpy().copy()
        data['encoding_gradient']=actual_encoding.grad.detach().numpy().copy()
        rows.append(dict(arm=model.arm,material=material,data=data,loss=float(loss.detach()),gradients=grad,probes=probes))
    require([m.arm for m in models]==list(ARMS),'Original representation order required')
    a,b,n=rows
    require(torch.equal(a['material'].physical,b['material'].physical) and torch.equal(a['material'].encoding,b['material'].encoding) and
        np.array_equal(a['data']['mean'],b['data']['mean']) and a['loss']==b['loss'] and
        torch.equal(models[0].latent.grad,models[1].latent.grad),'D0/N0 exact initial parity failed')
    exact={key:bool(np.array_equal(a['data'][key],b['data'][key])) for key in
        ('tangent','bitangent','normal','half_vectors','specular_directions','diffuse_directions','pdf','mask','per_tape','mean','encoding','encoding_gradient')}
    require(all(exact.values()),'D0/N0 exact sample/component/encoding-gradient parity failed: '+str(exact))
    p0,p1=(r['material'].physical[0].detach().numpy() for r in (a,n))
    q0,q1=(previous.coupling.accepted.tensor_q(p) for p in (p0,p1))
    require(np.array_equal(p0[[0,1,3,4,5,6]],p1[[0,1,3,4,5,6]]) and
        np.all(abs(q1-q0)<=2e-7+2e-6*abs(q0)),'Neural calibrated physical matching failed')
    d0,d1=a['data'],n['data'];checks={}
    for key in ('tangent','bitangent','normal'):checks[key]=bool(np.max(abs(d0[key]-d1[key]))<=2e-6)
    for key,atol in (('half_vectors',2e-6),('specular_directions',4e-6)):
        checks[key]=previous.accepted.bounded(d1[key].reshape(4,24,1024,3),d0[key].reshape(4,24,1024,3),atol,2e-5,True)['passed']
    checks['mask']=np.array_equal(d0['mask'],d1['mask'])
    checks['diffuse_directions']=np.array_equal(d0['diffuse_directions'],d1['diffuse_directions'])
    for i,name in enumerate(('diffuse','specular','total')):
        for key in ('per_tape','mean'):
            aa,bb=d1[key][...,i,:],d0[key][...,i,:]
            checks[key+'_'+name]=bool(np.array_equal(aa,bb) if i==0 else np.all(abs(aa.astype(float)-bb)<=2e-6+2e-5*abs(bb)))
    envelope=previous.accepted.objective_envelope(d0['mean'][:,2],d1['mean'][:,2],target.numpy(),a['loss'],n['loss'])
    checks['arithmetic']=envelope['passed']
    for row in rows:
        data=row['data'];replay=np.stack([torch.from_numpy(data['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
        checks[row['arm']+'_exact_replay']=bool(np.array_equal(replay,data['mean']) and previous.roundoff.mse32(data['mean'][:,2],target.numpy())[0]==row['loss'])
    require(all(checks.values()),'Mandatory neural renderer/frame/sample/component admission failed: '+str(checks))
    return dict(passed=True,checks=checks,arithmetic=envelope,gradients=[r['gradients'] for r in rows],
        D0_N0_exact=exact,
        initial_materials={r['arm']:material_diagnostics(r['material']) for r in rows},
        old_scalar_warning_passed=abs(n['loss']-a['loss'])<=1e-12+5e-7*abs(a['loss']))


def train_history(model,batch,shader,target,row,clock,persist,checkpoint,archived=None,paired_history=None,paired_trajectory=None,failure_state=None,expected_optimizer=None):
    """Candidate future 5000-update loop; never invoked to completion in Phase B."""
    require(row['status']=='admitted' and row['completed_updates']==0,'Fresh admitted history required')
    optimizer=optimizer_for(model);c=row['work'];row.update(status='optimizing',history=[],snapshots={})
    if model.arm in ('D0','N0'):row['trajectory_sha256']='0'*64
    try:
        require(not optimizer.state,'Fresh Adam state required; no implicit continuation')
        if expected_optimizer is not None:require(previous.lm.accepted.optimizer_options(optimizer)==expected_optimizer,'Effective Adam options differ from final freeze')
        with clock.active():
            for update in range(1,UPDATES+1):
                clock.check(Settings().endpoint_reserve);optimizer.zero_grad()
                values,material,_=batch.render(model,shader,c)
                loss=loss_value(values,target);grad=backward(model,loss,c)
                gradnorm=math.sqrt(sum(g['squared_norm'] for g in grad.values()))
                # Original vector norm is required for exact archived D0 reproduction.
                if model.arm in ('D0','N0'):gradnorm=float(torch.norm(model.latent.grad))
                reused_snapshot_objective(row,update,loss,material)
                clock.check(Settings().endpoint_reserve);step(model,optimizer,c);row['completed_updates']+=1
                if model.arm in ('D0','N0'):
                    row['trajectory_sha256']=vector_transition_digest(row['trajectory_sha256'],model,optimizer,loss,update)
                if archived is not None and update<=1000:prefix_check(row,model,optimizer,archived,update-1,loss.detach(),gradnorm)
                if update==1 or update%10==0:
                    log=dict(update=update,loss_at_update=update-1,loss=float(loss.detach()),gradient_norm=gradnorm,
                        physical_before_update=material.physical[0].detach().tolist(),material_before_update=material_diagnostics(material),
                        gradient_groups=grad,work=copy.deepcopy(c),active_wall_seconds=clock.elapsed())
                    row['history'].append(log)
                    if paired_history is not None:
                        expected=next(h for h in paired_history if h['update']==update)
                        for key in ('loss','gradient_norm','physical_before_update'):require(log[key]==expected[key],'N0/D0 exact paired history mismatch')
                    persist()
                if update in SNAPSHOTS:
                    row['snapshots'][str(update)]=checkpoint(update,model,optimizer,row)
                if update==1000:
                    with torch.no_grad():v,m,_=batch.render(model,shader,c,phase='prefix')
                    objective=float(loss_value(v,target));row['snapshots']['1000'].update(objective=objective,
                        objective_at_update=1000,objective_source='one counted no-grad prefix forward',material=material_diagnostics(m))
                    if archived is not None:require(objective==archived['snapshots']['1000']['objective'],'D0 archived post1000 objective mismatch')
                    row['prefix_reproduction_passed']=archived is not None
                check_counts(c,model.arm)
            if model.arm=='N0':require(row['trajectory_sha256']==paired_trajectory,'N0/D0 full transition trajectory mismatch')
            clock.check();optimizer.zero_grad()
            v,material,_=batch.render(model,shader,c,phase='terminal');loss=loss_value(v,target);grad=backward(model,loss,c)
            row.update(status='optimized',termination='exact_5000_update_ceiling',final_fit_MSE=float(loss.detach()),
                final_physical=material.physical[0].detach().tolist(),final_gradient_groups=grad,final_state=compact_state(model,optimizer))
            row['snapshots']['5000'].update(objective=row['final_fit_MSE'],objective_at_update=5000,
                objective_source='one counted terminal forward/backward; no update',material=material_diagnostics(material));clock.check()
    except (HistoryLimit,previous.lm.ArmLimit) as exc:
        row.update(status='censored',termination='active_time_cap',failure=str(exc))
    except NumericalFailure as exc:
        row.update(status='failed_numerical',termination='numerical_failure',failure=str(exc))
    except previous.lm.WholeLimit as exc:
        row.update(status='censored',termination='whole_process_resource_cap',failure=str(exc));raise
    except BaseException as exc:
        row.update(status='failed_integrity',failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        row.update(completed_updates=c['optimizer_returns'],active_wall_seconds=clock.wall,CPU_seconds=clock.cpu)
        if row['status']!='optimized' and failure_state is not None:failure_state(model,optimizer,row)
        persist()
    return optimizer


def validate_execution_review(review,root=None,run_id=None):
    """Phase B cannot manufacture the final Phase C/D authorization evidence."""
    require(type(review) is dict and review.get('phase')=='D' and review.get('scientific_execution_authorized') is True,
        'Phase D execution is not authorized by Phase B')
    require(review.get('external_watchdog')==dict(wall_seconds=28800,kill_grace_seconds=10),
        'Separate review must require the external whole-process watchdog')
    from experiments.freeze_neural_material_recovery_v1 import validate_freeze,check_output_selection
    path=Path(review.get('freeze_path',''))
    require(path.is_absolute() and path.resolve()==path,'Explicit unaliased final freeze path required')
    freeze=validate_freeze(path,review.get('freeze_sha256'))
    require(root is not None and run_id is not None and review.get('approved_output_root')==str(root) and
        review.get('approved_run_id')==run_id,'Separate Phase D approval must bind the exact output root/run ID')
    check_output_selection(freeze,Path(root),run_id)
    return freeze


@contextmanager
def controlled_runtime(report):
    require(all(os.environ.get(k)==v for k,v in previous.original.REQUIRED_ENV.items()),'Explicit hidden-GPU/single-thread CPU environment required')
    before_threads=torch.get_num_threads();rng=torch.get_rng_state().clone();nrng=np.random.get_state()
    address=resource.getrlimit(resource.RLIMIT_AS);handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'Do not replace another active process timer')
    whole=WholeClock();begin,cpu=time.perf_counter(),time.process_time()
    report['initial_torch_CPU_RNG']=field.record(rng)
    try:
        torch.set_num_threads(1)
        cap=min(Settings().address_bytes,address[0]) if address[0]>=0 else Settings().address_bytes
        if address[1]>=0:cap=min(cap,address[1])
        resource.setrlimit(resource.RLIMIT_AS,(cap,address[1]))
        for s in handlers:signal.signal(s,whole.alarm)
        signal.setitimer(signal.ITIMER_REAL,Settings().whole_seconds)
        yield whole
    finally:
        after=np.random.get_state()
        report.update(global_torch_CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),
            global_numpy_RNG_unchanged=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:])
        torch.set_rng_state(rng);np.random.set_state(nrng)
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        torch.set_num_threads(before_threads);resource.setrlimit(resource.RLIMIT_AS,address)
        report.update(threads_restored=torch.get_num_threads()==before_threads,memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS)==address,
            wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(all(report[k] for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged',
        'threads_restored','memory_limit_restored')),'Runtime/RNG integrity changed; no successful experiment admission')


def execute_reviewed(review,root,run_id):
    """Future Phase D API. There is deliberately no Phase B CLI route here.

    Phase C must supply its actual report and final reviewed source freeze;
    neither exists as a product of this implementation milestone.
    """
    freeze=validate_execution_review(review,root,run_id)  # Before allocation, model, shader or optimization.
    reference,ids=admit_reference();fit_tape,eval_tape,manifest=inputs(reference)
    require(manifest==freeze['tapes'],'Final tape identities changed')
    require(Settings().validate() and reference['runtime']==dict(python=platform.python_version(),numpy=np.__version__,
        torch=torch.__version__,device='cpu',torch_threads=1),'Exact accepted CPU numerical runtime required')
    report=dict(schema_version=1,protocol=PROTOCOL,run_id=run_id,status='running',phase='admission',base_commit=BASE,
        preparation_baseline=freeze['baseline'],repository=dict(
            HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()),
        settings=asdict(Settings()),reviewed_freeze_sha256=review['freeze_sha256'],accepted_identities=ids,
        tapes=manifest,material_panel=previous.reference.PANEL,outcomes=[],initial_admission=[],optimization_began=False,
        GPU_execution=False,learned_geometry=False,learned_lighting=False,failure_artifacts=[])
    provenance=dict(freeze_sha256=review['freeze_sha256'],configuration_sha256=freeze['configuration_sha256'],
        source_sha256=freeze['sha256'],run_id=run_id)
    root=Path(root)
    if not root.exists():root.mkdir(exist_ok=False)
    with OwnedOutput(root,run_id) as output:
        output.write_json('frozen-execution.json',dict(freeze_path=review['freeze_path'],freeze_sha256=review['freeze_sha256'],
            scientific_execution_authorized=True,approved_output_root=str(root),approved_run_id=run_id))
        output.persist(report)
        try:
            with controlled_runtime(report) as whole:
                scene=reference['scene'];obs=scene['observations']
                fit=MaterialBatch(scene,obs[:24],fit_tape);evaluation=MaterialBatch(scene,obs,eval_tape)
                shader=previous.original.MaterialShaderV1('A2','atan2')
                require(shader.identity==reference['shader'],'Accepted shader identity changed')
                models={};rows={};clocks={};targets={};report['phase']='initial_admission'
                def persist():output.persist(report)
                for material in previous.reference.PANEL:
                    stored=next(r for r in reference['materials'] if r['material']==material)
                    fine,target=previous.material_target(stored,material);targets[material]=(fine,target)
                    for seed in previous.SEEDS:
                        triple=[];work=[]
                        for arm in ARMS:
                            key=(material,seed,arm);model=field.make_material(arm,fresh_latent(seed),seed)
                            model_id=field.identity(model)
                            require(model_id==freeze['initial_models'][material+'/'+str(seed)+'/'+arm],'Frozen initial model/parameter/buffer identity mismatch')
                            row=dict(material=material,seed=seed,arm=arm,status='constructed',completed_updates=0,work=counts(arm),
                                initial_model=model_id,fit_target_float32_sha256=stored['fit_target_float32_sha256'])
                            models[key]=model;rows[key]=row;clocks[key]=HistoryClock(whole)
                            triple.append(model);work.append(row['work']);report['outcomes'].append(row)
                        # Whole-run admission work, followed by each history's original active cap.
                        def negative(audits,exc):
                            artifact=persist_failure_capture(output,'admission-'+material.lower()+'-'+str(seed),audits,exc)
                            report['failure_artifacts'].append(artifact)
                            report['failure_category']='renderer_sample_parity' if any(a.get('observer') is not None for a in audits) else 'initialization_admission'
                            persist()
                        gate=initial_admission(triple,fit,shader,target,work,evidence_sink=negative)
                        report['initial_admission'].append(dict(material=material,seed=seed,**gate))
                        for arm in ARMS:rows[(material,seed,arm)]['status']='admitted'
                        whole.check();persist()
                verify_tapes(fit_tape,eval_tape,manifest)
                require(len(report['initial_admission'])==9 and all(r['passed'] for r in report['initial_admission']),
                    'All 27 initial forwards/nine gates required before optimization')
                report.update(phase='optimization',optimization_began=True);persist()
                for arm in ARMS:  # All nine D0 controls first, then N0, then N1.
                    if arm!='D0':
                        require(all(rows[(m,s,'D0')]['status']=='complete' and rows[(m,s,'D0')]['prefix_reproduction_passed']
                            for m in previous.reference.PANEL for s in previous.SEEDS),'All D0 controls must reproduce and complete before neural runs')
                    if arm=='N1':require(all(rows[(m,s,'N0')]['status']=='complete' for m in previous.reference.PANEL for s in previous.SEEDS),
                        'Every exact N0 paired control must complete before N1')
                    for material in previous.reference.PANEL:
                        for seed in previous.SEEDS:
                            key=(material,seed,arm);model=models[key];row=rows[key];clock=clocks[key]
                            report['active_history']=dict(material=material,seed=seed,arm=arm,stage='optimization')
                            fine,target=targets[material]
                            require(previous.oldref.sha_array(target.numpy())==row['fit_target_float32_sha256'],'Material fit target changed')
                            archived=next(r for r in reference['Adam_runs'] if r['material']==material and r['seed']==seed) if arm=='D0' else None
                            def checkpoint(update,model,optimizer,current):
                                state=compact_state(model,optimizer)
                                if arm=='N0':require(state==rows[(material,seed,'D0')]['snapshots'][str(update)]['compact_state'],
                                    'N0/D0 exact checkpoint parameter/moment mismatch')
                                saved=save_checkpoint(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-u'+str(update),model,optimizer,update,provenance=provenance)
                                return dict(**saved,compact_state=state)
                            def failed_state(model,optimizer,current):
                                started=time.perf_counter()
                                current['failure_checkpoint']=save_checkpoint(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-failure',
                                    model,optimizer,current['work']['optimizer_returns'],diagnostic=True,provenance=provenance)
                                current['failure_persistence_wall_seconds']=time.perf_counter()-started
                            opt=train_history(model,fit,shader,target,row,clock,persist,checkpoint,archived,
                                rows[(material,seed,'D0')]['history'] if arm=='N0' else None,
                                rows[(material,seed,'D0')].get('trajectory_sha256') if arm=='N0' else None,failed_state,freeze['configuration']['optimizer_effective'])
                            if row['status']=='optimized':
                                if arm=='N0':require(row['final_state']==rows[(material,seed,'D0')]['final_state'],'N0/D0 final parameter/Adam state mismatch')
                                audit=dict(model=model)
                                report['active_history']['stage']='evaluation'
                                try:
                                    with clock.active():
                                        row['evaluation']=evaluate(model,evaluation,shader,row['work'],previous.reference.truth(material),fine,obs,audit=audit)
                                        clock.check()
                                    row['status']='complete';row['scientific_outcome']='complete_pass' if row['evaluation']['success']['all_passed'] else 'complete_fail'
                                    require(row['completed_updates']==UPDATES and row['work']==work_limits(arm),'Complete endpoint has incomplete work counters')
                                except (HistoryLimit,previous.lm.ArmLimit) as exc:
                                    row.update(status='censored',termination='evaluation_active_time_cap',failure=str(exc));failed_state(model,opt,row)
                                    report['failure_artifacts'].append(persist_failure_capture(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-evaluation',[audit],exc))
                                except NumericalFailure as exc:
                                    row.update(status='failed_evaluation',failure=str(exc));failed_state(model,opt,row)
                                    report['failure_artifacts'].append(persist_failure_capture(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-evaluation',[audit],exc))
                                except previous.lm.WholeLimit as exc:
                                    row.update(status='censored',termination='whole_process_resource_cap',failure=str(exc));failed_state(model,opt,row)
                                    report['failure_artifacts'].append(persist_failure_capture(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-evaluation',[audit],exc));raise
                                except BaseException as exc:
                                    row.update(status='failed_evaluation',failure_type=type(exc).__name__,failure=str(exc));failed_state(model,opt,row)
                                    report['failure_artifacts'].append(persist_failure_capture(output,material.lower()+'-'+str(seed)+'-'+arm.lower()+'-evaluation',[audit],exc));raise
                            row.update(active_wall_seconds=clock.wall,CPU_seconds=clock.cpu);persist();whole.check()
                            verify_tapes(fit_tape,eval_tape,manifest)
                            if arm in ('D0','N0'):require(row['status']=='complete','Required control is incomplete; stop before scientific comparisons')
                for name,sha in ids.items():require(digest(ROOT/name)==sha,'Inherited artifact changed during execution')
                verify_tapes(fit_tape,eval_tape,manifest)
                if all(r['status']=='complete' for r in rows.values()):
                    require({k:sum(r['work'][k] for r in rows.values()) for k in work_limits('D0')}==freeze['expected_total_work'],
                        'Complete experiment has incomplete aggregate work counters')
                report.update(status='complete' if all(r['status']=='complete' for r in rows.values()) else 'incomplete',phase='finished');persist()
        except previous.lm.WholeLimit as exc:
            report.update(status='censored',failure_category='whole_process_resource_censoring',failure_type=type(exc).__name__,failure=str(exc));raise
        except BaseException as exc:
            active=report.get('active_history',{})
            category=report.get('failure_category') or ('evaluation' if active.get('stage')=='evaluation' else
                'baseline_reproduction' if active.get('arm') in ('D0','N0') else
                'initialization_admission' if report['phase']=='initial_admission' else 'source_evidence_integrity')
            if isinstance(exc,OSError):category='source_evidence_integrity'
            report.update(status='failed',failure_category=category,failure_type=type(exc).__name__,failure=str(exc));raise
        finally:
            output.persist(report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true',required=True,help='Read-only provenance/settings; no model, shader or output allocation')
    parser.parse_args()
    reference,ids=admit_reference()
    print(json.dumps(dict(phase='C.2',execution_enabled=False,base_commit=BASE,admitted_identities=len(ids),
        reference_status=reference['status'],representations=list(ARMS),settings=asdict(Settings()),
        freeze_present=(ROOT/'docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json').is_file(),
        remaining_execution_gates=['separate Phase D authorization','approved fresh output selection',
            'execution-time exact D0 prefix and N0 full trajectory']),indent=2))


if __name__=='__main__':main()
