"""State-10 -> state-100 continuation. CLI is preparation only; no scientific imports."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from infrastructure import neural_material_observer_e2a as obs
from infrastructure.signatures import read_report,require
from infrastructure.runs import validate_run_id
from experiments import neural_material_early_step_e2a as prior

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/home/isavand/paper/evidence/neural-e2b-d1f760d630964a2786ff45bde7438825')
BASE='fea5cfb2ef32c0f5b7c10c6c83b8d314d0c0dcac'
F2='docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json'
F2_SHA='f9d8f28a46641c8f3aa777118b7f106f120918e7c60e6605387fcb7c46d984ad'
REPORT_SHA='bb12d3627e778634da17da7a2b484ebe608a88d94aab0bc6c033cd58a1f15373'
SEAL_SHA='cabee3c788264f60b127882274b8f03d81b28248defacc509bb4ee20843786f5'
RECEIPT_SHA='20f52bbcec66b1a06e1dfb9320f848304c536aababe63e8ac39630ec9938cbcd'
EVIDENCE_MANIFEST='docs/results/neural_material_lr_e2b/evidence_manifest.json'
EVIDENCE_MANIFEST_SHA='6b5661f553b0df096ead3dad88716233d0211faf07c759d2e0a22032bf651386'
FREEZE='docs/protocols/NEURAL_MATERIAL_CONTINUATION_E3_V1.freeze.json'
VALIDATION='docs/evidence/NEURAL_MATERIAL_CONTINUATION_E3_PREPARATION_V1.json'
CODE=('experiments/neural_material_continuation_e3.py','tests/scientific/test_neural_material_continuation_e3.py',
      'docs/protocols/NEURAL_MATERIAL_CONTINUATION_E3_V1.md')
CASES=prior.CASES;RATES=prior.RATES;START=10;END=100
CAPS=dict(forward_attempts=92,forwards=92,backward_attempts=91,backwards=91,optimizer_attempts=90,successful_updates=90,
          training_forwards=90,terminal_forwards=1,external_forwards=1)
SNAPSHOTS=(20,50,100)
_ADMITTED=object()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def options(opt):return json.loads(json.dumps({k:v for k,v in opt.param_groups[0].items() if k!='params'}))


def rng_tuple(arrays,meta):
    require(set(meta)=={'numpy_kind','position','has_gauss','cached_gaussian'} and meta['numpy_kind']=='MT19937' and
        type(meta['position']) is int and 0<=meta['position']<=624 and meta['has_gauss'] in (0,1) and
        math.isfinite(meta['cached_gaussian']),'Invalid saved NumPy RNG metadata')
    a=arrays['numpy_MT19937'];require(a.dtype==np.uint32 and a.shape==(624,),'Invalid NumPy RNG array')
    return (meta['numpy_kind'],a.copy(),meta['position'],meta['has_gauss'],meta['cached_gaussian'])


def install_rng(arrays,meta):
    import torch
    a=arrays['torch_CPU_RNG'];require(a.dtype==np.uint8 and a.shape==tuple(torch.get_rng_state().shape),'Invalid CPU RNG array')
    t=torch.from_numpy(a.copy());torch.Generator(device='cpu').set_state(t)
    state=rng_tuple(arrays,meta);np.random.RandomState(0).set_state(state)  # Local validation; global state untouched.
    torch.set_rng_state(t);np.random.set_state(state)


@contextmanager
def saved_rng_context(arrays,meta):
    """Install admitted common state before legacy limits snapshots it; restore caller on exit."""
    import torch
    old_t=torch.get_rng_state().clone();old_n=np.random.get_state()
    try:install_rng(arrays,meta);yield
    finally:torch.set_rng_state(old_t);np.random.set_state(old_n)


def restore_state(model,opt,arrays,rng_meta,expected_options,expected_identity=None):
    """Strict numeric restoration, never model execution or optimizer stepping."""
    import torch
    ps=dict(model.named_parameters());bs=dict(model.named_buffers())
    wanted={'parameter/'+n for n in ps}|{'buffer/'+n for n in bs}|{
        'adam/'+n+'/'+k for n in ps for k in ('step','exp_avg','exp_avg_sq')}|{'torch_CPU_RNG','numpy_MT19937'}
    require(set(arrays)==wanted and not opt.state and options(opt)==expected_options,'State inventory/fresh optimizer/options mismatch')
    for prefix,items in (('parameter/',ps),('buffer/',bs)):
        for n,t in items.items():
            a=arrays[prefix+n]
            require(t.device.type=='cpu' and t.dtype==torch.float32 and a.dtype==np.float32 and
                a.shape==tuple(t.shape) and np.isfinite(a).all(),'Parameter/buffer shape/dtype/value mismatch: '+n)
    for n,p in ps.items():
        for k in ('step','exp_avg','exp_avg_sq'):
            a=arrays['adam/'+n+'/'+k];shape=() if k=='step' else tuple(p.shape)
            require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Invalid Adam tensor: '+n+'/'+k)
            require(float(a)==START if k=='step' else bool((a>=0).all()) if k=='exp_avg_sq' else True,'Invalid Adam counter/second moment')
    # Validate RNG before touching parameters; validation itself consumes no global random numbers.
    a=arrays['torch_CPU_RNG'];require(a.dtype==np.uint8 and a.shape==tuple(torch.get_rng_state().shape),'Invalid CPU RNG array')
    torch.Generator(device='cpu').set_state(torch.from_numpy(a.copy()));np.random.RandomState(0).set_state(rng_tuple(arrays,rng_meta))
    parameters={n:arrays['parameter/'+n] for n in ps}
    require(expected_identity is None or obs.parameter_identity(parameters)==expected_identity,'Selected checkpoint parameter identity differs')
    with torch.no_grad():
        for prefix,items in (('parameter/',ps),('buffer/',bs)):
            for n,t in items.items():t.copy_(torch.from_numpy(arrays[prefix+n].copy()))
    for n,p in ps.items():opt.state[p]={k:torch.from_numpy(arrays['adam/'+n+'/'+k].copy()) for k in ('step','exp_avg','exp_avg_sq')}
    install_rng(arrays,rng_meta)
    actual,meta=prior.state_arrays(model,opt)
    require(meta==rng_meta and set(actual)==wanted and all(actual[k].dtype==arrays[k].dtype and actual[k].shape==arrays[k].shape and
        actual[k].tobytes()==arrays[k].tobytes() for k in wanted),'Exact complete state restoration failed')
    return dict(exact=True,arrays=len(wanted),parameter_identity=obs.parameter_identity(parameters),Adam_step=START,
                buffers_restored=True,CPU_Torch_and_NumPy_RNG_restored=True,optimizer_effective=options(opt))


class Work:
    def __init__(self):self.value={k:0 for k in CAPS}
    def attempt(self,kind,phase=None):
        k={'forward':'forward_attempts','backward':'backward_attempts','optimizer':'optimizer_attempts'}[kind]
        require(self.value[k]<CAPS[k],'E3 '+kind+' cap reached')
        if kind=='forward':
            require(phase in ('training','terminal','external'),'Invalid E3 forward phase');key=phase+'_forwards'
            require(self.value[key]<CAPS[key],'E3 phase cap reached')
        self.value[k]+=1
        if kind=='forward':self.value[key]+=1
    def done(self,kind):
        k={'forward':'forwards','backward':'backwards','optimizer':'successful_updates'}[kind]
        require(self.value[k]<CAPS[k],'E3 completion cap reached');self.value[k]+=1


def compact_measurement(core,store,prefix,state,check):
    core.opt.zero_grad();core.work.attempt('forward','terminal' if state==END else 'training')
    with obs.GraphObserver(core.model,expected_relus=22 if core.scientific else None) as observer:
        observer.begin();values,material=core.forward(observer.wrap_shader(core.shader),state==END)
        core.work.done('forward');loss=core.loss(values);core.work.attempt('backward')
        groups=core.backward(loss);core.work.done('backward');signals,nodes=observer.collect();nodes=nodes['nodes']
        row=dict(state=state,loss=float(loss.detach()),original_gradient_norm=math.sqrt(sum(g['squared_norm'] for g in groups.values())),
            physical=core.physical(material),parameter_gradient_groups=groups,shader={},heads={},relus={})
        for h in obs.HEADS:
            row['heads'][h]={k:signals[h+'.'+k].tolist() for k in ('logits','sigmoid','logits.loss_gradient','sigmoid.loss_gradient')}
            row['heads'][h]['boundary_channels']=nodes[h+'.sigmoid']['analytic_derivative']['zero']
        for name in ('mx','my','albedo','metallic','encoding'):row['shader'][name]=nodes['shader.'+name]['loss_gradient']
        row['shared_feature_gradient']=nodes['shared_features']['loss_gradient']
        for n,v in nodes.items():
            if n.startswith('relu.') and n.endswith('.post'):row['relus'][n]=dict(v['activity'],gradient_L2=v['loss_gradient']['L2'])
        if state in (START,*SNAPSHOTS):row['signals']=store.numeric(prefix+'/signals-'+str(state)+'.npz',signals,1024**2)
    check();return row


def prepare_arm(core,store,prefix,expected,check=lambda reserve=0:None):
    require(core.restoration['exact'],'Restoration gate missing')
    core.work=Work();row=compact_measurement(core,store,prefix,START,check)
    passed=row['loss']==expected['loss'] and row['original_gradient_norm']==expected['original_gradient_norm'] and row['physical']['physical']==expected['physical']
    store.json(prefix+'/state10-parity.json',dict(restoration=core.restoration,passed=passed,actual=row,expected=expected,
        first_fitting_pass_reused_for_update11=True))
    require(passed,'Exact restored state-10 loss/gradient/physical parity failed')
    core.cached_state10=row
    return row


def continue_arm(core,store,prefix,outcome,check=lambda reserve=0:None,_authorization=None):
    require(not core.scientific or _authorization is _ADMITTED,'Scientific continuation requires admitted E3 review')
    require(core.cached_state10['state']==START,'Admitted cached state-10 measurement required')
    result=outcome;result.update(status='running',completed_additional_updates=0,work=core.work.value,checkpoints={})
    log=store.path(prefix+'/trajectory.jsonl');log.parent.mkdir(parents=True,exist_ok=True);previous=None
    try:
        with log.open('xb') as f:
            for state in range(START,END+1):
                check(0 if state==END else 10)
                row=core.cached_state10 if state==START else compact_measurement(core,store,prefix,state,check)
                row.update(previous_sha256=previous,optimizer_effective=options(core.opt))
                if state<END:
                    before=obs.parameter_arrays(core.model);check(10);core.work.attempt('optimizer');core.step();core.work.done('optimizer')
                    result['completed_additional_updates']=state-START+1
                    after=obs.parameter_arrays(core.model);_,delta=obs.displacement(before,after)
                    row['actual_parameter_movement']=dict(before_identity=delta['before_identity'],after_identity=delta['after_identity'],per_group=delta['per_group'])
                    row['parameter_probes']={}
                    for n,a in before.items():
                        idx=np.linspace(0,a.size-1,min(8,a.size),dtype=np.int64)
                        row['parameter_probes'][n]=dict(indices=idx.tolist(),before=a.ravel()[idx].tolist(),after=after[n].ravel()[idx].tolist())
                    if state+1 in SNAPSHOTS:
                        arrays,rng=prior.state_arrays(core.model,core.opt)
                        result['checkpoints'][str(state+1)]=dict(selection=store.numeric(prefix+'/state-'+str(state+1)+'.npz',arrays,18*1024**2),
                            state=state+1,RNG=rng,optimizer_effective=options(core.opt))
                row['post_state']=min(state+1,END);row['Adam_counter']=core.native_work()['optimizer_returns'];row['work']=dict(core.work.value)
                raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                require(len(raw)<=65536,'Compact trajectory record exceeds frozen 64 KiB cap');store.account(len(raw))
                f.write(raw);f.flush();os.fsync(f.fileno());previous=hashlib.sha256(raw).hexdigest();check()
        core.work.attempt('forward','external');result['endpoint']=core.evaluate();core.work.done('forward');check()
        require(core.work.value==CAPS and core.native_work()['optimizer_returns']==END,'E3 bounded work mismatch')
        result.update(status='complete',native_work=core.native_work(),last_record_sha256=previous,
            trajectory=dict(path=str(log.relative_to(store.root)),sha256=sha(log),bytes=log.stat().st_size,records=91),hooks_removed=True)
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,prior.Censored) else 'failed',failure=type(exc).__name__+': '+str(exc),native_work=core.native_work())
        try:
            arrays,rng=prior.state_arrays(core.model,core.opt)
            result['failure_state']=dict(selection=store.numeric(prefix+'/failure.npz',arrays,18*1024**2,allow_nonfinite=True),RNG=rng)
        except BaseException as second:result['failure_capture_error']=str(second)
        raise
    finally:store.json(prefix+'/arm-result.json',result)
    return result


def accepted_inputs():
    old,_=read_report(ROOT/F2,F2_SHA);r,_=read_report(RAW/'result.json',REPORT_SHA)
    seal,_=read_report(RAW/'finalization/seal.json',SEAL_SHA)
    receipt,_=read_report(RAW.parent/(RAW.name+'.process.json'),RECEIPT_SHA)
    archive,_=read_report(ROOT/EVIDENCE_MANIFEST,EVIDENCE_MANIFEST_SHA)
    inventory={x['path']:x for x in archive['raw_experiment']['files']}
    require(seal['outcome']=='complete' and seal['report']['sha256']==REPORT_SHA and receipt['process_exit_code']==0 and
        r['scientific_work_completed'] and r['cleanup_verified'] and all(v['status']=='passed' for v in r['terminal_integrity'].values()),'E2b completion admission failed')
    identities={str(ROOT/F2):F2_SHA,str(ROOT/EVIDENCE_MANIFEST):EVIDENCE_MANIFEST_SHA,str(RAW/'result.json'):REPORT_SHA,str(RAW/'finalization/seal.json'):SEAL_SHA,
        str(RAW.parent/(RAW.name+'.process.json')):RECEIPT_SHA};arms=[];common=None
    for arm in r['outcomes']:
        require(arm['status']=='complete' and arm['completed_updates']==10,'Incomplete E2b arm')
        selection=arm['final_state'];m,_=read_report(RAW/selection['path'],selection['sha256'])
        identities[str(RAW/selection['path'])]=selection['sha256'];identities[str(RAW/m['archive'])]=m['archive_sha256']
        arrays=obs.read_numeric(RAW,selection)
        catalog=old['initial_models'][arm['material']+'/'+str(arm['seed'])]
        require(len(arrays)==350 and len([n for n in arrays if n.endswith('/step')])==84 and
            all(float(a)==START for n,a in arrays.items() if n.endswith('/step')),'Saved Adam inventory/counter differs')
        wanted={'parameter/'+n for n in catalog['parameters']}|{'buffer/'+n for n in catalog['buffers']}|{
            'adam/'+n+'/'+k for n in catalog['parameters'] for k in ('step','exp_avg','exp_avg_sq')}|{'torch_CPU_RNG','numpy_MT19937'}
        require(set(arrays)==wanted,'Saved state catalog differs from frozen architecture')
        for group,prefix in (('parameters','parameter/'),('buffers','buffer/')):
            for n,entry in catalog[group].items():
                a=arrays[prefix+n]
                require(a.dtype==np.float32 and list(a.shape)==entry['shape'],'Saved parameter/buffer layout differs: '+n)
        for n,entry in catalog['parameters'].items():
            for k in ('step','exp_avg','exp_avg_sq'):
                a=arrays['adam/'+n+'/'+k]
                require(a.dtype==np.float32 and list(a.shape)==([] if k=='step' else entry['shape']) and
                    (bool((a>=0).all()) if k=='exp_avg_sq' else True),'Saved Adam layout/value differs: '+n+'/'+k)
        require(obs.parameter_identity({n[10:]:a for n,a in arrays.items() if n.startswith('parameter/')})==arm['final_parameter_identity'],'Saved model identity differs')
        require(all(hashlib.sha256(arrays['buffer/'+n].tobytes()).hexdigest()==v['sha256'] for n,v in catalog['buffers'].items()),'Calibration buffer identity differs')
        signature=({k:obs.array_record(arrays[k]) for k in ('torch_CPU_RNG','numpy_MT19937')},arm['final_rng'])
        if common is None:common=signature
        require(signature==common,'E2b arms do not share an identical saved global RNG')
        terminal=arm['records'][-1];v,_=read_report(RAW/terminal['path'],terminal['sha256']);identities[str(RAW/terminal['path'])]=terminal['sha256']
        arms.append(dict(material=arm['material'],seed=arm['seed'],lr=arm['lr'],checkpoint=selection,RNG=arm['final_rng'],
            parameter_identity=arm['final_parameter_identity'],parity={k:v[k] for k in ('loss','original_gradient_norm')}|{'physical':v['physical']['physical']}))
    require([(a['material'],a['seed'],a['lr']) for a in arms]==[(m,s,lr) for m,s in CASES for lr in RATES],'Exact inherited matrix differs')
    for path,s in identities.items():
        if str(RAW)+'/' in path:require(inventory[path]['sha256']==s and Path(path).stat().st_size==inventory[path]['bytes'],'Selected input differs from accepted archival manifest')
    return old,arms,identities


def make_plan():
    old,arms,ids=accepted_inputs()
    sources=dict(old['inherited_sha256'],**old['source_sha256']);sources.update({p:sha(ROOT/p) for p in CODE})
    require(all(sha(ROOT/p)==s for p,s in sources.items()),'Scientific/source identity mismatch')
    return dict(schema_version=1,protocol='neural-material-state100-continuation-e3-v1',base_commit=BASE,execution_authorized=False,
        source_sha256=sources,input_sha256=ids,E2b_freeze_sha256=F2_SHA,arms=arms,start=START,end=END,work_per_arm=CAPS,
        total_work={k:10*v for k,v in CAPS.items()},snapshots=list(SNAPSHOTS),global_parity_before_any_update=True,
        runtime=old['runtime'],arithmetic_runtime=old['arithmetic_runtime'],inputs=old['inputs'],tapes=old['tapes'],optimizer=old['optimizer'],
        resources=dict(parity_stage_seconds=120,continuation_seconds_per_arm=120,whole_seconds=1800,kill_grace_seconds=10,RSS_bytes=4*1024**3,address_bytes=8*1024**3),
        storage=dict(output_cap_bytes=1024**3,minimum_free_bytes=2*1024**3,checkpoint_max_bytes=18*1024**2,
            trajectory_line_max_bytes=65536,trajectory_records_per_arm=91,signal_states=[10,20,50,100],
            full_snapshots_per_arm=[20,50,100],probe_indices='8 evenly spaced flattened indices per tensor; no RNG draw',
            limitation='Full per-update changes measured but only summaries/probes retained; do not claim full XOR reconstruction for all 900 transitions'),
        output_parent='/home/isavand/paper/evidence',run_prefix='neural-e3-')


def validate_freeze(selected_sha):
    plan,_=read_report(ROOT/FREEZE,selected_sha)
    require(plan['execution_authorized'] is False and plan['start']==START and plan['end']==END and
        plan['work_per_arm']==CAPS and plan['total_work']=={k:10*v for k,v in CAPS.items()} and
        [(a['material'],a['seed'],a['lr']) for a in plan['arms']]==[(m,s,lr) for m,s in CASES for lr in RATES] and
        all(p in plan['source_sha256'] for p in CODE),'E3 frozen contract differs')
    for p,s in plan['source_sha256'].items():require(sha(ROOT/p)==s,'Frozen source changed: '+p)
    for p,s in plan['input_sha256'].items():require(sha(p)==s,'Accepted E2b input changed: '+p)
    validation,_=read_report(ROOT/VALIDATION,plan['validation_sha256'])
    require(validation['exit_code']==0 and validation['failed']==validation['errors']==validation['skipped']==0 and
        validation['scientific_queries']==0,'E3 engineering validation failed')
    require(validation['new_source_sha256']=={p:plan['source_sha256'][p] for p in CODE},'Engineering source version differs')
    junit=ROOT/validation['JUnit']['path'];require(sha(junit)==validation['JUnit']['sha256'],'Engineering JUnit evidence changed')
    tree=ET.parse(junit);cases=tree.findall('.//testcase')
    require(len(cases)==validation['passed'] and all(not x.findall('failure') and not x.findall('error') and not x.findall('skipped') for x in cases) and
        sorted(x.attrib['name'] for x in cases)==validation['test_inventory'],'Engineering test inventory differs')
    return plan


def validate_review(review,freeze_sha):
    require(set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_commit','reviewed_branch','run_id','output_root','external_watchdog'} and
        review['phase']=='E.3' and review['scientific_execution_authorized'] is True,'Separate explicit E3 execution approval required')
    require(review['freeze_sha256']==freeze_sha and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10),'E3 freeze/watchdog review differs')
    plan=validate_freeze(freeze_sha);validate_run_id(review['run_id'])
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    require(git('rev-parse','HEAD')==review['reviewed_commit'] and git('branch','--show-current')==review['reviewed_branch']=='feat/neural-material-early-step-e2a' and not git('status','--porcelain'),'Exact clean reviewed E3 checkout required')
    path=Path(review['output_root']);require(path.is_absolute() and path.resolve()==path and path.parent==Path(plan['output_parent']) and
        path.name==review['run_id'] and path.name.startswith(plan['run_prefix']) and not path.exists(),'Fresh approved E3 output required')
    require(shutil.disk_usage(path.parent).free>=plan['storage']['minimum_free_bytes'],'Insufficient continuation storage')
    return plan


def execute_reviewed(review_path,review_sha,freeze_sha):
    review,_=read_report(Path(review_path),review_sha);plan=validate_review(review,freeze_sha)
    # No scientific import, construction or query is reachable before approval.
    from experiments import neural_material_recovery_v1 as h
    import platform
    require(dict(python=platform.python_version(),numpy=np.__version__,torch=h.torch.__version__,device=str(h.torch.get_default_device()),torch_threads=h.torch.get_num_threads())==plan['runtime'],'Frozen CPU runtime differs')
    require(h.previous.accepted.arithmetic_runtime()==plan['arithmetic_runtime'],'Arithmetic runtime differs')
    old,_=read_report(ROOT/F2,F2_SHA);reference,_=h.admit_reference();ft,et,tm=h.inputs(reference)
    require(tm==plan['tapes'],'Tape identity differs')
    store=obs.Store(review['output_root'],plan['output_parent'],total_cap=plan['storage']['output_cap_bytes']);store.json('execution-review.json',review)
    result=dict(protocol=plan['protocol'],status='running',freeze_sha256=freeze_sha,review_sha256=review_sha,
        outcomes=[dict(material=x['material'],seed=x['seed'],lr=x['lr'],status='not_started',completed_additional_updates=0) for x in plan['arms']],cleanup_verified=False,
        all_state10_parity_passed=False,scientific_work_completed=False,terminal_integrity_verified=False,evidence_finalized=False)
    common=obs.read_numeric(RAW,plan['arms'][0]['checkpoint']);stage='restoration/parity'
    try:
        with saved_rng_context(common,plan['arms'][0]['RNG']):
            with prior.limits() as (check,begin,end):
                shader=h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==plan['inputs']['shader'],'Shader differs')
                scene=reference['scene'];observations=scene['observations'];fit=h.MaterialBatch(scene,observations[:24],ft);evaluation=h.MaterialBatch(scene,observations,et)
                cores=[];begin()
                for spec in plan['arms']:
                    material,seed,lr=spec['material'],spec['seed'],spec['lr'];prefix=material.lower()+'-'+str(seed)+'-'+str(lr)
                    outcome=next(x for x in result['outcomes'] if (x['material'],x['seed'],x['lr'])==(material,seed,lr));outcome['status']='restoration/parity'
                    result['active_input']=spec
                    arrays=obs.read_numeric(RAW,spec['checkpoint'])
                    class Core:
                        scientific=True
                        def __init__(self):
                            self.model=h.field.make_material('N1',h.torch.from_numpy(arrays['buffer/initial_latent'].copy()),seed)
                            self.shader=shader;self.opt=h.optimizer_for(self.model)
                            for group in self.opt.param_groups:group['lr']=lr
                            self.opt.defaults['lr']=lr;self.c=h.counts('N1');self.c.update(optimizer_attempts=10,optimizer_returns=10)
                            self.truth=h.previous.reference.truth(material);self.fine,self.target=h.previous.material_target(next(x for x in reference['materials'] if x['material']==material),material)
                            require(h.previous.oldref.sha_array(self.target.numpy())==plan['inputs']['targets'][material]['fit_float32_sha256'] and
                                h.previous.oldref.sha_array(self.fine)==plan['inputs']['targets'][material]['fine_sha256'],'Original target differs')
                            identity=h.field.identity(self.model);expected=old['initial_models'][material+'/'+str(seed)]
                            require(identity['source']==expected['source'] and identity['configuration']==expected['configuration'],'Original N1 source/config differs')
                            self.restoration=restore_state(self.model,self.opt,arrays,spec['RNG'],dict(plan['optimizer'],lr=lr),spec['parameter_identity'])
                        def forward(self,view,terminal):
                            v,m,_=fit.render(self.model,view,self.c,phase='terminal' if terminal else 'update');return v,m
                        def loss(self,values):return h.loss_value(values,self.target)
                        def backward(self,loss):return h.backward(self.model,loss,self.c)
                        def step(self):h.step(self.model,self.opt,self.c)
                        def native_work(self):return dict(self.c)
                        def physical(self,m):return dict(physical=m.physical[0].detach().tolist(),encoding=m.encoding.detach().tolist(),
                            centered_radius=m.centered_radius.detach().tolist(),errors=h.previous.recovery_errors(m.physical[0].detach().numpy(),self.truth))
                        def evaluate(self):return h.evaluate(self.model,evaluation,shader,self.c,self.truth,self.fine,observations)
                    core=Core();prepare_arm(core,store,prefix,spec['parity'],check);outcome['status']='parity_verified';cores.append((core,prefix,outcome));check()
                end();result['all_state10_parity_passed']=True;stage='continuation'
                require(len({id(p) for core,_,_ in cores for p in core.model.parameters()})==840,'Unexpected cross-arm parameter sharing')
                for core,prefix,outcome in cores:
                    begin();continue_arm(core,store,prefix,outcome,check,_ADMITTED);end();h.verify_tapes(ft,et,plan['tapes'])
                require(all(x['status']=='complete' and x['work']==CAPS for x in result['outcomes']),'Incomplete continuation matrix')
            # Legacy runtime has now checked saved common RNG and restored resources.
        result['cleanup_verified']=True;stage='terminal_integrity';validate_freeze(freeze_sha);h.verify_tapes(ft,et,plan['tapes'])
        expected_native=dict(field_attempts=92,field_completed=92,calibration_forwards=1,shader_attempts=92,shader_completed=92,
            fit_forwards=91,evaluation_forwards=1,attempted_points=9072,completed_points=9072,diffuse_directions=9072*512,
            specular_directions=9072*1024,reverse_attempts=91,reverse_completed=91,optimizer_attempts=100,optimizer_returns=100,
            initial_forwards=0,prefix_forwards=0,terminal_forwards=1)
        require(all(x['native_work']==expected_native for x in result['outcomes']),'Native continuation counts differ')
        result.update(status='work_and_cleanup_verified',scientific_work_completed=True,terminal_integrity_verified=True,
            additional_totals={k:sum(x['work'][k] for x in result['outcomes']) for k in CAPS})
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,prior.Censored) else 'failed',failure_stage=stage,failure=type(exc).__name__+': '+str(exc));raise
    finally:
        selected=store.json('result.json',result)
        seal=store.json('finalization/seal.json',dict(report=selected,outcome='complete' if result['scientific_work_completed'] else result['status'],
            cleanup_verified=result['cleanup_verified'],successful_process_exit_also_required=True))
        verified,_=read_report(store.root/seal['path'],seal['sha256'])
        require(verified['report']==selected,'Finalization seal readback differs')
    result.update(evidence_finalized=True,seal=seal,status='complete');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--verify-sha256');a=p.parse_args()
    if a.prepare:
        plan=make_plan();validation=json.loads((ROOT/VALIDATION).read_text())
        require(validation['exit_code']==0 and validation['scientific_queries']==0 and validation['new_source_sha256']=={p:plan['source_sha256'][p] for p in CODE},'Preparation tests/source mismatch')
        plan['validation_sha256']=sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f:json.dump(plan,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(freeze=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)))
    elif a.verify_sha256:print(json.dumps(dict(verified=bool(validate_freeze(a.verify_sha256)),execution_authorized=False)))
    else:print(json.dumps(dict(start=10,end=100,arms=10,additional_optimizer_updates=900,forwards=920,backwards=910,execution_authorized=False)))


if __name__=='__main__':main()
