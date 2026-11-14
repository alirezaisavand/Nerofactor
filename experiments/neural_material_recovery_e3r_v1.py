"""Four unfinished E3 arms only. Preparation CLI; separate execution approval required."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import FunctionType

import numpy as np
from experiments import neural_material_continuation_e3 as old
from infrastructure import material_diagnostics_e3r_v1 as diag
from infrastructure import neural_material_observer_e2a as obs
from infrastructure.signatures import read_report,require
from infrastructure.runs import validate_run_id

ROOT=old.ROOT
BASE='6ce6afc22ce49ce4553bd95bc1c224a8d7cbbba7'
RAW=Path('/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae')
PARENT_FREEZE='ae258c5f6b7d2b0015f071f01c4bbb53c164f19bc2df1e412bb5752c2b98c606'
REPORT='79a876fffd11f3e5d44c637a2ca4b9f1aa05ebd6a0f2484744a7403ecbe2a490'
SEAL='e7cf6b4673652cd912e3ba8547c27106399a3b99b0e2b9eeb22fa8b622f26be3'
RECEIPT='281aac5e08d05863aa64fb14e090019e3010929736c9c3017c5ed57c4675216c'
INVENTORY='docs/results/neural_material_continuation_e3/evidence_manifest.json'
INVENTORY_SHA='a8fc588210bd6842d5798d000783e14a6adbb6c205ee332446655af53c7616c4'
FREEZE='docs/protocols/NEURAL_MATERIAL_RECOVERY_E3R_V1.freeze.json'
VALIDATION='docs/evidence/NEURAL_MATERIAL_RECOVERY_E3R_PREPARATION_V1.json'
CODE=('infrastructure/material_diagnostics_e3r_v1.py','experiments/neural_material_recovery_e3r_v1.py',
      'tests/scientific/test_neural_material_recovery_e3r_v1.py','docs/protocols/NEURAL_MATERIAL_RECOVERY_E3R_V1.md')
MATRIX=(('M3',101,.01,21),('M3',101,.001,10),('M3',103,.01,10),('M3',103,.001,10))
_ADMITTED=object()
sha=old.sha


def caps(start):
    require(type(start) is int and start in (10,21),'Only explicitly admitted states 10/21')
    n=100-start
    return dict(forward_attempts=n+2,forwards=n+2,backward_attempts=n+1,backwards=n+1,
        optimizer_attempts=n,successful_updates=n,training_forwards=n,terminal_forwards=1,external_forwards=1)


class Work:
    def __init__(self,start):self.caps=caps(start);self.value={k:0 for k in self.caps}
    def attempt(self,kind,phase=None):
        key={'forward':'forward_attempts','backward':'backward_attempts','optimizer':'optimizer_attempts'}[kind]
        require(self.value[key]<self.caps[key],'E3R '+kind+' cap reached')
        if kind=='forward':
            require(phase in ('training','terminal','external'),'Invalid phase');p=phase+'_forwards'
            require(self.value[p]<self.caps[p],'E3R phase cap reached')
        self.value[key]+=1
        if kind=='forward':self.value[p]+=1
    def done(self,kind):
        key={'forward':'forwards','backward':'backwards','optimizer':'successful_updates'}[kind]
        require(self.value[key]<self.caps[key],'E3R completion cap reached');self.value[key]+=1


def restore_state(model,opt,arrays,rng,options,identity,start):
    """Reuse the exact frozen strict restorer with a private expected-counter binding."""
    caps(start)
    fn=FunctionType(old.restore_state.__code__,dict(old.restore_state.__globals__,START=start),
        'restore_selected_state',old.restore_state.__defaults__,old.restore_state.__closure__)
    proof=fn(model,opt,arrays,rng,options,identity)
    require(old.START==10 and proof['Adam_step']==start,'Frozen restorer globals changed')
    return proof


def measurement(core,store,prefix,state,check):
    core.opt.zero_grad();core.work.attempt('forward','terminal' if state==100 else 'training')
    with obs.GraphObserver(core.model,expected_relus=22 if core.scientific else None) as observer:
        observer.begin();values,material=core.forward(observer.wrap_shader(core.shader),state==100);core.work.done('forward')
        loss=core.loss(values);core.work.attempt('backward');groups=core.backward(loss);core.work.done('backward')
        signals,nodes=observer.collect();nodes=nodes['nodes']
        # Persist selected raw signals BEFORE physical-domain diagnostics can reject.
        selected=store.numeric(prefix+'/signals-'+str(state)+'.npz',signals,1024**2) if state in (core.start,50,100) else None
        raw=core.raw_physical(material)
        try:physical=core.physical(material)
        except BaseException:
            store.json(prefix+'/rejected-physical-'+str(state)+'.json',dict(state=state,raw_physical=raw,signals=selected));raise
        row=dict(state=state,loss=float(loss.detach()),original_gradient_norm=math.sqrt(sum(g['squared_norm'] for g in groups.values())),
            physical=physical,parameter_gradient_groups=groups,shader={},heads={},relus={},diagnostic_protocol=diag.PROTOCOL)
        for h in obs.HEADS:
            row['heads'][h]={k:signals[h+'.'+k].tolist() for k in ('logits','sigmoid','logits.loss_gradient','sigmoid.loss_gradient')}
            row['heads'][h]['boundary_channels']=nodes[h+'.sigmoid']['analytic_derivative']['zero']
        for name in ('mx','my','albedo','metallic','encoding'):row['shader'][name]=nodes['shader.'+name]['loss_gradient']
        row['shared_feature_gradient']=nodes['shared_features']['loss_gradient']
        for n,v in nodes.items():
            if n.startswith('relu.') and n.endswith('.post'):row['relus'][n]=dict(v['activity'],gradient_L2=v['loss_gradient']['L2'])
        if selected:row['signals']=selected
    check();return row


def prepare_arm(core,store,prefix,expected,check=lambda reserve=0:None):
    require(core.restoration['exact'] and core.restoration['Adam_step']==core.start,'Exact selected-state restoration required')
    row=measurement(core,store,prefix,core.start,check)
    if core.start==10:
        passed=row['loss']==expected['loss'] and row['original_gradient_norm']==expected['original_gradient_norm'] and row['physical']['physical']==expected['physical']
        label='exact_E2b_state10_parity'
    else:
        p=np.asarray(row['physical']['physical'],np.float32)
        passed=bool(np.any(p[:2]==np.float32(.04)) and np.isfinite(p).all())
        label='new_state21_measurement_confirms_float32_floor_not_historical_parity'
    store.json(prefix+'/restoration-measurement.json',dict(restoration=core.restoration,passed=passed,check=label,
        actual=row,expected=expected,first_pass_reused_for_next_update=True))
    require(passed,'Restored-state parity/failure-mechanism gate failed')
    core.cached=row
    return row


def continue_arm(core,store,prefix,result,check=lambda reserve=0:None,_authorization=None):
    require(not core.scientific or _authorization is _ADMITTED,'Separate E3R authorization required')
    require(core.cached['state']==core.start,'Restoration measurement required')
    result.update(status='running',completed_additional_updates=0,work=core.work.value,checkpoints={})
    path=store.path(prefix+'/trajectory.jsonl');previous=None
    try:
        with path.open('xb') as f:
            for state in range(core.start,101):
                check(0 if state==100 else 10)
                row=core.cached if state==core.start else measurement(core,store,prefix,state,check)
                row.update(previous_sha256=previous,optimizer_effective=old.options(core.opt))
                if state<100:
                    before=obs.parameter_arrays(core.model);check(10);core.work.attempt('optimizer');core.step();core.work.done('optimizer')
                    result['completed_additional_updates']=state-core.start+1
                    after=obs.parameter_arrays(core.model);_,delta=obs.displacement(before,after)
                    row['actual_parameter_movement']={k:delta[k] for k in ('before_identity','after_identity','per_group')}
                    row['parameter_probes']={}
                    for n,a in before.items():
                        ix=np.linspace(0,a.size-1,min(8,a.size),dtype=np.int64)
                        row['parameter_probes'][n]=dict(indices=ix.tolist(),before=a.ravel()[ix].tolist(),after=after[n].ravel()[ix].tolist())
                    if state+1 in (50,100):
                        arrays,rng=old.prior.state_arrays(core.model,core.opt)
                        result['checkpoints'][str(state+1)]=dict(state=state+1,selection=store.numeric(prefix+'/state-'+str(state+1)+'.npz',arrays,18*1024**2),
                            RNG=rng,optimizer_effective=old.options(core.opt))
                row.update(post_state=min(state+1,100),Adam_counter=core.native_work()['optimizer_returns'],work=dict(core.work.value))
                data=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                require(len(data)<=65536,'Trajectory record cap');store.account(len(data));f.write(data);f.flush();os.fsync(f.fileno())
                previous=hashlib.sha256(data).hexdigest();check()
        final=obs.parameter_identity(obs.parameter_arrays(core.model))
        core.work.attempt('forward','external');result['endpoint']=core.evaluate();core.work.done('forward');check()
        require(obs.parameter_identity(obs.parameter_arrays(core.model))==final,'Evaluation changed weights')
        require(core.work.value==core.work.caps and core.native_work()['optimizer_returns']==100,'Recovery work mismatch')
        result.update(status='complete',native_work=core.native_work(),last_record_sha256=previous,
            trajectory=dict(path=str(path.relative_to(store.root)),sha256=sha(path),records=101-core.start),hooks_removed=True)
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,old.prior.Censored) else 'failed',failure=type(exc).__name__+': '+str(exc),native_work=core.native_work())
        try:
            arrays,rng=old.prior.state_arrays(core.model,core.opt)
            result['failure_state']=dict(selection=store.numeric(prefix+'/failure.npz',arrays,18*1024**2,allow_nonfinite=True),RNG=rng)
        except BaseException as other:result['failure_capture_error']=str(other)
        raise
    finally:store.json(prefix+'/arm-result.json',result)
    return result


def admitted_inputs():
    parent=old.validate_freeze(PARENT_FREEZE)
    r,_=read_report(RAW/'result.json',REPORT);seal,_=read_report(RAW/'finalization/seal.json',SEAL)
    receipt,_=read_report(RAW.parent/(RAW.name+'.process.json'),RECEIPT)
    inventory,_=read_report(ROOT/INVENTORY,INVENTORY_SHA)
    require(r['status']=='failed' and r['all_state10_parity_passed'] and seal['outcome']=='failed' and seal['report']['sha256']==REPORT and
        receipt['process_exit_code']==1,'Interrupted parent evidence changed')
    ids=dict(parent['input_sha256']);ids.update({str(ROOT/old.FREEZE):PARENT_FREEZE,str(ROOT/INVENTORY):INVENTORY_SHA})
    for item in inventory['raw_files']+inventory['sidecars']:
        p=Path(item['path']);require(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'Original E3 artifact changed: '+str(p));ids[str(p)]=item['sha256']
    require(sum(a['status']=='complete' and a['completed_additional_updates']==90 for a in r['outcomes'])==6,'Six completed arms must remain unchanged')
    arms=[];common=None
    for m,s,lr,start in MATRIX:
        a=next(x for x in r['outcomes'] if (x['material'],x['seed'],x['lr'])==(m,s,lr))
        original=next(x for x in parent['arms'] if (x['material'],x['seed'],x['lr'])==(m,s,lr))
        selected=a['failure_state'] if start==21 else dict(selection=original['checkpoint'],RNG=original['RNG'])
        input_root=RAW if start==21 else old.RAW;arrays=obs.read_numeric(input_root,selected['selection'])
        identity=obs.parameter_identity({n[10:]:v for n,v in arrays.items() if n.startswith('parameter/')})
        require(len(arrays)==350 and all(float(v)==start for n,v in arrays.items() if n.endswith('/step')),'Selected state inventory/counter differs')
        if start==21:
            require(a['status']=='failed' and a['native_work']['optimizer_returns']==21,'Explicit failed state21 required')
            last=json.loads((RAW/'m3-101-0.01/trajectory.jsonl').read_bytes().splitlines()[-1])
            require(last['post_state']==21 and last['actual_parameter_movement']['after_identity']==identity,'Failure state not attributable to saved update21')
        else:require(a['status']=='parity_verified' and a['completed_additional_updates']==0 and identity==original['parameter_identity'],'Only uncontinued state10 arms permitted')
        signature=({n:obs.array_record(arrays[n]) for n in ('torch_CPU_RNG','numpy_MT19937')},selected['RNG'])
        if common is None:common=signature
        require(common==signature,'Restored common RNG differs')
        arms.append(dict(material=m,seed=s,lr=lr,start=start,end=100,input_root=str(input_root),checkpoint=selected['selection'],RNG=selected['RNG'],
            parameter_identity=identity,parity=original['parity'] if start==10 else None,work=caps(start)))
    return parent,arms,ids


def make_plan():
    parent,arms,ids=admitted_inputs();sources=dict(parent['source_sha256']);sources.update({p:sha(ROOT/p) for p in CODE})
    require(all(sha(ROOT/p)==h for p,h in sources.items()),'Frozen source changed')
    return dict(schema_version=1,protocol='neural-material-e3-recovery-v1',base_commit=BASE,execution_authorized=False,
        source_sha256=sources,input_sha256=ids,parent_freeze_sha256=PARENT_FREEZE,parent_run=str(RAW),arms=arms,
        total_work={k:sum(a['work'][k] for a in arms) for k in caps(10)},
        runtime=parent['runtime'],arithmetic_runtime=parent['arithmetic_runtime'],optimizer=parent['optimizer'],inputs=parent['inputs'],tapes=parent['tapes'],
        resources=parent['resources'],storage=dict(output_cap_bytes=1024**3,minimum_free_bytes=2*1024**3),
        output_parent=parent['output_parent'],run_prefix='neural-e3r-',diagnostic_protocol=diag.PROTOCOL,
        six_completed_arms_not_executed=True,global_restoration_gate_before_updates=True)


def validate_freeze(selected):
    p,_=read_report(ROOT/FREEZE,selected)
    require(p['execution_authorized'] is False and [(a['material'],a['seed'],a['lr'],a['start']) for a in p['arms']]==list(MATRIX) and
        all(a['work']==caps(a['start']) for a in p['arms']) and p['total_work']=={k:sum(caps(s)[k] for _,_,_,s in MATRIX) for k in caps(10)},'Recovery matrix/work changed')
    for name,h in p['source_sha256'].items():require(sha(ROOT/name)==h,'Frozen source changed: '+name)
    for name,h in p['input_sha256'].items():require(sha(name)==h,'Accepted input changed: '+name)
    v,_=read_report(ROOT/VALIDATION,p['validation_sha256'])
    require(v['exit_code']==0 and v['failed']==v['errors']==v['skipped']==0 and v['scientific_queries']==0 and
        v['new_source_sha256']=={n:p['source_sha256'][n] for n in CODE},'Recovery engineering validation failed')
    require(sha(ROOT/v['JUnit']['path'])==v['JUnit']['sha256'],'JUnit evidence changed')
    return p


def validate_review(review,freeze):
    require(set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_commit','reviewed_branch','run_id','output_root','external_watchdog'} and
        review['phase']=='E.3R' and review['scientific_execution_authorized'] is True,'Separate explicit E3 recovery approval required')
    p=validate_freeze(freeze);git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    require(review['freeze_sha256']==freeze and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10),'Reviewed freeze/watchdog mismatch')
    require(git('rev-parse','HEAD')==review['reviewed_commit'] and git('branch','--show-current')==review['reviewed_branch']=='feat/neural-material-early-step-e2a' and not git('status','--porcelain'),'Clean reviewed recovery checkout required')
    validate_run_id(review['run_id']);path=Path(review['output_root'])
    require(path.is_absolute() and path.resolve()==path and path.parent==Path(p['output_parent']) and path.name==review['run_id'] and
        path.name.startswith(p['run_prefix']) and not path.exists(),'Fresh explicit recovery output required')
    require(shutil.disk_usage(path.parent).free>=p['storage']['minimum_free_bytes'],'Insufficient output storage')
    return p


def execute_reviewed(review_path,review_sha,freeze_sha):
    review,_=read_report(Path(review_path),review_sha);plan=validate_review(review,freeze_sha)
    from experiments import neural_material_recovery_v1 as h
    import platform
    require(dict(python=platform.python_version(),numpy=np.__version__,torch=h.torch.__version__,device=str(h.torch.get_default_device()),torch_threads=h.torch.get_num_threads())==plan['runtime'] and
        h.previous.accepted.arithmetic_runtime()==plan['arithmetic_runtime'],'Frozen CPU runtime differs')
    reference,_=h.admit_reference();ft,et,tm=h.inputs(reference);require(tm==plan['tapes'],'Tape identity differs')
    evaluate=diag.evaluation_adapter(h.evaluate)
    store=obs.Store(review['output_root'],plan['output_parent'],total_cap=plan['storage']['output_cap_bytes']);store.json('execution-review.json',review)
    result=dict(protocol=plan['protocol'],status='running',freeze_sha256=freeze_sha,parent_run=str(RAW),outcomes=[],cleanup_verified=False,
        scientific_work_completed=False,terminal_integrity_verified=False,evidence_finalized=False,all_restoration_gates_passed=False)
    for a in plan['arms']:result['outcomes'].append({k:a[k] for k in ('material','seed','lr','start','end')}|dict(status='not_started',completed_additional_updates=0))
    common=obs.read_numeric(plan['arms'][0]['input_root'],plan['arms'][0]['checkpoint']);stage='restoration'
    try:
        with old.saved_rng_context(common,plan['arms'][0]['RNG']):
            with old.prior.limits() as (check,begin,end):
                shader=h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==plan['inputs']['shader'],'Shader differs')
                observations=reference['scene']['observations'];fit=h.MaterialBatch(reference['scene'],observations[:24],ft);evaluation=h.MaterialBatch(reference['scene'],observations,et)
                oldf,_=read_report(ROOT/old.F2,old.F2_SHA);cores=[];begin()
                for spec,outcome in zip(plan['arms'],result['outcomes']):
                    m,s,lr,start=(spec[k] for k in ('material','seed','lr','start'));prefix=m.lower()+'-'+str(s)+'-'+str(lr)
                    outcome['status']='restoration';arrays=obs.read_numeric(spec['input_root'],spec['checkpoint'])
                    class Core:
                        scientific=True
                        def __init__(self):
                            self.start=start;self.model=h.field.make_material('N1',h.torch.from_numpy(arrays['buffer/initial_latent'].copy()),s)
                            self.shader=shader;self.opt=h.optimizer_for(self.model)
                            for group in self.opt.param_groups:group['lr']=lr
                            self.opt.defaults['lr']=lr;self.c=h.counts('N1');self.c.update(optimizer_attempts=start,optimizer_returns=start)
                            self.truth=h.previous.reference.truth(m);self.fine,self.target=h.previous.material_target(next(x for x in reference['materials'] if x['material']==m),m)
                            require(h.previous.oldref.sha_array(self.target.numpy())==plan['inputs']['targets'][m]['fit_float32_sha256'] and h.previous.oldref.sha_array(self.fine)==plan['inputs']['targets'][m]['fine_sha256'],'Target changed')
                            identity=h.field.identity(self.model);expected=oldf['initial_models'][m+'/'+str(s)]
                            require(identity['source']==expected['source'] and identity['configuration']==expected['configuration'],'N1 source/config changed')
                            self.restoration=restore_state(self.model,self.opt,arrays,spec['RNG'],dict(plan['optimizer'],lr=lr),spec['parameter_identity'],start)
                            self.work=Work(start)
                        def forward(self,view,terminal):
                            v,mat,_=fit.render(self.model,view,self.c,phase='terminal' if terminal else 'update');return v,mat
                        def loss(self,v):return h.loss_value(v,self.target)
                        def backward(self,loss):return h.backward(self.model,loss,self.c)
                        def step(self):h.step(self.model,self.opt,self.c)
                        def native_work(self):return dict(self.c)
                        def raw_physical(self,mat):return mat.physical[0].detach().tolist()
                        def physical(self,mat):return dict(physical=self.raw_physical(mat),encoding=mat.encoding.detach().tolist(),centered_radius=mat.centered_radius.detach().tolist(),
                            errors=diag.recovery_errors(mat.physical[0].detach().numpy(),self.truth))
                        def evaluate(self):return evaluate(self.model,evaluation,shader,self.c,self.truth,self.fine,observations)
                    core=Core();outcome['work']=core.work.value;cores.append((core,prefix,outcome))
                    try:prepare_arm(core,store,prefix,spec['parity'],check)
                    finally:outcome['native_work']=core.native_work()
                    outcome['status']='restoration_verified';check()
                end();result['all_restoration_gates_passed']=True;stage='continuation'
                for core,prefix,outcome in cores:
                    begin();continue_arm(core,store,prefix,outcome,check,_ADMITTED);end();h.verify_tapes(ft,et,plan['tapes'])
                require(all(x['status']=='complete' and x['work']==caps(x['start']) for x in result['outcomes']),'Recovery incomplete')
        result['cleanup_verified']=True;stage='terminal_integrity';validate_freeze(freeze_sha);h.verify_tapes(ft,et,plan['tapes'])
        for a in result['outcomes']:
            n=100-a['start'];points=(n+1)*96+336
            expected=dict(field_attempts=n+2,field_completed=n+2,calibration_forwards=1,shader_attempts=n+2,shader_completed=n+2,
                fit_forwards=n+1,evaluation_forwards=1,attempted_points=points,completed_points=points,diffuse_directions=points*512,
                specular_directions=points*1024,reverse_attempts=n+1,reverse_completed=n+1,optimizer_attempts=100,optimizer_returns=100,
                initial_forwards=0,prefix_forwards=0,terminal_forwards=1)
            require(a['native_work']==expected,'Native recovered counters differ')
        totals={k:sum(a['work'][k] for a in result['outcomes']) for k in caps(10)};require(totals==plan['total_work'],'Recovery aggregate counters differ')
        result.update(status='work_and_cleanup_verified',scientific_work_completed=True,terminal_integrity_verified=True,additional_totals=totals)
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,old.prior.Censored) else 'failed',failure_stage=stage,failure=type(exc).__name__+': '+str(exc));raise
    finally:
        selected=store.json('result.json',result);seal=store.json('finalization/seal.json',dict(report=selected,outcome='complete' if result['scientific_work_completed'] else result['status'],successful_process_exit_also_required=True))
        verified,_=read_report(store.root/seal['path'],seal['sha256']);require(verified['report']==selected,'Seal readback differs')
    result.update(status='complete',evidence_finalized=True,seal=seal);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--verify-sha256');args=p.parse_args()
    if args.prepare:
        plan=make_plan();v=json.loads((ROOT/VALIDATION).read_text());require(v['exit_code']==0 and v['scientific_queries']==0 and
            v['new_source_sha256']=={p:plan['source_sha256'][p] for p in CODE},'Preparation source/test mismatch');plan['validation_sha256']=sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f:json.dump(plan,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(freeze=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)))
    elif args.verify_sha256:print(json.dumps(dict(verified=bool(validate_freeze(args.verify_sha256)),execution_authorized=False)))
    else:print(json.dumps(dict(arms=4,updates=349,forwards=357,backwards=353,calibrations=4,execution_authorized=False)))


if __name__=='__main__':main()
