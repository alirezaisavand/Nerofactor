"""E4 AS1 five-case matched pilot. CLI preparation only; scientific execution gated."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from experiments import neural_material_recovery_e3r_v1 as prior
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostics
from infrastructure.signatures import require,read_report
from infrastructure.runs import validate_run_id

ROOT=prior.ROOT;BASE='f8cde24889fc44d0d2e25a08e625907662f08b87'
E2B=prior.old.RAW
RECOVERY_FREEZE='ff4dc8f7b6b3712731a584681770980ca9ee9f46c9352e5ea1de3f37a299f743'
PANEL='docs/results/neural_material_e3_complete_v1/comparison.json'
FREEZE='docs/protocols/MATERIAL_PARAMETERIZATION_E4_V1.freeze.json'
VALIDATION='docs/evidence/MATERIAL_PARAMETERIZATION_E4_PREPARATION_V1.json'
CODE=('experiments/material_chart_e4_v1.py','experiments/material_parameterization_e4_v1.py',
      'tests/scientific/test_material_parameterization_e4_v1.py','docs/protocols/MATERIAL_PARAMETERIZATION_E4_V1.md')
CASES=(('M1',101),('M1',103),('M2',103),('M3',101),('M3',103))
sha=prior.sha


def schedule(material):
    """Exactly match each historical low-LR arm's shader-query schedule, without rerunning it."""
    require(material in ('M1','M2','M3'),'Unknown material')
    result=[('fit',s,True) for s in range(10)]+[('fit',10,False),('evaluate',10,False)]
    if material=='M3':result.append(('fit',10,False))  # Recorded unused parity pass in interrupted E3.
    return result+[('fit',s,True) for s in range(10,100)]+[('fit',100,False),('evaluate',100,False)]


def caps(material):
    n=104+int(material=='M3')
    return dict(forwards=n,backwards=n-2,updates=100,evaluations=2)


class Work:
    def __init__(self,material):self.limit=caps(material);self.attempts={k:0 for k in self.limit};self.done={k:0 for k in self.limit}
    def attempt(self,kind):
        require(self.attempts[kind]<self.limit[kind],'E4 '+kind+' cap')
        if kind=='evaluations':require(self.attempts['forwards']<self.limit['forwards'],'E4 evaluation-forward cap')
        self.attempts[kind]+=1
        if kind=='evaluations':self.attempts['forwards']+=1
    def complete(self,kind):
        require(self.done[kind]<self.attempts[kind],'Unattempted E4 completion');self.done[kind]+=1
        if kind=='evaluations':self.done['forwards']+=1


def run_schedule(core,record,check=lambda reserve=0:None):
    """Dependency-injected engineering driver; initial fitting pass is already cached."""
    for index,(kind,state,update) in enumerate(schedule(core.material)):
        check(0 if state==100 else 10)
        if kind=='fit':
            row=core.cached if index==0 else core.measure(state)
            require(row['state']==state,'Fitting state mismatch')
            if update:
                before=obs.parameter_arrays(core.model) if core.scientific else None
                core.work.attempt('updates');core.step();core.work.complete('updates')
                if core.scientific:
                    after=obs.parameter_arrays(core.model);_,d=obs.displacement(before,after)
                    row['actual_parameter_movement']={k:d[k] for k in ('before_identity','after_identity','per_group')}
                row['post_state']=state+1
            else:row['post_state']=state
            record(dict(row,kind=kind,update=update,index=index))
        else:
            core.work.attempt('evaluations');endpoint=core.evaluate(state);core.work.complete('evaluations')
            core.endpoints[str(state)]=endpoint;record(dict(kind=kind,state=state,index=index,endpoint=endpoint))
        check()
    require(core.work.done==core.work.limit and core.work.attempts==core.work.limit,'E4 work incomplete')


def admitted_inputs():
    parent=prior.validate_freeze(RECOVERY_FREEZE)
    # Pin the accepted complete panel and its recovery receipt/seal; no old experiments repeated.
    panel_bytes=subprocess.check_output(['git','show',BASE+':'+PANEL],cwd=ROOT)
    require((ROOT/PANEL).read_bytes()==panel_bytes,'Accepted panel changed')
    panel=json.loads(panel_bytes);require(len(panel['endpoints'])==10,'Complete E3 panel required')
    manifest_path=ROOT/'docs/results/neural_material_e3_complete_v1/evidence_manifest.json'
    require(manifest_path.read_bytes()==subprocess.check_output(['git','show',BASE+':'+str(manifest_path.relative_to(ROOT))],cwd=ROOT),'Accepted panel inventory changed')
    manifest=json.loads(manifest_path.read_text());ids=dict(parent['input_sha256'])
    for item in manifest['new_raw_files']+manifest['new_sidecars']:
        p=Path(item['path']);require(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'Accepted recovered artifact changed');ids[str(p)]=item['sha256']
    ids[str(ROOT/PANEL)]=sha(ROOT/PANEL);ids[str(manifest_path)]=sha(manifest_path)
    ids[str(ROOT/prior.FREEZE)]=RECOVERY_FREEZE
    e2b,_=read_report(E2B/'result.json',prior.old.REPORT_SHA);arms=[];common_rng=None
    for material,seed in CASES:
        a=next(a for a in e2b['outcomes'] if (a['material'],a['seed'],a['lr'])==(material,seed,.001))
        selection=a['initial_state'];arrays=obs.read_numeric(E2B,selection)
        initial,_=read_report(E2B/a['records'][0]['path'],a['records'][0]['sha256'])
        signals=obs.read_numeric(E2B,initial['signals'])
        require(len(arrays)==98 and len([n for n in arrays if n.startswith('parameter/')])==84 and
            not any(n.startswith('adam/') for n in arrays),'Exact state0 weights/buffers required, no trained initialization')
        require(obs.parameter_identity({n[10:]:v for n,v in arrays.items() if n.startswith('parameter/')})==a['initial_parameter_identity'],'State0 parameter identity changed')
        signature=({n:obs.array_record(arrays[n]) for n in ('torch_CPU_RNG','numpy_MT19937')},a['initial_rng'])
        if common_rng is None:common_rng=signature
        require(signature==common_rng,'Saved initial global RNG differs across cases')
        for selected in (selection,a['records'][0],initial['signals']):
            p=E2B/selected['path'];require(sha(p)==selected['sha256'],'Initial capture hash mismatch');ids[str(p)]=selected['sha256']
        for selected in (selection,initial['signals']):
            m,_=read_report(E2B/selected['path'],selected['sha256']);ids[str(E2B/m['archive'])]=m['archive_sha256']
        require(signals['shader.encoding'].shape==(96,2),'Initial shader encoding capture missing')
        arms.append(dict(material=material,seed=seed,lr=.001,start=0,end=100,checkpoint=selection,RNG=a['initial_rng'],
            initial_record=a['records'][0],signals=initial['signals'],parameter_identity=a['initial_parameter_identity'],
            expected_loss=initial['loss'],expected_physical=initial['physical']['physical'],expected_groups=initial['original_gradient_groups'],
            work=caps(material),schedule=[list(x) for x in schedule(material)]))
    return parent,arms,ids


def make_plan():
    parent,arms,ids=admitted_inputs();sources=dict(parent['source_sha256']);sources.update({n:sha(ROOT/n) for n in CODE})
    require(all(sha(ROOT/n)==h for n,h in sources.items()),'Frozen source changed')
    return dict(schema_version=1,protocol='material-parameterization-e4-as1-v1',base_commit=BASE,execution_authorized=False,
        source_sha256=sources,input_sha256=ids,arms=arms,total_work={k:sum(a['work'][k] for a in arms) for k in caps('M1')},
        runtime=parent['runtime'],arithmetic_runtime=parent['arithmetic_runtime'],inputs=parent['inputs'],tapes=parent['tapes'],optimizer=dict(parent['optimizer'],lr=.001),
        resources=parent['resources'],storage=parent['storage'],output_parent=parent['output_parent'],run_prefix='material-e4-',
        initial_gate=dict(shader_inputs='bitwise equal to saved state0',physical='bitwise equal',loss='exact',gradient_group_norm_atol=1e-8,gradient_group_norm_rtol=5e-5),
        learned_capacity=dict(parameters=1485584,parameter_tensors=84,buffers=12),
        comparison='Historical N1 LR=.001 primary; LR=.01 descriptive only; exact query counts include historical duplicate diagnostics; no wall-time efficiency claim')


def validate_freeze(selected):
    plan,_=read_report(ROOT/FREEZE,selected)
    require(plan['execution_authorized'] is False and [(a['material'],a['seed']) for a in plan['arms']]==list(CASES) and
        all(a['work']==caps(a['material']) and a['schedule']==[list(x) for x in schedule(a['material'])] for a in plan['arms']),'Frozen E4 design changed')
    for n,h in plan['source_sha256'].items():require(sha(ROOT/n)==h,'Source changed: '+n)
    for n,h in plan['input_sha256'].items():require(sha(n)==h,'Input changed: '+n)
    v,_=read_report(ROOT/VALIDATION,plan['validation_sha256'])
    require(v['exit_code']==0 and v['failed']==v['errors']==v['skipped']==0 and v['scientific_queries']==0 and
        v['new_source_sha256']=={n:plan['source_sha256'][n] for n in CODE},'E4 synthetic validation mismatch')
    require(sha(ROOT/v['JUnit']['path'])==v['JUnit']['sha256'],'E4 JUnit changed')
    return plan


def validate_review(review,selected):
    require(set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_commit','reviewed_branch','run_id','output_root','external_watchdog'} and
        review['phase']=='E.4' and review['scientific_execution_authorized'] is True,'Separate E4 scientific execution approval required')
    plan=validate_freeze(selected);git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    require(review['freeze_sha256']==selected and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10),'E4 watchdog/freeze mismatch')
    require(git('rev-parse','HEAD')==review['reviewed_commit'] and git('branch','--show-current')==review['reviewed_branch']=='feat/neural-material-early-step-e2a' and not git('status','--porcelain'),'Clean reviewed E4 commit required')
    validate_run_id(review['run_id']);p=Path(review['output_root'])
    require(p.is_absolute() and p.resolve()==p and p.parent==Path(plan['output_parent']) and p.name==review['run_id'] and p.name.startswith(plan['run_prefix']) and not p.exists(),'Fresh E4 output required')
    require(shutil.disk_usage(p.parent).free>=plan['storage']['minimum_free_bytes'],'Insufficient storage')
    return plan


def execute_reviewed(review_path,review_sha,freeze):
    review,_=read_report(Path(review_path),review_sha);plan=validate_review(review,freeze)
    # Real N1/shader imports and construction occur only after the explicit gate.
    from experiments import neural_material_recovery_v1 as h
    from experiments import material_chart_e4_v1 as chart
    import platform
    require(dict(python=platform.python_version(),numpy=np.__version__,torch=h.torch.__version__,device=str(h.torch.get_default_device()),torch_threads=h.torch.get_num_threads())==plan['runtime'] and
        h.previous.accepted.arithmetic_runtime()==plan['arithmetic_runtime'],'Accepted CPU runtime mismatch')
    reference,_=h.admit_reference();ft,et,tm=h.inputs(reference);require(tm==plan['tapes'],'Tape identity changed')
    evaluate=diagnostics.evaluation_adapter(h.evaluate)
    store=obs.Store(review['output_root'],plan['output_parent'],total_cap=plan['storage']['output_cap_bytes']);store.json('execution-review.json',review)
    result=dict(protocol=plan['protocol'],status='running',freeze_sha256=freeze,outcomes=[dict(material=a['material'],seed=a['seed'],status='not_started') for a in plan['arms']],
        all_initial_gates_passed=False,cleanup_verified=False,terminal_integrity_verified=False,scientific_work_completed=False,evidence_finalized=False)
    common=obs.read_numeric(E2B,plan['arms'][0]['checkpoint']);common_rng=plan['arms'][0]['RNG'];stage='initial_admission'
    try:
        with prior.old.saved_rng_context(common,common_rng):
            with prior.old.prior.limits() as (check,begin,end):
                shader=h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==plan['inputs']['shader'],'Shader changed')
                observations=reference['scene']['observations'];fit=h.MaterialBatch(reference['scene'],observations[:24],ft);evaluation=h.MaterialBatch(reference['scene'],observations,et)
                cores=[];begin()
                for spec,outcome in zip(plan['arms'],result['outcomes']):
                    m,s=spec['material'],spec['seed'];prefix=m.lower()+'-'+str(s);arrays=obs.read_numeric(E2B,spec['checkpoint']);signals=obs.read_numeric(E2B,spec['signals'])
                    class Core:
                        scientific=True
                        def __init__(self):
                            self.material=m;self.seed=s;self.spec=spec;self.prefix=prefix;self.work=Work(m);self.endpoints={};self.c=h.counts('N1');self.evaluation_counters={}
                            base=h.field.make_material('N1',h.torch.from_numpy(arrays['buffer/initial_latent'].copy()),s)
                            with h.torch.no_grad():
                                for n,p in base.named_parameters():p.copy_(h.torch.from_numpy(arrays['parameter/'+n].copy()))
                                for n,b in base.named_buffers():b.copy_(h.torch.from_numpy(arrays['buffer/'+n].copy()))
                            require(obs.parameter_identity(obs.parameter_arrays(base))==spec['parameter_identity'],'Initial parameter bytes differ')
                            self.model=chart.ChartMaterial(base,h.field.Material);self.opt=h.optimizer_for(self.model)
                            for group in self.opt.param_groups:group['lr']=.001
                            self.opt.defaults['lr']=.001
                            require(prior.old.options(self.opt)==plan['optimizer'] and not self.opt.state,'Fresh matched Adam required')
                            actual,_=prior.old.prior.state_arrays(self.model,self.opt)
                            require(set(actual)==set(arrays) and all(actual[k].dtype==arrays[k].dtype and actual[k].shape==arrays[k].shape and actual[k].tobytes()==arrays[k].tobytes() for k in arrays if k not in ('torch_CPU_RNG','numpy_MT19937')),'Initial weights/buffers differ')
                            require(sum(p.numel() for p in self.model.parameters())==1485584 and len(list(self.model.parameters()))==84,'Capacity mismatch')
                            self.truth=h.previous.reference.truth(m);self.fine,self.target=h.previous.material_target(next(x for x in reference['materials'] if x['material']==m),m)
                            require(h.previous.oldref.sha_array(self.fine)==plan['inputs']['targets'][m]['fine_sha256'],'Target mismatch')
                        def measure(self,state):
                            self.opt.zero_grad();self.work.attempt('forwards')
                            with self.model.observe():
                                values,mat,_=fit.render(self.model,self.model.shader_view(shader),self.c,phase='terminal' if state==100 else 'update')
                                self.work.complete('forwards');loss=h.loss_value(values,self.target);self.work.attempt('backwards');groups=h.backward(self.model,loss,self.c);self.work.complete('backwards')
                                records={};numeric={}
                                for name,t in self.model.nodes.items():
                                    require(t.grad is not None,'Missing actual chart gradient: '+name)
                                    a=obs.tensor_array(t);g=obs.tensor_array(t.grad);records[name]=dict(value=obs.array_record(a),gradient=obs.array_record(g))
                                    if a.size<=8:records[name]['values']=a.tolist()
                                    numeric[name]=a;numeric[name+'.loss_gradient']=g
                                physical=mat.physical[0].detach().numpy()
                                row=dict(state=state,loss=float(loss.detach()),physical=physical.tolist(),physical_errors=diagnostics.recovery_errors(physical,self.truth),
                                    gradient_groups=groups,gradient_norm=math.sqrt(sum(g['squared_norm'] for g in groups.values())),nodes=records,
                                    shader_inputs={k:obs.array_record(numeric['shader.'+k]) for k in ('mx','my','albedo','metallic','encoding')})
                                if state in (0,10,100):
                                    row['signals']=store.numeric(self.prefix+'/signals-'+str(state)+'-'+str(self.work.done['forwards'])+'.npz',numeric,1024**2)
                                if state==0:
                                    same=all(np.array_equal(numeric['shader.'+k],signals['shader.'+k]) for k in ('mx','my','albedo','metallic','encoding'))
                                    store.json(self.prefix+'/initial-attempt.json',dict(actual=row,shader_inputs_equal=same,expected_loss=self.spec['expected_loss'],expected_physical=self.spec['expected_physical']))
                                    require(same and row['physical']==self.spec['expected_physical'] and row['loss']==self.spec['expected_loss'],'Initial physical/shader/loss parity failed')
                                    for group,g in groups.items():
                                        a=math.sqrt(g['squared_norm']);b=math.sqrt(self.spec['expected_groups'][group]['squared_norm'])
                                        require(abs(a-b)<=plan['initial_gate']['gradient_group_norm_atol']+plan['initial_gate']['gradient_group_norm_rtol']*b,'Initial group-gradient parity failed: '+group)
                                    row['initial_gate_passed']=True
                            return row
                        def step(self):h.step(self.model,self.opt,self.c)
                        def evaluate(self,state):
                            c=h.counts('N1');c['calibration_forwards']=0
                            self.evaluation_counters[str(state)]=c
                            return evaluate(self.model,evaluation,shader,c,self.truth,self.fine,observations)
                    core=Core();outcome.update(status='initial_admission',work=core.work.done,attempts=core.work.attempts,native_fit=core.c,native_evaluations=core.evaluation_counters);cores.append((core,outcome))
                    core.cached=core.measure(0);store.json(prefix+'/initial-admission.json',core.cached);outcome['status']='initial_verified';check()
                end();result['all_initial_gates_passed']=True;stage='pilot'
                for core,outcome in cores:
                    begin();path=store.path(core.prefix+'/trajectory.jsonl');previous=None
                    try:
                        with path.open('xb') as f:
                            def record(row):
                                nonlocal previous
                                row['previous_sha256']=previous;row['work']=dict(core.work.done)
                                data=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();require(len(data)<=131072,'E4 row cap')
                                store.account(len(data));f.write(data);f.flush();os.fsync(f.fileno());previous=hashlib.sha256(data).hexdigest()
                                if row['kind']=='fit' and row['update'] and row['post_state'] in (10,100):
                                    a,nr=prior.old.prior.state_arrays(core.model,core.opt)
                                    outcome.setdefault('checkpoints',{})[str(row['post_state'])]=dict(selection=store.numeric(core.prefix+'/state-'+str(row['post_state'])+'.npz',a,18*1024**2),RNG=nr)
                            run_schedule(core,record,check)
                        f=core.work.limit['backwards'];points=96*f
                        expected=dict(field_attempts=f,field_completed=f,calibration_forwards=1,shader_attempts=f,shader_completed=f,
                            fit_forwards=f,evaluation_forwards=0,attempted_points=points,completed_points=points,diffuse_directions=points*512,
                            specular_directions=points*1024,reverse_attempts=f,reverse_completed=f,optimizer_attempts=100,optimizer_returns=100,
                            initial_forwards=0,prefix_forwards=0,terminal_forwards=1)
                        require(core.c==expected,'Native E4 fitting counters differ')
                        expected_eval=dict(field_attempts=1,field_completed=1,calibration_forwards=0,shader_attempts=1,shader_completed=1,
                            fit_forwards=0,evaluation_forwards=1,attempted_points=336,completed_points=336,diffuse_directions=336*512,
                            specular_directions=336*1024,reverse_attempts=0,reverse_completed=0,optimizer_attempts=0,optimizer_returns=0,
                            initial_forwards=0,prefix_forwards=0,terminal_forwards=0)
                        require(set(core.evaluation_counters)=={'10','100'} and all(c==expected_eval for c in core.evaluation_counters.values()),'Native E4 evaluation counters differ')
                        outcome.update(status='complete',endpoints=core.endpoints,native_fit=core.c,native_evaluations=core.evaluation_counters,
                            trajectory=dict(path=str(path.relative_to(store.root)),sha256=sha(path)),hooks_removed=not core.model.nodes and not core.model._active)
                    except BaseException as exc:
                        outcome.update(status='censored' if isinstance(exc,prior.old.prior.Censored) else 'failed',failure=type(exc).__name__+': '+str(exc),native_fit=core.c,native_evaluations=core.evaluation_counters)
                        try:
                            a,nr=prior.old.prior.state_arrays(core.model,core.opt);outcome['failure_state']=dict(selection=store.numeric(core.prefix+'/failure.npz',a,18*1024**2,allow_nonfinite=True),RNG=nr)
                        except BaseException as other:outcome['failure_capture_error']=str(other)
                        raise
                    finally:store.json(core.prefix+'/arm-result.json',outcome)
                    end();h.verify_tapes(ft,et,plan['tapes'])
        result['cleanup_verified']=True;stage='terminal_integrity';validate_freeze(freeze);h.verify_tapes(ft,et,plan['tapes'])
        totals={k:sum(a['work'][k] for a in result['outcomes']) for k in caps('M1')}
        require(all(a['status']=='complete' and a['hooks_removed'] for a in result['outcomes']) and totals==plan['total_work'],'E4 matrix incomplete')
        result.update(status='work_and_cleanup_verified',scientific_work_completed=True,terminal_integrity_verified=True,totals=totals)
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,prior.old.prior.Censored) else 'failed',failure_stage=stage,failure=type(exc).__name__+': '+str(exc));raise
    finally:
        selected=store.json('result.json',result);seal=store.json('finalization/seal.json',dict(report=selected,outcome='complete' if result['scientific_work_completed'] else result['status'],external_success_required=True))
        verified,_=read_report(store.root/seal['path'],seal['sha256']);require(verified['report']==selected,'Seal mismatch')
    result.update(status='complete',evidence_finalized=True,seal=seal);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--verify-sha256');a=p.parse_args()
    if a.prepare:
        plan=make_plan();v=json.loads((ROOT/VALIDATION).read_text());require(v['exit_code']==0 and v['scientific_queries']==0 and v['new_source_sha256']=={n:plan['source_sha256'][n] for n in CODE},'E4 tests/source mismatch');plan['validation_sha256']=sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f:json.dump(plan,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(freeze=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)))
    elif a.verify_sha256:print(json.dumps(dict(verified=bool(validate_freeze(a.verify_sha256)),execution_authorized=False)))
    else:print(json.dumps(dict(arms=5,updates=500,forwards=522,backwards=512,external_evaluations=10,constructor_calibrations=5,execution_authorized=False)))


if __name__=='__main__':main()
