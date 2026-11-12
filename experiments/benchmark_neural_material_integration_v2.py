"""D.2 engineering fixtures only. Never invokes a material shader or recovery."""
import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
from types import FunctionType,SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments import neural_material_recovery_logged_v2 as integrated
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.scientific_reporting_adapter_v2 import plain,Tracker

ROOT=integrated.ROOT
BASE='40b7b301d02835d1b324ed54db6a6c2649f0e049'
RAW_SHA='4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49'
FIELD_STEPS=10


def timed(fn):
    start=time.perf_counter();value=fn();return value,time.perf_counter()-start


class SyntheticModel:
    """Tiny inert tensors. Not a neural module; no automatic differentiation."""
    def __init__(self,arm,latent,seed):
        self.arm=arm;self.seed=seed;self.source_identity={'synthetic_fixture':True}
        self.latent=torch.tensor(latent,dtype=torch.float32,requires_grad=True)
        self.parts=[('latent',self.latent)] if arm!='N1' else [(n,torch.full((2,),.1,requires_grad=True)) for n in
            ('features.weight','mx.weight','my.weight','rotation.weight','metallic.weight','kd.weight')]
    def parameters(self):return (p for _,p in self.parts)
    def named_parameters(self):return iter(self.parts)
    def named_buffers(self):return iter(())


class SyntheticOptimizer:
    """Deterministic arithmetic stand-in; explicitly NOT Adam."""
    def __init__(self,model,options):self.model=model;self.state={};self.param_groups=[dict(params=list(model.parameters()),**options)]
    def zero_grad(self):
        for p in self.model.parameters():p.grad=None
    def step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                old=self.state.get(p,{});n=float(old.get('step',0))+1
                self.state[p]=dict(step=torch.tensor(n,dtype=torch.float32),exp_avg=torch.full_like(p,n*.01),exp_avg_sq=torch.full_like(p,n*.001))
                p.add_(.001)


class SyntheticShader:
    identity={'synthetic_fixture':True}
    def __init__(self,*args):pass


def synthetic_pair(root,scenario='complete',capture_boundaries=False):
    """Run actual original/v2 orchestration with identical explicit test doubles.

    ONLY in this fixture: train_history's literal 250/1000/5000 milestones are
    scaled to 1/2/3, and the global ceiling to 3. Production graph is unchanged.
    No actual shader, optimizer, autograd, references or original targets run.
    """
    v1=integrated._scientific_v1();root=Path(root);root.mkdir(exist_ok=False)
    options=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,
        foreach=None,capturable=False,differentiable=False,fused=None)
    outcomes=[]
    for mode in ('v1','v2'):
        state={'current':None,'phase':'initial','calls':[],'boundaries':[],'writes':0}
        ns=integrated._execution_graph() if mode=='v2' else dict(vars(v1))
        if mode=='v1':ns['execute_reviewed']=FunctionType(v1.execute_reviewed.__code__,ns)
        model_field=SimpleNamespace(**vars(v1.field));model_field.make_material=SyntheticModel
        clock_id=[0]
        class Clock:
            def __init__(self,whole):self.wall=0.;self.cpu=0.;self.whole=whole;clock_id[0]+=1
            def check(self,*args):self.whole.check()
            def elapsed(self):return self.wall
            @contextmanager
            def active(self):
                try:yield
                finally:self.wall+=.001;self.cpu+=.001
        class Whole:
            def check(self):pass
        @contextmanager
        def runtime(report):
            state['report']=report;report['synthetic_fixture']=True
            try:yield Whole()
            finally:report.update(global_torch_CPU_RNG_unchanged=True,global_numpy_RNG_unchanged=True,
                threads_restored=True,memory_limit_restored=True,wall_seconds=1.,CPU_seconds=1.,peak_RSS_bytes=1234)
        def inject(where,model=None):
            arm=getattr(model,'arm',None)
            if scenario=='initial_failure' and where=='initial':raise ValueError('synthetic initial admission failure')
            if scenario=='whole_timeout' and where=='update':raise v1.previous.lm.WholeLimit('synthetic whole cap')
            if scenario=='before_step' and where=='update':raise ValueError('synthetic failure before first step')
            if scenario=='N0_timeout' and where=='update' and arm=='N0':raise v1.HistoryLimit('synthetic N0 cap')
            if scenario=='N1_timeout' and where=='update' and arm=='N1':raise v1.HistoryLimit('synthetic N1 cap')
            if scenario=='N1_numerical' and where=='update' and arm=='N1':raise v1.NumericalFailure('synthetic N1 nonfinite')
            if scenario=='evaluation_failure' and where=='evaluation':raise ValueError('synthetic external evaluation exception')
        class Batch:
            def __init__(self,scene,obs,tape):self.n=len(obs)
            def render(self,model,shader,c,evaluation=False,phase='update',**kw):
                state['current']=model;state['phase']=phase
                inject('evaluation' if evaluation else phase,model)
                for k in ('field_attempts','field_completed','shader_attempts','shader_completed'):c[k]+=1
                c['evaluation_forwards' if evaluation else 'fit_forwards']+=1
                for k in ('attempted_points','completed_points'):c[k]+=self.n*4
                c['diffuse_directions']+=self.n*4*512;c['specular_directions']+=self.n*4*1024
                if phase in ('initial','prefix','terminal'):c[phase+'_forwards']+=1
                p=model.latent.detach().clone().reshape(1,7)
                material=SimpleNamespace(physical=p,encoding=torch.tensor([[.6,.8]]),rotation_raw=None,centered_radius=None)
                return torch.full((self.n,3,3),float(p.sum())),material,None
        def backward(model,loss,c):
            c['reverse_attempts']+=1;c['reverse_completed']+=1
            for p in model.parameters():p.grad=torch.ones_like(p)
            return v1.field.gradient_summary(model)
        def admission(models,batch,shader,target,counters,evidence_sink=None):
            try:
                inject('initial')
                for model,c in zip(models,counters):
                    c['field_attempts']+=2;c['field_completed']+=2
                    values,_,_=batch.render(model,shader,c,phase='initial');backward(model,v1.loss_value(values,target),c)
                return dict(passed=True,synthetic_fixture=True,checks={'synthetic':True})
            except BaseException as exc:
                if evidence_sink:evidence_sink([],exc)
                raise
        def prefix(row,*args,**kw):
            if scenario=='D0_prefix':raise ValueError('synthetic D0 historical-prefix mismatch')
            row['prefix_verified_through']=min(row['completed_updates'],2)
        def digest(previous,model,*args):
            actual=v1.vector_transition_digest(previous,model,*args)
            return 'f'*64 if scenario=='N0_trajectory' and model.arm=='N0' else actual
        def save(output,stem,model,optimizer,completed_updates,**kw):
            if scenario=='checkpoint_failure' and not kw.get('diagnostic'):raise OSError('synthetic checkpoint disk failure')
            value=dict(synthetic_fixture=True,completed_updates=completed_updates,arm=model.arm,
                state=v1.compact_state(model,optimizer) if optimizer.state else dict(parameters={n:v1.field.record(p) for n,p in model.named_parameters()},adam={},optimizer_effective=options),execution_provenance=kw.get('provenance'))
            name='checkpoints/'+stem+'.json';output.write_json(name,value)
            return dict(manifest=name,manifest_sha256=log.file_sha(output.path(name)),completed_updates=completed_updates)
        def evaluate(model,batch,shader,c,truth,targets,obs,**kw):
            values,_,_=batch.render(model,shader,c,evaluation=True)
            return dict(synthetic_fixture=True,success={'all_passed':False},predictions=values.tolist(),
                radiance_errors={'synthetic':.125},recovery_errors={'synthetic':.25})
        def capture(output,stem,audits,exc):
            name='checkpoints/'+stem+'-negative.json';output.write_json(name,dict(synthetic_fixture=True,failure_type=type(exc).__name__,failure=str(exc)))
            return dict(manifest=name,manifest_sha256=log.file_sha(output.path(name)))
        def limits(arm):
            c=v1.counts(arm)
            c.update(field_attempts=9,field_completed=9,shader_attempts=7,shader_completed=7,fit_forwards=6,evaluation_forwards=1,
                attempted_points=912,completed_points=912,diffuse_directions=912*512,specular_directions=912*1024,
                reverse_attempts=5,reverse_completed=5,optimizer_attempts=3,optimizer_returns=3,initial_forwards=1,prefix_forwards=1,terminal_forwards=1)
            return c
        prev=SimpleNamespace(**vars(v1.previous));prev.original=SimpleNamespace(**vars(v1.previous.original))
        prev.original.MaterialShaderV1=SyntheticShader
        prev.reference=SimpleNamespace(PANEL=['M1','M2','M3'],truth=lambda m:[0.]*7)
        prev.material_target=lambda stored,material:(np.zeros((84,3,3)),torch.zeros((24,3)))
        manifest={'synthetic_fixture':True}
        reference=dict(runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1),
            scene={'observations':[{} for _ in range(84)]},shader=SyntheticShader.identity,
            materials=[dict(material=m,fit_target_float32_sha256=prev.oldref.sha_array(np.zeros((24,3),np.float32))) for m in prev.reference.PANEL],
            Adam_runs=[dict(material=m,seed=s,snapshots={'1000':{'objective':float(np.float32(np.float32(.102000005)*7)**2)}})
                for m in prev.reference.PANEL for s in prev.SEEDS])
        # Prefix objective is measured independently by the deterministic stand-in;
        # fake archived equality is established by a test-only scalar dictionary.
        # Use exact predicted float32 objective after two additive stand-in steps.
        probe=SyntheticModel('D0',[.1]*7,101);po=SyntheticOptimizer(probe,options)
        po.step();po.step();x=torch.full((24,3),float(probe.latent.detach().sum()))
        objective=float(torch.mean(x*x))
        for r in reference['Adam_runs']:r['snapshots']['2']=dict(objective=objective)
        freeze=dict(baseline=BASE,tapes=manifest,configuration_sha256='a'*64,sha256={},configuration={'optimizer_effective':options},
            initial_models={m+'/'+str(s)+'/'+a:v1.field.identity(SyntheticModel(a,[.1]*7,s)) for m in prev.reference.PANEL for s in prev.SEEDS for a in v1.ARMS},
            expected_total_work={k:sum(limits(a)[k] for m in prev.reference.PANEL for s in prev.SEEDS for a in v1.ARMS) for k in limits('D0')})
        baseout=integrated.JournalOutput if mode=='v2' else v1.OwnedOutput
        class Output(baseout):
            def write_json(self,name,value):
                if name=='frozen-execution.json':
                    name='synthetic-execution.json';value=dict(synthetic_fixture=True,scientific_execution_authorized=False)
                return super().write_json(name,value)
            def persist(self,report):
                super().persist(report)
                if capture_boundaries:state['boundaries'].append(log.digest(plain(report)))
        ns.update(previous=prev,field=model_field,OwnedOutput=Output,controlled_runtime=runtime,HistoryClock=Clock,
            validate_execution_review=lambda *a:freeze,admit_reference=lambda:(reference,{}),
            inputs=lambda ref:(None,None,manifest),verify_tapes=lambda *a:None,fresh_latent=lambda seed:[.1]*7,
            MaterialBatch=Batch,initial_admission=admission,backward=backward,
            optimizer_for=lambda model:SyntheticOptimizer(model,options),prefix_check=prefix,vector_transition_digest=digest,
            save_checkpoint=save,persist_failure_capture=capture,evaluate=evaluate,work_limits=limits,UPDATES=3,SNAPSHOTS=(1,2,3),
            reused_snapshot_objective=lambda *a:None,time=SimpleNamespace(perf_counter=lambda:0.,process_time=lambda:0.))
        for name in ('step','check_counts','compact_state'):
            ns[name]=FunctionType(getattr(v1,name).__code__,ns)
        # Scaled lifecycle TEST ONLY; numerical production source and graph remain pinned.
        tree=ast.parse(__import__('inspect').getsource(v1.train_history))
        class Scale(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is int and node.value in (5000,1000,250,251):node.value={5000:3,1000:2,250:1,251:2}[node.value]
                if node.value=='5000':node.value='3'
                if node.value=='1000':node.value='2'
                return node
        exec(compile(ast.fix_missing_locations(Scale().visit(tree)),'synthetic-scaled-lifecycle','exec'),ns)
        destination=root/mode
        try:
            with mock.patch.object(v1.previous.original,'MaterialShaderV1',side_effect=AssertionError('Real shader forbidden')),\
                 mock.patch.object(torch.optim,'Adam',side_effect=AssertionError('Real optimizer forbidden')),\
                 mock.patch.object(torch.autograd,'backward',side_effect=AssertionError('Real backward forbidden')):
                ns['execute_reviewed'](dict(freeze_path='synthetic-only',freeze_sha256='b'*64),destination,'synthetic-shared')
            failure=None
        except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc))
        report=log.decode((destination/'report.json').read_bytes()) if (destination/'report.json').is_file() else plain(state.get('report',{}))
        result=dict(mode=mode,report=report,failure=failure,boundaries=state['boundaries'],output=str(destination))
        if mode=='v2':
            selected=log.decode((destination/'reporting-result.json').read_bytes())
            result['reporting']=selected
            if selected['verification_status']=='pending_independent_scan':
                verified=integrated.verify_materialized(destination,selected['identity'],selected['final_manifest']['sha256'])
                result['verified']=verified['status'] if 'status' in verified else True
        outcomes.append(result)
    log.require(log.canonical(outcomes[0]['report'])==log.canonical(outcomes[1]['report']),'Integrated v1/v2 report mismatch: '+scenario)
    log.require(outcomes[0]['failure']==outcomes[1]['failure'],'Integrated failure classification differs')
    if capture_boundaries:log.require(outcomes[0]['boundaries']==outcomes[1]['boundaries'],'Persistence ordering differs')
    return dict(synthetic_fixture=True,scenario=scenario,semantic_equality=True,results=outcomes)


def field_fixture(output):
    """Exactly ten target-independent field-only updates; no material shader."""
    start=time.perf_counter();v1=integrated._scientific_v1();freeze=integrated.source_admission();output=Path(output)
    log.require(not output.exists() and output.is_absolute() and output.resolve()==output,'Fresh explicit engineering output required')
    log.require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (output,*output.parents)),'No engineering outputs inside Git')
    for forbidden in (ROOT.parent/'data',ROOT.parent/'final_results'):
        log.require(output!=forbidden and forbidden not in output.parents and output not in forbidden.parents,'Preserved input overlap')
    output.mkdir();result=dict(protocol='n1-field-only-performance-d2-v1',engineering_only=True,status='running',planned_updates=FIELD_STEPS,
        implementation_sha256=log.file_sha(Path(__file__).resolve()),attempted_updates=0,completed_updates=0,shader_forwards=0,recovery_updates=0,steps=[],source_audit=integrated.source_equivalence_audit())
    def limits(*args):
        log.require(time.perf_counter()-start<120,'Engineering 120-second wall cap')
        log.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*1024**3,'Engineering 2 GiB RSS cap')
    handler=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Engineering wall cap')))
    signal.setitimer(signal.ITIMER_REAL,max(.01,120-(time.perf_counter()-start)))
    torch.set_num_threads(1);rng=torch.get_rng_state().clone();nrng=np.random.get_state()
    try:
        with mock.patch.object(v1.previous.original,'MaterialShaderV1',side_effect=AssertionError('Material shader prohibited')):
            model,cost=timed(lambda:v1.field.make_material('N1',v1.fresh_latent(101),101));result['construction_seconds']=cost
            identity=v1.field.identity(model);log.require(identity==freeze['initial_models']['M1/101/N1'],'Original initial N1 identity mismatch')
            result['initial_model']=identity;optimizer=v1.optimizer_for(model)
            result['optimizer_effective']=v1.previous.lm.accepted.optimizer_options(optimizer)
            initial={n:p.detach().clone() for n,p in model.named_parameters()}
            for i in range(FIELD_STEPS):
                limits();optimizer.zero_grad();result['attempted_updates']+=1
                material,fwd=timed(lambda:model(torch.zeros((1,3),dtype=torch.float32)).validate())
                # Fixed synthetic field-only objective, no fitting/reference radiance.
                objective,obj=timed(lambda:(material.physical*torch.tensor([[1.,2.,3.,4.,5.,6.,7.]])).square().mean()+.1*material.encoding.sum())
                log.require(bool(torch.isfinite(objective)),'Nonfinite field-only synthetic objective')
                _,bwd=timed(objective.backward);groups,observer=timed(lambda:v1.field.gradient_summary(model))
                log.require(all(g['missing']==0 and g['finite'] for g in groups.values()),'Disconnected/nonfinite field gradient')
                _,adam=timed(optimizer.step)
                log.require(all(float(optimizer.state[p]['step'])==i+1 and bool(torch.isfinite(p).all()) and
                    all(bool(torch.isfinite(t).all()) for t in optimizer.state[p].values()) for p in model.parameters()),'Invalid continuous engineering Adam state')
                result['completed_updates']+=1
                result['steps'].append(dict(update=i+1,objective=float(objective.detach()),forward_seconds=fwd,objective_seconds=obj,
                    backward_seconds=bwd,Adam_seconds=adam,gradient_summary_seconds=observer,groups=groups));limits()
            result['parameter_changes']={n:dict(changed=not torch.equal(p,initial[n]),max_abs=float((p-initial[n]).abs().max())) for n,p in model.named_parameters()}
            class Output:
                root=output
                def path(self,name):
                    from infrastructure.paths import contained_path
                    return contained_path(output,name,writable=True)
                def write_json(self,name,value):
                    with self.path(name).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
            (output/'checkpoints').mkdir()
            selection,seconds=timed(lambda:v1.save_checkpoint(Output(),'engineering-only-n1-step10',model,optimizer,FIELD_STEPS,
                provenance=dict(engineering_only=True,not_recovery=True)))
            result['checkpoint']=dict(selection=selection,serialization_seconds=seconds)
            # Read and validate numeric state; not restore, rerun, or historical load.
            meta=log.decode((output/selection['manifest']).read_bytes());archive=output/'checkpoints'/meta['archive']
            def inspect():
                log.require(log.file_sha(archive)==meta['archive_sha256'],'Own numeric archive hash mismatch')
                with np.load(archive,allow_pickle=False) as values:
                    log.require(set(values.files)=={e['key'] for e in meta['entries']}|{'numpy_rng'},'Own array catalog mismatch')
                    for e in meta['entries']:
                        a=values[e['key']];log.require(str(a.dtype)==e['dtype'].replace('torch.','') and list(a.shape)==e['shape'] and
                            hashlib.sha256(a.tobytes()).hexdigest()==e['sha256'],'Own tensor identity mismatch')
            _,seconds=timed(inspect);result['checkpoint']['read_verify_seconds']=seconds;limits()
            result.update(status='complete',final_model=v1.field.identity(model),optimizer_steps=[int(optimizer.state[p]['step']) for p in model.parameters()])
    except BaseException as exc:result.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        after=np.random.get_state();result.update(wall_seconds=time.perf_counter()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            global_torch_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),global_numpy_RNG_unchanged=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:])
        with (output/'measurement.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
    return result


def reporting_benchmark(output,case):
    """Full 5000 synthetic-update mutation cadence; zero model/optimizer calls.

    R0 is measured on IDENTICAL in-memory reports at prespecified boundaries.
    Its total cost is projected, not a repeated multi-hour R0 execution.
    """
    from experiments import benchmark_neural_material_reporting_v1 as old
    start=time.perf_counter();output=Path(output)
    log.require(output.is_absolute() and output.resolve()==output and not output.exists(),'Fresh explicit benchmark output required')
    raw=old.read_selected(Path('/home/isavand/paper/evidence/3d2-recovery-v1-3e3001d85600/report.json'),RAW_SHA)
    desired=old.fixture(raw,'planned_27_synthetic')['report']
    template=copy.deepcopy(desired['outcomes'][0]['history'][0])
    n1_groups=copy.deepcopy(next(r for r in desired['outcomes'] if r['arm']=='N1')['history'][0]['gradient_groups'])
    snapshot_templates=[copy.deepcopy(r['snapshots']) for r in desired['outcomes']]
    for row in desired['outcomes']:
        row.update(history=[],snapshots={},status='admitted',completed_updates=0)
    desired.update(status='running',synthetic_fixture=True,scientific_execution_authorized=False)
    output.mkdir();tracker=Tracker(desired);report=tracker.root
    measurement=dict(protocol='reporting-integrated-benchmark-d2-v1',synthetic_fixture=True,case=case,status='running',
        histories=[],R0_sampled_boundaries=[],source_report_sha256=RAW_SHA,scientific_operations=0,
        plan=dict(updates=5000,histories=27 if case=='complete' else 16,censor_update=4860,
            R0_sample_history_indices=[0,13,26] if case=='complete' else [0,13,15],wall_cap_seconds=600))
    handler=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Synthetic benchmark 600 second cap')))
    signal.setitimer(signal.ITIMER_REAL,600)
    writer=integrated.JournalOutput(output,'synthetic-'+case)
    try:
        writer.persist(report)
        for i,row in enumerate(report['outcomes']):
            if case=='censored' and i>15:break
            begin=time.perf_counter();a0=tracker.append_seconds;s0=tracker.summary_seconds;p0=writer.persist_seconds
            row['status']='optimizing';limit=4860 if case=='censored' and i==15 else 5000
            for update in range(1,limit+1):
                # Same mutation cadence and original schema; fixed synthetic values.
                for k in ('field_attempts','field_completed','shader_attempts','shader_completed','fit_forwards',
                          'reverse_attempts','reverse_completed','optimizer_attempts','optimizer_returns'):
                    row['work'][k]=update
                for k,mult in (('attempted_points',96),('completed_points',96),('diffuse_directions',96*512),('specular_directions',96*1024)):
                    row['work'][k]=update*mult
                row['completed_updates']=update
                if row['arm']!='N1':row['trajectory_sha256']=hashlib.sha256(str(update).encode()).hexdigest()
                if update==1 or update%10==0:
                    record=old.synthetic_record(template,update,row['arm'],n1_groups)
                    row['history'].append(record);writer.persist(report)
                if update in (250,1000,5000):
                    # Includes representative N1 parameter/moment hash catalog.
                    snapshot=copy.deepcopy(snapshot_templates[i].get(str(update)))
                    if snapshot is None:snapshot=copy.deepcopy(snapshot_templates[i]['1000']);snapshot['completed_updates']=update
                    row['snapshots'][str(update)]=snapshot
                if update%100==0:
                    log.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Synthetic benchmark RSS cap')
            row['status']='censored' if limit<5000 else 'complete';row['termination']='active_time_cap' if limit<5000 else 'exact_5000_update_ceiling'
            writer.persist(report)
            measurement['histories'].append(dict(index=i,arm=row['arm'],updates=limit,wall_seconds=time.perf_counter()-begin,
                append_seconds=tracker.append_seconds-a0,summary_seconds=tracker.summary_seconds-s0,persistence_seconds=writer.persist_seconds-p0))
            if i in measurement['plan']['R0_sample_history_indices']:
                with (output/('r0-sample-'+str(i)+'.json')).open('x') as stream:
                    r0=old.persist_r0(stream,plain(report))
                measurement['R0_sampled_boundaries'].append(dict(history_index=i,**r0))
        report.update(status='complete' if case=='complete' else 'failed',phase='finished' if case=='complete' else 'optimization')
        if case=='censored':report['failure_category']='baseline_reproduction';report['failure']='synthetic required N0 control censored'
        writer.persist(report);writer.close()
        result=log.decode((output/'reporting-result.json').read_bytes())
        _,verify_seconds=timed(lambda:integrated.verify_materialized(output,result['identity'],result['final_manifest']['sha256'],verify_checkpoints=False))
        _,hash_seconds=timed(lambda:log.file_sha(output/'journal/events.jsonl'))
        measurement.update(status='complete',semantic_equality=True,reporting=result,
            verification_reconstruction_seconds=verify_seconds,independent_journal_hash_seconds=hash_seconds,
            failure_time_finalization_seconds=result['materialization_seconds'] if case=='censored' else None,
            logical_storage_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            projected_R0_seconds_per_5000_history=[r['total_wall_seconds']*501 for r in measurement['R0_sampled_boundaries']],
            checkpoint_payload_scope='inert synthetic metadata, no numeric checkpoint artifacts; artifact verification excluded only in this benchmark',
            R0_total_scope='sampled full-report save costs times 501; projection, not measured full R0 trajectory')
    except BaseException as exc:
        measurement.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
        writer.close(propagate=False);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        measurement.update(wall_seconds=time.perf_counter()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (output/'measurement.json').open('x') as stream:json.dump(measurement,stream,sort_keys=True,indent=2,allow_nan=False)
    return measurement


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=('field','complete','censored'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    log.require(all(os.environ.get(k)==v for k,v in dict(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1').items()),'Hidden GPU/one CPU thread required')
    soft,hard=resource.getrlimit(resource.RLIMIT_AS);cap=8*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(min(cap,soft) if soft>=0 else min(cap,hard) if hard>=0 else cap,hard))
    if a.case=='field':r=field_fixture(a.output)
    else:r=reporting_benchmark(a.output,a.case)
    print(json.dumps(dict(status=r['status'],output=str(a.output)),sort_keys=True))


if __name__=='__main__':main()
