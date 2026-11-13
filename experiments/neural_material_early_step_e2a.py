"""Prepared E2a experiment. Scientific execution requires a separate E2b review."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

from infrastructure.neural_material_observer_e2a import (GraphObserver,Store,parameter_arrays,
    parameter_identity,displacement,gradient_records,adam_records,array_record)
from infrastructure.signatures import require,read_report

BASE='06f837af623df76e194f60d475a1deb94fa4045a'
CASES=(('M1',101),('M1',103),('M2',103),('M3',101),('M3',103))
RATES=(.01,.001)
LIMITS=dict(forward_attempts=12,forwards=12,backward_attempts=11,backwards=11,
            optimizer_attempts=10,successful_updates=10,training_forwards=10,terminal_forwards=1,external_forwards=1)


def require_identity(actual,expected,kind):
    require(actual==expected,kind+' identity differs')


_ADMITTED_FACTORY=object()


class Censored(RuntimeError):pass


class Work:
    def __init__(self):self.value={k:0 for k in LIMITS}
    def attempt(self,kind,phase=None):
        k={'forward':'forward_attempts','backward':'backward_attempts','optimizer':'optimizer_attempts'}[kind]
        require(self.value[k]<LIMITS[k],'Frozen '+kind+' attempt cap reached')
        if kind=='forward':
            require(phase in ('training','terminal','external'),'Frozen forward phase required')
            key=phase+'_forwards';require(self.value[key]<LIMITS[key],'Frozen forward phase cap reached')
        # Rejected calls must leave every counter unchanged.
        self.value[k]+=1
        if kind=='forward':self.value[key]+=1
    def done(self,kind):
        key={'forward':'forwards','backward':'backwards','optimizer':'successful_updates'}[kind]
        require(self.value[key]<LIMITS[key],'Frozen completion cap reached');self.value[key]+=1
    def complete(self):require(self.value==LIMITS,'Incomplete bounded work')


def parity(state,loss,norm,physical,expected):
    require(state in (0,9),'Only original labels1/10 admitted for historical parity')
    old=expected[str(state+1)]
    require(old['loss_at_update']==state and old['update']==state+1 and
        loss==old['loss'] and norm==old['gradient_norm'] and physical==old['physical_before_update'],
        'Exact historical loss/gradient/physical parity failed at state '+str(state))


def caps(model):
    """Shape-derived planning; no model construction or neural query."""
    def size(d):return sum(int(np.prod(v['shape'],dtype=np.int64))*np.dtype(v['dtype'].split('.')[-1]).itemsize for v in d.values())
    p,b=size(model['parameters']),size(model['buffers']);n=len(model['parameters']);nb=len(model['buffers'])
    ceil=lambda v:((v+1024**2-1)//1024**2)*1024**2
    initial=ceil(p+b+5056+2496+1024*(n+nb+2))
    final=ceil(3*p+b+4*n+5056+2496+1024*(4*n+nb+2))
    xor=ceil(p+1024*n);capture=1024**2;json_cap=2*1024**2
    per_arm=initial+10*xor+2*final+11*capture+64*json_cap
    subtotal=10*per_arm+4*1024**2
    return dict(parameters=p,buffers=b,parameter_tensors=n,buffer_tensors=nb,
        initial_archive=initial,final_archive=final,failure_archive=final,xor_archive=xor,
        signal_archive=capture,json_record=json_cap,max_json_records_per_arm=64,
        per_arm_bytes=per_arm,root_metadata_bytes=4*1024**2,subtotal_bytes=subtotal,
        minimum_free_bytes=2*subtotal,maximum_signal_arrays=130,
        retained_signal_and_gradient_bytes=98528,uncompressed_signal_bytes=98528,
        signal_header_allowance_bytes=130*1024,checkpoint_roles='initial params/buffers/RNG; final/failure include moments/counters',
        xor_method='initial float32 parameters plus ten uint32 XOR deltas reconstruct all pre/post parameter bits')


def state_arrays(model,opt):
    import torch
    arrays={}
    for n,p in model.named_parameters():
        arrays['parameter/'+n]=p.detach().numpy().copy()
        for k,v in opt.state.get(p,{}).items():arrays['adam/'+n+'/'+k]=v.detach().numpy().copy()
    for n,b in model.named_buffers():arrays['buffer/'+n]=b.detach().numpy().copy()
    arrays['torch_CPU_RNG']=torch.get_rng_state().numpy().copy();nrng=np.random.get_state()
    arrays['numpy_MT19937']=nrng[1].copy()
    return arrays,dict(numpy_kind=nrng[0],position=nrng[2],has_gauss=nrng[3],cached_gaussian=nrng[4])


def run_arm(core,store,prefix,lr,expected,plan,check=lambda reserve=0:None,paired_initial=None,result_row=None,_authorization=None):
    """Dependency-injected core for engineering fixtures; only guarded factory is scientific."""
    require(not core.scientific or _authorization is _ADMITTED_FACTORY,'Scientific core requires admitted E2b factory')
    work=Work();result=dict(status='running',completed_updates=0,work=work.value,records=[],historical_parity={},first_boundary_states={},first_all_zero_gradient_state=None,
        first_zero_gradient_states={},first_zero_shader_gradient_states={},first_zero_logit_gradient_states={},
        horizon='10-update diagnostic, not 5000-step recovery',lr=lr)
    if result_row is not None:result_row.update(result);result=result_row
    obs=GraphObserver(core.model,expected_relus=22 if core.scientific else None)
    cap=plan['storage'];last=None;start=time.perf_counter()
    try:
        arrays,rng=state_arrays(core.model,core.opt)
        result['initial_state']=store.numeric(prefix+'/initial.npz',arrays,cap['initial_archive'])
        result['initial_rng']=rng;result['initial_parameter_identity']=parameter_identity(parameter_arrays(core.model))
        with obs:
            for state in range(11):
                terminal=state==10;check(0 if terminal else 10)
                core.opt.zero_grad();obs.begin();work.attempt('forward','terminal' if terminal else 'training')
                values,material=core.forward(obs.wrap_shader(core.shader),terminal)
                work.done('forward');loss=core.loss(values);work.attempt('backward');groups=core.backward(loss);work.done('backward')
                norm=math.sqrt(sum(g['squared_norm'] for g in groups.values()))
                signals,summary=obs.collect();physical=core.physical(material)
                if core.scientific:
                    require(summary['nodes']['shared_features']['value']['shape']==[1,256],'Shared feature layout changed')
                    require(all(v['value']['shape']==[1,256] for n,v in summary['nodes'].items() if n.startswith('relu.')),'ReLU layout changed')
                    for name,width in (('mx_predictor',1),('my_predictor',1),('metallic_predictor',1),('rotation_predictor',2),('kd_predictor',3)):
                        require(summary['nodes'][name+'.sigmoid']['value']['shape']==[1,width] and
                            summary['nodes'][name+'.logits']['value']['shape']==[1,width],'Calibrated head layout changed')
                    for name,width in (('mx',1),('my',1),('albedo',3),('metallic',1),('encoding',2)):
                        require(summary['nodes']['shader.'+name]['value']['shape']==[96,width],'Actual shader input layout changed')
                    require(sum(a.nbytes for a in signals.values())==cap['uncompressed_signal_bytes'],'Signal payload differs from shape-derived budget')
                require(len(signals)<=cap['maximum_signal_arrays'],'Signal array cap exceeded')
                if lr==.01 and state in (0,9):
                    parity(state,float(loss.detach()),norm,physical['physical'],expected)
                    result['historical_parity'][str(state)]=True
                signature=dict(loss=float(loss.detach()),gradient_norm=norm,physical=physical['physical'])
                if state==0:
                    result['initial_signature']=signature
                    require(paired_initial is None or paired_initial==signature,'Paired initial loss/gradient/physical differs')
                record=dict(pre_state=state,post_state=state if terminal else state+1,loss_state=state,
                    loss=float(loss.detach()),original_gradient_norm=norm,physical=physical,
                    parameter_gradients=gradient_records(core.model),original_gradient_groups=groups,
                    observations=summary,optimizer_effective=core.options(),previous_record_sha256=last,
                    Adam_before_return={} if state==0 else adam_records(core.model,core.opt,state),
                    phase='terminal_fitting_forward_backward_no_update' if terminal else 'ordinary_training')
                for head in ('mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor'):
                    if summary['nodes'][head+'.sigmoid']['analytic_derivative']['zero']:
                        result['first_boundary_states'].setdefault(head,state)
                    if summary['nodes'][head+'.logits']['loss_gradient']['nonzero']==0:
                        result['first_zero_logit_gradient_states'].setdefault(head,state)
                for group in ('feats_network','mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor'):
                    if all(g['nonzero']==0 for n,g in record['parameter_gradients'].items() if n.split('.')[0]==group):
                        result['first_zero_gradient_states'].setdefault(group,state)
                for name in ('mx','my','albedo','metallic','encoding'):
                    if summary['nodes']['shader.'+name]['loss_gradient']['nonzero']==0:
                        result['first_zero_shader_gradient_states'].setdefault(name,state)
                if all(g['nonzero']==0 for g in record['parameter_gradients'].values()) and result['first_all_zero_gradient_state'] is None:
                    result['first_all_zero_gradient_state']=state
                record['signals']=store.numeric(prefix+'/signals-'+str(state)+'.npz',signals,cap['signal_archive'])
                if not terminal:
                    before=parameter_arrays(core.model);check(10);work.attempt('optimizer');core.step();work.done('optimizer')
                    result['completed_updates']=state+1;after=parameter_arrays(core.model)
                    xor,record['actual_displacement']=displacement(before,after)
                    record['xor']=store.numeric(prefix+'/xor-'+str(state+1)+'.npz',xor,cap['xor_archive'])
                record['Adam_after_return']=adam_records(core.model,core.opt,10 if terminal else state+1)
                record['work']=dict(work.value);record['active_wall_seconds']=time.perf_counter()-start
                identity=store.json(prefix+'/record-'+str(state)+'.json',record);last=identity['sha256'];result['records'].append(identity)
                obs.release();check()
        final_identity=parameter_identity(parameter_arrays(core.model))
        work.attempt('forward','external');result['endpoint']=core.evaluate();work.done('forward');check()
        require(parameter_identity(parameter_arrays(core.model))==final_identity,'Evaluation modified network parameters')
        arrays,rng=state_arrays(core.model,core.opt)
        result['final_state']=store.numeric(prefix+'/final.npz',arrays,cap['final_archive']);result['final_rng']=rng
        result['final_parameter_identity']=final_identity;work.complete()
        result['native_work']=core.native_work()
        if core.scientific:
            c=result['native_work']
            require(all(c[k]==v for k,v in dict(field_attempts=12,field_completed=12,shader_attempts=12,shader_completed=12,reverse_attempts=11,reverse_completed=11,optimizer_attempts=10,optimizer_returns=10,fit_forwards=11,evaluation_forwards=1,initial_forwards=0,prefix_forwards=0,terminal_forwards=1).items()),'Native work differs from frozen plan')
            require(c['calibration_forwards']==1 and c['attempted_points']==c['completed_points']==1392 and c['diffuse_directions']==712704 and c['specular_directions']==1425408,'Native sample/calibration counts differ')
        result.update(status='complete',terminal_record_sha256=last)
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,Censored) else 'failed',failure_type=type(exc).__name__,failure=str(exc),
            native_work=core.native_work(),terminal_record_sha256=last)
        result['native_optimizer_returns']=result['native_work'].get('optimizer_returns',result['native_work'].get('optimizer',0))
        result['postcheck_verified_returns']=result['completed_updates']
        try:
            arrays,rng=state_arrays(core.model,core.opt)
            result['failure_state']=store.numeric(prefix+'/failure.npz',arrays,cap['failure_archive'],allow_nonfinite=True)
        except BaseException as second:result['failure_capture_error']=type(second).__name__+': '+str(second)
        raise
    finally:
        obs.close();result['hooks_removed']=not obs.handles;result['active_wall_seconds']=time.perf_counter()-start
        store.json(prefix+'/arm-result.json',result)
    return result


@contextmanager
def limits():
    import torch
    require(all(os.environ.get(k)==v for k,v in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items()),'Hidden GPUs/one-thread environment required')
    require(torch.get_num_threads()==1 and torch.get_default_dtype()==torch.float32 and str(torch.get_default_device())=='cpu','CPU float32 defaults required')
    rng=torch.get_rng_state().clone();nrng=np.random.get_state();address=resource.getrlimit(resource.RLIMIT_AS)
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)};whole=time.perf_counter();arm=[None]
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'Existing process timer cannot be replaced')
    def check(reserve=0):
        if time.perf_counter()-whole>=1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise Censored('Whole/RSS cap')
        if arm[0] is not None and time.perf_counter()-arm[0]>=120-reserve:raise Censored('Arm active cap')
    def end_arm():
        arm[0]=None;signal.setitimer(signal.ITIMER_REAL,max(.001,1800-(time.perf_counter()-whole)))
    def alarm(signum,frame):raise Censored('External termination or frozen runtime timer')
    def begin_arm():
        arm[0]=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,max(.001,min(120.,1800-(arm[0]-whole))))
    try:
        cap=min(v for v in (address[0],address[1],8*1024**3) if v>=0);resource.setrlimit(resource.RLIMIT_AS,(cap,address[1]))
        for s in handlers:signal.signal(s,alarm)
        signal.setitimer(signal.ITIMER_REAL,1800.);yield check,begin_arm,end_arm
    finally:
        after=np.random.get_state();same=torch.equal(rng,torch.get_rng_state()) and nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        torch.set_rng_state(rng);np.random.set_state(nrng);signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        resource.setrlimit(resource.RLIMIT_AS,address)
        require(same,'Scientific global RNG changed')


def validate_review(review,freeze_sha):
    from experiments.freeze_neural_material_early_step_e2a import validate_freeze
    require(type(review) is dict and set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_branch','reviewed_commit','output_root','run_id','external_watchdog'} and
        review['phase']=='E.2b' and review['scientific_execution_authorized'] is True,
        'Separate E.2b scientific authorization required; E.2a is preparation only')
    require(review['freeze_sha256']==freeze_sha and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10),'Reviewed freeze/watchdog mismatch')
    require(len(json.dumps(review,sort_keys=True,indent=2).encode())+1<=16384,'Execution review metadata cap exceeded')
    freeze=validate_freeze(freeze_sha)
    from infrastructure.runs import validate_run_id
    validate_run_id(review['run_id'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    require(review['reviewed_commit']==git('rev-parse','HEAD') and review['reviewed_branch']==git('branch','--show-current')=='feat/neural-material-early-step-e2a' and
        git('status','--porcelain')=='','Reviewed commit/branch must be exact and clean')
    root=Path(review['output_root']);require(root.is_absolute() and root.resolve()==root and root.parent==Path(freeze['output_parent']) and
        root.name==review['run_id'] and root.name.startswith('neural-e2b-') and not root.exists(),'Fresh explicit reviewed output required')
    import shutil
    require(shutil.disk_usage(root.parent).free>=freeze['storage']['minimum_free_bytes'],'Insufficient frozen output capacity')
    return freeze


def verify_completed_work(outcomes,plan):
    expected=[(m,s,lr) for m,s in CASES for lr in RATES]
    require([(r['material'],r['seed'],r['lr']) for r in outcomes]==expected,'Exact ten-arm ordered matrix required')
    for row in outcomes:
        require(row['status']=='complete' and row['completed_updates']==10 and row['hooks_removed'] is True and
            row['work']==LIMITS and row['native_work']==plan['native_work_per_arm'],'Incomplete arm/counter/hook verification')
    totals={k:sum(row['work'][k] for row in outcomes) for k in LIMITS}
    require(totals==plan['total_work'],'Frozen total work differs')
    return totals


def terminal_integrity(plan,outcomes,freeze_sha256,verify_tapes,checks,verify_freeze=None):
    """Metadata/hash and tape-array checks only; no renderer or field query."""
    if verify_freeze is None:
        from experiments.freeze_neural_material_early_step_e2a import validate_freeze
        verify_freeze=validate_freeze
    for name,check in (
        ('frozen_sources_and_evidence',lambda:require_identity(verify_freeze(freeze_sha256),plan,'terminal freeze/source/evidence')),
        ('immutable_tapes',verify_tapes),
        ('work_counters',lambda:verify_completed_work(outcomes,plan))):
        try:check()
        except BaseException as exc:
            checks[name]=dict(status='failed',reason=type(exc).__name__+': '+str(exc));raise
        checks[name]=dict(status='passed')


def finish_execution(result,store,work,integrity,plan,runtime=None):
    """Separate work, normal cleanup, terminal admission and durable reporting.

    Persisted result.json never claims finalization. A separate bounded seal
    selects its hash after fsync. A successful process exit is additionally
    required for a consumer to accept the seal as a complete run.
    """
    result.update(status='running',work_counters_verified=False,scientific_work_completed=False,
        cleanup_verified=False,evidence_finalized=False,report_written=False,
        runtime_cleanup_status='not_verified',
        terminal_integrity={k:dict(status='not_verified',reason='Terminal gate not reached') for k in
            ('frozen_sources_and_evidence','immutable_tapes','work_counters')})
    stage='runtime_setup'
    try:
        with (runtime or limits)() as handles:
            stage='scientific_work';work(*handles)
            result['verified_totals']=verify_completed_work(result['outcomes'],plan)
            result['work_counters_verified']=True;stage='runtime_cleanup'
        # This is reachable only following a normal, non-suppressed body/exit.
        require(result['work_counters_verified'],'Runtime suppressed an incomplete work failure')
        result.update(cleanup_verified=True,runtime_cleanup_status='passed')
        stage='terminal_integrity';integrity(result['terminal_integrity'])
        require(all(v['status']=='passed' for v in result['terminal_integrity'].values()),'Terminal integrity incomplete')
        result.update(scientific_work_completed=True,status='work_and_cleanup_verified',interpretation='short-horizon diagnostic only')
    except BaseException as exc:
        result.update(status='censored' if isinstance(exc,Censored) else 'failed',failure_stage=stage,
            failure_type=type(exc).__name__,failure=str(exc))
        if stage=='runtime_cleanup':result['runtime_cleanup_status']='failed'
        raise
    finally:
        result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        try:
            # Keep the immutable report explicitly pending until a seal exists.
            report=store.json('result.json',result)
            result.update(report_written=True,report_identity=report)
            seal=dict(schema_version=2,report=report,
                outcome='complete' if result['scientific_work_completed'] else result['status'],
                cleanup_verified=result['cleanup_verified'],successful_process_exit_also_required=True)
            require(len(json.dumps(seal,sort_keys=True,indent=2).encode())+1<=4096,'Finalization seal cap exceeded')
            selected=store.json('finalization/seal.json',seal)
            require_identity(read_report(store.root/selected['path'],selected['sha256'])[0],seal,'finalization seal')
            result.update(evidence_finalized=True,finalization_seal=selected)
            if result['scientific_work_completed']:result['status']='complete'
        except BaseException as exc:
            result.update(status='failed',evidence_finalized=False,finalization_failure_type=type(exc).__name__,
                finalization_failure=str(exc));raise
    return result


def execute_reviewed(review_path,review_sha256,freeze_sha256):
    """No default or CLI scientific route; explicit hash-selected E2b review only."""
    review,_=read_report(Path(review_path),review_sha256);plan=validate_review(review,freeze_sha256)
    # All scientific imports/construction are below the complete admission gate.
    from experiments import neural_material_recovery_v1 as h
    require(dict(python=__import__('platform').python_version(),numpy=np.__version__,torch=h.torch.__version__,
        device='cpu',torch_threads=h.torch.get_num_threads())==plan['runtime'],'Accepted runtime changed')
    require(h.previous.accepted.arithmetic_runtime()==plan['arithmetic_runtime'],'Arithmetic runtime changed')
    reference,_=h.admit_reference();ft,et,tm=h.inputs(reference)
    require_identity(tm,plan['tapes'],'tape')
    store=Store(review['output_root'],plan['output_parent'],total_cap=plan['storage']['subtotal_bytes']);store.json('execution-review.json',review)
    result=dict(protocol=plan['protocol'],status='running',scientific_execution_authorized=True,
        reviewed_freeze_sha256=freeze_sha256,review_sha256=review_sha256,run_id=review['run_id'],
        runtime=plan['runtime'],source_sha256=plan['source_sha256'],outcomes=[dict(material=m,seed=s,lr=lr,status='not_started',completed_updates=0) for m,s in CASES for lr in RATES],planned_arms=10)
    def scientific_work(check,begin_arm,end_arm):
        shader=h.previous.original.MaterialShaderV1('A2','atan2')
        require_identity(shader.identity,plan['inputs']['shader'],'shader source')
        scene=reference['scene'];obs=scene['observations']
        fit=h.MaterialBatch(scene,obs[:24],ft);evaluation=h.MaterialBatch(scene,obs,et)
        for material,seed in CASES:
            pair=[h.field.make_material('N1',h.fresh_latent(seed),seed) for _ in RATES]
            for m in pair:require_identity(h.field.identity(m),plan['initial_models'][material+'/'+str(seed)],'initial model/source/calibration')
            stored=next(x for x in reference['materials'] if x['material']==material);fine,target=h.previous.material_target(stored,material)
            require_identity(h.previous.oldref.sha_array(target.numpy()),plan['inputs']['targets'][material]['fit_float32_sha256'],'target')
            require_identity(h.previous.oldref.sha_array(fine),plan['inputs']['targets'][material]['fine_sha256'],'fine target')
            initial=None
            for model,lr in zip(pair,RATES):
                class Core:
                    scientific=True
                    def __init__(self):
                        self.model=model;self.shader=shader;self.opt=h.optimizer_for(model);self.c=h.counts('N1')
                        for group in self.opt.param_groups:group['lr']=lr
                        self.opt.defaults['lr']=lr
                        expected=dict(plan['optimizer'],lr=lr);require(self.options()==expected and not self.opt.state,'Only LR may differ; fresh Adam required')
                    def options(self):return h.previous.lm.accepted.optimizer_options(self.opt)
                    def forward(self,view,terminal):
                        v,m,_=fit.render(self.model,view,self.c,phase='terminal' if terminal else 'update');return v,m
                    def loss(self,v):return h.loss_value(v,target)
                    def backward(self,loss):return h.backward(self.model,loss,self.c)
                    def step(self):h.step(self.model,self.opt,self.c)
                    def physical(self,m):
                        from experiments.analyze_neural_material_failures_e1 import boundary_distances
                        p=m.physical[0].detach().numpy().tolist()
                        return dict(physical=p,boundary_distances=boundary_distances(p).tolist(),
                            centered_radius=m.centered_radius.detach().tolist(),encoding=m.encoding.detach().tolist(),
                            errors=h.previous.recovery_errors(np.array(p),h.previous.reference.truth(material)))
                    def evaluate(self):return h.evaluate(self.model,evaluation,shader,self.c,h.previous.reference.truth(material),fine,obs)
                    def native_work(self):return dict(self.c)
                core=Core();begin_arm();prefix=material.lower()+'-'+str(seed)+'-'+('baseline' if lr==.01 else 'lr001')
                row=next(r for r in result['outcomes'] if (r['material'],r['seed'],r['lr'])==(material,seed,lr))
                arm=run_arm(core,store,prefix,lr,plan['historical_parity'][material+'/'+str(seed)],plan,check,initial,row,_ADMITTED_FACTORY)
                end_arm()
                if lr==.01:initial=arm['initial_signature']
                h.verify_tapes(ft,et,tm);check()
    def integrity(checks):
        def tapes():
            require_identity(tm,plan['tapes'],'terminal tape manifest')
            h.verify_tapes(ft,et,plan['tapes'])
        terminal_integrity(plan,result['outcomes'],freeze_sha256,tapes,checks)
    return finish_execution(result,store,scientific_work,integrity,plan)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--describe',required=True,action='store_true');p.parse_args()
    print(json.dumps(dict(preparation_only=True,execution_authorized=False,cases=CASES,rates=RATES,updates_per_arm=10)))


if __name__=='__main__':main()
