"""E8R endpoint-corrected spatial pilot; frozen E8 scientific body preserved."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments import hybrid_material_e7_v1 as prior
from experiments import spatial_material_e8_v1 as method
from experiments import spatial_pilot_e8_v1 as legacy
from experiments import uniform_tapes_e8r_v1 as tape_policy
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report
from infrastructure.runs import validate_run_id

ROOT = prior.ROOT; BASE = '51eac252cc15060612c808286ca06a0bf4774ece'
PARENT = '425956188c550f3139996f70e06f70b19a36c9a574af88a2fddc8df862d68aca'
PROTOCOL = 'spatial-neural-physical-e8r-v1'
FREEZE = 'docs/protocols/SPATIAL_MATERIAL_E8R_V1.freeze.json'
VALIDATION = 'docs/evidence/SPATIAL_MATERIAL_E8R_PREPARATION_V1.json'
CODE = ('experiments/uniform_tapes_e8r_v1.py','experiments/spatial_pilot_e8r_v1.py',
        'tests/scientific/test_spatial_material_e8r_v1.py','docs/protocols/SPATIAL_MATERIAL_E8R_V1.md')
CAPS = dict(legacy.CAPS,reference_calls=0,reference_directions=0)
RAW = Path('/home/isavand/paper/evidence/material-e8-565980ba2b1245a39b221288ea2b7fac')
OLD_REPORT = dict(path='result.json',sha256='42abc3d3d4a672cc8faa4f1733701ca488b52642690a9eda632321a475fc9ecb')
sha = prior.sha


def same_design(old, new):
    keys = ('seeds','arms','train_points','held_out_points','truth','policy','initial_latents','spatial_network',
            'optimizer','scene','reference_grid_identities','runtime','arithmetic_runtime','inputs','output_parent','storage','resources')
    require(all(old[k] == new[k] for k in keys) and all(new['limits'][k] == v for k,v in old['limits'].items()
        if k not in ('reference_calls','reference_directions')), 'Frozen E8 scientific design changed')


class Work:
    def __init__(self): self.attempts = dict.fromkeys(CAPS,0); self.done = dict(self.attempts)
    def attempt(self, key, count=1):
        require(key in CAPS and type(count) is int and count >= 0 and self.attempts[key]+count <= CAPS[key], 'E8 cap: '+key)
        self.attempts[key] += count
    def complete(self, key, count=1):
        require(count >= 0 and self.done[key]+count <= self.attempts[key], 'Unattempted E8 completion')
        self.done[key] += count
    def verify(self): require(self.attempts == self.done, 'Incomplete E8 attempted work')


def matching_updates(guided_costs):
    """Total C: control16n+100; guided correction cost+74. Scores never enter."""
    require(len(guided_costs)==2 and all(528 <= c <= 624 for c in guided_costs), 'Invalid guided correction ledger')
    n = (min(guided_costs)-26)//16
    require(31 <= n <= 37, 'Work-matched endpoint outside fixed cap'); return n


def make_plan():
    parent = legacy.validate_freeze(PARENT); ids = dict(parent['input_sha256'])
    path = ROOT/'docs/results/spatial_material_e8_attempt1/evidence_manifest.json'
    require(path.read_bytes() == subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(ROOT))],cwd=ROOT), 'Accepted E8 manifest changed')
    ids[str(path)] = sha(path); manifest = json.loads(path.read_text())
    for item in manifest['raw_files']+manifest['sidecars']:
        p = Path(item['path']); require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'Accepted E8 artifact changed')
        ids[str(p)] = item['sha256']
    ids[str(ROOT/legacy.FREEZE)] = PARENT
    _, preflight = tape_policy.preflight(parent['tapes'])
    tape_policy.admit_references(RAW,OLD_REPORT,parent)
    plan = dict(parent); sources = dict(parent['source_sha256']); sources.update({n:sha(ROOT/n) for n in CODE})
    plan.update(protocol=PROTOCOL,base_commit=BASE,limits=CAPS,source_sha256=sources,input_sha256=ids,
        original_tapes=parent['tapes'],tapes=preflight['corrected_identities'],numerical_preflight=preflight,
        reused_reference_root=str(RAW),reused_reference_report=OLD_REPORT,reused_reference_freeze=PARENT,
        execution_authorized=False)
    plan.pop('validation_sha256',None);same_design(parent,plan);return plan


def validate_freeze(selected):
    p,_ = read_report(ROOT/FREEZE,selected)
    require(p['protocol'] == PROTOCOL and p['execution_authorized'] is False and p['limits'] == CAPS, 'E8R freeze differs')
    old,_ = read_report(ROOT/legacy.FREEZE,PARENT);same_design(old,p)
    require(p['original_tapes'] == old['tapes'] and p['reused_reference_root'] == str(RAW) and p['reused_reference_report'] == OLD_REPORT and
        p['reused_reference_freeze'] == PARENT, 'Reference/tape selection changed')
    for name,h in p['source_sha256'].items():require(sha(ROOT/name) == h, 'Source changed: '+name)
    for name,h in p['input_sha256'].items():require(sha(name) == h, 'Input changed: '+name)
    _, checked = tape_policy.preflight(p['original_tapes'],p['numerical_preflight'])
    require(checked['corrected_identities'] == p['tapes'], 'Corrected tape identities changed')
    tape_policy.admit_references(RAW,OLD_REPORT,p)
    v,_ = read_report(ROOT/VALIDATION,p['validation_sha256'])
    require(v['exit_code'] == v['failed'] == v['errors'] == v['skipped'] == v['scientific_queries'] == 0 and
        v['new_source_sha256'] == {n:p['source_sha256'][n] for n in CODE} and sha(ROOT/v['JUnit']['path']) == v['JUnit']['sha256'], 'E8R validation changed')
    return p


def validate_review(review, selected):
    require(set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_commit','reviewed_branch','run_id','output_root','external_watchdog'} and
        review['phase']=='E.8R' and review['scientific_execution_authorized'] is True, 'Explicit E8R approval required')
    p = validate_freeze(selected); git = lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    require(review['freeze_sha256']==selected and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10), 'E8 freeze/watchdog differs')
    require(review['reviewed_commit']==git('rev-parse','HEAD') and review['reviewed_branch']==git('branch','--show-current')=='feat/neural-material-early-step-e2a' and not git('status','--porcelain'), 'Clean reviewed E8 commit required')
    validate_run_id(review['run_id']); out = Path(review['output_root'])
    require(out.is_absolute() and out.resolve()==out and out.parent==Path(p['output_parent']) and out.name==review['run_id'] and out.name.startswith('material-e8r-') and not out.exists(), 'Fresh E8 output required')
    require(shutil.disk_usage(out.parent).free>=p['storage']['minimum_free_bytes'], 'Insufficient storage'); return p


def execute_reviewed(review_path, review_sha, freeze_sha):
    review,_ = read_report(Path(review_path),review_sha); plan = validate_review(review,freeze_sha)
    prepared_tapes, preflight = tape_policy.preflight(plan['original_tapes'],plan['numerical_preflight'])
    import platform
    import torch
    from experiments import neural_material_recovery_v1 as h
    from experiments import neural_material_continuation_e3 as runtime
    require(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=str(torch.get_default_device()),torch_threads=torch.get_num_threads())==plan['runtime'] and h.previous.accepted.arithmetic_runtime()==plan['arithmetic_runtime'], 'Accepted CPU runtime required')
    work = Work(); store = obs.Store(review['output_root'],plan['output_parent'],total_cap=plan['storage']['output_cap_bytes'])
    store.json('execution-review.json',review); started = time.perf_counter(); phase='setup'; cores=[]; tape_arrays=prepared_tapes
    result=dict(schema_version=1,protocol=PROTOCOL,freeze_sha256=freeze_sha,status='running',attempts=work.attempts,completed=work.done,
        numerical_preflight=preflight,reused_reference_root=plan['reused_reference_root'],references=[],arms=[],cleanup_verified=False,terminal_integrity_verified=False,scientific_work_completed=False,evidence_finalized=False)
    try:
        with runtime.prior.limits() as (check,begin,end):
            points=plan['train_points']+plan['held_out_points']; scene=plan['scene']; observations=scene['observations']; fine=[]
            phase='reused_reference_admission';begin()
            fine,reused=tape_policy.admit_references(plan['reused_reference_root'],plan['reused_reference_report'],plan)
            result['references']=reused;result['reference_role']='reused_successful_E8_evidence_not_new_measurements';end()
            phase='corrected_tape_batch_construction';begin()
            batches={}; shader=h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==plan['inputs']['shader'],'Shader changed')
            for i,point in enumerate(points):
                for name,start,count in (('fit',6000,24),('evaluation',7000,84)):
                    a=tape_arrays[i,name]
                    require(prior.prior.previous.tape_identity(a)==plan['tapes'][str(i)][name], 'Fixed tapes changed')
                    t=h.previous.original.RandomInputs(*(torch.from_numpy(a[n]) for n in ('u','v','azimuth')))
                    b=h.previous.original.ObservationBatch(scene,observations[:24] if name=='fit' else observations,t,4)
                    b.positions=torch.tensor(point,dtype=torch.float32).reshape(1,3).repeat(4*count,1);batches[i,name]=b
            end()
            class Core:
                def __init__(self,seed,arm):
                    self.seed=seed;self.arm=arm;self.prefix=str(seed)+'-'+arm;self.fitF=self.evalF=self.physF=self.revN=self.revP=0
                    self.row=dict(seed=seed,arm=arm,status='initializing',history=[]);result['arms'].append(self.row)
                    work.attempt('constructors');self.model=method.make_spatial(torch.tensor(plan['initial_latents'][str(seed)],dtype=torch.float32),seed);work.complete('constructors')
                    self.opt=torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                    self.native=h.counts('N1');self.initial_state={n:a.copy() for n,a in obs.parameter_arrays(self.model).items()}
                    self.initial_buffers={n:b.detach().numpy().copy() for n,b in self.model.named_buffers()}
                def cost(self):return 2*(self.fitF+self.physF+self.revN+self.revP)+7*self.evalF
                def probe(self,label):
                    work.attempt('field_extra');m,_,z=method.capture(self.model,torch.tensor(points,dtype=torch.float32));work.complete('field_extra')
                    return store.numeric(self.prefix+'/'+label+'.npz',dict(points=np.asarray(points),physical=m.physical.numpy(),encoding=m.encoding.numpy(),logits=z),1024**2),m.physical.numpy().copy()
                def joint(self,backward=False,label=None):
                    if backward:self.opt.zero_grad()
                    values=[];physical=[];encodings=[];losses=[];zs=[];probabilities=[];features={n:[] for n,_ in method.HEADS};gradients=[]
                    for i in range(4):
                        check();work.attempt('neural_fit');self.fitF+=1
                        with (obs.GraphObserver(self.model,expected_relus=22) if backward else nullcontext()) as observer:
                            if backward:observer.begin()
                            with method.monitor(self.model) as (f,z,q):
                                with (nullcontext() if backward else torch.no_grad()):
                                    m=self.model(torch.tensor(points[i],dtype=torch.float32).reshape(1,3))
                                    y=method.shade_material(observer.wrap_shader(shader) if backward else shader,batches[i,'fit'],m)
                                    loss=h.loss_value(y,torch.from_numpy(fine[i][:24,2].astype(np.float32)))
                            work.complete('neural_fit');values.append(y.detach().numpy().copy());physical.append(m.physical[0].detach().numpy().copy());encodings.append(m.encoding[0].detach().numpy().copy());losses.append(float(loss.detach()))
                            zs.append(np.concatenate([z[n] for n,_ in method.HEADS],-1)[0])
                            probabilities.append(np.concatenate([q[n] for n,_ in method.HEADS],-1)[0])
                            for n in features:features[n].append(f[n][0])
                            if backward:
                                work.attempt('neural_reverse');groups=h.backward(self.model,loss/4,self.native);work.complete('neural_reverse');self.revN+=1
                                signals,graph=observer.collect();gradients.append(dict(groups=groups,head_boundaries={n:graph['nodes'][n+'.sigmoid']['analytic_derivative']['zero'] for n,_ in method.HEADS},
                                    shader={n:graph['nodes']['shader.'+n]['loss_gradient'] for n in ('mx','my','albedo','metallic','encoding')}))
                                if label:store.numeric(self.prefix+'/'+label+'-signals-'+str(i)+'.npz',signals,1024**2)
                    return dict(values=np.stack(values),physical=np.stack(physical),encoding=np.stack(encodings),losses=np.asarray(losses),
                                logits=np.stack(zs),probabilities=np.stack(probabilities),features={n:np.stack(v) for n,v in features.items()},gradients=gradients)
                def evaluate(self,label):
                    rows=[];predictions=[];physicals=[]
                    for i,point in enumerate(points):
                        work.attempt('neural_eval');self.evalF+=1
                        with torch.no_grad():m=self.model(torch.tensor(point,dtype=torch.float32).reshape(1,3));y=method.shade_material(shader,batches[i,'evaluation'],m).numpy().astype(np.float64)
                        work.complete('neural_eval');p=m.physical[0].numpy();metrics={}
                        for split in ('fit','held_out','relight'):
                            mask=[o['split']==split for o in observations]
                            metrics[split]=dict(errors={k:h.previous.original.radiance_errors(y[mask,j],fine[i][mask,j]) for j,k in enumerate(('diffuse','specular','total'))},**prior.prior.previous.component_errors(y[mask],fine[i][mask]))
                        errors=diagnostic.recovery_errors(p,np.asarray(plan['truth'][i]));rows.append(dict(location=i,spatial_split='train' if i<4 else 'held_out',physical=p.tolist(),errors=errors,metrics=metrics,
                            success=h.previous.original.assess_success(errors,{s:v['errors'] for s,v in metrics.items()})))
                        predictions.append(y);physicals.append(p.copy())
                    return dict(locations=rows,arrays=store.numeric(self.prefix+'/'+label+'-evaluation.npz',dict(predictions=np.stack(predictions),physical=np.stack(physicals)),1024**2))
                def snapshot(self,label):
                    a,rng=runtime.prior.state_arrays(self.model,self.opt)
                    return dict(role='Adam_continuation' if self.arm=='Adam' else 'weights_only',RNG=rng,selection=store.numeric(self.prefix+'/'+label+'.npz',a,18*1024**2))
            phase='all_initial_admission';begin()
            for seed in method.SEEDS:
                paired=[]
                for arm in method.ARMS:
                    c=Core(seed,arm);cores.append(c);probe,p=c.probe('initial-probe');c.current=c.joint(True,'initial');c.cache=c.current['features'];paired.append(c)
                    require(float(np.ptp(p[:,:2],axis=0).max())>1e-6,'Spatial field is accidentally constant')
                    c.row.update(initial_probe=probe,initial_loss=c.current['losses'].tolist(),initial_physical=c.current['physical'].tolist(),initial_parameter_identity=obs.parameter_identity(c.initial_state),status='initial_verified')
                require(all(c.row['initial_parameter_identity']==paired[0].row['initial_parameter_identity'] and np.array_equal(c.current['physical'],paired[0].current['physical']) and
                    np.array_equal(c.current['values'],paired[0].current['values']) and all(np.array_equal(v,paired[0].initial_buffers[n]) for n,v in c.initial_buffers.items()) for c in paired), 'Unmatched spatial initialization')
            end();result['all_initial_gates_passed']=True
            for seed in method.SEEDS:
                paired=[c for c in cores if c.seed==seed];guided_costs=[]
                for c in paired[1:]:
                    phase=c.prefix;begin();cost0=c.cost();rowtime=time.perf_counter()
                    for round_index in range(12):
                        check(10);selected=round_index%4;p=c.current['physical'][selected].copy();target=torch.from_numpy(fine[selected][:24,2].astype(np.float32))
                        c.pending=dict(physical_before=c.current['physical'].copy(),predictions_before=c.current['values'].copy(),logits_before=c.current['logits'].copy())
                        def direct(x,phase='jacobian'):
                            work.attempt('physical');c.physF+=1;y=batches[selected,'fit'](shader,x);work.complete('physical');return y
                        with torch.no_grad():base=direct(torch.from_numpy(p.copy()))
                        bridge=float(abs(base.numpy()-c.current['values'][selected]).max());require(bridge<=1e-7+1e-5*float(abs(base).max()),'Direct/neural sampling bridge failed')
                        class JWork:
                            def attempt(self,*_):work.attempt('physical_reverse')
                            def complete(self,*_):work.complete('physical_reverse');c.revP+=1
                        work.attempt('Jacobians');j=prior.prior.previous.columns(lambda x,phase:direct(x)[:,2].reshape(72),torch.from_numpy(p),base[:,2].reshape(72),JWork(),c.pending,check).astype(np.float64)/math.sqrt(72);work.complete('Jacobians')
                        residual=(base[:,2].numpy()-target.numpy()).astype(np.float64).reshape(72)/math.sqrt(72)
                        work.attempt('proposals');teacher,detail=prior.prior.method.proposal(p,j,residual,.01);work.complete('proposals')
                        c.pending.update(J_scaled=j.copy(),teacher=teacher.copy())
                        record=dict(round=round_index+1,selected=selected,physical_before=c.current['physical'].tolist(),loss_before=c.current['losses'].tolist(),teacher=teacher.tolist(),proposal=detail,bridge_max_error=bridge,queried=False,accepted=False)
                        if detail['predicted_reduction']>0 and not np.array_equal(p,teacher):
                            work.attempt('projections')
                            def count_solve(stage):getattr(work,stage)('linear_solves')
                            updates,projection=method.proposed_updates(c.cache,c.current['logits'],c.current['physical'],selected,teacher,0. if c.arm=='no_retention' else 1.,count_solve)
                            with method.transfer(c.model,updates) as commit:
                                work.complete('projections');trial=c.joint()
                                require(all(np.array_equal(trial['features'][n],c.cache[n]) for n in c.cache),'Unintended trunk/hidden feature change')
                                decision=method.accept(c.current['physical'],trial['physical'],teacher,selected,c.current['losses'],trial['losses'],trial['probabilities'])
                                c.pending.update(trial_predictions=trial['values'].copy(),trial_physical=trial['physical'].copy(),trial_logits=trial['logits'].copy())
                                if decision['accepted']:commit();c.current=trial
                                record.update(queried=True,accepted=decision['accepted'],projection=projection,decision=decision,trial_physical=trial['physical'].tolist(),trial_loss=trial['losses'].tolist())
                        allowed={n+'.predictor.6.'+k for n,_ in method.HEADS for k in ('weight_g','weight_v','bias')}
                        require(all(np.array_equal(v,c.initial_state[n]) for n,v in obs.parameter_arrays(c.model).items() if n not in allowed) and
                            all(np.array_equal(b.detach().numpy(),c.initial_buffers[n]) for n,b in c.model.named_buffers()), 'Projection changed non-final-layer state')
                        c.pending['physical_after']=c.current['physical'].copy()
                        record['arrays']=store.numeric(c.prefix+'/round-'+str(round_index+1)+'.npz',c.pending,1024**2)
                        record['cumulative_C']=c.cost();c.row['history'].append(store.json(c.prefix+'/round-'+str(round_index+1)+'.json',record))
                    guided_costs.append(c.cost()-cost0);c.row['correction_C']=c.cost()-cost0
                    terminal=c.joint(True,'terminal');c.row['terminal_loss']=terminal['losses'].tolist();c.row['terminal_gradients']=terminal['gradients']
                    c.row['final_probe'],_=c.probe('final-probe');c.row['evaluation']=c.evaluate('final');c.row['checkpoint']=c.snapshot('final-state')
                    c.row.update(status='complete',total_C=c.cost(),arm_seconds=time.perf_counter()-rowtime);end()
                c=paired[0];n=matching_updates(guided_costs);phase=c.prefix;begin();rowtime=time.perf_counter()
                for k in range(n):
                    check(10);measurement=c.current if k==0 else c.joint(True);work.attempt('Adam_returns');h.step(c.model,c.opt,c.native);work.complete('Adam_returns')
                    c.row['history'].append(dict(update=k+1,pre_update_loss=measurement['losses'].tolist(),pre_update_physical=measurement['physical'].tolist(),gradients=measurement['gradients']))
                    if k+1==12:
                        c.row['equal_update_probe'],_=c.probe('state12-probe');c.row['equal_update_evaluation']=c.evaluate('state12');c.row['state12']=c.snapshot('state12')
                terminal=c.joint(True,'terminal');c.row['terminal_loss']=terminal['losses'].tolist();c.row['terminal_gradients']=terminal['gradients']
                c.row['final_probe'],_=c.probe('final-probe');c.row['evaluation']=c.evaluate('final');c.row['checkpoint']=c.snapshot('final-state')
                c.row.update(status='complete',updates=n,total_C=c.cost(),arm_seconds=time.perf_counter()-rowtime);end()
        result['cleanup_verified']=True;phase='terminal_integrity';validate_freeze(freeze_sha)
        for (i,name),a in tape_arrays.items():require(prior.prior.previous.tape_identity(a)==plan['tapes'][str(i)][name],'Immutable tapes changed')
        work.verify();require(len(cores)==6 and all(c.row['status']=='complete' for c in cores),'Incomplete arm panel')
        require(sum(c.native['reverse_completed'] for c in cores)==work.done['neural_reverse'] and sum(c.native['optimizer_returns'] for c in cores)==work.done['Adam_returns'],'Original backward/Adam counters differ')
        require(work.done['reference_calls']==0 and work.done['reference_directions']==0 and work.done['constructors']==6 and work.done['field_extra']==14 and work.done['Jacobians']==48 and work.done['physical']==384 and work.done['physical_reverse']==672,'Incomplete mandatory work')
        result.update(status='work_and_cleanup_verified',scientific_work_completed=True,terminal_integrity_verified=True)
    except BaseException as exc:
        result.update(status='failed',failure_stage=phase,failure=type(exc).__name__+': '+str(exc))
        for c in cores:
            try:
                c.row['partial_state']=c.snapshot('failure-state')
                if hasattr(c,'pending'):c.row['partial_arrays']=store.numeric(c.prefix+'/failure-evidence.npz',c.pending,1024**2,allow_nonfinite=True)
            except BaseException as capture_error:c.row['partial_capture_error']=str(capture_error)
        raise
    finally:
        import resource
        result.update(wall_seconds=time.perf_counter()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        report=store.json('result.json',result);seal=store.json('finalization/seal.json',dict(report=report,outcome='complete' if result['scientific_work_completed'] else 'failed',external_success_required=True))
        read_report(store.root/seal['path'],seal['sha256'])
    result.update(status='complete',evidence_finalized=True,seal=seal);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--verify-sha256');args=parser.parse_args()
    if args.prepare:
        p=make_plan();v=json.loads((ROOT/VALIDATION).read_text());require(v['exit_code']==v['scientific_queries']==0 and v['new_source_sha256']=={n:p['source_sha256'][n] for n in CODE},'Focused validation differs');p['validation_sha256']=sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(freeze=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)))
    elif args.verify_sha256:print(json.dumps(dict(verified=bool(validate_freeze(args.verify_sha256)),execution_authorized=False)))
    else:print(json.dumps(dict(protocol=PROTOCOL,limits=CAPS,scientific_execution_authorized=False)))


if __name__=='__main__':main()
