"""E9 head-only photometric Adam control; preparation CLI, execution approval required."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments import hybrid_material_e7_v1 as prior
from experiments import spatial_material_e8_v1 as method
from experiments import spatial_pilot_e8r_v1 as legacy
from experiments import head_only_adam_e9_v1 as heads
from experiments import uniform_tapes_e8r_v1 as tape_policy
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report
from infrastructure.runs import validate_run_id

ROOT = prior.ROOT; BASE = 'e21d3dda9da35c24c162bdc0cc2194089e3ef52b'
PARENT = '1ed9a8cb24e2dd9ad76fc3c6d7dd60c315b8d8f17e0bf3b4aeb9c5f72e1a2349'
PROTOCOL = 'spatial-head-only-Adam-e9-v1'
FREEZE = 'docs/protocols/SPATIAL_HEAD_ONLY_E9_V1.freeze.json'
VALIDATION = 'docs/evidence/SPATIAL_HEAD_ONLY_E9_PREPARATION_V1.json'
CODE = ('experiments/head_only_adam_e9_v1.py','experiments/spatial_head_only_e9_v1.py',
        'tests/scientific/test_spatial_head_only_e9_v1.py','docs/protocols/SPATIAL_HEAD_ONLY_E9_V1.md')
CAPS = dict(legacy.CAPS,neural_fit=304,neural_eval=24,neural_reverse=304,Adam_returns=74,field_extra=6,constructors=2,
            physical=0,physical_reverse=0,Jacobians=0,proposals=0,projections=0,linear_solves=0)
RAW = Path('/home/isavand/paper/evidence/material-e8r-2392ca1944dd4ba7a8111200e13598f3')
OLD_REPORT = dict(path='result.json',sha256='34d25fc9820bba74f29e50c87723d61664c0fee47e8466207ffb543b386fcad1')
sha = prior.sha
DESIGN = ('seeds','train_points','held_out_points','truth','initial_latents','spatial_network','optimizer','scene',
          'reference_grid_identities','runtime','arithmetic_runtime','inputs','output_parent','storage','resources',
          'tapes','original_tapes','numerical_preflight','reused_reference_root','reused_reference_report','reused_reference_freeze')


def admit_initials(ids):
    old,_ = read_report(RAW/OLD_REPORT['path'],OLD_REPORT['sha256'])
    require(old['scientific_work_completed'] and old['all_initial_gates_passed'] and old['cleanup_verified'] and old['terminal_integrity_verified'], 'Complete E8R evidence required')
    cases=[];common=None
    for seed in method.SEEDS:
        rows=[a for a in old['arms'] if a['seed']==seed];require(len(rows)==3 and len({a['initial_parameter_identity'] for a in rows})==1, 'E8R initial parity missing')
        initial=next(a for a in rows if a['arm']=='no_retention');control=next(a for a in rows if a['arm']=='Adam')
        require(len(initial['history'])==12 and not any(read_report(RAW/h['path'],h['sha256'])[0]['accepted'] for h in initial['history']), 'Rejected E8R ablation no longer establishes initialization')
        selection=initial['checkpoint'];a=obs.read_numeric(RAW,selection['selection'])
        require(len(a)==98 and not any(n.startswith('adam/') for n in a) and
            obs.parameter_identity({n[10:]:v for n,v in a.items() if n.startswith('parameter/')})==initial['initial_parameter_identity'], 'Saved initial weights differ')
        require(sum(a['parameter/'+n].size for n in heads.final_names())==2064, 'Wrong final-head parameter capacity')
        rng=dict(metadata=selection['RNG'],arrays={n:obs.array_record(a[n]) for n in ('torch_CPU_RNG','numpy_MT19937')})
        if common is None:common=rng
        require(rng==common,'Common saved initial RNG differs')
        signals=[]
        for i in range(4):
            path=RAW/(str(seed)+'-Adam/initial-signals-'+str(i)+'.npz.json');require(str(path) in ids,'Unbound E8R initial signals')
            signals.append(dict(path=str(path.relative_to(RAW)),sha256=ids[str(path)]))
        first,_=read_report(RAW/initial['history'][0]['path'],initial['history'][0]['sha256']);render=obs.read_numeric(RAW,first['arrays'])
        require(render['physical_before'].tolist()==control['initial_physical'] and first['loss_before']==control['initial_loss'], 'Saved initial radiance/physical bridge differs')
        cases.append(dict(seed=seed,initial_state=selection['selection'],RNG=selection['RNG'],parameter_identity=initial['initial_parameter_identity'],
            source_role='E8R_all_rejected_ablation_exact_initial_state',initial_probe=control['initial_probe'],initial_render=first['arrays'],initial_signals=signals,
            initial_loss=control['initial_loss'],initial_physical=control['initial_physical']))
    return cases


class Work:
    def __init__(self): self.attempts = dict.fromkeys(CAPS,0); self.done = dict(self.attempts)
    def attempt(self, key, count=1):
        require(key in CAPS and type(count) is int and count >= 0 and self.attempts[key]+count <= CAPS[key], 'E9 cap: '+key)
        self.attempts[key] += count
    def complete(self, key, count=1):
        require(count >= 0 and self.done[key]+count <= self.attempts[key], 'Unattempted E9 completion')
        self.done[key] += count
    def verify(self): require(self.attempts == self.done, 'Incomplete E9 attempted work')


def make_plan():
    parent = legacy.validate_freeze(PARENT);ids=dict(parent['input_sha256'])
    path=ROOT/'docs/results/spatial_material_e8r_v1/evidence_manifest.json'
    require(path.read_bytes()==subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(ROOT))],cwd=ROOT),'Accepted E8R manifest changed')
    ids[str(path)]=sha(path);manifest=json.loads(path.read_text())
    for item in manifest['raw_files']+manifest['sidecars']:
        p=Path(item['path']);require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'E8R evidence changed');ids[str(p)]=item['sha256']
    ids[str(ROOT/legacy.FREEZE)]=PARENT
    sources=dict(parent['source_sha256']);sources.update({n:sha(ROOT/n) for n in CODE})
    plan={n:parent[n] for n in DESIGN}
    plan.update(schema_version=1,protocol=PROTOCOL,execution_authorized=False,base_commit=BASE,source_sha256=sources,input_sha256=ids,limits=CAPS,
        updates=37,evaluation_states=[12,37],new_arm='head_only_Adam',trainable_names=sorted(heads.final_names()),
        initial_evidence_root=str(RAW),initial_evidence_report=OLD_REPORT,cases=admit_initials(ids))
    return plan


def validate_freeze(selected):
    p,_=read_report(ROOT/FREEZE,selected);parent,_=read_report(ROOT/legacy.FREEZE,PARENT)
    require(p['protocol']==PROTOCOL and p['execution_authorized'] is False and p['limits']==CAPS and p['updates']==37 and
        p['evaluation_states']==[12,37] and p['new_arm']=='head_only_Adam' and p['trainable_names']==sorted(heads.final_names()) and
        all(p[k]==parent[k] for k in DESIGN),'Frozen E9 design differs')
    require(p['initial_evidence_root']==str(RAW) and p['initial_evidence_report']==OLD_REPORT,'Initial evidence selection changed')
    for n,h in p['source_sha256'].items():require(sha(ROOT/n)==h,'Source changed: '+n)
    for n,h in p['input_sha256'].items():require(sha(n)==h,'Input changed: '+n)
    require(p['cases']==admit_initials(p['input_sha256']),'Saved initial identities changed')
    _,checked=tape_policy.preflight(p['original_tapes'],p['numerical_preflight']);require(checked['corrected_identities']==p['tapes'],'E8R tapes changed')
    tape_policy.admit_references(p['reused_reference_root'],p['reused_reference_report'],p)
    v,_=read_report(ROOT/VALIDATION,p['validation_sha256'])
    require(v['exit_code']==v['failed']==v['errors']==v['skipped']==v['scientific_queries']==0 and
        v['new_source_sha256']=={n:p['source_sha256'][n] for n in CODE} and sha(ROOT/v['JUnit']['path'])==v['JUnit']['sha256'],'E9 focused validation changed')
    return p


def validate_review(review, selected):
    require(set(review)=={'phase','scientific_execution_authorized','freeze_sha256','reviewed_commit','reviewed_branch','run_id','output_root','external_watchdog'} and
        review['phase']=='E.9' and review['scientific_execution_authorized'] is True, 'Explicit E9 approval required')
    p = validate_freeze(selected); git = lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    require(review['freeze_sha256']==selected and review['external_watchdog']==dict(wall_seconds=1800,kill_grace_seconds=10), 'E9 freeze/watchdog differs')
    require(review['reviewed_commit']==git('rev-parse','HEAD') and review['reviewed_branch']==git('branch','--show-current')=='feat/neural-material-early-step-e2a' and not git('status','--porcelain'), 'Clean reviewed E9 commit required')
    validate_run_id(review['run_id']); out = Path(review['output_root'])
    require(out.is_absolute() and out.resolve()==out and out.parent==Path(p['output_parent']) and out.name==review['run_id'] and out.name.startswith('material-e9-') and not out.exists(), 'Fresh E9 output required')
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
    store.json('execution-review.json',review); started = time.perf_counter(); cpu_started=time.process_time(); phase='setup'; cores=[]; tape_arrays=prepared_tapes
    result=dict(schema_version=1,protocol=PROTOCOL,freeze_sha256=freeze_sha,status='running',attempts=work.attempts,completed=work.done,
        numerical_preflight=preflight,reused_reference_root=plan['reused_reference_root'],references=[],arms=[],cleanup_verified=False,terminal_integrity_verified=False,scientific_work_completed=False,evidence_finalized=False)
    common=obs.read_numeric(RAW,plan['cases'][0]['initial_state'])
    try:
        with runtime.saved_rng_context(common,plan['cases'][0]['RNG']), runtime.prior.limits() as (check,begin,end):
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
                def __init__(self,spec):
                    self.spec=spec;seed=spec['seed'];self.seed=seed;self.arm='head_only_Adam';self.prefix=str(seed)+'-'+self.arm
                    self.fitF=self.evalF=self.physF=self.revN=self.revP=0
                    self.row=dict(seed=seed,arm=self.arm,status='initializing',history=[]);result['arms'].append(self.row)
                    work.attempt('constructors');self.model=method.make_spatial(torch.tensor(plan['initial_latents'][str(seed)],dtype=torch.float32),seed);work.complete('constructors')
                    selected=heads.freeze(self.model);require(sum(p.numel() for p in selected.values())==2064,'Trainable capacity differs')
                    self.opt=torch.optim.Adam(list(selected.values()),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                    self.view=heads.TrainableView(self.model);self.native=h.counts('N1');self.initial=obs.read_numeric(RAW,spec['initial_state'])
                    self.row['restoration']=heads.restore_initial(self.model,self.opt,self.initial,spec['RNG'],spec['parameter_identity'])
                    self.row['optimizer_parameter_order']=list(selected);self.row['optimizer_effective']=runtime.options(self.opt)
                def cost(self):return 2*(self.fitF+self.physF+self.revN+self.revP)+7*self.evalF
                def probe(self,label):
                    work.attempt('field_extra');m,_,z=method.capture(self.model,torch.tensor(points,dtype=torch.float32));work.complete('field_extra')
                    return store.numeric(self.prefix+'/'+label+'.npz',dict(points=np.asarray(points),physical=m.physical.numpy(),encoding=m.encoding.numpy(),logits=z),1024**2),m.physical.numpy().copy()
                def joint(self,backward=False,label=None):
                    if backward:self.opt.zero_grad()
                    values=[];physical=[];encodings=[];losses=[];zs=[];probabilities=[];features={n:[] for n,_ in method.HEADS};gradients=[]
                    for i in range(4):
                        check();work.attempt('neural_fit');self.fitF+=1
                        with (heads.HeadObserver(self.model) if backward else nullcontext()) as observer:
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
                                work.attempt('neural_reverse');groups=h.backward(self.view,loss/4,self.native);work.complete('neural_reverse');self.revN+=1
                                signals,graph=observer.collect();gradients.append(dict(groups=groups,head_boundaries={n:graph['nodes'][n+'.sigmoid']['analytic_derivative']['zero'] for n,_ in method.HEADS},
                                    shader={n:graph['nodes']['shader.'+n]['loss_gradient'] for n in ('mx','my','albedo','metallic','encoding')}))
                                if label=='initial':
                                    expected=obs.read_numeric(RAW,self.spec['initial_signals'][i])
                                    require(all(k in expected and np.array_equal(v,expected[k]) for k,v in signals.items()),'Saved E8R initial head/shader signal parity failed')
                                if label:store.numeric(self.prefix+'/'+label+'-signals-'+str(i)+'.npz',signals,1024**2)
                    heads.verify_frozen(self.model,self.initial,self.opt)
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
                    return dict(role='head_only_Adam_continuation',RNG=rng,selection=store.numeric(self.prefix+'/'+label+'.npz',a,18*1024**2))
            phase='all_initial_admission';begin()
            for spec in plan['cases']:
                c=Core(spec);cores.append(c);probe,p=c.probe('initial-probe');c.current=c.joint(True,'initial')
                expected=obs.read_numeric(RAW,spec['initial_probe']);render=obs.read_numeric(RAW,spec['initial_render'])
                require(np.array_equal(p,expected['physical']) and c.current['physical'].tolist()==spec['initial_physical'] and
                    c.current['losses'].tolist()==spec['initial_loss'] and np.array_equal(c.current['values'],render['predictions_before']),
                    'Exact saved E8R initial physical/RGB/loss parity failed')
                c.row.update(initial_probe=probe,initial_loss=c.current['losses'].tolist(),initial_physical=c.current['physical'].tolist(),
                    initial_parameter_identity=spec['parameter_identity'],status='initial_verified')
            end();result['all_initial_gates_passed']=True
            for c in cores:
                phase=c.prefix;begin();rowtime=time.perf_counter();cpu_start=time.process_time()
                for k in range(37):
                    check(10);measurement=c.current if k==0 else c.joint(True)
                    work.attempt('Adam_returns');h.step(c.view,c.opt,c.native);work.complete('Adam_returns');heads.verify_frozen(c.model,c.initial,c.opt)
                    c.row['history'].append(dict(update=k+1,pre_update_loss=measurement['losses'].tolist(),pre_update_physical=measurement['physical'].tolist(),gradients=measurement['gradients']))
                    if k+1==12:
                        c.row['equal_update_probe'],_=c.probe('state12-probe');c.row['equal_update_evaluation']=c.evaluate('state12');c.row['state12']=c.snapshot('state12')
                terminal=c.joint(True,'terminal');c.row['terminal_loss']=terminal['losses'].tolist();c.row['terminal_gradients']=terminal['gradients']
                c.row['final_probe'],_=c.probe('final-probe');c.row['evaluation']=c.evaluate('final');c.row['checkpoint']=c.snapshot('final-state')
                c.row.update(status='complete',updates=37,total_C=c.cost(),arm_seconds=time.perf_counter()-rowtime,arm_CPU_seconds=time.process_time()-cpu_start,
                    native_optimizer_backward=c.native,expected_frozen_gradient_parameters=1483520)
                require(c.cost()==692,'Fixed E9 C differs');end()
        result['cleanup_verified']=True;phase='terminal_integrity';validate_freeze(freeze_sha)
        for (i,name),a in tape_arrays.items():require(prior.prior.previous.tape_identity(a)==plan['tapes'][str(i)][name],'Immutable tapes changed')
        work.verify();require(len(cores)==2 and all(c.row['status']=='complete' for c in cores),'Incomplete arm panel')
        require(sum(c.native['reverse_completed'] for c in cores)==work.done['neural_reverse'] and sum(c.native['optimizer_returns'] for c in cores)==work.done['Adam_returns'],'Original backward/Adam counters differ')
        require(work.done==CAPS,'Incomplete fixed E9 work')
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
        result.update(wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu_started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
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
