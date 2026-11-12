"""Hash-selected, saved-data-only Phase D verification. No Torch/model imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.material_recovery_metrics_v1 import gauge_metrics
from infrastructure.signatures import require, _pairs, _nonfinite, _finite_tree

ROOT=Path(__file__).resolve().parents[1]
BASE='ffed2b075b997e7b6faf37840fdec7e1c489d8ef'
FREEZE='docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json'
FREEZE_SHA='730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad'
AGGREGATE='docs/evidence/NEURAL_MATERIAL_FIELD_VALIDATION_AGGREGATE_V1.json'
AGGREGATE_SHA='f9ef3477a4189958a6e0e16067b35dade62069a11797e90958515996b6cb9a2e'
REFERENCE='docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'
REFERENCE_SHA='3abc274191d0e84c456355a7c351ac1aa94a2c976c1f789979ddb550a4383e89'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def selected_json(path,sha):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'Explicit regular unaliased file required')
    before=path.stat();require(before.st_size<=128*1024**2,'Own JSON report exceeds bounded offline limit')
    require(digest(path)==sha,'Selected JSON hash mismatch before parsing')
    with path.open() as stream:value=json.load(stream,object_pairs_hook=_pairs,parse_constant=_nonfinite)
    _finite_tree(value);after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Selected report changed while reading')
    require(type(value) is dict,'JSON object required')
    return value,dict(path=str(path),sha256=sha,bytes=before.st_size)


def child(root,name):
    root=Path(root);p=root/name
    require(p.is_absolute() and p.resolve()==p and root in p.parents and p.is_file() and not p.is_symlink(),'Unsafe/missing evidence child')
    return p


def inspect_checkpoint(root,selection,row,freeze,run_id):
    path=child(root,selection['manifest']);value,meta=selected_json(path,selection['manifest_sha256'])
    require(value['schema_version']==1 and value['protocol']=='neural-material-recovery-v1' and value['arm']==row['arm'] and
            value['source']==row['initial_model']['source'] and value['model_configuration']==row['initial_model']['configuration'],
            'Checkpoint identity/configuration mismatch')
    require(value['completed_updates']==selection['completed_updates'] and value['optimizer_effective']==freeze['configuration']['optimizer_effective'],
            'Checkpoint count/Adam options mismatch')
    expected=dict(freeze_sha256=FREEZE_SHA,configuration_sha256=freeze['configuration_sha256'],source_sha256=freeze['sha256'],run_id=run_id)
    require(value['execution_provenance']==expected,'Checkpoint provenance mismatch')
    archive=child(path.parent,value['archive'])
    require(digest(archive)==value['archive_sha256'] and archive.stat().st_size==value['archive_bytes'],'Checkpoint archive hash/size mismatch before load')
    entries=value['entries'];require(len(entries)==len({e['key'] for e in entries}),'Duplicate array key')
    catalogue={};pairs=set()
    with np.load(archive,allow_pickle=False) as arrays:
        require(set(arrays.files)=={e['key'] for e in entries}|{'numpy_rng'},'Checkpoint array inventory mismatch')
        for entry in entries:
            pair=(entry['role'],entry['name']);require(pair not in pairs,'Duplicate array role/name');pairs.add(pair)
            a=arrays[entry['key']]
            require(a.dtype.kind!='O' and list(a.shape)==entry['shape'] and 'torch.'+str(a.dtype)==entry['dtype'] and
                    entry['device']=='cpu' and array_sha(a)==entry['sha256'] and bool(np.isfinite(a).all())==entry['finite'],
                    'Checkpoint tensor catalogue mismatch')
            require(entry['finite'],'Unexpected nonfinite state in this selected resource-censored run')
            if entry['role']=='adam':
                if entry['name'].endswith('/step'):require(a.shape==() and float(a)==value['completed_updates'],'Adam counter mismatch')
                if entry['name'].endswith('/exp_avg_sq'):require(bool((a>=0).all()),'Negative second moment')
            catalogue[pair]=entry
        nrng=arrays['numpy_rng'];require(nrng.dtype==np.uint32 and nrng.shape==(624,) and value['numpy_rng']['kind']=='MT19937' and
            array_sha(nrng)==value['numpy_rng']['array_sha256'],'NumPy RNG mismatch')
    parameters=row['initial_model']['parameters'];buffers=row['initial_model']['buffers']
    require(pairs=={('parameter',n) for n in parameters}|{('buffer',n) for n in buffers}|
            {('adam',n+'/'+k) for n in parameters for k in ('step','exp_avg','exp_avg_sq')}|{('torch_rng','CPU')},'Incomplete state roles')
    for name,p in parameters.items():
        for pair in [('parameter',name),('adam',name+'/exp_avg'),('adam',name+'/exp_avg_sq')]:
            require(all(catalogue[pair][k]==p[k] for k in ('shape','dtype','device')),'State parameter/moment layout mismatch')
    for name,b in buffers.items():require(all(catalogue[('buffer',name)][k]==b[k] for k in ('shape','dtype','device','sha256')),'Immutable buffer changed')
    if 'compact_state' in selection:
        compact=selection['compact_state']
        for name,p in compact['parameters'].items():require(catalogue[('parameter',name)]['sha256']==p['sha256'],'Snapshot parameter hash mismatch')
        for name,state in compact['adam'].items():
            for key,v in state.items():require(catalogue[('adam',name+'/'+key)]['sha256']==v['sha256'],'Snapshot Adam hash mismatch')
    return dict(manifest=meta,archive=dict(path=str(archive),sha256=value['archive_sha256'],bytes=value['archive_bytes']),
        purpose=value['purpose'],completed_updates=value['completed_updates'],verified_arrays=len(entries)+1,all_finite=True,allow_pickle=False)


def radiance_errors(prediction,target):
    delta=prediction-target
    return dict(RMSE=float(np.sqrt(np.mean(delta**2))),MAE=float(np.mean(abs(delta))),
                relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(target),1e-4)),max_absolute=float(np.max(abs(delta))))


def recovery_errors(prediction,truth):
    p,t=np.asarray(prediction,np.float64),np.asarray(truth,np.float64);g=gauge_metrics(p[:3],t[:3]);delta=p[3:6]-t[3:6]
    return dict(**g,sorted_axis_relative_errors=(abs(np.sort(p[:2])-np.sort(t[:2]))/np.sort(t[:2])).tolist(),
        orientation_degrees=float(np.rad2deg(g['major_axis_error_radians'])) if g['orientation_identifiable'] else None,
        albedo_absolute_errors=abs(delta).tolist(),albedo_RMSE=float(np.sqrt(np.mean(delta**2))),
        albedo_relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(t[3:6]),1e-4)),
        metallic_absolute_error=float(abs(p[6]-t[6])),metallic_near_boundary=bool(p[6]<=.01 or p[6]>=.99))


def assess(errors,metrics,limits):
    checks=dict(Q=errors['Q_relative_frobenius']<=limits['Q_relative'],
        orientation=errors['orientation_degrees'] is not None and errors['orientation_degrees']<=limits['orientation_degrees'],
        albedo=max(errors['albedo_absolute_errors'])<=limits['max_albedo_absolute'],metallic=errors['metallic_absolute_error']<=limits['metallic_absolute'],
        held_out_radiance=metrics['held_out']['total']['relative_L2']<=limits['held_out_relative_L2'])
    return dict(checks=checks,all_passed=bool(all(checks.values())))


def verify_evaluation(row,reference,freeze):
    evaluation=row['evaluation'];stored=next(x for x in reference['materials'] if x['material']==row['material'])
    target=np.asarray(stored['targets']['fine'],np.float64);prediction=np.asarray(evaluation['predictions'],np.float64)
    require(target.shape==prediction.shape==(84,3,3) and np.isfinite(prediction).all(),'Invalid saved prediction layout')
    require(array_sha(target)==stored['targets']['fine_sha256'] and array_sha(target[:24,2].astype(np.float32))==row['fit_target_float32_sha256'] and
            array_sha(prediction)==evaluation['prediction_sha256'],'Saved target/prediction hash mismatch')
    metrics={};compensation={};obs=reference['scene']['observations']
    for split in ('fit','held_out','relight','relight_seen_views','relight_unseen_views'):
        mask=[o['split']==split if split in ('fit','held_out','relight') else o['split']=='relight' and
            (o['view']<6 if split=='relight_seen_views' else o['view']>=6) for o in obs]
        metrics[split]={name:radiance_errors(prediction[mask,i],target[mask,i]) for i,name in enumerate(('diffuse','specular','total'))}
        if split in ('fit','held_out','relight'):
            d,s,t=((prediction-target)[mask,i] for i in range(3))
            compensation[split]=dict(diffuse_specular_inner_product=float(np.sum(d*s)),
                residual_correlation=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s),1e-30)),
                total_squared_error=float(np.sum(t*t)),component_squared_error=float(np.sum(d*d)+np.sum(s*s)))
    errors=recovery_errors(evaluation['physical'],reference['material_panel'][row['material']]);success=assess(errors,metrics,freeze['configuration']['success'])
    require(metrics==evaluation['radiance_errors'] and errors==evaluation['recovery_errors'] and success==evaluation['success'] and
            compensation==evaluation['component_compensation'],'Saved endpoint metrics differ from offline recomputation')
    return dict(physical=evaluation['physical'],encoding=evaluation['encoding'],recovery_errors=errors,radiance_errors=metrics,
        success=success,component_compensation=compensation,prediction_sha256=evaluation['prediction_sha256'],offline_recomputed=True)


def verify_controls(rows,reference):
    indexed={(x['material'],x['seed'],x['arm']):x for x in rows};direct=[];paired=[]
    for row in rows:
        if row['arm']=='D0':
            require(row['status']=='complete' and row['prefix_reproduction_passed'] and row['prefix_verified_through']==1000,'D0 reproduction incomplete')
            old=next(x for x in reference['Adam_runs'] if (x['material'],x['seed'])==(row['material'],row['seed']))
            logs={x['update']:x for x in row['history']};checked=0
            for entry in old['history']:
                actual=logs[entry['update']];require(actual['loss']==entry['loss'] and actual['gradient_norm']==entry['gradient_norm'],'Offline D0 prefix log mismatch');checked+=1
            for step in ('250','1000'):
                new,prior=row['snapshots'][step],old['snapshots'][step];state=new['compact_state']
                require(state['parameters']['latent']==prior['unconstrained'] and state['adam']['latent']==prior['adam'] and
                        state['optimizer_effective']==prior['optimizer_effective'] and new['objective']==prior['objective'] and
                        new['material']['physical']['values'][0]==prior['physical']['values'],'Offline D0 snapshot mismatch')
            direct.append(dict(material=row['material'],seed=row['seed'],runtime_full_prefix_passed=True,
                offline_exact_logged_loss_gradient_records=checked,offline_exact_snapshots=[250,1000],
                offline_scope='Recorded prefix losses/gradient norms, saved parameter/moment/options/physical/objective snapshots; no update replay'))
        elif row['arm']=='N0' and row['completed_updates']:
            counterpart=indexed[(row['material'],row['seed'],'D0')];oldlogs={x['update']:x for x in counterpart['history']}
            for log in row['history']:
                require(all(log[k]==oldlogs[log['update']][k] for k in ('loss','gradient_norm','physical_before_update')),'Offline paired history mismatch')
            for step,snapshot in row['snapshots'].items():require(snapshot['compact_state']==counterpart['snapshots'][step]['compact_state'],'Offline paired snapshot mismatch')
            complete=row['status']=='complete'
            if complete:require(row['trajectory_sha256']==counterpart['trajectory_sha256'] and row['final_state']==counterpart['final_state'] and
                row['evaluation']==counterpart['evaluation'],'Complete N0 paired state/trajectory/evaluation mismatch')
            paired.append(dict(material=row['material'],seed=row['seed'],completed_updates=row['completed_updates'],
                exact_logged_pairs=len(row['history']),exact_snapshots=list(row['snapshots']),full_trajectory_match=complete,
                final_state_and_evaluation_match=complete,scope='Full equality verified from runtime digest and saved endpoint' if complete else
                    'Censored partial logs/snapshots match; no full trajectory or endpoint claim'))
    return dict(D0=direct,N0=paired,per_update_values_archived=False,trajectory_digest_scope='Runtime instrumentation binds every transition; offline comparison of final digests, not a transition replay')


def statistics(rows):
    result={}
    for arm in ('D0','N0','N1'):
        selected=[x for x in rows if x['arm']==arm];complete=[x for x in selected if x['status']=='complete']
        result[arm]=dict(planned=len(selected),eligible_complete=len(complete),physical_pass=sum(x['evaluation']['success']['all_passed'] for x in complete),
            held_out_total_RGB_pass=sum(x['evaluation']['success']['checks']['held_out_radiance'] for x in complete),
            censored=sum(x['status']=='censored' for x in selected),unstarted=sum(x['completed_updates']==0 for x in selected),
            per_material={m:dict(planned=3,complete=sum(x['material']==m for x in complete),physical_pass=sum(x['material']==m and
                x['evaluation']['success']['all_passed'] for x in complete)) for m in ('M1','M2','M3')})
    return result


def analyze(report_path,report_sha,control):
    report_path=Path(report_path);control=Path(control);raw,raw_id=selected_json(report_path,report_sha)
    frozen,freeze_id=selected_json(ROOT/FREEZE,FREEZE_SHA);_,aggregate_id=selected_json(ROOT/AGGREGATE,AGGREGATE_SHA)
    reference,reference_id=selected_json(ROOT/REFERENCE,REFERENCE_SHA)
    require(all(digest(ROOT/name)==sha for name,sha in frozen['sha256'].items()),'Frozen source inventory changed')
    files={p.name:dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(control.iterdir()) if p.is_file()}
    process,_=selected_json(child(control,'process-outcome.json'),files['process-outcome.json']['sha256'])
    start,_=selected_json(child(control,'process-start.json'),files['process-start.json']['sha256'])
    authorization,_=selected_json(child(control,'authorization.json'),process['review_sha256'])
    preflight,_=selected_json(child(control,'preflight.json'),files['preflight.json']['sha256'])
    require(preflight['status']=='passed' and preflight['scientific_operations']==preflight['model_constructions']==preflight['optimizer_steps']==0 and
            preflight['output_allocated'] is False and preflight['global_RNG_unchanged'] is True and
            preflight['review']['sha256']==process['review_sha256'],'Preflight evidence mismatch')
    require(process['invocations']==start['invocations']==authorization['execution_invocations_authorized']==1 and
            process['run_id']==start['run_id']==raw['run_id']==authorization['approved_run_id'] and
            process['output']==str(report_path.parent)==authorization['approved_output_root'],'Execution identity/invocation mismatch')
    require(process['console_sha256']==files['console.log']['sha256'] and start['runner_sha256']==files['execute.py']['sha256'] and
            start['external_watchdog']==authorization['external_watchdog']==frozen['authorization']['external_watchdog'],'Execution watchdog/runner/console mismatch')
    require(start['command'][:4]==['timeout','--signal=TERM','--kill-after=10s','28800s'] and
            start['command'][-1]==process['review_sha256'],'Invocation not under frozen hash-selected watchdog')
    require(raw['repository']['HEAD']==BASE and raw['repository']['tracked_status']==[] and raw['reviewed_freeze_sha256']==FREEZE_SHA and
            authorization['freeze_sha256']==FREEZE_SHA and authorization['aggregate_evidence']['sha256']==AGGREGATE_SHA and
            authorization['configuration_sha256']==frozen['configuration_sha256'] and raw['settings']==frozen['settings'] and
            raw['tapes']==frozen['tapes'] and authorization['source_inventory_sha256']==identity(frozen['sha256']),'Frozen execution identity mismatch')
    rows=raw['outcomes'];cfg=frozen['configuration'];expected={(m,s,a) for m in cfg['materials'] for s in cfg['seeds'] for a in cfg['arms']}
    require(len(rows)==27 and {(x['material'],x['seed'],x['arm']) for x in rows}==expected,'Missing/duplicate planned histories')
    require(len(raw['initial_admission'])==9 and all(x['passed'] and all(x['checks'].values()) and all(x['D0_N0_exact'].values()) for x in raw['initial_admission']),
            'Incomplete initial admission')
    counters={k:0 for k in frozen['expected_total_work']};summaries=[];checkpoints=[]
    for row in rows:
        key=row['material']+'/'+str(row['seed'])+'/'+row['arm'];require(row['initial_model']==frozen['initial_models'][key],'Frozen initial identity mismatch')
        work=row['work'];limit=cfg['work_per_history'][row['arm']]
        require(set(work)==set(limit) and all(type(v) is int and 0<=v<=limit[k] for k,v in work.items()),'Invalid work counters')
        for k,v in work.items():counters[k]+=v
        require(work['optimizer_attempts']==work['optimizer_returns']==row['completed_updates'] and
                work['shader_attempts']==work['shader_completed']==work['fit_forwards']+work['evaluation_forwards'] and
                work['field_attempts']==work['field_completed']==work['shader_completed']+2 and
                work['reverse_attempts']==work['reverse_completed']==row['completed_updates']+1+work['terminal_forwards'] and
                work['attempted_points']==work['completed_points']==96*work['fit_forwards']+336*work['evaluation_forwards'] and
                work['diffuse_directions']==512*work['completed_points'] and work['specular_directions']==1024*work['completed_points'],
                'Inconsistent selected-run work accounting')
        if row['status']=='complete':require(work==limit and set(row['snapshots'])=={'250','1000','5000'} and
            row['final_state']==row['snapshots']['5000']['compact_state'],'Incomplete/inconsistent completed endpoint')
        selected=list(row.get('snapshots',{}).values())+([row['failure_checkpoint']] if 'failure_checkpoint' in row else [])
        states=[inspect_checkpoint(report_path.parent,v,row,frozen,raw['run_id']) for v in selected];checkpoints.extend(states)
        summary={k:row[k] for k in ('material','seed','arm','status','completed_updates','work')}
        summary.update(optimization_status='unstarted' if row['completed_updates']==0 else row['status'],
            initial_model_identity_sha256=identity(row['initial_model']),checkpoints=states,
            active_wall_seconds=row.get('active_wall_seconds'),CPU_seconds=row.get('CPU_seconds'),
            termination=row.get('termination'),failure=row.get('failure'),last_logged_update=row['history'][-1]['update'] if row.get('history') else None,
            final_fit_MSE=row.get('final_fit_MSE'),final_gradient_groups=row.get('final_gradient_groups'),
            trajectory_sha256=row.get('trajectory_sha256'))
        if 'evaluation' in row:summary['evaluation']=verify_evaluation(row,reference,frozen)
        summaries.append(summary)
    require(len(checkpoints)==len(list((report_path.parent/'checkpoints').glob('*.json')))==len(list((report_path.parent/'checkpoints').glob('*.npz'))),'Unreferenced checkpoint artifacts')
    controls=verify_controls(rows,reference);counts=Counter(x['status'] for x in rows)
    require(raw['status']=='failed' and process['returncode']==1 and not process['watchdog_expired'] and counts==dict(complete=15,censored=1,admitted=11),
            'Selected incomplete-run classification changed')
    require(all(raw[k] is True for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')),
            'Runtime restoration/RNG integrity failed')
    inventory={str(p.relative_to(report_path.parent)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(report_path.parent.rglob('*')) if p.is_file()}
    return dict(schema_version=1,protocol='neural-material-recovery-v1-phase-d-offline-report',status='incomplete_resource_censored_control',
        scientifically_complete=False,source_commit=BASE,run_id=raw['run_id'],output_root=str(report_path.parent),
        source_report=raw_id,freeze=freeze_id,aggregate=aggregate_id,accepted_3D1=reference_id,control_artifacts=files,
        process=process,configuration=cfg,source_inventory_verified=len(frozen['sha256']),initial_identities_verified=27,initial_admission_groups_passed=9,
        offline_implementation=dict(path='experiments/analyze_neural_material_recovery_v1.py',sha256=digest(Path(__file__).resolve())),
        statuses=dict(completed=15,numerically_failed=0,censored=1,unstarted=11,
            valid_recovery_pass=sum(x['status']=='complete' and x['evaluation']['success']['all_passed'] for x in summaries),
            valid_recovery_fail=sum(x['status']=='complete' and not x['evaluation']['success']['all_passed'] for x in summaries)),
        failure=dict(raw_global_status=raw['status'],raw_global_category=raw['failure_category'],message=raw['failure'],
            root_cause='N0 M3/101 active resource cap with endpoint reserve; incomplete required control stopped execution globally',
            active_wall_seconds=next(x['active_wall_seconds'] for x in rows if x['status']=='censored'),
            no_observed_control_mismatch=True,neural_optimization_updates=0),
        total_work=counters,planned_complete_work=frozen['expected_total_work'],control_verification=controls,statistics=statistics(rows),outcomes=summaries,
        runtime={k:raw[k] for k in ('wall_seconds','CPU_seconds','peak_RSS_bytes','global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')},
        storage=dict(output_bytes=sum(x['bytes'] for x in inventory.values()),output_file_count=len(inventory),
            checkpoint_count=len(checkpoints),checkpoint_bytes=sum(x['archive']['bytes'] for x in checkpoints)),output_inventory=inventory,
        offline_work=dict(shader_forwards=0,derivatives=0,optimizer_steps=0,reference_generation=0,checkpoint_model_restorations=0),
        limitations=['No N1 optimizer history or endpoint; the primary representation question is unresolved',
            'Unstarted means optimizer-unstarted: all27 models and their initial shader/backward checks executed',
            'Completed N0 cases omit M3; do not present2/6 as full-panel recovery or pair missing endpoints',
            'Runtime baseline_reproduction label records an incomplete control, not a numerical mismatch',
            'Every-transition tensors were not separately archived; offline checks compare the frozen runtime digests and saved records',
            'Censored latest state is diagnostic-only; no resume or retry authorized'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--control',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.report,args.report_sha256,args.control)
    require(args.output.resolve()!=args.report.resolve() and not args.output.exists(),'No evidence/report overwrite')
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],output=str(args.output),sha256=digest(args.output),statistics=result['statistics']),indent=2))


if __name__=='__main__':main()
