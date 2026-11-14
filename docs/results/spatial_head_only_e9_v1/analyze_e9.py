"""E9 saved-array analysis; Torch and scientific imports are blocked, no model queries."""
import argparse
import ast
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
BASE='e21d3dda9da35c24c162bdc0cc2194089e3ef52b'
HELPER=ROOT/'docs/results/material_optimizer_e6_v1/analyze_e6.py'
assert HELPER.read_bytes()==subprocess.check_output(['git','show',BASE+':'+str(HELPER.relative_to(ROOT))],cwd=ROOT)
spec=importlib.util.spec_from_file_location('accepted_offline_numpy_helpers',HELPER)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)  # Installs process-wide import blocker.
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import read_report,require

RUN='material-e9-5f825d760f92483e9d18070d8c4c0fb3'
FREEZE='836be3d443893a8dccb7379d2607b4fe6a56bc8bab8ac77d807cf1da74b7e7e5'
COMMIT='20807f3c035aadb106a57df363001a6f9c3e1794'
REPORT='6089dfbb3989a8ba47ef2fd6c6a6d8d656b7ff48807caf94bfe8acfa156ca156'
SEAL='881340bd8457fb8d549288d49a15b7955b04d955a2de02f15dbda10bff136c04'
RECEIPT='23a82990cc39b6343f4bf3febbb00aab012f146bb4ec79d2329d902d165efc03'
sha=helpers.sha;identity=helpers.identity


def pure_functions(path,names,namespace):
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-parent',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.evidence_parent;raw=parent/RUN;out=args.output_dir
    assert parent.is_absolute() and parent.resolve()==parent and out.is_absolute() and out.resolve()==out and out.parent==parent and not out.exists()
    plan,_=read_report(ROOT/'docs/protocols/SPATIAL_HEAD_ONLY_E9_V1.freeze.json',FREEZE)
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    for n,h in plan['input_sha256'].items():assert sha(n)==h
    report,_=read_report(raw/'result.json',REPORT);seal,_=read_report(raw/'finalization/seal.json',SEAL)
    receipt,_=read_report(parent/(RUN+'.process.json'),RECEIPT)
    assert report['all_initial_gates_passed'] and report['scientific_work_completed'] and report['cleanup_verified'] and report['terminal_integrity_verified']
    assert seal['report']['sha256']==REPORT and seal['outcome']=='complete' and receipt['process_exit_code']==receipt['retry_attempts']==0 and receipt['foreground_supervisor']
    for key in ('authorization','launch','console'):assert identity(Path(receipt[key]['path']))==receipt[key]
    launch=json.loads(Path(receipt['launch']['path']).read_text());assert launch['commit']==COMMIT and launch['freeze_sha256']==FREEZE and launch['foreground_supervisor']
    assert launch['automatic_retries']==0 and launch['environment_overrides']['CUDA_VISIBLE_DEVICES']==''
    assert report['attempts']==report['completed']==plan['limits'] and len(report['arms'])==2
    assert not report['evidence_finalized'] and report['status']=='work_and_cleanup_verified'  # External receipt + seal, never rewrite original report.
    numeric=helpers.numeric_metrics(plan)
    pure=pure_functions(ROOT/'experiments/uniform_tapes_e8r_v1.py',('convert','identity','preflight'),dict(np=np,require=require,POLICY=plan['numerical_preflight']['policy'],UPPER=np.nextafter(np.float32(1),np.float32(0))))
    _,preflight=pure['preflight'](plan['original_tapes'],plan['numerical_preflight']);assert preflight==report['numerical_preflight']
    embed=pure_functions(ROOT/'experiments/spatial_material_e8_v1.py',('embedding',),dict(np=np))['embedding']
    oldraw=Path(plan['initial_evidence_root']);old,_=read_report(oldraw/plan['initial_evidence_report']['path'],plan['initial_evidence_report']['sha256'])
    oldpath=ROOT/'docs/results/spatial_material_e8r_v1/comparison.json'
    assert oldpath.read_bytes()==subprocess.check_output(['git','show',BASE+':'+str(oldpath.relative_to(ROOT))],cwd=ROOT)
    oldanalysis=json.loads(oldpath.read_text())
    truth=np.asarray(plan['truth']);points=np.asarray(plan['train_points']+plan['held_out_points']);observations=plan['scene']['observations']
    fine=[obs.read_numeric(Path(plan['reused_reference_root']),r['arrays'])['fine'] for r in report['references']]
    assert len(fine)==6 and report['completed']['reference_calls']==report['completed']['reference_directions']==0
    numerical_archives=0
    for path in sorted(raw.rglob('*.npz.json')):
        obs.read_numeric(raw,dict(path=str(path.relative_to(raw)),sha256=sha(path)));numerical_archives+=1
    initials={case['seed']:obs.read_numeric(oldraw,case['initial_state']) for case in plan['cases']}
    allowed=set(plan['trainable_names']);assert len(allowed)==15
    endpoints=[];summaries=[];movement=[];gradients=[];curves=[];variation=[];progress=[]

    def verify_endpoints(base,arm,label,selection,C):
        arrays=obs.read_numeric(base,selection['arrays']);assert arrays['physical'].shape==(6,7) and arrays['predictions'].shape==(6,84,3,3)
        rows=[]
        for i,stored in enumerate(selection['locations']):
            p=arrays['physical'][i];assert p.tolist()==stored['physical'] and stored['location']==i
            errors=diagnostic.recovery_errors(p,truth[i]);assert errors==stored['errors'];metrics={}
            for split in ('fit','held_out','relight'):
                mask=[o['split']==split for o in observations];y=arrays['predictions'][i,mask];target=fine[i][mask]
                metrics[split]=dict(errors={k:numeric['radiance_errors'](y[:,j],target[:,j]) for j,k in enumerate(('diffuse','specular','total'))},**helpers.components(y,target))
                assert metrics[split]==stored['metrics'][split]
            success=numeric['assess_success'](errors,{s:v['errors'] for s,v in metrics.items()});assert success==stored['success']
            row=dict(seed=arm['seed'],arm=arm['arm'],endpoint=label,location=i,spatial_split='train' if i<4 else 'held_out',physical=p.tolist(),errors=errors,metrics=metrics,success=success)
            if arm['arm']!='head_only_Adam':assert row==next(e for e in oldanalysis['endpoints'] if all(e[k]==row[k] for k in ('seed','arm','endpoint','location')))
            endpoints.append(row);rows.append(row)
        summary=dict(seed=arm['seed'],arm=arm['arm'],endpoint=label,C=C,arm_seconds=arm['arm_seconds'] if label=='final' else None,
            arm_CPU_seconds=arm.get('arm_CPU_seconds') if label=='final' else None,
            fitting_MSE=float(np.mean(arm['history'][12]['pre_update_loss'])) if label=='equal_update_12' else float(np.mean(arm['terminal_loss'])))
        for split,indices in (('train',range(4)),('held_out_spatial',range(4,6))):
            rs=[rows[i] for i in indices];summary[split]=dict(locations=len(rs),full_recoveries=sum(r['success']['all_passed'] for r in rs),orientation_identifiable=sum(r['errors']['orientation_identifiable'] for r in rs))
            measures=dict(Q=[r['errors']['Q_relative_frobenius'] for r in rs],axis_max=[max(r['errors']['sorted_axis_absolute_errors']) for r in rs],
                albedo_max=[max(r['errors']['albedo_absolute_errors']) for r in rs],metallic=[r['errors']['metallic_absolute_error'] for r in rs])
            for s in ('fit','held_out','relight'):
                for component in ('diffuse','specular','total'):measures[s+'_'+component+'_RGB_L2']=[r['metrics'][s]['errors'][component]['relative_L2'] for r in rs]
                measures[s+'_component_residual_correlation']=[r['metrics'][s]['normalized_residual_inner_product'] for r in rs]
            for key,values in measures.items():summary[split][key+'_mean']=float(np.mean(values));summary[split][key+'_worst']=float(max(values))
            angles=[r['errors']['orientation_degrees'] for r in rs if r['errors']['orientation_identifiable']]
            summary[split]['orientation_degrees_mean']=float(np.mean(angles)) if angles else None
            summary[split]['orientation_degrees_worst']=float(max(angles)) if angles else None
        summaries.append(summary)
        pred=embed(arrays['physical']);target=embed(truth);pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
        delta=np.array([pred[i]-pred[j] for i,j in pairs]);actual=np.array([target[i]-target[j] for i,j in pairs])
        variation.append(dict(seed=arm['seed'],arm=arm['arm'],endpoint=label,relative_pairwise_embedding_error=float(np.linalg.norm(delta-actual)/np.linalg.norm(actual)),variation_amplitude_ratio=float(np.linalg.norm(delta)/np.linalg.norm(actual)),scope='Descriptive15 pairs across six sampled locations; no continuous smoothness criterion'))

    for base,r in ((oldraw,old),(raw,report)):
        for arm in r['arms']:
            seed=arm['seed'];name=arm['arm'];witness=initials[seed];prefix=str(seed)+'-'+name;is_new=name=='head_only_Adam'
            assert arm['status']=='complete' and arm['initial_parameter_identity']==obs.parameter_identity({k[10:]:v for k,v in witness.items() if k.startswith('parameter/')})
            state=obs.read_numeric(base,arm['checkpoint']['selection'])
            assert all(v.tobytes()==witness[k].tobytes() for k,v in state.items() if k.startswith('buffer/'))
            final={k[10:]:v for k,v in state.items() if k.startswith('parameter/')}
            assert len(final)==84 and sum(v.size for v in final.values())==1485584
            if name!='Adam':assert all(v.tobytes()==witness['parameter/'+n].tobytes() for n,v in final.items() if n not in allowed)
            delta=np.concatenate([(v-witness['parameter/'+n]).astype(float).reshape(-1) for n,v in final.items()]);active_delta=np.concatenate([(v-witness['parameter/'+n]).astype(float).reshape(-1) for n,v in final.items() if n in allowed])
            movement.append(dict(seed=seed,arm=name,changed_elements=int(np.count_nonzero(delta)),parameter_L2=float(np.linalg.norm(delta)),maximum_absolute=float(abs(delta).max()),active_head_L2=float(np.linalg.norm(active_delta)),frozen_other_parameters_verified=(name!='Adam')))
            if name in ('Adam','head_only_Adam'):
                assert arm['updates']==37 and len(arm['history'])==37 and arm['total_C']==692
                assert all(float(v)==37 for k,v in state.items() if k.endswith('/step'))
                midway=obs.read_numeric(base,arm['state12']['selection']);assert all(float(v)==12 for k,v in midway.items() if k.endswith('/step'))
                if is_new:
                    assert len(state)==143 and len([n for n in state if n.startswith('adam/')])==45
                    assert {n.split('/')[1] for n in state if n.startswith('adam/')}==allowed
                    assert all(np.isfinite(v).all() for v in state.values())
                    assert arm['native_optimizer_backward']['optimizer_returns']==37 and arm['native_optimizer_backward']['reverse_completed']==152
                    assert arm['restoration']['exact'] and arm['restoration']['fresh_Adam'] and arm['restoration']['Adam_step']==0
                    for snapshot in (midway,state):assert all(v.tobytes()==witness['parameter/'+n].tobytes() for n,v in final.items() if n not in allowed for v in (snapshot['parameter/'+n],))
                physical=[np.asarray(h['pre_update_physical']) for h in arm['history']]+[np.asarray([loc['physical'] for loc in arm['evaluation']['locations'][:4]])]
                losses=[float(np.mean(h['pre_update_loss'])) for h in arm['history']]+[float(np.mean(arm['terminal_loss']))]
                q=np.array([[diagnostic.recovery_errors(p,truth[i])['Q_relative_frobenius'] for i,p in enumerate(row)] for row in physical])
                ploss=np.array([h['pre_update_loss'] for h in arm['history']]+[arm['terminal_loss']])
                for k,h in enumerate(arm['history']):
                    assert h['update']==k+1
                    curves.append(dict(seed=seed,arm=name,state=k,C=16*(k+1)+42*int(k>=12),loss=losses[k],train_Q_mean=float(q[k].mean())))
                    assert all(g['finite'] and not g['missing'] for groups in h['gradients'] for g in groups['groups'].values())
                curves.append(dict(seed=seed,arm=name,state=37,C=650,loss=losses[-1],train_Q_mean=float(q[-1].mean())))
                progress.append(dict(seed=seed,arm=name,initial_Q_mean=float(q[0].mean()),final_Q_mean=float(q[-1].mean()),last10_Q_change=float(q[-1].mean()-q[-11].mean()),
                    initial_loss=losses[0],final_loss=losses[-1],last10_relative_loss_reduction=float((losses[-11]-losses[-1])/losses[-11]),
                    joint_loss_increasing_transitions=int(sum(np.diff(losses)>0)),mixed_location_loss_change_transitions=int(sum(np.any(np.diff(ploss,axis=0)<0,axis=1)&np.any(np.diff(ploss,axis=0)>0,axis=1))),
                    Q_worsened_location_transitions=int(np.count_nonzero(np.diff(q,axis=0)>0)),Q_transition_total=148,
                    interpretation='Saved joint-update trajectories; simultaneous spatial changes are not an independent causal interference test'))
            else:
                for k,selection in enumerate(arm['history']):
                    row,_=read_report(base/selection['path'],selection['sha256']);loss=row['trial_loss'] if row['accepted'] else row['loss_before']
                    curves.append(dict(seed=seed,arm=name,state=k+1,C=row['cumulative_C'],loss=float(np.mean(loss))))
                curves.insert(len(curves)-12,dict(seed=seed,arm=name,state=0,C=16,loss=float(np.mean(arm['initial_loss']))))
                curves.append(dict(seed=seed,arm=name,state=12,C=656,loss=float(np.mean(arm['terminal_loss']))))
            for phase in ('initial','terminal'):
                record=dict(seed=seed,arm=name,phase=phase,exact_sigmoid_boundaries=0,head_logit_gradient_L2={},shader_gradient_L2={})
                for i in range(4):
                    path=base/(prefix+'/'+phase+'-signals-'+str(i)+'.npz.json');signal=obs.read_numeric(base,dict(path=str(path.relative_to(base)),sha256=sha(path)))
                    if is_new and phase=='initial':
                        oldsignal=obs.read_numeric(oldraw,plan['cases'][(101,103).index(seed)]['initial_signals'][i]);assert all(np.array_equal(v,oldsignal[k]) for k,v in signal.items())
                    for head in ('mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor'):
                        v=signal[head+'.sigmoid'];record['exact_sigmoid_boundaries']+=int(np.count_nonzero((v==0)|(v==1)))
                        record['head_logit_gradient_L2'].setdefault(head,[]).append(float(np.linalg.norm(signal[head+'.logits.loss_gradient'])))
                        assert np.isfinite(signal[head+'.logits.loss_gradient']).all()
                    for n in ('mx','my','albedo','metallic','encoding'):record['shader_gradient_L2'].setdefault(n,[]).append(float(np.linalg.norm(signal['shader.'+n+'.loss_gradient'])))
                record['terminal_joint_parameter_gradient_L2']=math.sqrt(sum(g['squared_norm'] for g in arm['terminal_gradients'][-1]['groups'].values())) if phase=='terminal' else None
                gradients.append(record)
            if name in ('Adam','head_only_Adam'):verify_endpoints(base,arm,'equal_update_12',arm['equal_update_evaluation'],234)
            verify_endpoints(base,arm,'final',arm['evaluation'],arm['total_C'])
    assert len(endpoints)==72 and len(summaries)==12 and len(gradients)==16
    assert not any(n=='torch' or n.startswith(('torch.','experiments.')) for n in sys.modules)
    out.mkdir()
    def dump(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    dump('comparison.json',dict(schema_version=1,decision='FIX AND PROCEED',run=RUN,execution_commit=COMMIT,report_sha256=REPORT,freeze_sha256=FREEZE,
        summaries=summaries,endpoints=endpoints,movement=movement,gradients=gradients,curves=curves,spatial_variation=variation,progress=progress,
        inherited_guidance_rounds=oldanalysis['rounds'],completed=report['completed'],sources_verified=len(plan['source_sha256']),inputs_verified=len(plan['input_sha256']),
        numerical_archives_verified=numerical_archives,external_wall_seconds=receipt['wall_seconds'],external_CPU_seconds=receipt['CPU_seconds'],peak_RSS_bytes=receipt['peak_RSS_bytes'],additional_scientific_queries=0,
        note='Old endpoints independently reverified without rerendering. New curves include charged state12 evaluation cost; accepted E8R curves preserved unchanged.'))
    with (out/'endpoints.csv').open('x',newline='') as f:
        keys=['seed','arm','endpoint','location','spatial_split','Q_relative','sorted_width_errors','orientation_degrees','orientation_identifiable','albedo_errors','metallic_error','fit_RGB_L2','held_view_RGB_L2','relight_RGB_L2','held_diffuse_RGB_L2','held_specular_RGB_L2','held_component_residual_correlation','full_recovery','physical']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        for e in endpoints:
            er=e['errors'];writer.writerow(dict(seed=e['seed'],arm=e['arm'],endpoint=e['endpoint'],location=e['location'],spatial_split=e['spatial_split'],Q_relative=er['Q_relative_frobenius'],
                sorted_width_errors=json.dumps(er['sorted_axis_absolute_errors']),orientation_degrees=er['orientation_degrees'],orientation_identifiable=er['orientation_identifiable'],albedo_errors=json.dumps(er['albedo_absolute_errors']),metallic_error=er['metallic_absolute_error'],
                fit_RGB_L2=e['metrics']['fit']['errors']['total']['relative_L2'],held_view_RGB_L2=e['metrics']['held_out']['errors']['total']['relative_L2'],relight_RGB_L2=e['metrics']['relight']['errors']['total']['relative_L2'],
                held_diffuse_RGB_L2=e['metrics']['held_out']['errors']['diffuse']['relative_L2'],held_specular_RGB_L2=e['metrics']['held_out']['errors']['specular']['relative_L2'],held_component_residual_correlation=e['metrics']['held_out']['normalized_residual_inner_product'],full_recovery=e['success']['all_passed'],physical=json.dumps(e['physical'])))
    with tempfile.TemporaryDirectory(prefix='e9-mpl-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'Adam':'#666666','guided':'#2074b0','head_only_Adam':'#c35025'};labels={'Adam':'Full Adam37','guided':'Guided12','head_only_Adam':'Head Adam37'}
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for col,seed in enumerate((101,103)):
            for name,color in colors.items():
                rows=[e for e in endpoints if e['seed']==seed and e['arm']==name and e['endpoint']=='final']
                for row,key in enumerate(('Q_relative_frobenius','RGB')):
                    values=[e['errors'][key] if row==0 else e['metrics']['held_out']['errors']['total']['relative_L2'] for e in rows]
                    axes[row,col].plot(range(6),values,'-o',label=labels[name],color=color)
                    axes[row,col].axvspan(3.5,5.5,color='grey',alpha=.1);axes[row,col].set_xticks(range(6));axes[row,col].set_xlabel('Location (4–5: spatial holdouts)');axes[row,col].set_title('Seed '+str(seed));axes[row,col].legend(fontsize=8)
            axes[0,col].axhline(.05,color='black',ls=':');axes[0,col].set_ylabel('Relative Q error');axes[1,col].set_ylabel('Held-view RGB relative L2');axes[1,col].axhline(.01,color='black',ls=':')
        fig.savefig(out/'physical_and_RGB.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for col,seed in enumerate((101,103)):
            for name,color in colors.items():
                rows=[c for c in curves if c['seed']==seed and c['arm']==name];axes[col].plot([c['C'] for c in rows],[c['loss'] for c in rows],'-o',ms=2,label=labels[name],color=color)
            axes[col].set_yscale('log');axes[col].set_title('Seed '+str(seed));axes[col].set_xlabel('C charged at saved fitting measurement');axes[col].set_ylabel('Joint training-tape MSE');axes[col].legend(fontsize=8)
        fig.savefig(out/'objective_work.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for col,seed in enumerate((101,103)):
            rows=[s for s in summaries if s['seed']==seed and s['endpoint']=='final' and s['arm'] in colors]
            for row,key in enumerate(('Q_mean','held_out_total_RGB_L2_mean')):
                ax=axes[row,col];x=np.arange(3)
                for offset,split,label in ((-.18,'train','Training sites'),(.18,'held_out_spatial','Spatial holdouts')):ax.bar(x+offset,[s[split][key] for s in rows],width=.35,label=label)
                ax.set_xticks(x);ax.set_xticklabels([labels[s['arm']] for s in rows],fontsize=8);ax.set_title('Seed '+str(seed));ax.set_ylabel('Mean '+('relative Q error' if row==0 else 'held-view RGB relative L2'));ax.legend(fontsize=8)
        fig.savefig(out/'fixed_endpoint_comparison.png',dpi=170);plt.close(fig)
    manifest=dict(schema_version=1,run=RUN,execution_commit=COMMIT,report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,freeze_sha256=FREEZE,
        raw_files=[identity(p) for p in sorted(raw.rglob('*')) if p.is_file()],sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')],
        inherited_E8R_manifest=identity(ROOT/'docs/results/spatial_material_e8r_v1/evidence_manifest.json'),inherited_E8R_comparison=identity(oldpath),
        raw_evidence_copied_to_Git=False,analysis_source=identity(Path(__file__).resolve()),accepted_offline_helper=identity(HELPER),additional_scientific_queries=0)
    dump('evidence_manifest.json',manifest)
    dump('analysis_integrity.json',dict(schema_version=1,scientific_imports_blocked=True,additional_scientific_queries=0,new_endpoint_rows_recomputed=24,inherited_endpoint_rows_reverified=48,
        sources_verified=len(plan['source_sha256']),inputs_verified=len(plan['input_sha256']),numerical_archives_verified=numerical_archives,
        execution_complete_by_seal_and_exit=True,original_prefinalization_report_preserved=True,files=[identity(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps(dict(status='offline_analysis_complete',output=str(out),endpoints=72,new_scientific_queries=0)))


if __name__=='__main__':main()
