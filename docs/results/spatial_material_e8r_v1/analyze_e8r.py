"""E8R offline verification/analysis; no Torch or experimental module imports."""
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
HELPER=ROOT/'docs/results/material_optimizer_e6_v1/analyze_e6.py'
assert HELPER.read_bytes()==subprocess.check_output(['git','show','3ef5d53d5d184f5e9dacf81631b06f83a9110e6a:'+str(HELPER.relative_to(ROOT))],cwd=ROOT)
spec=importlib.util.spec_from_file_location('accepted_numpy_helpers',HELPER);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
# Accepted helper installs process-wide blockers for Torch and all experiments.
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import read_report,require

RUN='material-e8r-2392ca1944dd4ba7a8111200e13598f3'
FREEZE='1ed9a8cb24e2dd9ad76fc3c6d7dd60c315b8d8f17e0bf3b4aeb9c5f72e1a2349'
REPORT='34d25fc9820bba74f29e50c87723d61664c0fee47e8466207ffb543b386fcad1'
SEAL='0afb634b7be1268a80ddeacf96d032344171ddc503a46df7b11052ce0941feaf'
RECEIPT='520542d7917ced971f72c722d6bf039e29b3d0f2293a565be8e56508f95e529e'
COMMIT='3ef5d53d5d184f5e9dacf81631b06f83a9110e6a'
sha=helpers.sha;identity=helpers.identity


def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-parent',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.evidence_parent;raw=parent/RUN;out=args.output_dir
    assert raw.is_absolute() and raw.resolve()==raw and out.is_absolute() and out.resolve()==out and out.parent==parent and not out.exists()
    plan,_=read_report(ROOT/'docs/protocols/SPATIAL_MATERIAL_E8R_V1.freeze.json',FREEZE)
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    for n,h in plan['input_sha256'].items():assert sha(n)==h
    r,_=read_report(raw/'result.json',REPORT);seal,_=read_report(raw/'finalization/seal.json',SEAL);receipt,_=read_report(parent/(RUN+'.process.json'),RECEIPT)
    assert r['all_initial_gates_passed'] and r['scientific_work_completed'] and r['cleanup_verified'] and r['terminal_integrity_verified']
    assert seal['report']['sha256']==REPORT and seal['outcome']=='complete' and receipt['process_exit_code']==receipt['retry_attempts']==0
    for key in ('authorization','launch','console'):assert identity(Path(receipt[key]['path']))==receipt[key]
    launch=json.loads(Path(receipt['launch']['path']).read_text());assert launch['commit']==COMMIT and launch['freeze_sha256']==FREEZE and launch['foreground_supervisor']
    assert r['attempts']==r['completed']==plan['limits'] and r['completed']['reference_calls']==r['completed']['reference_directions']==0
    numeric=helpers.numeric_metrics(plan)
    pure=load_functions(ROOT/'experiments/uniform_tapes_e8r_v1.py',('convert','identity','preflight'),dict(np=np,require=require,POLICY=plan['numerical_preflight']['policy'],UPPER=np.nextafter(np.float32(1),np.float32(0))))
    _,preflight=pure['preflight'](plan['original_tapes'],plan['numerical_preflight']);assert preflight==r['numerical_preflight']
    phys=load_functions(ROOT/'experiments/spatial_material_e8_v1.py',('embedding',),dict(np=np));embed=phys['embedding']
    refroot=Path(plan['reused_reference_root']);fine=[]
    assert r['reference_role']=='reused_successful_E8_evidence_not_new_measurements' and r['reused_reference_root']==str(refroot)
    for row in r['references']:fine.append(obs.read_numeric(refroot,row['arrays'])['fine'])
    observations=plan['scene']['observations'];truth=np.asarray(plan['truth']);points=np.array(plan['train_points']+plan['held_out_points'])
    numerical_archives=0
    for path in sorted(raw.rglob('*.npz.json')):
        obs.read_numeric(raw,dict(path=str(path.relative_to(raw)),sha256=sha(path)));numerical_archives+=1
    initial_witness={}
    for seed in plan['seeds']:
        ablation=next(a for a in r['arms'] if a['seed']==seed and a['arm']=='no_retention')
        histories=[read_report(raw/s['path'],s['sha256'])[0] for s in ablation['history']]
        assert len(histories)==12 and not any(v['accepted'] for v in histories)
        state=obs.read_numeric(raw,ablation['checkpoint']['selection'])
        parameters={k[10:]:v for k,v in state.items() if k.startswith('parameter/')}
        assert obs.parameter_identity(parameters)==ablation['initial_parameter_identity']
        assert len(parameters)==84 and sum(v.size for v in parameters.values())==1485584
        initial_witness[seed]=state  # Actual all-rejected endpoint equals recorded original initialization.
    endpoints=[];summaries=[];rounds=[];gradients=[];movement=[];curves=[];variation=[]
    for arm in r['arms']:
        seed=arm['seed'];name=arm['arm'];witness=initial_witness[seed];initial=obs.read_numeric(raw,arm['initial_probe'])['physical']
        assert arm['initial_parameter_identity']==obs.parameter_identity({k[10:]:v for k,v in witness.items() if k.startswith('parameter/')})
        assert np.ptp(initial[:,:2],axis=0).max()>1e-6
        state=obs.read_numeric(raw,arm['checkpoint']['selection'])
        final_params={k[10:]:v for k,v in state.items() if k.startswith('parameter/')}
        allowed={n+'.predictor.6.'+k for n in ('mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor') for k in ('weight_g','weight_v','bias')}
        if name!='Adam':assert all(np.array_equal(v,witness['parameter/'+k]) for k,v in final_params.items() if k not in allowed)
        assert all(np.array_equal(v,witness[k]) for k,v in state.items() if k.startswith('buffer/'))
        delta=np.concatenate([(v-witness['parameter/'+k]).astype(np.float64).reshape(-1) for k,v in final_params.items()])
        movement.append(dict(seed=seed,arm=name,changed_elements=int(np.count_nonzero(delta)),parameter_L2=float(np.linalg.norm(delta)),maximum_absolute=float(abs(delta).max()),
            original_initial_identity_verified=True,non_final_layers_unchanged=(name!='Adam')))
        prefix=str(seed)+'-'+name
        if name=='Adam':
            assert arm['updates']==37 and len(arm['history'])==37 and len(state)==350
            assert all(float(v)==37 for k,v in state.items() if k.endswith('/step'))
            midway=obs.read_numeric(raw,arm['state12']['selection']);assert all(float(v)==12 for k,v in midway.items() if k.endswith('/step'))
            for k,row in enumerate(arm['history']):
                assert row['update']==k+1
                curves.append(dict(seed=seed,arm=name,C=16*(k+1),loss=float(np.mean(row['pre_update_loss'])),state=k))
            curves.append(dict(seed=seed,arm=name,C=608,loss=float(np.mean(arm['terminal_loss'])),state=37))
            assert arm['total_C']==692
        else:
            assert len(state)==98 and not any(k.startswith('adam/') for k in state)
            assert arm['correction_C']==624 and arm['total_C']==698
            records=[read_report(raw/s['path'],s['sha256'])[0] for s in arm['history']];assert len(records)==12
            curves.append(dict(seed=seed,arm=name,C=16,loss=float(np.mean(arm['initial_loss'])),state=0))
            for k,row in enumerate(records):
                assert row['round']==k+1 and row['selected']==k%4 and row['queried']
                a=obs.read_numeric(raw,row['arrays']);selected=row['selected'];b=embed(a['physical_before']);trial=embed(a['trial_physical']);teacher=embed(a['teacher'])
                drift=np.linalg.norm(trial[[i for i in range(4) if i!=selected]]-b[[i for i in range(4) if i!=selected]],axis=1)
                intended=np.linalg.norm(teacher-b[selected]);residual=np.linalg.norm(trial[selected]-teacher)
                np.testing.assert_allclose(drift,row['decision']['other_embedding_drift'],rtol=0,atol=1e-15)
                assert abs(intended-row['decision']['intended_embedding_L2'])<1e-15 and abs(residual-row['decision']['residual_embedding_L2'])<1e-15
                assert row['accepted']==row['decision']['accepted']==all(row['decision']['gates'].values())
                if not row['accepted']:assert np.array_equal(a['physical_after'],a['physical_before'])
                else:assert np.array_equal(a['physical_after'],a['trial_physical'])
                errors=[diagnostic.recovery_errors(p,truth[selected])['Q_relative_frobenius'] for p in (a['physical_before'][selected],a['teacher'],a['trial_physical'][selected])]
                singular=np.linalg.svd(a['J_scaled'],compute_uv=False)
                direction=teacher-b[selected];realized=trial[selected]-b[selected]
                rounds.append(dict(seed=seed,arm=name,round=k+1,selected=selected,accepted=row['accepted'],gates=row['decision']['gates'],
                    intended_embedding_L2=float(intended),relative_projection_residual=float(residual/intended),
                    realized_direction_cosine=float(direction@realized/max(np.linalg.norm(direction)*np.linalg.norm(realized),1e-30)),
                    other_drift_max=float(max(drift)),before_Q=errors[0],teacher_Q=errors[1],trial_Q=errors[2],teacher_improves_Q=errors[1]<errors[0],
                    alpha=row['projection']['alpha'],minimum_singular_value=float(singular[-1]),maximum_singular_value=float(singular[0]),
                    damping_over_minimum_eigenvalue=float(.01/singular[-1]**2),
                    bridge_max_error=row['bridge_max_error'],maximum_projection_condition=max(s['condition'] for s in row['projection']['systems'])))
                expected_loss=row['trial_loss'] if row['accepted'] else row['loss_before']
                assert row['cumulative_C']==16+52*(k+1)
                curves.append(dict(seed=seed,arm=name,C=row['cumulative_C'],loss=float(np.mean(expected_loss)),state=k+1))
        labels=['initial','terminal']
        for phase in labels:
            groups=[];head_norms={};shader_norms={};boundaries=0
            for i in range(4):
                signal=obs.read_numeric(raw,dict(path=prefix+'/'+phase+'-signals-'+str(i)+'.npz.json',sha256=sha(raw/(prefix+'/'+phase+'-signals-'+str(i)+'.npz.json'))))
                for h in ('mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor'):
                    q=signal[h+'.sigmoid'];boundaries+=int(np.count_nonzero((q==0)|(q==1)))
                    head_norms.setdefault(h,[]).append(float(np.linalg.norm(signal[h+'.logits.loss_gradient'])))
                for h in ('mx','my','albedo','metallic','encoding'):shader_norms.setdefault(h,[]).append(float(np.linalg.norm(signal['shader.'+h+'.loss_gradient'])))
            if phase=='terminal':
                g=arm['terminal_gradients'][-1]['groups'];assert all(x['finite'] and x['missing']==0 for x in g.values())
                full_norm=math.sqrt(sum(x['squared_norm'] for x in g.values()))
            else:full_norm=None
            gradients.append(dict(seed=seed,arm=name,phase=phase,exact_sigmoid_boundaries=boundaries,head_logit_gradient_L2=head_norms,shader_gradient_L2=shader_norms,
                terminal_joint_parameter_gradient_L2=full_norm,loss_scaling='Each shader/head gradient is for loss_i/4; final group accumulation is full joint loss'))
        for endpoint in (('equal_update_12',arm['equal_update_evaluation']) if name=='Adam' else (None,None),('final',arm['evaluation'])):
            label,saved=endpoint
            if label is None:continue
            a=obs.read_numeric(raw,saved['arrays']);location_rows=[]
            for i,stored in enumerate(saved['locations']):
                p=a['physical'][i];errors=diagnostic.recovery_errors(p,truth[i]);assert errors==stored['errors'];metrics={}
                for split in ('fit','held_out','relight'):
                    mask=[o['split']==split for o in observations];y=a['predictions'][i,mask];target=fine[i][mask]
                    metrics[split]=dict(errors={k:numeric['radiance_errors'](y[:,j],target[:,j]) for j,k in enumerate(('diffuse','specular','total'))},**helpers.components(y,target))
                    assert metrics[split]==stored['metrics'][split]
                success=numeric['assess_success'](errors,{s:v['errors'] for s,v in metrics.items()});assert success==stored['success']
                row=dict(seed=seed,arm=name,endpoint=label,location=i,spatial_split='train' if i<4 else 'held_out',physical=p.tolist(),errors=errors,metrics=metrics,success=success)
                endpoints.append(row);location_rows.append(row)
            summary=dict(seed=seed,arm=name,endpoint=label,C=234 if label=='equal_update_12' else arm['total_C'],arm_seconds=arm['arm_seconds'] if label=='final' else None,
                fitting_MSE=float(np.mean(arm['history'][12]['pre_update_loss'])) if label=='equal_update_12' else float(np.mean(arm['terminal_loss'])))
            for split,indices in (('train',range(4)),('held_out_spatial',range(4,6))):
                rs=[location_rows[i] for i in indices];q=[v['errors']['Q_relative_frobenius'] for v in rs]
                summary[split]=dict(Q_mean=float(np.mean(q)),Q_worst=float(max(q)),
                    held_view_RGB_L2_mean=float(np.mean([v['metrics']['held_out']['errors']['total']['relative_L2'] for v in rs])),
                    relight_RGB_L2_mean=float(np.mean([v['metrics']['relight']['errors']['total']['relative_L2'] for v in rs])),
                    full_recoveries=sum(v['success']['all_passed'] for v in rs),locations=len(rs))
            summaries.append(summary)
            pred=embed(a['physical']);target=embed(truth);pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
            delta=np.array([pred[i]-pred[j] for i,j in pairs]);actual=np.array([target[i]-target[j] for i,j in pairs])
            variation.append(dict(seed=seed,arm=name,endpoint=label,relative_pairwise_embedding_error=float(np.linalg.norm(delta-actual)/np.linalg.norm(actual)),
                variation_amplitude_ratio=float(np.linalg.norm(delta)/np.linalg.norm(actual)),
                mean_sampled_slope=float(np.mean([np.linalg.norm(pred[i]-pred[j])/np.linalg.norm(points[i]-points[j]) for i,j in pairs])),
                scope='Descriptive all15 six-location pairs; no continuous smoothness or new success criterion'))
    assert len(endpoints)==48 and len(summaries)==8 and len(rounds)==48
    assert sum(v['accepted'] for v in rounds if v['arm']=='guided')==20 and not any(v['accepted'] for v in rounds if v['arm']=='no_retention')
    assert not any(n=='torch' or n.startswith(('torch.','experiments.')) for n in sys.modules)
    out.mkdir()
    def dump(name,value):
        with (out/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    dump('comparison.json',dict(schema_version=1,decision='FIX AND PROCEED',run=RUN,execution_commit=COMMIT,summaries=summaries,endpoints=endpoints,
        rounds=rounds,gradients=gradients,movement=movement,spatial_variation=variation,curves=curves,
        completed=r['completed'],sources_verified=len(plan['source_sha256']),inputs_verified=len(plan['input_sha256']),numerical_archives_verified=numerical_archives,
        external_wall_seconds=receipt['wall_seconds'],peak_RSS_bytes=receipt['peak_RSS_bytes'],additional_scientific_queries=0,new_reference_measurements=0))
    with (out/'endpoints.csv').open('x',newline='') as f:
        keys=['seed','arm','endpoint','location','spatial_split','Q_relative','sorted_width_errors','orientation_degrees','albedo_errors','metallic_error','held_view_RGB_L2','relight_RGB_L2','full_recovery','physical']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        for e in endpoints:
            er=e['errors'];writer.writerow(dict(seed=e['seed'],arm=e['arm'],endpoint=e['endpoint'],location=e['location'],spatial_split=e['spatial_split'],Q_relative=er['Q_relative_frobenius'],
                sorted_width_errors=json.dumps(er['sorted_axis_absolute_errors']),orientation_degrees=er['orientation_degrees'],albedo_errors=json.dumps(er['albedo_absolute_errors']),metallic_error=er['metallic_absolute_error'],
                held_view_RGB_L2=e['metrics']['held_out']['errors']['total']['relative_L2'],relight_RGB_L2=e['metrics']['relight']['errors']['total']['relative_L2'],full_recovery=e['success']['all_passed'],physical=json.dumps(e['physical'])))
    with tempfile.TemporaryDirectory(prefix='nerofactor-e8r-mpl-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'Adam':'#666666','guided':'#2074b0','no_retention':'#d57b32'}
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for j,seed in enumerate((101,103)):
            ax=axes[j]
            for name in colors:
                es=[e for e in endpoints if e['seed']==seed and e['arm']==name and e['endpoint']=='final'];ax.plot(range(6),[e['errors']['Q_relative_frobenius'] for e in es],'-o',label=name,color=colors[name])
            es=[e for e in endpoints if e['seed']==seed and e['endpoint']=='equal_update_12'];ax.plot(range(6),[e['errors']['Q_relative_frobenius'] for e in es],'--',label='Adam state12',color='black',alpha=.5)
            ax.axhline(.05,color='black',ls=':');ax.axvspan(3.5,5.5,color='grey',alpha=.1);ax.set_xticks(range(6));ax.set_title('Seed'+str(seed));ax.set_xlabel('Location; shaded4–5 spatial holdouts');ax.set_ylabel('Relative Q error');ax.legend(fontsize=8)
        fig.savefig(out/'per_location_Q.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for j,seed in enumerate((101,103)):
            rows=[s for s in summaries if s['seed']==seed and s['endpoint']=='final'];x=np.arange(3)
            for i,key in enumerate(('Q_mean','held_view_RGB_L2_mean')):
                ax=axes[i,j]
                for shift,split,label in ((-.18,'train','Training sites'),(.18,'held_out_spatial','Spatial holdouts')):ax.bar(x+shift,[s[split][key] for s in rows],width=.35,label=label)
                ax.set_xticks(x);ax.set_xticklabels([s['arm'] for s in rows],fontsize=8);ax.set_ylabel('Mean '+('Q error' if i==0 else 'held-view RGB relative L2'));ax.set_title('Seed'+str(seed));ax.legend(fontsize=8)
        fig.savefig(out/'physical_and_RGB.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for j,seed in enumerate((101,103)):
            for name,color in colors.items():
                rows=[c for c in curves if c['seed']==seed and c['arm']==name];axes[j].plot([c['C'] for c in rows],[c['loss'] for c in rows],'-o',ms=2,label=name,color=color)
            axes[j].set_yscale('log');axes[j].set_title('Seed'+str(seed));axes[j].set_xlabel('Counted C at fitting measurement');axes[j].set_ylabel('Joint training-tape MSE');axes[j].legend(fontsize=8)
        fig.savefig(out/'objective_work.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for j,seed in enumerate((101,103)):
            for i,name in enumerate(('guided','no_retention')):
                rows=[v for v in rounds if v['seed']==seed and v['arm']==name];ax=axes[i,j]
                for accepted,marker in ((True,'o'),(False,'x')):
                    rs=[v for v in rows if v['accepted']==accepted];ax.scatter([v['round'] for v in rs],[v['other_drift_max'] for v in rs],marker=marker,label='accepted' if accepted else 'rejected',s=30)
                ax.axhline(.05,color='black',ls='--',label='retention cap');ax.set_title(str(seed)+' '+name);ax.set_xlabel('Proposal round');ax.set_ylabel('Max other-site embedding drift');ax.legend(fontsize=8)
        fig.savefig(out/'retention_ablation.png',dpi=170);plt.close(fig)
    dump('evidence_manifest.json',dict(schema_version=1,run=RUN,report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,freeze_sha256=FREEZE,execution_commit=COMMIT,
        reused_reference_root=str(refroot),reused_reference_report=plan['reused_reference_report'],raw_evidence_copied_to_Git=False,
        raw_files=[identity(p) for p in sorted(raw.rglob('*')) if p.is_file()],
        sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')],
        analysis_source=identity(Path(__file__).resolve()),accepted_offline_helper=identity(HELPER)))
    dump('analysis_integrity.json',dict(schema_version=1,scientific_imports_blocked=True,additional_scientific_queries=0,
        endpoint_rows_recomputed=48,physical_proposal_rows=48,numerical_archives_verified=numerical_archives,
        rejected_ablation_endpoints_verify_initial_state=True,files=[identity(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps(dict(status='offline_analysis_complete',output=str(out),endpoints=48,new_scientific_queries=0)))


if __name__=='__main__':main()
