"""E4 recorded-data analysis: no Torch, original model, shader or optimizer imports."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import subprocess
import sys


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name=='torch' or name.startswith(('torch.','experiments.')):raise RuntimeError('Offline analysis forbids '+name)


sys.meta_path.insert(0,NoScience());REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
import numpy as np
from infrastructure import neural_material_observer_e2a as audit
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require,read_report

BASE='f8cde24889fc44d0d2e25a08e625907662f08b87'
RUN='material-e4-5290efddb03c4842837176191a90558d'
FREEZE='a3318d2e2633d5fd7e952224e65293ec0fb810e4de04f2c38f887e2a3dadf643'
REPORT='d55a0303b33591f289fab7e3969525b8670281f5ccd06227d1d7b1086dd931df'
SEAL='fdb891aed767728c2fcf3a27276cc12490edd6e6810122ef0641141836b35522'
RECEIPT='a29bac04fb1c77e371d25ab0fe21eb71746e2454a84399ea2722518a5bc131b1'
CASES=('M1/101','M1/103','M2/103','M3/101','M3/103')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def norm(v):return math.sqrt(math.fsum(float(x)**2 for x in v))


def accepted_file(path):
    p=REPO/path;expected=subprocess.check_output(['git','show',BASE+':'+path],cwd=REPO)
    assert p.read_bytes()==expected,'Accepted baseline changed: '+path
    return p


def numeric_functions(plan):
    path='experiments/run_known_material_recovery_v1.py';assert sha(REPO/path)==plan['source_sha256'][path]
    tree=ast.parse((REPO/path).read_text());cfg=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets))
    success=next(ast.literal_eval(v) for k,v in zip(cfg.keys,cfg.values) if isinstance(k,ast.Constant) and k.value=='success')
    ns=dict(np=np,require=require,SETTINGS=dict(success=success));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('radiance_errors','assess_success')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REPO/path),'exec'),ns);return ns


def local_Q_jacobian(physical,width_gain,angle_gain):
    """Ideal coordinate-level 3x4 Jacobian in a Frobenius-isometric Q basis.

    NOT a measured neural/observation Jacobian. Own-state comparisons mix
    physical-state and coordinate effects; no rank/success threshold is added.
    """
    x,y,t=physical[:3];c,s=np.cos(t),np.sin(t);rotation=np.array([[c,-s],[s,c]])
    vec=lambda a:np.array([a[0,0],np.sqrt(2)*a[0,1],a[1,1]])
    dx=vec(2*x*np.outer(rotation[:,0],rotation[:,0]));dy=vec(2*y*np.outer(rotation[:,1],rotation[:,1]))
    dt=vec((x*x-y*y)*rotation@np.array([[0.,1.],[1.,0.]])@rotation.T)
    jac=np.column_stack((dx*width_gain[0],dy*width_gain[1],dt*angle_gain[0],dt*angle_gain[1]))
    sv=np.linalg.svd(jac,compute_uv=False)
    return dict(singular_values=sv.tolist(),condition_ratio=float(sv[0]/sv[-1]) if x!=y and sv[-1]>0 else None,
        structural_rank=2 if x==y else 3,limitation='Ideal output-coordinate Q Jacobian; not neural/observation information; states differ across methods')


def baseline_curves(parent):
    folder='docs/results/neural_material_e3_complete_v1/'
    with accepted_file(folder+'trajectories.csv').open() as f:rows=[r for r in csv.DictReader(f) if float(r['lr'])==.001]
    with accepted_file(folder+'gradient_routes.csv').open() as f:routes=[r for r in csv.DictReader(f) if float(r['lr'])==.001]
    with accepted_file(folder+'adam_displacements.csv').open() as f:motions=[r for r in csv.DictReader(f) if float(r['lr'])==.001]
    curves=[];head={};moves=[]
    for r in routes:head[r['case'],int(r['state']),r['head']]=dict(q=np.asarray(json.loads(r['sigmoid']),np.float64).ravel(),logits=np.asarray(json.loads(r['logits']),np.float64).ravel())
    for r in rows:
        curves.append(dict(case=r['case'],method='N1',state=int(r['state']),loss=float(r['loss']),Q_relative=float(r['Q_relative']),physical=json.loads(r['physical']),
            gradient_norm=float(r['gradient_norm']),nonzero=int(r['gradient_nonzero']),boundary_channels=int(r['boundary_channels']),shared_gradient=float(r['shared_feature_gradient_L2'])))
    for r in motions:moves.append(dict(case=r['case'],method='N1',update=int(r['update']),movement_L2=float(r['displacement_L2']),gradient_norm=float(r['gradient_norm'])))
    oldroot=parent/'neural-e2b-d1f760d630964a2786ff45bde7438825'
    old,_=read_report(oldroot/'result.json','bb12d3627e778634da17da7a2b484ebe608a88d94aab0bc6c033cd58a1f15373')
    for a in old['outcomes']:
        case=a['material']+'/'+str(a['seed'])
        if a['lr']!=.001 or case not in CASES:continue
        for state,selection in enumerate(a['records'][:10]):
            row,_=read_report(oldroot/selection['path'],selection['sha256']);nodes=row['observations']['nodes']
            boundary=0
            for h in audit.HEADS:
                q=np.asarray(nodes[h+'.sigmoid']['values'],np.float64).ravel();z=np.asarray(nodes[h+'.logits']['values'],np.float64).ravel()
                head[case,state,h]=dict(q=q,logits=z);boundary+=int(np.sum((q==0)|(q==1)))
            curves.append(dict(case=case,method='N1',state=state,loss=row['loss'],Q_relative=row['physical']['errors']['Q_relative_frobenius'],physical=row['physical']['physical'],
                gradient_norm=row['original_gradient_norm'],nonzero=sum(v['nonzero'] for v in row['original_gradient_groups'].values()),boundary_channels=boundary,shared_gradient=nodes['shared_features']['loss_gradient']['L2']))
            moves.append(dict(case=case,method='N1',update=state+1,movement_L2=norm(v['L2'] for v in row['actual_displacement']['per_group'].values()),gradient_norm=row['original_gradient_norm']))
    assert len(curves)==505 and len(moves)==500
    return curves,head,moves


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-parent',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    parent=args.evidence_parent;root=parent/RUN;out=args.output_dir
    assert root.is_absolute() and root.resolve()==root and out.is_absolute() and out.resolve()==out and out.parent==parent and not out.exists()
    plan,_=read_report(REPO/'docs/protocols/MATERIAL_PARAMETERIZATION_E4_V1.freeze.json',FREEZE)
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h
    for p,h in plan['input_sha256'].items():assert sha(p)==h
    report,_=read_report(root/'result.json',REPORT);seal,_=read_report(root/'finalization/seal.json',SEAL);receipt,_=read_report(parent/(RUN+'.process.json'),RECEIPT)
    assert report['scientific_work_completed'] and report['all_initial_gates_passed'] and report['cleanup_verified'] and report['terminal_integrity_verified']
    assert seal['outcome']=='complete' and seal['report']['sha256']==REPORT and receipt['process_exit_code']==0 and receipt['retry_attempts']==0
    assert report['totals']==plan['total_work']
    for k in ('authorization','launch','console'):assert identity(Path(receipt[k]['path']))==receipt[k]
    panel=json.loads(accepted_file('docs/results/neural_material_e3_complete_v1/comparison.json').read_text())
    baselines={e['case']:e for e in panel['endpoints'] if e['lr']==.001}
    reference_path=REPO/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'
    assert reference_path.stat().st_size==19679861 and sha(reference_path)==plan['inputs']['accepted_3D1_SHA256']
    reference=json.loads(reference_path.read_bytes());observations=reference['scene']['observations'];numeric=numeric_functions(plan)
    validation,_=read_report(REPO/'docs/evidence/MATERIAL_PARAMETERIZATION_E4_PREPARATION_V1.json',plan['validation_sha256'])
    curves,baseline_heads,motions=baseline_curves(parent);measurements=[];gradient_routes=[];conditioning=[];endpoints=[];snapshots=[];initial_gates=[]
    e2b=parent/'neural-e2b-d1f760d630964a2786ff45bde7438825';signal_count=0;chain_errors=[]
    for spec,a in zip(plan['arms'],report['outcomes']):
        case=a['material']+'/'+str(a['seed']);base=baselines[case];truth=np.asarray(plan['inputs']['targets'][a['material']]['truth'],np.float64)
        assert a['status']=='complete' and a['work']==a['attempts']==spec['work'] and a['hooks_removed']
        assert a['native_fit']['optimizer_returns']==100 and a['native_fit']['reverse_completed']==spec['work']['backwards']
        for k in ('10','100'):assert a['native_evaluations'][k]['shader_completed']==1
        initial=audit.read_numeric(e2b,spec['checkpoint']);initial_gate=json.loads((root/(a['material'].lower()+'-'+str(a['seed']))/'initial-admission.json').read_text())
        assert initial_gate['initial_gate_passed'] and initial_gate['physical']==spec['expected_physical'] and initial_gate['loss']==spec['expected_loss']
        for k,g in initial_gate['gradient_groups'].items():
            norm0=math.sqrt(spec['expected_groups'][k]['squared_norm']);actual=math.sqrt(g['squared_norm'])
            assert abs(actual-norm0)<=plan['initial_gate']['gradient_group_norm_atol']+plan['initial_gate']['gradient_group_norm_rtol']*norm0
        initial_gates.append(dict(case=case,physical_and_loss=True,matching_parameter_identity=spec['parameter_identity']))
        j0=np.asarray(next(v for v in validation['coordinate_only_saved_buffer_checks'] if (v['material'],v['seed'])==(a['material'],a['seed']))['orientation_tangent'],np.float64).ravel()
        path=root/a['trajectory']['path'];assert sha(path)==a['trajectory']['sha256'];previous=None;updates=0;unique={};rows=[]
        for line in path.read_bytes().splitlines(keepends=True):
            row=json.loads(line);assert row['previous_sha256']==previous;previous=hashlib.sha256(line).hexdigest();rows.append(row)
            if row['kind']=='evaluate':assert row['endpoint']==a['endpoints'][str(row['state'])];continue
            state=row['state'];assert row['post_state']==state+int(row['update']) and row['work']['updates']==row['post_state']
            assert row['gradient_norm']==math.sqrt(sum(row['gradient_groups'][g]['squared_norm'] for g in audit.GROUPS))
            p=np.asarray(row['physical'],np.float32);assert p.astype(np.float64).tolist()==row['physical'] and diagnostic.recovery_errors(p,truth)==row['physical_errors']
            boundary=0
            for h in ('mx_predictor','my_predictor','metallic_predictor','kd_predictor'):
                node=row['nodes'][h+'.sigmoid'];q=np.asarray(node['values'],np.float64);boundary+=int(np.sum((q==0)|(q==1)))
            if 'signals' in row:
                values=audit.read_numeric(root,row['signals']);signal_count+=1
                for n,v in row['nodes'].items():
                    assert audit.array_record(values[n])==v['value'] and audit.array_record(values[n+'.loss_gradient'])==v['gradient']
                if state==0:
                    expected=audit.read_numeric(e2b,spec['signals'])
                    for n in ('mx','my','encoding','metallic','albedo'):assert values['shader.'+n].tobytes()==expected['shader.'+n].tobytes()
            entry=dict(case=case,method='AS1',state=state,loss=row['loss'],Q_relative=row['physical_errors']['Q_relative_frobenius'],physical=row['physical'],
                gradient_norm=row['gradient_norm'],nonzero=sum(g['nonzero'] for g in row['gradient_groups'].values()),boundary_channels=boundary,
                shared_gradient=row['nodes']['shared_features']['gradient']['L2'])
            measurements.append(dict(entry,index=row['index'],update=row['update']));unique[state]=entry
            for h in ('mx_predictor','my_predictor'):
                q=float(row['nodes'][h+'.sigmoid']['values'][0][0]);d=float(row['nodes'][h+'.delta']['values'][0][0])
                upstream=row['nodes'][h+'.sigmoid']['gradient']['L2'];observed=row['nodes'][h+'.delta']['gradient']['L2']
                factor=q*(1-q)/np.sqrt(1+d*d);expected=upstream*factor
                chain_errors.append(abs(observed-expected)/max(abs(expected),1e-300))
                gradient_routes.append(dict(case=case,state=state,index=row['index'],head=h,raw_delta=d,q=q,coordinate_derivative=factor,
                    measured_upstream_L2=upstream,measured_raw_L2=observed,ideal_chain_raw_L2=expected,chain_relative_error=chain_errors[-1]))
            if row['update']:
                updates+=1;delta=row['actual_parameter_movement'];motions.append(dict(case=case,method='AS1',update=row['post_state'],
                    movement_L2=norm(g['L2'] for g in delta['per_group'].values()),gradient_norm=row['gradient_norm']))
                if updates>1:assert prior_identity==delta['before_identity']
                prior_identity=delta['after_identity']
        assert updates==100 and len(rows)==spec['work']['forwards'] and sorted(unique)==list(range(101));curves.extend(unique.values())
        for state,snapshot in a['checkpoints'].items():
            values=audit.read_numeric(root,snapshot['selection']);n=int(state);assert len(values)==350
            assert all(float(v)==n for k,v in values.items() if k.endswith('/step')) and snapshot['RNG']==spec['RNG']
            assert all(values[k].tobytes()==v.tobytes() for k,v in initial.items() if k.startswith('buffer/') or k in ('numpy_MT19937','torch_CPU_RNG'))
            transition=next(row for row in rows if row['kind']=='fit' and row['update'] and row['post_state']==n)
            assert audit.parameter_identity({k[10:]:v for k,v in values.items() if k.startswith('parameter/')})==transition['actual_parameter_movement']['after_identity']
            snapshots.append(dict(case=case,state=n,exact_parameter_identity=True,Adam_step=n,buffers_and_RNG_unchanged=True))
        target_info=next(v for v in reference['materials'] if v['material']==a['material']);target=np.asarray(target_info['targets']['fine'],np.float64)
        assert hashlib.sha256(target.tobytes()).hexdigest()==plan['inputs']['targets'][a['material']]['fine_sha256']
        for label,e in a['endpoints'].items():
            predictions=np.asarray(e['predictions'],np.float64);assert predictions.shape==target.shape==(84,3,3)
            assert hashlib.sha256(predictions.tobytes()).hexdigest()==e['prediction_sha256']
            recomputed={}
            for split in ('fit','held_out','relight','relight_seen_views','relight_unseen_views'):
                mask=[o['split']==split if split in ('fit','held_out','relight') else o['split']=='relight' and (o['view']<6 if split=='relight_seen_views' else o['view']>=6) for o in observations]
                recomputed[split]={name:numeric['radiance_errors'](predictions[mask,i],target[mask,i]) for i,name in enumerate(('diffuse','specular','total'))}
            assert recomputed==e['radiance_errors'] and diagnostic.recovery_errors(np.asarray(e['physical'],np.float32),truth)==e['recovery_errors']
            assert numeric['assess_success'](e['recovery_errors'],recomputed)==e['success']
        e=a['endpoints']['100'];er=e['recovery_errors'];endpoint=dict(case=case,baseline=base,
            AS1=dict(loss100=unique[100]['loss'],Q100=er['Q_relative_frobenius'],heldout100=e['radiance_errors']['held_out']['total']['relative_L2'],
                relight100=e['radiance_errors']['relight']['total']['relative_L2'],physical100=e['physical'],physical_errors100=er,radiance_errors100=e['radiance_errors'],
                success=e['success'],component_compensation=e['component_compensation'],global_gradient_norm100=unique[100]['gradient_norm'],
                exact_sigmoid_boundary_states=sum(v['boundary_channels']>0 for v in unique.values()),zero_global_gradient_states=sum(v['nonzero']==0 for v in unique.values())),
            Q_ratio_to_baseline=er['Q_relative_frobenius']/base['Q100'],heldout_ratio_to_baseline=e['radiance_errors']['held_out']['total']['relative_L2']/base['heldout100'])
        endpoints.append(endpoint)
        for state in (0,10,100):
            row=next(row for row in rows if row['kind']=='fit' and row['state']==state)
            q=np.array([row['nodes'][h+'.sigmoid']['values'][0][0] for h in ('mx_predictor','my_predictor')]);d=np.array([row['nodes'][h+'.delta']['values'][0][0] for h in ('mx_predictor','my_predictor')])
            conditioning.append(dict(case=case,method='AS1',state=state,**local_Q_jacobian(row['physical'],.96*q*(1-q)/np.sqrt(1+d*d),j0)))
            control=next(v for v in curves if v['case']==case and v['method']=='N1' and v['state']==state)
            bq=np.array([baseline_heads[case,state,h]['q'][0] for h in ('mx_predictor','my_predictor')]);qr=baseline_heads[case,state,'rotation_predictor']['q'];c=2*qr-1;r2=c@c
            j=np.array([-c[1]*qr[0]*(1-qr[0]),c[0]*qr[1]*(1-qr[1])])/r2
            conditioning.append(dict(case=case,method='N1',state=state,**local_Q_jacobian(control['physical'],.96*bq*(1-bq),j)))
    assert len(curves)==1010 and len(measurements)==512 and len(motions)==1000 and len(conditioning)==30 and len(snapshots)==10 and signal_count==22
    wins={k:sum(e['AS1'][k]<e['baseline'][k] for e in endpoints) for k in ('loss100','Q100','heldout100','relight100')}
    out.mkdir(exist_ok=False)
    def save(name,v):
        with (out/name).open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    def table(name,rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader()
            for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    save('comparison.json',dict(protocol='material-e4-as1-offline-v1',execution_status='complete',matched_initialization=True,AS1_wins=wins,
        AS1_physical_successes=sum(e['AS1']['success']['all_passed'] for e in endpoints),baseline_physical_successes=sum(e['baseline']['physical_success'] for e in endpoints),
        endpoints=endpoints,scientific_work=report['totals'],analysis_model_queries=0,analysis_shader_queries=0,analysis_optimizer_updates=0))
    save('integrity.json',dict(source_hashes=len(plan['source_sha256']),input_hashes=len(plan['input_sha256']),initial_gates=initial_gates,full_numeric_snapshots=snapshots,
        numeric_signal_archives=signal_count,terminal_metric_sets_recomputed=10,physical_rows_recomputed=512,analytic_chain_max_relative_error=max(chain_errors),
        conditioning_scope='3x4 output-coordinate Q Jacobian only; own fitted states differ; no full neural or observation Jacobian measured'))
    table('endpoints.csv',[dict(case=e['case'],baseline_Q=e['baseline']['Q100'],AS1_Q=e['AS1']['Q100'],Q_ratio=e['Q_ratio_to_baseline'],
        baseline_heldout=e['baseline']['heldout100'],AS1_heldout=e['AS1']['heldout100'],heldout_ratio=e['heldout_ratio_to_baseline'],
        baseline_loss=e['baseline']['loss100'],AS1_loss=e['AS1']['loss100'],baseline_relight=e['baseline']['relight100'],AS1_relight=e['AS1']['relight100'],
        physical_errors=e['AS1']['physical_errors100'],physical=e['AS1']['physical100'],success=e['AS1']['success'],radiance_errors=e['AS1']['radiance_errors100']) for e in endpoints])
    table('trajectories.csv',curves);table('candidate_measurements.csv',measurements);table('gradient_routes.csv',gradient_routes);table('conditioning.csv',conditioning);table('parameter_movement.csv',motions)
    save('evidence_manifest.json',dict(schema_version=1,run_id=RUN,execution_commit='91b82387b675adc5d92aaf144d8e693642e5b27e',freeze_sha256=FREEZE,
        report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,new_raw_files=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
        sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')],
        accepted_baseline=identity(REPO/'docs/results/neural_material_e3_complete_v1/comparison.json'),analysis_source=identity(Path(__file__).resolve()),
        original_evidence_unchanged=True,large_raw_arrays_not_copied=True))
    figures(out,curves,endpoints,conditioning,motions)
    print(json.dumps(dict(status='complete_saved_data_analysis',AS1_wins=wins,physical_successes=sum(e['AS1']['success']['all_passed'] for e in endpoints),output=str(out),scientific_queries=0)))


def figures(out,curves,endpoints,conditioning,motions):
    os.environ['MPLCONFIGDIR']=str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});colors={'N1':'#0072B2','AS1':'#D55E00'}
    folder=out/'figures';folder.mkdir()
    def emit(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,key,title in zip(axes,('Q_ratio_to_baseline','heldout_ratio_to_baseline'),('Physical Q error ratio','Held-out RGB relative-L2 ratio')):
        vals=[e[key] for e in endpoints];ax.bar(CASES,vals,color=['#D55E00' if v>1 else '#0072B2' for v in vals]);ax.axhline(1,color='black',ls='--',lw=.8)
        for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.3f}',ha='center')
        ax.set_ylim(0,1.7);ax.set_ylabel('AS1 / original N1');ax.set_title(title+'; lower is better');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched state-100 pilot: no full recovery in either arm');fig.tight_layout();emit(fig,'paired_endpoint_ratios')
    fig,axes=plt.subplots(5,3,figsize=(13,13),sharex=True)
    for i,case in enumerate(CASES):
        for method in ('N1','AS1'):
            rows=sorted([r for r in curves if r['case']==case and r['method']==method],key=lambda r:r['state'])
            for j,key in enumerate(('loss','Q_relative','gradient_norm')):
                axes[i,j].plot([r['state'] for r in rows],[r[key] for r in rows],color=colors[method],label=method);axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel(case)
        for ax in axes[i]:ax.set_yscale('log')
    for j,title in enumerate(('Fitting MSE','Relative Q error','Original parameter-gradient L2')):axes[0,j].set_title(title)
    for ax in axes[-1]:ax.set_xlabel('Completed updates / pre-update state')
    axes[0,0].legend();fig.tight_layout();emit(fig,'matched_trajectories')
    fig,axes=plt.subplots(5,2,figsize=(12,12))
    for i,case in enumerate(CASES):
        for method in ('N1','AS1'):
            rows=sorted([r for r in conditioning if r['case']==case and r['method']==method],key=lambda r:r['state'])
            for j,style in enumerate(('-','--',':')):axes[i,0].plot([r['state'] for r in rows],[r['singular_values'][j] for r in rows],style,marker='o',ms=3,color=colors[method],label=method+' sigma'+str(j+1))
            rows=sorted([r for r in motions if r['case']==case and r['method']==method],key=lambda r:r['update']);axes[i,1].plot([r['update'] for r in rows],[r['movement_L2'] for r in rows],color=colors[method],label=method)
        axes[i,0].set_ylabel(case)
        for ax in axes[i]:ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0,0].set_title('Own-state output-coordinate Q-Jacobian SVs\nNot a full neural/observation Jacobian');axes[0,1].set_title('Actual float32 weight displacement L2')
    axes[0,0].legend(fontsize=7,ncol=2);axes[0,1].legend()
    for ax in axes[-1]:ax.set_xlabel('Completed updates / measured state')
    fig.tight_layout();emit(fig,'coordinate_sensitivity_and_weight_motion')


if __name__=='__main__':main()
