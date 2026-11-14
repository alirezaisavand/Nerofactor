"""Saved-data E3/E3R integration only. No Torch, model, shader or optimizer execution."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname=='torch' or fullname.startswith(('torch.','experiments.')):
            raise RuntimeError('Offline analysis forbids scientific imports: '+fullname)


sys.meta_path.insert(0,NoScience())
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
import numpy as np
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure import neural_material_observer_e2a as audit
from infrastructure.signatures import read_report,require

PARENT='neural-e3-b4d1389b174e4b20b2f892725050a0ae'
RUN='neural-e3r-39baf93f9caa4d89b3fc2ddc549eb205'
FREEZE='ff4dc8f7b6b3712731a584681770980ca9ee9f46c9352e5ea1de3f37a299f743'
REPORT='328abbadd40cebff641daa2c0e4cc72cd3f0b1fbd40930d83b76dcfb42ef786f'
SEAL='92d65ae225990e01babcc88e6202109a36461e9fbf632ef2e79c31e9114f66a8'
RECEIPT='e3ed7ddb7789d4372aa5d9a389ca498e27c66a6303a493dc70a127ee300a44df'
OLD_REPORT='79a876fffd11f3e5d44c637a2ca4b9f1aa05ebd6a0f2484744a7403ecbe2a490'
E2B_REPORT='bb12d3627e778634da17da7a2b484ebe608a88d94aab0bc6c033cd58a1f15373'
CASES=('M1/101','M1/103','M2/103','M3/101','M3/103');RATES=(.01,.001)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def norm(values):return math.sqrt(math.fsum(float(v)**2 for v in values))
def key(a):return a['material']+'/'+str(a['seed']),a['lr']


def original_numeric_metrics(plan):
    """Closed analytical functions only, with unchanged success constants."""
    path='experiments/run_known_material_recovery_v1.py';assert sha(REPO/path)==plan['source_sha256'][path]
    tree=ast.parse((REPO/path).read_text())
    setting=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets))
    success=next(ast.literal_eval(v) for k,v in zip(setting.keys,setting.values) if isinstance(k,ast.Constant) and k.value=='success')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('radiance_errors','assess_success')]
    ns=dict(np=np,require=require,SETTINGS=dict(success=success));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REPO/path),'exec'),ns)
    return ns


def read_rows(root,arm,selection=None):
    if 'trajectory' not in arm:
        # The failed parent arm has a sealed-inventory-selected partial log.
        path=root/(arm['material'].lower()+'-'+str(arm['seed'])+'-'+str(arm['lr']))/'trajectory.jsonl'
    else:path=root/arm['trajectory']['path'];assert sha(path)==arm['trajectory']['sha256']
    rows=[];previous=None
    for line in path.read_bytes().splitlines(keepends=True):
        row=json.loads(line);assert row['previous_sha256']==previous;previous=hashlib.sha256(line).hexdigest()
        assert not rows or row['state']==rows[-1]['state']+1
        assert row['post_state']==min(row['state']+1,100) and row['Adam_counter']==row['post_state']
        assert row['original_gradient_norm']==math.sqrt(sum(row['parameter_gradient_groups'][g]['squared_norm'] for g in audit.GROUPS))
        if rows and 'actual_parameter_movement' in row:
            assert rows[-1]['actual_parameter_movement']['after_identity']==row['actual_parameter_movement']['before_identity']
            for name,probe in row['parameter_probes'].items():assert probe['before']==rows[-1]['parameter_probes'][name]['after']
        rows.append(row)
    if 'last_record_sha256' in arm:assert previous==arm['last_record_sha256']
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-parent',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.evidence_parent;root=parent/RUN;oldroot=parent/PARENT;out=args.output_dir
    assert parent.is_absolute() and parent.resolve()==parent and out.is_absolute() and out.resolve()==out and out.parent==parent and not out.exists()
    plan,_=read_report(REPO/'docs/protocols/NEURAL_MATERIAL_RECOVERY_E3R_V1.freeze.json',FREEZE)
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h
    for p,h in plan['input_sha256'].items():assert sha(p)==h
    r,_=read_report(root/'result.json',REPORT);seal,_=read_report(root/'finalization/seal.json',SEAL);receipt,_=read_report(parent/(RUN+'.process.json'),RECEIPT)
    old,_=read_report(oldroot/'result.json',OLD_REPORT)
    e2b,_=read_report(parent/'neural-e2b-d1f760d630964a2786ff45bde7438825/result.json',E2B_REPORT)
    assert r['scientific_work_completed'] and r['cleanup_verified'] and r['terminal_integrity_verified'] and r['all_restoration_gates_passed']
    assert seal['outcome']=='complete' and seal['report']['sha256']==REPORT and receipt['process_exit_code']==0 and receipt['retry_attempts']==0
    assert old['status']=='failed' and not old['scientific_work_completed']
    assert r['additional_totals']==plan['total_work'] and len(r['outcomes'])==4
    for name in ('authorization','launch','console'):
        item=receipt[name];assert identity(Path(item['path']))==item
    # This accepted 19.7-MB reference predates the generic small-report reader.
    # Admit this one file by exact known size/hash; never widen that reader's limits.
    refpath=REPO/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'
    assert refpath.stat().st_size==19679861 and sha(refpath)==plan['inputs']['accepted_3D1_SHA256']
    reference=json.loads(refpath.read_bytes())
    metrics=original_numeric_metrics(plan);observations=reference['scene']['observations']
    new={key(a):a for a in r['outcomes']};oldarms={key(a):a for a in old['outcomes']};starts={key(a):a for a in e2b['outcomes']}
    endpoints=[];trajectories=[];routes=[];motions=[];bridges=[];snapshots=[];series={};verified_signals=0
    for case in CASES:
        material,seed=case.split('/');target_info=next(v for v in reference['materials'] if v['material']==material)
        target=np.asarray(target_info['targets']['fine'],np.float64);truth=np.asarray(plan['inputs']['targets'][material]['truth'],np.float64)
        assert hashlib.sha256(target.tobytes()).hexdigest()==plan['inputs']['targets'][material]['fine_sha256']
        for lr in RATES:
            k=case,lr;prior=oldarms[k];first=starts[k];fresh=new.get(k)
            original_rows=read_rows(oldroot,prior) if prior['status'] in ('complete','failed') else [json.loads((oldroot/(material.lower()+'-'+seed+'-'+str(lr))/'state10-parity.json').read_text())['actual']]
            if fresh:
                assert fresh['status']=='complete' and fresh['completed_additional_updates']==100-fresh['start']
                assert fresh['work']==next(a for a in plan['arms'] if key(a)==k)['work'] and fresh['native_work']['optimizer_returns']==100
                newrows=read_rows(root,fresh);assert newrows[0]['state']==fresh['start'] and newrows[-1]['state']==100
                prefix=material.lower()+'-'+seed+'-'+str(lr);gate=json.loads((root/prefix/'restoration-measurement.json').read_text())
                assert gate['passed'] and gate['restoration']['exact'] and gate['restoration']['Adam_step']==fresh['start']
                if fresh['start']==10:
                    for field in ('loss','original_gradient_norm','physical','parameter_gradient_groups','heads','shader','shared_feature_gradient','relus'):
                        assert original_rows[0][field]==newrows[0][field],(case,lr,field)
                    joined=[(row,RUN,root) for row in newrows];label='exact_state10_measurement_parity'
                else:
                    assert original_rows[-1]['state']==20 and original_rows[-1]['post_state']==21
                    assert original_rows[-1]['actual_parameter_movement']['after_identity']==newrows[0]['actual_parameter_movement']['before_identity']==gate['restoration']['parameter_identity']
                    assert np.float32(newrows[0]['physical']['physical'][0])==np.float32(.04)
                    joined=[(row,PARENT,oldroot) for row in original_rows]+[(row,RUN,root) for row in newrows];label='exact_state21_checkpoint_bridge_new_measurement'
                bridges.append(dict(case=case,lr=lr,start=fresh['start'],verified=True,kind=label))
                arm=fresh;source=RUN
                spec=next(a for a in plan['arms'] if key(a)==k);initial=audit.read_numeric(spec['input_root'],spec['checkpoint'])
                for state,snapshot in arm['checkpoints'].items():
                    a=audit.read_numeric(root,snapshot['selection']);n=int(state)
                    assert len(a)==350 and all(float(v)==n for name,v in a.items() if name.endswith('/step'))
                    assert snapshot['RNG']==spec['RNG']
                    for name,v in initial.items():
                        if name.startswith('buffer/') or name in ('torch_CPU_RNG','numpy_MT19937'):assert a[name].tobytes()==v.tobytes() and a[name].dtype==v.dtype
                    transition=next(row for row,_,_ in joined if row['state']==n-1)
                    params={name[10:]:v for name,v in a.items() if name.startswith('parameter/')}
                    assert audit.parameter_identity(params)==transition['actual_parameter_movement']['after_identity']
                    for name,probe in transition['parameter_probes'].items():assert params[name].ravel()[probe['indices']].tolist()==probe['after']
                    snapshots.append(dict(case=case,lr=lr,state=n,parameter_identity=True,probes=True,Adam_counter=n,buffers_and_RNG_unchanged=True))
            else:
                assert prior['status']=='complete' and prior['completed_additional_updates']==90
                joined=[(row,PARENT,oldroot) for row in original_rows];arm=prior;source=PARENT
            assert [row['state'] for row,_,_ in joined]==list(range(10,101))
            local=[];localmotions=[]
            for row,source_run,artifact_root in joined:
                p=np.asarray(row['physical']['physical'],np.float32)
                assert p.astype(np.float64).tolist()==row['physical']['physical']
                assert diagnostic.recovery_errors(p,truth)==row['physical']['errors']
                nonzero=sum(g['nonzero'] for g in row['parameter_gradient_groups'].values());boundary=0;floor=int(np.sum(p[:2]==np.float32(.04)))
                for h in audit.HEADS:
                    value=row['heads'][h];q=np.asarray(value['sigmoid']);gq=np.asarray(value['sigmoid.loss_gradient']);gz=np.asarray(value['logits.loss_gradient'])
                    b=(q==0)|(q==1);assert int(b.sum())==value['boundary_channels'] and not np.any(b&(gz!=0));boundary+=int(b.sum())
                    routes.append(dict(case=case,lr=lr,state=row['state'],head=h,source_run=source_run,boundary_channels=int(b.sum()),
                        sigmoid=q.tolist(),logits=value['logits'],sigmoid_gradient_L2=norm(gq.ravel()),logit_gradient_L2=norm(gz.ravel()),blocked_channels=int(np.sum(b&(gq!=0)&(gz==0)))))
                if source_run==RUN and 'signals' in row:
                    signals=audit.read_numeric(artifact_root,row['signals']);verified_signals+=1
                    for h in audit.HEADS:
                        for field in ('logits','sigmoid','logits.loss_gradient','sigmoid.loss_gradient'):assert signals[h+'.'+field].tolist()==row['heads'][h][field]
                tr=dict(case=case,lr=lr,state=row['state'],source_run=source_run,loss=row['loss'],Q_relative=row['physical']['errors']['Q_relative_frobenius'],
                    physical=row['physical']['physical'],gradient_norm=row['original_gradient_norm'],gradient_nonzero=nonzero,gradient_nonzero_fraction=nonzero/1485584,
                    boundary_channels=boundary,physical_floor_channels=floor,shared_feature_gradient_L2=row['shared_feature_gradient']['L2'],
                    shader_gradient_L2=norm(v['L2'] for v in row['shader'].values()),sigmoid_gradient_L2=norm(v['sigmoid_gradient_L2'] for v in routes[-5:]),
                    logit_gradient_L2=norm(v['logit_gradient_L2'] for v in routes[-5:]))
                trajectories.append(tr);local.append(tr)
                if 'actual_parameter_movement' in row:
                    delta=row['actual_parameter_movement'];groups=delta['per_group']
                    item=dict(case=case,lr=lr,state=row['state'],update=row['post_state'],source_run=source_run,gradient_nonzero=nonzero,gradient_norm=tr['gradient_norm'],
                        displacement_L2=norm(g['L2'] for g in groups.values()),changed_elements=sum(g['changed_elements'] for g in groups.values()),
                        bitwise_changed_elements=sum(g['bitwise_changed_elements'] for g in groups.values()),groups=groups)
                    motions.append(item);localmotions.append(item)
            e=arm['endpoint'];assert e['recovery_errors']==diagnostic.recovery_errors(np.asarray(e['physical'],np.float32),truth)
            assert e['physical']==joined[-1][0]['physical']['physical']
            predictions=np.asarray(e['predictions'],np.float64);assert predictions.shape==target.shape==(84,3,3)
            assert hashlib.sha256(predictions.tobytes()).hexdigest()==e['prediction_sha256']
            recalculated={}
            for split in ('fit','held_out','relight','relight_seen_views','relight_unseen_views'):
                mask=[o['split']==split if split in ('fit','held_out','relight') else o['split']=='relight' and (o['view']<6 if split=='relight_seen_views' else o['view']>=6) for o in observations]
                recalculated[split]={name:metrics['radiance_errors'](predictions[mask,i],target[mask,i]) for i,name in enumerate(('diffuse','specular','total'))}
            assert recalculated==e['radiance_errors'] and metrics['assess_success'](e['recovery_errors'],recalculated)==e['success']
            er=e['recovery_errors'];old_e=first['endpoint']
            endpoints.append(dict(case=case,lr=lr,state=100,endpoint_source_run=source,restoration_state=fresh['start'] if fresh else None,
                loss10=local[0]['loss'],loss100=local[-1]['loss'],Q10=old_e['recovery_errors']['Q_relative_frobenius'],Q100=er['Q_relative_frobenius'],
                heldout10=old_e['radiance_errors']['held_out']['total']['relative_L2'],heldout100=recalculated['held_out']['total']['relative_L2'],
                relight100=recalculated['relight']['total']['relative_L2'],physical100=e['physical'],physical_errors100=er,radiance_errors100=recalculated,
                physical_success=e['success']['all_passed'],success_checks=e['success']['checks'],albedo_max_error=max(er['albedo_absolute_errors']),metallic_error=er['metallic_absolute_error'],
                orientation_error_degrees=er['orientation_degrees'],first_boundary_state=min(list(first['first_boundary_states'].values())+[t['state'] for t in local if t['boundary_channels']],default=None),
                first_all_zero_gradient_state=next(iter([first['first_all_zero_gradient_state']] if first['first_all_zero_gradient_state'] is not None else [t['state'] for t in local if not t['gradient_nonzero']]),None),
                first_physical_floor_state=next((t['state'] for t in local if t['physical_floor_channels']),None),
                zero_gradient_moving_updates=sum(t['gradient_nonzero']==0 and t['bitwise_changed_elements']>0 for t in localmotions),
                gradient_norm100=local[-1]['gradient_norm'],minimum_sigmoid100=min(min(np.asarray(joined[-1][0]['heads'][h]['sigmoid']).ravel()) for h in audit.HEADS)))
            series[k]=local
    assert len(endpoints)==10 and len(trajectories)==910 and len(motions)==900 and len(routes)==4550 and len(bridges)==4 and len(snapshots)==8 and verified_signals==12
    wins={}
    for metric in ('loss100','Q100','heldout100','relight100'):
        wins[metric]=sum(next(e for e in endpoints if e['case']==case and e['lr']==.001)[metric]<next(e for e in endpoints if e['case']==case and e['lr']==.01)[metric] for case in CASES)
    out.mkdir(exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    def table(name,rows):
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader()
            for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    save('comparison.json',dict(schema_version=1,protocol='e3-linked-complete-panel-analysis-v1',panel_status='ten terminal endpoints from two linked executions',
        original_run_status='failed',recovery_run_status='complete',endpoints=endpoints,lower_LR_wins=wins,physical_successes=sum(e['physical_success'] for e in endpoints),
        Q_improved_from10=sum(e['Q100']<e['Q10'] for e in endpoints),heldout_improved_from10=sum(e['heldout100']<e['heldout10'] for e in endpoints),
        zero_gradient_moving_updates=sum(e['zero_gradient_moving_updates'] for e in endpoints),recovery_work=dict(updates=349,forwards=357,backwards=353,calibrations=4),
        combined_work=dict(updates=900,forwards=924,backwards=914,calibrations=14),extra_repeated_restoration_passes=4,model_queries_in_analysis=0,shader_queries_in_analysis=0,optimizer_updates_in_analysis=0))
    save('integrity.json',dict(source_hashes_verified=len(plan['source_sha256']),input_hashes_verified=len(plan['input_sha256']),bridges=bridges,numeric_snapshots=snapshots,
        new_signal_archives_verified=verified_signals,unique_recorded_states=910,actual_update_records=900,physical_metric_rows_recomputed=910,
        terminal_radiance_and_success_checks_recomputed=10,original_failed_status_preserved=True,new_cleanup_and_terminal_integrity_verified=True))
    table('endpoints.csv',endpoints);table('trajectories.csv',trajectories);table('gradient_routes.csv',routes);table('adam_displacements.csv',motions)
    save('evidence_manifest.json',dict(schema_version=1,recovery_run=RUN,parent_run=PARENT,execution_commit='633d1e354bbac7ae480afe742a625339ec3ab78b',
        freeze_sha256=FREEZE,report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,original_report_sha256=OLD_REPORT,
        new_raw_files=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],new_sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')],
        original_inventory=identity(REPO/'docs/results/neural_material_continuation_e3/evidence_manifest.json'),
        analysis_source=identity(Path(__file__).resolve()),original_files_reverified_by_selected_input_hashes=True,large_archives_copied_to_Git=False))
    figures(out,series,endpoints,routes,motions)
    print(json.dumps(dict(status='complete_linked_panel',endpoints=10,physical_successes=sum(e['physical_success'] for e in endpoints),lower_LR_wins=wins,output=str(out),model_queries=0)))


def figures(out,series,endpoints,routes,motions):
    os.environ['MPLCONFIGDIR']=str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={.01:'#D55E00',.001:'#0072B2'};folder=out/'figures';folder.mkdir()
    def emit(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(5,3,figsize=(13,13),sharex=True)
    for i,case in enumerate(CASES):
        for lr in RATES:
            rows=series[case,lr];x=[r['state'] for r in rows]
            for j,k in enumerate(('loss','Q_relative','gradient_norm')):
                axes[i,j].plot(x,[r[k] for r in rows],color=colors[lr],label='LR '+str(lr));axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel(case);axes[i,0].set_yscale('log');axes[i,1].set_yscale('log');axes[i,2].set_yscale('symlog',linthresh=1e-12)
        if case=='M3/101':
            for ax in axes[i]:ax.axvline(21,color='0.5',ls=':',lw=.8)
    for j,title in enumerate(('Fitting MSE','Relative anisotropy-tensor Q error','Original parameter-gradient L2')):axes[0,j].set_title(title)
    for ax in axes[-1]:ax.set_xlabel('Completed updates / pre-update state');ax.set_xlim(10,100)
    axes[0,0].legend();fig.suptitle('E.3 complete panel: original six endpoints + separately authorized four-arm recovery',y=1.003)
    fig.tight_layout();emit(fig,'complete_trajectories')
    fig,axes=plt.subplots(5,2,figsize=(13,12),sharex=True,constrained_layout=True)
    positive=[r['logit_gradient_L2'] for r in routes if r['logit_gradient_L2']>0];lo=math.floor(math.log10(min(positive)));hi=math.ceil(math.log10(max(positive)))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d0d0d0')
    for i,case in enumerate(CASES):
        for j,lr in enumerate(RATES):
            a=np.full((5,91),np.nan);marks=[]
            for r in routes:
                if r['case']==case and r['lr']==lr:
                    if r['logit_gradient_L2']>0:a[audit.HEADS.index(r['head']),r['state']-10]=math.log10(r['logit_gradient_L2'])
                    if r['boundary_channels']:marks.append((r['state'],audit.HEADS.index(r['head'])))
            im=axes[i,j].imshow(np.ma.masked_invalid(a),aspect='auto',origin='upper',extent=(9.5,100.5,4.5,-.5),cmap=cmap,vmin=lo,vmax=hi)
            for x,y in marks:axes[i,j].plot(x,y,'r.',ms=2)
            axes[i,j].set_title(case+' | LR '+str(lr));axes[i,j].set_yticks(range(5));axes[i,j].set_yticklabels(('mx','my','metallic','rotation','albedo'))
    for ax in axes[-1]:ax.set_xlabel('Recorded state; gray = zero gradient; red = exact sigmoid endpoint')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 actual logit-gradient L2');emit(fig,'gradient_transport_and_saturation')
    fig,axes=plt.subplots(5,2,figsize=(12,12),sharex=True)
    for i,case in enumerate(CASES):
        for lr in RATES:
            rows=[r for r in motions if r['case']==case and r['lr']==lr]
            axes[i,0].plot([r['update'] for r in rows],[r['displacement_L2'] for r in rows],color=colors[lr],label='LR '+str(lr))
            for r in rows:
                if not r['gradient_nonzero'] and r['displacement_L2']>0:axes[i,0].plot(r['update'],r['displacement_L2'],'.',color=colors[lr],ms=3)
            rs=series[case,lr];axes[i,1].plot([r['state'] for r in rs],[r['gradient_nonzero_fraction'] for r in rs],color=colors[lr])
        axes[i,0].set_ylabel(case);axes[i,0].set_yscale('log');axes[i,1].set_ylim(-.02,1.02)
        for ax in axes[i]:ax.grid(alpha=.2)
    axes[0,0].set_title('Actual float32 displacement L2\nDots: zero current gradient, nonzero movement');axes[0,1].set_title('Nonzero parameter-gradient fraction');axes[0,0].legend()
    for ax in axes[-1]:ax.set_xlabel('Update (left) / recorded state (right)')
    fig.tight_layout();emit(fig,'adam_motion_and_gradients')
    fig,axes=plt.subplots(1,5,figsize=(16,3.5),sharex=True,sharey=True)
    for ax,case in zip(axes,CASES):
        for lr in RATES:
            e=next(e for e in endpoints if e['case']==case and e['lr']==lr)
            ax.plot(e['Q10'],e['heldout10'],'o',mfc='none',color=colors[lr]);ax.plot(e['Q100'],e['heldout100'],'o',color=colors[lr])
            ax.annotate('',xy=(e['Q100'],e['heldout100']),xytext=(e['Q10'],e['heldout10']),arrowprops=dict(arrowstyle='->',color=colors[lr],lw=1))
        ax.axvline(.05,color='0.4',ls=':',lw=.8);ax.axhline(.01,color='0.4',ls=':',lw=.8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.04,6);ax.set_ylim(.004,1);ax.set_title(case);ax.set_xlabel('Relative Q error');ax.grid(alpha=.2)
    axes[0].set_ylabel('Held-out RGB relative L2')
    fig.legend(handles=[Line2D([0],[0],color=colors[lr],marker='o',label='LR '+str(lr)) for lr in RATES],loc='upper center',ncol=2,bbox_to_anchor=(.5,1.09))
    fig.suptitle('State 10 (open) → state 100 (filled); dotted lines are unchanged individual success bounds',y=1.01)
    fig.tight_layout();emit(fig,'photometric_vs_physical')


if __name__=='__main__':main()
