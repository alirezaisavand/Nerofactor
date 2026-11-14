"""Offline analysis of the one interrupted E3 run. Never imports Torch or scientific models."""
import argparse
import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith(('torch.', 'experiments.')):
            raise RuntimeError('Offline analysis forbids scientific imports: '+fullname)


sys.meta_path.insert(0, NoScience())
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
from infrastructure import neural_material_observer_e2a as audit
from infrastructure.signatures import read_report

RUN = 'neural-e3-b4d1389b174e4b20b2f892725050a0ae'
FREEZE = 'ae258c5f6b7d2b0015f071f01c4bbb53c164f19bc2df1e412bb5752c2b98c606'
REPORT = '79a876fffd11f3e5d44c637a2ca4b9f1aa05ebd6a0f2484744a7403ecbe2a490'
SEAL = 'e7cf6b4673652cd912e3ba8547c27106399a3b99b0e2b9eeb22fa8b622f26be3'
RECEIPT = '281aac5e08d05863aa64fb14e090019e3010929736c9c3017c5ed57c4675216c'
E2B = 'bb12d3627e778634da17da7a2b484ebe608a88d94aab0bc6c033cd58a1f15373'
CASES = ('M1/101','M1/103','M2/103','M3/101','M3/103')
HEADS = audit.HEADS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):h.update(b)
    return h.hexdigest()


def identity(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def norm(values):return math.sqrt(math.fsum(float(v)**2 for v in values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True,help='Fresh external analysis directory')
    args=parser.parse_args();root=args.run;out=args.output_dir
    assert root.is_absolute() and root.resolve()==root and root.name==RUN
    assert out.is_absolute() and out.resolve()==out and out.parent==root.parent and not out.exists()
    plan,_=read_report(REPO/'docs/protocols/NEURAL_MATERIAL_CONTINUATION_E3_V1.freeze.json',FREEZE)
    report,_=read_report(root/'result.json',REPORT);seal,_=read_report(root/'finalization/seal.json',SEAL)
    receipt,_=read_report(root.parent/(RUN+'.process.json'),RECEIPT)
    oldroot=root.parent/'neural-e2b-d1f760d630964a2786ff45bde7438825'
    old,_=read_report(oldroot/'result.json',E2B)
    assert receipt['process_exit_code']==1 and receipt['retry_attempts']==0 and receipt['foreground_supervisor']
    assert report['status']==seal['outcome']=='failed' and seal['report']['sha256']==REPORT
    assert report['all_state10_parity_passed'] and not report['scientific_work_completed']
    assert not report['cleanup_verified'] and not report['terminal_integrity_verified']
    for p,s in plan['source_sha256'].items():assert sha(REPO/p)==s
    for p,s in plan['input_sha256'].items():assert sha(p)==s
    for suffix in ('.authorization.json','.launch.json','.console.log'):
        key={'.authorization.json':'authorization','.launch.json':'launch','.console.log':'console'}[suffix]
        assert identity(root.parent/(RUN+suffix))==receipt[key]
    out.mkdir(exist_ok=False)

    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    def table(name,rows):
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,keys);w.writeheader()
            for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})

    endpoints=[];trajectories=[];routes=[];motions=[];integrity=[];series={};totals=dict(updates=0,forwards=0,backwards=0,calibrations=10)
    for arm in report['outcomes']:
        case=arm['material']+'/'+str(arm['seed']);lr=arm['lr'];prefix=arm['material'].lower()+'-'+str(arm['seed'])+'-'+str(lr)
        spec=next(x for x in plan['arms'] if (x['material'],x['seed'],x['lr'])==(arm['material'],arm['seed'],lr))
        oldarm=next(x for x in old['outcomes'] if (x['material'],x['seed'],x['lr'])==(arm['material'],arm['seed'],lr))
        initial=audit.read_numeric(oldroot,spec['checkpoint'])
        gate=json.loads((root/prefix/'state10-parity.json').read_text())
        assert gate['passed'] and gate['restoration']['exact'] and gate['restoration']['Adam_step']==10
        assert gate['actual']['loss']==gate['expected']['loss']==spec['parity']['loss']
        assert gate['actual']['original_gradient_norm']==spec['parity']['original_gradient_norm']
        assert gate['actual']['physical']['physical']==spec['parity']['physical']
        log=root/prefix/'trajectory.jsonl';rows=[];previous=None;after_probes=None
        if log.exists():
            for state,line in enumerate(log.read_bytes().splitlines(keepends=True),10):
                row=json.loads(line);assert row['state']==state and row['post_state']==min(state+1,100)
                assert row['previous_sha256']==previous;previous=hashlib.sha256(line).hexdigest()
                assert row['Adam_counter']==min(state+1,100)
                assert row['optimizer_effective']==dict(plan['optimizer'],lr=lr)
                # Reproduce the frozen model's group reduction order, not JSON's sorted-key order.
                assert row['original_gradient_norm']==math.sqrt(sum(row['parameter_gradient_groups'][g]['squared_norm'] for g in audit.GROUPS))
                if state<100:
                    if rows:assert rows[-1]['actual_parameter_movement']['after_identity']==row['actual_parameter_movement']['before_identity']
                    if after_probes is not None:
                        assert all(p['before']==after_probes[n]['after'] and p['indices']==after_probes[n]['indices'] for n,p in row['parameter_probes'].items())
                    after_probes=row['parameter_probes']
                rows.append(row)
            assert previous==arm.get('last_record_sha256',previous)
        else:rows=[gate['actual']]
        if arm['status']=='complete':
            assert len(rows)==91 and sha(log)==arm['trajectory']['sha256']
            assert arm['completed_additional_updates']==90
            totals['updates']+=90;totals['forwards']+=92;totals['backwards']+=91
            assert arm['native_work']['optimizer_returns']==100 and arm['native_work']['field_completed']==92
            assert arm['work']==plan['work_per_arm']
        elif arm['status']=='failed':
            assert arm['completed_additional_updates']==11 and len(rows)==11
            assert arm['native_work']['optimizer_returns']==21 and arm['work']['forwards']==arm['work']['backwards']==12
            totals['updates']+=11;totals['forwards']+=12;totals['backwards']+=12
        else:
            assert arm['status']=='parity_verified' and not log.exists() and arm['completed_additional_updates']==0
            totals['forwards']+=1;totals['backwards']+=1
        snapshots=dict(arm.get('checkpoints',{}))
        if arm['status']=='failed':snapshots['21-failure']=arm['failure_state']
        checked_snapshots=[]
        for label,snapshot in snapshots.items():
            arrays=audit.read_numeric(root,snapshot['selection']);step=21 if label=='21-failure' else int(label)
            assert len(arrays)==350 and all(float(v)==step for n,v in arrays.items() if n.endswith('/step'))
            assert all(same(initial[n],v) for n,v in arrays.items() if n.startswith('buffer/') or n in ('torch_CPU_RNG','numpy_MT19937'))
            assert snapshot['RNG']==spec['RNG'] and snapshot.get('optimizer_effective',dict(plan['optimizer'],lr=lr))==dict(plan['optimizer'],lr=lr)
            transition=next(r for r in rows if r['state']==step-1)
            parameters={n[10:]:v for n,v in arrays.items() if n.startswith('parameter/')}
            assert audit.parameter_identity(parameters)==transition['actual_parameter_movement']['after_identity']
            for n,p in transition['parameter_probes'].items():assert parameters[n].ravel()[p['indices']].tolist()==p['after']
            checked_snapshots.append(dict(state=step,selection=snapshot['selection'],parameters=len(parameters),Adam_step=step,
                parameter_identity=True,probes_equal=True,buffers_and_RNG_unchanged=True,
                Adam_first_moment_L2=norm(np.linalg.norm(v.astype(np.float64)) for n,v in arrays.items() if n.endswith('/exp_avg'))))
        for row in rows:
            state=row['state'];groups=row['parameter_gradient_groups'];nonzero=sum(g['nonzero'] for g in groups.values())
            assert all(g['finite'] and g['missing']==0 for g in groups.values())
            boundary=0;min_boundary=1.
            for head in HEADS:
                h=row['heads'][head];q=np.asarray(h['sigmoid']);gq=np.asarray(h['sigmoid.loss_gradient']);gz=np.asarray(h['logits.loss_gradient'])
                b=(q==0)|(q==1);assert int(b.sum())==h['boundary_channels'] and not np.any(b&(gz!=0))
                boundary+=int(b.sum());min_boundary=min(min_boundary,float(np.minimum(q,1-q).min()))
                routes.append(dict(case=case,lr=lr,state=state,head=head,boundary_channels=int(b.sum()),
                    sigmoid=q.tolist(),logits=h['logits'],sigmoid_gradient_L2=norm(gq.ravel()),logit_gradient_L2=norm(gz.ravel()),
                    blocked_boundary_channels=int(np.sum(b&(gq!=0)&(gz==0)))))
                if 'signals' in row:
                    signals=audit.read_numeric(root,row['signals'])
                    for k in ('sigmoid','logits','sigmoid.loss_gradient','logits.loss_gradient'):assert signals[head+'.'+k].tolist()==h[k]
            tr=dict(case=case,lr=lr,state=state,loss=row['loss'],Q_relative=row['physical']['errors']['Q_relative_frobenius'],
                physical=row['physical']['physical'],physical_errors=row['physical']['errors'],gradient_norm=row['original_gradient_norm'],
                gradient_nonzero=nonzero,gradient_nonzero_fraction=nonzero/1485584,boundary_channels=boundary,min_boundary=min_boundary,
                shared_feature_gradient_L2=row['shared_feature_gradient']['L2'],sigmoid_gradient_L2=norm(r['sigmoid_gradient_L2'] for r in routes[-5:]),
                logit_gradient_L2=norm(r['logit_gradient_L2'] for r in routes[-5:]),shader_gradient_L2=norm(g['L2'] for g in row['shader'].values()))
            trajectories.append(tr)
            if 'actual_parameter_movement' in row:
                groups=row['actual_parameter_movement']['per_group']
                motions.append(dict(case=case,lr=lr,state=state,update=state+1,gradient_norm=tr['gradient_norm'],gradient_nonzero=nonzero,
                    displacement_L2=norm(g['L2'] for g in groups.values()),changed_elements=sum(g['changed_elements'] for g in groups.values()),
                    bitwise_changed_elements=sum(g['bitwise_changed_elements'] for g in groups.values()),groups=groups))
        series[case,lr]=[r for r in trajectories if r['case']==case and r['lr']==lr]
        e=arm.get('endpoint');old_e=oldarm['endpoint'];errs=e['recovery_errors'] if e else None
        selected=[r for r in motions if r['case']==case and r['lr']==lr]
        entry=dict(case=case,lr=lr,status=arm['status'],additional_updates=arm['completed_additional_updates'],
            last_completed_update=10+arm['completed_additional_updates'],last_recorded_fitting_state=rows[-1]['state'],
            loss10=spec['parity']['loss'],loss100=rows[-1]['loss'] if e else None,
            Q10=old_e['recovery_errors']['Q_relative_frobenius'],Q100=errs['Q_relative_frobenius'] if e else None,
            heldout10=old_e['radiance_errors']['held_out']['total']['relative_L2'],heldout100=e['radiance_errors']['held_out']['total']['relative_L2'] if e else None,
            relight100=e['radiance_errors']['relight']['total']['relative_L2'] if e else None,
            physical100=e['physical'] if e else None,physical_errors100=errs,success100=e['success'] if e else None,
            radiance_errors100=e['radiance_errors'] if e else None,
            first_continuation_boundary_state=next((r['state'] for r in series[case,lr] if r['boundary_channels']),None),
            first_continuation_all_zero_gradient_state=next((r['state'] for r in series[case,lr] if not r['gradient_nonzero']),None),
            zero_gradient_moving_updates=sum(m['gradient_nonzero']==0 and m['bitwise_changed_elements']>0 for m in selected),
            recorded_transitions=len(selected))
        endpoints.append(entry);integrity.append(dict(case=case,lr=lr,parity=True,trajectory_records=len(rows) if log.exists() else 0,
            hash_chain=bool(log.exists()),numeric_snapshots=checked_snapshots,complete=arm['status']=='complete'))
    assert totals==dict(updates=551,forwards=567,backwards=561,calibrations=10)
    assert len(motions)==551 and len(trajectories)==560 and len(routes)==2800
    save('comparison.json',dict(protocol='e3-interrupted-offline-analysis-v1',run_status='failed',completed_arms=6,failed_arms=1,
        parity_only_arms=3,scientific_totals=totals,counter_basis='Saved native/work counters for seven continued arms plus one verified parity forward/backward for each of the three uncontinued arms',
        full_recovery_successes=sum(bool(e['success100'] and e['success100']['all_passed']) for e in endpoints),endpoints=endpoints,
        float32_floor_as_float64=float(np.float32(.04)),float64_validator_floor=.04,model_queries=0,shader_queries=0,optimizer_updates=0,
        limitation='The state-21 failed physical output was not archived. No model query is made to recover it. Do not impute missing state-100 outcomes.'))
    save('integrity.json',dict(offline_source_hashes=len(plan['source_sha256']),offline_input_hashes=len(plan['input_sha256']),arms=integrity,
        runtime_cleanup_verified=False,runtime_terminal_integrity_verified=False,offline_file_hashes_do_not_promote_runtime_status=True))
    table('endpoints.csv',endpoints);table('trajectories.csv',trajectories);table('gradient_routes.csv',routes);table('adam_displacements.csv',motions)
    artifacts=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()]
    sidecars=[identity(root.parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')]
    save('evidence_manifest.json',dict(schema_version=1,run_id=RUN,run_status='failed',execution_commit='6ae92884fa362dbf7511d6b39bd437fa3b148468',
        freeze_sha256=FREEZE,report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,raw_root=str(root),
        raw_files=artifacts,raw_file_count=len(artifacts),raw_bytes=sum(a['bytes'] for a in artifacts),sidecars=sidecars,
        source_files_reverified=len(plan['source_sha256']),accepted_inputs_reverified=len(plan['input_sha256']),analysis_source=identity(Path(__file__).resolve()),
        accepted_e2b_evidence_unchanged=True,raw_arrays_copied_to_Git=False))
    # Standard plotting from saved numerical values only; no neural/scientific imports.
    os.environ['MPLCONFIGDIR']=str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={.01:'#D55E00',.001:'#0072B2'};figdir=out/'figures';figdir.mkdir()
    def emit(fig,name):
        fig.savefig(figdir/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(5,3,figsize=(13,13),sharex=True)
    for i,case in enumerate(CASES):
        for lr in (.01,.001):
            rs=series[case,lr];x=[r['state'] for r in rs]
            for j,k in enumerate(('loss','Q_relative','gradient_norm')):
                axes[i,j].plot(x,[r[k] for r in rs],color=colors[lr],label='LR '+str(lr),marker='o' if len(rs)==1 else None,ms=4)
                axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel(case);axes[i,0].set_yscale('log');axes[i,1].set_yscale('log');axes[i,2].set_yscale('symlog',linthresh=1e-12)
    for j,t in enumerate(('Fitting MSE','Relative anisotropy-tensor Q error','Original parameter-gradient L2')):axes[0,j].set_title(t)
    for ax in axes[-1]:ax.set_xlabel('Completed updates / fitting state');ax.set_xlim(9,101)
    axes[0,0].legend();fig.suptitle('Interrupted E.3: six endpoints; one partial arm; three state-10 parity captures only',y=1.002)
    fig.tight_layout();emit(fig,'continuation_trajectories')
    fig,axes=plt.subplots(5,2,figsize=(13,12),sharex=True,constrained_layout=True)
    positives=[r['logit_gradient_L2'] for r in routes if r['logit_gradient_L2']>0]
    low=math.floor(math.log10(min(positives)));high=math.ceil(math.log10(max(positives)))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d0d0d0')
    for i,case in enumerate(CASES):
        for j,lr in enumerate((.01,.001)):
            rs=series[case,lr];end=rs[-1]['state'];v=np.full((5,end-9),np.nan)
            for r in routes:
                if r['case']==case and r['lr']==lr:
                    if r['logit_gradient_L2']>0:v[HEADS.index(r['head']),r['state']-10]=math.log10(r['logit_gradient_L2'])
                    if r['boundary_channels']:axes[i,j].plot(r['state'],HEADS.index(r['head']),'r.',ms=2)
            im=axes[i,j].imshow(np.ma.masked_invalid(v),origin='upper',aspect='auto',extent=(9.5,end+.5,4.5,-.5),cmap=cmap,vmin=low,vmax=high)
            axes[i,j].set_xlim(9,101);axes[i,j].set_title(case+' | LR '+str(lr));axes[i,j].set_yticks(range(5));axes[i,j].set_yticklabels(('mx','my','metallic','rotation','albedo'))
    for ax in axes[-1]:ax.set_xlabel('Recorded fitting state; white beyond capture')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 actual logit-gradient L2; gray = zero; red = sigmoid endpoint')
    emit(fig,'logit_gradients_and_saturation')
    fig,axes=plt.subplots(5,2,figsize=(12,12),sharex=True)
    for i,case in enumerate(CASES):
        for lr in (.01,.001):
            ms=[r for r in motions if r['case']==case and r['lr']==lr]
            axes[i,0].plot([r['update'] for r in ms],[r['displacement_L2'] for r in ms],color=colors[lr],label='LR '+str(lr))
            for r in ms:
                if r['gradient_nonzero']==0 and r['displacement_L2']>0:axes[i,0].plot(r['update'],r['displacement_L2'],'.',color=colors[lr],ms=3)
            rs=series[case,lr];axes[i,1].plot([r['state'] for r in rs],[r['gradient_nonzero_fraction'] for r in rs],color=colors[lr],marker='o' if len(rs)==1 else None,ms=4)
        axes[i,0].set_ylabel(case);axes[i,0].set_yscale('log');axes[i,1].set_ylim(-.02,1.02)
        for ax in axes[i]:ax.grid(alpha=.2)
    axes[0,0].set_title('Actual float32 parameter movement L2\nDots: zero current gradient, nonzero movement')
    axes[0,1].set_title('Fraction of network parameters with nonzero gradients')
    axes[0,0].legend()
    for ax in axes[-1]:ax.set_xlabel('Update (left) / fitting state (right)');ax.set_xlim(9,101)
    fig.tight_layout();emit(fig,'adam_movement_and_gradient')
    print(json.dumps(dict(status='offline_analysis_of_failed_run',scientific_totals=totals,output=str(out),model_queries=0)))


if __name__=='__main__':main()
