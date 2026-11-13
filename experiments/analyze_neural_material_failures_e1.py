"""Deterministic saved-array E.1 analysis. No Torch/model/renderer imports."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import numpy as np

from experiments import analyze_neural_material_recovery_v1 as saved
from experiments.finalize_neural_material_journal_v2 import checkpoint_evidence
from infrastructure.signatures import require

ROOT=Path(__file__).resolve().parents[1]
PINS={
 'freeze':('docs/protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.freeze.json','dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af'),
 'original':('docs/evidence/NEURAL_MATERIAL_RECOVERY_LOGGED_V2_RESULTS.json','7a9ca7ae1e930c1a90143079f83e7376ebd00ed8d0bea514bf321686ea37084e'),
 'complete':('docs/evidence/NEURAL_MATERIAL_RECOVERY_CONTINUATION_V2_RESULTS.json','f60a3b8fa6db646efec916428c75a3bfb3196da89cd03a3ef3955f3609254135')}
GROUPS=('feats_network','mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor')
PHYSICAL_COLUMNS=(0,1,3,4,5,6)


def select(path,sha):return saved.selected_json(Path(path),sha)[0]
def file_id(path):
    path=Path(path);return dict(path=str(path),sha256=saved.digest(path),bytes=path.stat().st_size)


def coverage(rows):
    expected={(m,s,a) for m in ('M1','M2','M3') for s in (101,103,105) for a in ('D0','N0','N1')}
    actual=[(r['material'],r['seed'],r['arm']) for r in rows]
    require(len(actual)==27 and set(actual)==expected,'Exact unique 27-history coverage required')


def boundary_distances(physical):
    p=np.asarray(physical,np.float64)[list(PHYSICAL_COLUMNS)]
    lower=np.array([.04,.04,0,0,0,0],np.float32).astype(np.float64)
    require(p.shape==(6,) and np.isfinite(p).all() and ((p>=lower)&(p<=1)).all(),'Invalid physical bounds')
    return np.minimum(p-lower,1-p)


def orientation(raw):
    u=np.asarray(raw,np.float64);require(u.shape==(2,) and ((u>=0)&(u<=1)).all(),'Invalid raw channels')
    c=2*u-1;r=float(np.linalg.norm(c));slope=u*(1-u)
    if r<1e-6:
        return dict(raw=u.tolist(),centered_radius_float64=r,sigmoid_local_factors=slope.tolist(),
                    angular_logit_jacobian=None,normalization_operator_norm=None,source_branch='epsilon/invalid branch; no regular-branch inference')
    e=c/r;j=(-np.array([c[1],-c[0]])/(r*r))*slope
    return dict(raw=u.tolist(),centered_radius_float64=r,sigmoid_local_factors=slope.tolist(),
        angular_logit_jacobian=j.tolist(),angular_logit_jacobian_norm=float(np.linalg.norm(j)),
        normalization_operator_norm=1/r,normalized=e.tolist(),source_branch='regular radius>=1e-6',
        encoding_logit_jacobian_norm=float(np.linalg.norm((2/r)*(np.eye(2)-np.outer(e,e))*slope)))


def all_zero(groups):
    require(set(groups)==set(GROUPS),'Incomplete N1 gradient groups')
    require(all(g['finite'] and g['missing']==0 for g in groups.values()),'Missing/nonfinite gradient observation')
    require(all((g['nonzero']==0)==(g['squared_norm']==0) for g in groups.values()),'Inconsistent exact-zero gradient records')
    return all(g['nonzero']==0 for g in groups.values())


def history_records(row):
    records=row['history'];labels=[r['update'] for r in records]
    require(labels==[1]+list(range(10,5001,10)),'Missing/duplicate/unexpected logged updates; no interpolation')
    require(all(r['loss_at_update']==r['update']-1 for r in records),'Pre-update labels differ')
    return records


def segment_phase(state,resumed):
    return ('original_prefix' if state<1000 else 'authorized_replay' if state<4270 else 'new_progress') if resumed else 'single_segment'


def endpoint_counts(catalog):
    coverage(catalog)
    return {a:dict(physical_pass=sum(x['evaluation']['success']['all_passed'] for x in catalog if x['arm']==a),
        RGB_pass=sum(x['evaluation']['success']['checks']['held_out_radiance'] for x in catalog if x['arm']==a),
        total=sum(x['arm']==a for x in catalog)) for a in ('D0','N0','N1')}


def classify(row,evaluation):
    history=history_records(row);zero=[all_zero(r['gradient_groups']) for r in history]
    first=next((i for i,z in enumerate(zero) if z),None)
    onset=None
    if first is not None:
        point=history[first];previous=history[first-1] if first else None
        onset=dict(first_zero_log_label=point['update'],first_zero_state_update=point['loss_at_update'],
            last_recorded_nonzero_state_update=previous['loss_at_update'] if previous else None,
            possible_onset_completed_updates=[previous['loss_at_update']+1,point['loss_at_update']] if previous else None,
            first_zero_loss=point['loss'],previous_observed_loss=previous['loss'] if previous else None,
            observed_loss_increased=point['loss']>previous['loss'] if previous else None,
            later_nonzero_records=sum(not z for z in zero[first:]),zero_records=sum(zero),
            boundary_at_first_zero=bool((boundary_distances(point['physical_before_update'])==0).all()))
    if first is not None and all(zero[first:]) and onset['boundary_at_first_zero']:
        group='A_complete_boundary_saturation'
    elif evaluation['success']['all_passed']:group='B_physical_recovery'
    elif evaluation['success']['checks']['held_out_radiance']:group='C_photometric_near_miss'
    else:group='D_nonzero_gradient_failure_floor_approach'
    return dict(group=group,onset=onset,exact_zero_records=sum(zero),logged_records=len(history),
        endpoint_boundary_distances=boundary_distances(evaluation['physical']).tolist(),
        initial_boundary_distances=boundary_distances(history[0]['physical_before_update']).tolist(),
        scope='Saved pre-update observations only; onset between observations is interval-censored')


def stats(a):
    a=np.asarray(a,np.float64);require(np.isfinite(a).all(),'Nonfinite saved numeric data')
    return dict(elements=int(a.size),L2=float(np.linalg.norm(a)),RMS=float(np.sqrt(np.mean(a*a))),
        max_absolute=float(abs(a).max()),nonzero=int(np.count_nonzero(a)),
        absolute_quantiles=np.quantile(abs(a),[0,.5,.9,.99,1]).tolist())


def moment_direction(first,second,step,options):
    """Algebraic stored-moment magnitude, not a next gradient or Adam execution."""
    m,v=np.asarray(first,np.float64),np.asarray(second,np.float64)
    require(m.shape==v.shape and (v>=0).all() and step>0,'Invalid Adam moments')
    b1,b2=options['betas']
    return (options['lr']/(1-b1**step))*m/(np.sqrt(v)/np.sqrt(1-b2**step)+options['eps'])


def checkpoint_arrays(root,selection,row,freeze,run_id):
    path=saved.child(root,selection['manifest']);meta=select(path,selection['manifest_sha256'])
    expected=dict(freeze_sha256=PINS['freeze'][1],configuration_sha256=freeze['configuration_sha256'],
                  source_sha256=freeze['sha256'],run_id=run_id)
    require(meta['execution_provenance']==expected and meta['optimizer_effective']==freeze['configuration']['optimizer_effective'] and
            meta['completed_updates']==selection['completed_updates'],'Checkpoint provenance/options differ')
    archive=saved.child(path.parent,meta['archive']);require(saved.digest(archive)==meta['archive_sha256'],'NPZ identity differs')
    entries=meta['entries'];require(len({e['key'] for e in entries})==len(entries),'Duplicate numeric key')
    values={}
    with np.load(archive,allow_pickle=False) as arrays:
        require(set(arrays.files)=={e['key'] for e in entries}|{'numpy_rng'},'NPZ catalog differs')
        for e in entries:
            a=arrays[e['key']];require(a.dtype.kind!='O' and list(a.shape)==e['shape'] and
                'torch.'+str(a.dtype)==e['dtype'] and saved.array_sha(a)==e['sha256'] and np.isfinite(a).all(), 'Numeric array catalog/hash differs')
            key=(e['role'],e['name']);require(key not in values,'Duplicate numeric role/name')
            a.setflags(write=False);values[key]=a
        require(saved.array_sha(arrays['numpy_rng'])==meta['numpy_rng']['array_sha256'],'RNG array hash differs')
    return values,dict(manifest=file_id(path),archive=file_id(archive))


def checkpoint_summary(values,step,options):
    names=[n for role,n in values if role=='parameter'];out={}
    for group in sorted({n.split('.')[0] for n in names}):
        selected=[n for n in names if n.split('.')[0]==group]
        p=np.concatenate([values['parameter',n].ravel() for n in selected])
        m=np.concatenate([values['adam',n+'/exp_avg'].ravel() for n in selected])
        v=np.concatenate([values['adam',n+'/exp_avg_sq'].ravel() for n in selected])
        require(all(float(values['adam',n+'/step'])==step for n in selected),'Adam counter differs')
        direction=moment_direction(m,v,step,options)
        out[group]=dict(parameters=stats(p),first_moment=stats(m),second_moment=stats(v),
            stored_moment_direction=stats(direction),direction_to_parameter_L2_ratio=float(np.linalg.norm(direction)/np.linalg.norm(p)) if np.linalg.norm(p)>0 else None,
            coordinate_scope='Stored weight_g/weight_v/bias coordinates, not invariant function-space distance')
    return out


def difference(previous,current):
    names=[n for role,n in previous if role=='parameter'];result={}
    for group in sorted({n.split('.')[0] for n in names}):
        nn=[n for n in names if n.split('.')[0]==group]
        a=np.concatenate([previous['parameter',n].ravel() for n in nn]).astype(np.float64)
        b=np.concatenate([current['parameter',n].ravel() for n in nn]).astype(np.float64);delta=b-a
        result[group]=dict(**stats(delta),changed_elements=int(np.count_nonzero(a!=b)),
            relative_L2=float(np.linalg.norm(delta)/np.linalg.norm(a)) if np.linalg.norm(a)>0 else None,
            scope='Net displacement between saved states; not cumulative travel or one-step update')
    return result


def admit():
    evidence={k:select(ROOT/p,s) for k,(p,s) in PINS.items()};f=evidence['freeze'];old=evidence['original'];complete=evidence['complete']
    require(all(saved.digest(ROOT/p)==s for p,s in f['sha256'].items()),'Frozen source hash differs')
    require(complete['combined_catalog_complete'] and complete['continuation_verified_complete'] and
            not complete['original_run_scientifically_complete'] and not old['scientifically_complete'],'Segmented execution mislabeled')
    for record in list(old['identities'].values())+list(complete['inputs'].values())+[complete['continuation_materialized_report']]:
        if isinstance(record,dict) and 'path' in record:require(saved.digest(record['path'])==record['sha256'],'Evidence identity differs')
    for base,records in ((Path(old['output_root']),old['original_artifact_inventory']),
                         (Path(complete['authorization']['output_root']),complete['continuation_artifacts'])):
        for item in records:require(saved.digest(base/item['path'])==item['sha256'],'Original evidence changed')
    for p,s in complete['authorization']['continuation_sources'].items():require(saved.digest(ROOT/p)==s,'Continuation source differs')
    require(saved.digest(ROOT/'experiments/finalize_neural_material_continuation_v2.py')==complete['analysis_source_sha256'],'Finalizer source differs')
    original=select(old['identities']['materialized_report']['path'],old['identities']['materialized_report']['sha256'])
    continuation=select(complete['continuation_materialized_report']['path'],complete['continuation_materialized_report']['sha256'])
    checkpoint_evidence(Path(old['output_root']),original,f,PINS['freeze'][1])
    own=dict(run_id=continuation['run_id'],outcomes=[dict(r,snapshots={k:v for k,v in r['snapshots'].items()
        if not(r['material']=='M1' and k in ('250','1000'))}) for r in continuation['outcomes']])
    checkpoint_evidence(Path(complete['authorization']['output_root']),own,f,PINS['freeze'][1])
    rows={(r['material'],r['seed'],r['arm']):(r,'original',Path(old['output_root']),original['run_id']) for r in original['outcomes'] if r['status']=='complete'}
    require(len(rows)==20,'Original completed selection differs')
    evidence['initial_admission']=original['initial_admission']
    for r in continuation['outcomes']:
        key=(r['material'],r['seed'],r['arm']);require(key not in rows,'Duplicate endpoint source')
        rows[key]=(r,'continuation',Path(complete['authorization']['output_root']),continuation['run_id'])
    coverage([v[0] for v in rows.values()]);reference=select(ROOT/saved.REFERENCE,saved.REFERENCE_SHA)
    return evidence,rows,reference


def trajectory_row(record,material,seed,resumed):
    data=record['material_before_update'];raw=data['rotation_raw']['values'][0]
    state=record['loss_at_update'];o=orientation(raw);p=record['physical_before_update']
    phase=segment_phase(state,resumed)
    return dict(material=material,seed=seed,logged_update=record['update'],state_update=state,
        phase='physical/loss/gradients PRE-update',work_and_time_phase='POST optimizer return',segment_phase=phase,loss=record['loss'],
        physical=p,raw_orientation=raw,encoding=data['encoding']['values'][0],
        encoding_norm=float(np.linalg.norm(data['encoding']['values'][0])),
        recorded_encoding_norm=data['shader_input_norm']['values'][0][0],
        recorded_centered_radius=data['centered_radius']['values'][0][0],orientation_analysis=o,
        boundary_distances=boundary_distances(p).tolist(),gradient_norm=record['gradient_norm'],
        gradient_groups=record['gradient_groups'],work=record['work'],active_seconds=record['active_wall_seconds'])


def analyze():
    evidence,rows,reference=admit();f=evidence['freeze'];catalog=[];n1=[];checkpoints=[];trajectories=[];comparisons=[]
    for key,(row,segment,root,run_id) in sorted(rows.items()):
        require(row['status']=='complete' and row['completed_updates']==5000,'Incomplete endpoint')
        ev=saved.verify_evaluation(row,reference,f);hist=history_records(row);resumed=key==('M1',105,'N1')
        source=dict(run_id=run_id,segment=segment,restored_and_replayed=resumed,
                    initial_history_run_id=evidence['original']['run_id'] if resumed else run_id)
        gate=next(g for g in evidence['original']['verification']['controls']['D0'] if (g['material'],g['seed'])==key[:2])
        require(gate['runtime_full_prefix_passed'],'Missing admitted baseline')
        admission=next(g for g in evidence['initial_admission'] if (g['material'],g['seed'])==key[:2])
        cat=dict(material=key[0],seed=key[1],arm=key[2],source=source,completed_updates=5000,
            initial_physical=hist[0]['physical_before_update'],final_physical=ev['physical'],
            initial_fit_MSE=hist[0]['loss'],initial_fit_state_update=0,final_fit_MSE=row['final_fit_MSE'],
            final_fit_state_update=5000,evaluation=ev,available_records=len(hist),
            available_checkpoint_updates=sorted(map(int,row['snapshots'])),
            initial_gradient_groups=hist[0]['gradient_groups'],final_gradient_groups=row['final_gradient_groups'],
            initial_admission_gradient_groups=admission['gradients'][('D0','N0','N1').index(key[2])],
            provenance_limitations=['Original process exit/cause unavailable; not one uninterrupted run'] if segment=='original' or resumed else [],
            activation_and_physical_upstream_gradients_available=False,initial_full_weights_available=False if key[2]=='N1' else None)
        catalog.append(cat);previous=None;previous_step=None;state_summaries=[]
        for step,snap in sorted(row['snapshots'].items(),key=lambda kv:int(kv[0])):
            inherited=resumed and step in ('250','1000');sr=Path(evidence['original']['output_root']) if inherited else root
            sid=evidence['original']['run_id'] if inherited else run_id
            values,identity=checkpoint_arrays(sr,snap,row,f,sid)
            state=dict(material=key[0],seed=key[1],arm=key[2],update=int(step),phase='POST-update',source_run_id=sid,
                identity=identity,groups=checkpoint_summary(values,int(step),f['configuration']['optimizer_effective']),
                parameter_displacement_from_previous=difference(previous,values) if previous is not None else None,
                previous_checkpoint_update=previous_step,
                parameter_tensors_changed_from_initial_hash=sum(saved.array_sha(v)!=row['initial_model']['parameters'][name]['sha256']
                    for (role,name),v in values.items() if role=='parameter'))
            checkpoints.append(state);state_summaries.append(state);previous=values;previous_step=int(step)
        if key[2]=='N1':
            points=[trajectory_row(r,key[0],key[1],resumed) for r in hist];trajectories.extend(points)
            case=dict(material=key[0],seed=key[1],classification=classify(row,ev),
                initial_gradient_norm=hist[0]['gradient_norm'],endpoint=ev,
                initial_group_norms={g:float(np.sqrt(hist[0]['gradient_groups'][g]['squared_norm'])) for g in GROUPS},
                first_step_real_arithmetic_update_L2_upper_bound=.01*float(np.sqrt(sum(g['nonzero'] for g in hist[0]['gradient_groups'].values()))),
                first_step_bound_scope='LR*sqrt(initial nonzero coordinates); not measured rounded displacement or output change',
                minimum_recorded_radius=min(p['recorded_centered_radius'] for p in points),
                maximum_encoding_norm_error=max(abs(p['encoding_norm']-1) for p in points),
                minimum_angular_logit_jacobian_norm=min([p['orientation_analysis']['angular_logit_jacobian_norm'] for p in points
                    if p['orientation_analysis'].get('angular_logit_jacobian_norm') is not None],default=None),
                checkpoints=state_summaries,source=source)
            n1.append(case)
    coverage(catalog);idx={(x['material'],x['seed'],x['arm']):x for x in catalog}
    for m in ('M1','M2','M3'):
        for s in (101,103,105):
            d,n=(idx[m,s,a]['evaluation'] for a in ('D0','N1'))
            comparisons.append(dict(material=m,seed=s,D0=d,N1=n,
                N1_Q_improves=n['recovery_errors']['Q_relative_frobenius']<d['recovery_errors']['Q_relative_frobenius'],
                N1_held_out_improves=n['radiance_errors']['held_out']['total']['relative_L2']<d['radiance_errors']['held_out']['total']['relative_L2'],
                N1_relight_improves=n['radiance_errors']['relight']['total']['relative_L2']<d['radiance_errors']['relight']['total']['relative_L2'],
                N1_physical_pass=n['success']['all_passed'],D0_physical_pass=d['success']['all_passed']))
    counts=endpoint_counts(catalog)
    require([counts[a]['physical_pass'] for a in ('D0','N0','N1')]==[4,4,1],'Recovery counts differ')
    require(all(idx[m,s,'D0']['evaluation']==idx[m,s,'N0']['evaluation'] for m in ('M1','M2','M3') for s in (101,103,105)), 'Endpoint exact control differs')
    adam=Path(metadata.distribution('torch').locate_file('torch/optim/adam.py'))
    result=dict(schema_version=1,protocol='neural-material-failure-analysis-e1-v1',baseline_commit='e380a5a464b18de24204e6ac4c8a789a0ab73a14',
        inputs={k:dict(path=str(ROOT/p),sha256=s) for k,(p,s) in PINS.items()},
        source_sha256={p:saved.digest(ROOT/p) for p in ('experiments/neural_material_field_v1.py','scientific_sources/material_a/network/field.py',
            'experiments/material_shader_v1.py','experiments/neural_material_recovery_v1.py','experiments/analyze_neural_material_recovery_v1.py',
            'experiments/continue_neural_material_logged_v2.py','experiments/finalize_neural_material_continuation_v2.py')},
        optimizer_source_now=file_id(adam),optimizer_source_historical_identity_verified=False,
        optimizer_options=evidence['original']['optimizer'],physical_success_criteria=evidence['original']['success_thresholds'],
        verification=dict(original_files=131,continuation_files=45,frozen_sources=348,numeric_archives=81,
            endpoints_recomputed=27,unique_combinations=27,source_runs=2,physical_returns=138270,
            logical_returns=135000,duplicate_replay_returns=3270,scientific_forward_calls=0,backward_calls=0,optimizer_steps=0,
            original_run_scientifically_complete=False,combined_catalog_complete=True),
        counts=counts,catalog=catalog,N1_cases=n1,paired_endpoints=comparisons,
        D0_loss_curves={m+'/'+str(s):[[r['loss_at_update'],r['loss']] for r in rows[m,s,'D0'][0]['history']]
                        for m in ('M1','M2','M3') for s in (101,103,105)},
        checkpoint_diagnostics=checkpoints,analysis_source_sha256=saved.digest(Path(__file__).resolve()),
        numerical_scope='Float64 NumPy statistics of saved float32 arrays; no model queries or optimizer execution',
        limitations=['No per-update weights/logits or physical upstream loss gradients in updates1..9.',
            'Initial full neural weights are hashes only in selected run records; first full snapshot is250.',
            'Stored-moment algebra is not an observed last/next optimizer update; interval displacement is not total travel.',
            'Raw g/v/bias coordinate changes are not weight-normalization-invariant function changes.',
            'Exact original interruption cause/exit not independently known; continuation replay is explicit.'])
    return result,trajectories


def write_csv(path,rows):
    require(rows and not path.exists(),'Fresh nonempty CSV required')
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def artifacts(report,points,out):
    catalog=[];cases=[]
    for c in report['catalog']:
        e=c['evaluation'];r=e['recovery_errors'];catalog.append(dict(material=c['material'],seed=c['seed'],arm=c['arm'],
            source_run=c['source']['run_id'],segment=c['source']['segment'],restored_replay=c['source']['restored_and_replayed'],
            updates=5000,initial_fit=c['initial_fit_MSE'],final_fit=c['final_fit_MSE'],Q=r['Q_relative_frobenius'],
            orientation_degrees=r['orientation_degrees'],orientation_identifiable=r['orientation_identifiable'],
            max_albedo=max(r['albedo_absolute_errors']),metallic=r['metallic_absolute_error'],
            held_out=e['radiance_errors']['held_out']['total']['relative_L2'],relight=e['radiance_errors']['relight']['total']['relative_L2'],physical_pass=e['success']['all_passed']))
    write_csv(out/'catalog.csv',catalog)
    for c in report['N1_cases']:
        k=c['classification'];o=k['onset'] or {};cases.append(dict(material=c['material'],seed=c['seed'],group=k['group'],
            first_zero_log_label=o.get('first_zero_log_label'),first_zero_state=o.get('first_zero_state_update'),
            last_recorded_nonzero_state=o.get('last_recorded_nonzero_state_update'),zero_records=k['exact_zero_records'],
            observed_loss_increased=o.get('observed_loss_increased'),later_nonzero_records=o.get('later_nonzero_records'),
            min_radius=c['minimum_recorded_radius'],max_encoding_norm_error=c['maximum_encoding_norm_error']))
    write_csv(out/'classification.csv',cases)
    pairs=[]
    for p in report['paired_endpoints']:
        row=dict(material=p['material'],seed=p['seed'])
        for arm in ('D0','N1'):
            e=p[arm];c=e['component_compensation']['held_out']
            row.update({arm+'_Q':e['recovery_errors']['Q_relative_frobenius'],
                arm+'_held_out':e['radiance_errors']['held_out']['total']['relative_L2'],
                arm+'_relight':e['radiance_errors']['relight']['total']['relative_L2'],
                arm+'_physical_pass':e['success']['all_passed'],arm+'_held_out_residual_correlation':c['residual_correlation'],
                arm+'_component_cancellation_present':c['diffuse_specular_inner_product']<0})
        pairs.append(row)
    write_csv(out/'paired_endpoints.csv',pairs)
    with (out/'trajectories.json').open('x') as f:json.dump(points,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
    plot(report,points,out)


def plot(report,points,out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.hashsalt':'neural-material-E1'})
    lookup={(c['material'],c['seed']):c for c in report['N1_cases']}
    for kind in ('loss','roughness','color_metallic','orientation','gradients','boundary_distance'):
        fig,axes=plt.subplots(3,3,figsize=(11,8),sharex=True)
        for ax,((m,s),case) in zip(axes.flat,sorted(lookup.items())):
            pp=[p for p in points if (p['material'],p['seed'])==(m,s)];x=np.array([p['state_update'] for p in pp]);physical=np.array([p['physical'] for p in pp]);end=case['endpoint']['physical']
            if kind=='loss':
                ax.plot(x,[p['loss'] for p in pp],label='N1 pre-update');direct=np.asarray(report['D0_loss_curves'][m+'/'+str(s)]);ax.plot(direct[:,0],direct[:,1],color='gray',linestyle='--',label='paired D0');c=next(c for c in report['catalog'] if (c['material'],c['seed'],c['arm'])==(m,s,'N1'));ax.scatter([5000],[c['final_fit_MSE']],marker='s',s=18,label='N1 terminal');ax.set_yscale('log');ax.set_ylabel('Fitting linear-RGB MSE')
            elif kind in ('roughness','color_metallic'):
                indices=(0,1) if kind=='roughness' else (3,4,5,6);labels=('mx','my') if kind=='roughness' else ('kR','kG','kB','metallic')
                for i,label in zip(indices,labels):line,=ax.plot(x,physical[:,i],label=label);ax.scatter([5000],[end[i]],marker='s',s=12,color=line.get_color())
                ax.set_ylim((.02,1.02) if kind=='roughness' else (-.02,1.02));ax.set_ylabel('Physical width' if kind=='roughness' else 'Physical output')
            elif kind=='orientation':
                ax.plot(x,np.rad2deg(physical[:,2]),color='tab:orange',label='theta (deg)');ax.set_ylabel('Theta degrees (raw branch)');other=ax.twinx();other.plot(x,[p['encoding_norm'] for p in pp],label='encoding norm',color='tab:blue');other.plot(x,[p['recorded_centered_radius'] for p in pp],label='centered radius',color='tab:green',linestyle='--');other.set_ylim(0,2);other.set_ylabel('Norm/radius');ax.scatter([5000],[np.rad2deg(end[2])],marker='s',s=12,color='tab:orange')
            elif kind=='gradients':
                for group in GROUPS:
                    y=np.sqrt([p['gradient_groups'][group]['squared_norm'] for p in pp]);line,=ax.plot(x,y,label=group.replace('_predictor','').replace('_network',''));zero=y==0;ax.scatter(x[zero],y[zero],s=3,marker='x',color=line.get_color())
                ax.set_yscale('symlog',linthresh=1e-12);ax.set_ylabel('Gradient group L2 (zeros at 0)')
            else:
                d=np.array([p['boundary_distances'] for p in pp])
                for i,label in enumerate(('mx','my','kR','kG','kB','metallic')):ax.plot(x,d[:,i],label=label)
                ax.set_yscale('symlog',linthresh=1e-12);ax.set_ylabel('Distance to float32 bounds')
            ax.set_title(m+'/'+str(s)+' · '+case['classification']['group'][0]);ax.set_xlabel('Completed updates of observed state');ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(alpha=.2)
            if kind!='orientation' and m=='M1' and s==101:ax.legend(fontsize=6,loc='best')
        fig.suptitle('E1 saved observations: '+kind.replace('_',' ')+' (no interpolated intermediate measurements)',fontsize=11)
        if kind=='orientation':
            handles,labels=ax.get_legend_handles_labels();hh,ll=other.get_legend_handles_labels()
            fig.legend(handles+hh,labels+ll,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,fontsize=8)
        fig.tight_layout(rect=(0,0,1,.94 if kind=='orientation' else .97));fig.savefig(out/(kind+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'E1 offline NumPy analysis'})
        fig.savefig(out/(kind+'.png'),dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    require(out.is_absolute() and out.resolve()==out and not out.exists() and out.parent==Path('/home/isavand/paper/evidence') and
            out.parent.stat().st_uid==os.geteuid(),'Fresh explicit owned analysis root required')
    report,points=analyze();out.mkdir(exist_ok=False);artifacts(report,points,out)
    report['artifacts']={f.name:file_id(f) for f in sorted(out.iterdir()) if f.is_file()}
    with (out/'analysis.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    # Recheck every original artifact/source after analysis; no writes to inputs.
    admit()
    print(json.dumps(dict(report=file_id(out/'analysis.json'),counts=report['counts'],scientific_operations=0),sort_keys=True))


if __name__=='__main__':main()
