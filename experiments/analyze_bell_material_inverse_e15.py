"""File-only E.15 recovery analysis. No renderer, Torch, optimizer or new queries."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import bell_material_inverse_e15 as study
from experiments.analyze_bell_highlight_e11d import replay_equal,verify_layout,record
from experiments.bell_highlight_e11d import write_json,save_figure
from benchmarks.material_inverse_e15 import initial_parameters,physical,valid_parameters
from benchmarks.material_information_e14 import deterministic_codes,linear_bin_edges
from infrastructure.signatures import require
from infrastructure.stage_sources import digest,baseline,PINS

COMMIT='2ac4ada918f4706233a5fd241c2b0963187542b4'
FREEZE_SHA='ebc145566a47ff02268ca27ea98f0c0bbab9f202e05e6d7e04dbd23dc4419bfc'
AUTH_SHA='e28757b11c36efca33bc664565d4967476aaf5fca9e32200e57021e332875145'
ERRORS=('Q_world_relative','Q_local_relative','orientation_degrees','logmean_absolute',
        'logratio_absolute','major_absolute','minor_absolute')


def numeric(path):
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    require(all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in d.values()),'Invalid primitive numeric endpoint: '+str(path))
    return d


def independent_errors(p,truth,labels):
    def tensor(z):
        a=np.exp(2*z[...,1]+z[...,2]);b=np.exp(2*z[...,1]-z[...,2]);c=np.cos(z[...,0]);s=np.sin(z[...,0])
        return np.stack((a*c*c+b*s*s,(a-b)*c*s,(a-b)*c*s,a*s*s+b*c*c),-1).reshape(z.shape[:-1]+(2,2))
    q=tensor(p);t=tensor(truth);B=np.stack((labels['T'],labels['B']),-1)[:,None]
    world=B@q@B.swapaxes(-1,-2);target=B@t@B.swapaxes(-1,-2)
    widths=np.stack((np.exp(p[...,1]+p[...,2]/2),np.exp(p[...,1]-p[...,2]/2)),-1)
    tw=np.stack((np.exp(truth[...,1]+truth[...,2]/2),np.exp(truth[...,1]-truth[...,2]/2)),-1)
    angle=abs((p[...,0]-truth[...,0]+np.pi/2)%np.pi-np.pi/2)
    return dict(Q_world_relative=np.linalg.norm(world-target,axis=(-2,-1))/np.linalg.norm(target,axis=(-2,-1)),
        Q_local_relative=np.linalg.norm(q-t,axis=(-2,-1))/np.linalg.norm(t,axis=(-2,-1)),
        orientation_degrees=np.rad2deg(angle),logmean_absolute=abs(p[...,1]-truth[...,1]),
        logratio_absolute=abs(p[...,2]-truth[...,2]),major_absolute=abs(widths[...,0]-tw[...,0]),
        minor_absolute=abs(widths[...,1]-tw[...,1]))


def independent_objective(d,panel):
    use=d['train_views'];ids=d['observation_point_indices'];L=panel['lights']
    a=d['predicted_RGB'][use,:,:L];target=d['target_RGB'][use,:L][:,None]
    if panel['observation']=='linear':r=a-target
    else:
        code=d['target_codes'][use,:L][:,None].astype(int);e=linear_bin_edges()
        r=np.where(a<e[code],a-e[code],np.where(a>e[code+1],a-e[code+1],0.))
    result=np.zeros((84,3))
    for point in range(84):result[point]=(r[ids[use]==point]**2).mean((0,2,3))
    return result


def trajectory(t,opt,panel):
    initial=np.asarray(t['initial']);p=initial.copy();accepted=np.zeros((84,3),np.int64);trials=accepted.copy()
    active=np.ones((84,3),bool);termination=np.full((84,3),'outer_limit',dtype='<U32')
    forwards=0;jacobians=0;damping=np.full((84,3),study.POLICY['initial_damping'])
    require(t['panel']==panel and len(t['history'])<=20,'Trajectory panel/budget mismatch')
    for index,row in enumerate(t['history']):
        require(row['outer']==index and np.array_equal(row['parameters'],p) and
            np.array_equal(row['accepted_before'],accepted),'Trajectory state continuity differs')
        forwards+=1;jacobians+=1;gradient=np.asarray(row['gradient_infinity'])
        require(gradient.shape==(84,3) and np.isfinite(gradient).all() and (gradient>=0).all(),'Invalid saved gradient')
        stationary=active&(gradient<=study.POLICY['gradient_infinity_stop']);active[stationary]=False
        termination[stationary]='fitting_stationary';pending=active.copy();taken=np.zeros_like(active)
        require(len(row['trials'])<=3,'Trial ceiling exceeded')
        for trial,detail in enumerate(row['trials']):
            require(detail['index']==trial,'Trial order differs')
            eligible=np.asarray(detail['eligible']);take=np.asarray(detail['accepted']);step=np.asarray(detail['step_norm'])
            require(eligible.dtype==take.dtype==np.bool_ and eligible.shape==take.shape==(84,3) and
                not (eligible&~pending).any() and not (take&~eligible).any() and not (take&taken).any(),
                'Invalid accepted/pending trial classifications')
            require(np.isfinite(step).all() and (step>=0).all() and (step<=.35+1e-12).all(),'Step-radius metadata differs')
            if eligible.any():forwards+=1
            require((np.asarray(detail['loss_proposed'])[take]<np.asarray(row['loss_before'])[take]).all(),
                    'Accepted trial did not reduce fitting loss')
            next_damping=np.asarray(detail['damping'])
            rejected=pending&~take
            require(np.array_equal(next_damping[rejected],np.minimum(damping[rejected]*10,1e10)),'Rejected-trial damping differs')
            require(np.array_equal(next_damping[~pending],damping[~pending]),'Inactive damping changed')
            options=np.stack([np.clip(damping/3,1e-10,1e10),damping,np.clip(damping*2,1e-10,1e10)])
            require(np.any(options[:,take]==next_damping[take],axis=0).all(),'Accepted damping outside frozen policy')
            damping=next_damping;trials+=pending;accepted+=take;taken|=take
            tiny=take&(step<=1e-8);active[tiny]=False;termination[tiny]='accepted_step_stationary';pending&=~take
        after=np.asarray(row['parameters_after']);require(valid_parameters(after).all(),'Invalid saved physical parameters')
        require(np.array_equal(after[~taken],p[~taken]) and np.array_equal(row['accepted_after'],accepted),
                'Unaccepted state moved or accepted count differs')
        p=after
    require(np.array_equal(p,opt['parameters']) and np.array_equal(accepted,opt['accepted']) and
        np.array_equal(trials,opt['trials']) and np.array_equal(damping,opt['damping']) and
        np.array_equal(termination,t['termination']),'Terminal optimizer state/termination differs')
    return forwards,jacobians,initial


def verify(batch,authorization):
    fp=ROOT/'docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.freeze.json'
    require(digest(fp)==FREEZE_SHA and digest(authorization)==AUTH_SHA,'Approved freeze/authorization changed')
    f=json.loads(fp.read_text());a=json.loads(authorization.read_text());launch=json.loads((batch/'launch.json').read_text())
    r=json.loads((batch/'recovery/recovery.json').read_text());roots={k:Path(v) for k,v in f['host_inputs'].items()}
    require(str(batch)==f['host_output_directory']==a['host_output_directory']==launch['batch'] and
        r['commit']==a['commit']==launch['commit']==COMMIT and r['freeze_sha256']==a['freeze_sha256']==launch['freeze_sha256']==FREEZE_SHA and
        r['authorization_sha256']==launch['authorization_sha256']==AUTH_SHA and a['approved'] is True and
        a['output_directory']=='/outputs/recovery' and a['image_id']==r['backend']['image_id']==f['image_id'] and
        a['selected_gpu']==launch['gpu']=='0','Execution approval/launch identity mismatch')
    require(r['protocol']==a['protocol']==f['protocol']==study.PROTOCOL and r['files']==f['files'] and
        r['policy']==f['policy']==study.POLICY and r['limits']==f['limits']==a['limits']==study.LIMITS and
        r['starts']==f['starts'] and r['backend']==study.identity(),'Scientific identity mismatch')
    require(r['status']=='complete' and r['scientific_work_complete'] is True and r['cleanup_verified'] is True and
        r['admission']['status']=='pass' and (batch/'exit-code.txt').read_text().strip()=='0' and len(r['panels'])==16,
        'Incomplete execution, admission or cleanup')
    require(all(str(p)==launch[k] for k,p in roots.items()),'Input mount identity mismatch')
    _,data=study.definition(argparse.Namespace(**roots,freeze=fp,freeze_sha256=FREEZE_SHA))
    for stage in PINS:baseline(stage,PINS[stage])
    mounts=dict(v3=roots['benchmark'],e11d=roots['pilot'],e11e=roots['reference'],e11f=roots['factor'],e11g=roots['multiview'],e14=roots['two_light'])
    consumed=[]
    for row in r['inputs']:
        p=Path(row['path']);require(p.parts[:2]==('/','inputs') and p.parts[2] in mounts,'Unexpected consumed path')
        host=mounts[p.parts[2]].joinpath(*p.parts[3:]);require(host.resolve()==host and mounts[p.parts[2]] in host.parents and
            host.stat().st_size==row['bytes'] and digest(host)==row['sha256'],'Input artifact changed')
        consumed.append(record(host))
    require(consumed==data[4],'Consumed input list differs')
    native=batch/'recovery';expected={'panel-%02d-%s'%(i,s) for i in range(16) for s in ('progress.json','trajectory.json','optimizer.npz','endpoint.npz')}
    require(len(r['outputs'])==64 and {x['path'] for x in r['outputs']}==expected,'Output inventory differs')
    for row in r['outputs']:
        p=native/row['path'];require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Native output hash differs')
    verify_layout(native,expected|{'recovery.json'})
    require(r['attempted']==r['completed'] and r['forward_attempts']==r['forward_returns'] and
        r['analytic_jacobian_attempts']==r['analytic_jacobian_returns'],'Incomplete work counters')
    require(all(v<=study.LIMITS[k] for k,v in r['attempted'].items()) and
        all(r[k]==0 for k in ('Adam_returns','autograd_reverse_calls','neural_calls','new_geometry_queries','new_reference_datasets')),
        'Work ceilings/zero-query boundary failed')
    require(all(r['completed'][k]==study.LIMITS[k] for k in ('gate_point_directions','endpoint_point_directions',
        'environment_point_directions','coefficient_point_directions')) and
        r['forward_returns']['gate']==9 and r['forward_returns']['endpoint']==32 and
        r['analytic_jacobian_returns']['gate']==1 and r['analytic_jacobian_returns']['endpoint']==0,
        'Fixed admission/endpoint/preparation work differs')
    require(all(x['maximum_RGB_difference']<=2e-12 for x in r['admission']['forward_parity']) and
        all(x['maximum_absolute_error']<=2e-8 for x in r['admission']['derivative_errors']),'Recorded admission errors differ')
    require(r['elapsed_seconds']<3480 and r['host_peak_rss_bytes']<=64*1024**3 and
        sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<1024**3,'Recorded resource ceiling exceeded')
    console=[json.loads(s) for s in (batch/'console.log').read_text().splitlines() if s.strip()]
    require(console==r['panels'],'Console panel receipts differ')
    return f,r,data


def viewscores(d,selected,mode,lights,group):
    ids=d['observation_point_indices'];y=d['predicted_RGB'][np.arange(220),selected[ids]]
    codes=d['predicted_codes'][np.arange(220),selected[ids]];target=d['target_RGB']
    held=d['held_views'];train=d['train_views'];out={}
    for name,mask,light in [('train',train,slice(0,lights)),('held_L2',held,slice(0,1)),
                            ('held_rotated',held,slice(1,2)),('held_both',held,slice(0,2)),
                            ('rotated_all',np.ones(220,bool),slice(1,2))]:
        use=mask&group[ids]
        delta=y[use,light]-target[use,light];code_delta=codes[use,light].astype(int)-d['target_codes'][use,light].astype(int)
        out[name+'_observations']=int(use.sum())
        out[name+'_RGB_RMSE']=float(np.sqrt(np.mean(delta**2))) if use.any() else None
        out[name+'_RGB_relative_L2']=float(np.linalg.norm(delta)/max(np.linalg.norm(target[use,light]),1e-15)) if use.any() else None
        out[name+'_code_MAE']=float(np.mean(abs(code_delta))) if use.any() else None
    return out


def point_rows(d,opt,panel,index,pixels,history):
    rows=[];ids=d['observation_point_indices'];counts=np.bincount(ids[d['train_views']],minlength=84)
    selected=d['selected_by_training'];L=panel['lights']
    for point in range(84):
        mask=ids==point;train=mask&d['train_views'];held=mask&d['held_views']
        for start in range(3):
            y=d['predicted_RGB'][:,start];low=d['coarse_RGB'][:,start]
            grid_abs=float(abs(y[mask]-low[mask]).max());grid_rel=float(np.linalg.norm(y[mask]-low[mask])/max(np.linalg.norm(y[mask]),1e-15))
            code=d['predicted_codes'][train,start,:L];target=d['target_codes'][train,:L]
            row=dict(panel=index,**panel,probe_index=point,pixel_id=int(pixels[point]),start=start,selected_by_training=bool(start==selected[point]),
                fitting_views=int(counts[point]),held_view_available=bool(held.any()),training_objective=float(d['training_objective'][point,start]),
                accepted_updates=int(opt['accepted'][point,start]),trial_attempts=int(opt['trials'][point,start]),
                termination=history['termination'][point][start],orientation_eligible=bool(d['orientation_eligible'][point,0]),
                orientation_prediction_defined=bool(d['orientation_prediction_defined'][point,start]),
                theta=float(d['parameters'][point,start,0]),logmean=float(d['parameters'][point,start,1]),logratio=float(d['parameters'][point,start,2]),
                Q_pass=bool(d['Q_world_relative'][point,start]<=.05),
                orientation_pass=bool(d['orientation_degrees'][point,start]<=5 and d['orientation_eligible'][point,0] and d['orientation_prediction_defined'][point,start]),
                actual_fitting_codes_exact=bool(np.array_equal(code,target)),fitting_code_MAE=float(np.mean(abs(code.astype(int)-target.astype(int)))),
                grid_RGB_maximum=grid_abs,grid_RGB_relative_L2=grid_rel,grid_pass=bool(grid_abs<=.003 and grid_rel<=.002),
                last_pretrial_gradient_infinity=float(history['history'][-1]['gradient_infinity'][point][start]))
            row.update({k:float(d[k][point,start]) for k in ERRORS})
            for name,m in [('held_L2',held),('held_both',held),('train_L2',train)]:
                light=slice(0,2) if name=='held_both' else slice(0,1)
                delta=y[m,light]-d['target_RGB'][m,light]
                row[name+'_relative_L2']=float(np.linalg.norm(delta)/max(np.linalg.norm(d['target_RGB'][m,light]),1e-15)) if m.any() else None
            rows.append(row)
    return rows


def summarize(d,opt,panel,index,pixels,history):
    details=point_rows(d,opt,panel,index,pixels,history);counts=np.bincount(d['observation_point_indices'][d['train_views']],minlength=84)
    summary=[];selections=[('start'+str(i),np.full(84,i,np.int64)) for i in range(3)]+[('fitting_selected',d['selected_by_training'])]
    groups=[('all84',np.ones(84,bool)),('heldout_eligible',np.bincount(d['observation_point_indices'][d['held_views']],minlength=84)>0)]
    groups += [('train_views_'+str(n),counts==n) for n in (1,2,3)]
    for name,selected in selections:
        for group,mask in groups:
            rows=[details[p*3+selected[p]] for p in np.flatnonzero(mask)]
            row=dict(panel=index,**panel,selection=name,group=group,points=len(rows),Q_pass=sum(x['Q_pass'] for x in rows),
                eligible_orientation=sum(x['orientation_eligible'] for x in rows),orientation_pass=sum(x['orientation_pass'] for x in rows),
                grid_unresolved=sum(not x['grid_pass'] for x in rows),actual_fitting_codes_exact=sum(x['actual_fitting_codes_exact'] for x in rows),
                wrong_Q_exact_fitting_codes=sum(x['actual_fitting_codes_exact'] and not x['Q_pass'] for x in rows),
                exactly_zero_interval_objective=sum(x['training_objective']==0 for x in rows),
                outer_limit=sum(x['termination']=='outer_limit' for x in rows))
            for error in ERRORS+('training_objective',):
                values=[x[error] for x in rows if error!='orientation_degrees' or x['orientation_eligible']]
                row[error+'_median']=float(np.median(values)) if values else None
                row[error+'_p95']=float(np.quantile(values,.95)) if values else None
            row.update(viewscores(d,selected,panel['observation'],panel['lights'],mask));summary.append(row)
    return details,summary


def ambiguity(d,panel,index):
    rows=[];ids=d['observation_point_indices'];L=panel['lights']
    for point in range(84):
        use=(ids==point)&d['train_views']
        for a,b in ((0,1),(0,2),(1,2)):
            ya=d['predicted_RGB'][use,a,:L];yb=d['predicted_RGB'][use,b,:L];target=d['target_RGB'][use,:L]
            ca=d['predicted_codes'][use,a,:L];cb=d['predicted_codes'][use,b,:L];ct=d['target_codes'][use,:L]
            delta=float(np.sqrt(np.mean((ya-yb)**2)));angle=float(np.rad2deg(abs((d['parameters'][point,a,0]-d['parameters'][point,b,0]+np.pi/2)%np.pi-np.pi/2)))
            rows.append(dict(panel=index,**panel,probe_index=point,starts=str(a)+'/'+str(b),prediction_RMSE=delta,
                near_photometric=bool(delta<=1e-4),actual_codes_both_match_target=bool(np.array_equal(ca,ct) and np.array_equal(cb,ct)),
                target_RMSE_a=float(np.sqrt(np.mean((ya-target)**2))),target_RMSE_b=float(np.sqrt(np.mean((yb-target)**2))),
                Q_error_a=float(d['Q_world_relative'][point,a]),Q_error_b=float(d['Q_world_relative'][point,b]),
                orientation_separation_degrees=angle,logmean_separation=float(abs(d['parameters'][point,a,1]-d['parameters'][point,b,1])),
                logratio_separation=float(abs(d['parameters'][point,a,2]-d['parameters'][point,b,2]))))
    return rows


def figures(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='bell-physical-e15-offline-v1'
    allrows=[r for r in summary if r['group']=='all84']
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    labels=('L2/fixed','two/fixed','L2/free','two/free')
    for arm,material in enumerate(('original','stronger')):
        for col,observation in enumerate(('linear','codes8')):
            ax=axes[arm,col]
            for offset,selection,color in [(-.18,'start0','C0'),(.18,'fitting_selected','C1')]:
                rows=[next(r for r in allrows if r['material']==material and r['observation']==observation and r['ratio']==ratio and r['lights']==light and r['selection']==selection) for ratio in ('fixed','free') for light in (1,2)]
                ax.bar(np.arange(4)+offset,[100*r['Q_pass']/84 for r in rows],.36,label=selection,color=color)
            ax.set(xticks=np.arange(4),xticklabels=labels,ylim=(0,100),ylabel='Q relative error <=0.05 (%)',title=material+' / '+observation);ax.legend(fontsize=8)
    fig.suptitle('Primary start0 and secondary fitting-selected multistart; all84 retained');fig.tight_layout();save_figure(fig,output/'physical_recovery.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,material in zip(axes,('original','stronger')):
        for observation,color in [('linear','C0'),('codes8','C1')]:
            rows=[next(r for r in allrows if r['material']==material and r['observation']==observation and r['ratio']==ratio and r['lights']==light and r['selection']=='fitting_selected') for ratio in ('fixed','free') for light in (1,2)]
            ax.plot(np.arange(4),[100*r['held_both_RGB_relative_L2'] for r in rows],marker='o',label=observation,color=color)
        ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Held-view both-light RGB relative L2 (%)',title=material);ax.legend()
    fig.suptitle('Same83 withheld views; source and rotated-light errors retained separately in CSV');fig.tight_layout();save_figure(fig,output/'held_view_error.svg');plt.close(fig)


def analyze(batch,authorization,output):
    require(output.is_absolute() and output.resolve()==output and not output.exists() and
            batch not in output.parents and output not in batch.parents,'Fresh separate analysis output required')
    f,r,data=verify(batch,authorization);native=batch/'recovery';labels,pixels=data[:2]
    details=[];summary=[];pairs=[];work_f=0;work_j=0;verification=[]
    for i,panel in enumerate(study.PANELS):
        d=numeric(native/('panel-%02d-endpoint.npz'%i));opt=numeric(native/('panel-%02d-optimizer.npz'%i))
        t=json.loads((native/('panel-%02d-trajectory.json'%i)).read_text());fw,jw,initial=trajectory(t,opt,panel)
        work_f+=fw;work_j+=jw;truth=study.truth_parameters(labels,panel['material']=='stronger')
        views,points=np.nonzero(data[3]['valid'])
        require(d['parameters'].shape==(84,3,3) and d['parameters'].dtype==np.float64 and
            d['predicted_RGB'].shape==d['coarse_RGB'].shape==(220,3,2,3) and
            d['predicted_RGB'].dtype==d['coarse_RGB'].dtype==np.float64 and
            d['predicted_codes'].dtype==d['target_codes'].dtype==np.uint8 and
            np.array_equal(d['observation_view_indices'],views) and np.array_equal(d['observation_point_indices'],points) and
            np.array_equal(d['train_views'],data[6][views,points]) and np.array_equal(d['held_views'],data[7][views,points]),
            'Endpoint shapes/dtypes/split/observation identities differ')
        require(np.array_equal(d['target_RGB'],study.captured_target(data[5][512],int(panel['material']=='stronger'),views,points)),
            'Saved endpoint target differs from accepted independent capture')
        true_width=physical(truth);pred_width=physical(d['parameters'])
        require(np.array_equal(d['orientation_eligible'],true_width[...,0]-true_width[...,1]>=.1) and
            np.array_equal(d['orientation_prediction_defined'],pred_width[...,0]-pred_width[...,1]>1e-6),
            'Orientation eligibility/definition changed')
        require(np.array_equal(d['parameters'],opt['parameters']) and np.array_equal(d['truth'],truth) and
            np.array_equal(initial,initial_parameters(84,truth[:,0,2] if panel['ratio']=='fixed' else None)),
            'Endpoint/truth/initial state identity differs')
        if panel['ratio']=='fixed':require(np.array_equal(d['parameters'][...,2],initial[...,2]),'Oracle ratio moved')
        actual=independent_errors(d['parameters'],truth,labels);error={}
        for name in ERRORS:
            if name=='orientation_degrees':
                delta=float(abs(np.cos(np.deg2rad(actual[name]))-np.cos(np.deg2rad(d[name]))).max())
                require(delta<=64*np.finfo(float).eps,'Independent periodic orientation differs')
            else:
                delta=float(abs(actual[name]-d[name]).max());require(np.allclose(actual[name],d[name],atol=1e-12,rtol=1e-10),'Independent physical errors differ: '+name)
            error[name]=delta
        require(np.array_equal(d['predicted_codes'],deterministic_codes(d['predicted_RGB'])) and
            np.array_equal(d['target_codes'],deterministic_codes(d['target_RGB'])),'Actual code replay differs')
        objective=independent_objective(d,panel)
        require(np.allclose(objective,d['training_objective'],atol=1e-24,rtol=1e-10),'Independent fitting objective differs')
        require(np.array_equal(np.argmin(d['training_objective'],axis=1),d['selected_by_training']),'Fitting-only selection differs')
        require(int(opt['accepted'].sum())==r['panels'][i]['accepted_updates'] and int(opt['trials'].sum())==r['panels'][i]['trial_attempts'],
            'Saved update/trial counts differ')
        a,b=summarize(d,opt,panel,i,pixels,t);details+=a;summary+=b;pairs+=ambiguity(d,panel,i)
        verification.append(dict(panel=i,independent_error_maxima=error,trajectory_forward_calls=fw,trajectory_jacobian_calls=jw))
    require(work_f==r['forward_returns']['fit'] and work_j==r['analytic_jacobian_returns']['fit'] and
        r['completed']['fit_point_directions']==work_f*137*3*512**2 and
        r['analytic_derivative_point_directions']['fit']==3*work_j*137*3*512**2,
        'Trajectory-derived work accounting differs')
    output.mkdir()
    for name,rows in [('all_starts_and_selection',summary),('point_start_results',details),('multistart_pairs',pairs)]:
        with (output/(name+'.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    figures(output,summary)
    result=dict(schema_version=1,protocol='bell-physical-recovery-e15-offline-v1',verification_status='pass',decision='FIX_AND_PROCEED',
        full_neural_benchmark_admitted=False,source_commit=COMMIT,freeze_sha256=FREEZE_SHA,native_report=record(native/'recovery.json'),
        authorization=record(authorization),native_artifacts=[record(p) for p in sorted(batch.rglob('*')) if p.is_file()],
        input_artifacts=data[4],frozen_sources=f['files'],endpoint_verification=verification,
        summary=summary,admission=r['admission'],work=r['completed'],forward_calls=r['forward_returns'],analytic_jacobian_calls=r['analytic_jacobian_returns'],
        accepted_LM_steps=sum(p['accepted_updates'] for p in r['panels']),trial_attempts=sum(p['trial_attempts'] for p in r['panels']),
        termination_counts={k:sum(p['termination_counts'].get(k,0) for p in r['panels']) for k in ('fitting_stationary','outer_limit','accepted_step_stationary')},
        runtime=r['runtime'],backend=r['backend'],elapsed_seconds=r['elapsed_seconds'],
        resources={k:r[k] for k in ('host_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')},
        limitations=['No Docker-daemon inspection; cleanup success is a verified native receipt.',
            'Trial rho/full normal matrix not archived: source policy is frozen but numeric rho cannot be independently replayed.',
            'Per-point grid accuracy is flagged without dropping any result; native panel aggregates are retained.',
            'Truth-fixed ratio and same-kernel known-color/metal recovery are oracle diagnostics, not neural/material-model validation.'],
        new_scientific_queries=0,previous_evidence_replaced=False,
        analysis_source=dict(path='experiments/analyze_bell_material_inverse_e15.py',sha256=digest(Path(__file__))))
    write_json(output/'analysis.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','authorization','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=analyze(a.batch,a.authorization,a.output)
    print(json.dumps(dict(verification=r['verification_status'],decision=r['decision'],accepted_steps=r['accepted_LM_steps'],panels=16)))


if __name__=='__main__':main()
