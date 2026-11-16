"""File-only E16 analysis: primitive saved arrays, no model/render/optimizer calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import bell_neural_material_e16 as study
from benchmarks.neural_observation_e16 import world_Q,compatibility
from benchmarks.material_information_e14 import deterministic_codes
from infrastructure.stage_sources import digest,baseline,PINS
from infrastructure.signatures import require

COMMIT='71687127e36527052aff733e3fcb96ebd0b67887'
FREEZE='10c51b7d8ac1dc1ca59abe657c30f9769a0f50f93893d3761c26022f4a17ba48'
AUTH='ffc23701487396ce9bd2184eeac7ee69dd3340406be4de8d995ffbb03d6ed6da'
ACTIVE={'feats_network','mx_predictor','my_predictor','rotation_predictor'}
INACTIVE={'kd_predictor','metallic_predictor','inner_light','outer_light'}


def record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))


def numeric(p):
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    require(d and all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in d.values()),'Invalid numeric artifact: '+str(p))
    return d


def state_hash(d):
    h=hashlib.sha256()
    for n,a in sorted(d.items()):
        h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def physical(Q,mx,my,axis,other,truth,T,B):
    """Canonical major axis only for evaluation; never modify head ordering."""
    widths=np.stack((mx,my),-1);tw=np.exp(truth[:,1,None]+truth[:,2,None]*np.array([.5,-.5]))
    major_axis=np.where((mx>=my)[:,None],axis,other)
    target_axis=np.cos(truth[:,:1])*T+np.sin(truth[:,:1])*B
    cosine=abs((major_axis*target_axis).sum(-1))/np.linalg.norm(major_axis,axis=-1)/np.linalg.norm(target_axis,axis=-1)
    tq=world_Q(truth,T,B);basis=np.stack((T,B),-1)
    local=basis.swapaxes(1,2)@Q@basis;tl=basis.swapaxes(1,2)@tq@basis
    return dict(Q_world_relative=np.linalg.norm(Q-tq,axis=(1,2))/np.linalg.norm(tq,axis=(1,2)),
        Q_local_relative=np.linalg.norm(local-tl,axis=(1,2))/np.linalg.norm(tl,axis=(1,2)),
        orientation_degrees=np.degrees(np.arccos(np.clip(cosine,0,1))),
        major_absolute=abs(widths.max(1)-tw[:,0]),minor_absolute=abs(widths.min(1)-tw[:,1]),
        logmean_absolute=abs(np.log(np.sqrt(mx*my))-truth[:,1]),
        logratio_absolute=abs(abs(np.log(mx/my))-truth[:,2]),
        orientation_eligible=tw[:,0]-tw[:,1]>=.1,predicted_width_gap_lt_0p1=abs(mx-my)<.1)


def radiometry(draws,target,fine,coarse,mask):
    require(draws.ndim==4 and draws.shape[1:]==target.shape==fine.shape==coarse.shape,'RGB shape mismatch')
    a=draws[:,mask].astype(float);y=target[mask];mean=a.mean(0)
    mse=float(((mean-y)**2).mean());variance=float(((a-mean)**2).mean());single=float(((a-y)**2).mean())
    require(np.isclose(single,mse+variance,atol=1e-18,rtol=1e-12),'Saved-draw bias/variance identity failed')
    codes=deterministic_codes(mean);truth_codes=deterministic_codes(y)
    return dict(native_mean_relative_L2=float(np.linalg.norm(mean-y)/np.linalg.norm(y)),native_mean_MSE=mse,
        native_single_draw_MSE=single,native_empirical_variance=variance,
        independent_relative_L2=float(np.linalg.norm(fine[mask]-y)/np.linalg.norm(y)),
        native_independent_relative_L2=float(np.linalg.norm(mean-fine[mask])/np.linalg.norm(fine[mask])),
        independent_grid_maximum=float(abs(fine[mask]-coarse[mask]).max()),
        independent_grid_relative_L2=float(np.linalg.norm(fine[mask]-coarse[mask])/np.linalg.norm(fine[mask])),
        code_channels_exact=int((codes==truth_codes).sum()),code_channels=int(codes.size),
        code_RGB_exact=int((codes==truth_codes).all(-1).sum()),code_RGB=int(codes.size//3),
        single_draw_code_channel_fraction=float((deterministic_codes(a)==truth_codes).mean()),
        native_mean_clipped_channels=int(((mean<0)|(mean>1)).sum()))


def verify(batch,authorization):
    fp=ROOT/'docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json'
    require(digest(fp)==FREEZE and digest(authorization)==AUTH,'Freeze/authorization changed')
    f=json.loads(fp.read_text());a=json.loads(authorization.read_text());launch=json.loads((batch/'launch.json').read_text())
    r=json.loads((batch/'neural/neural-recovery.json').read_text())
    require(str(batch)==f['host_output_directory']==a['host_output_directory']==launch['batch'] and
        r['commit']==launch['commit']==a['commit']==COMMIT and r['freeze_sha256']==launch['freeze_sha256']==a['freeze_sha256']==FREEZE and
        r['authorization_sha256']==launch['authorization_sha256']==AUTH and a['approved'] is True and
        a['image_id']==f['image_id']==r['runtime']['image_id'] and a['selected_gpu']==launch['gpu']=='0','Launch identity mismatch')
    require(r['files']==f['files'] and r['policy']==f['policy']==study.POLICY and
        r['limits']==f['limits']==a['limits']==study.LIMITS and r['protocol']==a['protocol']==study.PROTOCOL,'Scientific definition differs')
    require((batch/'exit-code.txt').read_text().strip()=='0' and r['status']=='complete' and all(r[k] is True for k in
        ('cleanup_verified','scientific_work_complete','training_admitted','inactive_parameters_verified_unchanged','known_inputs_verified_unchanged')),
        'Incomplete execution or cleanup')
    roots={k:Path(v) for k,v in f['host_inputs'].items()}
    require(all(str(v)==launch[k] for k,v in roots.items()),'Input mounts differ')
    _,data=study.definition(argparse.Namespace(**roots,freeze=fp,freeze_sha256=FREEZE))
    for stage,pin in PINS.items():baseline(stage,pin)
    mapped={'v3':'benchmark','e11d':'pilot','e11e':'reference','e11f':'factor','e11g':'multiview','e14':'two_light','e15':'physical'}
    consumed=[]
    for x in r['inputs']:
        p=Path(x['path']);require(p.parts[:2]==('/','inputs') and p.parts[2] in mapped,'Unexpected native input')
        host=roots[mapped[p.parts[2]]].joinpath(*p.parts[3:])
        require(host.resolve()==host and host.stat().st_size==x['bytes'] and digest(host)==x['sha256'],'Input artifact changed')
        consumed.append(record(host))
    require(consumed==data['records'],'Consumed input order differs')
    native=batch/'neural';paths={x['path'] for x in r['outputs']}
    require(len(paths)==len(r['outputs'])==97 and paths=={p.name for p in native.iterdir() if p.is_file()}-{'neural-recovery.json'},'Output inventory differs')
    for x in r['outputs']:
        p=native/x['path'];require(p.parent==native and p.resolve()==p and p.stat().st_size==x['bytes'] and digest(p)==x['sha256'],'Native artifact changed')
    require(r['attempted']==r['completed'] and all(v<=study.LIMITS[k] for k,v in r['completed'].items()),'Work counter mismatch')
    require(len(r['snapshots'])==12 and r['elapsed_seconds']<3480 and r['host_peak_rss_bytes']<64*1024**3 and
        sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<study.LIMITS['output_bytes'],'Resource/completion mismatch')
    acquisition=numeric(native/'acquisition.npz')
    for k,v in data['known'].items():require(np.array_equal(acquisition[k],v),'Known acquisition input differs: '+k)
    require(np.array_equal(acquisition['pixel_ids'],data['legacy'][1]),'Physical point identities differ')
    views,points=np.nonzero(data['legacy'][3]['valid']);fit=data['legacy'][6][views,points]
    require(np.array_equal(acquisition['observation_views'],views) and np.array_equal(acquisition['observation_points'],points) and
        np.array_equal(acquisition['fitting_indices'],np.flatnonzero(fit)),'Correspondence identity differs')
    replay={}
    for material in study.FIELDS:
        z=numeric(native/(material+'-compatibility.npz'));checks={}
        require(np.array_equal(z['fitting_observations'],np.flatnonzero(fit)) and
            np.array_equal(z['gradient_observations'],np.flatnonzero((views==0)&np.isin(points,study.POLICY['gradient_probe_indices']))) and
            np.array_equal(z['reference'][:,:,0],data['compat_reference'][material]['predicted_RGB'][fit]) and
            np.array_equal(z['coarse_reference'][:,:,0],data['compat_reference'][material]['coarse_RGB'][fit]),'Compatibility reference/point identity mismatch')
        for name,n,k in [('forward','native','reference'),('gradient','native_gradient','reference_gradient')]:
            coarse='coarse_'+k
            result=compatibility(z[n],z[k],abs(z[k]-z[coarse]),study.POLICY['compatibility_absolute' if name=='forward' else 'gradient_absolute'],
                study.POLICY['compatibility_relative' if name=='forward' else 'gradient_relative'])
            for key,value in result.items():
                old=r['compatibility'][material][name][key]
                require(value==old if type(value) is bool or type(value) is int else np.isclose(value,old,atol=1e-14,rtol=1e-12),'Compatibility replay differs: '+key)
            require(result['passed'],'Saved scientific compatibility gate fails');checks[name]=result
        replay[material]=checks
    return f,r,data,acquisition,fit,replay


def weights_and_training(native,r):
    base=numeric(native/'original-A-initial-weights.npz');states={};gradients=[];counts=dict(photo=0,distillation=0)
    material={k:v for k,v in base.items() if k.split('.')[0] in ACTIVE|{'kd_predictor','metallic_predictor'}}
    require(len(material)==84 and sum(v.size for v in material.values())==1485584,'Full material architecture changed')
    for row in r['snapshots']:
        prefix=row['artifact'][:-4];w=numeric(native/(prefix+'-weights.npz'))
        require(set(w)==set(base) and state_hash(w)==row['weight_hash'],'Checkpoint signature/state differs')
        if row['stage']=='initial':require(state_hash(w)==r['initial_state_sha256'],'Initial weights differ')
        groups={}
        for name,v in base.items():
            group=name.split('.')[0];require(w[name].shape==v.shape and w[name].dtype==v.dtype,'Weight metadata changed')
            if group in INACTIVE or group not in ACTIVE:require(np.array_equal(w[name],v),'Inactive parameter/buffer changed')
            else:groups[group]=groups.get(group,0.)+float(((w[name].astype(float)-v)**2).sum())
        states[prefix]=dict(parameter_displacement_L2={k:float(np.sqrt(v)) for k,v in groups.items()})
        if row['stage']!='initial':
            z=numeric(native/(prefix+'-Adam.npz'));steps=[int(v) for k,v in z.items() if k.endswith('.step')]
            expected=200 if row['stage']=='post_distillation' else (2000 if row['stage']=='compute_endpoint' else 1000)
            require(len(steps)==60 and set(steps)=={expected},'Adam reset/step/cap mismatch')
            for k,v in z.items():
                require(k.split('.')[0] in ACTIVE,'Inactive optimizer state')
                if k.endswith('.exp_avg_sq'):require((v>=0).all() and (z[k[:-10]+'max_exp_avg_sq']>=v).all(),'Invalid AMSGrad second moments')
            states[prefix]['Adam_step']=expected;states[prefix]['Adam_parameter_tensors']=len(steps)
    for material in study.FIELDS:
        for arm in ('A','B'):
            rows=json.loads((native/(material+'-'+arm+'-training.json')).read_text())
            phases=('photo',) if arm=='A' else ('distillation','photo')
            require(len(rows)==(2000 if arm=='A' else 1200),'Training record count differs')
            for phase in phases:
                selected=[x for x in rows if x['phase']==phase];expected=200 if phase=='distillation' else (2000 if arm=='A' else 1000)
                require([x['step'] for x in selected]==list(range(expected)),'Step order/phase differs');counts[phase]+=len(selected)
                require(all(set(x['group_gradient_L2'])==ACTIVE for x in selected),'Missing/full-field gradient group')
                for key in ('loss','preclip_gradient_L2'):
                    v=np.array([x[key] for x in selected]);require(np.isfinite(v).all() and (v>=0).all(),'Invalid loss/gradient')
                row=dict(material=material,arm=arm,phase=phase,updates=len(selected),initial_loss=selected[0]['loss'],
                    final_preupdate_loss=selected[-1]['loss'],last100_mean_loss=float(np.mean([x['loss'] for x in selected[-100:]])),
                    clipped_steps=sum(x['preclip_gradient_L2']>.7 for x in selected))
                for group in sorted(ACTIVE):
                    v=np.array([x['group_gradient_L2'][group] for x in selected]);require(np.isfinite(v).all() and (v>=0).all(),'Nonfinite group gradient')
                    row[group+'_minimum']=float(v.min());row[group+'_median']=float(np.median(v));row[group+'_zero_steps']=int((v==0).sum())
                gradients.append(row)
    require(counts==dict(photo=6000,distillation=400) and r['completed']['photometric_Adam_returns']==counts['photo'] and
        r['completed']['distillation_Adam_returns']==counts['distillation'],'Trajectory/count mismatch')
    expected=dict(photo_shader_calls=counts['photo']*2,photo_backwards=counts['photo'],distillation_backwards=counts['distillation'],
        gate_shader_calls=2*32*2,gate_reverse_sweeps=2*32*2*3,eval_shader_calls=12*8*2,
        field_calls=counts['photo']+counts['distillation']+12)
    expected['shadow_rays']=(expected['photo_shader_calls']*137+expected['gate_shader_calls']*137+expected['eval_shader_calls']*220)*1536
    require(all(r['completed'][k]==v for k,v in expected.items()),'Trajectory/snapshot-derived work mismatch')
    return states,gradients


def summarize_errors(errors):
    eligible=errors['orientation_eligible']
    out={k+'_median':float(np.median(v)) for k,v in errors.items() if v.dtype.kind!='b'}
    out.update(Q_pass=int((errors['Q_world_relative']<=.05).sum()),Q_mean=float(errors['Q_world_relative'].mean()),
        orientation_eligible=int(eligible.sum()),orientation_pass=int(((errors['orientation_degrees']<=5)&eligible).sum()),
        predicted_width_gap_lt_0p1=int(errors['predicted_width_gap_lt_0p1'].sum()))
    return out


def snapshots(native,r,data,acq,fit):
    rows=[];points=[];detail={};labels=data['legacy'][0];views=acq['observation_views'];ids=acq['observation_points']
    initial=None
    for row in r['snapshots']:
        z=numeric(native/row['artifact']);material=row['material'];truth=study.previous.truth_parameters(labels,material=='stronger')[:,0]
        if row['stage']=='initial':
            if initial is None:initial=z
            else:require(np.array_equal(initial['native_components'],z['native_components']) and np.array_equal(initial['Q'],z['Q']),'Initial native prediction parity differs')
        require(np.array_equal(z['true_parameters'],truth) and np.array_equal(z['true_Q'],world_Q(truth,labels['T'],labels['B'])) and
            np.array_equal(z['teacher_Q'],data['teachers'][material]['Q']) and np.array_equal(z['fitting_mask'],fit) and
            np.array_equal(z['observation_views'],views) and np.array_equal(z['observation_points'],ids),'Snapshot identities differ')
        require(z['native_components'].shape==(8,220,2,3,3) and z['mx'].shape==z['my'].shape==(84,1),'Snapshot shapes differ')
        mx=z['mx'][:,0];my=z['my'][:,0];axis=z['axis'];other=z['other']
        rebuilt=mx[:,None,None]**2*axis[:,:,None]*axis[:,None,:]+my[:,None,None]**2*other[:,:,None]*other[:,None,:]
        require(np.allclose(z['Q'],rebuilt,atol=2e-7,rtol=2e-6) and (mx>=.04).all() and (mx<=1).all() and
            (my>=.04).all() and (my<=1).all() and np.allclose(np.linalg.norm(axis,axis=1),1,atol=2e-6,rtol=0),'Invalid material/frame capture')
        errors=physical(z['Q'].astype(float),mx.astype(float),my.astype(float),axis,other,truth,labels['T'],labels['B'])
        target=study.previous.captured_target(data['legacy'][5][512],study.FIELDS.index(material),views,ids)
        require(np.array_equal(target,z['target_RGB']) and np.array_equal(deterministic_codes(target),z['target_codes']),'Target mismatch')
        draws=z['native_components'][:,:,:,0].astype(float);mean=draws.mean(0)
        require(np.array_equal(deterministic_codes(mean),z['native_mean_codes']) and
            np.allclose(z['native_components'][:,:,:,0],z['native_components'][:,:,:,1]+z['native_components'][:,:,:,2],atol=3e-8,rtol=2e-6),'RGB/component replay mismatch')
        metrics={name:radiometry(draws,target,z['independent_fine'],z['independent_coarse'],mask) for name,mask in [('fit',fit),('held',~fit)]}
        require(int((errors['Q_world_relative']<=.05).sum())==row['Q_pass'] and np.isclose(np.median(errors['Q_world_relative']),row['Q_median'],rtol=1e-12) and
            np.isclose(metrics['held']['native_mean_relative_L2'],row['held_RGB_relative_L2'],rtol=1e-12),'Saved summary replay mismatch')
        name=material+'-'+row['arm']+'-'+row['stage'];detail[name]=dict(errors=errors,arrays=z,radiometry=metrics)
        result=dict(material=material,arm=row['arm'],stage=row['stage'],**summarize_errors(errors))
        for subset,v in metrics.items():result.update({subset+'_'+k:value for k,value in v.items()})
        result['teacher_relative_Q_median']=float(np.median(np.linalg.norm(z['Q']-z['teacher_Q'],axis=(1,2))/np.linalg.norm(z['teacher_Q'],axis=(1,2))))
        rows.append(result)
        for i,pixel in enumerate(acq['pixel_ids']):points.append(dict(material=material,arm=row['arm'],stage=row['stage'],point=i,pixel=int(pixel),**{k:float(v[i]) if v.dtype.kind!='b' else bool(v[i]) for k,v in errors.items()}))
    teachers={}
    for material in study.FIELDS:
        p=data['teachers'][material]['parameters'];t=study.previous.truth_parameters(labels,material=='stronger')[:,0]
        a=np.cos(p[:,:1])*labels['T']+np.sin(p[:,:1])*labels['B'];b=-np.sin(p[:,:1])*labels['T']+np.cos(p[:,:1])*labels['B']
        mx=np.exp(p[:,1]+p[:,2]/2);my=np.exp(p[:,1]-p[:,2]/2)
        errors=physical(data['teachers'][material]['Q'],mx,my,a,b,t,labels['T'],labels['B'])
        teachers[material]=summarize_errors(errors);detail[material+'-teacher']=dict(errors=errors)
        for i,pixel in enumerate(acq['pixel_ids']):points.append(dict(material=material,arm='teacher',stage='fitting_selected',point=i,pixel=int(pixel),
            **{k:float(v[i]) if v.dtype.kind!='b' else bool(v[i]) for k,v in errors.items()}))
    return rows,points,detail,teachers


def rank_correlation(a,b):
    def rank(x):
        values,ids,counts=np.unique(x,return_inverse=True,return_counts=True)
        return (np.cumsum(counts)-counts+(counts-1)/2.)[ids]
    x=rank(a);y=rank(b)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None


def csv_write(p,rows):
    with p.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures(out,rows,detail,native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Teacher','Initial','B distill200','A photo1000','B photo1000','A photo2000']
    stages=['teacher','A-initial','B-post_distillation','A-photo1000','B-photo1000','A-compute_endpoint']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,material in zip(axes,study.FIELDS):
        errors=[detail[material+'-'+s]['errors']['Q_world_relative'] for s in stages]
        counts=[name+'\n'+str(int((v<=.05).sum()))+'/84' for name,v in zip(labels,errors)]
        ax.boxplot(errors,labels=counts,showfliers=True);ax.set_yscale('log');ax.axhline(.05,color='black',ls='--',lw=1)
        ax.tick_params(axis='x',rotation=35);ax.set_title(material.capitalize())
    axes[0].set_ylabel('Relative world Q error (all 84 points)');fig.suptitle('Observation-guided initialization improves Q; photometric refinement erodes it')
    fig.tight_layout();fig.savefig(out/'physical_recovery.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,material in zip(axes,study.FIELDS):
        x=detail[material+'-A-photo1000']['errors']['Q_world_relative'];y=detail[material+'-B-photo1000']['errors']['Q_world_relative']
        ax.scatter(x,y,s=16,alpha=.7);ax.plot([.001,2],[.001,2],color='black',ls='--');ax.set_xscale('log');ax.set_yscale('log')
        ax.set_title(material.capitalize()+' (B better: '+str(int((y<x).sum()))+'/84)');ax.set_xlabel('A: photometric baseline');ax.set_ylabel('B: guided initialization')
    fig.suptitle('Matched 1,000 photometric updates: relative Q error');fig.tight_layout();fig.savefig(out/'paired_Q.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i,material in enumerate(study.FIELDS):
        for arm in ('A','B'):
            raw=json.loads((native/(material+'-'+arm+'-training.json')).read_text());v=np.array([x['loss'] for x in raw if x['phase']=='photo'])
            axes[i,0].plot(np.arange(24,len(v)),np.convolve(v,np.ones(25)/25,'valid'),label=arm)
        axes[i,0].set_yscale('log');axes[i,0].set_title(material.capitalize()+': fitting loss (25-step mean)');axes[i,0].legend()
        keys=['B-post_distillation','A-photo1000','B-photo1000','A-compute_endpoint'];x=np.arange(len(keys))
        for offset,(key,title) in enumerate([('native_mean_MSE','mean native MSE'),('native_empirical_variance','native draw variance')]):
            vals=[detail[material+'-'+s]['radiometry']['held'][key] for s in keys];axes[i,1].bar(x+offset*.35,vals,width=.35,label=title)
        axes[i,1].set_xticks(x+.175);axes[i,1].set_xticklabels(['B distill','A1000','B1000','A2000'],rotation=20)
        axes[i,1].set_yscale('log');axes[i,1].set_title('Withheld linear RGB: mean error and draw variance');axes[i,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'photometry_and_variance.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('batch','authorization','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();require(a.output.is_absolute() and a.output.resolve()==a.output and not a.output.exists(),'New explicit analysis directory required')
    f,r,data,acq,fit,replay=verify(a.batch,a.authorization);native=a.batch/'neural'
    states,gradients=weights_and_training(native,r);rows,points,detail,teachers=snapshots(native,r,data,acq,fit)
    a.output.mkdir();csv_write(a.output/'endpoints.csv',rows);csv_write(a.output/'point_errors.csv',points);csv_write(a.output/'gradients.csv',gradients)
    figures(a.output,rows,detail,native)
    comparisons={};confidence={}
    for material in study.FIELDS:
        A=detail[material+'-A-photo1000']['errors']['Q_world_relative'];B=detail[material+'-B-photo1000']['errors']['Q_world_relative']
        post=detail[material+'-B-post_distillation']['errors']['Q_world_relative'];extra=detail[material+'-A-compute_endpoint']['errors']['Q_world_relative']
        comparisons[material]=dict(B_lower_Q_than_A1000=int((B<A).sum()),B_lower_Q_than_A2000=int((B<extra).sum()),
            B_Q_worsened_after_distillation=int((B>post).sum()),initially_passing_lost=int(((post<=.05)&(B>.05)).sum()),
            newly_passing_after_photo=int(((post>.05)&(B<=.05)).sum()),matched_total_time=r['compute_comparisons'][material]['reached'])
        teacher_error=detail[material+'-teacher']['errors']['Q_world_relative'];c=data['confidence'][material]
        confidence[material]=dict(weighting_used=False,all_points=84,
            teacher_error_rank_vs_fitting_MSE=rank_correlation(teacher_error,c['fitting_MSE']),
            teacher_error_rank_vs_Q_disagreement=rank_correlation(teacher_error,c['relative_Q_disagreement']),
            teacher_error_rank_vs_minimum_normal_eigenvalue=rank_correlation(teacher_error,np.asarray(c['last_pretrial_normal_eigenvalues'])[:,0]),
            teacher_error_rank_vs_final_B_error=rank_correlation(teacher_error,B))
    result=dict(schema_version=1,status='verified_file_only',execution_decision='PROCEED',method_decision='FIX_AND_PROCEED_MIXED_RECOVERY',
        execution_commit=COMMIT,execution_report=record(native/'neural-recovery.json'),freeze=record(ROOT/'docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json'),
        authorization=record(a.authorization),batch=str(a.batch),analysis_directory=str(a.output),
        native_work=r['completed'],native_compatibility_replayed=replay,teacher_metrics=teachers,endpoints=rows,paired_comparisons=comparisons,
        observation_only_confidence_diagnostic=confidence,
        gradient_summaries=gradients,checkpoint_verification=states,compute_comparisons=r['compute_comparisons'],
        runtime=r['runtime'],elapsed_seconds=r['elapsed_seconds'],host_peak_rss_bytes=r['host_peak_rss_bytes'],
        cuda_peak_allocated_bytes=r['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=r['cuda_peak_reserved_bytes'],
        sources_verified=f['files'],input_artifacts_verified=data['records'],
        native_artifacts=[record(x) for x in sorted(a.batch.rglob('*')) if x.is_file()],
        analysis_code=record(Path(__file__).resolve()),new_scientific_queries=0,
        limitations=['One seed, sparse 84-point curved-surface field; no unseen positions or lighting.',
            'Cleanup verified by saved native report and exit0; Docker daemon not inspected by Codex.',
            'Final group gradients/moments/steps are independently checked; full Adam trajectory cannot be replayed without unsaved per-parameter gradients.',
            'Both baseline extensions hit the update cap before conservative time matching; no equal-total-compute claim.',
            'Teacher fitting ambiguity, finite quadrature, hard visibility and Monte Carlo variance remain; passing local gates does not establish global unbiasedness.',
            'Photometric refinement degrades distilled Q; endpoint correlations do not identify a unique causal mechanism.'])
    result['analysis_artifacts']=[record(x) for x in sorted(a.output.iterdir()) if x.is_file()]
    with (a.output/'analysis.json').open('x') as h:json.dump(result,h,indent=2,allow_nan=False);h.write('\n')
    print(json.dumps(dict(status=result['status'],teacher_metrics=teachers,comparisons=comparisons,endpoints=[{k:v for k,v in x.items() if k in
        ('material','arm','stage','Q_world_relative_median','Q_pass','orientation_pass','major_absolute_median','minor_absolute_median','held_native_mean_relative_L2','held_independent_relative_L2')} for x in rows]),indent=2))


if __name__=='__main__':main()
