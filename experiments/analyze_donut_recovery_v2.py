"""File-only donut D1/D2 verification, numerical summaries and paper figures.

No checkpoint decoding, renderer, native model, rays or optimizer calls.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BATCH=Path('/home/isavand/paper/evidence/donut-kurt-recovery-v2-4c47339a0d204bc0abe8480be2325b63')
REFERENCE=Path('/home/isavand/paper/evidence/donut-kurt-reference-v2-262b3cfcf73340a0867829c48c0198f7/reference')
FREEZE='d39ac455354bb444c61739a630d389fd59818788c2466046abb1e87359ba0f6a'
AUTH='7f53ad4ae75a8473ab2820ef76a17f52a68549d7fa71c9916c7b7b6e00811657'
GENERATION='2052306bdf87a0beaa0a3f9b0bddeeb12d787f1d65357d520b5ce2fe0a33ee64'
COMMIT='2b7caf883035b9107eb5815266fa55c7b8421482'
CAMERAS=[0,50,1,45,28,34]
STEPS=[0,2000,4000,6000,8000,10000]
COLOR=np.array([.10804546,1.,.03044616])
SCALAR_REPLAY=[]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def read(path):return json.loads(path.read_text())


def numeric(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    assert a and all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in a.values()),path
    return a


def equal(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=False)


def scalar32(value,saved,label,count):
    # CPU/GPU-process NumPy reductions may differ at float32 roundoff. This
    # guard applies only to report-scalar replay, never arrays or decisions.
    difference=abs(value-saved);ulp=float(abs(np.spacing(np.float32(saved))))
    # Standard gamma_n bound for nonnegative float32 reductions; factor four
    # covers the two reductions, square roots and ratio. No fitted tolerance.
    u=np.finfo(np.float32).eps/2;gamma=count*u/(1-count*u)
    bound=max((4*gamma+8*u)*max(abs(value),abs(saved)),1e-12)
    assert np.isfinite(value) and np.isfinite(saved) and difference<=bound,label
    if difference:SCALAR_REPLAY.append(dict(label=label,absolute_difference=difference,float32_ULPs=difference/ulp if ulp else 0,reduction_values=count,roundoff_envelope=bound))


def stats(a):
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))


def srgb(a):
    # Explicit display clipping only. All analysis uses unmodified linear arrays.
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,np.finfo(np.float32).eps)**(1/2.4)-.055)


def analyze(out):
    assert not out.exists() and all(p!=out and p not in out.parents for p in [BATCH,REFERENCE])
    freeze=read(BATCH/'freeze.json');auth=read(BATCH/'authorization.json')
    assert digest(BATCH/'freeze.json')==FREEZE and digest(BATCH/'authorization.json')==AUTH
    assert auth['approved'] and auth['commit']==COMMIT and auth['limits']==freeze['training_limits']
    assert auth['freeze_sha256']==FREEZE and auth['maximum_execution_launches']==1
    assert auth['host_output_directory']==str(BATCH) and auth['researcher_admitted_dataset']
    assert auth['selected_gpu']=='0' and auth['output_directories']==dict(D1='/outputs/D1',D2='/outputs/D2')
    assert auth['benchmark_generation_sha256']==GENERATION
    assert digest(Path(auth['admission']['path']))==auth['admission']['sha256']
    assert read(Path(auth['admission']['path']))['researcher_admitted_dataset']
    for name,sha in freeze['sources'].items():assert digest(ROOT/name)==sha,name
    for v in freeze['inputs'].values():assert digest(Path(v['path']))==v['sha256']
    assert digest(REFERENCE/'generation.json')==GENERATION
    generation=read(REFERENCE/'generation.json')
    assert generation['freeze_sha256']==freeze['reference_freeze_sha256']
    for name,v in generation['outputs'].items():assert digest(REFERENCE/name)==v['sha256']
    failed=read(ROOT/'docs/evidence/donut_training_preview_v2/failed-batch-identities.json')
    for name,v in failed['files'].items():assert digest(Path(failed['failed_batch'])/name)==v['sha256']
    truth={c:numeric(REFERENCE/f'view-{c}.npz') for c in CAMERAS}
    manifest={str(p):record(p) for p in [BATCH/'authorization.json',BATCH/'freeze.json',
        REFERENCE/'generation.json',Path(auth['admission']['path'])]}
    reports={};logs={};rows=[];gradient_rows=[];replayed=0
    for arm in ['D1','D2']:
        base=BATCH/arm;m=read(base/'recovery.json');reports[arm]=m
        assert m['status']=='complete' and m['scientific_work_complete'] and m['cleanup_verified']
        assert m['optimizer_returns']==10000 and m['commit']==COMMIT and m['freeze_sha256']==FREEZE
        assert m['benchmark_sha256']==GENERATION and m['runtime']['declared_image']==auth['image_id']
        assert m['policy']==freeze['policy'] and m['effective_shader']==freeze['shader']
        assert m['indirect_illumination']=='identically zero in D1/D2'
        assert m['elapsed_seconds']<freeze['training_limits']['inner_seconds']
        assert m['resources']['host_peak_RSS']<64*1024**3
        assert (BATCH/(arm+'-exit-code.txt')).read_text().strip()=='0'
        assert all(m['counters'][k]<=freeze['training_limits'][k] for k in m['counters'])
        assert [v['step'] for v in m['saved_endpoints']]==STEPS
        assert m['native_reference_gate']['passed']
        for name,v in m['outputs'].items():
            p=base/name
            assert p.resolve()==p and base in p.parents and p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],p
        for p in [base/'recovery.json',base/'training.jsonl',base/'initial.json',base/'native-reference-gate.npz',
                  BATCH/(arm+'-exit-code.txt'),BATCH/(arm+'-console.txt')]+sorted(base.glob('checkpoint-*.pth')):
            manifest[str(p)]=record(p)
        assert len(list(base.glob('checkpoint-*.pth')))==5
        log=[json.loads(s) for s in (base/'training.jsonl').read_text().splitlines()];logs[arm]=log
        assert [v['completed_updates'] for v in log]==list(range(1,10001))
        assert log[0]['lr']==0 and log[1]['lr']==5e-7
        assert log[0]['parameter_displacement_L1']==0 and log[1]['parameter_displacement_L1']>0
        groups=['feats_network','mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor']
        for v in log:
            assert all(np.isfinite(v[k]) for k in ['lr','RGB_loss','material_prior','diffuse_white_prior','preclip_gradient_l2'])
            assert all(np.isfinite(v['gradient_groups'][g]) for g in groups+['outer_light','inner_light'])
            assert v['gradient_groups']['inner_light']==0
            if arm=='D1':assert v['gradient_groups']['outer_light']==0
            if 'gradient_interactions' in v:
                gradient_rows.append(dict(arm=arm,completed_updates=v['completed_updates'],**v['gradient_interactions']))
        assert sum('gradient_interactions' in v for v in log)==6
        for endpoint in m['saved_endpoints']:
            step=endpoint['step'];destination=base/f'step-{step:05d}'
            assert read(destination/'metrics.json')==endpoint['metrics']
            pooled=[]
            for c in CAMERAS:
                d=truth[c];ix=np.flatnonzero(d['shader_valid']);pixels=d['pixel_indices'][ix]
                a=numeric(destination/f'camera-{c}/materials.npz');e=numeric(destination/f'camera-{c}-errors.npz')
                mask=np.zeros(128**2,bool);mask[pixels]=True
                assert np.array_equal(a['hit_mask'].reshape(-1),mask)
                assert np.array_equal(e['geometry_hit_mask'],d['hit_mask'])
                def flat(k):return a[k].reshape(128**2,*a[k].shape[2:])[pixels]
                equal(flat('position'),d['position'][ix].astype(np.float32))
                equal(flat('normal'),d['normals'][ix].astype(np.float32))
                equal(flat('linear_gt'),d['linear_components'][ix,2].astype(np.float32))
                mx,my=flat('mx'),flat('my');t,b=2*flat('tangents')-1,2*flat('bitangents')-1
                Q=mx[:,:,None]**2*t[:,:,None]*t[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]
                equal(Q,flat('Q_world'))
                widths=np.sort(np.concatenate([mx,my],-1),axis=-1)[:,::-1]
                axis=np.where(mx>=my,t,b)
                error=dict(Q_relative=np.linalg.norm(Q-d['Q_world'][ix],axis=(-1,-2))/np.linalg.norm(d['Q_world'][ix],axis=(-1,-2)),
                    color_max=abs(flat('kd')-COLOR).max(-1),metallic=abs(flat('metallic')[:,0]-.8),
                    major_width=abs(widths[:,0]-.35),minor_width=abs(widths[:,1]-.08),
                    orientation_degrees=np.degrees(np.arccos(np.clip(abs((axis*d['axis'][ix]).sum(-1)),0,1))),
                    orientation_eligible=(widths[:,0]**2-widths[:,1]**2)/(widths**2).sum(-1)>=.1,
                    tangent_error=abs(np.einsum('nij,nj->ni',Q,d['normals'][ix])).max(-1))
                for k,v in error.items():equal(v,e[k])
                physical=endpoint['metrics'][str(c)]['physical']
                for k,v in error.items():
                    if v.dtype!=bool:
                        for key,value in stats(v).items():
                            if v.dtype==np.float32:scalar32(value,physical[k][key],f'{arm}/{step}/{c}/{k}/{key}',len(v))
                            else:equal(value,physical[k][key])
                assert physical['Q_at_most_005']==int((error['Q_relative']<=.05).sum())
                assert physical['orientation_ineligible']==int((~error['orientation_eligible']).sum())
                delta=a['linear_rgb'][a['hit_mask']]-a['linear_gt'][a['hit_mask']]
                rgbtruth=a['linear_gt'][a['hit_mask']]
                scalar32(float(np.sqrt(np.mean(delta**2))),endpoint['metrics'][str(c)]['RGB_linear_RMSE'],f'{arm}/{step}/{c}/RGB_RMSE',delta.size)
                scalar32(float(np.linalg.norm(delta)/np.linalg.norm(rgbtruth)),endpoint['metrics'][str(c)]['RGB_relative_L2'],f'{arm}/{step}/{c}/RGB_relative',delta.size)
                scalar32(float(np.sqrt(np.mean(e['native_component_SE'].reshape(128,128,3,3)[a['hit_mask']]**2))),
                    endpoint['metrics'][str(c)]['native_standard_error_RMS'],f'{arm}/{step}/{c}/native_SE',len(pixels)*9)
                error.update(rgb_delta=delta.astype(float),rgb_truth=rgbtruth.astype(float),kd=flat('kd').astype(float),
                    metallic_value=flat('metallic')[:,0].astype(float),widths=widths.astype(float),
                    native_total_RGB_SE=e['native_component_SE'][pixels,0].astype(float),
                    log_mean_error=abs(np.log(np.sqrt(widths[:,0]*widths[:,1])/np.sqrt(.35*.08))),
                    log_ratio_error=abs(np.log((widths[:,0]/widths[:,1])/(.35/.08))))
                pooled.append((c,error));replayed+=1
                if step==0 and arm=='D2':
                    old=numeric(BATCH/'D1'/f'step-{step:05d}/camera-{c}/materials.npz')
                    for k in ['kd','mx','my','rotation','rotation_raw','Q_world']:
                        assert np.array_equal(a[k],old[k]),'Initial material maps differ'
            for split,cameras in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                values={k:np.concatenate([v[k] for c,v in pooled if c in cameras]) for k in pooled[0][1]}
                row=dict(arm=arm,completed_updates=step,split=split,observations=len(values['Q_relative']))
                for k in ['Q_relative','orientation_degrees','major_width','minor_width','color_max','metallic','log_mean_error','log_ratio_error']:
                    row.update({k+'_'+stat:value for stat,value in stats(values[k]).items()})
                eligible=values['orientation_eligible']
                row.update(Q_at_most_005=int((values['Q_relative']<=.05).sum()),orientation_eligible=int(eligible.sum()),
                    RGB_linear_RMSE=float(np.sqrt(np.mean(values['rgb_delta']**2))),
                    RGB_relative_L2=float(np.linalg.norm(values['rgb_delta'])/np.linalg.norm(values['rgb_truth'])),
                    native_total_RGB_SE_RMS=float(np.sqrt(np.mean(values['native_total_RGB_SE']**2))),
                    predicted_metallic_mean=float(values['metallic_value'].mean()),predicted_major_width_mean=float(values['widths'][:,0].mean()),
                    predicted_minor_width_mean=float(values['widths'][:,1].mean()),width_at_floor=int((values['widths'][:,1]<=.0401).sum()),
                    green_albedo_below_half_truth=int((values['kd'][:,1]<.5).sum()),
                    orientation_eligible_median=float(np.median(values['orientation_degrees'][eligible])) if eligible.any() else None)
                row.update({'kd_'+name+'_mean':float(values['kd'][:,i].mean()) for i,name in enumerate('RGB')})
                rows.append(row)
            light=numeric(destination/'illumination.npz');v=light['linear_RGB'];known=light['known_linear_RGB']
            theta=(np.arange(128)+.5)/128*np.pi;w=np.sin(theta)[:,None,None]
            illum=endpoint['metrics']['illumination']
            equal(np.sqrt(((v-known)**2*w).sum()/((known**2*w).sum())),illum['relative_L2'])
            assert int((v>1).sum())==illum['preview']['channels_above_one']
            if arm=='D1':assert np.array_equal(v,known)
        assert m['counters']['component_reverse_sweeps']==24
        assert m['counters']['shader_calls']==10000+32+6*sum(int(np.ceil(v['shader_valid'].sum()/512)) for v in truth.values())*8
        assert m['counters']['shadow_rays']==(10000*512+32*24+6*22630*8)*1536
    assert reports['D1']['initial']==reports['D2']['initial']
    out.mkdir(parents=True)
    with (out/'endpoint-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result=dict(schema_version=1,status='verified_complete',execution_commit=COMMIT,freeze_sha256=FREEZE,
        authorization_sha256=AUTH,reference_generation_sha256=GENERATION,
        checks=dict(frozen_source_records=len(freeze['sources']),verified_arm_output_records=sum(len(m['outputs']) for m in reports.values()),
            independently_replayed_camera_endpoints=replayed,matched_initial_state=True,matched_initial_material_maps=True,
            preserved_failed_batch_files=len(failed['files']),reference_artifacts=len(generation['outputs'])),
        arms={arm:dict(optimizer_returns=m['optimizer_returns'],elapsed_seconds=m['elapsed_seconds'],resources=m['resources'],
            counters=m['counters'],cleanup_verified=m['cleanup_verified'],cleanup_scope=m['cleanup_scope'],
            initial=m['initial'],native_reference_gate=m['native_reference_gate'],
            final_illumination=m['saved_endpoints'][-1]['metrics']['illumination'],
            final_surface_smoothness=m['saved_endpoints'][-1]['metrics']['surface_Q_smoothness_diagnostic'],
            gradient_group_zero_rows={g:sum(v['gradient_groups'][g]==0 for v in logs[arm]) for g in logs[arm][0]['gradient_groups']},
            warmup_displacement=[logs[arm][i]['parameter_displacement_L1'] for i in [0,1]]) for arm,m in reports.items()},
        metrics=rows,component_gradient_probes=gradient_rows,
        report_scalar_replay=dict(float32_reduction_guard='(4*gamma_n+8*u)*scale, gamma_n=n*u/(1-n*u), u=2^-24; nonnegative scalar reductions only',arrays_absolute_guard=1e-12,
            scientific_thresholds_or_decisions_changed=False,differences=SCALAR_REPLAY),
        analysis_execution=dict(numpy=np.__version__,new_model_calls=0,new_renderer_calls=0,new_optimizer_returns=0,checkpoint_deserializations=0),
        limitations=['One seed, one constant-parameter material with spatially varying director, six128x128 cameras.',
            'Scores count repeated visible observations, not area-uniform unique surface points.',
            'Held-out cameras have the same light; unseen relighting was not acquired or executed.',
            'Eight-draw native RGB averages retain Monte Carlo uncertainty; no additional queries performed.',
            'Cleanup is attested by native reports and zero process exits; Docker state was not independently queried.',
            'Saved checkpoints are hash-verified only; strict roundtrip occurred in the frozen native execution, not this analysis.',
            'The legacy energy proxy is not a proof of conservation; prior causality is not isolated here.'])
    figures(out,rows,reports,logs,truth)
    for name,value in [('analysis.json',result),('artifact-manifest.json',dict(schema_version=1,batch=str(BATCH),
        batch_bytes=sum(p.stat().st_size for p in BATCH.rglob('*') if p.is_file()),records=manifest,
        full_output_hashes='Each immutable recovery.json binds890 independently verified output identities; checkpoints are never deserialized here.'))]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    assert not any(n.split('.')[0] in {'torch','network','raytracing','bpy'} for n in sys.modules)
    return result


def figures(out,rows,reports,logs,truth):
    if 'MPLCONFIGDIR' not in os.environ:
        os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='donut-offline-mpl-')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['image.interpolation']='nearest'
    import matplotlib.pyplot as plt
    from PIL import Image
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    def view(arm,step,c):return numeric(BATCH/arm/f'step-{step:05d}/camera-{c}/materials.npz')
    def gt(c):
        d=truth[c];a=np.zeros((128*128,3));a[d['pixel_indices']]=d['linear_components'][:,2];return a.reshape(128,128,3)
    fig,axes=plt.subplots(6,5,figsize=(13,14))
    for i,c in enumerate(CAMERAS):
        a,b=view('D1',10000,c),view('D2',10000,c)
        for j,img in enumerate([srgb(gt(c)),a['rgb_pr'],b['rgb_pr']]):axes[i,j].imshow(np.clip(img,0,1))
        for j,v in enumerate([a,b],3):im=axes[i,j].imshow(np.sqrt(np.mean((v['linear_rgb']-v['linear_gt'])**2,-1)),vmin=0,vmax=.02,cmap='magma')
        axes[i,0].set_ylabel(f'Camera {c}\n'+('fit' if i<4 else 'held out'))
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Reference','D1 known light','D2 learned light','D1 linear RGB error','D2 linear RGB error']):ax.set_title(title)
    fig.subplots_adjust(left=.06,right=.88,bottom=.02,top=.96,hspace=.1,wspace=.05)
    colorbar_axis=fig.add_axes([.91,.20,.014,.60])
    fig.colorbar(im,cax=colorbar_axis,label='Per-pixel linear RGB RMSE (display cap .02)')
    fig.savefig(out/'RGB-comparison.png',dpi=150);plt.close(fig)
    c=0;d=truth[c];mask=np.zeros(128**2,bool);mask[d['pixel_indices'][d['shader_valid']]]=True;mask=mask.reshape(128,128)
    def paint(values,width):
        x=np.zeros((128**2,width));x[d['pixel_indices']]=values;return x.reshape(128,128,width)
    P=d['axis'][:,:,None]*d['axis'][:,None,:]
    oracle=[paint(np.tile(COLOR,(len(d['axis']),1)),3),np.full((128,128),.8),np.full((128,128),.35),np.full((128,128),.08),
        paint(np.stack([P[:,0,1],P[:,0,2],P[:,1,2]],-1)+.5,3),np.zeros((128,128))]
    fig,axes=plt.subplots(3,6,figsize=(15,7))
    for i,arm in enumerate(['Oracle','D1','D2']):
        if i:
            a=view(arm,10000,c);p=a['major_axis_projector_world'];e=numeric(BATCH/arm/'step-10000/camera-0-errors.npz')
            q=np.zeros(128**2);q[d['pixel_indices'][d['shader_valid']]]=e['Q_relative']
            images=[a['kd'],a['metallic'][...,0],a['major_width'][...,0],a['minor_width'][...,0],np.stack([p[...,0,1],p[...,0,2],p[...,1,2]],-1)+.5,q.reshape(128,128)]
        else:images=oracle
        for j,img in enumerate(images):
            img=np.where(mask[...,None],img,0) if img.ndim==3 else np.where(mask,img,np.nan)
            axes[i,j].imshow(np.clip(img,0,1) if img.ndim==3 else img,vmin=0,vmax=[1,1,1,.2,1,.5][j],cmap='magma' if j==5 else 'viridis')
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(arm)
    for ax,title in zip(axes[0],['Base color (linear)','Metallic [0,1]','Major width [0,1]','Minor width [0,.2]','World director projector','Relative Q error [0,.5]']):ax.set_title(title,fontsize=9)
    save(fig,'material-comparison-camera0')
    fig,axes=plt.subplots(6,6,figsize=(13,12))
    for i,step in enumerate(STEPS):
        for j,arm in enumerate(['D1','D2']):
            a=view(arm,step,0)
            for k,key in enumerate(['reconstructed','albedo','orientation_world_projector']):
                axes[i,j*3+k].imshow(Image.open(BATCH/arm/f'step-{step:05d}/camera-0'/f'{key}.png'));axes[i,j*3+k].axis('off')
        axes[i,0].text(-.08,.5,str(step),transform=axes[i,0].transAxes,rotation=90,va='center',ha='right')
    for ax,title in zip(axes[0],['D1 RGB','D1 albedo','D1 world director','D2 RGB','D2 albedo','D2 world director']):ax.set_title(title,fontsize=9)
    save(fig,'progression-camera0')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for arm,color in [('D1','tab:blue'),('D2','tab:orange')]:
        allrows=[r for r in rows if r['arm']==arm and r['split']=='all'];x=[r['completed_updates'] for r in allrows]
        axes[0,0].plot(x,[r['Q_relative_median'] for r in allrows],'-o',color=color,label=arm)
        axes[0,1].plot(x,[r['Q_at_most_005']/r['observations'] for r in allrows],'-o',color=color,label=arm)
        for split,style in [('fitting','-'),('held_out','--')]:
            rr=[r for r in rows if r['arm']==arm and r['split']==split]
            axes[0,2].plot(x,[r['RGB_linear_RMSE'] for r in rr],style+'o',color=color,label=arm+' '+split)
        axes[1,0].plot(x,[r['predicted_major_width_mean'] for r in allrows],'-o',color=color,label=arm+' major')
        axes[1,0].plot(x,[r['predicted_minor_width_mean'] for r in allrows],'--o',color=color,label=arm+' minor')
        axes[1,1].plot(x,[r['orientation_eligible_median'] for r in allrows],'-o',color=color,label=arm)
        axes[1,2].plot(x,[e['metrics']['illumination']['relative_L2'] for e in reports[arm]['saved_endpoints']],'-o',color=color,label=arm)
    axes[0,0].set_yscale('log');axes[0,2].set_yscale('log')
    for y in [.35,.08]:axes[1,0].axhline(y,color='grey',lw=.8)
    for ax,title in zip(axes.ravel(),['Median relative Q error','Fraction Q error <= .05','Linear RGB RMSE','Mean sorted widths (oracle lines)','Eligible median orientation error (degrees)','Environment relative L2 error']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Completed Adam updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'recovery-curves')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for arm,color in [('D1','tab:blue'),('D2','tab:orange')]:
        log=logs[arm];x=np.array([v['completed_updates'] for v in log])
        def smooth(v):return np.convolve(v,np.ones(200)/200,mode='valid')
        for k,style in [('RGB_loss','-'),('material_prior','--'),('diffuse_white_prior',':')]:axes[0,0].plot(x[199:],smooth([v[k] for v in log]),style,color=color,label=arm+' '+k)
        axes[0,1].plot(x[199:],smooth([v['kd_mean'][1] for v in log]),color=color,label=arm+' green kd')
        axes[0,2].plot(x[199:],smooth([v['metallic_mean'] for v in log]),color=color,label=arm)
        axes[1,0].plot(x[199:],smooth([v['diffuse_allocation'][1] for v in log]),color=color,label=arm)
        axes[1,1].plot(x[199:],smooth([v['preclip_gradient_l2'] for v in log]),color=color,label=arm)
        for g,style in [('rotation_predictor','-'),('outer_light','--')]:axes[1,2].plot(x[199:],smooth([v['gradient_groups'][g] for v in log]),style,color=color,label=arm+' '+g)
    axes[0,0].set_yscale('log');axes[0,1].axhline(1,color='grey',lw=.8);axes[0,2].axhline(.8,color='grey',lw=.8);axes[1,0].axhline(.2,color='grey',lw=.8)
    for ax,title in zip(axes.ravel(),['Weighted losses (200-update moving mean)','Green base color (truth=1)','Metallic (truth=.8)','Green diffuse allocation (truth=.2)','Pre-clipping total gradient norm','Rotation and direct-light gradient norms']):
        ax.set_title(title,fontsize=9);ax.set_xlabel('Completed Adam updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'losses-and-gradients')
    fig,axes=plt.subplots(3,4,figsize=(12,7))
    for i,arm in enumerate(['Oracle','D1','D2']):
        if i:
            a=view(arm,10000,0);images=[srgb(a[k]) for k in ['linear_diffuse','linear_specular','linear_rgb']]
            light=numeric(BATCH/arm/'step-10000/illumination.npz')['linear_RGB']
        else:
            images=[]
            for k in [0,1,2]:
                a=np.zeros((128**2,3));a[d['pixel_indices']]=d['linear_components'][:,k];images.append(srgb(a.reshape(128,128,3)))
            light=numeric(BATCH/'D1/step-10000/illumination.npz')['known_linear_RGB']
        for j,img in enumerate(images+[srgb(light)]):axes[i,j].imshow(img);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(arm)
    for ax,title in zip(axes[0],['Diffuse RGB','Specular RGB','Total RGB','Environment (display clips above1)']):ax.set_title(title,fontsize=9)
    save(fig,'components-and-illumination')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.output.resolve())
    print(json.dumps(dict(status=result['status'],checks=result['checks'],output=str(args.output.resolve()))))
