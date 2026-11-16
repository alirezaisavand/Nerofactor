"""File-only E.13 v2 verification and paper figures. No checkpoint deserialization.

Only saved JSON, numeric NPZ (allow_pickle=False), and PNG data are consumed.
No scientific model/shader imports or renderer/optimizer queries are permitted.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS,digest
from experiments.verify_bell_material_smoke_e12b import assess_updates

ARMS=('A_original','A_estimator_v1','A_corrected_core_v1')
COMMIT='c372859dfa38789528517859a49e7f0f132cbdab'
FREEZE=ROOT/'docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V2.freeze.json'
FREEZE_SHA='4fd53bd58ed55d2900b4e959b1bb8c48ca86b7f26db90aff87aaec87596cf7cb'
REPORT_SHAS=('140dc6e05fa8ff388d543bae8328757f5c7dde8f796c09470bca2b364f66b819',
             '9a1ba36ba74370200823fbc48ce05f5b1314902c7cdc5f3a7e1ab8310f9b5d9c',
             '1d1acf0da1c3603e3da31dc5827a2cda3b15ccc6bdc2af877eae6101ec1e24d0')
ERRORS=('Q_world_absolute','Q_world_relative','Q_local_absolute','Q_local_relative',
        'major_width_absolute','minor_width_absolute','orientation_degrees','albedo_max_absolute','metallic_absolute')
CHECK32=32*np.finfo(np.float32).eps  # Verification roundoff only; NOT a scientific acceptance threshold.
CHECK64=64*np.finfo(np.float64).eps


def record(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def load_json(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    require(a and all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in a.values()),'Invalid numeric archive: '+str(path))
    return a


def png(path):
    from PIL import Image
    with Image.open(path) as image:a=np.asarray(image).copy()
    require(a.dtype==np.uint8 and a.shape==(128,128,3),'Expected unchanged128x128 RGB8 PNG')
    return a


def compare(a,b,where='metadata'):
    if isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),'Key mismatch: '+where)
        for k in a:compare(a[k],b[k],where+'.'+k)
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'List mismatch: '+where)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,where+'.'+str(i))
    elif type(a) is float:
        require(type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and
                math.isclose(a,b,rel_tol=CHECK64,abs_tol=CHECK64),'Float mismatch: '+where)
    else:require(type(a) is type(b) and a==b,'Exact mismatch: '+where)


def rgb_weight(step,cfg):
    w0,peak=cfg['render_loss_weight_begin'],cfg['render_loss_weight_end']
    start,mid,end=(cfg[k] for k in ('render_weight_decay_begin','render_weight_decay_mid','render_weight_decay_end'))
    require(start<mid<end,'Invalid saved RGB-loss schedule')
    if step<=start or step>end:return float(w0)
    if step<=mid:return float(w0+.5*(1-np.cos(np.pi*(step-start)/(mid-start)))*(peak-w0))
    return float(peak-.5*(1-np.cos(np.pi*(step-mid)/(end-mid)))*(peak-w0))


def physical_reference(p,g):
    """Independent float64 tensor/axis reference, never used for optimization."""
    mx,my=p['mx'].astype(float).ravel(),p['my'].astype(float).ravel()
    angle=np.arctan2(p['rotation'][:,1].astype(float),p['rotation'][:,0].astype(float))/2
    t,b=p['T'].astype(float),p['B'].astype(float)
    u=t*np.cos(angle)[:,None]+b*np.sin(angle)[:,None]
    v=b*np.cos(angle)[:,None]-t*np.sin(angle)[:,None]
    Q=np.array([x*x*np.outer(a,a)+y*y*np.outer(c,c) for x,y,a,c in zip(mx,my,u,v)])
    basis=np.stack((g['T'],g['B']),axis=-1)
    local=np.matmul(np.swapaxes(basis,1,2),np.matmul(Q,basis))
    actual=np.stack((p['actual_T'],p['actual_B'],g['normals']),axis=-1).astype(float)
    gram=np.matmul(np.swapaxes(actual,1,2),actual)
    invalid=(np.max(np.abs(gram-np.eye(3)),axis=(1,2))>1e-5)|(np.abs(np.linalg.det(actual)-1)>1e-5)
    major=np.where((mx>=my)[:,None],u,v)
    truth_major=g['T']*np.cos(g['theta_major'])[:,None]+g['B']*np.sin(g['theta_major'])[:,None]
    cosine=np.clip(np.abs(np.sum(major*truth_major,axis=1)),0,1)
    result=dict(Q_world=Q,Q_local=local,
        Q_world_absolute=np.linalg.norm(Q-g['Q_world'],axis=(1,2)),
        Q_world_relative=np.linalg.norm(Q-g['Q_world'],axis=(1,2))/np.maximum(np.linalg.norm(g['Q_world'],axis=(1,2)),1e-15),
        Q_local_absolute=np.linalg.norm(local-g['Q_local'],axis=(1,2)),
        Q_local_relative=np.linalg.norm(local-g['Q_local'],axis=(1,2))/np.maximum(np.linalg.norm(g['Q_local'],axis=(1,2)),1e-15),
        major_width_absolute=np.abs(np.maximum(mx,my)-g['widths_major_minor'][:,0]),
        minor_width_absolute=np.abs(np.minimum(mx,my)-g['widths_major_minor'][:,1]),
        albedo_max_absolute=np.max(np.abs(p['color']-g['color']),axis=1),
        metallic_absolute=np.abs(p['metallic']-g['metallic']).ravel())
    require(np.array_equal(p['pixel_indices'],g['pixel_indices']),'Export/label pixel identities differ')
    require(np.array_equal(p['invalid_frame'],invalid),'Actual frame decision differs')
    # All material domains checked independently, with the original threshold.
    bad=(mx<=0)|(my<=0)|(mx>1)|(my>1)|(np.abs(np.linalg.norm(p['rotation'],axis=1)-1)>1e-5)
    bad|=((p['color']<0)|(p['color']>1)).any(1)|((p['metallic']<0)|(p['metallic']>1)).any(1)
    require(np.array_equal(p['invalid_material'],bad),'Material domain decision differs')
    require(np.array_equal(p['orientation_eligible'],g['orientation_defined']&g['orientation_eligible']) and
            np.array_equal(p['orientation_prediction_defined'],np.abs(mx-my)>1e-6),'Orientation eligibility differs')
    max_error={}
    for k,x in result.items():
        delta=float(np.max(np.abs(p[k]-x)));max_error[k]=delta
        require(np.all(np.abs(p[k]-x)<=CHECK32*np.maximum(1,np.abs(x))),'Independent material calculation differs: '+k)
    c=np.cos(np.deg2rad(p['orientation_degrees']))
    max_error['orientation_cosine']=float(np.max(np.abs(c-cosine)))
    require(max_error['orientation_cosine']<=CHECK32,'Independent orientation cosine differs')
    # Canonical acceptance remains based on the frozen saved errors/thresholds;
    # float64 reference comparisons above only verify arithmetic consistency.
    return max_error


def physical_summary(rows):
    a={k:np.concatenate([r[k] for r in rows]) for k in ERRORS+('invalid_frame','invalid_material','orientation_eligible','orientation_prediction_defined')}
    valid=~(a['invalid_frame']|a['invalid_material']);n=len(valid)
    out=dict(observations=n,invalid_frames=int(a['invalid_frame'].sum()),invalid_materials=int(a['invalid_material'].sum()),valid_observations=int(valid.sum()))
    for k in ERRORS:
        x=a[k][a['orientation_eligible']&a['orientation_prediction_defined']] if k=='orientation_degrees' else a[k]
        out[k]=dict(count=len(x),mean=float(np.mean(x)) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None,maximum=float(np.max(x)) if len(x) else None)
    out['descriptive_pass_fractions']={}
    for k,limit in [('Q_world_relative',.05),('Q_local_relative',.05),('orientation_degrees',5.),('albedo_max_absolute',.02),('metallic_absolute',.03)]:
        eligible=a['orientation_eligible'] if k=='orientation_degrees' else np.ones(n,bool)
        defined=a['orientation_prediction_defined'] if k=='orientation_degrees' else np.ones(n,bool)
        numerator=int(((a[k]<=limit)&valid&eligible&defined).sum());denom=int(eligible.sum())
        out['descriptive_pass_fractions'][k]=dict(threshold=limit,numerator=numerator,denominator=denom,fraction=numerator/denom if denom else None)
    out['gt_orientation_eligible']=int(a['orientation_eligible'].sum())
    out['predicted_orientation_undefined_on_gt_eligible']=int((a['orientation_eligible']&~a['orientation_prediction_defined']).sum())
    return out


def image_reference(a,ref):
    from skimage.metrics import structural_similarity
    p,g,hit=a['native_srgb'],a['ground_truth_srgb'],a['hit'];linear=a['linear_components']
    require(p.shape==g.shape==(128,128,3) and hit.shape==(128,128) and hit.any(),'Image evidence shapes differ')
    q=np.clip(p*255,0,255).astype(np.uint8);t=np.clip(g*255,0,255).astype(np.uint8)
    mse=float(np.mean((t.astype(np.float32).reshape(-1,3)-q.astype(np.float32).reshape(-1,3))**2,axis=0).mean())
    out=dict(native_psnr=10*float(np.log10(255**2/mse)) if mse else None,native_psnr_infinite=(mse==0),
             native_ssim=float(structural_similarity(t,q,win_size=11,channel_axis=2,data_range=255)),foreground_pixels=int(hit.sum()),clipped_prediction_channels=int((p>1).sum()))
    for pi,gi,k in [(0,2,'total'),(1,0,'diffuse'),(2,1,'specular')]:
        x=linear[:,:,pi][hit].astype(float);truth=ref[:,:,gi][hit];d=x-truth
        out[k]=dict(rmse=float(np.sqrt(np.mean(d*d))),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(truth),1e-15)),squared_error=float((d*d).sum()),reference_squared=float((truth*truth).sum()),channels=int(truth.size))
    def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,np.finfo(np.float32).eps)**(1/2.4)-.055)
    out['foreground_srgb_rmse']=float(np.sqrt(np.mean((p[hit]-srgb(ref[:,:,2])[hit])**2)))
    require(np.allclose(linear[:,:,0],linear[:,:,1]+linear[:,:,2],rtol=CHECK32,atol=CHECK32),'Captured diffuse/specular sum differs')
    require(np.max(np.abs(p-srgb(linear[:,:,0])))<=2e-6,'Frozen RGB transfer bound failed')
    return out


def verify_arm(r,seal,arm,freeze):
    require(r['status']=='complete' and r['phase']=='complete' and r['scientific_work_complete'] is True and
            r['cleanup_verified'] is True and r['cleanup_errors']==[],'Native arm incomplete/cleanup failed')
    require(r['variant']==seal['variant']==arm and r['run_id']==seal['run_id']==freeze['run_ids'][arm] and
            r['infrastructure_commit']==COMMIT and r['freeze_sha256']==FREEZE_SHA and r['source_files']==freeze['files'] and
            r['declared_image_id']==freeze['image_id'],'Native identity mismatch')
    require(r['optimizer_calls_completed']==seal['optimizer_returns']==2000 and
            len(r['trajectory'])==2000 and r['completed_training_work']==dict(original_forwards=2000,original_backwards=2000,original_adam_returns=2000,direction_slots=1572864000),'Native work mismatch')
    require(seal['status']=='complete' and seal['cleanup_verified'] is True and seal['scientific_work_complete'] is True,'Completion seal fields invalid')
    # Reuse accepted validation only for its two-record prefix; the full count
    # above remains2000. This does not alter or rewrite the saved native report.
    assess_updates(dict(r,optimizer_calls_completed=len(r['updates'])))
    for i,row in enumerate(r['trajectory']):
        require(row['update']==i+1 and all(math.isfinite(row[k]) for k in ('loss','gradient_l2','lr','elapsed_seconds')) and row['gradient_l2']>0,'Invalid trajectory row')
        expected=5e-4*(i/1000) if i<1000 else 5e-4*((np.cos(np.pi*((i-1000)/99000))+1)*.5*.8+.2)
        require(row['lr']==expected,'Learning-rate trajectory differs')
    initial=r['initial_optimizer']['groups'][0];final=r['final_optimizer']['groups'][0]
    require({k:v for k,v in initial.items() if k!='lr'}=={k:v for k,v in final.items() if k!='lr'},'Optimizer settings changed')
    require(len(r['final_optimizer']['state'])==84 and all(s['step']==2000 for s in r['final_optimizer']['state'].values()),'Final Adam state count differs')
    require(r['checkpoint']['strict'] is True and r['checkpoint']['optimizer_restored'] is False and r['checkpoint']['step']==2000,'Roundtrip evidence incomplete')
    for name,p in r['milestones']['2000']['parameters'].items():
        require(r['checkpoint']['tensor_hashes'][name]==p['sha256'],'Checkpoint/final-parameter identity differs')
    for update,m in r['milestones'].items():
        require(all(s['step']==int(update) for s in m['optimizer']['state'].values()),'Milestone Adam counter differs')
    require(r['elapsed_seconds']<freeze['limits']['arm_seconds'] and r['host_peak_rss_bytes']<freeze['limits']['host_memory_bytes'],'Recorded resource bound failed')


def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def execute(batch,out):
    require(batch.is_absolute() and batch.resolve()==batch and out.is_absolute() and out.resolve()==out and
            not out.exists() and batch not in out.parents,'Fresh separate analysis output required')
    require(digest(FREEZE)==FREEZE_SHA,'Frozen definition changed');freeze=load_json(FREEZE)
    for path,h in freeze['files'].items():require(digest(ROOT/path)==h,'Frozen source changed: '+path)
    for stage in PINS:baseline(stage,PINS[stage])
    bench=Path(freeze['host_benchmark'])
    for path,v in freeze['inputs'].items():require(digest(bench/path)==v['sha256'] and (bench/path).stat().st_size==v['bytes'],'Original input changed: '+path)
    failure=load_json(ROOT/'docs/evidence/BELL_THREE_ARM_E13_FAILURE_V1.json')
    for v in failure['records']:require(digest(Path(v['path']))==v['sha256'],'Earlier failed evidence changed')
    launch=load_json(batch/'launch.json');auth=load_json(batch/'authorization.json');panel=load_json(batch/'panel-status.json')
    require(launch['commit']==auth['commit']==COMMIT and launch['freeze_sha256']==auth['freeze_sha256']==FREEZE_SHA and
            digest(batch/'authorization.json')==launch['authorization_sha256'] and auth['approved'] is True and
            auth['run_ids']==freeze['run_ids'] and auth['host_output_directory']==str(batch)==freeze['host_output_directory'] and
            auth['limits']==freeze['limits'],'Authorization/launch identity differs')
    out.mkdir();reports={};all_outputs=[];endpoints=[];images=[];materials=[];bins=[];maxima={};distributions={}
    # Preserve a failed analysis as failed; never rewrite any native run evidence.
    try:
        for arm,sha in zip(ARMS,REPORT_SHAS):
            print('Verifying saved '+arm,flush=True)
            run=batch/arm;rp=run/'recovery.json';seal=load_json(run/'completion.json')
            require(digest(rp)==sha==seal['recovery_sha256'],'Report/completion hash differs')
            require((batch/(arm+'.exit-code.txt')).read_text().strip()=='0' and panel[arm]['exit_code']==0,'Nonzero/absent process exit')
            r=load_json(rp);verify_arm(r,seal,arm,freeze);reports[arm]=r
            for name,h in r['output_hashes'].items():
                p=run/name;require(p.resolve()==p and run in p.parents and digest(p)==h,'Native output hash differs: '+name);all_outputs.append(record(p))
            require(digest(run/'checkpoints/endpoint-2000.pth')==r['checkpoint']['sha256'],'Endpoint checkpoint hash differs')
            require(r['initial_parameters']==reports[ARMS[0]]['initial_parameters'] and r['splits']==reports[ARMS[0]]['splits'],'Initial parity differs')
            grouped={k:[] for k in ('train','validation','held_view_nvs')};maxima[arm]={}
            generation=load_json(bench/'geometry/generation.json')
            for image in map(str,range(128)):
                p=arrays(run/'materials'/(image+'.npz'));g=arrays(bench/'geometry/labels'/(image+'.npz'))
                errors=physical_reference(p,g)
                for k,v in errors.items():maxima[arm][k]=max(maxima[arm].get(k,0),v)
                summary=physical_summary([p]);compare(summary,r['material_recovery']['per_view'][image],arm+'.material.'+image)
                split=next(s for s,ids in generation['splits'].items() if image in ids);grouped[split].append({k:p[k] for k in ERRORS+('invalid_frame','invalid_material','orientation_eligible','orientation_prediction_defined')})
                materials.append(dict(arm=arm,source_id=image,split=split,observations=summary['observations'],eligible_orientation=summary['gt_orientation_eligible'],
                    **{k:summary[k]['mean'] for k in ERRORS},invalid_frames=summary['invalid_frames'],invalid_materials=summary['invalid_materials']))
            for split,rows in grouped.items():compare(physical_summary(rows),r['material_recovery']['split_aggregates'][split],arm+'.split.'+split)
            require(sum(len(x['invalid_frame']) for rows in grouped.values() for x in rows)==515698,'Export count differs')
            distributions[arm]={k:np.concatenate([x[k] for x in grouped['held_view_nvs']]) for k in ERRORS+('orientation_eligible','orientation_prediction_defined')}
            image_rows={}
            for key,reported in r['image_evaluation']['per_view'].items():
                mode,image=key.split('-');a=arrays(run/'evaluations'/(key+'.npz'))
                reference=arrays(bench/'references'/((('relight/' if mode=='relight' else '')+image)+'.npz'))['linear_components']
                gtpath=bench/('references/relight/'+image+'.png' if mode=='relight' else 'datasets/GlossySynthetic/bell/'+image+'.png')
                gt=png(gtpath).astype(np.float32)/255
                hit=arrays(bench/'geometry/rays'/(image+'.npz'))['hit'].reshape(128,128)
                if mode=='source':gt=gt*hit[:,:,None]
                require(np.array_equal(a['hit'],hit) and np.array_equal(a['ground_truth_srgb'],gt),'Image/mask/camera reference differs')
                expected=image_reference(a,reference)
                compare(expected,{k:reported[k] for k in expected},arm+'.image.'+key)
                image_rows[key]=expected
                images.append(dict(arm=arm,illumination=mode,source_id=image,split='validation' if image=='58' else 'held',
                    psnr=expected['native_psnr'],ssim=expected['native_ssim'],foreground_pixels=int(hit.sum()),
                    total_relative_l2=expected['total']['relative_l2'],diffuse_relative_l2=expected['diffuse']['relative_l2'],specular_relative_l2=expected['specular']['relative_l2']))
            require(len(image_rows)==12 and r['image_evaluation']['primary_rays']==196608,'Evaluation count differs')
            for mode in ('source','relight'):
                for name,ids in [('validation',generation['splits']['validation']),('held',generation['splits']['held_view_nvs'])]:
                    rows=[image_rows[mode+'-'+i] for i in ids]
                    independent=dict(native_psnr_mean=float(np.mean([x['native_psnr'] for x in rows])),native_ssim_mean=float(np.mean([x['native_ssim'] for x in rows])),pooled_linear_relative_l2=float(np.sqrt(sum(x['total']['squared_error'] for x in rows)/sum(x['total']['reference_squared'] for x in rows))))
                    compare(independent,r['image_evaluation']['aggregates'][mode+'-'+name],arm+'.aggregate')
            t=r['trajectory'];cfg=r['effective_losses'][0]['configuration']
            for start in range(0,2000,50):
                rows=t[start:start+50];raw=np.array([x['loss']/rgb_weight(x['update']-1,cfg) for x in rows])
                bins.append(dict(arm=arm,update_begin=start+1,update_end=start+50,weighted_loss_mean=float(np.mean([x['loss'] for x in rows])),
                    raw_charbonnier_mean=float(raw.mean()),excess_above_floor=float(raw.mean()-math.sqrt(.001)),
                    native_gradient_l2_mean=float(np.mean([x['gradient_l2'] for x in rows])),lr_mean=float(np.mean([x['lr'] for x in rows]))))
            held=r['material_recovery']['split_aggregates']['held_view_nvs'];image=r['image_evaluation']['aggregates']
            endpoints.append(dict(arm=arm,optimizer_returns=2000,held_observations=held['observations'],orientation_eligible=held['gt_orientation_eligible'],
                **{k:held[k]['mean'] for k in ERRORS},
                **{k+'_pass_fraction':v['fraction'] for k,v in held['descriptive_pass_fractions'].items()},
                held_source_psnr=image['source-held']['native_psnr_mean'],held_source_ssim=image['source-held']['native_ssim_mean'],held_source_relative_l2=image['source-held']['pooled_linear_relative_l2'],
                held_relight_psnr=image['relight-held']['native_psnr_mean'],held_relight_relative_l2=image['relight-held']['pooled_linear_relative_l2'],
                final_weighted_loss=t[-1]['loss'],final_rgb_loss_weight=rgb_weight(1999,cfg),final_raw_charbonnier=t[-1]['loss']/rgb_weight(1999,cfg),
                elapsed_seconds=r['elapsed_seconds'],torch_peak_MiB=r['gpu_memory']['peak_allocated_bytes']/1024**2,parent_peak_GiB=r['host_peak_rss_bytes']/1024**3,
                invalid_frames=held['invalid_frames'],invalid_materials=held['invalid_materials']))
        for name,rows in [('endpoints.csv',endpoints),('per_view_images.csv',images),('per_view_materials.csv',materials),('training_50_update_bins.csv',bins)]:table(out/name,rows)
        figures(out,bench,batch,reports,bins,distributions)
        receipt_files=[batch/'launch.json',batch/'authorization.json',batch/'panel-status.json']
        for arm in ARMS:receipt_files += [batch/arm/'recovery.json',batch/arm/'completion.json',batch/(arm+'.exit-code.txt'),batch/(arm+'.console.log'),batch/(arm+'.cid')]
        receipt_files += list(Path(str(batch)+'-launch').iterdir())
        result=dict(schema_version=1,protocol='bell-e13-v2-file-only-analysis-v1',decision='PROCEED_FOR_NATIVE_LIFECYCLE_ONLY',physical_recovery='NOT_ESTABLISHED',
            execution_commit=COMMIT,freeze=record(FREEZE),analysis_source=record(Path(__file__).resolve()),batch=str(batch),
            scientific_calls_this_analysis=0,checkpoint_deserializations=0,verified_frozen_sources=len(freeze['files']),verified_inputs=len(freeze['inputs']),verified_native_outputs=len(all_outputs),
            receipts=[record(p) for p in receipt_files],native_outputs=all_outputs,independent_material_reference_maximum_differences=maxima,
            numeric_verification=dict(float32_scaled_absolute_bound=CHECK32,float64_aggregate_bound=CHECK64,
                purpose='Offline arithmetic verification only; original scientific decisions, thresholds, identities and counts stay exact.'),
            all_initial_parameters_equal=True,training_endpoint_rng_equal=all(r['training_rng_endpoint']==reports[ARMS[0]]['training_rng_endpoint'] for r in reports.values()),
            evaluation_endpoint_rng_equal=all(r['evaluation_rng_endpoint']==reports[ARMS[0]]['evaluation_rng_endpoint'] for r in reports.values()),
            completed_original_training_forwards=6000,completed_original_backwards=6000,completed_original_Adam_returns=6000,
            evaluation_shader_chunks=sum(r['image_evaluation']['shader_chunks'] for r in reports.values()),
            evaluation_primary_rays=sum(r['image_evaluation']['primary_rays'] for r in reports.values()),evaluation_direction_slots=sum(r['image_evaluation']['direction_slots'] for r in reports.values()),
            total_native_arm_seconds=sum(r['elapsed_seconds'] for r in reports.values()),endpoints=endpoints,
            host_packages={n:importlib.metadata.version(n) for n in ('numpy','scikit-image','matplotlib','imageio')},
            limitations=['No independent Docker-engine/image inspection or checkpoint reload; native roundtrip/cleanup are inspected recorded evidence.',
                'Native camera positions/normals were gated but not saved as extra captured arrays; offline mask equality is verified.',
                'One scene, one seed, one finite-sample endpoint; shared RNG endpoints do not establish variance or significance.',
                'Only endpoint material exports exist; no intermediate physical-recovery trajectory is inferred.',
                'Weak-cue oracle is benchmark-defined geometry, not independently measured ground truth.'])
        with (out/'analysis.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        require(not any(n.split('.')[0] in ('torch','network','dataset','train','raytracing','_raytracing','scientific_variants') for n in sys.modules),'Scientific module imported by offline analysis')
        with (out/'completion.json').open('x') as stream:json.dump(dict(status='complete',analysis_sha256=digest(out/'analysis.json'),outputs=[record(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json']),stream,sort_keys=True,indent=2)
    except BaseException as exc:
        with (out/'failure.json').open('x') as stream:json.dump(dict(status='failed',error_type=type(exc).__name__,error=str(exc)),stream,indent=2)
        raise


def figures(out,bench,batch,reports,bins,distributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    colors=['#b34d30','#3373b0','#25865b'];short=['A original','B estimator','C core']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for arm,color,label in zip(ARMS,colors,short):
        rows=[x for x in bins if x['arm']==arm];x=[(r['update_begin']+r['update_end'])/2 for r in rows]
        for ax,key in zip(axes,['weighted_loss_mean','excess_above_floor','native_gradient_l2_mean']):ax.plot(x,[r[key] for r in rows],label=label,color=color)
    for ax,title in zip(axes,['Logged weighted RGB loss','Raw Charbonnier excess above sqrt(.001)','Native weighted gradient norm']):
        ax.set_title(title);ax.set_xlabel('Original optimizer returns');ax.grid(alpha=.2)
    axes[1].set_yscale('log');axes[2].set_yscale('log');axes[0].legend();fig.suptitle('Bell v3 weak-cue control: 50-update means; no additional model queries')
    fig.tight_layout();fig.savefig(out/'training.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    keys=['Q_world_relative','orientation_degrees','major_width_absolute','minor_width_absolute','albedo_max_absolute','metallic_absolute']
    for ax,key in zip(axes.ravel(),keys):
        for arm,color,label in zip(ARMS,colors,short):
            a=distributions[arm];v=a[key][a['orientation_eligible']&a['orientation_prediction_defined']] if key=='orientation_degrees' else a[key]
            q=np.linspace(0,1,501);ax.plot(np.quantile(v,q),q,label=label,color=color)
        limit={'Q_world_relative':.05,'orientation_degrees':5.,'albedo_max_absolute':.02,'metallic_absolute':.03}.get(key)
        if limit is not None:ax.axvline(limit,color='black',ls='--',lw=.8)
        ax.set_title(key);ax.set_ylabel('Empirical fraction');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Held-view material errors: 19846 observations; 17607 eligible orientations; frozen targets dashed')
    fig.tight_layout();fig.savefig(out/'physical_recovery.svg');plt.close(fig)
    from PIL import Image
    ids=['58','11','105','114','17','76'];fig,axes=plt.subplots(12,4,figsize=(8,23))
    for row,(mode,image) in enumerate((m,i) for m in ('source','relight') for i in ids):
        gt=bench/('datasets/GlossySynthetic/bell/'+image+'.png' if mode=='source' else 'references/relight/'+image+'.png')
        files=[gt]+[batch/arm/'renders'/(mode+'-'+image+'.png') for arm in ARMS]
        for col,p in enumerate(files):
            with Image.open(p) as im:axes[row,col].imshow(np.asarray(im),interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title((['Independent GT']+short)[col])
        axes[row,0].set_ylabel(mode+' / '+image)
    fig.suptitle('All predeclared evaluation views: native 8-bit sRGB, no exposure change')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'observations.svg',dpi=128);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();execute(args.batch,args.output)
