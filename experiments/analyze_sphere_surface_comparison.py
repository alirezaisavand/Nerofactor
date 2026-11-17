"""File-only comparison of the two accepted sphere surface forks; no Torch."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.sphere_visuals import uint8
from experiments.analyze_sphere_regularization import chart_statistics,q_numpy,prior_values

ARMS=('surface_chart_control_v1','surface_world_q_v1')
COMMIT='c1a06d4e685628974f8fcbd1f9791f8c6b6f8ab1'
FREEZE='5b1660f4c7fed73676e08646dd41e3ec95460214384660092160a240acb3627e'


def image_metrics(gt,prediction):
    """Independent transcription of pinned MaterialRenderMetrics, saved RGB only."""
    from skimage.metrics import structural_similarity
    gt,pr=uint8(gt,np),uint8(prediction,np)
    mse=((gt.reshape(-1,3).astype(np.float32)-pr.reshape(-1,3).astype(np.float32))**2).mean(0).mean()
    require(mse>0,'Unexpected exact image equality; retain explicit finite metrics')
    return dict(psnr=float(10*np.log10(255*255/mse)),
                ssim=float(structural_similarity(gt,pr,win_size=11,channel_axis=2,data_range=255)))


def acceptance(control,candidate):
    """The original frozen interpretation bounds; never fitted to these outcomes."""
    reduction=1-candidate['neighbor_Q_p95']/control['neighbor_Q_p95']
    gap=candidate['mean_gap']/control['mean_gap']-1
    trace=candidate['mean_Q_trace']/control['mean_Q_trace']-1
    checks=dict(Q_p95_reduction_at_least_20pct=bool(reduction>=.2),
                PSNR_loss_at_most_point2dB=bool(candidate['psnr']>=control['psnr']-.2),
                SSIM_loss_at_most_point005=bool(candidate['ssim']>=control['ssim']-.005))
    return dict(checks=checks,all_primary_checks_pass=all(checks.values()),
                neighbor_Q_p95_fraction_reduction=float(reduction),
                PSNR_candidate_minus_control=candidate['psnr']-control['psnr'],
                SSIM_candidate_minus_control=candidate['ssim']-control['ssim'],
                mean_gap_fraction_change=float(gap),mean_Q_trace_fraction_change=float(trace),
                isotropization_flag=bool(gap<=-.2),tensor_shrinkage_flag=bool(trace<=-.2))


def load_arrays(path):
    with np.load(path,allow_pickle=False) as data:a={k:data[k] for k in data.files}
    require(all(v.dtype.kind in 'bifu' and np.isfinite(v).all() for v in a.values()),'Non-numeric/nonfinite saved data')
    mask=a['hit_mask'];require(mask.dtype==bool and mask.any(),'Invalid saved hit mask')
    q=q_numpy(a)
    if 'Q_world' in a:
        error=float(np.max(abs(q[mask]-a['Q_world'][mask])))
        require(error<=2e-6,'Saved world Q disagrees with raw widths/frame reconstruction')
    return a


def summarize(a):
    m=a['hit_mask'];kd=a['kd'][m].astype(float);metal=a['metallic'][m].astype(float)
    d=kd*(1-metal);require(np.allclose(d,a['f_d'][m],atol=1e-7,rtol=1e-6),'Diffuse allocation identity mismatch')
    ld=a['linear_diffuse'][m].astype(float);ls=a['linear_specular'][m].astype(float)
    supported=d[:,0]>1e-8;light=ld[supported,0]/d[supported,0]
    mx=a['mx'][m].astype(float);my=a['my'][m].astype(float)
    gap=abs(mx**2-my**2)/np.maximum(mx**2+my**2,1e-12)
    neighbors=chart_statistics(a,m)
    return dict(**image_metrics(a['rgb_gt'],a['rgb_pr']),foreground_pixels=int(m.sum()),
        kd_red=float(kd[:,0].mean()),metallic=float(metal.mean()),diffuse_allocation_red=float(d[:,0].mean()),
        effective_diffuse_incident_red=float(light.mean()),light_ratio_supported_red=int(supported.sum()),
        linear_diffuse_red=float(ld[:,0].mean()),linear_specular_red=float(ls[:,0].mean()),
        mean_major_width=float(np.maximum(mx,my).mean()),mean_minor_width=float(np.minimum(mx,my).mean()),
        major_width_quantiles=np.quantile(np.maximum(mx,my),[.05,.5,.95]).tolist(),
        minor_width_quantiles=np.quantile(np.minimum(mx,my),[.05,.5,.95]).tolist(),
        mean_gap=float(gap.mean()),mean_Q_trace=float((mx**2+my**2).mean()),
        neighbor_Q_p95=neighbors['transported_Q_absolute_difference_p95'],neighbor_diagnostics=neighbors,
        diffuse_light_all_channels_clipped_fraction=float((a['diffuse_light'][m]>=1).all(-1).mean()),
        raw_rotation_norm_p05=float(np.quantile(np.linalg.norm(2*a['rotation_raw'][m]-1,axis=-1),.05)),
        validation_priors=prior_values(a,m))


def analyze(batch,output):
    from run_sphere_surface_experiment import preflight,authorize,completed_arm
    require(not output.exists(),'New analysis directory required; preserve originals')
    freeze_path=ROOT/'docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json'
    f=preflight(freeze_path,FREEZE);require(str(batch)==f['host_output_directory'],'Wrong selected batch')
    require(digest(batch/'freeze.json')==FREEZE,'Launch freeze changed')
    authorize(batch/'authorization.json',digest(batch/'authorization.json'),f,FREEZE,COMMIT)
    from infrastructure.sphere_replay import input_records
    inputs=input_records(Path('/home/isavand/nero/data/nerf_synthetic'),Path('/home/isavand/nero/final_results/meshes/sphere_shape-300000.ply'),Path('/home/isavand/nero/final_results'))
    require(inputs==json.loads((ROOT/f['input_manifest']['path']).read_text())['records'],'Original input identities changed')
    source_checkpoint=Path(f['initial_checkpoint']['host_path'])
    require(digest(source_checkpoint)==f['initial_checkpoint']['sha256'],'Initial checkpoint changed')
    artifacts={};reports={};logs={};components={};rows=[];displays={};pair_arrays={}
    def record(p):
        p=Path(p);require(p.resolve()==p and p.is_file(),'Missing/redirected evidence: '+str(p))
        artifacts[str(p)]=dict(bytes=p.stat().st_size,sha256=digest(p))
    for p in (batch/'freeze.json',batch/'authorization.json',source_checkpoint):record(p)
    original=Path('/home/isavand/paper/evidence/sphere-core-v2-ec8cee6b6d234133b0c7e1168fff02e1/sphere/renders/step-004000/validation/camera-36/materials.npz')
    initial_identity=json.loads((original.parent/'view.json').read_text())['files']['materials.npz']
    require(digest(original)==initial_identity['sha256'],'Initial saved material archive changed')
    record(original);record(original.parent/'view.json');initial=load_arrays(original)
    initial_stats=summarize(initial)
    displays['initial']=initial
    for arm in ARMS:
        base=batch/arm;root=base/'run';report_path=root/'surface-run.json'
        require((base/'exit-code.txt').read_text().strip()=='0','Arm process failed')
        r=completed_arm(report_path,arm,FREEZE,COMMIT);reports[arm]=r
        require(r['cleanup_errors']==[] and r['native_initialized'] and r['checkpoint_restore']['strict'] and
                r['checkpoint_restore']['optimizer_restored'] and r['checkpoint_restore']['sha256']==f['initial_checkpoint']['sha256'],
                'Native initialization/restoration/cleanup failure')
        require([p['zero_based_step'] for p in r['gradient_probes']]==[4000,5999,7999],'Missing derivative observations')
        require([p['absolute_completed_updates'] for p in r['optimizer_events']]==[4001,6000,8000] and
                all(p['active_state_counters_advanced'] and p['parameter_displacement_l1']>0 for p in r['optimizer_events']),
                'Missing finite optimizer progression evidence')
        require(r['ray_backend']['image_id']==f['image_id'] and r['ray_backend']['binary_sha256']==f['corrected_binary_sha256'],'Backend identity mismatch')
        for name,identity in r['outputs'].items():
            p=root/name;require(root in p.resolve().parents,'Escaped output identity')
            record(p);require(artifacts[str(p)]==identity,'Output identity mismatch: '+name)
        for p in (report_path,base/'exit-code.txt',base/'console.log',base/'container.cid',root/'training.jsonl',root/'components.jsonl',root/'config.effective.yaml',root/'config.original.yaml',root/'cache/surface-pairs.npz'):record(p)
        with np.load(root/'cache/surface-pairs.npz',allow_pickle=False) as pairs:pair_arrays[arm]={k:pairs[k] for k in pairs.files}
        require(int((pair_arrays[arm]['weights']>0).sum())==r['surface_pairs']['positive_weight_pairs'],'Pair coverage count mismatch')
        logs[arm]=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
        components[arm]=[json.loads(line) for line in (root/'components.jsonl').read_text().splitlines()]
        require([v['completed_updates'] for v in logs[arm] if v['prefix']=='train']==list(range(4010,8001,10)), 'Missing training log intervals')
        require([v['completed_updates'] for v in components[arm]]==list(range(4010,8001,10)),'Missing component log intervals')
        for step in (6000,8000):
            directory=root/'renders'/('step-%06d'%step)/'validation/camera-36';a=load_arrays(directory/'materials.npz')
            for key in ('hit_mask','position','normal','rgb_gt'):require(np.array_equal(a[key],initial[key]),'Camera/geometry/GT mismatch: '+key)
            summary=summarize(a);rows.append(dict(arm=arm,step=step,**summary))
            logged=next(v for v in logs[arm] if v['prefix']=='val' and v['completed_updates']==step)
            require(abs(logged['val-psnr']-summary['psnr'])<=1e-4 and abs(logged['val-ssim']-summary['ssim'])<=2e-6,'Saved-image metric replay mismatch')
            if step==8000:displays[arm]=a
    control,candidate=(reports[a] for a in ARMS)
    parity_keys=('initial_parameters','initial_buffers','initial_sample_tapes','initial_optimizer','initial_rng','split_ids','surface_pairs','effective_configuration','effective_shader','runtime','scheduler','imported_sources')
    require(all(control[k]==candidate[k] for k in parity_keys),'Paired initial/runtime/settings identity mismatch')
    require(candidate['control_packet_sha256']==digest(batch/ARMS[0]/'run/surface-run.json'),'Mounted control report identity mismatch')
    require(pair_arrays[ARMS[0]].keys()==pair_arrays[ARMS[1]].keys() and all(np.array_equal(pair_arrays[ARMS[0]][k],pair_arrays[ARMS[1]][k]) for k in pair_arrays[ARMS[0]]),'Saved surface-pair array mismatch')
    final=[next(r for r in rows if r['arm']==arm and r['step']==8000) for arm in ARMS]
    decision=acceptance(*final)
    result=dict(schema_version=1,status='saved_execution_and_outputs_verified',execution_commit=COMMIT,freeze_sha256=FREEZE,batch=str(batch),
        initial=initial_stats,observations=rows,decision=decision,original_input_files_verified=len(inputs),
        exact_scientific_work=dict(new_Adam_returns=8000,inherited_step_per_arm=4000,final_step_per_arm=8000,extra_component_reverse_sweeps=36,surface_predictor_points=8192000,extra_visibility_rays=2048,validation_views=4),
        initial_parity_checks=list(parity_keys),surface_pair_coverage=control['surface_pairs'],resources={a:reports[a]['resources'] for a in ARMS},
        runtime=control['runtime'],backend=control['ray_backend'],gradient_probes={a:reports[a]['gradient_probes'] for a in ARMS},optimizer_events={a:reports[a]['optimizer_events'] for a in ARMS},
        component_log_summary={a:dict(rows=len(components[a]),energy_nonzero_rows=sum(v['energy']!=0 for v in components[a]),mean_specular=float(np.mean([v['specular'] for v in components[a]]))) for a in ARMS},
        external_artifacts=artifacts,
        limitations=['One seed, one saved validation camera, no independent material ground truth or NVS in this bounded study.',
            'Common RNG/sampler reset; not an exact historical continuation. Chart control also changes legacy neighborhood selection.',
            'Screen-neighbor Q statistic excludes invalid/distant/normal-discontinuous neighbors; not a whole-mesh accuracy metric.',
            '192 of512 selected training-camera pairs are visible; a single image gate cannot cover all material boundaries.',
            'Pair descriptor hashes name pre-conversion float64 arrays; saved pair NPZ holds float32. Cache lacks original terminal file hash; current hash and cross-arm equality verified separately.',
            'Native ownership/runtime cleanup flags and zero process exits are verified; daemon-level container removal is not independently inspected here (no cleanup.log produced under --rm).',
            'Prior seven unresolved reflectance quadrature checks remain unresolved; this analysis performs no new integration.',
            'Three gradient probes cannot establish global loss causality, and400 component records do not observe every minibatch.'],
        new_GPU_calls=0,new_renderer_calls=0,new_checkpoint_deserialization=False)
    output.mkdir(parents=True)
    scalar_keys=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    with (output/'endpoints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    gradient_rows=[dict(arm=a,step=p['zero_based_step'],term=k,weighted_loss=v['weighted_loss'],gradient_l2=v['gradient_l2'],**{n+'_norm':x for n,x in v['groups'].items()}) for a in ARMS for p in reports[a]['gradient_probes'] for k,v in p['terms'].items()]
    with (output/'gradients.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(gradient_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(gradient_rows)
    figures(output,displays,rows,initial_stats,reports,logs,components)
    # Terminal byte checks include all consumed outputs/receipts, without loading weights.
    require(all(Path(p).stat().st_size==r['bytes'] and digest(p)==r['sha256'] for p,r in artifacts.items()),'Evidence changed during analysis')
    preflight(freeze_path,FREEZE)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def figures(output,displays,rows,initial,reports,logs,components):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from infrastructure.sphere_visuals import orientation_maps
    fig,axes=plt.subplots(3,8,figsize=(20,10))
    for row,(name,a) in enumerate(displays.items()):
        mask=a['hit_mask'];q=q_numpy(a)
        _,raw=orientation_maps(a['rotation'],np.ones_like(a['mx']),np.zeros_like(a['my']),mask,np)
        pr=a.get('major_axis_projector_world')
        if pr is None:
            t,b=2*a['tangents']-1,2*a['bitangents']-1;axis=np.where(a['my']>a['mx'],b,t);pr=axis[..., :,None]*axis[...,None,:]
        world=np.stack([pr[...,0,1],pr[...,0,2],pr[...,1,2]],-1)+.5
        fields=(a['rgb_pr'],a['kd'],a['metallic'][...,0],a['major_width'][...,0],a['minor_width'][...,0],raw,world*mask[...,None],np.stack([q[...,0,0],q[...,1,1],q[...,2,2]],-1))
        for col,(v,label) in enumerate(zip(fields,('RGB','albedo kd','metallic','major width','minor width','raw chart','world director projector','Qxx/Qyy/Qzz'))):
            axes[row,col].imshow(np.clip(v[::3,::3],0,1),vmin=0,vmax=1,cmap='gray' if v.ndim==2 else None)
            axes[row,col].set_title(('Initial4000' if name=='initial' else name.replace('surface_','').replace('_v1','')+'8000')+'\n'+label,fontsize=8);axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(output/'material-comparison.png',dpi=135);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm];label='chart' if arm==ARMS[0] else 'world-Q'
        for axis,key in zip(axes.flat,('kd_red','effective_diffuse_incident_red','neighbor_Q_p95','mean_minor_width','mean_gap','psnr')):
            axis.plot([4000,6000,8000],[initial[key]]+[r[key] for r in selected],'-o',label=label);axis.set_title(key);axis.set_xlabel('Absolute completed updates');axis.legend()
    fig.tight_layout();fig.savefig(output/'progression.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,arm in enumerate(ARMS):
        for key in ('loss_rgb','loss_mat_reg','loss_diffuse_light'):
            selected=[v for v in logs[arm] if v['prefix']=='train'];values=np.array([v[key] for v in selected]);axes[row,0].plot([v['completed_updates'] for v in selected],values,label=key)
        axes[row,0].set_yscale('log');axes[row,0].legend(fontsize=7);axes[row,0].set_title(arm+' weighted training losses',fontsize=8)
        terms=list(reports[arm]['gradient_probes'][0]['terms'])
        for column,field in ((1,'gradient_l2'),(2,'weighted_loss')):
            matrix=np.array([[p['terms'][k][field] for k in terms] for p in reports[arm]['gradient_probes']]);shown=np.ma.masked_where(matrix<=0,np.log10(np.maximum(matrix,1e-20)))
            im=axes[row,column].imshow(shown,vmin=-8,vmax=1,aspect='auto',cmap='viridis');axes[row,column].set_xticks(range(len(terms)));axes[row,column].set_xticklabels(terms,rotation=40,ha='right',fontsize=7);axes[row,column].set_yticks(range(3));axes[row,column].set_yticklabels([4000,5999,7999]);axes[row,column].set_title('log10 '+field+' (blank=zero)');fig.colorbar(im,ax=axes[row,column])
    fig.tight_layout();fig.savefig(output/'loss-gradient-diagnostics.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=analyze(args.batch,args.output);print(json.dumps(dict(status=r['status'],decision=r['decision']),indent=2))
