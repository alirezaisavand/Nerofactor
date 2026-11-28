"""Gate 13 file-only verification and paired material/geometry analysis.

Never constructs a model, deserializes a pickle, traces rays or evaluates a BRDF.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_material_geometry_gate13 as m
from experiments.analyze_donut_heterogeneous_training_v1 import read_endpoint
from benchmarks.donut_recovery_v1 import numeric
from benchmarks.neural_observation_e16 import compatibility
from benchmarks.bell_known_l2_v1 import route_plan,verify_routes,validate_map
from infrastructure.signatures import require


def distribution(a):
    a=np.asarray(a).reshape(-1)
    require(np.isfinite(a).all(),'Nonfinite Gate 13 metric')
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),
                minimum=float(a.min()),maximum=float(a.max())) if len(a) else dict(count=0)


def physical(a,d):
    """Both predictions and receiving truth transported into oracle's plane."""
    ids=d['pixel_indices'];mask=a['hit_mask']
    np.testing.assert_array_equal(np.flatnonzero(mask),ids)
    mx,my=a['mx'][mask],a['my'][mask]
    t,b=2*a['tangents'][mask]-1,2*a['bitangents'][mask]-1
    q=mx[:,:,None]**2*t[:,:,None]*t[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]
    axis=np.where(mx>=my,t,b);width=np.sort(np.c_[mx,my],axis=1)[:,::-1]
    R=m.g.minimal_rotation(d['normals'],d['oracle_normals'])
    common=m.g.rotate_tensor(q,R);truth=m.g.rotate_tensor(d['Q_world'],R)
    norm=lambda x:np.linalg.norm(x,axis=(1,2))
    receiving=norm(q-d['Q_world'])/norm(d['Q_world'])
    np.testing.assert_allclose(norm(common-truth)/norm(truth),receiving,atol=2e-6,rtol=0)
    director=np.einsum('nij,nj->ni',R,axis)
    gap=m.old.gap(width);eligible=gap>=.1
    values=dict(Q_receiving=receiving,Q_source=norm(common-d['oracle_Q'])/norm(d['oracle_Q']),
        Q_coordinate_frame_shift=norm(truth-d['oracle_Q'])/norm(d['oracle_Q']),
        major_bias=width[:,0]-d['widths_major_minor'][:,0],minor_bias=width[:,1]-d['widths_major_minor'][:,1],
        source_major_bias=width[:,0]-d['oracle_widths'][:,0],source_minor_bias=width[:,1]-d['oracle_widths'][:,1],
        log_mean_error=abs(np.log(width).mean(1)-np.log(d['widths_major_minor']).mean(1)),
        log_ratio_error=abs(np.log(width[:,0]/width[:,1])-np.log(d['widths_major_minor'][:,0]/d['widths_major_minor'][:,1])),
        orientation_receiving=np.degrees(np.arccos(np.clip(abs((axis*d['axis']).sum(1)),0,1))),
        orientation_source=np.degrees(np.arccos(np.clip(abs((director*d['oracle_axis']).sum(1)),0,1))),
        orientation_eligible=eligible&(m.old.gap(d['widths_major_minor'])>=.1),
        source_orientation_eligible=eligible&(m.old.gap(d['oracle_widths'])>=.1),
        color_error=abs(a['kd'][mask]-d['color']).mean(1),
        source_color_error=abs(a['kd'][mask]-d['oracle_color']).mean(1),
        metallic_bias=(a['metallic'][mask]-d['metallic']).reshape(-1),
        source_metallic_bias=(a['metallic'][mask]-d['oracle_metallic']).reshape(-1),
        gap=gap,major_predict=width[:,0],minor_predict=width[:,1])
    for prefix in ('','source_'):
        for component in ('major','minor','metallic'):
            values[prefix+component+'_error']=abs(values[prefix+component+'_bias'])
    values['major_relative_error']=values['major_error']/d['widths_major_minor'][:,0]
    values['minor_relative_error']=values['minor_error']/d['widths_major_minor'][:,1]
    require(all(np.isfinite(v).all() for v in values.values()),'Invalid physical export')
    return values,common


def verify_history(batch,arm,f,auth,views):
    r=m.completed(batch,arm,f,auth['commit'],auth['freeze_sha256'],m.digest(batch/'authorization.json'))
    root=batch/arm;recipe=f['arms'][arm];limits=recipe['limits']
    require(r['backend']==f['backend'] and r['effective_shader']==m.SHADER and r['policy']==m.POLICY and
        r['benchmark_sha256']==m.GENERATION and r['dataset_admission_sha256']==m.ADMISSION and
        r['historical_arm_initial_parity_verified'] and r['initial']==m.read(root/'initial.json') and
        r['elapsed_seconds']<limits['inner_seconds'] and r['resources']['host_peak_RSS']<64*1024**3 and
        sum(v['bytes'] for v in r['outputs'].values())<limits['output_bytes'], 'Native identity/resource gate differs')
    require(set(r['outputs'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
        and p.name not in ('recovery.json','progress.json')},'Unrecorded or missing native outputs')
    require(sorted(p.name for p in root.glob('checkpoint-*.pth'))==[f'checkpoint-{s:05d}.pth' for s in m.STEPS[1:]],
            'Own-checkpoint roundtrip schedule differs')
    history=[json.loads(v) for v in (root/'training.jsonl').read_text().splitlines()]
    require([v['completed_updates'] for v in history]==list(range(1,10001)) and
        history[0]['parameter_displacement_L1']==0<history[1]['parameter_displacement_L1'] and
        all(np.isfinite([v[k] for k in ('lr','RGB_loss','material_prior','diffuse_white_prior','preclip_gradient_l2')]).all()
            for v in history),'Incomplete/nonfinite Adam history')
    require(r['effective_lighting']=='known original Bell v3 environment' and r['diffuse_white_in_objective'] and
        all(r['trainable_parameters'][k]==0 for k in ('outer_light','inner_light')) and
        sum(r['trainable_parameters'].values())==1485584,'Known-light/full-capacity gate differs')
    audit=r['known_lighting_audit'];verify_routes(audit['routes'],route_plan(recipe['work'],False))
    require([v['step'] for v in audit['checkpoints']]==m.STEPS and
        len({v['state_sha256'] for v in audit['checkpoints']})==1 and
        all(v['optimizer_light_parameters']==v['optimizer_light_states']==v['learned_light_gradients']==0
            for v in audit['checkpoints']) and
        all(v['gradient_groups']['outer_light']==v['gradient_groups']['inner_light']==0 for v in history),
        'Illumination was optimized or disconnected from audited route')
    probes=[v for v in history if 'gradient_interactions' in v]
    require([v['completed_updates'] for v in probes]==[1,*m.STEPS[1:]] and
        all(v['gradient_interactions']['terms']['diffuse_white']['gradient_l2']==0 for v in probes),
        'Known-environment white term acquired a material gradient')
    gate=numeric(root/'native-reference-gate.npz');d=views[58]
    ix=np.linspace(0,len(d['pixel_indices'])-1,24).astype(int)
    np.testing.assert_array_equal(gate['pixel_indices'],d['pixel_indices'][ix])
    kind='oracle' if arm==m.ARMS[0] else 'O1';p=m.paths()
    for key,size in [('coarse',256),('reference',512)]:
        z=numeric(p['gate12']/f'58-{kind}-base-{size}.npz');j=np.searchsorted(z['pixels'],d['pixel_indices'][ix])
        np.testing.assert_array_equal(z['pixels'][j],d['pixel_indices'][ix])
        np.testing.assert_array_equal(gate[key],z['components'][j,0,2])
    replay=compatibility(gate['native_draws'],gate['reference'],abs(gate['reference']-gate['coarse']),
                         m.POLICY['compatibility_absolute'],m.POLICY['compatibility_relative'])
    require(replay['passed'] and r['native_reference_gate']['passed'],'Native/reference gate replay failed')
    previous=None
    for step in m.STEPS:
        z=numeric(root/f'step-{step:05d}/illumination.npz');validate_map(z['linear_RGB'],z['known_linear_RGB'])
        if previous is not None:np.testing.assert_array_equal(z['linear_RGB'],previous)
        previous=z['linear_RGB']
    require(len((root/'sampling-tapes.jsonl').read_text().splitlines())==10000,'Incomplete sampling ledger')
    return r,history


def paired_mc(draw0,draw1):
    """Matched eight-draw uncertainty; not independent quadrature accuracy."""
    require(draw0.shape==draw1.shape and draw0.shape[0]==8 and draw0.shape[2:]==(3,3), 'Draw inventory differs')
    delta=draw1.astype(float)-draw0
    return np.std(delta,axis=0,ddof=1)/np.sqrt(8)


def geometry_diagnostics(data,p):
    """No new rays: saved hits and Gate 12 incident-visibility counts only."""
    from scipy.spatial import cKDTree
    vertices,faces=m.g.mesh_arrays(p['O1']/'candidate.ply')
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=vertices[unique[counts==1]].mean(1)
    tree=cKDTree(boundary) if len(boundary) else None
    oracle_vertices,_=m.g.mesh_arrays(p['oracle']/'geometry/mesh.ply')
    bounds=np.stack([oracle_vertices.min(0),oracle_vertices.max(0)]).astype(float)
    result={}
    for camera,d in data.items():
        ids=d['pixel_indices'];R=m.g.minimal_rotation(d['normals'],d['oracle_normals'])
        intermediate=m.g.analytic(d['position'],d['oracle_normals'],bounds)['Q']
        receiving_in_oracle=m.g.rotate_tensor(d['Q_world'],R)
        denominator=np.linalg.norm(d['oracle_Q'],axis=(1,2))
        row=dict(near_open_boundary=tree.query(d['position'])[0]<=.02 if tree else np.zeros(len(ids),bool),
            coordinate_only_Q_shift=np.linalg.norm(intermediate-d['oracle_Q'],axis=(1,2))/denominator,
            normal_projection_vs_transport_Q_shift=np.linalg.norm(receiving_in_oracle-intermediate,axis=(1,2))/denominator)
        if str(camera) in m.g.PANEL:
            z=numeric(p['gate12']/f'{camera}-O1-base-512.npz');j=np.searchsorted(z['pixels'],ids)
            np.testing.assert_array_equal(z['pixels'][j],ids)
            row['incident_disagreement_any']=z['visibility'][j,2]>0
            row['incident_disagreement_fraction']=z['visibility'][j,2]/512**2
            row['fixed_material_RGB_squared']=((z['components'][j,0,2]-d['linear_components'][:,2])**2).mean(1)
        result[camera]=row
    return result


def known_support(d,a):
    """No prediction is invented at an unevaluated hit or undefined material."""
    common=a['hit_mask'].reshape(-1);background=~d['hit_mask'].reshape(-1)
    known=common|background;prediction=a['linear_rgb'].reshape(-1,3)
    require(not prediction[background].any(),'Nonzero unrendered background')
    target=d['full_linear_components'][:,2]
    return dict(raster_pixels=len(common),known_support=int(known.sum()),undefined=int((~known).sum()),
        common_evaluated=int(common.sum()),geometric_background=int(background.sum()),
        squared_error_sum=float(((prediction[known]-target[known])**2).sum()),
        RMSE_on_known_support=float(np.sqrt(((prediction[known]-target[known])**2).mean())),
        full_raster_score_defined=bool(known.all()))


def summarize(values,selection):
    answer={}
    for k,a in values.items():
        if a.dtype==bool:continue
        use=selection.copy()
        if k=='orientation_receiving':use &= values['orientation_eligible']
        if k=='orientation_source':use &= values['source_orientation_eligible']
        if k=='orientation_common_degrees':use &= values['orientation_common_eligible']
        if k=='source_orientation_common_degrees':use &= values['source_orientation_common_eligible']
        answer[k]=distribution(a[use])
    answer['Q_at_most_005']=int(((values['Q_receiving']<=.05)&selection).sum())
    answer['orientation_eligible']=int((values['orientation_eligible']&selection).sum())
    answer['source_orientation_eligible']=int((values['source_orientation_eligible']&selection).sum())
    answer['minor_at_floor']=int(((values['minor_predict']<=.0401)&selection).sum())
    answer['major_at_ceiling']=int(((values['major_predict']>=.9999)&selection).sum())
    return answer


def correlation(a,b):
    """Descriptive Pearson correlation only; no causal or significance claim."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None


def analyze(sha):
    f=m.freeze(m.ROOT/m.FREEZE,sha);batch=Path(f['host_output']);auth=m.read(batch/'authorization.json')
    require(auth==m.authorization(f,auth['commit'],sha) and m.digest(batch/'freeze.json')==sha,'Foreign execution approval')
    output=batch/'analysis';require(not output.exists(),'Preserve previous analysis evidence');output.mkdir()
    p=m.paths();datasets={arm:m.dataset(f,p,f['arms'][arm]['geometry'])[1] for arm in m.ARMS}
    receipts,histories={},{}
    for arm in m.ARMS:receipts[arm],histories[arm]=verify_history(batch,arm,f,auth,datasets[arm])
    require([r['lr'] for r in histories[m.ARMS[0]]]==[r['lr'] for r in histories[m.ARMS[1]]], 'Scheduler histories differ')
    diagnostics=geometry_diagnostics(datasets[m.ARMS[1]],p)
    rows=[];summary={};paired=[]
    for step in m.STEPS:
        split_values={arm:{s:[] for s in ('fitting','validation','held_out')} for arm in m.ARMS}
        for camera in m.panel():
            values={};arrays={};draws={}
            split='fitting' if str(camera) in m.split()['train'] else 'validation' if camera==58 else 'held_out'
            for arm in m.ARMS:
                root=batch/arm;d=datasets[arm][camera]
                errors,rgb,target,q,width=read_endpoint(root,step,camera,d)
                a=numeric(root/f'step-{step:05d}/camera-{camera}/materials.npz');arrays[arm]=a
                val,common=physical(a,d)
                np.testing.assert_allclose(val['Q_receiving'],errors['Q_relative'],atol=2e-12,rtol=0)
                z=numeric(root/f'native-draws/step-{step:05d}-camera-{camera}.npz')
                np.testing.assert_array_equal(z['component_reference_indices'],[2,0,1]);draws[arm]=z['draws']
                for j,key in enumerate(('linear_rgb','linear_diffuse','linear_specular')):
                    np.testing.assert_array_equal(z['draws'].mean(0)[:,j],a[key][a['hit_mask']])
                val.update(RGB_squared=((rgb-target)**2).mean(1),
                    RGB_reference_squared=(target**2).mean(1),
                    diffuse_squared=((a['linear_diffuse'][a['hit_mask']]-d['linear_components'][:,0])**2).mean(1),
                    specular_squared=((a['linear_specular'][a['hit_mask']]-d['linear_components'][:,1])**2).mean(1),
                    RGB_MC_SE_RMS=np.sqrt((np.std(z['draws'].astype(float),axis=0,ddof=1)[:,0]**2/8).mean(1)))
                values[arm]=val
            common_orientation=values[m.ARMS[0]]['orientation_eligible']&values[m.ARMS[1]]['orientation_eligible']
            source_common=values[m.ARMS[0]]['source_orientation_eligible']&values[m.ARMS[1]]['source_orientation_eligible']
            d0,d1=(datasets[arm][camera] for arm in m.ARMS)
            angle=np.degrees(np.arccos(np.clip((d0['normals']*d1['normals']).sum(1),-1,1)))
            displacement=np.linalg.norm(d1['position']-d0['position'],axis=1)
            cosine=np.minimum((d0['normals']*d0['outgoing']).sum(1),(d1['normals']*d1['outgoing']).sum(1))
            paired_se=paired_mc(*(draws[arm] for arm in m.ARMS))
            cohorts=dict(all=np.ones(len(angle),bool),normal_gt15=angle>15,grazing=cosine<.1,
                displacement_gt01=displacement>.01,weak_receiving_truth=m.old.gap(d0['widths_major_minor'])<.1)
            cohorts.update({k:v for k,v in diagnostics[camera].items() if v.dtype==bool})
            for arm in m.ARMS:
                v=values[arm];v.update(orientation_common_eligible=common_orientation,
                    source_orientation_common_eligible=source_common,
                    orientation_common_degrees=v['orientation_receiving'],source_orientation_common_degrees=v['orientation_source'])
                val=v
                split_values[arm][split].append(val)
                report={name:summarize(val,c) for name,c in cohorts.items()}
                report['orientation_common']=distribution(v['orientation_receiving'][common_orientation])
                report['source_orientation_common']=distribution(v['orientation_source'][source_common])
                report['RGB_linear_RMSE']=float(np.sqrt(v['RGB_squared'].mean()))
                report['RGB_relative_L2']=float(np.sqrt(v['RGB_squared'].sum()/max(v['RGB_reference_squared'].sum(),1e-30)))
                covariates=dict(normal_degrees=angle,displacement=displacement,min_view_cosine=cosine)
                if 'incident_disagreement_fraction' in diagnostics[camera]:
                    covariates['incident_disagreement_fraction']=diagnostics[camera]['incident_disagreement_fraction']
                associations={key:{name:correlation(v[key],z) for name,z in covariates.items()}
                    for key in ('Q_receiving','Q_source','minor_bias','RGB_squared')}
                rows.append(dict(arm=arm,step=step,camera=camera,split=split,metrics=report,descriptive_correlations=associations,
                    raster_support=known_support(datasets[arm][camera],arrays[arm])))
            paired.append(dict(step=step,camera=camera,split=split,points=len(angle),
                Q_O1_minus_oracle=distribution(values[m.ARMS[1]]['Q_receiving']-values[m.ARMS[0]]['Q_receiving']),
                RGB_squared_O1_minus_oracle=distribution(values[m.ARMS[1]]['RGB_squared']-values[m.ARMS[0]]['RGB_squared']),
                lower_RGB_error_with_worse_receiving_Q=int(((values[m.ARMS[1]]['RGB_squared']<values[m.ARMS[0]]['RGB_squared'])&
                    (values[m.ARMS[1]]['Q_receiving']>values[m.ARMS[0]]['Q_receiving'])).sum()),
                O1_native_MSE_minus_fixed_true_MSE=distribution(values[m.ARMS[1]]['RGB_squared']-
                    diagnostics[camera]['fixed_material_RGB_squared']) if 'fixed_material_RGB_squared' in diagnostics[camera] else None,
                paired_eight_draw_RGB_difference_SE_RMS=float(np.sqrt((paired_se[:,0]**2).mean())),
                initial_common_plane_Q_relative=distribution(np.linalg.norm(
                    physical(arrays[m.ARMS[1]],d1)[1]-physical(arrays[m.ARMS[0]],d0)[1],axis=(1,2))/np.linalg.norm(d0['Q_world'],axis=(1,2))) if step==0 else None))
            if step in (0,10000):
                fields=dict(pixel_indices=d0['pixel_indices'],normal_degrees=angle,displacement=displacement,grazing=cosine<.1,
                    orientation_common=common_orientation,source_orientation_common=source_common,paired_component_SE=paired_se)
                fields.update(diagnostics[camera])
                for arm in m.ARMS:fields.update({arm+'_'+k:v for k,v in values[arm].items()})
                np.savez_compressed(output/f'paired-{step:05d}-camera-{camera}.npz',**fields)
            if step==10000 and str(camera) in m.split()['held_view_nvs']:
                figure(output,camera,arrays,{arm:datasets[arm][camera] for arm in m.ARMS},values)
        summary[str(step)]={}
        for arm in m.ARMS:
            summary[str(step)][arm]={}
            for name,chunks in split_values[arm].items():
                v={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
                summary[str(step)][arm][name]=summarize(v,np.ones(len(v['Q_receiving']),bool))
    curves(output,summary,histories)
    m.write(output/'summary.json',dict(protocol=m.PROTOCOL,status='verified_complete',primary_step=10000,
        support=f['support'],coverage=m.read(m.ROOT/m.INVENTORY)['coverage'],metrics=summary,
        uncertainty='Eight paired native draws measure stochastic variation, not deterministic diffuse bias or absolute BRDF integration accuracy.',
        geometry='O1 perfect-synthetic-mask-assisted derivative; oracle-assisted correspondence restriction.',
        Gate12='Fixed true-material discrepancy is neither an attainable nor irreducible optimized fitting bound.'))
    m.write(output/'per-camera.json',rows);m.write(output/'paired.json',paired)
    m.write(output/'histories.json',histories)
    m.write(output/'artifact-manifest.json',dict(freeze_sha256=sha,commit=auth['commit'],native={arm:m.item(batch/arm/'recovery.json') for arm in m.ARMS},
        files={p.relative_to(output).as_posix():m.item(p) for p in output.rglob('*') if p.is_file()}))
    print(output/'summary.json')


def curves(output,summary,histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for arm in m.ARMS:
        hist=histories[arm];x=[v['completed_updates'] for v in hist]
        for col,key in enumerate(('RGB_loss','material_prior','preclip_gradient_l2')):
            ax[0,col].plot(x[::20],[v[key] for v in hist[::20]],label=arm,lw=.7);ax[0,col].set_title(key);ax[0,col].set_yscale('symlog',linthresh=1e-6)
        for col,(key,metric) in enumerate((('Q_receiving','median'),('minor_bias','mean'),('RGB_squared','mean'))):
            for split,style in [('fitting','-'),('held_out','--')]:
                y=[summary[str(s)][arm][split][key][metric] for s in m.STEPS]
                ax[1,col].plot(m.STEPS,y,style,label=arm+' '+split);ax[1,col].set_title(key+' '+metric)
    for a in ax.flat:a.set_xlabel('Original Adam returns');a.legend(fontsize=7);a.grid(alpha=.2)
    fig.savefig(output/'trajectories.png',dpi=220);fig.savefig(output/'trajectories.pdf');plt.close(fig)


def figure(out,camera,arrays,views,values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.bell_e11 import linear_to_srgb
    fig,ax=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    target=views[m.ARMS[0]]['full_linear_components'][:,2].reshape(128,128,3)
    # Explicit identical display-only clipping. Numerical arrays stay unclipped.
    ax[0,0].imshow(linear_to_srgb(np.clip(target,0,1)));ax[0,0].set_title('Original oracle target')
    for j,arm in enumerate(m.ARMS):
        a=arrays[arm];mask=a['hit_mask'];rgb=a['linear_rgb'].copy()
        rgb[views[arm]['hit_mask']&~mask]=[1,0,1]
        ax[0,j+1].imshow(linear_to_srgb(np.clip(rgb,0,1)));ax[0,j+1].set_title(arm+' (undefined magenta)')
        v=values[arm]
        for col,key,cap in [(j*2,'Q_receiving',1.),(j*2+1,'minor_bias',.15)]:
            image=np.full((128,128),np.nan);image[mask]=v[key]
            im=ax[1,col].imshow(image,vmin=-cap if key.endswith('bias') else 0,vmax=cap,cmap='coolwarm' if key.endswith('bias') else 'magma')
            ax[1,col].set_title(arm+' '+key);fig.colorbar(im,ax=ax[1,col],shrink=.65)
    delta=np.full((128,128),np.nan);mask=arrays[m.ARMS[0]]['hit_mask']
    delta[mask]=np.sqrt(values[m.ARMS[1]]['RGB_squared'])-np.sqrt(values[m.ARMS[0]]['RGB_squared'])
    im=ax[0,3].imshow(delta,vmin=-.02,vmax=.02,cmap='coolwarm');ax[0,3].set_title('RGB RMSE O1 − oracle');fig.colorbar(im,ax=ax[0,3],shrink=.65)
    for a in ax.flat:a.axis('off')
    fig.savefig(out/f'camera-{camera}-comparison.png',dpi=220);fig.savefig(out/f'camera-{camera}-comparison.pdf');plt.close(fig)


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__);c.add_argument('--freeze-sha256',required=True)
    analyze(c.parse_args().freeze_sha256)
