"""Verify Gate 8 terminal evidence and compare with saved K1, without rendering."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from benchmarks import donut_thesis_attribution_gate8_v1 as g
from benchmarks.bell_known_l2_v1 import verify_illumination_receipt
from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.neural_observation_e16 import compatibility
from experiments.analyze_donut_matched_geometry_v1 import distribution
from experiments.analyze_donut_relighting_gate6_v1 import raster, scores
from experiments.train_donut_thesis_attribution_gate8_v1 import valid_capture
from infrastructure.signatures import require


def member(root,receipt,name):
    return g.check(root/name,receipt['outputs'][name])


def endpoint(root,receipt,step,camera,view,original):
    path=f'step-{step:05d}/camera-{camera}'
    a=numeric(member(root,receipt,path+'/materials.npz'))
    mask=a['hit_mask']
    np.testing.assert_array_equal(np.flatnonzero(mask),view['pixel_indices'])
    for key,truth in [('linear_gt',view['linear_components'][:,2]),
                      ('position',view['position']),('normal',view['normals'])]:
        np.testing.assert_array_equal(a[key][mask],truth.astype(np.float32))
    frame=g.frame_checks(a)
    valid=np.logical_and.reduce(list(frame.values())) if original else np.ones(mask.sum(),bool)
    if original:
        np.testing.assert_array_equal(g.valid_mask(a),a['physical_valid'])
    f={k:a[k][mask] for k in ('kd','metallic','mx','my','rotation_raw')}
    t,b=2*a['tangents'][mask]-1,2*a['bitangents'][mask]-1
    mx,my=f['mx'],f['my']
    q=mx[:,:,None]**2*t[:,:,None]*t[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]
    np.testing.assert_array_equal(a['Q_world'][mask],q)
    w=np.sort(np.concatenate((mx,my),axis=1),axis=1)[:,::-1]
    orientation=np.where(mx>=my,t,b)
    linear=np.stack([a['linear_'+k][mask] for k in ('diffuse','specular','rgb')],axis=1).astype(float)
    require((linear>=0).all(),'Negative native L2 observations')
    if original:
        record=numeric(member(root,receipt,f'step-{step:05d}/camera-{camera}-errors.npz'))
        np.testing.assert_array_equal(record['physical_valid'],valid)
        np.testing.assert_array_equal(record['geometry_hit_mask'],view['hit_mask'])
    return dict(**f,Q_world=q,axis=orientation,widths=w,valid=valid,linear=linear,
                **{'valid_'+key:v for key,v in frame.items()})


def panel(pred,truth,selected,physical_scope):
    keep=selected&physical_scope
    width_diff=pred['widths']-truth['widths_major_minor']
    errors,physical={},None
    if keep.any():
        errors,physical=g.m.old.point_errors({k:pred[k][keep] for k in ('mx','my','kd','metallic','Q_world','axis')},
                                              {k:truth[k][keep] for k in g.m.TARGET_KEYS})
        physical['signed_width_bias']=width_diff[keep].mean(axis=0).tolist()
        physical['positive_minor_width_error_tail']=distribution(width_diff[keep,1][width_diff[keep,1]>0])
    reasons={key[6:]:int((selected&~values).sum()) for key,values in pred.items() if key.startswith('valid_')}
    scalar={}
    if selected.any():
        scalar=dict(color_max=distribution(abs(pred['kd'][selected]-truth['color'][selected]).max(axis=1)),
                    metallic=distribution(abs(pred['metallic'][selected]-truth['metallic'][selected]).reshape(-1)),
                    raw_head_width_MAE=abs(width_diff[selected]).mean(axis=0).tolist(),
                    raw_head_signed_width_bias=width_diff[selected].mean(axis=0).tolist())
    return dict(observations=int(selected.sum()),physical_frames_available=int((selected&pred['valid']).sum()),
                invalidity_counts_nonexclusive=reasons,physical=physical,scalar_and_raw_heads=scalar,
                L2_appearance=scores(pred['linear'],truth['linear_components'],selected)),errors


def comparison(pred,truth,mask):
    scope=mask&pred[g.ARM]['valid']&pred[g.CONTROL]['valid']
    rows,errors={},{}
    for arm in (g.ARM,g.CONTROL):
        rows[arm],errors[arm]=panel(pred[arm],truth,mask,scope)
    common=errors[g.ARM].get('orientation_eligible',np.zeros(0,bool)) & errors[g.CONTROL].get(
        'orientation_eligible',np.zeros(0,bool))
    return dict(methods=rows,common_physical_count=int(scope.sum()),common_orientation_count=int(common.sum()),
                common_orientation={arm:distribution(err['orientation_degrees'][common]) if common.any() else None
                                    for arm,err in errors.items()})


def concat(views,cams):
    return {key:np.concatenate([views[c][key] for c in cams]) for key in views[cams[0]]}


def endpoint_comparisons(pred,views,coverage,panels,adjacency,step):
    per_view={}
    for camera,view in views.items():
        pp={arm:pred[arm][camera] for arm in pred}
        scopes={'all':view['shader_valid']}
        if step==10000:
            scope=coverage[camera]
            scopes.update({key:scope[key][scope['pool']] for key in ('boundary','grazing','degenerate_adjacent','depth_outlier')})
            if camera in panels:
                scopes.update({key[5:]:value[view['pixel_indices']] for key,value in panels[camera].items()
                               if key.startswith('mask_') and key not in ('mask_full_raster','mask_extra_foreground')})
        per_view[str(camera)]={key:comparison(pp,view,mask) for key,mask in scopes.items()}
        if step==10000:
            pairs=metrics.boundary_pairs(view,adjacency)
            valid=pp[g.ARM]['valid'][pairs].all(axis=1)
            per_view[str(camera)]['matched_boundary_jumps']=dict(total_pairs=len(pairs),valid_pairs=int(valid.sum()),
                methods={arm:metrics.boundary_score(view,pairs[valid],p['Q_world'],p['widths']) for arm,p in pp.items()})
    splits={}
    for split,cameras in g.m.SPLITS.items():
        v=concat({c:{k:views[c][k] for k in (*g.m.TARGET_KEYS,'shader_valid','linear_components')} for c in cameras},cameras)
        pp={arm:concat(p,cameras) for arm,p in pred.items()}
        splits[split]=comparison(pp,v,v['shader_valid'])
    return dict(step=step,cameras=per_view,splits=splits)


def unseen(root,receipt,panels):
    views,image,uncertainty={},{},{}

    def evaluate(pred,truth,masks):
        return {arm:{key:scores(x,truth,mask) for key,mask in masks.items()} for arm,x in pred.items()}

    for camera,a in panels.items():
        z=numeric(member(root,receipt,f'unseen-camera-{camera}.npz'))
        np.testing.assert_array_equal(z['pixels'],a['pixels'])
        require(z['draws'].shape==(g.g6.DRAWS,len(z['pixels']),3,3),'Unseen draw shape differs')
        valid_capture(z['draws'])
        draws=z['draws'].astype(float)
        pred={g.ARM:raster(draws.mean(0),z['pixels']),g.CONTROL:a[g.CONTROL+'_native']}
        se=raster(draws.std(0,ddof=1)/np.sqrt(g.g6.DRAWS),z['pixels'])
        masks={k[5:]:v for k,v in a.items() if k.startswith('mask_')}
        views[str(camera)]=evaluate(pred,a['reference'],masks)
        common=masks['common']
        rms=lambda rgb: float(np.sqrt(np.mean(rgb**2)))
        mid=g.g6.DRAWS//2
        half=raster(draws[:mid].mean(0)-draws[mid:].mean(0),z['pixels'])
        uncertainty[str(camera)]=dict(
            original_RGB_standard_error_RMS=rms(se[common,2]),
            K1_RGB_standard_error_RMS=rms(a[g.CONTROL+'_native_SE'][common,2]),
            original_RGB_half_draw_difference_RMS=rms(half[common,2]),
            reference_RGB_grid_difference_RMS=rms((a['reference']-a['reference_coarse'])[common,2]))
        image[camera]=dict(pred=pred,reference=a['reference'],masks=masks)
    pred={arm:np.concatenate([v['pred'][arm] for v in image.values()]) for arm in (g.ARM,g.CONTROL)}
    truth=np.concatenate([v['reference'] for v in image.values()])
    masks={key:np.concatenate([v['masks'][key] for v in image.values()]) for key in next(iter(image.values()))['masks']}
    return dict(views=views,pooled=evaluate(pred,truth,masks),common_RGB_uncertainty=uncertainty),image


def figures(rows,image,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.bell_e11 import linear_to_srgb

    fig,ax=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for arm in (g.ARM,g.CONTROL):
        scores_=[r['splits']['held_out']['methods'][arm] for r in rows]
        ax[0].plot(g.m.STEPS,[s['physical']['Q_relative']['median'] if s['physical'] else np.nan for s in scores_],
                   'o-',label=arm)
        ax[1].plot(g.m.STEPS,[s['L2_appearance']['RGB']['relative_L2'] for s in scores_],'o-',label=arm)
    for a,yl in zip(ax,('Common-valid median Q error','Held-out L2 linear RGB relative L2')):
        a.set(xlabel='Optimizer returns',ylabel=yl)
        a.legend(frameon=False,fontsize=8)
    fig.savefig(output/'held-out-trajectory.pdf')
    plt.close(fig)
    fig,axs=plt.subplots(4,3,figsize=(7,8.7),layout='constrained')
    vmax=max(float(np.sqrt(((p[:,2]-v['reference'][:,2])**2).mean(1)).max())
             for v in image.values() for p in v['pred'].values())
    for i,(camera,v) in enumerate(image.items()):
        truth=v['reference'][:,2]
        axs[2*i,0].set_ylabel('Camera '+str(camera))
        for j,(name,data) in enumerate([('Reference',v['reference']),*v['pred'].items()]):
            rgb=linear_to_srgb(np.clip(data[:,2],0,1).reshape(g.g6.SIZE,g.g6.SIZE,3))
            axs[2*i,j].imshow(rgb)
            if i==0:
                axs[2*i,j].set_title(name)
            if j:
                err=np.sqrt(((data[:,2]-truth)**2).mean(-1)).reshape(g.g6.SIZE,g.g6.SIZE)
                heat=axs[2*i+1,j].imshow(err,cmap='magma',vmin=0,vmax=max(vmax,1e-15))
            else:
                axs[2*i+1,j].imshow(v['masks']['common'].reshape(g.g6.SIZE,g.g6.SIZE),cmap='gray')
        for ax in axs.flat:
            ax.set(xticks=[],yticks=[])
    fig.colorbar(heat,ax=axs[:,1:],label='Linear RGB pixel RMSE',shrink=.65)
    fig.savefig(output/'unseen-light-comparison.pdf')
    plt.close(fig)


def analyze(sha,output):
    f=g.freeze(g.ROOT/g.FREEZE,sha)
    batch=Path(f['host_output'])
    require(output.resolve()==output and not output.exists() and output.parent.is_dir()
            and batch not in output.parents,'Fresh Gate 8 offline results directory required')
    root,receipt=g.completed(batch,f,sha)
    control,k1=g.control_receipt(f)
    p=g.paths(f)
    _,views,refs,coverage,geometry=g.m.admit(g.gate7.gate5.parent(),p)
    views=views['oracle']
    panels=g.relight_reference(f,p,views)
    adjacency=metrics.mesh_adjacency(geometry['oracle']['triangles'])
    training={}
    for arm,path,r in ((g.ARM,root,receipt),(g.CONTROL,control,k1)):
        with member(path,r,'training.jsonl').open() as log:
            history=[json.loads(line) for line in log]
        require([row['completed_updates'] for row in history]==list(range(1,10001)), 'Training update sequence differs')
        for step in g.m.STEPS:
            member(path,r,f'step-{step:05d}/illumination.npz')
        verify_illumination_receipt(path,r,f['arm_recipe'],history)
        training[arm]=history
    for a,b in zip(training[g.ARM],training[g.CONTROL]):
        require(a['lr']==b['lr'],'Learning-rate stream differs')
    witness=numeric(member(root,receipt,'native-reference-gate.npz'))
    first=views[g.m.PANEL[0]]
    selection=np.linspace(0,len(first['pixel_indices'])-1,24).astype(int)
    np.testing.assert_array_equal(witness['pixel_indices'],first['pixel_indices'][selection])
    for i,k in enumerate(('coarse','reference')):
        np.testing.assert_array_equal(witness[k],refs['oracle',g.m.PANEL[0]][i,selection])
    gate=compatibility(witness['native_draws'],witness['reference'],abs(witness['reference']-witness['coarse']),
                       g.m.POLICY['compatibility_absolute'],g.m.POLICY['compatibility_relative'])
    require(gate==receipt['native_reference_gate'],'Original A compatibility evidence differs')
    rows=[]
    for step in g.m.STEPS:
        pred={arm:{c:endpoint(path,r,step,c,v,arm==g.ARM) for c,v in views.items()}
              for arm,path,r in ((g.ARM,root,receipt),(g.CONTROL,control,k1))}
        row=endpoint_comparisons(pred,views,coverage,panels,adjacency,step)
        if step==0:
            shifts={}
            for camera in views:
                a,b=(pred[arm][camera] for arm in (g.ARM,g.CONTROL))
                valid=a['valid']&b['valid']
                shifts[str(camera)]=dict(
                    raw_head_max_abs={k:float(abs(a[k]-b[k]).max()) for k in ('mx','my','kd','metallic','rotation_raw')},
                    common_physical_frames=int(valid.sum()),
                    Q_Frobenius=distribution(np.linalg.norm(a['Q_world'][valid]-b['Q_world'][valid],axis=(1,2))),
                    RGB_between_recipes=scores(a['linear'],b['linear'],np.ones(len(valid),bool)))
            row['initial_recipe_shifts']=shifts
        rows.append(row)
    relight,image=unseen(root,receipt,panels)
    output.mkdir()
    figures(rows,image,output)
    export={split:dict(observations=part['methods'][g.ARM]['observations'],valid=part['common_physical_count'])
            for split,part in rows[-1]['splits'].items()}
    for item in export.values():
        item['status']='complete' if item['valid']==item['observations'] else ('partial' if item['valid'] else 'unavailable')
    json_new(output/'comparison.json',dict(status='verified_complete',physical_export=export,rows=rows,unseen=relight,
        native_reference_diagnostic=gate,control_reference_diagnostic=k1['native_reference_gate'],
        training_diagnostics={arm:[r for r in history if 'gradient_interactions' in r]
                              for arm,history in training.items()},
        original_runtime=receipt['elapsed_seconds'],control_runtime=k1['elapsed_seconds'],
        original_initial=receipt['initial'],K1_initial=k1['initial'],optimizer_updates=0,renderer_calls=0))
    json_new(output/'artifact-manifest.json',dict(native_root=str(root),freeze_sha256=sha,
        original_receipt_sha256=g.digest(root/'recovery.json'),K1_receipt=f['K1_receipt'],
        original_outputs=receipt['outputs'],relight_panels=f['relight_panels'],
        files={p.name:dict(bytes=p.stat().st_size,sha256=g.digest(p)) for p in sorted(output.iterdir())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.freeze_sha256,args.output)
