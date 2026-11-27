"""Saved-array checkpoint evolution analysis. No model, CUDA or new ray queries."""
import csv
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks import bell_geometry_gate103_v2 as p
from infrastructure.signatures import require


def csv_new(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(rows):
    oracle=sum(x['oracle'] for x in rows);common=sum(x['common'] for x in rows)
    candidate=sum(x['reconstructed'] for x in rows);valid=sum(x['matched_support'] for x in rows)
    return dict(cameras=len(rows),oracle=oracle,reconstructed=candidate,common=common,
        extra=candidate-common,missing=oracle-common,iou=common/(oracle+candidate-common),
        precision=common/candidate,recall=common/oracle,correspondence=valid/oracle,
        passed_views=sum(x['passed'] for x in rows),min_iou=min(x['silhouette_iou'] for x in rows),
        extra_radial_interior=sum(x.get('extra_radial_interior',0) for x in rows),
        missing_inside_domain=sum(x['missing_inside_domain'] for x in rows),
        missing_outside_domain=sum(x['missing_outside_domain'] for x in rows))


def geometry_points(folder,rows,fitting,acquisition):
    buckets=dict(depth=[],displacement=[],normal_degrees=[])
    for row in rows:
        camera=row['camera']
        with np.load(folder/('fitting' if fitting else 'rasters')/f'{camera}.npz',allow_pickle=False) as z:
            if fitting:
                with np.load(acquisition/'rays'/f'{camera}.npz',allow_pickle=False) as a:
                    oracle={k:a[v] for k,v in dict(hit='hit',normal='normals',position='positions',depth='depth').items()}
                candidate={k:z[k] for k in ('hit','normal','position','depth')}
            else:
                oracle={k:z['oracle_'+k] for k in ('hit','normal','position','depth')}
                candidate={k:z['reconstructed_'+k] for k in ('hit','normal','position','depth')}
            common=oracle['hit']&candidate['hit']
            buckets['depth'].append(candidate['depth'][common].astype(float)-oracle['depth'][common].astype(float))
            buckets['displacement'].append(np.linalg.norm(candidate['position'][common]-oracle['position'][common],axis=1))
            ndot=(oracle['normal'][common]*candidate['normal'][common]).sum(1)
            buckets['normal_degrees'].append(np.degrees(np.arccos(np.clip(ndot,-1,1))))
    return {k:p.policy.original.distribution(np.concatenate(v)) for k,v in buckets.items()}


def neural_groups(row,folder,plan):
    selected=[x for x in row['fitting'] if 'neural' in x];groups={}
    for name in selected[0]['neural']:
        rows=[x['neural'][name] for x in selected];total=sum(x['count'] for x in rows)
        keys=('stable_false_positive','stable_false_negative','threshold_flips','mesh_false_positive','mesh_false_negative')
        groups[name]=dict(count=total,**{k:sum(x[k] for x in rows) for k in keys},
            mean_acc_by_draw=[sum(x['mean_acc_by_draw'][d]*x['count'] for x in rows)/total for d in (0,1)])
    coexist=dict(mesh_false_positive_stable_high_opacity=0,mesh_false_positive_stable_low_opacity=0,
                 mesh_false_positive_sampling_flip=0,stable_high_opacity_without_mesh_hit=0)
    for camera in plan['neural_panel']:
        values=[]
        for draw in (0,1):
            with np.load(folder/'neural'/f'{camera}-{draw}.npz',allow_pickle=False) as z:
                values.append(z['fg_acc']);oracle=z['oracle_hit'];mesh=z['mesh_hit']
        high=np.stack(values)>.5;extra=mesh&~oracle
        coexist['mesh_false_positive_stable_high_opacity']+=int((extra&high.all(0)).sum())
        coexist['mesh_false_positive_stable_low_opacity']+=int((extra&(~high).all(0)).sum())
        coexist['mesh_false_positive_sampling_flip']+=int((extra&(high[0]!=high[1])).sum())
        coexist['stable_high_opacity_without_mesh_hit']+=int((~oracle&~mesh&high.all(0)).sum())
    return groups,coexist


def finish_figure(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def figures(out,summary,receipt,plan):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    rows=summary['steps'];steps=[x['step'] for x in rows];x=np.arange(4)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for name,label in (('fitting','122 fitting cameras'),('validation','Validation 58'),('held_view_nvs','Five held views')):
        axes[0,0].plot(x,[r['splits'][name]['iou'] for r in rows],'-o',label=label)
    axes[0,0].axhline(.90,color='gray',ls=':',label='Per-view gate (not pooled)')
    axes[0,0].set(ylabel='Pooled mesh IoU');axes[0,0].legend(fontsize=7)
    for key,label in (('extra','Extra'),('missing','Missing')):
        axes[0,1].plot(x,[r['splits']['fitting'][key] for r in rows],'-o',label=label)
    axes[0,1].set(ylabel='Fitting pixels');axes[0,1].legend()
    for key,label in (('normal_degrees','Signed normal error (deg)'),('displacement','Displacement ×1000')):
        axes[0,2].plot(x,[r['splits']['held_view_nvs']['point_distributions'][key]['p95']*(1000 if key=='displacement' else 1) for r in rows],'-o',label=label)
    axes[0,2].set(ylabel='Held-view p95');axes[0,2].legend(fontsize=7)
    for key,label in (('mesh_false_positive','Mesh extra hits'),('stable_false_positive','Stable opacity > .5')):
        axes[1,0].plot(x,[r['neural_groups']['outer_band'][key] for r in rows],'-o',label=label)
    axes[1,0].set(ylabel='Outer-band sample count / 1024');axes[1,0].legend(fontsize=7)
    for key,label in (('inside','Inside supported sphere'),('outside','Outside (unsupported extrapolation)')):
        axes[1,1].plot(x,[r['negative_extension'][key]['negative']/r['negative_extension'][key]['nodes'] for r in rows],'-o',label=label)
    axes[1,1].set(ylabel='Raw negative-SDF node fraction');axes[1,1].legend(fontsize=7)
    train=np.genfromtxt(p.ROOT/'docs/results/publication_gate10/results-v4/training-loss-bin-medians.csv',delimiter=',',names=True)
    for key in ('loss_rgb','loss_occ','loss_mask','loss_opacity'):
        y=train[key];good=np.isfinite(y)&(y>0)
        axes[1,2].plot(train['step'][good]/1000,y[good],label=key.replace('loss_',''))
    for step in steps:axes[1,2].axvline(step/1000,color='gray',lw=.6,alpha=.6)
    axes[1,2].set(yscale='log',xlabel='Completed updates (thousands)',ylabel='Logged weighted bin median')
    axes[1,2].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        if ax is not axes[1,2]:ax.set_xticks(x,[f'{s//1000}k' for s in steps]);ax.set_xlabel('Completed updates')
    finish_figure(fig,out,'geometry-evolution')

    cameras=('14','58','105');fig,axes=plt.subplots(3,5,figsize=(11,6.5),constrained_layout=True)
    for ri,camera in enumerate(cameras):
        with np.load(Path(plan['acquisition'])/'geometry/rays'/f'{camera}.npz',allow_pickle=False) as a:truth=a['hit'].reshape(128,128)
        axes[ri,0].imshow(truth,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        for ci,row in enumerate(receipt['steps'],1):
            folder=Path(plan['host_output'])/'run'/str(row['step'])
            if camera=='14':
                with np.load(folder/'fitting'/f'{camera}.npz',allow_pickle=False) as z:hit=z['hit'].reshape(128,128)
            else:
                if row['step']==300000:folder=p.ACCEPTED/'run'
                with np.load(folder/'rasters'/f'{camera}.npz',allow_pickle=False) as z:hit=z['reconstructed_hit'].reshape(128,128)
            image=np.zeros((128,128,3));image[truth&hit]=.78;image[hit&~truth]=[.90,.24,.14];image[truth&~hit]=[.12,.45,.95]
            axes[ri,ci].imshow(image,interpolation='nearest')
        for ci in range(5):
            axes[ri,ci].set_xticks([]);axes[ri,ci].set_yticks([])
            if ri==0:axes[ri,ci].set_title(('Oracle','15k','30k','50k','300k')[ci])
        axes[ri,0].set_ylabel(f'Camera {camera}\n'+('Fitting' if camera=='14' else 'Validation' if camera=='58' else 'Held view'))
    fig.suptitle('Same calibrated rays. Gray: common; red: excess; blue: missing. Fixed cameras, no selection by score.')
    finish_figure(fig,out,'silhouette-progression')

    fig,axes=plt.subplots(2,4,figsize=(11,6),constrained_layout=True)
    # Camera 105 is a fixed original held view. All maps use identical scales.
    for ci,row in enumerate(receipt['steps']):
        folder=Path(plan['host_output'])/'run'/str(row['step']) if row['step']!=300000 else p.ACCEPTED/'run'
        with np.load(folder/'rasters/105.npz',allow_pickle=False) as z:
            common=z['common'];depth=np.full(128**2,np.nan);angle=depth.copy()
            depth[common]=z['reconstructed_depth'][common].astype(float)-z['oracle_depth'][common].astype(float)
            ndot=(z['oracle_normal'][common]*z['reconstructed_normal'][common]).sum(1)
            angle[common]=np.degrees(np.arccos(np.clip(ndot,-1,1)))
        im0=axes[0,ci].imshow(depth.reshape(128,128),cmap='coolwarm',vmin=-.05,vmax=.05)
        im1=axes[1,ci].imshow(angle.reshape(128,128),cmap='magma',vmin=0,vmax=40)
        axes[0,ci].set_title(f'{row["step"]//1000}k')
        for ax in axes[:,ci]:ax.set_xticks([]);ax.set_yticks([])
    axes[0,0].set_ylabel('Signed depth difference');axes[1,0].set_ylabel('Signed normal error (degrees)')
    fig.colorbar(im0,ax=axes[0,:],shrink=.7);fig.colorbar(im1,ax=axes[1,:],shrink=.7)
    fig.suptitle('Fixed held camera 105, common support only; white = undefined support')
    finish_figure(fig,out,'depth-normal-progression')


def main():
    plan=p.freeze('e027b4ab9320049086b4e3d7b83c47646f6b228cf9399b43a015a84b089bb862')
    root=Path(plan['host_output']);out=root.parent/(root.name+'-offline-v3')
    verification=p.read(out/'verification.json');receipt=p.read(root/'run/receipt.json')
    require(verification['status']=='complete_verified_diagnostic' and verification['receipt']==p.item(root/'run/receipt.json'),
            'Verified native captures required')
    require(not (out/'summary.json').exists(),'Analysis already finalized')
    outside=np.load(Path(p.policy.read(p.ROOT/p.policy.FREEZE)['diagnostic'])/'run/stages/outside-sphere.npy',allow_pickle=False,mmap_mode='r')
    steps=[];camera_rows=[];neural_rows=[]
    for row in receipt['steps']:
        step=row['step'];folder=root/'run'/str(step)
        raw_path=folder/'raw-SDF-grid.npy' if step!=300000 else Path(plan['raw300_path'])
        raw=np.load(raw_path,allow_pickle=False,mmap_mode='r')
        require(p.negative_extension(raw,outside)==row['negative_extension'],'Raw negative-grid summary replay differs');del raw
        split_rows=dict(fitting=row['fitting'],validation=[v for v in row['views'] if v['split']=='validation'],
                        held_view_nvs=[v for v in row['views'] if v['split']=='held_view_nvs'])
        splits={}
        for name,views in split_rows.items():
            splits[name]=aggregate(views)
            source=folder if name=='fitting' or step!=300000 else p.ACCEPTED/'run'
            splits[name]['point_distributions']=geometry_points(source,views,name=='fitting',Path(plan['acquisition'])/'geometry')
            for view in views:
                camera_rows.append(dict(step=step,split=name,camera=view['camera'],iou=view['silhouette_iou'],
                    correspondence=view['matched_fraction_oracle'],extra=view['reconstruction_only'],missing=view['oracle_only'],
                    displacement_p95=view['surface_displacement']['p95'],normal_p95=view['signed_normal_error_degrees']['p95'],
                    passed=view['passed']))
        groups,coexist=neural_groups(row,folder,plan)
        for name,values in groups.items():neural_rows.append(dict(step=step,group=name,**{k:v for k,v in values.items() if k!='mean_acc_by_draw'},
            mean_acc_draw0=values['mean_acc_by_draw'][0],mean_acc_draw1=values['mean_acc_by_draw'][1]))
        steps.append(dict(step=step,splits=splits,neural_groups=groups,mesh_volume_coexistence=coexist,
            negative_extension=row['negative_extension'],topology=row['topology_after'],SDF_gate=row['SDF_gate']))
    summary=dict(status='completed_verified_diagnostic',geometry_admitted=False,handoff_created=False,
        publication_endpoint=300000,endpoint_replacement_permitted=False,comparison_is_development_evidence=True,
        native_commit=receipt['commit'],native_freeze_sha256=receipt['freeze_sha256'],image=receipt['image'],
        native_receipt=p.item(root/'run/receipt.json'),offline_verification=p.item(out/'verification.json'),
        roundoff_events=verification['roundoff_events'],camera_precision=receipt['steps'][0]['camera_precision_diagnostic'],
        runtime=dict(wall_seconds=receipt['wall_seconds'],peak_RSS_bytes=receipt['peak_RSS_bytes'],
            peak_cuda_allocated_bytes=receipt['peak_cuda_allocated_bytes']),work=receipt['observed'],steps=steps,
        interpretation=dict(
            supported='Excess silhouette support is present on fitting views by 15k, decreases by 30k, then grows by the fixed 300k endpoint. Stable sampled background opacity accompanies most measured mesh excess.',
            not_established='These snapshots do not isolate the causal effects of mask, opacity, occlusion, appearance or background NeRF.',
            next_diagnostic='Optional matched perfect-mask background-opacity ablation; a realistic method must instead use admitted confident background and preserve unknown segmentation pixels.',
            realistic_candidate='Confidence-aware trimap opacity is the first lower-cost candidate; retain direct SDF supervision for measured low-opacity/mesh discrepancies.',
            prohibitions='No checkpoint substitution, trimming, threshold change, new optimization, segmentation inference or automatic geometry admission.'),
        additional_scientific_execution=dict(checkpoint_loads=0,SDF_queries=0,extractions=0,new_BVH_queries=0,neural_renders=0,optimizer_returns=0))
    csv_new(out/'per-camera.csv',camera_rows);csv_new(out/'neural-panel.csv',neural_rows)
    p.write(out/'summary.json',summary);figures(out,summary,receipt,plan)
    for name in ('verification.json','reported-offline-failure.json'):
        require((out/name).is_file(),'Missing offline evidence')
    native_files={x.relative_to(root).as_posix():p.item(x) for x in sorted(root.rglob('*')) if x.is_file()}
    analysis_files={x.relative_to(out).as_posix():p.item(x) for x in sorted(out.rglob('*')) if x.is_file()}
    p.write(out/'artifact-manifest.json',dict(native_root=str(root),native_files=native_files,
        analysis_root=str(out),analysis_files=analysis_files,source_files={
            name:p.item(p.ROOT/name) for name in ('experiments/analyze_bell_geometry_gate103_v3.py',
                'experiments/bell_geometry_gate103_offline_v3.py','benchmarks/bell_geometry_gate103_replay_v3.py',
                'docs/results/publication_gate10/results-v4/training-loss-bin-medians.csv')}))
    destination=p.ROOT/'docs/results/publication_gate103/results-v2';destination.mkdir()
    for path in out.iterdir():
        if path.is_file():shutil.copyfile(path,destination/path.name)
    print(destination/'summary.json')


if __name__=='__main__':main()
