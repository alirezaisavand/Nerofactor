"""Publication report from verified O1/O0 files; no checkpoint loads or inference."""
import ast
import csv
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR','/tmp/bell-gate11-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks import bell_geometry_gate11 as g
from benchmarks.bell_geometry_e11 import mesh_arrays
from experiments.analyze_bell_geometry_gate10_v4_failure import logs,table
from infrastructure.signatures import require


def joint_statistics(reference,candidates):
    """Compare both reconstructions on identical oracle-ray foreground support."""
    require(len(candidates)==2,'Exactly the frozen O0/O1 pair is required')
    common=reference['hit']&candidates[0]['hit']&candidates[1]['hit']
    arrays=[]
    for candidate in candidates:
        depth=candidate['depth'][common].astype(float)-reference['depth'][common].astype(float)
        distance=np.linalg.norm(candidate['position'][common]-reference['position'][common],axis=1)
        ndot=(candidate['normal'][common]*reference['normal'][common]).sum(1)
        normal=np.degrees(np.arccos(np.clip(ndot,-1,1)))
        require(all(np.isfinite(x).all() for x in (depth,distance,normal)),'Nonfinite joint-support error')
        arrays.append(dict(signed_depth=depth,absolute_depth=abs(depth),displacement=distance,normal_degrees=normal))
    counts=dict(oracle=int(reference['hit'].sum()),joint=int(common.sum()),
        O0_missing=int((reference['hit']&~candidates[0]['hit']).sum()),
        O1_missing=int((reference['hit']&~candidates[1]['hit']).sum()))
    return counts,arrays


def distribution(values):
    if not len(values):return None
    require(np.isfinite(values).all(),'Nonfinite distribution input')
    return dict(count=len(values),minimum=float(values.min()),median=float(np.median(values)),
        p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)),
        maximum=float(values.max()),mean=float(values.mean()))


def mask_loss_validation(trace):
    steps=[x['completed'] for x in trace]
    expected=sorted(set(range(20,300001,20))|{1,2,1000,5000,15000,15001,20001,30000,30001,50000,50001,100000,200000,300000})
    require(steps==expected,'Mask-component step inventory differs')
    source=g.ROOT/'scientific_sources/geometry_g/network/loss.py'
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MaskLoss')
    scope=dict(Loss=object,np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original-MaskLoss-CPU','exec'),scope)
    original=scope['MaskLoss']({});schedule_roundoff=[]
    for row in trace:
        require(all(np.isfinite(row[k]) for k in ('foreground','background','total','mask_weight',
                    'weighted_foreground','weighted_background')),'Nonfinite mask component')
        require(np.float32(row['foreground'])+np.float32(row['background'])==np.float32(row['total']),
                'Recorded total differs from actual float32 foreground+background addition')
        weight=float(original.get_mask_weight(row['completed']-1))
        if weight!=row['mask_weight']:
            require(float(np.nextafter(weight,row['mask_weight']))==row['mask_weight'],
                    'Mask schedule differs beyond one float64 ULP')
            schedule_roundoff.append(row['completed'])
        require(row['weighted_foreground']==row['mask_weight']*row['foreground'] and
                row['weighted_background']==row['mask_weight']*row['background'],
                'Mask component weighting differs')
    return dict(records=len(trace),float32_addition_exact=True,original_schedule_replayed=True,
                schedule_one_ULP_events=schedule_roundoff,extra_gradient_queries=0)


def read_pair(folder,camera,fitting,acquisition):
    with np.load(folder/('fitting' if fitting else 'rasters')/(camera+'.npz'),allow_pickle=False) as z:
        if fitting:
            with np.load(acquisition/'rays'/(camera+'.npz'),allow_pickle=False) as a:
                reference={k:a[v] for k,v in dict(hit='hit',depth='depth',position='positions',normal='normals').items()}
            candidate={k:z[k] for k in ('hit','depth','position','normal')}
        else:
            reference={k:z['oracle_'+k] for k in ('hit','depth','position','normal')}
            candidate={k:z['reconstructed_'+k] for k in ('hit','depth','position','normal')}
    return reference,candidate


def original_folder(f,step,camera):
    if camera in f['fitting_ids']:return Path(f['O0_evolution_root'])/'run'/str(step)
    if step==300000:return g.evolution.ACCEPTED/'run'
    return Path(f['O0_evolution_root'])/'run'/str(step)


def figure_save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def comparison_figures(f,root,out,trace,training_bins,validation,old_validation):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    cameras=('14','58','105');fig,axes=plt.subplots(3,3,figsize=(8.5,8),constrained_layout=True)
    for ri,camera in enumerate(cameras):
        fitting=camera in f['fitting_ids'];reference,c0=read_pair(original_folder(f,300000,camera),camera,fitting,Path(f['acquisition'])/'geometry')
        _,c1=read_pair(root/'evaluate/300000',camera,fitting,Path(f['acquisition'])/'geometry')
        mask=reference['hit'].reshape(128,128)
        axes[ri,0].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        for ci,prediction in enumerate((c0,c1),1):
            hit=prediction['hit'].reshape(128,128);rgb=np.zeros((128,128,3))
            rgb[mask&hit]=.78;rgb[hit&~mask]=[.9,.22,.12];rgb[mask&~hit]=[.1,.45,.95]
            axes[ri,ci].imshow(rgb,interpolation='nearest')
        for ci in range(3):
            axes[ri,ci].set_xticks([]);axes[ri,ci].set_yticks([])
            if ri==0:axes[ri,ci].set_title(('Oracle','O0 300k','O1 300k')[ci])
        axes[ri,0].set_ylabel(f'Camera {camera}\n'+('Fitting' if fitting else 'Validation' if camera=='58' else 'Held view'))
    fig.suptitle('Fixed endpoint: gray common, red excess, blue missing; no mesh repair')
    figure_save(fig,out,'endpoint-support')

    fig,axes=plt.subplots(3,4,figsize=(11,8),constrained_layout=True);display_counts=[]
    for ri,camera in enumerate(('58','105','114')):
        reference,c0=read_pair(original_folder(f,300000,camera),camera,False,Path(f['acquisition'])/'geometry')
        _,c1=read_pair(root/'evaluate/300000',camera,False,Path(f['acquisition'])/'geometry')
        counts,arrays=joint_statistics(reference,(c0,c1));common=reference['hit']&c0['hit']&c1['hit']
        for j,data in enumerate(arrays):
            for offset,key in ((0,'signed_depth'),(2,'normal_degrees')):
                values=np.full(128**2,np.nan);values[common]=data[key]
                lo,hi=(-.04,.04) if key=='signed_depth' else (0.,30.)
                im=axes[ri,offset+j].imshow(values.reshape(128,128),cmap='coolwarm' if offset==0 else 'magma',vmin=lo,vmax=hi)
                if ri==2:fig.colorbar(im,ax=axes[:,offset+j],shrink=.6)
                display_counts.append(dict(camera=camera,arm=('O0','O1')[j],field=key,
                    below_display=int((data[key]<lo).sum()),above_display=int((data[key]>hi).sum()),joint=counts['joint']))
        for ci in range(4):
            axes[ri,ci].set_xticks([]);axes[ri,ci].set_yticks([])
            if ri==0:axes[ri,ci].set_title(('O0 depth','O1 depth','O0 normal error','O1 normal error')[ci])
        axes[ri,0].set_ylabel('Camera '+camera)
    fig.suptitle('Identical joint foreground; depth ±.04 scene units, signed-normal 0–30°; white = undefined')
    figure_save(fig,out,'endpoint-common-depth-normals')

    old_bins=np.genfromtxt(g.ROOT/'docs/results/publication_gate10/results-v4/training-loss-bin-medians.csv',names=True,delimiter=',')
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
    bx=np.array([x['step'] for x in training_bins])/1000
    for key,label in (('weighted_foreground','O1 foreground'),('weighted_background','O1 background')):
        values=np.array([x[key] for x in training_bins]);axes[0].plot(bx,np.where(values>0,values,np.nan),label=label)
    axes[0].plot(old_bins['step']/1000,np.where(old_bins['loss_mask']>0,old_bins['loss_mask'],np.nan),label='O0 foreground-only',color='black',alpha=.7)
    axes[0].set(ylabel='Weighted mask bin median',yscale='log');axes[0].legend(fontsize=8)
    for key,label in (('total','O1 foreground+background'),):
        axes[1].plot(bx,[x['mask_weighted_total'] for x in training_bins],label=label)
    axes[1].set(ylabel='Combined mask bin median',yscale='log');axes[1].legend(fontsize=8)
    for rows,label in ((old_validation,'O0'),(validation,'O1')):
        axes[2].plot([x['step']/1000 for x in rows],[x['val-psnr'] for x in rows],label=label)
    axes[2].set(xlabel='Completed updates (thousands)',ylabel='Native validation PSNR (dB)');axes[2].legend()
    for ax in axes:
        for step in (15,20,30,50):ax.axvline(step,color='.6',lw=.7,ls=':')
        ax.grid(alpha=.2)
    figure_save(fig,out,'mask-training-and-validation')
    return display_counts


def main():
    sha='4b09eb7871eb2f0d8e52f142a814d7148808a3908320c464d8d4d05a003cc81c'
    f=g.freeze(sha);root=Path(f['host_output']);out=root.parent/(root.name+'-finalization-v1')
    require(g.read(out/'preservation-and-validation.json')['status']=='original_verifier_replayed_exactly',
            'Complete independent file-only replay is required first')
    require(not (out/'summary.json').exists(),'Final analysis already exists')
    t=g.read(root/'train/training.json');r=g.read(root/'evaluate/evaluation.json')
    summary=g.read(root/'analysis/comparison-summary.json');old=g.read(g.ROOT/'docs/results/publication_gate103/results-v2/summary.json')
    old_r=g.read(Path(f['O0_evolution_root'])/'run/receipt.json')
    trace=[json.loads(line) for line in (root/'train/mask-components.jsonl').read_text().splitlines()]
    loss_check=mask_loss_validation(trace)
    training=logs(root/'train/logs/train.txt','train',20);validation=logs(root/'train/logs/val.txt','val',1000)
    old_val=logs(Path(f['O0']['root'])/'logs/val.txt','val',1000)
    require(len(training)==15000 and len(validation)==300,'Native log inventory differs')
    bins=[]
    for end in range(1000,300001,1000):
        block=[x for x in trace if end-1000<x['completed']<=end]
        bins.append(dict(step=end,records=len(block),**{k:float(np.median([x[k] for x in block])) for k in ('weighted_foreground','weighted_background')},
            mask_weighted_total=float(np.median([x['mask_weight']*x['total'] for x in block]))))
    table(out/'mask-component-bins.csv',bins);table(out/'validation.csv',validation)
    joint={};joint_table=[]
    for split,ids in (('fitting',f['fitting_ids']),('validation',f['splits']['validation']),('held_view_nvs',f['splits']['held_view_nvs'])):
        counts=dict(oracle=0,joint=0,O0_missing=0,O1_missing=0);arrays=[{},{}]
        for camera in ids:
            fitting=split=='fitting'
            reference,c0=read_pair(original_folder(f,300000,camera),camera,fitting,Path(f['acquisition'])/'geometry')
            _,c1=read_pair(root/'evaluate/300000',camera,fitting,Path(f['acquisition'])/'geometry')
            n,data=joint_statistics(reference,(c0,c1))
            for k,v in n.items():counts[k]+=v
            for i in (0,1):
                for k,v in data[i].items():arrays[i].setdefault(k,[]).append(v)
        arms={name:{k:distribution(np.concatenate(v)) for k,v in arrays[i].items()} for i,name in enumerate(('O0','O1'))}
        joint[split]=dict(counts=counts,arms=arms)
        for arm,fields in arms.items():
            for field,stats in fields.items():joint_table.append(dict(split=split,arm=arm,field=field,**stats))
    table(out/'joint-support-errors.csv',joint_table)
    topology=[]
    for a,b in zip(r['steps'],old_r['steps']):
        for arm,row in (('O0',b),('O1',a)):
            areas=np.array(row['topology_after']['component_areas']);top=row['topology_after']
            topology.append(dict(step=row['step'],arm=arm,components=top['edge_connected_components'],
                open_edges=top['open_boundary_edges'],nonmanifold_edges=top['nonmanifold_edges'],area=float(areas.sum()),
                secondary_area=float(areas.sum()-areas.max()),secondary_area_fraction=float((areas.sum()-areas.max())/areas.sum())))
    table(out/'topology.csv',topology)
    oracle_v,oracle_f=mesh_arrays(Path(f['acquisition'])/'geometry/mesh.ply')
    triangles=oracle_v[oracle_f].astype(float)
    oracle_area=float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2)
    boundary={};band=f['domain']['cell_counts']['maximum_boundary_omission_depth']
    for arm,folder,mesh in (('O0',g.evolution.ACCEPTED/'run','domain-aware-300000.ply'),
                            ('O1',root/'evaluate/300000','candidate.ply')):
        vertices,_=mesh_arrays(folder/mesh)
        with np.load(folder/'topology-after-cleanup.npz',allow_pickle=False) as z:edges=z['boundary_edges']
        midpoint=vertices[edges].astype(float).mean(1);radius=np.linalg.norm(midpoint,axis=1)
        boundary[arm]=dict(edges=len(edges),midpoint_radius=distribution(radius),
            edges_deeper_than_domain_band=int((radius<1-band).sum()),domain_band_width=band)
    display=comparison_figures(f,root,out,trace,bins,validation,old_val)
    native_validation=g.read(root/'offline-verification.json')
    g.write(out/'summary.json',dict(status='complete_verified_eligible_pending_researcher_admission',
        geometry_admitted=False,handoff_created=False,quality_eligible=native_validation['quality_eligible'],
        primary_endpoint=300000,source_commit=t['commit'],freeze_sha256=sha,variant=g.VARIANT,
        native_training=g.item(root/'train/training.json'),native_evaluation=g.item(root/'evaluate/evaluation.json'),
        checkpoint=t['endpoint_checkpoint'],mesh=r['steps'][-1]['mesh'],initial_parity=t['initial_parity_passed'],
        native_verification=g.item(root/'analysis/verification.json'),offline_replay=g.item(out/'verification.json'),
        prepared_comparison=g.item(root/'analysis/comparison-summary.json'),joint_support=joint,
        mask_loss_replay=loss_check,topology=topology,oracle_total_mesh_area=oracle_area,
        boundary_domain_statistics=boundary,display_limits=display,
        runtime=summary['runtime'],observed_work=r['observed'],
        interpretation='Strong silhouette/opacity improvement with small recall loss and worse p95 geometry/normal errors; oracle-mask diagnostic only.',
        new_scientific_execution=dict(checkpoint_loads=0,SDF_queries=0,native_rays=0,neural_renders=0,optimizer_updates=0)))
    # Include compact, original analysis artifacts without rewriting their bytes.
    for path in (root/'analysis').iterdir():
        if path.name!='artifact-manifest.json':shutil.copyfile(path,out/('prepared-'+path.name))
    g.write(out/'artifact-manifest.json',dict(native_root=str(root),
        native_manifest=g.item(root/'analysis/artifact-manifest.json'),
        native_files=g.read(root/'analysis/artifact-manifest.json')['files'],
        finalization_root=str(out),files={x.name:g.item(x) for x in out.iterdir() if x.is_file()},
        source_files={name:g.item(g.ROOT/name) for name in
            ('experiments/report_bell_geometry_gate11_results_v1.py','experiments/analyze_bell_geometry_gate10_v4_failure.py')}))
    destination=g.ROOT/'docs/results/publication_gate11/results-v1';destination.mkdir()
    for path in out.iterdir():
        if path.is_file():shutil.copyfile(path,destination/path.name)
    print(destination/'summary.json')


if __name__=='__main__':main()
