"""Prepare fixed-endpoint O1/O0 comparison from verified saved captures only."""
import csv
import json
from pathlib import Path
import numpy as np
from benchmarks import bell_geometry_gate11 as g
from infrastructure.signatures import require
from experiments.analyze_bell_geometry_gate103_v3 import aggregate,geometry_points,neural_groups


def analyze(f,receipt,training,output):
    old=g.read(g.ROOT/'docs/results/publication_gate103/results-v2/summary.json')
    old_receipt=g.read(g.check(Path(f['O0_evolution_root'])/'run/receipt.json',f['O0_evolution_receipt']))
    old_training=g.read(g.check(Path(f['O0']['root'])/'training.json',f['O0']['training_receipt']))
    root=Path(f['host_output']);rows=[];comparisons=[];camera_rows=[]
    outside=np.load(f['outside_mask_path'],allow_pickle=False,mmap_mode='r')
    for row,baseline in zip(receipt['steps'],old['steps']):
        require(row['step']==baseline['step'],'O1/O0 checkpoint inventories differ')
        step=row['step'];folder=root/'evaluate'/str(step)
        raw=np.load(folder/'raw-SDF-grid.npy',allow_pickle=False,mmap_mode='r')
        require(g.negative_extension(raw,outside)==row['negative_extension'],'Negative SDF replay differs');del raw
        split_rows=dict(fitting=row['fitting'],validation=[v for v in row['views'] if v['split']=='validation'],
            held_view_nvs=[v for v in row['views'] if v['split']=='held_view_nvs'])
        splits={}
        for name,views in split_rows.items():
            current=aggregate(views)
            current['point_distributions']=geometry_points(folder,views,name=='fitting',Path(f['acquisition'])/'geometry')
            splits[name]=current;prior=baseline['splits'][name]
            comparisons.append(dict(step=step,split=name,
                **{key+'_O1_minus_O0':current[key]-prior[key] for key in
                    ('iou','precision','recall','extra','missing','correspondence')},
                depth_median_O1_minus_O0=current['point_distributions']['depth']['median']-prior['point_distributions']['depth']['median'],
                displacement_p95_O1_minus_O0=current['point_distributions']['displacement']['p95']-prior['point_distributions']['displacement']['p95'],
                normal_p95_O1_minus_O0=current['point_distributions']['normal_degrees']['p95']-prior['point_distributions']['normal_degrees']['p95']))
            prior_row=next(x for x in old_receipt['steps'] if x['step']==step)
            prior_views=prior_row['fitting'] if name=='fitting' else [x for x in prior_row['views'] if x['split']==name]
            prior_by_camera={x['camera']:x for x in prior_views}
            require(set(prior_by_camera)=={x['camera'] for x in views},'Paired per-camera inventory differs')
            for view in views:
                previous=prior_by_camera[view['camera']]
                camera_rows.append(dict(step=step,split=name,camera=view['camera'],
                    iou=view['silhouette_iou'],correspondence=view['matched_fraction_oracle'],
                    extra=view['reconstruction_only'],missing=view['oracle_only'],passed=view['passed'],
                    signed_depth_mean=view['signed_depth_error']['mean'],absolute_depth_p95=view['absolute_depth_error']['p95'],
                    displacement_max=view['surface_displacement']['maximum'],normal_p95=view['signed_normal_error_degrees']['p95'],
                    normal_max=view['signed_normal_error_degrees']['maximum'],
                    absolute_depth_p95_O1_minus_O0=view['absolute_depth_error']['p95']-previous['absolute_depth_error']['p95'],
                    normal_p95_O1_minus_O0=view['signed_normal_error_degrees']['p95']-previous['signed_normal_error_degrees']['p95']))
        groups,coexist=neural_groups(row,folder,f)
        rows.append(dict(step=step,splits=splits,neural_groups=groups,mesh_volume_coexistence=coexist,
            negative_extension=row['negative_extension'],topology=row['topology_after'],SDF_gate=row['SDF_gate']))
    for name,data in (('paired-comparison.csv',comparisons),('per-camera.csv',camera_rows)):
        with (output/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    summary=dict(protocol=g.PROTOCOL,variant=g.VARIANT,status='verified_fixed_endpoint_comparison',
        geometry_admitted=False,handoff_created=False,primary_endpoint=300000,steps=rows,
        paired_comparison=comparisons,initial_network_parity=training['initial_parity_passed'],
        O0_source_summary=f['O0_evolution_summary'],O1_training_receipt=g.item(root/'train/training.json'),
        O1_evaluation_receipt=g.item(root/'evaluate/evaluation.json'),
        runtime=dict(O1_training_seconds=training['wall_seconds'],O0_training_seconds=old_training['wall_seconds'],
            O1_evaluation_seconds=receipt['wall_seconds'],O1_training_peak_RSS=training['peak_RSS_bytes'],
            O1_training_peak_CUDA=training['peak_cuda_allocated_bytes'],
            interpretation='Total runtime includes diagnostics/system load, not an isolated timing of the background term.'),
        interpretation_limits=['Oracle-mask diagnostic only; no evidence of imperfect-mask robustness.',
            'Same original sampling policy; O0 did not archive full per-update ray/RNG tapes.',
            'All Bell cameras are development evidence; do not replace the fixed endpoint.'])
    g.write(output/'comparison-summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    steps=np.arange(4)
    panels=[('fitting','iou','Fitting IoU'),('fitting','recall','Fitting recall'),
        ('held_view_nvs','iou','Held pooled IoU'),('held_view_nvs','displacement','Held displacement p95'),
        ('held_view_nvs','normal_degrees','Held signed normal p95 (deg)'),('fitting','precision','Fitting precision')]
    for ax,(split,key,label) in zip(axes.flat,panels):
        for source,tag in ((old['steps'],'O0 archived'),(rows,'O1 background opacity')):
            values=[x['splits'][split]['point_distributions'][key]['p95'] if key in ('displacement','normal_degrees')
                    else x['splits'][split][key] for x in source]
            ax.plot(steps,values,'-o',label=tag)
        ax.set_xticks(steps,['15k','30k','50k','300k']);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.savefig(output/'O1-O0-evolution.png',dpi=190);fig.savefig(output/'O1-O0-evolution.pdf');plt.close(fig)
    with (root/'train/mask-components.jsonl').open() as stream:trace=[json.loads(line) for line in stream]
    require(len(trace)==15006 and all(np.isfinite([r['foreground'],r['background'],r['total']]).all() for r in trace),
            'Mask trace incomplete or nonfinite')
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    for key in ('weighted_foreground','weighted_background'):
        points=[(x['completed'],x[key]) for x in trace if x[key]>0]
        if points:ax.plot(*np.array(points).T,label=key)
    ax.set(xlabel='Completed updates',ylabel='Weighted component (positive entries)',yscale='log');ax.legend(fontsize=8)
    fig.savefig(output/'mask-components.png',dpi=180);plt.close(fig)
    g.write(output/'artifact-manifest.json',dict(root=str(root),files={x.relative_to(root).as_posix():g.item(x)
        for x in sorted(root.rglob('*')) if x.is_file()}))
