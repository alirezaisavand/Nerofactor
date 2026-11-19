"""Create Bell recovery figures and three-benchmark tables from verified files."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_heterogeneous_training_v1 as h
from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.donut_surface_q_d6 import NeighborhoodSampler
from experiments.analyze_donut_heterogeneous_training_v1 import read_endpoint
from experiments.plot_donut_heterogeneous_training_v1 import artifact, plot_curves
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


RESULTS = h.ROOT/'docs/results'
DONUT_D6 = RESULTS/'donut_surface_q_d6_v1/endpoint-metrics.csv'
DONUT_D8 = RESULTS/'donut_paired_tensor_d8_v1/endpoint-metrics.csv'
HET_DONUT = RESULTS/'donut_heterogeneous_training_v1'
METRICS = ('RGB_relative_L2','Q_median','Q_p95','major_width_MAE','minor_width_MAE')


def write_csv(path, header, rows):
    with path.open('x',newline='') as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def endpoint_comparison(analysis):
    rows = []
    for source, paths in (('Homogeneous Donut',(DONUT_D6,DONUT_D8)),
                          ('Heterogeneous Donut',(HET_DONUT/'checkpoint-metrics.csv',))):
        old = {}
        for path in paths:
            with path.open() as file:
                for row in csv.DictReader(file):
                    if row['step']=='10000' and row['split']=='held_out' and row['arm'] in h.ARMS:
                        keys = METRICS if 'Q_median' in row else ('RGB_relative_L2','q_median','q_p95',
                                                                 'major_abs_mean','minor_abs_mean')
                        values = [float(row[k]) for k in keys]
                        require(old.setdefault(row['arm'],values)==values,'Historical duplicate endpoint differs')
        require(set(old)==set(h.ARMS),'Historical held-out endpoint missing')
        rows.extend((source,arm,*old[arm]) for arm in h.ARMS)
    for row in analysis['results']:
        if row['step']==10000 and row['split']=='held_out':
            x = row['metrics']['all']['observation_weighted']
            rows.append(('Heterogeneous Bell',row['arm'],x['RGB_relative_L2'],x['Q_relative']['median'],
                         x['Q_relative']['p95'],x['major_width']['mean'],x['minor_width']['mean']))
    require(len(rows)==12,'Expected all three four-arm comparisons')
    return rows


def plot_views(batch, views, predictions, output, plt, *, arm_roots=None):
    from PIL import Image

    fig, axes = plt.subplots(6,5,figsize=(10,10.4),layout='constrained')
    roots = {arm:batch/arm for arm in h.ARMS} if arm_roots is None else arm_roots
    arms = tuple(roots)
    for row,camera in enumerate(h.SPLITS['held_out']):
        v = views[camera]
        visible = v['shader_valid']
        pixels = v['pixel_indices'][visible]
        for col,arm in enumerate(('Oracle',*arms)):
            root = roots[arm if col else arms[0]]/'step-10000'/f'camera-{camera}'
            ax = axes[3*row,col]
            with Image.open(root/('reconstructed.png' if col else 'ground_truth.png')) as png:
                ax.imshow(np.asarray(png),interpolation='nearest')
            ax.set_title(arm)
            minor = predictions[arm,camera]['minor_predict'] if col else v['widths_major_minor'][visible,1]
            err = predictions[arm,camera]['Q_relative'] if col else np.zeros(len(pixels))
            for offset,values,vmin,vmax,cmap in ((1,minor,.04,.24,'viridis'),(2,err,0,1,'magma')):
                xy = np.full(h.b.SIZE**2,np.nan)
                xy[pixels] = values
                layer = axes[3*row+offset,col].imshow(xy.reshape(h.b.SIZE,h.b.SIZE),
                    interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
                if col==len(arms):
                    fig.colorbar(layer,ax=axes[3*row+offset],fraction=.013,pad=.01,extend='max')
        for offset,label in enumerate((f'Camera {camera} RGB','Minor width','Relative Q error')):
            axes[3*row+offset,0].set_ylabel(label)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output/'held-out-views.png',dpi=165)
    plt.close(fig)


def load_spatial(batch, receipt, arm):
    root = batch/arm
    for name in ('surface-q-domain.npz','training.jsonl'):
        h.b.check_artifact(root/name,receipt['outputs'][name])
    return numeric(root/'surface-q-domain.npz'), load_history(root/'training.jsonl')


def load_history(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    require([row['completed_updates'] for row in rows]==list(range(1,10001)), 'Incomplete loss history')
    return rows


def spatial_scale(domain):
    distance = domain['distance'][domain['training_ids']]
    scale = (h.SPATIAL_POLICY['reference_distance']/distance)**2
    return dict(distance_min_median_max=np.quantile(distance,[0,.5,1]).tolist(),
                weight_min_median_max=np.quantile(scale,[0,.5,1]).tolist())


def shortest_edge_replay(f, domain, logs):
    minimum = int(domain['training_ids'][np.argmin(domain['distance'][domain['training_ids']])])
    sampler = NeighborhoodSampler(domain['training_ids'],len(domain['distance']),h.SPATIAL_POLICY['rng_seed'])
    hits = []
    fields = ('completed_updates','surface_q_loss','RGB_loss','preclip_gradient_l2','mx_mean','my_mean')
    for step,log in enumerate(logs,1):
        drawn = sampler.draw(h.SPATIAL_POLICY['batch_pairs'])
        if minimum in drawn:
            weight = (h.SPATIAL_POLICY['reference_distance']/domain['distance'][drawn])**2
            relative = float(weight[drawn==minimum].sum()/weight.sum())
            hits.append(dict(weight_fraction_of_batch=relative,**{k:log[k] for k in fields}))
        if step in h.STEPS:
            require(h.array_hash(sampler.counts)==f['sampler_snapshots'][str(step)],'Spatial replay count differs')
    peak = max(logs,key=lambda row:row['surface_q_loss'])
    step = peak['completed_updates']
    normal_cosine = np.dot(domain['normals0'][minimum],domain['normals1'][minimum])
    return dict(pair_id=minimum,face_ids=domain['face_pairs'][minimum].tolist(),
        dihedral_degrees=float(np.degrees(np.arccos(np.clip(normal_cosine,-1,1)))),
        shortest_native_distance=float(domain['distance'][minimum]),hits=hits,
        peak_spatial_loss_step=step,
        peak_neighborhood_training_values=[{k:log[k] for k in fields} for log in logs[step-2:step+7]])


def plot_spatial_failure(bell, bell_baseline, donut, event, output, plt):
    fig,(left,right) = plt.subplots(1,2,figsize=(9.7,3.8),layout='constrained')
    steps = np.arange(1,2201)
    for name,history in (('Bell D6',bell),('Heterogeneous Donut D6',donut)):
        loss = [row['surface_q_loss'] for row in history[:len(steps)]]
        left.semilogy(steps,loss,lw=.85,label=name)
    left.set(xlabel='Update',ylabel='Weighted surface-Q loss')
    for name,history in (('D2',bell_baseline),('D6',bell)):
        for coordinate,style in (('mx','-'),('my','--')):
            right.plot(steps,[v[coordinate+'_mean'] for v in history[:len(steps)]],style,
                       lw=1.,label=f'{name} {coordinate}',color='C0' if name=='D2' else 'C1')
    right.set(xlabel='Update',ylabel='Fitting-batch mean chart width',xlim=(1200,2200))
    for ax in (left,right):
        ax.axvline(event,color='.25',lw=.8,ls=':')
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(output/'spatial-collapse.png',dpi=165)
    plt.close(fig)


def package(freeze_sha, analysis_dir, analysis_sha, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h.freeze(h.ROOT/h.FREEZE,freeze_sha)
    batch = Path(f['host_output'])
    auth_path = batch/'authorization.json'
    approval = h.b.read(auth_path)
    a_path = analysis_dir/'analysis.json'
    require(digest(a_path)==analysis_sha, 'Explicit verified Bell analysis required')
    a = h.b.read(a_path)
    require(approval==h.authorization(f,approval['commit'],freeze_sha) and a['freeze_sha256']==freeze_sha
            and a['protocol']==h.PROTOCOL and a['status']=='verified_complete'
            and a['execution_commit']==approval['commit'] and a['authorization_sha256']==digest(auth_path),
            'Verified Bell execution or analysis differs')
    require(output.resolve()==output and not output.exists() and batch not in output.parents
            and Path(f['host_reference']) not in output.parents, 'Separate fresh results directory required')
    d = f['dataset']
    _,views = h.dataset(f,Path(f['host_reference']),Path(d['admission']['path']),Path(d['verification']['path']))
    manifest, receipts, predictions = {}, {}, {}
    for arm in h.ARMS:
        r_path = batch/arm/'recovery.json'
        identity = artifact(r_path)
        require(identity['sha256']==a['arms'][arm]['receipt_sha256'], 'Native Bell receipt differs')
        r = receipts[arm] = h.b.read(r_path)
        manifest[arm] = dict(receipt=identity,console=artifact(batch/f'{arm}-console.txt'),
            exit_code=artifact(batch/f'{arm}-exit-code.txt'),
            checkpoints={name:x for name,x in r['outputs'].items() if name.endswith('.pth')},
            outputs=len(r['outputs']),output_bytes=sum(v['bytes'] for v in r['outputs'].values()))
        for c in h.SPLITS['held_out']:
            root = f'step-10000/camera-{c}'
            for name in (f'{root}/materials.npz',f'{root}/reconstructed.png',
                         f'{root}/ground_truth.png',f'{root}-errors.npz'):
                h.b.check_artifact(batch/arm/name,r['outputs'][name])
            predictions[arm,c] = read_endpoint(batch/arm,10000,c,views[c])[0]

    angles = []
    for pair in (('D2','D8-T'),('D8-C','D8-T')):
        per_arm = {arm:{c:predictions[arm,c] for c in h.SPLITS['held_out']} for arm in pair}
        masks = {c:np.logical_and.reduce([per_arm[arm][c]['orientation_eligible'] for arm in pair])
                 for c in h.SPLITS['held_out']}
        angles.append(dict(arms=list(pair),observations=int(sum(mask.sum() for mask in masks.values())),
            degrees={arm:dict(zip(('median','p95'),np.quantile(np.concatenate(
                [per_arm[arm][c]['orientation_degrees'][mask] for c,mask in masks.items()]),[.5,.95]).tolist()))
                for arm in pair}))

    old_manifest = h.b.read(HET_DONUT/'artifact-manifest.json')
    donor = Path(old_manifest['batch'])
    donor_receipt = h.b.check_artifact(donor/'D6/recovery.json',old_manifest['arms']['D6']['receipt'])
    bell_domain,bell_logs = load_spatial(batch,receipts['D6'],'D6')
    donut_domain,donut_logs = load_spatial(donor,h.b.read(donor_receipt),'D6')
    d2 = h.b.check_artifact(batch/'D2/training.jsonl',receipts['D2']['outputs']['training.jsonl'])
    event = shortest_edge_replay(f,bell_domain,bell_logs)
    camera_rows = []
    for row in a['per_camera_endpoint']:
        for cohort,x in row['metrics'].items():
            if x is not None:
                camera_rows.append((row['arm'],row['camera'],cohort,x['observations'],x['RGB_relative_L2'],
                                    x['Q_relative']['median'],x['Q_relative']['p95'],x['orientation_eligible']))

    output.mkdir()
    shutil.copyfile(analysis_dir/'summary.csv',output/'checkpoint-metrics.csv')
    write_csv(output/'three-benchmark-endpoints.csv',('benchmark','arm',*METRICS),endpoint_comparison(a))
    write_csv(output/'per-camera-metrics.csv',('arm','camera','cohort','observations','RGB_relative_L2',
              'Q_median','Q_p95','orientation_eligible'),camera_rows)
    json_new(output/'endpoint-diagnostics.json',dict(orientation_common_four=[r for r in a['common_orientation']
        if r['step']==10000],held_out_orientation_matched_pairs=angles,boundary_by_arm={
        k:v['boundary_history'] for k,v in a['arms'].items()},
        distance_weighting=dict(bell=spatial_scale(bell_domain),heterogeneous_donut=spatial_scale(donut_domain)),
        D6_shortest_edge_event=event))
    plot_curves(a,output,plt)
    plot_views(batch,views,predictions,output,plt)
    plot_spatial_failure(bell_logs,load_history(d2),donut_logs,event['peak_spatial_loss_step'],output,plt)
    json_new(output/'artifact-manifest.json',dict(schema_version=1,batch=str(batch),execution_commit=approval['commit'],
        dataset=d,authorization=artifact(auth_path),freeze=artifact(batch/'freeze.json'),arms=manifest,
        offline_analysis=[artifact(a_path),artifact(analysis_dir/'summary.csv')],
        historical_inputs=[artifact(p) for p in (DONUT_D6,DONUT_D8,HET_DONUT/'checkpoint-metrics.csv',donor_receipt)],
        packaged_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(output.iterdir())}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--analysis-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    x = p.parse_args()
    package(x.freeze_sha256,x.analysis,x.analysis_sha256,x.output)
