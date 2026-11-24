"""Package the completed Dense Bell comparison from verified saved evidence."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_dense_known_l2_v1 as d
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_known_l2_v1 import endpoint
from experiments.package_bell_known_l2_v1 import checkpoint_rows
from experiments.package_bell_heterogeneous_training_v1 import artifact, write_csv
from infrastructure.signatures import require

b = d.b
ARMS = tuple(arm for dense,sparse in d.MATCH.items() for arm in (sparse,dense))


def curves(rows, output, plt):
    fields = [('RGB_relative_L2',None,'Linear RGB relative L2'),
              ('Q_relative','median','Relative world-Q error (median)'),
              ('Q_relative','p95','Relative world-Q error (p95)'),
              ('minor_width','mean','Minor-width MAE')]
    fig,axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for index,arm in enumerate(ARMS):
        held = sorted((r for r in rows if r['arm']==arm and r['split']=='held_out'),key=lambda r:r['step'])
        require([r['step'] for r in held]==d.STEPS,'Incomplete held-out recovery trajectory')
        for ax,(field,stat,label) in zip(axes.flat,fields):
            values = [r['metrics']['all']['observation_weighted'][field] for r in held]
            ax.semilogy(d.STEPS, [v[stat] for v in values] if stat else values,
                '-o' if arm in d.ARMS else '--s',color=f'C{index//2}',markersize=3,linewidth=1.2,label=arm)
            ax.set(xlabel='Adam updates',ylabel=label,xticks=d.STEPS)
            ax.grid(alpha=.23)
    axes[0,0].legend(frameon=False,ncol=2,fontsize=8)
    fig.savefig(output/'recovery-curves.png',dpi=165)
    plt.close(fig)


def held_views(views, roots, receipts, output, plt):
    from PIL import Image

    fig,axes = plt.subplots(8,7,figsize=(14.5,13.0),layout='constrained')
    for index,camera in enumerate(d.SPLITS['held_out']):
        view = views[camera]
        mask = view['shader_valid']
        pixels = view['pixel_indices'][mask]
        row = 4*index
        for col,arm in enumerate(('Oracle',*ARMS)):
            source = arm if col else 'K1'
            file = 'reconstructed.png' if col else 'ground_truth.png'
            key = f'step-10000/camera-{camera}/{file}'
            with Image.open(b.check_artifact(roots[source]/key,receipts[source]['outputs'][key])) as pic:
                axes[row,col].imshow(np.asarray(pic),interpolation='nearest')
            axes[row,col].set_title(arm,fontsize=10)
            if col:
                errors,_,_,_,widths = endpoint(roots[arm],receipts[arm],camera,view)
                q = errors['Q_relative']
            else:
                widths,q = view['widths_major_minor'][mask],np.zeros(len(pixels))
            layers = ((widths[:,0],(.10,.50),'viridis'),(widths[:,1],(.04,.24),'viridis'),(q,(0,1),'magma'))
            for offset,(values,limits,palette) in enumerate(layers,1):
                raster = np.full(view['hit_mask'].shape,np.nan)
                raster.flat[pixels] = values
                layer = axes[row+offset,col].imshow(raster,vmin=limits[0],vmax=limits[1],cmap=palette,
                                                    interpolation='nearest')
                if col==len(ARMS):
                    fig.colorbar(layer,ax=axes[row+offset],fraction=.013,pad=.012,extend='both')
        for offset,label in enumerate((f'Camera {camera} RGB','Major width','Minor width','Relative Q error')):
            axes[row+offset,0].set_ylabel(label)
    for ax in axes.flat:
        ax.set(xticks=[],yticks=[])
    fig.savefig(output/'held-out-views.png',dpi=165)
    plt.close(fig)


def matched_tables(matched, output):
    rows,boundaries = [],[]
    for arm,pair in matched['arms'].items():
        for split,methods in pair['groups'].items():
            for method,cohorts in methods.items():
                for cohort,values in cohorts.items():
                    for weighting in ('observation','area'):
                        score = values[weighting]
                        if score is None:
                            continue
                        angle = score['orientation_degrees']
                        bias = values['signed_major_minor_bias'] if weighting=='observation' else (None,None)
                        rows.append((arm,method,split,cohort,weighting,score['observations'],score['RGB_relative_L2'],
                                     score['Q_relative']['median'],score['Q_relative']['p95'],score['Q_at_most_005'],
                                     score['major_width']['mean'],score['minor_width']['mean'],*bias,
                                     score['orientation_eligible'],angle['median'] if angle else None,
                                     angle['p95'] if angle else None,score['width_at_floor'],score['width_at_ceiling']))
        for condition,x in pair['boundary'].items():
            method = arm if condition=='dense' else pair['sparse_control']
            boundaries.append((method,x['pairs'],x['Q_jump_projection_median'],x['absolute_Q_jump_error_mean'],
                               *x['predicted_signed_width_jump'],*x['reference_signed_width_jump'],*x['signed_width_jump_MAE']))
    write_csv(output/'matched-endpoints.csv',('pair','arm','split','cohort','weighting','observations',
        'linear_RGB_rel_L2','Q_median','Q_p95','Q_at_most_005','major_MAE','minor_MAE','major_bias','minor_bias',
        'orientation_eligible','angle_median_deg','angle_p95_deg','minor_at_floor','major_at_ceiling'),rows)
    write_csv(output/'boundary-jumps.csv',('arm','pairs','median_projected_Q_jump_fraction','absolute_Q_jump_MAE',
        'predicted_major_jump','predicted_minor_jump','oracle_major_jump','oracle_minor_jump',
        'major_jump_MAE','minor_jump_MAE'),boundaries)


def package(freeze_sha, analysis, analysis_sha, matched_sha, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = d.freeze(d.ROOT/d.FREEZE,freeze_sha)
    d.verify_prior_launch(f)
    d.verify_controls(f)
    batch = Path(f['host_output'])
    require(output.resolve()==output and not output.exists()
            and all(p not in output.parents for p in (analysis,batch,Path(f['host_reference']))),
            'Expected a fresh local results path')
    current,comparison = analysis/'analysis.json',analysis/'dense-vs-sparse.json'
    require(b.digest(current)==analysis_sha and b.digest(comparison)==matched_sha,'Verified offline reports differ')
    full,matched = b.read(current),b.read(comparison)
    sparse = d.read_artifact(f['sparse_analysis'])
    auth = b.read(batch/'authorization.json')
    require(auth==d.authorization(f,full['execution_commit'],freeze_sha)
            and full['freeze_sha256']==b.digest(batch/'freeze.json')==freeze_sha and full['status']=='verified_complete'
            and full['authorization_sha256']==b.digest(batch/'authorization.json')
            and set(full['arms'])==set(matched['arms'])==set(d.ARMS)
            and matched['photometric_stream_sha256']==f['photometric_index_sha256'],
            'Foreign or incomplete Dense Bell results')
    inputs = f['dataset']
    _,views = d.dataset(f,Path(f['host_reference']),Path(inputs['admission']['path']),Path(inputs['verification']['path']))
    roots = dict({arm:batch/arm for arm in d.ARMS},**{control:Path(f['controls'][arm]['path']).parent
                 for arm,control in d.MATCH.items()})
    rows = [row for report in (sparse,full) for row in report['results'] if row['arm'] in ARMS]
    require(len(rows)==72,'Incomplete recovery checkpoint metric panel')
    records,receipts = {},{}
    for arm,root in roots.items():
        p = root/'recovery.json'
        scored = (full if arm in d.ARMS else sparse)['arms'][arm]
        require(b.digest(p)==scored['receipt_sha256'],'Frozen arm receipt differs: '+arm)
        receipt = receipts[arm] = b.read(p)
        files = receipt['outputs']
        records[arm] = dict(receipt=artifact(p),exit_code=artifact(root.parent/(arm+'-exit-code.txt')),
            output_count=len(files),output_bytes=sum(v['bytes'] for v in files.values()),
            checkpoints={name:value for name,value in files.items() if name.endswith('.pth')})
        if arm in d.ARMS:
            stream = b.check_artifact(root/d.STREAM,files[d.STREAM])
            require(files[d.STREAM]['sha256']==f['photometric_index_sha256'],'Photometric stream differs')
            records[arm]['photometric_stream'] = artifact(stream)
    n = f['training_projection']['fitting_observations']
    x = np.fromfile(roots['K1-Dense']/d.STREAM,dtype=d.INDEX_DTYPE)
    require(x.size==f['training_projection']['training_slots'] and int(x.max())<n,'Photometric trace values differ')
    exposure = np.bincount(x,minlength=n)
    edges = np.cumsum([0,*[d.dense.COUNTS[c] for c in d.SPLITS['fitting']]])
    per_camera = {str(c):int(exposure[lo:hi].sum()) for c,lo,hi in zip(d.SPLITS['fitting'],edges,edges[1:])}
    output.mkdir()
    checkpoint_rows(rows,output)
    matched_tables(matched,output)
    cameras = [(row['arm'],row['camera'],row['metrics']['all']['RGB_relative_L2'],
                row['metrics']['all']['Q_relative']['median'],row['metrics']['all']['Q_relative']['p95'])
               for report in (sparse,full) for row in report['per_camera_endpoint']
               if row['arm'] in ARMS and row['camera'] in d.SPLITS['held_out']]
    write_csv(output/'held-out-cameras.csv',('arm','camera','RGB_relative_L2','Q_median','Q_p95'),cameras)
    json_new(output/'endpoint-diagnostics.json',dict(Q_median_K3_minus_K1=matched['Q_median_K3_minus_K1'],
        common_orientation=matched['common_orientation'],sparse_dense_orientation={
            arm:z['common_orientation'] for arm,z in matched['arms'].items()},
        fitting_exposure=dict(matched['exposure'],minimum=int(exposure.min()),maximum=int(exposure.max()),
            p05_median_p95=np.quantile(exposure,[.05,.5,.95]).tolist(),dense_camera_draws=per_camera),
        elapsed_seconds={arm:receipt['elapsed_seconds'] for arm,receipt in receipts.items()}))
    curves(rows,output,plt)
    held_views(views,roots,receipts,output,plt)
    json_new(output/'artifact-manifest.json',dict(schema_version=1,batch=str(batch),
        execution_commit=full['execution_commit'],image=f['image'],freeze=artifact(batch/'freeze.json'),
        authorization=artifact(batch/'authorization.json'),dataset=inputs,coverage=f['held_out_support'],
        previous_failed_launch=f['failed_launch'],arms=records,
        offline_analysis=[artifact(p) for p in (current,comparison,analysis/'summary.csv')],sparse_analysis=f['sparse_analysis'],
        packaged_files={p.name:dict(bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(output.iterdir())}))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--analysis',required=True,type=Path)
    cli.add_argument('--analysis-sha256',required=True)
    cli.add_argument('--matched-sha256',required=True)
    cli.add_argument('--output',required=True,type=Path)
    args = cli.parse_args()
    package(args.freeze_sha256,args.analysis,args.analysis_sha256,args.matched_sha256,args.output)
