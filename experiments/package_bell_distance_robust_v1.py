"""Package the verified B1/B2 Bell comparison against completed D2 and D6."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_distance_robust_v1 as h
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_heterogeneous_training_v1 import read_endpoint
from experiments.package_bell_heterogeneous_training_v1 import artifact, load_history, plot_views, write_csv
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


ARMS = ('D2','D6',*h.ARMS)
CONTROL = h.ROOT/'docs/results/bell_heterogeneous_training_v1/artifact-manifest.json'
COLORS = dict(D2='C0',D6='C1',B1='C2',B2='C3')


def curves(rows, histories, dest, plt):
    fig, axes = plt.subplots(2,2,figsize=(9.5,6.5),layout='constrained')
    for arm in ARMS:
        held = [r['metrics']['all']['observation_weighted'] for r in rows
                if r['arm']==arm and r['split']=='held_out']
        axes[0,0].semilogy(h.STEPS,[x['RGB_relative_L2'] for x in held],'-o',ms=3,
                           label=arm,color=COLORS[arm])
        axes[0,1].semilogy(h.STEPS,[x['Q_relative']['median'] for x in held],'-o',ms=3,color=COLORS[arm])
        steps = np.arange(1400,2001)
        logs = histories[arm][steps[0]-1:steps[-1]]
        if arm!='D2':
            axes[1,0].semilogy(steps,[r['surface_q_loss'] for r in logs],lw=.8,color=COLORS[arm],label=arm)
        for key,style in (('mx_mean','-'),('my_mean','--')):
            axes[1,1].plot(steps,[r[key] for r in logs],style,lw=.9,color=COLORS[arm])
    axes[0,0].legend(frameon=False,ncol=2)
    axes[1,0].legend(frameon=False)
    for ax,label in zip(axes.flat,('Held-out linear RGB relative L2','Held-out median Q error',
                                 'Weighted spatial loss','Fitting-batch mean chart widths')):
        ax.set(xlabel='Update',ylabel=label)
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.axvline(1696,color='.2',ls=':',lw=.8)
    axes[1,1].set_title('Solid mx  ·  Dashed my',fontsize=9)
    fig.savefig(dest/'training-and-edge-event.png',dpi=165)
    plt.close(fig)


def package(sha, analysis_dir, expected_analysis, dest):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h.freeze(h.ROOT/h.FREEZE,sha)
    batch = Path(f['host_output'])
    auth_path = batch/'authorization.json'
    approval = h.bell.b.read(auth_path)
    analysis_path = analysis_dir/'analysis.json'
    require(digest(analysis_path)==expected_analysis,'Explicit verified B1/B2 analysis required')
    a = h.bell.b.read(analysis_path)
    require(approval==h.authorization(f,approval['commit'],sha) and a['freeze_sha256']==sha
            and a['protocol']==h.PROTOCOL and a['status']=='verified_complete'
            and a['execution_commit']==approval['commit'] and a['authorization_sha256']==digest(auth_path),
            'Verified distance-robust execution differs')
    require(dest.resolve()==dest and not dest.exists() and batch not in dest.parents
            and Path(f['host_reference']) not in dest.parents,'Fresh separate results directory required')
    d = f['dataset']
    _,views = h.dataset(f,Path(f['host_reference']),Path(d['admission']['path']),Path(d['verification']['path']))
    historical = h.bell.b.read(CONTROL)
    old = h.bell.b.read(h.bell.b.check_artifact(Path(historical['offline_analysis'][0]['path']),
                                                historical['offline_analysis'][0]))
    require(old['status']=='verified_complete' and old['protocol']==h.bell.PROTOCOL
            and old['freeze_sha256']==h.BASE_FREEZE_SHA,'Wrong Bell controls')
    rows = [r for x in (old,a) for r in x['results'] if r['arm'] in ARMS]
    endpoints = {(r['arm'],r['split']):r for r in rows if r['step']==10000}
    require(len(rows)==48 and len(endpoints)==8,'Incomplete endpoint panel')
    roots = {arm:(Path(historical['batch']) if arm in ('D2','D6') else batch)/arm for arm in ARMS}
    manifest, hist, pred, features = {}, {}, {}, {}
    gate_anchor = None
    start_fields = {}
    for arm,root in roots.items():
        summary = (old if arm in ('D2','D6') else a)['arms'][arm]
        path = root/'recovery.json'
        require(digest(path)==summary['receipt_sha256'],'Bell arm receipt differs')
        receipt = h.bell.b.read(path)
        out = receipt['outputs']
        identity = lambda name: h.bell.b.check_artifact(root/name,out[name])
        gate = numeric(identity('native-reference-gate.npz'))
        if gate_anchor is None:
            gate_anchor = gate
        else:
            require(gate.keys()==gate_anchor.keys(),'Initial reference gate fields differ')
            for k,value in gate.items():
                np.testing.assert_array_equal(value,gate_anchor[k])
        hist[arm] = load_history(identity('training.jsonl'))
        for c in h.SPLITS['held_out']:
            inherited = numeric(identity(f'step-00000/camera-{c}/materials.npz'))
            for key in ('linear_rgb','Q_world'):
                start = (c,key)
                np.testing.assert_array_equal(inherited[key],start_fields.setdefault(start,inherited[key]))
            camera_root = f'step-10000/camera-{c}'
            for name in (f'{camera_root}/materials.npz',f'{camera_root}/reconstructed.png',
                         f'{camera_root}/ground_truth.png',f'{camera_root}-errors.npz'):
                identity(name)
            pred[arm,c] = read_endpoint(root,10000,c,views[c])[0]

        terminal = receipt['saved_endpoints'][-1]['metrics']
        features[arm] = dict(elapsed_seconds=receipt['elapsed_seconds'],surface_q_timing=receipt.get('surface_q_timing'),
                            illumination=summary['illumination_history'][-1],
                            surface_consistency=terminal['surface_Q_smoothness_diagnostic'],
                            boundary=summary['boundary_history'][-1],shortest_events=summary.get('weighted_shortest_edge_events'),
                            surface_coverage=terminal.get('surface_Q_training_coverage'))
        fields = ('completed_updates','surface_q_loss','preclip_gradient_l2','mx_mean','my_mean','edge_diagnostic')
        capture = lambda row: {k:row[k] for k in fields if k in row}
        features[arm]['update_1696_window'] = [capture(r) for r in hist[arm][1694:1703]]
        if arm in h.ARMS:
            features[arm].update(peak_spatial_loss=capture(max(hist[arm],key=lambda r:r['surface_q_loss'])),
                peak_preclip_gradient=capture(max(hist[arm],key=lambda r:r['preclip_gradient_l2'])))
        manifest[arm] = dict(receipt=artifact(path),train_log=artifact(root/'training.jsonl'),
            outputs=len(out),output_bytes=sum(v['bytes'] for v in out.values()),
            checkpoints={k:v for k,v in out.items() if k.endswith('.pth')})

    retained = ('D2',*h.ARMS)
    subset = {c:np.logical_and.reduce([pred[arm,c]['orientation_eligible'] for arm in retained])
              for c in h.SPLITS['held_out']}
    common = dict(observations=int(sum(x.sum() for x in subset.values())),statistics={
        arm:dict(zip(('median','p95'),np.quantile(np.concatenate(
            [pred[arm,c]['orientation_degrees'][keep] for c,keep in subset.items()]),[.5,.95]).tolist()))
        for arm in retained})
    dest.mkdir()
    shutil.copyfile(analysis_dir/'summary.csv',dest/'checkpoint-metrics.csv')
    metrics = []
    for (arm,split),r in endpoints.items():
        for cohort,methods in r['metrics'].items():
            for weighting,m in methods.items():
                if m is None:
                    continue
                orientation = m['orientation_degrees']
                metrics.append((arm,split,cohort,weighting,m['observations'],m['RGB_relative_L2'],
                    *(m['Q_relative'][k] for k in ('mean','median','p95')),m['Q_at_most_005'],
                    m['major_width']['mean'],m['minor_width']['mean'],m['width_at_floor'],m['width_at_ceiling'],
                    m['gap']['mean'],m['orientation_eligible'],orientation['p95'] if orientation else None,
                    m['color_max']['mean'],m['metallic']['mean']))
    write_csv(dest/'endpoint-cohorts.csv',('arm','split','cohort','weighting','observations','RGB_relative_L2',
        'Q_mean','Q_median','Q_p95','Q_at_most_005','major_width_MAE','minor_width_MAE','minor_at_floor',
        'major_at_ceiling','mean_anisotropy_gap','orientation_eligible','orientation_p95_degrees','albedo_max_MAE',
        'metallic_MAE'),metrics)
    cameras = [(row['arm'],row['camera'],cohort,m['observations'],m['RGB_relative_L2'],m['Q_relative']['median'],
                m['Q_relative']['p95']) for x in (old,a) for row in x['per_camera_endpoint'] if row['arm'] in ARMS
               for cohort,m in row['metrics'].items() if m is not None]
    write_csv(dest/'per-camera-metrics.csv',('arm','camera','cohort','observations','RGB_relative_L2','Q_median','Q_p95'),cameras)
    json_new(dest/'endpoint-diagnostics.json',dict(initial_gate_and_held_out_arrays_match=True,
              held_out_common_orientation=common,arms=features))
    curves(rows,hist,dest,plt)
    plot_views(batch,views,pred,dest,plt,arm_roots=roots)
    json_new(dest/'artifact-manifest.json',dict(schema_version=1,batch=str(batch),execution_commit=approval['commit'],
        dataset=d,authorization=artifact(auth_path),freeze=artifact(batch/'freeze.json'),weights=f['weight_archive'],
        controls=artifact(CONTROL),control_analysis=historical['offline_analysis'][0],arms=manifest,
        offline_analysis=[artifact(analysis_path),artifact(analysis_dir/'summary.csv')],
        packaged_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(dest.iterdir())}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--analysis-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    x = p.parse_args()
    package(x.freeze_sha256,x.analysis,x.analysis_sha256,x.output)
