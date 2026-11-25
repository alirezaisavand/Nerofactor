"""Verify Gate 4 receipts and score matched geometry effects from saved arrays."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_matched_geometry_recovery_v1 as m
from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from benchmarks.bell_known_l2_v1 import verify_illumination_receipt
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_heterogeneous_training_v1 import arm_receipt, read_endpoint
from infrastructure.signatures import require


class ReceiptSpec:
    PROTOCOL, GENERATION, ADMISSION = m.PROTOCOL,m.GENERATION,m.ADMISSION
    verify_illumination_receipt = staticmethod(verify_illumination_receipt)


def rgb_l2(pred, truth):
    return float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-15))


def distribution(a):
    if not len(a):
        return None
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))


def summarize(prediction, target, selected):
    err, pred, rgb, _, widths = prediction
    fields = {key:value[selected] for key,value in err.items()}
    score = metrics.score(fields,pred[selected],rgb[selected],np.ones(int(selected.sum())))
    if score:
        score['signed_width_bias'] = (widths[selected]-target[selected]).mean(0).tolist()
    return score


def cat_endpoints(by_camera, cams):
    group = [by_camera[c] for c in cams]
    fields = {k:np.concatenate([v[0][k] for v in group]) for k in group[0][0]}
    return (fields, *(np.concatenate([v[i] for v in group]) for i in range(1,5)))


def endpoint_rows(batch, step, views, refs, coverage, adjacency):
    values, by_arm, prediction = {}, {}, {}
    fixed = {}
    for split,cameras in m.SPLITS.items():
        o,r = (np.concatenate([refs[g,c][1] for c in cameras]) for g in m.rgb.ARMS)
        fixed[split] = rgb_l2(r,o)
    for arm in m.ARMS:
        kind = m.geometry_name(arm)
        by_camera = {c:read_endpoint(batch/arm,step,c,views[kind][c]) for c in m.PANEL}
        prediction[arm] = by_camera
        for split,cams in m.SPLITS.items():
            entries = cat_endpoints(by_camera,cams)
            widths = np.concatenate([views[kind][c]['widths_major_minor'] for c in cams])
            ix = np.ones(len(widths),bool)
            by_arm[arm,split] = entries
            cohorts = {'all':ix}
            if step==10000:
                for name in ('boundary','grazing','degenerate_adjacent','depth_outlier'):
                    cohorts[name] = np.concatenate([coverage[c][name][coverage[c]['pool']] for c in cams])
            values.setdefault(arm,{})[split] = {name:summarize(entries,widths,mask) for name,mask in cohorts.items()}
            values[arm][split]['fixed_material_Gate3_RGB_L2'] = 0. if kind=='oracle' else fixed[split]
        if step in (0,10000):
            by_view = {}
            for c, entry in by_camera.items():
                target = views[kind][c]
                by_view[str(c)] = summarize(entry,target['widths_major_minor'],target['shader_valid'])
            values[arm]['cameras'] = by_view

    comparisons = {}
    for oracle,recon in zip(m.ARMS[::2],m.ARMS[1::2]):
        pair = {}
        for split,cams in m.SPLITS.items():
            a,b = by_arm[oracle,split],by_arm[recon,split]
            keep = a[0]['orientation_eligible'] & b[0]['orientation_eligible']
            ma,mb = (values[arm][split]['all'] for arm in (oracle,recon))
            delta = {key:mb[key]-ma[key] for key in ('RGB_relative_L2','RGB_RMSE','orientation_eligible')}
            for key in ('Q_relative','major_width','minor_width','color_max','metallic'):
                delta[key] = {stat:mb[key][stat]-ma[key][stat] for stat in ('mean','median','p95')}
            delta['signed_width_bias'] = (np.asarray(mb['signed_width_bias'])-ma['signed_width_bias']).tolist()
            row = dict(reconstruction_minus_oracle=delta, common_orientation_count=int(keep.sum()),
                       common_orientation={arm:distribution(v[0]['orientation_degrees'][keep]) for arm,v in ((oracle,a),(recon,b))},
                       reconstruction_RGB_minus_fixed=float(mb['RGB_relative_L2']-fixed[split]),
                       lower_RGB_than_oracle_fit_with_worse_Q_median=delta['RGB_relative_L2']<0<delta['Q_relative']['median'],
                       lower_RGB_than_fixed_material_with_worse_Q_median=mb['RGB_relative_L2']<fixed[split] and delta['Q_relative']['median']>0)
            if step in (0,10000):
                shifts, versus, boundaries, material_shifts = [], [], {}, []
                for camera in cams:
                    vo,vr = (views[g][camera] for g in m.rgb.ARMS)
                    pa,pb = prediction[oracle][camera],prediction[recon][camera]
                    aligned = m.common_frame(vo,vr,pb[3])
                    norm = np.linalg.norm(vo['Q_world'],axis=(1,2))
                    shifts.append(np.linalg.norm(aligned-pa[3],axis=(1,2))/norm)
                    versus.append(np.linalg.norm(aligned-vo['Q_world'],axis=(1,2))/norm)
                    if step==0:
                        arrays = [numeric(batch/arm/'step-00000'/f'camera-{camera}'/'materials.npz') for arm in (oracle,recon)]
                        fields = [{key:a[key][a['hit_mask']] for key in ('kd','metallic')} for a in arrays]
                        material_shifts.append({key:abs(fields[0][key]-fields[1][key]).reshape(len(norm),-1).max(1)
                                                for key in ('kd','metallic')})
                    pairs = metrics.boundary_pairs(vo,adjacency)
                    mapped = dict(vo,Q_world=m.common_frame(vo,vr,vr['Q_world']),widths_major_minor=vr['widths_major_minor'])
                    boundaries[str(camera)] = {oracle:metrics.boundary_score(vo,pairs,pa[3],pa[4]),
                                                recon:metrics.boundary_score(mapped,pairs,aligned,pb[4])}
                row.update(common_frame_relative_Q_shift=distribution(np.concatenate(shifts)),
                           reconstructed_Q_relative_to_same_ray_oracle_truth=distribution(np.concatenate(versus)),
                           boundary_by_camera=boundaries)
                if material_shifts:
                    row['initial_scalar_shifts'] = {k:distribution(np.concatenate([a[k] for a in material_shifts])) for k in ('kd','metallic')}
            pair[split] = row
        comparisons[oracle.split('-')[0]] = pair
    return dict(step=step, arms=values, geometry_pairs=comparisons)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.4,6.1), layout='constrained')
    for arm in m.ARMS:
        scores = [r['arms'][arm]['held_out']['all'] for r in rows]
        groups = ([s['Q_relative']['median'] for s in scores], [s['Q_relative']['p95'] for s in scores],
                  [s['RGB_relative_L2'] for s in scores], [s['minor_width']['mean'] for s in scores])
        for ax,items,title in zip(axes.flat,groups,('Relative Q median','Relative Q p95','Linear RGB relative L2','Minor width MAE')):
            ax.plot(m.STEPS,items,'-o' if arm.endswith('Oracle') else '--s',markersize=3,label=arm,
                    color='#2b699b' if arm.startswith('K1') else '#ae5e2d')
            ax.set(xlabel='Adam updates',ylabel=title,xticks=m.STEPS)
            ax.grid(alpha=.22)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(output/'held-out-geometry-trajectories.png',dpi=160)
    plt.close(fig)


def verify(sha, output):
    f = m.freeze(m.ROOT/m.FREEZE,sha)
    batch = Path(f['host_output'])
    authfile = batch/'authorization.json'
    auth = m.rgb.geo.h.read(authfile)
    auth_sha = m.digest(authfile)
    require(output.resolve()==output and not output.exists() and batch not in output.parents
            and auth==m.authorization(f,auth['commit'],sha) and m.digest(batch/'freeze.json')==sha
            and auth_sha==m.digest(batch.with_name(batch.name+'-authorization.json')), 'Foreign or incomplete matched batch')
    _, views, refs, coverage, meshes = m.admit(f,m.paths(f))
    records, orders, initial = {}, {}, {}
    for arm in m.ARMS:
        r = m.completed(batch,arm,f,auth['commit'],sha,auth_sha)
        kind = m.geometry_name(arm)
        first = views[kind][m.PANEL[0]]
        pair = refs[kind,m.PANEL[0]]
        # Shared receipt verifier expects its input views to carry the
        # compatibility RGB arrays. Only the first view is read for this gate.
        fine,coarse = first['linear_components'].copy(),first['linear_components'].copy()
        fine[:,2],coarse[:,2] = pair[1],pair[0]
        compat = {m.PANEL[0]:dict(first,linear_components=fine,coarse_components=coarse)}
        checked,hist = arm_receipt(batch,arm,f,auth,compat,spec=ReceiptSpec)
        require(checked==r,'Native receipt changed during verification')
        initial[arm] = r['initial']
        orders[arm] = [row['lr'] for row in hist]
        records[arm] = dict(receipt_sha256=m.digest(batch/arm/'recovery.json'), output_count=len(r['outputs']),
                            initial=r['initial'], counters=r['counters'], elapsed_seconds=r['elapsed_seconds'], resources=r['resources'])
    require(orders[m.ARMS[0]]==orders[m.ARMS[1]]==orders[m.ARMS[2]]==orders[m.ARMS[3]]
            and all(initial[a]==initial[b] for a,b in zip(m.ARMS[::2],m.ARMS[1::2])), 'Initial state or schedule mismatch')
    adjacency = metrics.mesh_adjacency(meshes['oracle']['triangles'])
    rows = [endpoint_rows(batch,step,views,refs,coverage,adjacency) for step in m.STEPS]
    output.mkdir()
    from experiments.donut_reconstructed_geometry_v1 import save
    save(output/'coverage.npz',{f'{camera}_{key}':mask for camera,fields in coverage.items() for key,mask in fields.items()})
    coverage_counts = {str(c):{key:int(mask.sum()) for key,mask in v.items()} for c,v in coverage.items()}
    json_new(output/'matched-results.json',dict(protocol=m.PROTOCOL,status='verified_complete',freeze_sha256=sha,
             execution_commit=auth['commit'],authorization_sha256=auth_sha,rows=rows,receipts=records,
             coverage=coverage_counts,support=f['support'],historical_geometry_provenance='unresolved',
             offline_renderer_calls=0,offline_checkpoint_loads=0,offline_optimizer_updates=0))
    plot(rows,output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    verify(args.freeze_sha256,args.output)
