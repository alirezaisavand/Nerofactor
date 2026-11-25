"""Verify saved Donut histories and score the matched representation comparison."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_representation_gate5_v1 as g
from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_heterogeneous_training_v1 import arm_receipt, read_endpoint
from experiments.analyze_donut_matched_geometry_v1 import ReceiptSpec, cat_endpoints, distribution
from infrastructure.signatures import require

m = g.m


class ChartReceiptSpec(ReceiptSpec):
    PROTOCOL = g.PROTOCOL


def cohorts_and_views(oracle, reference, coverage):
    for camera,v in oracle.items():
        with np.load(reference/f'view-{camera}.npz',allow_pickle=False) as archive:
            rows = np.searchsorted(archive['pixel_indices'],v['pixel_indices'])
            np.testing.assert_array_equal(archive['pixel_indices'][rows],v['pixel_indices'])
            for k in ('near_boundary','nearly_isotropic','highlight_mask'):
                v[k] = archive[k][rows]
    groups = {}
    for camera,v in oracle.items():
        nearby,highlights = metrics.fitting_support(oracle,camera,fitting_cameras=m.SPLITS['fitting'])
        grazing = coverage[camera]['grazing'][coverage[camera]['pool']]
        groups[camera] = dict(all=np.ones(len(nearby),bool), region_0=v['material_region']==0,
            region_1=v['material_region']==1, boundary=v['near_boundary'],
            nonboundary=~v['near_boundary'], smooth_interior=~v['near_boundary'] & ~v['nearly_isotropic'],
            nearly_isotropic=v['nearly_isotropic'], grazing=grazing, nongrazing=~grazing,
            other_fitting_view=nearby>0, no_other_fitting_view=nearby==0,
            nearby_highlight=highlights>0, no_nearby_highlight=highlights==0,
            covered_without_highlight=(nearby>0)&(highlights==0), own_highlight=v['highlight_mask'])
    return groups


def signed_width_summary(delta):
    return {name:dict(mean=float(x.mean()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),
                      p95=float(np.quantile(x,.95))) for name,x in zip(('major','minor'),delta.T)}


def scores(endpoint, widths, groups):
    errors,rgb,truth,_,predicted_widths = endpoint
    result = {}
    for key,keep in groups.items():
        value = metrics.score({k:v[keep] for k,v in errors.items()},rgb[keep],truth[keep],np.ones(int(keep.sum())))
        if value is not None:
            value['signed_width_errors'] = signed_width_summary(predicted_widths[keep]-widths[keep])
        result[key] = value
    return result


def common_orientation(endpoints):
    joint = np.logical_and.reduce([v[0]['orientation_eligible'] for v in endpoints.values()])
    return dict(common_eligible=int(joint.sum()),observations=len(joint),
                by_arm={arm:distribution(value[0]['orientation_degrees'][joint]) for arm,value in endpoints.items()})


def lr_and_history(history):
    summary = []
    for end in m.STEPS[1:]:
        rows = history[end-2000:end]
        summary.append(dict(end=end,
            **{key:distribution(np.asarray([r[key] for r in rows])) for key in
               ('RGB_loss','preclip_gradient_l2')}))
    return summary


def initial_full_parity(roots, oracle):
    result = {}
    for camera in m.PANEL:
        a,b = (numeric(roots[arm]/'step-00000'/f'camera-{camera}'/'materials.npz')
               for arm in (g.AMBIENT,g.ARM))
        np.testing.assert_array_equal(a['hit_mask'],b['hit_mask'])
        mask = a['hit_mask']
        np.testing.assert_array_equal(np.flatnonzero(mask),oracle[camera]['pixel_indices'])
        shifts = {}
        for field,atol in (('kd',0.),('metallic',0.),('Q_tensor_direct',g.PARITY['initial_Q_atol']),
                           ('linear_rgb',g.PARITY['initial_RGB_atol'])):
            np.testing.assert_allclose(a[field][mask],b[field][mask],atol=atol,rtol=0)
            shifts[field] = float(np.max(abs(a[field][mask]-b[field][mask])))
        result[str(camera)] = shifts
    return result


def analyze(output, sha=None):
    f = g.parent()
    p = m.paths(f)
    roots = {}
    _,views,refs,coverage,mesh = m.admit(f,p)
    oracle = views['oracle']
    groups = cohorts_and_views(oracle,p['reference'],coverage)
    adjacency = metrics.mesh_adjacency(mesh['oracle']['triangles'])
    boundary = {camera:metrics.boundary_pairs(v,adjacency) for camera,v in oracle.items()}
    launch = []
    if sha:
        frozen = g.freeze(g.ROOT/g.FREEZE,sha)
        b = Path(frozen['host_output'])
        a = g.read(b/'authorization.json')
        authorization_sha = g.digest(b/'authorization.json')
        require(a==g.authorization(frozen,a['commit'],sha) and g.digest(b/'freeze.json')==sha
                and authorization_sha==g.digest(b.with_name(b.name+'-authorization.json')),
                'Exact Gate 5 launch evidence required')
        chart_support = g.completed(b,frozen,a,sha,authorization_sha)
        launch.append((b,g.ARM,frozen,a,ChartReceiptSpec,chart_support))
    batch,auth,records = g.control_receipts(f)
    launch[0:0] = [(batch,arm,f,auth,ReceiptSpec,records[arm]) for arm in g.CONTROLS]
    require(output.resolve()==output and not output.exists()
            and all(b not in output.parents for b,*_ in launch) and p['reference'] not in output.parents,
            'Fresh offline analysis location required')
    camera = m.PANEL[0]
    first = oracle[camera]
    fine,coarse = first['linear_components'].copy(),first['linear_components'].copy()
    coarse[:,2],fine[:,2] = refs['oracle',camera]
    compat = {camera:dict(first,linear_components=fine,coarse_components=coarse)}
    work, pre = {}, {}
    for b,arm,ff,aa,spec,receipt in launch:
        r,h = arm_receipt(b,arm,ff,aa,compat,spec=spec)
        require(r==receipt,'Receipt changed during verification')
        roots[arm] = b/arm
        pre[arm] = [row['lr'] for row in h]
        work[arm] = dict(receipt_sha256=g.digest(roots[arm]/'recovery.json'),
                         elapsed_seconds=r['elapsed_seconds'],counters=r['counters'],
                         resources=r['resources'],initial=r['initial'],
                         training_windows=lr_and_history(h))
    require(all(x==pre[g.AMBIENT] for x in pre.values()),'Learning-rate streams differ')
    initial = initial_full_parity(roots,oracle) if sha else None
    report = dict(protocol=g.PROTOCOL,status='verified_complete' if sha else 'verified_Gate4_controls_only',
                  support=f['support'],gate4_manifest_sha256=g.GATE4_MANIFEST_SHA,gate5_freeze_sha256=sha,
                  work=work,initial_full_raster_parity=initial,rows=[],
                  offline_renderer_calls=0,offline_checkpoint_loads=0,offline_optimizer_updates=0)
    for step in m.STEPS:
        row = dict(step=step,arms={},comparisons={})
        by_split = {split:{} for split in m.SPLITS}
        for arm,root in roots.items():
            views = {c:read_endpoint(root,step,c,v) for c,v in oracle.items()}
            split_scores = {}
            for split,cams in m.SPLITS.items():
                e = cat_endpoints(views,cams)
                by_split[split][arm] = e
                split_cohorts = {name:np.concatenate([groups[c][name] for c in cams])
                                 for name in groups[cams[0]] if step==10000 or name=='all'}
                w = np.concatenate([oracle[c]['widths_major_minor'] for c in cams])
                split_scores[split] = scores(e,w,split_cohorts)
            split_scores['cameras'] = {str(c):scores(views[c],v['widths_major_minor'],
                groups[c] if step==10000 else {'all':groups[c]['all']}) for c,v in oracle.items()}
            if step==10000:
                split_scores['boundary_by_camera'] = {str(c):metrics.boundary_score(v,boundary[c],views[c][3],views[c][4])
                                                      for c,v in oracle.items()}
            row['arms'][arm] = split_scores
        for split,ends in by_split.items():
            comparison = dict(recipe_ambient_vs_K1=common_orientation({arm:ends[arm] for arm in g.CONTROLS}))
            if sha:
                comparison['matched_chart_vs_ambient'] = common_orientation({arm:ends[arm] for arm in (g.ARM,g.AMBIENT)})
                comparison['all_three'] = common_orientation(ends)
                chart,ambient = [row['arms'][arm][split]['all'] for arm in (g.ARM,g.AMBIENT)]
                comparison['ambient_minus_chart'] = {
                    key:{stat:ambient[key][stat]-chart[key][stat] for stat in ('mean','median','p95')}
                    for key in ('Q_relative','major_width','minor_width','color_max','metallic')}
                for key in ('RGB_relative_L2','RGB_RMSE','orientation_eligible'):
                    comparison['ambient_minus_chart'][key] = ambient[key]-chart[key]
            row['comparisons'][split] = comparison
        report['rows'].append(row)
    output.mkdir()
    json_new(output/'representation-results.json',report)
    plot(report,output)


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = report['rows']
    fig,axes = plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for arm in report['work']:
        for split in m.SPLITS:
            scores = [row['arms'][arm][split]['all'] for row in rows]
            values = ([s['Q_relative']['median'] for s in scores], [s['Q_relative']['p95'] for s in scores],
                      [s['signed_width_errors']['minor']['mean'] for s in scores], [s['RGB_relative_L2'] for s in scores])
            for ax,value,title in zip(axes.flat,values,('Relative Q median','Relative Q p95','Signed minor-width bias',
                                                       'Linear RGB relative L2')):
                color = {'K1-Oracle':'#2866a2',g.AMBIENT:'#ac5023',g.ARM:'#24784d'}[arm]
                ax.plot(m.STEPS,value,'-o' if split=='held_out' else '--s',color=color,
                        markersize=3,label=f'{arm}, {split.replace("_"," ")}')
                ax.set(xlabel='Adam returns',ylabel=title,xticks=m.STEPS)
                ax.grid(alpha=.2)
    axes[1,0].axhline(0,color='.55',lw=.7)
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.savefig(output/'oracle-trajectories.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze-sha256',help='Verify and include the new chart execution only when specified')
    args = parser.parse_args()
    analyze(args.output,args.freeze_sha256)
