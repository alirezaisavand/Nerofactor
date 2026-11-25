"""Package verified Gate 5 tables and render paper figures from saved endpoints."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_representation_gate5_v1 as g
from benchmarks.donut_recovery_v1 import json_new, numeric
from infrastructure.signatures import require


ARMS = ('Final-Chart-Oracle', 'Ambient-Oracle', 'K1-Oracle')
LABELS = ('Final-Chart', 'Final-Ambient', 'K1')
COLORS = ('#0072B2', '#D55E00', '#555555')
STYLES = ('--', '-', ':')
HELD = 'held_out'


def identity(path, *, name=None):
    result = dict(bytes=path.stat().st_size, sha256=g.digest(path))
    if name is not None:
        result['path'] = str(name)
    return result


def endpoint_distributions(roots, receipts, reference):
    """Read both held-out cameras and keep the frozen native eligible masks."""
    generation = reference/'generation.json'
    require(g.digest(generation)==g.m.GENERATION, 'Oracle generation identity differs')
    admitted = g.read(generation)
    width, angles, eligible = ({arm:[] for arm in ARMS} for _ in range(3))
    for camera in g.m.SPLITS[HELD]:
        name = f'view-{camera}.npz'
        oracle = numeric(g.check(reference/name,admitted['outputs'][name]))
        masks = []
        for arm, root in roots.items():
            files = receipts[arm]['outputs']
            material = f'step-10000/camera-{camera}/materials.npz'
            errs = f'step-10000/camera-{camera}-errors.npz'
            a,e = (numeric(g.check(root/name,files[name])) for name in (material,errs))
            mask = a['hit_mask']
            pixels = np.flatnonzero(mask)
            indices = np.searchsorted(oracle['pixel_indices'],pixels)
            np.testing.assert_array_equal(oracle['pixel_indices'][indices],pixels)
            require(len(pixels)==g.m.COUNTS[camera] and oracle['shader_valid'][indices].all(),
                    'Held-out archive support differs')
            width[arm].append(np.minimum(a['mx'][mask,0],a['my'][mask,0])-oracle['widths_major_minor'][indices,1])
            angles[arm].append(e['orientation_degrees'])
            eligible[arm].append(e['orientation_eligible'])
            masks.append(mask)
        require(all(np.array_equal(masks[0],mask) for mask in masks[1:]), 'Held-out pixel masks differ')
    joint = np.logical_and.reduce([np.concatenate(eligible[arm]) for arm in ARMS])
    return {arm:np.concatenate(width[arm]) for arm in ARMS}, {
        arm:np.concatenate(angles[arm])[joint] for arm in ARMS}


def tables(data, out):
    rows = []
    for state in data['rows']:
        for split in ('fitting',HELD):
            joint = state['comparisons'][split]['all_three']
            for arm in ARMS:
                a = state['arms'][arm][split]['all']
                orientation = joint['by_arm'][arm]
                rows.append(dict(step=state['step'],split=split,arm=arm,observations=a['observations'],
                    Q_median=a['Q_relative']['median'],Q_p95=a['Q_relative']['p95'],
                    Q_at_most_005=a['Q_at_most_005'],major_MAE=a['major_width']['mean'],minor_MAE=a['minor_width']['mean'],
                    major_bias=a['signed_width_errors']['major']['mean'],minor_bias=a['signed_width_errors']['minor']['mean'],
                    minor_signed_error_p95=a['signed_width_errors']['minor']['p95'],
                    orientation_eligible=a['orientation_eligible'],orientation_common_eligible=joint['common_eligible'],
                    orientation_common_median=orientation['median'] if orientation else None,
                    orientation_common_p95=orientation['p95'] if orientation else None,
                    albedo_max_channel_MAE=a['color_max']['mean'],metallic_MAE=a['metallic']['mean'],
                    linear_RGB_L2=a['RGB_relative_L2'],linear_RGB_RMSE=a['RGB_RMSE']))
    with (out/'trajectory-endpoints.csv').open('x',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure(end, width, angles, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig,axes = plt.subplots(2,2,figsize=(9.3,6.4),layout='constrained')
    axes = axes.ravel()
    groups = ['no_other_fitting_view','covered_without_highlight','nearby_highlight']
    positions = np.arange(len(groups))
    for i,(arm,name,color,line) in enumerate(zip(ARMS,LABELS,COLORS,STYLES)):
        a = end['arms'][arm]
        signed = width[arm]
        angle = angles[arm]
        aggregate = a[HELD]['all']
        joint = end['comparisons'][HELD]['all_three']
        require(len(signed)==aggregate['observations'] and len(angle)==joint['common_eligible'],
                'Plot support differs from verified statistics')
        np.testing.assert_allclose(np.quantile(signed,.95),aggregate['signed_width_errors']['minor']['p95'])
        np.testing.assert_allclose(np.quantile(angle,.95),joint['by_arm'][arm]['p95'])
        for ax,values in zip(axes[:2],(signed,angle)):
            ax.step(np.sort(values),np.arange(1,len(values)+1)/len(values),where='post',
                    color=color,linestyle=line,lw=1.7,label=name)
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.set_ylim(0,1.02)
        axes[2].bar(positions+(i-1)*.23,[a[HELD][k]['minor_width']['mean'] for k in groups],
                    width=.22,color=color)
        pairs = [a['boundary_by_camera'][str(c)] for c in g.m.SPLITS[HELD]]
        jumps = [p['Q_jump_projection_median'] for p in pairs]
        axes[3].bar(np.arange(len(pairs))+(i-1)*.23,jumps,width=.22,color=color)
    axes[0].axvline(0,lw=.8,color='.65')
    axes[0].set(xlabel='Predicted minus oracle minor width',ylabel='Cumulative observations')
    axes[1].set(xlabel='Principal-axis error (degrees)',ylabel='Cumulative eligible observations',xlim=(0,90))
    axes[1].set_xticks([0,15,30,45,60,75,90])
    axes[2].set_xticks(positions,['No nearby\nfitting view','Nearby view,\nno highlight','Nearby fitting\nhighlight'])
    axes[2].set(ylabel='Minor-width MAE')
    axes[3].set_xticks([0,1],['Camera 28\n17 pairs','Camera 34\n5 pairs'])
    axes[3].set(ylabel='Median projected Q jump / oracle',ylim=(0,1.05))
    axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[3].axhline(1,lw=.8,color='.5',linestyle='--')
    axes[0].legend(loc='lower right',frameon=False,fontsize=9)
    for ax,title in zip(axes,('A  Signed minor-width error','B  Orientation error',
                              'C  Error by fitting coverage','D  Boundary recovery')):
        ax.set_title(title,loc='left',weight='semibold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.22)
        ax.set_axisbelow(True)
    for extension in ('pdf','png'):
        fig.savefig(out/f'held-out-physical-comparison.{extension}',dpi=240)
    plt.close(fig)


def package(freeze_sha, analysis, out):
    frozen = g.freeze(g.ROOT/g.FREEZE,freeze_sha)
    batch = Path(frozen['host_output'])
    run_auth = batch/'authorization.json'
    auth = g.read(run_auth)
    auth_sha = g.digest(run_auth)
    require(auth==g.authorization(frozen,auth['commit'],freeze_sha)
            and g.digest(batch/'freeze.json')==freeze_sha
            and g.digest(batch.with_name(batch.name+'-authorization.json'))==auth_sha,
            'Gate 5 launch identity differs')
    chart = g.completed(batch,frozen,auth,freeze_sha,auth_sha)
    parent = g.parent()
    past,_,archive = g.control_receipts(parent)
    receipts = {g.ARM:chart,**archive}
    roots = {arm:(batch if arm==g.ARM else past)/arm for arm in ARMS}
    report = g.read(analysis/'representation-results.json')
    require(report['status']=='verified_complete' and report['gate5_freeze_sha256']==freeze_sha
            and report['support']==frozen['support'] and report['rows'][-1]['step']==10000
            and all(report['work'][arm]['receipt_sha256']==g.digest(root/'recovery.json')
                    for arm,root in roots.items()), 'Verified offline results required')
    width,angles = endpoint_distributions(roots,receipts,g.m.paths(parent)['reference'])
    require(out.resolve()==out and not out.exists() and batch not in out.parents and analysis not in out.parents,
            'Fresh publication package required')
    out.mkdir()
    for name in ('representation-results.json','oracle-trajectories.png'):
        shutil.copyfile(analysis/name,out/name)
        require(identity(out/name)==identity(analysis/name),'Packaged copy differs')
    tables(report,out)
    figure(report['rows'][-1],width,angles,out)
    chart_root = roots[g.ARM]
    records = {}
    for arm,root in roots.items():
        receipt = receipts[arm]
        records[arm] = dict(receipt=identity(root/'recovery.json',name=root/'recovery.json'),
            verified_native_artifacts=len(receipt['outputs']),counters=receipt['counters'],
            optimizer_returns=receipt['optimizer_returns'],cleanup_verified=receipt['cleanup_verified'],
            exit_code=identity(root.parent/f'{arm}-exit-code.txt',name=root.parent/f'{arm}-exit-code.txt'),
            fitting_stream=receipt['outputs'][g.m.bell.dense.STREAM],
            checkpoints={name:item for name,item in receipt['outputs'].items() if name.endswith('.pth')})
    json_new(out/'artifact-manifest.json',dict(schema_version=1,protocol=g.PROTOCOL,status='verified_complete',
        commit=auth['commit'],image=auth['image_id'],corrected_backend=chart['backend'],support=frozen['support'],
        execution_freeze=identity(batch/'freeze.json',name=batch/'freeze.json'),
        authorization=identity(run_auth,name=run_auth), arms=records,
        gate4_manifest=identity(g.GATE4_MANIFEST,name=g.GATE4_MANIFEST.relative_to(g.ROOT)),
        native_gate_artifacts={name:identity(chart_root/name,name=chart_root/name) for name in g.CAPTURES.values()},
        evidence={name:identity(analysis/name,name=analysis/name)
                  for name in ('representation-results.json','oracle-trajectories.png')},
        packaged_files={p.name:identity(p) for p in sorted(out.iterdir())},
        offline_checkpoint_loads=0,offline_renderer_calls=0,offline_optimizer_updates=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    package(args.freeze_sha256,args.analysis,args.output)
