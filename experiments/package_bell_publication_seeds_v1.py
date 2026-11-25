"""Combine verified Bell seed replication with the accepted 6033 results."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_publication_seeds_v1 as s
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_publication_seeds_v1 import analyze_saved, summarize
from experiments.package_bell_heterogeneous_training_v1 import artifact, write_csv
from infrastructure.signatures import require


b = s.b
BASE = s.ROOT/'docs/results/bell_final_tensor_pair_v1'
SEEDS = (6033, *s.SEEDS)
FIELDS = ('Q_median', 'Q_p95', 'major_MAE', 'minor_MAE', 'signed_major_bias', 'signed_minor_bias',
          'RGB_L2', 'eligibility', 'common_orientation_median', 'common_orientation_p95',
          'Q_jump', 'major_jump', 'minor_jump', 'seconds')


def read_csv(path):
    with path.open(newline='') as handle:
        return {r['arm']: r for r in csv.DictReader(handle)}


def accepted_6033():
    record = b.read(BASE/'artifact-manifest.json')
    require(record['freeze']['sha256'] == s.PARENT_SHA and set(record['arms']) == set(s.METHODS),
            'Incorrect original seed provenance')
    names = ('held-out-endpoints.csv', 'boundary-jumps.csv', 'optimization-diagnostics.json', 'checkpoint-metrics.csv')
    for name in names:
        b.check_artifact(BASE/name, record['packaged_files'][name])
    rows, jumps, diag = read_csv(BASE/names[0]), read_csv(BASE/names[1]), b.read(BASE/names[2])
    common = diag['common_orientation']['10000']['all']
    rename = {'major_MAE': 'major_width_MAE', 'minor_MAE': 'minor_width_MAE',
              'signed_major_bias': 'major_width_bias', 'signed_minor_bias': 'minor_width_bias',
              'RGB_L2': 'RGB_relative_L2', 'eligibility': 'orientation_eligible', 'seconds': 'elapsed_seconds'}
    endpoints = {}
    for method in s.METHODS:
        orientation = common['angle_median_p95'][method]
        endpoints[method] = {key: float(rows[method][rename.get(key, key)]) for key in FIELDS[:8]}
        endpoints[method].update(eligibility=int(rows[method]['orientation_eligible']),
            common_orientation_median=orientation[0], common_orientation_p95=orientation[1],
            Q_jump=float(jumps[method]['median_projected_Q_jump_fraction']),
            major_jump=float(jumps[method]['predicted_major_jump']),
            minor_jump=float(jumps[method]['predicted_minor_jump']), seconds=float(rows[method]['elapsed_seconds']))
    return endpoints, common, record


def stability(receipt, run, score):
    p = b.check_artifact(run/'training.jsonl', receipt['outputs']['training.jsonl'])
    with p.open() as handle:
        norms = np.array([json.loads(line)['preclip_gradient_l2'] for line in handle])
    require(norms.shape == (10000,) and np.isfinite(norms).all() and (norms >= 0).all(),
            'Incomplete or nonfinite gradient log')
    return dict(max_preclip_L2=float(norms.max()), updates_above_clip=int((norms > receipt['policy']['gradient_clip']).sum()),
                held_out_width_at_floor=score['width_at_floor'], held_out_width_at_ceiling=score['width_at_ceiling'])


def plot(endpoints, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {'K1-Dense': ('o', '0.26'), 'Final-Chart': ('s', '#b36125'), 'Final-Ambient': ('D', '#275e9e')}
    fields = (('Q_median', 'Relative world-Q error · median'), ('Q_p95', 'Relative world-Q error · p95'),
              ('major_MAE', 'Major width · MAE'), ('minor_MAE', 'Minor width · MAE'),
              ('common_orientation_p95', 'Orientation error · common subset p95 (°)'),
              ('RGB_L2', 'Held-out linear RGB · relative L2'))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), layout='constrained')
    for ax, (key, label) in zip(axes.flat, fields):
        for method, (marker, color) in style.items():
            ax.plot(SEEDS, [endpoints[str(seed)][method][key] for seed in SEEDS], marker=marker,
                    linestyle='-', label=method, color=color, markersize=5)
        ax.set(xlabel='Seed', ylabel=label, xticks=SEEDS)
        ax.set_xticklabels([str(seed) for seed in SEEDS])
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(output/'held-out-across-seeds.png', dpi=160)
    plt.close(fig)


def package(freeze_sha, analysis, output):
    f = s.freeze(s.ROOT/s.FREEZE, freeze_sha)
    batch = Path(f['host_output'])
    require(output.resolve() == output and not output.exists() and output.parent.is_dir()
            and all(p not in output.parents for p in (analysis, batch, Path(f['host_reference']))),
            'Expected fresh results destination outside the evidence')
    replay, paired = b.read(analysis/'analysis.json'), b.read(analysis/'paired-seed-results.json')
    require(replay['status'] == 'verified_complete' and set(replay['arms']) == set(s.ARMS)
            and replay['freeze_sha256'] == freeze_sha and b.digest(batch/'authorization.json') == replay['authorization_sha256']
            and paired['parent_6033'] == s.PARENT_SHA and paired['offline_scientific_calls'] == 0
            and set(paired['seeds']) == {str(seed) for seed in s.SEEDS}, 'Replication analysis incomplete')
    previous, common, parent = accepted_6033()
    endpoints = {'6033': previous}
    common_by_seed = {'6033': common}
    pair_by_seed = {'6033': b.read(BASE/'optimization-diagnostics.json')['common_orientation']['10000']['paired']}
    stability_rows = {}
    prior_metrics = read_prior_stability()
    for method in s.METHODS:
        item = parent['arms'][method]['receipt']
        p = b.check_artifact(Path(item['path']), item)
        stability_rows[f'{method}-6033'] = stability(b.read(p), p.parent, prior_metrics[method])
    new_metrics = {r['arm']: r['metrics']['all']['observation_weighted'] for r in replay['results']
                   if r['step'] == 10000 and r['split'] == 'held_out'}
    receipts = {}
    for seed in s.SEEDS:
        row = paired['seeds'][str(seed)]
        endpoints[str(seed)] = row['endpoints']
        common_by_seed[str(seed)] = row['common_orientation']
        pair_by_seed[str(seed)] = row['pair_orientation']
        for method in s.METHODS:
            arm = f'{method}-{seed}'
            common_angles = row['common_orientation']['angle_median_p95'][method]
            endpoints[str(seed)][method]['common_orientation_median'] = common_angles[0] if common_angles else None
            p = batch/arm/'recovery.json'
            require(b.digest(p) == replay['arms'][arm]['receipt_sha256'], 'Arm receipt identity changed')
            r = s.completed(batch, arm, f, replay['execution_commit'], freeze_sha, replay['authorization_sha256'])
            inventory = r['outputs']
            receipts[arm] = dict(receipt=artifact(p), exit_code=artifact(batch/f'{arm}-exit-code.txt'),
                                 outputs=len(inventory), output_bytes=sum(item['bytes'] for item in inventory.values()),
                                 stream=inventory[s.dense.STREAM],
                                 checkpoints={name: item for name, item in inventory.items() if name.endswith('.pth')})
            stability_rows[arm] = stability(r, p.parent, new_metrics[arm])

    end_rows, comparisons = [], []
    for seed, rows in endpoints.items():
        count = common_by_seed[seed]['observations']
        for method, data in rows.items():
            end_rows.append((seed, method, count, *(data[key] for key in FIELDS)))
        for control in ('Final-Chart', 'K1-Dense'):
            a, c = rows['Final-Ambient'], rows[control]
            comparisons.append((seed, control, count, *(None if a[key] is None or c[key] is None
                                                       else a[key]-c[key] for key in FIELDS)))
    output.mkdir()
    write_csv(output/'held-out-endpoints.csv', ('seed', 'method', 'common_orientation_count', *FIELDS), end_rows)
    write_csv(output/'ambient-minus-control.csv', ('seed', 'control', 'common_orientation_count', *FIELDS), comparisons)
    json_new(output/'stability-and-orientation.json', dict(stability=stability_rows, common_all=common_by_seed,
                                                           common_pair=pair_by_seed))
    plot(endpoints, output)
    json_new(output/'artifact-manifest.json', dict(schema_version=1, execution_commit=replay['execution_commit'],
        batch=str(batch), freeze=artifact(batch/'freeze.json'), authorization=artifact(batch/'authorization.json'),
        dataset=f['dataset'], arms=receipts, accepted_6033=artifact(BASE/'artifact-manifest.json'),
        offline_analysis=[artifact(p) for p in sorted(analysis.iterdir()) if p.is_file()],
        packaged_files={p.name: dict(bytes=p.stat().st_size, sha256=b.digest(p)) for p in output.iterdir()}))


def read_prior_stability():
    with (BASE/'checkpoint-metrics.csv').open(newline='') as handle:
        return {row['arm']: dict(width_at_floor=int(row['minor_at_floor']),
                                 width_at_ceiling=int(row['major_at_ceiling'])) for row in csv.DictReader(handle)
                if (row['step'], row['split'], row['cohort'], row['weighting']) ==
                ('10000', 'held_out', 'all', 'observation_weighted')}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--analysis', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    cli.add_argument('--verify-native-artifacts', action='store_true',
                     help='Run the frozen verifier into a new analysis directory before packaging')
    args = cli.parse_args()
    if args.verify_native_artifacts:
        print(f'Verified {analyze_saved(args.freeze_sha256, args.analysis)} arm/state/split records', flush=True)
        summarize(args.freeze_sha256, args.analysis)
    package(args.freeze_sha256, args.analysis, args.output)
