"""Verify receipt inventories and compute paired seed-level Dense Bell differences from saved files."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_publication_seeds_v1 as s
from benchmarks.donut_recovery_v1 import json_new
from experiments import analyze_donut_heterogeneous_training_v1 as accepted
from experiments.analyze_bell_known_l2_v1 import common_orientation, endpoint
from infrastructure.signatures import require


def analyze_saved(sha, output):
    # Run the shared verifier with fresh-seed receipts. The predecessor remains byte-identical.
    source = Path(accepted.__file__)
    text = source.read_text()
    edits = {
        "r['historical_arm_initial_parity_verified']": "r['fresh_seed_initialization_verified']",
        "r['policy']==h.POLICY": "r['policy']==spec.policy(arm)",
        "==ref.read(Path(recipe['initial']['path']))": '',
    }
    for old, new in edits.items():
        require(text.count(old) == 1, 'Unexpected shared verifier receipt signature')
        text = text.replace(old, new)
    namespace = dict(__file__=str(source), __name__=__name__)
    exec(compile(text, 'bell-publication-offline-v1', 'exec'), namespace)
    return namespace['analyze'](sha, output, acquisition=s)


def summarize(sha, output):
    f = s.freeze(s.ROOT/s.FREEZE, sha)
    root = Path(f['host_output'])
    analysis = s.b.read(output/'analysis.json')
    require(analysis['status'] == 'verified_complete' and analysis['freeze_sha256'] == sha
            and set(analysis['arms']) == set(s.ARMS), 'Missing verified seed analysis')
    files = f['dataset']
    _, views = s.dataset(f, Path(f['host_reference']), Path(files['admission']['path']),
                         Path(files['verification']['path']))
    all_rows = {row['arm']: row['metrics']['all']['observation_weighted'] for row in analysis['results']
                if (row['step'], row['split']) == (10000, 'held_out')}
    by_seed = {}
    base_hashes = set()
    cameras = s.SPLITS['held_out']
    for seed in s.SEEDS:
        receipts = {}
        for method in s.METHODS:
            arm = f'{method}-{seed}'
            receipts[method] = s.completed(root, arm, f, analysis['execution_commit'], sha,
                                           analysis['authorization_sha256'])
        base = receipts['K1-Dense']['initial']
        s.validate_initial(base, f'K1-Dense-{seed}')
        require(base['state_sha256'] not in base_hashes, 'Seed base hashes are identical')
        base_hashes.add(base['state_sha256'])
        for method in s.final.ARMS:
            r = receipts[method]
            s.validate_initial(r['initial'], f'{method}-{seed}', base)
            require(r['seed_matched_base_initial_sha256'] == receipts['K1-Dense']['outputs']['initial.json']['sha256'],
                    'Native seed anchor differs')
        for name in ('paired-initial-gate.npz', 'tensor-physical-admission.npz', 'native-reference-gate.npz'):
            captures = [s.b.numeric(s.b.check_artifact(root/f'{method}-{seed}'/name, receipts[method]['outputs'][name]))
                        for method in s.final.ARMS]
            require(captures[0].keys() == captures[1].keys(), 'Paired native capture inventory differs')
            for key in captures[0]:
                np.testing.assert_array_equal(captures[0][key], captures[1][key])
        for camera in s.PANEL:
            name = f'step-00000/camera-{camera}/materials.npz'
            fields = [s.b.numeric(s.b.check_artifact(root/f'{method}-{seed}'/name, receipts[method]['outputs'][name]))
                      for method in s.final.ARMS]
            np.testing.assert_array_equal(fields[0]['hit_mask'], fields[1]['hit_mask'])
            mask = fields[0]['hit_mask']
            for key, tol in (('kd', 0), ('metallic', 0), ('Q_tensor_direct', s.d9.REPRESENTATION['initial_Q_atol']),
                             ('linear_rgb', s.d9.REPRESENTATION['initial_RGB_atol'])):
                np.testing.assert_allclose(fields[0][key][mask], fields[1][key][mask], atol=tol, rtol=0)
        predictions = {method: {c: endpoint(root/f'{method}-{seed}', r, c, views[c]) for c in cameras}
                       for method, r in receipts.items()}
        pair = common_orientation({key: predictions[key] for key in s.final.ARMS}, cameras)
        all_common = common_orientation(predictions, cameras)
        rows = {}
        for method, r in receipts.items():
            arm = f'{method}-{seed}'
            m = all_rows[arm]
            jumps = analysis['arms'][arm]['boundary_history'][-1]
            bias = np.concatenate([predictions[method][c][4]-views[c]['widths_major_minor'][views[c]['shader_valid']]
                                   for c in cameras]).mean(axis=0)
            rows[method] = dict(Q_median=m['Q_relative']['median'], Q_p95=m['Q_relative']['p95'],
                                major_MAE=m['major_width']['mean'], minor_MAE=m['minor_width']['mean'],
                                signed_major_bias=float(bias[0]), signed_minor_bias=float(bias[1]),
                                RGB_L2=m['RGB_relative_L2'], eligibility=m['orientation_eligible'],
                                common_orientation_p95=all_common['angle_median_p95'][method][1]
                                if all_common['observations'] else None, Q_jump=jumps['Q_jump_projection_median'],
                                major_jump=jumps['predicted_signed_width_jump'][0],
                                minor_jump=jumps['predicted_signed_width_jump'][1], seconds=r['elapsed_seconds'])
        ambient = rows['Final-Ambient']
        delta = {other: {metric: ambient[metric]-value if value is not None else None
                         for metric, value in rows[other].items()} for other in ('Final-Chart', 'K1-Dense')}
        by_seed[str(seed)] = dict(endpoints=rows, ambient_minus=delta, pair_orientation=pair,
                                  common_orientation=all_common)
    json_new(output/'paired-seed-results.json', dict(schema_version=1, parent_6033=s.PARENT_SHA, seeds=by_seed,
                                                     offline_scientific_calls=0))
    with (output/'paired-endpoints.csv').open('x', newline='') as handle:
        keys = list(by_seed[str(s.SEEDS[0])]['endpoints']['K1-Dense'])
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['seed', 'method', *keys])
        for seed, row in by_seed.items():
            for method, values in row['endpoints'].items():
                writer.writerow([seed, method, *(values[k] for k in keys)])


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    print(f'Verified {analyze_saved(args.freeze_sha256, args.output)} arm/state/split records')
    summarize(args.freeze_sha256, args.output)
