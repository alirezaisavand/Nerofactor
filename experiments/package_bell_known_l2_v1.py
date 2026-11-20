"""Package the completed fixed-light Bell comparison from verified saved files."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_known_l2_completion_v3 as completion
from benchmarks import bell_known_l2_v1 as known
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_known_l2_v1 import endpoint
from experiments.package_bell_heterogeneous_training_v1 import artifact, write_csv
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


ARMS = tuple(arm for pair in known.MATCH.items() for arm in pair[::-1])


def curves(rows, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), layout='constrained')
    fields = [('RGB_relative_L2', None, 'Linear RGB relative L2'),
              ('Q_relative', 'median', 'Relative world-Q error (median)'),
              ('Q_relative', 'p95', 'Relative world-Q error (p95)'),
              ('minor_width', 'mean', 'Minor-width MAE')]
    for arm_index, arm in enumerate(ARMS):
        held = sorted((r for r in rows if r['arm'] == arm and r['split'] == 'held_out'),
                      key=lambda r: r['step'])
        for ax, (key, stat, label) in zip(axes.flat, fields):
            values = [r['metrics']['all']['observation_weighted'][key] for r in held]
            if stat:
                values = [value[stat] for value in values]
            ax.semilogy(known.STEPS, values, '-o' if arm in known.ARMS else '--s',
                        color=f'C{arm_index // 2}', markersize=3, linewidth=1.2, label=arm)
            ax.set(xlabel='Adam updates', ylabel=label, xticks=known.STEPS)
            ax.grid(alpha=.22)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output/'recovery-curves.png', dpi=165)
    plt.close(fig)


def held_views(views, roots, receipts, output, plt):
    from PIL import Image

    fig, axes = plt.subplots(6, 7, figsize=(13.6, 9.7), layout='constrained')
    for index, camera in enumerate(known.SPLITS['held_out']):
        view = views[camera]
        valid = view['shader_valid']
        pixels = view['pixel_indices'][valid]
        for col, arm in enumerate(('Oracle', *ARMS)):
            row = 3 * index
            source = arm if col else 'K1'
            name = 'reconstructed.png' if col else 'ground_truth.png'
            key = f'step-10000/camera-{camera}/{name}'
            with Image.open(known.b.check_artifact(roots[source]/key, receipts[source]['outputs'][key])) as png:
                axes[row, col].imshow(np.asarray(png), interpolation='nearest')
            axes[row, col].set_title(arm)
            if col:
                errors, *_ = endpoint(roots[arm], receipts[arm], camera, view)
                minor, q = errors['minor_predict'], errors['Q_relative']
            else:
                minor, q = view['widths_major_minor'][valid, 1], np.zeros(len(pixels))
            for offset, values, bounds, palette in ((1, minor, (.04, .24), 'viridis'),
                                                    (2, q, (0, 1), 'magma')):
                image = np.full(view['hit_mask'].shape, np.nan)
                image.flat[pixels] = values
                layer = axes[row+offset, col].imshow(image, vmin=bounds[0], vmax=bounds[1],
                                                    cmap=palette, interpolation='nearest')
                if col == len(ARMS):
                    fig.colorbar(layer, ax=axes[row+offset], fraction=.012, pad=.012, extend='max')
        for offset, label in enumerate((f'Camera {camera} RGB', 'Minor width', 'Relative world-Q error')):
            axes[3*index+offset, 0].set_ylabel(label)
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    fig.savefig(output/'held-out-views.png', dpi=165)
    plt.close(fig)


def checkpoint_rows(rows, output):
    metrics = []
    for row in rows:
        for cohort, weighting in row['metrics'].items():
            if row['step'] != 10000 and cohort != 'all':
                continue
            for method, score in weighting.items():
                if score is None:
                    continue
                orientation = score['orientation_degrees']
                metrics.append((row['arm'], row['step'], row['split'], cohort, method, score['observations'],
                    score['RGB_relative_L2'], *(score['Q_relative'][k] for k in ('mean', 'median', 'p95')),
                    score['Q_at_most_005'], score['major_width']['mean'], score['minor_width']['mean'],
                    score['major_width_relative']['mean'], score['minor_width_relative']['mean'],
                    score['gap']['mean'], score['width_at_floor'], score['width_at_ceiling'],
                    score['orientation_eligible'], orientation['median'] if orientation else None,
                    orientation['p95'] if orientation else None, score['color_max']['mean'], score['metallic']['mean']))
    write_csv(output/'checkpoint-metrics.csv', ('arm', 'step', 'split', 'cohort', 'weighting', 'observations',
        'RGB_relative_L2', 'Q_mean', 'Q_median', 'Q_p95', 'Q_at_most_005', 'major_width_MAE', 'minor_width_MAE',
        'major_width_relative_MAE', 'minor_width_relative_MAE', 'mean_anisotropy_gap', 'minor_at_floor',
        'major_at_ceiling', 'orientation_eligible', 'orientation_median_degrees', 'orientation_p95_degrees',
        'albedo_max_MAE', 'metallic_MAE'), metrics)


def package(freeze_sha, analysis_dir, combined_sha, matched_sha, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = completion.freeze(known.ROOT/completion.FREEZE, freeze_sha)
    old = completion.prior_definition()
    native = Path(f['host_output'])
    prior = Path(old['host_output'])
    combined_path = analysis_dir/'combined-analysis.json'
    matched_path = analysis_dir/'matched-controls.json'
    require(digest(combined_path) == combined_sha and digest(matched_path) == matched_sha,
            'Expected explicit verified analysis identities')
    combined, matched = (known.b.read(p) for p in (combined_path, matched_path))
    authenticated = known.b.read(native/'authorization.json')
    require(authenticated == completion.authorization(f, combined['execution_commit'], freeze_sha)
            and combined['authorization_sha256'] == digest(native/'authorization.json')
            and combined['freeze_sha256'] == freeze_sha and combined['protocol'] == completion.PROTOCOL
            and combined['status'] == 'verified_complete' and set(combined['arms']) == set(known.ARMS),
            'Foreign or incomplete verified known-light comparison')
    item = matched['learned_analysis']
    learned = known.b.read(known.b.check_artifact(Path(item['path']), item))
    roots = {'K1': prior/'K1', **{a: native/a for a in completion.ARMS},
             **{control: Path(old['controls'][arm]['path']).parent for arm, control in known.MATCH.items()}}
    separate = (native, prior, Path(f['host_reference']), analysis_dir, *(root.parent for root in roots.values()))
    require(output.resolve() == output and not output.exists() and all(p not in output.parents for p in separate),
            'Expected fresh results destination separate from evidence')
    d = f['dataset']
    _, views = known.dataset(f, Path(f['host_reference']), Path(d['admission']['path']), Path(d['verification']['path']))
    rows = [row for report in (learned, combined) for row in report['results'] if row['arm'] in ARMS]
    require(len(rows) == 72 and {(r['arm'], r['split'], r['step']) for r in rows} ==
            {(arm, split, step) for arm in ARMS for split in known.SPLITS for step in known.STEPS},
            'Incomplete comparison panel')

    receipts, records, diagnostics = {}, {}, {}
    for arm in ARMS:
        root = roots[arm]
        verified = (combined if arm in known.ARMS else learned)['arms'][arm]
        receipt_path = root/'recovery.json'
        require(digest(receipt_path) == verified['receipt_sha256'], 'Arm receipt differs: '+arm)
        r = receipts[arm] = known.b.read(receipt_path)
        inventory = r['outputs']
        records[arm] = dict(receipt=artifact(receipt_path), exit_code=artifact(root.parent/(arm+'-exit-code.txt')),
            train_log=artifact(root/'training.jsonl'), outputs=len(inventory),
            output_bytes=sum(v['bytes'] for v in inventory.values()),
            checkpoints={k: v for k, v in inventory.items() if k.endswith('.pth')})
        diagnostics[arm] = dict(initial=r['initial'], elapsed_seconds=r['elapsed_seconds'], resources=r['resources'],
            illumination_history=verified['illumination_history'], boundary_history=verified['boundary_history'],
            gradient_probes=verified['gradient_probes'], surface_q_timing=r.get('surface_q_timing'),
            known_lighting_audit=r.get('known_lighting_audit'),
            reserved_edge_Q=r['saved_endpoints'][-1]['metrics']['surface_Q_smoothness_diagnostic'])
    output.mkdir()
    checkpoint_rows(rows, output)
    cameras = [(r['arm'], r['camera'], name, m['observations'], m['RGB_relative_L2'], m['Q_relative']['median'],
                m['Q_relative']['p95'], m['major_width']['mean'], m['minor_width']['mean'])
               for report in (learned, combined) for r in report['per_camera_endpoint'] if r['arm'] in ARMS
               for name, m in r['metrics'].items() if m is not None]
    write_csv(output/'per-camera-metrics.csv', ('arm', 'camera', 'cohort', 'observations', 'RGB_relative_L2',
              'Q_median', 'Q_p95', 'major_width_MAE', 'minor_width_MAE'), cameras)
    json_new(output/'endpoint-diagnostics.json', dict(arms=diagnostics, matched_orientation=matched['known_pairs'],
        matched_lighting={arm: {k: m[k] for k in ('learned_control', 'splits', 'last_2000_updates',
            'initial_material_and_reference_checks_exact', 'matching_learning_rates')} for arm, m in matched['arms'].items()}))
    curves(rows, output, plt)
    held_views(views, roots, receipts, output, plt)
    json_new(output/'artifact-manifest.json', dict(schema_version=1, dataset=d, image=completion.IMAGE,
        native_phase_files=[artifact(base/name) for base in (prior, native) for name in ('freeze.json', 'authorization.json')],
        v3_previous_cleanup=artifact(native/'previous-run-cleanup.json'),
        v1_failed_tensor_evidence=f['prior_evidence'], v2_failed_tensor_evidence=f['second_failure'],
        offline_analysis=[artifact(analysis_dir/name) for name in
                          ('analysis.json', 'combined-analysis.json', 'matched-controls.json', 'summary.csv')],
        learned_analysis=item, arms=records,
        packaged_files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.iterdir())}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--combined-sha256', required=True)
    p.add_argument('--matched-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    package(a.freeze_sha256, a.analysis, a.combined_sha256, a.matched_sha256, a.output)
