"""Package the verified heterogeneous Donut results from saved arrays."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from benchmarks import donut_heterogeneous_training_v1 as training
from benchmarks import donut_heterogeneous_v1 as reference
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_donut_heterogeneous_training_v1 import read_endpoint
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def artifact(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def plot_curves(analysis, directory, plt):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    measures = (('Relative linear RGB L2', 'all', 'RGB_relative_L2', None),
                ('Relative Q error (median)', 'all', 'Q_relative', 'median'),
                ('Relative Q error (p95)', 'all', 'Q_relative', 'p95'),
                ('Near-isotropic relative Q error (median)', 'nearly_isotropic', 'Q_relative', 'median'))
    for arm in training.ARMS:
        rows = [r for r in analysis['results'] if r['arm'] == arm and r['split'] == 'held_out']
        for ax, (label, cohort, field, stat) in zip(axes.flat, measures):
            values = [r['metrics'][cohort]['observation_weighted'][field] for r in rows]
            if stat:
                values = [v[stat] for v in values]
            ax.plot(training.STEPS, values, marker='o', markersize=3, label=arm)
            ax.set(xlabel='Adam returns', ylabel=label, yscale='log', xticks=training.STEPS)
            ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(directory / 'recovery-curves.png', dpi=165)
    plt.close(fig)


def plot_views(batch, views, predictions, directory, plt):
    from PIL import Image

    fig, axes = plt.subplots(6, 5, figsize=(10, 10.4), layout='constrained')
    for index, camera in enumerate(reference.SPLITS['held_out']):
        row = 3 * index
        view = views[camera]
        valid = view['shader_valid']
        pixels = view['pixel_indices'][valid]
        for col, label in enumerate(('Oracle', *training.ARMS)):
            source = batch / (label if col else 'D2') / 'step-10000' / f'camera-{camera}'
            with Image.open(source / ('reconstructed.png' if col else 'ground_truth.png')) as im:
                axes[row, col].imshow(np.asarray(im))
            axes[row, col].set_title(label)
            minor = predictions[label, camera]['minor_predict'] if col else view['widths_major_minor'][valid, 1]
            error = predictions[label, camera]['Q_relative'] if col else np.zeros(len(pixels))
            for offset, values, low, high, palette in ((1, minor, .04, .24, 'viridis'),
                                                         (2, error, 0, 1, 'magma')):
                panel = np.full(reference.SIZE ** 2, np.nan)
                panel[pixels] = values
                layer = axes[row + offset, col].imshow(
                    panel.reshape(reference.SIZE, reference.SIZE), vmin=low, vmax=high, cmap=palette)
                if col == len(training.ARMS):
                    fig.colorbar(layer, ax=axes[row + offset], fraction=.013, pad=.01, extend='max')
        for offset, label in enumerate((f'Camera {camera} RGB', 'Minor width', 'Relative Q error')):
            axes[row + offset, 0].set_ylabel(label)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(directory / 'held-out-views.png', dpi=165)
    plt.close(fig)


def package(freeze_sha, analysis_dir, directory):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = training.freeze(training.ROOT / training.FREEZE, freeze_sha)
    batch = Path(f['host_output'])
    approval = reference.read(batch / 'authorization.json')
    analysis = reference.read(analysis_dir / 'analysis.json')
    require(approval == training.authorization(f, approval['commit'], freeze_sha)
            and analysis['status'] == 'verified_complete' and analysis['freeze_sha256'] == freeze_sha
            and analysis['execution_commit'] == approval['commit']
            and analysis['authorization_sha256'] == digest(batch / 'authorization.json')
            and len(analysis['results']) == len(training.ARMS) * len(training.STEPS) * len(reference.SPLITS),
            'Wrong execution or offline analysis')
    require(directory.resolve() == directory and not directory.exists()
            and batch not in directory.parents and Path(f['host_reference']) not in directory.parents,
            'Fresh results directory required')
    dataset = f['dataset']
    _, views = training.dataset(f, Path(f['host_reference']), Path(dataset['admission']['path']),
                                Path(dataset['verification']['path']))
    arms, predictions, rows, widths, orientation = {}, {}, [], {}, {}
    supports = {c: metrics.fitting_support(views, c) for c in reference.PANEL}
    common = {}
    for arm in training.ARMS:
        r = reference.read(batch / arm / 'recovery.json')
        receipt = artifact(batch / arm / 'recovery.json')
        require(receipt['sha256'] == analysis['arms'][arm]['receipt_sha256'], 'Arm receipt differs')
        checkpoints = {name: x for name, x in r['outputs'].items() if name.endswith('.pth')}
        arms[arm] = dict(receipt=receipt, log=artifact(batch / f'{arm}-console.txt'),
                         exit_code=artifact(batch / f'{arm}-exit-code.txt'), checkpoints=checkpoints,
                         output_files=len(r['outputs']), output_bytes=sum(x['bytes'] for x in r['outputs'].values()))
        widths[arm] = {}
        for camera, view in views.items():
            base = batch / arm / 'step-10000'
            data = base / f'camera-{camera}/materials.npz'
            errors = base / f'camera-{camera}-errors.npz'
            for p in (data, errors, data.with_name('reconstructed.png'), data.with_name('ground_truth.png')):
                reference.check_artifact(p, r['outputs'][p.relative_to(batch / arm).as_posix()])
            e, rgb, gt, _, w = read_endpoint(batch / arm, 10000, camera, view)
            predictions[arm, camera] = e
            widths[arm][str(camera)] = (w - view['widths_major_minor'][view['shader_valid']]).mean(0).tolist()
            g, l = supports[camera]
            for name, keep in (('all', np.ones(len(w), dtype=bool)), ('no_fitting_support', g == 0),
                               ('no_highlight_support', l == 0)):
                s = metrics.score({k: v[keep] for k, v in e.items()}, rgb[keep], gt[keep], np.ones(keep.sum()))
                if s is not None:
                    rows.append((arm, camera, name, s['observations'], s['RGB_relative_L2'], s['RGB_RMSE'],
                                 s['Q_relative']['median'], s['Q_relative']['p95'], s['major_width']['mean'],
                                 s['minor_width']['mean'], s['orientation_eligible'],
                                 s['orientation_degrees']['p95'] if s['orientation_degrees'] else None))

    for split, cameras in reference.SPLITS.items():
        masks = {c: np.logical_and.reduce([predictions[arm, c]['orientation_eligible'] for arm in training.ARMS])
                 for c in cameras}
        common[split] = int(sum(x.sum() for x in masks.values()))
        orientation[split] = {}
        for arm in training.ARMS:
            values = np.concatenate([predictions[arm, c]['orientation_degrees'][masks[c]] for c in cameras])
            orientation[split][arm] = dict(zip(('median', 'p95'), np.quantile(values, [.5, .95]).tolist()))

    directory.mkdir()
    shutil.copyfile(analysis_dir / 'summary.csv', directory / 'checkpoint-metrics.csv')
    with (directory / 'per-camera-metrics.csv').open('x', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(('arm', 'camera', 'cohort', 'observations', 'linear_RGB_relative_L2', 'linear_RGB_RMSE',
                         'Q_median', 'Q_p95', 'major_width_MAE', 'minor_width_MAE',
                         'orientation_eligible', 'orientation_p95_degrees'))
        writer.writerows(rows)
    json_new(directory / 'endpoint-diagnostics.json', dict(
        common_orientation_observations=common, orientation_common_subset_degrees=orientation,
        signed_mean_width_error_by_camera=widths))
    plot_curves(analysis, directory, plt)
    plot_views(batch, views, predictions, directory, plt)
    json_new(directory / 'artifact-manifest.json', dict(
        schema_version=1, batch=str(batch), reference=str(f['host_reference']), dataset=dataset,
        authorization=artifact(batch / 'authorization.json'), freeze=artifact(batch / 'freeze.json'),
        execution_commit=approval['commit'], image_id=approval['image_id'], arms=arms,
        offline_analysis=[artifact(analysis_dir / name) for name in ('analysis.json', 'summary.csv')],
        packaged_files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(directory.iterdir())}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--analysis', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    package(args.freeze_sha256, args.analysis, args.output)
