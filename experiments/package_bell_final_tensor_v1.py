"""Package the verified final Bell comparison without model or renderer calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from benchmarks import bell_final_tensor_v1 as f
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_known_l2_v1 import endpoint
from experiments.package_bell_heterogeneous_training_v1 import artifact, write_csv
from experiments.package_bell_known_l2_v1 import checkpoint_rows
from infrastructure.signatures import require


b = f.b
ARMS = ('K1-Dense', *f.ARMS)


def plot_curves(rows, comparison, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout='constrained')
    for arm, color in zip(ARMS, ('0.35', '#b36125', '#275e9e')):
        held = sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['step'])
        scores = [r['metrics']['all']['observation_weighted'] for r in held]
        series = ([s['Q_relative']['median'] for s in scores], [s['Q_relative']['p95'] for s in scores],
                  [s['RGB_relative_L2'] for s in scores])
        for ax, values, label in zip(axes.flat, series, ('World-Q error (median)', 'World-Q error (p95)',
                                                        'Held-out linear RGB relative L2')):
            ax.semilogy(f.STEPS, values, '-o', color=color, markersize=3, label=arm)
            ax.set(ylabel=label, xticks=f.STEPS)
        windows = comparison['training_windows'][arm]
        axes[1, 1].semilogy([w['last_update'] for w in windows], [w['RGB_loss'] for w in windows], color=color)
    axes[1, 1].set(ylabel='Fitting RGB loss (500-update mean)', xticks=f.STEPS)
    for ax in axes.flat:
        ax.set_xlabel('Adam updates')
        ax.grid(alpha=.24)
    axes[0, 0].legend(frameon=False)
    fig.savefig(output/'recovery-curves.png', dpi=160)
    plt.close(fig)


def plot_views(views, images, scores, output, plt):
    fig, axes = plt.subplots(8, 4, figsize=(9.2, 13.5), layout='constrained')
    for index, camera in enumerate(f.SPLITS['held_out']):
        view = views[camera]
        valid = view['shader_valid']
        row = 4*index
        for col, arm in enumerate(('Oracle', *ARMS)):
            axes[row, col].imshow(images[camera, arm], interpolation='nearest')
            axes[row, col].set_title(arm, fontsize=10)
            if arm == 'Oracle':
                widths, q = view['widths_major_minor'][valid], np.zeros(valid.sum())
            else:
                errors, _, _, _, widths = scores[camera, arm]
                q = errors['Q_relative']
            layers = ((widths[:, 0], (.10, .50), 'viridis'), (widths[:, 1], (.04, .24), 'viridis'),
                      (q, (0, .65), 'magma'))
            for offset, (values, bounds, palette) in enumerate(layers, 1):
                image = np.full(view['hit_mask'].shape, np.nan)
                image.flat[view['pixel_indices'][valid]] = values
                layer = axes[row+offset, col].imshow(image, vmin=bounds[0], vmax=bounds[1], cmap=palette,
                                                   interpolation='nearest')
                if col == len(ARMS):
                    fig.colorbar(layer, ax=axes[row+offset], fraction=.022, pad=.02, extend='both')
        for offset, label in enumerate((f'Camera {camera} RGB', 'Major width', 'Minor width', 'Relative Q error')):
            axes[row+offset, 0].set_ylabel(label)
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    fig.savefig(output/'held-out-views.png', dpi=160)
    plt.close(fig)


def package(freeze_sha, analysis, analysis_sha, comparison_sha, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freeze = f.freeze(f.ROOT/f.FREEZE, freeze_sha)
    f.verify_controls(freeze)
    batch = Path(freeze['host_output'])
    require(output.resolve() == output and not output.exists() and all(
        p not in output.parents for p in (batch, analysis, Path(freeze['host_reference']))),
        'Expected fresh results destination separate from evidence')
    primary, comparison_path = analysis/'analysis.json', analysis/'method-comparison.json'
    require(b.digest(primary) == analysis_sha and b.digest(comparison_path) == comparison_sha,
            'Offline analysis identities differ')
    full, comparison = b.read(primary), b.read(comparison_path)
    auth_path = batch/'authorization.json'
    require(full['status'] == 'verified_complete' and full['freeze_sha256'] == freeze_sha == b.digest(batch/'freeze.json')
            and full['authorization_sha256'] == b.digest(auth_path)
            and b.read(auth_path) == f.authorization(freeze, full['execution_commit'], freeze_sha)
            and set(full['arms']) == set(f.ARMS) and comparison['primary_step'] == 10000
            and comparison['offline_scientific_queries'] == 0, 'Incomplete or foreign final pair evidence')
    prior = f.dense.read_artifact(freeze['dense_analysis'])
    inputs = freeze['dataset']
    _, views = f.dataset(freeze, Path(freeze['host_reference']), Path(inputs['admission']['path']),
                         Path(inputs['verification']['path']))
    roots = {'K1-Dense': Path(freeze['controls']['K1-Dense']['path']).parent, **{arm: batch/arm for arm in f.ARMS}}
    rows = [r for report in (prior, full) for r in report['results'] if r['arm'] in ARMS]
    require(len(rows) == 36 and {(r['arm'], r['step'], r['split']) for r in rows} ==
            {(arm, step, split) for arm in ARMS for step in f.STEPS for split in f.SPLITS},
            'Incomplete final comparison metrics')

    records, images, scores, endpoint_rows, camera_rows, boundary_rows = {}, {}, {}, [], [], []
    camera_panel = f.SPLITS['held_out']
    predicted, widths = 'predicted_signed_width_jump', 'reference_signed_width_jump'
    for arm, root in roots.items():
        reference = prior if arm == 'K1-Dense' else full
        receipt_path = root/'recovery.json'
        require(b.digest(receipt_path) == reference['arms'][arm]['receipt_sha256'], 'Arm receipt differs: '+arm)
        receipt = b.read(receipt_path)
        files = receipt['outputs']
        exit_code = root.parent/(arm+'-exit-code.txt')
        require(exit_code.read_text().strip() == '0', 'Arm exit code differs: '+arm)
        stream = b.check_artifact(root/f.dense.STREAM, files[f.dense.STREAM])
        checkpoints = {name: artifact(b.check_artifact(root/name, item))
                       for name, item in files.items() if name.endswith('.pth')}
        records[arm] = dict(receipt=artifact(receipt_path), exit_code=artifact(exit_code),
                           photometric_stream=artifact(stream), checkpoints=checkpoints, outputs=len(files),
                           output_bytes=sum(item['bytes'] for item in files.values()))
        biases = []
        for camera in camera_panel:
            view = views[camera]
            valid = view['shader_valid']
            scores[camera, arm] = endpoint(root, receipt, camera, view)
            _, _, _, _, prediction = scores[camera, arm]
            bias = prediction-view['widths_major_minor'][valid]
            biases.append(bias)
            names = {'reconstructed.png': arm}
            if arm == 'K1-Dense':
                names['ground_truth.png'] = 'Oracle'
            for name, label in names.items():
                key = f'step-10000/camera-{camera}/{name}'
                with Image.open(b.check_artifact(root/key, files[key])) as png:
                    images[camera, label] = np.asarray(png)
            metrics = next(r['metrics']['all'] for r in reference['per_camera_endpoint']
                           if r['arm'] == arm and r['camera'] == camera)
            camera_rows.append((arm, camera, int(valid.sum()), metrics['Q_relative']['median'],
                                metrics['Q_relative']['p95'], metrics['RGB_relative_L2'], *bias.mean(0)))
        pooled_bias = np.concatenate(biases).mean(0)
        arm_diag = comparison['arms'][arm]
        m = next(r['metrics']['all']['observation_weighted'] for r in rows
                 if (r['arm'], r['step'], r['split']) == (arm, 10000, 'held_out'))
        endpoint_rows.append((arm, m['Q_relative']['median'], m['Q_relative']['p95'], m['Q_at_most_005'],
                              m['major_width']['mean'], m['minor_width']['mean'], *pooled_bias,
                              m['RGB_relative_L2'], m['orientation_eligible'], m['gap']['mean'],
                              arm_diag['elapsed_seconds']))
        jump = arm_diag['boundary'][-1]
        boundary_rows.append((arm, jump['pairs'], jump['Q_jump_projection_median'], jump['absolute_Q_jump_error_mean'],
                              *jump[predicted], *jump[widths], *jump['signed_width_jump_MAE']))

    output.mkdir()
    checkpoint_rows(rows, output)
    write_csv(output/'held-out-endpoints.csv', ('arm', 'Q_median', 'Q_p95', 'Q_at_most_005', 'major_width_MAE',
        'minor_width_MAE', 'major_width_bias', 'minor_width_bias', 'RGB_relative_L2', 'orientation_eligible',
        'mean_anisotropy_gap', 'elapsed_seconds'), endpoint_rows)
    write_csv(output/'held-out-cameras.csv', ('arm', 'camera', 'observations', 'Q_median', 'Q_p95', 'RGB_relative_L2',
                                             'major_width_bias', 'minor_width_bias'), camera_rows)
    write_csv(output/'boundary-jumps.csv', ('arm', 'pairs', 'median_projected_Q_jump_fraction', 'absolute_Q_jump_MAE',
        'predicted_major_jump', 'predicted_minor_jump', 'oracle_major_jump', 'oracle_minor_jump',
        'major_jump_MAE', 'minor_jump_MAE'), boundary_rows)
    json_new(output/'optimization-diagnostics.json', {k: v for k, v in comparison.items() if k != 'trajectories'})
    plot_curves([r for r in rows if r['split'] == 'held_out'], comparison, output, plt)
    plot_views(views, images, scores, output, plt)
    json_new(output/'artifact-manifest.json', dict(schema_version=1, batch=str(batch),
        execution_commit=full['execution_commit'], image=freeze['image'], freeze=artifact(batch/'freeze.json'),
        authorization=artifact(auth_path), dataset=inputs, K1_Dense_analysis=freeze['dense_analysis'], arms=records,
        offline_analysis=[artifact(p) for p in (primary, comparison_path, analysis/'summary.csv')],
        packaged_files={p.name: dict(bytes=p.stat().st_size, sha256=b.digest(p)) for p in sorted(output.iterdir())}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--analysis', required=True, type=Path)
    cli.add_argument('--analysis-sha256', required=True)
    cli.add_argument('--comparison-sha256', required=True)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    package(args.freeze_sha256, args.analysis, args.analysis_sha256, args.comparison_sha256, args.output)
