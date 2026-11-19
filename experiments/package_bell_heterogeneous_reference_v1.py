"""Package review figures and numerical summaries from the verified Bell arrays."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_heterogeneous_v1 as b
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_heterogeneous_metrics_v1 import boundary_pairs, mesh_adjacency
from benchmarks.donut_recovery_v1 import json_new, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def artifact(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def label_figure(views, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(b.PANEL), 4, figsize=(8.0, 10.5), layout='constrained')
    cols = [('Major width', .20, .46, 'viridis'), ('Minor width', .09, .225, 'viridis'),
            ('Anisotropy gap', 0, 1, 'magma'), ('Region and boundary', 0, 2, 'viridis')]
    for row, camera in enumerate(b.PANEL):
        view = views[camera]
        ids = view['pixel_indices'][view['shader_valid']]
        widths = view['widths_major_minor'][view['shader_valid']]
        region = np.where(view['near_boundary'], 2, view['material_region'])
        for col, (label, lo, hi, colors) in enumerate(cols):
            values = (widths[:, 0], widths[:, 1], view['anisotropy_gap'], region)[col]
            panel = np.full(b.SIZE**2, np.nan)
            panel[ids] = values
            ax = axes[row, col]
            ax.imshow(panel.reshape(b.SIZE, b.SIZE), vmin=lo, vmax=hi, cmap=colors, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                split = 'Fit' if camera in b.SPLITS['fitting'] else 'Held out'
                ax.set_ylabel(f'{split} {camera}')
    for col in range(len(cols)):
        layer = axes[-1, col].images[0]
        scale = fig.colorbar(layer, ax=axes[:, col], orientation='horizontal', shrink=.94, pad=.015)
        if col == 3:
            scale.set_ticks([0, 1, 2])
            scale.ax.set_xticklabels(['Region 0', 'Region 1', 'Band'], fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def package(verification, verification_sha, output):
    require(verification.resolve() == verification and digest(verification) == verification_sha,
            'Explicit verified Bell review identity required')
    v = b.read(verification)
    f = b.freeze(b.ROOT/b.FREEZE, v['freeze_sha256'])
    batch, root = Path(f['host_output']), Path(v['package'])
    require(v['status'] == 'verified_pending_researcher_admission' and v['offline_replay_version'] == 2
            and v['protocol'] == b.PROTOCOL and v['researcher_admitted_dataset'] is False
            and v['training_authorized'] is False and root == batch/'reference',
            'Verified Bell reference identity differs')
    report = root/'generation.json'
    r = b.read(report)
    require(digest(report) == v['generation_sha256'] and r['outputs'] == v['outputs']
            and [row['camera'] for row in r['views']] == b.PANEL,
            'Generation report or verified artifact inventory differs')
    require(output.resolve() == output and not output.exists() and batch not in output.parents
            and Path(f['host_inputs']['oracle']) not in output.parents and output.parent.is_dir(),
            'Separate fresh Bell results directory required')
    mesh = b.SOURCE_PATHS['mesh']
    _, triangles = mesh_arrays(b.check_artifact(Path(f['host_inputs']['oracle'])/mesh, f['input_records'][mesh]))
    adjacency = mesh_adjacency(triangles)
    views, rows = {}, []
    for row in r['views']:
        c = row['camera']
        name = f'view-{c}.npz'
        view = views[c] = numeric(b.check_artifact(root/name, r['outputs'][name]))
        gate, energy = row['radiance_convergence'], row['reflectance_convergence']
        h = row['highlight_diagnostics']
        rows.append((c, 'fitting' if c in b.SPLITS['fitting'] else 'held_out', row['valid'],
                     gate['maximum_rgb_difference'], gate['relative_l2'], len(row['selected_pixels']),
                     energy['maximum_rgb_difference'], energy['relative_l2'], float(view['reflectance'].max()),
                     float(view['linear_components'][:, 2].max()), h['integrated_specular_fraction'],
                     h['highlight_coverage'], *[int(view[k].sum()) for k in
                     ('near_boundary', 'nearly_isotropic', 'exactly_isotropic')],
                     len(boundary_pairs(view, adjacency))))
    preview = b.check_artifact(root/'reference-components.png', r['outputs']['reference-components.png'])
    output.mkdir()
    shutil.copyfile(preview, output/preview.name)
    shutil.copyfile(verification, output/'verification.json')
    require(digest(output/preview.name) == digest(preview)
            and digest(output/'verification.json') == verification_sha, 'Copied reference assets differ')
    with (output/'view-metrics.csv').open('x', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(('camera', 'split', 'valid', 'rgb_max_delta', 'rgb_relative_l2', 'refined',
                         'reflectance_max_delta', 'reflectance_relative_l2', 'max_reflectance', 'max_linear_rgb',
                         'specular_fraction', 'highlight_fraction', 'boundary', 'near_isotropy', 'exact_isotropy',
                         'strict_image_boundary_pairs'))
        writer.writerows(rows)
    label_figure(views, output/'material-labels.png')
    json_new(output/'artifact-manifest.json', dict(schema_version=1, native_commit=r['commit'],
        generation=artifact(report), verification=artifact(verification), package=str(root),
        freeze=artifact(batch/'freeze.json'), authorization=artifact(batch/'authorization.json'),
        console=artifact(batch/'console.txt'), exit_code=artifact(batch/'exit-code.txt'),
        offline_replayer=artifact(b.ROOT/'experiments/verify_bell_heterogeneous_replay_v2.py'),
        scientific_outputs=r['outputs'], copied_reference_preview=artifact(preview),
        review_outputs={x.name: dict(bytes=x.stat().st_size, sha256=digest(x)) for x in sorted(output.iterdir())}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--verification', required=True, type=Path)
    cli.add_argument('--verification-sha256', required=True)
    cli.add_argument('--output', required=True, type=Path)
    a = cli.parse_args()
    package(a.verification, a.verification_sha256, a.output)
