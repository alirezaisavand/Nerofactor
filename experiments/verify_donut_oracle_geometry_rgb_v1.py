"""Replay fixed-material paired refinement and score the defined radiance support."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_oracle_geometry_rgb_v1 as rgb
from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.bell_e11 import linear_to_srgb
from experiments.donut_oracle_geometry_rgb_v1 import capture_name
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


LUMA = np.array([.2126, .7152, .0722])


def ratio(n, d):
    return float(n/d) if d else None


def score(oracle, reconstruction, mask, known):
    use = mask & known
    missing = mask & ~known
    a, b = oracle[:,use], reconstruction[:,use]
    diff = b-a
    uncertainty = diff[1]-diff[0]
    result = dict(pixels=int(mask.sum()), defined_pixels=int(use.sum()), undefined_pixels=int(missing.sum()),
                  undefined_fraction=ratio(missing.sum(), mask.sum()))
    for i,name in enumerate(('diffuse', 'specular', 'RGB')):
        delta = diff[1,:,i]
        err = float(np.square(delta).sum())
        target = float(np.square(a[1,:,i]).sum())
        all_target = float(np.square(oracle[1,mask,i]).sum())
        residue = float(np.square(uncertainty[:,i]).sum())
        result[name] = dict(squared_error=err, known_oracle_squared_energy=target,
                            undefined_oracle_energy_fraction=ratio(np.square(oracle[1,missing,i]).sum(), all_target),
                            relative_L2=ratio(np.sqrt(err), np.sqrt(target)),
                            known_support_RMSE=np.sqrt(err/delta.size).item() if delta.size else None,
                            signed_luminance=float((delta@LUMA).mean()) if len(delta) else None,
                            squared_relative_error_lower_bound=ratio(err, all_target),
                            coarse_fine_difference_RMSE=np.sqrt(residue/delta.size).item() if delta.size else None,
                            coarse_fine_difference_max_channel=float(abs(uncertainty[:,i]).max()) if len(delta) else None,
                            coarse_fine_difference_over_oracle_L2=ratio(np.sqrt(residue), np.sqrt(target)),
                            coarse_fine_difference_over_discrepancy_L2=ratio(np.sqrt(residue), np.sqrt(err)))
    return result


def summarize(views):
    a, b = (np.concatenate([v['images'][arm] for v in views], axis=1) for arm in rgb.ARMS)
    known = np.concatenate([v['known'] for v in views])
    cohorts = {key: np.concatenate([v['cohorts'][key] for v in views]) for key in views[0]['cohorts']}
    cohorts['raster'] = np.ones_like(known)
    report = {name: score(a, b, mask, known) for name,mask in cohorts.items()}
    total = report['raster']['RGB']['squared_error']
    for item in report.values():
        item['share_of_defined_raster_RGB_squared_error'] = ratio(item['RGB']['squared_error'], total)
    return report


def replay_gates(recorded, recomputed, selected):
    require(set(recorded) == {'refined_pixels','oracle_replay','convergence'} and recorded['refined_pixels'] == selected
            and set(recorded['convergence']) == set(rgb.ARMS), 'Offline refinement selection differs')
    pairs = [(recorded['oracle_replay'], recomputed['oracle_replay'])]
    pairs.extend((recorded['convergence'][arm], recomputed['convergence'][arm]) for arm in rgb.ARMS)
    for a,b in pairs:
        require(set(a) == set(b) and a['passed'] is b['passed'] is True
                and a['maximum_rgb_difference'] == b['maximum_rgb_difference']
                and np.isfinite(a['relative_l2']) and np.isclose(a['relative_l2'], b['relative_l2'], atol=1e-15, rtol=1e-12),
                'Offline convergence replay differs')


def figure(views, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(views), figsize=(11,7), layout='constrained')
    for j,(camera, view) in enumerate(views.items()):
        a,b = (view['images'][arm][1,:,2] for arm in rgb.ARMS)
        known = view['known']
        shape = (rgb.geo.SIZE, rgb.geo.SIZE)
        for i,value in enumerate((a,b)):
            preview = linear_to_srgb(np.clip(value,0,1))
            if i == 1:
                preview[~known] = (255,0,190)
            axes[i,j].imshow(preview.reshape(*shape,3))
        axes[0,j].set_title(str(camera))
        delta = (b-a)@LUMA
        delta[~known] = np.nan
        signed = axes[2,j].imshow(delta.reshape(shape), cmap='coolwarm', vmin=-.035, vmax=.035)
        coarse = (view['images']['reconstruction'][0,:,2]-view['images']['oracle'][0,:,2])@LUMA
        error = axes[3,j].imshow(abs(delta-coarse).reshape(shape), cmap='magma', vmin=0, vmax=.003)
    for row,label in zip(axes, ('Oracle geometry', 'Reconstruction', 'Luminance difference', 'Integration difference')):
        row[0].set_ylabel(label)
        for ax in row:
            ax.set(xticks=[], yticks=[])
    fig.colorbar(signed, ax=axes[2], extend='both')
    fig.colorbar(error, ax=axes[3], extend='max')
    fig.suptitle('Fixed-material Donut comparison · magenta marks undefined reconstruction')
    fig.savefig(output/'rgb-differences.png', dpi=150)
    plt.close(fig)


def verify(sha, output):
    f = rgb.freeze(sha)
    batch = Path(f['host_output'])
    root = batch/'render'
    a = batch/'authorization.json'
    require(output.resolve() == output and not output.exists() and batch not in output.parents,
            'Fresh offline destination required')
    r = rgb.geo.h.read(root/'receipt.json')
    require((batch/'exit-code.txt').read_text().strip() == '0' and r['status'] == 'complete' and r['cleanup_verified']
            and r['backend'] == f['backend'] and r['freeze_sha256'] == digest(batch/'freeze.json') == sha
            and digest(a) == r['authorization_sha256'] == digest(batch.with_name(batch.name+'-authorization.json'))
            and rgb.geo.h.read(a) == rgb.authorization(f, r['commit'], sha)
            and r['gate2_receipt_sha256'] == f['gate2_receipt']['sha256']
            and r['primary_rays'] == r['native_model_calls'] == r['checkpoint_loads'] == r['optimizer_updates'] == 0,
            'Completed bound fixed-material receipt required')
    _, data = rgb.admit(f, rgb.paths(f))
    reviewed = set()
    point_directions = total = 0
    outliers = []
    for camera, view in data.items():
        def fetch(arm, selected, grid, prefix):
            nonlocal point_directions
            name = capture_name(camera, arm, prefix, grid)
            require(name not in reviewed, 'Duplicate offline capture')
            capture = numeric(rgb.geo.h.check_artifact(root/name, r['outputs'][name]))
            reviewed.add(name)
            point_directions += len(selected['pixel_indices'])*grid**2
            return capture
        states, selected = rgb.paired(view['views'], fetch, total)
        total += len(selected)
        images, gates = rgb.raster(view, states)
        replay_gates(r['views'][str(camera)], gates, selected)
        view['images'] = images
        fine_o, fine_r = (images[arm][1,:,2] for arm in rgb.ARMS)
        for row in view['outliers']:
            p = int(row['pixel'])
            valid = bool(view['known'][p])
            outliers.append(dict(row, oracle_R=float(fine_o[p,0]), oracle_G=float(fine_o[p,1]), oracle_B=float(fine_o[p,2]),
                                 reconstruction_R=float(fine_r[p,0]) if valid else None,
                                 reconstruction_G=float(fine_r[p,1]) if valid else None,
                                 reconstruction_B=float(fine_r[p,2]) if valid else None,
                                 RGB_squared_error=float(np.square(fine_r[p]-fine_o[p]).sum()) if valid else None))
        view['preview_clipped_pixels'] = {arm: int(((im[1,:,2] > 1).any(1) & (view['known'] if arm == 'reconstruction' else True)).sum())
                                          for arm, im in images.items()}
    require(reviewed == set(r['outputs']) == {p.name for p in root.glob('*.npz')}
            and total == r['refined_pixels_total'] and set(r['views']) == {str(c) for c in data}
            and set(r['counters']) == set(rgb.LIMITS) and rgb.BASE_COST <= point_directions <= min(rgb.LIMITS.values())
            and all(n == point_directions for n in r['counters'].values()), 'Offline capture or work count differs')
    output.mkdir()
    json_new(output/'rgb-metrics.json', dict(protocol=rgb.PROTOCOL, status='verified_complete', receipt_sha256=digest(root/'receipt.json'),
        camera_metrics={str(c): summarize([v]) for c,v in data.items()},
        split_metrics={k: summarize([data[c] for c in cams]) for k,cams in rgb.geo.SPLITS.items()},
        pooled=summarize(list(data.values())), numerical_gates=r['views'], point_directions=point_directions,
        preview_clipped_pixels={c: v['preview_clipped_pixels'] for c,v in data.items()},
        provenance='historical, unresolved', interpretation='fixed-material geometry-induced discrepancy'))
    with (output/'depth-outlier-rgb.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, list(outliers[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(outliers)
    figure(data, output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    verify(args.freeze_sha256, args.output)
