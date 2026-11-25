"""Verify saved Donut geometry evidence and produce split metrics and geometry-only previews."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_reconstructed_geometry_v1 as g
from benchmarks.donut_recovery_v1 import json_new
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def statistics(values):
    require(np.isfinite(values).all(), 'Nonfinite geometry statistic')
    if not len(values):
        return None
    return dict(zip(('min', 'median', 'p95', 'max'), map(float, np.quantile(values, [0, .5, .95, 1]))))


def metrics(raw, c):
    o, r = (raw[key+'_hit'] for key in ('oracle', 'reconstruction'))
    both = o & r
    n_ok = both & raw['reconstruction_normal_valid']
    oracle_facing = -(raw['oracle_smooth_normal']*raw['direction']).sum(1)
    recon_facing = -(raw['reconstruction_normal']*raw['direction']).sum(1)
    cosine = (raw['oracle_smooth_normal'][n_ok]*raw['reconstruction_normal'][n_ok]).sum(1)
    pixel = c['pixel']
    np.testing.assert_array_equal(pixel, np.flatnonzero(r))
    expected = ~np.logical_or.reduce([c[key] for key in g.REASONS])
    np.testing.assert_array_equal(expected, c['valid'])
    np.testing.assert_array_equal(c['label_rows'], np.flatnonzero(expected))
    np.testing.assert_array_equal(c['valid'], c['reason_code'] == 0)
    require(c['search_completed'].all() and (c['visibility_tested'] | c['visibility_reused']).sum() == (c['forward_face'] >= 0).sum(),
            'Incomplete saved correspondence search or visibility inventory')
    q = c.get('transported_world_Q', np.empty((0, 3, 3)))
    if len(q):
        np.testing.assert_allclose(c['transported_widths'], c['labels_widths_major_minor'], atol=g.TOL['transport_width'], rtol=0)
        np.testing.assert_allclose(np.einsum('nij,nj->ni', q, raw['reconstruction_normal'][pixel[expected]]), 0,
                                   rtol=0, atol=g.TOL['transport_width'])
        np.testing.assert_allclose(c['total_delta_Q'], c['sampling_delta_Q']+c['frame_delta_Q'], rtol=0, atol=2e-15)
    out = dict(both_hit=int(both.sum()), oracle_only=int((o & ~r).sum()), reconstruction_only=int((r & ~o).sum()),
               both_miss=int((~r & ~o).sum()), undefined_reconstruction_normal=int((r & ~raw['reconstruction_normal_valid']).sum()),
               reconstruction_grazing=int((r & (recon_facing > 0) & (recon_facing <= g.TOL['grazing_cosine'])).sum()),
               oracle_grazing=int((o & (oracle_facing > 0) & (oracle_facing <= g.TOL['grazing_cosine'])).sum()),
               reconstruction_nonpositive_facing=int((r & raw['reconstruction_normal_valid'] & (recon_facing <= 0)).sum()),
               oracle_nonpositive_facing=int((o & (oracle_facing <= 0)).sum()), near_degenerate=int(raw['near_degenerate'].sum()),
               valid_common=int((o[pixel] & expected).sum()), valid_reconstruction_only=int((~o[pixel] & expected).sum()),
               orientation_eligible=int(c.get('orientation_eligible', np.empty(0, bool)).sum()),
               reason_any={key: int(c[key].sum()) for key in g.REASONS},
               reason_primary={key: int((c['reason_code'] == i).sum()) for i, key in enumerate(g.REASONS, 1)})
    dist = np.linalg.norm(raw['reconstruction_position'][both]-raw['oracle_position'][both], axis=1)
    vectors = dict(signed_depth=raw['reconstruction_depth'][both]-raw['oracle_depth'][both], displacement=dist,
                   normal_angle_degrees=np.degrees(np.arccos(np.clip(cosine, -1, 1))),
                   oracle_view_degrees=np.degrees(np.arccos(np.clip(oracle_facing[both], -1, 1))),
                   reconstruction_view_degrees=np.degrees(np.arccos(np.clip(recon_facing[n_ok], -1, 1))),
                   projection_distance=c['projection_distance'][expected], roundtrip_distance=c['roundtrip_distance'][expected])
    for key in ('sampling', 'frame', 'total'):
        value = c.get(key+'_delta_Q', np.empty((0, 3, 3)))
        vectors[key+'_relative_Q_shift'] = np.linalg.norm(value, axis=(1, 2))/c.get('source_Q_frobenius', np.empty(0))
    vectors['orientation_shift_degrees'] = c.get('orientation_shift_degrees', np.empty(0))[c.get('orientation_eligible', np.empty(0,bool))]
    widths = c.get('sampling_signed_widths', np.empty((0, 2)))
    vectors['sampling_signed_major_width_shift'], vectors['sampling_signed_minor_width_shift'] = widths.T
    return out, vectors


def aggregate(rows):
    counts = {key: sum(row[0][key] for row in rows) for key in rows[0][0] if not isinstance(rows[0][0][key], dict)}
    for group in ('reason_any', 'reason_primary'):
        counts[group] = {key: sum(r[0][group][key] for r in rows) for key in g.REASONS}
    common = counts['both_hit']
    union = common+counts['oracle_only']+counts['reconstruction_only']
    ratio = lambda n,d: n/d if d else None
    return dict(counts=counts, silhouette_IoU=ratio(common,union),
                coverage_common=ratio(counts['valid_common'],common),
                coverage_reconstruction_only=ratio(counts['valid_reconstruction_only'],counts['reconstruction_only']),
                coverage_oracle_only=ratio(0,counts['oracle_only']),
                distributions={key: statistics(np.concatenate([row[1][key] for row in rows])) for key in rows[0][1]})


def verify(sha, output):
    f = g.freeze(sha)
    batch = Path(f['host_output'])
    root = batch/'audit'
    require(not output.exists() and output.resolve() == output and batch not in output.parents, 'Fresh offline destination required')
    r = g.h.read(root/'receipt.json')
    a = batch/'authorization.json'
    require((batch/'exit-code.txt').read_text().strip() == '0' and r['status'] == 'complete' and r['cleanup_verified']
            and r['backend'] == f['backend'] and set(r['gates']) == {str(c) for c in g.PANEL} and all(r['gates'].values())
            and r['view_products'] == g.PANEL and r['freeze_sha256'] == digest(batch/'freeze.json') == sha
            and digest(a) == r['authorization_sha256'] == digest(batch.with_name(batch.name+'-authorization.json'))
            and g.h.read(a) == g.authorization(f, r['commit'], sha)
            and r['renderer_calls'] == r['optimizer_updates'] == r['checkpoint_loads'] == 0,
            'Completed bound geometry receipt required')
    require(set(r['counters']) == set(g.LIMITS) and all(0 <= r['counters'][key] <= cap for key, cap in g.LIMITS.items()),
            'Invalid work counters')
    names = {f'{kind}-{c}.{ext}' for c in g.PANEL
             for kind, ext in (('primary', 'npz'), ('witnesses', 'json'), ('correspondence', 'npz'))}
    require(names == set(r['outputs']) == {p.name for p in root.iterdir()}-{'receipt.json'},
            'Geometry evidence inventory differs')
    for name, item in r['outputs'].items():
        g.h.check_artifact(root/name, item)
    _, _, views = g.admit(f, g.paths(f))
    data, images = {}, {}
    counted = dict.fromkeys(('primary_rays', 'visibility_rays', 'cpu_ray_checks', 'nearest_searches'), 0)
    for camera in g.PANEL:
        with np.load(root/f'primary-{camera}.npz', allow_pickle=False) as z:
            raw = {key: z[key] for key in z.files}
        with np.load(root/f'correspondence-{camera}.npz', allow_pickle=False) as z:
            c = {key: z[key] for key in z.files}
        np.testing.assert_array_equal(views[camera]['hit_mask'].ravel(), raw['oracle_hit'])
        witnesses = g.h.read(root/f'witnesses-{camera}.json')['witnesses']
        require(set(witnesses) == {'oracle', 'reconstruction'}, 'Expected both primary witness sets')
        for w in witnesses.values():
            np.testing.assert_array_equal([v['pixel'] for v in w], g.sampled_pixels(raw['oracle_hit']))
            require(all(v['passed'] for v in w), 'Failed native witness')
            counted['cpu_ray_checks'] += len(w)
        counted['primary_rays'] += 2*len(raw['oracle_hit'])
        counted['nearest_searches'] += int(raw['reconstruction_hit'].sum())+int((c['forward_face'] >= 0).sum())
        counted['visibility_rays'] += int(c['visibility_tested'].sum())
        data[camera] = metrics(raw, c)
        images[camera] = raw, c
    require(all(value == r['counters'][key] for key, value in counted.items()), 'Recorded work differs from saved ray arrays')
    output.mkdir()
    json_new(output/'geometry-metrics.json', dict(protocol=g.PROTOCOL, receipt_sha256=digest(root/'receipt.json'),
        split_metrics={k: aggregate([data[c] for c in cameras]) for k, cameras in g.SPLITS.items()},
        camera_metrics={str(c): aggregate([row]) for c, row in data.items()}, pooled=aggregate(list(data.values())),
        historical_geometry_provenance='unresolved'))
    figure(images, output)


def figure(images, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(4, len(g.PANEL), figsize=(11, 7), layout='constrained')
    for j, (camera, (raw, c)) in enumerate(images.items()):
        o, r = raw['oracle_hit'], raw['reconstruction_hit']
        code = o.astype(int)+2*r
        coverage = np.zeros(len(o))
        coverage[o & ~r] = 1
        coverage[c['pixel']] = np.where(c['valid'], 3, 2)
        shape = (g.SIZE, g.SIZE)
        axes[0,j].imshow(code.reshape(shape), cmap=ListedColormap(['#151923','#ffb000','#de4968','#dce6e4']), vmin=0, vmax=3)
        axes[1,j].imshow(coverage.reshape(shape), cmap=ListedColormap(['#151923','#ffb000','#de4968','#5ed3a3']), vmin=0, vmax=3)
        common = o & r
        valid = common & raw['reconstruction_normal_valid']
        d, angle = np.full(len(o), np.nan), np.full(len(o), np.nan)
        d[common] = np.linalg.norm(raw['reconstruction_position'][common]-raw['oracle_position'][common], axis=1)
        dot = (raw['oracle_smooth_normal'][valid]*raw['reconstruction_normal'][valid]).sum(1)
        angle[valid] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        depth_map = axes[2,j].imshow(d.reshape(shape), cmap='magma', vmin=0, vmax=.01)
        normal_map = axes[3,j].imshow(angle.reshape(shape), cmap='viridis', vmin=0, vmax=15)
        axes[0,j].set_title(str(camera))
    for row, name in zip(axes, ('Silhouette', 'Transfer', 'Displacement', 'Normal error')):
        row[0].set_ylabel(name)
        for ax in row:
            ax.set(xticks=[], yticks=[])
    fig.colorbar(depth_map, ax=axes[2], label='native units', extend='max')
    fig.colorbar(normal_map, ax=axes[3], label='degrees', extend='max')
    fig.suptitle('Donut geometry · yellow oracle-only · pink reconstruction-only or invalid transfer')
    fig.savefig(output/'geometry-diagnostics.png', dpi=145)
    plt.close(fig)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    verify(args.freeze_sha256, args.output)
