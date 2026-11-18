"""Saved-data coverage and post-run attribution. This entry point never imports Torch."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_specular_d10 as d
from benchmarks.donut_recovery_v1 import SIZE, SPLITS, json_new, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def coverage(views, ids, ring):
    from scipy.spatial import cKDTree

    radii = np.array(d.POLICY['coverage_radii'])
    normal_cos = np.cos(np.deg2rad(d.POLICY['coverage_normal_degrees']))
    view = views[d.POLICY['camera']]
    p, n, triangle = (view[k][ids] for k in ('position', 'normals', 'triangle'))
    supported = np.zeros((len(ids), len(radii)), dtype=int)
    highlighted = np.zeros(len(ids), dtype=int)
    same_face = np.zeros(len(ids), dtype=int)
    peak_specular = np.zeros(len(ids))
    per_camera = {}
    for camera in SPLITS['fitting']:
        fit = views[camera]
        valid = fit['shader_valid']
        pf, nf = (fit[k][valid] for k in ('position', 'normals'))
        spec = fit['linear_components'][valid, 1] @ d.LUMA
        highlight = (spec >= d.POLICY['highlight_min_specular_luminance']) & (
            spec >= d.POLICY['highlight_min_view_median_multiple']*np.median(spec))
        require(np.array_equal(highlight, fit['highlight_mask'][valid]), 'Admitted highlight mask changed')
        same_face += np.isin(triangle, fit['triangle'][valid])
        nearby = cKDTree(pf).query_ball_point(p, radii[-1])
        supports = np.zeros(len(ids), dtype=bool)
        highlights = np.zeros(len(ids), dtype=bool)
        for i, candidates in enumerate(nearby):
            mask = np.asarray(candidates, dtype=int)
            mask = mask[(nf[mask] @ n[i]) >= normal_cos]
            if not len(mask):
                continue
            distance = np.linalg.norm(pf[mask]-p[i], axis=-1)
            supported[i] += (distance.min() <= radii)
            close = mask[distance <= radii[1]]
            if len(close):
                supports[i] = True
                highlights[i] = highlight[close].any()
                peak_specular[i] = max(peak_specular[i], spec[close].max())
        highlighted += highlights
        per_camera[str(camera)] = {
            region: dict(matches=int(supports[mask].sum()), highlighted_matches=int(highlights[mask].sum()))
            for region, mask in [('ring', ring), ('outside_ring', ~ring)]
        }
    nviews = len(SPLITS['fitting'])+1
    return dict(camera=d.POLICY['camera'], fitting=SPLITS['fitting'], radii_native=radii.tolist(),
                minimum_normal_cosine=float(normal_cos), highlight_radius_native=float(radii[1]),
                highlight_definition='Saved specular luminance >= .01 and >= 3 times the fitting-view median',
                interpretation='Nearby saved rays and same-face hits are coverage proxies. They do not measure point visibility or inversion conditioning.',
                per_camera=per_camera, groups={
                    region: dict(points=int(mask.sum()),
                                 nearby_camera_count=[np.bincount(a, minlength=nviews).tolist() for a in supported[mask].T],
                                 same_triangle_camera_count=np.bincount(same_face[mask], minlength=nviews).tolist(),
                                 highlight_camera_count=np.bincount(highlighted[mask], minlength=nviews).tolist(),
                                 median_peak_nearby_specular_luminance=float(np.median(peak_specular[mask])),
                                 median_peak_specular_when_supported=float(np.median(peak_specular[mask & (supported[:, 1] > 0)]))
                                 if (mask & (supported[:, 1] > 0)).any() else None)
                    for region, mask in [('ring', ring), ('outside_ring', ~ring)]
                })


def capture(path, receipt, role):
    name = role+'.npz'
    draws = numeric(d.checked(path/name, receipt['outputs'][name]))['component_draws']
    require(draws.shape == (d.DRAWS, d.POINTS, 3, 3) and draws.dtype == np.float32
            and (draws >= 0).all()
            and np.array_equal(draws[:, :, 0]+draws[:, :, 1], draws[:, :, 2]), 'Invalid D10 linear draw block')
    return draws


def plot(output, pixels, ring, ref, means):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from benchmarks.bell_e11 import linear_to_srgb

    def image(values):
        canvas = np.zeros((SIZE*SIZE, *values.shape[1:]))
        canvas[pixels] = values
        return canvas.reshape((SIZE, SIZE, *values.shape[1:]))

    labels = {'Oracle': 'Independent oracle', 'R1': 'R1 recovered',
              'R2': 'R2 oracle Q', 'R3': 'R3 oracle light', 'R4': 'R4 oracle Q + light',
              'R5': 'R5 fully oracle native', 'width_only': 'Oracle widths',
              'orientation_only': 'Oracle orientation'}
    estimates = dict(Oracle=ref, **means)
    errors = [(a[:, 2]-ref[:, 2])@d.LUMA for a in estimates.values()]
    scale = max(max(float(abs(a).max()) for a in errors), 1e-6)
    maximum_sse = max(max(float(np.sum((a[:, 2]-ref[:, 2])**2, axis=-1).max())
                          for a in estimates.values()), 1e-12)
    fig, axes = plt.subplots(5, len(estimates), figsize=(2.4*len(estimates), 11), squeeze=False,
                             layout='constrained')
    for i, (name, a) in enumerate(estimates.items()):
        axes[0, i].set_title(labels[name], fontsize=9)
        for row, component in enumerate((2, 1, 0)):
            axes[row, i].imshow(linear_to_srgb(np.clip(image(a[:, component]), 0, 1)))
        diff = axes[3, i].imshow(image(errors[i]), cmap='RdBu_r', vmin=-scale, vmax=scale)
        squared = axes[4, i].imshow(image(np.sum((a[:, 2]-ref[:, 2])**2, axis=-1)),
                                    cmap='magma', vmin=0, vmax=maximum_sse)
        for row in (3, 4):
            axes[row, i].contour(image(ring), levels=[.5], colors='gray', linewidths=.4)
    for row, label in enumerate(('RGB', 'Specular', 'Diffuse', 'Signed luminance error', 'RGB squared error')):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.set(xticks=[], yticks=[])
    fig.colorbar(diff, ax=axes[3, :].tolist(), fraction=.03)
    fig.colorbar(squared, ax=axes[4, :].tolist(), fraction=.03)
    fig.savefig(output/'camera28-attribution.png', dpi=160)
    plt.close(fig)


def analyze(plan, batch, output, selected, view, coverage_report):
    expected_batch = Path(plan['host_output'])
    require(batch == expected_batch and batch.resolve() == batch, 'Expected the approved D10 batch')
    freeze_path = batch/'freeze.json'
    freeze_id = digest(d.ROOT/d.FREEZE)
    require(digest(freeze_path) == freeze_id and (batch/'exit-code.txt').read_text().strip() == '0',
            'Run freeze or foreground exit differs')
    auth_path = batch/'authorization.json'
    auth = d.read(auth_path)
    require(auth == d.authorization(plan, auth['commit'], freeze_id), 'Run authorization differs')
    run = batch/'run'
    receipt_path = run/'render-receipt.json'
    receipt = d.read(receipt_path)
    require(receipt['status'] == 'complete' and receipt['cleanup_verified']
            and receipt['RNG_modes_and_weights_unchanged'] is True
            and receipt['commit'] == auth['commit'] and receipt['freeze_sha256'] == freeze_id
            and receipt['authorization_sha256'] == digest(auth_path)
            and receipt['image'] == d.IMAGE and receipt['protocol'] == d.PROTOCOL
            and receipt['checkpoint_sha256'] == auth['trusted_checkpoint_sha256']
            and receipt['optimizer_returns'] == receipt['primary_rays'] == 0, 'Incomplete D10 execution receipt')
    panel = numeric(d.checked(run/'panel.npz', receipt['outputs']['panel.npz']))
    ids, pixels, ring, archive, archive_se = selected
    eligible = panel['orientation_eligible']
    require(np.array_equal(panel['pixels'], pixels) and np.array_equal(panel['ring'], ring)
            and eligible.dtype == np.bool_ and eligible.shape == pixels.shape, 'D10 panel identity differs')
    ref = view['linear_components'][ids]
    expected_context = receipt['conditional_split_light']
    roles = d.BASE + (d.SWAPS if expected_context is not None else ())
    require(set(receipt['outputs']) == {'panel.npz', 'R1-archived-replay.npz', *(role+'.npz' for role in roles)}
            and receipt['shader_calls'] == len(roles)*d.DRAWS*((d.POINTS+d.CHUNK-1)//d.CHUNK)
            and receipt['shadow_slots'] == len(roles)*d.DRAWS*d.POINTS*1536, 'Incomplete D10 work or output inventory')
    first = numeric(d.checked(run/'R1-archived-replay.npz', receipt['outputs']['R1-archived-replay.npz']))
    masks = dict(foreground=np.ones(len(ids), dtype=bool), ring=ring, outside_ring=~ring)
    if expected_context is not None:
        masks.update(orientation_eligible=eligible, ring_orientation_eligible=ring & eligible)
    measures = {name: {} for name, mask in masks.items() if mask.any()}
    means = {}
    for role in roles:
        values = capture(run, receipt, role)
        if role == 'R1':
            require(np.array_equal(first['component_draws'], values[:d.ARCHIVED_DRAWS])
                    and d.verify_replay(values[:d.ARCHIVED_DRAWS], archive, archive_se) ==
                    receipt['archived_replay'] and receipt['archived_replay']['passed'], 'Offline R1 replay differs')
        means[role] = values.mean(axis=0, dtype=np.float64)
        for name, stats in measures.items():
            stats[role] = d.draw_summary(values, ref, masks[name])
        if role == 'R5':
            native_se = values.std(axis=0, ddof=1, dtype=np.float64)/np.sqrt(d.DRAWS)
    require(d.split_context(measures['ring']) == expected_context, 'Offline conditional decision differs')

    weights = dict(Q_at_recovered_light=dict(R2=1, R1=-1), Q_at_oracle_light=dict(R4=1, R3=-1),
                   light_at_recovered_Q=dict(R3=1, R1=-1), light_at_oracle_Q=dict(R4=1, R2=-1),
                   interaction=dict(R4=1, R3=-1, R2=-1, R1=1), oracle_scalar_change=dict(R5=1, R4=-1))
    if expected_context is not None:
        anchor = 'R1' if expected_context == 'recovered' else 'R3'
        weights.update(width_only={'width_only': 1, anchor: -1},
                       orientation_only={'orientation_only': 1, anchor: -1})
    grid_delta = abs(view['coarse_components'][ids]-ref)
    delta = means['R5']-ref
    allowance = .001+grid_delta+6*native_se+64*np.finfo(np.float32).eps
    compatibility = {region: dict(
        native_oracle_RMSE=np.sqrt(np.mean(np.square(delta[mask]), axis=(0, 2))).tolist(),
        empirical_MC_SE_RMS=np.sqrt(np.mean(np.square(native_se[mask]), axis=(0, 2))).tolist(),
        independent_grid_delta_RMS=np.sqrt(np.mean(np.square(grid_delta[mask]), axis=(0, 2))).tolist(),
        channels_beyond_descriptive_allowance=(abs(delta[mask]) > allowance[mask]).sum(axis=(0, 2)).tolist())
        for region, mask in masks.items() if mask.any()}
    report = dict(protocol=d.PROTOCOL, camera=d.POLICY['camera'], native=receipt,
                  component_order=list(d.COMPONENTS), draw_uncertainty='Paired whole-draw leave-one-out jackknife with t(63) intervals; excludes fixed diffuse quadrature, reference and spatial sampling uncertainty.',
                  sign_convention='Intervened minus baseline; negative SSE effect is an improvement.',
                  coverage=coverage_report, conditional_light=expected_context, groups={},
                  compatible_oracle_diagnostic=dict(absolute_allowance=.001, SE_multiplier=6,
                      interpretation='Descriptive channel exceedance, using grid discrepancy as a proxy, without simultaneous coverage.',
                      regions=compatibility),
                  execution_artifacts={str(f): dict(bytes=f.stat().st_size, sha256=digest(f))
                      for f in (freeze_path, auth_path, receipt_path, batch/'console.txt', batch/'exit-code.txt',
                                *(run/name for name in receipt['outputs']))})
    for region, stats in measures.items():
        measurements = {}
        for role, fields in stats.items():
            residual = (means[role]-ref)[masks[region]]
            mixed = 2*np.sum(residual[:, 0]*residual[:, 1])
            measurements[role] = dict(
                **{metric: d.interval(*args) for metric, args in fields.items()},
                linear_RMSE=np.sqrt(fields['SSE'][0]/(3*len(residual))).tolist(),
                diffuse_specular_cross_term=float(mixed))
        report['groups'][region] = dict(points=int(masks[region].sum()), measurements=measurements,
            effects={label: {metric: d.effect(stats, metric, values) for metric in stats['R1']}
                     for label, values in weights.items()})
    output.mkdir()
    json_new(output/'analysis.json', report)
    plot(output, pixels, ring, ref, means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=Path, help='Omit for coverage-only preflight')
    args = parser.parse_args()
    plan = d.freeze(d.ROOT/d.FREEZE, args.freeze_sha256)
    _, views, _, selected = d.panel(plan, {role: Path(item['host']) for role, item in plan['artifacts'].items()})
    summary = coverage(views, selected[0], selected[2])
    if args.batch is None:
        json_new(args.output, summary)
    else:
        analyze(plan, args.batch, args.output, selected, views[d.POLICY['camera']], summary)


if __name__ == '__main__':
    main()
