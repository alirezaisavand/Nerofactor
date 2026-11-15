"""File-only E11G admission and matched multiview comparisons; no scientific queries."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import bell_multiview_e11g as study
from experiments.analyze_bell_anisotropy_e11f import load_npz
from experiments.analyze_bell_highlight_e11d import record, replay_equal, response_fields, verify_layout
from experiments.bell_highlight_e11d import write_json, save_figure
from infrastructure.signatures import require
from infrastructure.stage_sources import digest, baseline, PINS
from infrastructure.verified_ray_backend import identity

ROOT = study.ROOT
COMMIT = 'b74edd6c1840c166248851d0e27a3a0319a607df'
FREEZE_SHA = 'c7486b2ce23836ba03abc1edccf3cc3c649c2f9364f9b6960d940814561eeba7'
REPORT_SHA = 'db1c0b3d5ce630501645d07612f940e46df6bfa77e551d839882be8503205e09'
ARMS = ('original', 'stronger')


def verify_geometry(data, g, plan, pixels):
    names = {'camera_ids', 'pixel_indices', 'camera_centers', 'outgoing_directions', 'camera_distances',
             'facing', 'uv', 'in_frame', 'geometric', 'queried', 'valid', 'occluded', 'unresolved', 'nearest_depth'}
    require(set(data) == names, 'Outgoing geometry inventory mismatch')
    ids = plan['camera_ids']
    require(data['camera_ids'].dtype == data['pixel_indices'].dtype == np.int64 and
            np.array_equal(data['camera_ids'], ids) and np.array_equal(data['pixel_indices'], pixels), 'Camera/probe IDs changed')
    errors = {}
    for name, key in [('camera_centers', 'centers'), ('outgoing_directions', 'directions'),
                      ('camera_distances', 'distance'), ('facing', 'facing'), ('uv', 'uv')]:
        a, b = data[name], g[key][ids]
        require(a.shape == b.shape and a.dtype == np.float64 and np.isfinite(a).all() and
                np.allclose(a, b, rtol=1e-12, atol=1e-12), 'Saved camera geometry changed: '+name)
        errors[name] = float(abs(a-b).max())
    for name in ('in_frame', 'geometric', 'queried', 'valid', 'occluded', 'unresolved'):
        require(data[name].shape == (4, 84) and data[name].dtype == np.bool_, 'Invalid correspondence mask')
    require(np.array_equal(data['in_frame'], g['in_frame'][ids]) and
            np.array_equal(data['geometric'], g['geometric'][ids]), 'Geometric eligibility changed')
    queried = g['geometric'][ids].copy(); queried[0] = False
    require(np.array_equal(data['queried'], queried) and int(queried.sum()) == 146, 'Visibility query selection changed')
    depth = data['nearest_depth']
    require(depth.shape == (4, 84) and depth.dtype == np.float64 and np.isnan(depth[~queried]).all(),
            'Unqueried depths fabricated')
    classified = study.classify_visibility(depth[queried], data['camera_distances'][queried])
    for name, key in [('valid', 'visible'), ('occluded', 'occluded'), ('unresolved', 'unresolved')]:
        mask = np.zeros((4, 84), dtype=np.bool_)
        if name == 'valid':
            mask[0] = True
        mask[queried] = classified[key]
        require(np.array_equal(mask, data[name]), 'Saved segment classification changed: '+name)
    coverage = [dict(camera_id=c, positive_facing=int((g['facing'][c] > 0).sum()),
        in_frame=int(g['in_frame'][c].sum()), queried=int(queried[v].sum()), valid=int(data['valid'][v].sum()),
        occluded=int(data['occluded'][v].sum()), unresolved=int(data['unresolved'][v].sum())) for v, c in enumerate(ids)]
    return coverage, errors


def verify_grid(data, valid, pixels, ids, cache):
    require(set(data) == {'components', 'valid', 'camera_ids', 'pixel_indices', 'completed_probes'}, 'Grid inventory mismatch')
    require(data['valid'].dtype == data['completed_probes'].dtype == np.bool_ and
            np.array_equal(data['valid'], valid) and data['completed_probes'].shape == (84,) and data['completed_probes'].all(),
            'Incomplete grid/correspondence mask')
    require(data['camera_ids'].dtype == data['pixel_indices'].dtype == np.int64 and
            np.array_equal(data['camera_ids'], ids) and np.array_equal(data['pixel_indices'], pixels), 'Grid IDs changed')
    a = data['components']
    require(a.shape == (2, 4, 84, 12, 3, 3) and a.dtype == np.float64 and np.isnan(a[:, ~valid]).all(),
            'Invalid grid shape/dtype or fabricated invalid views')
    b = a[:, valid]
    require(np.isfinite(b).all() and (b >= 0).all() and np.array_equal(b[..., 0, :]+b[..., 1, :], b[..., 2, :]),
            'Invalid measured radiance or component sum')
    for arm, name in enumerate(('original', 'stronger')):
        require(np.array_equal(a[arm, 0, :, :8], cache[name]), 'Accepted single-view baseline changed')
        measured = a[arm, valid]
        require(np.array_equal(measured[:, :, 0], np.repeat(measured[:, :1, 0], 12, axis=1)), 'Diffuse changed across cases')
    require(np.array_equal(a[0, valid, 7], a[1, valid, 7]) and
            np.array_equal(a[0, valid, :, 0], a[1, valid, :, 0]), 'Shared isotropic/diffuse controls changed')
    return a


def matched_statistics(comparison):
    rows = []
    for arm, label in enumerate(ARMS):
        for n in ('at_least_two', 1, 2, 3, 4):
            source = [r for r in comparison['per_probe_conditioning'] if r['material'] == arm and
                      (r['valid_view_count'] >= 2 if n == 'at_least_two' else r['valid_view_count'] == n)]
            for basis in ('legacy_scaled', 'theta_logmean_logratio'):
                v = [r[basis] for r in source]
                row = dict(material=label, valid_view_group=n, basis=basis, probes=len(v))
                for which in ('single', 'stacked_RMS'):
                    conditions = [r[which]['condition_number'] for r in v if r[which]['condition_number'] is not None]
                    row[which+'_finite_conditions'] = len(conditions)
                    row[which+'_median_condition'] = float(np.median(conditions)) if conditions else None
                    row[which+'_p95_condition'] = float(np.quantile(conditions, .95)) if conditions else None
                    row[which+'_median_smin'] = float(np.median([r[which]['singular_values'][-1] for r in v])) if v else None
                row['positive_grid_proxy_lower_count'] = sum(r['grid_proxy_smin_lower'] > 0 for r in v)
                rows.append(row)
    return rows


def response_tables(fine, coarse, valid, pixels, ids):
    details = []; summary = []
    for arm, label in enumerate(ARMS):
        all_fields = {}
        for view, camera in enumerate(ids):
            indices = np.flatnonzero(valid[view]); a = fine[arm, view, indices]; b = coarse[arm, view, indices]
            if not len(indices):
                continue
            fields = response_fields(a[:, :8], b[:, :8])
            for name, pair in [('logmean', (8, 9)), ('logratio', (10, 11))]:
                order = [0, *pair, 3, 4, 5, 6, 7]
                fields[name] = response_fields(a[:, order], b[:, order])['orientation']
            for name, values in fields.items():
                if name not in all_fields:
                    all_fields[name] = {k: np.zeros((4, 84), dtype=value.dtype) if value.dtype == np.bool_
                                       else np.full((4, 84), np.nan) for k, value in values.items()}
                for k, value in values.items():
                    all_fields[name][k][view, indices] = value
                for j, index in enumerate(indices):
                    details.append(dict(material=label, camera_id=camera, probe_index=int(index), pixel_index=int(pixels[index]),
                                        factor=name, **{k: v[j].item() for k, v in values.items()}))
        matched = valid.sum(0) >= 2
        for name, fields in all_fields.items():
            union = fields['resolved'].any(0); maximum = np.nanmax(fields['amplitude'], axis=0)
            for group, mask in [('all84', np.ones(84, bool)), ('at_least_two', matched)]:
                summary.append(dict(material=label, factor=name, group=group, probes=int(mask.sum()),
                    single_resolved=int(fields['resolved'][0, mask].sum()), multiview_any_resolved=int(union[mask].sum()),
                    single_median_response=float(np.median(fields['amplitude'][0, mask])) if mask.any() else None,
                    multiview_max_median_response=float(np.median(maximum[mask])) if mask.any() else None,
                    single_two_codes=int(fields['code_gate'][0, mask].sum()),
                    multiview_any_two_codes=int(fields['code_gate'][:, mask].any(0).sum()),
                    single_zero_codes=int((fields['code_difference'][0, mask] == 0).sum()),
                    multiview_all_zero_codes=int(np.all((fields['code_difference'] == 0) | ~valid, axis=0)[mask].sum())))
    return details, summary


def analyze(batch, benchmark, pilot, reference, factor, output):
    roots = dict(benchmark=benchmark, pilot=pilot, reference=reference, factor=factor)
    require(all(p.is_absolute() and p.resolve() == p for p in (batch, output, *roots.values())), 'Explicit unaliased paths required')
    require(not output.exists() and all(p not in output.parents for p in (batch, *roots.values())), 'Fresh separate analysis output required')
    native = batch/'multiview'; rp = native/'multiview.json'
    require(digest(rp) == REPORT_SHA, 'Selected completed report changed')
    r = json.loads(rp.read_text()); launch = json.loads((batch/'launch.json').read_text())
    ap = Path(str(batch)+'-authorization.json'); a = json.loads(ap.read_text())
    fp = ROOT/'docs/protocols/BELL_MULTIVIEW_E11G_V1.freeze.json'; require(digest(fp) == FREEZE_SHA, 'Frozen source definition changed')
    f = json.loads(fp.read_text())
    require(r['status'] == 'complete' and r['cleanup_verified'] is True and (batch/'exit-code.txt').read_text().strip() == '0', 'Incomplete work/cleanup/exit')
    require(r['protocol'] == a['protocol'] == f['protocol'] == study.PROTOCOL and
            r['commit'] == launch['commit'] == a['commit'] == COMMIT and
            r['freeze_sha256'] == launch['freeze_sha256'] == a['freeze_sha256'] == FREEZE_SHA and
            r['authorization_sha256'] == launch['authorization_sha256'] == digest(ap) and a['approved'] is True and
            launch['batch'] == a['host_output_directory'] == str(batch) and a['output_directory'] == '/outputs/multiview' and
            launch['gpu'] == a['selected_gpu'] == '0' and a['image_id'] == r['backend']['image_id'] == study.identity()['image_id'],
            'Launch/authorization identity mismatch')
    for name, path in roots.items():
        require(a['host_'+name] == launch[name] == str(path), 'Read-only input mount identity changed')
    require(r['source_files'] == f['files'] and r['limits'] == a['limits'] == f['limits'] == study.LIMITS and
            r['cases'] == f['cases'] == list(study.CASES) and r['lighting'] == f['lighting'] == study.LIGHT and
            r['geometry'] == f['geometry'] == study.GEOMETRY and r['backend'] == identity(), 'Scientific identity mismatch')
    for name, sha in f['files'].items():
        require(digest(ROOT/name) == sha, 'Frozen source changed: '+name)
    for stage in PINS:
        baseline(stage, PINS[stage])
    labels, wo, pixels, cache, geometry, consumed = study.inputs(**roots)
    plan = study.select_views(geometry); replay_equal(plan, f['view_plan']); replay_equal(plan, r['view_plan'])
    replay_equal(plan, a['view_plan'])
    mounts = dict(v3=benchmark, e11d=pilot, e11e=reference, e11f=factor); native_inputs = []
    for item in r['inputs']:
        p = Path(item['path']); require(p.parts[:2] == ('/', 'inputs') and p.parts[2] in mounts, 'Unexpected consumed input path')
        host = mounts[p.parts[2]].joinpath(*p.parts[3:])
        require(host.resolve() == host and mounts[p.parts[2]] in host.parents and digest(host) == item['sha256'] and
                host.stat().st_size == item['bytes'], 'Consumed artifact changed')
        native_inputs.append(record(host))
    require(native_inputs == consumed == a['immutable_inputs'], 'Consumed-input inventory mismatch')
    expected = {'comparison.json', 'grid-256.npz', 'grid-512.npz', 'outgoing-geometry.npz'}
    require(len(r['outputs']) == 4 and {i['path'] for i in r['outputs']} == expected, 'Native output inventory mismatch')
    for item in r['outputs']:
        p = native/item['path']; require(digest(p) == item['sha256'] and p.stat().st_size == item['bytes'], 'Native output changed')
    verify_layout(native, expected | {'progress.json', 'multiview.json'})
    g = load_npz(native/'outgoing-geometry.npz'); coverage, errors = verify_geometry(g, geometry, plan, pixels)
    replay_equal(coverage, r['view_coverage']); valid = g['valid']; ids = plan['camera_ids']
    grids = {size: verify_grid(load_npz(native/('grid-'+str(size)+'.npz')), valid, pixels, ids, cache[size]) for size in study.GRIDS}
    require(r['completed_grids'] == [256, 512] and r['any_additional_visible'] is bool(valid[1:].any()), 'Incomplete grid records')
    comparison = study.compare_views(grids[512], grids[256], valid)
    replay_equal(comparison, r['comparison']); replay_equal(comparison, json.loads((native/'comparison.json').read_text()))
    brdf = (8*84+23*int(valid[1:].sum()))*327680
    require(r['counters'] == dict(outgoing_visibility_attempts=146, outgoing_visibility_returns=146,
        brdf_attempts=brdf, brdf_returns=brdf, environment_attempts=27525120, environment_returns=27525120,
        new_primary_rays=0, new_incident_shadow_queries=0, optimizer_returns=0, model_calls=0) and brdf <= study.LIMITS['max_brdf_point_directions'],
        'Recorded scientific work differs from authorized exact formula')
    batch_bytes = sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    require(r['elapsed_seconds'] < 900 and r['host_peak_rss_bytes'] <= 64*1024**3 and batch_bytes <= study.LIMITS['output_bytes'], 'Recorded resource ceiling exceeded')
    stats = matched_statistics(comparison); detail, responses = response_tables(grids[512], grids[256], valid, pixels, ids)
    convergence = [dict(material=ARMS[row['material']], camera_id=ids[row['view_index']], valid_points=row['valid_points'],
                        case=case, **value) for row in comparison['convergence'] for case, value in row.get('cases', {}).items()]
    output.mkdir()
    result = dict(schema_version=1, protocol='bell-angular-diversity-e11g-offline-analysis-v1', verification_status='pass',
        decision='FIX_AND_PROCEED', full_dataset_admitted=False, native_report=record(rp), freeze=record(fp), authorization=record(ap),
        execution_receipts=[record(batch/p) for p in ('launch.json', 'exit-code.txt', 'console.log', 'container.cid')],
        native_outputs=[record(native/p) for p in sorted(expected)], input_records=consumed, source_files=f['files'],
        view_plan=plan, view_coverage=coverage, camera_geometry_replay_max_errors=errors,
        comparison=comparison, matched_statistics=stats, response_summary=responses, counters=r['counters'],
        runtime=r['runtime'], backend=r['backend'], elapsed_seconds=r['elapsed_seconds'], batch_bytes=batch_bytes,
        resources={k: r[k] for k in ('host_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')},
        previous_E11D_E11E_E11F_replaced=False, analysis_new_GPU_queries=0, analysis_new_BRDF_queries=0)
    write_json(output/'analysis.json', result)
    for name, rows in [('matched_conditioning', stats), ('response_summary', responses), ('probe_view_responses', detail), ('convergence_summary', convergence)]:
        with (output/(name+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    figures(output, stats, responses, valid, pixels)
    return result


def figures(output, stats, responses, valid, pixels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'e11g-analysis-v1'
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, basis in zip(axes, ('legacy_scaled', 'theta_logmean_logratio')):
        rows = [r for r in stats if r['valid_view_group'] == 'at_least_two' and r['basis'] == basis]
        for x, row in enumerate(rows):
            ax.bar(x-.18, row['single_median_condition'], .36, color='C0', label='single' if x == 0 else None)
            ax.bar(x+.18, row['stacked_RMS_median_condition'], .36, color='C1', label='multiview RMS' if x == 0 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(ARMS); ax.set_yscale('log'); ax.set_title(basis, fontsize=9); ax.legend(fontsize=8)
        ax.set_ylabel('Median condition number, same83 probes')
    fig.suptitle('Angular diversity separates local parameter responses'); fig.tight_layout(); save(fig, output, 'matched_conditioning')
    factors = ['orientation', 'major', 'minor', 'logmean', 'logratio', 'isotropic']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(len(factors))
    for ax, arm in zip(axes, ARMS):
        rows = [next(r for r in responses if r['material'] == arm and r['group'] == 'at_least_two' and r['factor'] == name) for name in factors]
        ax.bar(x-.18, [r['single_resolved'] for r in rows], .36, label='camera58')
        ax.bar(x+.18, [r['multiview_any_resolved'] for r in rows], .36, label='any admitted view')
        ax.axhline(.25*83, color='grey', linestyle=':', linewidth=.8, label='25% width screen')
        ax.set_xticks(x); ax.set_xticklabels(factors, rotation=35); ax.set_ylabel('Resolved matched probes /83'); ax.set_title(arm); ax.legend(fontsize=8)
    fig.suptitle('Better conditioning does not guarantee observable roughness contrasts'); fig.tight_layout(); save(fig, output, 'factor_coverage')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    sc = axes[0].scatter(pixels % 128, pixels//128, c=valid.sum(0), vmin=1, vmax=4, s=25, cmap='viridis')
    axes[0].set_xlim(0, 127); axes[0].set_ylim(127, 0); axes[0].set_aspect('equal'); axes[0].set_title('Valid views per original probe')
    fig.colorbar(sc, ax=axes[0], ticks=[1, 2, 3, 4])
    axes[1].bar([1, 2, 3, 4], [int((valid.sum(0) == n).sum()) for n in (1, 2, 3, 4)])
    axes[1].set_xticks([1, 2, 3, 4]); axes[1].set_xlabel('Valid view count'); axes[1].set_ylabel('Original probes /84')
    fig.suptitle('No visibility-based replacements; one probe retains only camera58'); fig.tight_layout(); save(fig, output, 'visibility_coverage')


def save(fig, output, name):
    import matplotlib.pyplot as plt
    save_figure(fig, output/(name+'.svg')); fig.savefig(output/(name+'.png'), dpi=130, bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'benchmark', 'pilot', 'reference', 'factor', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    result = analyze(**vars(p.parse_args()))
    print(json.dumps(dict(status=result['verification_status'], decision=result['decision'], matched_statistics=result['matched_statistics'][:2])))


if __name__ == '__main__':
    main()
