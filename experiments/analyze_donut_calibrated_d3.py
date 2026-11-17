"""Verify and compare the completed D1/D2/D3 runs from saved files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_donut_camera34_coverage import analyze as coverage_analysis
from analyze_donut_recovery_v2 import (
    BATCH, CAMERAS, COLOR, GENERATION, REFERENCE, ROOT, STEPS,
    SCALAR_REPLAY, digest, equal, numeric, read, record, scalar32, srgb, stats,
)

D3_BATCH = Path('/home/isavand/paper/evidence/donut-kurt-calibrated-d3-805284134859458fbc3da128c6e513bc')
FREEZE = 'ce32a272fb0e7f4ccf65420a858e2070bc16fcd778dc8fac19913596c53bf639'
AUTH = '93ba5ef21719a76a4da9c43be152b708351bf98770ad8a340de1ca33956915d5'
COMMIT = 'f116c8f4bfdbe27d8a1a06bb04997a744ab98482'
PRIOR = ROOT / 'docs/results/donut_kurt_recovery_v2'
ARMS = {'D1': BATCH / 'D1', 'D2': BATCH / 'D2', 'D3': D3_BATCH / 'D3'}
LABELS = {'D1': 'D1 known light', 'D2': 'D2 learned light', 'D3': 'D3 calibrated light'}
PHYSICAL = ('q', 'orientation_degrees', 'major_abs', 'minor_abs', 'log_mean_abs',
            'log_ratio_abs', 'color_max_abs', 'metallic_abs')


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def verify_files(root, records):
    for name, identity in records.items():
        path = root / name
        assert path.resolve() == path and root in path.parents
        assert path.stat().st_size == identity['bytes'] and digest(path) == identity['sha256'], path


def materials(arm, step, camera):
    return numeric(ARMS[arm] / f'step-{step:05d}/camera-{camera}/materials.npz')


def replay_camera(arm, step, camera, reference, metrics):
    a = materials(arm, step, camera)
    errors = numeric(ARMS[arm] / f'step-{step:05d}/camera-{camera}-errors.npz')
    valid = reference['shader_valid']
    pixels = reference['pixel_indices'][valid]
    mask = a['hit_mask']
    np.testing.assert_array_equal(np.flatnonzero(mask), pixels)
    np.testing.assert_array_equal(errors['geometry_hit_mask'], reference['hit_mask'])
    for key, original in [('position', 'position'), ('normal', 'normals')]:
        np.testing.assert_array_equal(a[key][mask], reference[original][valid].astype(np.float32))
    truth = a['linear_gt'][mask]
    np.testing.assert_array_equal(truth, reference['linear_components'][valid, 2].astype(np.float32))
    mx, my = a['mx'][mask], a['my'][mask]
    tangents, bitangents = 2 * a['tangents'][mask] - 1, 2 * a['bitangents'][mask] - 1
    tensor = (mx[:, :, None]**2 * tangents[:, :, None] * tangents[:, None, :]
              + my[:, :, None]**2 * bitangents[:, :, None] * bitangents[:, None, :])
    equal(a['Q_world'][mask], tensor)
    oracle_tensor = reference['Q_world'][valid]
    q = np.linalg.norm(tensor - oracle_tensor, axis=(1, 2)) / np.linalg.norm(oracle_tensor, axis=(1, 2))
    major, minor = np.maximum(mx, my)[:, 0], np.minimum(mx, my)[:, 0]
    axis = np.where(mx >= my, tangents, bitangents)
    theta = np.degrees(np.arccos(np.clip(abs((axis * reference['axis'][valid]).sum(-1)), 0, 1)))
    eligible = (major**2 - minor**2) / (major**2 + minor**2) >= .1
    kd, metal = a['kd'][mask], a['metallic'][mask, 0]
    major_abs, minor_abs = abs(major - .35), abs(minor - .08)
    for key, value in [('Q_relative', q), ('orientation_degrees', theta), ('orientation_eligible', eligible),
                       ('major_width', major_abs), ('minor_width', minor_abs),
                       ('color_max', abs(kd - COLOR).max(-1)), ('metallic', abs(metal - .8))]:
        equal(value, errors[key])
    assert metrics['physical']['Q_at_most_005'] == int((q <= .05).sum())
    assert metrics['physical']['orientation_ineligible'] == int((~eligible).sum())
    delta = a['linear_rgb'][mask] - truth
    scalar32(float(np.linalg.norm(delta) / np.linalg.norm(truth)), metrics['RGB_relative_L2'],
             f'{arm}/{step}/{camera}/RGB_relative_L2', delta.size)
    scalar32(float(np.sqrt(np.mean(delta**2))), metrics['RGB_linear_RMSE'],
             f'{arm}/{step}/{camera}/RGB_linear_RMSE', delta.size)
    return dict(q=q, orientation_degrees=theta, eligible=eligible, major=major, minor=minor, kd=kd, metallic=metal,
                major_abs=major_abs, minor_abs=minor_abs, log_mean_abs=abs(np.log(np.sqrt(major * minor / (.35 * .08)))),
                log_ratio_abs=abs(np.log((major / minor) / (.35 / .08))), color_max_abs=abs(kd - COLOR).max(-1),
                metallic_abs=abs(metal - .8), rgb_error=delta.astype(float), rgb_truth=truth.astype(float))


def summarize(values):
    eligible = values['eligible']
    observations = len(eligible)
    row = dict(observations=observations, orientation_eligible=int(eligible.sum()),
               q_at_most_005=int((values['q'] <= .05).sum()),
               RGB_relative_L2=float(np.linalg.norm(values['rgb_error']) / np.linalg.norm(values['rgb_truth'])),
               RGB_linear_RMSE=float(np.sqrt(np.mean(values['rgb_error']**2))),
               minor_at_floor=int((values['minor'] <= .0401).sum()))
    for name in PHYSICAL:
        samples = values[name][eligible] if name == 'orientation_degrees' else values[name]
        row.update({f'{name}_{k}': v for k, v in stats(samples).items()} if len(samples)
                   else {f'{name}_{k}': None for k in ('mean', 'median', 'p95')})
    for key in ('major', 'minor', 'metallic'):
        row[key + '_predicted_mean'] = float(values[key].mean(dtype=float))
    row.update({f'kd_{channel}_predicted_mean': float(values['kd'][:, i].mean(dtype=float))
                for i, channel in enumerate('RGB')})
    row['minor_signed_bias'] = row['minor_predicted_mean'] - .08
    return row


def replay_illumination(arm, step, saved, target, known_map):
    light = numeric(ARMS[arm] / f'step-{step:05d}/illumination.npz')
    value, known = light['linear_RGB'], light['known_linear_RGB']
    np.testing.assert_array_equal(known, known_map['known_linear_RGB'])
    np.testing.assert_array_equal(light['directions'], known_map['directions'])
    theta = (np.arange(128) + .5) / 128 * np.pi
    weight = np.broadcast_to(np.sin(theta)[:, None, None], (*value.shape[:2], 1))
    mean = (value * weight).sum((0, 1)) / weight.sum()
    equal(mean, saved['mean_spherical'])
    error = np.sqrt(((value - known)**2 * weight).sum() / (known**2 * weight).sum())
    equal(error, saved['relative_L2'])
    # Retrospective angular-shape comparison on the same saved grid only.
    adjusted = value * (target / mean)
    equalized_error = np.sqrt(((adjusted - known)**2 * weight).sum() / (known**2 * weight).sum())
    row = dict(arm=arm, completed_updates=step, mean_RGB=mean.tolist(),
               max_abs_mean_error=float(abs(mean - target).max()), relative_L2=float(error),
               mean_equalized_relative_L2=float(equalized_error),
               per_channel_relative_L2=np.sqrt(((value - known)**2 * weight).sum((0, 1))
                                               / (known**2 * weight).sum((0, 1))).tolist())
    if arm == 'D3':
        calibration = saved['calibration']
        equal(abs(mean - target), calibration['absolute_error'])
        equal(calibration['target_mean_RGB'], target)
        np.testing.assert_allclose(np.array(calibration['raw_mean_RGB']) * calibration['scale_RGB'],
                                   target, rtol=0, atol=2e-8)
        # The run's nested field was overwritten by the last camera chunk.
        # Preserve it in the original receipt; the map and mean_spherical agree.
        field = np.asarray(calibration['realized_mean_RGB'], dtype=np.float32)
        final_view = materials(arm, step, 34)
        final_chunk = int(final_view['hit_mask'].sum()) % 512 or 512
        np.testing.assert_array_equal(field, np.stack(
            [final_view[key][final_view['hit_mask']][-final_chunk:]
             for key in ('linear_rgb', 'linear_diffuse', 'linear_specular')], axis=1))
        row.update(saved_mislabeled_field_shape=list(field.shape),
                   raw_mean_RGB=calibration['raw_mean_RGB'], scale_RGB=calibration['scale_RGB'])
    return row


def read_log(root):
    with (root / 'training.jsonl').open() as stream:
        history = [json.loads(line) for line in stream]
    assert [r['completed_updates'] for r in history] == list(range(1, 10001))
    assert history[0]['lr'] == 0 and history[1]['lr'] == 5e-7
    assert history[0]['parameter_displacement_L1'] == 0 < history[1]['parameter_displacement_L1']
    for row in history:
        assert all(np.isfinite(row[key]) for key in ('RGB_loss', 'material_prior', 'diffuse_white_prior', 'preclip_gradient_l2', 'lr'))
        assert all(np.isfinite(value) for value in row['gradient_groups'].values())
        assert row['gradient_groups']['inner_light'] == 0
    return history


def analyze(out):
    out = out.resolve()
    assert not out.exists() and not any(root in (out, *out.parents) for root in (BATCH, D3_BATCH, REFERENCE))
    freeze_path, auth_path = D3_BATCH / 'freeze.json', D3_BATCH / 'authorization.json'
    assert digest(freeze_path) == digest(ROOT / 'docs/protocols/DONUT_KURT_CALIBRATED_D3_V1.freeze.json') == FREEZE
    assert digest(auth_path) == digest(Path(str(D3_BATCH) + '-authorization.json')) == AUTH
    freeze, auth = read(freeze_path), read(auth_path)
    definition = freeze['calibrated_training']
    assert auth['approved'] is True and auth['researcher_admitted_dataset'] is True and auth['maximum_execution_launches'] == 1
    assert auth['freeze_sha256'] == FREEZE and auth['commit'] == COMMIT and auth['limits'] == definition['limits']
    assert auth['protocol'] == definition['protocol'] and auth['output_directories'] == {'D3': '/outputs/D3'}
    assert auth['host_output_directory'] == freeze['host_calibrated_output'] == str(D3_BATCH)
    assert auth['selected_gpu'] == '0' and auth['image_id'] == freeze['image_id']
    assert auth['benchmark_generation_sha256'] == definition['benchmark_generation_sha256'] == GENERATION
    assert auth['initial_sha256'] == definition['initial_sha256'] == digest(Path(definition['initial_host_path']))
    admission = Path(auth['dataset_admission_path'])
    assert digest(admission) == auth['dataset_admission_sha256'] == definition['dataset_admission_sha256']
    assert read(admission)['researcher_admitted_dataset'] is True
    for name, sha in freeze['sources'].items():
        assert digest(ROOT / name) == sha, name
    for item in freeze['inputs'].values():
        assert digest(Path(item['path'])) == item['sha256']
    generation = read(REFERENCE / 'generation.json')
    assert digest(REFERENCE / 'generation.json') == GENERATION and generation['freeze_sha256'] == freeze['reference_freeze_sha256']
    verify_files(REFERENCE, generation['outputs'])
    target = np.asarray(definition['calibration']['mean_linear_RGB'])
    map_path = Path(definition['reference_map_host_path'])
    assert digest(map_path) == definition['calibration']['reference_map_sha256']
    known_map = numeric(map_path)
    references = {c: numeric(REFERENCE / f'view-{c}.npz') for c in CAMERAS}
    expected_init = read(Path(definition['initial_host_path']))
    old = read(PRIOR / 'artifact-manifest.json')['records']
    receipt_files = [freeze_path, auth_path, admission, REFERENCE / 'generation.json', map_path]
    rows, views, light_rows, probes, resources = [], [], [], [], {}
    histories, final_distributions = {}, {}
    for arm, root in ARMS.items():
        receipt = root / 'recovery.json'
        report = read(receipt)
        if arm == 'D3':
            assert report['commit'] == COMMIT and report['freeze_sha256'] == FREEZE
            assert report['initialization_anchor_sha256'] == definition['initial_sha256']
            assert report['calibration'] == definition['calibration'] and report['protocol'] == definition['protocol']
            assert report['counters'] == dict(shader_calls=12288, shadow_rays=9533964288,
                                               calibration_evaluations=10006, calibration_directions=327876608,
                                               component_reverse_sweeps=30)
            assert all(v <= definition['limits'][k] for k, v in report['counters'].items())
            assert report['elapsed_seconds'] < definition['limits']['inner_seconds']
            assert report['resources']['host_peak_RSS'] < 64 * 1024**3
            assert sum(item['bytes'] for item in report['outputs'].values()) < definition['limits']['output_bytes']
        else:
            assert digest(receipt) == old[str(receipt)]['sha256'] == definition[f'accepted_{arm}_report_sha256']
        assert report['arm'] == arm and report['status'] == 'complete' and report['optimizer_returns'] == 10000
        assert report['scientific_work_complete'] and report['cleanup_verified'] and report['native_reference_gate']['passed']
        assert report['initial'] == expected_init == read(root / 'initial.json')
        assert report['benchmark_sha256'] == GENERATION and report['runtime']['declared_image'] == freeze['image_id']
        assert report['policy'] == freeze['policy'] and report['effective_shader'] == freeze['shader']
        assert [p['step'] for p in report['saved_endpoints']] == STEPS
        exit_path = root.parent / f'{arm}-exit-code.txt'
        assert exit_path.read_text().strip() == '0'
        verify_files(root, report['outputs'])
        checkpoints = sorted(root.glob('checkpoint-*.pth'))
        assert [p.name for p in checkpoints] == [f'checkpoint-{step:05d}.pth' for step in STEPS[1:]]
        receipt_files += [receipt, exit_path, root / 'initial.json', root / 'native-reference-gate.npz',
                          root / 'training.jsonl', root.parent / f'{arm}-console.txt', *checkpoints]
        history = histories[arm] = read_log(root)
        assert all((row['gradient_groups']['outer_light'] == 0) == (arm == 'D1') for row in history)
        records = [row for row in history if 'gradient_interactions' in row]
        assert [r['completed_updates'] for r in records] == [1, *STEPS[1:]]
        for row in records:
            entry = row['gradient_interactions']
            if arm == 'D3':
                white = entry['terms']['diffuse_white']
                assert row['diffuse_white_prior'] == white['weighted_loss']
                assert white['groups']['outer_light'] > 0 and all(
                    v == 0 for k, v in white['groups'].items() if k != 'outer_light')
            probes.append(dict(arm=arm, completed_updates=row['completed_updates'], **entry))
        resources[arm] = {k: report[k] for k in ('elapsed_seconds', 'resources', 'counters', 'runtime', 'cleanup_verified')}
        resources[arm]['output_identities_verified'] = len(report['outputs'])
        if arm == 'D3':
            resources[arm]['calibration_timing'] = report['calibration_timing']
        for endpoint in report['saved_endpoints']:
            step = endpoint['step']
            metrics = endpoint['metrics']
            assert metrics == read(root / f'step-{step:05d}/metrics.json')
            pieces = {}
            for camera in CAMERAS:
                pieces[camera] = replay_camera(arm, step, camera, references[camera], metrics[str(camera)])
                if step == 0 and arm != 'D1':
                    prior, now = materials('D1', 0, camera), materials(arm, 0, camera)
                    for key in ('kd', 'mx', 'my', 'rotation', 'Q_world', 'metallic'):
                        np.testing.assert_array_equal(prior[key], now[key])
                views.append(dict(arm=arm, completed_updates=step, camera=camera, **summarize(pieces[camera])))
            for split, cameras in [('all', CAMERAS), ('fitting', CAMERAS[:4]), ('held_out', CAMERAS[4:])]:
                values = {k: np.concatenate([pieces[c][k] for c in cameras]) for k in pieces[cameras[0]]}
                rows.append(dict(arm=arm, completed_updates=step, split=split, **summarize(values)))
                if step == 10000:
                    final_distributions[arm, split] = values
            light_row = replay_illumination(arm, step, metrics['illumination'], target, known_map)
            light_rows.append(light_row)
            if arm == 'D3':
                assert light_row['max_abs_mean_error'] <= definition['calibration']['absolute_mean_tolerance']
        resources[arm]['final_surface_Q_diagnostic'] = report['saved_endpoints'][-1]['metrics']['surface_Q_smoothness_diagnostic']
    out.mkdir(parents=True)
    coverage = coverage_analysis(out / 'coverage', ARMS['D3'])
    frozen_coverage = ROOT / 'docs/results/donut_camera34_coverage_v1'
    original = read(frozen_coverage / 'camera34-coverage.json')
    assert original['arms'] == {k: coverage['arms'][k] for k in ('D1', 'D2')}
    for key, value in numeric(frozen_coverage / 'camera34-coverage.npz').items():
        np.testing.assert_array_equal(value, numeric(out / 'coverage/camera34-coverage.npz')[key])
    write_csv(out / 'endpoint-metrics.csv', rows)
    write_csv(out / 'per-camera-metrics.csv', views)
    write_json(out / 'analysis.json', dict(
        execution_commit=COMMIT, freeze_sha256=FREEZE, authorization_sha256=AUTH,
        reference_generation_sha256=GENERATION, status='verified_complete', frozen_source_identities=len(freeze['sources']),
        reference_artifact_identities=len(generation['outputs']), independently_replayed_camera_endpoints=len(views),
        matched_initial_states=True, matched_initial_material_maps=True, resources=resources,
        calibration_target=target.tolist(), illumination=light_rows, gradient_probes=probes,
        report_scalar_replay=SCALAR_REPLAY,
        metadata_issue='D3 calibration.realized_mean_RGB records the final camera chunk. Verified mean from hashed illumination maps and mean_spherical.',
        analysis_work=dict(new_model_calls=0, renderer_calls=0, optimizer_returns=0, checkpoint_deserializations=0)))
    write_json(out / 'artifact-manifest.json', dict(batch=str(D3_BATCH),
        bytes=sum(p.stat().st_size for p in D3_BATCH.rglob('*') if p.is_file()),
        records={str(p): record(p) for p in sorted(set(receipt_files))},
        retained_maps='Each recovery.json has the SHA-256 and size of every saved map and native output.'))
    figures(out, rows, light_rows, probes, histories, references, final_distributions)
    return len(views)


def figures(out, rows, illumination, probes, histories, references, distributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    colors = dict(D1='tab:blue', D2='tab:orange', D3='tab:green')

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def grid_image(ax, value, title='', **kwargs):
        rendered = ax.imshow(value, interpolation='nearest', **kwargs)
        ax.set(title=title, xticks=[], yticks=[])
        return rendered

    def reference_image(camera, component):
        ref = references[camera]
        value = np.zeros((128 * 128, 3))
        value[ref['pixel_indices']] = ref['linear_components'][:, component]
        return srgb(value.reshape(128, 128, 3))

    fig, axes = plt.subplots(6, 7, figsize=(15, 12))
    for i, camera in enumerate(CAMERAS):
        grid_image(axes[i, 0], reference_image(camera, 2), 'Reference' if not i else '')
        axes[i, 0].set_ylabel(f'Camera {camera}\n' + ('fit' if i < 4 else 'held out'))
        for j, arm in enumerate(ARMS):
            a = materials(arm, 10000, camera)
            grid_image(axes[i, 1 + j], srgb(a['linear_rgb']), LABELS[arm] if not i else '')
            rmse = np.sqrt(np.mean((a['linear_rgb'] - a['linear_gt'])**2, axis=-1))
            rmse = np.where(a['hit_mask'], rmse, np.nan)
            scale = grid_image(axes[i, 4 + j], rmse, arm + ' RGB error' if not i else '',
                               vmin=0, vmax=.02, cmap='magma')
    fig.tight_layout(rect=(0, 0, .94, 1))
    fig.colorbar(scale, cax=fig.add_axes([.945, .23, .012, .54]), label='Per-pixel linear RGB RMSE')
    fig.savefig(out / 'RGB-comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(4, 6, figsize=(14, 8))
    camera = 0
    ref = references[camera]
    mask = materials('D3', 10000, camera)['hit_mask']
    director = ref['axis']
    projector = np.zeros((128 * 128, 3))
    projector[ref['pixel_indices']] = np.stack([director[:, 0] * director[:, 1],
        director[:, 0] * director[:, 2], director[:, 1] * director[:, 2]], axis=-1) + .5
    oracle = [np.broadcast_to(COLOR, (128, 128, 3)), np.full((128, 128), .8),
              np.full((128, 128), .35), np.full((128, 128), .08),
              projector.reshape(128, 128, 3), np.zeros((128, 128))]
    titles = ('Linear base color', 'Metallic [0,1]', 'Major width [0,0.5]',
              'Minor width [0,0.15]', 'World director', 'Relative Q error [0,0.5]')
    for i, arm in enumerate(('Reference', *ARMS)):
        if arm == 'Reference':
            panels = oracle
        else:
            a = materials(arm, 10000, camera)
            p = a['major_axis_projector_world']
            lines = np.stack([p[..., 0, 1], p[..., 0, 2], p[..., 1, 2]], -1) + .5
            q_error = numeric(ARMS[arm] / 'step-10000/camera-0-errors.npz')['Q_relative']
            image = np.zeros(mask.shape)
            image[mask] = q_error
            panels = [a['kd'], a['metallic'][..., 0], a['major_width'][..., 0], a['minor_width'][..., 0], lines, image]
        for j, value in enumerate(panels):
            value = np.where(mask[..., None], value, 0) if value.ndim == 3 else np.where(mask, value, np.nan)
            grid_image(axes[i, j], value, titles[j] if not i else '', vmin=0, vmax=(1, 1, .5, .15, 1, .5)[j],
                       cmap='magma' if j == 5 else 'viridis')
        axes[i, 0].set_ylabel(arm)
    save(fig, 'material-comparison-camera0')

    fig, axes = plt.subplots(6, 6, figsize=(12, 11))
    for i, step in enumerate(STEPS):
        for j, name in enumerate(('reconstructed', 'albedo', 'major_width', 'minor_width',
                                  'orientation_world_projector', 'specular')):
            path = ARMS['D3'] / f'step-{step:05d}/camera-0/{name}.png'
            with Image.open(path) as source:
                grid_image(axes[i, j], np.array(source), name.replace('_', ' ') if not i else '', cmap='gray')
        axes[i, 0].set_ylabel(f'{step:,} updates')
    save(fig, 'progression-camera0')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for arm, color in colors.items():
        pooled = [row for row in rows if row['arm'] == arm and row['split'] == 'all']
        held = [row for row in rows if row['arm'] == arm and row['split'] == 'held_out']
        axes[0, 0].plot(STEPS, [r['q_median'] for r in pooled], '-o', color=color, label=arm)
        axes[0, 0].plot(STEPS, [r['q_p95'] for r in pooled], ':', color=color)
        axes[0, 1].plot(STEPS, [r['q_at_most_005'] / r['observations'] for r in pooled], '-o', color=color, label=arm)
        axes[0, 2].plot(STEPS, [r['RGB_relative_L2'] for r in held], '-o', color=color, label=arm)
        axes[1, 0].plot(STEPS, [r['minor_predicted_mean'] for r in pooled], '-o', color=color, label=arm)
        axes[1, 1].plot(STEPS, [r['orientation_degrees_median'] for r in pooled], '-o', color=color, label=arm)
        light = [row for row in illumination if row['arm'] == arm]
        axes[1, 2].plot(STEPS, [r['relative_L2'] for r in light], '-o', color=color, label=arm)
    axes[1, 0].axhline(.08, color='0.4', linestyle='--', label='Reference')
    for ax in (axes[0, 0], axes[0, 2]):
        ax.set_yscale('log')
    for ax, title in zip(axes.flat, ('Relative Q error (median and dotted p95)', 'Fraction with Q error ≤ 0.05',
                                     'Held-out linear RGB relative L2', 'Mean minor width',
                                     'Median eligible orientation error (°)', 'Illumination relative L2')):
        ax.set(title=title, xlabel='Completed Adam updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, 'recovery-curves')

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, split in enumerate(('all', 'held_out')):
        for j, key in enumerate(('q', 'minor_abs', 'orientation_degrees', 'log_ratio_abs')):
            ax = axes[i, j]
            for arm, color in colors.items():
                data = distributions[arm, split]
                array = data[key][data['eligible']] if key == 'orientation_degrees' else data[key]
                sorted_values = np.sort(array)
                ax.step(sorted_values, np.arange(1, len(array) + 1) / len(array), color=color, label=arm)
            ax.set(xlabel=('Relative Q error', 'Minor width absolute error', 'Eligible orientation error (°)',
                           'Absolute log anisotropy ratio error')[j], ylabel=split.replace('_', ' ') + ' fraction')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    save(fig, 'endpoint-distributions')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for arm, color in colors.items():
        h = histories[arm]
        for ax, key in zip(axes.flat, ('RGB_loss', 'diffuse_white_prior', 'preclip_gradient_l2')):
            values = [row[key] for row in h]
            ax.plot(np.arange(200, 10001), np.convolve(values, np.ones(200) / 200, mode='valid'),
                    color=color, label=arm)
    x = [p['completed_updates'] for p in probes if p['arm'] == 'D3']
    d3 = [p for p in probes if p['arm'] == 'D3']
    for key in ('photo', 'material', 'energy', 'specular', 'diffuse_white'):
        axes[1, 0].plot(x, [p['terms'][key]['gradient_l2'] for p in d3], '-o', label=key)
    axes[1, 1].plot(x, [p['gradient_cosines']['photo__diffuse_white'] for p in d3], '-o', label='All groups')
    axes[1, 1].plot(x, [p['group_gradient_cosines']['diffuse_white__photo']['outer_light'] for p in d3],
                    '-o', label='Outer light')
    axes[1, 2].plot(x, [p['terms']['diffuse_white']['groups']['outer_light'] /
                       p['terms']['photo']['groups']['outer_light'] for p in d3], '-o', label='D3')
    for ax, title in zip(axes.flat, ('Photometric loss (200-update mean)', 'Diffuse-white loss (200-update mean)',
                                     'Preclip gradient norm (200-update mean)', 'D3 component gradient norms',
                                     'D3 diffuse-white vs photometric cosine', 'D3 white/photo light gradient norm ratio')):
        ax.set(title=title, xlabel='Completed Adam updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[1, 0].set_yscale('symlog', linthresh=1e-6)
    axes[1, 1].axhline(0, color='0.5', linewidth=.8)
    save(fig, 'losses-and-gradients')

    fig, axes = plt.subplots(4, 5, figsize=(14, 8))
    reference_light = numeric(ARMS['D3'] / 'step-10000/illumination.npz')['known_linear_RGB']
    for i, arm in enumerate(('Reference', *ARMS)):
        if arm == 'Reference':
            components = [reference_image(0, c) for c in (0, 1, 2)]
            light = reference_light
        else:
            a = materials(arm, 10000, 0)
            components = [srgb(a[k]) for k in ('linear_diffuse', 'linear_specular', 'linear_rgb')]
            light = numeric(ARMS[arm] / 'step-10000/illumination.npz')['linear_RGB']
        panels = (*components, srgb(light), np.sqrt(np.mean((light - reference_light)**2, axis=-1)))
        for j, value in enumerate(panels):
            grid_image(axes[i, j], value, ('Diffuse', 'Specular', 'Total RGB', 'Environment (display clipped)',
                                          'Environment linear RMSE [0,.8]')[j] if not i else '',
                       vmin=0, vmax=.8, cmap='magma')
        axes[i, 0].set_ylabel(arm)
    save(fig, 'components-and-illumination')

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    cov = numeric(out / 'coverage/camera34-coverage.npz')
    x, y = cov['pixel_xy'].T
    values = np.full((128, 128), np.nan)
    values[y, x] = cov['nearby_camera_counts'][:, 1]
    count_map = grid_image(axes[0], values, 'Nearby fitting cameras', vmin=0, vmax=3, cmap='viridis')
    fig.colorbar(count_map, ax=axes[0], orientation='horizontal', ticks=[0, 1, 2, 3], shrink=.85)
    for ax, arm in zip(axes[1:], ARMS):
        q = numeric(ARMS[arm] / 'step-10000/camera-34-errors.npz')['Q_relative']
        error = np.full((128, 128), np.nan)
        error[y, x] = q
        rendered = grid_image(ax, error, arm + ' relative Q error', vmin=0, vmax=1.25, cmap='magma')
        fig.colorbar(rendered, ax=ax, orientation='horizontal', ticks=[0, .5, 1, 1.25], shrink=.85)
    save(fig, 'camera34-coverage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = analyze(args.output)
    print(f'Verified three arms and {count} saved camera/endpoints')
