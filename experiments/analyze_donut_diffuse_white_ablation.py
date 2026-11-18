"""File-only verification and matched Donut factorial comparison."""

import argparse
import sys
from pathlib import Path

import numpy as np

import analyze_donut_calibrated_d3 as replay
from analyze_donut_camera34_coverage import error_summary

ROOT = replay.ROOT
sys.path.insert(0, str(ROOT))
from benchmarks.neural_observation_e16 import compatibility

BATCH = Path('/home/isavand/paper/evidence/donut-kurt-diffuse-white-ablation-25c58aa9eb94402aba940efc6f28194a')
FREEZE = '1e0f78d52a171e048b3f5a7f453a6a5655c47e31814f92d9eba28d1f47f279aa'
AUTH = '7a38ee92538ed75a50cc4918372e309e0b593efa4058051042d6338cfa6ad41b'
COMMIT = '7234bb3dc06651350efe8cb03e46266904b8a3f6'
FREEZE_FILE = ROOT / 'docs/protocols/DONUT_KURT_DIFFUSE_WHITE_ABLATION_V1.freeze.json'
CONTROL = ROOT / 'docs/results/donut_calibrated_d3_v1'
ARMS = dict(replay.ARMS, D4=BATCH / 'D4', D5=BATCH / 'D5')
CAMERAS, STEPS = replay.CAMERAS, replay.STEPS
COLORS = dict(D1='tab:blue', D2='tab:orange', D3='tab:green', D4='tab:purple', D5='tab:red')


def verify_definition():
    manifest, authorization = BATCH / 'freeze.json', BATCH / 'authorization.json'
    assert replay.digest(manifest) == replay.digest(FREEZE_FILE) == FREEZE
    assert replay.digest(authorization) == replay.digest(Path(str(BATCH) + '-authorization.json')) == AUTH
    freeze, auth = replay.read(manifest), replay.read(authorization)
    definition = freeze['diffuse_white_ablation']
    assert auth['approved'] and auth['researcher_admitted_dataset'] and auth['maximum_execution_launches'] == 1
    assert auth['commit'] == COMMIT and auth['freeze_sha256'] == FREEZE
    assert auth['protocol'] == definition['protocol'] and auth['limits'] == definition['limits']
    assert auth['arms'] == definition['arms'] == ['D4', 'D5']
    assert auth['output_directories'] == {'D4': '/outputs/D4', 'D5': '/outputs/D5'}
    assert auth['host_output_directory'] == freeze['host_ablation_output'] == str(BATCH)
    assert auth['diffuse_white_in_objective'] is definition['diffuse_white_in_objective'] is False
    assert auth['selected_gpu'] == '0' and auth['image_id'] == freeze['image_id']
    assert auth['wall_seconds'] == definition['wall_seconds'] == 7500
    assert auth['calibration'] == definition['calibration']
    for name, sha in freeze['sources'].items():
        assert replay.digest(ROOT / name) == sha, name
    for item in freeze['inputs'].values():
        assert replay.digest(Path(item['path'])) == item['sha256']
    admission = Path(auth['dataset_admission_path'])
    assert replay.digest(admission) == auth['dataset_admission_sha256'] == definition['dataset_admission_sha256']
    assert replay.read(admission)['researcher_admitted_dataset'] is True
    generation = replay.REFERENCE / 'generation.json'
    assert replay.digest(generation) == auth['benchmark_generation_sha256'] == replay.GENERATION
    assert definition['benchmark_generation_sha256'] == replay.GENERATION
    dataset = replay.read(generation)
    assert dataset['freeze_sha256'] == freeze['reference_freeze_sha256']
    replay.verify_files(replay.REFERENCE, dataset['outputs'])
    initial = Path(definition['initial_host_path'])
    assert replay.digest(initial) == definition['initial_sha256'] == auth['initial_sha256']
    known = Path(definition['reference_map_host_path'])
    assert replay.digest(known) == definition['calibration']['reference_map_sha256']
    return freeze, auth, dataset, initial, known, [manifest, authorization, admission, generation, initial, known]


def verify_arm(arm, root, freeze, auth, expected_initial):
    receipt = root / 'recovery.json'
    m = replay.read(receipt)
    definition = freeze['diffuse_white_ablation']
    assert m['arm'] == arm and m['status'] == 'complete' and m['scientific_work_complete'] and m['cleanup_verified']
    assert m['optimizer_returns'] == 10000 and [p['step'] for p in m['saved_endpoints']] == STEPS
    assert m['initial'] == expected_initial == replay.read(root / 'initial.json')
    assert m['benchmark_sha256'] == replay.GENERATION and m['runtime']['declared_image'] == freeze['image_id']
    assert m['policy'] == freeze['policy'] and m['effective_shader'] == freeze['shader']
    exit_path = root.parent / f'{arm}-exit-code.txt'
    assert exit_path.read_text().strip() == '0'
    replay.verify_files(root, m['outputs'])
    assert len(m['outputs']) == 890
    checkpoints = sorted(root.glob('checkpoint-*.pth'))
    assert [p.name for p in checkpoints] == [f'checkpoint-{step:05d}.pth' for step in STEPS[1:]]
    gate_path = root / 'native-reference-gate.npz'
    z = replay.numeric(gate_path)
    gate = compatibility(z['native_draws'], z['reference'], abs(z['reference']-z['coarse']),
                         m['policy']['compatibility_absolute'], m['policy']['compatibility_relative'])
    for key, value in gate.items():
        if isinstance(value, (int, bool)):
            assert value == m['native_reference_gate'][key]
        else:
            replay.equal(value, m['native_reference_gate'][key])
    assert gate['passed']
    history = replay.read_log(root)
    if arm in definition['arms']:
        assert m['commit'] == COMMIT and m['freeze_sha256'] == FREEZE and m['protocol'] == definition['protocol']
        assert m['initialization_anchor_sha256'] == definition['initial_sha256']
        assert m['diffuse_white_in_objective'] is False and m['mean_calibration'] == (arm == 'D4')
        assert m['optimization_terms'] == ['RGB_loss', 'material_prior']
        assert m['diffuse_white_diagnostic_coefficient'] == .01
        assert m['counters']['shader_calls'] == 12288 and m['counters']['shadow_rays'] == 9533964288
        assert m['counters']['component_reverse_sweeps'] == 30
        limits = definition['limits'][arm]
        for key in ('calibration_evaluations', 'calibration_directions'):
            assert m['counters'].get(key, 0) == limits.get(key, 0)
        assert all(v <= limits[k] for k, v in m['counters'].items())
        assert m['elapsed_seconds'] < limits['inner_seconds']
        assert sum(item['bytes'] for item in m['outputs'].values()) < limits['output_bytes']
        assert m['resources']['host_peak_RSS'] < limits['host_memory_gib'] * 1024**3
        for row in history:
            optimized = np.float32(row['RGB_loss']) + np.float32(row['material_prior'])
            assert float(optimized) == row['optimization_loss']
        assert all(row['gradient_groups']['outer_light'] > 0 for row in history)
    else:
        control = auth['accepted_control_receipts'][arm]
        assert Path(control['path']) == receipt
        assert replay.digest(receipt) == control['sha256'] == definition[f'accepted_{arm}_report_sha256']
    paths = [receipt, exit_path, root.parent / f'{arm}-console.txt', root / 'initial.json',
             root / 'training.jsonl', gate_path, *checkpoints]
    return m, history, gate, paths


def extend_coverage(values, data, coverage):
    q = values['q']
    angle = values['orientation_degrees']
    rgb = values['rgb_error']
    counts = data['nearby_camera_counts']
    facing = data['facing_cosine']
    triangle = data['same_triangle_camera_count']
    assert len(q) == len(counts) == coverage['observations']
    for label, bins in [('within_001_camera_count', counts[:, 1]), ('same_triangle_camera_count', triangle)]:
        yield label, {str(i): error_summary(q, angle, rgb, bins == i) for i in range(5)}
    yield 'radius_sensitivity', {
        str(radius): {'zero': error_summary(q, angle, rgb, counts[:, i] == 0),
                      'one_or_more': error_summary(q, angle, rgb, counts[:, i] > 0)}
        for i, radius in enumerate(coverage['radius_native_units'])
    }
    yield 'facing_cross_tab_at_001', {
        f'{support}_{band}': error_summary(q, angle, rgb, mask & (facing >= lower) & (facing < upper))
        for support, mask in [('unsupported', counts[:, 1] == 0), ('supported', counts[:, 1] > 0)]
        for band, lower, upper in [('grazing', 0, .25), ('oblique', .25, .6), ('near_frontal', .6, 1.001)]
    }


def interaction_row(metric, values, **labels):
    d2, d3, d4, d5 = (values[arm] for arm in ('D2', 'D3', 'D4', 'D5'))
    if None in (d2, d3, d4, d5):
        cal = free = no_white = interaction = None
    else:
        cal, free, no_white = d4 - d3, d5 - d2, d4 - d5
        interaction = cal - free
    return dict(**labels, metric=metric, D2=d2, D3=d3, D4=d4, D5=d5,
                remove_calibrated=cal, remove_free=free, calibrate_without_white=no_white,
                interaction=interaction)


def mean_of_logs(history, key):
    return float(np.mean([row[key] for row in history[-200:]]))


def analyze(out):
    out = out.resolve()
    assert not out.exists() and not any(root in (out, *out.parents) for root in (BATCH, replay.BATCH, replay.D3_BATCH, replay.REFERENCE))
    freeze, auth, dataset, initial, known, artifact_paths = verify_definition()
    target = np.asarray(freeze['diffuse_white_ablation']['calibration']['mean_linear_RGB'])
    refs = {c: replay.numeric(replay.REFERENCE / f'view-{c}.npz') for c in CAMERAS}
    known_map = replay.numeric(known)
    coverage_paths = [CONTROL / 'coverage/camera34-coverage.json', CONTROL / 'coverage/camera34-coverage.npz']
    coverage_hashes = replay.read(CONTROL / 'finalization-validation.json')['result_hashes']
    for path in coverage_paths:
        assert replay.digest(path) == coverage_hashes[str(path.relative_to(ROOT))]
    coverage, cov = replay.read(coverage_paths[0]), replay.numeric(coverage_paths[1])
    artifact_paths += coverage_paths
    x, y = cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y + x, refs[34]['pixel_indices'][refs[34]['shader_valid']])
    # Add only the two admitted paths to the existing read-only replay helpers.
    replay.ARMS.update(ARMS)
    rows, cameras, illumination, probes = [], [], [], []
    histories, resources, gates, final = {}, {}, {}, {}
    for arm, root in ARMS.items():
        m, history, gate, paths = verify_arm(arm, root, freeze, auth, replay.read(initial))
        artifact_paths += paths
        gates[arm] = gate
        histories[arm] = history
        resources[arm] = {k: m[k] for k in ('elapsed_seconds', 'resources', 'counters', 'runtime', 'optimizer_returns', 'cleanup_verified')}
        resources[arm]['calibration_timing'] = m.get('calibration_timing')
        resources[arm]['output_identities_verified'] = len(m['outputs'])
        resources[arm]['last_200_loss_means'] = {k: mean_of_logs(history,k) for k in ('RGB_loss', 'material_prior', 'diffuse_white_prior')}
        for row in history:
            if 'gradient_interactions' in row:
                probe = row['gradient_interactions']
                if arm in ('D3', 'D4', 'D5'):
                    white = probe['terms']['diffuse_white']
                    assert white['weighted_loss'] == row['diffuse_white_prior']
                    assert white['groups']['outer_light'] > 0
                    assert all(v == 0 for name, v in white['groups'].items() if name != 'outer_light')
                probes.append(dict(arm=arm, completed_updates=row['completed_updates'], **probe))
        assert sum(p['arm'] == arm for p in probes) == 6
        for endpoint in m['saved_endpoints']:
            step, metrics = endpoint['step'], endpoint['metrics']
            assert replay.read(root / f'step-{step:05d}/metrics.json') == metrics
            pieces = {}
            for camera in CAMERAS:
                values = replay.replay_camera(arm, step, camera, refs[camera], metrics[str(camera)])
                pieces[camera] = values
                cameras.append(dict(arm=arm, completed_updates=step, camera=camera, **replay.summarize(values)))
            for split, selection in [('all', CAMERAS), ('fitting', CAMERAS[:4]), ('held_out', CAMERAS[4:])]:
                combined = {k: np.concatenate([pieces[c][k] for c in selection]) for k in pieces[selection[0]]}
                rows.append(dict(arm=arm, completed_updates=step, split=split, **replay.summarize(combined)))
                if step == 10000:
                    final[arm, split] = combined
            light = replay.replay_illumination(arm, step, metrics['illumination'], target, known_map)
            if arm == 'D4':
                cal = metrics['illumination']['calibration']
                assert np.shape(cal['realized_mean_RGB']) == (3,)
                replay.equal(cal['target_mean_RGB'], target)
                replay.equal(cal['realized_mean_RGB'], light['mean_RGB'])
                replay.equal(cal['absolute_error'], abs(np.asarray(light['mean_RGB']) - target))
                assert light['max_abs_mean_error'] <= freeze['diffuse_white_ablation']['calibration']['absolute_mean_tolerance']
            if arm == 'D5':
                assert 'calibration' not in metrics['illumination']
            light['transported_Q_mean_squared'] = metrics['surface_Q_smoothness_diagnostic']['predicted_mean_squared']
            assert metrics['surface_Q_smoothness_diagnostic']['pairs'] == 1024
            illumination.append(light)
            if step == 10000 and arm in ('D4','D5'):
                coverage['arms'][arm] = dict(recovery_report_sha256=replay.digest(root / 'recovery.json'),
                                            **dict(extend_coverage(pieces[34], cov, coverage)))
        for camera in CAMERAS:
            baseline = replay.materials('D1', 0, camera)
            actual = replay.materials(arm, 0, camera)
            for k in ('kd','mx','my','rotation','Q_world','metallic'):
                np.testing.assert_array_equal(baseline[k], actual[k])
    for new, control in [('D4', 'D3'), ('D5', 'D2')]:
        a = replay.numeric(ARMS[new] / 'step-00000/illumination.npz')['linear_RGB']
        b = replay.numeric(ARMS[control] / 'step-00000/illumination.npz')['linear_RGB']
        np.testing.assert_array_equal(a, b)
    assert len(rows) == 90 and len(cameras) == 180 and len(probes) == 30
    effects = []
    for step in STEPS:
        for split in ('all', 'fitting', 'held_out'):
            panel = {r['arm']: r for r in rows if r['completed_updates'] == step and r['split'] == split}
            for key in panel['D4'].keys() - {'arm', 'completed_updates', 'split', 'observations'}:
                effects.append(interaction_row(key, {a: r[key] for a, r in panel.items()}, step=step, split=split))
        panel = {r['arm']: r for r in illumination if r['completed_updates'] == step}
        for key in ('relative_L2', 'mean_equalized_relative_L2', 'max_abs_mean_error', 'transported_Q_mean_squared'):
            effects.append(interaction_row(key, {a: r[key] for a, r in panel.items()}, step=step, split='sphere_or_surface'))
    effects.sort(key=lambda row: (row['step'], row['split'], row['metric']))
    out.mkdir(parents=True)
    replay.write_csv(out/'endpoint-metrics.csv', rows)
    replay.write_csv(out/'per-camera-metrics.csv', cameras)
    replay.write_csv(out/'factorial-effects.csv', effects)
    replay.write_json(out/'camera34-coverage.json', coverage)
    replay.write_json(out/'analysis.json', dict(status='verified_complete', execution_commit=COMMIT,
        freeze_sha256=FREEZE, authorization_sha256=AUTH, generation_sha256=replay.GENERATION,
        sources=len(freeze['sources']), reference_artifacts=len(dataset['outputs']), independently_replayed_camera_endpoints=len(cameras),
        matched_initial_material_and_light=True, resources=resources, native_reference_gates=gates,
        illumination=illumination, gradient_probes=probes, report_scalar_replay=replay.SCALAR_REPLAY,
        interpretation='Interaction is (D4-D3)-(D5-D2); descriptive single-seed factorial. Directional mean equalization uses saved grids only.',
        analysis_work=dict(native_models=0, renderer_calls=0, optimizer_returns=0, checkpoint_deserializations=0)))
    replay.write_json(out/'artifact-manifest.json', dict(batch=str(BATCH),
        bytes=sum(p.stat().st_size for p in BATCH.rglob('*') if p.is_file()),
        records={str(p): replay.record(p) for p in sorted(set(artifact_paths))},
        retained_outputs={arm: dict(path=str(root), endpoints=STEPS, cameras=CAMERAS, maps_pattern='step-XXXXX/camera-ID/',
                                   all_output_identities_in=str(root/'recovery.json')) for arm, root in ARMS.items()}))
    figures(out, rows, illumination, histories, probes, refs, final, cov)
    return len(cameras)


def figures(out, rows, illumination, histories, probes, refs, final, coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=144, bbox_inches='tight')
        plt.close(fig)

    def image(ax, pixels, title='', **kwargs):
        ax.set(title=title, xticks=[], yticks=[])
        return ax.imshow(pixels, interpolation='nearest', **kwargs)

    def oracle(camera, component):
        ref = refs[camera]
        pixels = np.zeros((128**2, 3))
        pixels[ref['pixel_indices']] = ref['linear_components'][:, component]
        return replay.srgb(pixels.reshape(128, 128, 3))

    fig, axes = plt.subplots(6, 6, figsize=(12, 11), layout='constrained')
    for i, camera in enumerate(CAMERAS):
        image(axes[i, 0], oracle(camera, 2), 'Reference' if not i else '')
        axes[i, 0].set_ylabel(f'Camera {camera}\n' + ('fitting' if i < 4 else 'held out'))
        for j, arm in enumerate(ARMS, 1):
            image(axes[i, j], replay.srgb(replay.materials(arm, 10000, camera)['linear_rgb']), arm if not i else '')
    save(fig, 'RGB-comparison')

    fig, axes = plt.subplots(5, 6, figsize=(12, 9), layout='constrained')
    for i, arm in enumerate(ARMS):
        a = replay.materials(arm, 10000, 0)
        mask = a['hit_mask']
        p = a['major_axis_projector_world']
        director = np.stack([p[..., 0, 1], p[..., 0, 2], p[..., 1, 2]], -1) + .5
        q = np.full((128,128), np.nan)
        q[mask] = replay.numeric(ARMS[arm] / 'step-10000/camera-0-errors.npz')['Q_relative']
        maps = (a['kd'], a['metallic'][...,0], a['major_width'][...,0],
                a['minor_width'][...,0], np.where(mask[...,None],director,0), q)
        titles = ('Linear color','Metallic','Major width [0,.5]','Minor width [0,.15]',
                  'World director','Relative Q error [0,.8]')
        axes[i, 0].set_ylabel(arm)
        for j, value in enumerate(maps):
            array = np.where(mask,value,np.nan) if value.ndim == 2 else np.clip(value,0,1)
            image(axes[i, j], array, titles[j] if not i else '', vmin=0, vmax=(1,1,.5,.15,1,.8)[j],
                  cmap='magma' if j == 5 else 'viridis')
    save(fig, 'material-comparison-camera0')

    for arm in ('D4','D5'):
        fig, axes = plt.subplots(6, 7, figsize=(12, 10), layout='constrained')
        for i, step in enumerate(STEPS):
            axes[i,0].set_ylabel(f'{step:,} updates')
            for j, name in enumerate(('reconstructed', 'albedo', 'major_width', 'minor_width',
                                      'orientation_world_projector', 'diffuse', 'specular')):
                path = ARMS[arm] / f'step-{step:05d}/camera-0/{name}.png'
                with Image.open(path) as source:
                    image(axes[i,j], np.array(source), name.replace('_',' ') if not i else '', vmin=0,vmax=255,cmap='gray')
        fig.suptitle(f'{arm} · camera 0')
        save(fig, f'progression-{arm}-camera0')
        fig, axes = plt.subplots(6, 6, figsize=(11, 10), layout='constrained')
        for i, step in enumerate(STEPS):
            axes[i,0].set_ylabel(f'{step:,} updates')
            for j, camera in enumerate(CAMERAS):
                path = ARMS[arm] / f'step-{step:05d}/camera-{camera}/reconstructed.png'
                with Image.open(path) as source:
                    image(axes[i,j], np.array(source), f'Camera {camera}' if not i else '')
        fig.suptitle(f'{arm} · six-view RGB progression')
        save(fig, f'progression-{arm}-all-cameras')

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    for arm, color in COLORS.items():
        pooled = [r for r in rows if r['arm'] == arm and r['split'] == 'all']
        held = [r for r in rows if r['arm'] == arm and r['split'] == 'held_out']
        light = [r for r in illumination if r['arm'] == arm]
        for ax, points, key in zip(axes.flat, [pooled,pooled,pooled,held,pooled,pooled,light,light],
                ('q_median','q_p95','major_abs_mean','RGB_relative_L2','minor_abs_mean',
                 'orientation_degrees_median','relative_L2','transported_Q_mean_squared')):
            ax.plot(STEPS, [r[key] for r in points], '-o', color=color, label=arm)
    for i, (ax, title) in enumerate(zip(axes.flat, ('Median relative Q error','Relative Q p95','Major-width MAE',
            'Held-out linear RGB relative L2','Minor-width MAE','Eligible median orientation error (°)',
            'Illumination relative L2','Normal-transported surface-Q variation'))):
        ax.set(title=title,xlabel='Completed Adam updates')
        if i in (0,1,2,3,4,7):
            ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig, 'recovery-curves')

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout='constrained')
    for i, split in enumerate(('all','held_out')):
        for j, (key, title) in enumerate((('q','Relative Q error'),('major_abs','Major-width absolute error'),
                       ('minor_abs','Minor-width absolute error'),('orientation_degrees','Eligible orientation error (°)'))):
            ax = axes[i,j]
            for arm, color in COLORS.items():
                d = final[arm,split]
                values = np.sort(d[key][d['eligible']] if key == 'orientation_degrees' else d[key])
                ax.step(values, np.arange(1,len(values)+1)/len(values), color=color, label=arm)
            ax.set(xlabel=title,ylabel=f'{split.replace("_"," ")} fraction')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    save(fig,'endpoint-distributions')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for arm, color in COLORS.items():
        h = histories[arm]
        for ax, key in zip(axes.flat, ('RGB_loss','material_prior','diffuse_white_prior')):
            values = np.convolve([r[key] for r in h],np.full(200,1/200),mode='valid')
            ax.plot(np.arange(200,10001),values,label=arm,color=color)
    for arm in ('D3','D4','D5'):
        panel = [p for p in probes if p['arm'] == arm]
        x = [p['completed_updates'] for p in panel]
        w = [p['terms']['diffuse_white']['groups']['outer_light'] for p in panel]
        f = [p['terms']['photo']['groups']['outer_light'] for p in panel]
        cosine = [p['group_gradient_cosines']['diffuse_white__photo']['outer_light'] for p in panel]
        for ax, values in zip(axes[1],(w, np.array(w)/f, cosine)):
            ax.plot(x, values, '-o', color=COLORS[arm], label=arm)
    for ax,title in zip(axes.flat,('Photometric loss','Legacy material prior','Weighted diffuse-white term',
            'Diffuse-white light gradient norm','White / photo light gradient norm','White vs photo light-gradient cosine')):
        ax.set(title=title,xlabel='Completed Adam updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ax in (axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1]):
        ax.set_yscale('log')
    axes[1,2].axhline(0,color='0.4',linewidth=.8)
    save(fig,'losses-and-gradients')

    fig, axes = plt.subplots(6, 5, figsize=(12,11), layout='constrained')
    known = replay.numeric(ARMS['D4'] / 'step-10000/illumination.npz')['known_linear_RGB']
    for i,arm in enumerate(('Reference',*ARMS)):
        if arm == 'Reference':
            components, light = [oracle(0,k) for k in (0,1,2)], known
        else:
            a = replay.materials(arm,10000,0)
            components = [replay.srgb(a[key]) for key in ('linear_diffuse','linear_specular','linear_rgb')]
            light = replay.numeric(ARMS[arm] / 'step-10000/illumination.npz')['linear_RGB']
        axes[i,0].set_ylabel(arm)
        panels = (*components,replay.srgb(light),np.sqrt(np.mean((light-known)**2,axis=-1)))
        for j,v in enumerate(panels):
            image(axes[i,j],v,('Diffuse RGB','Specular RGB','Total RGB','Environment preview','Linear RMSE [0,.8]')[j] if not i else '',
                  vmin=0,vmax=.8,cmap='magma')
    save(fig,'components-and-illumination')

    fig, axes = plt.subplots(1,6,figsize=(14,3.3), layout='constrained')
    x,y = coverage['pixel_xy'].T
    points = np.full((128,128),np.nan)
    points[y,x] = coverage['nearby_camera_counts'][:,1]
    count = image(axes[0],points,'Nearby fitting views',vmin=0,vmax=3,cmap='viridis')
    fig.colorbar(count,ax=axes[0],orientation='horizontal',ticks=[0,1,2,3])
    for ax,arm in zip(axes[1:],ARMS):
        q = replay.numeric(ARMS[arm] / 'step-10000/camera-34-errors.npz')['Q_relative']
        error = np.full((128,128),np.nan)
        error[y,x] = q
        rendered = image(ax,error,arm,vmin=0,vmax=1.25,cmap='magma')
    fig.colorbar(rendered,ax=axes[1:],orientation='horizontal',label='Relative Q error')
    save(fig,'camera34-coverage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    cfg = parser.parse_args()
    print(f'Verified five arms and {analyze(cfg.output)} saved camera/endpoints')
