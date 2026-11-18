"""Verify D7 and compare its fixed endpoint with the archived D2 and D6."""

import argparse
from pathlib import Path

import numpy as np

import analyze_donut_surface_q_d6 as world
from analyze_donut_calibrated_d3 import ROOT
from analyze_donut_diffuse_white_ablation import extend_coverage
from benchmarks.donut_calibration_d3 import CALIBRATION
from benchmarks.donut_recovery_v1 import verify_freeze
from benchmarks.donut_surface_chart_d7 import array_hash

replay = world.replay
CAMERAS, STEPS = world.CAMERAS, world.STEPS
BATCH = Path('/home/isavand/paper/evidence/donut-kurt-surface-chart-d7-v2-0af2db9ce8e34fde93b1c33dd8797519')
COMMIT = '39d4d2ea7e7447271e3f1098aca5cfb8ea7a9f62'
FREEZE = '2f99b6684eb0db0bd86f2469f46e686475898de33d0e1ffec73428debfb4f638'
AUTH = '7c408ca3d49f2edbd3f448d258a5c5f85e1ea64dc4d93e59caaea4b22322ad93'
ARMS = dict(D2=world.ARMS['D2'], D6=world.ARMS['D6'], D7=BATCH/'D7')
PRIOR = ROOT/'docs/results/donut_surface_q_d6_v1'


def verify_definition():
    freeze_path = ROOT/'docs/protocols/DONUT_KURT_SURFACE_CHART_D7_V2.freeze.json'
    freeze = verify_freeze(freeze_path, FREEZE)
    fp, ap = BATCH/'freeze.json', BATCH/'authorization.json'
    host_authorization = BATCH.with_name(BATCH.name+'-authorization.json')
    assert replay.digest(fp) == FREEZE
    assert replay.digest(ap) == replay.digest(host_authorization) == AUTH
    plan = freeze['surface_chart_training']
    a = replay.read(ap)
    assert a['approved'] is True and a['researcher_admitted_dataset'] is True
    assert a['maximum_execution_launches'] == 1 and a['selected_gpu'] == '0'
    assert str(BATCH) == a['host_output_directory'] == freeze['host_chart_output']
    assert a['output_directories'] == {'D7': '/outputs/D7'}
    assert a['commit'] == COMMIT and a['freeze_sha256'] == FREEZE and a['image_id'] == freeze['image_id']
    assert a['diffuse_white_in_objective'] is True and a['input_identities'] == freeze['inputs']
    for key in ('protocol', 'limits', 'spatial_policy', 'benchmark_generation_sha256',
                'dataset_admission_sha256', 'initial_sha256'):
        assert a[key] == plan[key]
    assert {**a['accepted_control_receipts'], **a['accepted_neighborhood_artifacts']} == plan['controls']
    assert a['interrupted_attempt'] == freeze['interrupted_attempt']
    artifacts = [fp, ap, host_authorization]
    for item in [*freeze['inputs'].values(), *plan['controls'].values(), *a['interrupted_attempt']['records']]:
        p = Path(item['path'])
        assert p.resolve() == p and replay.digest(p) == item['sha256'], p
        if 'bytes' in item:
            assert p.stat().st_size == item['bytes']
        artifacts.append(p)
    initial, admitted = (Path(plan[k]) for k in ('initial_host_path', 'dataset_admission_host_path'))
    assert str(admitted) == a['dataset_admission_path'] and replay.read(admitted)['researcher_admitted_dataset']
    assert replay.digest(initial) == plan['initial_sha256'] and replay.digest(admitted) == plan['dataset_admission_sha256']
    reference = Path(plan['benchmark_host_path'])
    assert reference == replay.REFERENCE and str(reference) == a['benchmark_host_path']
    generation_path = reference/'generation.json'
    assert replay.digest(generation_path) == replay.GENERATION == plan['benchmark_generation_sha256']
    generation = replay.read(generation_path)
    assert generation['freeze_sha256'] == freeze['reference_freeze_sha256']
    replay.verify_files(reference, generation['outputs'])
    return freeze, plan, generation, replay.read(initial), artifacts+[initial, admitted, generation_path]


def chart_receipt(freeze, plan, initial):
    root = ARMS['D7']
    path = root/'recovery.json'
    r = replay.read(path)
    assert r['arm'] == 'D7' and r['status'] == 'complete'
    assert r['scientific_work_complete'] is True and r['cleanup_verified'] is True
    assert r['D6_domain_parity_verified'] is True and r['optimizer_returns'] == 10000
    assert [e['step'] for e in r['saved_endpoints']] == STEPS
    assert r['commit'] == COMMIT and r['freeze_sha256'] == FREEZE and r['protocol'] == plan['protocol']
    assert r['policy'] == freeze['policy'] and r['effective_shader'] == freeze['shader']
    assert r['initial'] == replay.read(root/'initial.json') == initial
    assert replay.digest(root/'initial.json') == r['initialization_anchor_sha256'] == plan['initial_sha256']
    assert r['benchmark_sha256'] == replay.GENERATION and r['runtime']['declared_image'] == freeze['image_id']
    assert r['surface_q_policy'] == plan['spatial_policy'] and r['diffuse_white_in_objective'] is True
    assert r['mean_calibration'] is False
    assert r['optimization_terms'] == ['RGB_loss', 'material_prior', 'diffuse_white_prior', 'surface_chart_loss']
    limits, work = plan['limits'], r['counters']
    assert all(work[k] == limits[k] for k in work if k not in ('shader_calls','shadow_rays'))
    assert all(v <= limits[k] for k, v in work.items())
    assert r['elapsed_seconds'] < limits['inner_seconds']
    assert r['resources']['host_peak_RSS'] < limits['host_memory_gib'] * 1024**3
    assert sum(item['bytes'] for item in r['outputs'].values()) < limits['output_bytes']
    exit_path = BATCH/'D7-exit-code.txt'
    assert exit_path.read_text().strip() == '0'
    assert set(r['outputs']) == {p.relative_to(root).as_posix() for p in root.rglob('*')
                                 if p.is_file() and p.name not in ('recovery.json','progress.json')}
    replay.verify_files(root, r['outputs'])
    checks = sorted(root.glob('checkpoint-*.pth'))
    assert [p.name for p in checks] == [f'checkpoint-{s:05d}.pth' for s in STEPS[1:]]
    capture = root/'native-reference-gate.npz'
    g = replay.numeric(capture)
    gate = world.compatibility(g['native_draws'], g['reference'], abs(g['reference']-g['coarse']),
                               r['policy']['compatibility_absolute'], r['policy']['compatibility_relative'])
    assert gate['passed']
    for key, value in gate.items():
        replay.equal(value, r['native_reference_gate'][key])
    history = replay.read_log(root)
    for row in history:
        photo, reg, white, chart = (np.float32(row[k]) for k in r['optimization_terms'])
        assert np.isfinite(chart) and chart >= 0 and row['optimization_loss'] == float(photo+reg+white+chart)
        assert row['gradient_groups']['outer_light'] > 0
    return r, history, [path, exit_path, BATCH/'D7-console.txt', capture,
                       root/'initial.json', root/'training.jsonl', *checks]


def verify_neighborhoods(r, plan, world_history):
    d6, d7 = ARMS['D6'], ARMS['D7']
    domain = replay.numeric(d7/'surface-q-domain.npz')
    assert {k:array_hash(v) for k,v in domain.items()} == plan['D6_domain_array_sha256']
    assert replay.digest(d7/'surface-q-domain.npz') == replay.digest(d6/'surface-q-domain.npz')
    pair_ids, faces, fit = (domain[k] for k in ('diagnostic_ids','face_pairs','fitting_camera_count'))
    support = (fit[faces[pair_ids]]>0).sum(axis=1)
    chart_history = []
    for endpoint, prior in zip(r['saved_endpoints'], world_history):
        step, metrics = endpoint['step'], endpoint['metrics']
        assert metrics['surface_Q_training_coverage'] == prior['training_coverage']
        z = replay.numeric(d7/f'step-{step:05d}/surface-q-diagnostic.npz')
        prior_arrays = replay.numeric(d6/f'step-{step:05d}/surface-q-diagnostic.npz')
        for key in ('pair_ids', 'training_pair_counts','oracle_transported_square'):
            np.testing.assert_array_equal(z[key], prior_arrays[key])
        np.testing.assert_array_equal(z['pair_ids'], pair_ids)
        assert array_hash(z['training_pair_counts']) == plan['D6_sampler_snapshots'][str(step)]
        assert z['training_pair_counts'].sum() == step*plan['spatial_policy']['batch_pairs']
        assert not z['training_pair_counts'][pair_ids].any()
        widths = z['widths']
        assert widths.shape == (len(pair_ids),2,2)
        sq = widths**2
        np.testing.assert_array_equal(z['anisotropy_gap'], abs(sq[...,0]-sq[...,1])/sq.sum(axis=-1))
        v = z['transported_square']
        raw = metrics['surface_Q_smoothness_diagnostic']
        assert raw['pairs'] == len(pair_ids)
        assert np.isfinite(v).all() and (v>=0).all()
        for array, key in ((v,'predicted_mean_squared'),(z['oracle_transported_square'],'oracle_mean_squared')):
            replay.scalar32(float(array.mean()),raw[key],f'D7/{step}/{key}',len(array))
        chart_history.append(dict(step=step, training_coverage=prior['training_coverage'],
                                  diagnostic_raw_transported_sq=replay.stats(v),
                                  gap_p05=float(np.quantile(z['anisotropy_gap'],.05)),
                                  gap_below_point_one=int((z['anisotropy_gap']<.1).sum()),
                                  diagnostic_groups_by_fitting_supported_face_count={
                                      str(k): dict(pairs=int((support==k).sum()),
                                                   transported_sq=replay.stats(v[support==k]))
                                      for k in (0,1,2)}))
    return chart_history


def analyze(out):
    out = out.resolve()
    assert not out.exists() and all(p not in (out,*out.parents)
                                    for p in (BATCH,world.BATCH,replay.BATCH,replay.REFERENCE))
    replay.SCALAR_REPLAY.clear()
    freeze, plan, generation, initial, artifacts = verify_definition()
    refs = {c:replay.numeric(replay.REFERENCE/f'view-{c}.npz') for c in CAMERAS}
    assert replay.digest(world.BATCH/'freeze.json') == world.FREEZE
    control_freeze = replay.read(world.BATCH/'freeze.json')
    control_plan = control_freeze['surface_q_training']
    results = {arm:world.arm_receipt(arm,ARMS[arm],control_freeze,control_plan,initial) for arm in ('D2','D6')}
    results['D7'] = chart_receipt(freeze,plan,initial)
    for field in ('shader_calls','shadow_rays'):
        assert len({r['counters'][field] for r,_,_ in results.values()}) == 1
    lr = [row['lr'] for row in results['D2'][1]]
    assert all([row['lr'] for row in h] == lr for _,h,_ in results.values())
    d6_spatial, count_hash = world.verify_surface_history(results['D6'][0],control_plan,control_freeze,refs)
    d7_spatial = verify_neighborhoods(results['D7'][0],plan,d6_spatial)
    strata = ROOT/'docs/results/donut_calibrated_d3_v1/coverage/camera34-coverage.npz'
    saved = PRIOR/'camera34-coverage.json'
    for p, validation, key in [(strata,strata.parents[1]/'finalization-validation.json','result_hashes'),
                                (saved,PRIOR/'finalization-validation.json','artifact_sha256')]:
        assert replay.digest(p) == replay.read(validation)[key][str(p.relative_to(ROOT))]
    cov, coverage = replay.numeric(strata), replay.read(saved)
    x,y = cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y+x,refs[34]['pixel_indices'][refs[34]['shader_valid']])
    coverage['arms'] = {k:coverage['arms'][k] for k in ('D2','D6')}
    artifacts += [strata,saved]
    replay.ARMS.update(ARMS)
    known_path=ARMS['D2']/'step-10000/illumination.npz'
    known, target = replay.numeric(known_path), np.array(CALIBRATION['mean_linear_RGB'])
    rows,cameras,light,probes,resources = [],[],[],[],{}
    for arm,(r,h,paths) in results.items():
        artifacts += paths
        resources[arm] = {k:r[k] for k in ('elapsed_seconds','resources','counters','runtime','optimizer_returns','cleanup_verified')}
        resources[arm]['output_identities_verified'] = len(r['outputs'])
        if arm in ('D6','D7'):
            resources[arm]['surface_timing'] = r['surface_q_timing']
            artifacts += [ARMS[arm]/f'step-{step:05d}/surface-q-diagnostic.npz' for step in STEPS]
        arm_probes = [dict(arm=arm, completed_updates=row['completed_updates'],**row['gradient_interactions'])
                      for row in h if 'gradient_interactions' in row]
        assert [v['completed_updates'] for v in arm_probes] == [1,*STEPS[1:]]
        if arm=='D7':
            for probe in arm_probes:
                g = probe['terms']['surface_chart']
                assert g['weighted_loss'] == h[probe['completed_updates']-1]['surface_chart_loss']
                assert all(g['groups'][k]==0 for k in ('outer_light','kd_predictor','metallic_predictor'))
                assert all(g['groups'][k]>0 for k in ('mx_predictor','my_predictor','rotation_predictor','feats_network'))
        probes += arm_probes
        for e in r['saved_endpoints']:
            step,m = e['step'],e['metrics']
            assert replay.read(ARMS[arm]/f'step-{step:05d}/metrics.json') == m
            values={c:replay.replay_camera(arm,step,c,refs[c],m[str(c)]) for c in CAMERAS}
            cameras += [dict(arm=arm,step=step,camera=c,**world.augmented_summary(v)) for c,v in values.items()]
            for split, views in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                pooled={k:np.concatenate([values[c][k] for c in views]) for k in values[views[0]]}
                rows.append(dict(arm=arm,step=step,split=split,**world.augmented_summary(pooled)))
            illum = replay.replay_illumination(arm,step,m['illumination'],target,known)
            illum['transported_Q_mean_squared']=m['surface_Q_smoothness_diagnostic']['predicted_mean_squared']
            light.append(illum)
            if step==10000:
                stratified = dict(recovery_report_sha256=replay.digest(ARMS[arm]/'recovery.json'),
                                  **dict(extend_coverage(values[34],cov,coverage)))
                if arm=='D7':
                    coverage['arms'][arm] = stratified
                else:
                    assert stratified == coverage['arms'][arm]
        for c in CAMERAS:
            a,b = replay.materials(arm,0,c),replay.materials('D2',0,c)
            for k in ('kd','mx','my','rotation','Q_world','metallic','linear_rgb'):
                np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(replay.numeric(ARMS[arm]/'step-00000/illumination.npz')['linear_RGB'],
                                      replay.numeric(ARMS['D2']/'step-00000/illumination.npz')['linear_RGB'])
    out.mkdir(parents=True)
    replay.write_csv(out/'endpoint-metrics.csv',rows)
    replay.write_csv(out/'per-camera-metrics.csv',cameras)
    replay.write_json(out/'camera34-coverage.json',coverage)
    replay.write_json(out/'analysis.json',dict(status='verified_complete',execution_commit=COMMIT,
        freeze_sha256=FREEZE,authorization_sha256=AUTH,reference_artifacts=len(generation['outputs']),
        matched_initial_maps=True,matched_learning_rates=True,matched_sampler_snapshots=len(STEPS),
        sampler_count_vector_sha256=count_hash,camera_endpoints=len(cameras),resources=resources,
        illumination=light,gradient_probes=probes,surface_history=dict(D6=d6_spatial,D7=d7_spatial),
        last_200_loss_means={arm:{k:float(np.mean([r[k] for r in h[-200:]])) for k in
            ('RGB_loss','material_prior','diffuse_white_prior','surface_q_loss','surface_chart_loss')
            if k in h[-1]} for arm,(_,h,_) in results.items()},
        report_scalar_replay=replay.SCALAR_REPLAY,
        analysis_work=dict(native_models=0,renderer_calls=0,optimizer_returns=0,checkpoint_deserializations=0)))
    artifacts += [known_path,world.BATCH/'freeze.json']
    replay.write_json(out/'artifact-manifest.json',dict(batch=str(BATCH),
        records={str(p):replay.record(p) for p in sorted(set(artifacts))},
        retained_outputs={arm:dict(path=str(p),endpoints=STEPS,cameras=CAMERAS,output_identities=str(p/'recovery.json'))
                          for arm,p in ARMS.items()}))
    figures(out,rows,light,probes,cov)
    return len(cameras)


def figures(out,rows,light,probes,cov):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    colors=dict(D2='tab:orange',D6='tab:green',D7='tab:blue')

    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=145,bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for arm,color in colors.items():
        held=[r for r in rows if r['arm']==arm and r['split']=='held_out']
        pool=[r for r in rows if r['arm']==arm and r['split']=='all']
        illum=[r for r in light if r['arm']==arm]
        for ax,series,key in zip(axes.flat,[held,held,held,pool,illum,illum],
                ['q_p95','orientation_degrees_p95','RGB_relative_L2','minor_predicted_mean',
                 'transported_Q_mean_squared','relative_L2']):
            ax.plot(STEPS,[r[key] for r in series],'-o',label=arm,color=color)
    axes[1,0].axhline(.08,color='0.4',linestyle='--',label='Reference')
    axes[1,1].set_yscale('log')
    for ax,title in zip(axes.flat,('Held-out relative Q, p95','Held-out eligible orientation, p95 (°)',
                 'Held-out RGB relative L2','Mean minor width','Reserved-edge Q variation','Illumination relative L2')):
        ax.set(title=title,xlabel='Completed Adam updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig,'recovery-curves')

    fig,axes=plt.subplots(2,4,figsize=(11,6),layout='constrained')
    x,y=cov['pixel_xy'].T
    support=np.full((128,128),np.nan)
    support[y,x]=cov['nearby_camera_counts'][:,1]
    axes[0,0].set_title('Nearby fitting views')
    views=axes[0,0].imshow(support,vmin=0,vmax=3,cmap='viridis',interpolation='nearest')
    fig.colorbar(views,ax=axes[0,0],orientation='horizontal',ticks=[0,1,2,3])
    axes[1,0].set_title('Camera 34 reference')
    axes[1,0].imshow(replay.srgb(replay.materials('D7',10000,34)['linear_gt']))
    for j,arm in enumerate(ARMS,1):
        error=np.full((128,128),np.nan)
        error[y,x]=replay.numeric(ARMS[arm]/'step-10000/camera-34-errors.npz')['Q_relative']
        mapped=axes[0,j].imshow(error,vmin=0,vmax=1.25,cmap='magma',interpolation='nearest')
        axes[0,j].set_title(f'{arm} relative Q')
        axes[1,j].set_title(f'{arm} RGB')
        axes[1,j].imshow(replay.srgb(replay.materials(arm,10000,34)['linear_rgb']))
    for ax in axes.flat:
        ax.set(xticks=[],yticks=[])
    fig.colorbar(mapped,ax=axes[0,1:],orientation='horizontal')
    save(fig,'camera34-coverage')

    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for arm,term in [('D6','surface_q'),('D7','surface_chart')]:
        p=[r for r in probes if r['arm']==arm]
        x=[r['completed_updates'] for r in p]
        axes[0].plot(x,[r['terms'][term]['weighted_loss'] for r in p],'-o',color=colors[arm],label=arm)
        axes[1].plot(x,[r['terms'][term]['gradient_l2'] for r in p],'-o',color=colors[arm],label=arm)
        axes[2].plot(x,[r['gradient_cosines']['photo__'+term] for r in p],'-o',color=colors[arm],label=arm)
    axes[2].axhline(0,color='0.4',linewidth=.8)
    for i,(ax,title) in enumerate(zip(axes,('Weighted surface loss','Surface gradient L2','Cosine with RGB gradient'))):
        ax.set(title=title,xlabel='Completed Adam updates')
        if i<2:
            ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig,'gradient-probes')

    names=('reconstructed','albedo','major_width','minor_width','orientation_world_projector','diffuse','specular')
    fig,axes=plt.subplots(len(STEPS),len(names),figsize=(12,10),layout='constrained')
    for i,step in enumerate(STEPS):
        for j,name in enumerate(names):
            ax=axes[i,j]
            with Image.open(ARMS['D7']/f'step-{step:05d}/camera-34/{name}.png') as source:
                ax.imshow(np.array(source),vmin=0,vmax=255,cmap='gray',interpolation='nearest')
            ax.set(xticks=[],yticks=[],title=name.replace('_',' ') if i==0 else '')
        axes[i,0].set_ylabel(f'{step:,} updates')
    save(fig,'D7-camera34-material-progression')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(f'Verified D2, D6, D7 and {analyze(parser.parse_args().output)} saved camera/endpoints')
