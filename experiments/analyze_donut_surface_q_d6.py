"""Replay the saved Donut D6 evidence against accepted D2 and D4 outputs."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

import analyze_donut_calibrated_d3 as replay
from analyze_donut_diffuse_white_ablation import BATCH as D45_BATCH, extend_coverage
from benchmarks.donut_calibration_d3 import CALIBRATION
from benchmarks.donut_surface_q_d6 import NeighborhoodSampler, coverage_summary, fitting_face_counts, neighborhoods
from benchmarks.neural_observation_e16 import compatibility

ROOT = replay.ROOT
BATCH = Path('/home/isavand/paper/evidence/donut-kurt-surface-q-d6-cde03768f4a14f13b231e82149b49ec7')
COMMIT = '1a66ff579b4a330942f4a78ea25374b0c2e87332'
FREEZE = '02b181b05bf499cb312f634622122977ea47a24f0c18d93ccb150409fa4ecec4'
AUTH = 'c0f096e194980d6932d50cd54d0f2882bbdef65147611e16e30fa389fc4d245a'
FREEZE_PATH = ROOT / 'docs/protocols/DONUT_KURT_SURFACE_Q_D6_V1.freeze.json'
PRIOR = ROOT / 'docs/results/donut_diffuse_white_ablation_v1'
PREPARATION = ROOT / 'docs/evidence/donut_surface_q_d6_preparation_v1'
CAMERAS, STEPS = replay.CAMERAS, replay.STEPS
ARMS = dict(D2=replay.BATCH/'D2', D4=D45_BATCH/'D4', D6=BATCH/'D6')


def verify_definition():
    saved, auth_path = BATCH/'freeze.json', BATCH/'authorization.json'
    assert replay.digest(saved) == replay.digest(FREEZE_PATH) == FREEZE
    assert replay.digest(auth_path) == replay.digest(Path(str(BATCH)+'-authorization.json')) == AUTH
    freeze, auth = replay.read(saved), replay.read(auth_path)
    d = freeze['surface_q_training']
    assert auth['approved'] and auth['researcher_admitted_dataset'] and auth['maximum_execution_launches'] == 1
    assert auth['host_output_directory'] == str(BATCH) == freeze['host_surface_output']
    assert auth['protocol'] == d['protocol'] and auth['limits'] == d['limits'] and auth['spatial_policy'] == d['spatial_policy']
    assert auth['output_directories'] == {'D6':'/outputs/D6'} and auth['diffuse_white_in_objective'] is True
    assert auth['commit'] == COMMIT and auth['freeze_sha256'] == FREEZE and auth['image_id'] == freeze['image_id']
    assert auth['accepted_control_receipts'] == d['controls'] and auth['selected_gpu'] == '0'
    for name, sha in freeze['sources'].items():
        assert replay.digest(ROOT/name) == sha, name
    for item in freeze['inputs'].values():
        assert replay.digest(Path(item['path'])) == item['sha256']
    admission, initial = Path(auth['dataset_admission_path']), Path(d['initial_host_path'])
    assert replay.digest(admission) == auth['dataset_admission_sha256'] == d['dataset_admission_sha256']
    assert replay.read(admission)['researcher_admitted_dataset']
    assert replay.digest(initial) == d['initial_sha256'] == auth['initial_sha256']
    generation = replay.REFERENCE/'generation.json'
    assert replay.digest(generation) == replay.GENERATION == d['benchmark_generation_sha256'] == auth['benchmark_generation_sha256']
    data = replay.read(generation)
    assert data['freeze_sha256'] == freeze['reference_freeze_sha256']
    replay.verify_files(replay.REFERENCE, data['outputs'])
    return freeze, d, data, replay.read(initial), [saved, auth_path, admission, initial, generation]


def arm_receipt(arm, root, freeze, definition, initial):
    path = root/'recovery.json'
    r = replay.read(path)
    assert r['arm'] == arm and r['status'] == 'complete' and r['scientific_work_complete'] and r['cleanup_verified']
    assert r['optimizer_returns'] == 10000 and [v['step'] for v in r['saved_endpoints']] == STEPS
    assert r['policy'] == freeze['policy'] and r['effective_shader'] == freeze['shader']
    assert r['initial'] == replay.read(root/'initial.json') == initial
    assert r['runtime']['declared_image'] == freeze['image_id'] and r['benchmark_sha256'] == replay.GENERATION
    exit_path = root.parent/f'{arm}-exit-code.txt'
    assert exit_path.read_text().strip() == '0'
    replay.verify_files(root, r['outputs'])
    checkpoints = sorted(root.glob('checkpoint-*.pth'))
    assert [p.name for p in checkpoints] == [f'checkpoint-{step:05d}.pth' for step in STEPS[1:]]
    npz = root/'native-reference-gate.npz'
    g = replay.numeric(npz)
    gate = compatibility(g['native_draws'],g['reference'],abs(g['reference']-g['coarse']),
                         r['policy']['compatibility_absolute'],r['policy']['compatibility_relative'])
    assert gate['passed']
    for k,v in gate.items():
        if isinstance(v,(int,bool)):
            assert v == r['native_reference_gate'][k]
        else:
            replay.equal(v,r['native_reference_gate'][k])
    history = replay.read_log(root)
    if arm != 'D6':
        assert definition['controls'][arm] == dict(path=str(path),sha256=replay.digest(path))
        assert len(r['outputs']) == 890
    else:
        assert r['commit'] == COMMIT and r['freeze_sha256'] == FREEZE and r['protocol'] == definition['protocol']
        assert r['surface_q_policy'] == definition['spatial_policy'] and r['initialization_anchor_sha256'] == definition['initial_sha256']
        assert r['diffuse_white_in_objective'] is True and r['mean_calibration'] is False
        assert r['optimization_terms'] == ['RGB_loss','material_prior','diffuse_white_prior','surface_q_loss']
        assert len(r['outputs']) == 897
        counters, limits = r['counters'], definition['limits']
        assert counters['shader_calls'] == 12288 and counters['shadow_rays'] == 9533964288
        assert all(counters[k] == limits[k] for k in ('component_reverse_sweeps','surface_q_batches','surface_q_pairs',
                                                        'surface_q_predictor_calls','surface_q_predictor_points'))
        assert all(v <= limits[k] for k,v in counters.items())
        assert r['elapsed_seconds'] < limits['inner_seconds']
        assert r['resources']['host_peak_RSS'] < limits['host_memory_gib'] * 1024**3
        assert sum(v['bytes'] for v in r['outputs'].values()) < limits['output_bytes']
        for row in history:
            photo, reg, white, spatial = (np.float32(row[k]) for k in r['optimization_terms'])
            assert np.isfinite(spatial) and spatial >= 0
            assert row['optimization_loss'] == float(photo+reg+white+spatial)
        assert all(row['gradient_groups']['outer_light'] > 0 for row in history)
    artifacts = [path,exit_path,root.parent/f'{arm}-console.txt',root/'initial.json',root/'training.jsonl',npz,*checkpoints]
    return r, history, artifacts


def augmented_summary(values):
    result = replay.summarize(values)
    major, minor = values['major'], values['minor']
    gap = (major**2-minor**2)/(major**2+minor**2)
    assert np.array_equal(gap >= .1,values['eligible'])
    result.update({f'anisotropy_gap_{k}':v for k,v in replay.stats(gap).items()})
    result.update(gap_p05=float(np.quantile(gap,.05)), major_at_ceiling=int((major >= .9999).sum()),
                  minor_signed_relative_bias=float(minor.mean(dtype=float)/.08-1))
    return result


def verify_surface_history(r, definition, freeze, refs):
    root=ARMS['D6']
    geometry = replay.numeric(Path(freeze['inputs']['oracle-geometry-and-frames.npz']['path']))
    p = definition['spatial_policy']
    actual = replay.numeric(root/'surface-q-domain.npz')
    reconstructed = neighborhoods(geometry,p)
    for k,v in reconstructed.items():
        np.testing.assert_array_equal(actual[k],v)
    fit = fitting_face_counts(refs,freeze['splits']['fitting'],len(actual['face_area']))
    np.testing.assert_array_equal(fit,actual['fitting_camera_count'])
    sampler = NeighborhoodSampler(actual['training_ids'],len(actual['face_pairs']),p['rng_seed'])
    pair_ids = actual['diagnostic_ids']
    support = (fit[actual['face_pairs'][pair_ids]]>0).sum(1)
    prior = replay.read(PREPARATION/'geometry-and-scale.json')
    preparation = replay.read(PREPARATION/'preparation.json')
    assert preparation['freeze_sha256'] == FREEZE
    assert replay.digest(PREPARATION/'geometry-and-scale.json') == preparation['geometry_and_scale_report_sha256']
    spatial = []
    previous = 0
    for e in r['saved_endpoints']:
        step = e['step']
        for _ in range(previous,step):
            sampler.draw(p['batch_pairs'])
        previous = step
        z = replay.numeric(root/f'step-{step:05d}/surface-q-diagnostic.npz')
        np.testing.assert_array_equal(z['pair_ids'],pair_ids)
        np.testing.assert_array_equal(z['training_pair_counts'],sampler.counts)
        assert not sampler.counts[pair_ids].any()
        cov = coverage_summary(actual,sampler.counts,fit)
        assert cov == e['metrics']['surface_Q_training_coverage']
        assert cov['pair_draws'] == step*p['batch_pairs']
        if step:
            assert cov['surface_area_fraction'] == 1.0 and cov['visited_faces'] == p['expected_mesh_faces']
        width = z['widths']
        assert width.shape == (len(pair_ids),2,2)
        sq = width**2
        np.testing.assert_array_equal(z['anisotropy_gap'],abs(sq[...,0]-sq[...,1])/sq.sum(axis=-1))
        v = z['transported_square']
        diagnostic = e['metrics']['surface_Q_smoothness_diagnostic']
        for array,key in ((v,'predicted_mean_squared'),(z['oracle_transported_square'],'oracle_mean_squared')):
            replay.scalar32(float(array.mean()),diagnostic[key],f'D6/{step}/{key}',len(array))
        groups = {str(k):dict(pairs=int((support==k).sum()), transported_sq=replay.stats(v[support==k]),
                              gap=replay.stats(z['anisotropy_gap'][support==k])) for k in (0,1,2)}
        spatial.append(dict(step=step,training_coverage=cov,diagnostic_raw_transported_sq=replay.stats(v),
            diagnostic_groups_by_fitting_supported_face_count=groups,
            diagnostic_gap=replay.stats(z['anisotropy_gap']),gap_p05=float(np.quantile(z['anisotropy_gap'],.05)),
            diagnostic_major_mean=float(width.max(-1).mean(dtype=float)),
            diagnostic_minor_mean=float(width.min(-1).mean(dtype=float)),
            diagnostic_gap_below_point_one=int((z['anisotropy_gap']<.1).sum())))
    final_hash=hashlib.sha256(sampler.counts.astype('<i8').tobytes()).hexdigest()
    assert final_hash == prior['dry_sampler']['count_vector_little_endian_int64_sha256']
    assert sampler.counts.sum() == r['counters']['surface_q_pairs']
    return spatial, final_hash


def analyze(out):
    out = out.resolve()
    assert not out.exists() and not any(root in (out,*out.parents) for root in (BATCH,replay.BATCH,D45_BATCH,replay.REFERENCE))
    freeze, definition, generation, initial, artifacts = verify_definition()
    refs = {c:replay.numeric(replay.REFERENCE/f'view-{c}.npz') for c in CAMERAS}
    control = replay.read(PRIOR/'finalization-validation.json')['result_hashes']
    strata_path = replay.ROOT/'docs/results/donut_calibrated_d3_v1/coverage/camera34-coverage.npz'
    strata_report = PRIOR/'camera34-coverage.json'
    assert replay.digest(strata_report) == control[str(strata_report.relative_to(ROOT))]
    saved_strata = replay.read(strata_path.parents[1]/'finalization-validation.json')['result_hashes']
    assert replay.digest(strata_path) == saved_strata[str(strata_path.relative_to(ROOT))]
    cov, coverage = replay.numeric(strata_path),replay.read(strata_report)
    x,y = cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y+x,refs[34]['pixel_indices'][refs[34]['shader_valid']])
    coverage['arms'] = {arm:coverage['arms'][arm] for arm in ('D2','D4')}
    artifacts += [strata_report,strata_path]
    replay.ARMS.update(ARMS)
    rows,cameras,light,probes = [],[],[],[]
    histories,resources,distributions = {},{},{}
    known_path=ARMS['D2']/'step-10000/illumination.npz'
    known = replay.numeric(known_path)
    for arm, root in ARMS.items():
        r, history, paths = arm_receipt(arm,root,freeze,definition,initial)
        artifacts += paths
        histories[arm] = history
        resources[arm] = {k:r[k] for k in ('elapsed_seconds','resources','counters','runtime','optimizer_returns','cleanup_verified')}
        resources[arm]['output_identities_verified'] = len(r['outputs'])
        resources[arm]['native_reference_gate'] = r['native_reference_gate']
        if arm=='D6':
            spatial, count_hash = verify_surface_history(r,definition,freeze,refs)
            resources[arm]['surface_q_timing']=r['surface_q_timing']
            artifacts.append(root/'surface-q-domain.npz')
        for row in history:
            if 'gradient_interactions' in row:
                terms=row['gradient_interactions']
                if arm=='D6':
                    q=terms['terms']['surface_q']
                    assert q['weighted_loss']==row['surface_q_loss']
                    assert q['groups']['outer_light']==0 and q['groups']['mx_predictor']>0
                    assert q['groups']['my_predictor']>0 and q['groups']['rotation_predictor']>0
                probes.append(dict(arm=arm,completed_updates=row['completed_updates'],**terms))
        assert sum(row['arm']==arm for row in probes)==6
        for endpoint in r['saved_endpoints']:
            step, m=endpoint['step'],endpoint['metrics']
            assert replay.read(root/f'step-{step:05d}/metrics.json') == m
            pieces={}
            for camera in CAMERAS:
                values=replay.replay_camera(arm,step,camera,refs[camera],m[str(camera)])
                pieces[camera]=values
                cameras.append(dict(arm=arm,step=step,camera=camera,**augmented_summary(values)))
            for split, views in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                combined={k:np.concatenate([pieces[c][k] for c in views]) for k in pieces[views[0]]}
                rows.append(dict(arm=arm,step=step,split=split,**augmented_summary(combined)))
                if step==10000:
                    distributions[arm,split]=combined
            illum=replay.replay_illumination(arm,step,m['illumination'],np.array(CALIBRATION['mean_linear_RGB']),known)
            illum['transported_Q_mean_squared']=m['surface_Q_smoothness_diagnostic']['predicted_mean_squared']
            assert m['surface_Q_smoothness_diagnostic']['pairs']==1024
            light.append(illum)
            if step==10000:
                result=dict(recovery_report_sha256=replay.digest(root/'recovery.json'),
                            **dict(extend_coverage(pieces[34],cov,coverage)))
                if arm=='D6':
                    coverage['arms'][arm]=result
                else:
                    assert coverage['arms'][arm]==result
        for camera in CAMERAS:
            a,b=replay.materials(arm,0,camera),replay.materials('D2',0,camera)
            for k in ('kd','mx','my','rotation','Q_world','metallic'):
                np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(replay.numeric(ARMS['D6']/'step-00000/illumination.npz')['linear_RGB'],
                                  replay.numeric(ARMS['D2']/'step-00000/illumination.npz')['linear_RGB'])
    artifacts += [known_path, PREPARATION/'geometry-and-scale.json']
    out.mkdir(parents=True)
    replay.write_csv(out/'endpoint-metrics.csv',rows)
    replay.write_csv(out/'per-camera-metrics.csv',cameras)
    replay.write_json(out/'camera34-coverage.json',coverage)
    replay.write_json(out/'analysis.json',dict(status='verified_complete',execution_commit=COMMIT,freeze_sha256=FREEZE,
        authorization_sha256=AUTH,generation_sha256=replay.GENERATION,frozen_sources=len(freeze['sources']),
        reference_artifacts=len(generation['outputs']),camera_endpoints=len(cameras),matched_initial_material=True,
        matched_D2_D6_initial_light=True,resources=resources,illumination=light,surface_history=spatial,
        verified_sampler_count_vector_sha256=count_hash,gradient_probes=probes,
        last_200_loss_means={arm:{k:float(np.mean([r[k] for r in h[-200:]])) for k in h[-1] if
                             k in ('RGB_loss','material_prior','diffuse_white_prior','surface_q_loss')} for arm,h in histories.items()},
        report_scalar_replay=replay.SCALAR_REPLAY,
        analysis_work=dict(native_models=0,renderer_calls=0,optimizer_returns=0,checkpoint_deserializations=0)))
    replay.write_json(out/'artifact-manifest.json',dict(batch=str(BATCH),
        records={str(path):replay.record(path) for path in sorted(set(artifacts))},
        retained_outputs={arm:dict(path=str(root),endpoints=STEPS,cameras=CAMERAS,
          output_identities=str(root/'recovery.json')) for arm,root in ARMS.items()}))
    figures(out,rows,light,histories,probes,distributions,cov)
    return len(cameras)


def figures(out,rows,light,histories,probes,distributions,cov):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    colors=dict(D2='tab:orange', D4='tab:purple', D6='tab:green')

    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=145,bbox_inches='tight')
        plt.close(fig)

    def image(ax,a,title='',**kwargs):
        ax.set(title=title,xticks=[],yticks=[])
        return ax.imshow(a,interpolation='nearest',**kwargs)

    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for arm,color in colors.items():
        pooled=[r for r in rows if r['arm']==arm and r['split']=='all']
        held=[r for r in rows if r['arm']==arm and r['split']=='held_out']
        illumination=[r for r in light if r['arm']==arm]
        for ax,series,key in zip(axes.flat,[pooled,held,held,held,pooled,pooled,illumination,illumination],
                ['q_median','q_p95','orientation_degrees_p95','RGB_relative_L2','minor_predicted_mean',
                 'anisotropy_gap_median','transported_Q_mean_squared','relative_L2']):
            ax.plot(STEPS,[r[key] for r in series],'-o',label=arm,color=color)
    axes[1,0].axhline(.08,color='0.4',linestyle='--',label='Oracle')
    axes[1,1].axhline((.35**2-.08**2)/(.35**2+.08**2),color='0.4',linestyle='--',label='Oracle')
    axes[1,2].axhline(3.0729609079571674e-6,color='0.4',linestyle='--',label='Oracle')
    for i,(ax,title) in enumerate(zip(axes.flat,('Overall median relative Q','Held-out relative Q p95',
            'Held-out eligible orientation p95 (°)','Held-out RGB relative L2','Mean minor width',
            'Median predicted anisotropy gap','Reserved-edge transported Q variation','Illumination relative L2'))):
        ax.set(title=title,xlabel='Completed Adam updates')
        if i in (0,1,3,6):ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig,'recovery-curves')

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,split in enumerate(('all','held_out')):
        for j,(key,title) in enumerate((('q','Relative Q error'),
                ('orientation_degrees','Eligible orientation error (°)'),('gap','Predicted anisotropy gap'))):
            ax=axes[i,j]
            for arm,color in colors.items():
                d=distributions[arm,split]
                if key=='gap':
                    major,minor=d['major'],d['minor']
                    values=(major**2-minor**2)/(major**2+minor**2)
                else:
                    values=d[key][d['eligible']] if key=='orientation_degrees' else d[key]
                ax.step(np.sort(values),np.arange(1,len(values)+1)/len(values),label=arm,color=color)
            ax.set(xlabel=title,ylabel=split.replace('_',' ')+' fraction')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    save(fig,'endpoint-distributions')

    fig,axes=plt.subplots(1,4,figsize=(11.5,3.4),layout='constrained')
    x,y=cov['pixel_xy'].T
    heat=np.full((128,128),np.nan)
    heat[y,x]=cov['nearby_camera_counts'][:,1]
    count=image(axes[0],heat,'Nearby fitting views',vmin=0,vmax=3,cmap='viridis')
    fig.colorbar(count,ax=axes[0],orientation='horizontal',ticks=[0,1,2,3])
    for ax,arm in zip(axes[1:],ARMS):
        q=replay.numeric(ARMS[arm]/'step-10000/camera-34-errors.npz')['Q_relative']
        error=np.full((128,128),np.nan)
        error[y,x]=q
        colorbar=image(ax,error,arm,vmin=0,vmax=1.25,cmap='magma')
    fig.colorbar(colorbar,ax=axes[1:],label='Relative Q error',orientation='horizontal')
    save(fig,'camera34-coverage')

    fig,axes=plt.subplots(6,4,figsize=(9,12),layout='constrained')
    for i,cam in enumerate(CAMERAS):
        for j,arm in enumerate(('Reference',*ARMS)):
            a=replay.materials('D6' if arm=='Reference' else arm,10000,cam)
            image(axes[i,j],replay.srgb(a['linear_gt' if arm=='Reference' else 'linear_rgb']),arm if i==0 else '')
        axes[i,0].set_ylabel(f'Camera {cam}\n'+('fitting' if i<4 else 'held out'))
    save(fig,'RGB-comparison')

    fig,axes=plt.subplots(6,6,figsize=(11,10),layout='constrained')
    for i,step in enumerate(STEPS):
        for j,cam in enumerate(CAMERAS):
            with Image.open(ARMS['D6']/f'step-{step:05d}/camera-{cam}/reconstructed.png') as source:
                image(axes[i,j],np.array(source),f'Camera {cam}' if i==0 else '')
        axes[i,0].set_ylabel(f'{step:,} updates')
    save(fig,'D6-six-view-progression')

    fig,axes=plt.subplots(6,7,figsize=(12,10),layout='constrained')
    names=('reconstructed','albedo','major_width','minor_width','orientation_world_projector','diffuse','specular')
    for i,step in enumerate(STEPS):
        for j,name in enumerate(names):
            with Image.open(ARMS['D6']/f'step-{step:05d}/camera-34/{name}.png') as source:
                image(axes[i,j],np.array(source),name.replace('_',' ') if i==0 else '',vmin=0,vmax=255,cmap='gray')
        axes[i,0].set_ylabel(f'{step:,} updates')
    save(fig,'D6-camera34-material-progression')

    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for arm,color in colors.items():
        history=histories[arm]
        keys=['RGB_loss','material_prior']+(['surface_q_loss'] if arm=='D6' else [])
        for key,style in zip(keys,('-','--',':')):
            means=np.convolve([r[key] for r in history],np.full(200,1/200),mode='valid')
            axes[0].plot(np.arange(200,10001),means,style,color=color,label=arm+' '+key)
    p=[r for r in probes if r['arm']=='D6']
    x=[r['completed_updates'] for r in p]
    for key in ('photo','material','surface_q'):
        axes[1].plot(x,[r['terms'][key]['gradient_l2'] for r in p],'-o',label=key)
    axes[2].plot(x,[r['gradient_cosines']['photo__surface_q'] for r in p],'-o',color=colors['D6'],label='Photo vs surface-Q')
    axes[2].axhline(0,color='0.4',linewidth=.8)
    for i,(ax,title) in enumerate(zip(axes,('Training losses (200-update mean)','D6 component gradient norms',
                                         'D6 gradient cosine'))):
        ax.set(title=title,xlabel='Completed Adam updates')
        if i<2:ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig,'losses-and-gradients')

    fig,axes=plt.subplots(3,3,figsize=(10,6),layout='constrained')
    for i,arm in enumerate(ARMS):
        z=replay.numeric(ARMS[arm]/'step-10000/illumination.npz')
        light,oracle=z['linear_RGB'],z['known_linear_RGB']
        axes[i,0].set_ylabel(arm)
        image(axes[i,0],replay.srgb(oracle),'Reference light' if i==0 else '')
        image(axes[i,1],replay.srgb(light),'Recovered light' if i==0 else '')
        image(axes[i,2],np.sqrt(np.mean((light-oracle)**2,axis=-1)),'Linear RMSE [0,.6]' if i==0 else '',
              vmin=0,vmax=.6,cmap='magma')
    save(fig,'illumination')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(f'Verified D2, D4, D6 and {analyze(args.output)} saved camera/endpoints')
