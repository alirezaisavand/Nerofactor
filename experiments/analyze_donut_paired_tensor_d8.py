"""File-only verification and comparison of the authorized D8 pair."""

import argparse
from pathlib import Path

import numpy as np

import analyze_donut_surface_q_d6 as world
from analyze_donut_diffuse_white_ablation import extend_coverage
from benchmarks.donut_calibration_d3 import CALIBRATION
from benchmarks.donut_recovery_v1 import verify_freeze
from benchmarks.donut_surface_chart_d7 import array_hash

replay = world.replay
ROOT, CAMERAS, STEPS = world.ROOT, world.CAMERAS, world.STEPS
BATCH = Path('/home/isavand/paper/evidence/donut-kurt-paired-tensor-d8-ad9ec0ee9e6347779a46a56b6fea6cd6')
COMMIT = '80594da3f0afca250b9f54703ce8277b651d8235'
FREEZE = '47d1d7b62b528207169a9ec5a70e70462c2da564d348b89c19924a69fa5a289a'
AUTH = '15897042bc2516eb074618e9dfe06ab771c620cefdcf66d233e47cb07779c2c8'
VARIANTS = {'D8-C':'donut_tensor_chart_v1','D8-T':'donut_tensor_ambient_v1'}
ARMS = dict(D6=world.ARMS['D6'], **{arm:BATCH/arm for arm in VARIANTS})
D6_RESULTS = ROOT/'docs/results/donut_surface_q_d6_v1'


def verify_definition():
    freeze_path = ROOT/'docs/protocols/DONUT_KURT_PAIRED_TENSOR_D8_V1.freeze.json'
    freeze = verify_freeze(freeze_path, FREEZE)
    plan = freeze['tensor_training']
    assert plan['arms'] == list(VARIANTS)
    fp, ap = BATCH/'freeze.json', BATCH/'authorization.json'
    host_authorization = BATCH.with_name(BATCH.name+'-authorization.json')
    assert replay.digest(fp) == FREEZE
    assert replay.digest(ap) == replay.digest(host_authorization) == AUTH
    a = replay.read(ap)
    assert a['approved'] is True and a['researcher_admitted_dataset'] is True
    assert a['maximum_execution_launches'] == 1 and a['selected_gpu'] == '0'
    assert str(BATCH) == a['host_output_directory'] == freeze['host_tensor_output']
    assert a['output_directories'] == {arm:'/outputs/'+arm for arm in VARIANTS}
    assert a['commit'] == COMMIT and a['freeze_sha256'] == FREEZE and a['image_id'] == freeze['image_id']
    assert a['diffuse_white_in_objective'] is True and a['input_identities'] == freeze['inputs']
    assert a['accepted_control_artifacts'] == plan['controls']
    for key in ('arms','protocol','limits','spatial_policy','representation',
                'benchmark_generation_sha256','dataset_admission_sha256','initial_sha256'):
        assert a[key] == plan[key]
    artifacts = [fp,ap,host_authorization]
    for item in [*freeze['inputs'].values(),*plan['controls'].values()]:
        p=Path(item['path'])
        assert p.resolve()==p and replay.digest(p)==item['sha256'], p
        if 'bytes' in item:
            assert p.stat().st_size==item['bytes']
        artifacts.append(p)
    initial, admission = (Path(plan[k]) for k in ('initial_host_path','dataset_admission_host_path'))
    assert str(initial)==a['initial_host_path'] and str(admission)==a['dataset_admission_path']
    assert replay.read(admission)['researcher_admitted_dataset'] is True
    assert replay.digest(initial)==plan['initial_sha256'] and replay.digest(admission)==plan['dataset_admission_sha256']
    assert str(replay.REFERENCE)==plan['benchmark_host_path']==a['benchmark_host_path']
    generation_path = replay.REFERENCE/'generation.json'
    assert replay.digest(generation_path)==replay.GENERATION==plan['benchmark_generation_sha256']
    generation = replay.read(generation_path)
    assert generation['freeze_sha256']==freeze['reference_freeze_sha256']
    replay.verify_files(replay.REFERENCE,generation['outputs'])
    return freeze,plan,generation,replay.read(initial),artifacts+[initial,admission,generation_path]


def arm_receipt(arm,freeze,plan,initial,reference,d6_counters):
    root = ARMS[arm]
    path = root/'recovery.json'
    r = replay.read(path)
    assert r['arm']==arm and r['status']=='complete' and r['optimizer_returns']==10000
    assert r['cleanup_verified'] is True and r['scientific_work_complete'] is True
    assert r['D6_domain_parity_verified'] is True and r['diffuse_white_in_objective'] is True
    assert r['mean_calibration'] is False
    assert r['effective_regularization']=='legacy_scalar_energy_specular_white_plus_surface_Q_v1'
    assert r['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior','surface_q_loss']
    assert r['representation']==plan['representation'] and r['surface_q_policy']==plan['spatial_policy']
    assert [e['step'] for e in r['saved_endpoints']]==STEPS
    assert r['commit']==COMMIT and r['freeze_sha256']==FREEZE and r['protocol']==plan['protocol']
    assert r['policy']==freeze['policy'] and r['effective_shader']==freeze['shader']
    assert r['runtime']['declared_image']==freeze['image_id'] and r['benchmark_sha256']==replay.GENERATION
    assert r['initialization_anchor_sha256']==plan['initial_sha256']
    assert r['initial']==replay.read(root/'initial.json') and r['initial']['variant']==VARIANTS[arm]
    assert r['initial']['seed']==initial['seed'] and r['initial']['common_base_state_sha256']==initial['state_sha256']
    inventory=r['trainable_parameters']
    assert inventory['outer_light']==151814 and inventory['inner_light']==0
    assert sum(v for k,v in inventory.items() if not k.endswith('_light'))==1485584+(516 if arm=='D8-T' else 0)
    counters=dict(d6_counters)
    counters['shader_calls']+=2
    counters['shadow_rays']+=2*plan['representation']['native_gate_points']*1536
    assert r['counters']==counters and all(v<=plan['limits'][k] for k,v in counters.items())
    assert r['elapsed_seconds']<plan['limits']['inner_seconds']
    assert r['resources']['host_peak_RSS']<plan['limits']['host_memory_gib']*1024**3
    assert sum(v['bytes'] for v in r['outputs'].values())<plan['limits']['output_bytes']
    exit_path = BATCH/f'{arm}-exit-code.txt'
    assert exit_path.read_text().strip()=='0' and len(r['outputs'])==898
    assert set(r['outputs'])=={p.relative_to(root).as_posix() for p in root.rglob('*')
                                if p.is_file() and p.name not in ('recovery.json','progress.json')}
    replay.verify_files(root,r['outputs'])
    checks=sorted(root.glob('checkpoint-*.pth'))
    assert [p.name for p in checks]==[f'checkpoint-{s:05d}.pth' for s in STEPS[1:]]
    capture=root/'native-reference-gate.npz'
    g=replay.numeric(capture)
    valid=np.flatnonzero(reference['shader_valid'])
    idx=valid[np.linspace(0,len(valid)-1,plan['representation']['native_gate_points']).astype(int)]
    for key,value in [('pixel_indices',reference['pixel_indices'][idx]),('reference',reference['linear_components'][idx,2]),
                      ('coarse',reference['coarse_components'][idx,2])]:
        np.testing.assert_array_equal(g[key],value)
    gate=world.compatibility(g['native_draws'],g['reference'],abs(g['reference']-g['coarse']),
                             r['policy']['compatibility_absolute'],r['policy']['compatibility_relative'])
    assert gate['passed']
    for key,value in gate.items():
        replay.equal(value,r['native_reference_gate'][key])
    group_path=root/'paired-initial-gate.npz'
    paired=replay.numeric(group_path)
    saved=r['paired_initial_gate']
    cfg=plan['representation']
    assert saved['passed'] is True and saved['scalar_predictions_equal'] is True and saved['shared_state_equal'] is True
    assert saved['local_isotropic_Q_max_abs']<=cfg['initial_Q_atol']
    for key in ('kd','metallic'):
        np.testing.assert_array_equal(paired[f'D8-C_{key}'],paired[f'D8-T_{key}'])
    def difference(key):
        return float(abs(paired[f'D8-C_{key}']-paired[f'D8-T_{key}']).max())
    rgb=max(difference(k) for k in ('linear_rgb','linear_diffuse','linear_specular'))
    assert rgb==saved['RGB_component_max_abs']<=cfg['initial_RGB_atol']
    assert difference('world_Q')==saved['world_Q_max_abs']<=cfg['initial_Q_atol']
    history=replay.read_log(root)
    for row in history:
        photo,reg,white,q=(np.float32(row[k]) for k in r['optimization_terms'])
        assert np.isfinite(q) and q>=0 and row['optimization_loss']==float(photo+reg+white+q)
        assert row['gradient_groups']['outer_light']>0
    return r,history,paired,[path,exit_path,BATCH/f'{arm}-console.txt',capture,group_path,
                             root/'initial.json',root/'training.jsonl',*checks]


def surface_history(arm,r,plan,d6_history):
    root=ARMS[arm]
    path=root/'surface-q-domain.npz'
    assert replay.digest(path)==replay.digest(ARMS['D6']/'surface-q-domain.npz')
    domain=replay.numeric(path)
    assert {k:array_hash(v) for k,v in domain.items()}==plan['D6_domain_array_sha256']
    pair_ids,faces,fit=(domain[k] for k in ('diagnostic_ids','face_pairs','fitting_camera_count'))
    support=(fit[faces[pair_ids]]>0).sum(axis=1)
    history=[]
    for endpoint,prior in zip(r['saved_endpoints'],d6_history):
        step,metrics=endpoint['step'],endpoint['metrics']
        assert metrics['surface_Q_training_coverage']==prior['training_coverage']
        z=replay.numeric(root/f'step-{step:05d}/surface-q-diagnostic.npz')
        control=replay.numeric(ARMS['D6']/f'step-{step:05d}/surface-q-diagnostic.npz')
        for key in ('pair_ids','training_pair_counts','oracle_transported_square'):
            np.testing.assert_array_equal(z[key],control[key])
        assert array_hash(z['training_pair_counts'])==plan['D6_sampler_snapshots'][str(step)]
        assert not z['training_pair_counts'][pair_ids].any()
        width=z['widths']
        assert width.shape==(len(pair_ids),2,2) and (width[...,0]>=width[...,1]).all()
        sq=width**2
        np.testing.assert_array_equal(z['anisotropy_gap'],abs(sq[...,0]-sq[...,1])/sq.sum(-1))
        v=z['transported_square']
        assert (v>=0).all()
        d=metrics['surface_Q_smoothness_diagnostic']
        assert d['pairs']==len(pair_ids)
        for array,key in ((v,'predicted_mean_squared'),(z['oracle_transported_square'],'oracle_mean_squared')):
            replay.scalar32(float(array.mean()),d[key],f'{arm}/{step}/{key}',len(array))
        history.append(dict(step=step,training_coverage=prior['training_coverage'],
                            diagnostic_raw_transported_sq=replay.stats(v),gap=replay.stats(z['anisotropy_gap']),
                            ineligible_endpoints=int((z['anisotropy_gap']<.1).sum()),
                            by_supported_faces={str(k):replay.stats(v[support==k]) for k in (0,1,2)}))
    return history


def direct_q_check(arm,step,camera,ref,reported,limit):
    a=replay.materials(arm,step,camera)
    mask=a['hit_mask']
    q=a['Q_tensor_direct'][mask]
    reconstructed=a['Q_world'][mask]
    normal=ref['normals'][ref['shader_valid']]
    truth=ref['Q_world'][ref['shader_valid']]
    reconstruction_error=float(abs(q-reconstructed).max())
    tangent_error=float(abs(np.einsum('nij,nj->ni',q,normal)).max())
    assert reconstruction_error<=limit and tangent_error<=limit
    eig=np.linalg.eigvalsh(q)
    assert (abs(eig[:,0])<=limit).all() and (eig[:,1]>=.04**2-limit).all() and (eig[:,2]<=1+limit).all()
    error=np.linalg.norm(q-truth,axis=(1,2))/np.linalg.norm(truth,axis=(1,2))
    return dict(arm=arm,step=step,camera=camera,reconstruction_max_abs=reconstruction_error,
                projection_max_abs=tangent_error,relative_Q_max_difference=float(abs(error-reported['q']).max()),
                Q_cutoff_disagreements=int(((error<=.05)!=(reported['q']<=.05)).sum()))


def summarize_pair_eligibility(endpoint):
    result={}
    for label,cameras in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:]),
                          ('camera28',[28]),('camera34',[34])]:
        estimates={arm:{k:np.concatenate([endpoint[arm][c][k] for c in cameras])
                        for k in ('eligible','orientation_degrees','q')} for arm in ARMS}
        both=np.logical_and.reduce([estimates[arm]['eligible'] for arm in VARIANTS])
        result[label]=dict(observations=len(both),both_D8_eligible=int(both.sum()),
            by_arm={arm:dict(eligible=int(values['eligible'].sum()),
                orientation_on_joint_D8_subset=replay.stats(values['orientation_degrees'][both]) if both.any() else None,
                Q_on_joint_D8_subset=replay.stats(values['q'][both]) if both.any() else None)
                for arm,values in estimates.items()})
    return result


def analyze(out):
    out=out.resolve()
    assert not out.exists() and all(p not in (out,*out.parents) for p in (BATCH,world.BATCH,replay.REFERENCE))
    replay.SCALAR_REPLAY.clear()
    freeze,plan,generation,initial,artifacts=verify_definition()
    refs={c:replay.numeric(replay.REFERENCE/f'view-{c}.npz') for c in CAMERAS}
    d6_fp=world.BATCH/'freeze.json'
    assert replay.digest(d6_fp)==world.FREEZE
    control=replay.read(d6_fp)
    r6,h6,paths=world.arm_receipt('D6',ARMS['D6'],control,control['surface_q_training'],initial)
    surface,count_hash=world.verify_surface_history(r6,control['surface_q_training'],control,refs)
    results={'D6':(r6,h6)}
    surfaces={'D6':surface}
    artifacts+=[d6_fp,*paths]
    captures={}
    for arm in VARIANTS:
        r,h,capture,paths=arm_receipt(arm,freeze,plan,initial,refs[0],r6['counters'])
        results[arm]=(r,h)
        artifacts+=paths
        surfaces[arm]=surface_history(arm,r,plan,surface)
        captures[arm]=capture
        assert [row['lr'] for row in h]==[row['lr'] for row in h6]
    assert results['D8-C'][0]['initial']['state_sha256']!=results['D8-T'][0]['initial']['state_sha256']
    assert captures['D8-C'].keys()==captures['D8-T'].keys()
    for key in captures['D8-C']:
        np.testing.assert_array_equal(captures['D8-C'][key],captures['D8-T'][key])

    strata=ROOT/'docs/results/donut_calibrated_d3_v1/coverage/camera34-coverage.npz'
    saved=D6_RESULTS/'camera34-coverage.json'
    for p,validation,key in [(strata,strata.parents[1]/'finalization-validation.json','result_hashes'),
                              (saved,D6_RESULTS/'finalization-validation.json','artifact_sha256')]:
        assert replay.digest(p)==replay.read(validation)[key][str(p.relative_to(ROOT))]
    cov,coverage=replay.numeric(strata),replay.read(saved)
    x,y=cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y+x,refs[34]['pixel_indices'][refs[34]['shader_valid']])
    coverage['arms']={'D6':coverage['arms']['D6']}
    artifacts += [strata,saved]
    replay.ARMS.update(ARMS)
    known_path=ARMS['D6']/'step-10000/illumination.npz'
    known,target=replay.numeric(known_path),np.array(CALIBRATION['mean_linear_RGB'])
    rows,cameras,light,probes,direct=[],[],[],[],[]
    resources,final,paired_initial_maps={},{},{}
    for arm,(r,h) in results.items():
        resources[arm]={k:r[k] for k in ('elapsed_seconds','resources','counters','runtime','optimizer_returns',
                                        'cleanup_verified','surface_q_timing','trainable_parameters')}
        resources[arm]['output_identities_verified']=len(r['outputs'])
        artifacts += [ARMS[arm]/'surface-q-domain.npz',
                      *(ARMS[arm]/f'step-{s:05d}/surface-q-diagnostic.npz' for s in STEPS)]
        if arm in VARIANTS:
            for k in ('paired_initial_gate','native_reference_gate','tensor_training_forward_CUDA_seconds'):
                resources[arm][k]=r[k]
        arm_probes=[dict(arm=arm,completed_updates=row['completed_updates'],**row['gradient_interactions'])
                    for row in h if 'gradient_interactions' in row]
        assert [v['completed_updates'] for v in arm_probes]==[1,*STEPS[1:]]
        for row in arm_probes:
            terms=row['terms']
            assert terms['surface_q']['weighted_loss']==h[row['completed_updates']-1]['surface_q_loss']
            assert all(terms['surface_q']['groups'][k]==0 for k in ('outer_light','kd_predictor','metallic_predictor'))
        probes+=arm_probes
        for e in r['saved_endpoints']:
            step,m=e['step'],e['metrics']
            assert replay.read(ARMS[arm]/f'step-{step:05d}/metrics.json')==m
            values={c:replay.replay_camera(arm,step,c,refs[c],m[str(c)]) for c in CAMERAS}
            if arm in VARIANTS:
                direct.extend(direct_q_check(arm,step,c,refs[c],v,plan['representation']['export_world_Q_atol'])
                              for c,v in values.items())
            cameras.extend(dict(arm=arm,step=step,camera=c,**world.augmented_summary(v)) for c,v in values.items())
            for split,cs in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                pooled={k:np.concatenate([values[c][k] for c in cs]) for k in values[cs[0]]}
                rows.append(dict(arm=arm,step=step,split=split,**world.augmented_summary(pooled)))
            illum=replay.replay_illumination(arm,step,m['illumination'],target,known)
            illum['transported_Q_mean_squared']=m['surface_Q_smoothness_diagnostic']['predicted_mean_squared']
            light.append(illum)
            if step==10000:
                final[arm]=values
                entry=dict(recovery_report_sha256=replay.digest(ARMS[arm]/'recovery.json'),
                           **dict(extend_coverage(values[34],cov,coverage)))
                if arm=='D6':
                    assert entry==coverage['arms'][arm]
                else:
                    coverage['arms'][arm]=entry
        np.testing.assert_array_equal(replay.numeric(ARMS[arm]/'step-00000/illumination.npz')['linear_RGB'],
                                      replay.numeric(ARMS['D6']/'step-00000/illumination.npz')['linear_RGB'])
    for c in CAMERAS:
        a,b=(replay.materials(arm,0,c) for arm in VARIANTS)
        for k in ('kd','metallic'):
            np.testing.assert_array_equal(a[k],b[k])
        paired_initial_maps[str(c)]={k:float(abs(a[k]-b[k]).max()) for k in
                                     ('Q_tensor_direct','linear_rgb','linear_diffuse','linear_specular')}
    out.mkdir(parents=True)
    replay.write_csv(out/'endpoint-metrics.csv',rows)
    replay.write_csv(out/'per-camera-metrics.csv',cameras)
    replay.write_json(out/'camera34-coverage.json',coverage)
    replay.write_json(out/'analysis.json',dict(status='verified_complete',execution_commit=COMMIT,freeze_sha256=FREEZE,
        authorization_sha256=AUTH,frozen_sources=len(freeze['sources']),reference_artifacts=len(generation['outputs']),
        camera_endpoints=len(cameras),matched_learning_rates=True,matched_D6_sampler_snapshots=len(STEPS),
        matched_initial_light=True,matched_initial_probe_captures=True,initial_view_differences=paired_initial_maps,
        sampler_count_vector_sha256=count_hash,resources=resources,illumination=light,gradient_probes=probes,
        surface_history=surfaces,raw_tensor_export_checks=direct,paired_eligibility=summarize_pair_eligibility(final),
        last_200_loss_means={arm:{k:float(np.mean([r[k] for r in h[-200:]])) for k in
            ('RGB_loss','material_prior','diffuse_white_prior','surface_q_loss')} for arm,(_,h) in results.items()},
        report_scalar_replay=replay.SCALAR_REPLAY,
        analysis_work=dict(native_models=0,renderer_calls=0,optimizer_returns=0,checkpoint_deserializations=0)))
    artifacts.append(known_path)
    replay.write_json(out/'artifact-manifest.json',dict(batch=str(BATCH),
        records={str(p):replay.record(p) for p in sorted(set(artifacts))},
        retained_outputs={arm:dict(path=str(p),cameras=CAMERAS,endpoints=STEPS,output_identities=str(p/'recovery.json'))
                          for arm,p in ARMS.items()}))
    figures(out,rows,light,final,cov)
    return len(cameras)


def figures(out,rows,light,final,cov):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors={'D6':'tab:green','D8-C':'tab:orange','D8-T':'tab:blue'}

    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=150,bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for arm,color in colors.items():
        held=[r for r in rows if r['arm']==arm and r['split']=='held_out']
        pool=[r for r in rows if r['arm']==arm and r['split']=='all']
        lighting=[v for v in light if v['arm']==arm]
        for ax,series,key in zip(axes.flat,[held,held,pool,pool,held,lighting],
                ('q_p95','RGB_relative_L2','major_predicted_mean','minor_predicted_mean',
                 'orientation_eligible','relative_L2')):
            ax.plot(STEPS,[v[key] for v in series],'-o',label=arm,color=color)
    axes[0,2].axhline(.35,color='0.4',linestyle='--',label='Oracle')
    axes[1,0].axhline(.08,color='0.4',linestyle='--',label='Oracle')
    for ax,label in zip(axes.flat,('Held-out Q error (p95)','Held-out RGB relative L2','Mean major width',
                'Mean minor width','Held-out orientation-eligible count','Illumination relative L2')):
        ax.set(title=label,xlabel='Completed Adam updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig,'recovery-curves')

    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for arm,color in colors.items():
        d=[final[arm][i] for i in CAMERAS[4:]]
        major,minor=(np.concatenate([v[k] for v in d]) for k in ('major','minor'))
        q=np.concatenate([v['q'] for v in d])
        gap=(major**2-minor**2)/(major**2+minor**2)
        angles=np.concatenate([v['orientation_degrees'] for v in d])
        for ax,values in zip(axes,(q,gap,angles[gap>=.1])):
            if len(values):
                ax.step(np.sort(values),np.arange(1,len(values)+1)/len(values),label=arm,color=color)
    axes[0].axvline(.05,color='0.5',linestyle='--')
    axes[1].axvline(.1,color='0.5',linestyle='--')
    for ax,name in zip(axes,('Relative Q error','Anisotropy gap','Orientation error among eligible (°)')):
        ax.set(xlabel=name,ylabel='Held-out fraction')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig,'held-out-distributions')

    fig,axes=plt.subplots(3,4,figsize=(10,7.5),layout='constrained')
    x,y=cov['pixel_xy'].T
    support=np.full((128,128),np.nan)
    support[y,x]=cov['nearby_camera_counts'][:,1]
    views=axes[2,0].imshow(support,vmin=0,vmax=3,cmap='viridis',interpolation='nearest')
    axes[2,0].set_title('Camera 34 fitting views',fontsize=9)
    fig.colorbar(views,ax=axes[2,0],orientation='horizontal',ticks=[0,1,2,3])
    for row,camera in enumerate(CAMERAS[4:]):
        for j,arm in enumerate(('Oracle',*ARMS)):
            m=replay.materials('D6' if arm=='Oracle' else arm,10000,camera)
            axes[row,j].imshow(replay.srgb(m['linear_gt' if arm=='Oracle' else 'linear_rgb']))
            axes[row,j].set_title(f'Camera {camera} · {arm}',fontsize=9)
    for j,arm in enumerate(ARMS,1):
        error=np.full((128,128),np.nan)
        error[y,x]=final[arm][34]['q']
        mapped=axes[2,j].imshow(error,vmin=0,vmax=1.25,cmap='magma',interpolation='nearest')
        axes[2,j].set_title(f'{arm} camera 34 Q error',fontsize=9)
    fig.colorbar(mapped,ax=axes[2,1:],orientation='horizontal')
    for ax in axes.flat:
        ax.set(xticks=[],yticks=[])
    save(fig,'held-out-views')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(f'Verified D6, D8-C, D8-T and {analyze(parser.parse_args().output)} saved camera/endpoints')
