"""Verify D9 archives and compare against the accepted D8/D6 tables."""

import argparse
import csv
from pathlib import Path

import numpy as np

import analyze_donut_paired_tensor_d8 as d8
from analyze_donut_camera28_ring_d9 import LUMA
from analyze_donut_diffuse_white_ablation import extend_coverage
from benchmarks.donut_calibration_d3 import CALIBRATION
from benchmarks.donut_recovery_v1 import verify_freeze
from benchmarks.donut_surface_chart_d7 import array_hash
from benchmarks.donut_tensor_d9 import ARMS

replay, world = d8.replay, d8.world
ROOT, CAMERAS, STEPS = d8.ROOT, d8.CAMERAS, d8.STEPS
BATCH = Path('/home/isavand/paper/evidence/donut-kurt-paired-tensor-d9-58f62392c4f84f019fde495849b317c8')
FREEZE = '59811c789ba222e6bcf17507c99d3bc5e49987dc936d0dd69bb76996f323bd85'
AUTH = '6c820cd19be1f7eb3a4001c37bfbcfb15beee403b127756d327a23e56dc3f0be'
COMMIT = '96435ff1991a0cbfd6c85bb5f34c72bae6b87846'
VARIANTS = dict(zip(ARMS,('donut_tensor_chart_weak_start_v1','donut_tensor_ambient_weak_start_v1')))
RUNS = {arm:BATCH/arm for arm in ARMS}
PRIOR = ROOT/'docs/results/donut_paired_tensor_d8_v1'


def verify_definition():
    path=ROOT/'docs/protocols/DONUT_KURT_PAIRED_TENSOR_D9_V1.freeze.json'
    freeze=verify_freeze(path,FREEZE)
    plan=freeze['tensor_training']
    saved,ap=BATCH/'freeze.json',BATCH/'authorization.json'
    host=BATCH.with_name(BATCH.name+'-authorization.json')
    assert replay.digest(saved)==FREEZE and replay.digest(ap)==replay.digest(host)==AUTH
    a=replay.read(ap)
    assert a['approved'] is True and a['researcher_admitted_dataset'] is True
    assert a['maximum_execution_launches']==1 and a['selected_gpu']=='0'
    assert a['commit']==COMMIT and a['freeze_sha256']==FREEZE and a['image_id']==freeze['image_id']
    assert a['host_output_directory']==str(BATCH)==freeze['host_tensor_output']
    assert a['output_directories']=={arm:'/outputs/'+arm for arm in ARMS}
    assert a['arms']==list(ARMS) and a['input_identities']==freeze['inputs']
    assert a['accepted_control_artifacts']==plan['controls']
    for key in ('protocol','limits','spatial_policy','representation','diffuse_white_in_objective',
                'benchmark_generation_sha256','dataset_admission_sha256','initial_sha256'):
        assert a[key]==plan[key]
    files=[saved,ap,host]
    for item in [*freeze['inputs'].values(),*plan['controls'].values()]:
        p=Path(item['path'])
        assert p.resolve()==p and replay.digest(p)==item['sha256'],p
        if 'bytes' in item:
            assert p.stat().st_size==item['bytes']
        files.append(p)
    for label,path_key,sha_key in [('initial_host_path','initial_host_path','initial_sha256'),
                                  ('dataset_admission_path','dataset_admission_host_path','dataset_admission_sha256')]:
        p=Path(plan[path_key])
        assert str(p)==a[label] and replay.digest(p)==plan[sha_key]
        files.append(p)
    assert replay.read(Path(a['dataset_admission_path']))['researcher_admitted_dataset'] is True
    assert str(replay.REFERENCE)==plan['benchmark_host_path']==a['benchmark_host_path']
    generation=replay.REFERENCE/'generation.json'
    assert replay.digest(generation)==replay.GENERATION==plan['benchmark_generation_sha256']
    g=replay.read(generation)
    assert g['freeze_sha256']==freeze['reference_freeze_sha256']
    replay.verify_files(replay.REFERENCE,g['outputs'])
    return freeze,plan,g,replay.read(Path(a['initial_host_path'])),files+[generation]


def verify_arm(arm,freeze,plan,reference,initial,d8_report):
    root=RUNS[arm]
    path,exit_path=root/'recovery.json',BATCH/f'{arm}-exit-code.txt'
    r=replay.read(path)
    assert exit_path.read_text().strip()=='0' and r['status']=='complete'
    assert r['optimizer_returns']==10000 and r['cleanup_verified'] is True and r['scientific_work_complete'] is True
    assert r['arm']==arm and r['commit']==COMMIT and r['freeze_sha256']==FREEZE and r['protocol']==plan['protocol']
    assert r['policy']==freeze['policy'] and r['effective_shader']==freeze['shader']
    assert r['effective_regularization']=='legacy_scalar_energy_specular_white_plus_surface_Q_v1'
    assert r['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior','surface_q_loss']
    assert r['runtime']['declared_image']==freeze['image_id'] and r['benchmark_sha256']==replay.GENERATION
    assert r['representation']==plan['representation'] and r['surface_q_policy']==plan['spatial_policy']
    assert r['D6_domain_parity_verified'] is True and r['diffuse_white_in_objective'] is True
    assert r['mean_calibration'] is False and r['initialization_anchor_sha256']==plan['initial_sha256']
    assert r['initial']==replay.read(root/'initial.json') and r['initial']['variant']==VARIANTS[arm]
    assert r['initial']['seed']==initial['seed'] and r['initial']['common_base_state_sha256']==initial['state_sha256']
    assert r['trainable_parameters']==d8_report['trainable_parameters']
    assert [p['step'] for p in r['saved_endpoints']]==STEPS
    assert r['counters']==d8_report['counters'] and all(v<=plan['limits'][k] for k,v in r['counters'].items())
    assert r['elapsed_seconds']<plan['limits']['inner_seconds']
    assert r['resources']['host_peak_RSS']<plan['limits']['host_memory_gib']*1024**3
    assert len(r['outputs'])==898 and sum(v['bytes'] for v in r['outputs'].values())<plan['limits']['output_bytes']
    assert set(r['outputs'])=={p.relative_to(root).as_posix() for p in root.rglob('*')
                                if p.is_file() and p.name not in ('recovery.json','progress.json')}
    replay.verify_files(root,r['outputs'])
    checks=sorted(root.glob('checkpoint-*.pth'))
    assert [p.name for p in checks]==[f'checkpoint-{step:05d}.pth' for step in STEPS[1:]]
    native=root/'native-reference-gate.npz'
    g=replay.numeric(native)
    valid=np.flatnonzero(reference['shader_valid'])
    ix=valid[np.linspace(0,len(valid)-1,plan['representation']['native_gate_points']).astype(int)]
    for key,value in [('pixel_indices',reference['pixel_indices'][ix]),('reference',reference['linear_components'][ix,2]),
                      ('coarse',reference['coarse_components'][ix,2])]:
        np.testing.assert_array_equal(g[key],value)
    gate=world.compatibility(g['native_draws'],g['reference'],abs(g['reference']-g['coarse']),
                             r['policy']['compatibility_absolute'],r['policy']['compatibility_relative'])
    assert gate['passed']
    for k,v in gate.items():
        replay.equal(v,r['native_reference_gate'][k])
    assert d8_report['native_reference_gate']==r['native_reference_gate']
    pair_path=root/'paired-initial-gate.npz'
    pair=replay.numeric(pair_path)
    stored,cfg=r['paired_initial_gate'],plan['representation']
    assert stored['passed'] is True and stored['shared_state_equal'] is True and stored['scalar_predictions_equal'] is True
    assert stored['local_initial_Q_max_abs']<=cfg['initial_Q_atol']
    for key in ('kd','metallic'):
        np.testing.assert_array_equal(pair['D9-C_'+key],pair['D9-T_'+key])
    def diff(key):
        return float(abs(pair['D9-C_'+key]-pair['D9-T_'+key]).max())
    assert diff('world_Q')==stored['world_Q_max_abs']<=cfg['initial_Q_atol']
    assert max(diff(k) for k in ('linear_rgb','linear_diffuse','linear_specular'))==stored['RGB_component_max_abs']<=cfg['initial_RGB_atol']
    history=replay.read_log(root)
    for row in history:
        photo,reg,white,q=(np.float32(row[k]) for k in r['optimization_terms'])
        assert np.isfinite(q) and q>=0 and row['optimization_loss']==float(photo+reg+white+q)
        assert row['gradient_groups']['outer_light']>0
    log=root/'training.jsonl'
    return r,history,pair,[path,exit_path,BATCH/f'{arm}-console.txt',native,pair_path,root/'initial.json',log,*checks]


def verify_surface(arm,r,plan,accepted):
    root=RUNS[arm]
    p=root/'surface-q-domain.npz'
    domain=replay.numeric(p)
    assert replay.digest(p)==replay.digest(d8.ARMS['D6']/'surface-q-domain.npz')
    assert {k:array_hash(v) for k,v in domain.items()}==plan['D6_domain_array_sha256']
    ids,faces,fit=(domain[k] for k in ('diagnostic_ids','face_pairs','fitting_camera_count'))
    support=(fit[faces[ids]]>0).sum(axis=1)
    result=[]
    for e,a in zip(r['saved_endpoints'],accepted):
        step,m=e['step'],e['metrics']
        assert m['surface_Q_training_coverage']==a['training_coverage']
        z=replay.numeric(root/f'step-{step:05d}/surface-q-diagnostic.npz')
        ctrl=replay.numeric(d8.ARMS['D6']/f'step-{step:05d}/surface-q-diagnostic.npz')
        for k in ('pair_ids','training_pair_counts','oracle_transported_square'):
            np.testing.assert_array_equal(z[k],ctrl[k])
        np.testing.assert_array_equal(z['pair_ids'],ids)
        assert array_hash(z['training_pair_counts'])==plan['D6_sampler_snapshots'][str(step)]
        assert not z['training_pair_counts'][ids].any()
        width=z['widths']
        assert width.shape==(len(ids),2,2) and (width[...,0]>=width[...,1]).all()
        square=width**2
        np.testing.assert_array_equal(z['anisotropy_gap'],abs(square[...,0]-square[...,1])/square.sum(-1))
        v=z['transported_square']
        assert (v>=0).all()
        d=m['surface_Q_smoothness_diagnostic']
        assert d['pairs']==len(ids)
        for array,key in ((v,'predicted_mean_squared'),(z['oracle_transported_square'],'oracle_mean_squared')):
            replay.scalar32(float(array.mean()),d[key],f'{arm}/{step}/{key}',len(array))
        result.append(dict(step=step,training_coverage=a['training_coverage'],
                           transported_sq=replay.stats(v),gap=replay.stats(z['anisotropy_gap']),
                           ineligible_endpoints=int((z['anisotropy_gap']<.1).sum()),
                           by_supported_faces={str(i):replay.stats(v[support==i]) for i in range(3)}))
    return result


def anchor_world(normal,config):
    n=normal/np.linalg.norm(normal,axis=-1,keepdims=True)
    proj=np.eye(3)-n[:,:,None]*n[:,None,:]
    px=proj[:,:,0]
    norm2=(px*px).sum(-1)
    spec=config['initialization']
    low,high=spec['minimum_minor_width']**2,spec['major_width']**2
    midpoint,span=(1+.04**2)/2,(1-.04**2)/2
    def inv(width2):
        y=(width2-midpoint)/span
        return y/np.sqrt(1-y*y)
    z=inv(high)+(inv(low)-inv(high))*norm2
    minor=midpoint+span*z/np.sqrt(1+z*z)
    multiplier=(minor-high)/np.where(norm2>0,norm2,1)
    return high*proj+multiplier[:,None,None]*px[:,:,None]*px[:,None,:]


def ring_statistics(ref,ring,arms):
    valid=ref['shader_valid']
    pixels=ref['pixel_indices'][valid]
    y,x=np.divmod(pixels,128)
    radius=np.linalg.norm(np.column_stack((x,y))-ring['center_xy'],axis=1)
    quartiles=np.digitize(radius,ring['radius_quartiles'])
    region=np.isin(pixels,ring['D8_T_positive_excess_decile']['pixels'])
    truth=ref['linear_components'][valid].astype(float)
    sets={'foreground':np.ones(len(pixels),bool),'D8_T_positive_excess_decile':region,
          'elsewhere':~region,**{f'radius_Q{i+1}':quartiles==i for i in range(4)}}
    result={}
    for arm in arms:
        a=replay.materials(arm,10000,28)
        mask=a['hit_mask']
        np.testing.assert_array_equal(np.flatnonzero(mask),pixels)
        rgb=np.stack([a[k][mask] for k in ('linear_diffuse','linear_specular','linear_rgb')],axis=1).astype(float)
        d=rgb-truth
        result[arm]={}
        for label,use in sets.items():
            s=d[use,1]
            v=d[use,0]
            total=float((d[use,2]**2).sum())
            rs=float((s*s).sum())
            rd=float((v*v).sum())
            cross=float((2*s*v).sum())
            np.testing.assert_allclose(rs+rd+cross,total,rtol=1e-5,atol=1e-7)
            result[arm][label]=dict(observations=int(use.sum()),RGB_SSE=total,specular_SSE=rs,
                diffuse_SSE=rd,cross_SSE=cross,mean_linear_luminance_residual=(d[use]@LUMA).mean(0).tolist(),
                major_mean=float(a['major_width'][mask][use].mean(dtype=float)),
                minor_mean=float(a['minor_width'][mask][use].mean(dtype=float)))
    return result


def analyze(out):
    out=out.resolve()
    assert not out.exists() and all(p not in out.parents for p in (BATCH,d8.BATCH,world.BATCH,replay.REFERENCE))
    replay.SCALAR_REPLAY.clear()
    freeze,plan,g,initial,files=verify_definition()
    refs={c:replay.numeric(replay.REFERENCE/f'view-{c}.npz') for c in CAMERAS}
    sealed=replay.read(PRIOR/'finalization-validation.json')['artifact_sha256']
    p_artifacts=[PRIOR/name for name in ('analysis.json','artifact-manifest.json','endpoint-metrics.csv',
                                         'per-camera-metrics.csv','camera34-coverage.json')]
    for p in p_artifacts:
        assert replay.digest(p)==sealed[str(p.relative_to(ROOT))]
    files+=p_artifacts
    prior=replay.read(PRIOR/'analysis.json')
    ring_path=ROOT/'docs/results/donut_camera28_ring_d9_v1/ring-analysis.json'
    assert replay.digest(ring_path)==freeze['sources'][str(ring_path.relative_to(ROOT))]
    files.append(ring_path)
    ring=replay.read(ring_path)
    cov_path=ROOT/'docs/results/donut_calibrated_d3_v1/coverage/camera34-coverage.npz'
    assert replay.digest(cov_path)==replay.read(cov_path.parents[1]/'finalization-validation.json')['result_hashes'][str(cov_path.relative_to(ROOT))]
    files.append(cov_path)
    cov=replay.numeric(cov_path)
    x,y=cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y+x,refs[34]['pixel_indices'][refs[34]['shader_valid']])
    coverage=replay.read(PRIOR/'camera34-coverage.json')
    replay.ARMS.update({**d8.ARMS,**RUNS})
    for arm,root in d8.ARMS.items():
        accepted=replay.read(root/'recovery.json')
        names=['step-10000/camera-28/materials.npz','step-10000/camera-34/materials.npz']
        if arm=='D6':
            names.append('step-10000/illumination.npz')
        else:
            names.extend(('step-00000/illumination.npz','training.jsonl'))
        replay.verify_files(root,{name:accepted['outputs'][name] for name in names})
        files.extend(root/name for name in names)
    known_path=d8.ARMS['D6']/'step-10000/illumination.npz'
    known=replay.numeric(known_path)
    files.append(known_path)
    target=np.array(CALIBRATION['mean_linear_RGB'])
    rows,cameras,light,probes,direct=[],[],[],[],[]
    resources,final,surface,captures,last200={},{},{},{},{}
    for arm in ARMS:
        control='D8-'+arm[-1]
        old_root=d8.ARMS[control]
        old=replay.read(Path(plan['controls'][control]['path']))
        r,h,pair,paths=verify_arm(arm,freeze,plan,refs[0],initial,old)
        captures[arm]=pair
        files+=paths
        assert [row['lr'] for row in h]==[row['lr'] for row in replay.read_log(old_root)]
        surface[arm]=verify_surface(arm,r,plan,prior['surface_history']['D6'])
        resources[arm]={k:r[k] for k in ('elapsed_seconds','resources','counters','runtime','cleanup_verified',
                         'optimizer_returns','surface_q_timing','tensor_training_forward_CUDA_seconds',
                         'trainable_parameters','native_reference_gate','paired_initial_gate')}
        resources[arm]['output_identities_verified']=len(r['outputs'])
        last200[arm]={k:float(np.mean([v[k] for v in h[-200:]])) for k in r['optimization_terms']}
        arm_probes=[dict(arm=arm,completed_updates=v['completed_updates'],**v['gradient_interactions'])
                    for v in h if 'gradient_interactions' in v]
        assert [v['completed_updates'] for v in arm_probes]==[1,*STEPS[1:]]
        for v in arm_probes:
            q=v['terms']['surface_q']
            assert q['weighted_loss']==h[v['completed_updates']-1]['surface_q_loss']
            assert all(q['groups'][k]==0 for k in ('outer_light','kd_predictor','metallic_predictor'))
        probes+=arm_probes
        files += [RUNS[arm]/'surface-q-domain.npz',
                  *(RUNS[arm]/f'step-{s:05d}/surface-q-diagnostic.npz' for s in STEPS)]
        for e in r['saved_endpoints']:
            step,m=e['step'],e['metrics']
            assert replay.read(RUNS[arm]/f'step-{step:05d}/metrics.json')==m
            values={c:replay.replay_camera(arm,step,c,refs[c],m[str(c)]) for c in CAMERAS}
            direct.extend(d8.direct_q_check(arm,step,c,refs[c],v,plan['representation']['export_world_Q_atol'])
                          for c,v in values.items())
            cameras.extend(dict(arm=arm,step=step,camera=c,**world.augmented_summary(v)) for c,v in values.items())
            for split,cs in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                merged={k:np.concatenate([values[c][k] for c in cs]) for k in values[cs[0]]}
                rows.append(dict(arm=arm,step=step,split=split,**world.augmented_summary(merged)))
            illumination=replay.replay_illumination(arm,step,m['illumination'],target,known)
            illumination['transported_Q_mean_squared']=m['surface_Q_smoothness_diagnostic']['predicted_mean_squared']
            light.append(illumination)
            if step==10000:
                final[arm]=values
                coverage['arms'][arm]=dict(recovery_report_sha256=replay.digest(RUNS[arm]/'recovery.json'),
                                          **dict(extend_coverage(values[34],cov,coverage)))
        np.testing.assert_array_equal(replay.numeric(RUNS[arm]/'step-00000/illumination.npz')['linear_RGB'],
                                      replay.numeric(old_root/'step-00000/illumination.npz')['linear_RGB'])
    assert captures[ARMS[0]].keys()==captures[ARMS[1]].keys()
    for key in captures[ARMS[0]]:
        np.testing.assert_array_equal(captures[ARMS[0]][key],captures[ARMS[1]][key])
    initial_check={}
    for c in CAMERAS:
        a,b=(replay.materials(arm,0,c) for arm in ARMS)
        mask=a['hit_mask']
        np.testing.assert_array_equal(mask,b['hit_mask'])
        for k in ('kd','metallic'):
            np.testing.assert_array_equal(a[k],b[k])
        q=anchor_world(refs[c]['normals'][refs[c]['shader_valid']],plan['representation'])
        anchor=max(float(abs(v['Q_tensor_direct'][mask]-q).max()) for v in (a,b))
        assert anchor<=plan['representation']['export_world_Q_atol']
        initial_check[str(c)]=dict(analytical_anchor_max_abs=anchor,
            **{k:float(abs(a[k]-b[k]).max()) for k in ('Q_tensor_direct','linear_rgb','linear_diffuse','linear_specular')})
    eligibility={}
    for label,cs in [('all',CAMERAS),('held_out',CAMERAS[4:]),('camera28',[28]),('camera34',[34])]:
        cols={a:{k:np.concatenate([final[a][c][k] for c in cs]) for k in ('eligible','orientation_degrees')} for a in ARMS}
        together=cols[ARMS[0]]['eligible']&cols[ARMS[1]]['eligible']
        eligibility[label]=dict(joint_count=int(together.sum()),summary={a:replay.stats(v['orientation_degrees'][together])
                                                                       for a,v in cols.items()})
    out.mkdir(parents=True)
    for name,new in [('endpoint-metrics.csv',rows),('per-camera-metrics.csv',cameras)]:
        with (PRIOR/name).open(newline='') as stream:
            accepted=list(csv.DictReader(stream))
        replay.write_csv(out/name,accepted+new)
    replay.write_json(out/'camera34-coverage.json',coverage)
    replay.write_json(out/'camera28-radial-errors.json',ring_statistics(refs[28],ring,(*d8.ARMS,*ARMS)))
    replay.write_json(out/'analysis.json',dict(status='verified_complete',execution_commit=COMMIT,freeze_sha256=FREEZE,
        authorization_sha256=AUTH,reference_artifacts=len(g['outputs']),frozen_sources=len(freeze['sources']),
        replayed_camera_endpoints=len(cameras),accepted_control_camera_endpoints=108,
        matched_learning_rates=True,matched_D6_sampler_snapshots=len(STEPS),
        initial_view_checks=initial_check,resources=resources,native_reference_captures_agree=True,
        illumination=light,gradient_probes=probes,surface_history=surface,raw_tensor_export_checks=direct,
        joint_orientation_eligibility=eligibility,last_200_loss_means=last200,scalar_replay=replay.SCALAR_REPLAY,
        analysis_work=dict(native_models=0,renderer_calls=0,optimizer_returns=0,checkpoint_deserializations=0)))
    replay.write_json(out/'artifact-manifest.json',dict(batch=str(BATCH),
        records={str(p):replay.record(p) for p in sorted(set(files))},
        retained_outputs={arm:dict(path=str(p),cameras=CAMERAS,endpoints=STEPS,output_identities=str(p/'recovery.json'))
                          for arm,p in {**d8.ARMS,**RUNS}.items()}))
    figures(out)
    return len(cameras)


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'D6':'tab:green','D8-C':'tab:orange','D8-T':'tab:blue','D9-C':'tab:red','D9-T':'tab:purple'}
    with (out/'endpoint-metrics.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    fig,axs=plt.subplots(2,3,figsize=(12,6.3),layout='constrained')
    for arm,color in colors.items():
        for ax,(key,split,title) in zip(axs.flat,[
                ('q_median','held_out','Held-out median Q error'),('q_p95','held_out','Held-out Q p95'),
                ('RGB_relative_L2','held_out','Held-out RGB relative L2'),
                ('minor_predicted_mean','all','Mean minor width'),('minor_at_floor','all','Observations at minor-width floor'),
                ('orientation_degrees_p95','held_out','Eligible orientation p95 (°)')]):
            series=[v for v in rows if v['arm']==arm and v['split']==split]
            ax.plot(STEPS,[float(v[key]) if v[key] else np.nan for v in series],marker='o',color=color,label=arm,
                    linestyle='--' if arm.startswith('D8') else '-')
            ax.set(title=title,xlabel='Completed Adam updates')
            ax.grid(alpha=.2)
    axs[1,0].axhline(.08,color='.5',linestyle=':')
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.savefig(out/'recovery-curves.png',dpi=145,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(2,6,figsize=(14,5.3),layout='constrained')
    for i,c in enumerate((28,34)):
        for j,arm in enumerate(('Oracle',*colors)):
            m=replay.materials('D6' if arm=='Oracle' else arm,10000,c)
            axs[i,j].imshow(replay.srgb(m['linear_gt' if arm=='Oracle' else 'linear_rgb']))
            axs[i,j].set(title=f'Camera {c} · {arm}',xticks=[],yticks=[])
    fig.savefig(out/'held-out-views.png',dpi=145,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(f'Verified D9 and {analyze(parser.parse_args().output)} saved D9 camera/endpoints')
