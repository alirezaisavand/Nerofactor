"""Replay saved D11 predictions, compare with D8-T and report D6 context."""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[i]) for i in (0, 1)]

import numpy as np

import analyze_donut_paired_tensor_d8 as d8
from analyze_donut_paired_tensor_d9 import ring_statistics
from analyze_donut_diffuse_white_ablation import extend_coverage
from benchmarks import donut_log_tensor_d11 as d11
from benchmarks.donut_calibration_d3 import CALIBRATION
from benchmarks.donut_recovery_v1 import verify_freeze
from benchmarks.donut_surface_chart_d7 import array_hash

ROOT, CAMERAS, STEPS = d8.ROOT, d8.CAMERAS, d8.STEPS
replay, world = d8.replay, d8.world
RING = ROOT/'docs/results/donut_camera28_ring_d9_v1/ring-analysis.json'
COVERAGE = ROOT/'docs/results/donut_calibrated_d3_v1/coverage/camera34-coverage.npz'


def summary(values):
    result = world.augmented_summary(values)
    for name, width in [('major',.35),('minor',.08)]:
        result[name+'_signed_relative_bias'] = float(values[name].mean(dtype=float)/width-1)
        result.update({name+'_relative_abs_'+k: v
                       for k,v in replay.stats(values[name+'_abs'].astype(float)/width).items()})
    return result


def saved_residuals(arm, step):
    p = replay.ARMS[arm]/f'step-{step:05d}/surface-q-diagnostic.npz'
    z = replay.numeric(p)
    estimated, guard = d11.archived_log_square(z['widths'],z['transported_square'])
    if arm == 'D11':
        log = z['log_transported_square']
        assert log.shape==estimated.shape and (log>=0).all()
        guard['direct_vs_archive_log_sq_max_absolute'] = float(abs(log-estimated).max())
    else:
        log = estimated
    return z, dict(step=step, raw_Q_square=replay.stats(z['transported_square']),
                   log_Q_square=replay.stats(log), archive_roundoff=guard)


def verify_run(f, freeze_sha, auth_path):
    plan=f['tensor_training']
    batch=Path(f['host_tensor_output'])
    root=batch/'D11'
    auth=replay.read(auth_path)
    assert replay.digest(auth_path)==replay.digest(batch/'authorization.json')
    assert auth==d11.authorization(f,auth['commit'],freeze_sha)
    assert replay.digest(batch/'freeze.json')==freeze_sha
    assert (batch/'D11-exit-code.txt').read_text().strip()=='0'
    receipt=replay.read(root/'recovery.json')
    assert receipt['status']=='complete' and receipt['scientific_work_complete'] and receipt['cleanup_verified']
    assert receipt['optimizer_returns']==10000 and receipt['arm']=='D11'
    assert receipt['commit']==auth['commit'] and receipt['freeze_sha256']==freeze_sha and receipt['protocol']==d11.PROTOCOL
    assert receipt['benchmark_sha256']==plan['benchmark_generation_sha256']
    assert receipt['effective_shader']==f['shader'] and receipt['policy']==f['policy']
    assert receipt['runtime']['declared_image']==f['image_id'] and receipt['representation']==plan['representation']
    assert receipt['surface_q_policy']==plan['spatial_policy'] and receipt['spatial_regularizer']==plan['spatial_regularizer']
    assert receipt['coefficient_sha256']==plan['coefficient_sha256'] and receipt['diffuse_white_in_objective'] is True
    assert receipt['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior','surface_log_q_loss']
    assert receipt['D8_T_initial_state_verified'] and receipt['D8_T_initial_capture_verified']
    assert receipt['D6_domain_parity_verified'] and receipt['native_reference_gate']['passed']
    assert [e['step'] for e in receipt['saved_endpoints']]==STEPS
    assert (receipt['elapsed_seconds']<plan['limits']['inner_seconds'] and
            sum(v['bytes'] for v in receipt['outputs'].values())<plan['limits']['output_bytes'])
    assert all(v<=plan['limits'][k] for k,v in receipt['counters'].items())
    assert set(receipt['outputs'])=={p.relative_to(root).as_posix() for p in root.rglob('*')
                                    if p.is_file() and p.name not in ('recovery.json','progress.json')}
    assert sorted(p.name for p in root.glob('checkpoint-*.pth'))==[f'checkpoint-{s:05d}.pth' for s in STEPS[1:]]
    replay.verify_files(root,receipt['outputs'])
    controls=Path(plan['D8_T_initial_host_path'])
    d11.verify_initial_controls(controls,plan['D8_T_initial_controls'])
    assert receipt['initial']==replay.read(root/'initial.json')==replay.read(controls/'initial.json')
    archived=replay.numeric(controls/'paired-initial-gate.npz')
    current=replay.numeric(root/'paired-initial-gate.npz')
    assert archived.keys()==current.keys()
    for k in current:
        assert current[k].dtype == archived[k].dtype
        np.testing.assert_array_equal(current[k],archived[k])
    return root,receipt


def analyze(freeze_sha, auth_path, output):
    f=verify_freeze(ROOT/d11.FREEZE,freeze_sha)
    plan=f['tensor_training']
    root,receipt=verify_run(f,freeze_sha,auth_path)
    prior={a:Path(plan['controls'][a]['path']) for a in ('D6','D8-T')}
    reports={a:replay.read(p) for a,p in prior.items()}
    for a,p in prior.items():
        assert replay.digest(p)==plan['controls'][a]['sha256'] and reports[a]['status']=='complete'
    reports['D11']=receipt
    replay.ARMS.update({**{a:p.parent for a,p in prior.items()},'D11':root})
    assert not output.exists() and not any(p==output or p in output.parents for p in
        (root.parent, replay.REFERENCE, *(p.parent for p in prior.values())))
    ref=Path(plan['benchmark_host_path'])
    gen=ref/'generation.json'
    assert replay.digest(gen)==plan['benchmark_generation_sha256']
    replay.verify_files(ref,replay.read(gen)['outputs'])
    refs={c:replay.numeric(ref/f'view-{c}.npz') for c in CAMERAS}
    assert receipt['counters']==reports['D8-T']['counters']
    assert receipt['trainable_parameters']==reports['D8-T']['trainable_parameters']
    known=replay.numeric(replay.ARMS['D6']/'step-10000/illumination.npz')
    target=np.asarray(CALIBRATION['mean_linear_RGB'])
    rows, per_camera, illumination, surfaces, final, probes = [], [], {}, {}, {}, {}
    d8_lr=[v['lr'] for v in replay.read_log(replay.ARMS['D8-T'])]
    d6_domain=replay.numeric(replay.ARMS['D6']/'surface-q-domain.npz')
    assert {k:array_hash(v) for k,v in d6_domain.items()}==plan['D6_domain_array_sha256']
    for arm,p in replay.ARMS.items():
        if arm not in reports: continue
        r=reports[arm]
        names=['training.jsonl','surface-q-domain.npz']
        for s in STEPS:
            prefix=f'step-{s:05d}/'
            names += [prefix+n for n in ('metrics.json','surface-q-diagnostic.npz','illumination.npz')]
            names += [prefix+n for c in CAMERAS for n in (f'camera-{c}/materials.npz',f'camera-{c}-errors.npz')]
        replay.verify_files(p,{n:r['outputs'][n] for n in names})
        assert replay.digest(p/'surface-q-domain.npz')==replay.digest(replay.ARMS['D6']/'surface-q-domain.npz')
        history=replay.read_log(p)
        assert [v['lr'] for v in history]==d8_lr
        term='surface_log_q' if arm=='D11' else 'surface_q'
        for v in history:
            components=[np.float32(v[k]) for k in ('RGB_loss','material_prior','diffuse_white_prior',term+'_loss')]
            assert np.isfinite(components).all() and v[term+'_loss']>=0
            assert float(components[0]+components[1]+components[2]+components[3])==v['optimization_loss']
        probes[arm]=[dict(step=v['completed_updates'],**v['gradient_interactions'])
                     for v in history if 'gradient_interactions' in v]
        surfaces[arm], illumination[arm] = [], []
        for e in r['saved_endpoints']:
            s,m=e['step'],e['metrics']
            assert replay.read(p/f'step-{s:05d}/metrics.json')==m
            z,residual=saved_residuals(arm,s)
            np.testing.assert_array_equal(z['pair_ids'],d6_domain['diagnostic_ids'])
            assert array_hash(z['training_pair_counts'])==plan['D6_sampler_snapshots'][str(s)]
            assert not z['training_pair_counts'][z['pair_ids']].any()
            surfaces[arm].append(residual)
            illumination[arm].append(replay.replay_illumination(arm,s,m['illumination'],target,known))
            values={c:replay.replay_camera(arm,s,c,refs[c],m[str(c)]) for c in CAMERAS}
            per_camera.extend(dict(arm=arm,step=s,camera=c,**summary(v)) for c,v in values.items())
            for split,cams in [('all',CAMERAS),('fitting',CAMERAS[:4]),('held_out',CAMERAS[4:])]:
                merged={k:np.concatenate([values[c][k] for c in cams]) for k in values[cams[0]]}
                rows.append(dict(arm=arm,step=s,split=split,**summary(merged)))
            if arm=='D11':
                for c,v in values.items():
                    d8.direct_q_check(arm,s,c,refs[c],v,plan['representation']['export_world_Q_atol'])
                    if s==0:
                        initial, control = (replay.materials(k,s,c) for k in (arm,'D8-T'))
                        for name in ('Q_tensor_direct','linear_rgb','linear_diffuse','linear_specular','kd','metallic'):
                            np.testing.assert_array_equal(initial[name],control[name])
                if s==0:
                    light='step-00000/illumination.npz'
                    np.testing.assert_array_equal(replay.numeric(p/light)['linear_RGB'],
                                                  replay.numeric(replay.ARMS['D8-T']/light)['linear_RGB'])
            if s==10000: final[arm]=values
    cov=replay.numeric(COVERAGE)
    x,y=cov['pixel_xy'].T
    np.testing.assert_array_equal(128*y+x,refs[34]['pixel_indices'][refs[34]['shader_valid']])
    coverage=replay.read(ROOT/'docs/results/donut_paired_tensor_d8_v1/camera34-coverage.json')
    coverage['arms']['D11']=dict(extend_coverage(final['D11'][34],cov,coverage))
    coverage['arms']={a:coverage['arms'][a] for a in reports}
    ring_record=replay.read(RING)
    ring=ring_statistics(refs[28],ring_record,tuple(reports))
    view28=refs[28]
    region=np.isin(view28['pixel_indices'][view28['shader_valid']],ring_record['D8_T_positive_excess_decile']['pixels'])
    assert int(region.sum())==485
    for arm,v in final.items():
        for label,keep in [('D8_T_positive_excess_decile',region),('elsewhere',~region)]:
            ring[arm][label]['physical_and_RGB']=summary({k:a[keep] for k,a in v[28].items()})
        counts=cov['nearby_camera_counts'][:,1]
        coverage['arms'][arm]['physical_by_within_001_camera_count']={
            str(n):summary({k:a[counts==n] for k,a in v[34].items()}) if (counts==n).any() else None
            for n in range(5)}
    output.mkdir()
    replay.write_csv(output/'endpoint-metrics.csv',rows)
    replay.write_csv(output/'per-camera-metrics.csv',per_camera)
    replay.write_json(output/'analysis.json',dict(protocol=d11.PROTOCOL,status='verified_complete',
        freeze_sha256=freeze_sha,native_receipt=replay.record(root/'recovery.json'),coefficient=plan['spatial_policy']['coefficient'],
        resources={a:{k:r.get(k) for k in ('elapsed_seconds','resources','surface_q_timing','counters')}
                   for a,r in reports.items()}, surface_history=surfaces, illumination=illumination,
        gradient_probes=probes,camera28_fixed_ring=ring,camera34_coverage=coverage,
        frozen_sources=len(f['sources']),analysis_work=dict(new_native_models=0,new_renderer_calls=0,checkpoint_deserializations=0)))
    figures(output,rows,surfaces)
    return len(per_camera)


def figures(out, rows, surface):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for arm, label in [('D6','D6'),('D8-T','D8-T'),('D11','D11')]:
        r=[v for v in rows if v['arm']==arm and v['split']=='held_out']
        for ax,key,title in zip(axes.flat,('q_median','q_p95','RGB_relative_L2','minor_relative_abs_mean'),
                                ('Held-out Q median','Held-out Q p95','Held-out RGB relative L2','Held-out minor relative MAE')):
            ax.plot(STEPS,[v[key] for v in r],'.-',label=label)
            ax.set_title(title)
        for ax,k,title in zip(axes.flat[4:],('raw_Q_square','log_Q_square'),('Reserved raw-Q distance²','Reserved log-Q distance²')):
            ax.plot(STEPS,[v[k]['mean'] for v in surface[arm]],'.-',label=label)
            ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Adam returns')
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.savefig(out/'recovery-curves.png', dpi=145)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--authorization', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    print('Verified camera endpoints', analyze(a.freeze_sha256,a.authorization,a.output.resolve()))
