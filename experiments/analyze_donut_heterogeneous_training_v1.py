"""File-only verification and physical scoring of the four trained arms."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_heterogeneous_v1 as ref
from benchmarks import donut_heterogeneous_training_v1 as h
from benchmarks import donut_heterogeneous_metrics_v1 as m
from benchmarks.donut_recovery_v1 import inputs, json_new, numeric
from benchmarks.neural_observation_e16 import compatibility
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.verified_ray_backend import identity


def arm_receipt(batch, arm, f, auth, views):
    root=batch/arm
    path=root/'recovery.json'
    r=ref.read(path)
    recipe=f['arms'][arm]
    require((batch/f'{arm}-exit-code.txt').read_text().strip()=='0'
            and r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified']
            and r['arm']==arm and r['commit']==auth['commit'] and r['freeze_sha256']==auth['freeze_sha256']
            and r['authorization_sha256']==digest(batch/'authorization.json') and r['protocol']==h.PROTOCOL
            and r['benchmark_sha256']==h.GENERATION and r['dataset_admission_sha256']==h.ADMISSION
            and r['counters']==recipe['work'] and r['historical_arm_initial_parity_verified']
            and r['optimizer_returns']==10000 and [e['step'] for e in r['saved_endpoints']]==h.STEPS
            and r['backend']==identity() and r['effective_shader']==h.SHADER and r['policy']==h.POLICY
            and r['elapsed_seconds']<recipe['limits']['inner_seconds']
            and r['resources']['host_peak_RSS']<recipe['limits']['host_memory_gib']*1024**3
            and sum(item['bytes'] for item in r['outputs'].values())<recipe['limits']['output_bytes'],
            'Incomplete or foreign heterogeneous arm receipt: '+arm)
    require(r['initial']==ref.read(root/'initial.json')==ref.read(Path(recipe['initial']['path'])),
            'Native initial state differs from accepted arm')
    if arm!='D2':
        require(r['D6_domain_parity_verified'] and r['surface_q_policy']==recipe['spatial_policy']
                and r['diffuse_white_in_objective'] is True, 'Surface parity differs')
    if arm in h.d8.ARMS:
        require(r['paired_initial_gate']['passed'] and r['representation']==recipe['representation'],
                'D8 paired initialization gate did not pass')
    require(set(r['outputs'])=={p.relative_to(root).as_posix() for p in root.rglob('*')
            if p.is_file() and p.name not in ('recovery.json','progress.json')},
            'Arm output set differs')
    for name,item in r['outputs'].items():
        p=root/name
        require(root in p.parents,'Arm output escaped path')
        ref.check_artifact(p,item)
    require(sorted(p.name for p in root.glob('checkpoint-*.pth'))==[f'checkpoint-{s:05d}.pth' for s in h.STEPS[1:]],
            'Checkpoint schedule differs')
    gate=numeric(root/'native-reference-gate.npz')
    view=views[0]
    valid=np.flatnonzero(view['shader_valid'])
    ix=valid[np.linspace(0,len(valid)-1,24).astype(int)]
    for name,key in [('reference','linear_components'),('coarse','coarse_components')]:
        np.testing.assert_array_equal(gate[name],view[key][ix,2])
    np.testing.assert_array_equal(gate['pixel_indices'],view['pixel_indices'][ix])
    replay=compatibility(gate['native_draws'],gate['reference'],abs(gate['reference']-gate['coarse']),
                         h.POLICY['compatibility_absolute'],h.POLICY['compatibility_relative'])
    require(replay['passed']==r['native_reference_gate']['passed'] is True, 'Known-light compatibility replay differs')
    history=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
    require([v['completed_updates'] for v in history]==list(range(1,10001))
            and history[0]['parameter_displacement_L1']==0<history[1]['parameter_displacement_L1']
            and all(np.isfinite([v[k] for k in ('lr','RGB_loss','material_prior','diffuse_white_prior',
                                                'preclip_gradient_l2')]).all() for v in history),
            'Incomplete/nonfinite optimization history')
    return r,history


def read_endpoint(root, step, camera, view):
    base=root/f'step-{step:05d}'
    a=numeric(base/f'camera-{camera}/materials.npz')
    saved=numeric(base/f'camera-{camera}-errors.npz')
    i=view['shader_valid']
    mask=a['hit_mask']
    np.testing.assert_array_equal(np.flatnonzero(mask),view['pixel_indices'][i])
    np.testing.assert_array_equal(saved['geometry_hit_mask'],view['hit_mask'])
    truth=view['linear_components'][i,2]
    np.testing.assert_array_equal(a['linear_gt'][mask],truth.astype(np.float32))
    np.testing.assert_array_equal(a['position'][mask],view['position'][i].astype(np.float32))
    np.testing.assert_array_equal(a['normal'][mask],view['normals'][i].astype(np.float32))
    mx,my,t,b=a['mx'][mask],a['my'][mask],2*a['tangents'][mask]-1,2*a['bitangents'][mask]-1
    q=mx[:,:,None]**2*t[:,:,None]*t[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]
    np.testing.assert_array_equal(a['Q_world'][mask],q)
    if 'Q_tensor_direct' in a:
        np.testing.assert_allclose(a['Q_tensor_direct'][mask],q,atol=h.d8.REPRESENTATION['export_world_Q_atol'],rtol=0)
    errors,_=h.point_errors(dict(mx=mx,my=my,kd=a['kd'][mask],metallic=a['metallic'][mask],
                         Q_world=q,axis=np.where(mx>=my,t,b)),{k:view[k][i] for k in h.TARGET_KEYS})
    for k,v in errors.items():
        np.testing.assert_allclose(saved[k],v,atol=2e-12,rtol=0)
    width=np.sort(np.concatenate((mx,my),1),axis=1)[:,::-1]
    errors.update(major_predict=width[:,0],minor_predict=width[:,1])
    return errors,a['linear_rgb'][mask].astype(float),truth,q,width


def analyze(sha, output):
    f=h.freeze(h.ROOT/h.FREEZE,sha)
    batch=Path(f['host_output'])
    p=batch/'authorization.json'
    auth=ref.read(p)
    require(auth==h.authorization(f,auth['commit'],sha) and digest(batch/'freeze.json')==sha
            and digest(p)==digest(batch.with_name(batch.name+'-authorization.json')),
            'Exact heterogeneous launch evidence required')
    require(output.resolve()==output and not output.exists() and output.parent.is_dir()
            and batch not in output.parents and Path(f['host_reference']) not in output.parents,
            'Fresh analysis-only output required')
    dataset=f['dataset']
    _,views=h.dataset(f,Path(f['host_reference']),Path(dataset['admission']['path']),Path(dataset['verification']['path']))
    geometry,_,_=inputs(Path(f['host_inputs']['oracle']),Path(f['host_inputs']['cameras']))
    _,area=ref.geometry_fields(geometry)
    adj=m.mesh_adjacency(geometry['triangles'])
    pairs={c:m.boundary_pairs(views[c],adj) for c in ref.PANEL}
    support={c:m.fitting_support(views,c) for c in ref.PANEL}
    reports, rows, previous_lr, paired = {}, [], None, None
    for arm in h.ARMS:
        root=batch/arm
        r,hist=arm_receipt(batch,arm,f,auth,views)
        lr=[v['lr'] for v in hist]
        require(previous_lr is None or previous_lr==lr,'Learning-rate streams differ')
        previous_lr=lr
        if arm in h.d8.ARMS:
            capture=numeric(root/'paired-initial-gate.npz')
            if paired is None:
                paired=capture
            else:
                require(capture.keys()==paired.keys(),'D8 paired evidence fields differ')
                for name,a in paired.items():
                    np.testing.assert_array_equal(capture[name],a)
        reports[arm]=dict(receipt_sha256=digest(root/'recovery.json'),counters=r['counters'],elapsed_seconds=r['elapsed_seconds'],
            resources=r['resources'],surface_q_timing=r.get('surface_q_timing'),
            gradient_probes=[dict(completed_updates=x['completed_updates'],**x['gradient_interactions'])
                             for x in hist if 'gradient_interactions' in x])
        for step in h.STEPS:
            by_camera={c:read_endpoint(root,step,c,views[c]) for c in ref.PANEL}
            for split, cameras in ref.SPLITS.items():
                valid=lambda key: np.concatenate([views[c][key][views[c]['shader_valid']] for c in cameras])
                e={k:np.concatenate([by_camera[c][0][k] for c in cameras]) for k in by_camera[cameras[0]][0]}
                rgb,gt=(np.concatenate([by_camera[c][slot] for c in cameras]) for slot in (1,2))
                weights,coverage=ref.seen_area_weights(valid('triangle'),area)
                boundary=valid('near_boundary')
                weak=valid('nearly_isotropic')
                geom,light=(np.concatenate([support[c][slot] for c in cameras]) for slot in (0,1))
                region=valid('material_region')
                cohorts=dict(all=np.ones(len(weights),dtype=bool),boundary=boundary,smooth_interior=~boundary&~weak,
                             nearly_isotropic=weak,boundary_region_0=boundary&(region==0),
                             boundary_region_1=boundary&(region==1))
                for n in range(5):
                    cohorts[f'fitting_cameras_{n}']=geom==n
                    cohorts[f'highlight_cameras_{n}']=light==n
                result={}
                for name,i in cohorts.items():
                    values={k:v[i] for k,v in e.items()}
                    result[name]=dict(observation_weighted=m.score(values,rgb[i],gt[i],np.ones(int(i.sum()))),
                                      observed_triangle_area_weighted=m.score(values,rgb[i],gt[i],weights[i]))
                rows.append(dict(arm=arm,step=step,split=split,coverage=coverage,metrics=result,
                    boundary_by_camera={str(c):m.boundary_score(views[c],pairs[c],*by_camera[c][3:5]) for c in cameras}))
        light=np.sin((np.arange(128)+.5)*np.pi/128)[:,None,None]
        for step,e in zip(h.STEPS,r['saved_endpoints']):
            a=numeric(root/f'step-{step:05d}/illumination.npz')
            value,known=(a[k].astype(float) for k in ('linear_RGB','known_linear_RGB'))
            rgb=np.sqrt(((value-known)**2*light).sum()/(known**2*light).sum())
            mean=(value*light).sum((0,1))/(light.sum()*256)
            row=dict(step=step,RGB_relative_L2=float(rgb),spherical_mean_RGB=mean.tolist())
            np.testing.assert_allclose(row['RGB_relative_L2'],e['metrics']['illumination']['relative_L2'],atol=1e-6,rtol=0)
            reports[arm].setdefault('illumination_history',[]).append(row)
    output.mkdir()
    json_new(output/'analysis.json',dict(schema_version=1,protocol=h.PROTOCOL,status='verified_complete',
        freeze_sha256=sha,execution_commit=auth['commit'],authorization_sha256=digest(p),arms=reports,results=rows,
        boundary_pairs_per_camera={str(c):len(v) for c,v in pairs.items()},offline_native_calls=0,
        offline_checkpoint_loads=0,offline_optimizer_updates=0))
    with (output/'summary.csv').open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(('arm','step','split','RGB_relative_L2','Q_median','Q_p95','major_width_MAE',
                         'minor_width_MAE','orientation_p95','boundary_Q_median','near_isotropic_Q_median'))
        for r in rows:
            metrics=r['metrics']
            a=metrics['all']['observation_weighted']
            writer.writerow((r['arm'],r['step'],r['split'],a['RGB_relative_L2'],a['Q_relative']['median'],
                a['Q_relative']['p95'],a['major_width']['mean'],a['minor_width']['mean'],
                a['orientation_degrees']['p95'] if a['orientation_degrees'] else None,
                metrics['boundary']['observation_weighted']['Q_relative']['median'],
                metrics['nearly_isotropic']['observation_weighted']['Q_relative']['median']))
    return len(rows)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    print(f'Verified and scored {analyze(args.freeze_sha256,args.output)} arm/checkpoint/split combinations')
