"""Explicit training admission and observation adapter for the heterogeneous panel."""
import os
from pathlib import Path

import numpy as np

from benchmarks import donut_heterogeneous_v1 as ref
from benchmarks import donut_surface_q_d6 as d6, donut_tensor_d8 as d8
from benchmarks.donut_recovery_v1 import IMAGE, POLICY, SHADER, TRAIN_LIMITS, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.verified_ray_backend import identity

PROTOCOL = 'donut-kurt-heterogeneous-four-arm-recovery-v1'
FREEZE = 'docs/protocols/DONUT_HETEROGENEOUS_TRAINING_V1.freeze.json'
ARMS = ('D2','D6',*d8.ARMS)
REFERENCE_FREEZE = '0bf3836aaa58491224df419c0bc5feaf921393137b196f2a252f4589e2badb38'
GENERATION = 'de8dda85559b286456d926e09731c6c501dfcbd6707d1aaf3aef9cf8784c58a0'
VERIFICATION = '22e7894fdba2f84ed0355299999cc74a1784cb40c74667f7b0b9e432cb598340'
ADMISSION = 'a6eab416054c5bf74b43834c3be224847e2d78d9af7eb4989745fe37333618fb'
FOREGROUND_SECONDS = 14760
STEPS = [0,2000,4000,6000,8000,10000]
FIT_KEYS = ('position','normals','outgoing','linear_components')
TARGET_KEYS = ('normals','axis','Q_world','widths_major_minor','color','metallic')
COVERAGE_RADIUS = .01
BOUNDARY_PAIR_RADIUS = .02
NORMAL_COSINE = float(np.cos(np.pi/6))


def accepted_limits(arm):
    if arm=='D2':
        return dict(TRAIN_LIMITS)
    if arm=='D6':
        return dict(TRAIN_LIMITS,**d6.LIMITS)
    require(arm in d8.ARMS, 'Invalid heterogeneous recovery arm')
    return d8.limits(TRAIN_LIMITS)


def planned_counters(arm):
    require(arm in ARMS, 'Invalid heterogeneous recovery arm')
    chunk_count=sum((n+511)//512 for n in ref.COUNTS)
    gate=(32+2*(arm in d8.ARMS))*24
    counts=dict(shader_calls=10000+32+2*(arm in d8.ARMS)+6*8*chunk_count,
                shadow_rays=(10000*512+gate+6*8*sum(ref.COUNTS))*1536,
                component_reverse_sweeps=24 if arm=='D2' else 30)
    if arm!='D2':
        counts.update({k:d6.LIMITS[k] for k in ('surface_q_batches','surface_q_pairs','surface_q_predictor_calls',
                                               'surface_q_predictor_points')})
    return counts


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha, 'Heterogeneous training freeze differs')
    f=ref.read(path)
    require(f['protocol']==PROTOCOL and f['image']==IMAGE and f['order']==list(ARMS)
            and f['steps']==STEPS and f['foreground_seconds']==FOREGROUND_SECONDS
            and f['policy']==POLICY and f['shader']==SHADER
            and f['splits']==ref.SPLITS and f['reference_freeze_sha256']==REFERENCE_FREEZE
            and f['dataset']['generation']['sha256']==GENERATION
            and f['dataset']['verification']['sha256']==VERIFICATION and f['dataset']['admission']['sha256']==ADMISSION,
            'Heterogeneous training definition differs')
    for arm in ARMS:
        recipe=f['arms'][arm]
        require(recipe['limits']==accepted_limits(arm) and recipe['work']==planned_counters(arm)
                and recipe['spatial_policy']==(None if arm=='D2' else d6.POLICY)
                and recipe['representation']==(d8.REPRESENTATION if arm in d8.ARMS else None),
                'Historical method definition changed: '+arm)
    for name,value in f['sources'].items():
        file=ROOT/name
        require(file.resolve()==file and ROOT in file.parents and digest(file)==value,'Frozen input source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, freeze_sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=freeze_sha,image_id=IMAGE,
                selected_gpu='0',host_output=f['host_output'],order=list(ARMS),maximum_execution_launches=1,
                benchmark_generation_sha256=GENERATION,verification_sha256=VERIFICATION,
                researcher_admitted_dataset=True,dataset_admission_sha256=ADMISSION,
                recipes=f['arms'],foreground_seconds=FOREGROUND_SECONDS)


def dataset(f, root, admission, verification):
    files={k:ref.check_artifact(p,f['dataset'][k]) for k,p in
           [('generation',root/'generation.json'),('admission',admission),('verification',verification)]}
    g,a,v=(ref.read(files[k]) for k in ('generation','admission','verification'))
    require(a['researcher_admitted_dataset'] is True and a['training_authorized'] is False
            and a['generation']==f['dataset']['generation'] and a['verification']==f['dataset']['verification']
            and a['package']==f['host_reference'] and a['dataset_splits']==ref.SPLITS
            and v['status']=='verified_pending_researcher_admission' and v['generation_sha256']==GENERATION
            and g['freeze_sha256']==v['freeze_sha256']==REFERENCE_FREEZE
            and g['scientific_work_complete'] and g['cleanup_verified'] and g['backend']==identity()
            and g['field']==ref.FIELD and g['environment']==ref.ENVIRONMENT
            and g['limits']['primary_rays']==0 and g['splits']==ref.SPLITS
            and g['outputs']==a['output_identities']==v['outputs'] and len(g['outputs'])==39,
            'Dataset admission, runtime or scientific package differs')
    for name,item in g['outputs'].items():
        path=root/name
        require(root in path.parents, 'Invalid dataset member')
        ref.check_artifact(path,item)
    return g,{c:numeric(root/f'view-{c}.npz') for c in ref.PANEL}


def runtime(args):
    f=freeze(args.freeze,args.freeze_sha256)
    require(args.phase in ARMS and args.image_id==IMAGE and digest(args.authorization)==args.authorization_sha256
            and ref.read(args.authorization)==authorization(f,args.commit,args.freeze_sha256),
            'Exact separate four-arm training authorization required')
    require(os.geteuid()!=0 and args.output==Path('/outputs')/args.phase
            and args.output.resolve()==args.output and not args.output.exists()
            and not os.statvfs('/outputs').f_flag&os.ST_RDONLY,
            'Fresh non-root experiment output required')
    for p in (Path('/'),ROOT,args.freeze,args.oracle,args.cameras,args.authorization,args.benchmark,
              args.admission,args.verification,args.initial,args.arm_initial):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only source, inputs and root filesystem required')
    require({line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Networkless execution required')
    ref.check_artifact(args.initial,f['base_initial'])
    recipe=f['arms'][args.phase]
    ref.check_artifact(args.arm_initial,recipe['initial'])
    require(args.initial_sha256==f['base_initial']['sha256'] and args.benchmark_sha256==GENERATION,
            'Wrong benchmark or base initialization selected')
    r=dict(recipe,initial_sha256=args.initial_sha256,benchmark_generation_sha256=GENERATION,protocol=PROTOCOL,
           D6_domain_array_sha256=f['D6_domain_array_sha256'],D6_sampler_snapshots=f['D6_sampler_snapshots'])
    active=dict(f,arm_training=r)
    if args.phase=='D6':
        active['surface_q_training']=r
    if args.phase in d8.ARMS:
        active['tensor_training']=dict(r,arms=list(d8.ARMS))
    return active,ref.read(args.authorization)


def fit_pool(data):
    """Only observed RGB and the geometry needed to render it reach optimization."""
    return {k:np.concatenate([data[c][k][data[c]['shader_valid']] for c in ref.SPLITS['fitting']])
            for k in FIT_KEYS}


def gap(widths):
    squared=np.asarray(widths)**2
    return (squared[:,0]-squared[:,1])/squared.sum(-1)


def point_errors(pred, truth):
    """The historical physical errors, scored against pointwise oracle values."""
    widths=np.sort(np.concatenate([pred['mx'],pred['my']],axis=1),axis=1)[:,::-1]
    w=truth['widths_major_minor']
    pred_gap=gap(widths)
    truth_gap=gap(w)
    threshold=ref.FIELD['orientation_gap']
    error=abs(widths-w)
    axis=np.sum(pred['axis']*truth['axis'],axis=1)
    arrays=dict(Q_relative=np.linalg.norm(pred['Q_world']-truth['Q_world'],axis=(1,2))
                            /np.linalg.norm(truth['Q_world'],axis=(1,2)),
                color_max=abs(pred['kd']-truth['color']).max(1),
                metallic=abs(pred['metallic']-truth['metallic']).reshape(-1),
                major_width=error[:,0],minor_width=error[:,1],
                major_width_relative=error[:,0]/w[:,0],minor_width_relative=error[:,1]/w[:,1],
                orientation_degrees=np.degrees(np.arccos(np.clip(abs(axis),0,1))),
                orientation_eligible=(pred_gap>=threshold)&(truth_gap>=threshold),
                orientation_prediction_eligible=pred_gap>=threshold,orientation_truth_eligible=truth_gap>=threshold,
                gap=pred_gap,gap_reference=truth_gap,
                tangent_error=abs(np.einsum('nij,nj->ni',pred['Q_world'],truth['normals'])).max(1))
    require(all(np.isfinite(x).all() for x in arrays.values()), 'Nonfinite heterogeneous physical metric')
    summary={}
    for k,a in arrays.items():
        if a.dtype==bool:
            continue
        if k=='orientation_degrees':
            a=a[arrays['orientation_eligible']]
        summary[k]=dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95))) if len(a) else None
    summary.update(points=len(w),Q_at_most_005=int((arrays['Q_relative']<=.05).sum()),
                   orientation_ineligible=int((~arrays['orientation_eligible']).sum()),
                   width_at_floor=int((widths[:,1]<=.0401).sum()), width_at_ceiling=int((widths[:,0]>=.9999).sum()))
    return arrays,summary
