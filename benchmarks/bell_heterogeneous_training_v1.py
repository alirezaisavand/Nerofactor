"""Admitted Bell observations and geometry for the existing four-arm trainer."""
import os
from pathlib import Path

import numpy as np

from benchmarks import bell_heterogeneous_v1 as b
from benchmarks import donut_surface_q_d6 as d6, donut_tensor_d8 as d8
from benchmarks.donut_heterogeneous_training_v1 import (ARMS, FIT_KEYS, FOREGROUND_SECONDS, STEPS, TARGET_KEYS,
    accepted_limits, point_errors)
from benchmarks.donut_recovery_v1 import IMAGE, POLICY, SHADER, numeric
from benchmarks.donut_surface_chart_d7 import array_hash
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.verified_ray_backend import identity


PROTOCOL = 'bell-kurt-heterogeneous-four-arm-recovery-v1'
FREEZE = 'docs/protocols/BELL_HETEROGENEOUS_TRAINING_V1.freeze.json'
PANEL, SPLITS = b.PANEL, b.SPLITS
REFERENCE_FREEZE = '56efcb40527e67ce8848b62a293da246d7571c27e5ebd3caef94d457468059e3'
GENERATION = 'c2d6c2996b8f19b99417d173724ef4eaf953818bf7008285d6b987165ada7e88'
VERIFICATION = 'e46e1e214bc352bd8db2f86a24cb6beaadf04d6b123817c02f695cfd9430454f'
ADMISSION = 'a1ec274ae406d1a47f07daa49aad0d73308f87e731e8e9ba69634f0fefb3c769'
SPATIAL_POLICY = dict(d6.POLICY, expected_mesh_faces=248908, expected_pairs=362520)
BOUNDARY_POLICY = dict(order='plane, original face index', crossing='one strict plane only',
    offset_fraction=.5, minimum_native_distance=1e-6, maximum_native_distance=.02, ratio_floor=1e-12,
    used_for_training=False)


def planned_counters(arm, probes):
    require(arm in ARMS and probes > 0, 'Invalid Bell arm or probe inventory')
    chunks = sum((n+511)//512 for n in b.COUNTS)
    gate_calls = 32+2*(arm in d8.ARMS)
    work = dict(shader_calls=10000+gate_calls+len(STEPS)*8*chunks,
        shadow_rays=(10000*512+gate_calls*24+len(STEPS)*8*b.POINTS)*1536,
        component_reverse_sweeps=24 if arm=='D2' else 30,
        boundary_predictor_calls=2*len(STEPS), boundary_predictor_points=2*len(STEPS)*probes)
    if arm!='D2':
        work.update({k:d6.LIMITS[k] for k in ('surface_q_batches','surface_q_pairs',
                                             'surface_q_predictor_calls','surface_q_predictor_points')})
    return work


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha, 'Bell training freeze differs')
    f = b.read(path)
    require(f['protocol']==PROTOCOL and f['image']==IMAGE and f['order']==list(ARMS)
            and f['steps']==STEPS and f['foreground_seconds']==FOREGROUND_SECONDS and f['panel']==PANEL
            and f['policy']==POLICY and f['shader']==SHADER and f['splits']==SPLITS
            and f['boundary_policy']==BOUNDARY_POLICY and f['reference_freeze_sha256']==REFERENCE_FREEZE
            and f['mesh']['sha256']==b.MESH
            and {k:f['dataset'][k]['sha256'] for k in ('generation','verification','admission')}
                ==dict(generation=GENERATION,verification=VERIFICATION,admission=ADMISSION),
            'Bell training definition differs')
    for arm in ARMS:
        recipe = f['arms'][arm]
        work = planned_counters(arm,f['boundary_probe_count'])
        require(recipe['limits']=={**accepted_limits(arm),**work} and recipe['work']==work
                and recipe['spatial_policy']==(None if arm=='D2' else SPATIAL_POLICY)
                and recipe['representation']==(d8.REPRESENTATION if arm in d8.ARMS else None),
                'Accepted method definition differs: '+arm)
    for name,value in f['sources'].items():
        file = ROOT/name
        require(file.resolve()==file and ROOT in file.parents and digest(file)==value,
                'Frozen Bell training source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,image_id=IMAGE,
        selected_gpu='0',host_output=f['host_output'],order=list(ARMS),maximum_execution_launches=1,
        benchmark_generation_sha256=GENERATION,verification_sha256=VERIFICATION,
        researcher_admitted_dataset=True,dataset_admission_sha256=ADMISSION,
        recipes=f['arms'],foreground_seconds=FOREGROUND_SECONDS)


def dataset(f, root, admission, verification):
    paths = dict(generation=root/'generation.json',admission=admission,verification=verification)
    g,a,v = (b.read(b.check_artifact(paths[k],f['dataset'][k])) for k in ('generation','admission','verification'))
    require(a['researcher_admitted_dataset'] is True and a['training_authorized'] is False
            and a['generation']==f['dataset']['generation'] and a['verification']==f['dataset']['verification']
            and a['package']==f['host_reference'] and a['dataset_splits']==SPLITS and a['scope']==list(ARMS)
            and v['status']=='verified_pending_researcher_admission' and v['generation_sha256']==GENERATION
            and g['freeze_sha256']==v['freeze_sha256']==REFERENCE_FREEZE and g['protocol']==b.PROTOCOL
            and g['scientific_work_complete'] and g['cleanup_verified'] and g['backend']==identity()
            and g['field']==b.FIELD and g['environment']==b.ENVIRONMENT and g['BRDF']==b.BRDF
            and g['indirect']=='zero' and g['point_light']=='absent' and g['splits']==SPLITS
            and g['outputs']==a['output_identities']==v['outputs'] and len(g['outputs'])==38,
            'Bell dataset admission, lighting or reference package differs')
    for name,item in g['outputs'].items():
        require(root in (root/name).parents, 'Invalid Bell dataset member')
        b.check_artifact(root/name,item)
    views = {c:numeric(root/f'view-{c}.npz') for c in PANEL}
    for c,count in zip(PANEL,b.COUNTS):
        view = views[c]
        require(view['shader_valid'].dtype==bool and len(view['shader_valid'])==count and view['shader_valid'].all()
                and np.array_equal(view['pixel_indices'],np.flatnonzero(view['hit_mask'])),
                'Admitted Bell view lost valid observations')
    return g,views


def geometry(f, mesh):
    vertices,triangles = b.mesh_arrays(b.check_artifact(mesh,f['mesh']))
    vertices = vertices.astype(np.float64)
    _,_,area = b.face_data(vertices,triangles)
    bounds = np.stack((vertices.min(0),vertices.max(0)))
    return dict(vertices_native=vertices,triangles=triangles,area=area,bounds=bounds),None,dict(mesh=f['mesh'])


def runtime(args, *, spec=None):
    read_freeze, approve, arms = (freeze,authorization,ARMS) if spec is None else (spec.freeze,spec.authorization,spec.ARMS)
    f = read_freeze(args.freeze,args.freeze_sha256)
    a = b.read(args.authorization)
    require(args.phase in arms and args.image_id==IMAGE and digest(args.authorization)==args.authorization_sha256
            and a==approve(f,args.commit,args.freeze_sha256), 'Exact separate Bell training authorization required')
    require(os.geteuid()!=0 and args.output==Path('/outputs')/args.phase and args.output.resolve()==args.output
            and not args.output.exists() and not os.statvfs('/outputs').f_flag&os.ST_RDONLY,
            'Fresh non-root Bell arm output required')
    for p in (Path('/'),ROOT,args.freeze,args.oracle,args.authorization,args.benchmark,args.admission,
              args.verification,args.initial,args.arm_initial):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only Bell source, inputs and root filesystem required')
    require({line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Networkless execution required')
    b.check_artifact(args.initial,f['base_initial'])
    recipe = f['arms'][args.phase]
    b.check_artifact(args.arm_initial,recipe['initial'])
    require(args.initial_sha256==f['base_initial']['sha256'] and args.benchmark_sha256==GENERATION,
            'Wrong Bell dataset or base initialization')
    r = dict(recipe,initial_sha256=args.initial_sha256,benchmark_generation_sha256=GENERATION,
        protocol=PROTOCOL if spec is None else spec.PROTOCOL,
        D6_domain_array_sha256=f['integer_domain_hashes'],D6_sampler_snapshots=f['sampler_snapshots'])
    active = dict(f,arm_training=r)
    if args.phase=='D6' or 'distance_weighting' in recipe:
        active['surface_q_training'] = r
    if args.phase in d8.ARMS:
        active['tensor_training'] = dict(r,arms=list(d8.ARMS))
    return active,a


def fit_pool(data):
    return {k:np.concatenate([data[c][k][data[c]['shader_valid']] for c in SPLITS['fitting']]) for k in FIT_KEYS}


def boundary_probes(geometry):
    """Pair points within faces that are transversely cut by one region plane."""
    vertices,triangles,bounds = (geometry[k] for k in ('vertices_native','triangles','bounds'))
    faces,raw,_ = b.face_data(vertices,triangles)
    u = 2*(vertices-bounds.mean(0))/(bounds[1]-bounds[0])
    signed = (u@np.asarray(b.FIELD['plane_normals']).T+b.FIELD['plane_offsets'])[triangles]
    captured = []
    for plane in (0,1):
        values,other = signed[:,:,plane],signed[:,:,1-plane]
        cut = (values.min(1)<0)&(values.max(1)>0)&(values!=0).all(1)
        one_plane = (other>0).all(1)|(other<0).all(1)
        ids = np.flatnonzero(cut&one_plane)
        v,d = faces[ids],values[ids]
        next_d = np.roll(d,-1,axis=1)
        crosses = (d<0)!=(next_d<0)
        require((crosses.sum(1)==2).all(),'Bell boundary intersections are not segments')
        t = np.divide(d,d-next_d,out=np.zeros_like(d),where=crosses)
        edge = v+(np.roll(v,-1,axis=1)-v)*t[...,None]
        center = edge[crosses].reshape(-1,2,3).mean(1)
        index = np.arange(len(ids))
        low,high = (center+(v[index,i]-center)*BOUNDARY_POLICY['offset_fraction']
                    for i in (d.argmin(1),d.argmax(1)))
        region0_high = other[ids,0]>0
        p0 = np.where(region0_high[:,None],high,low)
        p1 = np.where(region0_high[:,None],low,high)
        distance = np.linalg.norm(p0-p1,axis=1)
        keep = (distance>=BOUNDARY_POLICY['minimum_native_distance'])&(distance<=BOUNDARY_POLICY['maximum_native_distance'])
        face = ids[keep]
        captured.append(dict(face_ids=face,plane_ids=np.full(len(face),plane,dtype=np.int64),
                             positions0=p0[keep],positions1=p1[keep],normals=-raw[face],distance=distance[keep]))
    out = {k:np.concatenate([row[k] for row in captured]) for k in captured[0]}
    points = np.stack((out.pop('positions0'),out.pop('positions1')))
    require(len(out['face_ids'])>0, 'Bell needs mesh-surface boundary probes')
    targets = [b.material_field(p,out['normals'],bounds) for p in points]
    require(all((mat['material_region']==i).all() for i,mat in enumerate(targets)),
            'Bell boundary probes do not straddle their declared regions')
    out.update(positions=points,Q_world=np.stack([mat['Q_world'] for mat in targets]),
               widths_major_minor=np.stack([mat['widths_major_minor'] for mat in targets]))
    return out


def diagnostics(geometry, f):
    domain,_ = b.neighborhoods(geometry)
    probes = boundary_probes(geometry)
    require(len(probes['face_ids'])==f['boundary_probe_count']
            and all(array_hash(probes[k])==v for k,v in f['integer_probe_hashes'].items()),
            'Frozen Bell boundary probe identities differ')
    return domain,domain['diagnostic_ids'],probes


def diagnostic_frames(geometry, domain, indices):
    frames = []
    for suffix in ('0','1'):
        normals = domain['normals'+suffix][indices]
        mat = b.material_field(domain['points'+suffix][indices],normals,geometry['bounds'])
        frames.append(dict(normals=normals,Q_world=mat['Q_world']))
    return frames


def boundary_score(probes, predicted):
    q,w = (probes[k] for k in ('Q_world','widths_major_minor'))
    qp,wp = (predicted[k] for k in ('Q_world','widths_major_minor'))
    require(q.shape==qp.shape and w.shape==wp.shape and np.isfinite(qp).all()
            and np.isfinite(wp).all(), 'Invalid Bell boundary predictions')
    q,qp = q[0]-q[1],qp[0]-qp[1]
    w,wp = w[0]-w[1],wp[0]-wp[1]
    magnitude = np.linalg.norm(q,axis=(1,2))
    valid = magnitude>BOUNDARY_POLICY['ratio_floor']
    stats = dict(pairs=len(w),ratio_eligible=int(valid.sum()),near_zero_oracle_Q_jump=int((~valid).sum()),
                 signed_width_jump_MAE=abs(w-wp).mean(0).tolist(),
                 reference_signed_width_jump=w.mean(0).tolist(),predicted_signed_width_jump=wp.mean(0).tolist(),
                 absolute_Q_jump_error_mean=float(np.linalg.norm(qp-q,axis=(1,2)).mean()))
    stats['Q_jump_projection_median'] = (float(np.median((qp[valid]*q[valid]).sum((1,2))/magnitude[valid]**2))
                                         if valid.any() else None)
    return stats


def check_probe_archive(probes, captured):
    require(probes.keys()==captured.keys(), 'Bell boundary archive fields differ')
    for name,value in probes.items():
        require(value.dtype==captured[name].dtype, 'Bell boundary archive dtype differs: '+name)
        if value.dtype.kind=='f':
            np.testing.assert_allclose(captured[name],value,rtol=0,atol=2e-12)
        else:
            np.testing.assert_array_equal(captured[name],value)


def check_domain_archive(f, domain, views, receipt_hashes, captured):
    expected = dict(domain,fitting_camera_count=d6.fitting_face_counts(views,SPLITS['fitting'],len(domain['face_area'])))
    check_probe_archive(expected,captured)
    hashes = {k:array_hash(v) for k,v in captured.items()}
    require(receipt_hashes==hashes and all(hashes[k]==v for k,v in f['integer_domain_hashes'].items()),
            'Bell neighborhood archive hash differs')


def check_sampler_capture(f, domain, step, capture):
    np.testing.assert_array_equal(capture['pair_ids'],domain['diagnostic_ids'])
    counts = capture['training_pair_counts']
    require(array_hash(counts)==f['sampler_snapshots'][str(step)]
            and int(counts.sum())==step*SPATIAL_POLICY['batch_pairs']
            and not counts[domain['diagnostic_ids']].any(), 'Bell surface sampling receipt differs')
