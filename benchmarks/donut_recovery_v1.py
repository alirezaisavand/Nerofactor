"""Fixed small donut acquisition and physical diagnostics; metadata/NumPy only."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.verified_ray_backend import CORRECTED_IMAGE as IMAGE

REFERENCE = 'donut-kurt-reference-v1'
TRAINING = 'donut-kurt-recovery-v1'
SIZE = 128
GRIDS = (256, 512)
COLOR = [0.10804546, 1.0, 0.03044616]
METALLIC = .8
WIDTHS = [.35, .08]
PANEL = [0, 50, 1, 45, 28, 34]
SPLITS = dict(fitting=PANEL[:4], held_out=PANEL[4:])
# Reuse E.11D's explicitly specified L2, without selecting light by new results.
from experiments.bell_highlight_e11d import candidates
ENVIRONMENT = candidates()['L2_six_cardinal']
INPUTS = {
    'oracle-geometry-and-frames.npz': 'cfe903c177dfe404c47390aed2a3fa025ea9499ca0af644dba667b721c0139f1',
    'native-chart-orientation-reference.npz': 'cc498faabe0b8b3e2ebf63a87a51d48316396b90e8650098cf19a559f442834c',
    'transforms_train.json': '89fc7b8d05db69437f3654cac1a8c6e68a23cfb411c28703824e18f0fdcaa82a',
}
REFERENCE_LIMITS = dict(wall_seconds=7200, inner_seconds=7080, grace_seconds=30,
    cpu_threads=4, host_memory_gib=64, cuda_allocator_fraction=.25, output_bytes=512*1024**2,
    primary_rays=len(PANEL)*SIZE**2,
    shadow_directions=len(PANEL)*SIZE**2*sum(g*g for g in GRIDS),
    BRDF_point_directions=len(PANEL)*SIZE**2*sum(g*g for g in GRIDS),
    CPU_primary_crosschecks=48, point_batch=4, direction_chunk=16384)
TRAIN_LIMITS = dict(wall_seconds=7500, arm_seconds=3600, inner_seconds=3540, grace_seconds=30,
    cpu_threads=4, host_memory_gib=64, cuda_allocator_fraction=.25, output_bytes=8*1024**3,
    updates_per_arm=10000, batch_rays=512, interval=2000, evaluation_draws=8,
    component_reverse_sweeps=24, smoothness_field_calls=12, illumination_map_calls=6,
    # Per arm: training + six full-panel checkpoints (including initialization),
    # and32 independent-tape compatibility draws at24 geometric probes.
    shader_calls=10000+6*len(PANEL)*32*8+32,
    shadow_rays=(10000*512+6*len(PANEL)*SIZE**2*8+32*24)*1536)
SHADER = dict(diffuse_sample_num=512, specular_sample_num=1024, human_lights=False,
    outer_light_version='direction', light_exp_max=20, inner_light_exp_max=20,
    anisotropy=True, random_azimuth=True, is_real=False,
    regularization_version='legacy_v1', reg_change=True, change_eps=.02,
    change_type='gaussian', reg_lambda1=.05, reg_energy_loss=True,
    reg_energy_loss_lambda=.05, reg_spec_loss=True, reg_spec_loss_lambda=.05,
    illumination_mode='known_env_v1', indirect_mode='zero_bounce_v1',
    known_environment=ENVIRONMENT)
POLICY = dict(seed=6033, optimizer='original A Adam', lr_initial=.001, eps=.0001,
    betas=[.9,.999], amsgrad=True, weight_decay=0, gradient_clip=.7,
    scheduler='original WarmUpCosLR', end_warm=1000, end_iter=100000,
    scheduled_lr=.0005, RGB_loss='original Charbonnier on linear RGB, epsilon=.001',
    diffuse_white_prior=.01, material_prior='unchanged legacy_v1',
    compatibility_absolute=.001, compatibility_relative=.02, compatibility_SE_multiplier=6,
    orientation_eligibility_gap=.1)


def json_new(path, value):
    # Validate the complete payload before creating an exclusive output file.
    # A reporting type/NaN error must not leave a truncated terminal record.
    text=json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(text)


def numeric(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    require(result and all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in result.values()),
            'Finite primitive numerical evidence required: '+str(path))
    return result


def unit(a):
    length = np.linalg.norm(a, axis=-1, keepdims=True)
    require(np.isfinite(a).all() and (length > 1e-12).all(), 'Undefined/nonfinite unit vector')
    return a/length


def select_panel(meta):
    poses = np.asarray([f['transform_matrix'] for f in meta['frames']], dtype=np.float64)
    require(poses.shape == (100,4,4), 'Original100 cameras required')
    directions = unit(poses[:,:3,3]); selected = [0]
    while len(selected) < 6:
        score = (1-directions@directions[selected].T).min(1)
        score[selected] = -1
        selected.append(int(score.argmax()))  # Fixed lowest-index tie break.
    return selected


def camera_rays(meta, camera, size=SIZE):
    require(camera in PANEL and type(size) is int and size > 0, 'Explicit panel camera/resolution required')
    pose = np.asarray(meta['frames'][camera]['transform_matrix'], dtype=np.float64)
    R = pose[:3,:3]
    require(np.max(abs(R.T@R-np.eye(3))) < 1e-6 and abs(np.linalg.det(R)-1) < 1e-6,
            'Invalid original camera rotation')
    focal = .5*size/np.tan(float(meta['camera_angle_x'])/2)
    y,x = np.mgrid[:size,:size]
    local = np.stack([(x-size/2)/focal, -(y-size/2)/focal, -np.ones_like(x)], -1)
    # Original A NeRF integer-index rays, no hidden +.5 offset. Normalize only
    # at the explicit oracle/shader interface, retaining poses and world axes.
    directions = unit(local.reshape(-1,3)@R.T)
    origins = np.broadcast_to(pose[:3,3]/2, directions.shape).copy()
    return origins, directions, dict(focal_px=focal, principal_point=[size/2]*2,
        pixel_offset=0, c2w=pose.tolist(), world_to_native_scale=.5)


def inputs(oracle, cameras):
    paths = {name: oracle/name for name in INPUTS if name.endswith('.npz')}
    paths['transforms_train.json'] = cameras
    records = {}
    for name,path in paths.items():
        require(path.is_absolute() and path.resolve() == path and path.is_file() and
                digest(path) == INPUTS[name], 'Admitted input changed: '+name)
        records[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=INPUTS[name])
    meta = json.loads(cameras.read_text())
    require(select_panel(meta) == PANEL, 'Geometric camera selection changed')
    return numeric(paths['oracle-geometry-and-frames.npz']), meta, records


def surface_frames(geometry, points):
    """Locate actual BVH points on immutable triangles; interpolate loop frames.

    A fixed centroid-neighbor candidate search must contain a matching face or
    fails. Tiny barycentric extrapolation reflects float32 BVH positions; no
    point projection, normal flip or mesh repair is performed.
    """
    from scipy.spatial import cKDTree
    v = geometry['vertices_native']; faces = geometry['triangles']
    tri = v[faces]; loops = geometry['triangle_loops']
    _, candidates = cKDTree(tri.mean(1)).query(points, k=min(32,len(tri)))
    candidates = np.asarray(candidates).reshape(len(points),-1)
    t = tri[candidates]; a = t[:,:,0]; e = t[:,:,1]-a; f = t[:,:,2]-a
    q = points[:,None]-a
    ee = (e*e).sum(-1); ff = (f*f).sum(-1); ef = (e*f).sum(-1)
    determinant = ee*ff-ef*ef
    require((determinant > 1e-20).all(), 'Degenerate oracle triangle candidate')
    u = ((q*e).sum(-1)*ff-(q*f).sum(-1)*ef)/determinant
    w = ((q*f).sum(-1)*ee-(q*e).sum(-1)*ef)/determinant
    weights = np.stack([1-u-w,u,w],-1)
    residual = np.linalg.norm(a+u[:,:,None]*e+w[:,:,None]*f-points[:,None],axis=-1)
    valid = (weights.min(-1)>=-2e-4)&(weights.max(-1)<=1+2e-4)&(residual<=2e-6)
    require(valid.any(1).all(), 'BVH point not attributable to an oracle triangle')
    # Shared-edge ties use original PLY/API triangle index, not BRDF responses.
    order = np.where(valid,candidates,np.iinfo(np.int32).max).argmin(1)
    face = candidates[np.arange(len(points)),order]
    bary = weights[np.arange(len(points)),order]
    li = loops[face]
    n = unit((geometry['shading_normals_world'][li]*bary[:,:,None]).sum(1))
    axes = geometry['rotated_Principled_tangent_world'][li]
    axes = unit(axes-(axes*n[:,None]).sum(-1,keepdims=True)*n[:,None])
    projector = np.einsum('ni,nij,nik->njk',bary,axes,axes)
    eigenvalues,eigenvectors = np.linalg.eigh(projector)
    require((eigenvalues[:,-1]-eigenvalues[:,-2] > .5).all(), 'Ambiguous interpolated principal director')
    axis = unit(eigenvectors[:,:,-1]); other = unit(np.cross(n,axis))
    face_normal = unit(np.cross(tri[face,1]-tri[face,0],tri[face,2]-tri[face,0]))
    require(((face_normal*n).sum(-1)>0).all(), 'Oracle shading/face normal hemisphere mismatch')
    Q = WIDTHS[0]**2*axis[:,:,None]*axis[:,None,:]+WIDTHS[1]**2*other[:,:,None]*other[:,None,:]
    return dict(normals=n, axis=axis, Q_world=Q, triangle=face, barycentric=bary,
        face_normal=face_normal, surface_residual=residual[np.arange(len(points)),order])


def physical_errors(pred, truth):
    """All points retained; orientation is modulo axis sign and width swapping."""
    n=truth['normals']; Q=pred['Q_world']; axis=pred['axis']
    require(np.isfinite(Q).all() and np.isfinite(axis).all(), 'Nonfinite physical prediction')
    widths=np.sort(np.concatenate([pred['mx'],pred['my']],-1),axis=-1)[:,::-1]
    gap=(widths[:,0]**2-widths[:,1]**2)/np.maximum((widths**2).sum(-1),1e-15)
    # Truth is anisotropic at every point. Near-isotropic predicted axes remain
    # reported as ineligible, never counted as orientation successes.
    angle=np.degrees(np.arccos(np.clip(abs((axis*truth['axis']).sum(-1)),0,1)))
    errors=dict(Q_relative=np.linalg.norm(Q-truth['Q_world'],axis=(-1,-2))/np.linalg.norm(truth['Q_world'],axis=(-1,-2)),
        color_max=abs(pred['kd']-np.asarray(COLOR)).max(-1),
        metallic=abs(pred['metallic'][:,0]-METALLIC),
        major_width=abs(widths[:,0]-WIDTHS[0]),minor_width=abs(widths[:,1]-WIDTHS[1]),
        orientation_degrees=angle,orientation_eligible=gap>=POLICY['orientation_eligibility_gap'],
        tangent_error=abs(np.einsum('nij,nj->ni',Q,n)).max(-1))
    summary={k:dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))
             for k,a in errors.items() if a.dtype!=bool}
    summary.update(points=len(n),Q_at_most_005=int((errors['Q_relative']<=.05).sum()),
        orientation_ineligible=int((~errors['orientation_eligible']).sum()),
        albedo_below_half_truth=int((pred['kd'][:,1]<COLOR[1]/2).sum()),
        width_at_floor=int((np.minimum(pred['mx'],pred['my'])[:,0]<=.0401).sum()),
        width_at_ceiling=int((np.maximum(pred['mx'],pred['my'])[:,0]>=.9999).sum()))
    return errors,summary


def verify_freeze(path, expected):
    require(digest(path)==expected, 'Freeze hash mismatch')
    f=json.loads(path.read_text())
    reference_limits=REFERENCE_LIMITS
    if f.get('reference_protocol')=='donut-kurt-reference-v2':
        from benchmarks.donut_reference_v2 import LIMITS as reference_limits,ORDERS,ORDINARY_COSINE_MIN,ORDINARY_COUNT
        require(f['reflectance_orders']==list(ORDERS) and f['ordinary_cosine_min']==ORDINARY_COSINE_MIN and
                f['ordinary_count']==ORDINARY_COUNT,'Frozen slope validation definition differs')
    else:
        require(f.get('reference_protocol',REFERENCE)==REFERENCE,'Unknown reference definition')
    require(f['reference_limits']==reference_limits and f['training_limits']==TRAIN_LIMITS and
            f['policy']==POLICY and f['shader']==SHADER and f['panel']==PANEL and f['splits']==SPLITS,
            'Frozen scientific definition differs')
    if 'calibrated_training' in f:
        from benchmarks.donut_calibration_d3 import CALIBRATION, NEW_LIMITS, PROTOCOL
        d3=f['calibrated_training']
        require(d3['protocol']==PROTOCOL and d3['calibration']==CALIBRATION and
                d3['limits']==dict(TRAIN_LIMITS, **NEW_LIMITS),
                'Frozen D3 calibration or work limits differ')
    for name,sha in f['sources'].items():
        p=ROOT/name;require(p.resolve()==p and ROOT in p.parents and digest(p)==sha,'Frozen source changed: '+name)
    baseline('material',PINS['material'])
    return f


def runtime(args, phase):
    import os
    f=verify_freeze(args.freeze,args.freeze_sha256)
    a=json.loads(args.authorization.read_text());limits=f['reference_limits'] if phase=='reference' else TRAIN_LIMITS
    reference_protocol=f.get('reference_protocol',REFERENCE)
    d3=f.get('calibrated_training')
    require((phase=='D3')==(d3 is not None), 'D3 requires its dedicated execution freeze')
    protocol=reference_protocol if phase=='reference' else TRAINING
    if phase=='D3':
        protocol,limits=d3['protocol'],d3['limits']
        require(a.get('host_output_directory')==f['host_calibrated_output']
                and a.get('benchmark_generation_sha256')==d3['benchmark_generation_sha256']
                and a.get('dataset_admission_sha256')==d3['dataset_admission_sha256']
                and a.get('initial_sha256')==d3['initial_sha256']
                and a.get('selected_gpu')=='0' and a.get('maximum_execution_launches')==1,
                'D3 authorization differs from the frozen dataset, initial state or launch')
    require(digest(args.authorization)==args.authorization_sha256 and a.get('approved') is True and
        a.get('protocol')==protocol and a.get('commit')==args.commit and
        a.get('freeze_sha256')==args.freeze_sha256 and a.get('image_id')==args.image_id==IMAGE and
        a.get('limits')==limits and a.get('output_directories',{}).get(phase)==str(args.output),
        'Separate exact reference/training authorization required')
    require(os.geteuid()!=0 and args.output.parent==Path('/outputs') and
        args.output.resolve()==args.output and not args.output.exists(), 'Fresh owned non-root /outputs child required')
    for p in (Path('/'),ROOT,args.oracle,args.cameras,args.freeze,args.authorization):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only source/oracle/cameras/root required')
    require(not os.statvfs('/outputs').f_flag&os.ST_RDONLY,'Writable separate output required')
    require({line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network execution required')
    return f,a
