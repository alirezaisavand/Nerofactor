"""Frozen inputs, common physical material evaluation and convergence for Gate 6."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from benchmarks import donut_representation_gate5_v1 as gate5
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import ENVIRONMENT, SIZE, numeric
from benchmarks.material_information_e14 import rotated_light
from infrastructure.signatures import require
from infrastructure.verified_ray_backend import identity

ROOT = gate5.ROOT
read, check, digest = gate5.read, gate5.check, gate5.digest
PROTOCOL = 'donut-unseen-light-gate6-v1'
FREEZE = 'docs/protocols/DONUT_RELIGHTING_GATE6_V1.freeze.json'
MANIFEST = 'docs/results/publication_gate5/results-v1/artifact-manifest.json'
MANIFEST_SHA = 'b52100c7eb3f91bf8efe85b1b73aca2ff87b444a6cfb638eae8b31a1439d87e0'
ARMS = ('K1-Oracle', gate5.ARM, gate5.AMBIENT)
ROLES = ('oracle', *ARMS)
VARIANTS = ('A_corrected_core_v1', 'donut_tensor_chart_weak_start_v1', 'donut_tensor_ambient_weak_start_v1')
TENSOR_ARMS = dict(zip(ARMS, (None, 'D9-C', 'D9-T')))
CAMERAS = (28, 34)
HITS = (4849, 2497)
COUNTS = (4849, 2495)
MATCHED = (4830, 2476)
DRAWS, CHUNK, STEP = 32, 512, 10000
BASE, HIGH = (256, 512), (1024, 2048)
MAX_VIEW, MAX_TOTAL = 64, 128
LIGHT = rotated_light(ENVIRONMENT)
TOL = gate5.m.rgb.geo.h.TOLERANCES
MATERIAL_ATOL = 1e-6  # Existing Donut tensor export and checkpoint field parity allowance.
FRAME_ATOL = 2e-6
POINT_BATCH, DIRECTION_CHUNK = 4, 16384
DIRECTIONS = sum(COUNTS)*sum(n*n for n in BASE)
MAX_DIRECTIONS = DIRECTIONS+MAX_TOTAL*sum(n*n for n in HIGH)
LIMITS = dict(checkpoint_loads=3, native_shader_calls=len(ARMS)*DRAWS*sum((n+CHUNK-1)//CHUNK for n in COUNTS),
              native_shadow_slots=len(ARMS)*DRAWS*sum(COUNTS)*1536, independent_shadow_directions=MAX_DIRECTIONS,
              independent_BRDF_point_directions=len(ROLES)*MAX_DIRECTIONS,
              primary_rays=0, optimizer_updates=0)
RESOURCES = dict(gpu='0', cpu_threads=4, host_memory_gib=64, pids=128, cuda_allocator_fraction=.25,
                 output_bytes=512*1024**2, watchdog_seconds=1800, kill_grace_seconds=30,
                 inner_seconds=1740, inner_kill_grace_seconds=20)
SOURCES = (
    'benchmarks/donut_relighting_gate6_v1.py', 'experiments/donut_relighting_gate6_v1.py',
    'experiments/analyze_donut_relighting_gate6_v1.py', 'experiments/run_donut_relighting_gate6_v1.sh',
    'tests/scientific/test_donut_relighting_gate6_v1.py', 'docs/protocols/DONUT_RELIGHTING_GATE6_V1.md',
    MANIFEST, gate5.FREEZE, 'benchmarks/material_information_e14.py',
    'benchmarks/bell_reference_e11.py', 'benchmarks/bell_e11.py',
    'experiments/bell_quadrature_e11e.py', 'experiments/bell_highlight_pilot_e11d.py',
    'experiments/bell_highlight_e11d.py',
    'benchmarks/generate_bell_e11.py', 'benchmarks/generate_donut_recovery_v1.py',
    'scientific_variants/material_a_v1.py', 'scientific_variants/donut_tensor_d8.py',
    'scientific_variants/donut_tensor_shader_d8.py', 'scientific_variants/donut_tensor_initialization_d9.py',
    'infrastructure/material_recovery_e13_evaluation.py', 'infrastructure/sphere_replay.py',
    'infrastructure/verified_ray_backend.py', 'infrastructure/stage_sources.py',
    'scientific_sources/manifest-v1.json', 'experiments/bell_neural_material_e16.py',
    'experiments/donut_reconstructed_geometry_v1.py',
    'experiments/analyze_donut_representation_gate5_v1.py',
)
VIEW_FIELDS = ('pixel_indices','hit_mask','shader_valid','position','normals','outgoing','axis',
               'widths_major_minor','color','metallic','Q_world','material_region','near_boundary',
               'nearly_isotropic','highlight_mask')


def with_path(identity_, path):
    return dict(identity_, path=str(path))


def definition(out, support):
    require(digest(ROOT/MANIFEST)==MANIFEST_SHA,'Accepted Gate 5 manifest differs')
    a = read(ROOT/MANIFEST)
    require(a['status']=='verified_complete' and a['image']==identity()['image_id']
            and a['support']==gate5.parent()['support'],'Gate 5 result identity differs')
    geometry = gate5.parent()['gate3']['geometry_audit']
    artifacts = dict(oracle=geometry['oracle'], cameras=geometry['cameras'], **geometry['dataset'])
    methods = {}
    for i, (arm, variant) in enumerate(zip(ARMS, VARIANTS)):
        row = a['arms'][arm]
        receipt = row['receipt']
        folder = Path(receipt['path']).parent
        artifacts[f'receipt_{i}'] = receipt
        artifacts[f'checkpoint_{i}'] = with_path(row['checkpoints']['checkpoint-10000.pth'],folder/'checkpoint-10000.pth')
        methods[arm] = dict(variant=variant, tensor_arm=TENSOR_ARMS[arm], folder=str(folder),
                            receipt=f'receipt_{i}', checkpoint=f'checkpoint_{i}')
    ref = Path(artifacts['generation']['path']).parent
    for camera in CAMERAS:
        name=f'view-{camera}.npz'
        artifacts[f'view_{camera}'] = with_path(geometry['views'][name],ref/name)
    require(set(support)==set(map(str,CAMERAS)) and all(set(v)=={'foreground_sha256','matched_sha256'}
            for v in support.values()), 'Unknown Gate 6 pixel inventory')
    return dict(schema_version=1, protocol=PROTOCOL, host_output=str(out), cameras=list(CAMERAS), size=SIZE,
                methods=methods, roles=list(ROLES), artifacts=artifacts, endpoint=STEP, image=a['image'],
                backend=identity(), gate5_manifest_sha256=MANIFEST_SHA, gate5_support=a['support'], support=support,
                oracle_foreground=dict(zip(map(str,CAMERAS),COUNTS)), matched=dict(zip(map(str,CAMERAS),MATCHED)),
                environment=LIGHT, fitting_environment=ENVIRONMENT, indirect='zero', point_light='absent',
                BRDF='Kurt-a0-metallic-additive-v1', native_draws=DRAWS, native_chunk=CHUNK,
                native_samples=dict(diffuse=512,specular=1024),
                native_tape=dict(base=6100000,camera_stride=1000000,row_stride=64,tensor_offset=100000000),
                independent_chunk=dict(points=POINT_BATCH,directions=DIRECTION_CHUNK),
                base_grids=list(BASE), high_grids=list(HIGH), max_refined_per_camera=MAX_VIEW,
                max_refined_total=MAX_TOTAL, tolerances=TOL, material_max_abs=MATERIAL_ATOL,
                frame_max_abs=FRAME_ATOL, limits=LIMITS, resources=RESOURCES,
                training_authorized=False, sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    path = ROOT/FREEZE
    require(digest(path)==sha, 'Gate 6 freeze identity differs')
    f = read(path)
    out = Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-relight-gate6-') and f==definition(out,f['support']),
            'Frozen relighting definition differs')
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, commit=commit, freeze_sha256=sha,
                host_output=f['host_output'], image=f['image'], selected_gpu='0',
                checkpoint_cpu_pickle_trust=True, checkpoints={a:f['artifacts'][r['checkpoint']]['sha256']
                      for a,r in f['methods'].items()}, maximum_execution_launches=1,
                training_authorized=False, limits=LIMITS, resources=RESOURCES)


def base_paths(f, native=False):
    return {key:Path(f'/inputs/{key}{Path(v["path"]).suffix}' if native else v['path'])
            for key,v in f['artifacts'].items()}


def endpoint_paths(f, native=False):
    return {(arm,camera):Path(f'/inputs/export-{i}-{camera}.npz') if native
            else Path(f['methods'][arm]['folder'])/endpoint_name(camera)
            for i,arm in enumerate(ARMS) for camera in CAMERAS}


def endpoint_name(camera):
    return f'step-{STEP:05d}/camera-{camera}/materials.npz'


def pixels_sha(pixels):
    return sha256(np.asarray(pixels,dtype='<u4').tobytes()).hexdigest()


def admit(f, paths, exports, *, check_support=True):
    for key,item in f['artifacts'].items():
        check(paths[key],item)
    gen, verify, admitted = (read(paths[k]) for k in ('generation','verification','admission'))
    require(gen['cleanup_verified'] and gen['scientific_work_complete'] and gen['environment']==ENVIRONMENT
            and gen['backend']==f['backend'] and gen['outputs']==verify['outputs']==admitted['output_identities']
            and admitted['researcher_admitted_dataset'] is True, 'Expected admitted Donut oracle dataset')
    receipts, materials = {}, {}
    for arm, recipe in f['methods'].items():
        r = receipts[arm] = read(paths[recipe['receipt']])
        require(r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified']
                and r['optimizer_returns']==STEP and r['saved_endpoints'][-1]['step']==STEP
                and r['initial']['variant']==recipe['variant'] and r['arm']==arm and r['geometry']=='oracle'
                and r['benchmark_sha256']==gate5.m.GENERATION and r['dataset_admission_sha256']==gate5.m.ADMISSION
                and r['backend']==f['backend'] and r['known_lighting_audit']['learned_module_calls']==0
                and r['matched_support']==f['gate5_support']
                and r['outputs']['checkpoint-10000.pth']=={k:f['artifacts'][recipe['checkpoint']][k]
                      for k in ('bytes','sha256')}, 'Checkpoint-producing history identity differs')
        for camera in CAMERAS:
            materials[arm,camera] = numeric(check(exports[arm,camera],r['outputs'][endpoint_name(camera)]))
    views = {}
    for camera,hits,count,common in zip(CAMERAS,HITS,COUNTS,MATCHED):
        key = f'view_{camera}'
        name = f'view-{camera}.npz'
        require(all(f['artifacts'][key][k]==gen['outputs'][name][k] for k in ('bytes','sha256')),
                'Donut view identity differs')
        raw = numeric(paths[key])
        mask = raw['hit_mask']
        valid = raw['shader_valid']
        ids = raw['pixel_indices']
        require(mask.dtype==valid.dtype==np.bool_ and mask.shape==(SIZE,SIZE)
                and np.array_equal(ids,np.flatnonzero(mask)) and len(ids)==hits and valid.sum()==count,
                'Unexpected oracle raster support')
        np.testing.assert_array_equal(valid, (raw['normals']*raw['outgoing']).sum(1)>0)
        require((raw['linear_components'][~valid]==0).all()
                and ((raw['normals'][valid].astype('f4')*raw['outgoing'][valid].astype('f4')).sum(1)>0).all(),
                'Archived black support or native facing differs')
        v = {k:raw[k][valid] for k in VIEW_FIELDS if k not in {'hit_mask','shader_valid'}}
        v['hit_mask'] = mask
        matched = [np.flatnonzero(materials[arm,camera]['hit_mask']) for arm in ARMS]
        require(all(materials[arm,camera]['hit_mask'].dtype==np.bool_
                    and materials[arm,camera]['hit_mask'].shape==mask.shape for arm in ARMS)
                and len(matched[0])==common and all(np.array_equal(matched[0],m) for m in matched[1:])
                and np.isin(matched[0],v['pixel_indices']).all(), 'Archived masks differ')
        v['matched_pixels'] = matched[0]
        rows = np.searchsorted(v['pixel_indices'],matched[0])
        for arm in ARMS:
            maps = materials[arm,camera]
            i = maps['hit_mask']
            np.testing.assert_array_equal(maps['position'][i],v['position'][rows].astype('f4'))
            np.testing.assert_array_equal(maps['normal'][i],v['normals'][rows].astype('f4'))
            np.testing.assert_array_equal(maps['linear_gt'][i],raw['linear_components'][valid][rows,2].astype('f4'))
        v['support'] = dict(foreground_sha256=pixels_sha(v['pixel_indices']),matched_sha256=pixels_sha(matched[0]))
        if check_support:
            require(v['support']==f['support'][str(camera)],'Frozen relighting pixels differ')
        views[camera] = v
    return receipts,views,materials


def tape(camera, begin, draw):
    seed = 6100000+camera*1000000+begin*64+draw
    return seed,seed+100000000


def tensor_from_axes(widths,axis,n):
    b=np.cross(n,axis)
    return (widths[:,0,None,None]**2*axis[:,:,None]*axis[:,None,:]
            +widths[:,1,None,None]**2*b[:,:,None]*b[:,None,:])


def physical_axes(q, n):
    """Recover principal widths/director from world Q in the unchanged oracle plane."""
    q,n = np.asarray(q,dtype=np.float64),np.asarray(n,dtype=np.float64)
    require(q.shape==(len(n),3,3) and n.shape==(len(n),3) and np.isfinite(q).all() and np.isfinite(n).all()
            and np.max(abs(np.linalg.norm(n,axis=1)-1))<=FRAME_ATOL
            and np.max(abs(q-q.swapaxes(1,2)))<=MATERIAL_ATOL
            and np.max(abs(np.einsum('nij,nj->ni',q,n)))<=MATERIAL_ATOL,'Invalid world Q or shading plane')
    t=np.cross(n,np.eye(3)[abs(n).argmin(1)])
    t/=np.linalg.norm(t,axis=1,keepdims=True)
    u=np.stack((t,np.cross(n,t)),axis=-1)
    local=u.swapaxes(1,2)@q@u
    eigenvalues, vectors=np.linalg.eigh(local)
    require((eigenvalues[:,0]>0).all(),'Non-positive tangent Q')
    widths=np.sqrt(eigenvalues[:,::-1])
    require((widths>=.04-MATERIAL_ATOL).all() and (widths<=1+MATERIAL_ATOL).all(),'Widths outside admitted domain')
    axis=np.einsum('nij,nj->ni',u,vectors[:,:,1])
    error=float(abs(tensor_from_axes(widths,axis,n)-q).max())
    require(error<=MATERIAL_ATOL,'Principal axes fail to reconstruct Q')
    return widths,axis,error


def verify_grid(pixels, grid, a):
    require(set(a)=={'pixels','grid','components','visibility'} and int(a['grid'])==grid
            and np.array_equal(pixels,a['pixels']), 'Grid or pixel identity differs')
    rgb,count=a['components'],a['visibility']
    require(rgb.shape==(len(ROLES),len(pixels),3,3) and rgb.dtype==np.float64
            and np.isfinite(rgb).all() and (rgb>=0).all() and np.array_equal(rgb[:,:,0]+rgb[:,:,1],rgb[:,:,2])
            and count.shape==(len(pixels),) and count.dtype.kind in 'iu'
            and ((count>=0)&(count<=grid**2)).all(),'Invalid grid capture')


def next_pixel(pixels, rgb, refined):
    """Absolute failures first, then largest squared RGB residual for pooled failure."""
    candidates=[]
    seen=np.isin(pixels,refined)
    for i, (coarse,fine) in enumerate(zip(rgb[0],rgb[1])):
        if convergence(fine[:,2],coarse[:,2])['passed']:
            continue
        difference=fine[:,2]-coarse[:,2]
        error=abs(difference).max(1)
        absolute=error>TOL['convergence_rgb_abs']
        require(not (absolute&seen).any(),'Refined absolute radiance convergence failed')
        use_absolute=bool(absolute.any())
        eligible=np.flatnonzero(~seen & (absolute if use_absolute else True))
        require(len(eligible)>0,'Pooled convergence unresolved after refinement')
        score=error if use_absolute else np.square(difference).sum(1)
        row=eligible[np.lexsort((pixels[eligible],-score[eligible]))[0]]
        candidates.append((not use_absolute,-float(score[row]),int(pixels[row]),i))
    return min(candidates)[2] if candidates else None


def refine(pixels, fetch, prior):
    def pair(rows, grids, prefix):
        captures=[fetch(rows,n,prefix) for n in grids]
        for n,a in zip(grids,captures):
            verify_grid(pixels[rows],n,a)
        return tuple(np.stack([a[k] for a in captures]) for k in ('components','visibility'))

    rgb,count=pair(np.arange(len(pixels)),BASE,'base')
    grids=np.broadcast_to(np.array(BASE)[:,None],count.shape).copy()
    done=[]
    while True:
        pixel=next_pixel(pixels,rgb,done)
        if pixel is None:
            return rgb,count,grids,done
        require(len(done)<MAX_VIEW and prior+len(done)<MAX_TOTAL,'Gate 6 refinement budget exhausted')
        row=np.array([np.searchsorted(pixels,pixel)])
        patch,visibility=pair(row,HIGH,f'pixel-{pixel}')
        rgb[:,:,row]=patch
        count[:,row]=visibility
        grids[:,row]=np.array(HIGH)[:,None]
        done.append(pixel)
