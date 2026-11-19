"""Frozen Bell geometry, material labels and normal-transport domain."""
from ast import literal_eval
from pathlib import Path

import numpy as np

from audits.bell_calibration_e11 import rays
from benchmarks.bell_e11 import MESH, TOL, geometry_gate
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_heterogeneous_v1 import BRDF, TOLERANCES, check_artifact, read, seen_area_weights, smoothstep
from benchmarks.donut_heterogeneous_v2 import HIGH, MAX_PER_VIEW, MAX_TOTAL
from benchmarks.beckmann_reflectance_v2 import NODES_PER_POINT, ORDERS
from benchmarks.donut_recovery_v1 import ENVIRONMENT, GRIDS, IMAGE, REFERENCE_LIMITS, SIZE, numeric, unit
from benchmarks.donut_surface_q_d6 import POLICY as D6, coverage_summary, fitting_face_counts
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest
from infrastructure.surface_q_regularization import surface_pairs
from infrastructure.verified_ray_backend import identity


PROTOCOL = 'bell-kurt-heterogeneous-reference-v1'
FREEZE = 'docs/protocols/BELL_HETEROGENEOUS_REFERENCE_V1.freeze.json'
SOURCE_GENERATION = '5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7'
SOURCE_ADMISSION = 'f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605'
CALIBRATION = '4d8f4e13d7fd5fffec8d258bfc7c5599ef6d81525547d678ba088e76b1656c9c'
PANEL = [0, 4, 19, 2, 105, 114]
SPLITS = dict(fitting=PANEL[:4], held_out=PANEL[4:])
COUNTS = [4395, 4365, 4392, 4421, 3607, 3475]
POINTS = sum(COUNTS)
FIELD = dict(coordinates='2*(position-midpoint)/(mesh_max-mesh_min)',
             plane_normals=[[1., 0., .30], [0., 1., -.25]], plane_offsets=[.10, -.08],
             width_families=[[.42, .17], [.30, .112]], width_modulation=[.035, .016],
             weak_cap_coefficients=[1., 0., .35], weak_cap_start=.58, weak_cap_end=.86,
             isotropic_width=.22, grain=[.35, .80, .48], projection_zero=.05, projection_full=.20,
             boundary_band=.035, orientation_gap=.1)
FACE_MATCH = dict(position_abs=1e-6, normal_dot_min=1-2e-6, tie_break='plane distance, original PLY face index')
SOURCE_PATHS = dict(generation='geometry/generation.json', completion='completion.json',
                    mesh='geometry/mesh.ply', calibration='geometry/calibrations-safe.npz')
INPUT_NAMES = (*SOURCE_PATHS.values(), *(f'geometry/rays/{c}.npz' for c in PANEL),
               *(f'datasets/GlossySynthetic/bell/masks_train/{c}.png' for c in PANEL))
REFINEMENT = dict(baseline_grids=list(GRIDS), high_grids=list(HIGH),
                  per_view=MAX_PER_VIEW, total=MAX_TOTAL, max_refinements_per_pixel=1,
                  selection='Donut v2 maximum channel for absolute, RGB squared difference for pooled, lower pixel ID ties')
WORK = ('primary_rays', 'CPU_primary_crosschecks', 'shadow_directions', 'BRDF_point_directions',
        'slope_nodes_attempted', 'slope_nodes_returned')


def counters(points, refined):
    require(0 <= refined <= MAX_TOTAL, 'Bell refinement budget exceeded')
    directions = points*sum(g*g for g in GRIDS) + refined*sum(g*g for g in HIGH)
    return dict(primary_rays=0, CPU_primary_crosschecks=0, shadow_directions=directions,
                BRDF_point_directions=directions, slope_nodes_attempted=points*NODES_PER_POINT,
                slope_nodes_returned=points*NODES_PER_POINT)


LIMITS = {**REFERENCE_LIMITS, **counters(POINTS, MAX_TOTAL)}


def freeze(path, sha):
    require(path.resolve() == path and digest(path) == sha, 'Bell reference freeze differs')
    f = read(path)
    expected = dict(protocol=PROTOCOL, panel=PANEL, counts=COUNTS, splits=SPLITS, field=FIELD,
                    face_match=FACE_MATCH, refinement=REFINEMENT, limits=LIMITS, environment=ENVIRONMENT,
                    tolerances=TOLERANCES, image=IMAGE, backend=identity(), reflectance_orders=list(ORDERS),
                    source_generation_sha256=SOURCE_GENERATION, source_admission_sha256=SOURCE_ADMISSION)
    require(all(f[k] == value for k, value in expected.items()) and set(f['input_records']) == set(INPUT_NAMES),
            'Bell scientific definition differs')
    for name, value in f['sources'].items():
        p = ROOT/name
        require(p.resolve() == p and ROOT in p.parents and digest(p) == value, 'Frozen source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, commit=commit, freeze_sha256=sha, image_id=IMAGE,
                selected_gpu='0', host_output=f['host_output'], source_generation_sha256=SOURCE_GENERATION,
                source_admission_sha256=SOURCE_ADMISSION, limits=LIMITS,
                maximum_execution_launches=1, training_authorized=False)


def select_cameras(poses, splits):
    centers = -np.einsum('nij,ni->nj', poses[:, :, :3].astype(float), poses[:, :, 3].astype(float))
    directions = unit(centers)
    selected = []
    for candidates, count in ((splits['train'], 4), (splits['validation']+splits['held_view_nvs'], 2)):
        ids = sorted(map(int, candidates))
        for _ in range(count):
            idx = 0 if not selected else int(np.argmin((directions[ids]@directions[selected].T).max(1)))
            selected.append(ids.pop(idx))
    return selected


def face_data(vertices, triangles):
    faces = vertices[triangles]
    cross = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    require((double_area > 0).all(), 'Bell face attribution requires positive triangle areas')
    return faces, cross/double_area[:, None], double_area/2


def attribute_faces(points, raw, faces, geometric_normals, area):
    """Associate an archived point and normal with a PLY face, without tracing a ray."""
    from scipy.spatial import cKDTree

    centers = faces.mean(1)
    tol = FACE_MATCH['position_abs']
    radius = np.linalg.norm(faces-centers[:, None], axis=-1).max() + tol
    candidates = cKDTree(centers).query_ball_point(points, np.nextafter(radius, np.inf))
    edges = np.roll(faces, -1, axis=1)-faces
    edge_length = np.linalg.norm(edges, axis=-1)
    indices, weights, residuals, alternative_counts = [], [], [], []
    for row, pool in enumerate(candidates):
        ix = np.asarray(pool, dtype=np.int64)
        ix = ix[(geometric_normals[ix]@raw[row]) >= FACE_MATCH['normal_dot_min']]
        delta = points[row]-faces[ix]
        normals = geometric_normals[ix]
        plane = abs(np.einsum('ij,ij->i', delta[:, 0], normals))
        numerator = np.einsum('ijk,ik->ij', np.cross(edges[ix], delta), normals)
        possible = (plane <= tol) & (numerator >= -tol*edge_length[ix]).all(1)
        ids = np.flatnonzero(possible)
        require(len(ids) > 0, f'No normal-compatible Bell face for saved hit row {row}')
        pick = ids[np.lexsort((ix[ids], plane[ids]))[0]]
        indices.append(ix[pick])
        weights.append(np.roll(numerator[pick], -1)/(2*area[ix[pick]]))
        residuals.append(plane[pick])
        alternative_counts.append(len(ids)-1)
    return (np.asarray(indices, dtype=np.int64), np.asarray(weights), np.asarray(residuals),
            np.asarray(alternative_counts, dtype=np.int64))


def material_field(points, normals, bounds):
    half = (bounds[1]-bounds[0])/2
    require(points.shape == normals.shape and points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
            and np.isfinite(points).all() and np.isfinite(normals).all() and (half > 0).all()
            and np.max(abs(np.linalg.norm(normals, axis=1)-1)) <= 2e-12, 'Invalid Bell field coordinates')
    u = (points-bounds.mean(0))/half
    x, y, z = u.T
    planes = np.asarray(FIELD['plane_normals'])
    signed = u@planes.T + FIELD['plane_offsets']
    region = (signed[:, 0]*signed[:, 1] < 0).astype(np.uint8)
    distance = (abs(signed)/np.linalg.norm(planes/half, axis=1)).min(1)
    modulation = np.stack((np.sin(np.pi*(x+.4*y)), np.cos(np.pi*(z-.35*y))), axis=1)
    widths = np.asarray(FIELD['width_families'])[region]+FIELD['width_modulation']*modulation
    cap = u@FIELD['weak_cap_coefficients']
    weak = smoothstep((cap-FIELD['weak_cap_start'])/(FIELD['weak_cap_end']-FIELD['weak_cap_start']))
    widths = (1-weak[:, None])*widths+weak[:, None]*FIELD['isotropic_width']
    grain = unit(np.asarray(FIELD['grain']))
    proj = grain-(normals@grain)[:, None]*normals
    length = np.linalg.norm(proj, axis=1)
    taper = smoothstep((length-FIELD['projection_zero'])/(FIELD['projection_full']-FIELD['projection_zero']))
    fallback = np.cross(normals, [1., 0., 0.])
    fallback = np.where(np.linalg.norm(fallback, axis=1)[:, None] < 1e-6,
                        np.cross(normals, [0., 1., 0.]), fallback)
    axis = unit(np.where((taper > 0)[:, None], proj, fallback))
    squared = widths**2
    average = squared.mean(1)
    delta = (squared[:, 0]-squared[:, 1])*taper/2
    major, minor = average+delta, average-delta
    P = np.eye(3)-normals[:, :, None]*normals[:, None, :]
    Q = minor[:, None, None]*P+(2*delta)[:, None, None]*axis[:, :, None]*axis[:, None, :]
    color = np.stack((.20+.05*np.sin(np.pi*(x-y)), .58+.10*np.cos(np.pi*z),
                      .13+.04*np.cos(np.pi*(x+y))), axis=1)
    metallic = (.68+.08*np.sin(np.pi*(y-z)))[:, None]
    gap = delta/average
    widths = np.sqrt(np.stack((major, minor), axis=1))
    require(((widths >= .04)&(widths <= 1)).all() and (delta >= 0).all()
            and ((color >= 0)&(color <= 1)).all() and ((metallic >= 0)&(metallic <= 1)).all(),
            'Bell material outside the supported Kurt domain')
    return dict(axis=axis, Q_world=Q, color=color, metallic=metallic, widths_major_minor=widths,
                anisotropy_gap=gap, orientation_defined=delta > 0,
                orientation_eligible=gap >= FIELD['orientation_gap'], exactly_isotropic=delta == 0,
                nearly_isotropic=gap < FIELD['orientation_gap'], projection_length=length,
                projection_taper=taper, material_region=region, boundary_distance_native=distance,
                near_boundary=distance <= FIELD['boundary_band'], weak_patch_weight=weak)


def source(f, package, admission):
    """Read pledged v3 geometry bytes only. Read source masks from admitted v3 output."""
    require(digest(admission) == SOURCE_ADMISSION, 'Original Bell admission differs')
    a = read(admission)
    p = {k: check_artifact(package/name, f['input_records'][name]) for k, name in SOURCE_PATHS.items()}
    r, seal = read(p['generation']), read(p['completion'])
    require(digest(p['generation']) == SOURCE_GENERATION and digest(p['mesh']) == MESH
            and digest(p['calibration']) == CALIBRATION and a['researcher_reviewed_generation'] is True
            and a['researcher_admission']['generation_sha256'] == SOURCE_GENERATION
            and a['mesh_sha256'] == r['mesh_sha256'] == MESH and literal_eval(a['split_policy']) == r['splits']
            and r['status'] == 'complete' and r['scientific_work_complete'] and r['cleanup_verified']
            and seal['generation_sha256'] == SOURCE_GENERATION and seal['cleanup_verified']
            and seal['scientific_work_complete']
            and r['ray_backend'] == identity() and r['runtime']['backend']['sha256'] == identity()['binary_sha256'],
            'Bell v3 geometry, backend or generation seal differs')
    outputs = {x['path']: {k: x[k] for k in ('bytes', 'sha256')} for x in r['outputs']}
    for name, item in f['input_records'].items():
        if name not in (SOURCE_PATHS['generation'], SOURCE_PATHS['completion']):
            require(item == outputs[name], 'Geometry artifact differs from admitted v3 inventory')
    calibration = numeric(p['calibration'])
    P, K = calibration['pose'], calibration['K']
    require(P.shape == (128, 3, 4) and K.shape == (128, 3, 3)
            and np.array_equal(calibration['source_camera_ids'], np.arange(128))
            and select_cameras(P, r['splits']) == PANEL, 'Frozen Bell camera selection differs')
    vertices, triangles = mesh_arrays(p['mesh'])
    vertices = vertices.astype(np.float64)
    faces, face_normals, area = face_data(vertices, triangles)
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    views = {}
    from PIL import Image

    for camera, count in zip(PANEL, COUNTS):
        ray = f'geometry/rays/{camera}.npz'
        mask_file = f'datasets/GlossySynthetic/bell/masks_train/{camera}.png'
        data = numeric(check_artifact(package/ray, f['input_records'][ray]))
        with Image.open(check_artifact(package/mask_file, f['input_records'][mask_file])) as im:
            mask = np.asarray(im) > 127
        hit = data['hit']
        keys = ('origins', 'directions', 'positions', 'raw_normals', 'normals', 'depth', 'hit')
        require(mask.shape == (SIZE, SIZE) and np.array_equal(mask.ravel(), hit),
                'Archived Bell hit silhouette differs')
        gate = geometry_gate(*(data[k] for k in keys), mask.ravel(), P[camera], K[camera])
        o, d = rays(P[camera], K[camera], SIZE, SIZE)
        original = r['views'][camera]['geometry']
        require(gate['passed'] and original['passed'] and original['silhouette_iou'] >= TOL['silhouette_iou_min']
                and int(hit.sum()) == count and max(abs(o-data['origins']).max(),
                                                   abs(d-data['directions']).max()) <= 2e-6,
                f'Archived Bell camera geometry does not replay: {camera}')
        pos = data['positions'][hit].astype(float)
        raw = unit(data['raw_normals'][hit].astype(float))
        normals = unit(data['normals'][hit].astype(float))
        require(np.max(abs(raw+normals)) <= TOL['unit_normal_abs'], 'Archived normal sign differs')
        faces_ix, bary, residual, alternatives = attribute_faces(pos, raw, faces, face_normals, area)
        outgoing = -unit(data['directions'][hit].astype(float))
        require(((normals*outgoing).sum(1) > 0).all(), 'A valid Bell hit was lost')
        views[camera] = dict(pixel_indices=np.flatnonzero(hit), hit_mask=mask, shader_valid=np.ones(count, bool),
                             position=pos, raw_BVH_normals=data['raw_normals'][hit], normals=normals,
                             outgoing=outgoing, depth=data['depth'][hit], triangle=faces_ix,
                             barycentric=bary, face_normal=-face_normals[faces_ix], surface_residual=residual,
                             additional_face_candidates=alternatives, **material_field(pos, normals, bounds))
    return dict(vertices_native=vertices, triangles=triangles, bounds=bounds, area=area), views


def neighborhoods(geometry):
    """Use D6's geometric rule with Bell's final negated flat face normals."""
    pairs = surface_pairs(geometry['vertices_native'], geometry['triangles'],
                          max_distance=D6['max_surface_distance'], max_angle_degrees=D6['max_face_dihedral_degrees'])
    faces = pairs['face_pairs']
    count = D6['diagnostic_pairs']
    require(len(faces) >= count + D6['batch_pairs'], 'Insufficient Bell training and reserved neighborhoods')
    diagnostic = np.linspace(0, len(faces)-1, count, dtype=np.int64)
    train = np.setdiff1d(np.arange(len(faces)), diagnostic, assume_unique=True)
    domain = {k: pairs[k] for k in ('face_pairs', 'distance', 'points0', 'points1')}
    domain.update(normals0=-pairs['normals0'], normals1=-pairs['normals1'], face_area=geometry['area'],
                  training_ids=train, diagnostic_ids=diagnostic)
    require(((domain['normals0']*domain['normals1']).sum(1) > -1+1e-6).all(),
            'Invalid Bell normal transport')
    return domain, {k: v for k, v in pairs.items() if k.startswith('excluded_')}


def view_geometry_summary(view):
    axes, normals, Q = (view[k] for k in ('axis', 'normals', 'Q_world'))
    eigen_error = abs(np.linalg.eigvalsh(Q)[:, 1:] - view['widths_major_minor'][:, ::-1]**2).max()
    tangent_error = abs(np.einsum('nij,nj->ni', Q, normals)).max()
    require(max(eigen_error, tangent_error, abs((axes*normals).sum(1)).max()) <= 2e-12,
            'Bell field tensor, widths or orientation do not agree')
    return dict(
        **{k: int(view[k].sum()) for k in ('shader_valid', 'near_boundary', 'nearly_isotropic', 'exactly_isotropic')},
        normal_dot_view_min=float((normals*view['outgoing']).sum(1).min()),
        max_face_plane_residual=float(view['surface_residual'].max()),
        hits_with_alternative_face_candidate=int((view['additional_face_candidates'] > 0).sum()),
        projection_isotropy_count=int((view['projection_taper'] == 0).sum()),
        world_Q_eigenvalue_error=float(eigen_error), world_Q_normal_null_error=float(tangent_error))


def coverage(geometry, views):
    domain, exclusions = neighborhoods(geometry)
    area = domain['face_area']
    fit_counts = fitting_face_counts(views, SPLITS['fitting'], len(area))
    train = np.zeros(len(domain['distance']), dtype=np.int64)
    train[domain['training_ids']] = 1
    measured = coverage_summary(domain, train, fit_counts)
    for field in ('pair_draws', 'unique_pairs'):
        measured.pop(field)
    observed_area = {}
    for name, cameras in SPLITS.items():
        triangles = np.concatenate([views[c]['triangle'] for c in cameras])
        _, observed_area[name] = seen_area_weights(triangles, area)
    union = np.unique(domain['face_pairs'])
    dihedral = np.degrees(np.arccos(np.clip((domain['normals0']*domain['normals1']).sum(1), -1., 1.)))
    return dict(eligible_pairs=len(train), training_pairs=int(train.sum()),
                reserved_pairs=len(domain['diagnostic_ids']), transport_exclusions=exclusions,
                eligible_faces=len(union), eligible_area_fraction=float(area[union].sum()/area.sum()),
                training_domain=measured, observed_area_fraction=observed_area,
                faces=len(area), area_native=float(area.sum()),
                transport_distance_native_min=float(domain['distance'].min()),
                transport_distance_native_max=float(domain['distance'].max()),
                maximum_eligible_dihedral_degrees=float(dihedral.max()))
