"""Geometry-only extension of the admitted heterogeneous Bell acquisition."""
from math import ceil
from pathlib import Path

import numpy as np

from benchmarks import bell_heterogeneous_training_v1 as sparse
from benchmarks import bell_heterogeneous_v1 as b
from benchmarks.donut_heterogeneous_training_v1 import COVERAGE_RADIUS, NORMAL_COSINE
from infrastructure.signatures import require


PROTOCOL = 'bell-kurt-heterogeneous-dense-reference-v1'
FREEZE = 'docs/protocols/BELL_DENSE_REFERENCE_V1.freeze.json'
ORIGINAL = [*b.SPLITS['fitting']]
NEW = [127, 71, 73, 56, 75, 95, 93, 110, 91, 7, 5, 3]
NEW_COUNTS = [3301, 4025, 4087, 4195, 4022, 3813, 3822, 3556, 3891, 4375, 4418, 4341]
SPLITS = dict(fitting=ORIGINAL+NEW, held_out=b.SPLITS['held_out'])
PANEL = [*SPLITS['fitting'], *SPLITS['held_out']]
COUNTS = dict(zip(b.PANEL+NEW, b.COUNTS+NEW_COUNTS))
LIMITS = {**b.LIMITS, **b.counters(sum(NEW_COUNTS), b.MAX_TOTAL)}
SELECTION = 'Continue original fitting maximin from origin-to-camera unit directions, ascending ID breaks ties'
SPARSE_FILES = [name for c in b.PANEL for name in
                (f'view-{c}.npz', *(f'view-{c}-{tag}.png' for tag in ('RGB','diffuse','specular')))]
PATH_KEYS = ('oracle','admission','sparse','sparse_admission','sparse_verification')


def input_names():
    return (*b.SOURCE_PATHS.values(), *(f'geometry/rays/{c}.npz' for c in NEW),
            *(f'datasets/GlossySynthetic/bell/masks_train/{c}.png' for c in NEW))


def select(poses, splits, originals, count):
    """Continue the original angular maximin order in the original training IDs."""
    require(poses.shape[1:] == (3, 4) and np.isfinite(poses).all() and count >= 0,
            'Invalid Bell pose inventory or request')
    groups = [list(map(int, splits[key])) for key in ('train','validation','held_view_nvs')]
    all_ids = [camera for group in groups for camera in group]
    training = groups[0]
    require(sorted(all_ids) == list(range(len(poses))) and originals and
            len(set(originals)) == len(originals) and set(originals) <= set(training),
            'Invalid or overlapping Bell splits')
    ids = sorted(set(training)-set(originals))
    require(len(ids) >= count, 'Insufficient training cameras')
    p = poses.astype(np.float64)
    centers = -np.einsum('nij,ni->nj', p[:, :, :3], p[:, :, 3])
    require((np.linalg.norm(centers, axis=1)>0).all(), 'Camera center coincides with acquisition origin')
    directions = b.unit(centers)
    picked = list(originals)
    angles = []
    for _ in range(count):
        scores = (directions[ids]@directions[picked].T).max(1)
        index = int(scores.argmin())
        picked.append(ids.pop(index))
        angles.append(float(np.degrees(np.arccos(np.clip(scores[index], -1, 1)))))
    return picked[len(originals):], angles, centers


def training_projection():
    """Work and exposure counts for the later admission-bound training stage."""
    n = sum(COUNTS[c] for c in SPLITS['fitting'])
    batches, tail = divmod(n, 512)
    evaluations = len(sparse.STEPS)*8
    slots = evaluations*sum(ceil(COUNTS[c]/512) for c in PANEL)
    return dict(fitting_observations=n, held_observations=sum(COUNTS[c] for c in SPLITS['held_out']),
        training_slots=10000*512, effective_passes=10000*512/n, batch_size=512,
        full_batches_per_permutation=batches, unused_tail_per_full_permutation=tail,
        evaluation_shader_calls=slots, evaluation_shading_observations=evaluations*sum(COUNTS.values()))


def freeze(path, sha):
    require(path.resolve() == path and b.digest(path) == sha, 'Dense reference freeze differs')
    f = b.read(path)
    expected = dict(protocol=PROTOCOL, selection=SELECTION, splits=SPLITS, panel=PANEL, new_cameras=NEW,
                    counts={str(c): COUNTS[c] for c in PANEL}, limits=LIMITS, field=b.FIELD,
                    image=b.IMAGE, environment=b.ENVIRONMENT, BRDF=b.BRDF, refinement=b.REFINEMENT,
                    tolerances=b.TOLERANCES, backend=b.identity(), reflectance_orders=list(b.ORDERS),
                    original_geometry_generation=b.SOURCE_GENERATION, original_geometry_admission=b.SOURCE_ADMISSION)
    require(all(f[k] == v for k,v in expected.items()) and set(f['input_records']) == set(input_names())
            and set(f['host_inputs']) == set(PATH_KEYS)
            and set(f['sparse']['outputs']) == set(SPARSE_FILES)
            and f['sparse']['host_reference'] == f['host_inputs']['sparse']
            and {k:f['sparse']['dataset'][k]['sha256'] for k in ('admission','verification','generation')}
                == dict(admission=sparse.ADMISSION, verification=sparse.VERIFICATION, generation=sparse.GENERATION),
            'Dense reference or sparse control differs')
    for name, sha256 in f['sources'].items():
        file = b.ROOT/name
        require(file.resolve() == file and b.ROOT in file.parents and b.digest(file) == sha256,
                'Frozen dense reference source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, commit=commit, freeze_sha256=sha,
                image_id=b.IMAGE, selected_gpu='0', maximum_execution_launches=1,
                host_output=f['host_output'], new_cameras=NEW, limits=LIMITS,
                source_geometry_sha256=b.SOURCE_GENERATION,
                sparse_admission_sha256=sparse.ADMISSION, training_authorized=False)


def host_paths(f):
    return tuple(Path(f['host_inputs'][key]) for key in PATH_KEYS)


def source(f, oracle, admission, sparse_root, sparse_admission, sparse_verification):
    """Verify all bound input bytes and construct labels for new views from saved hits."""
    from PIL import Image

    old_path = b.ROOT/b.FREEZE
    require(b.digest(old_path) == sparse.REFERENCE_FREEZE, 'Sparse geometry freeze differs')
    old = b.read(old_path)
    geometry, _ = b.source(old, oracle, admission)
    previous, reused = sparse.dataset(f['sparse'], sparse_root, sparse_admission, sparse_verification)
    require(all(f['sparse']['outputs'][name] == previous['outputs'][name] for name in SPARSE_FILES),
            'Reused sparse reference view inventory differs')
    p = {name: b.check_artifact(oracle/name, f['input_records'][name]) for name in input_names()}
    report = b.read(p[b.SOURCE_PATHS['generation']])
    outputs = {x['path']: {k:x[k] for k in ('bytes','sha256')} for x in report['outputs']}
    require(all(f['input_records'][name] == old['input_records'][name] for name in b.SOURCE_PATHS.values())
            and all(outputs[name] == f['input_records'][name] for name in set(p)-set(b.SOURCE_PATHS.values())),
            'Dense source differs from admitted original geometry')
    pose = b.numeric(p[b.SOURCE_PATHS['calibration']])
    poses, intrinsics = pose['pose'], pose['K']
    selected, angles, centers = select(poses, report['splits'], ORIGINAL, len(NEW))
    require(selected == NEW and set(SPLITS['held_out']) <= set(map(int, report['splits']['held_view_nvs'])),
            'Dense view selection differs')
    faces, raw_faces, area = b.face_data(geometry['vertices_native'], geometry['triangles'])
    views = dict(reused)
    for camera in NEW:
        ray = b.numeric(p[f'geometry/rays/{camera}.npz'])
        with Image.open(p[f'datasets/GlossySynthetic/bell/masks_train/{camera}.png']) as image:
            mask = np.asarray(image)>127
        keys = ('origins', 'directions', 'positions', 'raw_normals', 'normals', 'depth', 'hit')
        hit = ray['hit']
        width = b.SIZE
        require(mask.shape == (width,width) and hit.dtype == bool and np.array_equal(mask.ravel(), hit),
                f'Dense archived silhouette differs: {camera}')
        gate = b.geometry_gate(*(ray[k] for k in keys), mask.ravel(), poses[camera], intrinsics[camera])
        origins, directions = b.rays(poses[camera], intrinsics[camera], width, width)
        recorded = report['views'][camera]
        require(recorded['camera_id'] == str(camera) and gate['passed'] and recorded['geometry']['passed']
                and recorded['geometry']['silhouette_iou'] >= b.TOL['silhouette_iou_min']
                and int(hit.sum()) == COUNTS[camera] and max(abs(origins-ray['origins']).max(),
                    abs(directions-ray['directions']).max()) <= 2e-6, f'Dense archived geometry differs: {camera}')
        position = ray['positions'][hit].astype(float)
        raw = b.unit(ray['raw_normals'][hit].astype(float))
        normals = b.unit(ray['normals'][hit].astype(float))
        outgoing = -b.unit(ray['directions'][hit].astype(float))
        require(np.max(abs(raw+normals)) <= b.TOL['unit_normal_abs'] and
                ((normals*outgoing).sum(1)>0).all(), f'Dense archived normal differs: {camera}')
        face, bary, residual, alternatives = b.attribute_faces(position, raw, faces, raw_faces, area)
        views[camera] = dict(pixel_indices=np.flatnonzero(hit), hit_mask=mask, shader_valid=np.ones(COUNTS[camera], bool),
            position=position, raw_BVH_normals=ray['raw_normals'][hit], normals=normals, outgoing=outgoing,
            depth=ray['depth'][hit], triangle=face, barycentric=bary, face_normal=-raw_faces[face],
            surface_residual=residual, additional_face_candidates=alternatives,
            **b.material_field(position, normals, geometry['bounds']))
        b.view_geometry_summary(views[camera])
    return geometry, {c: views[c] for c in PANEL}, centers, angles


def support(views, camera, fitting, centers, highlights):
    """Describe outgoing diversity where a nearby normal-compatible fit exists."""
    from scipy.spatial import cKDTree

    current = views[camera]
    valid = current['shader_valid']
    points, normals = (current[k][valid] for k in ('position','normals'))
    seen = np.zeros((len(fitting), len(points)), bool)
    lit = np.zeros_like(seen)
    outgoing = b.unit(centers[np.array(fitting), None] - points[None])
    for idx, c in enumerate(fitting):
        v = views[c]
        mask = v['shader_valid']
        neighborhood = cKDTree(v['position'][mask]).query_ball_point(points, COVERAGE_RADIUS)
        vn = v['normals'][mask]
        bright = v['highlight_mask'][mask] if highlights else None
        for row, neighbors in enumerate(neighborhood):
            if neighbors:
                ids = np.asarray(neighbors)
                ids = ids[(vn[ids]@normals[row]) >= NORMAL_COSINE]
                seen[idx,row] = ids.size > 0
                if highlights:
                    lit[idx,row] = bright[ids].any()
    count = seen.sum(0)
    max_angle = np.zeros(len(points))
    for i in range(len(fitting)):
        for j in range(i):
            ok = seen[i]&seen[j]
            cos = np.einsum('ij,ij->i', outgoing[i,ok], outgoing[j,ok])
            max_angle[ok] = np.maximum(max_angle[ok], np.degrees(np.arccos(np.clip(cos,-1,1))))
    record = dict(nearby_fitting_count=count, max_outgoing_angle_degrees=max_angle)
    if highlights:
        record['nearby_highlight_count'] = lit.sum(0)
    return record


def coverage(geometry, views, centers, *, highlights):
    summary, arrays = {}, {}
    for name, fitting in (('sparse',ORIGINAL),('dense',SPLITS['fitting'])):
        ids = np.concatenate([views[c]['triangle'][views[c]['shader_valid']] for c in fitting])
        _, area = b.seen_area_weights(ids, geometry['area'])
        row = summary[name] = dict(fitting_observations=len(ids), observed_triangle_area=area, held_out={})
        for camera in SPLITS['held_out']:
            result = support(views, camera, fitting, centers, highlights)
            count = result['nearby_fitting_count']
            angles = result['max_outgoing_angle_degrees'][count>=2]
            row['held_out'][str(camera)] = dict(observations=len(count), no_fitting=int((count==0).sum()),
                fitting_counts=np.bincount(count,minlength=len(fitting)+1).tolist(),
                two_or_more_views=len(angles), maximum_outgoing_angle_median=float(np.median(angles)) if len(angles) else None)
            if highlights:
                h = result['nearby_highlight_count']
                row['held_out'][str(camera)].update(no_highlight=int((h==0).sum()),
                    highlight_counts=np.bincount(h,minlength=len(fitting)+1).tolist())
            arrays.update({f'{name}_{camera}_{k}':v for k,v in result.items()})
    summary['matched_angular_change'] = {}
    for camera in SPLITS['held_out']:
        for key in ('nearby_fitting_count','max_outgoing_angle_degrees',*(['nearby_highlight_count'] if highlights else [])):
            require(np.all(arrays[f'dense_{camera}_{key}'] >= arrays[f'sparse_{camera}_{key}']),
                    'Dense acquisition lost a sparse correspondence')
        multiple = arrays[f'sparse_{camera}_nearby_fitting_count'] >= 2
        diff = arrays[f'dense_{camera}_max_outgoing_angle_degrees'] - arrays[f'sparse_{camera}_max_outgoing_angle_degrees']
        summary['matched_angular_change'][str(camera)] = dict(observations=int(multiple.sum()),
            median_increase_degrees=float(np.median(diff[multiple])) if multiple.any() else None)
    return summary, arrays
