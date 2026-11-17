"""Describe camera 34 fitting coverage from saved positions and normals."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from analyze_donut_recovery_v2 import BATCH, CAMERAS, GENERATION, REFERENCE, ROOT, digest, numeric, read


RADII = (.005, .01, .02)
MIN_NORMAL_COSINE = np.cos(np.pi / 6)


def geometric_coverage(points, normals, faces, fitting_views):
    counts = np.zeros((len(points), len(RADII)), dtype=int)
    same_face = np.zeros(len(points), dtype=int)
    for view in fitting_views:
        valid = view['shader_valid']
        positions, directions = view['position'][valid], view['normals'][valid]
        same_face += np.isin(faces, view['triangle'][valid])
        candidates = cKDTree(positions).query_ball_point(points, RADII[-1])
        for i, ids in enumerate(candidates):
            if not ids:
                continue
            eligible = directions[ids] @ normals[i] >= MIN_NORMAL_COSINE
            if eligible.any():
                nearest = np.linalg.norm(positions[np.asarray(ids)[eligible]] - points[i], axis=-1).min()
                counts[i] += nearest <= np.array(RADII)
    return counts, same_face


def checked(path, sha):
    if digest(path) != sha:
        raise ValueError('Saved artifact identity differs: ' + str(path))
    return path


def error_summary(q, angle, rgb, mask):
    if not mask.any():
        return None
    return dict(observations=int(mask.sum()), Q_median=float(np.median(q[mask])),
                Q_p95=float(np.quantile(q[mask], .95)), Q_above_005=int((q[mask] > .05).sum()),
                orientation_median_degrees=float(np.median(angle[mask])),
                linear_RGB_RMSE=float(np.sqrt(np.mean(rgb[mask] ** 2))))


def analyze(output, d3=None):
    generation_path = checked(REFERENCE / 'generation.json', GENERATION)
    generation = read(generation_path)
    views = {}
    for c in [*CAMERAS[:4], 34]:
        name = f'view-{c}.npz'
        views[c] = numeric(checked(REFERENCE / name, generation['outputs'][name]['sha256']))
    view = views[34]
    valid = view['shader_valid']
    points, normals, faces = (view[key][valid] for key in ('position', 'normals', 'triangle'))
    counts, same_face = geometric_coverage(points, normals, faces, [views[c] for c in CAMERAS[:4]])
    facing = (normals * view['outgoing'][valid]).sum(-1)
    focal = next(row['calibration']['focal_px'] for row in generation['views'] if row['camera'] == 34)
    width = view['hit_mask'].shape[1]
    pixels = view['pixel_indices'][valid]
    records = read(ROOT / 'docs/results/donut_kurt_recovery_v2/artifact-manifest.json')['records']
    report = dict(reference_generation_sha256=GENERATION, camera=34, fitting_cameras=CAMERAS[:4],
                  observations=len(points), radius_native_units=list(RADII), minimum_normal_cosine=float(MIN_NORMAL_COSINE),
                  median_camera34_pixel_footprint_native_units=float(np.median(view['depth'][valid] / focal)),
                  coverage_counts_by_radius=[np.bincount(a, minlength=5).tolist() for a in counts.T],
                  same_triangle_camera_counts=np.bincount(same_face, minlength=5).tolist(), arms={},
                  interpretation='Nearby compatible normal hits and same-triangle samples are coverage proxies, '
                                 'not exact visibility, photometric conditioning or causal attribution.')
    arm_paths = {'D1': BATCH / 'D1', 'D2': BATCH / 'D2'}
    if d3 is not None:
        arm_paths['D3'] = d3
    for arm, base in arm_paths.items():
        receipt_path = base / 'recovery.json'
        if arm != 'D3':
            checked(receipt_path, records[str(receipt_path)]['sha256'])
        receipt = read(receipt_path)
        if (receipt['arm'] != arm or receipt['status'] != 'complete' or
                receipt['benchmark_sha256'] != GENERATION or receipt['optimizer_returns'] != 10000 or
                not receipt['scientific_work_complete'] or not receipt['cleanup_verified']):
            raise ValueError('Expected a completed frozen recovery arm')
        def load(name):
            return numeric(checked(base / name, receipt['outputs'][name]['sha256']))
        errors = load('step-10000/camera-34-errors.npz')
        materials = load('step-10000/camera-34/materials.npz')
        if not np.array_equal(np.flatnonzero(materials['hit_mask']), pixels):
            raise ValueError('Camera 34 pixel ordering differs')
        rgb = (materials['linear_rgb'] - materials['linear_gt']).reshape(-1, 3)[pixels].astype(float)
        q, angle = errors['Q_relative'], errors['orientation_degrees']
        arm_report = dict(recovery_report_sha256=digest(receipt_path))
        for label, bins in [('within_001_camera_count', counts[:, 1]), ('same_triangle_camera_count', same_face)]:
            arm_report[label] = {str(i): error_summary(q, angle, rgb, bins == i) for i in range(5)}
        arm_report['radius_sensitivity'] = {
            str(r): {'zero': error_summary(q, angle, rgb, counts[:, i] == 0),
                     'one_or_more': error_summary(q, angle, rgb, counts[:, i] > 0)}
            for i, r in enumerate(RADII)
        }
        arm_report['facing_cross_tab_at_001'] = {
            f'{support}_{band}': error_summary(q, angle, rgb, mask & (facing >= a) & (facing < b))
            for support, mask in [('unsupported', counts[:, 1] == 0), ('supported', counts[:, 1] > 0)]
            for band, a, b in [('grazing', 0, .25), ('oblique', .25, .6), ('near_frontal', .6, 1.001)]
        }
        report['arms'][arm] = arm_report
    output.mkdir(parents=True)
    with (output / 'camera34-coverage.json').open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')
    np.savez_compressed(output / 'camera34-coverage.npz', pixel_xy=np.stack([pixels % width, pixels // width], -1),
                        nearby_camera_counts=counts, same_triangle_camera_count=same_face, facing_cosine=facing)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d3', type=Path, help='Optional completed D3 arm directory')
    args = parser.parse_args()
    result = analyze(args.output, args.d3)
    print(json.dumps(result, indent=2))
