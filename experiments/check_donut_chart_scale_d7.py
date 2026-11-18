"""Recompute the pre-execution D7 scale check from D2 fitting arrays only."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.donut_recovery_v1 import numeric
from benchmarks.donut_surface_chart_d7 import POLICY, chart_difference
from benchmarks.donut_surface_q_d6 import neighborhoods
from infrastructure.stage_sources import digest
from infrastructure.surface_q_regularization import transported_q_loss, world_q


def check():
    import torch

    baseline = ROOT / 'docs/evidence/donut_surface_q_d6_preparation_v1/geometry-and-scale.json'
    prep = json.loads(baseline.read_text())
    f = ROOT / 'docs/protocols/DONUT_KURT_SURFACE_Q_D6_V1.freeze.json'
    assert digest(f) == prep['freeze_sha256']
    freeze = json.loads(f.read_text())
    definition = freeze['surface_q_training']
    source = freeze['inputs']['oracle-geometry-and-frames.npz']
    geometry_path = Path(source['path'])
    assert digest(geometry_path) == source['sha256']
    geometry = numeric(geometry_path)
    domain = neighborhoods(geometry, POLICY)
    centers = geometry['vertices_native'][geometry['triangles']].mean(axis=1)
    dataset = Path(definition['benchmark_host_path'])
    generation = dataset / 'generation.json'
    assert digest(generation) == definition['benchmark_generation_sha256']
    admitted = json.loads(generation.read_text())
    fitting = freeze['splits']['fitting']
    material_files = prep['coefficient_check']['fitting_material_prediction_identities']
    positions, normals, faces, coordinates, saved_q = [], [], [], [], []
    for camera, item in zip(fitting, material_files):
        view = dataset / ('view-%d.npz' % camera)
        assert digest(view) == admitted['outputs'][view.name]['sha256']
        path = Path(item['path'])
        assert path.parent.name == 'camera-%d' % camera and digest(path) == item['sha256']
        maps, data = numeric(path), numeric(view)
        valid = data['shader_valid']
        pixel = data['pixel_indices'][valid]
        assert np.array_equal(np.flatnonzero(maps['hit_mask']), pixel)
        positions.append(data['position'][valid])
        normals.append(data['normals'][valid])
        faces.append(data['triangle'][valid])
        widths = [maps[k].reshape(-1)[pixel] for k in ('mx', 'my')]
        r = maps['rotation'].reshape(-1, 2)[pixel]
        coordinates.append(np.stack([*widths, .5 * np.arctan2(r[:, 1], r[:, 0])], axis=-1))
        saved_q.append(maps['Q_world'].reshape(-1, 3, 3)[pixel])
    p, n, face, chart, q = map(np.concatenate, (positions, normals, faces, coordinates, saved_q))
    squared_distance = np.sum((p - centers[face]) ** 2, axis=-1)
    order = np.lexsort((np.arange(len(face)), squared_distance, face))
    ids, first = np.unique(face[order], return_index=True)
    chosen = np.full(len(centers), -1)
    chosen[ids] = order[first]
    endpoints = chosen[domain['face_pairs'][domain['training_ids']]]
    supported = (endpoints >= 0).all(axis=1)
    endpoints = endpoints[supported]
    assert len(endpoints) == prep['coefficient_check']['fitting_pairs']
    scale = torch.from_numpy((POLICY['reference_distance'] / domain['distance'][domain['training_ids']][supported]) ** 2)

    normals = torch.tensor(n[endpoints], dtype=torch.float64)
    chart = chart[endpoints]
    coords = torch.tensor(chart, dtype=torch.float64, requires_grad=True)
    a, b = coords.unbind(1)
    vectors = [torch.stack([torch.cos(2*z[:, 2]), torch.sin(2*z[:, 2])], dim=-1) for z in (a, b)]
    tensors = [world_q(z[:, :1], z[:, 1:2], r, normals[:, i], torch)
               for i, (z, r) in enumerate(zip((a, b), vectors))]
    q_difference = float(np.max(abs(torch.stack(tensors, 1).detach().numpy() - q[endpoints])))
    assert q_difference < 1e-6
    _, physical = transported_q_loss(*tensors, normals[:, 0], normals[:, 1], torch.ones_like(scale), torch)
    observed_gap = float(np.median(abs(chart[:, :, 0] ** 2 - chart[:, :, 1] ** 2)))
    assert round(observed_gap ** 2 / 2, 5) == POLICY['angle_weight']
    local = chart_difference(a[:, :2], b[:, :2], *vectors, POLICY['angle_weight'], torch)
    result = dict(fitting_cameras=fitting, fitting_pairs=len(endpoints), material_files=material_files,
                  fitting_state=10000, observed_squared_width_gap_median=observed_gap,
                  orientation_weight=POLICY['angle_weight'], Q_from_chart_max_absolute_difference=q_difference,
                  coefficients=dict(D6=freeze['surface_q_training']['spatial_policy']['coefficient'],
                                    D7=POLICY['coefficient']))
    for name, loss in [('D6', physical), ('D7', local)]:
        weighted = result['coefficients'][name] * loss * scale
        values = weighted.detach().numpy()
        # Gradients of each pair wrt its own saved width and angle coordinates.
        grad = torch.autograd.grad(weighted.sum(), coords, retain_graph=name=='D6')[0]
        result[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                            p95=float(np.quantile(values, .95)),
                            local_coordinate_gradient_RMS=float(grad.square().sum((1, 2)).mean().sqrt()),
                            local_coordinate_gradient_RMS_per_component=grad.square().sum(1).mean(0).sqrt().tolist())
    result['D7_to_D6_ratios'] = {key: result['D7'][key] / result['D6'][key]
                                 for key in ('mean', 'local_coordinate_gradient_RMS')}
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path)
    path = cli.parse_args().output
    result = check()
    with path.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')
