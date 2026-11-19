"""Geometry-only neighborhoods and the additive normal-transported Q term."""

import numpy as np

from infrastructure.signatures import require
from infrastructure.surface_q_regularization import surface_pairs, transported_q_loss, world_q

PROTOCOL = 'donut-kurt-surface-wide-q-d6-v1'
POLICY = dict(coefficient=.05, reference_distance=.02, max_surface_distance=.02,
              max_face_dihedral_degrees=15., rng_seed=26061, batch_pairs=128,
              diagnostic_pairs=1024, pair_sampling='uniform edge permutations without replacement',
              expected_mesh_faces=18432, expected_pairs=27648)
LIMITS = dict(wall_seconds=3720, component_reverse_sweeps=30, surface_q_batches=10000,
              surface_q_pairs=1280000, surface_q_predictor_calls=20000,
              surface_q_predictor_points=2560000)


def neighborhoods(geometry, policy):
    """Reserve the existing D2 diagnostic edges and interpolate centroid normals."""
    vertices, triangles = geometry['vertices_native'], geometry['triangles']
    pairs = surface_pairs(vertices, triangles, max_distance=policy['max_surface_distance'],
                          max_angle_degrees=policy['max_face_dihedral_degrees'])
    faces = pairs['face_pairs']
    count = policy['diagnostic_pairs']
    require(0 < count < len(faces), 'Insufficient disjoint mesh neighborhoods')
    diagnostic = np.linspace(0, len(faces)-1, count).astype(np.int64)
    mask = np.ones(len(faces), dtype=bool)
    mask[diagnostic] = False
    train = np.flatnonzero(mask)
    require(np.array_equal(np.unique(faces[train]), np.arange(len(triangles))),
            'Training neighborhood domain does not cover every oracle face')
    smooth = geometry['shading_normals_world'][geometry['triangle_loops']].mean(axis=1)
    length = np.linalg.norm(smooth, axis=-1, keepdims=True)
    require(np.isfinite(smooth).all() and (length > 1e-12).all(), 'Invalid smooth centroid normals')
    smooth = smooth / length
    n0, n1 = smooth[faces[:,0]], smooth[faces[:,1]]
    require(((n0 * pairs['normals0']).sum(-1) > 0).all() and
            ((n1 * pairs['normals1']).sum(-1) > 0).all() and ((n0 * n1).sum(-1) > -1 + 1e-6).all(),
            'Invalid centroid normal orientation or ambiguous transport')
    xyz = vertices[triangles]
    area = np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0]), axis=-1) * .5
    return dict(face_pairs=faces, distance=pairs['distance'], points0=pairs['points0'],
                points1=pairs['points1'], normals0=n0, normals1=n1, face_area=area,
                training_ids=train, diagnostic_ids=diagnostic)


class NeighborhoodSampler:
    """An independent CPU stream; diagnostics and photometric samples never draw here."""

    def __init__(self, training_ids, pair_count, seed):
        self.training_ids = training_ids
        self.rng = np.random.RandomState(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.counts = np.zeros(pair_count, dtype=np.int64)

    def draw(self, count):
        require(0 < count <= len(self.training_ids), 'Invalid surface sampling budget')
        chunks = []
        remaining = count
        while remaining:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.training_ids)
                self.cursor = 0
            used = min(remaining, len(self.order) - self.cursor)
            chunks.append(self.order[self.cursor:self.cursor + used])
            self.cursor += used
            remaining -= used
        ids = np.concatenate(chunks)
        np.add.at(self.counts, ids, 1)
        return ids

    def checkpoint(self):
        return dict(rng=self.rng.get_state(), order=self.order, cursor=self.cursor, counts=self.counts)


def fitting_face_counts(data, fitting_cameras, face_count):
    """Coverage accounting only. No role in selecting or weighting neighborhoods."""
    counts = np.zeros(face_count, dtype=np.int64)
    for camera in fitting_cameras:
        d = data[camera]
        indices = np.unique(d['triangle'][d['shader_valid']])
        require(indices.dtype.kind in 'iu' and (indices >= 0).all() and
                (indices < face_count).all(), 'Invalid fitting face IDs')
        counts[indices] += 1
    return counts


def coverage_summary(domain, counts, fitting_counts):
    faces = domain['face_pairs']
    seen = counts > 0
    visited_faces = np.zeros(len(fitting_counts), dtype=bool)
    visited_faces[faces[seen].ravel()] = True
    result = dict(pair_draws=int(counts.sum()), unique_pairs=int(seen.sum()),
                  visited_faces=int(visited_faces.sum()),
                  surface_area_fraction=float(domain['face_area'][visited_faces].sum() / domain['face_area'].sum()))
    for key, mask in [('sampled_in_fit', fitting_counts > 0), ('unsampled_in_fit', fitting_counts == 0)]:
        result[key] = dict(faces=int(mask.sum()), visited_faces=int((visited_faces & mask).sum()),
                           area=float(domain['face_area'][mask].sum()),
                           visited_area=float(domain['face_area'][visited_faces & mask].sum()))
    return result


def surface_q_term(shader, batch, policy, torch, *, distance_weighting=None):
    """Use the Q-only branch of surface_world_q_v1 without adding its scalar loss."""
    q = []
    for suffix in ('0', '1'):
        mx, my, _, _, rotation, _ = shader.predict_anisotropic_components(batch['points'+suffix])
        q.append(world_q(mx, my, rotation, batch['normals'+suffix], torch))
    weights = torch.ones_like(batch['distance'])
    _, per_pair = transported_q_loss(q[0], q[1], batch['normals0'], batch['normals1'], weights, torch)
    multiplier = (policy['reference_distance'] / batch['distance']).square()
    if distance_weighting is not None:
        bounded = multiplier.clamp_max(distance_weighting['cap']) * distance_weighting['scale']
        normalized = per_pair * bounded
        return policy['coefficient'] * normalized.mean(), normalized.detach() * (policy['coefficient']/len(normalized))
    normalized = per_pair * multiplier
    return policy['coefficient'] * normalized.mean()
