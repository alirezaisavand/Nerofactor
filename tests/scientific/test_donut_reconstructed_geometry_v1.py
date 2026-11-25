import numpy as np
import pytest

from benchmarks import donut_reconstructed_geometry_v1 as g
from benchmarks.bell_geometry_e11 import intersect
from benchmarks.donut_heterogeneous_v1 import material_field
from experiments import donut_reconstructed_geometry_v1 as audit
from experiments.verify_donut_reconstructed_geometry_v1 import aggregate, metrics


def test_exact_projection_interior_edge_vertex():
    tri = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]]])
    normal = np.array([[0., 0., 1.]])
    for point, expected in (([.2, .25, 1.], [.2, .25, 0.]),
                            ([.4, -2., 1.], [.4, 0., 0.]),
                            ([-.4, -.3, 0.], [0., 0., 0.]),
                            ([2., 1., 0.], [1.6, .2, 0.])):
        projected, bary = g.closest_points(np.array(point), tri, normal)
        np.testing.assert_allclose(projected[0], expected)
        np.testing.assert_allclose(bary.sum(1), 1)
        assert (bary >= 0).all()


def test_aabb_nearest_and_charge_precede_exact_projection(monkeypatch):
    v = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., .05]])
    surface = g.Surface(v, np.array([[0, 1, 2], [3, 3, 3]]))
    charges = []
    charge = lambda key, count: charges.append((key, count))
    hit = surface.nearest(np.array([.004, .002, .001]), charge)
    assert hit[0] == 0 and hit[3] == pytest.approx(.001)
    assert charges == [('nearest_searches', 1), ('projection_candidates', 1)]
    assert surface.nearest(np.array([0., 0., .05]), charge) is None
    def stop(key, count):
        if key == 'projection_candidates':
            raise RuntimeError('cap')
    monkeypatch.setattr(g, 'closest_points', lambda *args: pytest.fail('projection ran after cap failure'))
    with pytest.raises(RuntimeError, match='cap'):
        surface.nearest(np.array([.004, .002, .001]), stop)


def test_cpu_closest_hit_agrees_with_existing_reference_and_keeps_tied_face():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 1.]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    surface = g.Surface(vertices, faces)
    direction = np.array([0., 0., -1.])
    origin = np.array([.2, .2, 1.])
    native = intersect(origin, direction, vertices, faces)
    hit = surface.independent_hit(origin, direction)
    assert hit[:2] == (native['face'], native['depth'])
    origin[1] = 0.
    result = dict(hit=np.array([True]), position=np.array([[.2, 0, 0.]]),
                  raw=surface.normal[[1]], depth=np.ones(1))
    witness = audit.cpu_witness(surface, result, origin, direction, 0)
    assert witness['passed'] and witness['normal_witness_face'] == 1 and witness['shared_edge']
    result['raw'] = -result['raw']
    assert not audit.cpu_witness(surface, result, origin, direction, 0)['passed']
    assert surface.independent_hit(np.array([3., 0, 1.]), direction) is None


def test_correspondence_keeps_unmatched_and_separates_field_and_frame(monkeypatch):
    v = np.array([[.30, .10, 0.], [.50, .10, 0.], [.30, .20, 0.]])
    normal = np.tile([0., 0., 1.], (3, 1))
    triangles = np.array([[0, 1, 2]])
    geometry = dict(shading_normals_world=normal, triangle_loops=triangles)
    surfaces = dict(oracle=g.Surface(v, triangles), reconstruction=g.Surface(v+[0., 0., .001], triangles[:, ::-1]))
    positions = np.array([[.4, .12, 0.], [.7, .12, 0.]], dtype='f4').astype(float)
    raw = dict(oracle_hit=np.array([True, False]), reconstruction_hit=np.ones(2, bool),
               oracle_position=positions, reconstruction_position=positions+[0., 0., .001],
               oracle_depth=np.array([1., 10.]), reconstruction_depth=np.full(2, .999),
               oracle_smooth_normal=np.tile([0., 0., 1.], (2, 1)), reconstruction_normal=np.tile([0., 0., 1.], (2, 1)),
               reconstruction_normal_valid=np.ones(2, bool), direction=np.tile([0., 0., -1.], (2, 1)),
               origin=(positions+[0., 0., 1.]).astype('f4'), near_degenerate=np.zeros(2, bool))
    view = dict(pixel_indices=np.array([0]), **material_field(positions[:1], normal[:1], .33))
    monkeypatch.setattr(audit, 'trace', lambda *args: pytest.fail('coincident ray must be reused'))
    arrays = {}
    audit.correspondence(arrays, raw, surfaces, geometry, view, .33, None, lambda *args: None, None)
    np.testing.assert_array_equal(arrays['valid'], [True, False])
    assert arrays['no_oracle_candidate'][1]
    np.testing.assert_allclose(arrays['transported_world_Q'][0], view['Q_world'][0], atol=2e-15)
    stats = aggregate([metrics(raw, arrays)])
    assert stats['silhouette_IoU'] == .5 and stats['coverage_common'] == 1 and stats['coverage_reconstruction_only'] == 0


def test_normal_rotation_preserves_eigenvalues_and_frame_sign():
    source = np.array([[0., 0., 1.]])
    target = np.array([[0., .5, np.sqrt(.75)]])
    Q = np.diag([.12, .03, 0.])[None]
    R = g.minimal_rotation(source, target)
    Qt = g.rotate_tensor(Q, R)
    np.testing.assert_allclose(np.einsum('nij,nj->ni', R, source), target)
    np.testing.assert_allclose(np.linalg.eigvalsh(Qt), np.linalg.eigvalsh(Q), atol=1e-15)
    np.testing.assert_allclose(np.einsum('nij,nj->ni', Qt, target), 0, atol=1e-15)
