import numpy as np
import pytest

from benchmarks import bell_heterogeneous_v1 as b
from experiments.verify_bell_heterogeneous_v1 import compare_geometry_metadata
from infrastructure.signatures import SignatureError


BOUNDS = np.array([[-1., -2., -1.], [1., 2., 1.]])


def test_width_step_and_continuous_scalar_materials():
    points = np.array([[-.1-1e-8, 1., 0.], [-.1+1e-8, 1., 0.]])
    n = np.broadcast_to([0., 0., 1.], points.shape)
    field = b.material_field(points, n, BOUNDS)
    np.testing.assert_array_equal(field['material_region'], [1, 0])
    np.testing.assert_allclose(np.diff(field['widths_major_minor'], axis=0), [[.12, .058]], atol=5e-9)
    np.testing.assert_allclose(field['color'][0], field['color'][1], atol=5e-9, rtol=0)
    np.testing.assert_allclose(field['metallic'][0], field['metallic'][1], atol=5e-9, rtol=0)
    assert field['near_boundary'].all()


@pytest.mark.parametrize('cap', [True, False])
def test_exact_isotropy_and_tangent_plane_at_degeneracy(cap):
    grain = b.unit(np.array(b.FIELD['grain']))
    n = np.array([[1., 0., 0.] if cap else grain])
    point = np.array([[.95 if cap else 0., 0., 0.]])
    field = b.material_field(point, n, BOUNDS)
    widths = field['widths_major_minor'][0]
    assert widths[0] == widths[1]
    assert field['exactly_isotropic'][0] and field['nearly_isotropic'][0]
    assert not field['orientation_defined'][0] and not field['orientation_eligible'][0]
    np.testing.assert_allclose(field['Q_world'][0], widths[0]**2*(np.eye(3)-n.T@n), atol=2e-16)
    np.testing.assert_allclose(field['axis']@n.T, 0, atol=2e-15)


def test_face_index_barycentrics_and_deterministic_edge_tie():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces, n, area = b.face_data(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
    points = np.array([[.8, .3, 0.], [.3, .8, 0.], [.5, .5, 0.]])
    raw = np.broadcast_to([0., 0., 1.], points.shape)
    chosen, barycentric, residual, alternatives = b.attribute_faces(points, raw, faces, n, area)
    np.testing.assert_array_equal(chosen, [0, 1, 0])
    np.testing.assert_array_equal(alternatives, [0, 0, 1])
    np.testing.assert_allclose(np.einsum('ni,nij->nj', barycentric, faces[chosen]), points)
    np.testing.assert_array_equal(residual, 0.)
    with pytest.raises(SignatureError, match='No normal-compatible'):
        b.attribute_faces(points, -raw, faces, n, area)


def test_faceted_transport_retains_negated_normals_and_partial_domain(monkeypatch):
    monkeypatch.setattr(b, 'D6', dict(b.D6, diagnostic_pairs=1, batch_pairs=1))
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                         [2., 0., 0.], [2., 1., .10], [3., 0., 0.], [3., 1., 0.], [4., 1., 0.]])*.005
    triangles = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2], [6, 7, 8]])
    _, raw, area = b.face_data(vertices, triangles)
    d, reasons = b.neighborhoods(dict(vertices_native=vertices, triangles=triangles, area=area))
    np.testing.assert_allclose(d['normals0'], -raw[d['face_pairs'][:, 0]], atol=1e-15)
    np.testing.assert_allclose(d['normals1'], -raw[d['face_pairs'][:, 1]], atol=1e-15)
    assert np.any(np.linalg.norm(d['normals0']-d['normals1'], axis=1) > .01)
    assert 4 not in d['face_pairs']
    assert reasons['excluded_non_two_face_edges'] > 0
    assert not np.intersect1d(d['diagnostic_ids'], d['training_ids']).size


def test_work_counts_cover_full_panel_and_refinement():
    base = b.counters(b.POINTS, 0)
    cap = b.counters(b.POINTS, b.MAX_TOTAL)
    assert b.POINTS == 24655 and base['primary_rays'] == cap['primary_rays'] == 0
    assert base['shadow_directions'] == b.POINTS*(256**2+512**2)
    assert cap['shadow_directions'] == cap['BRDF_point_directions'] == 8750039040
    assert base['slope_nodes_returned'] == cap['slope_nodes_attempted'] == 2019737600
    with pytest.raises(SignatureError, match='refinement budget'):
        b.counters(b.POINTS, b.MAX_TOTAL+1)


def test_geometry_replay_preserves_integer_counts():
    expected = {'geometry': {'count': 153, 'error': 1e-7}}
    compare_geometry_metadata({'geometry': {'count': 153, 'error': 1e-7+1e-15}}, expected)
    with pytest.raises(SignatureError, match='count differs'):
        compare_geometry_metadata({'geometry': {'count': 152, 'error': 1e-7}}, expected)
    with pytest.raises(AssertionError, match='Not equal'):
        compare_geometry_metadata({'geometry': {'count': 153, 'error': 1e-5}}, expected)
