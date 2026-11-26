"""Exact mesh-degeneracy accounting without repairing or tracing any mesh."""
import numpy as np
from experiments.analyze_bell_geometry_gate10_v4_failure import face_diagnostics


def test_repeated_indices_and_equal_coordinates_have_distinct_accounting():
    vertices=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[1.,0,0]],np.float32)
    faces=np.array([[0,1,2],[0,1,1],[0,1,3]])
    result,rows=face_diagnostics(vertices,faces)
    assert result['zero_area_faces']==2
    assert result['repeated_index_faces']==1
    assert result['distinct_index_zero_area_faces']==1
    assert result['coincident_coordinate_zero_area_faces']==2
    assert result['minimum_positive_area']==.5
    assert [r['face'] for r in rows]==[1,2]
    assert np.array_equal(faces,[[0,1,2],[0,1,1],[0,1,3]])


def test_float32_serialization_can_collapse_distinct_float64_points():
    vertices=np.array([[.5,0,0],[.5+1e-9,0,0],[.5,.1,0],[.4,0,0]],np.float64)
    faces=np.array([[0,1,2],[0,2,3]])
    before,_=face_diagnostics(vertices,faces)
    after,_=face_diagnostics(vertices.astype(np.float32),faces)
    assert before['zero_area_faces']==0 and after['zero_area_faces']==1
    assert after['distinct_index_zero_area_faces']==1
