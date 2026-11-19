import numpy as np
import pytest

from benchmarks import bell_heterogeneous_training_v1 as bell
from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from infrastructure.signatures import SignatureError


def point(camera, highlight=False):
    return dict(position=np.array([[.001*camera,0.,0.]]), normals=np.array([[0.,0.,1.]]),
                outgoing=np.array([[0.,0.,1.]]), shader_valid=np.array([True]),
                linear_components=np.full((1,3,3),camera), highlight_mask=np.array([highlight]),
                Q_world='evaluation only', material_region='evaluation only')


def test_bell_fit_pool_keeps_only_geometry_and_observations():
    views = {c:point(c) for c in bell.PANEL}
    pooled = bell.fit_pool(views)
    assert set(pooled) == set(bell.FIT_KEYS)
    np.testing.assert_array_equal(pooled['linear_components'][:,2,0],bell.SPLITS['fitting'])


def test_fitting_support_uses_bell_ids_and_excludes_same_view():
    views = {c:point(0,highlight=True) for c in bell.PANEL}
    for camera in bell.PANEL:
        support,highlight = metrics.fitting_support(views,camera,fitting_cameras=bell.SPLITS['fitting'])
        count = 3 if camera in bell.SPLITS['fitting'] else 4
        assert support.tolist()==highlight.tolist()==[count]


def test_boundary_probes_are_same_face_opposite_regions():
    vertices = np.array([[-.108,.493,0.],[-.090,.495,0.],[-.092,.506,0.],
                         [.03,.49,0.],[.036,.49,0.],[.033,.50,0.]])
    faces = np.array([[0,1,2],[3,4,5]])
    geo = dict(vertices_native=vertices,triangles=faces,bounds=np.array([[-1.,-1.,-1.],[1.,1.,1.]]))
    probes = bell.boundary_probes(geo)
    assert probes['face_ids'].tolist()==probes['plane_ids'].tolist()==[0]
    assert probes['positions'][0,0,0]>-.1>probes['positions'][1,0,0]
    np.testing.assert_array_equal(probes['normals'],[[0.,0.,-1.]])
    assert bell.boundary_score(probes,probes)['Q_jump_projection_median']==pytest.approx(1.)
    flat = {k:np.repeat(probes[k][:1],2,axis=0) for k in ('Q_world','widths_major_minor')}
    assert bell.boundary_score(probes,flat)['Q_jump_projection_median']==0.
    with pytest.raises(AssertionError):
        bell.check_probe_archive(probes,dict(probes,face_ids=np.array([1])))


def test_isotropic_observation_has_no_orientation_error_cohort():
    widths = np.array([[.22,.22]])
    q = np.diag([.22**2,.22**2,0.])[None]
    truth = dict(normals=np.array([[0.,0.,1.]]),axis=np.array([[1.,0.,0.]]),Q_world=q,
                 widths_major_minor=widths,color=np.array([[.3,.5,.1]]),metallic=np.array([[.75]]))
    pred = dict(mx=widths[:,:1],my=widths[:,1:],Q_world=q,axis=np.array([[0.,1.,0.]]),
                kd=truth['color'],metallic=truth['metallic'])
    errors,stats = bell.point_errors(pred,truth)
    assert errors['Q_relative'].tolist()==[0.]
    assert errors['orientation_eligible'].tolist()==[False] and stats['orientation_degrees'] is None


def test_sampler_capture_protects_reserved_edges():
    counts = np.array([0,128],dtype=np.int64)
    f = dict(sampler_snapshots={'1':bell.array_hash(counts)})
    domain = dict(diagnostic_ids=np.array([0]))
    bell.check_sampler_capture(f,domain,1,dict(pair_ids=np.array([0]),training_pair_counts=counts))
    with pytest.raises(SignatureError):
        bell.check_sampler_capture(f,domain,1,dict(pair_ids=np.array([0]),training_pair_counts=counts[::-1]))


def test_batch_caps_include_only_the_extra_predictor_points():
    budget = [bell.planned_counters(arm,2474) for arm in bell.ARMS]
    assert sum(w['shader_calls'] for w in budget)==49924
    assert sum(w['boundary_predictor_calls'] for w in budget)==48
    assert sum(w['boundary_predictor_points'] for w in budget)==118752
    assert budget[2]==budget[3]
