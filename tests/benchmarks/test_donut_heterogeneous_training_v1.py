from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import donut_heterogeneous_metrics_v1 as m
from benchmarks import donut_heterogeneous_training_v1 as h
from benchmarks import donut_heterogeneous_v1 as ref
from infrastructure.signatures import SignatureError


def test_exact_sequence_and_historical_work():
    assert h.ARMS==('D2','D6','D8-C','D8-T')
    expected=[12288,12288,12290,12290]
    assert [h.planned_counters(a)['shader_calls'] for a in h.ARMS]==expected
    assert sum(h.planned_counters(a)['shadow_rays'] for a in h.ARMS)==38136004608
    assert sum(h.planned_counters(a)['component_reverse_sweeps'] for a in h.ARMS)==114
    for arm in h.ARMS:
        assert all(v<=h.accepted_limits(arm)[k] for k,v in h.planned_counters(arm).items())


def test_admitted_data_alone_cannot_start_training(monkeypatch,tmp_path):
    class UnexpectedFilesystem:
        def __call__(self,*a):
            pytest.fail('Runtime guard touched execution environment before rejecting authorization')
    freeze={'host_output':str(tmp_path/'batch'),'arms':{arm:{} for arm in h.ARMS}}
    p=tmp_path/'authorization.json'
    p.write_text('{"researcher_admitted_dataset": true, "training_authorized": false}')
    monkeypatch.setattr(h,'freeze',lambda *_:freeze)
    monkeypatch.setattr(h.os,'statvfs',UnexpectedFilesystem())
    args=SimpleNamespace(freeze=Path('unused'),freeze_sha256='',phase='D2',image_id=h.IMAGE,
                         authorization=p,authorization_sha256=h.digest(p),commit='s')
    with pytest.raises(SignatureError,match='Exact separate four-arm'):
        h.runtime(args)


def test_fitting_pool_never_reads_material_labels_or_held_out():
    class OnlyObservations(dict):
        def __getitem__(self,key):
            if key not in (*h.FIT_KEYS,'shader_valid'):
                pytest.fail('Material label accessed inside fitting: '+key)
            return super().__getitem__(key)
    data={c:OnlyObservations(shader_valid=np.array([True,False]),
         **{k:np.array([c,-1]) for k in h.FIT_KEYS}) for c in ref.SPLITS['fitting']}
    pool=h.fit_pool(data)
    assert tuple(pool)==h.FIT_KEYS
    for array in pool.values():
        np.testing.assert_array_equal(array,ref.SPLITS['fitting'])


def examples():
    widths=np.array([[.45,.12],[.223,.214]])
    q=np.zeros((2,3,3))
    q[:,0,0]=widths[:,0]**2
    q[:,1,1]=widths[:,1]**2
    color=np.array([[.32,.44,.2],[.17,.24,.07]])
    metal=np.array([[.52],[.43]])
    truth=dict(widths_major_minor=widths,Q_world=q,color=color,metallic=metal,
               axis=np.tile([1.,0.,0.],(2,1)),normals=np.tile([0.,0.,1.],(2,1)))
    pred=dict(mx=widths[:,1:2],my=widths[:,0:1],Q_world=q,kd=color,metallic=metal,axis=-truth['axis'])
    return pred,truth


def test_physical_truth_per_point_and_joint_orientation_gap():
    pred,truth=examples()
    e,s=h.point_errors(pred,truth)
    for k in ('major_width','major_width_relative','minor_width','color_max','metallic','orientation_degrees','Q_relative'):
        assert not e[k].any()
    assert e['orientation_eligible'].tolist()==[True,False]
    assert s['orientation_ineligible']==1
    pred['mx']=np.array([[.444],[.14]])
    e,s=h.point_errors(pred,truth)
    assert not e['orientation_eligible'].any()
    assert s['orientation_degrees'] is None
    np.testing.assert_allclose(e['minor_width_relative'],[.324/.12,.074/.214])


def test_geometry_boundary_rejects_nearby_disconnected_faces():
    _,truth=examples()
    v=dict(truth,shader_valid=np.ones(3,dtype=bool),pixel_indices=np.arange(3),normals=np.tile([0.,0.,1.],(3,1)),
           position=np.array([[0,0,0],[.01,0,0],[.012,0,0]]),triangle=np.arange(3),
           material_region=np.array([0,1,0]),
           Q_world=truth['Q_world'][[0,1,0]],widths_major_minor=truth['widths_major_minor'][[0,1,0]])
    adjacency=m.mesh_adjacency(np.array([[0,1,2],[1,2,3],[4,5,6]]))
    pairs=m.boundary_pairs(v,adjacency)
    assert pairs.tolist()==[[0,1]]
    recovered=m.boundary_score(v,pairs,v['Q_world'],v['widths_major_minor'])
    assert recovered['Q_jump_magnitude_ratio_median']==1
    assert recovered['Q_jump_projection_median']==pytest.approx(1)
    assert recovered['Q_jump_relative_error_mean']==0
    erased=m.boundary_score(v,pairs,np.repeat(v['Q_world'][0:1],3,axis=0),
                             np.repeat(v['widths_major_minor'][0:1],3,axis=0))
    assert erased['Q_jump_magnitude_ratio_median']==0
    assert erased['Q_jump_relative_error_mean']==1


def test_area_weighted_moments_follow_area_not_observation_count():
    pred,truth=examples()
    e,_=h.point_errors(pred,truth)
    e['Q_relative']=np.array([.1,.3])
    e.update(major_predict=truth['widths_major_minor'][:,0],minor_predict=truth['widths_major_minor'][:,1])
    rgb=np.tile([.2,.2,.2],(2,1))
    result=m.score(e,rgb,rgb+.1,np.array([.75,.25]))
    assert result['Q_relative']['mean']==pytest.approx(.15)
    assert result['Q_relative']['median']==.1
    assert result['orientation_eligible']==1
    assert result['RGB_RMSE']==pytest.approx(.1)


def test_fitting_support_excludes_own_camera_and_incompatible_normals():
    def sample(normal, highlight):
        return dict(shader_valid=np.array([True]),position=np.zeros((1,3)),
                    normals=np.asarray([normal],dtype=float),highlight_mask=np.array([highlight]))
    views={0:sample([0,0,1],True),50:sample([0,0,1],False),
           1:sample([0,0,-1],True),45:sample([0,0,1],True)}
    n,hits=m.fitting_support(views,0)
    assert n.tolist()==[2] and hits.tolist()==[1]
