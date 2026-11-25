from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import donut_matched_geometry_recovery_v1 as m
from experiments.train_donut_matched_geometry_v1 import Acquisition, trainer_source


def test_support_filters_positive_and_transfer_valid_in_source_order():
    view = dict(pixel_indices=np.array([0,1,3,5]),shader_valid=np.array([True,False,True,True]))
    c = dict(pixel=np.array([0,1,2,3,5]),label_rows=np.array([0,1,2,4]))
    normal = np.tile([0.,0.,1.],(6,1))
    normal[5,2] = -1
    primary = dict(reconstruction_normal=normal,direction=np.tile([0.,0.,-1.],(6,1)))
    np.testing.assert_array_equal(m.common_rows(view,c,primary),[0])


def test_fitting_pool_blocks_labels_and_heldout_and_records_actual_indices(tmp_path):
    instance = Acquisition.__new__(Acquisition)
    instance.record = None
    instance.args = SimpleNamespace(output=tmp_path)
    instance.f = {'support':{'fitting_observations':1024}}
    data = {}
    for c in m.PANEL:
        data[c] = dict(shader_valid=np.ones(256,bool), Q_world=np.full((256,3,3),c),
                       **{key:np.full((256,3),c) for key in m.FIT_KEYS})
    pool = instance.fit_pool(data)
    assert set(pool)==set(m.FIT_KEYS)
    assert not set(pool['normals'][:,0]) & set(m.SPLITS['held_out'])
    idx = next(m.bell.dense.index_batches(1024,6033,1,512))
    np.testing.assert_array_equal(pool['position'][idx],pool['normals'][idx])
    np.testing.assert_array_equal(np.fromfile(instance.record.path,dtype='<u4'),idx)


def test_common_frame_preserves_width_eigenvalues():
    o = dict(normals=np.array([[1.,0.,0.]]))
    r = dict(normals=np.array([[0.,0.,1.]]))
    q = np.diag([.36,.16,0.])[None]
    aligned = m.common_frame(o,r,q)
    np.testing.assert_allclose(np.linalg.eigvalsh(aligned),np.linalg.eigvalsh(q),atol=1e-15)
    np.testing.assert_allclose(np.einsum('nij,nj->ni',aligned,o['normals']),0,atol=1e-15)


def test_work_and_overlays_keep_accepted_loop_and_separate_compatibility():
    for ambient in (False,True):
        code = trainer_source(ambient)
        compile(code,'<native Gate 4 overlay>','exec')
        assert 'optimizer.step()' in code and 'acquisition.compatibility_rgb(panel[0],ix)' in code
        assert "truth[d['pixel_indices']]=d['linear_components'][:,2]" in code
        assert 'boundary-predictions.npz' not in code
        assert ('physical_tensor_admission' in code)==ambient
        assert m.work(ambient)['shader_calls']==12288+2*ambient
        assert m.work(ambient)['shadow_rays']==9524158464+73728*ambient


def test_support_identity_fails_on_target_mismatch():
    view = dict(pixel_indices=np.array([0]),linear_components=np.ones((1,3,3)),outgoing=np.ones((1,3)))
    other = dict(view,linear_components=np.zeros((1,3,3)))
    with pytest.raises(AssertionError,match='Arrays are not equal'):
        m.support_identity(dict(oracle={0:view},reconstruction={0:other}))


def test_metadata_parity_from_accepted_inputs():
    f = m.definition(Path('/home/isavand/paper/evidence/donut-matched-geometry-gate4-metadata-only'),{})
    _,views,refs,coverage,_ = m.admit(f,m.paths(f),check_support=False)
    assert [len(views['oracle'][c]['pixel_indices']) for c in m.PANEL]==list(m.COUNTS.values())
    assert sum(int((~v['defined_reconstruction']).sum()) for v in coverage.values())==61
    assert sum(int((v['depth_outlier']&v['pool']).sum()) for v in coverage.values())==1
    assert m.support_identity(views)['fitting_observations']==15191
    for c in m.PANEL:
        np.testing.assert_array_equal(refs['oracle',c][1],views['oracle'][c]['linear_components'][:,2])
