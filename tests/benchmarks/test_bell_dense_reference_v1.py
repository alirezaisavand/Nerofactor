import numpy as np
import pytest

from benchmarks import bell_dense_reference_v1 as d
from experiments import bell_dense_reference_v1 as ref
from infrastructure.signatures import SignatureError


def test_maximin_continues_training_only_and_breaks_ties_by_id():
    # Identity world-to-camera rotations with centers at the negative translations.
    centers = np.array([[1,0,0], [0,-1,0], [0,1,0], [-1,0,0], [0,0,1], [1,1,0]])
    poses = np.zeros((len(centers),3,4))
    poses[:,:,:3] = np.eye(3)
    poses[:,:,3] = -centers
    split = dict(train=['2','0','1','5'], validation=['3'], held_view_nvs=['4'])
    new, angles, measured = d.select(poses, split, [0], 2)
    assert new == [1,2]
    np.testing.assert_array_equal(angles, [90,90])
    np.testing.assert_array_equal(measured, centers)
    with pytest.raises(SignatureError, match='overlapping'):
        d.select(poses, dict(split, train=['0','2','3','5']), [0], 1)
    with pytest.raises(SignatureError, match='overlapping'):
        d.select(poses, split, [3], 1)


def test_support_filters_distance_and_normals_and_preserves_sparsity():
    def observation(p, n, lit):
        return dict(position=np.asarray(p, dtype=float), normals=np.asarray(n, dtype=float),
                    shader_valid=np.ones(len(p),bool), highlight_mask=np.asarray(lit,bool))

    q = [[0,0,0],[1,0,0]]
    n = [[0,0,1],[0,0,1]]
    views = {0: observation(q, n, [False, True]),
             1: observation([[.001,0,0],[1,0,0]], [[0,0,1],[0,0,-1]], [True,True]),
             2: observation([[0,0,0],[1.2,0,0]], n, [True,True]),
             3: observation(q, n, [False,False])}
    centers = np.array([[0,0,2],[2,0,0],[0,2,0],[0,0,-2]])
    result = d.support(views, 3, [0,1,2], centers, True)
    np.testing.assert_array_equal(result['nearby_fitting_count'], [3,1])
    np.testing.assert_array_equal(result['nearby_highlight_count'], [2,1])
    np.testing.assert_array_equal(result['max_outgoing_angle_degrees'], [90,0])


def test_copy_sparse_preserves_bytes_and_requires_fresh_target(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'SPARSE_FILES', ['view-0.npz'])
    src = tmp_path/'old'
    src.mkdir()
    out = tmp_path/'new'
    out.mkdir()
    path = src/'view-0.npz'
    path.write_bytes(b'old scientific bytes')
    records = {path.name: dict(bytes=path.stat().st_size, sha256=d.b.digest(path))}
    ref.copy_sparse(src,out,records)
    assert (out/path.name).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        ref.copy_sparse(src,out,records)
    other = tmp_path/'different'
    other.mkdir()
    path.write_bytes(b'other')
    with pytest.raises(SignatureError, match='Frozen artifact'):
        ref.copy_sparse(src,other,records)
    assert not list(other.iterdir())


def test_reuses_original_baseline_and_refinement_for_appended_views(tmp_path, monkeypatch):
    # Stub only numerical quadrature. Exercise real generation, exact array
    # assembly, physical checks, RGB previews and offline replay.
    camera = 127
    n = 8
    color = np.tile([.2,.3,.4],(n,1))
    metallic = np.full((n,1),.5)
    diffuse = color*(1-metallic)
    specular = np.full_like(diffuse,.05)
    reflectance = diffuse+specular
    normal = np.tile([0.,0.,1.],(n,1))
    data = dict(shader_valid=np.ones(n,bool), pixel_indices=np.arange(n),
                color=color, metallic=metallic, normals=normal, outgoing=normal)
    hi = np.stack((diffuse,specular,reflectance),axis=1)
    lo = hi.copy()
    lo[0,1] += .005
    lo[0,2] = lo[0,0]+lo[0,1]
    monkeypatch.setattr(d,'NEW',[camera])
    monkeypatch.setattr(d,'COUNTS',{camera:n})
    monkeypatch.setattr(ref,'baseline',lambda view,c,*_:
        (dict(view, linear_components=hi, coarse_components=lo, visible_counts=np.full(n,8),
              reflectance=reflectance, coarse_reflectance=reflectance, reflectance_diffuse=diffuse,
              reflectance_specular=specular, incident_reflectance=reflectance, coarse_incident_reflectance=reflectance),
         np.full(n,4)))

    def fixed_patch(view, indices, _):
        np.testing.assert_array_equal(indices,[0])
        return dict(pixels=np.array([0]),grids=np.asarray(refinement_grids),
                    components=np.stack([hi[:1]]*2), incident_reflectance=np.stack([reflectance[:1]]*2),
                    visible_counts=np.array([[16],[32]]))

    refinement_grids = ref.refinement.HIGH
    monkeypatch.setattr(ref,'capture',fixed_patch)
    report = dict(refined_points=0,views=[])
    ref.generate_new(tmp_path, {0: data, camera: data}, None, None, report, lambda: None)
    assert report['refined_points'] == 1 and report['views'][0]['selected_pixels']==[0]
    assert {p.name for p in tmp_path.iterdir()}==ref.view_outputs(camera,[0])
    verified, summary = ref.replay_view(tmp_path, report['views'][0], data, lambda: None)
    np.testing.assert_array_equal(verified['linear_components'],hi)
    assert summary['refined_pixels']==[0]
    with pytest.raises(SignatureError, match='order differs'):
        ref.replay_view(tmp_path,dict(report['views'][0],selected_pixels=[1]),data,lambda: None)


def test_work_and_exposure_use_appended_hits_only():
    assert d.LIMITS['primary_rays']==0
    assert d.LIMITS['BRDF_point_directions']==d.LIMITS['shadow_directions']==16_349_265_920
    assert d.training_projection()['fitting_observations']==65_419
    assert d.training_projection()['held_observations']==7_082
    assert d.training_projection()['effective_passes']==5_120_000/65_419
    assert d.training_projection()['full_batches_per_permutation']==127
