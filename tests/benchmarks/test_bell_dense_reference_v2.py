import numpy as np
import pytest

from benchmarks import bell_dense_reference_v1 as v1
from benchmarks import bell_dense_reference_v2 as v2
from experiments import bell_dense_reference_v2 as run
from infrastructure.signatures import SignatureError


def fixture(n):
    color = np.tile([.2,.3,.4],(n,1))
    metal = np.full((n,1),.5)
    diffuse = color*(1-metal)
    specular = np.full_like(diffuse,.05)
    truth = diffuse+specular
    normals = np.tile([0.,0.,1.],(n,1))
    data = dict(shader_valid=np.ones(n,bool),pixel_indices=np.arange(n),color=color,metallic=metal,
                normals=normals,outgoing=normals)
    rgb = np.stack((diffuse,specular,truth),axis=1)
    coarse = rgb.copy()
    coarse[:,1] += .005
    coarse[:,2] = coarse[:,0]+coarse[:,1]
    view = dict(data,linear_components=rgb,coarse_components=coarse,visible_counts=np.full(n,8),
                reflectance=truth,coarse_reflectance=truth,reflectance_diffuse=diffuse,reflectance_specular=specular,
                incident_reflectance=truth,coarse_incident_reflectance=truth)
    return data,view,np.full(n,4)


def patch(view,row,pixel):
    return dict(pixels=np.array([pixel]),grids=np.array([1024,2048]),
                components=np.stack([view['linear_components'][row:row+1]]*2),
                incident_reflectance=np.stack([view['reflectance'][row:row+1]]*2),
                visible_counts=np.array([[8],[8]]))


def test_reuse_baseline_and_patch_and_generate_only_unseen_rows(tmp_path, monkeypatch):
    camera = v2.CAMERA
    n = 8
    data,view,counts = fixture(n)
    monkeypatch.setattr(v1,'COUNTS',{camera:n})
    # Keep the cap policy and use one archived patch in a small synthetic panel.
    name = tmp_path/f'baseline-{camera}.npz'
    run.archive(name,dict(view,baseline_coarse_visible_counts=counts))
    run.archive(tmp_path/f'patch-{camera}-0.npz',patch(view,0,0))
    baseline_hash = v2.b.digest(name)
    called = []

    def fail(*args):
        raise AssertionError('Reused camera must not call baseline or render directly')

    def capture(old, rows, _):
        row = int(rows[0])
        called.append(row)
        return patch(old,row,row)

    monkeypatch.setattr(run,'baseline',fail)
    monkeypatch.setattr(run,'capture',capture)
    report = dict(new_refined_points=0,views=[])
    run.complete_camera(tmp_path,data,camera,fail,fail,report,[0])
    assert called==list(range(1,n))
    assert report['new_refined_points']==n-1
    assert report['views'][0]['selected_pixels']==list(range(n))
    assert v2.b.digest(name)==baseline_hash
    charges = []
    view_out,summary = run.replay_camera(tmp_path,report['views'][0],data,[0],lambda: charges.append(1))
    assert summary['gates']['radiance_convergence']['passed'] and len(charges)==n-1
    np.testing.assert_array_equal(view_out['linear_components'],view['linear_components'])


def test_changed_prefix_fails_before_fetch():
    _,view,counts = fixture(8)
    calls = []
    with pytest.raises(SignatureError,match='Archived patch order differs'):
        v2.assemble(view,counts,lambda *args: calls.append(args),[1])
    assert not calls


def test_per_view_budget_stops_before_the_sixty_fifth_patch():
    _,view,counts = fixture(65)
    fetched = []

    def fetch(row,pixel,index):
        fetched.append(index)
        return patch(view,row,pixel)

    with pytest.raises(SignatureError,match='Per-view refinement budget exhausted'):
        v2.assemble(view,counts,fetch)
    assert fetched==list(range(64))


def test_original_panel_ceiling_covers_reused_and_new_work():
    reused = v2.b.counters(v1.COUNTS[v2.CAMERA],v2.PREFIX)
    assert all(v2.LIMITS[key]+reused[key]==v1.LIMITS[key] for key in v2.b.WORK)
    assert v2.POLICY['per_view']==64 and v2.POLICY['total']==128
