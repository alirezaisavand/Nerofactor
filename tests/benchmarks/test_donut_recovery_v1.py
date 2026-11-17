"""Small deterministic fixtures; no native field/shader, real rays or rendering."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from benchmarks import donut_recovery_v1 as d
from experiments.donut_material_recovery_v1 import shader_lighting


def triangle():
    return dict(vertices_native=np.array([[0.,0,0],[1,0,0],[0,1,0]]),
        triangles=np.array([[0,1,2]],np.int32),triangle_loops=np.array([[0,1,2]],np.int32),
        shading_normals_world=np.tile([0.,0,1.],(3,1)),
        rotated_Principled_tangent_world=np.tile([1.,0,0],(3,1)))


def test_barycentric_smooth_frame_and_exact_physical_Q():
    g=triangle();p=np.array([[.2,.3,0],[.1,.1,0]])
    f=d.surface_frames(g,p)
    assert np.allclose(f['barycentric'][0],[.5,.2,.3])
    assert np.allclose(f['normals'],[0,0,1])
    assert np.allclose(f['Q_world'],np.diag([.35**2,.08**2,0]))
    assert np.allclose(f['Q_world']@np.array([0,0,1]),0)


def test_director_signs_do_not_change_interpolated_Q():
    g=triangle();p=np.array([[.2,.3,0]])
    a=d.surface_frames(g,p)
    g['rotated_Principled_tangent_world'][1]*=-1
    b=d.surface_frames(g,p)
    np.testing.assert_allclose(a['Q_world'],b['Q_world'],atol=1e-14)


@pytest.mark.parametrize('point',[[2,2,0],[.2,.3,.01]])
def test_unattributable_hit_is_rejected(point):
    with pytest.raises(ValueError,match='not attributable'):
        d.surface_frames(triangle(),np.array([point]))


def test_inverted_source_shading_normals_fail_without_flip():
    g=triangle();g['shading_normals_world']*=-1
    with pytest.raises(ValueError,match='hemisphere'):
        d.surface_frames(g,np.array([[.2,.3,0]]))


def test_camera_ray_order_and_intrinsic_scaling():
    pose=np.eye(4);pose[:3,3]=[0,0,4]
    meta=dict(camera_angle_x=np.pi/2,frames=[dict(transform_matrix=pose.tolist())]*100)
    o,ray,c=d.camera_rays(meta,0,4)
    np.testing.assert_allclose(o,[ [0,0,2] ]*16)
    np.testing.assert_allclose(ray[2*4+2],[0,0,-1],atol=1e-15)
    assert ray[0,0]<ray[1,0] and ray[0,1]>ray[4,1]
    assert c['focal_px']==pytest.approx(2) and c['pixel_offset']==0
    assert d.camera_rays(meta,0,8)[2]['focal_px']==pytest.approx(4)
    with pytest.raises(ValueError):d.camera_rays(meta,99,4)


def test_non_rotation_camera_rejected():
    pose=np.eye(4);pose[0,0]=2
    meta=dict(camera_angle_x=1,frames=[dict(transform_matrix=pose.tolist())]*100)
    with pytest.raises(ValueError,match='rotation'):d.camera_rays(meta,0)


def test_geometric_panel_is_deterministic_and_response_free():
    theta=np.linspace(0,2*np.pi,100,endpoint=False)
    poses=np.tile(np.eye(4),(100,1,1));poses[:,0,3]=np.cos(theta);poses[:,1,3]=np.sin(theta);poses[:,2,3]=.2
    meta=dict(frames=[dict(transform_matrix=p.tolist()) for p in poses])
    assert d.select_panel(meta)==d.select_panel(meta)
    assert len(set(d.select_panel(meta)))==6 and d.select_panel(meta)[0]==0


def test_physical_error_respects_sorted_axis_equivalence():
    truth=d.surface_frames(triangle(),np.array([[.2,.3,0]]))
    pred=dict(mx=np.array([[.08]]),my=np.array([[.35]]),axis=np.array([[-1.,0,0]]),
        Q_world=truth['Q_world'],kd=np.array([d.COLOR]),metallic=np.array([[d.METALLIC]]))
    errors,summary=d.physical_errors(pred,truth)
    assert errors['orientation_degrees'][0]==0 and summary['Q_at_most_005']==1
    assert errors['major_width'][0]==errors['minor_width'][0]==0


def test_invalid_npz_objects_are_not_deserialized(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,value=np.array([{'bad':'object'}],object))
    with pytest.raises(ValueError,match='allow_pickle=False'):d.numeric(p)


def test_nonfinite_npz_rejected(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,value=np.array([np.nan]))
    with pytest.raises(ValueError,match='Finite primitive'):d.numeric(p)


def test_original_artifact_hash_mismatch_fails_early(tmp_path):
    (tmp_path/'oracle-geometry-and-frames.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Admitted input changed'):d.inputs(tmp_path,tmp_path/'camera.json')


def test_D2_changes_only_direct_light_callback_and_restores_it():
    class Original:
        @staticmethod
        def predict_outer_lights(shader,p,wi):return wi*shader.scale
    class Shader:
        def __init__(self):self.cfg=deepcopy(d.SHADER);self.callback=None;self.scale=2
        @contextmanager
        def environment(self,callback,identity):
            assert self.callback is None
            self.callback=callback
            try:yield
            finally:self.callback=None
    s=Shader();cfg=deepcopy(s.cfg)
    with shader_lighting(s,Original,'D1'):assert s.callback is None
    with shader_lighting(s,Original,'D2'):
        assert np.array_equal(s.callback(np.zeros((1,3)),np.ones((1,3))),np.full((1,3),2))
        assert s.cfg['indirect_mode']=='zero_bounce_v1'
    assert s.callback is None and s.cfg==cfg
    with pytest.raises(RuntimeError):
        with shader_lighting(s,Original,'D2'):raise RuntimeError('synthetic failure')
    assert s.callback is None


def test_wrong_indirect_mode_fails_closed():
    s=SimpleNamespace(cfg={'indirect_mode':'native_learned'})
    with pytest.raises(ValueError,match='direct-only'):shader_lighting(s,None,'D2')


def test_actual_frozen_inner_light_method_returns_zero_without_querying_network():
    import ast
    from pathlib import Path
    source=ast.parse((d.ROOT/'scientific_variants/material_a_v1.py').read_text())
    method=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='get_inner_lights')
    namespace={'torch':SimpleNamespace(zeros_like=np.zeros_like)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<frozen method fixture>','exec'),namespace)
    x=np.ones((4,3));s=SimpleNamespace(cfg={'indirect_mode':'zero_bounce_v1'})
    np.testing.assert_array_equal(namespace['get_inner_lights'](s,x,x,x),np.zeros_like(x))


def test_reference_authorization_does_not_permit_training(tmp_path,monkeypatch):
    import json
    from infrastructure.stage_sources import digest
    auth=tmp_path/'auth.json';auth.write_text(json.dumps(dict(approved=True,protocol=d.REFERENCE)))
    monkeypatch.setattr(d,'verify_freeze',lambda *a:{})
    args=SimpleNamespace(freeze=tmp_path/'freeze.json',freeze_sha256='x',authorization=auth,
        authorization_sha256=digest(auth),commit='x',image_id=d.IMAGE,output=tmp_path/'out')
    with pytest.raises(ValueError,match='Separate exact'):d.runtime(args,'D1')


def test_frozen_successful_update_and_work_counts():
    assert d.TRAIN_LIMITS['updates_per_arm']==10000
    assert d.TRAIN_LIMITS['interval']==2000 and d.TRAIN_LIMITS['evaluation_draws']==8
    assert d.REFERENCE_LIMITS['primary_rays']==6*128**2
    assert d.REFERENCE_LIMITS['shadow_directions']==d.REFERENCE_LIMITS['BRDF_point_directions']==32212254720
    assert d.SHADER['regularization_version']=='legacy_v1'
    assert not d.SHADER['human_lights'] and d.SHADER['indirect_mode']=='zero_bounce_v1'
