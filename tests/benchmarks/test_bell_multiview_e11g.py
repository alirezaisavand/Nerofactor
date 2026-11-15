"""New deterministic geometric/stacking fixtures, no native kernels or models."""
import subprocess
import sys

import numpy as np
import pytest

from experiments.bell_multiview_e11g import (select_views, outgoing_geometry, classify_visibility,
    compare_views, spectrum, transformed_material, CASES, LIMITS)
from infrastructure.signatures import SignatureError


def test_geometry_camera_center_projection_and_positive_facing():
    positions = np.zeros((84, 3)); normals = np.tile([0., 0., -1.], (84, 1))
    poses = np.tile(np.c_[np.eye(3), [0., 0., 3.]], (128, 1, 1))
    K = np.tile([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]], (128, 1, 1))
    g = outgoing_geometry(positions, normals, poses, K)
    np.testing.assert_allclose(g['centers'], np.tile([0., 0., -3.], (128, 1)))
    assert g['geometric'].all() and np.array_equal(g['uv'], np.full((128, 84, 2), 64.))
    poses[0, 0, 0] = 2
    with pytest.raises(SignatureError):
        outgoing_geometry(positions, normals, poses, K)


def test_view_selection_geometry_only_deterministic_ties_and_no_replacement():
    angle = np.linspace(0, 2*np.pi, 128, endpoint=False)
    directions = np.stack([np.cos(angle), np.sin(angle), np.zeros(128)], -1)
    g = dict(directions=np.repeat(directions[:, None], 84, axis=1), geometric=np.ones((128, 84), bool))
    first = select_views(g)
    assert first == select_views(g) and first['camera_ids'][0] == 58
    assert len(set(first['camera_ids'])) == 4 and first['outgoing_visibility_queries'] == 252
    g['geometric'][:] = False
    with pytest.raises(SignatureError):
        select_views(g)


def test_segment_visibility_keeps_endpoint_ambiguity_and_blockers():
    distance = np.ones(5)*3
    d = np.array([10., 1., 3.-1e-5, 3.-1e-5+1e-4, 3.-1e-5-1e-4])
    v = classify_visibility(d, distance)
    assert v['visible'].tolist() == [True, False, False, False, False]
    assert v['occluded'].tolist() == [False, True, False, False, False]
    assert v['unresolved'].tolist() == [False, False, True, True, True]
    assert np.all(v['visible'].astype(int)+v['occluded']+v['unresolved'] == 1)
    with pytest.raises(SignatureError):
        classify_visibility(np.array([np.nan]), np.array([3.]))


def test_rms_stacking_replicated_views_does_not_manufacture_conditioning_gain():
    J = np.array([[1., .2, 0.], [.1, .8, 0.], [0., 0., .03]])
    single = spectrum(J); duplicate = spectrum(np.tile(J, (4, 1))/2)
    np.testing.assert_allclose(single['singular_values'], duplicate['singular_values'], rtol=1e-13)
    assert np.isclose(single['condition_number'], duplicate['condition_number'])
    assert spectrum(np.array([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]]))['numerical_full_rank'] is False


def panel():
    a = np.full((2, 4, 84, 12, 3, 3), np.nan)
    valid = np.zeros((4, 84), bool); valid[0] = True; valid[1, :42] = True
    for arm in range(2):
        for v in range(4):
            for p in np.flatnonzero(valid[v]):
                a[arm, v, p] = .01
                a[arm, v, p, :, 2] = .02
                a[arm, v, p, 1, 1, 0] += .006
                a[arm, v, p, 1, 2, 0] += .006
                if v:
                    a[arm, v, p, 3, 1, 1] += .005
                    a[arm, v, p, 3, 2, 1] += .005
    return a, valid


def test_matched_probe_groups_quantization_and_invalid_slots():
    a, valid = panel(); r = compare_views(a, a, valid)
    one = next(x for x in r['matched_coverage'] if x['material'] == 0 and x['valid_view_count'] == 1)
    two = next(x for x in r['matched_coverage'] if x['material'] == 0 and x['valid_view_count'] == 2)
    assert one['matched_probes'] == two['matched_probes'] == 42
    assert one['factors']['major']['multiview_any_resolved'] == 0
    assert two['factors']['major']['single_resolved'] == 0
    assert two['factors']['major']['multiview_any_resolved'] == 42
    assert len(r['per_probe_conditioning']) == 168
    assert r['all_measured_cases_converged'] is True
    bad_coarse = a.copy(); bad_coarse[:, 0, :, 0, 2] *= .5
    failed = compare_views(a, bad_coarse, valid)
    assert failed['all_measured_cases_converged'] is False
    assert not any(x['exploratory_screen_passed'] for x in failed['matched_conditioning_summary'])
    a[0, 1, 0, 0, 2, 0] = np.nan
    with pytest.raises(SignatureError):
        compare_views(a, a, valid)


def test_log_coordinates_are_actual_product_and_ratio_changes():
    import torch
    from experiments.bell_anisotropy_e11f import admit_isotropic, strengthen
    labels = dict(normals=torch.tensor([[0., 0., 1.]], dtype=torch.float64),
                  T=torch.tensor([[1., 0., 0.]], dtype=torch.float64), B=torch.tensor([[0., 1., 0.]], dtype=torch.float64),
                  theta_major=torch.tensor([.2], dtype=torch.float64),
                  widths_major_minor=torch.tensor([[.4, .2]], dtype=torch.float64),
                  color=torch.tensor([[.3, .4, .5]], dtype=torch.float64), metallic=torch.tensor([[.2]], dtype=torch.float64))
    admit_isotropic(torch, labels, strengthen(torch, labels))
    old = labels['widths_major_minor'].numpy()
    for case in CASES[8:]:
        w = transformed_material(torch, labels, case)['widths'].numpy()
        delta_u = .5*np.log(w.prod(-1)/old.prod(-1))
        delta_v = np.log((w[:, 0]/w[:, 1])/(old[:, 0]/old[:, 1]))
        sign = 1 if case.endswith('plus') else -1
        np.testing.assert_allclose(delta_u, sign*.1 if case.startswith('mean_') else 0., atol=1e-15)
        np.testing.assert_allclose(delta_v, sign*.1 if case.startswith('ratio_') else 0., atol=1e-15)
    assert np.array_equal(labels['widths_major_minor'].numpy(), old)


def test_budget_and_metadata_scientific_import_blocker():
    assert LIMITS['max_brdf_point_directions'] == (8*84+23*146)*(256**2+512**2)
    assert LIMITS['max_outgoing_visibility_queries'] == 146
    script = '''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.bell_multiview_e11g
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True)


def test_native_loop_layout_and_exact_work_with_CPU_standins_only(tmp_path, monkeypatch):
    """Exercise the real orchestration/indexing with toy kernels and no CUDA/BVH."""
    import time
    from types import SimpleNamespace
    import torch
    import experiments.bell_multiview_e11g as m
    import infrastructure.verified_ray_backend as backend
    import benchmarks.bell_geometry_e11 as geometry
    import benchmarks.bell_reference_e11 as reference
    from experiments.bell_quadrature_e11e import midpoint_grid
    monkeypatch.setattr(m, 'GRIDS', (4, 8))  # Tiny disposable quadrature, not a scientific run.
    monkeypatch.setattr(torch, '__version__', '1.8.1+cu111')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda n: 'A100 synthetic')
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda n: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    zeros = torch.zeros
    monkeypatch.setattr(torch, 'zeros', lambda *a, **kw: zeros(*a, **{k: v for k, v in kw.items() if k != 'device'}))
    threads = torch.get_num_threads()
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(backend, 'load', lambda *a: {'synthetic_only': True})
    monkeypatch.setattr(reference, 'environment', lambda wi, light: torch.ones_like(wi))
    monkeypatch.setattr(m, 'baseline', lambda *a: (tmp_path, {}))
    (tmp_path/'raytracing').mkdir()
    (tmp_path/'raytracing/raytracer.py').write_text('''
import torch
class RayTracer:
 def __init__(self,*a): pass
 def trace(self,o,d): return o.clone(),d.clone(),torch.full((len(o),),10.)
''')
    monkeypatch.setattr(geometry, 'mesh_arrays', lambda p: (np.zeros((3, 3), np.float32), np.zeros((9, 3), np.int32)))
    calls = []
    def toy_kernel(torch, wi, wo, material):
        calls.append(wi.shape[0]*wi.shape[1])
        return torch.ones_like(wi)*.01, torch.ones_like(wi)*.02
    monkeypatch.setattr(m, 'kernel_axes', toy_kernel)
    labels = dict(position=np.zeros((84, 3)), normals=np.tile([0., 0., 1.], (84, 1)),
                  T=np.tile([1., 0., 0.], (84, 1)), B=np.tile([0., 1., 0.], (84, 1)),
                  theta_major=np.zeros(84), widths_major_minor=np.tile([.4, .2], (84, 1)),
                  color=np.full((84, 3), .3), metallic=np.full((84, 1), .2))
    g = dict(directions=np.tile([0., 0., 1.], (128, 84, 1)), distance=np.full((128, 84), 3.),
             centers=np.tile([0., 0., 3.], (128, 1)), facing=np.ones((128, 84)),
             uv=np.full((128, 84, 2), 64.), in_frame=np.ones((128, 84), bool), geometric=np.zeros((128, 84), bool))
    ids = [58, 44, 9, 0]
    for camera, n in zip(ids, (84, 42, 42, 62)):
        g['geometric'][camera, :n] = True
    cache = {}
    for size in (4, 8):
        nodes, weight = midpoint_grid(size)
        wi = torch.from_numpy(np.tile(nodes[None], (4, 1, 1)))
        pair = m.weighted_components(torch, torch.ones_like(wi)*.01, torch.ones_like(wi)*.02,
                                    torch.ones_like(wi), wi[:, :, 2], torch.ones((4, size*size), dtype=torch.bool), weight)
        components = torch.cat([pair, pair.sum(1, keepdim=True)], 1).numpy()[0]
        original = np.tile(components[None, None], (84, 8, 1, 1))
        cache[size] = dict(nodes=nodes, weight=weight, visibility=np.full((84, size*size//8), 255, np.uint8),
                           original=original.copy(), stronger=original.copy())
    args = SimpleNamespace(benchmark=tmp_path, output=tmp_path/'out', image_id='synthetic_only')
    args.output.mkdir(); counters = dict(outgoing_visibility_attempts=0, outgoing_visibility_returns=0,
        brdf_attempts=0, brdf_returns=0, environment_attempts=0, environment_returns=0)
    report = dict(counters=counters, completed_grids=[])
    result = m.render(args, (labels, None, np.arange(84), cache, g, []),
                      dict(camera_ids=ids, outgoing_visibility_queries=146), report, time.monotonic(), torch)
    assert sum(calls) == counters['brdf_returns'] == (8*84+23*146)*(4**2+8**2)
    assert counters['outgoing_visibility_returns'] == 146 and result['cuda_peak_allocated_bytes'] == 0
    for size in (4, 8):
        with np.load(args.output/('grid-'+str(size)+'.npz'), allow_pickle=False) as z:
            a = z['components']; valid = z['valid']
            assert np.isnan(a[:, ~valid]).all() and np.array_equal(a[0, 0, :, :8], cache[size]['original'])
            np.testing.assert_allclose(a[0, :, :, 7], a[1, :, :, 7], equal_nan=True)
    assert torch.get_num_threads() == threads
