"""Synthetic reporting checks only; no previous experiment or shader repeated."""
import subprocess
import sys

import numpy as np
import pytest

from experiments.analyze_bell_multiview_e11g import verify_grid, verify_geometry, response_tables, matched_statistics
from experiments.bell_multiview_e11g import compare_views
from infrastructure.signatures import SignatureError


def panel():
    pixels = np.arange(84, dtype=np.int64); ids = np.array([58, 44, 9, 0], dtype=np.int64)
    valid = np.zeros((4, 84), bool); valid[0] = True; valid[1, :42] = True
    valid[2, 21:63] = True; valid[3, 21:83] = True
    a = np.full((2, 4, 84, 12, 3, 3), np.nan)
    for arm in range(2):
        for view in range(4):
            a[arm, view, valid[view], :, 0] = .01
            a[arm, view, valid[view], :, 1] = .02
            a[arm, view, valid[view], :, 2] = .03
    for k, channel in [(1, 0), (3, 1), (5, 2), (8, 1), (10, 2)]:
        for arm in range(2):
            for view in range(4):
                a[arm, view, valid[view], k, 1, channel] += .005
                a[arm, view, valid[view], k, 2, channel] = a[arm, view, valid[view], k, 0, channel]+a[arm, view, valid[view], k, 1, channel]
    data = dict(components=a, valid=valid, camera_ids=ids, pixel_indices=pixels, completed_probes=np.ones(84, bool))
    cache = dict(original=a[0, 0, :, :8].copy(), stronger=a[1, 0, :, :8].copy())
    return data, cache, pixels, ids


def test_grid_admission_preserves_baselines_and_rejects_fabricated_missing_views():
    data, cache, pixels, ids = panel()
    verify_grid(data, data['valid'], pixels, ids, cache)
    data['components'][0, 1, 83] = 0
    with pytest.raises(SignatureError):
        verify_grid(data, data['valid'], pixels, ids, cache)
    data['components'][0, 1, 83] = np.nan
    data['components'][0, 0, 0, 1, 1] += .001
    data['components'][0, 0, 0, 1, 2] += .001
    with pytest.raises(SignatureError):
        verify_grid(data, data['valid'], pixels, ids, cache)


def test_geometry_replay_rejects_camera_or_classification_substitution():
    data, cache, pixels, ids = panel(); queried = data['valid'].copy(); queried[0] = False
    geometry = dict(centers=np.tile([0., 0., 3.], (128, 1)), directions=np.tile([0., 0., 1.], (128, 84, 1)),
        distance=np.full((128, 84), 3.), facing=np.ones((128, 84)), uv=np.full((128, 84, 2), 64.),
        in_frame=np.ones((128, 84), bool), geometric=np.zeros((128, 84), bool))
    geometry['geometric'][ids] = data['valid']
    saved = dict(camera_ids=ids, pixel_indices=pixels, queried=queried, valid=data['valid'],
        occluded=np.zeros((4, 84), bool), unresolved=np.zeros((4, 84), bool),
        nearest_depth=np.where(queried, 10., np.nan))
    for dst, src in [('camera_centers', 'centers'), ('outgoing_directions', 'directions'),
                     ('camera_distances', 'distance'), ('facing', 'facing'), ('uv', 'uv'),
                     ('in_frame', 'in_frame'), ('geometric', 'geometric')]:
        saved[dst] = geometry[src][ids].copy()
    coverage, errors = verify_geometry(saved, geometry, dict(camera_ids=ids.tolist()), pixels)
    assert [row['queried'] for row in coverage] == [0, 42, 42, 62] and max(errors.values()) == 0
    saved['camera_centers'][1, 0] += .01
    with pytest.raises(SignatureError):
        verify_geometry(saved, geometry, dict(camera_ids=ids.tolist()), pixels)
    saved['camera_centers'][1, 0] -= .01; saved['occluded'][1, 0] = True
    with pytest.raises(SignatureError):
        verify_geometry(saved, geometry, dict(camera_ids=ids.tolist()), pixels)


def test_matched_statistics_and_responses_keep_missing_probe_and_quantization():
    data, cache, pixels, ids = panel(); a = data['components']; valid = data['valid']
    comparison = compare_views(a, a, valid); stats = matched_statistics(comparison)
    subset = [r for r in stats if r['valid_view_group'] == 'at_least_two']
    assert all(r['probes'] == 83 for r in subset)
    for row in subset:
        assert np.isclose(row['single_median_condition'], row['stacked_RMS_median_condition'])
    details, summary = response_tables(a, a, valid, pixels, ids)
    assert len(details) == 2*int(valid.sum())*6
    assert all(r['probes'] == 83 for r in summary if r['group'] == 'at_least_two')
    assert all(r['multiview_any_two_codes'] == r['single_two_codes'] for r in summary)
    assert all(r['camera_id'] in ids for r in details)


def test_file_only_analysis_import_has_no_scientific_dependency():
    script = '''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.analyze_bell_multiview_e11g
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True)
