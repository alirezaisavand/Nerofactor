"""Synthetic saved-array checks only; no scientific calls or GPU imports."""
import subprocess
import sys

import numpy as np
import pytest

from experiments.analyze_bell_anisotropy_e11f import verify_grid, verify_material, response_rows
from infrastructure.signatures import SignatureError


def fixture():
    a = np.zeros((84, 8, 3, 3), np.float64)
    a[:, :, 0] = .01; a[:, :, 1] = .02; a[:, :, 2] = a[:, :, 0]+a[:, :, 1]
    ids = np.arange(84, dtype=np.int64)
    return a, dict(components=a.copy(), pixel_indices=ids.copy(),
                   reused_case_mask=np.array([False]*7+[True])), ids


def test_exact_reused_control_diffuse_and_probe_inventory():
    a, data, ids = fixture()
    assert np.array_equal(verify_grid(data, a, ids), a)
    data['components'][:, 0, 1] += .001
    data['components'][:, 0, 2] = data['components'][:, 0, 0]+data['components'][:, 0, 1]
    assert np.array_equal(verify_grid(data, a, ids), data['components'])
    data['components'][0, 7, 1] += .001; data['components'][0, 7, 2] += .001
    with pytest.raises(SignatureError):
        verify_grid(data, a, ids)


@pytest.mark.parametrize('defect', ['diffuse', 'probe', 'mask', 'nonfinite', 'extra'])
def test_invalid_saved_arrays_rejected(defect):
    a, data, ids = fixture()
    if defect == 'diffuse':
        data['components'][:, :, 0] += .001
        data['components'][:, :, 2] = data['components'][:, :, 0]+data['components'][:, :, 1]
    elif defect == 'probe':
        data['pixel_indices'][0] = 100
    elif defect == 'mask':
        data['reused_case_mask'][0] = True
    elif defect == 'nonfinite':
        data['components'][0, 0, 0, 0] = np.nan
    else:
        data['extra'] = np.array(0)
    with pytest.raises(SignatureError):
        verify_grid(data, a, ids)


def test_requested_factor_and_canonical_isotropic_identity():
    ids = np.arange(84, dtype=np.int64); old = np.tile([.4, .2], (84, 1))
    factor = np.sqrt(old[:, 0]/old[:, 1]); new = np.stack([old[:, 0]*factor, old[:, 1]/factor], -1)
    data = dict(pixel_indices=ids, original_widths=old, stronger_widths=new, canonical_mean=np.sqrt(old.prod(-1)))
    result = verify_material(data, {'widths_major_minor': old}, ids)
    np.testing.assert_allclose(result['stronger_ratio_min_median_max'], [4., 4., 4.],
                               atol=0, rtol=16*np.finfo(np.float64).eps)
    import hashlib
    assert result['canonical_two_axis_width_sha256'] == hashlib.sha256(
        np.repeat(data['canonical_mean'][:, None], 2, axis=1).astype('<f8').tobytes()).hexdigest()
    data['stronger_widths'] = old.copy()
    with pytest.raises(SignatureError):
        verify_material(data, {'widths_major_minor': old}, ids)


def test_counts_include_losses_and_quantization_not_only_gains():
    a, _, _ = fixture(); b = a.copy()
    a[:, 1, 1] += .004; a[:, 1, 2] = a[:, 1, 0]+a[:, 1, 1]
    b[:42, 1, 1] += .008; b[:42, 1, 2] = b[:42, 1, 0]+b[:42, 1, 1]
    rows, fields = response_rows(a, a, b, b)
    theta = rows[0]
    assert theta['original_resolved'] == 84 and theta['stronger_resolved'] == 42
    assert theta['newly_resolved'] == 0 and theta['no_longer_resolved'] == 42
    assert theta['stronger_zero_codes'] == 42 and not fields['stronger']['orientation']['resolved'][42:].any()


def test_analysis_import_blocks_scientific_modules():
    script = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}:
            raise AssertionError('forbidden scientific import: '+fullname)
sys.meta_path.insert(0, Block())
import experiments.analyze_bell_anisotropy_e11f
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True)
