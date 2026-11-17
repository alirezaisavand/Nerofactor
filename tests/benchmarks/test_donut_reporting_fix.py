"""Actual highlight-helper/report regression; synthetic arrays, no rendering."""
import json
import numpy as np
import pytest

from benchmarks.donut_recovery_v1 import json_new
from benchmarks.generate_donut_recovery_v1 import highlight_record
from benchmarks.bell_e11 import convergence
from experiments.bell_highlight_e11d import measures


@pytest.mark.parametrize('scale',[.1,10.])
def test_actual_highlight_result_full_row_and_terminal_report(tmp_path,scale):
    diffuse=np.full((8,3),.01)*scale
    specular=np.arange(8)[:,None]*np.array([[.02,.03,.01]])*scale
    components=np.stack([diffuse,specular,diffuse+specular],1)
    original_summary,original_mask=measures(components)
    summary,mask=highlight_record(components)
    assert summary==original_summary
    np.testing.assert_array_equal(mask,original_mask)
    # This is the previously failing object returned by the real reused helper.
    with pytest.raises(TypeError):json.dumps(dict(highlight_diagnostics=(original_summary,original_mask)))
    row=dict(camera=0,highlight_diagnostics=summary,radiance_convergence=dict(passed=True),
             energy_convergence=dict(passed=False))
    assert json.loads(json.dumps(row))['highlight_diagnostics']==original_summary
    report=dict(status='failed',cleanup_verified=False,views=[row],error='Convergence failed')
    path=tmp_path/'generation.json';json_new(path,report)
    restored=json.loads(path.read_text())
    assert restored==report and not restored['views'][0]['energy_convergence']['passed']
    np.savez_compressed(tmp_path/'view.npz',highlight_mask=mask)
    with np.load(tmp_path/'view.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['highlight_mask'],original_mask)


@pytest.mark.parametrize('value,error',[(dict(mask=np.ones(3,bool)),TypeError),(dict(loss=float('nan')),ValueError)])
def test_bad_report_never_creates_partial_terminal_file(tmp_path,value,error):
    path=tmp_path/'generation.json'
    with pytest.raises(error):json_new(path,value)
    assert not path.exists()


def test_existing_terminal_record_cannot_be_overwritten(tmp_path):
    path=tmp_path/'generation.json';path.write_text('original evidence\n')
    with pytest.raises(FileExistsError):json_new(path,dict(status='complete'))
    assert path.read_text()=='original evidence\n'


def test_serialization_correction_cannot_accept_failed_reflectance():
    fine=np.ones((5,3))*.8;coarse=fine.copy();coarse[0,1]-=.0084
    result=convergence(fine,coarse)
    assert not result['passed'] and result['maximum_rgb_difference']>.003
    assert json.loads(json.dumps(result))['passed'] is False
