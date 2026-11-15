"""New saved-array/reporting fixtures only; no old suites or native execution."""
import copy
import subprocess
import sys

import numpy as np
import pytest

from experiments.analyze_bell_highlight_e11d import response_fields,replay_equal,verify_capture,verify_layout


def fixture():
    fine=np.zeros((3,8,3,3),dtype=np.float64)
    fine[:,:,0]=.04;fine[:,:,1]=.01
    fine[0,1,1]+=.01  # resolved orientation response
    fine[1,1,1]+=.00001  # no native code response
    fine[2,1,1]+=.01  # large response, insufficient grid agreement
    fine[:,:,2]=fine[:,:,0]+fine[:,:,1]
    low=fine.copy();low[2,1,1]+=.004;low[2,1,2]=low[2,1,0]+low[2,1,1]
    return fine,low


def test_all_frozen_gates_preserved_and_quantization_not_confused_with_visibility():
    a,b=fixture();field=response_fields(a,b)['orientation']
    assert field['resolved'].tolist()==[True,False,False]
    assert field['code_difference'][1]==0
    assert field['absolute_gate'][2] and not field['grid_gate'][2]
    assert set(response_fields(a,b))=={'orientation','major','minor','isotropic'}


def test_replay_tolerance_does_not_permit_scientific_decision_or_schema_changes():
    replay_equal({'x':.001,'passed':False,'count':3},{'x':.001+1e-16,'passed':False,'count':3})
    for saved in [{'x':.001,'passed':True,'count':3},{'x':.001,'passed':False,'count':3.0},
                  {'x':.001,'passed':False},{'x':.01,'passed':False,'count':3},
                  {'x':float('nan'),'passed':False,'count':3}]:
        with pytest.raises(ValueError):replay_equal({'x':.001,'passed':False,'count':3},saved)


@pytest.mark.parametrize('kind',['shape','nonfinite','outside_png'])
def test_invalid_paired_data_rejected(kind):
    a,b=fixture()
    if kind=='shape':a=a[:,:7];b=b[:,:7]
    if kind=='nonfinite':a[0,0,2,0]=np.nan
    if kind=='outside_png':a*=100;b*=100
    with pytest.raises(ValueError):response_fields(a,b)


def capture():
    hit=np.zeros((128,128),dtype=bool);hit[1,2]=True
    full=np.zeros((128,128,3,3),dtype=np.float64);full[1,2,0]=.04;full[1,2,1]=.01;full[1,2,2]=.05
    pairs=np.repeat(full[1,2][None,None],8,axis=1)
    return dict(linear_components=full,coarse_components=full.copy(),hit=hit,pixel_indices=np.array([130],dtype=np.int64),
                paired_components=pairs,paired_coarse_components=pairs.copy(),component_effective_samples=np.ones((128,128,2)),
                component_max_sample_fraction=np.ones((128,128,2))*.1)


def test_capture_identity_sum_and_probe_pixel_checks():
    z=capture();verify_capture(z,{'hit':z['hit']},np.array([130]))
    for key in ('pixel_indices','paired_components','linear_components'):
        bad=copy.deepcopy(z)
        if key=='pixel_indices':bad[key][0]=131
        else:bad[key].flat[0]+=.01
        with pytest.raises(ValueError):verify_capture(bad,{'hit':z['hit']},np.array([130]))
    bad=copy.deepcopy(z);bad['extra']=np.ones(1)
    with pytest.raises(ValueError):verify_capture(bad,{'hit':z['hit']},np.array([130]))


def test_metadata_import_works_with_all_torch_and_native_imports_blocked():
    code='''
import builtins
real=builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0] in ('torch','network','train','raytracing','_raytracing','scientific_variants'):
        raise RuntimeError('Forbidden scientific import '+name)
    return real(name,*a,**k)
builtins.__import__=guarded
import experiments.analyze_bell_highlight_e11d
'''
    r=subprocess.run(['python3','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_empty_cuda_home_directories_are_not_unmanifested_scientific_files(tmp_path):
    (tmp_path/'pilot.json').write_text('{}');(tmp_path/'runtime/.nv/ComputeCache').mkdir(parents=True)
    assert verify_layout(tmp_path,{'pilot.json'})==['runtime','runtime/.nv','runtime/.nv/ComputeCache']
    (tmp_path/'runtime/.nv/ComputeCache/unreviewed').write_text('payload')
    with pytest.raises(ValueError,match='unmanifested'):verify_layout(tmp_path,{'pilot.json'})
