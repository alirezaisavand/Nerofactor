"""Recorded E11C float32 replay differences; no builds or native calls."""
import ast,json,math,subprocess
from copy import deepcopy
import numpy as np
import pytest
from infrastructure.ray_backend_acceptance_e11c import replay_cases,FACING_DOT_REPLAY_ABS
from infrastructure.stage_sources import ROOT


def row():return [dict(hit=True,expected_hit=True,passed=True,depth=2.75,normal_abs=2e-8,facing_dot=float(np.float32(.9585405)))]


@pytest.mark.parametrize('sign',[-1,1])
def test_recorded_one_ulp_facing_difference_is_diagnostic_only(sign):
    a=row();b=deepcopy(a);b[0]['facing_dot']=float(np.nextafter(np.float32(a[0]['facing_dot']),np.float32(sign*np.inf)))
    result=replay_cases(a,b,'bell')
    assert result['passed'] and not result['mismatches'] and len(result['permitted_facing_roundoff'])==1
    assert result['permitted_facing_roundoff'][0]['absolute_difference']==5.960464477539063e-8==FACING_DOT_REPLAY_ABS


def test_exact_rows_preserve_empty_roundoff_inventory():
    result=replay_cases(row(),row(),'bell');assert result==dict(passed=True,mismatches=[],permitted_facing_roundoff=[])


@pytest.mark.parametrize('field,value',[('hit',False),('passed',False),('depth',2.750000000000001),('normal_abs',2.0000001e-8),('expected_hit',1)])
def test_no_other_acceptance_classification_numeric_or_type_tolerance(field,value):
    a=row();b=deepcopy(a);b[0][field]=value;assert not replay_cases(a,b,'bell')['passed']


@pytest.mark.parametrize('fault',['missing','extra','short','large_facing','nonfinite_facing','nonfinite_other','wrong_type'])
def test_missing_extra_nonfinite_or_substantive_values_fail(fault):
    a=row();b=deepcopy(a)
    if fault=='missing':del b[0]['facing_dot']
    if fault=='extra':b[0]['new_field']=0
    if fault=='short':b=[]
    if fault=='large_facing':b[0]['facing_dot']+=2*FACING_DOT_REPLAY_ABS
    if fault=='nonfinite_facing':a[0]['facing_dot']=b[0]['facing_dot']=float('nan')
    if fault=='nonfinite_other':a[0]['depth']=b[0]['depth']=float('inf')
    if fault=='wrong_type':b[0]['facing_dot']=str(b[0]['facing_dot'])
    assert not replay_cases(a,b,'bell')['passed']


def test_facing_tolerance_does_not_apply_to_identity_or_nested_metadata():
    a={'facing_dot':.75};b={'facing_dot':.75+FACING_DOT_REPLAY_ABS}
    assert not replay_cases(a,b,'identity')['passed']


def test_native_comparison_build_and_probe_are_unchanged():
    old=subprocess.check_output(['git','show','6af0474:run_ray_backend_e11c.py'],cwd=ROOT,text=True)
    new=(ROOT/'run_ray_backend_e11c.py').read_text()
    funcs=lambda s:{n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(old),funcs(new)
    for name in ('guarded','build','probe'):assert a[name]==b[name],name
    previous=subprocess.check_output(['git','show','6af0474:infrastructure/ray_backend_e11c.py'],cwd=ROOT,text=True)
    assert previous==(ROOT/'infrastructure/ray_backend_e11c.py').read_text()
