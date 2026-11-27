"""Focused CPU checks for one-ULP summaries without scientific decision drift."""
import copy
import numpy as np
import pytest

from benchmarks.bell_geometry_gate10_v1 import paired_surface
from benchmarks.bell_geometry_gate103_replay_v3 import compare_surface
from infrastructure.signatures import SignatureError


def fixture():
    n0=np.tile([0.,0.,1.],(4,1));angle=np.radians([2.,4.,8.,16.])
    n1=np.c_[np.sin(angle),np.zeros(4),np.cos(angle)]
    hit=np.array([True,True,True,False]);position=np.zeros((4,3));depth=np.ones(4)
    oracle=dict(hit=hit,normal=n0,position=position,depth=depth)
    reconstruction=dict(hit=hit,normal=n1,position=position+.001,depth=depth+.001)
    return paired_surface(oracle,reconstruction,np.tile([0.,0.,-1.],(4,1)))[0]


def test_actual_geometry_metrics_match_exactly_without_rounding_events():
    observed=fixture()
    assert compare_surface(copy.deepcopy(observed),observed)==[]


@pytest.mark.parametrize('captured,replay,stat',[
    (20.87159936365109,20.871599363651086,'p95'),
    (5.776839143688098,5.776839143688097,'mean')])
def test_recorded_cross_runtime_one_ULP_differences(captured,replay,stat):
    a=fixture();b=copy.deepcopy(a)
    a['signed_normal_error_degrees'][stat]=captured;b['signed_normal_error_degrees'][stat]=replay
    events=compare_surface(a,b)
    assert len(events)==1 and events[0]['float64_ULPs']==1
    assert a['signed_normal_error_degrees'][stat]==captured


@pytest.mark.parametrize('defect',['two_ULPs','substantive_angle','depth_ULP','passed','count','type','missing','extra'])
def test_substantive_or_unapproved_differences_fail(defect):
    a=fixture();b=copy.deepcopy(a)
    if defect=='two_ULPs':
        x=b['signed_normal_error_degrees']['mean']
        b['signed_normal_error_degrees']['mean']=float(np.nextafter(np.nextafter(x,np.inf),np.inf))
    elif defect=='substantive_angle':b['signed_normal_error_degrees']['p95']+=.0001
    elif defect=='depth_ULP':b['signed_depth_error']['mean']=float(np.nextafter(b['signed_depth_error']['mean'],np.inf))
    elif defect=='passed':b['passed']=not b['passed']
    elif defect=='count':b['matched_support']+=1
    elif defect=='type':b['matched_support']=float(b['matched_support'])
    elif defect=='missing':del b['signed_normal_error_degrees']['maximum']
    elif defect=='extra':b['signed_normal_error_degrees']['other']=0.
    with pytest.raises(SignatureError):compare_surface(a,b)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_summary_rejected_even_when_both_sides_identical(bad):
    a=fixture();a['signed_normal_error_degrees']['mean']=bad
    with pytest.raises(SignatureError):compare_surface(a,copy.deepcopy(a))
