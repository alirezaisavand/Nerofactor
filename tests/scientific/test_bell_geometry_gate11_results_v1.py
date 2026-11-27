"""Focused checks for common-support accounting and exact component arithmetic."""
import numpy as np
import pytest
from experiments.report_bell_geometry_gate11_results_v1 import joint_statistics,distribution,mask_loss_validation
from infrastructure.signatures import SignatureError


def test_joint_support_keeps_missing_observations_in_coverage_accounting():
    truth=dict(hit=np.array([True,True,True,False]),position=np.zeros((4,3)),
               depth=np.ones(4),normal=np.tile([0.,0.,1.],(4,1)))
    a={k:v.copy() for k,v in truth.items()};b={k:v.copy() for k,v in truth.items()}
    a['depth']+=.01;b['depth']+=.02;b['hit'][1]=False
    a['position'][:,0]=.01;b['position'][:,0]=.02
    counts,data=joint_statistics(truth,(a,b))
    assert counts==dict(oracle=3,joint=2,O0_missing=0,O1_missing=1)
    assert all(len(x['signed_depth'])==2 for x in data)
    assert distribution(data[0]['absolute_depth'])['median']==pytest.approx(.01)
    assert distribution(data[1]['displacement'])['median']==pytest.approx(.02)
    assert np.array_equal(data[0]['normal_degrees'],[0.,0.])


def test_distribution_preserves_extreme_tail_and_signed_bias():
    values=np.array([-.3,0.,.01,.8]);r=distribution(values)
    assert r['count']==4 and r['minimum']==-.3 and r['maximum']==.8
    assert r['median']==.005 and r['p99']>.7
    assert distribution(np.array([])) is None
    with pytest.raises(SignatureError):distribution(np.array([np.nan]))


def test_component_validator_rejects_incomplete_log_without_loading_a_model():
    with pytest.raises(SignatureError):mask_loss_validation([])
