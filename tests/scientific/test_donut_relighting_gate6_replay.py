from copy import deepcopy

import numpy as np
import pytest

from benchmarks import donut_relighting_gate6_v2 as g
from experiments.finalize_donut_relighting_gate6_v2 import compare_convergence
from infrastructure.signatures import SignatureError


def pair():
    replay={arm:dict(maximum_rgb_difference=.0005,relative_l2=.001,passed=True) for arm in g.ROLES}
    native=dict(refined_pixels=[14,3],convergence=deepcopy(replay))
    return native,replay


@pytest.mark.parametrize('direction',[-np.inf,np.inf])
def test_adjacent_fp64_norm_reducers_agree(direction):
    native,replay=pair()
    native['convergence']['oracle']['relative_l2']=float(np.nextafter(.001,direction))
    record=compare_convergence(34,native,[14,3],replay)
    assert record['roles']['oracle']['different_float64_steps']==1
    assert all(v['different_float64_steps']==0 for k,v in record['roles'].items() if k!='oracle')


def test_two_float64_steps_are_rejected():
    native,replay=pair()
    native['convergence']['oracle']['relative_l2']=float(np.nextafter(np.nextafter(.001,np.inf),np.inf))
    with pytest.raises(SignatureError,match='more than one float64 step'):
        compare_convergence(34,native,[14,3],replay)


@pytest.mark.parametrize('field,value',[
    ('passed',False),('relative_l2',np.nan),('maximum_rgb_difference',float(np.nextafter(.0005,np.inf))),
])
def test_other_fields_stay_fail_closed(field,value):
    native,replay=pair()
    native['convergence']['oracle'][field]=value
    with pytest.raises(SignatureError):
        compare_convergence(34,native,[14,3],replay)


def test_adjacent_norm_above_scientific_bound_and_reordered_pixels_fail():
    native,replay=pair()
    with pytest.raises(SignatureError,match='Refinement'):
        compare_convergence(34,native,[3,14],replay)
    replay['oracle']['relative_l2']=.002
    native['convergence']['oracle']['relative_l2']=float(np.nextafter(.002,np.inf))
    with pytest.raises(SignatureError):
        compare_convergence(34,native,[14,3],replay)
