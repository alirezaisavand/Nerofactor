import copy

import pytest

from experiments.analyze_donut_learned_light_relighting_gate9_v2 import (
    RESIDUAL, RESIDUAL_ROUNDOFF, check_endpoint_parity, g,
)
from infrastructure.signatures import SignatureError


@pytest.fixture
def archive():
    return {str(c): {arm: dict(pixels=count, **{RESIDUAL: 8.87715409514378e-09}) for arm in g.ARMS}
            for c,count in zip(g.CAMERAS,g.g6.COUNTS)}


def test_roundoff_on_the_observed_residual(archive):
    replay=copy.deepcopy(archive)
    replay['28']['D8-C'][RESIDUAL]=8.877154053510417e-09
    check_endpoint_parity(archive,replay)


@pytest.mark.parametrize('difference', [2*RESIDUAL_ROUNDOFF, 1e-7, float('nan'), float('inf')])
def test_larger_or_nonfinite_residual_is_rejected(archive,difference):
    replay=copy.deepcopy(archive)
    replay['28']['D8-C'][RESIDUAL]+=difference
    with pytest.raises(SignatureError,match='residuals differ'):
        check_endpoint_parity(archive,replay)


@pytest.mark.parametrize('field,value', [('pixels', 4850), ('pixels',4849.0), ('additional',True)])
def test_non_residual_fields_require_exact_parity(archive,field,value):
    replay=copy.deepcopy(archive)
    replay['28']['D8-C'][field]=value
    with pytest.raises(SignatureError,match='count or diagnostic'):
        check_endpoint_parity(archive,replay)
