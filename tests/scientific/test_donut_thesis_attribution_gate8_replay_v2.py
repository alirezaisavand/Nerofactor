import numpy as np
import pytest

from experiments.analyze_donut_thesis_attribution_gate8_v2 import check_diagnostic
from infrastructure.signatures import SignatureError


def test_recorded_rounding_is_allowed_and_larger_change_rejected():
    first=0.03248042072714239
    second=0.0324804207271424
    assert np.nextafter(first,np.inf)==second
    check_diagnostic(dict(pooled_limit=first,passed=False),dict(pooled_limit=second,passed=False))
    with pytest.raises(SignatureError,match='more than one'):
        check_diagnostic(dict(pooled_limit=first),dict(pooled_limit=float(np.nextafter(second,np.inf))))


@pytest.mark.parametrize('replay',[
    dict(pooled_limit=0.1,passed=True,gap=0.2),
    dict(pooled_limit=0.1,passed=0,gap=0.2),
    dict(pooled_limit=0.1,passed=False,gap=float(np.nextafter(.2,np.inf))),
    dict(pooled_limit=0.1,passed=False,gap=float('nan')),
    dict(pooled_limit=float('inf'),passed=False,gap=0.2),
    dict(pooled_limit=0.1,passed=False),
    dict(pooled_limit=0.1,passed=False,gap=0.2,extra=1),
])
def test_decisions_and_other_fields_are_exact(replay):
    with pytest.raises(SignatureError):
        check_diagnostic(replay,dict(pooled_limit=0.1,passed=False,gap=0.2))
