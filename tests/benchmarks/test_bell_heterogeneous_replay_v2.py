import numpy as np
import pytest

from experiments.verify_bell_heterogeneous_replay_v2 import ROUNDING_FIELDS, highlight_replay
from infrastructure.signatures import SignatureError


@pytest.mark.parametrize('key,value', zip(ROUNDING_FIELDS, (.014639412977349348, 10.850086081145415,
                                                          .011241994404346563)))
def test_accept_one_step_and_reject_two(key, value):
    neighbor = float(np.nextafter(value, -np.inf))
    assert highlight_replay({key: value}, {key: neighbor}) == {key: abs(value-neighbor)}
    second = float(np.nextafter(neighbor, -np.inf))
    with pytest.raises(SignatureError, match='exceeds one ULP'):
        highlight_replay({key: value}, {key: second})


@pytest.mark.parametrize('actual', [dict(pixels=31), dict(pixels=32, added=0), dict(pixels=32.)])
def test_changed_counts_fields_or_types_are_rejected(actual):
    with pytest.raises(SignatureError):
        highlight_replay(dict(pixels=32), actual)


def test_other_summaries_are_exact_and_nonfinite_always_fails():
    with pytest.raises(SignatureError, match='summary differs'):
        highlight_replay(dict(highlight_coverage=.5), dict(highlight_coverage=float(np.nextafter(.5, np.inf))))
    key = ROUNDING_FIELDS[0]
    with pytest.raises(SignatureError, match='finite'):
        highlight_replay({key: np.inf}, {key: np.inf})
