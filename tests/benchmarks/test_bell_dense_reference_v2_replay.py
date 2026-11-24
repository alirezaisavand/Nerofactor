import math

import numpy as np
import pytest

from experiments.verify_bell_dense_reference_v2_replay import highlight_replay
from infrastructure.signatures import SignatureError


def adjacent(value, count):
    for _ in range(count):
        value = float(np.nextafter(value,math.inf))
    return value


def test_record_camera_127_and_four_ulp_quantization_roundoff():
    saved = dict(integrated_specular_to_diffuse=1.889260277660826, highlight_count=71,
                 quantization_rmse_over_specular_rms=.009762216855511669)
    new = dict(saved,integrated_specular_to_diffuse=1.8892602776608258,
               quantization_rmse_over_specular_rms=.009762216855511676)
    result = highlight_replay(saved,new)
    assert result['integrated_specular_to_diffuse']['ulps']==1
    assert result['quantization_rmse_over_specular_rms']['ulps']==4
    assert 'highlight_count' not in result


@pytest.mark.parametrize('field, limit',[('quantization_rmse_over_specular_rms',4),
                                        ('integrated_specular_fraction',1)])
def test_roundoff_over_bound_fails(field,limit):
    with pytest.raises(ValueError,match='exceeds replay bound'):
        highlight_replay({field:.01},{field:adjacent(.01,limit+1)})


@pytest.mark.parametrize('actual',[{}, {'highlight_count':2}, {'highlight_count':1.0},
                                  {'highlight_count':1,'unknown':0}])
def test_scientific_accounting_remains_exact(actual):
    with pytest.raises(SignatureError):
        highlight_replay(dict(highlight_count=1),actual)


def test_nonfinite_or_unapproved_field_roundoff_fails():
    with pytest.raises(SignatureError,match='finite'):
        highlight_replay(dict(integrated_specular_fraction=.75),dict(integrated_specular_fraction=math.nan))
    with pytest.raises(SignatureError,match='differs'):
        highlight_replay(dict(highlight_coverage=.25),dict(highlight_coverage=adjacent(.25,1)))
