"""Result-only controls: output preservation and conditional residual accounting."""
from pathlib import Path
import numpy as np
import pytest
from experiments.finalize_bell_geometry_rgb_gate12 import output_only_rebind,controls,diagnostic_replay_comparison


def test_only_frozen_output_destination_changes():
    source="def verify(f,args):\n    root=Path('/original')\n    out=root/'analysis'\n    return root,out,f['passed']\n"
    namespace=dict(Path=Path)
    exec(compile(output_only_rebind(source,Path('/separate')), '<fixture>', 'exec'),namespace)
    assert namespace['verify'](dict(passed=True),None)==(Path('/original'),Path('/separate'),True)
    assert namespace['verify'](dict(passed=False),None)[2] is False
    with pytest.raises(ValueError):output_only_rebind(source.replace("root/'analysis'","root/'different'"),Path('/separate'))
    with pytest.raises(ValueError):output_only_rebind(source.replace('return root,out',"out=root/'analysis'\n    return root,out"),Path('/separate'))


def test_conditional_norms_are_not_added_as_causal_error_shares():
    target=np.full((2,2,5,3,3),.2);pred=target.copy()
    pred[:,:,0]=.3;pred[:,:,1]=.35;pred[:,:,2]=.4;pred[:,:,4]=.25
    # Opposing conditional residuals: -.05 visibility + .15 remainder = .1 total.
    yes=np.ones(2,bool);stats=np.zeros((2,2,13));stats[:,:,10:]=1
    state=dict(visibility=stats,pixels=np.arange(2),grids=np.tile(np.array([256,512])[:,None],(1,2)))
    record=dict(images=dict(oracle=target,O0=pred,O1=pred),states=dict(O0=state,O1=state),
                cohorts={arm:dict(valid=yes,known=yes) for arm in ('O0','O1')})
    r=controls([record],'valid')['O1']
    assert r['primary']['RGB']['known_support_RMSE']==pytest.approx(.1)
    assert r['visibility_substitution']['RGB']['known_support_RMSE']==pytest.approx(.05)
    assert r['oracle_visibility_vs_original']['RGB']['known_support_RMSE']==pytest.approx(.15)
    assert r['conditional_residual_alignment']['cosine']==pytest.approx(-1)


def test_diagnostic_roundoff_never_changes_counts_flags_or_RGB_errors():
    a=dict(passed=True,count=42,RGB_RMSE=.001,signed_luminance=.0001,transfer_roundoff_events=[])
    b=dict(a,signed_luminance=.0001+1e-19)
    assert len(diagnostic_replay_comparison(a,b)['diagnostic_events'])==1
    for change in [dict(passed=False),dict(count=43),dict(RGB_RMSE=.001+1e-18),dict(signed_luminance=.0001+1e-10)]:
        with pytest.raises(ValueError):diagnostic_replay_comparison(a,dict(b,**change))
