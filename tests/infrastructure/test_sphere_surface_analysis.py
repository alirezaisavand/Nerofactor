"""New offline-analysis checks only; never load checkpoints or scientific models."""
import numpy as np
import pytest

from experiments.analyze_sphere_surface_comparison import acceptance,image_metrics


def test_frozen_primary_rule_cannot_be_replaced_by_better_RGB():
    control=dict(neighbor_Q_p95=1.,mean_gap=1.,mean_Q_trace=1.,psnr=35.,ssim=.95)
    candidate=dict(control,neighbor_Q_p95=.81,psnr=36.)
    r=acceptance(control,candidate)
    assert not r['all_primary_checks_pass'] and not r['checks']['Q_p95_reduction_at_least_20pct']
    candidate['neighbor_Q_p95']=.75
    assert acceptance(control,candidate)['all_primary_checks_pass']
    candidate['psnr']=34.79
    assert not acceptance(control,candidate)['all_primary_checks_pass']
    candidate.update(psnr=35.,ssim=.9449)
    assert not acceptance(control,candidate)['all_primary_checks_pass']


def test_isotropization_and_shrinkage_flags_are_not_hidden_by_primary_pass():
    control=dict(neighbor_Q_p95=1.,mean_gap=1.,mean_Q_trace=1.,psnr=35.,ssim=.95)
    candidate=dict(control,neighbor_Q_p95=.5,mean_gap=.7,mean_Q_trace=.7)
    r=acceptance(control,candidate)
    assert r['all_primary_checks_pass'] and r['isotropization_flag'] and r['tensor_shrinkage_flag']


def test_saved_RGB_metric_matches_analytic_constant_image_case():
    a=np.full((16,16,3),128/255);b=np.full_like(a,127/255)
    r=image_metrics(a,b)
    assert r['psnr']==pytest.approx(10*np.log10(255**2),abs=1e-12)
    expected=(2*128*127+(.01*255)**2)/(128**2+127**2+(.01*255)**2)
    assert r['ssim']==pytest.approx(expected,abs=1e-12)
