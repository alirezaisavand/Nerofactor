"""New analytic/noise controls only; no BRDF, tracer, model or reference rendering."""
import numpy as np
import pytest
from benchmarks.material_information_e14 import (rotated_light,linear_bin_edges,gaussian_bins,
    information_row,line_interval,matrix_summary,SIGMA)
from experiments.bell_highlight_e11d import candidates,mean_environment
from infrastructure.signatures import SignatureError


def test_prescribed_rotation_preserves_axes_power_RGB_and_sphere_energy():
    old=candidates()['L2_six_cardinal'];new=rotated_light(old)
    a=np.array([x['axis'] for x in old['lobes']]);b=np.array([x['axis'] for x in new['lobes']])
    np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-14)
    np.testing.assert_allclose(mean_environment(old),mean_environment(new),rtol=0,atol=0)
    assert old['ambient']==new['ambient']
    assert all(x['power']==y['power']==64 and x['RGB']==y['RGB'] for x,y in zip(old['lobes'],new['lobes']))
    axis=np.ones(3)/np.sqrt(3)
    np.testing.assert_allclose(b[[0,2,4]].sum(0)/np.sqrt(3),axis,atol=1e-14)
    assert np.isclose(np.trace(b[[0,2,4]].T),1+2*np.cos(np.pi/4))


def test_full_support_bins_are_correct_inverse_sRGB_boundaries():
    e=linear_bin_edges();assert len(e)==257 and e[0]==-np.inf and e[-1]==np.inf
    s=np.where(e[1:-1]<=.0031308,e[1:-1]*12.92,1.055*e[1:-1]**(1/2.4)-.055)
    np.testing.assert_allclose(s,(np.arange(255)+.5)/255,atol=2e-15,rtol=0)


def test_probabilities_score_and_data_processing_including_extreme_tails():
    mu=np.r_[-.02,0,.005,.02,.1,.5,.99,1,1.02]
    result=gaussian_bins(mu)
    np.testing.assert_allclose(result['probabilities'].sum(1),1,atol=5e-13)
    np.testing.assert_allclose(result['dP_dmu'].sum(1),0,atol=5e-9)
    assert np.all(result['information']<=1/SIGMA**2*(1+1e-10))
    assert result['underflow_probability_bins']>0
    assert result['information'][0]<result['information'][3] and result['information'][-1]<result['information'][-2]


def test_Gaussian_derivative_matches_independent_finite_difference():
    mu=np.array([.00015,.005,.03,.1,.8,.999]);h=1e-7
    mid=gaussian_bins(mu);plus=gaussian_bins(mu+h);minus=gaussian_bins(mu-h)
    np.testing.assert_allclose((plus['probabilities']-minus['probabilities'])/(2*h),mid['dP_dmu'],atol=4e-5,rtol=3e-7)


def test_one_bin_zero_information_and_binary_threshold_analytic_information():
    one=gaussian_bins(np.array([-.1,0,.1]),edges=np.array([-np.inf,np.inf]))
    assert np.array_equal(one['information'],np.zeros(3))
    binary=gaussian_bins(np.array([0.]),edges=np.array([-np.inf,0.,np.inf]))
    np.testing.assert_allclose(binary['information'],2/(np.pi*SIGMA**2),rtol=1e-13)


def test_fine_bin_limit_and_small_noise_in_cell_interior():
    e=np.r_[-np.inf,np.linspace(-.05,.05,2001),np.inf]
    fine=gaussian_bins(np.array([.01313]),edges=e)
    assert .999<fine['information'][0]*SIGMA**2<=1
    bins=linear_bin_edges();mu=(bins[100]+bins[101])/2;width=bins[101]-bins[100]
    interior=gaussian_bins(np.array([mu]),sigma=width/100)
    assert interior['information'][0]==0  # Natural IEEE underflow, not a probability floor.


@pytest.mark.parametrize('mu,sigma,edges',[(np.array([np.nan]),SIGMA,None),(np.array([0.]),0,None),(np.array([0.]),SIGMA,np.array([0,1]))])
def test_invalid_noise_inputs_fail(mu,sigma,edges):
    with pytest.raises(SignatureError):gaussian_bins(mu,sigma,edges)


def capture():
    J=np.array([[.02,.01,0],[0,.03,.005],[.001,0,.01]])
    a=np.zeros((2,12,3,3));a[:,:,0]=.02;a[:,:,1]=.03;a[:,:,2]=.05
    for plus,minus,col,h in [(1,2,0,np.pi/12),(8,9,1,.1),(10,11,2,.1)]:
        for k,sign in ((plus,1),(minus,-1)):
            a[:,k,1]+=sign*h*J[:,col];a[:,k,2]=a[:,k,0]+a[:,k,1]
    return a


def test_duplicate_observations_only_double_total_information_not_conditioning():
    a=capture();one=information_row(a,a);two=information_row(np.tile(a,(2,1,1,1)),np.tile(a,(2,1,1,1)))
    assert two['rgb_observations']==2*one['rgb_observations']
    for name in ('ideal','gaussian_linear','gaussian_quantized8'):
        np.testing.assert_allclose(one[name]['singular_values'],two[name]['singular_values'],rtol=1e-13)
        np.testing.assert_allclose(one[name]['per_RGB_observation_JtJ'],two[name]['per_RGB_observation_JtJ'],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(2*np.array(one[name]['total_JtJ']),two[name]['total_JtJ'],rtol=1e-13,atol=1e-13)
    assert 'total_information_eigenvalues' not in one['ideal']
    assert one['deterministic8']['coordinate_line_sections']==two['deterministic8']['coordinate_line_sections']


def test_noiseless_intervals_and_rank_deficiency_are_not_fictitious_Fisher():
    mu=np.full(3,.05);J=np.diag([1.,2.,0.])
    interval=line_interval(mu,J,np.array([0.,0.,1.]))
    assert interval['lower_unbounded'] and interval['upper_unbounded'] and interval['width'] is None
    s=matrix_summary(J);assert s['invertibility']=='numerically_rank_deficient' and s['inverse_gram_axis_scales'] is None


def test_quantized_information_matrix_cannot_exceed_analogue_information():
    a=capture();r=information_row(a,a)
    difference=np.array(r['gaussian_linear']['total_JtJ'])-np.array(r['gaussian_quantized8']['total_JtJ'])
    assert np.linalg.eigvalsh(difference).min()>=-1e-12
