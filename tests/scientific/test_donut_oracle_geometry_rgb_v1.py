import numpy as np
import pytest

from benchmarks import donut_oracle_geometry_rgb_v1 as rgb
from experiments.verify_donut_oracle_geometry_rgb_v1 import replay_gates, score


def capture(pixels, size, light, error):
    colors = np.zeros((len(pixels), 3, 3))
    colors[:,0] = (np.asarray(light)+error)[:,None]
    colors[:,2] = colors[:,0]
    return dict(components=colors, visibility=np.full(len(pixels), size), pixels=np.array(pixels), grid=np.array(size))


def test_absolute_failure_precedes_pooled_and_pixel_breaks_ties():
    # Oracle has a larger total residual but stays below the absolute tolerance.
    a = np.full((2,5,3,3), .1)
    a[1,:,2] += .0028
    b = np.full((2,3,3,3), .1)
    b[1,[0,1],2] += .004
    states = dict(oracle=dict(pixels=np.arange(5), rgb=a), reconstruction=dict(pixels=np.array([40,8,90]), rgb=b))
    assert rgb.choose(states, []) == 8
    with pytest.raises(ValueError, match='Refined absolute'):
        rgb.choose(states, [8])


def test_pair_fetches_both_high_grids_only_at_selected_pixel():
    views = dict(oracle=dict(pixel_indices=np.array([1,2])), reconstruction=dict(pixel_indices=np.array([2,3])))
    calls = []

    def fetch(arm, view, size, prefix):
        pixels = view['pixel_indices']
        calls.append((arm, size, prefix, pixels.tolist()))
        error = np.where(pixels == 2, .01, 0.) if size == 512 else np.zeros(len(pixels))
        return capture(pixels, size, np.full(len(pixels),.2), error)

    states, done = rgb.paired(views, fetch, 0)
    assert done == [2]
    assert calls[4:] == [(arm, size, 'pixel-2', [2]) for arm in rgb.ARMS for size in rgb.HIGH]
    np.testing.assert_array_equal(states['oracle']['grids'], [[256,1024],[512,2048]])
    np.testing.assert_array_equal(states['reconstruction']['grids'], [[1024,256],[2048,512]])
    with pytest.raises(ValueError, match='budget exhausted'):
        rgb.paired(views, fetch, rgb.MAX_TOTAL)


def test_refinement_with_no_counterpart_does_not_render_missing_mesh():
    views = dict(oracle=dict(pixel_indices=np.array([1,2])), reconstruction=dict(pixel_indices=np.array([2,3])))
    def fetch(arm, view, size, prefix):
        pixels = view['pixel_indices']
        if prefix != 'base':
            assert arm == 'oracle' and pixels.tolist() == [1]
        error = (pixels == 1)*.01 if size == 512 else np.zeros(len(pixels))
        return capture(pixels, size, np.full(len(pixels), .25), error)
    _, done = rgb.paired(views, fetch, 0)
    assert done == [1]


def test_undefined_radiance_is_excluded_from_error_but_keeps_its_energy():
    a = np.zeros((2,4,3,3))
    b = a.copy()
    a[:,:,2] = np.array([1.,2.,0.,0.])[None,:,None]
    b[:,:,2] = np.array([.5,100.,.25,0.])[None,:,None]
    b[0,0,2] = .75
    yes = np.ones(4,bool)
    known = np.array([True,False,True,True])
    s = score(a,b,yes,known)
    m = s['RGB']
    assert (s['defined_pixels'],s['undefined_pixels']) == (3,1)
    assert m['undefined_oracle_energy_fraction'] == .8
    assert m['squared_error'] == 3*(.5**2+.25**2)
    assert m['squared_relative_error_lower_bound'] == (.5**2+.25**2)/5
    assert m['coarse_fine_difference_RMSE'] == pytest.approx(np.sqrt(3*.25**2/9))
    rec_only = score(a,b,np.array([False,False,True,False]),known)
    assert rec_only['RGB']['relative_L2'] is None
    assert rec_only['RGB']['squared_error'] > 0


def test_grid_checks_visibility_and_component_identity():
    a = capture([4],256,[.2],np.zeros(1))
    view = dict(pixel_indices=np.array([4]))
    rgb.verify_grid(view,256,a)
    a['components'][0,2,0] += .01
    with pytest.raises(ValueError, match='quadrature grid'):
        rgb.verify_grid(view,256,a)
    assert rgb.BASE_COST == 14_794_752_000
    assert rgb.BASE_COST+rgb.HIGH_COST == 16_136_929_280


def test_offline_norm_roundoff_keeps_exact_gates_and_selection():
    gate = dict(passed=True, maximum_rgb_difference=.001, relative_l2=.0005)
    measured = dict(oracle_replay=gate, convergence={arm: gate for arm in rgb.ARMS})
    saved = dict(measured, refined_pixels=[42], oracle_replay=dict(gate, relative_l2=.0005+1e-18))
    replay_gates(saved, measured, [42])
    with pytest.raises(ValueError, match='convergence replay'):
        replay_gates(dict(saved, oracle_replay=dict(gate, passed=False)), measured, [42])
    with pytest.raises(ValueError, match='selection differs'):
        replay_gates(saved, measured, [43])
