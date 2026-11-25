import numpy as np
import pytest

from benchmarks import donut_relighting_gate6_v1 as g
from experiments.analyze_donut_relighting_gate6_v1 import raster, score, scopes
from infrastructure.signatures import SignatureError


def test_unseen_light_preserves_lobe_shape_and_spherical_RGB_integral():
    def integral(light):
        return 4*np.pi*np.asarray(light['ambient'])+sum(2*np.pi*np.asarray(l['RGB'])/(l['power']+1)
                                                        for l in light['lobes'])

    fit,novel=g.ENVIRONMENT,g.LIGHT
    np.testing.assert_array_equal(integral(fit),integral(novel))
    a,b=[np.asarray([l['axis'] for l in light['lobes']]) for light in (fit,novel)]
    np.testing.assert_allclose(b@b.T,a@a.T,rtol=0,atol=1e-14)
    assert np.degrees(np.arccos((a@b.T).max()))==pytest.approx(36.4150539484)
    assert all(x['power']==y['power']==64 and x['RGB']==y['RGB']
               for x,y in zip(fit['lobes'],novel['lobes']))


def test_Q_roundtrip_preserves_director_sign_and_isotropy():
    n=np.tile([0.,0.,1.],(2,1))
    axis=np.array([[.6,.8,0.],[1.,0.,0.]])
    widths=np.array([[.43,.07],[.20,.20]])
    q=g.tensor_from_axes(widths,axis,n)
    np.testing.assert_array_equal(g.tensor_from_axes(widths,-axis,n),q)
    np.testing.assert_allclose(g.tensor_from_axes(widths[:,::-1],np.cross(n,axis),n),q,atol=2e-16)
    recovered,new_axis,error=g.physical_axes(q,n)
    np.testing.assert_allclose(recovered,widths,atol=2e-16)
    np.testing.assert_allclose(abs((new_axis[0]*axis[0]).sum()),1,atol=2e-16)
    assert error<2e-16


@pytest.mark.parametrize('diagonal',[(.25,.04,.001),(.25,-.005,0),(.25,.0001,0)])
def test_invalid_Q_is_rejected(diagonal):
    with pytest.raises(SignatureError):
        g.physical_axes(np.diag(diagonal)[None],np.array([[0.,0.,1.]]))


def test_refinement_uses_union_across_roles_with_shared_captures():
    pixels=np.array([17,30,310])
    fetched=[]
    def fetch(rows,grid,prefix):
        shape=(len(g.ROLES),len(rows),3,3)
        rgb=np.ones(shape,np.float64)
        if prefix=='base' and grid==g.BASE[1]:
            rgb[1,1,1]+=.005  # largest absolute failure
            rgb[2,0,0]+=.004
        rgb[:,:,2]=rgb[:,:,0]+rgb[:,:,1]
        fetched.append((prefix,grid))
        return dict(pixels=pixels[rows],grid=np.array(grid),components=rgb,visibility=np.zeros(len(rows),np.int64))
    rgb,_,resolution,done=g.refine(pixels,fetch,0)
    assert done==[30,17] and len(fetched)==6
    np.testing.assert_array_equal(rgb[0],rgb[1])
    np.testing.assert_array_equal(resolution,[(*[g.HIGH[0]]*2,g.BASE[0]),(*[g.HIGH[1]]*2,g.BASE[1])])


def test_pooled_failure_and_already_refined_failure():
    pix=np.array([3,7])
    rgb=np.full((2,len(g.ROLES),2,3,3),.1)
    rgb[1,0,1,2]+=.001
    assert g.next_pixel(pix,rgb,[])==7
    rgb[1,0,1,2]+=.003
    with pytest.raises(SignatureError,match='Refined absolute'):
        g.next_pixel(pix,rgb,[7])


def test_capture_rejects_inconsistent_components_or_pixel_order():
    rgb=np.ones((len(g.ROLES),2,3,3),np.float64)
    grid=dict(pixels=np.array([7,8]),grid=np.array(256),components=rgb,visibility=np.array([3,4]))
    with pytest.raises(SignatureError,match='Invalid grid capture'):
        g.verify_grid(np.array([7,8]),256,grid)
    rgb[:,:,2]=2
    with pytest.raises(SignatureError,match='pixel identity'):
        g.verify_grid(np.array([8,7]),256,grid)


def test_fixed_unit_linear_metrics_and_explicit_true_black_raster():
    truth=np.ones((2,3))*.5
    prediction=truth+.1
    m=score(prediction,truth)
    assert m['relative_L2']==pytest.approx(.2) and m['PSNR_unit_peak_dB']==pytest.approx(20.)
    px=[19,111]
    full=score(raster(prediction,px),raster(truth,px))
    assert full['relative_L2']==pytest.approx(m['relative_L2'])
    assert full['rmse']==pytest.approx(m['rmse']*np.sqrt(2/g.SIZE**2))
    assert score(truth,truth)['exact'] and score(np.zeros_like(truth),np.zeros_like(truth))['relative_L2'] is None


def test_cohorts_do_not_admit_extra_foreground_to_primary_support():
    v=dict(pixel_indices=np.array([5,12]),matched_pixels=np.array([12]),material_region=np.array([0,1]),
           nearly_isotropic=np.array([True,False]),near_boundary=np.array([False,True]),
           normals=np.array([[0.,0.,1.],[0.,0.,1.]]),outgoing=np.array([[0.,0.,1.],[0.,0.,1.]]),
           highlight_mask=np.array([True,False]))
    original={k:np.array([True]) for k in ('other_fitting_view','no_other_fitting_view',
                                          'nearby_highlight','covered_without_highlight')}
    masks=scopes(v,np.array([True,False]),original)
    np.testing.assert_array_equal(np.flatnonzero(masks['extra_foreground']),[5])
    np.testing.assert_array_equal(np.flatnonzero(masks['common']),[12])
    assert not masks['relight_highlight'].any() and masks['full_raster'].all()


def test_matched_tape_schedule_and_work_ceiling():
    seeds=[g.tape(c,b,d) for c,n in zip(g.CAMERAS,g.COUNTS)
           for b in range(0,n,g.CHUNK) for d in range(g.DRAWS)]
    assert len(set(seeds))==len(seeds)==480
    assert len({s for pair in seeds for s in pair})==960
    assert g.LIMITS['native_shader_calls']==len(g.ARMS)*len(seeds)
    assert g.LIMITS['independent_BRDF_point_directions']==len(g.ROLES)*g.LIMITS['independent_shadow_directions']


def test_real_frozen_mapped_support_and_prior_fields_without_checkpoint_loads():
    f=g.freeze(g.digest(g.ROOT/g.FREEZE))
    paths=g.base_paths(f)
    _,views,archives=g.admit(f,paths,g.endpoint_paths(f))
    for c,v in views.items():
        rows=np.searchsorted(v['pixel_indices'],v['matched_pixels'])
        for arm in g.ARMS:
            a=archives[arm,c]
            key='Q_world' if g.TENSOR_ARMS[arm] is None else 'Q_tensor_direct'
            widths,_,_=g.physical_axes(a[key][a['hit_mask']],v['normals'][rows])
            expected=np.sort(np.concatenate([a[k][a['hit_mask']] for k in ('mx','my')],axis=1),axis=1)[:,::-1]
            np.testing.assert_allclose(widths,expected,atol=g.MATERIAL_ATOL,rtol=0)
    assert len({p for p in g.base_paths(f,True).values()})==len(paths)
    assert all('checkpoint-10000.pth'==paths[m['checkpoint']].name for m in f['methods'].values())
    assert f['training_authorized'] is False


def test_inherited_quadrature_solid_angle_and_cosine_controls():
    from experiments.bell_quadrature_e11e import controls

    assert controls()['status']=='pass'
