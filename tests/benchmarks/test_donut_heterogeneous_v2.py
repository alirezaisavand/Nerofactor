import numpy as np
import pytest

from benchmarks import donut_heterogeneous_v2 as d
from experiments.generate_donut_heterogeneous_v2 import capture
from infrastructure.signatures import SignatureError


def components(values):
    x=np.broadcast_to(np.asarray(values,dtype=np.float64)[:,None],(len(values),3))
    return np.stack([x/4,3*x/4,x],1)


def view(deltas,pixels=None):
    n=len(deltas)
    return dict(pixel_indices=np.arange(n) if pixels is None else np.asarray(pixels),
                shader_valid=np.ones(n,dtype=bool),
                linear_components=components([.4]*n),coarse_components=components(.4-np.asarray(deltas)),
                visible_counts=np.full(n,10,dtype=np.int64),incident_reflectance=np.full((n,3),.3),
                coarse_incident_reflectance=np.full((n,3),.3),reflectance=np.full((n,3),.3))


def patch(pixels, delta=0):
    pix=np.asarray(pixels)
    n=len(pix)
    return dict(pixels=pix,grids=np.asarray(d.HIGH),
                components=np.stack([components([.4-delta]*n),components([.4]*n)]),
                incident_reflectance=np.full((2,n,3),.3),visible_counts=np.full((2,n),100,dtype=np.int64))


def test_replay_chooses_actual_camera50_witness_coordinates():
    assert d.PROBE_PIXELS==[57*128+21,62*128+19,6577]
    v=view([.0001,.007,.0006],[22,7317,7955])
    calls=[]
    def fetch(row):
        pix=v['pixel_indices'][row]
        calls.append(pix)
        return patch([pix])
    out,q=d.assemble(v,d.reused_coarse_counts(v),fetch)
    assert calls==[7317]
    assert q['refined_pixels'].tolist()==[7317]
    assert q['coarse_grid'].tolist()==[256,1024,256]
    assert q['fine_grid'].tolist()==[512,2048,512]
    assert q['coarse_visible_counts'].tolist()==[-1,100,-1]
    for k in ('linear_components','coarse_components','reflectance','visible_counts'):
        np.testing.assert_array_equal(out[k][[0,2]],v[k][[0,2]])
    assert d.rgb_gate(out)['passed']
    assert not d.rgb_gate(v)['passed']


def test_pooled_only_failure_refines_largest_residual_and_stops_early():
    v=view([.0009,.0012,.0009],[4,7,11])
    assert not d.rgb_gate(v)['passed']
    assert d.rgb_gate(v)['maximum_rgb_difference']<.003
    out,q=d.assemble(v,d.reused_coarse_counts(v),lambda i:patch(v['pixel_indices'][[i]]))
    assert q['refined_pixels'].tolist()==[7]
    assert d.rgb_gate(out)['passed']


def test_invalid_and_passing_rows_are_preserved_and_never_called():
    v=view([.0001,0])
    v['shader_valid'][1]=False
    v['linear_components'][1]=v['coarse_components'][1]=0
    v['visible_counts'][1]=0
    def unexpected(_):
        pytest.fail('Attempted scientific work for a passing view')
    out,q=d.assemble(v,d.reused_coarse_counts(v),unexpected)
    for key in v:
        np.testing.assert_array_equal(v[key],out[key])
    assert q['refined_pixels'].size==0
    assert q['fine_grid'].tolist()==[512,0]
    assert q['coarse_visible_counts'].tolist()==[-1,0]


def test_ties_use_lowest_pixel_id_and_cap_is_checked_before_fetch(monkeypatch):
    v=view([.006,.006],[9,5])
    calls=[]
    def fetch(row):
        calls.append(int(v['pixel_indices'][row]))
        return patch([calls[-1]])
    monkeypatch.setattr(d,'MAX_PER_VIEW',1)
    with pytest.raises(SignatureError,match='budget exhausted'):
        d.assemble(v,d.reused_coarse_counts(v),fetch)
    assert calls==[5]


def test_refined_absolute_failure_stops_without_replacing_other_points():
    v=view([.006,.008])
    calls=[]
    def fetch(row):
        calls.append(row)
        return patch([row],delta=.005)
    with pytest.raises(SignatureError,match='Refined absolute RGB'):
        d.assemble(v,d.reused_coarse_counts(v),fetch)
    assert calls==[1]


def test_diagnostic_capture_order_cache_and_fail_flag():
    obs=view([0,0,0],d.PROBE_PIXELS)
    for k in ('position','normals','outgoing','axis','color','widths_major_minor','metallic'):
        obs[k]=np.zeros((3,3))
    calls=[]
    def render(p,grid):
        calls.append(grid)
        return components([.4]*3),np.full((3,3),.2),np.full(3,10,dtype=np.int64)
    data=capture(obs,np.arange(3),render)
    assert calls==list(d.HIGH)
    assert d.diagnostic_gates(data)['RGB']['passed']
    p=d.probe_slice(data,d.PROBE_PIXELS[1])
    d.check_capture(np.array([d.PROBE_PIXELS[1]]),p)
    np.testing.assert_array_equal(p['components'],data['components'][:,1:2])
    data['components'][1,0]=components([.410])[0]
    assert not d.diagnostic_gates(data)['RGB']['passed']


def test_exact_work_caps_include_diagnostic_and_four_remaining_views():
    assert d.NEW_POINTS==15458
    assert d.counters(0)['shadow_directions']==5081006080
    assert d.counters(128)['shadow_directions']==d.LIMITS['shadow_directions']==5752094720
    assert d.counters(128)['slope_nodes_attempted']==1266319360
    with pytest.raises(SignatureError,match='cap'):
        d.counters(129)


def test_camera0_copy_rechecks_highlights_and_keeps_exact_bytes(tmp_path):
    import json
    from benchmarks.donut_recovery_v1 import numeric
    from benchmarks.generate_donut_recovery_v1 import highlight_record
    from experiments.generate_donut_heterogeneous_v1 import previews
    from experiments.generate_donut_heterogeneous_v2 import finalize_view
    from experiments.verify_donut_heterogeneous_v2 import validate_physical_view

    v=view([.0001]*8)
    v.update(normals=np.tile([0.,0.,1.],(8,1)),outgoing=np.tile([0.,0.,1.],(8,1)),
             color=np.full((8,3),.25),metallic=np.full((8,1),.2),
             reflectance_diffuse=np.full((8,3),.2),reflectance_specular=np.full((8,3),.1),
             coarse_reflectance=v['reflectance'].copy())
    _,v['highlight_mask']=highlight_record(v['linear_components'])
    prior,out=tmp_path/'prior',tmp_path/'out'
    prior.mkdir();out.mkdir()
    p=prior/'view-0.npz'
    np.savez_compressed(p,**v)
    images=previews(prior,0,v)
    (prior/'generation.json').write_text(json.dumps(dict(views=[dict(camera=0,previews=images)])))
    assembled,q=d.assemble(v,d.reused_coarse_counts(v),lambda _:pytest.fail('unexpected refinement'))
    row=finalize_view(0,assembled,q,out,prior)
    assert out.joinpath(p.name).read_bytes()==p.read_bytes()
    for tag in ('RGB','diffuse','specular'):
        name=f'view-0-{tag}.png'
        assert out.joinpath(name).read_bytes()==prior.joinpath(name).read_bytes()
    assert not row['selected_pixels'] and row['radiance_convergence']['passed']
    validate_physical_view(numeric(out/p.name),numeric(out/'quadrature-0.npz'))
