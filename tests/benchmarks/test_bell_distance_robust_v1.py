import numpy as np
import pytest

from benchmarks import bell_distance_robust_v1 as b
from benchmarks.donut_surface_q_d6 import surface_q_term
from infrastructure.signatures import SignatureError


def test_geometry_cap_and_median_normalization():
    raw = np.array([1.,10.,1000.])
    weights = b.weighting_definitions(.02/np.sqrt(raw),np.array([.01]),2)
    np.testing.assert_allclose(b.weight_values(raw,weights['B1']),[1.,10.,40.])
    np.testing.assert_allclose(b.weight_values(raw,weights['B2']),[.4,4.,16.])
    with pytest.raises(SignatureError):
        b.multipliers(np.array([0.]))


def test_same_sampler_retains_shortest_pair_and_scale_preserves_shares():
    domain = dict(distance=np.r_[.00001,np.full(127,.01),.01],training_ids=np.arange(128),
                  diagnostic_ids=np.array([128]))
    reference = dict(distance=np.array([.02]),training_ids=np.array([0]))
    summary,arrays = b.replay(domain,reference,{},steps=3)
    assert summary['shortest_pair_steps']==[1,2,3] and summary['capped_edge_count']==1
    assert summary['capped_draw_count']==3
    np.testing.assert_array_equal(arrays['max_share_pair_id_B2'],[0,0,0])
    np.testing.assert_allclose(arrays['max_share_B1'],arrays['max_share_B2'])
    assert arrays['max_share_D6'].min()>.99
    assert arrays['max_share_B1'].max()<.04


def test_weighted_term_preserves_transport_loss_and_autograd():
    torch = pytest.importorskip('torch')
    mx = torch.tensor([[.35],[.40],[.25],[.30]],dtype=torch.float64,requires_grad=True)
    normals = torch.tensor([[0.,0.,1.],[0.,0.,1.]],dtype=mx.dtype)

    class Fixture:
        def predict_anisotropic_components(self, pts):
            width = mx[pts]
            return width,width/2,None,None,width.new_tensor([[1.,0.],[1.,0.]]),None

    batch = dict(points0=torch.tensor([0,1]),points1=torch.tensor([2,3]),normals0=normals,normals1=normals,
                 distance=mx.new_tensor([.02,.001]))
    plain = surface_q_term(Fixture(),batch,b.bell.SPATIAL_POLICY,torch)
    uncapped,_ = surface_q_term(Fixture(),batch,b.bell.SPATIAL_POLICY,torch,distance_weighting=dict(cap=1000.,scale=1.))
    torch.testing.assert_close(plain,uncapped,rtol=0,atol=0)
    weighted,edges = surface_q_term(Fixture(),batch,b.bell.SPATIAL_POLICY,torch,distance_weighting=dict(cap=4.,scale=.5))
    contrast = (mx[:2,0].square()-mx[2:,0].square()).square()*(1+1/16)
    expected = .05*(contrast*mx.new_tensor([.5,2.])).mean()
    torch.testing.assert_close(weighted,expected)
    torch.testing.assert_close(edges.sum(),expected)
    a = torch.autograd.grad(weighted,mx,retain_graph=True)[0]
    z = torch.autograd.grad(expected,mx)[0]
    torch.testing.assert_close(a,z)
    assert not edges.requires_grad
    log = b.edge_diagnostic(np.array([20,19]),edges,dict(mx=mx,my=mx/2),dict(shortest_pair_id=19),torch)
    assert log['max_pair_id']==19 and log['shortest_contribution']==log['max_contribution']
    assert log['minor_at_floor']==0 and np.shape(log['minor_major_min_mean_max'])==(3,2)


def test_weighted_history_rejects_edge_outside_sample():
    domain = dict(distance=np.full(129,.01),training_ids=np.arange(128),diagnostic_ids=np.array([128]))
    f = dict(arms={'B1':dict(distance_weighting=dict(shortest_pair_id=0))})
    row = dict(surface_q_loss=.2,preclip_gradient_l2=.1,edge_diagnostic=dict(
        max_contribution=.02,max_pair_id=3,shortest_contribution=.01,minor_at_floor=0,
        minor_major_min_mean_max=[[.1,.2],[.2,.3],[.3,.4]]))
    assert b.audit_weighted_history(f,domain,'B1',[row])[0]['update']==1
    row['edge_diagnostic']['max_pair_id']=128
    with pytest.raises(SignatureError,match='Invalid weighted diagnostic'):
        b.audit_weighted_history(f,domain,'B1',[row])


def test_native_cli_dispatches_weighted_mode_only_when_selected(monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments import train_bell_heterogeneous_v1 as cli

    observed = []
    monkeypatch.setattr(b,'freeze',lambda path,sha:dict(base_initial=dict(sha256='initial')))
    monkeypatch.setitem(sys.modules,'experiments.donut_material_recovery_v1',
                        SimpleNamespace(train=lambda args,**kw:observed.append((args.phase,kw))))
    common = ['train','--phase','B1','--freeze-sha256','f','--execute']
    for name in ('freeze','oracle','benchmark','admission','verification','initial','arm-initial','output',
                 'authorization','authorization-sha256','commit','image-id'):
        common.extend(['--'+name,'fixture'])
    monkeypatch.setattr(sys,'argv',common)
    with pytest.raises(SignatureError,match='Arm outside'):
        cli.main()
    monkeypatch.setattr(sys,'argv',common+['--distance-robust'])
    cli.main()
    assert observed==[('B1',dict(acquisition=b))]
