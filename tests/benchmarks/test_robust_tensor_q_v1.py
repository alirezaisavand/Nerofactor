import sys

import numpy as np
import pytest

from benchmarks import robust_tensor_q_v1 as robust
from benchmarks.donut_surface_q_d6 import POLICY, surface_q_term
from infrastructure.signatures import SignatureError
from infrastructure.surface_q_regularization import tensor_from_axes, transported_q_loss, world_q


def test_penalty_and_derivative_match_declared_formula():
    torch = pytest.importorskip('torch')
    delta = .03
    s = torch.tensor([0., 1e-12, delta**2, .3, 1e4],dtype=torch.float64,requires_grad=True)
    rho,multiplier = robust.terms(s,delta,torch)
    expected = np.sqrt(1+s.detach().numpy()/delta**2)
    np.testing.assert_allclose(rho.detach(),2*s.detach().numpy()/(expected+1),rtol=2e-15,atol=0)
    np.testing.assert_allclose(multiplier.detach(),1/expected,rtol=2e-15,atol=0)
    derivative, = torch.autograd.grad(rho.sum(),s)
    torch.testing.assert_close(derivative,multiplier,atol=0,rtol=2e-14)
    assert derivative[0]==1 and derivative[1]>.99999999
    assert rho[-1]/(2*delta*s[-1].sqrt())>.999 and derivative[-1]<.001
    with pytest.raises(SignatureError,match='Positive finite'):
        robust.terms(s,0.,torch)


def test_robust_residual_gradient_is_zero_at_agreement_and_bounded_at_large_disagreement():
    torch = pytest.importorskip('torch')
    delta = .0085
    residual = torch.tensor([0., 1e-9, 1000.],dtype=torch.float64,requires_grad=True)
    rho,_ = robust.terms(residual.square(),delta,torch)
    gradient, = torch.autograd.grad(rho.sum(),residual)
    assert gradient[0]==0 and gradient[1]==pytest.approx(2e-9)
    assert gradient[2]==pytest.approx(2*delta,rel=1e-9)


@pytest.mark.parametrize('delta',[.030024479708747484,.008516476041898608])
def test_float32_stress_keeps_penalty_and_autograd_finite(delta):
    torch = pytest.importorskip('torch')
    limit = torch.finfo(torch.float32).max
    s = torch.tensor([0.,limit],dtype=torch.float32,requires_grad=True)
    rho,multiplier = robust.terms(s,delta,torch)
    derivative, = torch.autograd.grad(rho.sum(),s)
    assert torch.isfinite(rho).all() and torch.isfinite(derivative).all()
    torch.testing.assert_close(derivative,multiplier,rtol=1e-5,atol=0)
    assert (multiplier>0).all()
    loss,quad,edges,summary = robust.reduce(s[:1],delta,.05,torch)
    assert loss==quad==0 and edges.item()==0
    torch.testing.assert_close(summary,s.new_tensor([1.,1.,1.,0.]),rtol=torch.finfo(s.dtype).eps,atol=0)


def test_transported_tensor_gauge_and_rigid_rotation_invariance():
    torch = pytest.importorskip('torch')
    dtype = torch.float64
    n = torch.tensor([[0.,0.,1.],[0.,.6,.8]],dtype=dtype)
    widths = torch.tensor([[.4,.12],[.36,.08]],dtype=dtype)
    angles = torch.tensor([.23,-.39],dtype=dtype)

    def predicted(a,w):
        return world_q(w[:,:1],w[:,1:],torch.stack((torch.cos(2*a),torch.sin(2*a)),1),n,torch)

    q = predicted(angles,widths)
    alternate = [predicted(angles+torch.pi,widths),predicted(angles+torch.pi/2,widths.flip(1))]
    t = torch.stack((torch.zeros(2,dtype=dtype),n[:,2],-n[:,1]),1)
    b = torch.cross(n,t,dim=1)
    alternate.append(tensor_from_axes(widths[:,:1],widths[:,1:],
        angles.cos()[:,None]*t+angles.sin()[:,None]*b,-angles.sin()[:,None]*t+angles.cos()[:,None]*b))
    for other in alternate:
        torch.testing.assert_close(q,other,atol=2e-16,rtol=0)

    def penalty(q,n):
        _,s = transported_q_loss(q[:1],q[1:],n[:1],n[1:],n.new_ones(1),torch)
        return robust.terms(4*s,.03,torch)[0]

    R = n.new_tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    torch.testing.assert_close(penalty(q,n),penalty(R@q@R.T,n@R.T),atol=1e-16,rtol=0)


def test_loss_acts_after_distance_weighting_and_leaves_legacy_route():
    torch = pytest.importorskip('torch')
    mx = torch.tensor([[.35],[.4],[.25],[.3]],dtype=torch.float64,requires_grad=True)
    n = mx.new_tensor([[0.,0.,1.],[0.,0.,1.]])

    class Shader:
        def predict_anisotropic_components(self, points):
            w = mx[points]
            return w,w/2,None,None,mx.new_tensor([[1.,0.],[1.,0.]]),None

    batch = dict(points0=torch.tensor([0,1]),points1=torch.tensor([2,3]),normals0=n,normals1=n,
                 distance=mx.new_tensor([.02,.001]))
    weighting = dict(cap=4.,scale=.5)
    loss,quadratic,edges,stats = surface_q_term(Shader(),batch,POLICY,torch,distance_weighting=weighting,robust_delta=.03)
    original,original_edges = surface_q_term(Shader(),batch,POLICY,torch,distance_weighting=weighting)
    torch.testing.assert_close(quadratic,original,rtol=0,atol=0)
    expected = robust.terms(original_edges/(.05/2),.03,torch)[0]
    torch.testing.assert_close(edges,expected*.05/2)
    torch.testing.assert_close(edges.sum(),loss)
    assert not edges.requires_grad and not stats.requires_grad
    recovered, = torch.autograd.grad(loss,mx)
    assert torch.isfinite(recovered).all() and torch.count_nonzero(recovered)==len(mx)
    assert loss<original and 0<stats[2]<=stats[1]<1


def test_calibration_filters_both_endpoints_and_uses_the_applied_weights():
    ids = np.arange(34)
    points = np.zeros((34,3))
    other = points.copy()
    other[-2,0] = .011
    normals = np.tile([0.,0.,1.],(34,1))
    end = normals.copy()
    end[-1] *= -1
    d = dict(diagnostic_ids=ids,points0=points,points1=other,normals0=normals,normals1=end,
             distance=np.full(len(ids),.01))
    square = np.square(ids+1).astype(float)
    chosen = robust.calibrate(d,square,points[:1],normals[:1],POLICY,dict(cap=2.,scale=.5))
    assert chosen['selected_pair_ids']==list(range(32))
    assert chosen['delta']==np.quantile(np.arange(1,33),.9)
    with pytest.raises(SignatureError,match='Insufficient'):
        robust.calibrate(d,square,points[:1],-normals[:1],POLICY,None)


def test_transition_bins_keep_sides_and_empty_observations():
    distance = np.array([0.,.005,.01,.035,.036])
    region = np.array([0,0,1,1,1])
    truth = np.tile([.3,.2],(len(region),1))
    rows = robust.width_profiles(distance,region,truth+.01,truth,np.ones(len(region)))
    assert [r['observations'] for r in rows]==[1,1,0,0,0,0,1,1]
    assert rows[6]['signed_distance_interval']==[-.02,-.01]
    assert rows[5]['observation'] is None
    np.testing.assert_allclose(rows[7]['observation']['signed_width_error'],[.01,.01])


@pytest.mark.parametrize('arm,faceted',[('Donut-R1',False),('Bell-R1',True)])
def test_native_cli_dispatches_selected_arm_with_explicit_execution(monkeypatch,arm,faceted):
    from experiments import donut_material_recovery_v1, train_robust_tensor_q_v1 as cli

    monkeypatch.setattr(cli.h.Acquisition,'freeze',lambda *args: dict(base_initial={'sha256':'fixture'}))
    seen = []

    def train(args,*,acquisition):
        seen.append((args.phase,args.initial_sha256,args.benchmark_sha256,acquisition.FACETED))

    monkeypatch.setattr(donut_material_recovery_v1,'train',train)
    argv = ['runner','--phase',arm,'--freeze','freeze','--freeze-sha256','sha',
            '--oracle','mesh','--benchmark','benchmark','--admission','admission','--verification','verify',
            '--initial','init','--arm-initial','arm-init','--output','output','--authorization','auth',
            '--authorization-sha256','auth-sha','--commit','commit','--image-id','image','--execute']
    if not faceted:
        argv += ['--cameras','cameras']
    monkeypatch.setattr(sys,'argv',argv)
    cli.main()
    assert seen==[(arm,'fixture',cli.h.Acquisition(arm).GENERATION,faceted)]
