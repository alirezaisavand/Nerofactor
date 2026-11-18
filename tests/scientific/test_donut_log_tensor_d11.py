import numpy as np
import pytest
import torch

from benchmarks.donut_log_tensor_d11 import archived_log_square, coefficient
from infrastructure.signatures import SignatureError
from infrastructure.surface_q_regularization import normal_transport
from scientific_variants.donut_log_tensor_d11 import tangent_log, world_log_tensor, surface_log_q_term
from scientific_variants.donut_tensor_d8 import frame, chart_q


def spd_reference(a):
    vals, axes = np.linalg.eigh(a)
    return (axes*np.log(vals)[...,None,:]) @ axes.swapaxes(-1,-2)


def test_independent_log_reference_and_normal_component():
    q = torch.tensor([[[.1225,.009],[.009,.0064]],
                      [[.0016,0],[0,1.]],[[.0625,0],[0,.0625]]],dtype=torch.double)
    normals = torch.nn.functional.normalize(torch.tensor([[0.,0,1],[1.,2,3],[0.,1,0]],dtype=q.dtype))
    u = frame(normals)
    a = u @ q @ u.transpose(-1,-2) + normals[:,:,None]*normals[:,None,:]
    result = world_log_tensor(q,u)
    np.testing.assert_allclose(spd_reference(a.numpy()),result.numpy(),rtol=3e-13,atol=3e-13)
    assert (result@normals[:,:,None]).abs().max() <= 2e-14


def test_gauge_axis_swap_and_director_sign():
    n = torch.tensor([[.6,0.,.8]],dtype=torch.double)
    u = frame(n)
    a = torch.tensor([[.35]],dtype=n.dtype)
    b = torch.tensor([[.08]],dtype=n.dtype)
    s, c = np.sin(.5),np.cos(.5)
    rot = torch.tensor([[c,-s],[s,c]],dtype=n.dtype)
    basis = u@rot
    director = .6*u[...,0]+.8*u[...,1]
    orthogonal = -.8*u[...,0]+.6*u[...,1]
    expected = world_log_tensor(chart_q(a,b,director,u),u)
    for wa, wb, axis in ((a,b,-director),(b,a,orthogonal),(b,a,-orthogonal)):
        assert torch.allclose(world_log_tensor(chart_q(wa,wb,axis,basis),basis),expected,rtol=0,atol=2e-14)


def test_normal_transport_covariance_and_proportional_width_changes():
    n = torch.tensor([[0.,0,1.],[.6,0,.8]],dtype=torch.double)
    u = frame(n)
    r = normal_transport(n[0:1], n[1:2], torch)
    ua = u[1:2].transpose(-1,-2) @ r @ u[0:1]
    for width in (.06,.25,.65):
        q0 = torch.diag_embed(torch.tensor([[width**2,(width*.8)**2]],dtype=n.dtype))
        q1 = 1.1**2 * ua @ q0 @ ua.transpose(-1,-2)
        b0, b1 = world_log_tensor(q0,u[0:1]),world_log_tensor(q1,u[1:2])
        measured = (b1-r@b0@r.transpose(-1,-2)).square().sum()
        assert float(measured) == pytest.approx(8*np.log(1.1)**2,abs=2e-13)
        measured_archive,_ = archived_log_square(
            np.array([[[width,width*.8],[width*1.1,width*.8*1.1]]]),
            (q1-ua@q0@ua.transpose(-1,-2)).square().sum((-1,-2)).numpy())
        assert measured_archive[0] == pytest.approx(float(measured),abs=2e-13)


@pytest.mark.parametrize('entries', [
    [.0625,.0625,0.],
    [.0625,.062500000001,1e-13],
    [.00160005,.99999995,1e-6],
    [.00160003,.00160002,1e-9],
])
def test_gradients_at_isotropy_and_width_bounds(entries):
    x = torch.tensor(entries,dtype=torch.double,requires_grad=True)
    def from_entries(a):
        q = torch.stack([a[0],a[2],a[2],a[1]]).reshape(1,2,2)
        return tangent_log(q)
    assert torch.autograd.gradcheck(from_entries,(x,),eps=1e-7,atol=3e-5,rtol=2e-6)
    assert torch.isfinite(torch.autograd.grad(from_entries(x).sum(),x)[0]).all()


def test_archived_noncommuting_axes_and_bad_input():
    rng = np.random.RandomState(35)
    s = rng.uniform(.04,1,size=(64,2,2))
    a = rng.uniform(-np.pi,np.pi,size=(64,2))
    c,t = np.cos(a),np.sin(a)
    rot = np.stack((c,-t,t,c),axis=-1).reshape(64,2,2,2)
    q = (rot*s[...,None,:]**2)@rot.swapaxes(-1,-2)
    raw = np.square(q[:,0]-q[:,1]).sum((1,2))
    ref = spd_reference(q)
    result,report = archived_log_square(s,raw)
    np.testing.assert_allclose(result,np.square(ref[:,0]-ref[:,1]).sum((1,2)),atol=1e-10)
    assert not report['raw_square_clamped']
    with pytest.raises(SignatureError,match='inconsistent'):
        archived_log_square(s,raw+100)
    with pytest.raises(SignatureError,match='positive definite'):
        tangent_log(torch.tensor([[[-1.,0.],[0.,.25]]]))
    assert coefficient(4.,16.)==.0125


def test_training_loss_uses_both_endpoints_and_distance_scale():
    p=torch.tensor([[.15,0,0],[.2,0,0]],dtype=torch.double)
    n=torch.tensor([[0.,0,1.],[0.,0.,1.]],dtype=p.dtype)
    class Predictor:
        def predict_tensor(self,points,normals):
            q=torch.diag_embed(torch.stack((points[:,0],points[:,0]*.5),-1))
            return q,frame(normals),None,None
    a,b=p.clone().requires_grad_(),(1.1*p).requires_grad_()
    loss=surface_log_q_term(Predictor(),
        dict(points0=a,points1=b,normals0=n,normals1=n,distance=torch.tensor([.01,.02],dtype=p.dtype)),
        dict(coefficient=.02,reference_distance=.02),torch)
    assert float(loss) == pytest.approx(.02*2*np.log(1.1)**2*2.5)
    assert all(torch.isfinite(g).all() and (g[:,0]!=0).all() for g in torch.autograd.grad(loss,(a,b)))
