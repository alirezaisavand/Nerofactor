import numpy as np
import pytest

from benchmarks import bell_robust_completion_v1 as completion
from experiments import material_field_expressivity_v1 as expr
from infrastructure.signatures import SignatureError


def test_bell_completion_rejects_a_donut_arm():
    with pytest.raises(SignatureError,match='no other arm'):
        completion.Acquisition('Donut-R1')


def test_reports_existing_raw_orientation_guard_without_fallback():
    torch = pytest.importorskip('torch')
    from experiments.neural_material_field_v1 import source_definitions

    definitions,_ = source_definitions()
    field = expr.build_field(torch,definitions,6033,'D8-C')
    points = torch.tensor([[.1,.2,.3]])
    with torch.no_grad():
        field.rotation_predictor[6].bias.zero_()
        field.mx_predictor[6].bias.fill_(.2)
    with pytest.raises(ValueError,match='Undefined zero/near-zero orientation') as failure:
        field.predict_anisotropic_components(points)
    captured = completion.orientation_failure(failure.value,29)
    assert captured['pending_update']==29 and captured['offending_rows']==1
    row = captured['rows'][0]
    assert row['point_native']==points[0].tolist() and row['raw_orientation_norm']==0
    x,y = np.square(row['chart_widths'])
    assert row['anisotropy_gap']==pytest.approx(abs(x-y)/(x+y))
    assert completion.orientation_failure(ValueError('unrelated'),1) is None
    from benchmarks.donut_surface_q_d6 import POLICY, surface_q_term
    normal = torch.tensor([[0.,0.,1.]])
    batch = dict(points0=points,points1=points,normals0=normal,normals1=normal,distance=torch.ones(1))
    with pytest.raises(ValueError,match='Undefined zero/near-zero orientation') as spatial_failure:
        surface_q_term(field,batch,POLICY,torch,robust_delta=.03)
    assert completion.orientation_failure(spatial_failure.value,29)['surface_endpoint']=='0'


def test_actual_field_components_start_at_the_same_q_and_receive_gradients():
    torch = pytest.importorskip('torch')
    from experiments.neural_material_field_v1 import source_definitions
    from scientific_variants.donut_tensor_d8 import world_tensor

    defs,_ = source_definitions()
    p = torch.tensor([[.2,.1,.2],[0.,.1,0.]])
    n = torch.tensor([[0.,0.,1.],[1.,0.,0.]])
    target = torch.diag(p.new_tensor([.15,.04,.08])).double()
    fields = [expr.build_field(torch,defs,6033,arm) for arm in expr.DESIGN['arms']]
    states = [field.feats_network.state_dict() for field in fields]
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    predictions = []
    for field in fields:
        assert not hasattr(field,'outer_light') and not hasattr(field,'inner_light')
        q,u,_,_ = field.predict_tensor(p,n)
        world = world_tensor(q,u)
        predictions.append(world)
        loss = (world-target).square().sum((1,2)).mean()
        loss.backward()
        final = field.mx_predictor[6].bias.grad
        assert torch.isfinite(final).all() and torch.linalg.vector_norm(final)>0
        assert all(v.grad is None for v in field.kd_predictor.parameters())
    torch.testing.assert_close(*predictions,rtol=0,atol=expr.DESIGN['initial_Q_atol'])


def test_boundary_scoring_uses_signed_transported_jumps():
    torch = pytest.importorskip('torch')
    n = np.tile([0.,0.,1.],(2,1,1))
    q = np.stack((np.diag([.4,.08,0.]),np.diag([.3,.04,0.])))[:,None]
    w = np.sqrt(np.stack((q[...,0,0],q[...,1,1]),-1))
    data = dict(normals=n,Q_world=q,widths=w)
    recovered = expr.boundary_scores(torch,data,dict(Q_world=q,widths=w))
    reversed_jump = expr.boundary_scores(torch,data,dict(Q_world=q[::-1],widths=w[::-1]))
    assert recovered['Q_jump_projection_median']==pytest.approx(1.)
    assert reversed_jump['Q_jump_projection_median']==pytest.approx(-1.)


def test_area_sampler_lies_on_selected_faces():
    geo = dict(vertices_native=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]),
               triangles=np.array([[0,1,2],[3,4,5]]))

    def target(p,n):
        return dict(Q_world=np.zeros((len(p),3,3)),axis=n,widths_major_minor=np.zeros((len(p),2)),anisotropy_gap=np.zeros(len(p)),
                    near_boundary=np.ones(len(p),bool),nearly_isotropic=np.ones(len(p),bool))

    result = expr.sample_surface(geo,np.array([.5,.5]),np.tile([0.,0.,1.],(2,1)),np.array([1]),100,4,target)
    np.testing.assert_array_equal(result['face'],np.ones(100,dtype=int))
    np.testing.assert_allclose(result['position'][:,2],1.)
    assert (result['position'][:,:2]>=0).all() and (result['position'][:,:2].sum(1)<=1).all()
