import numpy as np
import pytest

from experiments import material_field_expressivity_v2 as x
from infrastructure.signatures import SignatureError


def test_geometry_rule_is_bounded_and_anisotropic_in_grid_shape():
    xyz = np.array([[0.,0.,0.],[1.,1.,.29]])
    line = np.linspace(.01,.99,101)
    points = np.stack([line,line*.7,line*.1],axis=1)
    result = x.grid_definition(xyz,points)
    levels = result['levels']
    assert levels[1]['pitch']>=3*result['sparse_nn_median']
    assert levels[1]['long_axis_cells']==2*levels[0]['long_axis_cells']
    assert result['additional_parameters']==sum(p['parameters'] for p in levels)<=result['parameter_cap']
    assert all(p['cells'][2]<p['cells'][0] for p in levels)


def test_local_interpolation_is_zero_initially_and_preserves_affine_values():
    torch = pytest.importorskip('torch')
    from scientific_variants.localized_ambient_tensor_v1 import LocalTensorResidual

    grid = dict(lower=[0.,0.,0.],levels=[dict(cells=[2,2,1],pitch=.5),dict(cells=[4,4,2],pitch=.25)])
    branch = LocalTensorResidual(grid)
    points = torch.tensor([[0.,0.,0.],[1.,1.,.5],[.1,.3,.2],[.38,.53,.3]])
    assert torch.eq(branch(points),0).all()
    branch(points).sum().backward()
    for table in branch.tables:
        assert torch.count_nonzero(table.grad)>0
        assert table.grad.sum().item()==pytest.approx(len(points)*6)
    with torch.no_grad():
        coeff = torch.tensor([1.,2.,-4.])
        for i,table in enumerate(branch.tables):
            _,_,z,y,x = table.shape
            h = grid['levels'][i]['pitch']
            zz,yy,xx = torch.meshgrid(torch.arange(z),torch.arange(y),torch.arange(x),indexing='ij')
            value = (torch.stack([xx,yy,zz],-1)@coeff.to(torch.int64))*h
            table.copy_(value.expand_as(table))
    expected = 2*(points@coeff)[:,None].expand(len(points),6)
    torch.testing.assert_close(branch(points),expected,atol=1e-6,rtol=0)
    with pytest.raises(SignatureError,match='outside frozen'):
        branch(torch.tensor([[2.,0.,0.]]))


def test_zero_residual_preserves_d8_tensor_and_material_and_is_trainable():
    torch = pytest.importorskip('torch')
    from experiments.neural_material_field_v1 import source_definitions
    definitions,_ = source_definitions()
    grid = dict(lower=[-1.,-1.,-1.],levels=[dict(cells=[2,2,2],pitch=1.),dict(cells=[4,4,4],pitch=.5)])
    base = x.make_field(torch,definitions,6033,False,grid)
    local = x.make_field(torch,definitions,6033,True,grid)
    p = torch.tensor([[.2,.1,.2],[0.,.25,0.]])
    n = torch.tensor([[0.,0.,1.],[1.,0.,0.]])
    keys = base.state_dict().keys()
    assert local.global_field.state_dict().keys()==keys
    assert all(torch.equal(base.state_dict()[k],local.global_field.state_dict()[k]) for k in keys)
    bn = base.predict_tensor(p,n)
    ln = local.predict_tensor(p,n)
    assert all(torch.equal(u,v) for u,v in zip(bn,ln))
    target = torch.tensor([[.2,.03],[.03,.09]],dtype=torch.float64)
    (ln[0]-target).square().mean().backward()
    assert all(torch.isfinite(t.grad).all() and torch.count_nonzero(t.grad)>0 for t in local.local.tables)


def test_spacing_strata_are_fixed_and_include_empty_and_open_bins():
    q = np.broadcast_to(np.eye(3),(4,3,3)).copy()
    held = dict(Q_world=q,near_boundary=np.array([False,True,False,True]),
                nearly_isotropic=np.array([False,False,True,False]))
    pred = dict(Q_world=q*np.array([1,2,3,4])[:,None,None])
    scores = x.local_spacing_scores(held,pred,np.array([.001,.012,.033,.057]),.01)
    assert scores['all']['spearman']==1
    assert [u['count'] for u in scores['all']['bins']]==[1,0,1,1,1]
    assert scores['all']['bins'][-1]['interval']==[4,None]
    assert scores['all']['bins'][-1]['Q_median']==pytest.approx(3.)
    assert scores['smooth_interior']['spearman'] is None


def test_work_includes_two_local_levels_and_no_radiometry():
    work = x.expected_work(dict(donut={'evaluation_points':8576},bell={'evaluation_points':8704}))
    assert work['updates']==32000 and work['fitting_points']==8192000
    assert work['illumination_updates']==work['RGB_renders']==0
    assert work['localized_point_levels']==work['fitting_points']+work['evaluation_points']
