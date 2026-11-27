"""Focused synthetic CPU controls only; no Bell rendering, queries or weights."""
import numpy as np
import pytest
from benchmarks import bell_geometry_rgb_gate12 as g
from benchmarks.bell_sam2_ordering_gate12 import camera_order
from experiments.bell_geometry_rgb_gate12 import integrate,validate_capture
from experiments.verify_bell_geometry_rgb_gate12 import gates_equal


def test_support_keeps_unmatched_material_undefined():
    a=np.array([1,1,0,0,1],bool);b=np.array([1,0,1,0,1],bool);valid=np.array([1,0,0,0,0],bool)
    c=g.support(a,b,valid)
    assert c['known'].tolist()==[True,True,False,True,False]
    assert c['oracle_only'].sum()==c['reconstruction_only'].sum()==1 and c['undefined'].sum()==2
    with pytest.raises(ValueError):g.support(a,b,np.ones(5,bool))


def test_transport_and_isotropy_preserve_the_physical_tensor():
    n=np.array([[0.,0.,1.],[0.,0.,1.]])
    m=np.array([[0.,.6,.8],[.6,0.,.8]])
    axis=np.array([[1.,0.,0.],[1.,0.,0.]]);width=np.array([[.4,.2],[.3,.3]])
    q=g.tensor(axis,width,n);r=g.minimal_rotation(n,m);a=np.einsum('nij,nj->ni',r,axis)
    np.testing.assert_allclose(g.tensor(a,width,m),g.rotate_tensor(q,r),atol=2e-16)
    np.testing.assert_allclose(g.tensor(-axis,width,n),q,atol=0)
    np.testing.assert_allclose(g.tensor(np.cross(n,axis),width[:,::-1],n),q,atol=0)
    np.testing.assert_allclose((g.rotate_tensor(q,r)*m[:,None]).sum(2),0,atol=1e-16)


def test_primary_field_is_the_original_global_projected_tensor_not_ray_labels():
    p=np.array([[.2,.3,.4],[-.1,.1,.2]]);n=np.array([[0.,0.,1.],[.6,0.,.8]])
    bounds=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
    values=g.analytic(p,n,bounds);P=np.eye(3)-n[:,:,None]*n[:,None,:]
    grain=np.array([1.,2.,3.])/np.sqrt(14);a=.23+.03*p[:,0];b=.35+.03*p[:,1]
    ambient=a[:,None,None]**2*np.eye(3)+b[:,None,None]**2*np.outer(grain,grain)
    np.testing.assert_allclose(values['Q'],P@ambient@P,atol=3e-17,rtol=1e-14)
    eigen=np.linalg.eigvalsh(values['Q'])[:,1:][:,::-1]
    np.testing.assert_allclose(eigen,values['widths_major_minor']**2,atol=3e-17)


def test_toy_integrator_visibility_substitution_and_components():
    import torch
    class Tracer:
        def __init__(self,blocked):self.blocked=blocked
        def trace(self,o,d):
            depth=torch.ones(d.shape[:-1],dtype=d.dtype)*(.5 if self.blocked else 100.)
            return o,d,depth
    v=dict(pixels=np.array([7]),position=np.array([[0.,0.,0.]]),normals=np.array([[0.,0.,1.]]),
        outgoing=np.array([[0.,0.,1.]]),source_position=np.array([[0.,0.,0.]]),axis=np.array([[1.,0.,0.]]),
        widths_major_minor=np.array([[.4,.2]]),color=np.array([[.3,.4,.5]]),metallic=np.array([[.2]]))
    v.update({f'transported_{k}':v[k] for k in ('axis','widths_major_minor','color','metallic')})
    counters=dict(shadow_directions=0,BRDF_point_directions=0,environment_point_directions=0)
    def charge(k,n):counters[k]+=n
    with torch.no_grad():a=integrate(v,4,Tracer(False),Tracer(True),False,torch,charge,.0068,'cpu')
    validate_capture(v,4,a)
    assert counters==dict(shadow_directions=32,BRDF_point_directions=32,environment_point_directions=16)
    assert (a['components'][:,0]>0).all() and (a['components'][:,1]==0).all()
    np.testing.assert_array_equal(a['components'][:,0],a['components'][:,2])
    np.testing.assert_array_equal(a['components'][:,0],a['components'][:,3])
    assert a['visibility'][0,:5].tolist()==[16,0,16,16,0]


def test_refinement_ranks_any_component_and_refuses_a_failed_refined_point():
    x=np.full((2,2,len(g.KINDS),3,3),.1);x[1,1,2,1,0]+=.004
    states={a:dict(pixels=np.array([5,9]),components=x.copy()) for a in g.ARMS}
    assert g.choose(states,[])==9
    with pytest.raises(ValueError,match='Refined'):g.choose(states,[9])


def test_paired_refinement_preserves_missing_counterparts_and_exact_budget():
    views={a:dict(pixels=np.array([5,9]) if a!='O0' else np.array([9])) for a in g.ARMS};calls=[]
    def fetch(arm,v,size,prefix):
        ids=v['pixels'];x=np.full((len(ids),len(g.KINDS),3,3),.1);x[:,:,2]=.2
        if size==512:
            x[ids==5,:,1]+=.01;x[ids==5,:,2]+=.01
        calls.append((arm,prefix,size,ids.tolist()))
        return dict(components=x,visibility=np.zeros((len(ids),13)),pixels=ids,grid=np.array(size))
    states,selected=g.paired(views,fetch,0)
    assert selected==[5]
    assert all(a!='O0' for a,p,s,i in calls if p!='base')
    np.testing.assert_array_equal(states['O1']['grids'],[[1024,256],[2048,512]])
    with pytest.raises(ValueError,match='budget exhausted'):g.paired(views,fetch,g.MAX_TOTAL)


def test_convergence_decisions_exact_and_only_norm_has_roundoff_allowance():
    a=dict(passed=True,maximum_rgb_difference=.001,relative_l2=.0005)
    gates_equal(dict(a,relative_l2=.0005+1e-18),a)
    with pytest.raises(ValueError):gates_equal(dict(a,passed=False),a)
    with pytest.raises(ValueError):gates_equal(dict(a,maximum_rgb_difference=.0010000000001),a)


def test_calibration_only_order_keeps_anchors_and_no_camera_replacement():
    ids=[0,42,21,127];theta=np.array([0.,.3,.6,.9]);centers=np.stack([np.cos(theta),np.sin(theta),np.zeros(4)],-1)
    pose=np.tile(np.eye(3,4),(4,1,1));pose[:,:,3]=-centers
    a=camera_order(ids,pose);b=camera_order(ids,pose)
    assert a==b and a['frame_to_camera']==['0','42','21','127']
    assert a['angular_order']['sum']<a['numerical_order']['sum']
