import numpy as np
import pytest

from benchmarks import donut_heterogeneous_v1 as d
from experiments.generate_donut_heterogeneous_v1 import compare, incident_grid, slope_pair
from infrastructure.signatures import SignatureError


def torus(u,v):
    u,v=np.broadcast_arrays(u,v)
    radial=.35+.14*np.cos(v)
    return (np.stack([radial*np.cos(u),radial*np.sin(u),.14*np.sin(v)],axis=-1),
            np.stack([np.cos(v)*np.cos(u),np.cos(v)*np.sin(u),np.sin(v)],axis=-1))


def test_roughness_is_a_real_boundary_without_albedo_or_director_jump():
    p,n=torus(np.array([np.pi/4-1e-8,np.pi/4+1e-8]),.2)
    m=d.material_field(p,n,.35)
    assert m['material_region'].tolist()==[0,1]
    assert m['near_boundary'].tolist()==[True,True]
    np.testing.assert_allclose(m['widths_major_minor'][0]-m['widths_major_minor'][1],[.12,.058],atol=1e-8)
    np.testing.assert_allclose(m['color'][0],m['color'][1],atol=1e-8)
    np.testing.assert_allclose(m['metallic'][0],m['metallic'][1],atol=1e-8)
    np.testing.assert_allclose(m['axis'][0],m['axis'][1],atol=1e-7)


def test_periodic_weak_patch_and_tensor_eigenvalues():
    u=np.array([0.,np.pi-1e-9,-np.pi+1e-9,.4,1.1])
    p,n=torus(u,.5)
    m=d.material_field(p,n,.35)
    assert m['nearly_isotropic'].tolist()==[True,False,False,False,False]
    np.testing.assert_allclose(m['widths_major_minor'][0],[.223,.214],atol=1e-15)
    np.testing.assert_allclose(m['Q_world'][1],m['Q_world'][2],atol=1e-9)
    np.testing.assert_allclose(np.linalg.eigvalsh(m['Q_world'])[:,1:],m['widths_major_minor'][:,::-1]**2,atol=1e-15)
    np.testing.assert_allclose(np.einsum('ijk,ik->ij',m['Q_world'],n),0,atol=1e-15)


def test_area_weights_are_invariant_to_duplicate_measurements_per_triangle():
    area=np.array([.01,.02,.03])
    w,coverage=d.seen_area_weights(np.array([0,2,2,2]),area)
    np.testing.assert_allclose(w,[.01,.01,.01,.01])
    assert coverage==pytest.approx(2/3)
    with pytest.raises(SignatureError,match='triangle'):
        d.seen_area_weights(np.array([3]),area)


def test_slope_and_incident_interfaces_keep_all_points_including_grazing():
    torch=pytest.importorskip('torch')
    p,n=torus(np.array([0.,.9,1.1]),.5)
    material=d.material_field(p,n,.35)
    wo=n.copy()
    wo[1]=.001*n[1]+np.sqrt(1-.001**2)*material['axis'][1]
    wo[2]=-n[2]
    view=dict(material,position=p,normals=n,outgoing=wo,shader_valid=np.array([True,True,False]))
    counts={}
    def charge(k,v):counts[k]=counts.get(k,0)+v
    class OpenHemisphere:
        def trace(self,origin,direction):
            assert origin.dtype==direction.dtype==torch.float32
            return origin,direction,torch.full(direction.shape[:-1],10.)
    c,e,vis=incident_grid(view,8,OpenHemisphere(),torch,charge,device='cpu')
    assert counts==dict(shadow_directions=128,BRDF_point_directions=128)
    assert vis.tolist()==[64,64,0]
    assert c.shape==(3,3,3) and np.array_equal(c[:,0]+c[:,1],c[:,2]) and (c[2]==0).all()
    slope=slope_pair(view,charge)
    assert counts['slope_nodes_attempted']==counts['slope_nodes_returned']==2*d.NODES_PER_POINT
    assert (slope[128]['total'][2]==0).all()
    np.testing.assert_array_equal(slope[128]['diffuse'][:2],material['color'][:2]*(1-material['metallic'][:2]))
    assert np.isfinite(e).all() and np.isfinite(slope[128]['total']).all()


def test_convergence_fails_on_substantive_difference():
    truth=np.ones((8,3))*.1
    normal=np.tile([0.,0.,1.],(8,1))
    rgb=np.ones((8,3,3))*.1
    v=dict(shader_valid=np.ones(8,dtype=bool),normals=normal,outgoing=normal,
           linear_components=rgb,coarse_components=rgb.copy(),reflectance=truth,coarse_reflectance=truth,
           incident_reflectance=truth,coarse_incident_reflectance=truth)
    assert all(g['passed'] for g in compare(v).values())
    v['coarse_components'][0,2,0]+=.004
    assert not compare(v)['radiance_convergence']['passed']
