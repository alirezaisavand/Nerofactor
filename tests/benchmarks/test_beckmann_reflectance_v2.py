"""Independent synthetic controls, not execution of the five real witnesses."""
import numpy as np
import pytest
from scipy.integrate import quad
from benchmarks.beckmann_reflectance_v2 import integrate,radial_domain,nodes,local_views


def test_reflection_jacobian_matches_finite_difference_surface_area():
    wo=np.array([.3,.2,np.sqrt(.87)])
    slope=np.array([.15,-.2]);eps=1e-6
    def reflect(s):
        h=np.r_[s,1.];h/=np.linalg.norm(h)
        return 2*(wo@h)*h-wo
    dx=(reflect(slope+[eps,0])-reflect(slope-[eps,0]))/(2*eps)
    dy=(reflect(slope+[0,eps])-reflect(slope-[0,eps]))/(2*eps)
    h=np.r_[slope,1.];h/=np.linalg.norm(h)
    assert np.linalg.norm(np.cross(dx,dy))==pytest.approx(4*(wo@h)*h[2]**3,rel=2e-9)


def test_exact_radial_domain_including_grazing_and_negative_A():
    c=np.array([1.,.6,1e-8]);wo=np.stack([np.sqrt(1-c*c),np.zeros(3),c],-1)
    widths=np.tile([.3,.2],(3,1));phi=np.tile(np.linspace(0,2*np.pi,31,endpoint=False),(3,1))
    A,B,r=radial_domain(wo,widths,phi)
    def cosine(radius):return 2*(c[:,None]+A*radius)/(1+B*radius**2)-c[:,None]
    assert np.max(abs(cosine(r)))<1e-14
    assert (cosine(r*.99)>0).all() and (cosine(r*1.01)<0).all()
    assert (A[2]<0).any() and np.isfinite(r).all()


def test_probability_weights_and_analytic_diffuse():
    wo=np.array([[0.,0,1]]);width=np.array([[.5,.5]])
    r=integrate(wo,width,np.array([[.2,.4,.6]]),np.array([.3]),32)
    np.testing.assert_allclose(r['diffuse'],[[.14,.28,.42]],atol=1e-16)
    assert r['valid_halfvector_probability'][0]==pytest.approx(1-np.exp(-4),abs=1e-14)
    u,wu,phi,wp=nodes(32)
    assert wu.sum()==pytest.approx(1,abs=1e-14) and wp.sum()==pytest.approx(2*np.pi,abs=1e-14)
    assert ((u>0)&(u<1)).all()


def test_normal_isotropic_perfect_Fresnel_against_independent_1D_integral():
    alpha=.5
    exact=quad(lambda r:2*r*np.exp(-r*r)*(1-alpha*alpha*r*r)/np.sqrt(1+alpha*alpha*r*r),0,1/alpha,
               epsabs=1e-12,epsrel=1e-12)[0]
    result=integrate(np.array([[0.,0,1.]]),np.array([[alpha,alpha]]),np.ones((1,3)),np.ones(1),64)
    np.testing.assert_allclose(result['total'],exact,atol=2e-10)


def incident_reference(wo,width,color,metal,size=256):
    # Direct, independently evaluated BRDF in incident solid angle; no slope
    # density/Jacobian cancellation from the implementation under test.
    z=(np.arange(size)+.5)/size;phi=(np.arange(size)+.5)*2*np.pi/size
    zz,pp=np.meshgrid(z,phi,indexing='ij');wi=np.stack([np.sqrt(1-zz**2)*np.cos(pp),np.sqrt(1-zz**2)*np.sin(pp),zz],-1).reshape(-1,3)
    h=wi+wo;h/=np.linalg.norm(h,axis=-1,keepdims=True)
    D=np.exp(-((h[:,0]/width[0])**2+(h[:,1]/width[1])**2)/h[:,2]**2)/(np.pi*np.prod(width)*h[:,2]**4)
    oh=h@wo;f0=.04*(1-metal)+metal*color
    F=f0+(1-f0)*(1-np.clip(oh,0,1))[:,None]**5
    spec=F*D[:,None]/(4*oh[:,None])
    return color*(1-metal)+(spec*wi[:,2,None]).sum(0)*2*np.pi/len(wi)


@pytest.mark.parametrize('angle',[0.,.5,1.])
def test_independent_incident_hemisphere_at_ordinary_synthetic_angles(angle):
    wo=np.array([np.sin(angle),0,np.cos(angle)]);m=np.array([.4,.3]);color=np.array([.2,.5,.8]);metal=.6
    a=integrate(wo[None],m[None],color[None],np.array([metal]),64)['total'][0]
    b=incident_reference(wo,m,color,metal)
    np.testing.assert_allclose(a,b,atol=2e-4,rtol=0)


def test_axis_swap_and_director_sign_equivalence():
    wo=np.array([[.3,.4,np.sqrt(.75)]]);width=np.array([[.4,.2]]);color=np.array([[.4,.5,.6]]);metal=np.array([.7])
    a=integrate(wo,width,color,metal,32)['total']
    b=integrate(wo*np.array([-1,-1,1]),width,color,metal,32)['total']
    c=integrate(wo[:,[1,0,2]]*np.array([1,-1,1]),width[:,::-1],color,metal,32)['total']
    np.testing.assert_allclose(a,b,atol=2e-14);np.testing.assert_allclose(a,c,atol=2e-14)


def test_world_frame_conversion_and_explicit_work_accounting():
    wo=np.array([[.6,0,.8]]);normal=np.array([[0.,0,1.]]);axis=np.array([[0.,1.,0.]])
    local=local_views(wo,normal,axis);np.testing.assert_allclose(local,[[0,-.6,.8]])
    calls=[]
    integrate(local,np.array([[.3,.2]]),np.full((1,3),.5),np.array([.5]),8,charge=lambda n,returned:calls.append((n,returned)))
    assert calls==[(4*8**2,False),(4*8**2,True)]


def test_invalid_domain_fails_without_repair_or_charge():
    calls=[]
    with pytest.raises(ValueError):
        integrate(np.array([[1.,0,-.1]]),np.array([[.3,.2]]),np.full((1,3),.5),np.array([.5]),8,
                  charge=lambda *x:calls.append(x))
    assert calls==[]
