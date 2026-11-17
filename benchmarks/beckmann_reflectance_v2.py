"""Independent float64 Kurt a=0 reflectance in Beckmann Gaussian slope space.

No Torch, native shader, renderer, lighting, geometry or visibility calls.
Diffuse is analytic. Specular uses dOmega_i=4*(wo.h)*dOmega_h and
dOmega_h=cos(h)^3*dsx*dsy; see the v2 protocol for the full Jacobian.
"""
from functools import lru_cache
import numpy as np
from infrastructure.signatures import require

ORDERS = (64,128)  # radial nodes and nodes in EACH of four azimuth quadrants
NODES_PER_POINT = sum(4*n*n for n in ORDERS)


@lru_cache(maxsize=8)
def nodes(order):
    require(type(order) is int and order>=2,'Integer quadrature order>=2 required')
    x,w=np.polynomial.legendre.leggauss(order)
    u=(x+1)/2;wu=w/2
    phi=np.concatenate([(x+1)*np.pi/4+k*np.pi/2 for k in range(4)])
    wp=np.tile(w*np.pi/4,4)
    for a in (u,wu,phi,wp):a.setflags(write=False)
    return u,wu,phi,wp


def radial_domain(local_views,widths,phi):
    """Positive root of c+2*A*r-c*B*r^2>0, without negative-root cancellation."""
    wo=np.asarray(local_views,dtype=np.float64);m=np.asarray(widths,dtype=np.float64)
    angle=np.asarray(phi,dtype=np.float64)
    require(wo.ndim==2 and wo.shape[1]==3 and m.shape==(len(wo),2) and
        angle.ndim==2 and angle.shape[0]==len(wo) and np.isfinite(wo).all() and
        np.isfinite(m).all() and np.isfinite(angle).all() and (wo[:,2]>0).all() and
        abs(np.linalg.norm(wo,axis=1)-1).max()<=2e-12 and ((m>=.04)&(m<=1)).all(),
        'Unit upper-hemisphere views and supported positive Beckmann widths required')
    c=wo[:,2:3];cos=np.cos(angle);sin=np.sin(angle)
    A=m[:,0:1]*wo[:,0:1]*cos+m[:,1:2]*wo[:,1:2]*sin
    B=(m[:,0:1]*cos)**2+(m[:,1:2]*sin)**2
    root=np.sqrt(A*A+c*c*B);rmax=np.empty_like(A);positive=A>=0
    numerator=A+root;denominator=np.broadcast_to(c,B.shape)*B
    rmax[positive]=numerator[positive]/denominator[positive]
    rmax[~positive]=np.broadcast_to(c,A.shape)[~positive]/(root[~positive]-A[~positive])
    require(np.isfinite(rmax).all() and (rmax>0).all(),'Invalid exact radial support')
    return A,B,rmax


def integrate(local_views,widths,color,metallic,order,batch=8,charge=None):
    """Return analytic diffuse + independent specular directional reflectance.

    phi is shifted ONLY by the known local view direction; quadrant boundaries
    align with A=0, without response-dependent adaptation. u=1-exp(-r^2)
    maps the complete valid radial support to a finite interval. No finite tail
    cutoff, accepted-sample renormalization or reflectance clipping is used.
    charge(nodes,returned) observes work before/after each successful CPU batch.
    """
    wo=np.asarray(local_views,dtype=np.float64);m=np.asarray(widths,dtype=np.float64)
    kd=np.asarray(color,dtype=np.float64);metal=np.asarray(metallic,dtype=np.float64).reshape(-1,1)
    require(len(wo)>0 and kd.shape==(len(wo),3) and metal.shape==(len(wo),1) and
        np.isfinite(kd).all() and np.isfinite(metal).all() and ((kd>=0)&(kd<=1)).all() and
        ((metal>=0)&(metal<=1)).all() and type(batch) is int and batch>0,'Invalid reflectance material/batch')
    u,wu,phi,wp=nodes(order);specular=np.empty_like(kd);mass=np.empty(len(wo))
    minimum_ci=np.inf
    for start in range(0,len(wo),batch):
        end=min(start+batch,len(wo));v=wo[start:end];width=m[start:end]
        base=np.arctan2(width[:,1]*v[:,1],width[:,0]*v[:,0])
        angles=base[:,None]+phi[None]
        A,B,rmax=radial_domain(v,width,angles)
        cost=(end-start)*4*order*order
        if charge is not None:charge(cost,False)
        umax=-np.expm1(-rmax*rmax)
        radial_u=umax[:,:,None]*u[None,None]
        require(((radial_u>0)&(radial_u<1)).all(),'Quadrature nodes must stay inside radial CDF')
        radius=np.sqrt(-np.log1p(-radial_u))
        ch=1/np.sqrt(1+B[:,:,None]*radius*radius)
        oh=(v[:,2,None,None]+A[:,:,None]*radius)*ch
        ci=2*(v[:,2,None,None]+A[:,:,None]*radius)*ch*ch-v[:,2,None,None]
        require(np.isfinite(ci).all() and (ci>0).all() and (ci<=1+2e-12).all() and
                (oh>0).all() and (oh<=1+2e-12).all(),'Reflected direction left the analytic upper support')
        # Fresnel cosine handling is identical to the independent incident
        # reference. It is not clipping of reflectance or of integration weights.
        fresnel5=(1-np.clip(oh,0,1))**5
        probability=umax[:,:,None]*wu[None,None]*wp[None,:,None]/(2*np.pi)
        weight=ci/ch*probability
        I0=weight.sum((1,2));I5=(weight*fresnel5).sum((1,2))
        f0=.04*(1-metal[start:end])+metal[start:end]*kd[start:end]
        specular[start:end]=f0*I0[:,None]+(1-f0)*I5[:,None]
        mass[start:end]=probability.sum((1,2))
        minimum_ci=min(minimum_ci,float(ci.min()))
        require(np.isfinite(specular[start:end]).all() and (specular[start:end]>=0).all(), 'Nonfinite/negative slope-space reflectance')
        if charge is not None:charge(cost,True)
    diffuse=kd*(1-metal)
    return dict(diffuse=diffuse,specular=specular,total=diffuse+specular,
                valid_halfvector_probability=mass,minimum_sampled_incident_cosine=minimum_ci,
                nodes=len(wo)*4*order*order)


def local_views(outgoing,normals,major_axis):
    """Known world frame to local coordinates; no normal flips or transforms."""
    wo,n,a=(np.asarray(x,dtype=np.float64) for x in (outgoing,normals,major_axis))
    require(wo.shape==n.shape==a.shape and wo.ndim==2 and wo.shape[1]==3 and
        all(np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<=2e-12 for x in (wo,n,a)) and
        abs((n*a).sum(-1)).max()<=2e-12,'Orthogonal unit world reference frame required')
    b=np.cross(n,a)
    return np.stack([(wo*a).sum(-1),(wo*b).sum(-1),(wo*n).sum(-1)],-1)
