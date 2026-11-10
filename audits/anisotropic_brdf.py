"""Independent NumPy analytical references, never used by research rendering.

Ideal continuous distributions are distinct from finite frozen sample patterns,
float32 clamps and conditional estimators. No proposed scientific fix is wired
into the renderer. Directions point away from the surface, n=(0,0,1).
"""
import numpy as np


def unit(v,eps=1e-12):
    v=np.asarray(v,np.float64)
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),eps)


def frame(theta=0.):
    # Frozen base at +Z: n cross +X = +Y, n cross +Y = -X.
    t=np.array([0.,1.,0.]);b=np.array([-1.,0.,0.])
    return np.cos(theta)*t+np.sin(theta)*b,-np.sin(theta)*t+np.cos(theta)*b


def ndf(h,mx,my,theta=0.):
    """Beckmann projected-area NDF from independent Cartesian slope density."""
    h=np.asarray(h);c=h[...,2];positive=c>0;z=np.maximum(c,1e-300);t,b=frame(theta)
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        slope_exponent=(h@t/(mx*z))**2+(h@b/(my*z))**2
        logD=-slope_exponent-np.log(np.pi*mx*my)-4*np.log(z)
        result=np.exp(logD)
    return np.where(positive,result,0.)


def fresnel(c,f0):return f0+(1-f0)*(1-np.clip(c,0,1))**5


def kernel(wi,wo,mx,my,theta=0.,f0=1.):
    """Reciprocal specular kernel inferred before accepted-count normalization.

    This is Kurt alpha=0, ks=1 in the ideal unclamped domain. It is NOT the
    total finite renderer estimator and does not impose a Cook-Torrance G term.
    """
    wi=np.asarray(wi);wo=np.asarray(wo);h=unit(wi+wo);dot=h@wo
    valid=(wi[...,2]>0)&(wo[...,2]>0)&(dot>0)
    return np.where(valid,fresnel(dot,f0)*ndf(h,mx,my,theta)/(4*np.maximum(dot,1e-300)),0.)


def directional_pdf(wi,wo,mx,my,theta=0.):
    """Upper-reflection subdensity. Integral=P(accepted), generally below one."""
    h=unit(np.asarray(wi)+wo);dot=h@wo
    return np.where((np.asarray(wi)[...,2]>0)&(dot>0),ndf(h,mx,my,theta)*h[...,2]/(4*np.maximum(dot,1e-300)),0.)


def hemisphere(n_mu=192,n_phi=384):
    c,w=np.polynomial.legendre.leggauss(n_mu);c=(c+1)/2;w=w/2
    phi=(np.arange(n_phi)+.5)*2*np.pi/n_phi
    z,az=np.meshgrid(c,phi,indexing='ij');r=np.sqrt(1-z*z)
    return np.stack([r*np.cos(az),r*np.sin(az),z],-1).reshape(-1,3),np.broadcast_to(w[:,None]*2*np.pi/n_phi,z.shape).reshape(-1)


def half_samples(u,v,mx,my,theta=0.):
    """Elliptical Gaussian slopes, independent of original spherical inversion."""
    r=np.sqrt(-np.log(np.asarray(u)));phi=2*np.pi*np.asarray(v);t,b=frame(theta)
    slopes=mx*r[...,None]*np.cos(phi)[...,None]*t+my*r[...,None]*np.sin(phi)[...,None]*b
    return unit(slopes+np.array([0.,0.,1.]))


def reflect(wo,h):
    return unit(2*np.clip(h@wo,0,1)[...,None]*h-wo)


def acceptance(wo,mx,my,theta=0.,n_phi=16384):
    """Independent 1-D integration of a Gaussian slope-domain acceptance circle."""
    wo=np.asarray(wo);co=wo[2]
    if co==0:return .5
    if co<0:raise ValueError('Upper outgoing hemisphere required')
    t,b=frame(theta);phi=(np.arange(n_phi)+.5)*2*np.pi/n_phi
    x=mx*np.cos(phi);y=my*np.sin(phi);A=x*x+y*y;B=(wo@t)*x+(wo@b)*y
    discriminant=np.sqrt(B*B+co*co*A);radial=np.empty_like(B);pos=B>=0
    radial[pos]=(discriminant[pos]+B[pos])/(co*A[pos])
    radial[~pos]=co/(discriminant[~pos]-B[~pos])
    return float(np.mean(-np.expm1(-radial**2)))


def diffuse_density(c):
    """Continuum generating density of frozen sample_sphere -> elevation map.

    Per solid angle, ignoring explicit 1e-7 length perturbations. The actual
    finite code uses a golden-angle radial pattern, not independent random pairs.
    """
    c=np.asarray(c)
    return np.where((c>=0)&(c<=1),.5*c*np.cos(np.pi*c*c/2),0.)


def observations():
    """Fixed analytical counterexamples, not fitted material or thesis scores."""
    wi,w=hemisphere();mx,my=.6,.9;normal=np.array([0.,0.,1.]);oblique=unit([1.,0.,.25])
    p0=acceptance(normal,mx,my);p1=acceptance(oblique,mx,my)
    base=float(kernel(oblique,normal,mx,my));reverse=float(kernel(normal,oblique,mx,my))
    rows=[]
    for label,wo in [('normal',normal),('oblique',oblique)]:
        prob=acceptance(wo,mx,my)
        for metallic,kd in [(.001,.999),(.5,.7),(.999,.999)]:
            f0=.04*(1-metallic)+metallic*kd;d=kd*(1-metallic)
            rho=float(np.sum(kernel(wi,wo,mx,my,f0=f0)*wi[:,2]*w))
            rows.append({'view':label,'mx':mx,'my':my,'metallic':metallic,'kd':kd,'F0':f0,'valid_probability':prob,
                         'unconditioned_specular_reflectance':rho,'conditional_specular_reflectance':rho/prob,
                         'unconditioned_constant_light_total':d+rho,
                         'actual_constant_light_total_limit':d+rho/prob})
    return {'schema_version':1,'protocol':'anisotropic-brdf-analytical-v1','quadrature':{'mu':192,'phi':384,'acceptance_phi':16384},
        'scope':'Independent ideal-domain CPU mathematics; source float32 checks live in focused tests, no native results',
        'specular_reciprocity':{'kernel_forward':base,'kernel_reverse':reverse,'P_normal':p0,'P_oblique':p1,
            'conditional_forward':base/p0,'conditional_reverse':reverse/p1},
        'diffuse_cosine_squared_illumination':{'Lambertian':.5,'frozen_continuum':1-2/np.pi},
        'directional_energy':rows}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    with Path(args.output).open('x') as stream:json.dump(observations(),stream,indent=2,allow_nan=False);stream.write('\n')
