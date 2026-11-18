"""D8 tangent tensor math, shared by both representations."""

import math

import torch
import torch.nn.functional as F

from infrastructure.signatures import require

LOW, HIGH = .04**2, 1.
MID, SPAN = (HIGH+LOW)/2, (HIGH-LOW)/2
INITIAL_WIDTH = .25


def frame(normal):
    ref=F.one_hot(normal.abs().argmin(-1),3).to(normal)
    t=F.normalize(torch.cross(normal,ref,dim=-1),dim=-1)
    b=F.normalize(torch.cross(normal,t,dim=-1),dim=-1)
    return torch.stack((t,b),dim=-1)


def symmetric(xx, yy, xy):
    return torch.stack((xx,xy,xy,yy),dim=-1).reshape(*xx.shape,2,2)


def bounded_local(x):
    """Apply f(z)=z/sqrt(1+z²) to a symmetric 2x2 matrix without eigenvectors."""
    x=x.double()
    a,d,b=x[...,0,0],x[...,1,1],x[...,0,1]
    center=(a+d)/2
    radius=torch.linalg.norm(torch.stack(((a-d)/2,b),dim=-1),dim=-1)
    p,m=center+radius,center-radius
    u,v=torch.sqrt(1+p.square()),torch.sqrt(1+m.square())
    prod=p*m
    same=prod>=0
    # uv-pm via rationalization for like signs, avoiding catastrophic subtraction.
    safe=torch.where(same,u*v+prod,torch.ones_like(prod))
    residual=torch.where(same,(1+p.square()+m.square())/safe,u*v-prod)
    divided=(1+residual)/(u*v*(u+v))
    average=(p/u+m/v)/2
    iso=MID+SPAN*average
    stretch=SPAN*divided
    return symmetric(iso+stretch*(a-d)/2,iso-stretch*(a-d)/2,stretch*b)


def ambient_q(logits, tangent, *, initial_width=INITIAL_WIDTH):
    xx,yy,zz,xy,xz,yz=logits.double().unbind(-1)
    h=torch.stack((xx,xy,xz,xy,yy,yz,xz,yz,zz),-1).reshape(-1,3,3)
    u=tangent.double()
    local=u.transpose(-1,-2)@h@u
    y=(initial_width**2-MID)/SPAN
    offset=y/math.sqrt(1-y*y)
    eye=torch.eye(2,device=u.device,dtype=u.dtype)
    # Analytic projection of offset*I. Keep floating-point frame error out of
    # the isotropic anchor and express zero as the exact common tensor.
    base=eye*offset
    anchor=bounded_local(base)
    return bounded_local((local+local.transpose(-1,-2))/2+base)+(eye*initial_width**2-anchor)


def chart_q(mx,my,axis,tangent):
    # Trace + deviator form gives the exact same isotropic tensor for any chart.
    coordinate=(axis.double().unsqueeze(-2)@tangent.double()).squeeze(-2)
    coordinate=F.normalize(coordinate,dim=-1)
    x,y=coordinate.unbind(-1)
    a=mx[:,0].double().square()
    b=my[:,0].double().square()
    mean=(a+b)/2
    gap=(a-b)/2
    diag=gap*(x.square()-y.square())
    return symmetric(mean+diag,mean-diag,2*gap*x*y)


def factor(q):
    """Batched principal square root, inverse, and log sqrt(det), in float64."""
    q=q.double()
    a,c,b=q[...,0,0],q[...,1,1],q[...,0,1]
    det=a*c-b.square()
    trace=a+c
    require(bool((torch.isfinite(q).all() & (det>0).all() & (trace>0).all())),
            'Invalid tangent covariance')
    s=det.sqrt()
    root=symmetric(a+s,c+s,b)/(trace+2*s).sqrt()[...,None,None]
    inverse=symmetric(c,a,-b)/det[...,None,None]
    return root,inverse,s.log()


def slope_log_ndf(slope, cosine, inverse, log_det):
    quad=((slope@inverse)*slope).sum(-1,keepdim=True)
    return -quad-math.log(math.pi)-log_det[:,None,None]-4*cosine.log()


def ndf(q,tangent,normal,half):
    """Direct Beckmann D for unit half-vectors. Lower support is exactly zero."""
    _,inverse,log_det=factor(q)
    h, n = half.double(), normal.double()
    cosine=(h*n[:,None]).sum(-1,keepdim=True)
    # On the unit sphere with eigenvalues <=1, D at c<=1/32 underflows
    # even in float64. Mask before division to keep horizon gradients finite.
    support=cosine>1/32
    safe=torch.where(support,cosine,torch.ones_like(cosine))
    s=(h@tangent.double())/safe
    result=slope_log_ndf(s,safe,inverse,log_det).exp()
    return torch.where(support,result,torch.zeros_like(result)).to(half.dtype)


def sample(q,tangent,normal,outgoing,gaussian,*,density=False):
    """A shared world Gaussian tape transformed into slopes and reflected rays."""
    root,inverse,log_det=factor(q)
    slope=((gaussian @ tangent) @ root.to(gaussian.dtype))/math.sqrt(2)
    half=F.normalize(slope @ tangent.transpose(-1,-2)+normal[:,None],dim=-1)
    dot=(outgoing[:,None]*half).sum(-1,keepdim=True)
    reflected=2*dot*half-outgoing[:,None]
    cosine=(half*normal[:,None]).sum(-1,keepdim=True)
    # Historical specular prior detaches D before applying its sample softmax.
    with torch.no_grad():
        log_d=slope_log_ndf(slope.double(),cosine.double(),inverse.detach(),log_det.detach())
        d=log_d.exp().to(half.dtype)
        pdf=None
        if density:
            support=dot!=0
            safe=torch.where(support,dot.abs(),torch.ones_like(dot)).double()
            value=(log_d+cosine.double().log()-math.log(4)-safe.log()).exp()
            pdf=torch.where(support,value,torch.zeros_like(value)).to(half.dtype)
    return half,reflected,cosine,d,pdf


def brdf(q,tangent,normal,wo,wi,kd,metallic):
    h=F.normalize(wi+wo[:,None],dim=-1)
    oh=(h*wo[:,None]).sum(-1,keepdim=True)
    valid=((wi*normal[:,None]).sum(-1,keepdim=True)>0)&((wo*normal).sum(-1)[:,None,None]>0)&(oh>0)
    safe=torch.where(valid,oh,torch.ones_like(oh))
    f0=(.04*(1-metallic)+metallic*kd)[:,None]
    fresnel=f0+(1-f0)*(1-oh.clamp(0,1))**5
    diffuse=(kd*(1-metallic)/math.pi)[:,None]
    spec=fresnel*ndf(q,tangent,normal,h)/(4*safe)
    return torch.where(valid,diffuse,0.),torch.where(valid,spec,0.)


def world_tensor(q,tangent):
    u=tangent.double()
    return u@q@u.transpose(-1,-2)


def sorted_widths(q):
    a,c,b=q[...,0,0],q[...,1,1],q[...,0,1]
    radius=torch.linalg.norm(torch.stack(((a-c)/2,b),dim=-1),dim=-1)
    largest=(a+c)/2+radius
    smallest=(a*c-b.square())/largest
    return torch.stack((largest,smallest),dim=-1).sqrt()


@torch.no_grad()
def export_axes(q,tangent):
    """Analytic principal axes for display and axial scoring outside AD."""
    widths=sorted_widths(q).to(tangent.dtype)
    a,d,b=q[:,0,0:1],q[:,1,1:2],q[:,0,1:2]
    angle=.5*torch.atan2(2*b,a-d)
    c,s=angle.cos(),angle.sin()
    x,y=tangent.double().unbind(-1)
    major=(c*x+s*y).to(tangent.dtype)
    minor=(-s*x+c*y).to(tangent.dtype)
    return widths[:,0:1],widths[:,1:2],major,minor
