"""Independent float64 Cartesian additive Kurt reference and GGX/cosine MIS proposal.

No production shader, neural model, optimizer or BRDF/PDF cancellation imported.
GGX is ONLY a reference importance proposal; physical NDF remains Beckmann.
"""
import math
import torch


def basis(n):
    axis=torch.nn.functional.one_hot(n.abs().argmin(-1),3).to(n)
    t=torch.nn.functional.normalize(torch.cross(n,axis,dim=-1),dim=-1)
    return t,torch.cross(n,t,dim=-1)


def field(p,n,bounds):
    center=bounds.mean(0);scale=(bounds[1]-bounds[0]).max()/2;q=(p-center)/scale
    a=.23+.03*q[:,0:1];b=.35+.03*q[:,1:2];grain=p.new_tensor([1.,2.,3.])/math.sqrt(14)
    u=grain-(n*grain).sum(-1,keepdim=True)*n
    color=torch.stack([.35+.08*q[:,0],.50+.06*q[:,1],.20+.04*q[:,2]],-1)
    metal=.25+.05*torch.sin(math.pi*q[:,0:1])*torch.cos(math.pi*q[:,1:2])
    Q=a[:,:,None]**2*(torch.eye(3,dtype=p.dtype,device=p.device)[None]-n[:,:,None]*n[:,None,:])+b[:,:,None]**2*u[:,:,None]*u[:,None,:]
    return dict(a=a,b=b,u=u,color=color,metallic=metal,Q_world=Q)


def kernel(wi,wo,n,material):
    h=torch.nn.functional.normalize(wi+wo[:,None],dim=-1)
    ci=(wi*n[:,None]).sum(-1);co=(wo*n).sum(-1);ch=(h*n[:,None]).sum(-1);oh=(h*wo[:,None]).sum(-1)
    valid=(ci>0)&(ch>0)&(oh>0)&(co[:,None]>0)
    safe_ch=torch.where(valid,ch,torch.ones_like(ch));safe_dot=torch.where(valid,oh,torch.ones_like(oh))
    ht=h-ch[...,None]*n[:,None];a2=material['a']**2;b2=material['b']**2
    norm_u2=material['u'].square().sum(-1,keepdim=True)
    quad=ht.square().sum(-1)/a2-b2*(ht*material['u'][:,None]).sum(-1).square()/(a2*(a2+b2*norm_u2))
    determinant=material['a']*torch.sqrt(a2+b2*norm_u2)
    D=torch.exp(-quad.clamp(min=0)/safe_ch.square())/(math.pi*determinant*safe_ch**4)
    f0=.04*(1-material['metallic'])+material['metallic']*material['color']
    F=f0[:,None]+(1-f0[:,None])*(1-oh.clamp(0,1))[...,None]**5
    diffuse=material['color']*(1-material['metallic'])/math.pi
    specular=F*D[...,None]/(4*safe_dot[...,None])
    return torch.where(valid[...,None],diffuse[:,None],torch.zeros_like(specular)),torch.where(valid[...,None],specular,torch.zeros_like(specular))


def samples(n,wo,diffuse_tape,half_tape,alpha=.3):
    t,b=basis(n);u,v=diffuse_tape.T;az=2*math.pi*v
    wi_d=torch.sqrt(u)[None,:,None]*n[:,None]+torch.sqrt(1-u)[None,:,None]*(torch.cos(az)[None,:,None]*t[:,None]+torch.sin(az)[None,:,None]*b[:,None])
    u,v=half_tape.T;ch=torch.sqrt((1-u)/(1+(alpha*alpha-1)*u));az=2*math.pi*v
    h=ch[None,:,None]*n[:,None]+torch.sqrt(1-ch*ch)[None,:,None]*(torch.cos(az)[None,:,None]*t[:,None]+torch.sin(az)[None,:,None]*b[:,None])
    wi_s=2*(wo[:,None]*h).sum(-1,keepdim=True)*h-wo[:,None]
    wi=torch.cat([wi_d,wi_s],1);H=torch.nn.functional.normalize(wi+wo[:,None],dim=-1)
    ci=(wi*n[:,None]).sum(-1);ch=(H*n[:,None]).sum(-1);oh=(H*wo[:,None]).sum(-1).abs()
    D=alpha**2/(math.pi*(1+(alpha**2-1)*ch.square())**2)
    q_s=D*ch.clamp(min=0)/(4*oh.clamp(min=torch.finfo(wi.dtype).tiny))
    pdf=.5*ci.clamp(min=0)/math.pi+.5*q_s
    valid=ci>0
    return wi,pdf,valid


def environment(directions,definition):
    result=directions.new_tensor(definition['ambient']).expand_as(directions)
    for lobe in definition['lobes']:
        c=(directions*directions.new_tensor(lobe['axis'])).sum(-1).clamp(min=0)
        result=result+c[...,None]**lobe['power']*directions.new_tensor(lobe['RGB'])
    return result


def integrate(p,n,wo,material,wi,pdf,visible,definition,coarse,return_energy=False):
    fd,fs=kernel(wi,wo,n,material);ci=(wi*n[:,None]).sum(-1).clamp(min=0)
    weight=ci/torch.where(pdf>0,pdf,torch.ones_like(pdf))*visible
    light=environment(wi,definition)
    values=torch.stack([fd*light*weight[...,None],fs*light*weight[...,None]],2)
    fine=values.mean(1);half=wi.shape[1]//2;indices=torch.cat([torch.arange(coarse//2,device=p.device),torch.arange(half,half+coarse//2,device=p.device)])
    low=values[:,indices].mean(1)
    result=(torch.cat([fine,fine.sum(1,keepdim=True)],1),torch.cat([low,low.sum(1,keepdim=True)],1))
    if return_energy:
        # Unoccluded unit-radiance directional reflectance, not a conservation claim.
        energy=(fd+fs)*(ci/torch.where(pdf>0,pdf,torch.ones_like(pdf)))[...,None]
        return result+(energy.mean(1),energy[:,indices].mean(1))
    return result
