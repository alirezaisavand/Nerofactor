"""Explicit surface_world_q_v1 candidate. Frozen A/core classes are never edited.

Q has eigenvalues mx**2,my**2,0 in world coordinates. Pairwise normal transport
compares physical tensors, not signed directors or local chart coordinates.
"""
import math
import numpy as np
from infrastructure.signatures import require

VERSION = 'surface_world_q_v1'


def preserve_legacy_spatial_draws(pts,cfg,torch):
    """Keep the ORIGINAL two Gaussian-neighborhood RNG requests in both arms.

    Their off-surface points are not used by the surface loss. Discarding these
    exact draws preserves the next renderer sample stream versus legacy_v1.
    """
    torch.rand(pts.shape[0],1,device=pts.device)
    if cfg['change_type']=='gaussian':
        torch.normal(0.,cfg['change_eps'],size=(pts.shape[0],1),device=pts.device)
    else:require(cfg['change_type']=='constant','Unsupported legacy spatial sampling mode')


def tensor_from_axes(mx, my, t, b):
    return mx[...,None]**2*t[..., :,None]*t[...,None,:] + my[...,None]**2*b[..., :,None]*b[...,None,:]


def world_q(mx, my, rotation, normals, torch):
    """Same corrected-core X/Y tangent chart and atan2 decoder, new tensor loss."""
    require(bool(torch.isfinite(normals).all()) and bool((torch.linalg.norm(normals,dim=-1)-1).abs().le(1e-5).all()),'Unit normals required')
    norm=torch.linalg.norm(rotation,dim=-1,keepdim=True)
    require(bool(torch.isfinite(norm).all()) and bool((norm>=1e-6).all()),'Invalid orientation encoding')
    x=normals.new_tensor([1.,0.,0.]).expand_as(normals)
    t=torch.cross(normals,x,dim=-1)
    alt=torch.cross(normals,normals.new_tensor([0.,1.,0.]).expand_as(normals),dim=-1)
    t=torch.where((torch.linalg.norm(t,dim=-1)<1e-6)[:,None],alt,t)
    t=torch.nn.functional.normalize(t,dim=-1);b=torch.nn.functional.normalize(torch.cross(normals,t,dim=-1),dim=-1)
    encoded=rotation/norm
    angle=.5*torch.atan2(encoded[:,1:2],encoded[:,0:1]);c,s=torch.cos(angle),torch.sin(angle)
    return tensor_from_axes(mx,my,c*t+s*b,-s*t+c*b)


def normal_transport(source, target, torch):
    """Unique minimal SO(3) rotation, undefined at antipodes; never flip normals."""
    for value in (source,target):
        require(bool(torch.isfinite(value).all()) and bool((torch.linalg.norm(value,dim=-1)-1).abs().le(1e-5).all()),'Unit transport normals required')
    v=torch.cross(source,target,dim=-1);c=(source*target).sum(-1)
    require(bool((c>-1+1e-6).all()),'Antipodal transport is undefined')
    z=torch.zeros_like(c);vx,vy,vz=v.unbind(-1)
    k=torch.stack([z,-vz,vy,vz,z,-vx,-vy,vx,z],-1).reshape(-1,3,3)
    eye=torch.eye(3,dtype=source.dtype,device=source.device).expand_as(k)
    return eye+k+(k@k)/(1+c)[:,None,None]


def transported_q_loss(q0,q1,n0,n1,weights,torch):
    require(q0.shape==q1.shape and q0.shape[-2:]==(3,3),'Q pair shape mismatch')
    require(bool(torch.isfinite(q0).all() & torch.isfinite(q1).all()),'Nonfinite Q')
    w=weights.detach().reshape(-1)
    require(w.shape==(len(q0),) and bool(torch.isfinite(w).all() & (w>=0).all()),'Invalid detached edge weights')
    r=normal_transport(n1,n0,torch);difference=q0-r@q1@r.transpose(-1,-2)
    # Widths live in [.04,1]; absolute squared Frobenius error has a fixed
    # physical scale. Do not divide by learned anisotropy or tune from truth.
    per_pair=(difference*difference).sum((-1,-2))
    return (per_pair*w).sum()/w.sum().clamp(min=1),per_pair


def surface_pairs(vertices,faces,max_distance=.02,max_angle_degrees=15.):
    """Deterministic edge-connected mesh-surface samples (no Euclidean kNN).

    Interior pairs share an actual manifold mesh edge. Centroids are on the
    selected triangles; exclude boundaries/nonmanifold/large dihedrals. Input
    winding/vertices are read only. No projection onto an unrelated surface.
    """
    v=np.asarray(vertices,np.float64);f=np.asarray(faces)
    require(v.ndim==2 and v.shape[1]==3 and f.ndim==2 and f.shape[1]==3 and
            f.dtype.kind in 'iu' and np.isfinite(v).all() and f.min()>=0 and f.max()<len(v),'Invalid triangle mesh')
    xyz=v[f];cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);length=np.linalg.norm(cross,axis=-1)
    # A zero-area face has no normal/surface neighborhood. Exclude it ONLY
    # from this new loss's pair domain; never remove/repair it in the renderer.
    nondegenerate=length>0
    normals=np.divide(cross,length[:,None],out=np.zeros_like(cross),where=length[:,None]>0);centers=xyz.mean(1)
    e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);owner=np.tile(np.arange(len(f)),3)
    e=np.sort(e,axis=1);order=np.lexsort((e[:,1],e[:,0]));e=e[order];owner=owner[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(e[1:]!=e[:-1],axis=1))];counts=np.diff(np.r_[starts,len(e)])
    start=starts[counts==2];i,j=owner[start],owner[start+1]
    distance=np.linalg.norm(centers[i]-centers[j],axis=-1)
    midpoint=v[e[start]].mean(1)
    surface_distance=np.linalg.norm(centers[i]-midpoint,axis=-1)+np.linalg.norm(centers[j]-midpoint,axis=-1)
    keep=nondegenerate[i]&nondegenerate[j]&(distance>0)&(surface_distance>=1e-6)&(surface_distance<=max_distance)&((normals[i]*normals[j]).sum(-1)>=math.cos(math.radians(max_angle_degrees)))
    require(keep.any(),'No admissible mesh-surface pairs')
    return dict(face_pairs=np.stack([i[keep],j[keep]],-1),points0=centers[i[keep]],points1=centers[j[keep]],
                normals0=normals[i[keep]],normals1=normals[j[keep]],distance=surface_distance[keep],
                excluded_non_two_face_edges=int((counts!=2).sum()),excluded_degenerate_faces=int((~nondegenerate).sum()),
                excluded_distance_or_dihedral=int((~keep).sum()))


def material_boundary_weights(color0,color1,metal0,metal1,color_scale=.1,metal_scale=.1):
    """Frozen observation-derived gates. Never use trainable values to evade loss.

    Fixed training-image colors are imperfect reflectance-boundary evidence;
    illumination edges can cause false boundaries. Never pass learned predictions
    without separately freezing and declaring their provenance.
    """
    delta=np.linalg.norm(np.asarray(color0)-color1,axis=-1)/color_scale
    delta+=np.abs(np.asarray(metal0)-metal1).reshape(-1)/metal_scale
    return np.exp(-delta)


def legacy_energy_specular(fd,proxy,L_spec,energy_coefficient,specular_coefficient,torch):
    """Exact current channel SUM/point MEAN; observer or unchanged candidate terms."""
    energy=torch.relu(fd+proxy-1).sum(-1).mean()*energy_coefficient
    specular=L_spec.sum(-1).mean()*specular_coefficient
    return energy,specular


def energy_proxy_candidate(diffuse_allocation,specular_weights,valid_mask,torch,version='all_proposals_v1'):
    """Separately selectable analytical energy-prior candidate; NOT enabled below.

    Weights are the unchanged corrected shader's F*cos_i/cos_h with zeroes for
    invalid samples. Only the PRIOR denominator changes; never the RGB estimator.
    """
    require(version in ('legacy_accepted_count_v1','all_proposals_v1'),'Unknown energy proxy version')
    require(specular_weights.ndim==3 and specular_weights.shape[-1]==3 and
            specular_weights.shape[1]>0 and diffuse_allocation.shape==(specular_weights.shape[0],3) and
            valid_mask.shape==specular_weights.shape[:2]+(1,) and valid_mask.dtype==torch.bool,
            'Explicit per-proposal specular weights and validity mask required')
    require(bool(torch.isfinite(specular_weights).all() & (specular_weights>=0).all()) and
            bool((specular_weights*~valid_mask).eq(0).all()),'Invalid-direction weights must be finite nonnegative zero')
    denominator=specular_weights.shape[1] if version=='all_proposals_v1' else valid_mask.sum(1).clamp(min=1)
    return diffuse_allocation+specular_weights.sum(1)/denominator


def candidate_pair_loss(shader,pairs,torch,version=VERSION,q_coefficient=.05,step=0):
    """Selectable, fully differentiable surface-pair spatial term only.

    Does not invoke or replace BRDF, lighting, sampling, energy/specular losses.
    The caller adds this term explicitly; frozen regularizers remain selectable.
    """
    require(version in (VERSION,'surface_chart_control_v1'),'Unknown surface spatial regularizer')
    p0,p1,n0,n1,w=(pairs[k] for k in ('points0','points1','normals0','normals1','weights'))
    w=w.detach()
    a=shader.predict_anisotropic_components(p0);b=shader.predict_anisotropic_components(p1)
    mx0,my0,m0,kd0,r0,_=a;mx1,my1,m1,kd1,r1,_=b
    distance=pairs['distance'].reshape(-1)
    require(bool(torch.isfinite(distance).all() & (distance>=1e-6).all()),'Invalid surface distance')
    scale=.02/distance  # Historical .02 perturbation scale, in unchanged mesh units.
    if version==VERSION:
        q0=world_q(mx0,my0,r0,n0,torch);q1=world_q(mx1,my1,r1,n1,torch)
        _,difference=transported_q_loss(q0,q1,n0,n1,w,torch)
        anisotropy=(difference*scale**2*w.reshape(-1)).sum()/w.sum().clamp(min=1)
    else:
        # Declared chart-coordinate control on IDENTICAL surface pairs.
        decay=(1e-3-1e-4)*(.001*(math.cos((step/100000.)*math.pi)+1)/2)+1e-4
        value=((mx0-mx1).abs().reshape(-1)+(my0-my1).abs().reshape(-1))*scale+decay*((r0-r1)**2).sum(-1)*scale**2
        anisotropy=(value*w.reshape(-1)).sum()/w.sum().clamp(min=1)
    scalar=((kd0-kd1).abs().mean(-1)+(m0-m1).abs().reshape(-1))*scale
    scalar=(scalar*w.reshape(-1)).sum()/w.sum().clamp(min=1)
    return dict(loss_surface_q=q_coefficient*anisotropy,loss_surface_scalar=.05*scalar)


def gradient_diagnostics(terms,parameters,torch,names=None):
    """Read-only extra AD probes; .grad and RNG are untouched. Not historical AD."""
    parameters=list(parameters)
    names=list(names) if names is not None else ['parameter_'+str(i) for i in range(len(parameters))]
    require(len(names)==len(parameters),'Gradient name inventory mismatch')
    named=[(n,p) for n,p in zip(names,parameters) if p.requires_grad]
    active=[p for n,p in named];vectors={};summary={};group_vectors={}
    def group(n):
        for key in ('feats_network','mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor','outer_light','inner_light'):
            if key in n:return key
        return 'other'
    for name,term in terms.items():
        require(bool(torch.isfinite(term)),'Nonfinite diagnostic loss')
        if term.requires_grad:
            gs=torch.autograd.grad(term,active,retain_graph=True,allow_unused=True)
            v=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(active,gs)])
        else:v=torch.cat([torch.zeros_like(p).reshape(-1) for p in active])
        require(bool(torch.isfinite(v).all()),'Nonfinite diagnostic derivative')
        vectors[name]=v
        offset=0;groups={}
        for n,p in named:
            groups.setdefault(group(n),[]).append(v[offset:offset+p.numel()]);offset+=p.numel()
        grouped={key:torch.cat(values) for key,values in groups.items()};group_vectors[name]=grouped
        summary[name]=dict(weighted_loss=float(term.detach()),gradient_l2=float(torch.linalg.norm(v)),
                          groups={key:float(torch.linalg.norm(value)) for key,value in grouped.items()})
    cosine={}
    for i,(a,va) in enumerate(vectors.items()):
        for b,vb in list(vectors.items())[i+1:]:
            den=torch.linalg.norm(va)*torch.linalg.norm(vb)
            cosine[a+'__'+b]=float((va@vb)/den) if bool(den>0) else None
    grouped_cosine={}
    for a in group_vectors:
        for b in group_vectors:
            if a>=b:continue
            grouped_cosine[a+'__'+b]={}
            for key,va in group_vectors[a].items():
                vb=group_vectors[b][key];den=torch.linalg.norm(va)*torch.linalg.norm(vb)
                grouped_cosine[a+'__'+b][key]=float((va@vb)/den) if bool(den>0) else None
    return dict(terms=summary,gradient_cosines=cosine,group_gradient_cosines=grouped_cosine,
                interpretation='Additional local AD probes; not archived training gradients')
