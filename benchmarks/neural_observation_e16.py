"""E16 policies, observation-only teacher extraction and physical tensor helpers.

No native imports or scientific construction at module import.
"""
import numpy as np
from infrastructure.signatures import require

POLICY=dict(seed=6033,photometric_updates=1000,distillation_updates=200,baseline_extra_max=1000,
    lr=.001,betas=[.9,.999],eps=1e-8,amsgrad=True,weight_decay=0.,gradient_clip=.7,
    compatibility_repeats=32,evaluation_repeats=8,compatibility_SE_multiplier=6.,
    compatibility_absolute=.001,compatibility_relative=.02,gradient_absolute=.001,
    gradient_relative=.10,float32_guard=64*float(np.finfo(np.float32).eps),
    gradient_probe_indices=[0,16,33,49,66,83],confidence_near_fit_RMSE=1e-4)
LIMITS=dict(wall_seconds=3600,inner_seconds=3480,grace_seconds=30,cpu_threads=4,host_memory_gib=64,
    cuda_allocator_fraction=.25,output_bytes=2*1024**3,photometric_Adam_returns=6000,
    distillation_Adam_returns=400,gate_shader_calls=128,photo_shader_calls=12000,eval_shader_calls=192,
    gate_reverse_sweeps=384,photo_backwards=6000,distillation_backwards=400,
    shadow_rays=2616999936,field_calls=6412,independent_point_directions=869007360,
    coefficient_point_directions=72089600,environment_point_directions=55050240)


def world_Q(parameters,T,B):
    """theta,u,v chart; no width sorting, preserving swapped-axis equivalence."""
    p=np.asarray(parameters);c=np.cos(p[...,0]);s=np.sin(p[...,0])
    a=c[...,None]*T+s[...,None]*B;b=-s[...,None]*T+c[...,None]*B
    return np.exp(2*p[...,1]+p[...,2])[...,None,None]*a[...,None]*a[...,None,:]+np.exp(2*p[...,1]-p[...,2])[...,None,None]*b[...,None]*b[...,None,:]


def teacher(archive,trajectory,T,B):
    """Read only observed fitting selection, estimates and training diagnostics.

    In particular, no truth, held-view predictions or physical-error arrays are
    read from the accepted archive to build the pseudo-label/confidence outputs.
    """
    p=archive['parameters'];loss=archive['training_objective'];selected=archive['selected_by_training']
    require(p.shape==(84,3,3) and p.dtype==np.float64 and loss.shape==(84,3) and
        np.isfinite(p).all() and np.isfinite(loss).all() and (loss>=0).all() and
        selected.shape==(84,) and selected.dtype==np.int64 and np.array_equal(selected,np.argmin(loss,axis=1)),
        'Accepted fitting-only teacher selection mismatch')
    Q=world_Q(p,T[:,None],B[:,None]);target=Q[np.arange(84),selected]
    norm=np.linalg.norm(target,axis=(1,2));require((norm>0).all(),'Invalid teacher tensor')
    near=np.sqrt(loss)<=np.sqrt(loss.min(1))[:,None]+POLICY['confidence_near_fit_RMSE']
    spread=[]
    for point in range(84):
        q=Q[point,near[point]]
        spread.append(float(np.linalg.norm(q[:,None]-q[None],axis=(-2,-1)).max()/norm[point]))
    eigen=np.asarray(trajectory['history'][-1]['normal_eigenvalues'])[np.arange(84),selected]
    confidence=dict(near_fit_starts=near.sum(1),relative_Q_disagreement=np.array(spread),
        fitting_MSE=loss[np.arange(84),selected],last_pretrial_normal_eigenvalues=eigen,
        weighting_used=False,interpretation='Observation-only training diagnostics; no calibrated uncertainty or GT filtering.')
    return p[np.arange(84),selected].copy(),target,confidence


def compatibility(samples,reference,grid_error,absolute,relative):
    """Repeated native estimates, deterministic reference and separate grid proxy.

    Six empirical SEs are descriptive, not a rigorous simultaneous confidence
    guarantee. A deterministic proxy is never folded into stochastic variance.
    """
    a=np.asarray(samples,dtype=np.float64);r=np.asarray(reference,dtype=np.float64);g=np.asarray(grid_error)
    require(a.shape[0]==POLICY['compatibility_repeats'] and a.shape[1:]==r.shape==g.shape and
        np.isfinite(a).all() and np.isfinite(r).all() and np.isfinite(g).all() and (g>=0).all(),
        'Incomplete/nonfinite compatibility panel')
    mean=a.mean(0);std=a.std(0,ddof=1);se=std/np.sqrt(len(a));delta=abs(mean-r)
    numerical=g+POLICY['float32_guard'];allow=POLICY['compatibility_SE_multiplier']*se+numerical
    component=delta<=absolute+allow
    pooled=float(np.linalg.norm(mean-r));limit=relative*float(np.linalg.norm(r))+float(np.linalg.norm(allow))
    precision_limit=absolute*np.sqrt(r.size)+relative*float(np.linalg.norm(r))
    precision=bool(np.linalg.norm(se)<=precision_limit)
    return dict(passed=bool(component.all() and pooled<=limit and precision),precision_adequate=precision,
        empirical_SE_L2=float(np.linalg.norm(se)),empirical_SE_limit=float(precision_limit),maximum_mean_absolute=float(delta.max()),
        pooled_relative_L2=pooled/max(float(np.linalg.norm(r)),1e-15),
        maximum_empirical_SE=float(se.max()),RMS_single_draw_std=float(np.sqrt(np.mean(std**2))),
        maximum_unexplained_absolute=float(np.maximum(delta-allow,0).max()),
        component_failures=int((~component).sum()),pooled_difference=pooled,pooled_limit=limit)


def tensor_quantities(torch,shader,known):
    mx,my,_,_,rotation,raw=shader.predict_anisotropic_components(known['positions'])
    from scientific_variants.material_a_v1 import orientation
    c,s=orientation(rotation);T,B=shader.compute_tangent_bitangent_flat(known['normals'],None)
    a=c*T+s*B;b=-s*T+c*B
    Q=mx[:,:,None]**2*a[:,:,None]*a[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]
    return dict(mx=mx,my=my,rotation=rotation,raw=raw,Q=Q,axis=a,other=b,T=T,B=B)


def Q_loss(torch,predicted,target):
    denominator=(target*target).sum((-1,-2))
    require(bool((denominator>0).all()),'Positive observed teacher Q required')
    return (((predicted-target)**2).sum((-1,-2))/denominator).mean()


def photo_loss(torch,predicted,fitting_RGB):
    require(predicted.shape==fitting_RGB.shape and predicted.ndim==3 and predicted.shape[1:]==(2,3),
            'Only matched fitting RGB observations enter photometric loss')
    return ((predicted-fitting_RGB)**2).mean()


def compute_match(active_A,active_B,distillation,teacher_charge,extra):
    goal=active_B+distillation+teacher_charge
    return dict(goal_seconds=goal,baseline_active_seconds=active_A,
        reached=bool(active_A>=goal),continue_baseline=bool(active_A<goal and extra<POLICY['baseline_extra_max']),
        capped=bool(active_A<goal and extra>=POLICY['baseline_extra_max']))
