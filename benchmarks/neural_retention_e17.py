"""E17 single-coefficient policy; metadata and dependency-injected loss helpers."""
import numpy as np
from benchmarks.neural_observation_e16 import Q_loss,photo_loss
from infrastructure.signatures import require

COEFFICIENT=1e-4
POLICY=dict(photometric_updates=1000,retention_coefficient=COEFFICIENT,gradient_probe_steps=[0,999],
    initialization='exact_E16_B_post_distillation',optimizer_state='fresh',confidence_weighting=False)
LIMITS=dict(wall_seconds=1800,inner_seconds=1680,grace_seconds=30,cpu_threads=4,host_memory_gib=64,
    cuda_allocator_fraction=.25,output_bytes=2*1024**3,photometric_Adam_returns=2000,photo_backwards=2000,
    photo_shader_calls=4000,eval_shader_calls=32,retention_reverse_sweeps=8,field_calls=2004,
    shadow_rays=852541440,independent_point_directions=144179200,
    coefficient_point_directions=72089600,environment_point_directions=55050240)


def scale_evidence(captures,training):
    """ONLY saved Q, observation-derived teacher Q and fitting losses select lambda."""
    rows=[]
    for material in ('original','stronger'):
        losses={}
        for stage in ('post_distillation','photo1000'):
            z=captures[material,stage];q=np.asarray(z['Q'],np.float64);target=z['teacher_Q']
            require(q.shape==target.shape==(84,3,3) and np.isfinite(q).all() and np.isfinite(target).all(),'Invalid Q scale archive')
            norm=(target*target).sum((1,2));require((norm>0).all(),'Invalid teacher denominator')
            losses[stage]=float((((q-target)**2).sum((1,2))/norm).mean())
        fitting=[r['loss'] for r in training[material] if r['phase']=='photo']
        require(len(fitting)==1000 and np.isfinite(fitting).all() and min(fitting)>=0,'Incomplete fitting scale log')
        rows.append(dict(material=material,post_distillation_Q_loss=losses['post_distillation'],
            final_Q_loss=losses['photo1000'],Q_loss_increase=losses['photo1000']-losses['post_distillation'],
            last100_fitting_loss=float(np.mean(fitting[-100:]))))
    photo=float(np.mean([x['last100_fitting_loss'] for x in rows]));drift=float(np.mean([x['Q_loss_increase'] for x in rows]))
    require(photo>0 and drift>0,'Positive historical scale required')
    ratio=photo/drift;rounded=float(10.**np.round(np.log10(ratio)))
    require(rounded==COEFFICIENT,'Accepted fitting-only scale no longer supports frozen coefficient')
    return dict(fields=rows,pooled_fitting_loss=photo,pooled_Q_loss_increase=drift,ratio=ratio,
        nearest_decade_coefficient=rounded,uses_GT_or_withheld=False,policy='One nearest-decade scale; no candidate search or outcome tuning.')


def objective(torch,rgb,target,Q,teacher_Q):
    rgb_loss=photo_loss(torch,rgb,target);retention=Q_loss(torch,Q,teacher_Q)
    return rgb_loss+COEFFICIENT*retention,rgb_loss,retention


def strict_restore(torch,model,arrays,expected_hash):
    from experiments.analyze_bell_neural_material_e16 import state_hash
    require(state_hash(arrays)==expected_hash,'Selected post-distillation state identity mismatch')
    current=model.state_dict();require(set(current)==set(arrays),'Strict checkpoint keys differ')
    for name,value in current.items():
        require(value.shape==arrays[name].shape and value.detach().cpu().numpy().dtype==arrays[name].dtype,
            'Checkpoint shape/dtype mismatch: '+name)
    state={name:torch.as_tensor(arrays[name],device=value.device) for name,value in current.items()}
    result=model.load_state_dict(state,strict=True)
    require(not result.missing_keys and not result.unexpected_keys,'Strict checkpoint restoration failed')
    require(state_hash({k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})==expected_hash,'Restored state differs')
