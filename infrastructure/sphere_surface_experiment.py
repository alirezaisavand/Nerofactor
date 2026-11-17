"""Two matched4000-update forks of the sphere's state4000; native entry only.

No scientific modules are imported until execute() is explicitly authorized.
"""
import hashlib
import json
from pathlib import Path
import types
import time

import numpy as np

from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,PINS,digest
from infrastructure.surface_q_regularization import (VERSION,surface_pairs,material_boundary_weights,
    candidate_pair_loss,legacy_energy_specular,gradient_diagnostics,preserve_legacy_spatial_draws)

PROTOCOL='sphere-surface-regularizer-comparison-v1'
ARMS=('surface_chart_control_v1',VERSION)
RETURNS=4000
START=4000
END=START+RETURNS
PROBE_STEPS=(4000,5999,7999)


def prepare_pairs(model,torch):
    """Fixed mesh pairs and TRAINING-camera bilateral gate, no held-out leakage."""
    mesh=model.mesh
    raw=surface_pairs(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    count=512
    require(len(raw['face_pairs'])>=count,'Insufficient fixed manifold mesh pairs')
    indices=np.linspace(0,len(raw['face_pairs'])-1,count,dtype=np.int64)
    p0,p1=raw['points0'][indices],raw['points1'][indices]
    # Exactly A's existing trace normal convention. No mesh winding changes.
    n0,n1=-raw['normals0'][indices],-raw['normals1'][indices]
    camera=str(model.train_ids[0]);db=model.database
    pose=np.asarray(db.get_pose(camera));center=pose[:3,3];K=db.get_K(camera);image=db.get_image(camera)
    colors=[];validity=[]
    device=next(model.parameters()).device
    for pts,normals in ((p0,n0),(p1,n1)):
        local=(pts-center)@pose[:3,:3];z=-local[:,2]
        u=K[0,0]*local[:,0]/np.maximum(z,1e-12)+K[0,2]
        v=-K[1,1]*local[:,1]/np.maximum(z,1e-12)+K[1,2]
        x,y=np.rint(u).astype(int),np.rint(v).astype(int)
        inside=(z>0)&(x>=0)&(x<image.shape[1])&(y>=0)&(y<image.shape[0])
        delta=pts-center;direction=delta/np.linalg.norm(delta,axis=-1,keepdims=True)
        origins=torch.tensor(np.broadcast_to(center,pts.shape).copy(),dtype=torch.float32,device=device)
        rays=torch.tensor(direction,dtype=torch.float32,device=device)
        with torch.no_grad():hits,actual_normals,depth,hit_mask=model.trace(origins,rays)
        hit_positions=hits.detach().cpu().numpy();actual_normals=actual_normals.detach().cpu().numpy()
        valid=inside & hit_mask.detach().cpu().numpy().reshape(-1)
        valid &= (np.linalg.norm(hit_positions-pts,axis=-1)<=2e-4)&((actual_normals*normals).sum(-1)>1-1e-4)
        valid &= (normals*(-direction)).sum(-1)>0
        color=np.zeros((count,3));color[inside]=image[y[inside],x[inside]]
        colors.append(color);validity.append(valid)
    valid=validity[0]&validity[1]
    # The original image is sRGB-like loader data, not an oracle albedo. This
    # fixed gate can also suppress shading edges; report that limitation.
    weights=material_boundary_weights(colors[0],colors[1],np.zeros((count,1)),np.zeros((count,1)))*valid
    require((weights>0).any(),'No visible fixed training-camera surface pairs; no replacements')
    packet=dict(points0=p0,points1=p1,normals0=n0,normals1=n1,weights=weights,distance=raw['distance'][indices])
    descriptor=dict(camera_id=camera,selected_pairs=count,visible_pairs=int(valid.sum()),
        positive_weight_pairs=int((weights>0).sum()),visibility_queries=2*count,
        excluded_degenerate_faces=raw['excluded_degenerate_faces'],
        face_pairs_sha256=hashlib.sha256(raw['face_pairs'][indices].tobytes()).hexdigest(),
        pair_array_hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in packet.items()},
        boundary_gate='Fixed visible training-image RGB contrast exp(-||delta RGB||/.1); no GT material/held-out supervision',
        normal_convention='negative raw mesh face normals, matching original A.trace')
    return {k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in packet.items()},descriptor


def install_spatial(shader,pairs,mode,torch):
    require(mode in ARMS and shader.cfg['regularization_version']=='legacy_v1' and shader.cfg['reg_change'],
            'Explicit matched surface mode plus unchanged active legacy energy/specular required')
    def regularize(self,pts,normals,t,b,mx,my,metallic,kd,fd,proxy,L_spec,rotation,rotation_raw,step):
        preserve_legacy_spatial_draws(pts,self.cfg,torch)
        parts=candidate_pair_loss(self,pairs,torch,mode,step=step)
        energy,spec=legacy_energy_specular(fd,proxy,L_spec,
            self.cfg['reg_energy_loss_lambda'],self.cfg['reg_spec_loss_lambda'],torch)
        parts.update(energy=energy,specular=spec)
        self._surface_terms=parts
        return fd.sum(-1)*0+sum(parts.values())
    shader.anisotropic_regularization=types.MethodType(regularize,shader)


class ControlledEndpoint(BaseException):
    pass


def trainer_class(base,output,report,persist,budget,torch,mode,control,checkpoint=None):
    from infrastructure.sphere_replay import trainer_adapter
    parent=trainer_adapter(base,output,report,persist,budget,torch,np)
    class SurfaceTrainer(parent):
        def _init_network(self):
            super()._init_network()
            require(checkpoint is not None,'Explicit admitted checkpoint required for paired fork')
            path,record=checkpoint
            require(digest(path)==record['sha256'],'Admitted initial checkpoint changed')
            saved=torch.load(str(path),map_location='cpu')  # Trusted own checkpoint; isolated approved runtime ONLY.
            require(saved['step']==START,'Wrong checkpoint absolute step')
            self.network.load_state_dict(saved['network_state_dict'],strict=True)
            self.optimizer.load_state_dict(saved['optimizer_state_dict'])
            require(all(g['eps']==1e-4 and g['amsgrad'] and tuple(g['betas'])==(.9,.999) and
                        g['weight_decay']==0 for g in self.optimizer.param_groups),'Restored Adam configuration differs')
            require(all(bool(torch.isfinite(v).all()) for state in self.optimizer.state.values()
                        for v in state.values() if isinstance(v,torch.Tensor)),'Nonfinite restored optimizer state')
            require(digest(path)==record['sha256'],'Checkpoint changed during CPU deserialization')
            self._fork_best=float(saved['best_para'])
            require(np.isfinite(self._fork_best),'Nonfinite saved selection metadata')
            report['optimizer_returns']=START;report['new_optimizer_returns']=0
            report['checkpoint_restore']=dict(path=str(path),sha256=record['sha256'],strict=True,
                optimizer_restored=True,absolute_step=START,RNG_sampler_restored=False)
            report['optimizer']={k:v for k,v in self.optimizer.param_groups[0].items() if k!='params'}
            optimizer=self.optimizer.state_dict()
            def primitive_hash(value):
                if isinstance(value,torch.Tensor):return dict(sha256=digest_tensor(value),shape=list(value.shape),dtype=str(value.dtype))
                if isinstance(value,dict):return {str(k):primitive_hash(v) for k,v in value.items()}
                if isinstance(value,(tuple,list)):return [primitive_hash(v) for v in value]
                return value
            report['initial_optimizer']=primitive_hash(optimizer)
            require(report['initial_optimizer']==primitive_hash(saved['optimizer_state_dict']),
                    'Restored Adam shapes/dtypes/values/configuration differ from the admitted checkpoint')
            # Record actual weights before installing a behavior tag, without
            # new random draws or copying parameters from the other arm.
            weights={name:digest_tensor(p) for name,p in self.network.named_parameters()}
            report['initial_parameters']=weights
            report['initial_buffers']={name:digest_tensor(p) for name,p in self.network.named_buffers()}
            report['initial_sample_tapes']={name:digest_tensor(getattr(self.network.shader_network,name))
                for name in ('diffuse_direction_samples','specular_direction_samples')}
            if control is not None:
                require(weights==control['initial_parameters'] and report['split_ids']==control['split_ids'],
                        'Paired original initialization/split mismatch; stop before updates')
                require(report['initial_buffers']==control['initial_buffers'] and report['initial_sample_tapes']==control['initial_sample_tapes'],
                        'Paired buffer/sample-tape initialization mismatch')
                require(report['initial_optimizer']==control['initial_optimizer'],'Paired exact Adam state mismatch')
            from infrastructure.bell_inference import rng_identity
            report['initial_rng']=rng_identity(torch,np)
            pairs,description=prepare_pairs(self.network,torch)
            require(rng_identity(torch,np)==report['initial_rng'],'Geometry/boundary setup consumed training RNG')
            if control is not None:require(description==control['surface_pairs'],'Cross-arm neighborhood mismatch')
            if control is not None:require(report['initial_rng']==control['initial_rng'],'Cross-arm initial RNG mismatch')
            report['surface_pairs']=description
            np.savez_compressed(output/'cache/surface-pairs.npz',**{k:v.detach().cpu().numpy() for k,v in pairs.items()})
            shader=self.network.shader_network
            install_spatial(shader,pairs,mode,torch)
            tag=hashlib.sha256(json.dumps(dict(protocol=PROTOCOL,mode=mode,pairs=description,
                sources={n:digest(ROOT/n) for n in ('infrastructure/surface_q_regularization.py','infrastructure/sphere_surface_experiment.py')}),sort_keys=True).encode()).digest()
            self.network.register_buffer('_surface_regularizer_signature',torch.tensor(list(tag),dtype=torch.uint8,device=next(shader.parameters()).device))
            original=self.train_losses[-1];rgb_loss=self.train_losses[0];network=self.network
            class ObserveLoss:
                def __call__(self,outputs,data,step,*args,**kwargs):
                    result=original(outputs,data,step,*args,**kwargs)
                    if 'loss_rgb' in outputs:
                        parts=shader._surface_terms
                        if (step+1)%10==0:
                            with (output/'components.jsonl').open('a') as f:
                                f.write(json.dumps(dict(completed_updates=step+1,**{k:float(v.detach()) for k,v in parts.items()}))+'\n')
                        if step in PROBE_STEPS:
                            terms=dict(parts,rgb=outputs['loss_rgb'].mean()*rgb_loss.cosine_weight_schedule(step),
                                       diffuse_light=outputs['loss_diffuse_light'].mean())
                            named=list(network.named_parameters())
                            report.setdefault('gradient_probes',[]).append(dict(zero_based_step=step,
                                **gradient_diagnostics(terms,[p for n,p in named],torch,names=[n for n,p in named])))
                            persist()
                    return result
            self.train_losses=list(self.train_losses);self.train_losses[-1]=ObserveLoss()
            original_step=self.optimizer.step
            def bounded_step(*args,**kwargs):
                require(START<=report['optimizer_returns']<END and report['new_optimizer_returns']<RETURNS,
                        'No optimizer return beyond paired-fork budget authorized')
                before_step=report['optimizer_returns']
                inspect=before_step in PROBE_STEPS
                active=[p for p in self.network.parameters() if p.grad is not None]
                previous=[p.detach().clone() for p in active] if inspect else []
                counters=[int(self.optimizer.state.get(p,{}).get('step',0)) for p in active] if inspect else []
                value=original_step(*args,**kwargs);report['new_optimizer_returns']+=1
                if inspect:
                    require(all(int(self.optimizer.state[p]['step'])==old+1 for p,old in zip(active,counters)),
                            'Adam counters did not advance once')
                    displacement=sum(float((p.detach()-old).abs().sum().cpu()) for p,old in zip(active,previous))
                    require(displacement>0,'No parameter displacement after restored Adam return')
                    report.setdefault('optimizer_events',[]).append(dict(absolute_zero_based_step=before_step,
                        absolute_completed_updates=report['optimizer_returns'],new_returns=report['new_optimizer_returns'],
                        parameter_displacement_l1=displacement,lr=[g['lr'] for g in self.optimizer.param_groups],
                        active_state_counters_advanced=True))
                    persist()
                return value
            self.optimizer.step=bounded_step
            persist()

        def _load_model(self):
            require(report.get('checkpoint_restore',{}).get('strict') is True and
                    not Path(self.pth_fn).exists() and not Path(self.best_pth_fn).exists(),
                    'Explicit admitted fork restoration missing or own run is not fresh')
            return self._fork_best,START

        def _save_model(self,step,best_para,save_fn=None):
            super()._save_model(step,best_para,save_fn)
            if save_fn is None:
                sidecar=output/'checkpoints'/('step-%06d.json'%step)
                record=json.loads(sidecar.read_text());record['external_spatial_regularizer']=mode
                record['comparison_protocol']=PROTOCOL;sidecar.write_text(json.dumps(record,indent=2)+'\n')
            if save_fn is None and step==END:
                require(report['optimizer_returns']==END and report['new_optimizer_returns']==RETURNS,'Wrong endpoint count')
                raise ControlledEndpoint()
    return SurfaceTrainer


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native(cfg,run,report,persist,mode,image,control,checkpoint):
    import resource
    started=time.monotonic()
    import torch
    from infrastructure.sphere_replay import IMAGE,Budget,renderer_adapter
    require(torch.__version__=='1.8.1+cu111' and torch.cuda.is_available() and torch.cuda.device_count()==1 and
            torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe' and image==IMAGE,'Reviewed single-A100 runtime required')
    from infrastructure.verified_ray_backend import load
    report['ray_backend']=load('predicate_fp64',image)
    from infrastructure.stage_sources import activate,baseline
    science,_=activate('material',PINS['material'])
    from train.trainer import Trainer
    import network.renderer as renderers
    import network.field as fields
    from infrastructure.material_variants import install
    from infrastructure.stage_execution import run_material_owned
    from infrastructure.geometry_training_smoke import runtime_identity
    require(Path(Trainer.run.__code__.co_filename).resolve()==science/'train/trainer.py','Wrong original Trainer')
    report['runtime']=runtime_identity(torch)
    class ArmBudget(Budget):
        def __init__(self):super().__init__(run.root,seconds=1600);self.next_arm_check=0.
        def check(self):
            super().check()
            if time.monotonic()>=self.next_arm_check:
                from infrastructure.sphere_replay import BudgetStop
                size=sum(p.stat().st_size for p in run.root.rglob('*') if p.is_file() and not p.is_symlink())
                if size>=8*1024**3:raise BudgetStop('Paired-arm8GiB output cap reached')
                self.next_arm_check=time.monotonic()+20
    budget=ArmBudget()
    base=install(renderers,fields,'A_corrected_core_v1',cfg)
    observed=renderer_adapter(base,run.root,report,budget,torch,np)
    class BoundedRenderer(observed):
        def train_step(self,step):
            require(START<=step<END,'No scientific forward outside paired-fork budget authorized')
            return super().train_step(step)
    renderers.name2renderer['material']=BoundedRenderer
    trainer=trainer_class(Trainer,run.root,report,persist,budget,torch,mode,control,checkpoint)(cfg)
    stopped=False
    try:
        try:run_material_owned(trainer,report,persist)
        except ControlledEndpoint:stopped=True
    finally:
        report['resources']=dict(native_wall_seconds=time.monotonic()-started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        persist()
    require(stopped and report['optimizer_returns']==END and report['new_optimizer_returns']==RETURNS and report.get('cleanup_errors')==[],
            'Endpoint or owned cleanup gate failed')
    require([r['completed_updates'] for r in report['saved_views']]==[6000,8000], 'Expected fixed validation-camera maps')
    require(len(report.get('gradient_probes',[]))==3,'Missing component-gradient probes')
    baseline('material',PINS['material']);baseline('geometry',PINS['geometry'])
    from infrastructure.stage_sources import imported_sources
    report['imported_sources']=imported_sources(science)
    require(digest(checkpoint[0])==checkpoint[1]['sha256'],'Initial checkpoint changed during the fork')
    report['owned_cleanup_verified']=True
