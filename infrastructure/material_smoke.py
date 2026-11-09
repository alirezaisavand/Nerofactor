"""Bounded A-material diagnostics; native dependencies are supplied after isolation."""
from contextlib import contextmanager
import inspect
import math
from pathlib import Path
import struct
import sys

from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,digest
from infrastructure.training_probe import (TrainingProbe,parameter_copies,snapshot,
                                           optimizer_record,array_record,tensor_copy)

PROTOCOL='bear-A-two-update-material-smoke-v1'
MESH_SHA256='abfead88fc664463b83b273c41f90cecb6e02ed578efb6d7037c8b4494625d24'
CONFIG='configs/experiments/material_a_bear_smoke_v1.yaml'
DOC='docs/MATERIAL_TRAINING_SMOKE.md'
GROUPS=('mx_predictor','my_predictor','rotation_predictor','metallic_predictor','kd_predictor',
        'feats_network','outer_light','inner_light','human_light')


def inspect_mesh(path,expected_sha256):
    """Read-only structural admission for this reviewed VCGLIB PLY layout only.

    No Open3D/BVH execution or claim of alignment/producing-run provenance.
    Fixtures may use their own SHA; native callers use the pinned historical SHA.
    """
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file(),'Missing/aliased explicit mesh')
    before=path.stat();require(digest(path)==expected_sha256,'Historical diagnostic mesh SHA-256 mismatch')
    with path.open('rb') as stream:
        lines=[];header_lines=0
        while True:
            line=stream.readline(256)
            header_lines+=1
            require(line and header_lines<=32 and len(line)<256,'Invalid/oversized PLY header')
            text=line.decode('ascii').strip()
            if not text.startswith('comment '):lines.append(text)
            if text=='end_header':break
        require(len(lines)==9 and lines[:2]==['ply','format binary_little_endian 1.0'] and
                lines[3:6]==['property float x','property float y','property float z'] and
                lines[7:]==['property list uchar int vertex_indices','end_header'],
                'Unsupported diagnostic PLY layout')
        vertices=int(lines[2].split()[-1])
        faces=int(lines[6].split()[-1])
        require(lines[2]=='element vertex '+str(vertices) and lines[6]=='element face '+str(faces) and vertices>0 and faces>0,
                'Empty or malformed mesh elements')
        require(before.st_size==stream.tell()+12*vertices+13*faces,'Truncated/extra mesh payload')
        xyz=stream.read(12*vertices);bounds=[[float('inf')]*3,[-float('inf')]*3]
        for point in struct.iter_unpack('<fff',xyz):
            require(all(math.isfinite(v) for v in point),'Nonfinite mesh vertices')
            for axis,value in enumerate(point):bounds[0][axis]=min(bounds[0][axis],value);bounds[1][axis]=max(bounds[1][axis],value)
        require(all(a<b for a,b in zip(*bounds)),'Collapsed mesh bounds')
        nondegenerate=0
        for count,a,b,c in struct.iter_unpack('<Biii',stream.read()):
            require(count==3 and 0<=a<vertices and 0<=b<vertices and 0<=c<vertices,'Invalid triangle index/count')
            p,q,r=(struct.unpack_from('<fff',xyz,12*i) for i in (a,b,c))
            u=[q[i]-p[i] for i in range(3)];v=[r[i]-p[i] for i in range(3)]
            nondegenerate+=int(any(x!=0 for x in (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])))
        require(nondegenerate>0,'No nondegenerate triangles')
    after=path.stat()
    require((before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)==
            (after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns) and digest(path)==expected_sha256,'Mesh changed while inspecting')
    return {'path':str(path),'sha256':expected_sha256,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,
            'vertices':vertices,'triangles':faces,'nondegenerate_triangles':nondegenerate,'bounds':bounds,
            'provenance':'historical/diagnostic; producing geometry revision and alignment unverified',
            'research_handoff_admitted':False,'native_Open3D_BVH_compatibility':'pending native construction'}


def diagnostic_files():
    return {name:digest(ROOT/name) for name in [CONFIG,DOC,'run_material_training_smoke.py']}


def admit_mesh(p):
    from infrastructure.stage_sources import PINS
    import yaml
    require(p['stage']=='material' and p['scientific_commit']==PINS['material'] and p['arm']=='C' and
            p['scene']=='bear' and p['configuration']['database_name']=='real/bear/raw_1024',
            'Historical diagnostic exception is only for pinned A bear material')
    require(p['raw']==(p['science']/'configs/material/real/bear.yaml').read_bytes(), 'Original bear YAML required')
    require(p['configuration']==yaml.load(p['raw'],Loader=yaml.FullLoader),'Diagnostic cannot override scientific bear values')
    require(p['selected'].checkpoint is None,'No material smoke checkpoint initialization/resume')
    return inspect_mesh(p['selected'].mesh,MESH_SHA256)


class MaterialProbe(TrainingProbe):
    """Reuse finite reductions, clipping, Adam accounting, two-update stop/cleanup."""
    database_name='real/bear/raw_1024'
    is_real=True
    human_lights=True
    groups=GROUPS
    def bind_calls(self,renderer,shader,rgb_loss,regularizer,trainer_class,evaluator):
        self.calls={getattr(shader,name).__code__:'shader.'+name for name in
            ['forward','anisotropic_forward','shade_anisotropic_mixed','sample_aniso_ggx_directions',
             'compute_tangent_bitangent_flat','compute_radiance','predict_anisotropic_components',
             'anisotropic_regularization','get_lights','predict_outer_lights','get_inner_lights','get_human_light']}
        self.calls.update({renderer.compute_rgb_loss.__code__:'renderer.compute_rgb_loss',
                           rgb_loss.__call__.__code__:'loss.nerf_render',regularizer.__call__.__code__:'loss.mat_reg'})
        # Torch decorators share wrapper code with Adam.step; forbid original bodies.
        self.forbidden={inspect.unwrap(getattr(renderer,n)).__code__ for n in ['test_step','nvs','predict_materials']}
        self.forbidden.update(inspect.unwrap(f).__code__ for f in [trainer_class._save_model,evaluator.__call__])

    @contextmanager
    def observe_calls(self):
        require(sys.getprofile() is None,'Material diagnostic requires an unprofiled process')
        sys.setprofile(self.profile)
        try:yield
        finally:sys.setprofile(None)

    def profile(self,frame,event,value):
        code=frame.f_code
        require(not (event=='call' and code in self.forbidden),'Validation/checkpoint/NVS reached during material smoke')
        if self.current is None or code not in self.calls:return
        name=self.calls[code]
        owner=frame.f_locals.get('self')
        require((name.startswith('shader.') and owner is self.trainer.network.shader_network) or
                (name.startswith('renderer.') and owner is self.trainer.network) or
                (name.startswith('loss.') and any(owner is v for v in self.trainer.train_losses)),
                'Anisotropic diagnostic observed a foreign model/loss object')
        counts=self.current['calls'].setdefault(name,{'entered':0,'returned':0})
        if event=='call':
            counts['entered']+=1
            if 'is_train' in frame.f_locals:require(frame.f_locals['is_train'] is True,'Nontraining shader call')
        elif event=='return' and value is not None:
            counts['returned']+=1
            if name in ('shader.anisotropic_regularization','renderer.compute_rgb_loss'):
                require(self.torch.is_tensor(value) and value.requires_grad,'Original RGB/regularization output lost autograd connection')
                self.current.setdefault('differentiable_outputs',{})[name]=array_record(tensor_copy(value,self.np,name),self.np)
            if name=='shader.get_lights':
                v=frame.f_locals;active=v['depth'].reshape(v['hit_mask'].shape)>v['eps'];hits=v['hit_mask']
                self.current['light_paths']['inner']+=int((active & hits).sum().item())
                if bool((~hits).any().item()):
                    weights=v['human_weights'].reshape(-1);near=active[~hits]
                    self.current['light_paths']['outer']+=int((near & (weights<1)).sum().item())
                    self.current['light_paths']['human']+=int((near & (weights>0)).sum().item())

    def initialize(self):
        t,np=self.trainer,self.np
        require(type(t.optimizer) is self.torch.optim.Adam and t.train_network is t.network,'Original single-GPU A Adam required')
        for key,value in {'total_step':100000,'random_seed':6033,'worker_num':8,'val_interval':2000,'save_interval':2000,'train_log_step':10}.items():
            require(type(t.cfg[key]) is int and t.cfg[key]==value,'Original bear setting changed: '+key)
        require(t.cfg['lr_cfg']=={'end_warm':1000,'end_iter':100000} and t.cfg['lr_type']=='warm_up_cos' and
                t.cfg['optimizer_type']=='adam' and not t.cfg['multi_gpus'],'Original bear optimizer/schedule required')
        cfg=t.network.cfg;shader=t.network.shader_network.cfg
        vertices=np.asarray(t.network.mesh.vertices);triangles=np.asarray(t.network.mesh.triangles)
        require(list(vertices.shape)==[self.report['mesh_identity']['vertices'],3] and
                list(triangles.shape)==[self.report['mesh_identity']['triangles'],3] and
                np.isfinite(vertices).all(),'Native Open3D mesh differs from the selected PLY')
        self.report['native_geometry']={'vertices':len(vertices),'triangles':len(triangles),
            'bounds':[vertices.min(0).tolist(),vertices.max(0).tolist()],
            'has_vertex_normals':bool(t.network.mesh.has_vertex_normals()),
            'ray_tracer_class':type(t.network.ray_tracer).__module__+'.'+type(t.network.ray_tracer).__name__,
            'normal_warning_observed':bool(t.network.warned_normal),
            'limit':'Original construction/ray-pool succeeded; no scientific mesh provenance/orientation claim.'}
        require(cfg['train_ray_num']==512 and cfg['rgb_loss']=='charbonier' and cfg['reg_mat'] and cfg['reg_diffuse_light'] and
                cfg['reg_diffuse_light_lambda']==.1 and cfg['database_name']==self.database_name,'Material scientific settings changed')
        require(shader['anisotropy'] is True and shader['is_real'] is self.is_real and shader['human_lights'] is self.human_lights and
                shader['diffuse_sample_num']==512 and shader['specular_sample_num']==1024 and shader['reg_change'] is True and
                shader['reg_energy_loss'] is True and shader['reg_spec_loss'] is True and
                shader['reg_energy_loss_lambda']==shader['reg_spec_loss_lambda']==.05,'Effective A shader/regularizer settings changed')
        initial=parameter_copies(t.network,np);require(initial,'No material parameters')
        for group in self.groups:require(any(n.startswith('shader_network.'+group+'.') for n in initial),'Missing predictor/light group: '+group)
        require(all(any(n.startswith('shader_network.'+g+'.') for g in self.groups) for n in initial),'Unknown material parameter group')
        self.previous_state=optimizer_record(t.optimizer,t.network,np)
        g=self.previous_state['groups'][0]
        require(len(self.previous_state['groups'])==1 and not self.previous_state['state'] and g['lr']==.001 and g['eps']==1e-4 and
                g['amsgrad'] is True and tuple(g['betas'])==(.9,.999) and g['weight_decay']==0,'Original fresh A Adam required')
        require(t.lr_manager.warm_up_end==1000 and t.lr_manager.end_iter==100000 and t.lr_manager.learning_rate==5e-4 and
                t.lr_manager.learning_rate_alpha==.2,'Original A scheduler required')
        self.report.update(initial_parameters=snapshot(initial,np),initial_optimizer=self.previous_state,
            initial_buffers={n:array_record(tensor_copy(v,np),np) for n,v in t.network.named_buffers()},
            effective_trainer=t.cfg,effective_renderer=cfg,effective_shader=shader,
            declared_top_level_regularization={k:t.cfg[k] for k in ['reg_energy_loss','reg_energy_loss_lambda','reg_spec_loss','reg_spec_loss_lambda']},
            effective_losses=[{'registered_name':n,'class':type(v).__module__+'.'+type(v).__name__,'configuration':v.cfg} for n,v in zip(t.cfg['loss'],t.train_losses)],
            scheduler={'class':type(t.lr_manager).__name__,'attributes':dict(vars(t.lr_manager)),'hard_coded_cosine_endpoint':100000},
            loader={'batch_size':t.train_set.batch_size,'num_workers':t.train_set.num_workers,
                    'dataset_size':len(t.train_set.dataset),'sampler':type(t.train_set.sampler).__name__},
            splits={n:[str(i) for i in getattr(t.network,n)] for n in ['train_ids','test_ids','nvs_ids']})
        self.persist()

    def event(self,boundary,values):
        if boundary=='fresh_restore_gate':
            require(not Path(self.trainer.pth_fn).exists() and not Path(self.trainer.best_pth_fn).exists(),'Implicit material resume refused')
            self.mark(boundary)
        else:super().event(boundary,values)

    def before_forward(self,values):
        t,c,np=self.trainer,self.current,self.np
        require(c is not None and c['step'] in (0,1) and values['train_data']=={'step':c['step']},'Unexpected material dispatch')
        lr=5e-4*(c['step']/1000)
        require(values['lr']==lr and all(g['lr']==lr for g in t.optimizer.param_groups),'Original bear warm-up LR changed')
        require(t.network.training and self.torch.is_grad_enabled(),'Material training autograd is disabled')
        start,count=t.network.train_batch_i,t.network.cfg['train_ray_num']
        require(start==c['step']*count and start+count<t.network.tbn,'Unexpected material ray cursor/reshuffle')
        c.update(lr=float(lr),rng_before_forward=self.rng(),memory_before=self.memory(),calls={},light_paths={'inner':0,'outer':0,'human':0},
                 sampling={'cursor_before':start,'ray_count':count,'pool_size':t.network.tbn,
                    'selected':{k:array_record(tensor_copy(v[start:start+count],np),np) for k,v in t.network.train_batch.items()}})
        self.before_parameters=parameter_copies(t.network,np);c['parameters_before']=snapshot(self.before_parameters,np)

    def before_backward(self,values):
        super().before_backward(values)
        required=['shader.'+n for n in ['forward','anisotropic_forward','shade_anisotropic_mixed','sample_aniso_ggx_directions',
                  'compute_tangent_bitangent_flat','compute_radiance','anisotropic_regularization']]+['renderer.compute_rgb_loss','loss.nerf_render','loss.mat_reg']
        require(all(self.current['calls'].get(n)=={'entered':1,'returned':1} for n in required),'Missing/repeated original anisotropic/RGB/regularizer path')
        require(all(n in self.current['losses'] for n in ['loss_rgb','loss_mat_reg','loss_diffuse_light']),'Original material loss missing')
        self.current['forward_path_verified']=True

    def group_gradients(self,records):
        groups={}
        for group in self.groups:
            selected={n:v for n,v in records.items() if n.startswith('shader_network.'+group+'.')}
            norm=sum(v['l2']**2 for v in selected.values() if v is not None)**.5
            active=True;reason='material prediction/shared features'
            if group in ('outer_light','inner_light','human_light'):
                path={'outer_light':'outer','inner_light':'inner','human_light':'human'}[group]
                active=self.current['light_paths'][path]>0
                reason='original light-path samples' if active else 'no contributing original light-path samples (branch/near/human mask)'
            groups[group]={'gradient_l2_cpu64':norm,'parameters':len(selected),'gradients_present':sum(v is not None for v in selected.values()),
                           'expected_active':active,'reason':reason,'status':'active' if active and norm>0 else ('expected_inactive' if not active else 'unexpected_missing_or_zero')}
            require(not active or (selected and all(v is not None for v in selected.values()) and norm>0),'Unexpected missing/zero material gradients: '+group)
            require(active or norm==0,'Unexpected gradients on an inactive light path: '+group)
        return groups

    def before_clip(self,values):
        super().before_clip(values)
        self.current['backward_returned']=True
        self.current['groups_before_clip']=self.group_gradients(self.current['gradients_before_clip'])
        self.persist()

    def before_optimizer(self,values):
        super().before_optimizer(values)
        self.current['groups_after_clip']=self.group_gradients(self.current['gradients_after_clip'])
        self.persist()

    def after_optimizer(self,values,completed):
        super().after_optimizer(values,completed)
        update=self.report['updates'][-1]
        require(any(v['exp_avg']['l2']>0 and v['exp_avg_sq']['l2']>0 for v in update['optimizer']['state'].values()),
                'No meaningful Adam moments after material update')
        update['group_changed_elements']={g:sum(v['changed_elements'] for n,v in update['parameter_changes'].items()
            if n.startswith('shader_network.'+g+'.')) for g in self.groups}
        self.persist()


def run_bounded(trainer,science,torch,np,report,persist,probe_class=MaterialProbe):
    from network.renderer import NeROMaterialRenderer
    from network.field import MCShadingNetwork
    from network.loss import NeRFRenderLoss,MaterialRegLoss
    from train.trainer import Trainer
    from train.train_valid import ValidationEvaluator
    from infrastructure.bounded_training import TwoUpdateTrace
    probe=probe_class(trainer,torch,np,report,persist)
    probe.bind_calls(NeROMaterialRenderer,MCShadingNetwork,NeRFRenderLoss,MaterialRegLoss,Trainer,ValidationEvaluator)
    with probe.observe_calls():TwoUpdateTrace(trainer,probe,science/'train/trainer.py').run()
    require(report['optimizer_calls_completed']==2 and len(report['updates'])==2,'Material smoke did not complete exactly two updates')
