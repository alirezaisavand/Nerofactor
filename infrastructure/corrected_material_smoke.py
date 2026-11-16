"""Two original A optimizer returns for explicit native derivatives, strict G handoff.

No historical-mesh exception. Reuse the accepted trace, resource cleanup, Adam,
clipping and parameter-change guards. No model imports outside native runtime.
"""
import inspect

from infrastructure.signatures import require
from infrastructure.material_smoke import MaterialProbe,GROUPS
from infrastructure.training_probe import TrainingProbe,parameter_copies,snapshot,optimizer_record,array_record,tensor_copy


class CorrectedProbe(MaterialProbe):
    def prepare_calls(self,renderer,rgb_loss,regularizer,trainer_class,evaluator):
        """Guard forbidden bodies now; actual shader identity requires lazy init."""
        require(not hasattr(self,'_binding_types'),'Corrected observer guards already prepared')
        self._binding_types=(renderer,rgb_loss,regularizer,trainer_class,evaluator)
        self._binding_count=0
        self.calls={}
        self.forbidden={inspect.unwrap(getattr(renderer,n)).__code__ for n in
                        ('test_step','nvs','predict_materials')}
        self.forbidden.update(inspect.unwrap(f).__code__ for f in (trainer_class._save_model,evaluator.__call__))
        self.report['observer_lifecycle']={'guards_prepared_before_run':True,'method_bindings':0}
        self.persist()

    def event(self,boundary,values):
        require(hasattr(self,'_binding_types'),'Corrected observer guards were not prepared')
        if boundary=='initialized':
            require(self._binding_count==0,'Corrected observer initialized more than once')
            network=getattr(self.trainer,'network',None)
            shader=getattr(network,'shader_network',None)
            require(network is not None and shader is not None,'Original network/shader missing at initialized boundary')
            renderer,rgb_loss,regularizer,trainer_class,evaluator=self._binding_types
            require(type(network) is renderer,'Initialized renderer differs from selected registry class')
            early_guards=self.forbidden.copy()
            self.bind_calls(type(network),type(shader),rgb_loss,regularizer,trainer_class,evaluator)
            require(early_guards<=self.forbidden,'Lazy binding removed an early forbidden-call guard')
            self._binding_count=1
            self.report['observer_lifecycle']['method_bindings']=1
            self.persist()
        elif boundary in ('fresh_restore_gate','iteration','before_forward','before_backward','before_clip','before_optimizer'):
            require(self._binding_count==1,'Missing initialized observer boundary before training')
        super().event(boundary,values)

    def initialize(self):
        require(getattr(self,'_binding_count',0)==1,'Corrected observer methods not bound at initialization')
        t=self.trainer;shader=t.network.shader_network
        require(t.train_network is t.network and not t.cfg['multi_gpus'],'Single original A network required')
        require(t.cfg['total_step'] in (100000,2000) and t.cfg['random_seed']==6033 and t.cfg['worker_num']==8 and
                t.cfg['val_interval']==t.cfg['save_interval']==2000 and t.cfg['train_log_step']==10,'Corrected smoke budget/configuration differs')
        # Original-A configurations preserve their implicit legacy variant; the
        # missing key is not permission to select a corrected renderer/shader.
        selected_variant=t.cfg.get('material_variant','A_original')
        require(t.network.material_variant==selected_variant==shader.material_variant,'Corrected shader/renderer not selected')
        self.groups=tuple(n for n in GROUPS if n!='human_light' or shader.cfg['human_lights'])
        initial=parameter_copies(t.network,self.np);require(initial,'No corrected native parameters')
        self.previous_state=optimizer_record(t.optimizer,t.network,self.np)
        g=self.previous_state['groups'][0]
        require(type(t.optimizer) is self.torch.optim.Adam and len(self.previous_state['groups'])==1 and not self.previous_state['state'] and
                g['eps']==1e-4 and g['amsgrad'] and g['lr']==.001 and tuple(g['betas'])==(.9,.999) and g['weight_decay']==0,'Original A fresh Adam required')
        require(t.lr_manager.warm_up_end==1000 and t.lr_manager.end_iter==100000 and t.lr_manager.learning_rate==5e-4 and t.lr_manager.learning_rate_alpha==.2,'Original A schedule required')
        self.report.update(initial_parameters=snapshot(initial,self.np),initial_optimizer=self.previous_state,effective_trainer=t.cfg,
            effective_renderer=t.network.cfg,effective_shader=shader.cfg,
            initial_buffers={n:array_record(tensor_copy(v,self.np),self.np) for n,v in t.network.named_buffers()},
            expected_absent_groups={'human_light':'disabled by explicit shader_cfg'} if 'human_light' not in self.groups else {},
            splits={n:list(map(str,getattr(t.network,n))) for n in ('train_ids','test_ids','nvs_ids')})
        self.persist()

    def group_gradients(self,records):
        if self.trainer.network.shader_network.cfg.get('illumination_mode')!='known_env_v1':
            return super().group_gradients(records)
        old=self.current['light_paths']
        self.current['light_paths']={name:0 for name in old}
        try:groups=super().group_gradients(records)
        finally:self.current['light_paths']=old
        for name in ('outer_light','inner_light'):
            groups[name]['reason']='Explicit known_env_v1 / zero_bounce_v1; native learned light networks are intentionally inactive'
        return groups

    def before_backward(self,values):
        TrainingProbe.before_backward(self,values)
        names=['shader.'+n for n in ('forward','anisotropic_forward','shade_anisotropic_mixed','sample_aniso_ggx_directions',
                'compute_tangent_bitangent_flat','compute_radiance')]+['renderer.compute_rgb_loss','loss.nerf_render']
        if self.trainer.network.cfg['reg_mat']:names+=['shader.anisotropic_regularization','loss.mat_reg']
        require(all(self.current['calls'].get(n)=={'entered':1,'returned':1} for n in names),'Corrected native forward/RGB path missing or repeated')
        require('loss_rgb' in self.current['losses'],'Native RGB loss missing')
        self.current['forward_path_verified']=True


def run_bounded(trainer,science,torch,np,report,persist):
    from network.renderer import name2renderer
    from network.loss import NeRFRenderLoss,MaterialRegLoss
    from train.trainer import Trainer
    from train.train_valid import ValidationEvaluator
    from infrastructure.bounded_training import TwoUpdateTrace
    probe=CorrectedProbe(trainer,torch,np,report,persist)
    probe.prepare_calls(name2renderer[trainer.cfg['network']],NeRFRenderLoss,MaterialRegLoss,Trainer,ValidationEvaluator)
    with probe.observe_calls():TwoUpdateTrace(trainer,probe,science/'train/trainer.py').run()
    require(probe._binding_count==1 and report['optimizer_calls_completed']==2 and len(report['updates'])==2,
            'Corrected smoke did not bind once and stop after two original returns')
