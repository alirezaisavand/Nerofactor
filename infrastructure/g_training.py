"""Observe the pristine G loop, with no clipping operation or A acceptance policy."""
import ast
from pathlib import Path
import time

from infrastructure.bounded_training import TwoUpdateTrace
from infrastructure.training_probe import (TrainingProbe, parameter_copies, parameter_changes,
                                           optimizer_record, snapshot, tensor_copy)
from infrastructure.signatures import require

PROTOCOL = 'bell-G-two-update-training-v1'


def boundaries(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in run.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    def name(n):
        if isinstance(n,ast.Call): return name(n.func)
        if isinstance(n,ast.Attribute): return name(n.value)+'.'+n.attr
        return n.id if isinstance(n,ast.Name) else ''
    def find(body,target):
        matches=[n for n in body if name(getattr(n,'value',None))==target]
        require(len(matches)==1,'Unrecognized pristine G boundary: '+target)
        return matches[0]
    require(not any(isinstance(n,ast.Call) and 'clip_grad' in name(n) for n in ast.walk(run)), 'G loop contains clipping')
    opt=find(loop.body,'self.optimizer.step')
    points={'dataset_initialization':find(run.body,'self._init_dataset').lineno,
            'network_initialization':find(run.body,'self._init_network').lineno,
            'initialized':find(run.body,'self._init_logger').lineno,
            'fresh_restore_gate':find(run.body,'self._load_model').lineno,
            'iteration':loop.body[0].lineno,'before_forward':find(loop.body,'self.train_network').lineno,
            'before_backward':find(loop.body,'loss.backward').lineno,
            'before_optimizer':opt.lineno,'after_optimizer':loop.body[loop.body.index(opt)+1].lineno}
    require(len(set(points.values()))==len(points),'Ambiguous G trace boundaries')
    return run.lineno,points


class GTwoUpdateTrace(TwoUpdateTrace):
    """Reuse only the stage-neutral stop/cleanup transport, not A boundary policy."""
    def __init__(self,trainer,observer,path):
        self.trainer,self.observer=trainer,observer
        self.code=trainer.run.__func__.__code__
        first,points=boundaries(path)
        require(Path(self.code.co_filename).resolve()==Path(path).resolve() and self.code.co_firstlineno==first,
                'Run body is not the pristine G trainer')
        self.points={line:name for name,line in points.items()}
        self.completed=0;self.iterator=self.progress=self.frame=None;self.cleanup_errors=[];self.started=False


class GTrainingProbe(TrainingProbe):
    """G-specific checks; inherited reductions/forward diagnostics are read-only."""
    def initialize(self):
        t,np,torch=self.trainer,self.np,self.torch
        require(type(t.optimizer) is torch.optim.Adam and t.train_network is t.network,'Expected original single-GPU G Adam')
        require(t.cfg['random_seed']==6033 and t.cfg['total_step']==300000 and t.cfg['lr_cfg']=={} and
                t.cfg['worker_num']==8 and t.cfg['val_interval']==t.cfg['save_interval']==1000 and t.cfg['train_log_step']==20,
                'Pinned G bell settings changed')
        require([str(v) for v in t.network.train_ids]==[str(i) for i in range(1,128)] and
                t.network.test_ids==['0'] and not hasattr(t.network,'isolated_geometry'),'G two-way split was replaced')
        initial=parameter_copies(t.network,np)
        self.report.update(initial_parameters=snapshot(initial,np),
            initial_buffers={n: snapshot({n:tensor_copy(v,np)},np)[n] for n,v in t.network.named_buffers()},
            splits={'train_ids':[str(v) for v in t.network.train_ids],'test_ids':list(t.network.test_ids),'nvs_ids':None},
            effective_trainer=t.cfg,effective_renderer=t.network.cfg,effective_shader=t.network.color_network.cfg,
            effective_losses=[{'registered_name':name,'class':type(loss).__module__+'.'+type(loss).__name__,
                               'configuration':getattr(loss,'cfg',None)} for name,loss in zip(t.cfg['loss'],t.train_losses)],
            scheduler={'class':type(t.lr_manager).__name__,'attributes':dict(vars(t.lr_manager))},
            clipping={'enabled':False,'source':'pristine G Trainer.run; no clipping call'},
            loader={'batch_size':t.train_set.batch_size,'num_workers':t.train_set.num_workers,
                    'dataset_size':len(t.train_set.dataset),'sampler':type(t.train_set.sampler).__name__})
        self.previous_state=optimizer_record(t.optimizer,t.network,np)
        require(not self.previous_state['state'] and len(self.previous_state['groups'])==1,'Optimizer not fresh')
        g=self.previous_state['groups'][0]
        require(g['lr']==1e-3 and g['eps']==1e-8 and g['amsgrad'] is False and tuple(g['betas'])==(0.9,0.999)
                and g['weight_decay']==0,'Unexpected G native Adam defaults')
        require(t.lr_manager.learning_rate_alpha==0.05 and t.lr_manager.end_iter==300000 and
                t.lr_manager.warm_up_end==5000 and t.lr_manager.learning_rate==5e-4,'G scheduler was replaced')
        self.report['initial_optimizer']=self.previous_state
        self.persist()

    def event(self,boundary,values):
        self.mark(boundary)
        if boundary=='initialized':self.initialize()
        elif boundary=='fresh_restore_gate':
            require(not Path(self.trainer.pth_fn).exists() and not Path(self.trainer.best_pth_fn).exists(),'Fresh G checkpoint collision')
        elif boundary=='iteration':
            require(values['start_step']==0 and values['best_para']==0,'G run is not fresh')
            self.current={'step':values['step'],'rng_before_batch':self.rng(),'started_monotonic':time.monotonic()}
        elif boundary=='before_forward':self.before_forward(values)
        elif boundary=='before_backward':self.before_backward(values)
        elif boundary=='before_optimizer':self.before_optimizer(values)
        else:require(boundary in ('dataset_initialization','network_initialization'),'Unknown G observation boundary')

    def before_forward(self,values):
        t,c,np=self.trainer,self.current,self.np
        require(c is not None and values['train_data']=={'step':c['step']} and values['step']==c['step'],
                'Unexpected G training dispatch')
        # Source-verified G warm-up, evaluated diagnostically without another scheduler call.
        expected=t.lr_manager.learning_rate*(c['step']/t.lr_manager.warm_up_end)
        require(values['lr']==expected and all(g['lr']==expected for g in t.optimizer.param_groups),
                'G warm-up learning rate differs')
        require(t.network.training and self.torch.is_grad_enabled(),'G training gradients disabled')
        c.update(lr=float(values['lr']),rng_before_forward=self.rng(),memory_before=self.memory())
        start,count=t.network.train_batch_i,t.network.cfg['train_ray_num']
        require(start==c['step']*count and start+count<t.network.tbn,'G ray cursor/reshuffle changed')
        from infrastructure.training_probe import array_record
        c['sampling']={'cursor_before':start,'ray_count':count,'pool_size':t.network.tbn,
                       'selected':{k:array_record(tensor_copy(v[start:start+count],np),np) for k,v in t.network.train_batch.items()}}
        self.before_parameters=parameter_copies(t.network,np)
        c['parameters_before']=snapshot(self.before_parameters,np)

    def before_optimizer(self,values):
        gradients,norm=self.gradients()
        require(norm>0,'No finite nonzero G gradients')
        self.current.update(gradients_before_optimizer=gradients,gradient_l2_before_optimizer_cpu64=norm,
                            clipping_applied=False)
        self.persist()

    def after_optimizer(self,values,completed):
        self.mark('after_optimizer')
        c,t,np=self.current,self.trainer,self.np
        self.report['optimizer_calls_completed']=completed
        require(c is not None and c['step']+1==completed and 'gradients_before_optimizer' in c,'Missing G update evidence')
        after=parameter_copies(t.network,np);changes=parameter_changes(self.before_parameters,after,np)
        changed=sum(v['changed_elements'] for v in changes.values())
        require((completed==1 and changed==0) or (completed==2 and changed>0),'Unexpected G warm-up parameter changes')
        state=optimizer_record(t.optimizer,t.network,np)
        active={n for n,v in c['gradients_before_optimizer'].items() if v is not None}
        previous=self.previous_state['state']
        require(set(state['state'])==set(previous)|active,'G optimizer state mismatch')
        require(all(set(v)=={'step','exp_avg','exp_avg_sq'} and
                    v['step']==previous.get(n,{}).get('step',0)+int(n in active) for n,v in state['state'].items()),
                'G Adam step/state differs; AMSGrad is not enabled')
        c.update(update_index=completed,parameters_after=snapshot(after,np),parameter_changes=changes,
                 changed_elements=changed,optimizer=state,rng_after=self.rng(),memory_after=self.memory(),
                 elapsed_seconds=time.monotonic()-c.pop('started_monotonic'),cursor_after=t.network.train_batch_i)
        require(c['cursor_after']==completed*t.network.cfg['train_ray_num'],'G ray cursor changed')
        self.report['updates'].append(c);self.previous_state=state;self.current=self.before_parameters=None
        self.persist()
