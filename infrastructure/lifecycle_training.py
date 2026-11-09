"""One original optimizer return at zero-based step2 after explicit restoration."""
import inspect
from pathlib import Path
import sys
import time

from infrastructure.signatures import require
from infrastructure.bounded_training import run_boundaries
from infrastructure.g_training import boundaries,GTrainingProbe
from infrastructure.bell_material_integration import BellProbe
from infrastructure.training_probe import parameter_copies,parameter_changes,snapshot,optimizer_record,array_record,tensor_copy


class _ReloadUpdateComplete(BaseException):
    pass


class ReloadTrace:
    """Separate fixed budget; old two-update trace/acceptance remain untouched."""
    def __init__(self,trainer,probe,path,stage):
        self.trainer=trainer;self.probe=probe;self.code=trainer.run.__func__.__code__
        first,points=(boundaries if stage=='geometry' else run_boundaries)(path)
        require(self.code.co_firstlineno==first and Path(self.code.co_filename).resolve()==Path(path).resolve(),'Unrecognized original Trainer.run')
        self.points={line:name for name,line in points.items()};self.completed=0;self.resources={};self.frame=None
    def observe(self,frame,event,arg):
        if frame.f_code is not self.code:return None
        if event=='call':
            require(self.frame is None and frame.f_locals['self'] is self.trainer,'Reentrant/foreign Trainer.run')
            self.frame=frame
        if event!='line':return self.observe
        v=frame.f_locals
        for name in ['train_iter','pbar']:
            if name in v:self.resources[name]=v[name]
        name=self.points.get(frame.f_lineno)
        if name is None:return self.observe
        if name in ('iteration','before_forward','before_backward','before_clip','before_optimizer','after_optimizer'):
            require(self.completed==0 and v['step']==2 and v['start_step']==2 and v['best_para']==0.,'Post-load step/budget/progress mismatch')
        if name=='after_optimizer':
            self.completed=1
            self.probe.after_optimizer(v,1)
            raise _ReloadUpdateComplete()
        self.probe.event(name,v)
        return self.observe
    def run(self):
        require(sys.implementation.name=='cpython' and sys.gettrace() is None,'Untraced CPython required')
        stopped=False;errors=[]
        try:
            sys.settrace(self.observe)
            require(sys.gettrace()==self.observe,'Reload observer unavailable')
            try:self.trainer.run()
            except _ReloadUpdateComplete:stopped=True
        finally:
            sys.settrace(None)
            for name,obj,method in [('workers',self.resources.get('train_iter'),'_shutdown_workers'),
                    ('progress',self.resources.get('pbar'),'close'),('writer',getattr(getattr(self.trainer,'logger',None),'writer',None),'close')]:
                if callable(getattr(obj,method,None)):
                    try:getattr(obj,method)()
                    except Exception as exc:errors.append({'resource':name,'error':str(exc)})
            self.resources.clear();self.frame=None;self.probe.cleanup(self.completed,errors)
        require(stopped and self.completed==1 and not errors,'Missing bounded post-load completion or cleanup failed')


class ResumeObservation:
    def event(self,boundary,v):
        self.mark(boundary)
        if boundary=='initialized':self.initialize()  # Still fresh here; original load follows.
        elif boundary=='fresh_restore_gate':require(Path(self.trainer.pth_fn).is_file(),'Explicit reload input missing')
        elif boundary=='iteration':
            require(self.report.get('restoration',{}).get('exact_restoration') is True,'No admitted original restoration')
            self.previous_state=optimizer_record(self.trainer.optimizer,self.trainer.network,self.np)
            self.current={'step':2,'rng_before_batch':self.rng(),'started_monotonic':time.monotonic()}
        elif boundary=='before_forward':self.before_forward(v)
        elif boundary=='before_backward':self.before_backward(v)
        elif boundary=='before_clip':self.before_clip(v)
        elif boundary=='before_optimizer':self.before_optimizer(v)
        else:require(boundary in ('dataset_initialization','network_initialization'),'Unknown resume observation boundary')
    def before_forward(self,v):
        t,c,np=self.trainer,self.current,self.np
        expected=t.lr_manager.learning_rate*(2/t.lr_manager.warm_up_end)  # Original order, including float rounding.
        require(v['train_data']=={'step':2} and v['lr']==expected and all(g['lr']==expected for g in t.optimizer.param_groups),
                'Original scheduler failed step2 continuity')
        require(t.network.training and self.torch.is_grad_enabled(),'Original training autograd disabled')
        cursor=t.network.train_batch_i;count=t.network.cfg['train_ray_num']
        require(cursor==0 and count<t.network.tbn,'Fresh-process original ray cursor was not reset as audited')
        c.update(lr=float(v['lr']),rng_before_forward=self.rng(),memory_before=self.memory(),calls={},
                 light_paths={'inner':0,'outer':0,'human':0},sampling={'cursor_before':cursor,'ray_count':count,'pool_size':t.network.tbn,
                    'selected':{k:array_record(tensor_copy(x[:count],np),np) for k,x in t.network.train_batch.items()}})
        self.before_parameters=parameter_copies(t.network,np);c['parameters_before']=snapshot(self.before_parameters,np)
    def after_optimizer(self,v,completed):
        c,t,np=self.current,self.trainer,self.np
        self.report['optimizer_calls_completed']=completed
        require(completed==1 and c['step']==2,'Unexpected restored update count')
        key='gradients_before_optimizer' if isinstance(self,GResumeProbe) else 'gradients_after_clip'
        require(key in c,'Missing original backward/gradient evidence')
        after=parameter_copies(t.network,np);changes=parameter_changes(self.before_parameters,after,np)
        changed=sum(x['changed_elements'] for x in changes.values());require(changed>0,'Restored nonzero-LR update did not change parameters')
        state=optimizer_record(t.optimizer,t.network,np);previous=self.previous_state['state']
        active={n for n,g in c[key].items() if g is not None}
        require(set(state['state'])==set(previous)|active and all(x['step']==previous.get(n,{}).get('step',0)+int(n in active)
                for n,x in state['state'].items()),'Restored Adam counters/state trajectory mismatch')
        require(any(x['exp_avg']['sha256']!=previous.get(n,{}).get('exp_avg',{}).get('sha256') for n,x in state['state'].items()),
                'Restored optimizer moments did not change')
        require(t.network.train_batch_i==t.network.cfg['train_ray_num'],'Unexpected original resumed sampling cursor')
        c.update(update_index=1,parameters_after=snapshot(after,np),parameter_changes=changes,changed_elements=changed,
                 optimizer=state,rng_after=self.rng(),memory_after=self.memory(),cursor_after=t.network.train_batch_i,
                 elapsed_seconds=time.monotonic()-c.pop('started_monotonic'))
        self.report['updates'].append(c);self.current=self.before_parameters=None
        self.report['scheduler_continuity']={'loaded_lr':self.report['restoration']['restored_progress_lr'],
            'actual_step2_lr':c['lr'],'original_scheduler_step':2,'verified':True}
        self.report['continuation']={'RNG_restored':False,'DataLoader_restored':False,'ray_cursor_restored':False,
            'actual_cursor_before':c['sampling']['cursor_before'],'actual_cursor_after':c['cursor_after'],
            'sampling_compared_to_save_prefix':c['sampling']['selected']==self.report['producer_sampling']['selected'],
            'uninterrupted_bitwise_equivalence_proven':False}
        self.persist()


class GResumeProbe(ResumeObservation,GTrainingProbe):
    pass


class AResumeProbe(ResumeObservation,BellProbe):
    pass


class ForbidCalls:
    """Block genuine bodies, never shared Torch decorator wrapper code."""
    def __init__(self,functions):self.codes={inspect.unwrap(f).__code__ for f in functions}
    def observe(self,frame,event,value):
        require(not(event=='call' and frame.f_code in self.codes),'Forbidden validation/checkpoint/NVS/update during lifecycle action')
    def __enter__(self):
        require(sys.getprofile() is None,'Unprofiled process required');sys.setprofile(self.observe);return self
    def __exit__(self,*args):sys.setprofile(None)
