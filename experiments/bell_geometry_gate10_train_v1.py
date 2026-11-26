"""Split-scoped acquisition and lifecycle observation around pristine G Trainer.run."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
from shutil import copyfile
import sys

import numpy as np

from benchmarks import bell_geometry_gate10_v1 as g
from infrastructure.g_checkpoint import identities, load_checked, optimizer_preflight, tensor_catalog
from infrastructure.g_training import boundaries
from infrastructure.signatures import require
from infrastructure.stage_sources import imported_sources


class Acquisition:
    def __init__(self, root, cameras, splits):
        self.root,self.cameras,self.splits=root,cameras,splits
        self.scope=None
        self.split_calls=0
        self.data_reads={}

    @contextmanager
    def visiting(self, name):
        require(self.scope is None and name in self.splits and name!='held_view_nvs','Unexpected geometry data scope')
        self.scope=name
        try:
            yield
        finally:
            self.scope=None

    def access(self, key, camera):
        allowed=self.splits.get(self.scope,[])
        require(str(camera) in allowed and (key!='depth' or self.scope=='validation'),
                'Unauthorized G view/depth access '+str(camera))
        counts=self.data_reads.setdefault(self.scope,{})
        counts[key]=counts.get(key,0)+1

    def bind(self, database, renderer, train_dataset):
        parent=database.GlossySyntheticDatabase
        a=self

        class BellViews(parent):
            def __init__(self, name):
                database.BaseDatabase.__init__(self,name)
                self.root=str(a.root/'bell')
                self.scale_factor=1.

            def get_img_ids(self):
                require(a.scope is not None,'Unscoped camera inventory read')
                return list(a.splits[a.scope])

            def get_image(self, camera):
                a.access('rgb',camera)
                return super().get_image(camera)

            def get_mask(self, camera):
                a.access('mask',camera)
                return super().get_mask(camera)

            def get_depth(self, camera):
                a.access('depth',camera)
                return super().get_depth(camera)

            def get_K(self, camera):
                a.access('K',camera)
                return a.cameras[str(camera)][1].copy()

            def get_pose(self, camera):
                a.access('pose',camera)
                return a.cameras[str(camera)][0].copy()

        def parse(name, dataset_dir):
            require(name=='syn/bell' and Path(dataset_dir)==a.root,'Unexpected G dataset address')
            return BellViews(name)

        def split(instance, split_type='validation'):
            require(isinstance(instance,BellViews) and split_type=='validation','Unexpected G split request')
            # Pristine helper resets the Python RNG on both validation-split calls.
            database.random.seed(g.SEED)
            a.split_calls+=1
            return list(a.splits['train']),list(a.splits['validation'])

        build=renderer.build_imgs_info
        def build_scoped(instance, ids, is_nerf=False):
            ordered=list(map(str,ids))
            roles=[role for role in ('train','validation') if ordered==a.splits[role]]
            require(isinstance(instance,BellViews) and len(roles)==1 and not is_nerf,'Unexpected G raster dispatch')
            with a.visiting(roles[0]):
                return build(instance,ids,is_nerf)

        for module in (renderer,train_dataset):
            module.parse_database_name=parse
            module.get_database_split=split
        renderer.build_imgs_info=build_scoped


def state_sha(values):
    rows=identities(values)
    h=hashlib.sha256()
    for name,row in sorted(rows.items()):
        h.update(repr((name,row)).encode())
    return h.hexdigest()


def own_checkpoint(path, step, network, torch, optimizer=None, expected=None):
    identity=g.item(path)
    require(expected is None or identity==expected,'Owned G checkpoint differs from training receipt')
    ckpt=load_checked(path,identity['sha256'],torch,trusted=True)
    require(set(ckpt)=={'step','best_para','network_state_dict','optimizer_state_dict'}
            and type(ckpt['step']) is int and ckpt['step']==step and math.isfinite(float(ckpt['best_para'])),
            'Unexpected native checkpoint schema or step')
    state=ckpt['network_state_dict']
    require(tensor_catalog(state,torch)==tensor_catalog(network.state_dict(),torch)
            and all(x.device.type=='cpu' for x in state.values()), 'Owned G weight layout differs')
    if optimizer is not None:
        optimizer_preflight(optimizer,ckpt['optimizer_state_dict'],network,torch)
        require(identities(state)==identities(network.state_dict()),'Natural G checkpoint weight bytes differ')
    return identity,ckpt


def run(out, source, f, report, torch, cfg, budget, progress):
    from dataset import database, train_dataset
    from network import renderer
    from train import trainer as trainer_module
    from train.trainer import Trainer
    from infrastructure.bell_geometry_workers_v2 import spawned_loaders

    train_root=Path('/inputs/training')
    acquisition=Acquisition(train_root,g.read_staged(f,train_root),f['splits'])
    acquisition.bind(database,renderer,train_dataset)
    cfg=dict(cfg,dataset_dir=str(train_root),train_dataset_cfg=dict(cfg['train_dataset_cfg'],dataset_dir=str(train_root)))
    require(cfg['total_step']==g.STEP and cfg['val_interval']==cfg['save_interval']==g.CADENCE,
            'Unexpected G Bell training configuration')
    (out/'checkpoints').mkdir()
    (out/'logs').mkdir()
    (out/'workspace').mkdir()
    (out/'workspace'/'assets').symlink_to(source/'assets',target_is_directory=True)
    (out/'workspace'/'configs').symlink_to(source/'configs',target_is_directory=True)

    class OwnedTrainer(Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.model_dir=str(out/'logs')
            self.pth_fn=str(out/'checkpoints/model.pth')
            self.best_pth_fn=str(out/'checkpoints/model_best.pth')

        def _load_model(self):
            require(not Path(self.pth_fn).exists() and not Path(self.best_pth_fn).exists(),'Fresh G only')
            return super()._load_model()

        def _save_model(self, step, best_para, save_fn=None):
            require(step==report['adam_returns'] and step % g.CADENCE==0 and
                    report['validation_calls']==step//g.CADENCE, 'Checkpoint outside natural validation boundary')
            best=save_fn is not None
            key='best_saves' if best else 'periodic_saves'
            maximum=g.LIMITS['best_saves_max' if best else 'periodic_saves']
            require(report[key]<maximum and (not best or save_fn==self.best_pth_fn),'Unexpected native G checkpoint destination')
            super()._save_model(step,best_para,save_fn)
            report[key]+=1
            item=g.item(Path(self.best_pth_fn if best else self.pth_fn))
            row=dict(step=step,kind=key,identity=item,best_validation_PSNR=float(best_para))
            if not best and step==g.CADENCE:
                budget(first=True)
                artifact,saved=own_checkpoint(Path(self.pth_fn),step,self.network,torch,self.optimizer)
                require(item==artifact,'First natural G checkpoint changed')
                report['step_1000_gate']=dict(passed=True,validation=report['last_validation'],
                                             checkpoint=artifact,own_cpu_roundtrip=True,tensors=len(saved['network_state_dict']))
            if not best and step in g.KEEP_STEPS:
                preserved=out/f'checkpoints/model-step{step}.pth'
                copyfile(self.pth_fn,preserved)
                require(g.item(preserved)==item,'Intermediate checkpoint preservation failed')
                report.setdefault('retained_checkpoints',{})[str(step)]=item
            if not best and step==g.STEP:
                report['endpoint_checkpoint']=item
                report['endpoint_network_sha256']=state_sha(self.network.state_dict())
            with (out/'checkpoint-events.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            budget()
            progress()

    def initialize():
        n=t.network
        require(list(n.train_ids)==f['splits']['train'] and n.test_ids==f['splits']['validation']
                and acquisition.split_calls==2 and n.tbn==122*g.SIZE**2
                and n.cfg['train_ray_num']==512 and n.cfg['test_ray_num']==512
                and n.cfg['test_downsample_ratio'] and n.cfg['downsample_ratio']==.25 and not n.cfg['val_geometry']
                and n.cfg['apply_occ_loss'] and n.cfg['occ_loss_step']==20000 and n.cfg['occ_loss_max_pn']==2048
                and [n.cfg[k] for k in ('n_samples','n_bg_samples','n_importance','up_sample_steps')]==[64,32,64,4]
                and len(t.val_set_list)==1 and len(t.val_set_list[0])==1
                and type(t.optimizer) is torch.optim.Adam and not t.optimizer.state
                and t.train_set.num_workers==8 and t.cfg['random_seed']==g.SEED
                and t.lr_manager.warm_up_end==5000 and t.lr_manager.end_iter==g.STEP
                and t.lr_manager.learning_rate==5e-4 and t.lr_manager.learning_rate_alpha==.05,
                'Effective G training configuration or split differs')
        require(t.optimizer.defaults==dict(lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
                and len(t.optimizer.param_groups)==1,'Unexpected G Adam defaults')
        report.update(initial_state_sha256=state_sha(n.state_dict()),effective_trainer=t.cfg,effective_renderer=n.cfg,
                      effective_shader=n.color_network.cfg,scheduler=dict(vars(t.lr_manager)),
                      optimizer_defaults=dict(t.optimizer.defaults),ray_pool_size=n.tbn)
        original_test=n.test_step
        def test_scoped(index,step):
            require(index==0 and step==report['adam_returns']-1,'Unexpected validation view/step')
            with acquisition.visiting('validation'):
                return original_test(index,step)
        n.test_step=test_scoped
        original_validation=t.val_evaluator
        def validation(*args,**kwargs):
            completed=args[3]+1
            require(completed==report['adam_returns'] and completed==(report['validation_calls']+1)*g.CADENCE,
                    'Validation occurred outside native schedule')
            metrics,score=original_validation(*args,**kwargs)
            require(math.isfinite(float(score)) and all(np.isfinite(v).all() for v in metrics.values()),
                    'Nonfinite G validation outputs')
            report['validation_calls']+=1
            report['last_validation']=dict(checkpoint_step=completed,view=f['splits']['validation'][0],
                                           native_PSNR=float(score),metric_means={k:float(np.mean(v)) for k,v in metrics.items()})
            return metrics,score
        t.val_evaluator=validation
        progress()

    os.chdir(out/'workspace')
    t=OwnedTrainer()
    code=t.run.__func__.__code__
    first,site=boundaries(source/'train/trainer.py')
    require(Path(code.co_filename).resolve()==source/'train/trainer.py' and code.co_firstlineno==first
            and sys.gettrace() is None,'Pinned untraced G trainer required')
    sites={v:k for k,v in site.items()}
    owned={}
    warm={}
    probes={1,2,1000,5000,15000,15001,20001,30000,30001,50000,50001,100000,200000,g.STEP}
    def observe(frame,event,arg):
        if frame.f_code is not code:
            return None
        require(frame.f_locals.get('self') is t,'Unexpected G Trainer frame')
        if event!='line':
            return observe
        vals=frame.f_locals
        for name in ('train_iter','pbar'):
            if name in vals:
                owned[name]=vals[name]
        name=sites.get(frame.f_lineno)
        if name=='initialized':
            initialize()
        if name=='iteration':
            require(vals['step']==report['adam_returns']<g.STEP and vals['start_step']==0,
                    'G update order differs')
            if vals['step']==g.CADENCE:
                require(report.get('step_1000_gate',{}).get('passed') is True,'First natural G gate missing')
            if vals['step']%g.CADENCE==0:
                budget(first=vals['step']==0)
        if name=='before_backward':
            require(torch.is_tensor(vals['loss']) and bool(torch.isfinite(vals['loss']).item()),'Nonfinite G training loss')
        if name=='before_optimizer' and vals['step']+1 in probes:
            active=[p for p in t.network.parameters() if p.grad is not None]
            grads=[p.grad for p in active]
            require(grads and bool(torch.stack([x.isfinite().all() for x in grads]).all().item()),
                    'Missing or nonfinite sampled G gradient')
            warm['active_parameters']=len(grads)
            warm['gradient_norm']=float(torch.sqrt(torch.stack([x.detach().square().sum() for x in grads]).sum()).item())
            warm['loss_components']={key:float(torch.mean(value).detach().cpu().item())
                                     for key,value in vals['log_info'].items() if key.startswith('loss')}
            require(math.isfinite(warm['gradient_norm']) and all(math.isfinite(v) for v in warm['loss_components'].values()),
                    'Nonfinite weighted G loss/gradient witness')
            if vals['step']<2:
                warm['before']=state_sha(dict(t.network.named_parameters()))
                warm['prior_steps']={p:t.optimizer.state.get(p,{}).get('step',0) for p in active}
        if name=='after_optimizer':
            report['adam_returns']+=1
            k=report['adam_returns']
            if k in probes:
                row=dict(completed=k,LR=float(vals['lr']),active_parameters=warm['active_parameters'],
                         weighted_loss_components=warm['loss_components'],preclip_gradient_norm=warm['gradient_norm'])
                if k<=2:
                    after=state_sha(dict(t.network.named_parameters()))
                    require((warm['before']==after)==(k==1),'Unexpected G warm-up displacement')
                    require(all(t.optimizer.state[p]['step']==prior+1 and
                                all(torch.isfinite(t.optimizer.state[p][key]).all().item() for key in ('exp_avg','exp_avg_sq'))
                                for p,prior in warm['prior_steps'].items()),
                            'Expected finite advancing Adam moments during warm-up')
                    row['parameters_changed']=warm['before']!=after
                report['training_witnesses'].append(row)
                progress()
        return observe

    cleanup=[]
    try:
        sys.settrace(observe)
        with spawned_loaders(trainer_module,report):
            t.run()
    finally:
        sys.settrace(None)
        for target,method in ((owned.get('train_iter'),'_shutdown_workers'),(owned.get('pbar'),'close'),
                              (getattr(getattr(t,'logger',None),'writer',None),'close')):
            callback=getattr(target,method,None)
            if callable(callback):
                try:
                    callback()
                except Exception as exc:
                    cleanup.append(str(exc))
        report['cleanup_errors']=cleanup
        report['data_access']=acquisition.data_reads
        report['imported_scientific_modules']=imported_sources(source)
        progress()
    require(not cleanup and report['adam_returns']==g.STEP and report['validation_calls']==g.LIMITS['validation_calls']
            and report['periodic_saves']==g.LIMITS['periodic_saves']
            and report['data_access']==dict(train={k:122 for k in ('rgb','mask','K','pose')},
                                           validation={**{k:1 for k in ('rgb','mask','K','pose')},'depth':300}),
            'G training did not satisfy the authorized lifecycle or data access budget')
