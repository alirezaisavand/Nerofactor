"""Disposable CPU autograd/Adam with the original Trainer class AST and I/O fakes.

This never imports the scientific renderer. Fake datasets/loaders/renderers/loss
producers are explicit; real CPU Torch supplies autograd, clipping and Adam.
"""

import ast
from copy import deepcopy
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def class_node(path, name):
    return next(n for n in ast.parse((REPO/path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)


def defaults(path, name):
    cls=class_node(path,name)
    return next(ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='default_cfg' for t in n.targets))


def trainer_fixture():
    events=[]; loaders=[]; networks=[]; writers=[]
    class Dataset:
        def __init__(self,cfg,is_train,**kwargs):self.is_train=is_train
        def __len__(self):return 99999999 if self.is_train else 1
    class RandomSampler:pass
    def dummy_collate_fn(data):return data[0]
    class Iterator:
        def __init__(self,loader):self.loader=loader;self.closed=False;torch.rand(1,device='cpu')
        def __next__(self):
            events.append('next_batch');torch.rand(1,device='cpu');return {}
        def _shutdown_workers(self):self.closed=True;events.append('shutdown_workers')
    class Loader:
        def __init__(self,dataset,batch_size,shuffle,num_workers,collate_fn):
            self.dataset=dataset;self.batch_size=batch_size;self.num_workers=num_workers
            self.collate_fn=collate_fn;self.drop_last=False;self.sampler=RandomSampler();self.shuffle=shuffle
            self.iterator=None;loaders.append(self)
        def __len__(self):return len(self.dataset)
        def __iter__(self):self.iterator=Iterator(self);return self.iterator
    class Renderer(torch.nn.Module):
        def __init__(self,cfg,*,preprocessing=None,isolated_geometry=False):
            super().__init__();self.cfg={**defaults('network/renderer.py','NeROShapeRenderer'),**deepcopy(cfg)}
            self.isolated_geometry=isolated_geometry;self.weight=torch.nn.Parameter(torch.rand(4,device='cpu')*.1)
            self.inactive=torch.nn.Parameter(torch.tensor([.3],device='cpu'));self.register_buffer('lookup',torch.tensor([.1,.2],device='cpu'))
            from infrastructure.bell_inference import proposed_split
            split=proposed_split();self.database=SimpleNamespace(get_img_ids=lambda:[str(i) for i in range(128)])
            for key,value in split.items():setattr(self,key,np.asarray(value) if key=='train_ids' else value)
            self.tbn=2048;self.train_batch_i=0
            self.train_batch={'dirs':torch.arange(2048*3,dtype=torch.float32,device='cpu').reshape(2048,3),
                              'idxs':torch.arange(2048,dtype=torch.int64,device='cpu').reshape(2048,1)}
            self.color_network=SimpleNamespace(cfg={'CPU_fixture_shader':True})
            self.fail_step=None;self.bad_loss=False;self.bad_gradient=False;self.backward_calls=0
            networks.append(self)
        def cuda(self):return self  # Explicit CPU fixture; never used by the native entry point.
        def forward(self,data):
            step=data['step'];events.append(('forward',step));self.train_batch_i+=512
            if self.fail_step==step:raise RuntimeError('fixture forward failure')
            noise=torch.rand_like(self.weight)*.001
            result=((self.weight-noise-.7)**2).mean()
            if self.bad_loss:result=result*float('nan')
            def gradient_hook(gradient):
                self.backward_calls+=1;events.append('backward')
                return gradient*float('nan') if self.bad_gradient else gradient
            self.weight.register_hook(gradient_hook) if self.backward_calls==0 else None
            return {'CPU_fixture_loss':result}
    class Loss:
        def __init__(self,cfg,name):self.name=name;self.cfg={**cfg,'CPU_fixture_loss':name}
        def __call__(self,outputs,data,step):
            if self.name=='nerf_render':return {'loss_rgb':outputs['CPU_fixture_loss']}
            if self.name=='std':return {'std':outputs['CPU_fixture_loss'].detach()}
            return {}
    class Writer:
        def __init__(self,root):
            self.closed=False;self.path=Path(root)/'events.fixture';self.path.write_text('CPU fixture only\n');writers.append(self)
        def close(self):self.closed=True
    class Logger:
        def __init__(self,root):self.writer=Writer(root)
        def log(self,*args):events.append('unexpected_natural_log')
    class Progress:
        def __init__(self,**kwargs):self.closed=False;events.append(('progress_total',kwargs['total']))
        def update(self,*args):pass
        def set_postfix(self,**kwargs):pass
        def close(self):self.closed=True
    from train.lr_common_manager import name2lr_manager
    scope={'os':os,'random':random,'Path':Path,'torch':torch,'np':np,'Adam':torch.optim.Adam,'SGD':torch.optim.SGD,
           'DataLoader':Loader,'tqdm':Progress,'name2dataset':{'dummy':Dataset},'name2renderer':{'shape':Renderer},
           'name2loss':{name:(lambda cfg,n=name:Loss(cfg,n)) for name in
                ['nerf_render','eikonal','std','init_sdf_reg','occ','mask','curv','opacity']},
           'name2metrics':{'shape_render':lambda *args,**kwargs:SimpleNamespace()},
           'name2lr_manager':name2lr_manager,'ValidationEvaluator':lambda cfg:SimpleNamespace(),
           'Logger':Logger,'dummy_collate_fn':dummy_collate_fn,'to_cuda':lambda v:v}
    node=class_node('train/trainer.py','Trainer')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(REPO/'train/trainer.py'),'exec'),scope)
    return SimpleNamespace(Trainer=scope['Trainer'],events=events,loaders=loaders,networks=networks,writers=writers)


def cpu_rng():
    import hashlib
    state=np.random.get_state()
    return {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'numpy':hashlib.sha256(repr((state[0],state[2:])).encode()+state[1].tobytes()).hexdigest(),
            'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_cuda':[]}
