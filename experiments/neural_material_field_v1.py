"""3D-2 constant material representations, with exact admitted A components.

No native renderer imports, global source patches, or scientific execution.
"""
import ast
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import run_known_material_recovery_v1 as direct
from experiments.material_shader_v1 import encoding_status
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline, PINS, digest

PROTOCOL = 'constant-neural-material-field-v1'
SOURCE_SHA = 'b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe'
DEFINITIONS = ('Embedder','get_embedder','IdentityActivation','ExpActivation',
               'make_predictor','MaterialFeatsNetwork')
HEADS = (('mx_predictor',1),('my_predictor',1),('metallic_predictor',1),
         ('rotation_predictor',2),('kd_predictor',3))
RHO = .5
PARAMETER_COUNT = 1485584
SEEDS = (101,103,105)


def record(tensor, values=False):
    a=tensor.detach().cpu().numpy()
    out=dict(dtype=str(tensor.dtype),device=str(tensor.device),shape=list(tensor.shape),
             sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
    if values:
        out['values']=a.tolist()
    return out


def cpu_float(value, shape, label):
    require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and
            tuple(value.shape)==tuple(shape) and bool(torch.isfinite(value).all()), 'CPU finite float32 '+label+' required')


def source_definitions():
    """Compile a closed list of unmodified AST definitions, not the field module."""
    root,item=baseline('material',PINS['material'])
    path=root/'network/field.py'
    require(digest(path)==SOURCE_SHA,'Original A material source identity mismatch')
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in DEFINITIONS]
    require({n.name for n in nodes}==set(DEFINITIONS),'Missing original material definition')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork')
    predict=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='predict_anisotropic_components')
    nodes.append(predict)
    for node in nodes:
        require(not node.decorator_list and not any(isinstance(n,(ast.Import,ast.ImportFrom)) or
            isinstance(n,ast.Attribute) and n.attr=='cuda' for n in ast.walk(node)), 'Unexpected dependency/decorator in admitted CPU definition')
    namespace={'torch':torch,'nn':nn,'F':F,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    identity=dict(protocol=PROTOCOL,scientific_commit=item['commit'],source='scientific_sources/material_a/network/field.py',
        source_sha256=SOURCE_SHA,definition_AST_sha256={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in nodes})
    return namespace,identity


@dataclass(frozen=True)
class Material:
    """Physical [N,7] diagnostics and the ACTUAL [N,2] shader encoding."""
    physical: torch.Tensor
    encoding: torch.Tensor
    rotation_raw: torch.Tensor = None
    centered_radius: torch.Tensor = None

    def validate(self):
        require(isinstance(self.physical,torch.Tensor) and self.physical.ndim==2,'Physical material [N,7] required')
        n=self.physical.shape[0]
        cpu_float(self.physical,(n,7),'physical material')
        cpu_float(self.encoding,(n,2),'double-angle encoding')
        require(n>0,'Nonempty material batch required')
        p=self.physical
        require(bool(((p[:,:2]>=.04)&(p[:,:2]<=1)).all()) and
                bool(((p[:,3:]>=0)&(p[:,3:]<=1)).all()),'Material outside original physical bounds')
        require(bool(encoding_status(self.encoding)['valid'].all()),'Invalid zero/near-zero shader orientation encoding')
        return self

    def expand(self,n):
        require(type(n) is int and n>0 and self.physical.shape[0]==1,'One constant material required for expansion')
        return Material(self.physical.expand(n,7),self.encoding.expand(n,2),
            self.rotation_raw.expand(n,2) if self.rotation_raw is not None else None,
            self.centered_radius.expand(n,1) if self.centered_radius is not None else None)


class _Vector:
    def forward(self,positions):
        cpu_float(positions,(len(positions),3),'query positions')
        cpu_float(self.latent,(7,),'latent coordinates')
        p=direct.physical(self.latent)  # Original operation order, including raw theta.
        e=torch.stack((torch.cos(2*p[2]),torch.sin(2*p[2]))).reshape(1,2)
        return Material(p.unsqueeze(0),e).validate().expand(len(positions))


class DirectMaterial(_Vector):
    arm='D0'
    def __init__(self,latent):
        cpu_float(latent,(7,),'initial coordinates')
        self.latent=latent.detach().clone().requires_grad_()
    __call__=_Vector.forward
    def named_parameters(self):return (('latent',self.latent),)
    def parameters(self):return (self.latent,)
    def named_buffers(self):return ()


class ConstantMaterial(_Vector,nn.Module):
    arm='N0'
    def __init__(self,latent):
        super().__init__()
        cpu_float(latent,(7,),'initial coordinates')
        self.latent=nn.Parameter(latent.detach().clone())


class _CenteredHead(nn.Module):
    """Keep original modules/hooks once; insert only declared fixed logit centering."""
    def __init__(self,predictor,features,target):
        super().__init__()
        self.predictor=predictor
        center=self.pre_sigmoid(features).detach().clone()
        self.register_buffer('center',center)
        self.register_buffer('target',target.detach().reshape_as(center).clone())

    def pre_sigmoid(self,x):
        # Avoid registering sliced Sequential copies or replacing weight_norm.
        for i in range(len(self.predictor)-1):
            x=self.predictor[i](x)
        return x

    def forward(self,x):
        logits=self.target+(self.pre_sigmoid(x)-self.center)
        return self.predictor[-1](logits)


class NeuralMaterial(nn.Module):
    arm='N1'
    def __init__(self,latent,seed):
        super().__init__()
        cpu_float(latent,(7,),'initial coordinates')
        require(type(seed) is int and seed in SEEDS,'Only approved material initialization seeds permitted')
        require(torch.get_default_dtype()==torch.float32 and str(torch.get_default_device())=='cpu',
                'Original CPU float32 defaults required before component construction')
        definitions,self.source_identity=source_definitions()
        self.seed=seed
        self._predict=definitions['predict_anisotropic_components']
        self.register_buffer('anchor',torch.zeros((1,3),dtype=torch.float32,device='cpu'))
        self.register_buffer('initial_latent',latent.detach().clone())
        e=torch.stack((torch.cos(2*latent[2]),torch.sin(2*latent[2])))
        q=(1+RHO*e)/2
        rotation_logits=torch.log(q/(1-q))  # Declared float32 calibration, not a new decoder.
        targets=(latent[:1],latent[1:2],latent[6:7],rotation_logits,latent[3:6])
        # CPU-only local RNG; do not seed CUDA or mutate any scientific module.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(10000+seed)
            self.feats_network=definitions['MaterialFeatsNetwork']()
            heads=[definitions['make_predictor'](259,d) for _,d in HEADS]
            f=self.feats_network(self.anchor)
            features=torch.cat((f,self.anchor),-1)
            for (name,_),head,target in zip(HEADS,heads,targets):
                setattr(self,name,_CenteredHead(head,features,target))
        require(sum(p.numel() for p in self.parameters())==PARAMETER_COUNT,'Original A architecture/parameter count changed')
        require(len(list(self.named_parameters()))==84,'Unexpected duplicate/missing parameter registration')

    def forward(self,positions):
        cpu_float(positions,(len(positions),3),'query positions')
        require(len(positions)>0,'Nonempty query positions required')
        for name,p in self.named_parameters():
            cpu_float(p,tuple(p.shape),'parameter '+name)
        # BOTH feature input and each raw-point skip receive the constant anchor.
        mx,my,metal,kd,encoding,raw=self._predict(self,self.anchor)
        status=encoding_status(encoding)
        require(bool(status['valid'].all()),'Invalid zero/near-zero shader orientation encoding')
        unit=encoding/status['norm']
        theta=.5*torch.atan2(unit[:,1:2],unit[:,:1])  # Diagnostics only; never recoded for shading.
        p=torch.cat((mx,my,theta,kd,metal),-1)
        radius=torch.norm(raw*2-1,dim=-1,keepdim=True)
        return Material(p,encoding,raw,radius).validate().expand(len(positions))


def make_material(arm,latent,seed):
    require(arm in ('D0','N0','N1'),'Unknown representation')
    if arm=='D0':return DirectMaterial(latent)
    if arm=='N0':return ConstantMaterial(latent)
    return NeuralMaterial(latent,seed)


def identity(model):
    return dict(protocol=PROTOCOL,arm=model.arm,
        configuration=dict(anchor=[0.,0.,0.] if model.arm=='N1' else None,rho=RHO if model.arm=='N1' else None,
            head_order=[name for name,_ in HEADS] if model.arm=='N1' else None,
            weight_seed=10000+model.seed if model.arm=='N1' else None,
            calibration='target+(original_pre_sigmoid-fixed_initial_pre_sigmoid)' if model.arm=='N1' else None,dtype='float32',device='cpu'),
        source=getattr(model,'source_identity',None),
        parameters={name:dict(**record(p),requires_grad=p.requires_grad) for name,p in model.named_parameters()},
        buffers={name:record(b) for name,b in model.named_buffers()},
        parameter_count=sum(p.numel() for p in model.parameters()))


def gradient_summary(model):
    """Detached observations, never repair gradients or weaken a whole-head gate."""
    groups={}
    for name,p in model.named_parameters():
        group=name.split('.')[0];out=groups.setdefault(group,dict(parameters=0,missing=0,finite=True,nonzero=0,squared_norm=0.))
        out['parameters']+=p.numel()
        if p.grad is None:
            out['missing']+=p.numel();continue
        g=p.grad.detach();out['finite'] &= bool(torch.isfinite(g).all())
        out['nonzero']+=int(torch.count_nonzero(g))
        out['squared_norm']+=float(torch.sum(g.double()**2))
    return groups
