"""E4 AS1: anchored asinh roughness logits and SO(2) orientation transport.

Generic tensor algebra/components only. Never imports or constructs real N1.
"""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from infrastructure.signatures import require

PROTOCOL='anisotropic-anchored-chart-as1-v1'


def compressed_probability(delta,target):
    return torch.sigmoid(target+torch.asinh(delta))


def orientation_constants(target):
    q=torch.sigmoid(target);c=2*q-1;r2=(c*c).sum(-1,keepdim=True)
    require(bool((r2>=1e-12).all()),'Invalid original orientation anchor; no fallback')
    e=F.normalize(c,dim=-1,eps=1e-6)
    j=torch.cat((-c[:,1:2]*q[:,:1]*(1-q[:,:1]),c[:,:1]*q[:,1:2]*(1-q[:,1:2])),dim=-1)/r2
    return e,j


def transport(delta,target):
    e,j=orientation_constants(target);angle=(delta*j).sum(-1,keepdim=True)
    c,s=torch.cos(2*angle),torch.sin(2*angle)
    encoding=torch.cat((c*e[:,:1]-s*e[:,1:2],s*e[:,:1]+c*e[:,1:2]),dim=-1)
    return encoding,angle


def tensor_q(widths,encoding):
    """Actual Q=R diag(mx^2,my^2) R^T, not a GGX parameter substitution."""
    e=F.normalize(encoding,dim=-1,eps=1e-6);a,b=widths[:,:1]**2,widths[:,1:2]**2
    mean=(a+b)/2;difference=(a-b)/2;x,y=e[:,:1],e[:,1:2]
    return torch.stack((torch.cat((mean+difference*x,difference*y),-1),torch.cat((difference*y,mean-difference*x),-1)),-2)


class AsinhHead(nn.Module):
    def __init__(self,original,keep,name):
        super().__init__();self.predictor=original.predictor
        self.register_buffer('center',original.center);self.register_buffer('target',original.target)
        self.keep=keep;self.label=name
    def pre_sigmoid(self,x):
        for i in range(len(self.predictor)-1):x=self.predictor[i](x)
        return x
    def forward(self,x):
        delta=self.pre_sigmoid(x)-self.center;compressed=torch.asinh(delta);logits=self.target+compressed
        value=self.predictor[-1](logits)
        for name,t in (('delta',delta),('compressed',compressed),('logits',logits),('sigmoid',value)):self.keep(self.label+'.'+name,t)
        return value


class ChartMaterial(nn.Module):
    arm='N1'
    def __init__(self,base,material_type):
        super().__init__();self._active=False;self.nodes={};self.material_type=material_type
        self.feats_network=base.feats_network
        self.mx_predictor=AsinhHead(base.mx_predictor,self.keep,'mx_predictor')
        self.my_predictor=AsinhHead(base.my_predictor,self.keep,'my_predictor')
        self.metallic_predictor=base.metallic_predictor;self.rotation_predictor=base.rotation_predictor;self.kd_predictor=base.kd_predictor
        self.register_buffer('anchor',base.anchor);self.register_buffer('initial_latent',base.initial_latent)
        self.seed=base.seed;self.source_identity=base.source_identity
        require([(n,id(p)) for n,p in self.named_parameters()]==[(n,id(p)) for n,p in base.named_parameters()],
            'Variant changed trainable inventory/identity')
        require({n:tuple(v.shape) for n,v in self.named_buffers()}=={n:tuple(v.shape) for n,v in base.named_buffers()},'Variant changed buffer inventory')
    def keep(self,name,value):
        if self._active:
            require(value.requires_grad and name not in self.nodes,'Invalid/repeated actual chart node: '+name)
            value.retain_grad();self.nodes[name]=value
    @contextmanager
    def observe(self):
        require(not self._active and not self.nodes,'Nested observation')
        handles=[];self._active=True
        try:
            for name in ('kd_predictor','metallic_predictor'):
                last=getattr(self,name).predictor[-1]
                handles.append(last.register_forward_pre_hook(lambda m,args,n=name:self.keep(n+'.logits',args[0])))
                handles.append(last.register_forward_hook(lambda m,args,out,n=name:self.keep(n+'.sigmoid',out)))
            yield
        finally:
            for h in handles:h.remove()
            self.nodes.clear();self._active=False
    def shader_view(self,shader):
        owner=self
        class View:
            def shade(self,*args,**kwargs):
                for name,t in zip(('mx','my','albedo','metallic','encoding'),args[3:8]):owner.keep('shader.'+name,t)
                return shader.shade(*args,**kwargs)
        return View()
    def forward(self,positions):
        require(positions.ndim==2 and positions.shape[1]==3 and len(positions)>0 and positions.device.type=='cpu' and
            positions.dtype==torch.float32 and bool(torch.isfinite(positions).all()),'Explicit finite CPU float32 positions required')
        feats=self.feats_network(self.anchor);self.keep('shared_features',feats)
        mxq=self.mx_predictor(torch.cat((feats,self.anchor),-1));mx=.04+.96*mxq
        myq=self.my_predictor(torch.cat((feats,self.anchor),-1));my=.04+.96*myq
        metal=self.metallic_predictor(torch.cat((feats,self.anchor),-1));kd=self.kd_predictor(torch.cat((feats,self.anchor),-1))
        delta=self.rotation_predictor.pre_sigmoid(torch.cat((feats,self.anchor),-1))-self.rotation_predictor.center
        encoding,angle=transport(delta,self.rotation_predictor.target)
        self.keep('rotation_predictor.delta',delta);self.keep('orientation.angle_delta',angle);self.keep('orientation.encoding',encoding)
        unit=encoding/torch.norm(encoding,dim=-1,keepdim=True);theta=.5*torch.atan2(unit[:,1:2],unit[:,:1])
        physical=torch.cat((mx,my,theta,kd,metal),-1)
        require(bool(torch.isfinite(physical).all()),'Nonfinite chart output')
        return self.material_type(physical,encoding).validate().expand(len(positions))
