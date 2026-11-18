"""Explicit corrected-core extension used only by Donut D8-C and D8-T."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.signatures import require
from infrastructure.surface_q_regularization import transported_q_loss
from scientific_variants import donut_tensor_d8 as tensors
from scientific_variants.material_a_v1 import finite, orientation, unit

ARMS=('D8-C','D8-T')
VARIANTS=dict(zip(ARMS,('donut_tensor_chart_v1','donut_tensor_ambient_v1')))


def legacy_radiance(fd,diffuse_light,specular_light,fresnel,reflected,normal,cosine,ndf):
    ci=(reflected*normal[:,None]).sum(-1,keepdim=True).clamp(min=0)
    valid=ci>0
    require(bool((cosine>0).all()),'Nonpositive sampled half-vector cosine')
    geometric=ci/cosine*valid
    weight=fresnel*ci/cosine*valid
    diffuse=(diffuse_light*fd[:,None]).mean(1)
    specular=(specular_light*weight).sum(1)/specular_light.shape[1]
    count=valid.sum(1).clamp(min=1)
    proxy=weight.sum(1)/count
    spec_prior=(torch.softmax(ndf.detach(),dim=1)*fd[:,None]*valid).sum(1)/(count*3)
    return diffuse,specular,proxy,spec_prior,(geometric*specular_light).mean(1)


def chart_width_bias():
    target=(tensors.INITIAL_WIDTH-.04)/.96
    return math.log(target/(1-target))


def scalar_perturbation(shader,points,normals,kd,metallic):
    n=F.normalize(normals,dim=-1,eps=1e-6)
    x=shader.get_orthogonal_directions(n)
    y=torch.cross(n,x,dim=-1)
    ang=torch.rand((len(points),1),device=points.device)*math.pi*2
    require(shader.cfg['change_type']=='gaussian','D8 requires the legacy Gaussian perturbation')
    distance=torch.normal(mean=0.,std=shader.cfg['change_eps'],size=(len(points),1),device=points.device)
    changed=points+(torch.cos(ang)*x+torch.sin(ang)*y)*distance
    kd1,m1=shader.scalar_material(changed)
    return (abs(kd-kd1)+abs(metallic-m1)).mean(1)*shader.cfg['reg_lambda1']


def tensor_shader(core, to_srgb):
    class TensorShader(core):
        def select_tensor_representation(self,arm):
            require(arm in ARMS and not hasattr(self,'tensor_arm'),'Select one tensor representation')
            self.tensor_arm=arm
            self.tensor_variant=VARIANTS[arm]
            self.initial_width=tensors.INITIAL_WIDTH
            self.register_buffer('_donut_tensor_signature',
                                 torch.tensor(list(self.tensor_variant.encode('ascii')),dtype=torch.uint8))
            with torch.no_grad():
                for head in (self.mx_predictor,self.my_predictor,self.rotation_predictor):
                    if arm=='D8-T':
                        final=head[6]
                        if final.out_features!=2:
                            # Match hidden topology. The paired construction uses an RNG fork.
                            head[6]=nn.utils.weight_norm(nn.Linear(final.in_features,2))
                        head[7]=nn.Identity()
                    head[6].weight_g.zero_()
                    head[6].bias.zero_()
                if arm=='D8-C':
                    for head in (self.mx_predictor,self.my_predictor):
                        head[6].bias.fill_(chart_width_bias())
                    # Valid axial circle vector (+1,0) up to normalization.
                    self.rotation_predictor[6].bias[0]=math.log(3)

        def load_state_dict(self,state_dict,strict=True):
            tag=state_dict.get('_donut_tensor_signature')
            require(tag is not None and torch.equal(tag.cpu(),self._donut_tensor_signature.cpu()),
                    'Foreign D8 representation checkpoint')
            return super().load_state_dict(state_dict,strict=strict)

        def forward(self,points,outgoing,normals,human_poses,step,is_train,*,tape_seed=None,export=False):
            require(type(tape_seed) is int and human_poses is None,'D8 requires an explicit world tape and no human light')
            q,frame,kd,metallic=self.predict_tensor(points,normals)
            return self.shade_tensor(points,normals,outgoing,q,frame,kd,metallic,tape_seed,is_train,export=export)

        def features(self,points):
            return torch.cat([self.feats_network(points),points],dim=-1)

        def scalar_material(self,points):
            x=self.features(points)
            return self.kd_predictor(x),self.metallic_predictor(x)

        def predict_tensor(self,points,normals):
            unit(normals,'tensor normals')
            frame=tensors.frame(normals)
            if self.tensor_arm=='D8-C':
                mx,my,metallic,kd,rotation,_=self.predict_anisotropic_components(points)
                a,b=self.compute_tangent_bitangent_flat(normals,None)
                c,s=orientation(rotation)
                q=tensors.chart_q(mx,my,c*a+s*b,frame)
            else:
                f=self.features(points)
                logits=torch.cat([self.mx_predictor(f),self.my_predictor(f),self.rotation_predictor(f)],-1)
                finite(logits,'ambient tensor logits')
                q=tensors.ambient_q(logits,frame,initial_width=self.initial_width)
                kd,metallic=self.kd_predictor(f),self.metallic_predictor(f)
            return q,frame,kd,metallic

        def regularize_tensor(self,points,normals,o):
            require(self.cfg['reg_change'] and self.cfg['reg_energy_loss'] and self.cfg['reg_spec_loss'],
                    'D8 requires all retained legacy regularizers')
            scalar=scalar_perturbation(self,points,normals,o['kd'],o['metallic'])
            energy=F.relu(o['f_d']+o['f_s_sum']-1).sum(-1).mean()*self.cfg['reg_energy_loss_lambda']
            spec=o['L_spec'].sum(-1).mean()*self.cfg['reg_spec_loss_lambda']
            return scalar,energy,spec

        def shade_tensor(self,points,normals,outgoing,q,frame,kd,metallic,seed,is_train,*,export=False):
            unit(outgoing,'tensor views')
            require(bool(((outgoing*normals).sum(-1)>0).all()),'Upper-hemisphere outgoing views required')
            for value in (kd,metallic):
                require(bool((torch.isfinite(value)&(value>=0)&(value<=1)).all()),'Invalid scalar material')
            shape=(len(points),self.cfg['specular_sample_num'])
            # Preserve the core default RNG position before diffuse and legacy perturbation.
            torch.rand(shape,device=points.device)
            torch.rand(shape,device=points.device)
            tape=torch.Generator(device=points.device).manual_seed(seed)
            gaussian=torch.randn((*shape,3),generator=tape,dtype=points.dtype,device=points.device)
            _,wi,ch,ndf,_=tensors.sample(q,frame,normals,outgoing,gaussian)
            diffuse=self.sample_diffuse_directions(normals,is_train)
            dirs=torch.cat((diffuse,wi),dim=1)
            positions=points[:,None].repeat(1,dirs.shape[1],1)
            lights,_,_,_,_=self.get_lights(positions,dirs,None)
            finite(lights,'tensor incident light')
            require(bool((lights>=0).all()),'Negative incident radiance')
            dn=diffuse.shape[1]
            f0=.04*(1-metallic)+metallic*kd
            fresnel,_,_=self.fresnel_schlick_directions(f0[:,None],outgoing[:,None],wi)
            fd=self.diffuse_term(kd,metallic,is_seperate=False)
            d,s,proxy,prior,weighted=legacy_radiance(fd,lights[:,:dn],lights[:,dn:],fresnel,wi,normals,ch,ndf)
            o=dict(kd=kd,metallic=metallic,linear_rgb=d+s,linear_diffuse=d,linear_specular=s,
                   f_d=fd,f_s_sum=proxy,L_spec=prior,
                   diffuse_light=to_srgb(lights[:,:dn].mean(1)).clamp(0,1),
                   specular_light=to_srgb(weighted).clamp(0,1),Q_local=q)
            if export:
                require(not torch.is_grad_enabled(),'Axes are only exported outside training AD')
                mx,my,t,b=tensors.export_axes(q,frame)
                a0,a1=self.compute_tangent_bitangent_flat(normals,None)
                x,y=(t*a0).sum(-1,keepdim=True),(t*a1).sum(-1,keepdim=True)
                rotation=F.normalize(torch.cat((x.square()-y.square(),2*x*y),-1),dim=-1)
                o.update(mx=mx,my=my,tangents=(t+1)/2,bitangents=(b+1)/2,normals=(normals+1)/2,
                         rotation=rotation,rotation_raw=(rotation+1)/2,Q=tensors.world_tensor(q,frame))
            for key,value in o.items():
                finite(value,'tensor shader '+key)
            return o

    return TensorShader


def tensor_surface_term(shader,batch,policy,torch):
    """The existing D6 distance and transport loss, applied directly to Q."""
    n0,n1=(batch['normals'+suffix] for suffix in ('0','1'))
    q=[]
    for suffix,n in (('0',n0),('1',n1)):
        local,frame,_,_=shader.predict_tensor(batch['points'+suffix],n)
        q.append(tensors.world_tensor(local,frame).to(n.dtype))
    _,delta=transported_q_loss(q[0],q[1],n0,n1,torch.ones_like(batch['distance']),torch)
    return policy['coefficient']*(delta*(policy['reference_distance']/batch['distance']).square()).mean()


def paired_initial_gate(first,second,p,n,wo,policy):
    """Paired real-input gate, with no gradients or optimizer mutation."""
    require(not torch.is_grad_enabled() and len(p)==policy['native_gate_points'],
            'D8 initial gate is a fixed evaluation')
    shared=('feats_network','kd_predictor','metallic_predictor','outer_light','inner_light')
    def shared_state(model):
        return {name:value for name,value in model.state_dict().items() if name.split('.')[0] in shared}
    a,b=shared_state(first),shared_state(second)
    require(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'D8 shared initialization differs')
    capture={}
    max_initial_q=0.
    seed=policy['initial_gate_seed']
    for model in (first,second):
        model.eval()
        q,u,kd,m=model.predict_tensor(p,n)
        target=torch.eye(2,device=q.device,dtype=q.dtype)*policy['initial_width']**2
        max_initial_q=max(max_initial_q,float((q-target).abs().max()))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result=model.shade_tensor(p,n,wo,q,u,kd,m,seed,False)
        prefix=model.tensor_arm+'_'
        for key in ('linear_rgb','linear_diffuse','linear_specular','kd','metallic'):
            capture[prefix+key]=result[key].cpu().numpy()
        capture[prefix+'world_Q']=tensors.world_tensor(q,u).cpu().numpy()
    def delta(key):
        return float(abs(capture['D8-C_'+key]-capture['D8-T_'+key]).max())
    q_delta=delta('world_Q')
    rgb_delta=max(delta(k) for k in ('linear_rgb','linear_diffuse','linear_specular'))
    scalar_equal=delta('kd')==delta('metallic')==0
    passed=(max_initial_q<=policy['initial_Q_atol'] and q_delta<=policy['initial_Q_atol']
            and rgb_delta<=policy['initial_RGB_atol'] and scalar_equal)
    return dict(passed=passed,shared_state_equal=True,scalar_predictions_equal=scalar_equal,
                world_Q_max_abs=q_delta,local_isotropic_Q_max_abs=max_initial_q,
                RGB_component_max_abs=rgb_delta),capture
