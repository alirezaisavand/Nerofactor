"""Native, versioned A derivatives. Never imported into original scientific snapshots.

Core retains Kurt a=0/ks=1 and metallic allocation; it is NOT energy conserving.
The separate energy-budget variant uses d*Lambert+(1-d)*Kurt, a reciprocal
convex component mixture with an ideal directional-integral energy bound.
All directions point away from the surface. Invalid encodings fail before Adam.
"""
from contextlib import contextmanager
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from infrastructure.signatures import require
from infrastructure.material_variants import CORE, ESTIMATOR, ORIGINAL, ENERGY, configuration, identity

ORIENTATION_MIN=1e-6  # Same invalid-domain boundary as accepted A2/atan2.


def uniform32(values):
    """E8R endpoint rule for explicit float64 tapes; never clip invalid uniforms."""
    a=np.asarray(values)
    require(a.dtype==np.float64 and np.isfinite(a).all() and (a>=0).all() and (a<1).all(),'Expected valid float64 uniform tape')
    b=a.astype(np.float32);b[b==np.float32(1)]=np.nextafter(np.float32(1),np.float32(0))
    return b


def finite(t,name):require(bool(torch.isfinite(t).all()),'Nonfinite '+name)

def unit(t,name):
    finite(t,name);require(t.ndim==2 and t.shape[-1]==3 and bool((torch.abs(torch.linalg.norm(t,dim=-1)-1)<=1e-5).all()),'Unit '+name+' required; no normal repair')


def orientation(r):
    finite(r,'orientation');require(r.ndim==2 and r.shape[-1]==2,'Double-angle encoding shape required')
    norm=torch.linalg.norm(r,dim=-1,keepdim=True)
    finite(norm,'orientation norm')
    require(bool((norm>=ORIENTATION_MIN).all()),'Undefined zero/near-zero orientation: terminate pending update before optimizer.step; no fallback')
    v=r/norm;angle=.5*torch.atan2(v[:,1:2],v[:,0:1])
    return torch.cos(angle),torch.sin(angle)


def state_gate(module,state,prefix=''):
    """Reject missing/foreign behavior identity BEFORE loading any weights."""
    key=prefix+'_a_variant_signature'
    require(key in state and isinstance(state[key],torch.Tensor) and state[key].dtype==torch.uint8 and torch.equal(state[key].cpu(),module._a_variant_signature.cpu()),
            'Checkpoint variant/configuration signature mismatch; original-A checkpoints are not corrected training states')
    selected={n[len(prefix):]:v for n,v in state.items() if n.startswith(prefix)};expected=module.state_dict()
    require(selected.keys()==expected.keys(),'Strict scientific state key inventory mismatch')
    for name,value in expected.items():
        candidate=selected[name]
        require(isinstance(candidate,torch.Tensor) and candidate.shape==value.shape and candidate.dtype==value.dtype,
                'Scientific state shape/dtype mismatch: '+name)
        finite(candidate,'checkpoint tensor: '+name)


def native_classes(base_shader,base_renderer,variant,to_srgb):
    require(variant in (CORE,ESTIMATOR,ORIGINAL,ENERGY),'Explicit native material identity required')
    class CorrectedShader(base_shader):
        material_variant=variant
        def __init__(self,cfg,ray_trace_fun):
            require(variant!=ORIGINAL or cfg.get('illumination_mode')=='known_env_v1',
                    'Standard original A must use its untouched class; this adapter only supplies the explicit known-light control')
            configuration(dict(network='material',reg_mat=cfg.get('regularization_version')!='disabled_v1',shader_cfg=cfg),variant)
            super().__init__(cfg,ray_trace_fun)
            implementation=identity(variant,dict(network='material',reg_mat=cfg.get('regularization_version')!='disabled_v1',shader_cfg=cfg))
            signature=hashlib.sha256(json.dumps(dict(implementation=implementation,shader=self.cfg),sort_keys=True,allow_nan=False).encode()).digest()
            self.register_buffer('_a_variant_signature',torch.tensor(list(signature),dtype=torch.uint8))
            self._environment=None
            def guard(name):
                def checked(gradient):
                    finite(gradient,'parameter gradient: '+name)
                    return gradient  # Observe/reject only; never modify a finite derivative.
                return checked
            self._finite_gradient_handles=[p.register_hook(guard(name)) for name,p in self.named_parameters() if p.requires_grad]

        def load_state_dict(self,state_dict,strict=True):
            require(strict is True,'Non-strict corrected weight loading forbidden');state_gate(self,state_dict)
            return super().load_state_dict(state_dict,strict=True)

        def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
            # Parent containers otherwise bypass a child's load_state_dict override.
            state_gate(self,state_dict,prefix)
            return super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

        @contextmanager
        def environment(self,callback,identity):
            """Explicit relighting of outer rays; native visibility/inner light stay active."""
            require(callable(callback) and isinstance(identity,str) and bool(identity) and self._environment is None,'Explicit non-nested environment required')
            self._environment=(callback,identity)
            try:yield
            finally:self._environment=None

        def predict_outer_lights(self,points,directions):
            if self._environment is None and self.cfg.get('illumination_mode','learned_native')=='known_env_v1':
                e=self.cfg['known_environment'];value=torch.tensor(e['ambient'],device=directions.device,dtype=directions.dtype).expand_as(directions)
                for lobe in e['lobes']:
                    axis=torch.tensor(lobe['axis'],device=directions.device,dtype=directions.dtype)
                    value=value+torch.tensor(lobe['RGB'],device=directions.device,dtype=directions.dtype)*(directions@axis).clamp(min=0)[:,None]**lobe['power']
                return value
            if self._environment is None:return super().predict_outer_lights(points,directions)
            value=self._environment[0](points,directions)
            require(isinstance(value,torch.Tensor) and value.shape==directions.shape and value.device==directions.device and value.dtype==directions.dtype,'Environment must return matching linear RGB')
            finite(value,'environment');require(bool((value>=0).all()),'Negative incident radiance')
            return value

        def get_inner_lights(self,points,directions,normals):
            if self.cfg.get('indirect_mode','native_learned')=='zero_bounce_v1':return torch.zeros_like(points)
            return super().get_inner_lights(points,directions,normals)

        def predict_anisotropic_components(self,pts):
            values=super().predict_anisotropic_components(pts)
            if variant in (CORE,ENERGY):
                try:orientation(2*values[-1]-1)  # Check raw vector BEFORE original F.normalize epsilon hides its norm.
                except ValueError as exc:
                    batch=hashlib.sha256(pts.detach().cpu().numpy().tobytes()+values[-1].detach().cpu().numpy().tobytes()).hexdigest()
                    raise ValueError(str(exc)+'; point/raw-orientation batch SHA256='+batch) from exc
            return values

        def compute_tangent_bitangent_flat(self,normals,sources):
            if variant==ESTIMATOR:return super().compute_tangent_bitangent_flat(normals,sources)
            unit(normals,'normals')
            ref=torch.tensor([1.,0.,0.],dtype=normals.dtype,device=normals.device).expand_as(normals) if sources is None else sources
            finite(ref,'reference tangent')
            t=torch.cross(normals,ref,dim=-1);small=torch.linalg.norm(t,dim=-1)<1e-6
            if sources is None and bool(small.any()):
                ref_y=torch.tensor([0.,1.,0.],dtype=normals.dtype,device=normals.device).expand_as(normals)
                t=torch.where(small[:,None],torch.cross(normals,ref_y,dim=-1),t)
            require(bool((torch.linalg.norm(t,dim=-1)>=1e-6).all()),'Undefined parallel/near-parallel tangent reference')
            t=F.normalize(t,dim=-1);b=F.normalize(torch.cross(normals,t,dim=-1),dim=-1)
            return t,b  # Original X/Y seam remains; no global continuity/normal flip claim.

        def sample_diffuse_directions(self,normals,is_train):
            if variant in (CORE,ENERGY):unit(normals,'normals')
            x=self.get_orthogonal_directions(normals);y=torch.cross(normals,x,dim=-1)
            az=self.diffuse_direction_samples[:,0:1].to(normals.device).unsqueeze(0)*2*np.pi
            if is_train and self.cfg['random_azimuth']:az=(az+torch.rand(normals.shape[0],1,1,device=normals.device)*2*np.pi)%(2*np.pi)
            count=self.cfg['diffuse_sample_num'];z=((torch.arange(count,device=normals.device,dtype=normals.dtype)+.5)/count).reshape(1,-1,1)
            return torch.sqrt(1-z)*torch.cos(az)*x[:,None]+torch.sqrt(1-z)*torch.sin(az)*y[:,None]+torch.sqrt(z)*normals[:,None]

        def sample_aniso_ggx_directions(self,rotation,pts,m_x,m_y,wo,M,normals,sources,device=None,eps=1e-6):
            if variant==ESTIMATOR:return super().sample_aniso_ggx_directions(rotation,pts,m_x,m_y,wo,M,normals,sources,device,eps)
            unit(wo,'views');unit(normals,'normals');require(bool(((wo*normals).sum(-1)>0).all()),'Upper-hemisphere outgoing views required')
            for m in (m_x,m_y):finite(m,'Beckmann widths');require(bool(((m>=.04)&(m<=1)).all()),'Beckmann width outside [.04,1]')
            x,y=self.compute_tangent_bitangent_flat(normals,sources);c,s=orientation(rotation)
            if sources is None:x,y=self.rotate_tangent_bitangent(x,y,c,s)
            # Same two random requests, draw order and elliptical Gaussian law.
            # Only exact zero maps to smallest NORMAL float: no 1e-6 ring atom.
            u=torch.rand((wo.shape[0],M),device=wo.device);v=torch.rand((wo.shape[0],M),device=wo.device)
            finite(u,'uniform');finite(v,'uniform');require(bool(((u>=0)&(u<1)&(v>=0)&(v<1)).all()),'Uniform outside [0,1)')
            u=torch.where(u==0,torch.full_like(u,torch.finfo(u.dtype).tiny),u)
            radius=torch.sqrt(-torch.log(u));phi=2*np.pi*v
            sx=m_x*radius*torch.cos(phi);sy=m_y*radius*torch.sin(phi)
            h=F.normalize(sx[...,None]*x[:,None]+sy[...,None]*y[:,None]+normals[:,None],dim=-1)
            ch=(h*normals[:,None]).sum(-1,keepdim=True);dot=(wo[:,None]*h).sum(-1,keepdim=True)
            wi=2*dot*h-wo[:,None]  # Signed reflection: invisible facets are below horizon, carry zero weight.
            theta=torch.atan(torch.sqrt(sx*sx+sy*sy));az=torch.atan2(sy,sx)
            pdf=self.compute_pdf_aniso_ggx(m_x,m_y,wo,h,theta,az)
            finite(h,'half-vectors');finite(wi,'reflections');finite(pdf,'PDF')
            return h,wi,ch,pdf,x,y,normals,theta,az

        def compute_Dh(self,m_x,m_y,theta_h,phi_h):
            if variant==ESTIMATOR:return super().compute_Dh(m_x,m_y,theta_h,phi_h)
            ch=torch.cos(theta_h);safe=torch.where(ch>0,ch,torch.ones_like(ch))
            exponent=-(torch.tan(theta_h)**2)*((torch.cos(phi_h)/m_x)**2+(torch.sin(phi_h)/m_y)**2)
            logd=exponent-torch.log(np.pi*m_x*m_y)-4*torch.log(safe)
            return torch.where(ch>0,torch.exp(logd),torch.zeros_like(logd)).unsqueeze(-1)

        def compute_pdf_aniso_ggx(self,m_x,m_y,wo,h,theta_h,phi_h,eps=1e-6):
            if variant==ESTIMATOR:return super().compute_pdf_aniso_ggx(m_x,m_y,wo,h,theta_h,phi_h,eps)
            ch=torch.cos(theta_h).unsqueeze(-1);dot=(wo[:,None]*h).sum(-1,keepdim=True).abs()
            # p(h)=D(h)*ch, reflection Jacobian=4*abs(wo.h). No epsilon floor/tail lift.
            valid=(ch>0)&(dot>0);den=torch.where(valid,4*dot,torch.ones_like(dot))
            return torch.where(valid,self.compute_Dh(m_x,m_y,theta_h,phi_h)*ch/den,torch.zeros_like(dot))

        def compute_radiance(self,f_d,diffuse_lights,specular_lights,Fresnel,diffuse_directions,wi,n,cos_theta_h,wo,pdf,theta_h,phi_h,mx,my,eps=1e-6):
            if variant in (CORE,ENERGY):
                for value in (diffuse_lights,specular_lights):finite(value,'incident light');require(bool((value>=0).all()),'Negative incident radiance')
            M=specular_lights.shape[1];ci=(wi*n[:,None]).sum(-1,keepdim=True).clamp(min=0);mask=ci>0
            denominator=cos_theta_h.clamp(min=eps) if variant==ESTIMATOR else cos_theta_h
            require(bool((denominator>0).all()),'Invalid half-vector cosine')
            weight=Fresnel*ci/denominator*mask
            budget=1-f_d[:,None] if variant==ENERGY else 1
            weight=weight*budget
            diffuse=(diffuse_lights*f_d[:,None]).mean(1);specular=(specular_lights*weight).sum(1)/M
            unconditional=weight.sum(1)/M;weighted=ci/denominator*mask*specular_lights*budget
            if self.cfg['regularization_version']=='legacy_v1':
                count=mask.sum(1).clamp(min=1)
                proxy=weight.sum(1)/count
                L_spec=(self.compute_spec_loss(1,mx,my,theta_h,phi_h)*f_d[:,None]*mask).sum(1)/(count*3)
            else:
                proxy=unconditional
                # New, explicitly named diffuse-allocation prior; NOT historical L_spec.
                # Detached specular proxy: only d=kd*(1-metallic) gets a direct derivative.
                L_spec=f_d*unconditional.detach()/3
            for value in (diffuse,specular,proxy,L_spec):finite(value,'radiance/regularizer')
            return diffuse+specular,diffuse,specular,proxy,L_spec,weighted

        def shade_anisotropic_mixed(self,pts,normals,view_dirs,mx,my,metallic,kd,rotation,rotation_raw,human_poses,is_train):
            for value in (metallic,kd):finite(value,'material');require(bool(((value>=0)&(value<=1)).all()),'Material coefficient outside [0,1]')
            hs,wis,ch,pdf,t,b,n,theta,phi=self.sample_aniso_ggx_directions(rotation,pts,mx,my,view_dirs,self.cfg['specular_sample_num'],normals,None,pts.device)
            diffuse=self.sample_diffuse_directions(normals,is_train);dn=diffuse.shape[1];directions=torch.cat([diffuse,wis],1)
            samples=directions.shape[1];positions=pts[:,None].repeat(1,samples,1)
            poses=human_poses[:,None].repeat(1,samples,1,1) if human_poses is not None else None
            lights,hl,_,_,_=self.get_lights(positions,directions,poses)  # Native BVH + learned direct/indirect/human lighting.
            f0=.04*(1-metallic)+metallic*kd
            fresnel,_,_=self.fresnel_schlick_directions(f0[:,None],view_dirs[:,None],wis)
            fd=self.diffuse_term(kd,metallic,is_seperate=False)
            linear,dc,sc,proxy,spec_prior,weighted=self.compute_radiance(fd,lights[:,:dn],lights[:,dn:],fresnel,diffuse,wis,n,ch,view_dirs,pdf,theta,phi,mx,my)
            return to_srgb(linear),dict(tangents=(t+1)/2,bitangents=(b+1)/2,normals=(n+1)/2,human_lights=hl.reshape(-1,3),
                kd=kd,metallic=metallic,mx=mx,my=my,diffuse_color=to_srgb(dc),specular_color=to_srgb(sc),
                diffuse_light=to_srgb(lights[:,:dn].mean(1)).clamp(0,1),specular_light=to_srgb(weighted.mean(1)).clamp(0,1),
                f_d=fd,f_s_sum=proxy,L_spec=spec_prior,rotation=rotation,rotation_raw=rotation_raw,
                linear_rgb=linear,linear_diffuse=dc,linear_specular=sc)

        def anisotropic_regularization(self,pts,normals,t,b,mx,my,metallic,kd,f_d,f_s_sum,L_spec,rotation,rotation_raw,step):
            mode=self.cfg['regularization_version']
            if mode=='legacy_v1':return super().anisotropic_regularization(pts,normals,t,b,mx,my,metallic,kd,f_d,f_s_sum,L_spec,rotation,rotation_raw,step)
            reg=f_d.sum(-1)*0
            if mode=='disabled_v1':return reg
            if self.cfg['reg_change']:
                # Original spatial prior, draw order and unusual orientation schedule retained verbatim algebraically.
                normals=F.normalize(normals,dim=-1,eps=1e-6);x=self.get_orthogonal_directions(normals);y=torch.cross(normals,x,dim=-1)
                ang=torch.rand(pts.shape[0],1,device=pts.device)*2*np.pi
                if self.cfg['change_type']=='constant':distance=self.cfg['change_eps']
                elif self.cfg['change_type']=='gaussian':distance=torch.normal(0.,self.cfg['change_eps'],size=(pts.shape[0],1),device=pts.device)
                else:raise ValueError('Unknown spatial change_type')
                mx1,my1,m1,kd1,r1,_=self.predict_anisotropic_components(pts+(torch.cos(ang)*x+torch.sin(ang)*y)*distance)
                decay=(1e-3-1e-4)*(.001*(np.cos((step/100000.)*np.pi)+1)/2)+1e-4
                reg=reg+(torch.abs(kd-kd1)+torch.abs(mx-mx1)+torch.abs(my-my1)+torch.abs(metallic-m1)+((rotation-r1)**2).sum(-1,keepdim=True)*decay).mean(-1)*self.cfg['reg_lambda1']
            # Independent gates: no hidden reg_change dependency. Caller renderer.reg_mat still gates all material priors.
            if self.cfg['reg_energy_loss']:reg=reg+F.relu(f_d+f_s_sum-1).sum(-1)*self.cfg['reg_energy_loss_lambda']
            if self.cfg['reg_spec_loss']:reg=reg+L_spec.sum(-1)*self.cfg['reg_spec_loss_lambda']
            return reg

    # Original-A known-light comparator: EVERY shading/decoding/regularization
    # body is the original method object, not a corrected function with a label.
    class OriginalKnownLight(CorrectedShader):
        pass
    for name in ('predict_anisotropic_components','compute_tangent_bitangent_flat','sample_diffuse_directions',
                 'sample_aniso_ggx_directions','compute_Dh','compute_pdf_aniso_ggx','compute_radiance',
                 'shade_anisotropic_mixed','anisotropic_regularization'):
        setattr(OriginalKnownLight,name,getattr(base_shader,name))
    SelectedShader=OriginalKnownLight if variant==ORIGINAL else CorrectedShader

    class CorrectedRenderer(base_renderer):
        material_variant=variant
        def forward(self,data):
            # Original forward changes the global default tensor type. Restore it
            # even when fail-stop orientation/domain validation raises.
            previous=torch.tensor([]).type()
            try:return super().forward(data)
            finally:torch.set_default_tensor_type(previous)

        def _init_shader(self):
            cfg=configuration(self.cfg,variant);require(cfg.get('material_variant',ORIGINAL)==variant,'Native renderer variant mismatch')
            self.cfg=cfg;self.cfg['shader_cfg']['is_real']=self.cfg['database_name'].startswith('real')
            self.shader_network=SelectedShader(self.cfg['shader_cfg'],lambda o,d:self.trace(o,d))

        def load_state_dict(self,state_dict,strict=True):
            require(strict is True,'Non-strict corrected renderer loading forbidden');state_gate(self.shader_network,state_dict,'shader_network.')
            return super().load_state_dict(state_dict,strict=True)

        @torch.no_grad()
        def predict_materials(self,batch_size=8192):
            require(type(batch_size) is int and batch_size>0,'Positive material export batch size required')
            device=next(self.shader_network.parameters()).device
            vertices=torch.as_tensor(np.asarray(self.mesh.vertices,np.float32),device=device);result={k:[] for k in ('mx','my','metallic','albedo','rotation','Q')}
            for i in range(0,len(vertices),batch_size):
                mx,my,m,kd,r,_=self.shader_network.predict_anisotropic_components(vertices[i:i+batch_size]);c,s=orientation(r)
                qxx=mx*mx*c*c+my*my*s*s;qyy=mx*mx*s*s+my*my*c*c;qxy=(mx*mx-my*my)*c*s
                for key,value in zip(result,(mx,my,m,kd,r,torch.cat([qxx,qxy,qxy,qyy],-1).reshape(-1,2,2))):result[key].append(value.cpu().numpy())
            require(len(vertices)>0,'Empty material export mesh')
            exported={k:np.concatenate(v) for k,v in result.items()}  # Linear physical values, no fake isotropic roughness/gamma conversion.
            if variant==ENERGY:
                exported['diffuse_budget']=exported['albedo']*(1-exported['metallic'])
                exported['specular_budget']=1-exported['diffuse_budget']
            return exported

        def relight(self,pts,views,normals,human_poses,environment,environment_identity,is_train=False):
            require(self.shader_network.cfg['human_lights'] is False,'Explicit environment relighting requires human_lights=false')
            with self.shader_network.environment(environment,environment_identity):
                outputs=self.shade(pts,views,normals,human_poses,is_train,step=None)
                outputs['relight_environment_identity']=environment_identity
                return outputs

    return SelectedShader,CorrectedRenderer
