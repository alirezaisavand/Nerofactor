"""The D9 start in the chart and ambient coordinates of the shared tensor shader."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.donut_tensor_d9 import ARMS, REPRESENTATION
from infrastructure.signatures import require
from scientific_variants import donut_tensor_d8 as tensor
from scientific_variants.material_a_v1 import finite, orientation, unit

VARIANTS=dict(zip(ARMS,('donut_tensor_chart_weak_start_v1','donut_tensor_ambient_weak_start_v1')))


def inverse_bound(width):
    y=(width**2-tensor.MID)/tensor.SPAN
    return y/(1-y*y)**.5


def anchor_logits(frame):
    """Project z_major I + (z_minor-z_major) e_x e_x^T into the tangent plane."""
    spec=REPRESENTATION['initialization']
    major=inverse_bound(spec['major_width'])
    minor=inverse_bound(spec['minimum_minor_width'])
    x=frame[:,0].double()
    eye=torch.eye(2,dtype=x.dtype,device=x.device)
    return major*eye+(minor-major)*x[:,:,None]*x[:,None,:]


def ambient_from_start(residual,frame):
    xx,yy,zz,xy,xz,yz=residual.double().unbind(-1)
    h=torch.stack((xx,xy,xz,xy,yy,yz,xz,yz,zz),-1).reshape(-1,3,3)
    u=frame.double()
    local=u.transpose(-1,-2)@h@u
    return tensor.bounded_local((local+local.transpose(-1,-2))/2+anchor_logits(frame))


def chart_from_start(logits_x,logits_y,rotation,frame):
    x=frame[:,0].double()
    norm=torch.linalg.norm(x,dim=-1,keepdim=True)
    # A's n-cross-X chart with its Y fallback, evaluated in the computational frame.
    chart_ref=torch.where(norm<1e-6,frame[:,1].double(),x)
    chart_ref=F.normalize(chart_ref,dim=-1)
    a=torch.stack((-chart_ref[:,1],chart_ref[:,0]),-1)
    b=-chart_ref
    c,s=orientation(rotation.double())
    axis=c*a+s*b
    q0=tensor.bounded_local(anchor_logits(frame))
    width=tensor.sorted_widths(q0)
    bias=torch.log((width-.04)/(1-width))
    m=.04+.96*(torch.cat((logits_x,logits_y),dim=-1).double()+bias).sigmoid()
    x,y=axis.unbind(-1)
    mean=(m[:,0].square()+m[:,1].square())/2
    gap=(m[:,0].square()-m[:,1].square())/2
    diag=gap*(x.square()-y.square())
    return tensor.symmetric(mean+diag,mean-diag,2*gap*x*y)


def weak_start_shader(parent):
    class WeakStartShader(parent):
        def select_tensor_representation(self,arm):
            require(arm in ARMS,'D9 representation required')
            # Reuse the identical D8 subnet construction and seeded hidden weights.
            super().select_tensor_representation({'D9-C':'D8-C','D9-T':'D8-T'}[arm])
            self.tensor_arm=arm
            self.tensor_variant=VARIANTS[arm]
            self._donut_tensor_signature=torch.tensor(list(self.tensor_variant.encode('ascii')),dtype=torch.uint8)
            if arm=='D9-C':
                with torch.no_grad():
                    for head in (self.mx_predictor,self.my_predictor):
                        head[6].bias.zero_()
                        head[7]=nn.Identity()

        def predict_tensor(self,points,normals):
            unit(normals,'tensor normals')
            frame=tensor.frame(normals)
            f=self.features(points)
            a,b,r=self.mx_predictor(f),self.my_predictor(f),self.rotation_predictor(f)
            if self.tensor_arm=='D9-C':
                q=chart_from_start(a,b,2*r-1,frame)
            else:
                residual=torch.cat((a,b,r),dim=-1)
                finite(residual,'ambient tensor logits')
                q=ambient_from_start(residual,frame)
            return q,frame,self.kd_predictor(f),self.metallic_predictor(f)

    return WeakStartShader


def paired_initial_gate(first,second,p,n,wo,policy):
    """Compare physical initialization with the predeclared geometry-only anchor."""
    require(not torch.is_grad_enabled() and len(p)==policy['native_gate_points'],
            'D9 initial gate is a fixed evaluation')
    shared=('feats_network','kd_predictor','metallic_predictor','outer_light','inner_light')
    def shared_state(model):
        return {k:v for k,v in model.state_dict().items() if k.split('.')[0] in shared}
    a,b=shared_state(first),shared_state(second)
    require(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'D9 shared initialization differs')
    capture={}
    max_q=0.
    seed=policy['initial_gate_seed']
    for model in (first,second):
        model.eval()
        q,u,kd,m=model.predict_tensor(p,n)
        target=tensor.bounded_local(anchor_logits(u))
        max_q=max(max_q,float((q-target).abs().max()))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result=model.shade_tensor(p,n,wo,q,u,kd,m,seed,False)
        for k in ('linear_rgb','linear_diffuse','linear_specular','kd','metallic'):
            capture[model.tensor_arm+'_'+k]=result[k].cpu().numpy()
        capture[model.tensor_arm+'_world_Q']=tensor.world_tensor(q,u).cpu().numpy()
    def delta(k):
        return float(abs(capture['D9-C_'+k]-capture['D9-T_'+k]).max())
    q_delta=delta('world_Q')
    rgb_delta=max(delta(k) for k in ('linear_rgb','linear_diffuse','linear_specular'))
    scalar_equal=delta('kd')==delta('metallic')==0
    passed=(max_q<=policy['initial_Q_atol'] and q_delta<=policy['initial_Q_atol']
            and rgb_delta<=policy['initial_RGB_atol'] and scalar_equal)
    return dict(passed=passed,shared_state_equal=True,scalar_predictions_equal=scalar_equal,
                world_Q_max_abs=q_delta,local_initial_Q_max_abs=max_q,RGB_component_max_abs=rgb_delta),capture
