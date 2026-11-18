"""Independent and synthetic CPU checks for the paired tensor path."""

import math
import json
import os
from pathlib import Path
from types import SimpleNamespace

from benchmarks import donut_recovery_v1 as recovery
from benchmarks import donut_tensor_d8 as definition
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.bell_reference_e11 import kernel
from scientific_variants import donut_tensor_d8 as d8
from scientific_variants.donut_tensor_shader_d8 import legacy_radiance, paired_initial_gate, tensor_shader
from benchmarks.donut_tensor_d8 import REPRESENTATION
from infrastructure.stage_sources import digest


def normal_view():
    n=torch.tensor([[.3,.5,.8]],dtype=torch.float64)
    n=F.normalize(n,dim=-1)
    u=d8.frame(n)
    wo=F.normalize(.4*u[...,0]+n,dim=-1)
    return n,u,wo


def close(a,b,atol=1e-7,rtol=1e-6):
    assert torch.allclose(a,b,atol=atol,rtol=rtol)


def test_independent_cartesian_kurt_brdf_reciprocity_and_equivalences():
    n,u,wo=normal_view()
    axis=math.cos(.71)*u[...,0]+math.sin(.71)*u[...,1]
    other=torch.cross(n,axis,dim=-1)
    a,b=n.new_tensor([[.43]]),n.new_tensor([[.13]])
    q=d8.chart_q(a,b,axis,u)
    for alternate in (d8.chart_q(b,a,other,u),d8.chart_q(a,b,-axis,u)):
        close(q,alternate)
    g=torch.randn((1,40,3),dtype=n.dtype,generator=torch.Generator().manual_seed(11))
    wi=F.normalize(g+n[:,None]*.2,dim=-1)
    kd=n.new_tensor([[.1,.7,.4]])
    m=n.new_tensor([[.77]])
    # Independent reference expresses Q as minor^2 I plus a rank-one uplift.
    material=dict(a=b,b=(a.square()-b.square()).sqrt(),u=axis,color=kd,metallic=m)
    ours=d8.brdf(q,u,n,wo,wi,kd,m)
    target=kernel(wi,wo,n,material)
    for value,ref in zip(ours,target):
        close(value,ref,atol=2e-13,rtol=1e-12)
        assert (value>=0).all() and torch.isfinite(value).all()
        assert value[(wi*n[:,None]).sum(-1)<=0].eq(0).all()
    swapped=d8.brdf(q.expand(40,2,2),u.expand(40,3,2),n.expand(40,3),wi[0],wo[:,None].expand(40,1,3),
                    kd.expand(40,3),m.expand(40,1))
    close(ours[1][0],swapped[1][:,0])


def test_world_gaussian_covariance_proposal_density_and_matched_tape():
    n,u,wo=normal_view()
    q=d8.symmetric(n.new_tensor([.47**2]),n.new_tensor([.27**2]),n.new_tensor([.03]))
    tape=torch.randn((1,49152,3),dtype=n.dtype,generator=torch.Generator().manual_seed(27))
    h,wi,ch,nd,pdf=d8.sample(q,u,n,wo,tape,density=True)
    alternate=d8.sample(q.clone(),u,n,wo,tape,density=True)
    assert all(torch.equal(a,b) for a,b in zip((h,wi,ch,nd,pdf),alternate))
    slope=(h@u)/ch
    centered=slope-slope.mean(1,keepdim=True)
    close(centered.transpose(-1,-2)@centered/slope.shape[1],q/2,atol=.002,rtol=.02)
    close(d8.ndf(q,u,n,h),nd,atol=1e-12,rtol=1e-12)
    close(pdf,nd*ch/(4*(h*wo[:,None]).sum(-1,keepdim=True).abs()))
    assert ((wi*n[:,None]).sum(-1)<0).any()
    # Integrate projected area density independently in dcos(theta) dphi.
    c=((torch.arange(160,dtype=n.dtype)+.5)/160)[:,None].expand(160,220)
    p=((torch.arange(220,dtype=n.dtype)+.5)*(2*math.pi/220))[None,:].expand(160,220)
    xy=(1-c.square()).sqrt()[...,None]*(p.cos()[...,None]*u[0,:,0]+p.sin()[...,None]*u[0,:,1])
    h=(xy+c[...,None]*n).reshape(1,-1,3)
    mass=(2*math.pi*d8.ndf(q,u,n,h).reshape_as(c)*c).mean()
    assert abs(mass-1)<.002


def test_world_and_computational_frame_covariance():
    n,u,wo=normal_view()
    x=n.new_tensor([[.3,-.8,-.4,.1,.4,-.3]])
    q=d8.ambient_q(x,u)
    phi=.7; c,s=math.cos(phi),math.sin(phi)
    o=n.new_tensor([[c,-s],[s,c]])
    new=u@o
    qp=d8.ambient_q(x,new)
    close(qp,o.T@q@o,atol=1e-12,rtol=1e-12)
    tape=n.new_tensor([[[.1,.2,.3],[.4,.5,-.1]]])
    a=d8.sample(q,u,n,wo,tape,density=True)
    b=d8.sample(qp,new,n,wo,tape,density=True)
    for va,vb in zip(a,b):
        close(va,vb)
    rot=n.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotated=d8.sample(q,rot@u,n@rot.T,wo@rot.T,tape@rot.T,density=True)
    close(a[0]@rot.T,rotated[0])
    close(a[1]@rot.T,rotated[1])
    close(a[-1],rotated[-1])


@pytest.mark.parametrize('logits', [[0.,0.,0.,0.,0.,0.],[.01,-.01,0.,.002,0.,0.],
    [-80.,70.,-20.,1.,-2.,3.]])
def test_tensor_and_sample_gradient_checks(logits):
    n,u,wo=normal_view()
    x=n.new_tensor([logits]).requires_grad_()
    g=n.new_tensor([[[.6,.2,-.1],[-.3,.2,1.1]]])
    def evaluate(z):
        q=d8.ambient_q(z,u)
        h=d8.sample(q,u,n,wo,g)[0]
        return torch.cat((q.reshape(-1),h.reshape(-1),d8.ndf(q,u,n,wo[:,None]).reshape(-1)))
    assert torch.autograd.gradcheck(evaluate,(x,),eps=1e-5,atol=3e-8,rtol=5e-4)


def test_extreme_logits_horizon_and_factor_against_eigen_reference():
    n,u,_=normal_view()
    for scale in (0.,1e-9,1.,1e6,1e37):
        x=(n.new_tensor([[1.,1.,-.7,-1.,.1,-.2]])*scale).requires_grad_()
        q=d8.ambient_q(x,u)
        root,inv,_=d8.factor(q)
        e=torch.linalg.eigvalsh(q)
        assert bool(((e>=d8.LOW-5e-15)&(e<=d8.HIGH+5e-15)).all())
        close(root@root,q,atol=1e-14,rtol=1e-13)
        close(q@inv,torch.eye(2,dtype=n.dtype)[None],atol=1e-13,rtol=1e-13)
        h=torch.stack((u[0,:,0],-n[0],n[0],F.normalize(u[0,:,0]+1e-15*n[0],dim=0)))[None]
        d=d8.ndf(q,u,n,h)
        assert d[0,2,0]>0 and d[0,[0,1,3]].eq(0).all()
        assert torch.isfinite(torch.autograd.grad(d.sum()+root.sum(),x)[0]).all()


def test_axes_export_reconstructs_isotropic_and_anisotropic_q():
    n,u,_=normal_view()
    u=u.float()
    for components in ([0.,0.,0.,0.,0.,0.],[2.,-4.,7.,3.,-5.,19.]):
        x=torch.tensor([components],requires_grad=True)
        q=d8.ambient_q(x,u)
        mx,my,t,b=d8.export_axes(q,u)
        assert not any(value.requires_grad for value in (mx,my,t,b))
        reconstructed=mx[...,None].square()*t[...,None]*t[:,None]+my[...,None].square()*b[...,None]*b[:,None]
        close(reconstructed.double(),d8.world_tensor(q,u),atol=1e-6)


def test_runtime_binds_both_arm_identities_and_rejects_single_arm(tmp_path,monkeypatch):
    plan=dict(protocol=definition.PROTOCOL,spatial_policy=definition.POLICY,
              representation=definition.REPRESENTATION,limits=definition.limits(recovery.TRAIN_LIMITS),
              benchmark_generation_sha256='benchmark',dataset_admission_sha256='admitted',initial_sha256='anchor')
    freeze=dict(tensor_training=plan,host_tensor_output='fresh')
    monkeypatch.setattr(recovery,'verify_freeze',lambda *_:freeze)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    authorization=tmp_path/'authorization.json'
    payload=dict(plan,arms=list(definition.ARMS),output_directories={arm:'/outputs/'+arm for arm in definition.ARMS},
        host_output_directory='fresh',selected_gpu='0',maximum_execution_launches=1,approved=True,
        wall_seconds=plan['limits']['wall_seconds'],diffuse_white_in_objective=True,commit='commit',
        freeze_sha256='freeze',image_id=recovery.IMAGE)
    args=SimpleNamespace(freeze=tmp_path/'freeze.json',freeze_sha256='freeze',authorization=authorization,
                         commit='commit',image_id=recovery.IMAGE,output=Path('/outputs/D8-C'))
    authorization.write_text(json.dumps(payload));args.authorization_sha256=digest(authorization)
    with pytest.raises(ValueError,match='Fresh owned'):
        recovery.runtime(args,'D8-C')
    with pytest.raises(ValueError,match='D8 requires'):
        recovery.runtime(args,'D2')
    authorization.write_text(json.dumps(dict(payload,arms=['D8-C'])))
    args.authorization_sha256=digest(authorization)
    with pytest.raises(ValueError,match='Exact paired'):
        recovery.runtime(args,'D8-C')


def test_legacy_prior_uses_valid_count_detached_softmax_and_keeps_proposal_count():
    fd=torch.tensor([[.3,.1,.2]],requires_grad=True)
    d=fd.new_tensor([[[2.,1.,.5]]])
    s=fd.new_tensor([[[2.,1.,.5],[4.,1.,.5],[3.,1.,.5]]])
    fr=fd.new_ones((1,3,3))*.4
    wi=fd.new_tensor([[[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.]]])
    n=fd.new_tensor([[0.,0.,1.]])
    ndf=fd.new_tensor([[[.2],[1.1],[.4]]]).requires_grad_()
    diffuse,spec,energy,prior,_=legacy_radiance(fd,d,s,fr,wi,n,fd.new_ones((1,3,1)),ndf)
    close(diffuse,d[:,0]*fd)
    close(spec,(s[:,0]+s[:,2])*.4/3)
    close(energy,fd.new_full((1,3),.4))
    keep=ndf.softmax(1)[:,[0,2]].sum(1)
    close(prior,keep*fd/6)
    grads=torch.autograd.grad(prior.sum(),(fd,ndf),allow_unused=True)
    assert grads[1] is None
    close(grads[0],(keep/6).expand_as(fd))


class SyntheticCore(nn.Module):
    """Shape and call contract of the pinned core without graphics imports."""
    def __init__(self):
        super().__init__()
        self.cfg=dict(specular_sample_num=19,reg_change=True,reg_energy_loss=True,reg_spec_loss=True,
                      reg_lambda1=.05,reg_energy_loss_lambda=.05,reg_spec_loss_lambda=.05,
                      change_type='gaussian',change_eps=.02)
        self.feats_network=nn.Linear(3,4)
        for name,dim in [('mx_predictor',1),('my_predictor',1),('rotation_predictor',2),
                         ('kd_predictor',3),('metallic_predictor',1),('outer_light',3)]:
            setattr(self,name,nn.Sequential(nn.Linear(7,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),
                    nn.Linear(8,8),nn.ReLU(),nn.utils.weight_norm(nn.Linear(8,dim)),nn.Sigmoid()))

    def predict_anisotropic_components(self,p):
        f=self.features(p)
        mx,my=.04+.96*self.mx_predictor(f),.04+.96*self.my_predictor(f)
        r=self.rotation_predictor(f)
        return mx,my,self.metallic_predictor(f),self.kd_predictor(f),F.normalize(2*r-1,dim=-1),r

    def compute_tangent_bitangent_flat(self,n,sources):
        return d8.frame(n).unbind(-1)

    def get_orthogonal_directions(self,n):
        return d8.frame(n)[...,0]

    def sample_diffuse_directions(self,n,is_train):
        return n[:,None]

    def get_lights(self,p,w,poses):
        return .5+.1*w+0*p,None,None,None,None

    def fresnel_schlick_directions(self,f0,wo,wi):
        return f0,None,None

    def diffuse_term(self,kd,m,is_seperate=False):
        return kd*(1-m)


def test_selected_shader_paired_admission_checkpoint_and_real_backward(monkeypatch):
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *_:None)
    cls=tensor_shader(SyntheticCore,lambda x:x)
    torch.manual_seed(6033); chart=cls();chart.select_tensor_representation('D8-C')
    torch.manual_seed(6033); ambient=cls();ambient.select_tensor_representation('D8-T')
    n=torch.zeros((24,3));n[:,2]=1
    p=torch.rand_like(n)
    with torch.no_grad():
        gate,_=paired_initial_gate(chart,ambient,p,n,n,REPRESENTATION)
    assert gate['passed']
    with pytest.raises(ValueError,match='Foreign'):
        chart.load_state_dict(ambient.state_dict())
    with pytest.raises(ValueError,match='explicit world tape'):
        ambient(p,n,n,None,0,True)
    with pytest.raises(ValueError,match='Axes are only exported'):
        q,u,kd,m=ambient.predict_tensor(p,n)
        ambient.shade_tensor(p,n,n,q,u,kd,m,5,True,export=True)
    for model in (chart,ambient):
        torch.manual_seed(4)
        out=model(p,n,n,None,0,True,tape_seed=5)
        reg=sum(model.regularize_tensor(p,n,out)).mean()
        advanced=torch.get_rng_state()
        torch.manual_seed(4)
        torch.rand(24,19);torch.rand(24,19)
        torch.rand(24,1);torch.normal(0.,.02,size=(24,1))
        assert torch.equal(advanced,torch.get_rng_state())
        (out['linear_rgb'].mean()+reg).backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all()
                   for head in (model.mx_predictor,model.my_predictor,model.rotation_predictor)
                   for v in head.parameters())
        rotation_out=model.rotation_predictor[6].weight_g.grad
        if model is chart:
            assert rotation_out.eq(0).all()
        else:
            # z-normal means xy is tangent; xz and yz are in the pointwise nullspace.
            assert model.my_predictor[6].bias.grad[1].abs()>1e-9
