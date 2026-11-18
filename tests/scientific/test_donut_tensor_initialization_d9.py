import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmarks import donut_recovery_v1 as recovery
from benchmarks import donut_tensor_d9 as definition
from benchmarks.donut_tensor_d9 import REPRESENTATION
from infrastructure.stage_sources import digest
from scientific_variants import donut_tensor_d8 as d8
from scientific_variants import donut_tensor_initialization_d9 as d9
from scientific_variants.donut_tensor_shader_d8 import tensor_shader
from test_donut_tensor_d8 import SyntheticCore


def test_projected_anchor_matches_chart_covaries_and_has_the_declared_degeneracy():
    n=F.normalize(torch.tensor([[1.,0,0],[0.,1,0],[1.,.4,.2],[1.,1e-7,0]],dtype=torch.float64),dim=-1)
    frame=d8.frame(n)
    zeros=n.new_zeros((len(n),6))
    rot=n.new_tensor([[1.,0.]]).expand(len(n),2)
    a=d9.ambient_from_start(zeros,frame)
    b=d9.chart_from_start(zeros[:,:1],zeros[:,:1],rot,frame)
    assert torch.allclose(a,b,rtol=0,atol=1e-14)
    widths=d8.sorted_widths(a)
    assert torch.allclose(widths[:2],n.new_tensor([[.52,.52],[.52,.48]]),rtol=0,atol=1e-14)
    assert bool(((widths>=.48-1e-14)&(widths<=.52+1e-14)).all())
    rot2=n.new_tensor([[.6,-.8],[.8,.6]])
    changed=d9.ambient_from_start(zeros,frame@rot2)
    assert torch.allclose(changed,rot2.T@a@rot2,rtol=0,atol=1e-14)


def test_chart_and_ambient_initial_derivatives():
    n=torch.tensor([[1.,0,0],[0.,1,0]],dtype=torch.float64)
    frame=d8.frame(n)
    theta=n.new_zeros(2,requires_grad=True)
    raw=n.new_zeros((2,1),requires_grad=True)
    def chart(x,y,t):
        rotation=torch.stack(((2*t).cos(),(2*t).sin()),dim=-1)
        return d9.chart_from_start(x,y,rotation,frame)
    assert torch.autograd.gradcheck(chart,(raw,raw.detach().clone().requires_grad_(),theta),atol=1e-9,rtol=1e-5)
    # Frobenius derivative of a director rotation is sqrt(2) times squared-width gap.
    j=torch.autograd.functional.jacobian(lambda t:chart(raw,raw,t),theta)
    norm=torch.stack([j[i,:,:,i].norm() for i in range(2)])
    assert torch.allclose(norm,n.new_tensor([0.,2**.5*(.52**2-.48**2)]),rtol=0,atol=1e-14)
    logits=n.new_zeros((2,6),requires_grad=True)
    assert torch.autograd.gradcheck(lambda h:d9.ambient_from_start(h,frame),(logits,),atol=1e-9,rtol=1e-5)
    jac=torch.autograd.functional.jacobian(lambda h:d9.ambient_from_start(h,frame),logits)
    assert all(int(torch.linalg.matrix_rank(jac[i,:,:,i,:].reshape(4,6)))==3 for i in range(2))


def test_selected_shader_passes_paired_q_rgb_gate_on_varied_normals(monkeypatch):
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *_:None)
    cls=d9.weak_start_shader(tensor_shader(SyntheticCore,lambda x:x))
    torch.manual_seed(6033); chart=cls();chart.select_tensor_representation('D9-C')
    torch.manual_seed(6033); ambient=cls();ambient.select_tensor_representation('D9-T')
    n=F.normalize(torch.randn((24,3)),dim=-1)
    n[0]=torch.tensor([1.,0,0])
    p=torch.rand_like(n)
    with torch.no_grad():
        gate,arrays=d9.paired_initial_gate(chart,ambient,p,n,n,REPRESENTATION)
    assert gate['passed'] and gate['world_Q_max_abs']<2e-8
    assert chart.tensor_variant!=ambient.tensor_variant
    assert (arrays['D9-C_linear_rgb']==arrays['D9-T_linear_rgb']).all()
    for model in (chart,ambient):
        out=model(p,n,n,None,0,True,tape_seed=813)
        out['linear_rgb'].sum().backward()
        assert model.rotation_predictor[6].bias.grad.abs().sum()>0


def test_d9_runtime_rejects_d8_arm_and_unpaired_approval(tmp_path,monkeypatch):
    plan=dict(protocol=definition.PROTOCOL,spatial_policy=definition.POLICY,
              representation=REPRESENTATION,limits=definition.limits(recovery.TRAIN_LIMITS),
              benchmark_generation_sha256='dataset',dataset_admission_sha256='admitted',initial_sha256='anchor')
    freeze=dict(tensor_training=plan,host_tensor_output='fresh')
    monkeypatch.setattr(recovery,'verify_freeze',lambda *_:freeze)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    path=tmp_path/'authorization.json'
    payload=dict(plan,arms=list(definition.ARMS),output_directories={a:'/outputs/'+a for a in definition.ARMS},
        approved=True,host_output_directory='fresh',selected_gpu='0',maximum_execution_launches=1,
        wall_seconds=plan['limits']['wall_seconds'],diffuse_white_in_objective=True,commit='commit',
        freeze_sha256='freeze',image_id=recovery.IMAGE)
    args=SimpleNamespace(freeze=tmp_path/'freeze.json',freeze_sha256='freeze',authorization=path,
                         commit='commit',image_id=recovery.IMAGE,output=Path('/outputs/D9-C'))
    path.write_text(json.dumps(payload));args.authorization_sha256=digest(path)
    with pytest.raises(ValueError,match='Fresh owned'):
        recovery.runtime(args,'D9-C')
    with pytest.raises(ValueError,match='dedicated execution freeze'):
        recovery.runtime(args,'D8-C')
    path.write_text(json.dumps(dict(payload,arms=['D9-C'])))
    args.authorization_sha256=digest(path)
    with pytest.raises(ValueError,match='Exact paired tensor'):
        recovery.runtime(args,'D9-C')
