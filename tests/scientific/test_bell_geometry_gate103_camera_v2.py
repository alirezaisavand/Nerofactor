"""Focused CPU regressions for calibrated camera processing and capture reuse."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from benchmarks import bell_geometry_gate103_v2 as p
from experiments import bell_geometry_gate103_v2 as run
from infrastructure.signatures import SignatureError


def original_camera():
    tree=ast.parse((p.ROOT/'scientific_sources/geometry_g/network/renderer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NeROShapeRenderer')
    names={'_construct_ray_batch','_process_ray_batch','near_far_from_sphere','get_human_coordinate_poses'}
    cls=ast.ClassDef(name='Camera',bases=[],keywords=[],decorator_list=[],body=[
        n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names])
    scope=dict(torch=torch,F=torch.nn.functional)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'original-camera','exec'),scope)
    camera=scope['Camera']();camera.cfg={'fixed_camera':False}
    return camera


def test_actual_16_camera_prefixes_match_archives_through_production_adapter():
    frozen=p.read(p.ROOT/p.original.FREEZE);base=Path(frozen['acquisition'])/'geometry'
    camera=original_camera();rng=torch.get_rng_state().clone();total=0
    with np.load(base/'calibrations-safe.npz',allow_pickle=False) as cal:
        for name,panel in frozen['neural_panel'].items():
            indices=np.array(panel['indices'],np.int64)
            info=dict(imgs=torch.zeros(1,3,128,128),Ks=torch.from_numpy(cal['K'][int(name)].copy())[None],
                      poses=torch.from_numpy(cal['pose'][int(name)].copy())[None])
            values,batch,poses=run.camera_prefix_cpu(camera,info,indices,torch)
            assert all(v.device.type=='cpu' for v in values) and batch['dirs'].shape==(len(indices),3)
            with np.load(base/'rays'/f'{name}.npz',allow_pickle=False) as archive:
                errors=run.ray_error(values,archive['origins'][indices],archive['directions'][indices])
            assert max(errors.values())<=2e-6,(name,errors)
            assert values[2].shape==values[3].shape==(len(indices),1)
            assert values[4].shape==(len(indices),3,4)
            total+=len(indices)
    assert total==4055 and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('fail',[False,True])
def test_precision_witness_restores_policy_and_never_selects_gpu_rays(tmp_path,monkeypatch,fail):
    flag=NS(allow_tf32=True);fake=NS(no_grad=torch.no_grad,backends=NS(cuda=NS(matmul=flag)))
    cpu=(torch.zeros(2,3),torch.ones(2,3),torch.zeros(2,1),torch.ones(2,1),torch.zeros(2,3,4))
    calls=[]
    class Camera:
        def _process_ray_batch(self,batch,poses):
            calls.append(flag.allow_tf32)
            if fail and not flag.allow_tf32:raise RuntimeError('second-probe failure')
            return tuple(v+(0.001 if flag.allow_tf32 else 0) for v in cpu)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(run,'cuda_factories',lambda unused:nullcontext())
    row={};path=tmp_path/'precision.npz'
    args=(Camera(),{'dirs':cpu[1]},torch.zeros(1,3,4),cpu,cpu[0].numpy(),cpu[1].numpy(),fake,path,row)
    if fail:
        with pytest.raises(RuntimeError):run.camera_precision_witness(*args)
        assert not path.exists()
    else:
        run.camera_precision_witness(*args)
        assert row['camera_precision_probe_calls']==2
        assert row['camera_precision_diagnostic']['used_for_rendering']=='cpu'
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['cpu_origins'],cpu[0])
            assert not np.array_equal(z['default_origins'],z['cpu_origins'])
            assert np.array_equal(z['TF32_disabled_origins'],z['cpu_origins'])
    assert calls==[True,False] and flag.allow_tf32 is True


def completed_fixture(tmp_path):
    identity=dict(bytes=1,sha256='a'*64)
    row=dict(step=15000,MC_attempts=1,MC_returns=1,grid_tiles=512,primary_BVH_rays=196608,
             CPU_ray_checks=384,views=[{}]*6,SDF_gate={'passed':True},checkpoint=identity)
    receipt=dict(status='failed',commit='d19f71fee36edd109a0999baed0536180f46423e',
        image='fixture-image',backend={'variant':'fixture'},
        freeze_sha256='8d0337fe10c02aa00f7ecf797ac02620888daa4e5a2766698ecf856918f20c82',
        error=dict(type='SignatureError',message='Neural/mesh rays differ'),
        observed=dict(loads=1,renders=0),optimizer_returns=0,steps=[row])
    (tmp_path/'run').mkdir();p.write(tmp_path/'run/receipt.json',receipt)
    (tmp_path/'exit-code.txt').write_text('1\n');(tmp_path/'container-absent.txt').write_text('absent\n')
    files={str(x.relative_to(tmp_path)):p.item(x) for x in tmp_path.rglob('*') if x.is_file()}
    return dict(completed_15k=dict(files=files),checkpoints={'15000':{'identity':identity}},
                image=receipt['image'],backend=receipt['backend']),receipt


@pytest.mark.parametrize('defect',[None,'identity','renders','cleanup'])
def test_capture_reuse_rejects_wrong_identity_or_incomplete_phase(tmp_path,defect):
    plan,r=completed_fixture(tmp_path)
    if defect=='identity':plan['completed_15k']['files']['run/receipt.json']['sha256']='0'*64
    if defect=='renders':
        r['observed']['renders']=1
        path=tmp_path/'run/receipt.json';path.write_text(__import__('json').dumps(r))
        plan['completed_15k']['files']['run/receipt.json']=p.item(path)
    if defect=='cleanup':
        path=tmp_path/'container-absent.txt';path.write_text('present\n')
        plan['completed_15k']['files']['container-absent.txt']=p.item(path)
    if defect:
        with pytest.raises(SignatureError):p.completed_15k(plan,tmp_path)
    else:assert p.completed_15k(plan,tmp_path)==r['steps'][0]


def test_continuation_retains_science_and_reduces_only_completed_work():
    old=p.original.read(p.ROOT/p.original.FREEZE)
    assert p.NEURAL['ray_builder']=='original CPU construction AND processing, then CUDA transfer'
    assert {k:v for k,v in p.NEURAL.items() if k not in ('ray_builder','precision_witness')}=={
        k:v for k,v in old['neural'].items() if k!='ray_builder'}
    assert p.LIMITS['dense_SDF_points']==2*512**3 and p.LIMITS['MC_calls']==2
    assert p.LIMITS['audit_BVH_rays']==393216 and p.LIMITS['CPU_ray_checks']==1024
    assert p.LIMITS['consistency_SDF_points']==3072 and p.LIMITS['consistency_gradient_points']==1024
    assert p.RESOURCES==old['resources'] and p.LIMITS['neural_render_calls']==128
    assert p.LIMITS['optimizer_returns']==0
    shell=(p.ROOT/'experiments/run_bell_geometry_gate103_v2.sh').read_text()
    assert 'freeze='+p.FREEZE in shell and 'dst=/inputs/completed15k,readonly' in shell


def test_camera_error_rejects_nonfinite_values_instead_of_masking_with_max():
    origin=torch.zeros(2,3);direction=torch.ones(2,3);direction[0,0]=float('nan')
    with pytest.raises(SignatureError):
        run.ray_error((origin,direction),np.zeros((2,3)),np.ones((2,3)))
