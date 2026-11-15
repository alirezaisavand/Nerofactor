"""New E11E CPU controls only: analytic integrals, toy planes, fake execution."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.bell_quadrature_e11e import (midpoint_grid,controls,query_visibility,weighted_components,compare,LIMITS,
                                             PROTOCOL,GRIDS,LIGHT,CASES)


@pytest.mark.parametrize('size',[4,8,32])
def test_midpoint_domain_pdf_solid_angle_and_exact_analytic_moments(size):
    d,w=midpoint_grid(size);assert d.shape==(size*size,3) and d.dtype==np.float64
    assert (d[:,2]>0).all() and (d[:,2]<1).all()
    np.testing.assert_allclose(np.linalg.norm(d,axis=1),1,atol=3e-15,rtol=0)
    assert w*len(d)==pytest.approx(2*np.pi,abs=2e-14)
    assert ((d[:,2]/np.pi)*w).sum()==pytest.approx(1,abs=1e-14)
    assert (d[:,2]**2*w).sum()==pytest.approx(2*np.pi*(1/3-1/(12*size**2)),abs=2e-14)
    np.testing.assert_allclose((d[:,:2]*w).sum(0),[0,0],atol=2e-14,rtol=0)
    # Counting an extra sin(theta) or averaging weighted sums would fail this.
    assert abs((np.sqrt(1-d[:,2]**2)*d[:,2]*w).sum()-np.pi)>.1


def test_cpu_controls_and_world_frame_rotation_without_original_shader():
    assert controls()['status']=='pass'
    d,w=midpoint_grid(16)
    normals=np.array([[0,0,1.],[1,0,0.],[1,2,3.]])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    for n in normals:
        a=np.eye(3)[np.abs(n).argmin()];t=np.cross(n,a);t/=np.linalg.norm(t);b=np.cross(n,t)
        world=d[:,0,None]*t+d[:,1,None]*b+d[:,2,None]*n
        np.testing.assert_allclose(world@n,d[:,2],rtol=0,atol=2e-15)
        assert (world@n*w).sum()==pytest.approx(np.pi,abs=2e-14)
        # Match the exact finite-grid moment, including its known O(1/m^2) bias;
        # a rotated midpoint grid is not exactly a continuum-isotropic moment.
        qzz=2*np.pi*(1/3-1/(12*16**2));qxy=(2*np.pi-qzz)/2
        frame=np.stack([t,b,n],axis=1)
        np.testing.assert_allclose(world.T@world*w,frame@np.diag([qxy,qxy,qzz])@frame.T,atol=2e-14,rtol=0)


def test_actual_weighted_accumulator_integrates_lambert_and_directional_controls():
    size=16;directions,w=midpoint_grid(size)
    ci=torch.from_numpy(directions[None,:,2]);fd=torch.ones((1,size*size,3),dtype=torch.float64)/np.pi
    fs=torch.zeros_like(fd);color=torch.tensor([.2,.5,.8],dtype=torch.float64)
    light=color[None,None].expand_as(fd);visible=torch.ones_like(ci,dtype=torch.bool)
    answer=weighted_components(torch,fd,fs,light,ci,visible,w)
    np.testing.assert_allclose(answer[:,0].numpy(),color[None].numpy(),atol=2e-14,rtol=0)
    directional=weighted_components(torch,fd,fs,light*ci[...,None]**2,ci,visible,w)
    np.testing.assert_allclose(directional[:,0].numpy(),color[None].numpy()*(.5-1/(4*size**2)),atol=2e-14,rtol=0)
    assert torch.equal(answer[:,1],torch.zeros_like(answer[:,1]))


@pytest.mark.parametrize('value',[0,1,4.0,True])
def test_invalid_grid_specifications_fail(value):
    with pytest.raises(ValueError):midpoint_grid(value)


def test_visibility_matches_offset_float32_and_existing_depth_cutoff_with_analytic_plane():
    class Plane:
        def trace(self,o,d):
            assert o.dtype==d.dtype==torch.float32
            self.origin=o.clone();self.direction=d.clone()
            # Independent plane z=2; both rays hit it, but the legacy depth cutoff
            # intentionally treats the distant hit as visible environment.
            depth=(2-o[...,2])/d[...,2]
            return o,d,depth
    tracer=Plane();p=torch.zeros((1,3),dtype=torch.float64)
    wi=torch.tensor([[[0.,0.,1.],[np.sqrt(.99),0.,.1]]],dtype=torch.float64)
    v,depth=query_visibility(torch,tracer,p,wi)
    assert v.tolist()==[[False,True]]
    assert torch.equal(tracer.origin,p.float()[:,None]+1e-5*wi.float())
    assert torch.equal(tracer.direction,wi.float())
    assert depth[0,0]==pytest.approx(2-1e-5,abs=1e-6)


def test_nonfinite_shadow_depth_fails_without_redraw():
    class Bad:
        def trace(self,o,d):return o,d,torch.full(d.shape[:-1],float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):
        query_visibility(torch,Bad(),torch.zeros((1,3)),torch.ones((1,2,3)))


def components():
    a=np.zeros((84,8,3,3),dtype=np.float64);a[:,:,0]=.04;a[:,:,1]=.01;a[:,:,2]=a[:,:,0]+a[:,:,1]
    return a


def test_followup_comparison_never_changes_historical_failed_admission():
    a=components();r=compare(a,a.copy(),a.copy(),a.copy())
    assert r['all_cases_two_grid_converged'] and r['E11D_admission_changed'] is False
    assert r['original_E11D_decision']=='FIX_AND_PROCEED' and r['original_E11D_recommended'] is None
    assert r['full_dataset_or_training_authorized'] is False
    assert r['comparison']['truth']['fine_quadrature_minus_MIS_fine']['max_absolute']==0
    b=a.copy();b[:,:,1]+=.001;b[:,:,2]=b[:,:,0]+b[:,:,1]
    r=compare(b,a,a,a);assert not r['all_cases_two_grid_converged']
    assert r['comparison']['truth']['fine_quadrature_minus_coarse']['signed_mean_rgb'][0]<0
    with pytest.raises(ValueError):compare(a[:80],a,a,a)


def test_visibility_packing_preserves_probe_and_direction_order():
    mask=np.arange(4*32).reshape(4,32)%3==0
    packed=np.zeros((4,4),dtype=np.uint8)
    for start in (0,16):packed[:,start//8:(start+16)//8]=np.packbits(mask[:,start:start+16],axis=1,bitorder='big')
    assert np.array_equal(np.unpackbits(packed,axis=1,bitorder='big'),mask)
    assert np.array_equal(np.unpackbits(packed,axis=1).sum(1),mask.sum(1))


def test_metadata_imports_with_torch_and_scientific_modules_blocked():
    code='''
import builtins
real=builtins.__import__
def guard(name,*args,**kw):
    if name.split('.')[0] in ('torch','network','train','raytracing','_raytracing','scientific_variants'):
        raise RuntimeError('Forbidden scientific import '+name)
    return real(name,*args,**kw)
builtins.__import__=guard
from experiments.bell_quadrature_e11e import controls
assert controls()['status']=='pass'
'''
    p=subprocess.run(['python3','-B','-c',code],capture_output=True,text=True);assert p.returncode==0,p.stderr


@pytest.mark.parametrize('field',['approved','commit','freeze_sha256','image_id','output_directory','limits'])
def test_authorization_mismatch_precedes_all_native_work(tmp_path,monkeypatch,field):
    import experiments.bell_quadrature_e11e as m
    p=tmp_path/'source';p.write_text('synthetic immutable')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_paths',lambda:['source'])
    args=SimpleNamespace(freeze=tmp_path/'freeze',freeze_sha256='f'*64,authorization=tmp_path/'auth',
        authorization_sha256='a'*64,commit='b'*40,image_id=m.identity()['image_id'],output=tmp_path/'output')
    f=dict(protocol=PROTOCOL,grids=list(GRIDS),lighting=LIGHT,limits=LIMITS,cases=list(CASES),camera_id='58',probes=84,files={'source':m.digest(p)})
    a=dict(protocol=PROTOCOL,approved=True,commit=args.commit,freeze_sha256=args.freeze_sha256,
           image_id=args.image_id,output_directory=str(args.output),limits=LIMITS);a[field]=None
    monkeypatch.setattr(m,'read_report',lambda path,sha:(f if path==args.freeze else a,None))
    with pytest.raises(ValueError,match='authorization'):m.authorize(args)
    assert not args.output.exists()


def test_work_budget_and_python38_grammar():
    assert 84*sum(s*s for s in GRIDS)==LIMITS['shadow_directions']==27525120
    assert 8*LIMITS['shadow_directions']==LIMITS['brdf_point_directions']==220200960
    ast.parse(Path('experiments/bell_quadrature_e11e.py').read_text(),feature_version=(3,8))
