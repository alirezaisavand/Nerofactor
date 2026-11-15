"""New small E11F CPU fixtures only; no original experiment, tracer or model."""
import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.bell_anisotropy_e11f import strengthen,admit_isotropic,summarize,coupling,LIMITS,GRIDS,NEW_CASES


def labels():
    n=torch.tensor([[0.,0.,1.]]*3,dtype=torch.float64)
    return dict(normals=n,T=torch.tensor([[1.,0.,0.]]*3,dtype=torch.float64),
        B=torch.tensor([[0.,1.,0.]]*3,dtype=torch.float64),theta_major=torch.tensor([.2,.3,-.1],dtype=torch.float64),
        widths_major_minor=torch.tensor([[.4,.2],[.3,.3],[.21,.2]],dtype=torch.float64),
        color=torch.full((3,3),.4,dtype=torch.float64),metallic=torch.full((3,1),.25,dtype=torch.float64))


def test_exact_requested_factor_no_mutation_product_axis_and_isotropic_limit():
    a=labels();before={k:v.clone() for k,v in a.items()};b=strengthen(torch,a);w=a['widths_major_minor'];v=b['widths_major_minor'];r=w[:,0]/w[:,1]
    assert torch.equal(v[:,0],w[:,0]*torch.sqrt(r)) and torch.equal(v[:,1],w[:,1]/torch.sqrt(r))
    np.testing.assert_allclose(v.prod(-1).numpy(),w.prod(-1).numpy(),rtol=16*np.finfo(float).eps,atol=0)
    np.testing.assert_allclose((v[:,0]/v[:,1]).numpy(),(r*r).numpy(),rtol=16*np.finfo(float).eps,atol=0)
    assert torch.equal(v[1],w[1])
    assert all(torch.equal(a[k],before[k]) for k in a)
    assert all(torch.equal(a[k],b[k]) for k in a if k!='widths_major_minor')


def test_isotropic_equivalence_is_explicit_and_nonwidth_changes_are_forbidden():
    a=labels();b=strengthen(torch,a);r=admit_isotropic(torch,a,b)
    assert r['mathematical_equivalence'] and r['naive_recomputed_widths_claimed_bit_identical'] is False
    assert len(r['canonical_mean_sha256'])==64 and 'no new isotropic BRDF calls' in r['reused_case']
    b['color']=b['color']+.01
    with pytest.raises(ValueError,match='Non-width'):admit_isotropic(torch,a,b)


@pytest.mark.parametrize('kind',['unordered','nan','outside_domain','wrong_shape','wrong_dtype'])
def test_invalid_physical_inputs_stop_without_clipping(kind):
    a=labels()
    if kind=='unordered':a['widths_major_minor'][0]=torch.tensor([.2,.4])
    if kind=='nan':a['widths_major_minor'][0,0]=float('nan')
    if kind=='outside_domain':a['widths_major_minor'][0]=torch.tensor([.99,.04])
    if kind=='wrong_shape':a['widths_major_minor']=torch.ones((3,3),dtype=torch.float64)
    if kind=='wrong_dtype':a['widths_major_minor']=a['widths_major_minor'].float()
    with pytest.raises(ValueError):strengthen(torch,a)


def components():
    a=np.zeros((84,8,3,3),dtype=np.float64);a[:,:,0]=.04;a[:,:,1]=.01;a[:,:,2]=a[:,:,0]+a[:,:,1]
    return a


def test_screens_retain_original_gates_and_do_not_authorize_full_dataset():
    original=components();strong=components();strong[:,:7,1]=.03
    for case in [1,3,5]:strong[:,case,1]+=.015
    for case in [2,4,6]:strong[:,case,1]-=.005
    strong[:,:,2]=strong[:,:,0]+strong[:,:,1]
    a={'color':np.full((84,3),.4),'metallic':np.full((84,1),.25)}
    r=summarize(strong,strong,original,original,a,np.full(84,512**2),np.arange(84,dtype=np.int64))
    assert r['all_cases_two_grid_converged'] and r['single_panel_screen']['all_original_25percent_floors']
    assert r['single_panel_screen']['full_dataset_authorized'] is False and r['E11D_admission_changed'] is False
    assert r['spatial_coverage']['stronger']['joint_orientation_major_minor']['resolved_probes']==84
    assert 'MIS' not in json.dumps(r)
    np.testing.assert_allclose(r['coupling_diagnostics']['original']['irradiance_RGB_min_median_max'][0],np.pi*.04/.3)


def test_nonzero_parallel_columns_are_reported_as_coupled_not_darkness():
    a=components()
    for case in [1,3,5]:a[:,case,1]+=.01
    a[:,:,2]=a[:,:,0]+a[:,:,1]
    data={'color':np.full((84,3),.4),'metallic':np.full((84,1),.25)}
    r=coupling(a,data,np.full(84,512**2),512**2)
    assert all(v['median_absolute_column_cosine']==pytest.approx(1) for v in r['column_alignment'].values())
    assert min(r['irradiance_RGB_min_median_max'][0])>0 and r['visible_fraction_range']==[1,1]


def test_metadata_import_and_zero_ray_boundary():
    code='''
import builtins
real=builtins.__import__
def guard(name,*args,**kw):
    if name.split('.')[0] in ('torch','network','train','raytracing','_raytracing','scientific_variants'):
        raise RuntimeError('Forbidden scientific import '+name)
    return real(name,*args,**kw)
builtins.__import__=guard
import experiments.bell_anisotropy_e11f
'''
    r=subprocess.run(['python3','-B','-c',code],capture_output=True,text=True);assert r.returncode==0,r.stderr
    tree=ast.parse(Path('experiments/bell_anisotropy_e11f.py').read_text(),feature_version=(3,8))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='trace' for n in ast.walk(tree))
    assert len(NEW_CASES)==7 and 84*sum(n*n for n in GRIDS)*7==LIMITS['brdf_point_directions']==192675840
    assert LIMITS['new_shadow_queries']==LIMITS['new_primary_rays']==0


@pytest.mark.parametrize('field',['approved','commit','freeze_sha256','image_id','output_directory','limits'])
def test_new_execution_requires_its_own_exact_authorization(tmp_path,monkeypatch,field):
    import experiments.bell_anisotropy_e11f as m
    p=tmp_path/'source';p.write_text('immutable')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_paths',lambda:['source'])
    args=SimpleNamespace(freeze=tmp_path/'freeze',freeze_sha256='a'*64,authorization=tmp_path/'authorization',
        authorization_sha256='b'*64,commit='c'*40,image_id=m.identity()['image_id'],output=tmp_path/'new')
    f=dict(protocol=m.PROTOCOL,grids=list(m.GRIDS),new_cases=list(m.NEW_CASES),limits=m.LIMITS,
           lighting=m.LIGHT,reference_sha256=m.REFERENCE_SHA,files={'source':m.digest(p)})
    a=dict(protocol=m.PROTOCOL,approved=True,commit=args.commit,freeze_sha256=args.freeze_sha256,
           image_id=args.image_id,output_directory=str(args.output),limits=m.LIMITS);a[field]=None
    monkeypatch.setattr(m,'read_report',lambda p,sha:(f if p==args.freeze else a,None))
    with pytest.raises(ValueError,match='approval'):m.authorize(args)
    assert not args.output.exists()
