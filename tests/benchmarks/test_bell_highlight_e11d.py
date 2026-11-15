"""New E11D synthetic CPU fixtures only; no original renderer or real-data integration."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.bell_highlight_e11d import (candidates, mean_environment, subset, measures, ENV, CASES)
from experiments.bell_highlight_pilot_e11d import counterfactual, kernel_axes, sensitivity, LIMITS


def fixture():
    d=torch.float64
    n=torch.tensor([[0.,0.,1.]],dtype=d)
    return dict(normals=n,theta_major=torch.tensor([.3],dtype=d),T=torch.tensor([[1.,0.,0.]],dtype=d),
                B=torch.tensor([[0.,1.,0.]],dtype=d),widths_major_minor=torch.tensor([[.5,.2]],dtype=d),
                color=torch.tensor([[.3,.5,.2]],dtype=d),metallic=torch.tensor([[.25]],dtype=d))


def test_lighting_has_equal_global_rgb_radiance_and_fixed_unit_axes():
    for name,env in candidates().items():
        np.testing.assert_allclose(mean_environment(env),mean_environment(ENV),rtol=0,atol=3e-17)
        assert all(np.isclose(np.linalg.norm(x['axis']),1) for x in env['lobes'])
        assert all(np.min(x['RGB'])>0 for x in env['lobes'])
    assert len(candidates())==3 and candidates()['L0_v3']==ENV


def test_subset_is_spatial_deterministic_and_independent_of_material_values():
    hit=np.ones((128,128),bool);ids=subset(hit)
    assert len(ids)==192 and np.array_equal(ids,subset(hit)) and len(np.unique(ids))==192
    for y in range(8):
        for x in range(8):assert sum(((ids//128)//16==y)&((ids%128)//16==x))==3
    hit[:]=False;hit[106,53]=True;assert np.array_equal(subset(hit),[13621])


@pytest.mark.parametrize('hit',[np.ones(16384,bool),np.ones((128,128),float)])
def test_invalid_selection_inputs(hit):
    with pytest.raises(ValueError):subset(hit)


def test_saved_component_metrics_and_specular_quantization():
    c=np.zeros((100,3,3));c[:,0]=.04;c[:,1]=.01;c[:,2]=c[:,0]+c[:,1]
    r,h=measures(c);assert r['integrated_specular_to_diffuse']==pytest.approx(.25)
    assert r['integrated_specular_fraction']==pytest.approx(.2) and not h.any()
    assert r['specular_zero_code_fraction']==0 and r['specular_at_least_two_codes_fraction']==1
    c[:,2]+=.01
    with pytest.raises(ValueError,match='sum'):measures(c)


def test_clipping_is_reported_without_hidden_exposure_or_clamp():
    c=np.ones((2,3,3));c[:,2]=2;r,h=measures(c)
    assert r['out_of_png_domain'] and 'specular_zero_code_fraction' not in r


def test_counterfactuals_preserve_color_metallic_and_have_expected_physical_effects():
    data=fixture();original={k:v.clone() for k,v in data.items()}
    for name in CASES:
        m=counterfactual(torch,data,name)
        assert torch.equal(m['color'],data['color']) and torch.equal(m['metallic'],data['metallic'])
        assert torch.allclose((m['axis']*m['n']).sum(-1),torch.zeros(1,dtype=torch.float64),atol=1e-15)
        assert torch.allclose(torch.linalg.norm(m['axis'],dim=-1),torch.ones(1,dtype=torch.float64))
    iso=counterfactual(torch,data,'isotropic')['widths']
    assert torch.allclose(iso.prod(-1),data['widths_major_minor'].prod(-1))
    assert all(torch.equal(data[k],v) for k,v in original.items())


def test_independent_principal_axis_kernel_matches_existing_cartesian_reference():
    # Tiny stand-in materials and fixed directions only; no actual Bell integration.
    from benchmarks.bell_reference_e11 import kernel
    d=fixture();m=counterfactual(torch,d,'truth');n=d['normals'];a=m['axis']
    old=dict(a=torch.tensor([[.2]],dtype=torch.float64),b=torch.tensor([[np.sqrt(.25-.04)]],dtype=torch.float64),
             u=a,color=d['color'],metallic=d['metallic'])
    wi=torch.tensor([[[.1,.2,np.sqrt(.95)],[0.,0.,1.],[0.,0.,-1.]]],dtype=torch.float64)
    wo=n.clone();fd,fs=kernel_axes(torch,wi,wo,m);od,os=kernel(wi,wo,n,old)
    np.testing.assert_allclose(fd,od,atol=1e-15,rtol=1e-13);np.testing.assert_allclose(fs,os,atol=1e-15,rtol=1e-13)
    # Rotate both ellipse and world directions: physical covariance, no decoder use.
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    m['axis']=m['axis']@R.T;m['n']=m['n']@R.T
    _,rot=kernel_axes(torch,wi@R.T,wo@R.T,m);assert torch.allclose(fs,rot,atol=1e-15)


def test_integration_concentration_separates_sparse_and_uniform_contributions():
    from experiments.bell_highlight_pilot_e11d import concentration
    values=torch.ones((1,16,2,3),dtype=torch.float64);values[:,1:,1]=0
    ess,mass=concentration(values)
    np.testing.assert_allclose(ess,[[16,1]],rtol=1e-14);np.testing.assert_allclose(mass,[[1/16,1]],rtol=1e-14)


def test_sensitivity_detects_known_signals_and_grid_or_quantization_ambiguity():
    hi=np.zeros((4,8,3,3));hi[:,:,0]=.04;hi[:,:,1]=.01
    for case in [1,3,5,7]:hi[:,case,1]+=.01
    for case in [2,4,6]:hi[:,case,1]-=.005
    hi[:,:,2]=hi[:,:,0]+hi[:,:,1]
    result=sensitivity(hi,hi.copy());assert result['orientation']['resolved_fraction']==1
    assert result['conditional_three_column_jacobian']['full_numerical_rank_fraction']==0
    low=hi.copy();low[:,:,1]+=.02;low[:,:,2]=low[:,:,0]+low[:,:,1]
    assert sensitivity(hi,low)['orientation']['resolved_fraction']==0
    with pytest.raises(ValueError,match='eight'):sensitivity(hi[:,:7],hi[:,:7])
    assert sensitivity(hi*30,hi*30)['png_domain'] is False


def test_import_is_metadata_only_and_no_scientific_modules_enter_namespace():
    code='''
import builtins,sys
real=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','network','train','raytracing','_raytracing','scientific_variants'):
        raise RuntimeError('Forbidden scientific import: '+name)
    return real(name,*args,**kwargs)
builtins.__import__=guarded
import experiments.bell_highlight_e11d
import experiments.bell_highlight_pilot_e11d
'''
    r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_caps_and_python38_syntax_are_frozen():
    assert LIMITS['shadow_directions']==805306368
    assert LIMITS['brdf_point_directions']==871366656
    assert LIMITS['light_weighted_point_directions']==2614099968
    for path in ['experiments/bell_highlight_e11d.py','experiments/bell_highlight_pilot_e11d.py']:
        ast.parse(Path(path).read_text(),feature_version=(3,8))


@pytest.mark.parametrize('bad',['approved','commit','freeze_sha256','image_id','output_directory','limits'])
def test_pilot_requires_exact_authorization_before_runtime_or_render(tmp_path,monkeypatch,bad):
    import experiments.bell_highlight_pilot_e11d as m
    source=tmp_path/'X';source.write_text('immutable source')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_paths',lambda:['X'])
    f=dict(protocol=m.PROTOCOL,limits=m.LIMITS,lighting=m.candidates(),views=list(m.VIEWS),
           cases=list(m.CASES),files={'X':m.digest(source)},generation_sha256=m.GENERATION_SHA,image_id=m.identity()['image_id'])
    args=SimpleNamespace(freeze=tmp_path/'freeze',freeze_sha256='a'*64,authorization=tmp_path/'auth',
                         authorization_sha256='b'*64,commit='c'*40,image_id=m.identity()['image_id'],output=tmp_path/'new')
    auth=dict(protocol=m.PROTOCOL,approved=True,commit=args.commit,freeze_sha256=args.freeze_sha256,
              image_id=args.image_id,output_directory=str(args.output),limits=m.LIMITS)
    auth[bad]=None
    monkeypatch.setattr(m,'read_report',lambda path,sha:(f if path==args.freeze else auth,None))
    with pytest.raises(ValueError,match='authorization'):m.authorize(args)
    assert not args.output.exists()


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_terminal_reporting_waits_for_actual_cleanup(tmp_path,monkeypatch,cleanup_fails):
    import experiments.bell_highlight_pilot_e11d as m
    import benchmarks.verify_bell_e11 as v
    root=tmp_path/'root';(root/'geometry').mkdir(parents=True);(root/'geometry/generation.json').write_text('{}')
    monkeypatch.setattr(m,'GENERATION_SHA',m.digest(root/'geometry/generation.json'))
    monkeypatch.setattr(m,'authorize',lambda a:{'files':{}});monkeypatch.setattr(v,'verify',lambda *a:None)
    monkeypatch.setattr(m,'render',lambda *a:{'synthetic_work_returned':True})
    monkeypatch.setattr(m,'recommend',lambda rows:{'synthetic_fixture':True})
    def synchronize():
        if cleanup_fails:raise RuntimeError('synthetic cleanup failure')
    fake=SimpleNamespace(get_num_threads=lambda:1,set_num_threads=lambda n:None,
                         cuda=SimpleNamespace(synchronize=synchronize,empty_cache=lambda:None))
    monkeypatch.setitem(sys.modules,'torch',fake)
    args=SimpleNamespace(root=root,output=tmp_path/'outputs',commit='a'*40,freeze_sha256='b'*64,authorization_sha256='c'*64)
    if cleanup_fails:
        with pytest.raises(RuntimeError,match='cleanup'):m.execute(args)
    else:m.execute(args)
    report=json.loads((args.output/'pilot.json').read_text())
    assert report['cleanup_verified'] is (not cleanup_fails)
    assert report['status']==('failed' if cleanup_fails else 'complete')
    assert report['native_model_calls']==report['optimizer_returns']==0


def test_predeclared_selection_uses_signals_not_brightness_and_preserves_failures():
    import copy
    from experiments.bell_highlight_pilot_e11d import recommend
    rows=[]
    for camera in ('0','58','105'):
        for light,value in [('L0_v3',.1),('L1_three_oblique',.5),('L2_six_cardinal',.6)]:
            rows.append(dict(camera_id=camera,lighting=light,eligible=True,
                metrics={'highlight_coverage':.1},convergence={'relative_l2':.001},
                paired_convergence={'truth':{'relative_l2':.001}},
                sensitivity={**{k:{'resolved_fraction':value} for k in ('orientation','major','minor','isotropic')},'probes':80}))
    original=copy.deepcopy(rows);assert recommend(rows)['recommended']=='L2_six_cardinal';assert rows==original
    for r in rows:
        if r['lighting']=='L2_six_cardinal':r['eligible']=False
    assert recommend(rows)['recommended']=='L1_three_oblique'
    rows[0]['eligible']=False;assert recommend(rows)['recommended'] is None
    with pytest.raises(ValueError):recommend(rows[:-1])
