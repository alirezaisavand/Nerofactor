"""E14-only synthetic orchestration checks. Never render real inputs here."""
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from infrastructure.signatures import SignatureError
from experiments import bell_two_light_e14 as study
from experiments.bell_two_light_e14_analysis import analyze


def test_exact_work_ceiling_and_cold_metadata_import():
    assert study.LIMITS['brdf_point_directions']==220*23*(256**2+512**2)==1658060800
    assert study.LIMITS['environment_point_directions']==84*(256**2+512**2)
    assert all(study.LIMITS[k]==0 for k in ('new_primary_rays','new_shadow_queries','new_outgoing_queries','new_L2_brdf_queries'))
    script='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.bell_two_light_e14
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)


def test_native_grid_orchestration_uses_cached_geometry_and_no_first_light_kernel(tmp_path,monkeypatch):
    import torch
    import benchmarks.bell_reference_e11 as reference
    from experiments.bell_quadrature_e11e import midpoint_grid
    monkeypatch.setattr(study,'GRIDS',(4,8))
    monkeypatch.setattr(torch,'__version__','1.8.1+cu111');monkeypatch.setattr(torch.version,'cuda','11.1')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1);monkeypatch.setattr(torch.cuda,'get_device_name',lambda n:'A100 synthetic')
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda x:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self);monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    zeros=torch.zeros;monkeypatch.setattr(torch,'zeros',lambda *a,**kw:zeros(*a,**{k:v for k,v in kw.items() if k!='device'}))
    monkeypatch.setattr(reference,'environment',lambda wi,light:torch.ones_like(wi))
    monkeypatch.setattr(study,'baseline',lambda *a:None)
    binary=tmp_path/'binary.fixture';binary.write_bytes(b'CPU fixture only')
    monkeypatch.setattr(study,'identity',lambda:dict(binary_path=str(binary),binary_sha256=study.digest(binary)))
    labels=dict(position=np.zeros((84,3)),normals=np.tile([0.,0.,1.],(84,1)),T=np.tile([1.,0.,0.],(84,1)),B=np.tile([0.,1.,0.],(84,1)),
                theta_major=np.zeros(84),widths_major_minor=np.tile([.4,.2],(84,1)),color=np.full((84,3),.3),metallic=np.full((84,1),.2))
    valid=np.zeros((4,84),bool)
    for v,n in enumerate((84,36,40,60)):valid[v,:n]=True
    outgoing=dict(valid=valid,outgoing_directions=np.tile([0.,0.,1.],(4,84,1)),camera_ids=np.array([58,44,9,0]))
    cache={}
    for size in (4,8):
        nodes,weight=midpoint_grid(size)
        a=np.full((2,4,84,12,3,3),np.nan);a[:,valid]=.01;a[:,valid,:,2,:]=.02
        cache[size]=dict(nodes=nodes,weight=weight,visibility=np.full((84,size*size//8),255,np.uint8),L2=a)
    calls=[]
    def toy(torch,wi,wo,material):
        calls.append(len(wi)*wi.shape[1]);return torch.ones_like(wi)*.01,torch.ones_like(wi)*.02
    monkeypatch.setattr(study,'kernel_axes',toy)
    r=dict(counters=dict(brdf_attempts=0,brdf_returns=0,environment_attempts=0,environment_returns=0),completed_grids=[],lighting=list(study.LIGHTS))
    args=SimpleNamespace(output=tmp_path/'out');args.output.mkdir()
    results,resources=study.render(args,(labels,np.arange(84),cache,outgoing,[]),r,time.monotonic(),torch)
    assert sum(calls)==r['counters']['brdf_attempts']==r['counters']['brdf_returns']==220*23*80
    assert r['counters']['environment_attempts']==r['counters']['environment_returns']==84*80
    for size in (4,8):
        np.testing.assert_equal(results[size][:,0],cache[size]['L2'])
        assert np.isnan(results[size][0,1,~valid]).all()
        np.testing.assert_equal(results[size][0,1,valid,7],results[size][1,1,valid,7])
    assert resources['cuda_peak_allocated_bytes']==0


def test_invalid_valid_capture_rejected_before_information():
    valid=np.ones((4,84),bool);a=np.ones((2,2,4,84,12,3,3))*.01;a[...,2,:]=.02
    a[0,1,0,0,0,0,0]=np.nan
    with pytest.raises(SignatureError):analyze(a,a,valid)


def test_two_light_coverage_uses_matched_points_and_normalized_actual_counts(monkeypatch):
    import experiments.bell_two_light_e14_analysis as analysis
    calls=[]
    def toy_information(f,c):
        calls.append(len(f));return {'rgb_observations':len(f),'synthetic_only':True}
    monkeypatch.setattr(analysis,'information_row',toy_information)
    valid=np.zeros((4,84),bool)
    for v,n in enumerate((84,36,40,60)):valid[v,:n]=True
    a=np.full((2,2,4,84,12,3,3),np.nan)
    for arm in range(2):
        for light in range(2):
            for view in range(4):
                for point in np.flatnonzero(valid[view]):
                    sample=np.full((12,3,3),.02);sample[:,2]=.04
                    for plus,minus,col in [(1,2,0),(3,4,1),(5,6,2),(8,9,1),(10,11,2)]:
                        delta=.006 if light else [.001,.0002,.0001][col]
                        sample[plus,1,col]+=delta;sample[minus,1,col]-=delta
                    sample[:,2]=sample[:,0]+sample[:,1];a[arm,light,view,point]=sample
    result=analysis.analyze(a,a,valid)
    assert result['all_cases_converged'] and len(result['convergence'])==192
    assert len(result['per_probe_information'])==168 and len(calls)==504
    for row in result['per_probe_information']:
        V=row['valid_view_count'];assert row['information']['L2']['rgb_observations']==V
        assert row['information']['two_light']['rgb_observations']==2*V
    for row in result['response_coverage']:
        if row['factor'] in ('major','minor'):
            assert row['L2_resolved']==0 and row['two_light_resolved']==row['matched_points']
    assert result['old_decisions_replaced'] is False and result['full_dataset_admitted'] is False
