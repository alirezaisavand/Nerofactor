"""New file-only E16 metric/receipt fixtures; no scientific model imports."""
import subprocess
import sys
import json
import numpy as np
import pytest
from experiments.analyze_bell_neural_material_e16 import physical,radiometry,state_hash,numeric,rank_correlation
from benchmarks.neural_observation_e16 import world_Q


def test_physical_metrics_preserve_head_swap_and_periodic_equivalence():
    T=np.array([[1.,0.,0.]]);B=np.array([[0.,1.,0.]])
    truth=np.array([[.3,np.log(.3),np.log(2.)]])
    a=np.cos(truth[:,:1])*T+np.sin(truth[:,:1])*B;b=-np.sin(truth[:,:1])*T+np.cos(truth[:,:1])*B
    mx=np.exp(truth[:,1]+truth[:,2]/2);my=np.exp(truth[:,1]-truth[:,2]/2);Q=world_Q(truth,T,B)
    x=physical(Q,mx,my,a,b,truth,T,B);y=physical(Q,my,mx,b,-a,truth,T,B)
    for key in x:np.testing.assert_allclose(x[key],y[key],atol=1e-12)
    assert x['Q_world_relative'][0]==0 and x['orientation_degrees'][0]<1e-5 and x['orientation_eligible'][0]


def test_saved_draw_variance_and_mean_error_are_distinct():
    target=np.full((2,2,3),.2);draws=np.tile(target,(8,1,1,1));draws[:4]+=.01;draws[4:]-=.01
    result=radiometry(draws,target,target,target,np.array([True,True]))
    assert result['native_mean_MSE']<1e-30 and result['native_single_draw_MSE']==pytest.approx(.0001)
    assert result['native_empirical_variance']==pytest.approx(.0001) and result['code_channels_exact']==12
    with pytest.raises(ValueError):radiometry(draws[:,:,:1],target,target,target,np.ones(2,bool))


def test_checkpoint_identity_and_primitive_numeric_loading(tmp_path):
    a={'b':np.array([1.,2.],np.float32),'a':np.array(2,np.int64)}
    assert state_hash(a)==state_hash(dict(reversed(list(a.items()))))
    b={**a,'b':np.array([1.,3.],np.float32)};assert state_hash(a)!=state_hash(b)
    np.savez(tmp_path/'good.npz',**a);assert numeric(tmp_path/'good.npz')['b'].dtype==np.float32
    np.savez(tmp_path/'bad.npz',payload=np.array([{}],object))
    with pytest.raises(ValueError):numeric(tmp_path/'bad.npz')


def test_rank_diagnostic_handles_ties_and_constant_vectors():
    assert rank_correlation([1,1,2],[3,3,4])==pytest.approx(1)
    assert rank_correlation([1,2,3],[3,2,1])==pytest.approx(-1)
    assert rank_correlation([1,1,1],[1,2,3]) is None


def test_analysis_import_and_help_block_native_scientific_modules():
    code='''import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','network','train','dataset','raytracing','_raytracing','open3d','nvdiffrast'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=['analyze_bell_neural_material_e16.py','--help']
runpy.run_path('experiments/analyze_bell_neural_material_e16.py',run_name='__main__')
'''
    subprocess.run(['python3','-B','-c',code],check=True,stdout=subprocess.PIPE)


def test_png_figure_export_uses_binary_writer(tmp_path):
    from experiments.analyze_bell_neural_material_e16 import figures,study
    detail={}
    for material in study.FIELDS:
        for stage in ('teacher','A-initial','B-post_distillation','A-photo1000','B-photo1000','A-compute_endpoint'):
            detail[material+'-'+stage]=dict(errors=dict(Q_world_relative=np.array([.02,.1,.2])),
                radiometry=dict(held=dict(native_mean_MSE=1e-5,native_empirical_variance=1e-6)))
        for arm in ('A','B'):
            (tmp_path/(material+'-'+arm+'-training.json')).write_text(json.dumps([dict(phase='photo',loss=1e-5) for _ in range(30)]))
    figures(tmp_path,[],detail,tmp_path)
    for name in ('physical_recovery.png','paired_Q.png','photometry_and_variance.png'):
        assert (tmp_path/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
