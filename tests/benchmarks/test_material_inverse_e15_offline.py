"""Small E15 evidence-analysis fixtures only; no model/BRDF/optimizer calls."""
import subprocess
import sys
import numpy as np
import pytest

from experiments.analyze_bell_material_inverse_e15 import independent_errors,independent_objective,trajectory,numeric
from benchmarks.material_inverse_e15 import initial_parameters,metrics


def test_independent_tensor_and_periodic_orientation_reference():
    truth=initial_parameters(2)[:,:1];p=truth.copy();p[...,0]+=np.pi;p[...,1]+=.01
    labels=dict(T=np.tile([1.,0.,0.],(2,1)),B=np.tile([0.,1.,0.],(2,1)))
    a=independent_errors(p,truth,labels);b=metrics(p,truth,(labels['T'],labels['B']))
    for k in a:
        if k=='orientation_degrees':assert (a[k]<1e-12).all()
        else:np.testing.assert_allclose(a[k],b[k],atol=1e-14,rtol=1e-12)


def test_training_objective_excludes_held_views_and_unselected_light():
    target=np.full((168,2,3),.1);y=np.repeat(target[:,None],3,axis=1)
    y[84:]+=1;y[:84,:,1]+=1  # Held views and the unobserved light must not influence L2-only fitting.
    from benchmarks.material_information_e14 import deterministic_codes
    d=dict(train_views=np.arange(168)<84,observation_point_indices=np.tile(np.arange(84),2),
           predicted_RGB=y,target_RGB=target,target_codes=deterministic_codes(target))
    assert (independent_objective(d,dict(lights=1,observation='linear'))==0).all()
    assert (independent_objective(d,dict(lights=1,observation='codes8'))==0).all()
    assert (independent_objective(d,dict(lights=2,observation='linear'))>.1).all()


def test_stationary_trajectory_exact_counters_and_tamper_rejection():
    p=initial_parameters(84);zero=np.zeros((84,3),np.int64);damping=np.full((84,3),.001)
    panel=dict(lights=1,ratio='free',material='original',observation='linear')
    row=dict(outer=0,parameters=p.tolist(),accepted_before=zero.tolist(),gradient_infinity=zero.tolist(),
        trials=[],parameters_after=p.tolist(),accepted_after=zero.tolist())
    t=dict(panel=panel,initial=p.tolist(),history=[row],termination=np.full((84,3),'fitting_stationary').tolist())
    opt=dict(parameters=p,accepted=zero.copy(),trials=zero.copy(),damping=damping)
    f,j,_=trajectory(t,opt,panel);assert f==j==1
    opt['accepted'][0,0]=1
    with pytest.raises(ValueError):trajectory(t,opt,panel)


def test_object_payload_rejected_without_pickle(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,payload=np.array([dict(untrusted=True)],dtype=object))
    with pytest.raises(ValueError):numeric(p)


def test_analysis_import_scientific_modules_blocked():
    script='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.analyze_bell_material_inverse_e15
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
