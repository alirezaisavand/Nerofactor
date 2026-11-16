"""Only saved-evidence verifier fixtures; no BRDF or native execution."""
import subprocess
import sys

import numpy as np
import pytest

from experiments.analyze_bell_two_light_e14 import check_grid,check_work,replay_equal
from infrastructure.signatures import SignatureError


def receipt():
    return dict(status='complete',scientific_work_complete=True,cleanup_verified=True,completed_grids=[256,512],
        counters=dict(brdf_attempts=1658060800,brdf_returns=1658060800,environment_attempts=27525120,
            environment_returns=27525120,new_primary_rays=0,new_shadow_queries=0,new_outgoing_queries=0,
            new_L2_brdf_queries=0,model_calls=0,optimizer_returns=0))


@pytest.mark.parametrize('fault',['extra_query','incomplete_cleanup','missing_return'])
def test_receipt_fails_closed(fault):
    r=receipt();check_work(r)
    if fault=='extra_query':r['counters']['new_shadow_queries']=1
    elif fault=='incomplete_cleanup':r['cleanup_verified']=False
    else:r['counters']['brdf_returns']-=1
    with pytest.raises(SignatureError):check_work(r)


def test_saved_grid_preserves_L2_and_unavailable_correspondences():
    valid=np.ones((4,84),bool);valid[1,0]=False
    a=np.full((2,2,4,84,12,3,3),np.nan)
    for arm in range(2):
        for light in range(2):
            a[arm,light,valid]=.01;a[arm,light,valid,:,2]=.02
    z=dict(components=a,valid=valid,camera_ids=np.array([58,44,9,0],dtype=np.int64),
           pixel_indices=np.arange(84,dtype=np.int64),completed_probes=np.ones(84,bool))
    original=a[:,0].copy()
    check_grid(z,valid,z['pixel_indices'],z['camera_ids'],original)
    a[0,0,0,0,0,0,0]+=.0001
    with pytest.raises(SignatureError):check_grid(z,valid,z['pixel_indices'],z['camera_ids'],original)
    a[:,0]=original;a[0,1,1,0]=0
    with pytest.raises(SignatureError):check_grid(z,valid,z['pixel_indices'],z['camera_ids'],original)


def test_replay_roundoff_never_changes_scientific_decisions_or_counts():
    replay_equal(dict(passed=False,count=83,value=.001),dict(passed=False,count=83,value=.001+1e-14))
    for bad in [dict(passed=True,count=83,value=.001),dict(passed=False,count=84,value=.001),
                dict(passed=False,count=83,value=float('nan')),dict(passed=False,count=83,value=.002)]:
        with pytest.raises(SignatureError):replay_equal(dict(passed=False,count=83,value=.001),bad)


def test_offline_entry_import_has_no_model_or_tracing_dependencies():
    script='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.analyze_bell_two_light_e14
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
