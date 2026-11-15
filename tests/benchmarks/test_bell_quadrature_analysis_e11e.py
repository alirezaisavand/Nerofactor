"""New E11E saved-artifact controls only; no BRDF or GPU execution."""
import hashlib
import subprocess

import numpy as np
import pytest

from experiments.analyze_bell_quadrature_e11e import verify_nodes,verify_grid,contrast_stability
from experiments.bell_quadrature_e11e import midpoint_grid


def test_saved_node_hash_domain_order_and_weight_controls():
    d,w=midpoint_grid(4)
    info={'node_float64_little_endian_sha256':hashlib.sha256(d.astype('<f8').tobytes()).hexdigest(),
          'solid_angle_weight':w,'pdf':1/(2*np.pi)}
    data={'local_directions':d,'solid_angle_weight':np.array(w),'proposal_pdf':np.array(1/(2*np.pi))}
    verify_nodes(4,data,info)
    bad=dict(data);bad['solid_angle_weight']=np.array(w/2)
    with pytest.raises(ValueError,match='weight'):verify_nodes(4,bad,info)
    bad=dict(data);bad['local_directions']=d[::-1].copy()
    with pytest.raises(ValueError,match='bytes'):verify_nodes(4,bad,info)


def data_fixture():
    a=np.zeros((84,8,3,3),dtype=np.float64);a[:,:,0]=.04;a[:,:,1]=.01;a[:,:,2]=a[:,:,0]+a[:,:,1]
    data={'components':a,'pixel_indices':np.arange(84,dtype=np.int64),
          'visibility_packed':np.full((84,2),255,dtype=np.uint8),'visible_counts':np.full(84,16,dtype=np.int64)}
    return data


def test_visibility_bit_count_and_diffuse_counterfactual_control():
    d=data_fixture();r=verify_grid(4,d,np.arange(84),{'visible_count':84*16});assert r['visible_fraction']==1
    d['visibility_packed'][0,0]=0
    with pytest.raises(ValueError,match='Visibility'):verify_grid(4,d,np.arange(84),{'visible_count':84*16})
    d=data_fixture();d['components'][0,1,0]+=.001;d['components'][0,1,2]=d['components'][0,1,0]+d['components'][0,1,1]
    with pytest.raises(ValueError,match='Diffuse'):verify_grid(4,d,np.arange(84),{'visible_count':84*16})


def test_equal_counts_do_not_hide_different_resolved_probe_membership():
    mis=data_fixture()['components'];fine=mis.copy()
    mis[0,1,1]+=.01;mis[0,1,2]=mis[0,1,0]+mis[0,1,1]
    fine[1,1,1]+=.01;fine[1,1,2]=fine[1,1,0]+fine[1,1,1]
    rows,q,m=contrast_stability(fine.copy(),fine,mis,mis.copy())
    r=next(x for x in rows if x['factor']=='orientation')
    assert r['MIS_resolved']==r['quadrature_resolved']==1 and r['resolved_membership_changes']==2
    assert r['maximum_quad512_vs_quad256_response_change']==0


def test_analysis_metadata_does_not_import_torch_or_native_science():
    code='''
import builtins
real=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','network','train','raytracing','_raytracing','scientific_variants'):
        raise RuntimeError('Forbidden scientific import '+name)
    return real(name,*a,**kw)
builtins.__import__=guard
import experiments.analyze_bell_quadrature_e11e
'''
    r=subprocess.run(['python3','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
