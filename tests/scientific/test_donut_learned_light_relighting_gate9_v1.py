"""Focused Gate 9 material and role-adapter checks. No ray or model calls."""
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmarks import donut_learned_light_relighting_gate9_v1 as g
from infrastructure.signatures import SignatureError


def test_illumination_scale_is_reported_without_erasing_color_error():
    t=np.array([[[.2,.5,.1]],[[.3,.1,.2]]])
    w=np.array([[1.],[2.]])
    p=t*2
    a=g.illumination_score(p,t,w)
    assert a['raw_relative_L2']==pytest.approx(1)
    assert a['diagnostic_global_gain']==pytest.approx(.5)
    assert a['gain_aligned_relative_L2']==pytest.approx(0)
    p=p.copy()
    p[...,1]*=2
    assert g.illumination_score(p,t,w)['gain_aligned_relative_L2']>0


def test_three_role_refinement_reuses_exact_gate6_policy():
    ids=np.array([4,8])
    calls=[]
    def fetch(rows,grid,prefix):
        calls.append((prefix,grid))
        x=np.full((3,len(rows),3,3),.2)
        if prefix=='base' and grid==g.g6.BASE[0]:
            x[1,0,0,0]+=.004
        x[:,:,2]=x[:,:,0]+x[:,:,1]
        return dict(pixels=ids[rows],grid=np.array(grid),components=x,visibility=np.ones(len(rows),int))
    a,_,res,selected=g.refine(ids,fetch,0)
    assert a.shape==(2,3,2,3,3) and selected==[4]
    assert list(res[1])==[2048,512]
    assert calls==[('base',256),('base',512),('pixel-4',1024),('pixel-4',2048)]
    assert g.g6.ROLES[0]=='oracle' and len(g.g6.ROLES)==4


def test_missing_or_invalid_physical_frame_fails_closed():
    dots=np.zeros((g.SIZE,g.SIZE),bool)
    dots[0,0]=True
    xyz=np.zeros((g.SIZE,g.SIZE,3),np.float32)
    q=np.zeros((g.SIZE,g.SIZE,3,3))
    q[0,0]=np.diag([.09,.04,0])
    v=dict(pixel_indices=np.array([0]),position=np.zeros((1,3)),normals=np.array([[0.,0.,1.]]))
    m=dict(hit_mask=dots,position=xyz,normal=xyz.copy(),Q_world=q,
           Q_tensor_direct=q.copy(),kd=xyz,metallic=xyz[...,:1])
    m['normal'][0,0,2]=1
    fields,err=g.material_fields(m,v,True)
    assert err<=g.g6.MATERIAL_ATOL
    np.testing.assert_allclose(fields['widths_major_minor'],[[.3,.2]])
    m['Q_tensor_direct'][0,0,2,2]=.01
    with pytest.raises(SignatureError,match='disagrees'):
        g.material_fields(m,v,True)
    m['Q_world'][0,0,2,2]=.01
    with pytest.raises(SignatureError,match='Invalid world Q'):
        g.material_fields(m,v,True)


def test_frozen_no_oracle_image_or_checkpoint_mount_and_zero_work():
    p=g.ROOT/g.FREEZE
    f=g.freeze(g.digest(p))
    native=g.paths(f,True)
    assert not any(k.startswith(('reference_','illumination_')) for k in native)
    assert not any(a.suffix=='.pth' for a in native.values())
    assert not Path(f['host_output']).exists()
    auth=g.authorization(f,'1'*40,g.digest(p))
    assert all(auth['limits'][k]==0 for k in ('primary_rays','native_shader_calls','checkpoint_loads',
                                             'new_reference_integrations','optimizer_updates'))
    assert not auth['checkpoint_loading_authorized'] and not auth['training_authorized']
    assert g.expected_work(g.g6.MAX_TOTAL)==g.LIMITS
    assert not any(s=='network' or s.startswith('network.') for s in sys.modules)
