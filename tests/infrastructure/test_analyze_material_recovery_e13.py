"""File-only analysis fixtures: no scientific model, native renderer or optimizer."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.analyze_bell_material_recovery_e13 import arrays,compare,rgb_weight,physical_summary,physical_reference,execute,png
from infrastructure.signatures import SignatureError


def test_original_loss_weight_schedule_and_charbonnier_floor():
    cfg=dict(render_loss_weight_begin=1,render_loss_weight_end=5,render_weight_decay_begin=0,render_weight_decay_mid=5000,render_weight_decay_end=10000)
    assert [rgb_weight(s,cfg) for s in (0,5000,10000,10001)]==[1,5,1,1]
    assert rgb_weight(1999,cfg)==pytest.approx(2.3807710004578606,rel=1e-14)
    raw=np.sqrt(.001);assert raw==pytest.approx(.03162277660168379)
    assert raw*rgb_weight(1999,cfg)/rgb_weight(1999,cfg)==raw


def test_exact_decisions_and_float_verification_are_distinct():
    compare({'count':3,'passed':True,'x':1.0},{'count':3,'passed':True,'x':1+1e-15})
    for changed in ({'count':4,'passed':True,'x':1.0},{'count':3,'passed':False,'x':1.0},{'count':3,'passed':True,'x':1.01}):
        with pytest.raises(SignatureError):compare({'count':3,'passed':True,'x':1.0},changed)


def test_numeric_readers_reject_pickle_and_nonfinite(tmp_path):
    p=tmp_path/'a.npz';np.savez(p,x=np.array([1.,2.]));assert arrays(p)['x'].tolist()==[1.,2.]
    np.savez(p,x=np.array([{'pickle':'not allowed'}],object))
    with pytest.raises(ValueError):arrays(p)
    np.savez(p,x=np.array([np.inf]))
    with pytest.raises(SignatureError):arrays(p)


def test_png_decoder_preserves_all_uint8_values(tmp_path):
    from PIL import Image
    a=np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3)
    p=tmp_path/'known.png';Image.fromarray(a).save(p)
    assert np.array_equal(png(p),a)


def test_independent_material_reference_and_frozen_statistics():
    n=2;T=np.tile([1.,0.,0.],(n,1));B=np.tile([0.,1.,0.],(n,1));N=np.tile([0.,0.,1.],(n,1))
    g=dict(T=T,B=B,normals=N,theta_major=np.zeros(n),Q_world=np.tile(np.diag([.36,.09,0]),(n,1,1)),Q_local=np.tile(np.diag([.36,.09]),(n,1,1)),widths_major_minor=np.tile([.6,.3],(n,1)),color=np.full((n,3),.5),metallic=np.full((n,1),.2),pixel_indices=np.array([0,1]),orientation_defined=np.ones(n,bool),orientation_eligible=np.ones(n,bool))
    p=dict(mx=np.full((n,1),.6),my=np.full((n,1),.3),rotation=np.tile([1.,0.],(n,1)),T=T,B=B,actual_T=T,actual_B=B,color=g['color'],metallic=g['metallic'],pixel_indices=g['pixel_indices'],invalid_material=np.zeros(n,bool),invalid_frame=np.zeros(n,bool),orientation_degrees=np.zeros(n),orientation_eligible=np.ones(n,bool),orientation_prediction_defined=np.ones(n,bool),Q_world=g['Q_world'],Q_local=g['Q_local'])
    for k in ['Q_world_absolute','Q_world_relative','Q_local_absolute','Q_local_relative','major_width_absolute','minor_width_absolute','albedo_max_absolute','metallic_absolute']:p[k]=np.zeros(n)
    error=physical_reference(p,g);assert max(error.values())==0
    result=physical_summary([p]);assert result['observations']==2
    assert all(v['fraction']==1 for v in result['descriptive_pass_fractions'].values())
    broken=deepcopy(p);broken['invalid_frame'][0]=True
    with pytest.raises(SignatureError):physical_reference(broken,g)


def test_analysis_refuses_reuse_without_reading_native_evidence(tmp_path):
    batch=tmp_path/'batch';batch.mkdir();out=tmp_path/'analysis';out.mkdir()
    with pytest.raises(SignatureError,match='Fresh separate'):execute(batch,out)
