"""CPU fixtures for original method dispatch, frame validity and committed match."""
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from benchmarks import donut_thesis_attribution_gate8_v1 as g
from experiments import train_donut_thesis_attribution_gate8_v1 as t
from experiments.analyze_donut_thesis_attribution_gate8_v1 import comparison
from infrastructure.signatures import SignatureError
from infrastructure.material_variants import CORE, identity


def test_original_dispatch_and_linear_capture_without_native_import():
    class Buffer:
        def cpu(self):
            return self

        def tolist(self):
            return [7]*32

        def copy_(self,value):
            self.identity=value

    class Original:
        def compute_radiance(self):
            return linear

        def shade_anisotropic_mixed(self):
            raw=self.compute_radiance()
            assert raw is linear
            return display,{'mx':width}

    linear=tuple(object() for _ in range(6))
    display,width=object(),object()
    for key in t.ORIGINAL_METHODS:
        setattr(Original,key,lambda self:'pinned')
    class Lighting(Original):
        def __init__(self,cfg,trace):
            self._a_variant_signature=Buffer()

    for key in t.ORIGINAL_METHODS:
        setattr(Lighting,key,lambda self:'replacement')
    torch=SimpleNamespace(as_tensor=lambda value,dtype:value,uint8=None)
    cls=t.shader_type(SimpleNamespace(MCShadingNetwork=Original,linear_to_srgb=None),
                      SimpleNamespace(NeROMaterialRenderer=None),
                      lambda *args:(Lighting,None),torch)
    instance=cls({},None)
    for key in t.ORIGINAL_METHODS:
        assert getattr(cls,key) is getattr(Original,key)
    assert cls.compute_radiance is not Original.compute_radiance
    colors,outputs=instance.shade_anisotropic_mixed()
    assert colors is display
    assert outputs==dict(mx=width,linear_rgb=linear[0],linear_diffuse=linear[1],linear_specular=linear[2])
    assert not instance._capturing and instance._captured is None
    with pytest.raises(SignatureError,match='one original radiance'):
        instance.compute_radiance()
    assert not any(module=='network' or module.startswith('network.') for module in sys.modules)


def test_original_capture_preserves_fp32_components():
    rgb=np.array([[[.1,.2,.3],[.2,.3,.4],[0,0,0]]],np.float32)
    rgb[:,2]=rgb[:,0]+rgb[:,1]
    t.valid_capture(rgb)
    rgb[:,2,0]+=.01
    with pytest.raises(SignatureError,match='non-additive'):
        t.valid_capture(rgb)


def test_initial_parity_excludes_only_the_behavior_tag():
    as_tensor=lambda a,dtype:np.asarray(a,dtype=dtype)

    class Tensor:
        def __init__(self,a):
            self.a=a

        def cpu(self):
            return self.a

    cfg=dict(g.m.SHADER)
    i=identity(CORE,dict(network='material',reg_mat=True,shader_cfg=cfg))
    tag=sha256(json.dumps(dict(implementation=i,shader=cfg),sort_keys=True,allow_nan=False).encode()).digest()
    state={'_a_variant_signature':Tensor(np.zeros(32,np.uint8)),'weight':np.arange(4,dtype=np.float32)}
    def hash_state(model):
        return sha256(b''.join(np.asarray(v.cpu() if isinstance(v,Tensor) else v).tobytes()
                               for _,v in sorted(model.state_dict().items()))).hexdigest()
    expected=dict(seed=g.m.POLICY['seed'],variant=CORE,
        state_sha256=hash_state(SimpleNamespace(state_dict=lambda:dict(state,
                     _a_variant_signature=np.frombuffer(tag,np.uint8)))))
    torch=SimpleNamespace(as_tensor=as_tensor,uint8=np.uint8,equal=np.array_equal)
    shader=SimpleNamespace(cfg=cfg,state_dict=lambda:state)
    assert t.initial_identity(shader,cfg,expected,hash_state,torch)['K1_projection_sha256']==expected['state_sha256']
    assert not state['_a_variant_signature'].cpu().any()
    state['weight'][0]=21
    with pytest.raises(SignatureError,match='parameters or buffers'):
        t.initial_identity(shader,cfg,expected,hash_state,torch)


def test_frame_exclusion_includes_raw_zero_and_original_sqrt_sign_degeneracy():
    n=np.tile([0.,0.,1.],(4,1))
    x=np.tile([1.,0.,0.],(4,1))
    y=np.tile([0.,1.,0.],(4,1))
    raw=np.array([[1,.5],[0,.5],[.5,.5],[1,.5]])
    # At cos(2 theta)=-1 and sin(2 theta)=0, the pinned formulas give
    # cos(theta)=sqrt(eps), sin(theta)=0, even with a nonzero raw vector.
    x[1]*=1e-3
    y[1]*=1e-3
    mx=np.array([[.4],[.4],[.4],[1.1]])
    my=np.full((4,1),.2)
    re=g.vector_validity(mx,my,raw,x,y,n,n)
    assert all(re['normal'])
    np.testing.assert_array_equal(re['axes'],[True,False,True,True])
    np.testing.assert_array_equal(re['raw_orientation'],[True,True,False,True])
    np.testing.assert_array_equal(re['widths'],[True,True,True,False])


def test_no_physical_frames_still_reports_all_rgb_and_scalar_rows():
    widths=np.array([[.4,.2]])
    rgb=np.ones((1,3,3))
    truth=dict(widths_major_minor=widths,color=np.ones((1,3)),metallic=np.ones((1,1)),
               linear_components=rgb)
    control=dict(valid=np.array([True]),widths=widths,kd=truth['color'],
                 metallic=truth['metallic'],linear=rgb)
    original=dict(control,valid=np.array([False]),valid_axes=np.array([False]),linear=rgb+.1)
    result=comparison({g.ARM:original,g.CONTROL:control},truth,np.array([True]))
    assert result['common_physical_count']==result['common_orientation_count']==0
    assert result['methods'][g.ARM]['physical'] is None
    assert result['methods'][g.ARM]['L2_appearance']['RGB']['pixels']==1
    assert result['methods'][g.ARM]['invalidity_counts_nonexclusive']=={'axes':1}


def test_compiled_overlay_preserves_optimizer_loop_and_distinct_authorization():
    old=t.parent.trainer_source(False)
    new=t.trainer_source()
    ast.parse(new,feature_version=(3,8))
    a,b='                with (args.output/\'training.jsonl\').open(\'x\') as log:',"                require(report['optimizer_returns']==10000"
    assert new[new.index(a):new.index(b)]==old[old.index(a):old.index(b)]
    assert new.count("acquisition.relight(shader,acquisition.unseen")==1
    f=g.read(g.ROOT/g.FREEZE)
    auth=g.authorization(f,'a'*40,g.digest(g.ROOT/g.FREEZE))
    assert auth['arm']==g.ARM and auth['maximum_execution_launches']==1
    assert auth['historical_checkpoint_loads_authorized'] is False
    assert g.RELIGHT=={'shader_calls':480,'shadow_rays':360972288}
    assert g.WORK=={'shader_calls':12288,'shadow_rays':9524158464,'component_reverse_sweeps':30}
    assert (g.ROOT/Path(g.FREEZE)).is_file()
