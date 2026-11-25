import ast
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import donut_nero_gate7_v1 as g
from infrastructure.signatures import SignatureError


def test_official_source_and_acquisition_remain_pinned():
    f=g.freeze(g.digest(g.ROOT/g.FREEZE))
    p=g.paths(f)
    views,relight,_=g.admit(f,p)
    assert len(f['upstream']['files'])==44 and f['upstream']['commit']==g.UPSTREAM_COMMIT
    assert not any(n in sys.modules for n in ('network.field','network.renderer','raytracing.raytracer'))
    fit=g.m.old.fit_pool(views)
    assert set(fit)==set(g.m.FIT_KEYS) and len(fit['position'])==15191
    assert g.m.bell.dense.index_hash(len(fit['position']))==f['support']['photometric_index_sha256']
    assert sum(v['masks']['common'].sum() for v in relight.values())==7306
    assert sum(len(v['view']['pixel_indices']) for v in relight.values())==7344
    for v in relight.values():
        np.testing.assert_array_equal(v['reference'][~v['masks']['expanded_foreground']],0)


def test_raw_transfer_capture_matches_official_shader_sites():
    code=(g.UPSTREAM_HOST/'network/field.py').read_text()
    module=ast.parse(code)
    shader=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork')
    shade=next(n for n in shader.body if isinstance(n,ast.FunctionDef) and n.name=='shade_mixed')
    calls=sorted((n for n in ast.walk(shade) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                  and n.func.id=='linear_to_srgb'),key=lambda n:n.lineno)
    assert len(calls)==6
    assert [calls[i].args[0].id for i in (0,3,4)]==['colors','diffuse_colors','specular_colors']
    converter=lambda x:x+13
    field=SimpleNamespace(linear_to_srgb=converter)
    events=[]
    inputs=np.arange(6)
    with g.observe_linear(field,events):
        assert [field.linear_to_srgb(x) for x in inputs]==list(inputs+13)
    assert field.linear_to_srgb is converter and g.linear_components(events)==(3,4,0)


def test_capture_restores_original_and_extra_transfer_fails():
    converter=lambda x:x
    field=SimpleNamespace(linear_to_srgb=converter)
    with pytest.raises(SignatureError,match='inventory'):
        with g.observe_linear(field,[]):
            g.linear_components([1,2,3])
    assert field.linear_to_srgb is converter


def test_work_keeps_same_training_slots_and_deterministic_evaluation_budget():
    limits=g.LIMITS
    assert limits['optimizer_returns']==limits['backwards']==10000
    assert limits['reference_renders']==limits['primary_rays']==limits['checkpoint_loads']==0
    assert limits['shadow_slots']==7910155776 and limits['shader_calls']==10062
    assert g.EVAL_POINTS==29841 and g.EVAL_CALLS==62


def test_source_closure_refuses_mixed_tree(tmp_path):
    f=g.freeze(g.digest(g.ROOT/g.FREEZE))
    with pytest.raises(SignatureError,match='source closure'):
        g.source_check(tmp_path,f)
