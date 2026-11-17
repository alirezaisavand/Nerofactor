"""Synthetic continuation gates and immutable reuse; no real reference queries."""
import ast
import builtins
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import generate_donut_reference_v2 as g
from benchmarks.donut_reference_v2 import validation_indices,GRAZING_PIXELS,LIMITS
from benchmarks.donut_recovery_v1 import ROOT


def fixture():
    count=15;cosine=np.r_[np.full(5,.01),np.linspace(.3,1,10)]
    diffuse=np.full((count,3),.02);spec=np.full((count,3),.03)
    return dict(pixel_indices=np.array(GRAZING_PIXELS+list(range(10))),shader_valid=np.ones(count,bool),
        outgoing=np.stack([np.sqrt(1-cosine*cosine),np.zeros(count),cosine],-1),
        normals=np.tile([0.,0,1.],(count,1)),axis=np.tile([1.,0,0],(count,1)),
        widths_major_minor=np.tile([.3,.2],(count,1)),color=np.full((count,3),.2),metallic=np.full((count,1),.5),
        linear_components=np.stack([diffuse,spec,diffuse+spec],1),coarse_components=np.stack([diffuse,spec,diffuse+spec],1),
        reflectance=np.full((count,3),.8),coarse_reflectance=np.full((count,3),.8))


def test_camera0_reuse_preserves_numeric_arrays_without_touching_old_evidence():
    data=fixture();before={k:v.copy() for k,v in data.items()}
    low=np.full((15,3),.7);high=np.full((15,3),.7001)
    result,summary=g.completed_view_arrays(data,low,high)
    for key,value in before.items():
        np.testing.assert_array_equal(data[key],value)
        if key not in ('reflectance','coarse_reflectance'):np.testing.assert_array_equal(result[key],value)
    np.testing.assert_array_equal(result['hemisphere_reflectance_diagnostic'],before['reflectance'])
    np.testing.assert_array_equal(result['reflectance'],high)
    assert summary['pixels']==15


def test_validation_points_are_fixed_grazing_and_geometrically_selected_ordinary():
    d=fixture();a,b=validation_indices(d)
    np.testing.assert_array_equal(d['pixel_indices'][a],GRAZING_PIXELS)
    assert len(b)==8 and not np.intersect1d(a,b).size
    assert np.all((d['outgoing'][b]*d['normals'][b]).sum(-1)>=.25)
    d['linear_components']*=100  # Radiance must not affect the selection.
    aa,bb=validation_indices(d);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)


@pytest.mark.parametrize('stage',['grazing','ordinary'])
def test_failed_validation_never_imports_CUDA_or_generates_remaining_views(tmp_path,monkeypatch,stage):
    data=fixture();root=tmp_path/'previous';root.mkdir()
    args=SimpleNamespace(previous=root,output=tmp_path/'new',oracle=tmp_path,cameras=tmp_path/'camera.json',
        commit='synthetic',freeze_sha256='synthetic',image_id='synthetic')
    monkeypatch.setattr(g,'runtime',lambda *x:({},{}))
    monkeypatch.setattr(os,'statvfs',lambda *x:SimpleNamespace(f_flag=os.ST_RDONLY))
    monkeypatch.setattr(g,'previous',lambda *x:(data,{'original_artifacts':{}}))
    monkeypatch.setattr(g,'inputs',lambda *x:({}, {}, {}))
    def fake_integrate(wo,width,color,metal,order,**kw):
        # Both coarse/fine1.0 pass the grazing gate in the ordinary-failure
        # fixture; old independent incident0.8 then rejects the discrepancy.
        value=1.01 if stage=='grazing' and order==128 else 1.
        return dict(total=np.full((len(wo),3),value),diffuse=np.zeros((len(wo),3)))
    monkeypatch.setattr(g,'integrate',fake_integrate)
    actual_import=builtins.__import__;native=[]
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('torch','network','raytracing','bpy'):
            native.append(name);raise AssertionError('Native import before admission')
        return actual_import(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='validation failed'):g.generate(args)
    report=json.loads((args.output/'generation.json').read_text())
    assert report['status']=='failed' and not native
    assert report['counters']['primary_rays']==report['counters']['shadow_directions']==0
    assert not list(args.output.glob('view-*.npz'))
    assert (args.output/'grazing-validation.npz').exists()
    assert (args.output/'ordinary-validation.npz').exists()==(stage=='ordinary')


def test_D1_D2_scientific_training_body_and_policies_remain_exactly_accepted():
    commit='3a4288032b9f81f26e8f223318894020dc5d8699'
    def previous(path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT,text=True)
    def function(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    path='experiments/donut_material_recovery_v1.py'
    assert ast.dump(function(previous(path),'train'))==ast.dump(function((ROOT/path).read_text(),'train'))
    path='benchmarks/donut_recovery_v1.py'
    def policies(source):
        return {n.targets[0].id:ast.dump(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and
                isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SHADER','POLICY','TRAIN_LIMITS')}
    assert policies(previous(path))==policies((ROOT/path).read_text())
    assert LIMITS['primary_rays']==5*128**2 and LIMITS['CPU_primary_crosschecks']==40
