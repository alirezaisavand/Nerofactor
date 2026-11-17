"""HDR visualization regression: synthetic arrays only, no native imports."""
from types import SimpleNamespace
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from benchmarks.bell_e11 import linear_to_srgb
from experiments import donut_material_recovery_v1 as recovery


def test_HDR_preview_clips_only_display_and_preserves_raw_arrays(tmp_path):
    # Actual map shape and failure range, including distinct pixel/channel order.
    light=np.full((128,256,3),.01,np.float32)
    light[0,0]=[0.,.5,3.4031355]
    light[47,91]=[1.,2.,.0031308]
    light[-1,-1]=[3.,4.,5.]
    original=light.copy();light.flags.writeable=False
    raw=tmp_path/'illumination.npz'
    np.savez_compressed(raw,linear_RGB=light,known_linear_RGB=light.copy())
    raw_bytes=raw.read_bytes()
    with pytest.raises(ValueError,match='outside PNG domain'):
        linear_to_srgb(light)  # Reproduce the actual former call.
    png=tmp_path/'illumination-preview.png'
    info=recovery.illumination_preview(png,light)
    np.testing.assert_array_equal(np.asarray(Image.open(png)),linear_to_srgb(np.minimum(original,1)))
    assert info['channels_above_one']==5
    assert info['pixels_with_channel_above_one']==3
    assert info['display_only_upper_clip']==info['exposure']==1.0
    np.testing.assert_array_equal(light,original)
    assert raw.read_bytes()==raw_bytes
    with np.load(raw,allow_pickle=False) as z:
        assert z['linear_RGB'].dtype==original.dtype
        np.testing.assert_array_equal(z['linear_RGB'],original)


def test_in_domain_preview_is_identical_to_existing_encoder(tmp_path):
    light=np.array([[[0.,.0031308,.5],[1.,.25,.1]]],np.float64)
    png=tmp_path/'preview.png';info=recovery.illumination_preview(png,light)
    np.testing.assert_array_equal(np.asarray(Image.open(png)),linear_to_srgb(light))
    assert info['channels_above_one']==info['pixels_with_channel_above_one']==0


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,-.001])
def test_invalid_illumination_fails_without_preview(tmp_path,value):
    light=np.ones((2,2,3));light[0,0,1]=value
    png=tmp_path/'preview.png'
    with pytest.raises(ValueError,match='Invalid illumination preview'):
        recovery.illumination_preview(png,light)
    assert not png.exists()


@pytest.mark.parametrize('shape',[(0,2,3),(4,3),(2,2,4)])
def test_invalid_preview_shape_is_rejected(tmp_path,shape):
    with pytest.raises(ValueError,match='Invalid illumination preview'):
        recovery.illumination_preview(tmp_path/'preview.png',np.ones(shape))


def test_actual_evaluation_wires_explicit_preview_and_unmodified_HDR():
    source=ast.parse(Path(recovery.__file__).read_text())
    evaluate=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    save=next(n for n in ast.walk(evaluate) if isinstance(n,ast.Call) and
              isinstance(n.func,ast.Attribute) and n.func.attr=='savez_compressed' and
              any(k.arg=='linear_RGB' for k in n.keywords))
    assert next(k.value.id for k in save.keywords if k.arg=='linear_RGB')=='light'
    preview=next(n for n in ast.walk(evaluate) if isinstance(n,ast.Call) and
                 isinstance(n.func,ast.Name) and n.func.id=='illumination_preview')
    assert preview.args[1].id=='light'
    assert any(isinstance(n,ast.keyword) and n.arg=='preview' and
               isinstance(n.value,ast.Name) and n.value.id=='preview' for n in ast.walk(evaluate))


def test_scientific_settings_and_sources_remain_at_accepted_definition():
    from benchmarks import donut_recovery_v1 as d
    old=json.loads((d.ROOT/'docs/protocols/DONUT_KURT_REFERENCE_V2.freeze.json').read_text())
    assert d.TRAIN_LIMITS==old['training_limits']
    assert d.POLICY==old['policy'] and d.SHADER==old['shader']
    assert d.PANEL==old['panel'] and d.SPLITS==old['splits']
    changed={'AGENTS.md','experiments/donut_material_recovery_v1.py','experiments/run_donut_recovery_v1.sh'}
    for name,sha in old['sources'].items():
        if name not in changed:
            assert hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest()==sha,name


@pytest.mark.parametrize('generation,freeze,current,passes',[
    ('old',{},'old',True),
    ('old',{'reference_freeze_sha256':'old'},'new',True),
    ('wrong',{'reference_freeze_sha256':'old'},'new',False),
    ('old',{},'new',False),
])
def test_training_freeze_explicitly_binds_unchanged_reference(monkeypatch,tmp_path,generation,freeze,current,passes):
    class InputBoundary(Exception):pass
    monkeypatch.setattr(recovery,'runtime',lambda *a:(freeze,dict(researcher_admitted_dataset=True,benchmark_generation_sha256='dataset')))
    monkeypatch.setattr(recovery,'benchmark',lambda *a:({'freeze_sha256':generation},{}))
    monkeypatch.setattr(os,'statvfs',lambda *a:SimpleNamespace(f_flag=os.ST_RDONLY))
    def inputs(*a):raise InputBoundary()
    monkeypatch.setattr(recovery,'inputs',inputs)
    args=SimpleNamespace(phase='D1',benchmark=tmp_path,benchmark_sha256='dataset',freeze_sha256=current,oracle=tmp_path,cameras=tmp_path/'cameras')
    with pytest.raises(InputBoundary if passes else ValueError):recovery.train(args)
