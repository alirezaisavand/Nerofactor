"""Focused E11C metadata/CPU fixtures only; no builds, CUDA, shaders or models."""
import ast,json,subprocess
from pathlib import Path
import numpy as np
import pytest
from infrastructure import ray_backend_e11c as r
from infrastructure.stage_sources import ROOT,digest


def good_arrays():
    v,f,o,d=r.cube();refs=r.reference(v,f,o,d);depth=np.array([x['depth'] if x['hit'] else 10 for x in refs],np.float32)
    position=o+depth[:,None]*d;raw=np.array([-np.array(x['normal']) if x['hit'] else np.zeros(3) for x in refs],np.float32)
    return o,d,position,raw,depth,refs


def test_cube_full_bvh_fixture_has_native_minimum_faces_and_hit_miss_controls():
    v,f,o,d=r.cube();assert f.shape==(12,3) and len(o)==len(d)==8
    refs=r.reference(v,f,o,d)
    assert [x['hit'] for x in refs]==[True,True,True,False,False,True,True,False]
    assert all(x['depth']==2 for x in [refs[0],refs[6]])
    rows=r.compare(*good_arrays());assert all(x['passed'] for x in rows)


@pytest.mark.parametrize('fault',['depth','position','normal','false_positive','false_negative'])
def test_actual_result_comparison_rejects_bad_intersections(fault):
    o,d,p,n,depth,refs=good_arrays()
    if fault=='depth':depth[0]+=.01
    if fault=='position':p[0,0]+=.01
    if fault=='normal':n[0]*=-1
    if fault=='false_positive':depth[3]=1
    if fault=='false_negative':depth[0]=10
    assert not all(x['passed'] for x in r.compare(o,d,p,n,depth,refs))


def test_nan_miss_cannot_be_recorded_as_valid_visibility():
    o,d,p,n,depth,refs=good_arrays();depth[3]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):r.compare(o,d,p,n,depth,refs)


def test_authorization_binds_source_freeze_root_and_limits(tmp_path):
    path=tmp_path/'authorization';root=tmp_path/'batch'
    a=dict(protocol=r.PROTOCOL,approved=True,commit='a'*40,freeze_sha256='b'*64,base_image=r.BASE,work_limits=r.LIMITS,host_evidence_directory=str(root))
    path.write_text(json.dumps(a));assert r.authorize(path,digest(path),'b'*64,'a'*40,root)==a
    for key in ('approved','commit','freeze_sha256','base_image','work_limits','host_evidence_directory'):
        b=dict(a);b[key]=None;path.write_text(json.dumps(b))
        with pytest.raises(ValueError):r.authorize(path,digest(path),'b'*64,'a'*40,root)


def test_backend_selection_is_explicit_and_rejects_fallback(tmp_path,monkeypatch):
    original=tmp_path/'original.so';original.write_bytes(b'original synthetic bytes only')
    prefix=tmp_path/'sidecar';(prefix/'corrected').mkdir(parents=True);new=prefix/'corrected/_raytracing.so';new.write_bytes(b'corrected synthetic bytes only')
    monkeypatch.setattr(r,'ORIGINAL_BINARY',digest(original))
    build=dict(original_binary=r.file_record(original),corrected_binary=r.file_record(new))
    assert r.selected_binary(build,'original',prefix)==original and r.selected_binary(build,'predicate_fp64',prefix)==new
    new.write_bytes(original.read_bytes())
    with pytest.raises(ValueError):r.selected_binary(build,'predicate_fp64',prefix)
    with pytest.raises(ValueError):r.selected_binary(build,'implicit',prefix)


def test_cpu_entry_imports_no_native_scientific_modules_and_python38_grammar():
    code='''
import builtins,importlib.util
real=builtins.__import__
def guard(name,*a,**kw):
 assert name.split('.')[0] not in ('torch','network','_raytracing','train','dataset'),name
 return real(name,*a,**kw)
builtins.__import__=guard
import run_ray_backend_e11c,prepare_ray_backend_e11c
'''
    p=subprocess.run(['python3','-B','-c',code],cwd=ROOT,text=True,capture_output=True);assert p.returncode==0,p.stderr
    for name in ('run_ray_backend_e11c.py','prepare_ray_backend_e11c.py','infrastructure/ray_backend_e11c.py'):
        ast.parse((ROOT/name).read_text(),feature_version=8)


def test_build_and_runtime_boundaries_are_static_and_preserve_original_binary():
    docker=(ROOT/'docker/paper/e11c/Dockerfile').read_text()
    assert 'COPY e11c/ /opt/e11c/' in docker and not any(x.startswith('RUN ') for x in docker.splitlines())
    source=(ROOT/'run_ray_backend_e11c.py').read_text()
    assert "'--build-lib','/outputs/corrected'" in source and 'pip install' not in source
    assert digest(ROOT/r.PATCH)==r.PATCH_SHA
    shell=ROOT/'docker/paper/e11c/run.sh';p=subprocess.run(['bash','-n',str(shell)],text=True,capture_output=True);assert p.returncode==0,p.stderr
    s=shell.read_text();assert 'timeout --foreground --signal=TERM --kill-after=30s 1800s' in s
    assert '--env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=' in s
    assert 'run_stage original' in s and 'run_stage corrected' in s and 'new[:len(old)]==old' in s
    assert r.LIMITS['max_native_ray_queries']==2*(44+32768*(5+20))
