"""Metadata/stand-in tests for admitted common tracing; never loads a real .so."""
import ast,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from infrastructure import verified_ray_backend as b
from infrastructure.stage_sources import ROOT,PINS,digest


def test_native_admission_and_generation_use_the_verified_image():
    proof=json.loads(b.ADMISSION.read_text());assert proof['status']=='PROCEED' and proof['corrected_reference_matches']==44
    assert digest(b.ADMISSION)==b.ADMISSION_SHA and b.identity()['binary_sha256']==proof['corrected_binary']['sha256']
    from benchmarks.bell_e11 import IMAGE
    assert IMAGE==b.CORRECTED_IMAGE and b.image_for('original')==b.BASE


@pytest.mark.parametrize('variant,filename',[('A_original','original'),('A_estimator_v1','estimator'),('A_corrected_core_v1','core')])
def test_all_three_material_oracle_plans_share_backend_without_numeric_changes(tmp_path,variant,filename):
    from infrastructure.stage_execution import plan
    import yaml
    data=tmp_path/'datasets';(data/'GlossySynthetic/bell').mkdir(parents=True)
    geo=tmp_path/'geometry';geo.mkdir();(geo/'mesh.ply').write_bytes(b'path-only fixture')
    thesis=tmp_path/'thesis';thesis.mkdir()
    p=plan('material',PINS['material'],'C',data,thesis,tmp_path/'outputs',
           ROOT/('configs/experiments/material_a_bell_'+filename+'_known_v1.yaml'),ROOT,{'geometry':geo},material_variant=variant,oracle_benchmark=True)
    assert p['ray_backend']=='predicate_fp64'
    original=yaml.safe_load((ROOT/('configs/material_corrected/bell_'+filename+'_known_v1.yaml')).read_text())
    expected=dict(original);expected['train_dataset_type']='dummy'
    if variant!='A_original':expected['material_variant']=variant
    assert p['configuration']==expected
    assert not (tmp_path/'outputs').exists()


def test_selection_rejects_bad_image_missing_binary_and_preloaded_namespace(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'_raytracing',raising=False)
    with pytest.raises(ValueError,match='image mismatch'):b.load('predicate_fp64',b.BASE)
    monkeypatch.setitem(b.PATHS,'predicate_fp64',tmp_path/'missing.so')
    with pytest.raises(ValueError,match='no fallback'):b.load('predicate_fp64',b.CORRECTED_IMAGE)
    monkeypatch.setitem(sys.modules,'_raytracing',SimpleNamespace())
    with pytest.raises(ValueError,match='already imported'):b.load('predicate_fp64',b.CORRECTED_IMAGE)


def test_loader_checks_hash_before_import_and_uses_exact_sidecar(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'_raytracing',raising=False)
    path=tmp_path/'synthetic.so';path.write_bytes(b'not a real binary; never executed')
    monkeypatch.setitem(b.PATHS,'predicate_fp64',path)
    called=[]
    monkeypatch.setattr(b.importlib.util,'spec_from_file_location',lambda name,p:called.append(p) or SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)))
    monkeypatch.setattr(b.importlib.util,'module_from_spec',lambda spec:SimpleNamespace())
    with pytest.raises(ValueError,match='no fallback'):b.load('predicate_fp64',b.CORRECTED_IMAGE)
    assert not called
    real_identity=b.identity
    monkeypatch.setattr(b,'identity',lambda variant:dict(real_identity(variant),binary_sha256=digest(path)))
    b.load('predicate_fp64',b.CORRECTED_IMAGE);assert called==[path]
    monkeypatch.delitem(sys.modules,'_raytracing')


def test_integration_metadata_imports_do_not_construct_or_import_models():
    code='''
import builtins
real=builtins.__import__
def blocked(name,*a,**kw):
 assert name.split('.')[0] not in ('torch','network','_raytracing','train','dataset'),name
 return real(name,*a,**kw)
builtins.__import__=blocked
from infrastructure.verified_ray_backend import identity
assert identity()['variant']=='predicate_fp64'
from benchmarks.bell_e11 import IMAGE
'''
    r=subprocess.run(['python3','-B','-c',code],cwd=ROOT,text=True,capture_output=True);assert r.returncode==0,r.stderr


def test_generation_scientific_render_body_only_adds_backend_loading():
    old=subprocess.check_output(['git','show','6af0474:benchmarks/generate_bell_e11.py'],cwd=ROOT,text=True)
    new=(ROOT/'benchmarks/generate_bell_e11.py').read_text()
    def render(s):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='render')
    a,c=render(old),render(new)
    c.body=[n for n in c.body if not (isinstance(n,ast.ImportFrom) and n.module=='infrastructure.verified_ray_backend') and
        not (isinstance(n,ast.Assign) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='load_ray_backend' for x in ast.walk(n)))]
    assert ast.dump(a)==ast.dump(c)
