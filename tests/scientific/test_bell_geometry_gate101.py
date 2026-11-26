"""CPU numerical and authorization-boundary fixtures for Gate 10.1 preparation."""
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import bell_geometry_gate101 as p
from benchmarks.bell_geometry_e11 import mesh_arrays
from infrastructure.signatures import SignatureError


def test_stages_localize_float32_collapse_and_preserve_positive_faces():
    v=np.array([[.5,0,0],[.5+1e-9,0,0],[.5,.1,0],[.4,0,0]],float)
    f=np.array([[0,1,2],[0,2,3]])
    before,_=p.mesh_stage(v,f)
    after,_=p.mesh_stage(v.astype('f4'),f)
    assert before['zero_area_face_ids']==[] and after['zero_area_face_ids']==[0]
    cv,cf,retained=p.candidate_arrays(v.astype('f4'),f,expected=[0])
    assert np.array_equal(cv,v.astype('f4')) and np.array_equal(cf,f[[1]])
    assert retained.tolist()==[1]
    with pytest.raises(SignatureError):p.candidate_arrays(v.astype('f4'),f,expected=[])


def test_exact_vertex_mapping_detects_merges_without_losing_face_identity():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,0,0]])
    f=np.array([[0,1,2],[0,3,2]])
    processed=v[:3]
    pf=np.array([[0,1,2],[0,1,2]])
    mapping,r=p.exact_vertex_mapping(v,processed,pf,f)
    assert mapping.tolist()==[0,1,2,1] and r['merged_vertices']==1
    assert r['maximum_coordinate_shift']==0


def test_binary_export_precision_and_original_compatibility(tmp_path):
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    f=np.array([[0,1,2]],int)
    path=tmp_path/'single.ply'
    p.write_ply(path,v,f)
    a,b=mesh_arrays(path)
    assert np.array_equal(a,v) and np.array_equal(b,f)
    path=tmp_path/'double.ply'
    p.write_ply(path,v,f,dtype='<f8')
    with path.open('rb') as stream:
        lines=[]
        while not lines or lines[-1]!=b'end_header\n':lines.append(stream.readline())
        assert b'property double x\n' in lines
        actual=np.frombuffer(stream.read(3*3*8),dtype='<f8').reshape(3,3)
    assert np.array_equal(actual,v)


def test_radial_labels_distinguish_sentinel_raw_and_absent_crossing():
    values=np.ones((384,5))
    values[0,:]=-1
    values[1,:2]=-1
    values[2,:]=0
    labels=p.radial_labels(values)
    assert np.flatnonzero(labels['sentinel_crossing']).tolist()==[0]
    assert np.flatnonzero(labels['raw_radial_crossing']).tolist()==[1]
    assert np.flatnonzero(labels['ambiguous']).tolist()==[2]
    d=p.sphere_directions()
    assert d.shape==(384,3) and np.allclose(np.linalg.norm(d,axis=1),1,atol=1e-7)
    assert ((d[256:,1]>-1)&(d[256:,1]<-.65)).all()


def test_raw_zero_crossing_uses_original_thresholds():
    y=np.array([0.,.1,0.,0.])
    gradient=np.tile([0.,0.,1.],(4,1))
    gradient[3]=0
    assert p.raw_support(y,gradient,np.array([-1.,-1.,1.,-1.]),np.ones(4)).tolist()==[True,False,False,False]


def test_capture_boundary_still_matches_original_function():
    source=p.ROOT/'scientific_sources/geometry_g/network/field.py'
    tree=ast.parse(source.read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_geometry')
    call=f.body[1]
    assert isinstance(call,ast.Assign) and call.value.func.attr=='marching_cubes'
    assert isinstance(f.body[2],ast.Assign)  # bbox values, before normalized vertices assignment
    assert any(isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple) for n in f.body)


def test_diagnosis_module_imports_without_torch_or_scientific_namespaces():
    code="""import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','network','dataset','train','raytracing','open3d','mcubes','trimesh','_raytracing'):
   raise RuntimeError('Scientific import '+name)
sys.meta_path.insert(0,Block())
import experiments.bell_geometry_gate101
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=p.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_mapping_preserves_unreferenced_vertex_as_explicit_missing():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[9,9,9]])
    f=np.array([[0,1,2]])
    mapping,r=p.exact_vertex_mapping(v,v[:3],f,f)
    assert mapping.tolist()==[0,1,2,-1] and r['unreferenced_raw_vertices']==1


def test_freeze_validation_is_mount_independent_and_has_no_checkpoint_read(tmp_path,monkeypatch):
    out=p.PRIOR.parent/'bell-geometry-gate101-fixture-not-allocated'
    f=p.definition(out)
    freeze=tmp_path/'freeze.json'
    p.write(freeze,f)
    monkeypatch.setattr(p,'FREEZE',str(freeze))
    def forbidden(*args,**kwargs):
        raise AssertionError('Host artifact access during native freeze validation')
    monkeypatch.setattr(p,'verify_host_inputs',forbidden)
    monkeypatch.setattr(p,'check',forbidden)
    assert p.freeze(p.digest(freeze))==f
    assert not out.exists()


def test_actual_capture_observer_with_synthetic_functions(tmp_path):
    from types import SimpleNamespace
    import inspect
    from experiments.bell_geometry_gate101 import save_array
    from infrastructure.signatures import require
    source=ast.parse((p.ROOT/'experiments/bell_geometry_gate101.py').read_text())
    capture=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    observer=next(n for n in capture.body if isinstance(n,ast.FunctionDef) and n.name=='observer')
    raw=np.zeros((2,2,2),np.float32)
    outside=np.zeros((2,2,2),bool)
    def fields_fixture():
        return raw.copy()
    def geometry_fixture():
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        triangles=np.array([[0,1,2]])
        marker=True
        return vertices,triangles
    boundary=geometry_fixture.__code__.co_firstlineno+3
    captured={}
    scope=dict(field=SimpleNamespace(extract_fields=fields_fixture,extract_geometry=geometry_fixture),
               captured=captured,shape=raw.shape,np=np,raw=raw,outside=outside,
               folder=tmp_path,save_array=save_array,boundary=boundary,p=p,require=require)
    exec(compile(ast.Module(body=[observer],type_ignores=[]),'<capture-observer-fixture>','exec'),scope)
    previous=sys.gettrace()
    try:
        sys.settrace(scope['observer'])
        fields_fixture()
        original=geometry_fixture()
    finally:
        sys.settrace(previous)
    assert captured['masked'] and np.array_equal(captured['marching_cubes'][0],original[0])
    assert np.array_equal(np.load(tmp_path/'extraction-field.npy',allow_pickle=False),raw)
    assert np.array_equal(np.load(tmp_path/'marching-faces.npy',allow_pickle=False),original[1])


def test_substantive_stop_owns_the_only_original_audit_call():
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate101.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    work=next(n for n in execute.body if isinstance(n,ast.FunctionDef) and n.name=='work')
    branch=next(n for n in work.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call)
                and n.test.func.id=='broad_diagnosis')
    calls=[n for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='remaining_audit']
    assert len(calls)==1 and any(calls[0] is n for child in branch.body for n in ast.walk(child))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='remaining_audit'
                   for child in branch.orelse for n in ast.walk(child))
    assert p.LIMITS['diagnostic_SDF_points']==384*5+384*3+256*3
    assert p.LIMITS['diagnostic_gradient_points']==384+256
    assert p.LIMITS['original_SDF_points']==p.old.LIMITS['sdf_witness_evaluations_max']


def test_nonfinite_raw_support_cannot_satisfy_zero_level_gate():
    with pytest.raises(SignatureError):
        p.raw_support(np.array([0.]),np.array([[np.inf,0,0]]),np.array([-1.]),np.array([1.]))


def test_unsigned_marching_indices_and_signed_trimesh_indices_keep_integer_mapping():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    raw_faces=np.array([[0,1,2]],dtype=np.uint64)
    processed_faces=raw_faces.astype(np.int64)
    mapping,r=p.exact_vertex_mapping(v,v,processed_faces,raw_faces)
    assert mapping.dtype==np.int64 and mapping.tolist()==[0,1,2]


def test_rejected_query_preserves_budget_and_failed_query_records_attempt(monkeypatch):
    import experiments.bell_geometry_gate101 as native
    class FailedQuery:
        def __init__(self,*args):
            self.points=self.gradient_points=0
        def __call__(self,*args):
            raise RuntimeError('synthetic query failure')
    monkeypatch.setattr(native,'TorchQueries',FailedQuery)
    record={}
    q=native.Queries(None,None,1,1,record)
    with pytest.raises(SignatureError):q(np.zeros((2,3)),True)
    assert record==dict(point_slots_attempted=0,gradient_slots_attempted=0)
    with pytest.raises(RuntimeError):q(np.zeros((1,3)),True)
    assert record==dict(point_slots_attempted=1,gradient_slots_attempted=1,query_point_slots=0,completed_gradient_points=0)
