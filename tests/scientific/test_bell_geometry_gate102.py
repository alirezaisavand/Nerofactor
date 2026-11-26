"""Synthetic domain/precision/topology checks. No real extraction or model queries."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import bell_geometry_gate102 as p
from infrastructure.signatures import SignatureError


def test_cells_require_all_eight_inside_corners():
    outside=np.zeros((3,3,3),bool);outside[2,2,2]=True
    valid,any_inside=p.domain_cells(outside)
    assert valid.sum()==7 and any_inside.all() and not valid[1,1,1]
    result=p.cell_counts(valid,any_inside)
    assert result['boundary_cells']==1 and result['fully_outside']==0


def test_ambiguous_shared_face_requires_every_consistent_cell():
    valid=np.ones((2,2,2),bool);valid[1,0,0]=False
    any_inside=np.ones_like(valid)
    vertices=np.array([[1.,.1,.1],[1.,.9,.1],[1.,.1,.9]])
    keep,reason,possible,good=p.face_domain(vertices,np.array([[0,1,2]]),valid,any_inside)
    assert possible.tolist()==[2] and good.tolist()==[1]
    assert keep.tolist()==[False] and reason.tolist()==[1]


def test_unique_inside_cell_and_outside_cell_have_explicit_reasons():
    valid=np.zeros((2,2,2),bool);valid[0,0,0]=True
    any_inside=valid.copy();any_inside[1,0,0]=True
    lo=np.array([[.1,.1,.1],[1.1,.1,.1],[1.1,1.1,1.1]])
    hi=lo+.5
    keep,reason,_,_=p.bounds_support(lo,hi,valid,any_inside)
    assert keep.tolist()==[True,False,False] and reason.tolist()==[0,2,3]


def test_domain_classifier_keeps_outside_oracle_points_as_unsupported():
    valid=np.ones((2,2,2),bool)
    result=p.point_domain(np.array([[0.,0.,0.],[1.5,0.,0.],[-1.5,0.,0.]]),valid,valid)
    assert result.tolist()==[True,False,False]


def test_native_precision_weld_removes_exact_zero_only_and_preserves_tiny_positive():
    v=np.array([[.5,0,0],[.5+1e-9,0,0],[.5,.1,0],[.4,0,0],
                [.4+1e-6,0,0],[.4,1e-6,0]],float)
    f=np.array([[0,1,2],[0,2,3],[3,4,5]],np.uint64)
    a,b,mapping,retained,removed,info=p.numerical_cleanup(v,f,np.ones(3,bool))
    assert removed.tolist()==[0] and retained.tolist()==[1,2]
    assert np.array_equal(a[b],v[f[retained]].astype(np.float32))
    assert info['minimum_positive_area']>0 and info['exact_coordinate_merges']==1
    assert mapping.shape==(len(v),)


def test_components_and_open_boundary_are_retained():
    v=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[.5,0,0],[.6,0,0],[.5,.1,0]])
    f=np.array([[0,1,2],[3,4,5]])
    out,labels,edges=p.topology(v,f)
    assert out['edge_connected_components']==2 and labels.tolist()==[0,1]
    assert out['open_boundary_edges']==6 and out['nonmanifold_edges']==0
    assert edges.shape==(6,2)


def test_no_supported_surface_or_all_zero_stops_without_fallback():
    v=np.array([[0.,0,0],[0,0,0],[0,0,0]])
    f=np.array([[0,1,2]])
    with pytest.raises(SignatureError):p.numerical_cleanup(v,f,np.array([False]))
    with pytest.raises(SignatureError):p.numerical_cleanup(v,f,np.array([True]))


def test_original_quality_thresholds_are_reused_and_module_import_is_cold():
    from benchmarks import bell_geometry_gate10_v1 as original
    assert p.LIMITS['SDF_points']==original.LIMITS['sdf_witness_evaluations_max']
    code="""import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','network','dataset','train','raytracing','open3d','mcubes','trimesh','_raytracing'):
   raise RuntimeError('Scientific import '+name)
sys.meta_path.insert(0,Block())
import experiments.bell_geometry_gate102
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=p.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_native_mesh_layout_and_vertex_mapping_are_explicit():
    v=np.array([[0.,0,0],[.1,0,0],[0,.1,0],[9,9,9]],float)
    f=np.array([[0,1,2]],np.uint64)
    cv,cf,mapping,retained,_,_=p.numerical_cleanup(v,f,np.array([True]))
    assert cv.dtype==np.float32 and cf.dtype==np.int32
    assert mapping[-1]==-1 and retained.tolist()==[0]
    assert np.array_equal(cv[cf],v[f].astype(np.float32))


def test_multi_cell_triangles_fail_without_geometric_trimming():
    valid=np.ones((3,3,3),bool)
    v=np.array([[0.,0,0],[2.,0,0],[0,2.,0]])
    with pytest.raises(SignatureError):p.face_domain(v,np.array([[0,1,2]]),valid,valid)


def test_source_defers_checkpoint_to_consistency_phase():
    import ast
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate102.py').read_text())
    extract=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
                   n.func.id in ('own_checkpoint','strict_network_load','NeROShapeRenderer')
                   for n in ast.walk(extract))
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native')
    work=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='work')
    calls=[n.func.id for n in ast.walk(work) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id in ('extract','geometry_audit','sdf_audit')]
    assert calls==['extract','geometry_audit','sdf_audit']


def test_cell_edge_and_corner_ambiguity_requires_all_generators():
    valid=np.ones((2,2,2),bool);any_inside=valid.copy()
    lo=np.array([[1.,1.,.1],[1.,1.,1.]])
    hi=np.array([[1.,1.,.9],[1.,1.,1.]])
    keep,_,possible,good=p.bounds_support(lo,hi,valid,any_inside)
    assert keep.all() and possible.tolist()==[4,8] and np.array_equal(possible,good)
    valid[1,1,1]=False
    keep,reason,_,good=p.bounds_support(lo,hi,valid,any_inside)
    assert keep.tolist()==[True,False] and reason.tolist()==[0,1] and good.tolist()==[4,7]


def test_native_freeze_validation_needs_only_mounted_source_metadata(tmp_path,monkeypatch):
    # Host evidence is verified at preflight; its absolute paths are not container paths.
    out=p.previous.PRIOR.parent/'bell-geometry-gate102-synthetic-freeze-test'
    plan=p.definition(out)
    target=tmp_path/'freeze.json';p.write(target,plan)
    monkeypatch.setattr(p,'FREEZE',target)
    def forbidden(*args,**kwargs):raise AssertionError('Host path accessed by native freeze validation')
    monkeypatch.setattr(p,'host_inputs',forbidden)
    monkeypatch.setattr(p,'check',forbidden)
    assert p.freeze(p.digest(target))==plan
