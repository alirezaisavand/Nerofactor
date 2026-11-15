"""Saved scalar arithmetic/patch fixtures only; no rendering, models or GPU."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pytest
from audits.bell_triangle_diagnostic_e11 import (ray_hits,iq_predicate,fma_edge_witness,face_geometry,edge_neighbors)
from infrastructure.stage_sources import ROOT,digest


@pytest.fixture
def case():
    with np.load(ROOT/'tests/fixtures/bell_triangle_e11/case.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}


def test_saved_ray_identifies_farther_exit_and_closer_front_face(case):
    hits=ray_hits(case['origin'],case['direction'],case['triangles'])
    assert [int(case['face_indices'][h['face']]) for h in hits]==[181142,158220]
    assert abs(hits[0]['depth']-2.751337749748485)<1e-12
    assert abs(hits[1]['depth']-float(case['saved_depth']))<1e-6
    assert hits[1]['depth']-hits[0]['depth']>.85
    tri=case['triangles'][2].astype(float);a,b,c=tri;n=-np.cross(b-a,c-a);n/=np.linalg.norm(n)
    np.testing.assert_allclose(n,case['saved_normal'],atol=5e-8,rtol=0)
    assert np.dot(n,-case['direction'])<-.72


def test_permitted_float32_fma_order_loses_both_shared_edge_faces(case):
    rows=fma_edge_witness(case['triangles'][:2],case['origin'],case['direction'])
    assert [r['accepted'] for r in rows]==[False,False]
    assert rows[0]['sum_uv']==float(np.float32(1.0000001192092896))
    assert rows[1]['v']<0
    # A scalar non-contracted CPU order does not prove CUDA uses that order.
    assert iq_predicate(case['triangles'][0],case['origin'],case['direction'],np.float32)['accepted']


def test_candidate_double_predicate_agrees_with_independent_intersection(case):
    double=[iq_predicate(t,case['origin'],case['direction']) for t in case['triangles']]
    assert [r['accepted'] for r in double]==[True,False,True]
    assert double[0]['output_depth']==float(np.float32(2.751337749748485))
    reference=ray_hits(case['origin'],case['direction'],case['triangles'],np.longdouble)[0]
    assert abs(double[0]['depth']-reference['depth'])<1e-12
    assert min(double[0]['barycentric'])>0


@pytest.mark.parametrize('direction,accepted',[(np.array([0.,0.,-1.]),True),(np.array([0.,0.,1.]),False),(np.array([1.,0.,0.]),False)])
def test_predicate_preserves_forward_and_miss_semantics(direction,accepted):
    triangle=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]],dtype=np.float32)
    result=iq_predicate(triangle,np.array([0.,0.,1.]),direction)
    assert result['accepted'] is accepted
    assert result['output_depth']==(1. if accepted else 1e6)


def test_shared_edge_control_is_covered_without_barycentric_epsilon():
    triangles=np.array([[[-1,-1,0],[1,-1,0],[1,1,0]],[[-1,-1,0],[1,1,0],[-1,1,0]]],dtype=float)
    for x in [-1e-12,0,1e-12]:
        row=[iq_predicate(t,np.array([x,0,1]),np.array([0,0,-1])) for t in triangles]
        assert any(r['accepted'] for r in row)


def test_consistent_closed_tetrahedron_edges_are_manifold():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    rows=edge_neighbors(v,f,0)
    assert all(r['manifold'] and r['neighbors'][0]['opposite_shared_edge'] for r in rows)
    assert face_geometry(v,f,0)['area']==.5


def test_pinned_header_patch_applies_only_to_intersection_and_keeps_normals(tmp_path):
    fixture=ROOT/'tests/fixtures/raytracing_293292_triangle/triangle.cuh'
    assert digest(fixture)=='b19bcfba69334ebfa5c788f47f3066953898254a531b9c1fe4d5a2c9a67f19c7'
    target=tmp_path/'include/raytracing/triangle.cuh';target.parent.mkdir(parents=True);shutil.copyfile(fixture,target)
    patch=ROOT/'docker/paper/patches/raytracing-triangle-fp64-v1.patch'
    for mode in ('--check',None):
        args=['git','apply']+([mode] if mode else [])+[str(patch)]
        r=subprocess.run(args,cwd=tmp_path,text=True,capture_output=True);assert r.returncode==0,r.stderr
    before=fixture.read_text();after=target.read_text()
    begin=before.index('    __host__ __device__ float ray_intersect(')
    end=before.index('    __host__ __device__ float ray_intersect(',begin+1)
    first=after.index('    __host__ __device__ float ray_intersect(');last=after.index('    __host__ __device__ float ray_intersect(',first+1)
    assert before[:begin]==after[:first] and before[end:]==after[last:]
    changed=after[first:last]
    assert 'n = (b - a).cross(c - a);' in changed and 'return static_cast<float>(t);' in changed
    assert 'cast<double>()' in changed and 'double u' in changed and 'double v' in changed
    assert '(u+v)>1.0' in changed and 'epsilon' not in changed and '1e-6' not in changed
