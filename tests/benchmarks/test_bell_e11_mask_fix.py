"""Focused E11 shape regression: CPU/synthetic inputs only, no BRDF/GPU/model."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from benchmarks.bell_e11 import (mask_for_primary_rays,geometry_gate,linear_to_srgb,
                                convergence,LIMITS,TOL,ENV,RELIGHT,MESH,LAW)
from benchmarks.generate_bell_e11 import primary,render,write_arrays
from infrastructure.stage_sources import ROOT,digest


@pytest.fixture
def plane():
    import torch
    P=np.concatenate([np.diag([1.,-1.,-1.]),np.array([[0.],[0.],[3.]])],1).astype(np.float32)
    K=np.array([[128.,0,64.],[0,128.,64.],[0,0,1.]],dtype=np.float32)
    state=torch.get_rng_state().clone();o,d=primary(torch,P,K)
    assert torch.equal(state,torch.get_rng_state())
    o,d=o.numpy(),d.numpy();depth=-3/d[:,2];p=o+depth[:,None]*d
    n=np.broadcast_to([0.,0.,1.],o.shape);return o,d,p,-n,n,depth,P,K


def gate(plane,hit,mask):
    o,d,p,raw,n,depth,P,K=plane
    depth=np.where(hit,depth,1e10)
    return geometry_gate(o,d,p,raw,n,depth,hit,mask,P,K)


@pytest.mark.parametrize('hits,masks,expected,passed',[
    ([0,127,128,2259],[0,127,128,2259],1.,True),
    ([0,127],[0,127,128,2259],.5,False),
    ([0,127,128,2259],[0,127,128],.75,False),
    (list(range(20)),list(range(19)),.95,True),
    ([127],[128],0.,False),
])
def test_actual_geometry_gate_receives_c_order128_mask(plane,hits,masks,expected,passed):
    hit=np.zeros(16384,dtype=bool);hit[hits]=True
    source=np.zeros((128,128),dtype=bool);source.ravel(order='C')[masks]=True;before=source.copy()
    mask=mask_for_primary_rays(source)
    assert mask.shape==(16384,) and np.array_equal(source,before)
    result=gate(plane,hit,mask)
    assert result['silhouette_iou']==expected and result['passed'] is passed
    assert result['reprojection_pixels']<=TOL['reprojection_pixels']


def test_primary_pixel_order_is_y_times128_plus_x(plane):
    coords=[(0,127),(1,0),(17,83),(127,2)]
    source=np.zeros((128,128),dtype=bool)
    for y,x in coords:source[y,x]=True
    mask=mask_for_primary_rays(np.asfortranarray(source))
    ids=np.array([y*128+x for y,x in coords]);assert np.array_equal(np.flatnonzero(mask),ids)
    o,d,p,raw,n,depth,P,K=plane
    for (y,x),j in zip(coords,ids):
        local=d[j]@P[:,:3].T;pixel=local@K.T;pixel=pixel[:2]/pixel[2]
        np.testing.assert_allclose(pixel,[x+.5,y+.5],atol=2e-5,rtol=0)
    assert gate(plane,mask,mask)['silhouette_iou']==1
    assert gate(plane,mask,source.ravel(order='F'))['silhouette_iou']<1


def test_original800_mask_resize_assignment_feeds_actual_geometry_gate(plane):
    # Execute only the actual generator's two data-only packing assignments.
    # This covers the call site which the former pre-flattened fixture missed.
    tree=ast.parse(inspect.getsource(render));assignments=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('index','mask') for t in node.targets):
            if any(isinstance(v,ast.Attribute) and v.attr=='ix_' for v in ast.walk(node)) or any(isinstance(t,ast.Name) and t.id=='index' for t in node.targets):
                assignments.append(node)
    assert len(assignments)==2
    image=np.zeros((800,800),dtype=np.uint8);index=np.floor((np.arange(128)+.5)*800/128).astype(int)
    selected=[(1,127),(23,45),(126,0)]
    for y,x in selected:image[index[y],index[x]]=255
    before=image.copy();scope={'np':np,'mask':image,'mask_for_primary_rays':mask_for_primary_rays}
    module=ast.Module(body=assignments,type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'<mask packing>','exec'),scope)
    mask=scope['mask'];assert np.array_equal(image,before)
    assert np.array_equal(np.flatnonzero(mask),[y*128+x for y,x in selected])
    assert gate(plane,mask,mask)['passed']


@pytest.mark.parametrize('bad',['mask_image','hit_image','depth_column','origins_image'])
def test_geometry_gate_rejects_mismatched_shapes_before_broadcast(plane,bad):
    o,d,p,raw,n,depth,P,K=plane;hit=np.ones(16384,dtype=bool);mask=hit.copy()
    if bad=='mask_image':mask=mask.reshape(128,128)
    if bad=='hit_image':hit=hit.reshape(128,128)
    if bad=='depth_column':depth=depth[:,None]
    if bad=='origins_image':o=o.reshape(128,128,3)
    with pytest.raises(ValueError,match='Matching flattened'):geometry_gate(o,d,p,raw,n,depth,hit,mask,P,K)


def test_nonempty_union_and_no_hit_are_distinct_failures(plane):
    empty=np.zeros(16384,dtype=bool)
    with pytest.raises(ValueError,match='Nonempty hit/mask union'):gate(plane,empty,empty)
    source=empty.copy();source[0]=True
    with pytest.raises(ValueError,match='Empty/invalid primary'):gate(plane,empty,source)


@pytest.mark.parametrize('mask',[np.zeros((128,128),dtype=np.uint8),np.zeros(16384,dtype=bool)])
def test_mask_packing_rejects_wrong_layout_or_dtype(mask):
    with pytest.raises(ValueError,match='128x128 Boolean'):mask_for_primary_rays(mask)


def test_saved_rgb_depth_labels_and_relight_image_shapes(tmp_path):
    # The nearby producer/consumer expressions inspected in render/verify:
    # flat N x(component3) x RGB3 <-> H x W x component3 x RGB3.
    N=128*128;pixels=np.array([127,128,2259,16382]);mask=np.zeros((128,128),dtype=bool)
    mask.ravel()[pixels]=True;fine=np.arange(N*9,dtype=float).reshape(N,3,3)/(N*9)
    path=tmp_path/'reference.npz';write_arrays(path,linear_components=fine.reshape(128,128,3,3),hit=mask)
    with np.load(path,allow_pickle=False) as saved:
        image=saved['linear_components'];hit=saved['hit']
        assert np.array_equal(image[hit,2],fine[pixels,2])
        assert np.array_equal(linear_to_srgb(image[:,:,2]),linear_to_srgb(fine[:,2].reshape(128,128,3)))
        assert convergence(image[hit,2],fine[pixels,2])['passed']
        assert np.array_equal(np.flatnonzero(hit),pixels)
        depth=np.arange(N,dtype=float)/N;encoded=np.rint(np.where(hit.ravel(),depth,15).reshape(128,128)/15*65535).astype(np.uint16)
        assert np.array_equal(encoded[hit],np.rint(depth[pixels]/15*65535).astype(np.uint16))
        # Relighting follows precisely the same RGB/component indexing.
        relight=image*.5;assert np.array_equal(relight[hit,2],fine[pixels,2]*.5)


def test_only_render_mask_packing_changes_and_scientific_identities_stay_fixed():
    original=subprocess.check_output(['git','show','1657a2c:benchmarks/generate_bell_e11.py'],cwd=ROOT,text=True)
    old=next(x for x in ast.parse(original).body if isinstance(x,ast.FunctionDef) and x.name=='render')
    new=ast.parse(inspect.getsource(render)).body[0]
    class PackingOnly(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='mask' for t in node.targets) and any(isinstance(v,ast.Attribute) and v.attr=='ix_' for v in ast.walk(node)):
                return ast.Pass()
            return self.generic_visit(node)
    assert ast.dump(PackingOnly().visit(old))==ast.dump(PackingOnly().visit(new))
    frozen=ROOT/'docs/protocols/BELL_ORACLE_GENERATION_E11_V1.freeze.json'
    assert digest(frozen)=='4682a6304be08d3f03d525dde10c27ecc90910e9f6581395f8c09bd83d1fb3af'
    f=json.loads(frozen.read_text());assert f['work_limits']==LIMITS and f['tolerances']==TOL
    assert f['known_environment']==ENV and f['relight_environment']==RELIGHT and f['mesh_sha256']==MESH and f['BRDF']==LAW
    changed={'benchmarks/bell_e11.py','benchmarks/generate_bell_e11.py','benchmarks/run_bell_e11.sh','AGENTS.md'}
    for name,sha in f['files'].items():
        if name not in changed:assert digest(ROOT/name)==sha,name
