"""Exercise the production NPZ call expressions with the real existing writer."""
import ast
import copy
import inspect

import numpy as np
import pytest

from benchmarks import bell_geometry_gate102 as p
from experiments.donut_reconstructed_geometry_v1 import save
from experiments.bell_geometry_gate102 import read_completed_extraction
from infrastructure.signatures import SignatureError


def calls():
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate102.py').read_text())
    return sorted((n for n in ast.walk(tree) if isinstance(n,ast.Call) and
                   isinstance(n.func,ast.Name) and n.func.id=='save'),key=lambda n:n.lineno)


def test_every_actual_npz_call_matches_existing_writer_signature():
    assert len(calls())==7
    for call in calls():
        inspect.signature(save).bind(*call.args,**{kw.arg:kw.value for kw in call.keywords})


def test_all_production_npz_calls_roundtrip_exact_keys_values_and_dtypes(tmp_path):
    (tmp_path/'rasters').mkdir()
    keep=np.array([True,False]);reason=np.array([0,1],np.uint8)
    possible=np.array([1,2],np.int64);good=np.array([1,1],np.uint8)
    labels=np.array([0],np.int64);edges=np.array([[0,1]],np.int64)
    mapping=np.array([0,1,-1],np.int64);retained=np.array([0],np.int64);removed=np.array([1],np.int64)
    old_labels=labels.copy();old_edges=edges.copy();oa=np.array([0.,1.])
    arrays={'points':np.array([[.1,.2,.3]],np.float32),'hit':np.array([True])}
    faces=np.array([2],np.int64);vertices=np.array([3],np.int64)
    values=dict(locals(),out=tmp_path,camera='synthetic',save=save,np=np)
    # Evaluate only actual serialization expressions. No extraction, tracing or model call.
    code=ast.Module(body=[ast.Expr(value=copy.deepcopy(c)) for c in calls()],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),'production-save-calls','exec'),values)
    expected={
        'face-domain.npz':dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good),
        'topology-before-cleanup.npz':dict(face_labels=labels,boundary_edges=edges),
        'cleanup-mappings.npz':dict(raw_vertex_to_candidate=mapping,candidate_face_to_raw=retained,
                                    removed_exact_zero_raw_faces=removed),
        'topology-after-cleanup.npz':dict(face_labels=labels,boundary_edges=edges),
        'historical-positive-area-topology.npz':dict(face_labels=old_labels,boundary_edges=old_edges,
                                                   face_to_original=np.array([1],np.int64)),
        'rasters/synthetic.npz':arrays,
        'sdf-witnesses.npz':dict(**arrays,selected_faces=faces,selected_vertices=vertices)}
    assert {x.relative_to(tmp_path).as_posix() for x in tmp_path.rglob('*.npz')}==set(expected)
    for name,fields in expected.items():
        with np.load(tmp_path/name,allow_pickle=False) as z:
            assert set(z.files)==set(fields)
            for key,a in fields.items():
                assert z[key].dtype==a.dtype and np.array_equal(z[key],a)
    with pytest.raises(FileExistsError):save(tmp_path/'face-domain.npz',{})


def test_continuation_has_no_new_extraction_call():
    tree=ast.parse((p.ROOT/'experiments/bell_geometry_gate102.py').read_text())
    extract=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr=='marching_cubes' for n in ast.walk(extract))
    assert p.LIMITS['MC_calls']==0


def test_scientific_policies_and_remaining_limits_match_original_freeze():
    old=p.read(p.ROOT/'docs/protocols/BELL_DOMAIN_AWARE_EXTRACTION_GATE102_V1.freeze.json')
    assert dict(old['limits'],MC_calls=0)==p.LIMITS
    assert old['quality']==p.original.QUALITY and old['resources']==p.RESOURCES
    new=p.definition(p.previous.PRIOR.parent/'bell-geometry-gate102-v2-test')
    for key in ('image','backend','scientific_commit','checkpoint','network_sha256','original_mesh',
                'acquisition','eval_files','raw_grid','outside_mask','grid_shape','bounds','level',
                'domain','cell_policy','numerical_policy','quality','resources'):
        assert old[key]==new[key],key


@pytest.mark.parametrize('defect',[None,'hash','dtype','coordinate','index'])
def test_completed_extraction_capture_requires_identity_and_numerical_layout(tmp_path,defect):
    voxel=np.array([[1.,2,3],[2,2,3],[1,3,3]],np.float64)
    world=voxel/511*2-1
    faces=np.array([[0,1,2]],np.uint64)
    if defect=='dtype':faces=faces.astype(np.int32)
    if defect=='coordinate':world[0,0]+=.01
    if defect=='index':faces[0,0]=len(voxel)
    data=dict(zip(('raw-voxel-vertices.npy','raw-world-vertices.npy','raw-faces.npy'),(voxel,world,faces)))
    for name,value in data.items():np.save(tmp_path/name,value,allow_pickle=False)
    captured=dict(layout=dict(vertices=3,faces=1,vertex_dtype='float64',face_dtype='uint64'),
                  arrays={name:p.item(tmp_path/name) for name in data})
    if defect=='hash':captured['arrays']['raw-faces.npy']['sha256']='0'*64
    if defect:
        with pytest.raises(SignatureError):read_completed_extraction(captured,tmp_path)
    else:
        loaded=read_completed_extraction(captured,tmp_path)
        assert all(np.array_equal(a,b) and a.dtype==b.dtype for a,b in zip(loaded,data.values()))
