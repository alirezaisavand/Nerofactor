"""E8R correction/reuse regression fixtures only; real scientific imports blocked."""
import ast
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments import uniform_tapes_e8r_v1 as policy
from experiments import spatial_pilot_e8r_v1 as pilot

FORBIDDEN=('torch','experiments.neural_material_field_v1','experiments.material_shader_v1',
 'experiments.neural_material_recovery_v1','experiments.run_known_material_recovery_v1',
 'experiments.anisotropic_estimator_v1','experiments.known_material_reference_v1','experiments.multi_material_reference_v1')


@pytest.fixture(autouse=True)
def no_science():
    assert not any(n==k or n.startswith(k+'.') for n in sys.modules for k in FORBIDDEN)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,*args):
            if any(name==k or name.startswith(k+'.') for k in FORBIDDEN):raise AssertionError('Scientific import blocked: '+name)
    b=Block();sys.meta_path.insert(0,b);old=np.random.get_state()
    try:yield
    finally:
        sys.meta_path.remove(b);new=np.random.get_state()
        assert old[0]==new[0] and np.array_equal(old[1],new[1]) and old[2:]==new[2:]


def test_immediately_below_one_roundup_only_and_all_other_bits_unchanged():
    source=np.array([-0.,0.,.1,.5,float(policy.UPPER),np.nextafter(1.,0.),.9999999982641566],np.float64)
    old,new,mask=policy.convert(source)
    assert mask.tolist()==[False,False,False,False,False,True,True]
    assert new[-1]==new[-2]==np.nextafter(np.float32(1),np.float32(0))
    assert np.array_equal(old[~mask].view(np.uint32),new[~mask].view(np.uint32))
    assert new[0].view(np.uint32)==np.uint32(0x80000000)  # Preserve negative zero, too.
    assert (new>=0).all() and (new<1).all()


@pytest.mark.parametrize('value',[1.,np.nextafter(1.,2.),-.1,float('nan'),float('inf'),-float('inf')])
def test_invalid_float64_source_rejected_before_conversion(value):
    with pytest.raises(Exception,match='original float64'):policy.convert(np.array([value],np.float64))


def test_no_float32_source_or_hidden_dtype_reinterpretation():
    with pytest.raises(Exception,match='float64'):policy.convert(np.array([.5],np.float32))
    a=np.array([[.2,.4],[.6,.8]],np.float64);old,new,mask=policy.convert(a)
    assert not mask.any() and old.tobytes()==new.tobytes() and new.shape==(2,2)


def test_complete_panel_deterministic_identities_and_single_known_correction():
    frozen=json.loads((pilot.ROOT/pilot.legacy.FREEZE).read_text());a,summary=policy.preflight(frozen['tapes'])
    b,again=policy.preflight(frozen['tapes'],summary)
    assert summary==again and summary['checked_values']==5311008 and summary['arrays_checked']==36
    assert summary['changed_arrays']==len(summary['corrections'])==1
    changed=[(i,name,k) for (i,name),columns in a.items() for k in columns
             if summary['corrected_identities'][str(i)][name][k]!=frozen['tapes'][str(i)][name][k]]
    assert changed==[(5,'evaluation','v')]
    assert a[5,'evaluation']['v'][42,982]==policy.UPPER
    assert all(np.array_equal(v,b[key][k]) for key,columns in a.items() for k,v in columns.items())
    assert all(np.isfinite(v).all() and (v>=0).all() and (v<1).all() for columns in a.values() for v in columns.values())


def test_original_or_corrected_identity_mismatch_fails_preflight():
    frozen=json.loads((pilot.ROOT/pilot.legacy.FREEZE).read_text())
    original=json.loads(json.dumps(frozen['tapes']));original['0']['fit']['u']['sha256']='0'*64
    with pytest.raises(Exception,match='Original E8'):policy.preflight(original)
    _,summary=policy.preflight(frozen['tapes']);summary['checked_values']-=1
    with pytest.raises(Exception,match='Frozen corrected'):policy.preflight(frozen['tapes'],summary)


def reference_fixture(tmp_path):
    root=tmp_path/'source';store=pilot.obs.Store(root,tmp_path);truth=np.tile([.3,.5,.2,.3,.4,.5,.2],(6,1)).tolist();rows=[]
    for i in range(6):
        fine=np.full((84,3,3),.1);coarse=fine.copy()
        row=dict(location=i,truth=truth[i],passed=True,arrays=store.numeric('reference/'+str(i)+'.npz',
            dict(coarse=coarse,fine=fine,absolute_delta=abs(coarse-fine),limit=2e-6+5e-5*abs(fine)),1024**2))
        if i<4:
            gate=dict(explicit_truth=truth[i],passed=True,perturbation_convergence_passed=True,jacobian_gate_passed=True,
                stable_rank_under_observed_discrepancy=True,jacobians=np.tile(np.eye(72,7),(2,2,1,1)).tolist(),
                steps=[dict(ratio=1.),dict(ratio=1.)],smallest_singular_value=1.,step_spectral_discrepancy=0.,grid_spectral_discrepancies=[0.,0.])
            row['rank']=store.json('rank/'+str(i)+'.json',gate)
        rows.append(row)
    selection=store.json('result.json',dict(status='failed',scientific_work_completed=False,completed=dict(reference_calls=118,reference_directions=487587840,constructors=0),references=rows))
    return root,selection,dict(truth=truth)


def test_cached_reference_admission_has_no_generation(tmp_path):
    root,selected,design=reference_fixture(tmp_path)
    fine,rows=policy.admit_references(root,selected,design)
    assert len(fine)==len(rows)==6 and all(a.shape==(84,3,3) for a in fine)


@pytest.mark.parametrize('bad',['hash','truth','missing'])
def test_unsafe_or_incompatible_reference_evidence_rejected(tmp_path,bad):
    root,selected,design=reference_fixture(tmp_path)
    if bad=='hash':selected=dict(selected,sha256='0'*64)
    if bad=='truth':design['truth'][0][0]=.9
    if bad=='missing':(root/'reference/0.npz').unlink()  # Disposable synthetic fixture only.
    with pytest.raises(Exception):policy.admit_references(root,selected,design)


def test_scientific_core_and_both_optimization_loops_are_exact_frozen_E8_AST():
    def parsed(module):return ast.parse(Path(module.__file__).read_text())
    old=parsed(pilot.legacy);new=parsed(pilot)
    def core(t):return next(n for n in ast.walk(t) if isinstance(n,ast.ClassDef) and n.name=='Core')
    assert ast.dump(core(old))==ast.dump(core(new))
    def loops(t):return [ast.dump(n) for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)=='seed' and ast.unparse(n.iter)=='method.SEEDS']
    assert loops(old)==loops(new) and len(loops(old))==2
    for name in ('matching_updates','Work'):
        assert ast.dump(next(n for n in old.body if getattr(n,'name',None)==name))==ast.dump(next(n for n in new.body if getattr(n,'name',None)==name))
    assert all(pilot.CAPS[k]==v for k,v in pilot.legacy.CAPS.items() if k not in ('reference_calls','reference_directions'))
    assert pilot.CAPS['reference_calls']==pilot.CAPS['reference_directions']==0


def test_preflight_precedes_scientific_import_and_reference_calls_are_absent(tmp_path):
    source=Path(pilot.__file__).read_text();tree=ast.parse(source)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_reviewed')
    pre=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tape_policy.preflight')
    scientific=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
    assert pre<scientific and 'make_grids' not in source and 'identifiability(' not in source
    path=tmp_path/'review.json';path.write_text('{}\n')
    with pytest.raises(Exception,match='approval'):pilot.execute_reviewed(path,pilot.sha(path),'0'*64)
    work=pilot.Work()
    with pytest.raises(Exception,match='cap'):work.attempt('reference_calls')
    assert work.attempts['reference_calls']==0
