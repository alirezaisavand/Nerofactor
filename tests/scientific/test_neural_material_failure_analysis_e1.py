"""New deterministic NumPy fixtures. No old scientific tests or model execution."""
import copy
import hashlib
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import analyze_neural_material_failures_e1 as a


def fixture(collapse=False):
    records=[]
    for u in [1]+list(range(10,5001,10)):
        zero=collapse and u>=10
        records.append(dict(update=u,loss_at_update=u-1,loss=.03 if zero else .02,
            physical_before_update=[1,1,0,0,1,0,1] if zero else [.2,.4,.2,.3,.4,.5,.6],
            gradient_groups={g:dict(finite=True,missing=0,nonzero=0 if zero else 1,
                                  squared_norm=0. if zero else 1.) for g in a.GROUPS}))
    return dict(history=records)


def test_unique_complete_catalog_and_duplicate_rejection():
    rows=[dict(material=m,seed=s,arm=arm) for m in ('M1','M2','M3') for s in (101,103,105) for arm in ('D0','N0','N1')]
    a.coverage(rows)
    with pytest.raises(Exception,match='27-history'):a.coverage(rows[:-1]+[rows[0]])


def test_counts_come_from_all_endpoint_decisions_without_rescoring():
    rows=[]
    for arm,count in (('D0',4),('N0',4),('N1',1)):
        for i,(m,s) in enumerate((m,s) for m in ('M1','M2','M3') for s in (101,103,105)):
            rows.append(dict(material=m,seed=s,arm=arm,evaluation=dict(success=dict(all_passed=i<count,checks=dict(held_out_radiance=i<3)))))
    out=a.endpoint_counts(rows)
    assert [out[x]['physical_pass'] for x in ('D0','N0','N1')]==[4,4,1]
    assert all(x['total']==9 for x in out.values())


def test_segmented_progress_is_separate_from_duplicate_replay():
    assert a.segment_phase(999,True)=='original_prefix'
    assert a.segment_phase(1000,True)=='authorized_replay'
    assert a.segment_phase(4269,True)=='authorized_replay'
    assert a.segment_phase(4270,True)=='new_progress'
    assert a.segment_phase(0,False)=='single_segment'


def test_preupdate_onset_is_interval_censored_and_not_update_ten():
    row=fixture(True);before=copy.deepcopy(row)
    e=dict(physical=[1,1,0,0,1,0,1],success=dict(all_passed=False,checks=dict(held_out_radiance=False)))
    c=a.classify(row,e)
    assert c['group'].startswith('A_') and c['exact_zero_records']==500
    assert c['onset']['first_zero_log_label']==10 and c['onset']['first_zero_state_update']==9
    assert c['onset']['possible_onset_completed_updates']==[1,9]
    assert c['onset']['later_nonzero_records']==0 and row==before


def test_missing_intermediate_record_is_rejected_not_fabricated():
    row=fixture();row['history'].pop(2)
    with pytest.raises(Exception,match='no interpolation'):a.history_records(row)


def test_gradient_zero_is_exact_and_all_six_groups_are_required():
    groups=fixture(True)['history'][1]['gradient_groups']
    assert a.all_zero(groups)
    groups['mx_predictor'].update(nonzero=1,squared_norm=1e-300)
    assert not a.all_zero(groups)
    groups.pop('my_predictor')
    with pytest.raises(Exception,match='Incomplete'):a.all_zero(groups)


def test_near_zero_raw_orientation_is_not_zero_centered_vector():
    d=a.orientation([0,0]);assert d['centered_radius_float64']==np.sqrt(2)
    assert np.isclose(d['normalization_operator_norm'],1/np.sqrt(2))
    assert d['angular_logit_jacobian_norm']==0
    z=a.orientation([.5,.5]);assert z['angular_logit_jacobian'] is None


def test_orientation_factor_matches_independent_analytical_expression():
    q=np.array([.3,.8]);d=a.orientation(q);c=2*q-1
    expected=np.array([-c[1],c[0]])/(c@c)*q*(1-q)
    np.testing.assert_allclose(d['angular_logit_jacobian'],expected,rtol=1e-14,atol=0)


def test_physical_bounds_use_float32_floor_and_exclude_angular_gauge():
    p=[float(np.float32(.04)),1,100*np.pi,0,1,0,1]
    assert (a.boundary_distances(p)==0).all()
    p[0]=.04+1e-9;assert a.boundary_distances(p)[0]>0


def test_stored_moment_diagnostic_never_modifies_weights_or_moments():
    m=np.array([.1,0.],np.float32);v=np.array([.03,.01],np.float32);before=(m.copy(),v.copy())
    d=a.moment_direction(m,v,250,dict(lr=.01,betas=[.9,.999],eps=1e-8))
    assert np.isfinite(d).all() and d[1]==0
    assert np.array_equal(m,before[0]) and np.array_equal(v,before[1])


def test_null_orientation_is_retained_and_classification_never_rescores_it():
    row=fixture();e=dict(physical=[.4,.4,0,.3,.4,.5,.6],
        recovery_errors=dict(orientation_degrees=None),success=dict(all_passed=False,checks=dict(held_out_radiance=True)))
    assert a.classify(row,e)['group'].startswith('C_')
    assert e['recovery_errors']['orientation_degrees'] is None


def test_corrupt_evidence_rejected_before_json(tmp_path):
    p=tmp_path/'bad.json';p.write_text('not json')
    with pytest.raises(Exception,match='hash mismatch'):a.select(p,'0'*64)


def test_csv_bytes_are_reproducible_and_no_torch_import(tmp_path):
    rows=[dict(name='M1/105',segment='restored+replay',orientation=None,logical=5000,duplicate=3270)]
    a.write_csv(tmp_path/'a.csv',rows);a.write_csv(tmp_path/'b.csv',rows)
    assert (tmp_path/'a.csv').read_bytes()==(tmp_path/'b.csv').read_bytes()
    subprocess.run([sys.executable,'-B','-c',
        'import sys;from experiments import analyze_neural_material_failures_e1;assert "torch" not in sys.modules'],check=True)


def test_object_archive_is_rejected_without_deserialization(tmp_path):
    archive=tmp_path/'object.npz'
    np.savez(archive,a0=np.asarray([{}],dtype=object),numpy_rng=np.zeros(624,np.uint32))
    freeze=dict(configuration_sha256='config',sha256={},configuration=dict(optimizer_effective={}))
    meta=dict(archive=archive.name,archive_sha256=a.saved.digest(archive),completed_updates=250,
        execution_provenance=dict(freeze_sha256=a.PINS['freeze'][1],configuration_sha256='config',source_sha256={},run_id='test'),
        optimizer_effective={},entries=[dict(key='a0',role='parameter',name='x',shape=[1],dtype='torch.float32',sha256='unused')])
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(meta))
    selection=dict(manifest=manifest.name,manifest_sha256=a.saved.digest(manifest),completed_updates=250)
    before=archive.read_bytes()
    with pytest.raises(ValueError,match='allow_pickle=False'):
        a.checkpoint_arrays(tmp_path,selection,{},freeze,'test')
    assert archive.read_bytes()==before


def test_saved_analysis_catalog_counts_segments_and_numeric_findings():
    r=json.loads((a.ROOT/'docs/evidence/NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.json').read_text())
    a.coverage(r['catalog']);assert r['counts']==a.endpoint_counts(r['catalog'])
    assert all(c['initial_fit_state_update']==0 and c['final_fit_state_update']==5000 and
               c['available_records']==501 and c['available_checkpoint_updates']==[250,1000,5000] for c in r['catalog'])
    assert all(s['phase']=='POST-update' for s in r['checkpoint_diagnostics'])
    assert [r['counts'][arm]['physical_pass'] for arm in ('D0','N0','N1')]==[4,4,1]
    groups=[c['classification']['group'][0] for c in r['N1_cases']]
    assert [groups.count(g) for g in 'ABCD']==[5,1,2,1]
    resumed=next(c for c in r['catalog'] if (c['material'],c['seed'],c['arm'])==('M1',105,'N1'))
    assert resumed['source']['restored_and_replayed'] and resumed['source']['run_id']!=resumed['source']['initial_history_run_id']
    for c in r['N1_cases']:
        if c['classification']['group'].startswith('A_'):
            state=next(s for s in c['checkpoints'] if s['update']==5000)
            assert sum(g['changed_elements'] for g in state['parameter_displacement_from_previous'].values())==0
    figures=a.ROOT/'docs/figures/neural-material-e1'
    for name in ('catalog.csv','classification.csv','paired_endpoints.csv'):
        assert a.saved.digest(figures/name)==r['artifacts'][name]['sha256']
