import ast

import numpy as np
import pytest

from benchmarks import donut_relighting_gate6_v2 as g
from experiments.analyze_donut_relighting_gate6_v2 import verifier_source
from experiments.donut_relighting_gate6_v2 import runner_source
from infrastructure.signatures import SignatureError


def test_model_batches_replay_signed_pixel_order_and_isolate_extras():
    matching=np.arange(20,20+g.CHUNK+9)*2
    extras=np.array([1,43,59,2000])
    view=dict(pixel_indices=np.sort(np.r_[matching,extras]),matched_pixels=matching)
    calls=[]

    def predict(rows):
        pixels = view['pixel_indices'][rows]
        calls.append(pixels)
        # Deliberate batch-dependent material drift to expose the V1 layout.
        return {'color':pixels.astype('f4')[:,None]/2048+len(rows)*1e-5}

    values=g.reconstruct_material(view,predict)['color']
    keep,extra=g.layout(view)
    np.testing.assert_array_equal(calls[0],matching[:g.CHUNK])
    np.testing.assert_array_equal(calls[1],matching[g.CHUNK:])
    np.testing.assert_array_equal(calls[2],extras)
    archive=np.concatenate([p.astype('f4')/2048+len(p)*1e-5 for p in calls[:2]])
    np.testing.assert_array_equal(values[keep,0],archive)
    np.testing.assert_array_equal(values[extra,0],extras.astype('f4')/2048+4e-5)
    expanded=np.concatenate([p.astype('f4')/2048+len(p)*1e-5 for p in np.split(view['pixel_indices'],[g.CHUNK])])
    assert abs(expanded[keep]-archive).max()>g.MATERIAL_ATOL


def test_replay_rejects_missing_or_nonunique_pixels():
    with pytest.raises(SignatureError,match='Archived pixels missing'):
        g.layout(dict(pixel_indices=np.array([1,2]),matched_pixels=np.array([3])))
    with pytest.raises(SignatureError,match='increasing unique'):
        g.layout(dict(pixel_indices=np.array([1,2]),matched_pixels=np.array([1,1])))


def test_localized_failure_is_reported_with_unchanged_threshold():
    view=dict(pixel_indices=np.array([7,19,27]),matched_pixels=np.array([7,27]))
    old=dict(hit_mask=np.isin(np.arange(32),view['matched_pixels']).reshape(4,8),
             Q_world=np.zeros((4,8,3,3),np.float32),kd=np.zeros((4,8,3),np.float32),
             metallic=np.zeros((4,8,1),np.float32))
    fields=dict(Q_world=np.zeros((3,3,3)),color=np.zeros((3,3)),metallic=np.zeros((3,1)))
    fields['Q_world'][1]=1.  # Extra foreground is not a parity witness.
    fields['color'][2,0]=g.MATERIAL_ATOL/2
    record=g.field_parity(g.ARMS[0],view,fields,old)
    assert record['passed'] and record['maximum_error_pixels']['color']==27
    assert record['field_max_abs']['Q_world']==0
    fields['color'][2,0]=2*g.MATERIAL_ATOL
    record=g.field_parity(g.ARMS[0],view,fields,old)
    assert not record['passed'] and record['maximum_error_pixels']['color']==27


def test_real_accepted_batches_and_exact_K1_numpy_tensor_export():
    f=g.freeze(g.digest(g.ROOT/g.FREEZE))
    _,views,archived=g.admit(f,g.base_paths(f),g.endpoint_paths(f))
    for c,tail in zip(g.CAMERAS,(222,428)):
        view=views[c]
        keep,extra=g.layout(view)
        np.testing.assert_array_equal(view['pixel_indices'][keep],view['matched_pixels'])
        assert g.batch_record(view)['archived'][-1]==tail and len(extra)==19
        old=archived[g.ARMS[0],c]
        mask=old['hit_mask']
        q=g.archived_q(*(old[k][mask] for k in ('mx','my','tangents','bitangents')))
        np.testing.assert_array_equal(q,old['Q_world'][mask])


def test_v2_freeze_preserves_scientific_inputs_counts_and_native_tapes():
    f=g.freeze(g.digest(g.ROOT/g.FREEZE))
    v1=g.v1.freeze(g.V1_SHA)
    changed={k for k in v1 if v1[k]!=f[k]}
    assert changed=={'host_output','limits','protocol','sources'}
    assert {k:f['limits'][k] for k in v1['limits']}==v1['limits']
    assert f['limits']['material_field_calls']==51
    assert f['limits']['material_field_points']==22032
    assert f['material_max_abs']==v1['material_max_abs']==1e-6
    assert f['environment']==v1['environment']
    a=g.authorization(f,'v2-test',g.digest(g.ROOT/g.FREEZE))
    assert a['checkpoints']==g.v1.authorization(v1,'v1-test',g.V1_SHA)['checkpoints']
    assert a['checkpoint_cpu_pickle_trust'] and a['maximum_execution_launches']==1


def test_source_adapter_keeps_independent_integrator_and_retains_failure_witness():
    native=runner_source()
    analyze=verifier_source()
    compile(native,'<gate6_v2_native>','exec')
    compile(analyze,'<gate6_v2_offline>','exec')
    path=g.ROOT/'experiments/donut_relighting_gate6_v1.py'
    def integrate_source(source):
        node=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='integrate')
        return ast.get_source_segment(source,node)
    assert integrate_source(native)==integrate_source(path.read_text())
    assert 'native_pass(shader,arm,views,archives,torch,charge,save_capture,r[\'field_admission\'])' in native
    assert "g.field_parity(arm,v,d,old)" in analyze
