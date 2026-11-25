from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import donut_representation_gate5_v1 as g
from benchmarks.donut_recovery_v1 import json_new
from experiments.train_donut_representation_gate5_v1 import Acquisition, gate4
from infrastructure.signatures import SignatureError


def test_only_representation_and_corresponding_initial_state_change():
    parent = g.parent()
    recipe = parent['arms'][g.AMBIENT]
    paired = g.m.bell.freeze(g.ROOT/g.m.bell.FREEZE,g.m.BELL_SHA)
    chart = paired['arms']['Final-Chart']
    expected = dict(recipe,learned_control=chart['learned_control'],initial=chart['initial'])
    f = g.definition(Path('/home/isavand/paper/evidence/donut-representation-gate5-metadata'),
                     {key:dict(bytes=1,sha256='0'*64) for key in g.CAPTURES})
    assert f['arms']=={g.ARM:expected}
    assert f['support']==parent['support']
    assert f['arms'][g.ARM]['trainer_sha256']==sha256(gate4.trainer_source(True).encode()).hexdigest()
    assert f['arms'][g.ARM]['work']==g.m.work(True)
    assert f['steps'][-1]==10000
    assert g.paths(f,native=True)['initial']==Path(g.m.MOUNTS['initial'])
    assert g.paths(f,native=True)['chart_initial']!=g.paths(f,native=True)['paired']


def test_gate5_dispatch_requires_exact_approval_and_uses_unchanged_training_definition(tmp_path):
    parent = g.parent()
    f = g.definition(tmp_path/'fresh',
                     {key:dict(bytes=1,sha256='0'*64) for key in g.CAPTURES})
    a = Acquisition.__new__(Acquisition)
    a.frozen,a.f,a.recipe = f,dict(parent,arms=f['arms']),f['arms'][g.ARM]
    args = SimpleNamespace(authorization=tmp_path/'authorization.json',commit='test',
            freeze_sha256='test-freeze',image_id=g.m.IMAGE,benchmark_sha256=g.m.GENERATION,
            initial_sha256=parent['base_initial']['sha256'])
    json_new(args.authorization,g.authorization(f,args.commit,args.freeze_sha256))
    args.authorization_sha256 = g.digest(args.authorization)
    active,auth = a.runtime(args)
    assert active['arm_training']['learned_control']=='D9-C'
    assert active['arm_training']['geometry']=='oracle'
    assert active['arm_training']['protocol']==g.PROTOCOL
    assert active['tensor_training']['initialization_policy']==g.m.bell.INITIALIZATION
    assert active['tensor_training']['surface_q_enabled'] is False
    assert active['support']==parent['support'] and auth['arms']==[g.ARM]
    args.freeze_sha256 = 'different-freeze'
    with pytest.raises(SignatureError,match='Exact Gate 5 approval'):
        a.runtime(args)


@pytest.mark.parametrize('kind,key,tolerance',[
    ('paired','D9-C_world_Q',g.PARITY['initial_Q_atol']),
    ('paired','D9-C_linear_diffuse',g.PARITY['initial_RGB_atol']),
    ('physical','D9-T_pdf',g.PARITY['physical_atol']),
    ('native','native_draws',g.PARITY['native_RGB_atol']),
])
def test_comparator_float_budget_is_field_specific(kind,key,tolerance):
    old = {key:np.array([0.,0.])}
    result = g.compare_capture(kind,old,{key:np.array([tolerance/2,0.])})
    assert result=={key:tolerance/2}
    with pytest.raises(SignatureError,match='Historical comparator differs'):
        g.compare_capture(kind,old,{key:np.array([tolerance*2,0.])})


@pytest.mark.parametrize('kind,key', [('paired','D9-T_kd'),('native','reference'),('native','pixel_indices')])
def test_comparator_exact_fields_fail_on_any_substitution(kind,key):
    old = {key:np.array([0.,1.])}
    with pytest.raises(SignatureError,match='Historical comparator differs'):
        g.compare_capture(kind,old,{key:np.array([1e-10,1.])})


@pytest.mark.parametrize('other',[{}, {'D9-C_world_Q':np.array([np.nan])},
    {'D9-C_world_Q':np.array([0],dtype=np.float32)}, {'D9-C_world_Q':np.zeros((1,1))}])
def test_comparator_rejects_incomplete_or_invalid_capture(other):
    with pytest.raises(SignatureError,match='[Cc]apture field'):
        g.compare_capture('paired',{'D9-C_world_Q':np.array([0.])},other)


def test_replay_uses_accepted_capture_hashes_and_keeps_both_representation_keys():
    parent = g.parent()
    batch,_,controls = g.control_receipts(parent)
    ambient = batch/g.AMBIENT
    captures = {k:controls[g.AMBIENT]['outputs'][name] for k,name in g.CAPTURES.items()}
    f = g.definition(Path('/home/isavand/paper/evidence/donut-representation-gate5-metadata'),captures)
    p = g.paths(f)
    g.check_inputs(f,p)
    replay = g.replay_captures(f,p,ambient)
    assert all(delta==0. for capture in replay.values() for delta in capture.values())
    assert {'D9-C_world_Q','D9-T_world_Q'}<=replay['paired'].keys()
    wrong = deepcopy(f)
    wrong['ambient_captures']['paired']['sha256']='0'*64
    with pytest.raises(SignatureError):
        g.replay_captures(wrong,p,ambient)
