from pathlib import Path

import numpy as np
import pytest

from benchmarks import bell_publication_seeds_v1 as s
from experiments.audit_donut_geometry_readiness_v1 import transport_axis
from experiments.train_bell_publication_seeds_v1 import SeededPositions, trainer_source


def test_seeded_recorder_matches_native_policy(tmp_path):
    x = np.arange(1025)
    a = SeededPositions(x, tmp_path, 6034)
    batch = next(s.dense.index_batches(len(x), 6034, 1, 512))
    np.testing.assert_array_equal(x[batch], a[batch])
    np.testing.assert_array_equal(np.fromfile(a.path, dtype='<u4'), batch)
    held = next(s.dense.index_batches(len(x), 6035, 1, 512))
    with pytest.raises(ValueError, match='indices differ'):
        a[held]
    assert a.steps == 1 and a.path.stat().st_size == 512*4


def test_fresh_base_and_tensor_records_fail_on_wrong_seed_or_base():
    base = dict(seed=6034, variant='A_corrected_core_v1', state_sha256='a'*64)
    s.validate_initial(base, 'K1-Dense-6034')
    with pytest.raises(ValueError, match='initialization identity'):
        s.validate_initial(base, 'K1-Dense-6035')
    tensor = dict(seed=6034, variant='donut_tensor_ambient_weak_start_v1',
                  common_base_state_sha256='a'*64, state_sha256='b'*64)
    s.validate_initial(tensor, 'Final-Ambient-6034', base)
    with pytest.raises(ValueError, match='base weights differ'):
        s.validate_initial(tensor, 'Final-Ambient-6034', dict(base, state_sha256='d'*64))


def test_exact_overlays_keep_original_loop_and_conditional_tensor_gate():
    for method in s.METHODS:
        code, _ = trainer_source(method)
        compile(code, method, 'exec')
        assert "sample_rng=np.random.RandomState(POLICY['seed'])" in code
        assert 'optimizer.step()' in code
        assert 'fresh_seed_initialization_verified' in code
        assert 'args.arm_initial' not in code
        assert ('acquisition.tensor_admission(' in code) == (method != 'K1-Dense')


def test_definition_and_authorization_inherit_exact_science_and_caps():
    out = Path('/home/isavand/paper/evidence/bell-publication-seeds-v1-synthetic')
    f = s.definition(out)
    assert len(f['order']) == 6
    assert set(f['photometric_index_sha256']) == {'6034', '6035'}
    assert len(set(f['photometric_index_sha256'].values())) == 2
    assert sum(r['work']['shader_calls'] for r in f['arms'].values()) == 103112
    assert sum(r['work']['shadow_rays'] for r in f['arms'].values()) == 79265415168
    for arm, recipe in f['arms'].items():
        seed, method = s.seed_method(arm)
        assert recipe['limits']['updates_per_arm'] == 10000
        assert recipe['spatial_policy'] is None and recipe['illumination'] == 'known_L2'
        assert s.policy(arm) == dict(s.POLICY, seed=seed)
        assert (recipe['representation'] is None) == (method == 'K1-Dense')
    a = s.authorization(f, 'a'*40, 'd'*64)
    assert a['host_output'] == str(out) and a['order'] == f['order'] and a['recipes'] == f['arms']
    assert a['fresh_seeds'] == [6034, 6035] and a['maximum_execution_launches'] == 1


def test_geometry_transport_preserves_length_tangency_and_sign_equivalence():
    source = np.array([0., 0., 1.])
    target = np.array([1., 0., 0.])
    axis = np.array([1., 0., 0.])
    moved = transport_axis(axis, source, target)
    np.testing.assert_allclose(moved, -source)
    np.testing.assert_allclose(transport_axis(-axis, source, target), -moved)
    assert moved@target == 0 and transport_axis(axis, source, -source) is None
