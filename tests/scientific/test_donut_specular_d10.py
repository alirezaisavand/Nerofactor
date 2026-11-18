"""Focused offline checks for the exact replay and paired intervention logic."""
from pathlib import Path

import numpy as np
import pytest

from benchmarks import donut_specular_d10 as d
from infrastructure.signatures import SignatureError


def test_archive_se_and_fixed_tape_schedule():
    assert d.tape(0, 0) == (828000, 2428000)
    assert d.tape(512, 7) == (828519, 2428519)
    assert len({d.tape(start, draw) for start in range(0, d.POINTS, d.CHUNK)
                for draw in range(d.DRAWS)}) == 640
    draws = np.arange(8*2*3*3, dtype=np.float32).reshape(8, 2, 3, 3)/100
    mean = draws.mean(0)
    std_error = draws.std(0, ddof=1)/np.sqrt(8)
    assert d.verify_replay(draws, mean, std_error)['passed']
    changed = std_error.copy()
    changed[0, 1, 0] += 2e-6
    assert not d.verify_replay(draws, mean, changed)['passed']
    with pytest.raises(SignatureError, match='Incomplete archived replay'):
        d.verify_replay(draws[:-1], mean, std_error)


def test_paired_factorial_effect_reports_sse_on_mean_and_uncertainty():
    ref = np.zeros((2, 3, 3))
    mask = np.array([True, False])
    noise = np.linspace(-.01, .01, d.DRAWS)[:, None, None, None]
    stats = {}
    for name, scale in [('R1', 3.), ('R2', 2.), ('R3', 2.), ('R4', .5)]:
        draws = np.full((d.DRAWS, 2, 3, 3), scale)+noise
        stats[name] = d.draw_summary(draws, ref, mask)
    changed = d.effect(stats, 'SSE', dict(R2=1, R1=-1))
    assert changed['value'] == pytest.approx([-15.]*3)
    assert 0 < changed['paired_MC_SE'][-1] < .1
    interaction = d.effect(stats, 'SSE', dict(R4=1, R3=-1, R2=-1, R1=1))
    assert interaction['value'] == pytest.approx([3.75]*3)
    assert d.split_context(stats) == 'recovered'
    stats['R2'] = stats['R1']
    assert d.split_context(stats) == 'oracle'
    stats['R4'] = stats['R3']
    assert d.split_context(stats) is None


def test_orientation_eligibility_excludes_near_isotropy():
    assert d.orientation_eligible(np.array([.5, .501, .35]), np.array([.5, .5, .08])).tolist() == [
        False, False, True]


def test_hash_identity_then_check_one_launch_scope(tmp_path):
    p = tmp_path/'artifact'
    p.write_bytes(b'exact D8-T checkpoint identity only')
    record = dict(bytes=p.stat().st_size, sha256=d.digest(p))
    assert d.checked(p, record) == p
    p.write_bytes(b'inexact D8-T checkpoint identity xx')
    with pytest.raises(SignatureError, match='Frozen artifact differs'):
        d.checked(p, record)
    plan = dict(artifacts=dict(checkpoint=record), host_output=str(Path(tmp_path)/'fresh'))
    auth = d.authorization(plan, 'frozen-commit', 'frozen-sha')
    assert auth['maximum_execution_launches'] == 1
    assert auth['trusted_checkpoint_sha256'] == record['sha256']
    assert auth['isolated_cpu_pickle_authorized'] and auth['restore_optimizer'] is False
    assert auth['limits']['optimizer_returns'] == 0
