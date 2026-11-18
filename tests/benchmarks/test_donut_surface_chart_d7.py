import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import donut_recovery_v1 as recovery
from benchmarks import donut_surface_chart_d7 as d7
from benchmarks import donut_surface_q_d6 as d6
from infrastructure.stage_sources import digest
from infrastructure.surface_q_regularization import transported_q_loss, world_q


def angle_vector(theta):
    return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta)], dim=-1)


def test_axial_chord_matches_tensor_angle_component_in_common_frame():
    u, v = [torch.tensor([a], dtype=torch.float64) for a in (.2, -.7)]
    widths = u.new_tensor([[.3, .15]])
    r0, r1 = angle_vector(u), angle_vector(v)
    weight = float((widths[0, 0].square() - widths[0, 1].square()).square() / 2)
    actual = d7.chart_difference(widths, widths, r0, r1, weight, torch)
    normal = u.new_tensor([[0, 0, 1]])
    q0, q1 = [world_q(widths[:, :1], widths[:, 1:], r, normal, torch) for r in (r0, r1)]
    _, expected = transported_q_loss(q0, q1, normal, normal, u.new_ones(1), torch)
    assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-12)
    assert torch.allclose(actual, d7.chart_difference(
        widths, widths, angle_vector(u+math.pi), r1, weight, torch))
    # The control intentionally penalizes one endpoint's physically equivalent swap.
    swapped = d7.chart_difference(widths, widths.flip(-1), r0, -r0, weight, torch)
    assert swapped.item() > 0
    assert torch.allclose(q0, world_q(widths[:, 1:], widths[:, :1], -r0, normal, torch))


def test_actual_surface_term_gradients_distance_and_rng():
    x = torch.tensor([.05, .03, .4], requires_grad=True, dtype=torch.float64)
    rng = torch.get_rng_state()

    class Predictor:
        calls = 0

        def predict_anisotropic_components(self, p):
            self.calls += 1
            mx, my = .3 + x[0] * p[:, :1], .15 + x[1] * p[:, :1]
            return mx, my, None, None, angle_vector(x[2] * p[:, 0]), None

    p = Predictor()
    batch = dict(points0=x.new_tensor([[0., 0, 0]]), points1=x.new_tensor([[1., 0, 0]]),
                 distance=x.new_tensor([.01]))
    loss = d7.surface_chart_term(p, batch, d7.POLICY, torch)
    assert p.calls == 2 and torch.equal(rng, torch.get_rng_state())
    loss.backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs() > 0).all()
    double_distance = d7.surface_chart_term(
        p, dict(batch, distance=batch['distance'] * 2), d7.POLICY, torch)
    assert torch.allclose(double_distance, loss / 4)
    with pytest.raises(ValueError, match='Invalid surface distance'):
        d7.surface_chart_term(p, dict(batch, distance=batch['distance'] * 0), d7.POLICY, torch)
    with pytest.raises(ValueError, match='Invalid chart orientation'):
        d7.chart_difference(x[None, :2], x[None, :2], x.new_zeros((1, 2)),
                            x.new_ones((1, 2)), .00638, torch)


def test_d6_domain_policy_and_archive_hash_contract():
    assert {k: d7.POLICY[k] for k in d6.POLICY} == d6.POLICY
    assert d7.LIMITS == d6.LIMITS
    a = np.arange(16, dtype='<i8').reshape(4, 4)
    assert d7.array_hash(a) == d7.array_hash(a.astype('>i8')) == d7.array_hash(np.asfortranarray(a))
    assert d7.array_hash(a) != d7.array_hash(a.reshape(16))
    assert d7.array_hash(a) != d7.array_hash(a.astype('<i4'))
    assert d7.array_hash(a) != d7.array_hash(a[::-1])


def test_d7_runtime_requires_dedicated_phase_and_exact_objective(tmp_path, monkeypatch):
    definition = dict(protocol=d7.PROTOCOL, limits=d7.LIMITS, spatial_policy=d7.POLICY,
                      benchmark_generation_sha256='dataset', dataset_admission_sha256='admitted',
                      initial_sha256='initial')
    freeze = dict(surface_chart_training=definition, host_chart_output='batch')
    monkeypatch.setattr(recovery, 'verify_freeze', lambda *args: freeze)
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    auth = tmp_path / 'authorization.json'
    value = dict(definition, approved=True, commit='commit', freeze_sha256='freeze',
                 image_id=recovery.IMAGE, selected_gpu='0', maximum_execution_launches=1,
                 host_output_directory='batch', diffuse_white_in_objective=True,
                 output_directories={'D7': '/outputs/D7'})
    args = SimpleNamespace(freeze=tmp_path/'freeze.json', freeze_sha256='freeze', authorization=auth,
                           commit='commit', image_id=recovery.IMAGE, output=Path('/outputs/D7'))
    auth.write_text(json.dumps(value))
    args.authorization_sha256 = digest(auth)
    with pytest.raises(ValueError, match='Fresh owned'):
        recovery.runtime(args, 'D7')
    for phase in ('D2', 'D6'):
        with pytest.raises(ValueError, match='requires its dedicated'):
            recovery.runtime(args, phase)
    for alteration in (dict(diffuse_white_in_objective=False),
                       dict(spatial_policy=dict(d7.POLICY, angle_weight=.01))):
        auth.write_text(json.dumps(dict(value, **alteration)))
        args.authorization_sha256 = digest(auth)
        with pytest.raises(ValueError, match='D7 authorization differs'):
            recovery.runtime(args, 'D7')
