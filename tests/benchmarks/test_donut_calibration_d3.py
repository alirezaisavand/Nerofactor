"""CPU checks of the real D3 callback using synthetic positive radiance."""
from contextlib import contextmanager
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.donut_calibration_d3 import CALIBRATION, NODES, CalibratedEnvironment, quadrature
from benchmarks import donut_recovery_v1 as recovery
from experiments.donut_material_recovery_v1 import shader_lighting
from infrastructure.stage_sources import digest


def test_quadrature_integrates_constant_and_first_moments():
    directions, weights = quadrature()
    assert directions.size == 3 * NODES
    assert weights.shape == directions.shape[:2]
    assert (weights > 0).all()
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-15)
    np.testing.assert_allclose((directions * weights[..., None]).sum(axis=(0, 1)), 0, atol=1e-14)


def setup(raw_environment):
    directions, weights = quadrature()
    dirs = torch.tensor(directions.reshape(-1, 3), dtype=torch.float64)
    weights = torch.tensor(weights.reshape(-1, 1), dtype=torch.float64)
    target = torch.tensor(CALIBRATION['mean_linear_RGB'], dtype=torch.float64)
    env = CalibratedEnvironment(raw_environment, dirs, weights, target, torch)
    timing = {stage: dict(setup_wall_seconds=0., grid_forward_and_reduction_CUDA_seconds=0.)
              for stage in ('training', 'evaluation')}
    return env, timing


def test_callback_gradient_includes_denominator_and_cancels_channel_gain():
    gain = torch.tensor([.5, -.4, .9], dtype=torch.float64, requires_grad=True)
    angular = torch.tensor(.22, dtype=torch.float64, requires_grad=True)
    calls = []

    def raw(points, directions):
        calls.append(len(directions))
        return (gain + angular * directions[:, :1]).exp().expand_as(directions)

    env, timing = setup(raw)
    work = []
    budget = lambda key, count: work.append((key, count))
    shader = SimpleNamespace(cfg={'indirect_mode': 'zero_bounce_v1', 'outer_light_version': 'direction'})

    @contextmanager
    def install(callback, identity):
        shader.callback = callback
        try:
            yield
        finally:
            shader.callback = None

    shader.environment = install
    rng = torch.get_rng_state().clone()
    query = env.directions[:17]
    with shader_lighting(shader, None, 'D3', env):
        with env.scope(budget, timing, torch, evaluation=False):
            output = shader.callback(env.points[:17], query)
        assert env.scale is None  # Autograd still owns the denominator graph.
        dg, da = torch.autograd.grad(output.sum(), (gain, angular))
        with torch.no_grad(), env.scope(budget, timing, torch, evaluation=True):
            grid = env.grid_radiance()
            assert torch.equal(grid, shader.callback(env.points, env.directions))
            assert torch.allclose((grid * env.weights).sum(0), env.target, atol=1e-14, rtol=0)
    assert shader.callback is None and torch.equal(rng, torch.get_rng_state())
    assert calls == [NODES, len(query), NODES, NODES]
    assert work == [('calibration_evaluations', 1), ('calibration_directions', NODES)] * 2
    assert torch.allclose(dg, torch.zeros_like(gain), atol=1e-13, rtol=0)

    def independently_evaluate(a):
        d, w = quadrature()
        denominator = (np.exp(a * d[..., 0]) * w).sum()
        return float(env.target.sum()) * np.exp(a * query[:, 0].numpy()).sum() / denominator

    h = 1e-5
    derivative = (independently_evaluate(.22 + h) - independently_evaluate(.22 - h)) / (2 * h)
    assert abs(derivative) > 1e-3
    assert float(da) == pytest.approx(derivative, abs=1e-9)


@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
def test_invalid_calibration_fails_without_installing_scale(value):
    env, timing = setup(lambda points, directions: torch.full_like(directions, value))
    with pytest.raises(ValueError, match='Invalid D3 calibration'):
        with env.scope(lambda *a: None, timing, torch, evaluation=True):
            pytest.fail('Invalid denominator entered calibration scope')
    assert env.scale is None
    with pytest.raises(ValueError, match='outside calibration scope'):
        env(env.points, env.directions)


def test_no_grad_denominator_rejected_during_training_and_scope_restored_on_error():
    env, timing = setup(lambda points, directions: torch.ones_like(directions))
    with pytest.raises(ValueError, match='lost its gradient'):
        with env.scope(lambda *a: None, timing, torch, evaluation=False):
            pytest.fail('Detached denominator entered training')
    with pytest.raises(RuntimeError, match='fixture'):
        with env.scope(lambda *a: None, timing, torch, evaluation=True):
            raise RuntimeError('fixture')
    assert env.scale is env.grid is env.raw_mean is None


def test_float32_grid_meets_the_frozen_mean_tolerance():
    directions, weights = quadrature()
    dirs = torch.from_numpy(directions.reshape(-1, 3)).float()
    weights_torch = torch.from_numpy(weights.reshape(-1, 1)).float()
    target = torch.tensor(CALIBRATION['mean_linear_RGB'], dtype=torch.float32)

    def raw(points, directions):
        return (.01 + directions.clamp(min=0).pow(64)) * target

    env = CalibratedEnvironment(raw, dirs, weights_torch, target, torch)
    timing = {'evaluation': dict(setup_wall_seconds=0., grid_forward_and_reduction_CUDA_seconds=0.)}
    with torch.no_grad(), env.scope(lambda *a: None, timing, torch, evaluation=True):
        grid = env.grid_radiance().numpy().reshape(directions.shape)
    mean = (grid * weights[..., None]).sum((0, 1))
    np.testing.assert_allclose(mean, CALIBRATION['mean_linear_RGB'], rtol=0,
                               atol=CALIBRATION['absolute_mean_tolerance'])


def test_d3_runtime_rejects_other_phases_and_mismatched_approval(monkeypatch, tmp_path):
    d3 = dict(protocol='d3-protocol', limits={'updates_per_arm':10000}, benchmark_generation_sha256='generation',
              dataset_admission_sha256='admitted', initial_sha256='initial')
    freeze = dict(calibrated_training=d3, host_calibrated_output='batch')
    monkeypatch.setattr(recovery, 'verify_freeze', lambda *a: freeze)
    monkeypatch.setattr(os, 'geteuid', lambda: 1109)
    authorization = tmp_path / 'authorization.json'
    approved = dict(approved=True, protocol=d3['protocol'], limits=d3['limits'], commit='commit',
                    freeze_sha256='freeze', image_id=recovery.IMAGE, output_directories={'D3':str(tmp_path/'D3')},
                    host_output_directory='batch', benchmark_generation_sha256='generation',
                    dataset_admission_sha256='admitted', initial_sha256='initial',
                    selected_gpu='0', maximum_execution_launches=1)
    args = SimpleNamespace(freeze=tmp_path/'freeze.json', freeze_sha256='freeze', authorization=authorization,
                           commit='commit', image_id=recovery.IMAGE, output=tmp_path/'D3')
    authorization.write_text(json.dumps(approved))
    args.authorization_sha256 = digest(authorization)
    with pytest.raises(ValueError, match='dedicated execution freeze'):
        recovery.runtime(args, 'D2')
    with pytest.raises(ValueError, match='Fresh owned'):
        recovery.runtime(args, 'D3')  # Exact binding gets as far as the independent mount gate.
    for name in ['dataset_admission_sha256', 'initial_sha256', 'benchmark_generation_sha256']:
        authorization.write_text(json.dumps(dict(approved, **{name:'other'})))
        args.authorization_sha256 = digest(authorization)
        with pytest.raises(ValueError, match='D3 authorization differs'):
            recovery.runtime(args, 'D3')
