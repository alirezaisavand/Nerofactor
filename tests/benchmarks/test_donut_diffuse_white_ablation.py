"""Synthetic CPU checks for the D4/D5 objective, reporting, and launch binding."""

from contextlib import contextmanager
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import donut_recovery_v1 as benchmark
from benchmarks.donut_calibration_d3 import CALIBRATION, quadrature
from experiments import donut_material_recovery_v1 as adapter
from experiments.donut_material_recovery_v1 import (
    calibration_mean_record,
    shader_lighting,
    training_objective,
)
from infrastructure.stage_sources import digest
from infrastructure.surface_q_regularization import gradient_diagnostics


def test_diffuse_white_probe_has_gradient_but_contributes_nothing_to_update():
    material = torch.tensor(2., requires_grad=True)
    light = torch.tensor(3., requires_grad=True)
    photo = (material * light - 1).square()
    prior = material.square() * .05
    white = (light - 1).square() * .01
    report = gradient_diagnostics({'photo': photo, 'material': prior, 'diffuse_white': white},
                                  [material, light], torch, ['mx_predictor.weight', 'outer_light.weight'])
    assert report['terms']['diffuse_white']['groups']['outer_light'] == pytest.approx(.04)
    assert material.grad is None and light.grad is None
    with_white = training_objective(photo, prior, white, include_diffuse_white=True)
    removed = training_objective(photo, prior, white, include_diffuse_white=False)
    assert torch.equal(removed, photo + prior)
    white_on_gradient = torch.autograd.grad(with_white, light, retain_graph=True)[0]
    removed.backward()
    assert light.grad.item() == pytest.approx(20.)
    assert material.grad.item() == pytest.approx(30.2)
    assert white_on_gradient.item() == pytest.approx(20.04)


def test_calibrated_mean_report_retains_three_channels_after_camera_processing():
    directions, weights = quadrature()
    target = np.array(CALIBRATION['mean_linear_RGB'])
    grid = target + .01 * directions
    calibration_record = calibration_mean_record(grid, weights, CALIBRATION)
    chunk = np.ones((447, 3, 3), dtype=np.float32)
    # Replay the actual endpoint assignment with the old shadowing variable present.
    source = ast.parse(Path(adapter.__file__).read_text())
    evaluate = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name=='evaluate')
    field = ast.dump(ast.parse("metrics['illumination']['calibration'] = None").body[0].targets[0])
    assignment = next(n for n in ast.walk(evaluate) if isinstance(n, ast.Assign) and
                      ast.dump(n.targets[0]) == field)
    metrics = {'illumination': {}}
    scope = dict(metrics=metrics, calibration_record=calibration_record, mean=chunk,
                 mean_error=np.zeros(3), d3={'calibration': CALIBRATION},
                 calibrated=SimpleNamespace(raw_mean=torch.full((3,), .5), scale=torch.tensor(2 * target)))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<endpoint metadata>', 'exec'), scope)
    report = metrics['illumination']['calibration']
    assert np.asarray(report['realized_mean_RGB']).shape == (3,)
    np.testing.assert_allclose(report['realized_mean_RGB'], target, atol=1e-13, rtol=0)
    assert max(report['absolute_error']) <= CALIBRATION['absolute_mean_tolerance']
    with pytest.raises(ValueError, match='violates the calibrated grid mean'):
        calibration_mean_record(grid + .001, weights, CALIBRATION)


@pytest.mark.parametrize('arm,expected',[('D4', 2), ('D5', 3)])
def test_ablation_arms_select_calibrated_and_free_light_respectively(arm, expected):
    shader = SimpleNamespace(cfg={'indirect_mode': 'zero_bounce_v1', 'outer_light_version': 'direction'})

    @contextmanager
    def environment(callback, identity):
        shader.callback = callback
        try:
            yield
        finally:
            del shader.callback

    shader.environment = environment
    original = SimpleNamespace(predict_outer_lights=lambda shader, p, d: 3)
    calibrated = lambda p, d: 2
    with shader_lighting(shader, original, arm, calibrated):
        assert shader.callback(None, None) == expected
    assert not hasattr(shader, 'callback')


def test_ablation_authorization_is_bound_to_both_ordered_arms(monkeypatch, tmp_path):
    limits = {'D4': {'updates_per_arm': 10000}, 'D5': {'updates_per_arm': 10000}}
    definition = dict(protocol=benchmark.ABLATION, limits=limits, wall_seconds=7500,
                      benchmark_generation_sha256='generation', dataset_admission_sha256='admitted',
                      initial_sha256='initial')
    freeze = dict(diffuse_white_ablation=definition, host_ablation_output='batch')
    monkeypatch.setattr(benchmark, 'verify_freeze', lambda *args: freeze)
    monkeypatch.setattr(os, 'geteuid', lambda: 0)  # Cannot cross the independent non-root gate.
    auth = tmp_path / 'authorization.json'
    record = dict(definition, approved=True, commit='commit', freeze_sha256='freeze',
                  image_id=benchmark.IMAGE, host_output_directory='batch', arms=['D4','D5'],
                  output_directories={'D4':'/outputs/D4','D5':'/outputs/D5'}, selected_gpu='0',
                  maximum_execution_launches=1)
    for phase in ('D4','D5'):
        args = SimpleNamespace(freeze=tmp_path/'freeze.json', freeze_sha256='freeze', authorization=auth,
                               commit='commit', image_id=benchmark.IMAGE, output='/outputs/'+phase)
        auth.write_text(json.dumps(record))
        args.authorization_sha256 = digest(auth)
        with pytest.raises(ValueError, match='Fresh owned non-root'):
            benchmark.runtime(args, phase)
        with pytest.raises(ValueError, match='D4/D5 require their dedicated execution freeze'):
            benchmark.runtime(args, 'D2')
        auth.write_text(json.dumps(dict(record, arms=['D5','D4'])))
        args.authorization_sha256 = digest(auth)
        with pytest.raises(ValueError, match='authorization differs'):
            benchmark.runtime(args, phase)
