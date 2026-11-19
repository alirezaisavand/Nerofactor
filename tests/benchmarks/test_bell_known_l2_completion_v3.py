import ast
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bell_known_l2_completion_v3 as completion
from benchmarks import bell_known_l2_v1 as known
from benchmarks.bell_reference_e11 import environment
from benchmarks.donut_recovery_v1 import ENVIRONMENT, json_new
from benchmarks.donut_surface_chart_d7 import array_hash
from benchmarks.donut_surface_q_d6 import coverage_summary
from infrastructure.signatures import require
from infrastructure.sphere_visuals import write_view
from infrastructure.surface_q_regularization import normal_transport


def source(path):
    return ast.parse((completion.ROOT / path).read_text(), feature_version=(3, 8))


def adapter_members(tree, aliases):
    return {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }


def test_completion_supplies_the_training_and_offline_evaluation_interface():
    trainer = source('experiments/donut_material_recovery_v1.py')
    analyst = source('experiments/analyze_donut_heterogeneous_training_v1.py')
    required = adapter_members(trainer, {'hetero', 'acquisition'}) | adapter_members(analyst, {'spec'})
    # Distance-weighted histories and smooth-normal Donut use different protocols.
    required -= {'edge_diagnostic', 'audit_weighted_history', 'check_donut_domain'}
    required.update(('ARMS', 'ROOT', 'FREEZE', 'IMAGE', 'SPATIAL_POLICY'))
    assert sorted(name for name in required if not hasattr(completion, name)) == []
    assert completion.TARGET_KEYS is known.TARGET_KEYS
    assert completion.point_errors is known.point_errors
    assert completion.validate_map is known.validate_map
    assert completion.ARMS == ('K2', 'K3')


@pytest.mark.parametrize('arm', completion.ARMS)
def test_native_endpoint_evaluator_with_completion_adapter(tmp_path, monkeypatch, arm):
    torch = pytest.importorskip('torch')
    from benchmarks.donut_tensor_d8 import REPRESENTATION
    from scientific_variants import donut_tensor_d8 as tensor_math

    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda _: None)
    points = np.zeros((1, 3), dtype=np.float32)
    normals = np.array([[0., 0., 1.]], dtype=np.float32)
    widths = np.array([[.25, .125]], dtype=np.float32)
    color = np.full((1, 3), .375, dtype=np.float32)
    metal = np.full((1, 1), .5, dtype=np.float32)
    calls = []

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32)

    class ShaderFixture(torch.nn.Module):
        def predict_tensor(self, points, normals):
            local = torch.diag_embed(tensor(widths).double().square()).expand(len(points), 2, 2)
            return local, tensor_math.frame(normals), tensor(color), tensor(metal)

        def predict_outer_lights(self, points, directions):
            return environment(directions, ENVIRONMENT)

    shader = ShaderFixture()

    def shade(p, wo, n, step, training, seed, *, export):
        assert (step, training) == (0, False)
        calls.append((seed, export))
        q, u, kd, metallic = shader.predict_tensor(p, n)
        mx, my, t, b = tensor_math.export_axes(q, u)
        rotation = torch.tensor([[1., 0.]])
        return dict(kd=kd, metallic=metallic, mx=mx, my=my, rotation=rotation, rotation_raw=rotation,
                    tangents=(t+1)/2, bitangents=(b+1)/2, normals=(n+1)/2, Q=tensor_math.world_tensor(q, u),
                    linear_rgb=kd, linear_diffuse=kd/2, linear_specular=kd/2)

    q, u, _, _ = shader.predict_tensor(tensor(points), tensor(normals))
    world_q = tensor_math.world_tensor(q, u).numpy()
    saved_rgb = np.stack([color/2, color/2, color], axis=1)
    view = dict(position=points, normals=normals, outgoing=normals, shader_valid=np.array([True]),
                pixel_indices=np.array([2]), hit_mask=np.array([[False, False], [True, False]]),
                linear_components=saved_rgb, axis=u[..., 0].numpy(), Q_world=world_q,
                widths_major_minor=widths, color=color, metallic=metal)
    counts = np.array([0])
    report = dict(saved_endpoints=[], effective_lighting='known analytic L2')
    checkpoints = []
    boundary_work = []

    def no_state_changes(*args):
        return nullcontext()

    scope = dict(
        args=SimpleNamespace(output=tmp_path, phase=arm), known_audit=SimpleNamespace(checkpoint=lambda step, *_: checkpoints.append(step)),
        shader=shader, optimizer=None, lighting_anchor=None, torch=torch, np=np, tensor=tensor, SIZE=2,
        evaluation_scope=no_state_changes, native_tensor_default=nullcontext, calibration_scope=nullcontext,
        light_path=nullcontext, calibration=False, shade=shade, data={0: view}, field=SimpleNamespace(linear_to_srgb=lambda x: x),
        tensor_plan=dict(representation=REPRESENTATION), initial=dict(variant=known.MATCH[arm]), write_view=write_view,
        require=require, heterogeneous=True, hetero=completion, acquisition=completion, splits=dict(fitting=[0]),
        pair_points=[points, points], pair_frames=[view, view], pair_ix=np.array([0]), spatial=True,
        tensor_math=tensor_math, normal_transport=normal_transport, log_tensor=False, faceted=True,
        boundary=dict(positions=[points, points], normals=normals),
        budget=lambda key, n: boundary_work.append((key, n)), charge=lambda key, n: boundary_work.append((key, n)),
        parity_plan=dict(D6_sampler_snapshots={'0': array_hash(counts)}), array_hash=array_hash,
        sampler=SimpleNamespace(counts=counts), coverage_summary=coverage_summary,
        domain=dict(face_pairs=np.array([[0, 1]]), face_area=np.ones(2)), fitting_counts=np.array([1, 0]),
        illumination_preview=lambda path, light: {}, json_new=json_new, report=report, progress=lambda: None)
    trainer = source('experiments/donut_material_recovery_v1.py')
    train = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    evaluate = next(node for node in ast.walk(train) if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    exec(compile(ast.Module(body=[evaluate], type_ignores=[]), '<native endpoint fixture>', 'exec'), scope)
    scope['evaluate'](0)

    assert checkpoints == [0] and shader.training
    assert calls == [(800000+k, k == 7) for k in range(8)]
    assert boundary_work == [('boundary_predictor_calls', 1), ('boundary_predictor_points', 1)] * 2
    destination = tmp_path / 'step-00000'
    metrics = completion.b.read(destination / 'metrics.json')
    assert report['saved_endpoints'] == [dict(step=0, metrics=metrics)]
    assert metrics['0']['physical']['Q_at_most_005'] == 1
    assert metrics['0']['RGB_linear_RMSE'] == 0
    assert metrics['illumination']['analytic_agreement']['relative_L2'] == 0
    with np.load(destination / 'camera-0-errors.npz', allow_pickle=False) as errors:
        assert errors['Q_relative'].tolist() == [0]
        assert errors['orientation_degrees'].tolist() == [0]
    for name in ('boundary-predictions.npz', 'surface-q-diagnostic.npz', 'illumination.npz', 'camera-0/view.json'):
        assert (destination / name).is_file()
