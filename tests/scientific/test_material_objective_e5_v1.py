"""E5 synthetic algebra/accounting only; real shader and neural imports blocked."""
import ast
import importlib.abc
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments import material_objective_e5_v1 as e5


@pytest.fixture(autouse=True)
def no_scientific_execution():
    forbidden = ('experiments.material_shader_v1', 'experiments.neural_material_field_v1',
                 'experiments.neural_material_recovery_v1', 'experiments.run_known_material_recovery_v1',
                 'experiments.anisotropic_estimator_v1')
    assert not any(n.startswith(forbidden) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, *args):
            if name.startswith(forbidden): raise AssertionError('Scientific import forbidden: '+name)
    block = Block(); sys.meta_path.insert(0, block)
    t = torch.get_rng_state().clone(); n = np.random.get_state()
    try: yield
    finally:
        sys.meta_path.remove(block)
        assert torch.equal(t, torch.get_rng_state())
        after = np.random.get_state()
        assert n[0] == after[0] and np.array_equal(n[1], after[1]) and n[2:] == after[2:]
        assert not any(s.startswith(forbidden) for s in sys.modules)


def toy(p):
    # Seven distinct smooth directions in 216 synthetic observations; no BRDF.
    weights = torch.arange(1, 1513, dtype=p.dtype).reshape(216, 7)/1512
    return (weights@torch.sin(p)+.01*(weights@p)**2).reshape(24, 3, 3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scaled_derivatives_match_autodiff_and_both_fixed_steps(dtype):
    p = torch.tensor([.3, .6, -.4, .2, .4, .7, .3], dtype=dtype)
    work = e5.Work(); pending = {}
    def f(p, phase):
        work.attempt('forward', phase); y = toy(p); work.complete('forward', phase); return y
    j = e5.columns(f, p, toy(p), work, pending)
    fd = e5.finite_differences(f, p, pending)
    independent = torch.autograd.functional.jacobian(toy, p).numpy()*e5.SCALES
    np.testing.assert_allclose(j, independent, atol=1e-6, rtol=1e-5)
    assert e5.derivative_agreement(j, fd)['passed']
    assert work.done['forward']['jacobian'] == 7 and work.done['forward']['finite_difference'] == 28
    assert work.done['reverse']['JVP'] == 14


def test_original_direct_transform_derivative_without_importing_runner():
    path = e5.ROOT/'experiments/run_known_material_recovery_v1.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'physical')
    ns = {'torch': torch}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    z = torch.tensor([.2, -.8, .7, .4, -.3, -.9, .6], dtype=torch.float64, requires_grad=True)
    p = ns['physical'](z); j = torch.autograd.functional.jacobian(ns['physical'], z)
    q = torch.sigmoid(z.detach()); diagonal = q*(1-q); diagonal[:2] *= .96; diagonal[2] = 1
    assert torch.allclose(j, torch.diag(diagonal), atol=1e-15, rtol=1e-13)
    assert torch.autograd.gradcheck(ns['physical'], (z,))
    assert torch.equal(p[2], z[2])


def q_vector(p):
    c, s = torch.cos(p[2]), torch.sin(p[2]); r = torch.stack((c, -s, s, c)).reshape(2, 2)
    q = r@torch.diag(p[:2]**2)@r.T
    return torch.stack((q[0, 0], math.sqrt(2)*q[0, 1], q[1, 1]))


def test_gauge_equivalence_and_isotropic_null_angle_without_frame_substitution():
    p = torch.tensor([.3, .6, .7, .2, .4, .7, .3], dtype=torch.float64)
    swapped = p.clone(); swapped[:2] = p[:2].flip(0); swapped[2] -= math.pi/2
    periodic = p.clone(); periodic[2] += math.pi
    assert torch.allclose(q_vector(p), q_vector(swapped), atol=1e-15, rtol=1e-14)
    assert torch.allclose(q_vector(p), q_vector(periodic), atol=1e-15, rtol=1e-14)
    p[:2] = .4
    j = torch.autograd.functional.jacobian(q_vector, p)
    assert torch.max(abs(j[:, 2])) < 1e-16
    # Reporting-only permutation agrees for ideal gauge-equivalent Q Jacobians.
    def fake_observations(v): return q_vector(v).reshape(1, 1, 3).expand(24, 3, 3)
    p[:2] = torch.tensor([.3, .6]); swapped[:2] = p[:2].flip(0)
    ja = torch.autograd.functional.jacobian(fake_observations, p).numpy()*e5.SCALES
    jb = torch.autograd.functional.jacobian(fake_observations, swapped).numpy()*e5.SCALES
    a = e5.sensitivity(ja, p.numpy()); b = e5.sensitivity(jb, swapped.numpy())
    np.testing.assert_allclose(a['total']['singular_values'], b['total']['singular_values'], atol=1e-14)
    assert a['reporting_column_order'] == [0, 1, 2, 3, 4, 5, 6]
    assert b['reporting_column_order'] == [1, 0, 2, 3, 4, 5, 6]


def test_fixed_samples_are_reproducible_independent_and_match_accepted_identity():
    a = e5.tape_arrays(e5.TAPE_SEEDS['accepted'], 24)
    b = e5.tape_arrays(e5.TAPE_SEEDS['accepted'], 24)
    independent = e5.tape_arrays(e5.TAPE_SEEDS['independent_1'], 24)
    assert e5.tape_identity(a) == e5.tape_identity(b)
    assert e5.tape_identity(a) != e5.tape_identity(independent)
    plan = json.loads((e5.ROOT/'docs/protocols/MATERIAL_PARAMETERIZATION_E4_V1.freeze.json').read_text())
    assert e5.tape_identity(a) == plan['tapes']['fit']['arrays']
    for x in a.values(): assert x.dtype == np.float32 and np.all((x >= 0) & (x < 1))


def test_all_phase_caps_and_rejected_call_accounting():
    w = e5.Work()
    for kind, phases in (('forward', e5.FORWARDS), ('reverse', e5.REVERSES), ('update', {'direct': e5.UPDATES})):
        for phase, count in phases.items():
            for _ in range(count): w.attempt(kind, phase); w.complete(kind, phase)
            before = json.loads(json.dumps(w.attempts))
            with pytest.raises(Exception, match='cap'): w.attempt(kind, phase)
            assert w.attempts == before
    w.verify_complete()
    assert sum(w.done['forward'].values()) == 385 and sum(w.done['reverse'].values()) == 186
    assert w.done['update']['direct'] == 50
    w.done['update']['direct'] -= 1
    with pytest.raises(Exception, match='incomplete'): w.verify_complete()


def test_derivative_disagreement_and_zero_sensitivity_are_not_success():
    j = np.zeros((24, 3, 3, 7)); fd = np.zeros((2, *j.shape)); fd[1, 0, 0, 0, 2] = .1
    assert not e5.derivative_agreement(j, fd)['passed']
    spectrum = e5.spectra(np.zeros((72, 7)))
    assert spectrum['effective_rank'] == 0 and spectrum['condition_ratio'] is None
    assert not e5.sensitivity(j, [.4, .4, .2, .3, .3, .3, .4])['angle_scored']


def test_component_cancellation_is_distinct_from_material_identifiability():
    p = np.ones((24, 3, 3)); t = p.copy(); p[:, 0] += .2; p[:, 1] -= .2
    result = e5.component_errors(p, t)
    assert result['component_squared_error'] > 0 and result['total_squared_error'] == 0
    assert abs(result['normalized_residual_inner_product']+1) < 1e-14


def test_archived_angle_and_material_gradient_chain_against_synthetic_autodiff():
    theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    encoding = torch.stack((torch.cos(2*theta), torch.sin(2*theta))).repeat(4, 1)
    loss = (encoding[:, 0]**2+2*encoding[:, 1]).sum()
    eg, gt = torch.autograd.grad(loss, (encoding, theta))
    signals = {'shader.encoding': encoding.detach().numpy(), 'shader.encoding.loss_gradient': eg.numpy()}
    for name, width in (('mx', 1), ('my', 1), ('metallic', 1), ('albedo', 3)):
        signals['shader.'+name+'.loss_gradient'] = np.full((4, width), .2)
    g = e5.archived_physical_gradient(signals)
    np.testing.assert_allclose(g['physical_gradient'], [.8, .8, float(gt), .8, .8, .8, .8], atol=1e-14)


def test_missing_review_fails_before_any_scientific_import(tmp_path):
    path = tmp_path/'review.json'; path.write_text('{}\n')
    with pytest.raises(Exception, match='approval'): e5.execute_reviewed(path, e5.sha(path), '0'*64)


def test_observer_is_read_only_and_archived_encoding_override_is_explicit():
    class FakeShader:
        def shade(self, *args, **kwargs):
            self.args = args; self.kwargs = kwargs
            x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
            return dict(diffuse_rgb=x, specular_rgb=2*x, linear_rgb=3*x)
    fake = FakeShader(); wrapper = e5.CapturedShader(fake)
    args = tuple(object() for _ in range(10)); result = wrapper.shade(*args, is_train=True)
    assert fake.args == args and fake.kwargs == dict(is_train=True) and wrapper.calls == 1
    assert wrapper.per_tape.shape == (4, 1, 3, 3)
    assert torch.equal(result['linear_rgb'], torch.arange(12).reshape(4, 3)*3)
    encoding = torch.ones((4, 2)); replay = e5.CapturedShader(fake, encoding)
    replay.shade(*args)
    assert fake.args[:7] == args[:7] and fake.args[7] is encoding and fake.args[8:] == args[8:]


def test_runtime_cleanup_acknowledgement_is_after_normal_exit_and_no_execution_cli():
    tree = ast.parse(Path(e5.__file__).read_text())
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_reviewed')
    attempt = next(n for n in execute.body if isinstance(n, ast.Try))
    assert isinstance(attempt.body[0], ast.With)
    clean = attempt.body[1]
    assert isinstance(clean, ast.Assign) and isinstance(clean.value, ast.Constant) and clean.value.value is True
    assert 'cleanup_verified' in ast.unparse(clean)
    assert 'scientific_work_completed=True' not in ast.unparse(attempt.body[0])
    assert '--execute' not in Path(e5.__file__).read_text()
