"""E7 synthetic projection/restoration tests; no original field or shader."""
import ast
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments import hybrid_material_e7_v1 as pilot
from experiments import neural_projection_e7_v1 as projection
from experiments import neural_material_continuation_e3 as restorer

FORBIDDEN = ('experiments.neural_material_field_v1', 'experiments.material_shader_v1',
             'experiments.neural_material_recovery_v1', 'experiments.run_known_material_recovery_v1',
             'experiments.anisotropic_estimator_v1')


@pytest.fixture(autouse=True)
def synthetic_only():
    assert not any(n.startswith(FORBIDDEN) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.startswith(FORBIDDEN): raise AssertionError('Scientific import forbidden: '+fullname)
    block = Block(); sys.meta_path.insert(0, block); old = torch.get_rng_state().clone(); nrng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]): yield
    finally:
        sys.meta_path.remove(block)
        assert torch.equal(old, torch.get_rng_state())
        after = np.random.get_state()
        assert nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:]
        assert not any(n.startswith(FORBIDDEN) for n in sys.modules)


class ToyHead(torch.nn.Module):
    def __init__(self, width):
        super().__init__(); self.predictor = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Sigmoid())
        self.register_buffer('center', self.predictor[0](torch.ones(1, 2)).detach().clone())
        self.register_buffer('target', torch.full((1, width), .2))
    def forward(self, x):
        return self.predictor[-1](self.target+(self.predictor[0](x)-self.center))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name, width in projection.HEADS: setattr(self, name, ToyHead(width))
        self.register_buffer('anchor', torch.ones(1, 2))
    def forward(self, positions):
        q = [getattr(self, n)(self.anchor) for n, _ in projection.HEADS]
        e = torch.nn.functional.normalize(2*q[3]-1, dim=-1)
        theta = .5*torch.atan2(e[:, 1:2], e[:, :1])
        p = torch.cat((.04+.96*q[0], .04+.96*q[1], theta, q[4], q[2]), -1)
        return SimpleNamespace(physical=p.expand(len(positions), 7), encoding=e.expand(len(positions), 2))


def teacher(): return torch.tensor([.31, .62, -.48, .3, .4, .5, .6])


def test_projection_logit_derivatives_and_identity_bias_jacobian():
    p = teacher().double().requires_grad_()
    assert torch.autograd.gradcheck(projection.target_logits, (p,), atol=1e-7, rtol=1e-5)
    model = Toy(); _, logits = projection.capture(model, torch.zeros(1, 3))
    biases = projection.bias_parameters(model)
    for i, (name, width) in enumerate(projection.HEADS):
        out = getattr(model, name)(model.anchor)
        logit = torch.log(out/(1-out))
        key = name+'.predictor.0.bias'
        j = torch.stack([torch.autograd.grad(logit[0, k], biases[key], retain_graph=True)[0] for k in range(width)])
        torch.testing.assert_close(j, torch.eye(width))
    assert logits.shape == (8,)


def test_realized_projection_preserves_all_other_state_and_has_no_saturation():
    model = Toy(); before = {n: t.detach().clone() for n, t in model.state_dict().items()}
    _, old = projection.capture(model, torch.zeros(9, 3)); target = teacher()
    with projection.bias_transaction(model, old, target) as txn:
        mat, logits = projection.capture(model, torch.zeros(9, 3))
        assert projection.realization(mat.physical[0].numpy(), mat.encoding[0].numpy(), target.numpy())['passed']
        assert torch.all((torch.sigmoid(logits) > 0) & (torch.sigmoid(logits) < 1))
        assert len(txn['bias_names']) == 5 and len(txn['delta']) == 8
        txn['commit']()
    for name, value in model.state_dict().items():
        if name not in txn['bias_names']: assert torch.equal(before[name], value)
    assert torch.equal(mat.physical, mat.physical[:1].expand_as(mat.physical))


@pytest.mark.parametrize('failure', ['reject', 'exception', 'exception_after_commit'])
def test_rejected_or_interrupted_projection_restores_exact_biases(failure):
    model = Toy(); before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _, old = projection.capture(model, torch.zeros(1, 3))
    try:
        with projection.bias_transaction(model, old, teacher()) as txn:
            if failure == 'exception_after_commit': txn['commit']()
            if failure != 'reject': raise RuntimeError('synthetic interruption')
    except RuntimeError: pass
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert all(not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize('index,value', [(0, .04), (1, 1.), (3, 0.), (6, 1.), (2, float('nan'))])
def test_invalid_teacher_fails_without_clipping(index, value):
    p = teacher(); p[index] = value
    with pytest.raises(Exception): projection.target_logits(p)


def test_gauge_behavior_retains_teacher_and_near_isotropic_orientation():
    t = teacher().numpy(); swapped = t[[1, 0, 2, 3, 4, 5, 6]].copy(); swapped[2] += np.pi/2
    e = np.array([np.cos(2*swapped[2]), np.sin(2*swapped[2])])
    # Q is equivalent, but the finite-tape sampling gauge must NOT be replaced.
    result = projection.realization(swapped, e, t)
    assert result['relative_Q_error'] < 1e-6 and not result['passed']
    t[1] = t[0]+1e-8
    e = [np.cos(2*t[2]), np.sin(2*t[2])]
    assert projection.realization(t, e, t)['passed']
    assert not projection.realization(t, [0., 0.], t)['passed']
    shifted = t.copy(); shifted[3] += 1e-4
    assert not projection.realization(shifted, e, t)['passed']


def test_acceptance_uses_fitting_objective_and_realization_only():
    assert projection.accept(1e-5, 1e-6, True)
    assert not projection.accept(1e-5, 1e-6, False)
    assert not projection.accept(1e-5, 1e-5, True)
    assert not projection.accept(1., 1.-1e-8, True)
    with pytest.raises(Exception): projection.accept(1., float('nan'), True)


def synthetic_state(model, optimizer):
    for p in model.parameters():
        optimizer.state[p] = dict(step=torch.tensor(100.), exp_avg=torch.full_like(p, .002), exp_avg_sq=torch.full_like(p, .003))
    return restorer.prior.state_arrays(model, optimizer)


@pytest.mark.parametrize('corruption', [None, 'step', 'shape', 'identity', 'options'])
def test_complete_numeric_state100_restore_or_fail_closed(corruption):
    model = Toy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    arrays, rng = synthetic_state(model, optimizer); options = restorer.options(optimizer)
    expected = pilot.obs.parameter_identity({n[10:]: a for n, a in arrays.items() if n.startswith('parameter/')})
    new = Toy(); fresh = torch.optim.Adam(new.parameters(), lr=.001)
    first = next(iter(dict(model.named_parameters())))
    if corruption == 'step': arrays['adam/'+first+'/step'] = np.array(99., np.float32)
    if corruption == 'shape': arrays['parameter/'+first] = np.zeros(99, np.float32)
    if corruption == 'identity': expected = 'changed'
    if corruption == 'options': options['lr'] = .01
    old_start = restorer.START
    if corruption:
        with pytest.raises(Exception): projection.restore100(new, fresh, arrays, rng, options, expected)
        assert not fresh.state
    else:
        report = projection.restore100(new, fresh, arrays, rng, options, expected)
        actual, meta = restorer.prior.state_arrays(new, fresh)
        assert report['exact'] and report['Adam_step'] == 100 and meta == rng
        assert all(np.array_equal(actual[n], a) for n, a in arrays.items())
    assert restorer.START == old_start


def test_cached_control_gradient_survives_read_only_projection_probes():
    a = Toy(); b = Toy(); b.load_state_dict(a.state_dict())
    oa = torch.optim.Adam(a.parameters(), lr=.001); ob = torch.optim.Adam(b.parameters(), lr=.001)
    for model in (a, b): model(torch.zeros(1, 3)).physical.square().sum().backward()
    grad = {n: p.grad.clone() for n, p in b.named_parameters()}
    projection.capture(b, torch.zeros(9, 3))
    assert all(torch.equal(p.grad, grad[n]) for n, p in b.named_parameters())
    oa.step(); ob.step()  # Two tiny synthetic optimizer calls, not scientific work.
    assert all(torch.equal(p, dict(b.named_parameters())[n]) for n, p in a.named_parameters())


def test_caps_check_before_counting_and_native_counter_mismatch():
    w = pilot.Work(); w.attempts['Adam_returns'] = w.done['Adam_returns'] = 277
    before = dict(w.attempts)
    with pytest.raises(Exception, match='cap'): w.attempt('Adam_returns')
    assert w.attempts == before
    w.attempt('field_probes')
    with pytest.raises(Exception, match='Incomplete'): w.verify()
    w.complete('field_probes'); w.verify()
    with pytest.raises(Exception, match='mismatch'): pilot.verify_native(w, [])
    assert pilot.CAPS['neural_forwards']+pilot.CAPS['diagnostic_forwards'] == 301
    assert (4*115+11, 4*162+11) == (471, 659)


def test_native_completed_panel_and_rejected_projection_short_panel():
    w = pilot.Work(); w.attempts.update(pilot.CAPS); w.done.update(pilot.CAPS)
    native = []
    for i in range(12):
        n = dict(shader_attempts=0, shader_completed=0, reverse_attempts=0, reverse_completed=0,
                 optimizer_attempts=100 if i < 6 else 0, optimizer_returns=100 if i < 6 else 0,
                 calibration_forwards=int(i < 6))
        native.append(n)
    native[0].update(shader_attempts=293, shader_completed=293, reverse_attempts=283, reverse_completed=283,
                     optimizer_attempts=377, optimizer_returns=377)
    pilot.verify_native(w, native)
    # Failed realization may legitimately omit its trial F/B, not invent work.
    for k in ('neural_forwards', 'backwards'): w.attempts[k] -= 1; w.done[k] -= 1
    for k in ('shader_attempts', 'shader_completed', 'reverse_attempts', 'reverse_completed'): native[0][k] -= 1
    pilot.verify_native(w, native)
    native[0]['reverse_attempts'] += 1
    with pytest.raises(Exception, match='Incomplete native'): pilot.verify_native(w, native)


def test_capture_exception_removes_instrumentation():
    model = Toy()
    def fail(*args): raise RuntimeError('synthetic field failure')
    model.forward = fail
    with pytest.raises(RuntimeError): projection.capture(model, torch.zeros(1, 3))
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_execution_gate_and_accepted_sources_unchanged(tmp_path):
    path = tmp_path/'review.json'; path.write_text('{}\n')
    with pytest.raises(Exception, match='approval'): pilot.execute_reviewed(path, pilot.sha(path), '0'*64)
    parent = json.loads((pilot.ROOT/pilot.prior.FREEZE).read_text())
    for name, digest in parent['source_sha256'].items(): assert pilot.sha(pilot.ROOT/name) == digest
    source = Path(pilot.__file__).read_text(); tree = ast.parse(source)
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_reviewed')
    attempt = next(n for n in execute.body if isinstance(n, ast.Try))
    assert isinstance(attempt.body[0], ast.With)
    assert 'cleanup_verified' in ast.unparse(attempt.body[1])
    assert 'scientific_work_completed=True' not in ast.unparse(attempt.body[0])
    assert '--execute' not in source
