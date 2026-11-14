"""New E6 synthetic optimizer tests only; scientific shader/neural imports blocked."""
import ast
import importlib.abc
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from experiments import physical_lm_e6_v1 as lm
from experiments import material_optimizer_e6_v1 as pilot


@pytest.fixture(autouse=True)
def no_scientific_imports():
    forbidden = ('experiments.material_shader_v1', 'experiments.neural_material_field_v1',
                 'experiments.neural_material_recovery_v1', 'experiments.run_known_material_recovery_v1',
                 'experiments.anisotropic_estimator_v1')
    assert not any(n.startswith(forbidden) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, *args):
            if name.startswith(forbidden): raise AssertionError('Scientific module forbidden: '+name)
    block = Block(); sys.meta_path.insert(0, block); old = torch.get_rng_state().clone(); n = np.random.get_state()
    try: yield
    finally:
        sys.meta_path.remove(block); assert torch.equal(old, torch.get_rng_state())
        after = np.random.get_state(); assert n[0] == after[0] and np.array_equal(n[1], after[1]) and n[2:] == after[2:]
        assert not any(s.startswith(forbidden) for s in sys.modules)


def test_scaled_total_residual_JVP_and_independent_finite_difference():
    p = torch.tensor([.3, .6, .2, .3, .4, .5, .6], dtype=torch.float64)
    a = torch.arange(1, 505, dtype=torch.float64).reshape(72, 7)/504
    f = lambda x: a@torch.sin(x)+.1*(a@x)**2
    work = lm.Work('LM'); pending = {}
    def counted(x, phase):
        work.attempt('forward', phase); y = f(x); work.complete('forward', phase); return y
    j = pilot.previous.columns(counted, p, f(p), work, pending)
    independent = torch.autograd.functional.jacobian(f, p).numpy()*lm.SCALES
    np.testing.assert_allclose(j, independent, rtol=1e-12, atol=1e-14)
    h = 1e-5
    for i in range(7):
        d = torch.zeros(7, dtype=torch.float64); d[i] = h*lm.SCALES[i]
        np.testing.assert_allclose(j[:, i], ((f(p+d)-f(p-d))/(2*h)).numpy(), rtol=1e-8, atol=1e-9)
    assert work.done['fit_forward'] == 7 and work.done['reverse'] == 14


@pytest.mark.parametrize('boundary', [.04001, .99999])
def test_cholesky_scaling_trust_radius_bounds_and_represented_prediction(boundary):
    p = np.array([boundary, .4, .2, boundary, .4, .5, boundary], np.float32)
    j = np.vstack((np.eye(7), np.zeros((65, 7)))); r = np.ones(72)*(-3 if boundary > .9 else 3)
    candidate, d = lm.proposal(p, j, r, .01); lm.valid(candidate)
    step = (candidate.astype(float)-p.astype(float))/lm.SCALES
    np.testing.assert_array_equal(step, d['represented_scaled_delta'])
    assert np.linalg.norm(step) <= .1+1e-6 and 0 <= d['boundary_fraction'] <= 1
    assert d['predicted_reduction'] == float(-2*(j.T@r)@step-step@(j.T@j)@step)
    assert d['solve_residual_L2'] < 1e-12


def test_ideal_axis_permutation_covariance_does_not_canonicalize_queries():
    p = np.array([.3, .6, .2, .3, .4, .5, .6], np.float32); swapped = p[[1, 0, 2, 3, 4, 5, 6]].copy(); swapped[2] -= math.pi/2
    a = np.vstack((np.diag(np.arange(1, 8)), np.zeros((65, 7)))); r = np.linspace(-.3, .5, 72)
    x, detail = lm.proposal(p, a, r, .01)
    order = [1, 0, 2, 3, 4, 5, 6]; y, other = lm.proposal(swapped, a[:, order], r, .01)
    np.testing.assert_allclose(np.array(detail['raw_scaled_delta'])[order], other['raw_scaled_delta'], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(x[:2][::-1], y[:2], rtol=0, atol=0)
    assert abs(float(y[2])-float(x[2])+math.pi/2) < 2e-7
    # Actual finite-tape shader symmetry is NOT assumed or queried by this test.


def test_acceptance_damping_and_machine_precision_floor():
    take, rho, _ = lm.acceptance(1., .8, .2); assert take and abs(rho-1) < 1e-14
    assert not lm.acceptance(1., 1.1, .2)[0]
    assert not lm.acceptance(1., 1., .2)[0]
    assert not lm.acceptance(1., .8, -1)[0]
    assert not lm.acceptance(1., 1.-1e-8, 1e-8)[0]
    assert lm.next_damping(.01, True, .9) == .01/3
    assert lm.next_damping(.01, True, .2) == .02
    assert lm.next_damping(.01, False, None) == .1
    assert lm.next_damping(1e8, False, None) == 1e8


@pytest.mark.parametrize('changed', ['physical', 'radiance', 'objective'])
def test_paired_admission_accepts_only_identical_states_and_objectives(changed):
    p = np.array([.4, .5, 0., .3, .4, .5, .6], np.float32); y = np.zeros((24, 3, 3), np.float32)
    assert lm.verify_start(p, p.copy(), y, y.copy(), .1, .1)
    other = p.copy(); predictions = y.copy(); loss = .1
    if changed == 'physical': other[0] = np.nextafter(other[0], np.float32(1.))
    if changed == 'radiance': predictions[0, 0, 0] = .001
    if changed == 'objective': loss = .1000000001
    with pytest.raises(Exception, match='differ'): lm.verify_start(p, other, y, predictions, .1, loss)


def test_cost_matching_includes_reverse_and_all_evaluations_without_selecting_scores():
    for cost in range(15, 1022):
        n = lm.matched_adam_updates(cost)
        assert lm.adam_cost(n) <= cost and (n == 250 or lm.adam_cost(n+1) > cost)
    assert lm.matched_adam_updates(1013) == 248
    assert lm.matched_adam_updates(1021) == 250
    assert lm.adam_cost(20) == 91 and lm.adam_cost(21) == 102
    # Complete target comparisons have n>20 and only 1 or 3 half-units unmatched.
    for fit in range(443, 508):
        cost = 2*fit+7; n = lm.matched_adam_updates(cost)
        assert n > 20 and cost-lm.adam_cost(n) in (1, 3)
    with pytest.raises(Exception, match='too small'): lm.matched_adam_updates(1)


def test_rejected_caps_do_not_increment_work_and_partial_work_is_not_complete():
    w = lm.Work('LM'); w.attempts['reverse'] = w.done['reverse'] = 281
    before = json.loads(json.dumps(w.attempts))
    with pytest.raises(Exception, match='cap'): w.attempt('reverse', 'JVP')
    assert w.attempts == before
    w.attempt('forward', 'trial')
    with pytest.raises(Exception, match='Incomplete'): w.verify()
    w.complete('forward', 'trial'); w.verify()


class FakeCore:
    """Explicit scalar-exponential fixture and fake native counters, not a renderer."""
    def __init__(self, reject=False):
        self.p = np.array([.4, .5, 0., .3, .4, .5, .6], np.float32); self.work = lm.Work('LM'); self.reject = reject
        self.current = self.forward(self.p, initial=True)
    def forward(self, p, initial=False):
        self.work.attempt('forward', 'initial' if initial else 'trial')
        r = np.zeros(72); r[0] = np.exp(float(p[2])); value = dict(loss=float(np.float32(r@r)), residual=r)
        if self.reject and not initial: value['loss'] = self.current['loss']+1
        self.work.complete('forward', 'initial' if initial else 'trial'); return value
    def jacobian(self):
        for _ in range(7):
            self.work.attempt('forward', 'jacobian'); self.work.complete('forward', 'jacobian')
            for _ in range(2): self.work.attempt('reverse', 'JVP'); self.work.complete('reverse', 'JVP')
        j = np.zeros((72, 7)); j[0, 2] = math.pi*np.exp(float(self.p[2])); return j


def test_exact_twenty_successful_LM_updates_and_initial_state_preservation():
    core = FakeCore(); initial = core.p.copy(); records = []
    result = lm.run_lm(core, lambda row, j: records.append(row))
    assert result['status'] == 'complete' and result['accepted_updates'] == len(records) == 20
    assert core.work.events['accepted_steps'] == 20
    assert all(r['loss_after'] < r['loss_before'] for r in records)
    np.testing.assert_array_equal(initial[[0, 1, 3, 4, 5, 6]], core.p[[0, 1, 3, 4, 5, 6]])
    assert core.work.done['fit_forward'] == 161 and core.work.done['reverse'] == 280
    core.work.verify()


def test_four_rejections_are_counted_and_stop_without_extra_Jacobian_or_retry():
    core = FakeCore(reject=True); initial = core.p.copy(); records = []
    result = lm.run_lm(core, lambda row, j: records.append(row))
    assert result['status'] == 'censored_algorithm' and result['termination'] == 'four_trial_limit'
    assert result['accepted_updates'] == 0 and not result['matched_update_available']
    assert len(records) == 1 and len(records[0]['trials']) == 4
    assert core.work.events['rejected_trials'] == core.work.events['trials'] == core.work.events['solves_completed'] == 4
    assert core.work.events['jacobians_completed'] == 1 and core.work.done['fit_forward'] == 12
    np.testing.assert_array_equal(core.p, initial); core.work.verify()


def test_invalid_system_fails_without_alternate_solve_or_domain_repair():
    p = np.array([.4, .5, 0., .3, .4, .5, .6], np.float32)
    with pytest.raises(Exception, match='system'): lm.proposal(p, np.full((72, 7), np.nan), np.ones(72), .01)
    p[0] = .03
    with pytest.raises(Exception, match='bounded'): lm.proposal(p, np.ones((72, 7)), np.ones(72), .01)


def test_observing_physical_gradient_does_not_change_original_Adam_update():
    start = torch.tensor([.2, -.7, .3, .6, -.1, .5, .4], dtype=torch.float32)
    a = start.clone().requires_grad_(); b = start.clone().requires_grad_()
    options = dict(lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    oa = torch.optim.Adam([a], **options); ob = torch.optim.Adam([b], **options)
    pa, pb = torch.sigmoid(a), torch.sigmoid(b)
    la = ((pa-.3)**2).sum(); lb = ((pb-.3)**2).sum()
    ga, = torch.autograd.grad(la, a); gb, physical = torch.autograd.grad(lb, (b, pb))
    assert torch.equal(ga, gb) and torch.isfinite(physical).all()
    a.grad = ga; b.grad = gb; oa.step(); ob.step()
    assert torch.equal(a, b)
    for name in ('step', 'exp_avg', 'exp_avg_sq'): assert torch.equal(oa.state[a][name], ob.state[b][name])


def test_execution_gate_blocks_real_imports_and_optimizer_has_no_oracle(tmp_path):
    path = tmp_path/'review.json'; path.write_text('{}\n')
    with pytest.raises(Exception, match='approval'): pilot.execute_reviewed(path, pilot.sha(path), '0'*64)
    tree = ast.parse(Path(lm.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert not any(isinstance(n, ast.Name) and n.id in ('truth', 'heldout', 'reference') for n in ast.walk(node))
    assert '--execute' not in Path(pilot.__file__).read_text()


def test_cleanup_and_terminal_snapshot_binding_are_explicit():
    tree = ast.parse(Path(pilot.__file__).read_text())
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_reviewed')
    attempt = next(n for n in execute.body if isinstance(n, ast.Try)); assert isinstance(attempt.body[0], ast.With)
    assert 'cleanup_verified' in ast.unparse(attempt.body[1]) and 'scientific_work_completed=True' not in ast.unparse(attempt.body[0])
    assert 'final_state=store.numeric' in Path(pilot.__file__).read_text()
