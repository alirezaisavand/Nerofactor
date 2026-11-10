"""NEW restoration/control fixtures only; no scientific shader or prior experiments."""
import ast
import copy
import json
from pathlib import Path
import platform
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import material_recovery_continuation_v1 as run
from infrastructure.signatures import SignatureError


def toy_state():
    z = torch.tensor([-.2, .3, .4, -.5, .6, .7, -.8], requires_grad=True)
    opt = torch.optim.Adam([z], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    for i in range(250):
        opt.zero_grad(); ((z - .2)**2).sum().backward(); opt.step()
    row = dict(seed=101, status='complete', completed_updates=250,
               last_unconstrained=z.detach().tolist(), final=run.original.physical(z).detach().tolist(),
               last_physical=run.original.physical(z).detach().tolist(),
               optimizer_final_state={k: v.detach().tolist() for k, v in opt.state[z].items()},
               optimizer_effective=run.optimizer_options(opt))
    return json.loads(json.dumps(row)), z, opt


class ToyBatch:
    n, seeds = 24, 4

    def __call__(self, shader, p):
        return p.mean().expand(24, 3, 3)


def budget():
    return dict(optimizer_returns=0, fit_shader_points=0, fit_specular_directions=0,
                fit_diffuse_directions=0, update_forward_calls=0, endpoint_gradient_calls=0)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved, _, _ = toy_state()

    def test_json_restored_continuation_is_bitwise_uninterrupted(self):
        saved, z, opt = toy_state()
        restored, other = run.restore(saved)
        self.assertEqual(run.snapshot(z, opt), run.snapshot(restored, other))
        for i in range(5):
            for value, optimizer in ((z, opt), (restored, other)):
                optimizer.zero_grad(); ((value - .2)**2).sum().backward(); optimizer.step()
            self.assertEqual(run.snapshot(z, opt), run.snapshot(restored, other))
        self.assertEqual(float(other.state[restored]['step']), 255)

    def test_restoration_performs_no_update_no_rng_and_keeps_evidence(self):
        before = copy.deepcopy(self.saved); rng = torch.get_rng_state().clone()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No restoration update')):
            z, opt = run.restore(self.saved)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, self.saved)
        self.assertEqual(set(opt.state[z]), {'step', 'exp_avg', 'exp_avg_sq'})
        self.assertTrue(all(v.dtype == torch.float32 and v.device.type == 'cpu' for v in opt.state[z].values()))

    def test_bad_counter_or_incomplete_state_rejected(self):
        for field, value in [('step', 0.), ('exp_avg', [0.]*6), ('exp_avg_sq', [-1.]*7)]:
            row = copy.deepcopy(self.saved); row['optimizer_final_state'][field] = value
            with self.assertRaises(SignatureError): run.restore(row)
        row = copy.deepcopy(self.saved); del row['optimizer_final_state']['exp_avg']
        with self.assertRaises(SignatureError): run.restore(row)

    def test_wrong_options_or_physical_or_start_rejected(self):
        for field, value in [('lr', .02), ('amsgrad', True), ('foreach', True), ('weight_decay', .01)]:
            row = copy.deepcopy(self.saved); row['optimizer_effective'][field] = value
            with self.assertRaises(SignatureError): run.restore(row)
        row = copy.deepcopy(self.saved); row['final'][0] = .125
        with self.assertRaises(SignatureError): run.restore(row)
        row = copy.deepcopy(self.saved); row['completed_updates'] = 249
        with self.assertRaises(SignatureError): run.restore(row)

    def test_exact_float32_no_silent_rounding_or_invalid_shapes(self):
        for a, shape in [([.1]*7, (7,)), ([float('nan')]*7, (7,)), ([0.]*6, (7,)), ([250.], ())]:
            with self.assertRaises(SignatureError): run.float32_exact(a, shape, 'fixture')
        self.assertEqual(run.float32_exact(250., (), 'step').dtype, torch.float32)

    def test_nonoriginal_default_dtype_rejected(self):
        with patch.object(torch, 'get_default_dtype', return_value=torch.float64):
            with self.assertRaises(SignatureError): run.restore(self.saved)

    def test_exact_750_returns_without_replaying_prefix_or_extra_endpoint_step(self):
        z, opt = run.restore(self.saved); row = dict(seed=101, status='restored', completed_updates=250,
                                                    additional_updates=0, history=[])
        counts = budget(); calls = []
        original_step = opt.step
        def step():
            calls.append(float(opt.state[z]['step'])); return original_step()
        with patch.object(opt, 'step', side_effect=step), patch('builtins.print'):
            result = run.continue_start(row, z, opt, None, ToyBatch(), torch.zeros(24, 3), counts, lambda: None, lambda: None)
        self.assertEqual(calls, list(range(250, 1000)))
        self.assertEqual((row['completed_updates'], row['additional_updates'], counts['optimizer_returns']), (1000, 750, 750))
        self.assertEqual((counts['update_forward_calls'], counts['endpoint_gradient_calls']), (750, 1))
        self.assertEqual(row['history'][0]['loss_at_update_count'], 250)
        self.assertEqual(row['history'][-1]['loss_at_update_count'], 999)
        self.assertEqual(row['latest_state']['adam']['step']['values'], 1000.)
        self.assertEqual(result.tolist(), row['final']); self.assertTrue(np.isfinite(row['final_gradient']).all())

    def test_failed_forward_preserves_state_no_update_or_retry(self):
        z, opt = run.restore(self.saved); before = run.snapshot(z, opt)
        row = dict(status='restored', completed_updates=250, additional_updates=0, history=[])
        class Invalid(ToyBatch):
            def __call__(self, *args): return torch.full((24, 3, 3), float('nan'))
        with patch.object(opt, 'step', side_effect=AssertionError('No step')):
            with self.assertRaises(SignatureError):
                run.continue_start(row, z, opt, None, Invalid(), torch.zeros(24, 3), budget(), lambda: None, lambda: None)
        self.assertEqual(row['status'], 'failed'); self.assertEqual(row['latest_state'], before)
        self.assertEqual(row['additional_updates'], 0)

    def test_stop_preserves_successful_return_count_and_adam_state(self):
        z, opt = run.restore(self.saved)
        row = dict(status='restored', completed_updates=250, additional_updates=0, history=[])
        b = budget()
        def deadline():
            if b['optimizer_returns'] == 2: raise run.original.RuntimeLimit('fixture timeout')
        with self.assertRaises(run.original.RuntimeLimit):
            run.continue_start(row, z, opt, None, ToyBatch(), torch.zeros(24, 3), b, deadline, lambda: None)
        self.assertEqual(row['additional_updates'], 2)
        self.assertEqual(row['latest_state']['adam']['step']['values'], 252.)

    def test_budget_blocks_before_shader_and_never_extends(self):
        b = budget(); b['fit_shader_points'] = run.MAX_FIT_POINTS
        with self.assertRaises(SignatureError): run.count_fit(b, 96)
        b = budget(); b['update_forward_calls'] = 3750
        with self.assertRaises(SignatureError): run.count_fit(b, 96)
        with self.assertRaises(SignatureError): run.count_fit(budget(), 95)

    def test_input_hash_failure_precedes_parsing_and_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'wrong.json'; p.write_text('bad JSON')
            with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer')):
                with self.assertRaisesRegex(SignatureError, 'SHA-256'): run.admit(p, p)

    def test_runtime_failure_restores_threads_and_retains_failure(self):
        report = {}; threads = torch.get_num_threads(); rng = torch.get_rng_state().clone()
        runtime = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu', torch_threads=1)
        with patch.object(run.platform, 'python_version', return_value='incompatible'):
            with self.assertRaises(SignatureError):
                run.execute({'runtime': runtime}, {}, report, lambda: None)
        self.assertEqual(report['status'], 'failed'); self.assertTrue(report['threads_restored'])
        self.assertEqual(threads, torch.get_num_threads()); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(report['budget']['optimizer_returns'], 0)

    def test_only_budget_changes_and_original_success_criteria_reused(self):
        self.assertEqual(run.original.SETTINGS['success'], dict(Q_relative=.05, orientation_degrees=5.,
            max_albedo_absolute=.02, metallic_absolute=.03, held_out_relative_L2=.01))
        tree = ast.parse(Path(run.__file__).read_text())
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        forbidden = ('fit_start', 'initializations', 'render', 'identifiability', 'make_grids',
                     'load', 'clip_grad_norm_', 'manual_seed', 'cuda')
        self.assertEqual([f.attr for f in calls if isinstance(f, ast.Attribute) and f.attr in forbidden], [])

    def test_output_collision_never_admits_or_executes_and_handlers_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json'
            target.parent.mkdir(parents=True); target.write_text('preserve')
            protocol = root/'protocol.md'; protocol.write_text('fixture')
            source = root/'fixture.py'; source.write_text('fixture')
            handler = signal.getsignal(signal.SIGTERM)
            with patch.object(run, 'ROOT', root), patch.object(run, 'PROTOCOL', protocol), patch.object(run, '__file__', str(source)), \
                 patch.dict(run.os.environ, run.original.REQUIRED_ENV), \
                 patch.object(run, 'admit', side_effect=AssertionError('No admission')), \
                 patch.object(sys, 'argv', ['continue', '--input', str(root/'in.json'), '--diagnostic', str(root/'d.json'), '--output', str(target)]):
                with self.assertRaises(FileExistsError): run.main()
            self.assertEqual(target.read_text(), 'preserve')
            self.assertEqual(handler, signal.getsignal(signal.SIGTERM))

    def test_admission_failure_records_partial_and_restores_signal_handlers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root/'docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json'
            target.parent.mkdir(parents=True)
            protocol = root/'protocol.md'; protocol.write_text('fixture')
            source = root/'fixture.py'; source.write_text('fixture')
            handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGALRM)}
            with patch.object(run, 'ROOT', root), patch.object(run, 'PROTOCOL', protocol), patch.object(run, '__file__', str(source)), \
                 patch.dict(run.os.environ, run.original.REQUIRED_ENV), \
                 patch.object(run, 'admit', side_effect=SignatureError('fixture failed gate')), \
                 patch.object(sys, 'argv', ['continue', '--input', str(root/'in.json'), '--diagnostic', str(root/'d.json'), '--output', str(target)]):
                with self.assertRaises(SignatureError): run.main()
            result = json.loads(target.read_text())
            self.assertEqual(result['status'], 'failed'); self.assertEqual(result['restoration_gate'], 'not_started')
            self.assertIn('finished_at_utc', result)
            self.assertEqual(handlers, {s: signal.getsignal(s) for s in handlers})


if __name__ == '__main__': unittest.main()
