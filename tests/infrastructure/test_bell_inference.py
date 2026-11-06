"""Disposable CPU/report fixtures; no historical loading, renderer or GPU execution."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import compare_bell_inference as cli
from infrastructure import bell_inference as bell
from infrastructure.bell_comparison import artifact, compare_sequence, load_arm, pair_difference, rgb_metrics
from infrastructure.signatures import ADMITTED_REPORT_SHA256, CHECKPOINT_SHA256, SignatureError


class Tensor:
    def __init__(self, shape=(2, 3), dtype='float32', device='cpu'):
        self.shape, self.dtype, self.device = shape, dtype, SimpleNamespace(type=device)


class BellRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture_arm(self, arm, offset=0.0):
        directory = self.root / arm; directory.mkdir()
        (directory / 'arrays').mkdir(); (directory / 'renders').mkdir()
        gt = np.full((12, 12, 3), .5, dtype=np.float32)
        prediction = np.full_like(gt, .4 + offset)
        mask = np.ones((12, 12, 1), dtype=np.int32)
        arrays = {'prediction': prediction, 'ground_truth': gt, 'mask': mask}
        for name, values in arrays.items(): np.save(str(directory / 'arrays' / (name + '.npy')), values, allow_pickle=False)
        for name in ['prediction', 'ground_truth']:
            (directory / 'renders' / (name + '.png')).write_bytes(b'explicit synthetic codec placeholder')
        metrics = rgb_metrics(prediction, gt)
        (directory / 'metrics.json').write_text(json.dumps(metrics))
        report = {'schema_version': 1, 'protocol': bell.PROTOCOL, 'status': 'complete', 'phase': 'complete',
                  'arm': arm, 'run_id': {'R': 'reference', 'R-repeat': 'reference-repeat', 'C': 'candidate'}[arm],
                  'step': bell.STORED_STEP, 'validation_calls': 1, 'checkpoint_sha256': CHECKPOINT_SHA256,
                  'evidence': {'metadata': ADMITTED_REPORT_SHA256, 'reference': bell.REFERENCE_SHA256,
                               'comparison': bell.COMPARISON_SHA256},
                  'splits': bell.proposed_split(), 'used_inputs': [{'path': '58.png', 'sha256': '0' * 64}],
                  'effective_trainer': {'random_seed': 6033}, 'effective_renderer': {'test_ray_num': 512},
                  'effective_shader': {'sphere_direction': False}, 'rng_before': {'fixture': 'same'},
                  'rng_after': {'fixture': 'same-after'}, 'preprocessing': {'version': 'fixture'},
                  'implementation': {'fixture': '0' * 64}, 'legacy_sha256': '0' * 64,
                  'declared_source_commit': '0' * 40, 'declared_image_id': 'sha256:' + '0' * 64,
                  'runtime': {'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-image']}},
                  'metrics': metrics, 'arrays': {n: 'arrays/' + n + '.npy' for n in arrays}}
        report['output_hashes'] = {p.relative_to(directory).as_posix(): bell.file_hash(p)
                                  for p in directory.rglob('*') if p.is_file()}
        path = directory / 'inference.json'; path.write_text(json.dumps(report))
        return path, bell.file_hash(path), report, arrays

    def output(self, name='comparison'):
        path = self.root / name; path.mkdir(); return path

    def test_direct_and_candidate_configuration_parity_and_independence(self):
        source = self.root / 'code'; cfgdir = source / 'configs'
        (cfgdir / 'shape/syn').mkdir(parents=True); (cfgdir / 'experiments').mkdir()
        real = Path(bell.ROOT)
        for rel in ['shape/syn/bell.yaml', 'experiments/geometry.example.yaml']:
            (cfgdir / rel).write_bytes((real / 'configs' / rel).read_bytes())
        datasets, thesis, outputs = [self.root / name for name in ['datasets', 'thesis', 'outputs']]
        for p in [datasets / 'GlossySynthetic/bell/masks_train', thesis, outputs]: p.mkdir(parents=True)
        for name in ['0-camera.pkl', '0.png', '0-depth.png']:
            (datasets / 'GlossySynthetic/bell' / name).write_bytes(b'metadata fixture only')
        with patch.object(bell, 'ROOT', source):
            with patch('infrastructure.config.load_config', side_effect=AssertionError('R must not call schema loader')):
                r = bell.resolve_reference(datasets, thesis, outputs)
            c = bell.resolve_candidate(datasets, thesis, outputs)
        self.assertEqual(r[0], c[0]); self.assertEqual(r[3], c[3]); self.assertEqual(r[2], c[2])
        self.assertNotEqual(r[4], c[4]); self.assertEqual(r[0]['lr_cfg'], {})

    def test_seed_policy_shallow_defaults_and_local_split_rng(self):
        calls = []
        torch = SimpleNamespace(manual_seed=lambda s: calls.append(('torch', s)))
        np_fake = SimpleNamespace(random=SimpleNamespace(seed=lambda s: calls.append(('numpy', s))))
        with patch.object(bell.random, 'seed', side_effect=lambda s: calls.append(('python', s))):
            effective = bell.seed_and_merge({'lr_cfg': {}}, {'random_seed': 6033, 'lr_cfg': {'lr_init': .002}}, torch, np_fake)
        self.assertEqual(calls, [('torch', 6033), ('numpy', 6033), ('python', 6033)])
        self.assertEqual(effective['lr_cfg'], {})
        before = random.getstate(); split = bell.proposed_split()
        model = SimpleNamespace(database=SimpleNamespace(get_img_ids=lambda: [str(i) for i in range(128)]), **split)
        self.assertEqual(bell.ordered_split(model), split); self.assertEqual(random.getstate(), before)
        model.test_ids = ['0']
        with self.assertRaises(SignatureError): bell.ordered_split(model)

    def test_strict_network_only_loading_and_mismatch_rejection(self):
        model = SimpleNamespace(state_dict=lambda: {'weight': Tensor()}, load_state_dict=Mock())
        torch = SimpleNamespace(Tensor=Tensor, nn=SimpleNamespace(Parameter=Tensor))
        state = {'step': 150000, 'network_state_dict': {'weight': Tensor()}, 'optimizer_state_dict': object()}
        bell.strict_load(model, state, torch)
        model.load_state_dict.assert_called_once_with(state['network_state_dict'], strict=True)
        for bad in [dict(state, step=0), dict(state, network_state_dict={'module.weight': Tensor()}),
                    dict(state, network_state_dict={'weight': Tensor(shape=(3, 2))}),
                    dict(state, network_state_dict={'weight': Tensor(dtype='float16')})]:
            model.load_state_dict.reset_mock()
            with self.assertRaises(SignatureError): bell.strict_load(model, bad, torch)
            model.load_state_dict.assert_not_called()

    def test_selected_input_identities_read_only_and_no_unpickling(self):
        scene = self.root / 'bell'; (scene / 'masks_train').mkdir(parents=True)
        for i in range(128):
            (scene / (str(i) + '-camera.pkl')).write_bytes(b'not a pickle; opaque hashing only')
        splits = bell.proposed_split()
        self.assertEqual(splits['test_ids'], ['58'])
        self.assertEqual(splits['nvs_ids'], ['11', '105', '114', '17', '76'])
        for i in splits['train_ids'] + splits['test_ids']:
            (scene / (i + '.png')).write_bytes(b'RGB identity fixture')
            (scene / 'masks_train' / (i + '.png')).write_bytes(b'mask identity fixture')
        (scene / '58-depth.png').write_bytes(b'depth identity fixture')
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in scene.rglob('*') if p.is_file()}
        first = bell.used_inputs(scene, splits)
        self.assertEqual(len(first), 375)
        self.assertEqual(first, bell.used_inputs(scene, splits))
        self.assertTrue(all((p.stat().st_mtime_ns, p.read_bytes()) == value for p, value in before.items()))
        (scene / '58.png').write_bytes(b'different RGB')
        self.assertNotEqual(first, bell.used_inputs(scene, splits))
        (scene / '58.png').unlink(); (scene / '58.png').symlink_to(scene / '58-depth.png')
        with self.assertRaises(SignatureError): bell.used_inputs(scene, splits)

    def test_metrics_equal_original_psnr_and_uint8_conversion(self):
        path = Path(bell.ROOT) / 'network/metrics.py'; text = path.read_text()
        fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'compute_psnr')
        namespace = {'np': np}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        gt, pred = np.full((12, 12, 3), .501, np.float32), np.full((12, 12, 3), .401, np.float32)
        convert = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8)
        expected = namespace['compute_psnr'](convert(gt), convert(pred))
        self.assertEqual(rgb_metrics(pred, gt)['psnr']['value'], float(expected))
        self.assertEqual(rgb_metrics(gt, gt)['psnr']['status'], 'positive_infinity')

    def test_valid_report_and_original_artifact_hashes_preserved(self):
        path, digest, original, _ = self.fixture_arm('R')
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in path.parent.rglob('*') if p.is_file()}
        report, arrays, identity = load_arm(path, digest, 'R')
        self.assertEqual(report, original); self.assertEqual(arrays['prediction'].shape, (12, 12, 3))
        self.assertEqual(identity['sha256'], digest)
        self.assertTrue(all((p.stat().st_mtime_ns, p.read_bytes()) == value for p, value in before.items()))

    def test_report_identity_metric_and_artifact_tampering_rejected(self):
        path, digest, report, _ = self.fixture_arm('R')
        with self.assertRaises(SignatureError): load_arm(path, '0' * 64, 'R')
        for key, value in [('status', 'failed'), ('step', 0), ('validation_calls', True),
                           ('splits', dict(report['splits'], test_ids=['0']))]:
            changed = deepcopy(report); changed[key] = value; path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(SignatureError): load_arm(path, bell.file_hash(path), 'R')
        path.write_text(json.dumps(report)); (path.parent / 'renders/prediction.png').write_bytes(b'changed')
        with self.assertRaises(SignatureError): load_arm(path, bell.file_hash(path), 'R')

    def test_output_artifact_traversal_and_alias_rejection(self):
        path, _, report, _ = self.fixture_arm('R'); root = path.parent
        for name in ['../escape.npy', '/tmp/out.npy']:
            with self.assertRaises(ValueError): artifact(root, name, {})
        alias = root / 'arrays/alias.npy'; alias.symlink_to(root / 'arrays/prediction.npy')
        with self.assertRaises(SignatureError): artifact(root, 'arrays/alias.npy', {'arrays/alias.npy': bell.file_hash(alias)})

    def test_repeatability_pass_then_candidate_exact_and_full_differences(self):
        values = [self.fixture_arm(a) for a in ['R', 'R-repeat', 'C']]
        loaded = [load_arm(v[0], v[1], a) for v, a in zip(values, ['R', 'R-repeat', 'C'])]
        out = self.output(); result = compare_sequence(*loaded, out)
        self.assertEqual(result['result'], 'regression_passed'); self.assertTrue(result['candidate_interpreted'])
        self.assertEqual(set(result['thresholds']), set(bell.THRESHOLDS))
        self.assertEqual(np.load(out / 'C-minus-R-signed-difference.npy', allow_pickle=False).shape, (12, 12, 3))
        self.assertTrue(all(bell.file_hash(out / n) == h for n, h in result['output_hashes'].items()))

    def test_repeatability_failure_prevents_candidate_loading(self):
        r, rr = self.fixture_arm('R'), self.fixture_arm('R-repeat', .01)
        candidate = Mock(side_effect=AssertionError('C must not be read'))
        result = compare_sequence(load_arm(r[0], r[1], 'R'), load_arm(rr[0], rr[1], 'R-repeat'), candidate, self.output())
        self.assertEqual(result['result'], 'repeatability_failed'); self.assertFalse(result['candidate_interpreted'])
        candidate.assert_not_called()

    def test_candidate_numeric_identity_and_rng_mismatch(self):
        r, c = self.fixture_arm('R'), self.fixture_arm('C', .01)
        left, la, _ = load_arm(r[0], r[1], 'R'); right, ra, _ = load_arm(c[0], c[1], 'C')
        result, _, _ = pair_difference(left, right, la, ra)
        self.assertFalse(result['passed']); self.assertGreater(result['float_violation_count'], 0)
        right = deepcopy(left); right['run_id'] = 'other'; right['rng_before'] = {'different': True}
        result, _, _ = pair_difference(left, right, la, la)
        self.assertFalse(result['passed']); self.assertIn('rng_before', result['identity_mismatches'])

    def test_candidate_small_error_and_infinite_metric_rules(self):
        r, rr, c = self.fixture_arm('R'), self.fixture_arm('R-repeat'), self.fixture_arm('C', .00001)
        loaded = [load_arm(x[0], x[1], arm) for x, arm in [(r, 'R'), (rr, 'R-repeat'), (c, 'C')]]
        result = compare_sequence(*loaded, self.output())
        self.assertEqual(result['result'], 'regression_passed')
        self.assertGreater(result['candidate_comparison']['max_absolute_error'], 0)
        from infrastructure.bell_comparison import metric_delta
        self.assertEqual(metric_delta({'status': 'positive_infinity'}, {'status': 'positive_infinity'}), 0)
        self.assertIsNone(metric_delta({'status': 'positive_infinity'}, {'status': 'finite', 'value': 20}))

    def test_candidate_reused_id_is_not_accepted(self):
        data = [self.fixture_arm(a) for a in ['R', 'R-repeat', 'C']]
        loaded = [load_arm(x[0], x[1], a) for x, a in zip(data, ['R', 'R-repeat', 'C'])]
        loaded[2][0]['run_id'] = loaded[1][0]['run_id']
        with self.assertRaises(SignatureError): compare_sequence(*loaded, self.output())

    def test_cli_comparison_and_collision_preserve_existing_evidence(self):
        r, rr, c = [self.fixture_arm(a) for a in ['R', 'R-repeat', 'C']]; out = self.output()
        args = ['--reference', str(r[0]), '--reference-sha256', r[1], '--repeat', str(rr[0]),
                '--repeat-sha256', rr[1], '--candidate', str(c[0]), '--candidate-sha256', c[1], '--output-root', str(out)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
            before = (out / 'regression-comparison.json').read_bytes()
            self.assertEqual(cli.main(args), 2)
            self.assertEqual((out / 'regression-comparison.json').read_bytes(), before)

    def test_blocked_candidate_keeps_repeatability_evidence(self):
        r, rr = self.fixture_arm('R'), self.fixture_arm('R-repeat')
        out = self.output()
        args = ['--reference', str(r[0]), '--reference-sha256', r[1], '--repeat', str(rr[0]),
                '--repeat-sha256', rr[1], '--candidate', str(self.root / 'missing/inference.json'),
                '--candidate-sha256', '0' * 64, '--output-root', str(out)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args), 2)
        result = json.loads((out / 'regression-comparison.json').read_text())
        self.assertEqual(result['result'], 'blocked')
        self.assertTrue(result['repeatability']['passed'])
        self.assertFalse(result['candidate_interpreted'])
        self.assertTrue(result['error'])
        self.assertEqual(len(result['output_hashes']), 2)

    def test_failed_native_fixture_retains_phase_logs_and_restores_runtime(self):
        source = self.root / 'code'; (source / 'configs/shape/syn').mkdir(parents=True)
        original = (Path(bell.ROOT) / 'configs/shape/syn/bell.yaml').read_bytes()
        (source / 'configs/shape/syn/bell.yaml').write_bytes(original)
        for name in bell.IMPLEMENTATION_FILES:
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('explicit CPU source identity fixture')
        datasets, thesis, outputs = [self.root / p for p in ['datasets', 'thesis', 'output']]
        for path in [datasets / 'GlossySynthetic/bell', thesis / 'model/bell_shape', outputs]: path.mkdir(parents=True)
        checkpoint = thesis / 'model/bell_shape/model_best.pth'; checkpoint.write_bytes(b'synthetic unparsed bytes')
        packet = {'reference': {'configuration': {'sha256': hashlib.sha256(original).hexdigest()},
                               'declared_image_id': 'sha256:' + '0' * 64}, 'identities': {'fixture': True}}
        def fake_backend(cfg, cache, run, checkpoint, packet, mark):
            mark('one_validation_view'); print('explicit CPU failure fixture'); raise RuntimeError('fixture')
        read = Path.read_text; old_hash = bell.file_hash
        def fake_read(path, *args, **kwargs):
            return 'header\nheader\n lo: 0\n' if str(path) == '/proc/net/dev' else read(path, *args, **kwargs)
        with patch.object(bell, 'ROOT', source), patch.object(bell, 'evidence', return_value=packet), \
                patch.object(bell, 'native_arm', side_effect=fake_backend) as backend, \
                patch.object(bell, 'file_hash', side_effect=lambda p: CHECKPOINT_SHA256 if Path(p) == checkpoint else old_hash(p)), \
                patch.object(os, 'statvfs', return_value=SimpleNamespace(f_flag=os.ST_RDONLY)), \
                patch.object(os, 'geteuid', return_value=1109), patch.object(Path, 'read_text', fake_read), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cwd, home = Path.cwd(), os.environ.get('HOME')
            with patch.object(bell, 'file_hash', return_value='1' * 64), self.assertRaises(SignatureError):
                bell.execute_arm('R', 'bad-hash', datasets, thesis, outputs, source, source, source,
                                 'sha256:' + '0' * 64, '0' * 40)
            backend.assert_not_called(); self.assertEqual(list(outputs.iterdir()), [])
            with self.assertRaises(RuntimeError):
                bell.execute_arm('R', 'fixture', datasets, thesis, outputs, source, source, source,
                                 'sha256:' + '0' * 64, '0' * 40)
            self.assertEqual(Path.cwd(), cwd); self.assertEqual(os.environ.get('HOME'), home)
        report = json.loads((outputs / 'inference.json').read_text())
        self.assertEqual(report['status'], 'failed'); self.assertEqual(report['phase'], 'one_validation_view')
        self.assertIn('explicit CPU failure fixture', (outputs / 'logs/console.log').read_text())
        self.assertEqual(checkpoint.read_bytes(), b'synthetic unparsed bytes')
        self.assertEqual(list((outputs / 'checkpoints').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
