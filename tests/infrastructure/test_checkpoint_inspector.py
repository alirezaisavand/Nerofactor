"""CPU metadata fixtures only; never open the historical checkpoint or launch isolation."""

from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

SPEC = importlib.util.spec_from_file_location('bell_metadata_inspector',
                                             Path(__file__).resolve().parents[2] / 'inspect_bell_checkpoint.py')
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


class Device:
    def __init__(self, name):
        self.type = name

    def __str__(self):
        return self.type


class Tensor:
    def __init__(self, shape=(2, 3), device='cpu', dtype='torch.float32'):
        self.shape, self.dtype = shape, dtype
        self.device = Device(device)


FAKE_TORCH = SimpleNamespace(Tensor=Tensor, nn=SimpleNamespace(Parameter=Tensor))


def fixture_state():
    return {'step': 1200, 'best_para': 24.5,
            'network_state_dict': OrderedDict([('sdf.weight_v', Tensor()), ('sdf.weight_g', Tensor((2, 1)))]),
            'optimizer_state_dict': {'secret': 'OPTIMIZER_CONTENT_MUST_NOT_APPEAR'},
            'provenance': {'commit': 'declared-revision', 'seed': 6033,
                           'config': {'credentials': 'NESTED_CONTENT_MUST_NOT_APPEAR'}}}


class MetadataInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'synthetic.pth'
        self.checkpoint.write_bytes(b'synthetic fixture bytes; not a real historical checkpoint')
        self.outputs = self.root / 'outputs'
        self.outputs.mkdir()

    def args(self):
        return ['--checkpoint', str(self.checkpoint), '--output-root', str(self.outputs),
                '--image-id', 'sha256:' + '0' * 64, '--execute',
                '--acknowledge-trusted-checkpoint', '--confirm-isolated-runtime']

    def run_main(self, args=None, digest=None, guards=None, torch=None):
        torch = torch or SimpleNamespace(__version__=inspector.EXPECTED_TORCH,
                                       Tensor=Tensor, nn=FAKE_TORCH.nn,
                                       load=Mock(return_value=fixture_state()))
        with patch.object(inspector, 'CHECKPOINT', self.checkpoint), \
                patch.object(inspector, 'OUTPUT_ROOT', self.outputs), \
                patch.object(inspector, 'APPROVED_SHA256', digest or hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()), \
                patch.object(inspector, '_runtime_guards', guards or Mock(return_value={'fixture': True})), \
                patch.object(inspector, '_import_torch', return_value=torch) as importer, \
                patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = inspector.main(self.args() if args is None else args)
        return result, importer, torch

    def test_required_execution_acknowledgements(self):
        for flag in ['--execute', '--acknowledge-trusted-checkpoint', '--confirm-isolated-runtime']:
            args = self.args(); args.remove(flag)
            with self.subTest(flag=flag), self.assertRaises(SystemExit) as error:
                self.run_main(args)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(list(self.outputs.iterdir()), [])

    def test_no_checkpoint_hash_or_output_override_options(self):
        for flag, value in [('--checkpoint', '/tmp/other.pth'), ('--output-root', '/tmp/escape')]:
            args = self.args(); args[args.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                self.run_main(args)
        with self.assertRaises(SystemExit):
            self.run_main(self.args() + ['--expected-sha256', '0' * 64])

    def test_guards_and_bad_hash_fail_before_torch_or_output(self):
        for kwargs in [{'guards': Mock(side_effect=inspector.InspectionError('fixture guard failure'))},
                       {'digest': '0' * 64}]:
            with self.subTest(kwargs=kwargs):
                result, importer, torch = self.run_main(**kwargs)
                self.assertEqual(result, 1); importer.assert_not_called(); torch.load.assert_not_called()
                self.assertEqual(list(self.outputs.iterdir()), [])

    def test_fixed_hash_verification_and_stream_rewind(self):
        stream = io.BytesIO(b'fixture')
        with self.assertRaises(inspector.InspectionError):
            inspector._verify_hash(stream)
        with patch.object(inspector, 'APPROVED_SHA256', hashlib.sha256(b'fixture').hexdigest()):
            self.assertEqual(inspector._verify_hash(stream), inspector.APPROVED_SHA256)
        self.assertEqual(stream.tell(), 0)

    def test_metadata_excludes_values_and_optimizer_payload(self):
        state = fixture_state()
        class NoTraversal(dict):
            def __iter__(self): raise AssertionError('optimizer traversed')
            def __repr__(self): raise AssertionError('optimizer represented')
        state['optimizer_state_dict'] = NoTraversal(state['optimizer_state_dict'])
        report = inspector.summarize_checkpoint(state, FAKE_TORCH)
        self.assertEqual(report['stored_step']['value'], 1200)
        self.assertEqual(report['best_para']['value'], 24.5)
        self.assertEqual(report['network_tensors'][0]['shape'], [2, 1])
        self.assertEqual(report['primitive_provenance'][0]['value'], 'declared-revision')
        encoded = json.dumps(report, allow_nan=False)
        self.assertNotIn('OPTIMIZER_CONTENT', encoded); self.assertNotIn('NESTED_CONTENT', encoded)
        self.assertNotIn('values', report['network_tensors'][0])

    def test_invalid_container_tensor_or_device_rejected(self):
        for state in [[], {}, {'network_state_dict': {}},
                      {'network_state_dict': {'bad': 'not tensor'}},
                      {'network_state_dict': {'gpu': Tensor(device='cuda')}}]:
            with self.subTest(state_type=type(state)), self.assertRaises(inspector.InspectionError):
                inspector.summarize_checkpoint(state, FAKE_TORCH)

    def test_scalar_and_provenance_limits(self):
        state = fixture_state(); state.update(step=True, best_para=float('nan'))
        state['provenance'].update(branch='x' * 513, stage=Tensor())
        report = inspector.summarize_checkpoint(state, FAKE_TORCH)
        self.assertEqual(report['stored_step']['status'], 'unsupported_or_nonfinite')
        self.assertEqual(report['best_para']['status'], 'unsupported_or_nonfinite')
        self.assertTrue(all('value' not in r for r in report['primitive_provenance'] if r['path'] in ['provenance.stage', 'provenance.branch']))
        del state['step']; del state['best_para']
        report = inspector.summarize_checkpoint(state, FAKE_TORCH)
        self.assertEqual(report['stored_step']['status'], 'absent')

    def test_success_maps_cpu_without_restoration_or_input_changes(self):
        before = (self.checkpoint.read_bytes(), self.checkpoint.stat().st_mtime_ns)
        result, importer, torch = self.run_main()
        self.assertEqual(result, 0); importer.assert_called_once()
        self.assertEqual(torch.load.call_count, 1)
        self.assertEqual(torch.load.call_args.kwargs, {'map_location': 'cpu'})
        report = json.loads((self.outputs / inspector.REPORT_NAME).read_text())
        self.assertEqual(report['status'], 'metadata_extracted')
        self.assertTrue(report['checkpoint_deserialization_attempted'])
        self.assertEqual(report['declared_image_id'], 'sha256:' + '0' * 64)
        self.assertEqual(before, (self.checkpoint.read_bytes(), self.checkpoint.stat().st_mtime_ns))

    def test_collision_rejected_before_torch_without_overwrite(self):
        path = self.outputs / inspector.REPORT_NAME; path.write_text('existing evidence')
        result, importer, torch = self.run_main()
        self.assertEqual(result, 1); importer.assert_not_called(); torch.load.assert_not_called()
        self.assertEqual(path.read_text(), 'existing evidence')

    def test_symlink_and_nonregular_input_rejected(self):
        self.checkpoint.unlink(); self.checkpoint.symlink_to(self.root / 'absent')
        result, importer, _ = self.run_main(digest='0' * 64)
        self.assertEqual(result, 1); importer.assert_not_called()
        self.checkpoint.unlink(); self.checkpoint.mkdir()
        result, importer, _ = self.run_main(digest='0' * 64)
        self.assertEqual(result, 1); importer.assert_not_called()

    def test_failure_report_does_not_include_exception_values(self):
        torch = SimpleNamespace(__version__=inspector.EXPECTED_TORCH,
                                load=Mock(side_effect=RuntimeError('PRIVATE_OPTIMIZER_CONTENT tensor([99])')))
        result, _, _ = self.run_main(torch=torch)
        self.assertEqual(result, 1)
        text = (self.outputs / inspector.REPORT_NAME).read_text(); report = json.loads(text)
        self.assertEqual(report['phase'], 'checkpoint_deserialization'); self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['checkpoint_deserialization_attempted'])
        self.assertNotIn('PRIVATE_OPTIMIZER', text); self.assertNotIn('tensor([99])', text)

    def test_version_mismatch_stops_before_load(self):
        torch = SimpleNamespace(__version__='different', load=Mock())
        result, _, torch = self.run_main(torch=torch)
        self.assertEqual(result, 1); torch.load.assert_not_called()
        report = json.loads((self.outputs / inspector.REPORT_NAME).read_text())
        self.assertFalse(report['checkpoint_deserialization_attempted'])

    def test_input_mutation_during_load_is_failure(self):
        def mutate(stream, **kwargs):
            self.checkpoint.write_bytes(b'changed synthetic fixture')
            return fixture_state()
        torch = SimpleNamespace(__version__=inspector.EXPECTED_TORCH, Tensor=Tensor,
                                nn=FAKE_TORCH.nn, load=Mock(side_effect=mutate))
        result, _, _ = self.run_main(torch=torch)
        self.assertEqual(result, 1)
        report = json.loads((self.outputs / inspector.REPORT_NAME).read_text())
        self.assertEqual(report['phase'], 'input_recheck'); self.assertIsNone(report['inspection'])

    def test_runtime_guard_invariants_with_disposable_mount_fixtures(self):
        script = self.root / 'inspector.py'; script.write_text('fixture')
        base = '1 0 0:1 / / ro - overlay overlay ro\n'
        base += '2 1 0:2 / /outputs rw - ext4 fixture rw\n'
        base += '3 1 0:3 / ' + str(script) + ' ro - ext4 fixture ro\n'
        base += '4 1 0:4 / ' + str(self.checkpoint) + ' ro - ext4 fixture ro\n'
        read_text, exists, glob = Path.read_text, Path.exists, Path.glob
        state = {'mounts': base, 'interfaces': 'header\nheader\n lo: 0\n', 'gpu': False}
        def read(p, *args, **kwargs):
            if str(p) == '/proc/self/mountinfo': return state['mounts']
            if str(p) == '/proc/net/dev': return state['interfaces']
            return read_text(p, *args, **kwargs)
        def node_exists(p):
            return False if str(p) in ['/dev/dri', '/dev/kfd'] else exists(p)
        def nodes(p, pattern):
            if str(p) == '/dev': return [Path('/dev/nvidia0')] if state['gpu'] else []
            return glob(p, pattern)
        with patch.object(inspector, 'CHECKPOINT', self.checkpoint), \
                patch.object(inspector, 'OUTPUT_ROOT', self.outputs), \
                patch.object(inspector.os, 'geteuid', return_value=1109) as uid, \
                patch.object(inspector.sys, 'platform', 'linux'), \
                patch.object(inspector.os, 'statvfs', return_value=SimpleNamespace(f_flag=os.ST_RDONLY)), \
                patch.object(Path, 'read_text', read), patch.object(Path, 'exists', node_exists), \
                patch.object(Path, 'glob', nodes), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void'}):
            self.assertEqual(inspector._runtime_guards(script)['interfaces'], ['lo'])
            uid.return_value = 0
            with self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)
            uid.return_value = 1109
            for old, new in [(' / ro ', ' / rw '), (' /outputs rw ', ' /outputs ro '),
                             (' ' + str(script) + ' ro ', ' ' + str(script) + ' rw '),
                             (' ' + str(self.checkpoint) + ' ro ', ' ' + str(self.checkpoint) + ' rw ')]:
                state['mounts'] = base.replace(old, new)
                with self.subTest(mount=old), self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)
            state['mounts'] = base
            state['interfaces'] += ' eth0: 0\n'
            with self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)
            state['interfaces'] = 'header\nheader\n lo: 0\n'; state['gpu'] = True
            with self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)
            state['gpu'] = False; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            with self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''; (self.outputs / 'existing').write_text('fixture')
            with self.assertRaises(inspector.InspectionError): inspector._runtime_guards(script)

    def test_mount_parser_and_host_cli_fail_closed(self):
        mounts = inspector._mounts('1 0 0:1 / / ro - overlay overlay ro\n2 1 0:2 / /outputs rw - ext4 host rw\n')
        self.assertEqual(mounts['/outputs'], {'rw'}); self.assertEqual(mounts['/'], {'ro'})
        # Direct host use never reaches Torch. No historical checkpoint is opened.
        with patch.object(inspector, '_import_torch') as importer, \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void'}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = inspector.main(['--checkpoint', str(inspector.CHECKPOINT), '--output-root', '/outputs',
                                     '--image-id', 'sha256:' + '0' * 64, '--execute',
                                     '--acknowledge-trusted-checkpoint', '--confirm-isolated-runtime'])
        self.assertEqual(result, 1); importer.assert_not_called()


class SyntheticTorchFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest('Local CPU Torch import unavailable: ' + type(exc).__name__)
        cls.torch = torch

    def test_real_synthetic_checkpoint_and_numpy_scalars(self):
        torch = self.torch
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'disposable-synthetic.pth'
            state = {'step': np.int64(42), 'best_para': np.float64(23.5),
                     'network_state_dict': OrderedDict([('weight_v', torch.zeros((2, 3), device='cpu')),
                                                        ('buffer', torch.ones((), dtype=torch.int64, device='cpu'))]),
                     'optimizer_state_dict': {'private': torch.ones(5, device='cpu')}}
            torch.save(state, str(path)); before = path.read_bytes()
            with path.open('rb') as stream, warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                report = inspector._load_metadata(stream, torch)
            self.assertEqual(report['stored_step']['value'], 42)
            self.assertEqual(report['best_para']['value'], 23.5)
            self.assertEqual(report['network_tensors'][0]['shape'], [])
            self.assertEqual(report['network_tensors'][0]['dtype'], 'torch.int64')
            self.assertTrue(all(t['device'] == 'cpu' for t in report['network_tensors']))
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn('private', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
