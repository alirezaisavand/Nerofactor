"""Synthetic JSON/CPU fixtures; never deserializes historical weights or runs GPU code."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import compare_bell_signatures as cli
import generate_bell_reference as generator
from infrastructure.signatures import (CHECKPOINT_SHA256, REFERENCE_PROTOCOL, SignatureError,
                                       compare_catalogs, read_report, reserve_output, tensor_catalog,
                                       validate_checkpoint, validate_reference)


def entries(count=3):
    result = [{'name': 'layer.weight_g', 'shape': [2, 1], 'dtype': 'torch.float32',
               'device': 'cpu', 'role': 'parameter'},
              {'name': 'layer.weight_v', 'shape': [2, 3], 'dtype': 'torch.float32',
               'device': 'cpu', 'role': 'parameter'},
              {'name': 'buffer', 'shape': [], 'dtype': 'torch.float32',
               'device': 'cpu', 'role': 'persistent_buffer'}]
    result += [{'name': 'parameter_' + str(i), 'shape': [1], 'dtype': 'torch.float32',
                'device': 'cpu', 'role': 'parameter'} for i in range(count - 3)]
    return result


def checkpoint(values):
    values = deepcopy(values)
    for entry in values:
        entry.pop('role', None)
    return {'schema_version': 1, 'protocol': 'bell-checkpoint-metadata-v1',
            'status': 'metadata_extracted', 'phase': 'complete', 'checkpoint': {'sha256': CHECKPOINT_SHA256},
            'inspection': {'network_tensors': values, 'tensor_count': len(values),
                           'stored_step': {'status': 'recorded', 'value': 150000},
                           'best_para': {'status': 'recorded', 'value': 23.787122565451323}}}


def reference(values):
    original = 'network: shape\ndatabase_name: syn/bell\n'
    digest = hashlib.sha256(original.encode()).hexdigest()
    return {'schema_version': 1, 'protocol': REFERENCE_PROTOCOL, 'status': 'reference_generated',
            'phase': 'complete', 'algorithm': 'maintained-A-bell-geometry',
            'constructor': {'class': 'NeROShapeRenderer', 'training': False},
            'network_tensors': deepcopy(values), 'tensor_count': len(values),
            'state_dict_keys': sorted(e['name'] for e in values),
            'configuration': {'original_yaml': original, 'sha256': digest,
                              'raw_values': {'network': 'shape', 'database_name': 'syn/bell'},
                              'overrides': {}, 'effective_renderer_configuration': {'network': 'shape', 'database_name': 'syn/bell'},
                              'effective_shader_configuration': {}},
            'source': {'declared_commit': '0' * 40, 'declared_diff_sha256': '0' * 64, 'declared_tree_state': 'dirty',
                       'files_sha256': {'network/renderer.py': '0' * 64, 'network/field.py': '1' * 64,
                                       'configs/shape/syn/bell.yaml': digest}},
            'runtime': {'torch': '1.8.1+cu111', 'cuda_compiled': '11.1', 'python': 'synthetic fixture', 'gpu': 'fake fixture'}}


class SignatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def catalogs(self, values=None):
        values = values or entries()
        return validate_checkpoint(checkpoint(values), required_count=len(values)), validate_reference(reference(values))

    def test_exact_match_and_buffer_weightnorm_coverage(self):
        saved, expected = self.catalogs()
        result = compare_catalogs(saved, expected)
        self.assertEqual(result['result'], 'exact_structural_match')
        self.assertTrue(result['key_sets_equal'])
        self.assertEqual(result['coverage']['reference_persistent_buffers'], ['buffer'])
        self.assertTrue(result['coverage']['weight_normalization'][0]['checkpoint_pair'])
        expected['buffer']['device'] = 'cuda:0'
        self.assertEqual(compare_catalogs(saved, expected)['result'], 'exact_structural_match')

    def test_missing_unexpected_keys_are_not_normalized(self):
        saved, expected = self.catalogs()
        saved['module.buffer'] = saved.pop('buffer')
        result = compare_catalogs(saved, expected)
        self.assertEqual(result['result'], 'structural_mismatch')
        self.assertEqual(result['missing_keys'], ['buffer'])
        self.assertEqual(result['unexpected_keys'], ['module.buffer'])
        self.assertFalse(result['coverage']['checkpoint_contains_reference_buffers'])

    def test_shape_and_dtype_mismatches(self):
        saved, expected = self.catalogs()
        saved['layer.weight_v']['shape'] = [3, 2]
        saved['buffer']['dtype'] = 'torch.float64'
        result = compare_catalogs(saved, expected)
        self.assertTrue(result['key_sets_equal'])
        self.assertEqual(result['shape_mismatches'][0]['name'], 'layer.weight_v')
        self.assertEqual(result['dtype_mismatches'][0]['name'], 'buffer')
        self.assertEqual(result['result'], 'structural_mismatch')

    def test_duplicate_entries_bad_shapes_dtypes_and_counts(self):
        for bad in [entries() + [entries()[0]], [dict(entries()[0], shape=[True])],
                    [dict(entries()[0], shape=[-1])], [dict(entries()[0], dtype='float32')],
                    [dict(entries()[0], name='')], [dict(entries()[0], device='bad')]]:
            with self.subTest(entry_count=len(bad)), self.assertRaises(SignatureError):
                tensor_catalog(bad, len(bad))
        with self.assertRaises(SignatureError): tensor_catalog(entries(), True)
        with self.assertRaises(SignatureError): tensor_catalog(entries(), 2)

    def test_malformed_checkpoint_and_reference(self):
        for field, value in [('status', 'failed'), ('schema_version', True), ('inspection', None),
                             ('checkpoint', {'sha256': '0' * 64})]:
            bad = checkpoint(entries()); bad[field] = value
            with self.subTest(field=field), self.assertRaises(SignatureError): validate_checkpoint(bad, 3)
        for field, value in [('status', 'running'), ('state_dict_keys', []), ('source', {}), ('runtime', {}),
                             ('constructor', {'class': 'NeROShapeRenderer', 'training': True})]:
            bad = reference(entries()); bad[field] = value
            with self.subTest(field=field), self.assertRaises(SignatureError): validate_reference(bad)
        bad = reference(entries()); bad['network_tensors'][0]['role'] = 'persistent_buffer'
        with self.assertRaises(SignatureError): validate_reference(bad)
        bad = checkpoint(entries()); bad['inspection']['stored_step']['value'] = True
        with self.assertRaises(SignatureError): validate_checkpoint(bad, 3)

    def test_report_identity_duplicate_json_and_nonfinite_rejection(self):
        path = self.root / 'report.json'
        for text in ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}', '[]', 'invalid']:
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(SignatureError):
                read_report(path, hashlib.sha256(path.read_bytes()).hexdigest())
        path.write_text(json.dumps(checkpoint(entries())))
        with self.assertRaises(SignatureError): read_report(path, '0' * 64)
        data, identity = read_report(path, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(data['status'], 'metadata_extracted'); self.assertEqual(identity['path'], str(path))
        alias = self.root / 'alias.json'; alias.symlink_to(path)
        with self.assertRaises(SignatureError): read_report(alias, identity['sha256'])

    def test_output_containment_and_collision(self):
        output = self.root / 'out'; output.mkdir()
        with reserve_output(output, 'report.json') as stream: stream.write('preserved')
        with self.assertRaises(SignatureError): reserve_output(output, 'report.json')
        self.assertEqual((output / 'report.json').read_text(), 'preserved')
        empty = self.root / 'empty'; empty.mkdir()
        for name in ['../escape.json', '/tmp/escape.json']:
            with self.assertRaises(SignatureError): reserve_output(empty, name)
        alias = self.root / 'alias'; alias.symlink_to(empty, target_is_directory=True)
        with self.assertRaises(SignatureError): reserve_output(alias, 'x.json')
        with self.assertRaises(SignatureError): reserve_output(empty, 'x.json', [empty / 'input.json'])

    def run_cli(self, count=137, mutate=None, missing_reference=False):
        cpdir, refdir, out = [self.root / p for p in ('cp', 'ref', 'out')]
        for p in (cpdir, refdir, out): p.mkdir()
        cp, ref = cpdir / 'report.json', refdir / 'reference.json'
        cp.write_text(json.dumps(checkpoint(entries(count))))
        refdata = reference(entries(count))
        if mutate: mutate(refdata)
        ref.write_text(json.dumps(refdata)); cpbytes = cp.read_bytes(); refbytes = ref.read_bytes()
        cp_hash = hashlib.sha256(cpbytes).hexdigest(); ref_hash = hashlib.sha256(refbytes).hexdigest()
        if missing_reference: ref.unlink()
        args = ['--checkpoint-report', str(cp), '--checkpoint-report-sha256', cp_hash,
                '--reference-signature', str(ref), '--reference-sha256', ref_hash, '--output-root', str(out)]
        with patch.object(cli, 'ADMITTED_REPORT_SHA256', cp_hash), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = cli.main(args)
        report = json.loads((out / 'architecture-comparison.json').read_text())
        self.assertEqual(cp.read_bytes(), cpbytes)
        if not missing_reference: self.assertEqual(ref.read_bytes(), refbytes)
        return status, report

    def test_cli_exact_and_identity_reporting(self):
        status, report = self.run_cli()
        self.assertEqual(status, 0); self.assertEqual(report['result'], 'exact_structural_match')
        self.assertEqual(report['checkpoint_tensor_count'], 137)
        self.assertEqual(report['reference_identity']['bytes'], (self.root / 'ref/reference.json').stat().st_size)

    def test_cli_mismatch_has_complete_diagnostics(self):
        status, report = self.run_cli(mutate=lambda r: r['network_tensors'][0].update(shape=[4, 4]))
        self.assertEqual(status, 1); self.assertEqual(report['result'], 'structural_mismatch')
        self.assertEqual(len(report['shape_mismatches']), 1)

    def test_cli_missing_reference_is_blocked(self):
        status, report = self.run_cli(missing_reference=True)
        self.assertEqual(status, 2); self.assertEqual(report['result'], 'blocked')
        self.assertIsNone(report['reference_identity'])

    def test_constructor_and_state_signature_use_real_protocol_without_forward(self):
        class Tensor:
            shape, dtype, device = (2, 3), 'torch.float32', 'cpu'
        class Model:
            def __init__(self, cfg, training=True):
                self.cfg = deepcopy(cfg); self.color_network = SimpleNamespace(cfg={'sphere_direction': False})
                self.training_argument = training; calls.append(training)
            def state_dict(self): return {'weight': Tensor(), 'persistent': Tensor()}
            def named_parameters(self): return [('weight', Tensor())]
            def named_buffers(self): return [('persistent', Tensor()), ('nonpersistent', Tensor())]
            def forward(self, *args): raise AssertionError('forward forbidden')
        calls = []; cfg = {'network': 'shape', 'database_name': 'syn/bell', 'shader_config': {}}
        before = deepcopy(cfg); result = generator.construct_reference(cfg, Model)
        self.assertEqual(calls, [False]); self.assertEqual(cfg, before)
        self.assertEqual([e['role'] for e in result['network_tensors']], ['persistent_buffer', 'parameter'])
        self.assertNotIn('nonpersistent', [e['name'] for e in result['network_tensors']])

    def test_generator_guard_failure_never_enters_native_backend(self):
        out = self.root / 'out'; out.mkdir()
        args = ['--output-root', str(out), '--source-commit', '0' * 40, '--source-diff-sha256', '0' * 64,
                '--source-tree-state', 'dirty', '--image-id', 'sha256:' + '0' * 64,
                '--execute', '--confirm-isolated-runtime']
        with patch.object(generator, 'runtime_guards', side_effect=SignatureError('fixture rejection')), \
                patch.object(generator, 'native_reference') as backend, redirect_stderr(io.StringIO()):
            self.assertEqual(generator.main(args), 1)
        backend.assert_not_called(); self.assertEqual(list(out.iterdir()), [])

    def test_generator_full_fixture_reports_real_backend_results_without_overrides(self):
        try:
            import yaml
        except ImportError:
            self.skipTest('YAML fixture parser unavailable in this interpreter')
        out = self.root / 'out'; out.mkdir()
        ref = reference(entries()); config = self.root / 'bell.yaml'
        config.write_text(ref['configuration']['original_yaml'])
        args = ['--output-root', str(out), '--source-commit', '0' * 40, '--source-diff-sha256', '0' * 64,
                '--source-tree-state', 'dirty', '--image-id', 'sha256:' + '0' * 64,
                '--execute', '--confirm-isolated-runtime']
        fixture = {'network_tensors': sorted(entries(), key=lambda e: e['name']),
                   'effective_renderer_configuration': {'network': 'shape', 'database_name': 'syn/bell'},
                   'effective_shader_configuration': {'sphere_direction': False}, 'runtime': ref['runtime']}
        with patch.object(generator, 'CONFIG', config), \
                patch.object(generator, 'runtime_guards', return_value={'fixture': True}), \
                patch.object(generator, 'source_identity', return_value=ref['source']) as source, \
                patch.object(generator, 'native_reference', return_value=fixture) as backend, \
                patch.dict(os.environ), redirect_stdout(io.StringIO()):
            self.assertEqual(generator.main(args), 0)
        self.assertEqual(source.call_count, 2)
        backend.assert_called_once_with({'network': 'shape', 'database_name': 'syn/bell'})
        result = json.loads((out / 'reference-signature.json').read_text())
        self.assertEqual(result['configuration']['overrides'], {})
        self.assertEqual(len(validate_reference(result)), 3)


class SyntheticCPUStateTests(unittest.TestCase):
    def test_actual_cpu_module_buffers_and_weight_norm_without_forward(self):
        try:
            import torch
        except (ImportError, OSError) as exc:
            self.skipTest('CPU Torch unavailable: ' + type(exc).__name__)
        model = torch.nn.Module()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            model.layer = torch.nn.utils.weight_norm(torch.nn.Linear(3, 2))
        model.register_buffer('persistent', torch.ones(1, device='cpu'))
        model.register_buffer('temporary', torch.zeros(1, device='cpu'), persistent=False)
        result = generator.extract_signature(model)
        names = {e['name'] for e in result}
        self.assertIn('layer.weight_g', names); self.assertIn('layer.weight_v', names)
        self.assertIn('persistent', names); self.assertNotIn('temporary', names)
        self.assertTrue(all('value' not in e and e['device'] == 'cpu' for e in result))


if __name__ == '__main__':
    unittest.main()
