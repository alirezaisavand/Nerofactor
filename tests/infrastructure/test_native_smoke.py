"""CPU fixtures only; native smoke/GPU entry points are never launched here."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from infrastructure.config import ConfigError, load_config
from infrastructure.geometry import CODE_ROOT
from infrastructure.native_smoke import (check_mounts, execute_smoke, one_forward,
                                         tensor_summary, validate_smoke)
from infrastructure.paths import PathResolver
from infrastructure.runs import RunError
from dataset_fixtures import deny_input_writes, snapshot


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        # Mocked CUDA metadata only; storage/calculation is explicitly NumPy CPU.
        self.shape, self.dtype, self.device = self.value.shape, self.value.dtype, 'cuda:0'

    def detach(self):
        return self

    def all(self):
        return Tensor(self.value.all())

    def item(self):
        return self.value.item()


class NativeSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs, self.outputs = self.root / 'inputs', self.root / 'outputs'
        self.configs = self.inputs / 'configs'
        self.configs.mkdir(parents=True)
        self.thesis = self.inputs / 'thesis'
        self.thesis.mkdir()
        self.datasets = self.inputs / 'datasets'
        scene = self.datasets / 'nerf/scene'
        for name in ['train', 'depth', 'masks_train']:
            (scene / name).mkdir(parents=True)
        (scene / 'transforms_train.json').write_text('{}')
        self.mesh = self.thesis / 'mesh.ply'
        self.mesh.write_text('ply\nformat ascii 1.0\nelement face 12\nend_header\n')
        self.resolver = PathResolver({'datasets': self.datasets, 'thesis': self.thesis}, self.outputs)
        self.events, self.models = [], []
        self.torch = SimpleNamespace(
            is_tensor=lambda value: isinstance(value, Tensor),
            isfinite=lambda value: Tensor(np.isfinite(value.value)),
            __version__='cpu-fixture', version=SimpleNamespace(cuda='fixture'),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 synchronize=lambda: None, get_device_name=lambda _: 'cpu-fixture'))
        output = redirect_stdout(io.StringIO())
        error = redirect_stderr(io.StringIO())
        output.__enter__()
        error.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.addCleanup(error.__exit__, None, None, None)

    def config(self, stage='geometry', checkpoint=False):
        payload = {'name': 'scene_' + stage, 'network': 'shape' if stage == 'geometry' else 'material',
                   'database_name': 'nerf/scene', 'is_nerf': True, 'dataset_dir': 'legacy/root',
                   'train_dataset_type': 'dummy', 'train_dataset_cfg': {'database_name': 'nerf/scene'},
                   'val_set_list': [{'name': 'val', 'type': 'dummy', 'cfg': {'database_name': 'nerf/scene'}}],
                   'loss': ['nerf_render'], 'val_metric': ['shape_render' if stage == 'geometry' else 'mat_render'],
                   'key_metric_name': 'psnr', 'lr_type': 'warm_up_cos', 'lr_cfg': {},
                   'shader_cfg': {'diffuse_sample_num': 512, 'specular_sample_num': 1024},
                   'reg_energy_loss_lambda': 0.01}
        (self.configs / 'legacy.yaml').write_text(yaml.safe_dump(payload))
        envelope = {'schema_version': 1, 'stage': stage, 'legacy_config': 'legacy.yaml',
                    'inputs': {'dataset': 'nerf'}, 'preprocessing': {'cache': 'cache/datasets'}, 'overrides': {}}
        if stage == 'material':
            envelope['inputs']['mesh'] = {'root': 'thesis', 'path': 'mesh.ply'}
        if checkpoint:
            envelope['inputs']['checkpoint'] = {'root': 'thesis', 'path': 'never-load.pth'}
        path = self.configs / 'smoke.yaml'
        path.write_text(yaml.safe_dump(envelope))
        return load_config(path, config_root=self.configs)

    def readonly(self, path):
        path = Path(path)
        return path != self.root and not (path == self.outputs or self.outputs in path.parents)

    def backend(self, experiment, mark):
        events, models = self.events, self.models
        class Dummy:
            def __getitem__(self, index):
                assert index == 0
                return {}
        class Trainer:
            def __init__(self, cfg, *, experiment):
                self.cfg = {'random_seed': 6033, **cfg}
                experiment.record_owner('trainer', self.cfg)
            def _init_dataset(self):
                events.append('datasets')
                self.train_set = SimpleNamespace(dataset=Dummy())
            def _init_network(self):
                raise AssertionError('optimizer-producing initializer forbidden')
            def run(self):
                raise AssertionError('training loop forbidden')
        class Model:
            def __init__(self, cfg, **options):
                self.cfg = {'train_ray_num': 512, **cfg}
                self.options, self.calls = options, []
                self.shader_network = SimpleNamespace(cfg={'reg_energy_loss_lambda': 0.05, **cfg['shader_cfg']})
                self.tbn, self.train_ids, self.test_ids, self.nvs_ids = 512, ['1', '2'], ['3'], ['4']
                models.append(self)
                assert options['preprocessing'] is experiment.preprocessing
                if cfg['network'] == 'material':
                    assert cfg['mesh'] == str(experiment.plan.selected.mesh)
                    assert options['output_paths'] is experiment.run
                else:
                    assert options['isolated_geometry'] is True
            def cuda(self):
                events.append('cuda')
                return self
            def train(self):
                events.append('train_mode')
            def __call__(self, data):
                self.calls.append(data)
                assert data == {'step': 0}
                assert experiment.run.root in Path(os.environ['TORCH_EXTENSIONS_DIR']).parents
                assert experiment.run.root in Path(os.environ['HOME']).parents
                primary = 'ray_rgb' if self.cfg['network'] == 'shape' else 'rgb_pr'
                return {primary: Tensor(np.ones((2, 3), dtype=np.float32)), 'loss_rgb': Tensor([0.1])}
        return one_forward(experiment, Trainer, {'shape': Model, 'material': Model}, self.torch, mark)

    def test_one_forward_each_stage_preserves_inputs_settings_and_cache_isolation(self):
        for stage in ['geometry', 'material']:
            cfg = self.config(stage)
            plan = validate_smoke(cfg, self.resolver, stage)
            before = snapshot(self.inputs)
            with deny_input_writes(self.inputs):
                run, report = execute_smoke(plan, stage, backend=self.backend, readonly=self.readonly)
            self.assertEqual(before, snapshot(self.inputs))
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['backend'], 'injected_cpu_fixture')
            self.assertEqual(report['result']['forward_calls'], 1)
            self.assertEqual(self.models[-1].calls, [{'step': 0}])
            self.assertEqual(list(run.checkpoints.iterdir()), [])
            self.assertEqual(self.models[-1].cfg['shader_cfg'], cfg.legacy['shader_cfg'])
            self.assertEqual(self.models[-1].shader_network.cfg['reg_energy_loss_lambda'], 0.05)
            self.assertTrue(run.output('logs/console.log').exists())
            self.assertTrue(run.output('config.original.yaml').exists())
            self.assertEqual(json.loads(run.output('smoke.json').read_text())['phase'], 'complete')

    def test_preflight_and_cli_import_no_native_modules_allocate_nothing_and_preserve_rng(self):
        cfg = self.config()
        state, numpy_state = random.getstate(), np.random.get_state()
        validate_smoke(cfg, self.resolver, 'preflight')
        self.assertEqual(state, random.getstate())
        self.assertEqual(numpy_state[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        script = """
import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    assert name.split('.')[0] not in {'torch','numpy','network','train','dataset','open3d','nvdiffrast','raytracing'}
    return original(name,*args,**kwargs)
builtins.__import__=guard
from run_native_smoke import main
raise SystemExit(main(sys.argv[1:]))
"""
        result = subprocess.run([sys.executable, '-B', '-c', script, '--cfg', str(cfg.source),
                                 '--config-root', str(self.configs), '--datasets-root', str(self.datasets),
                                 '--thesis-root', str(self.thesis), '--output-root', str(self.outputs),
                                 '--run-id', 'cli', '--validate-only'], cwd=CODE_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.outputs.exists())
        # Actual temporary inputs are writable: explicit execute must fail its
        # mount gate before attempting any guarded native/scientific import.
        execute_args = result.args[:-1] + ['--execute']
        rejected = subprocess.run(execute_args, cwd=CODE_ROOT, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2, rejected.stderr)
        self.assertIn('read-only', rejected.stderr)
        self.assertFalse(self.outputs.exists())

    def test_writable_input_mount_rejected_before_allocation_or_backend(self):
        plan = validate_smoke(self.config(), self.resolver, 'unsafe')
        with self.assertRaisesRegex(RunError, 'read-only'):
            execute_smoke(plan, 'unsafe', backend=lambda *_: self.fail('backend reached'), readonly=lambda _: False)
        self.assertFalse(self.outputs.exists())

    def test_checkpoint_configuration_and_unsafe_run_id_rejected(self):
        with self.assertRaisesRegex(ConfigError, 'checkpoint'):
            validate_smoke(self.config(checkpoint=True), self.resolver, 'checkpoint')
        with self.assertRaises(ValueError):
            validate_smoke(self.config(), self.resolver, '../escape')
        self.assertFalse(self.outputs.exists())

    def test_selected_input_filesystems_and_output_mount_are_checked(self):
        plan = validate_smoke(self.config('material'), self.resolver, 'mounts')
        for path in [self.mesh, plan.selected.dataset, self.configs,
                     plan.selected.dataset / 'scene', self.thesis]:
            with self.subTest(path=path), self.assertRaisesRegex(RunError, 'read-only'):
                check_mounts(plan, lambda p: False if p == path else self.readonly(p))
        with self.assertRaisesRegex(RunError, 'output filesystem'):
            check_mounts(plan, lambda _: True)
        self.assertFalse(self.outputs.exists())

    def test_collision_never_overwrites_or_implicitly_resumes(self):
        plan = validate_smoke(self.config(), self.resolver, 'unique')
        run, _ = execute_smoke(plan, 'unique', backend=self.backend, readonly=self.readonly)
        before = snapshot(run.root)
        with self.assertRaises(RunError):
            execute_smoke(plan, 'unique', backend=self.backend, readonly=self.readonly)
        self.assertEqual(before, snapshot(run.root))

    def test_failure_records_phase_and_restores_runtime_environment(self):
        plan = validate_smoke(self.config(), self.resolver, 'failure')
        prior = {k: os.environ.get(k) for k in ['HOME', 'TMPDIR', 'TORCH_EXTENSIONS_DIR']}
        cwd = Path.cwd()
        def failure(experiment, mark):
            mark('initializing')
            raise RuntimeError('CPU simulated native failure')
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            execute_smoke(plan, 'failure', backend=failure, readonly=self.readonly)
        self.assertEqual(prior, {k: os.environ.get(k) for k in prior})
        self.assertEqual(cwd, Path.cwd())
        record = json.loads((self.outputs / 'failure/smoke.json').read_text())
        self.assertEqual((record['status'], record['phase']), ('failed', 'initializing'))
        self.assertIn('CPU simulated', (self.outputs / 'failure/logs/console.log').read_text())

    def test_nonfinite_empty_and_missing_primary_outputs_fail(self):
        for value in [{'loss': Tensor([0.1])}, {'ray_rgb': Tensor(np.empty((0, 3)))},
                      {'ray_rgb': Tensor([[float('nan'), 0, 0]])}]:
            with self.subTest(value=value), self.assertRaises(RunError):
                tensor_summary(value, self.torch, 'geometry')

    def test_native_backend_requires_exactly_one_visible_gpu(self):
        plan = validate_smoke(self.config(), self.resolver, 'gpu')
        self.torch.cuda.device_count = lambda: 2
        with self.assertRaisesRegex(RunError, 'exactly one'):
            execute_smoke(plan, 'gpu', backend=self.backend, readonly=self.readonly)
        self.assertEqual(self.models, [])


if __name__ == '__main__':
    unittest.main()
