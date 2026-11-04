import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from infrastructure.config import ConfigError, load_config
from infrastructure.material import (ALGORITHM, APPROVED_ORIGINS, check_material_selection,
                                     prepare_material, validate_material)
from infrastructure.material_execution import execute_material
from infrastructure.paths import PathError, PathResolver
from infrastructure.runs import RunError
from dataset_fixtures import deny_input_writes, snapshot
from geometry_fakes import REPO, Tensor
from material_fakes import material_fixture


BASELINE = '66037fb00a4662693e7bc35694e760429d3d0076'


class MaterialTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets, self.thesis, self.configs, self.outputs = [self.root / x for x in ['datasets', 'thesis', 'configs', 'outputs']]
        for directory in [self.datasets, self.thesis, self.configs]:
            directory.mkdir()
        scene = self.datasets / 'nerf/scene'
        for directory in ['train', 'depth', 'masks_train']:
            (scene / directory).mkdir(parents=True)
        (scene / 'transforms_train.json').write_text('{}')
        self.mesh = self.thesis / 'mesh.ply'
        self.mesh.write_text('ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n')
        self.cfg_path = self.configs / 'experiment.yaml'
        self.legacy = {'name': 'scene_material', 'network': 'material', 'database_name': 'nerf/scene', 'is_nerf': True,
                       'mesh': 'legacy/mesh.ply', 'dataset_dir': 'legacy/datasets',
                       'train_dataset_type': 'dummy', 'train_dataset_cfg': {'database_name': 'nerf/scene'},
                       'val_set_list': [{'name': 'val', 'type': 'dummy', 'cfg': {'database_name': 'nerf/scene'}}],
                       'loss': ['nerf_render', 'mat_reg'], 'val_metric': ['mat_render'], 'key_metric_name': 'psnr',
                       'lr_type': 'warm_up_cos', 'lr_cfg': {}, 'total_step': 1, 'val_interval': 1,
                       'save_interval': 1, 'train_log_step': 1, 'worker_num': 0,
                       'reg_energy_loss_lambda': 0.0, 'reg_spec_loss_lambda': 0.0,
                       'shader_cfg': {'diffuse_sample_num': 8, 'specular_sample_num': 16}, 'n_lobes': 2}
        self.envelope = {'schema_version': 1, 'stage': 'material', 'legacy_config': 'legacy.yaml',
                         'inputs': {'dataset': 'nerf', 'mesh': {'root': 'thesis', 'path': 'mesh.ply'}},
                         'preprocessing': {'cache': 'cache/selected'}}
        self.resolver = PathResolver({'datasets': self.datasets, 'thesis': self.thesis}, self.outputs)

    def plan(self, manifest=None, resolver=None):
        (self.configs / 'legacy.yaml').write_text(yaml.safe_dump(self.legacy))
        self.cfg_path.write_text(yaml.safe_dump(self.envelope))
        self.configuration = load_config(self.cfg_path, config_root=self.configs)
        return validate_material(self.configuration, resolver or self.resolver, manifest)

    def execute(self, experiment, fixture=None):
        fixture = fixture or material_fixture()
        with patch.dict(sys.modules, fixture.modules), redirect_stdout(io.StringIO()):
            execute_material(experiment, fixture.Trainer)
        return fixture

    def test_preflight_and_setup_preserve_rng_and_original_configuration(self):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        plan = self.plan()
        check_material_selection(plan, 'material')
        self.assertFalse(self.outputs.exists())
        experiment = prepare_material(plan, 'material')
        self.assertEqual(python_rng, random.getstate())
        for actual, expected in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_array_equal(actual, expected)
        expected = deepcopy(self.legacy)
        expected['mesh'], expected['dataset_dir'] = str(self.mesh), str(self.datasets / 'nerf')
        self.assertEqual(plan.pipeline_config, expected)
        self.assertEqual(plan.configuration.legacy, self.legacy)
        self.assertEqual(experiment.run.output('config.original.yaml').read_bytes(), self.cfg_path.read_bytes())
        identity = json.loads(experiment.run.output('material.json').read_text())
        self.assertEqual(identity['mode'], 'new')
        self.assertEqual(identity['compatibility']['mesh']['sha256'], hashlib.sha256(self.mesh.read_bytes()).hexdigest())
        self.assertTrue(identity['provenance_incomplete'])

    def test_actual_material_io_methods_and_fake_loop_isolate_all_outputs(self):
        plan = self.plan()
        experiment = prepare_material(plan, 'material')
        before = snapshot(self.datasets), snapshot(self.thesis)
        with deny_input_writes(self.datasets), deny_input_writes(self.thesis):
            fixture = self.execute(experiment)
            with patch.dict(sys.modules, fixture.modules):
                fixture.networks[0].save_pointcloud(Tensor([[1, 2, 3]]))
        self.assertEqual(before, (snapshot(self.datasets), snapshot(self.thesis)))
        self.assertEqual([e[1] for e in fixture.events if isinstance(e, tuple) and e[0] == 'mesh_read'], [str(self.mesh)])
        self.assertTrue(all(root == str(self.datasets / 'nerf') and cache is experiment.preprocessing
                            for name, root, cache in fixture.parses))
        for relative in ['checkpoints/model.pth', 'checkpoints/model_best.pth', 'logs/train.txt', 'logs/val.txt',
                         'logs/events.out.tfevents.fixture', 'logs/nvs/log.txt', 'logs/console.log',
                         'renders/validation/scene_material-val/0-index-0.jpg', 'debug/cloud.ply']:
            self.assertTrue(experiment.run.output(relative).is_file(), relative)
        self.assertEqual(len(list(experiment.run.output('renders/nvs').glob('*.png'))), 5)
        self.assertEqual(len([e for e in fixture.events if isinstance(e, tuple) and e[0] == 'test_step']), 5)
        self.assertFalse(any(isinstance(e, tuple) and e[0] == 'load' for e in fixture.events))
        self.assertIn(('clip', 0.7), fixture.events)
        effective = yaml.load(experiment.run.output('config.effective.yaml').read_text(), Loader=yaml.FullLoader)
        shader = effective['owner_defaults']['shader']
        self.assertEqual((shader['reg_energy_loss_lambda'], shader['reg_spec_loss_lambda']), (0.05, 0.05))
        self.assertEqual((shader['diffuse_sample_num'], shader['specular_sample_num']), (8, 16))
        self.assertFalse(shader['is_real'])
        self.assertEqual(effective['owner_defaults']['renderer']['reg_energy_loss_lambda'], 0.0)
        self.assertTrue(fixture.writers[0].closed)

    def test_legacy_and_isolated_validation_nvs_bytes_and_optimizer_agree(self):
        plan = self.plan()
        legacy = self.root / 'legacy'
        legacy.mkdir()
        previous = Path.cwd()
        old = material_fixture()
        try:
            os.chdir(legacy)
            with patch.dict(sys.modules, old.modules), redirect_stdout(io.StringIO()):
                trainer = old.Trainer(deepcopy(plan.pipeline_config))
                trainer.run()
        finally:
            os.chdir(previous)
        experiment = prepare_material(plan, 'new')
        new = self.execute(experiment)
        for old_name, new_name in [('data/model/scene_material/val.txt', 'logs/val.txt'),
                                   ('data/train_vis/scene_material-val/0-index-0.jpg', 'renders/validation/scene_material-val/0-index-0.jpg'),
                                   ('logs/scene_material/log.txt', 'logs/nvs/log.txt')]:
            self.assertEqual((legacy / old_name).read_bytes(), experiment.run.output(new_name).read_bytes())
        for path in (legacy / 'nvs_imgs/scene_material').glob('*.png'):
            self.assertEqual(path.read_bytes(), experiment.run.output('renders/nvs/' + path.name).read_bytes())
        self.assertEqual(old.networks[0].cfg, new.networks[0].cfg)
        self.assertEqual(old.networks[0].shader_network.cfg, new.networks[0].shader_network.cfg)
        np.testing.assert_array_equal(old.networks[0].train_ids, new.networks[0].train_ids)
        self.assertEqual(old.networks[0].test_ids, new.networks[0].test_ids)
        self.assertEqual(old.networks[0].nvs_ids, new.networks[0].nvs_ids)
        with (legacy / 'data/model/scene_material/model_best.pth').open('rb') as stream:
            old_state = pickle.load(stream)
        with experiment.run.output('checkpoints/model_best.pth').open('rb') as stream:
            new_state = pickle.load(stream)
        for key in ['step', 'best_para', 'optimizer_state_dict']:
            self.assertEqual(old_state[key], new_state[key])
        groups = new_state['optimizer_state_dict']['param_groups']
        self.assertEqual((groups[0]['eps'], groups[0]['amsgrad']), (1e-4, True))
        self.assertTrue(all(cache is None for _, _, cache in old.parses))

    def test_previous_geometry_root_is_explicit_and_disjoint_from_outputs(self):
        prior = self.root / 'previous_geometry'
        prior.mkdir()
        (prior / 'surface.ply').write_bytes(self.mesh.read_bytes())
        resolver = PathResolver({'datasets': self.datasets, 'thesis': self.thesis, 'geometry': prior}, self.outputs)
        self.envelope['inputs']['mesh'] = {'root': 'geometry', 'path': 'surface.ply'}
        plan = self.plan(resolver=resolver)
        self.assertEqual(plan.selected.mesh, prior / 'surface.ply')
        with self.assertRaises(PathError):
            PathResolver({'datasets': self.datasets, 'thesis': self.thesis, 'geometry': prior}, self.root)

    def test_invalid_fields_inputs_and_repository_outputs_fail_before_allocation(self):
        original = deepcopy(self.legacy)
        for change in [{'val_set_list': []}, {'loss': ['eikonal']}, {'val_metric': ['shape_render']},
                       {'name': '../escape'}, {'n_lobes': 3}, {'is_nerf': False}]:
            self.legacy = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ConfigError):
                self.plan()
        self.legacy = original
        plan = self.plan()
        with self.assertRaises(RunError):
            check_material_selection(replace(plan, identity={**plan.identity, 'stage': 'geometry'}), 'wrong-stage')
        with self.assertRaises(ConfigError):
            self.plan(resolver=PathResolver({'datasets': self.datasets, 'thesis': self.thesis}, REPO / 'outputs'))
        del self.envelope['inputs']['mesh']
        with self.assertRaises(ConfigError):
            self.plan()
        self.assertFalse(self.outputs.exists())

    def test_collisions_resume_and_changed_mesh_are_rejected_explicitly(self):
        plan = self.plan()
        experiment = prepare_material(plan, 'run')
        self.execute(experiment)
        for mode, checkpoint in [('new', None), ('resume', None), ('resume', '../outside.pth')]:
            with self.subTest(mode=mode), self.assertRaises((PathError, RunError)):
                prepare_material(plan, 'run', mode, checkpoint)
        resumed = prepare_material(plan, 'run', 'resume', 'model_best.pth')
        fixture = self.execute(resumed)
        self.assertEqual(fixture.events.count('optimizer_step'), 0)
        self.assertEqual([e[1] for e in fixture.events if isinstance(e, tuple) and e[0] == 'load'],
                         [str(experiment.run.output('checkpoints/model_best.pth'))])
        self.mesh.write_text('different mesh identity')
        with self.assertRaises(RunError):
            prepare_material(self.plan(), 'run', 'resume', 'model.pth')

    def historical(self, state):
        checkpoint = self.thesis / 'historical.pth'
        with checkpoint.open('wb') as stream:
            pickle.dump(state, stream)
        self.envelope['inputs']['checkpoint'] = {'root': 'thesis', 'path': 'historical.pth'}
        return checkpoint

    def test_untagged_initialization_requires_reviewed_origin_and_loads_weights_only(self):
        fixture = material_fixture()
        plan = self.plan()
        network = fixture.MaterialRenderer(deepcopy(plan.pipeline_config))
        checkpoint = self.historical({'network_state_dict': network.state_dict(), 'step': 9000,
                                      'best_para': 99, 'optimizer_state_dict': {'ignored': True}})
        plan = self.plan()
        with self.assertRaises(RunError):
            prepare_material(plan, 'implicit')
        experiment = prepare_material(plan, 'unchecked', 'initialize')
        optimizer = fixture.Optimizer([])
        with self.assertRaises(RunError):
            experiment.restore(network, optimizer, fixture.torch.load, 1)
        self.assertIsNone(network.loaded)
        manifest = self.configs / 'initialization.json'
        data = {'schema_version': 1, 'algorithm': ALGORITHM, 'source_commit': sorted(APPROVED_ORIGINS)[0],
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        manifest.write_text(json.dumps(data))
        experiment = prepare_material(self.plan(manifest), 'checked', 'initialize')
        before = snapshot(self.thesis)
        self.assertEqual(experiment.restore(network, optimizer, fixture.torch.load, 1), (0, 0))
        self.assertIsNotNone(network.loaded)
        self.assertIsNone(optimizer.loaded)
        self.assertEqual(snapshot(self.thesis), before)
        for change in [{'source_commit': 'ablation_anisotropic'}, {'source_commit': []}, {'checkpoint_sha256': 'wrong'}]:
            manifest.write_text(json.dumps({**data, **change}))
            with self.subTest(change=change), self.assertRaises(ConfigError):
                self.plan(manifest)
        with self.assertRaises(ConfigError):
            self.plan(self.root / 'outside.json')

    def test_incompatible_material_heads_keys_dtypes_and_foreign_tags_reject_before_loading(self):
        fixture = material_fixture()
        plan = self.plan()
        network = fixture.MaterialRenderer(deepcopy(plan.pipeline_config))
        state = {'network_state_dict': network.state_dict(), 'experiment': {'stage': 'material', 'algorithm': ALGORITHM}}
        checkpoint = self.historical(state)
        for index, change in enumerate([
                {'network_state_dict': {'different_head': Tensor([1])}},
                {'network_state_dict': {**state['network_state_dict'], 'shader_network.rotation_predictor.weight': Tensor([[1]])}},
                {'network_state_dict': {**state['network_state_dict'], 'shader_network.rotation_predictor.weight': np.ones((2, 1), np.float64)}},
                {'experiment': {'stage': 'material', 'algorithm': 'aniso_geometry_safe'}}]):
            with checkpoint.open('wb') as stream:
                pickle.dump({**state, **change}, stream)
            experiment = prepare_material(self.plan(), 'bad-' + str(index), 'initialize')
            with self.subTest(change=change), self.assertRaises(RunError):
                experiment.restore(network, fixture.Optimizer([]), fixture.torch.load, 1)
            self.assertIsNone(network.loaded)

    def test_mesh_change_fails_before_model_construction(self):
        experiment = prepare_material(self.plan(), 'changed')
        self.mesh.write_text('changed after validation')
        factory = Mock(side_effect=AssertionError('must not construct'))
        with self.assertRaises(RunError), redirect_stdout(io.StringIO()):
            execute_material(experiment, factory)
        factory.assert_not_called()

    def test_nvs_failure_is_logged_writer_closed_and_environment_restored(self):
        fixture = material_fixture()
        fixture.MaterialRenderer.test_step = Mock(side_effect=RuntimeError('fake NVS failure'))
        experiment = prepare_material(self.plan(), 'failure')
        previous = Path.cwd(), tempfile.tempdir, os.environ.get('TORCH_HOME')
        with self.assertRaises(RuntimeError):
            self.execute(experiment, fixture)
        self.assertEqual(previous, (Path.cwd(), tempfile.tempdir, os.environ.get('TORCH_HOME')))
        self.assertIn('fake NVS failure', experiment.run.output('logs/console.log').read_text())
        self.assertTrue(fixture.writers[0].closed)

    def test_resumed_writer_paths_reject_symlinks_to_protected_files(self):
        for index, relative in enumerate(['logs/nvs/log.txt', 'renders/nvs/test_0.png', 'logs/console.log', 'logs/train.txt',
                                         'renders/validation/scene_material-val/0-index-0.jpg', 'debug/cloud.ply']):
            experiment = prepare_material(self.plan(), 'symlink-' + str(index))
            target = experiment.run.output(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(self.mesh)
            before = snapshot(self.thesis)
            fixture = material_fixture()
            with self.subTest(relative=relative), deny_input_writes(self.thesis), self.assertRaises(PathError):
                if relative.startswith('debug'):
                    network = fixture.MaterialRenderer(deepcopy(experiment.plan.pipeline_config), output_paths=experiment.run)
                    with patch.dict(sys.modules, fixture.modules):
                        network.save_pointcloud(Tensor([[1, 2, 3]]))
                else:
                    self.execute(experiment, fixture)
            self.assertEqual(before, snapshot(self.thesis))

    def test_material_cli_preflight_imports_no_models_and_extra_roots_are_explicit(self):
        self.plan()
        script = """
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch','numpy','train','network','dataset','open3d','nvdiffrast'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from run_material_experiment import main
raise SystemExit(main(sys.argv[1:]))
"""
        args = ['--cfg', str(self.cfg_path), '--config-root', str(self.configs), '--datasets-root', str(self.datasets),
                '--thesis-root', str(self.thesis), '--output-root', str(self.outputs), '--run-id', 'validate', '--validate-only']
        result = subprocess.run([sys.executable, '-B', '-c', script, *args], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no run allocated', result.stdout)
        self.envelope['inputs']['mesh'] = {'root': 'geometry', 'path': 'mesh.ply'}
        self.plan(resolver=PathResolver({'datasets': self.datasets, 'thesis': self.thesis, 'geometry': self.thesis}, self.outputs))
        extra = ['--input-root', 'geometry=' + str(self.thesis)]
        result = subprocess.run([sys.executable, '-B', '-c', script, *args, *extra], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-B', '-c', script, *args, '--input-root', 'thesis=' + str(self.thesis)],
                                cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.outputs.exists())

    def test_tagged_weights_initialization_nested_shader_overrides_and_fresh_checkpoint_guard(self):
        self.legacy['shader_cfg']['reg_energy_loss_lambda'] = 0.003
        plan = self.plan()
        fixture = material_fixture()
        network = fixture.MaterialRenderer(deepcopy(plan.pipeline_config))
        self.assertEqual(network.shader_network.cfg['reg_energy_loss_lambda'], 0.003)
        self.historical({'network_state_dict': network.state_dict(),
                         'experiment': {'stage': 'material', 'algorithm': ALGORITHM}})
        experiment = prepare_material(self.plan(), 'tagged', 'initialize')
        self.assertEqual(experiment.restore(network, fixture.Optimizer([]), fixture.torch.load, 1), (0, 0))
        self.assertIsNotNone(network.loaded)
        del self.envelope['inputs']['checkpoint']
        fresh = prepare_material(self.plan(), 'fresh')
        loader = Mock(side_effect=AssertionError('no implicit checkpoint loading'))
        self.assertEqual(fresh.restore(None, None, loader, 1), (0, 0))
        fresh.run.output('checkpoints/unexpected.pth').write_bytes(b'unexpected')
        with self.assertRaises(RunError):
            fresh.restore(None, None, loader, 1)
        loader.assert_not_called()

    def test_hard_linked_logs_are_rejected_before_execution_or_console_writes(self):
        experiment = prepare_material(self.plan(), 'hardlink')
        os.link(self.mesh, experiment.run.output('logs/train.txt'))
        before = snapshot(self.thesis)
        with self.assertRaises(PathError), deny_input_writes(self.thesis):
            self.execute(experiment)
        self.assertFalse(experiment.run.output('logs/console.log').exists())
        self.assertEqual(before, snapshot(self.thesis))


class MaterialSourceCompatibilityTests(unittest.TestCase):
    def test_only_authorized_io_hooks_change_and_geometry_renderer_is_unchanged(self):
        def previous(path, commit=BASELINE):
            return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)
        def cls(source, name):
            return next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
        def method(node, name):
            return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for path in ['run_training.py', 'run_experiment.py', 'infrastructure/geometry.py',
                     'infrastructure/geometry_execution.py', 'network/field.py', 'network/loss.py',
                     'train/lr_common_manager.py', 'train/train_tools.py', 'train/train_valid.py',
                     'dataset/database.py', 'dataset/train_dataset.py']:
            self.assertEqual(previous(path), (REPO / path).read_bytes(), path)
        for origin in APPROVED_ORIGINS:
            self.assertEqual(previous('network/field.py', origin), (REPO / 'network/field.py').read_bytes())
            self.assertEqual(ast.dump(cls(previous('network/renderer.py', origin), 'NeROMaterialRenderer')),
                             ast.dump(cls(previous('network/renderer.py'), 'NeROMaterialRenderer')))
        old_source, new_source = previous('network/renderer.py'), (REPO / 'network/renderer.py').read_text()
        self.assertEqual(ast.dump(cls(old_source, 'NeROShapeRenderer')), ast.dump(cls(new_source, 'NeROShapeRenderer')))
        old, new = cls(old_source, 'NeROMaterialRenderer'), cls(new_source, 'NeROMaterialRenderer')
        constructor = method(new, '__init__')
        constructor.args.kwonlyargs.clear()
        constructor.args.kw_defaults.clear()
        constructor.body = [n for n in constructor.body if not (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr in {'preprocessing', 'output_paths'} for t in n.targets))]
        self.assertEqual(ast.dump(method(old, '__init__')), ast.dump(constructor))
        initialize = method(new, '_init_dataset')
        initialize.body[0] = initialize.body[0].body[0]  # same legacy parse call
        self.assertEqual(ast.dump(method(old, '_init_dataset')), ast.dump(initialize))

        class StripDestinations(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'self.output_paths is not None':
                    return None
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id in {'filename', 'destination'} for t in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == 'imsave':
                    node.args[0] = deepcopy(original_image_path)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'write_point_cloud':
                    node.args[0] = ast.Constant('cloud.ply')
                return node

        old_nvs, new_nvs = method(old, 'nvs'), method(new, 'nvs')
        original_image_path = next(n.args[0] for n in ast.walk(old_nvs)
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'imsave')
        self.assertEqual(ast.dump(old_nvs), ast.dump(StripDestinations().visit(new_nvs)))
        self.assertEqual(ast.dump(method(old, 'save_pointcloud')),
                         ast.dump(StripDestinations().visit(method(new, 'save_pointcloud'))))
        for node in old.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {'__init__', '_init_dataset', 'save_pointcloud', 'nvs'}:
                self.assertEqual(ast.dump(node), ast.dump(method(new, node.name)), node.name)
        old_metrics, new_metrics = previous('network/metrics.py'), (REPO / 'network/metrics.py').read_text()
        self.assertEqual(ast.dump(cls(old_metrics, 'ShapeRenderMetrics')), ast.dump(cls(new_metrics, 'ShapeRenderMetrics')))
        old_metric = method(cls(old_metrics, 'MaterialRenderMetrics'), '__call__')
        original_image_path = next(n.args[0] for n in ast.walk(old_metric)
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'imsave')
        self.assertEqual(ast.dump(old_metric), ast.dump(StripDestinations().visit(
            method(cls(new_metrics, 'MaterialRenderMetrics'), '__call__'))))


if __name__ == '__main__':
    unittest.main()
