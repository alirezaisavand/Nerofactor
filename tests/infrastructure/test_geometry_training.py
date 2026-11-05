import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from infrastructure.config import ConfigError, load_config
from infrastructure.geometry import (CODE_ROOT, check_geometry_selection, prepare_geometry,
                                     validate_geometry)
from infrastructure.geometry_execution import execute_geometry, execution_environment
from infrastructure.paths import PathError, PathResolver
from infrastructure.runs import RunError
from dataset_fixtures import deny_input_writes, snapshot
from geometry_fakes import REPO, Tensor, trainer_fixture


BASELINE = "52783b0f23dccdf3e46a707faf090c95b311873a"
INTERFACE_BASELINE = "8194baaf4fba2182e456534d95d21ed27ade5a74"


class GeometryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        scene = self.datasets / "nerf/scene"
        for directory in ["train", "masks_train", "depth"]:
            (scene / directory).mkdir(parents=True)
        (scene / "transforms_train.json").write_text("{}")
        (scene / "train/fixture.png").write_bytes(b"fixture only; fake renderer does not decode")
        self.thesis = self.root / "thesis"
        self.thesis.mkdir()
        self.output_root = self.root / "runs"
        self.resolver = PathResolver({"datasets": self.datasets, "thesis": self.thesis}, self.output_root)
        self.configs = self.root / "configs"
        self.configs.mkdir()
        self.legacy = {
            "name": "scene_shape", "network": "shape", "database_name": "nerf/scene", "is_nerf": True,
            "dataset_dir": "data/nerf", "train_dataset_type": "dummy",
            "train_dataset_cfg": {"database_name": "nerf/scene", "dataset_dir": "old/root"},
            "val_set_list": [{"name": "val", "type": "dummy", "cfg": {"database_name": "nerf/scene"}}],
            "loss": ["nerf_render"], "val_metric": ["shape_render"], "key_metric_name": "psnr",
            "lr_type": "warm_up_cos", "lr_cfg": {}, "total_step": 1, "val_interval": 1,
            "save_interval": 1, "train_log_step": 1, "worker_num": 0,
        }
        self.envelope = {"schema_version": 1, "stage": "geometry", "legacy_config": "legacy.yaml",
                         "inputs": {"dataset": "nerf"}, "preprocessing": {"cache": "cache/preprocessing"}}
        self.path = self.configs / "experiment.yaml"

    def plan(self):
        (self.configs / "legacy.yaml").write_text(yaml.safe_dump(self.legacy, sort_keys=False))
        self.path.write_text(yaml.safe_dump(self.envelope, sort_keys=False))
        self.cfg = load_config(self.path, config_root=self.configs)
        return validate_geometry(self.cfg, self.resolver)

    def test_validation_and_preparation_do_not_load_models_or_consume_rng(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        plan = self.plan()
        check_geometry_selection(plan, "new-run")
        self.assertFalse(self.output_root.exists())
        self.assertEqual(random.getstate(), python_state)
        for a, b in zip(np.random.get_state(), numpy_state):
            np.testing.assert_array_equal(a, b)
        experiment = prepare_geometry(plan, "new-run")
        self.assertEqual(random.getstate(), python_state)
        for a, b in zip(np.random.get_state(), numpy_state):
            np.testing.assert_array_equal(a, b)
        effective = yaml.load(experiment.run.output("config.effective.yaml").read_text(), Loader=yaml.FullLoader)
        self.assertEqual(effective["owner_defaults"], {})
        self.assertTrue(effective["unresolved"])
        self.assertEqual(experiment.run.output("config.original.yaml").read_bytes(), self.path.read_bytes())
        self.assertEqual(self.cfg.legacy["dataset_dir"], "data/nerf")

    def test_pipeline_settings_differ_only_in_explicit_dataset_paths(self):
        plan = self.plan()
        expected = deepcopy(self.cfg.legacy)
        expected["dataset_dir"] = str(self.datasets / "nerf")
        expected["train_dataset_cfg"]["dataset_dir"] = str(self.datasets / "nerf")
        self.assertEqual(plan.pipeline_config, expected)
        self.assertEqual(plan.pipeline_config["lr_cfg"], {})
        self.assertNotIn("experiment", plan.pipeline_config)

    def test_fresh_run_collision_and_material_stage_are_rejected(self):
        plan = self.plan()
        prepare_geometry(plan, "same")
        with self.assertRaises(RunError):
            prepare_geometry(plan, "same")
        with self.assertRaises(RunError):
            check_geometry_selection(plan, "../bad")
        self.envelope["stage"] = "material"
        with self.assertRaises(ConfigError):
            self.plan()

    def test_invalid_geometry_fields_or_missing_inputs_fail_before_allocation(self):
        original = deepcopy(self.legacy)
        changes = [{"val_set_list": []}, {"train_dataset_type": "unknown"}, {"name": "../escape"},
                   {"loss": ["mat_reg"]}, {"key_metric_name": "unknown"}, {"multi_gpus": True},
                   {"val_metric": ["mat_render"]}, {"lr_type": "unknown"},
                   {"train_dataset_cfg": {"database_name": "nerf/other"}}]
        for change in changes:
            self.legacy = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ConfigError):
                self.plan()
        self.legacy = original
        del self.legacy["train_dataset_cfg"]
        with self.assertRaises(ConfigError):
            self.plan()
        self.legacy = original
        (self.datasets / "nerf/scene/transforms_train.json").unlink()
        with self.assertRaises(ConfigError):
            self.plan()
        self.assertFalse(self.output_root.exists())

    def test_cpu_fake_loop_routes_datasets_checkpoints_logs_metrics_and_cache(self):
        plan = self.plan()
        experiment = prepare_geometry(plan, "geometry")
        fixture = trainer_fixture()
        before_inputs, before_thesis = snapshot(self.datasets), snapshot(self.thesis)
        with deny_input_writes(self.datasets), deny_input_writes(self.thesis), redirect_stdout(io.StringIO()):
            execute_geometry(experiment, fixture.Trainer)
        self.assertEqual(snapshot(self.datasets), before_inputs)
        self.assertEqual(snapshot(self.thesis), before_thesis)
        self.assertEqual(len(fixture.parses), 2)  # validation DummyDataset, then renderer
        for name, root, cache in fixture.parses:
            self.assertEqual(name, "nerf/scene")
            self.assertEqual(root, str(self.datasets / "nerf"))
            self.assertIs(cache, experiment.preprocessing)
        self.assertTrue(list(experiment.run.output("cache/preprocessing").rglob("identity.json")))
        self.assertTrue(experiment.run.output("checkpoints/model.pth").is_file())
        self.assertTrue(experiment.run.output("checkpoints/model_best.pth").is_file())
        self.assertTrue(experiment.run.output("logs/train.txt").is_file())
        self.assertTrue(experiment.run.output("logs/val.txt").is_file())
        self.assertTrue(experiment.run.output("logs/events.out.tfevents.fixture").is_file())
        self.assertTrue(experiment.run.output("renders/validation/scene_shape-val/0-index-0.jpg").is_file())
        self.assertTrue(experiment.run.output("logs/console.log").is_file())
        self.assertTrue(fixture.networks[0].isolated_geometry)
        self.assertIn("Final geometry NVS deferred", experiment.run.output("logs/console.log").read_text())
        self.assertEqual(json.loads(experiment.run.output("geometry.json").read_text())["final_nvs"]["status"],
                         "deferred")
        self.assertFalse(experiment.run.output("logs/nvs").exists())
        self.assertFalse(experiment.run.output("renders/nvs").exists())
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "nvs" for e in fixture.events))
        self.assertEqual(fixture.events.count(("torch_seed", 6033)), 1)
        self.assertIn(("clip", 0.7), fixture.events)
        self.assertEqual(fixture.events.count("optimizer_step"), 1)
        self.assertTrue(fixture.writers[0].closed)
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "load" for e in fixture.events))
        with experiment.run.output("checkpoints/model.pth").open("rb") as stream:
            checkpoint = pickle.load(stream)
        self.assertEqual(checkpoint["experiment"], experiment.checkpoint_tag)
        effective = yaml.load(experiment.run.output("config.effective.yaml").read_text(), Loader=yaml.FullLoader)
        self.assertEqual(effective["owner_defaults"]["trainer"]["random_seed"], 6033)
        self.assertEqual(effective["owner_defaults"]["renderer"]["n_samples"], 64)

    def test_legacy_fake_loop_retains_original_paths_and_scientific_sequence(self):
        plan = self.plan()
        legacy_dir = self.root / "legacy_work"
        legacy_dir.mkdir()
        old = trainer_fixture(legacy_nvs=True)
        previous = Path.cwd()
        try:
            os.chdir(legacy_dir)
            with redirect_stdout(io.StringIO()):
                trainer = old.Trainer(deepcopy(plan.pipeline_config))
                trainer.run()
        finally:
            os.chdir(previous)
        self.assertTrue((legacy_dir / "data/model/scene_shape/model.pth").is_file())
        self.assertTrue((legacy_dir / "data/train_vis/scene_shape-val/0-index-0.jpg").is_file())
        self.assertTrue((legacy_dir / "logs/scene_shape/fixture.txt").is_file())
        self.assertTrue(all(cache is None for _, _, cache in old.parses))
        self.assertFalse(old.networks[0].isolated_geometry)
        new = trainer_fixture()
        with redirect_stdout(io.StringIO()):
            execute_geometry(prepare_geometry(plan, "isolated"), new.Trainer)
        self.assertTrue(new.networks[0].isolated_geometry)
        for key in ["loss", "lr_cfg", "random_seed", "n_samples", "rgb_loss", "shader_config"]:
            self.assertEqual(old.networks[0].cfg[key], new.networks[0].cfg[key])
        self.assertEqual([e for e in old.events if isinstance(e, tuple) and e[0] in {"torch_seed", "clip"}],
                         [e for e in new.events if isinstance(e, tuple) and e[0] in {"torch_seed", "clip"}])
        self.assertEqual(old.events.count("optimizer_step"), new.events.count("optimizer_step"))
        self.assertEqual((legacy_dir / "data/model/scene_shape/val.txt").read_bytes(),
                         (self.output_root / "isolated/logs/val.txt").read_bytes())
        self.assertEqual((legacy_dir / "data/train_vis/scene_shape-val/0-index-0.jpg").read_bytes(),
                         (self.output_root / "isolated/renders/validation/scene_shape-val/0-index-0.jpg").read_bytes())
        with (legacy_dir / "data/model/scene_shape/model_best.pth").open("rb") as stream:
            old_best = pickle.load(stream)
        with (self.output_root / "isolated/checkpoints/model_best.pth").open("rb") as stream:
            new_best = pickle.load(stream)
        self.assertEqual((old_best["step"], old_best["best_para"]), (new_best["step"], new_best["best_para"]))
        self.assertEqual(old_best["optimizer_state_dict"], new_best["optimizer_state_dict"])

    def test_resume_requires_explicit_checkpoint_and_matching_run_metadata(self):
        plan = self.plan()
        experiment = prepare_geometry(plan, "run1")
        fixture = trainer_fixture()
        with redirect_stdout(io.StringIO()):
            execute_geometry(experiment, fixture.Trainer)
        with self.assertRaises(RunError):
            prepare_geometry(plan, "run1", mode="resume")
        with self.assertRaises((PathError, RunError)):
            prepare_geometry(plan, "run1", mode="resume", checkpoint="../outside.pth")
        resumed = prepare_geometry(plan, "run1", mode="resume", checkpoint="model_best.pth")
        fresh_fixture = trainer_fixture()
        with redirect_stdout(io.StringIO()):
            execute_geometry(resumed, fresh_fixture.Trainer)
        loaded = [e for e in fresh_fixture.events if isinstance(e, tuple) and e[0] == "load"]
        self.assertEqual(loaded, [("load", str(experiment.run.output("checkpoints/model_best.pth")))])
        self.assertEqual(fresh_fixture.events.count("optimizer_step"), 0)
        self.legacy["total_step"] = 2
        with self.assertRaises(RunError):
            prepare_geometry(self.plan(), "run1", mode="resume", checkpoint="model.pth")

    def test_legacy_checkpoint_cannot_masquerade_as_same_run_resume(self):
        plan = self.plan()
        experiment = prepare_geometry(plan, "run1")
        fixture = trainer_fixture()
        state = {"network_state_dict": {"geometry": Tensor([1])}, "optimizer_state_dict": {},
                 "step": 1, "best_para": 1.0}
        fixture.torch.save(state, experiment.run.output("checkpoints/old.pth"))
        resumed = prepare_geometry(plan, "run1", mode="resume", checkpoint="old.pth")
        network, optimizer = fixture.Renderer(plan.pipeline_config), fixture.Optimizer([])
        with self.assertRaises(RunError):
            resumed.restore(network, optimizer, fixture.torch.load, 1)
        self.assertIsNone(network.loaded)

    def test_historical_checkpoint_initialization_is_explicit_weights_only(self):
        fixture = trainer_fixture()
        historical = self.thesis / "historical.pth"
        fixture.torch.save({"network_state_dict": {"geometry": Tensor([7])}, "step": 9000,
                            "best_para": 99, "optimizer_state_dict": {"old": True}}, historical)
        self.envelope["inputs"]["checkpoint"] = {"root": "thesis", "path": "historical.pth"}
        plan = self.plan()
        before = snapshot(self.thesis)
        with self.assertRaises(RunError):
            prepare_geometry(plan, "implicit")
        self.assertFalse(self.output_root.exists())
        experiment = prepare_geometry(plan, "initialized", mode="initialize")
        network, optimizer = fixture.Renderer(plan.pipeline_config), fixture.Optimizer([])
        self.assertEqual(experiment.restore(network, optimizer, fixture.torch.load, 1), (0, 0))
        self.assertIsNotNone(network.loaded)
        self.assertIsNone(optimizer.loaded)
        self.assertEqual(before, snapshot(self.thesis))
        bad = {"network_state_dict": {"geometry": Tensor([[1]])}}
        with self.assertRaises(RunError):
            experiment.restore(network, optimizer, lambda path: bad, 1)

    def test_fresh_restore_never_implicitly_loads_a_checkpoint(self):
        experiment = prepare_geometry(self.plan(), "new")
        loader = Mock(side_effect=AssertionError("must not load"))
        self.assertEqual(experiment.restore(None, None, loader, 1), (0, 0))
        experiment.run.output("checkpoints/unexpected.pth").write_bytes(b"unexpected")
        with self.assertRaises(RunError):
            experiment.restore(None, None, loader, 1)
        loader.assert_not_called()

    def test_runtime_cache_and_temporary_environment_is_owned_and_restored(self):
        experiment = prepare_geometry(self.plan(), "runtime")
        previous_cwd, previous_tmp, previous_mpl = Path.cwd(), tempfile.tempdir, os.environ.get("MPLCONFIGDIR")
        with execution_environment(experiment):
            self.assertEqual(Path.cwd(), CODE_ROOT)
            self.assertEqual(Path(tempfile.gettempdir()), experiment.run.output("tmp"))
            for name in ["MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TMPDIR"]:
                self.assertIn(experiment.run.root, Path(os.environ[name]).parents)
        self.assertEqual((Path.cwd(), tempfile.tempdir, os.environ.get("MPLCONFIGDIR")),
                         (previous_cwd, previous_tmp, previous_mpl))

    def test_runtime_failure_trace_is_captured_and_environment_restored(self):
        experiment = prepare_geometry(self.plan(), "failed")
        previous = Path.cwd(), tempfile.tempdir, os.environ.get("TORCH_HOME")
        factory = Mock(side_effect=RuntimeError("CPU simulated construction failure"))
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            execute_geometry(experiment, factory)
        self.assertIn("CPU simulated construction failure", experiment.run.output("logs/console.log").read_text())
        self.assertEqual(previous, (Path.cwd(), tempfile.tempdir, os.environ.get("TORCH_HOME")))

    def test_mounted_geometry_run_uses_the_selected_output_root(self):
        plan = self.plan()
        experiment = prepare_geometry(plan, "mounted", layout="mounted")
        self.assertEqual(experiment.run.root, self.output_root)
        self.assertTrue((self.output_root / "geometry.json").is_file())
        self.assertFalse((self.output_root / "mounted").exists())
        with self.assertRaises(RunError):
            prepare_geometry(plan, "other", layout="mounted")

    def test_resume_rejects_wrong_run_and_optimizer_groups_before_state_loading(self):
        plan = self.plan()
        experiment = prepare_geometry(plan, "run")
        fixture = trainer_fixture()
        fixture.torch.save({}, experiment.run.output("checkpoints/selected.pth"))
        resumed = prepare_geometry(plan, "run", mode="resume", checkpoint="selected.pth")
        network, optimizer = fixture.Renderer(plan.pipeline_config), fixture.Optimizer([])
        base = {"network_state_dict": network.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "experiment": resumed.checkpoint_tag, "step": 0, "best_para": 0.0}
        wrong_run = {**base, "experiment": {**resumed.checkpoint_tag, "run_id": "other"}}
        wrong_optimizer = {**base, "optimizer_state_dict": {"param_groups": [], "state": {}}}
        for state in [wrong_run, wrong_optimizer, {**base, "step": 2}, {**base, "best_para": float("nan")}]:
            with self.subTest(state=state), self.assertRaises(RunError):
                resumed.restore(network, optimizer, lambda path: state, 1)
        self.assertIsNone(network.loaded)
        self.assertIsNone(optimizer.loaded)

    def test_cli_validation_imports_no_scientific_modules_and_allocates_nothing(self):
        self.plan()
        script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch','numpy','train','network','dataset','open3d','nvdiffrast'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from run_experiment import main
raise SystemExit(main(sys.argv[1:]))
"""
        args = ["--cfg", str(self.path), "--config-root", str(self.configs),
                "--datasets-root", str(self.datasets), "--thesis-root", str(self.thesis),
                "--output-root", str(self.output_root), "--run-id", "validate", "--validate-only"]
        result = subprocess.run([sys.executable, "-B", "-c", script, *args], cwd=REPO,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no run allocated", result.stdout)
        self.assertFalse(self.output_root.exists())


class GeometrySourceCompatibilityTests(unittest.TestCase):
    def test_scientific_sources_and_legacy_dispatch_are_unchanged(self):
        def previous(path):
            return subprocess.check_output(["git", "show", INTERFACE_BASELINE + ":" + path], cwd=REPO)

        for path in ["run_training.py", "dataset/database.py", "dataset/train_dataset.py", "network/field.py",
                     "network/loss.py", "train/lr_common_manager.py",
                     "train/train_valid.py", "train/train_tools.py"]:
            self.assertEqual(previous(path), (REPO / path).read_bytes(), path)
        old = ast.parse(previous("train/trainer.py"))
        new = ast.parse((REPO / "train/trainer.py").read_text())
        def run(tree):
            trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
            return next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        legacy = run(new).body[-2].body
        old_body = run(old).body
        start = next(i for i, statement in enumerate(old_body) if isinstance(statement, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "logs_dir" for t in statement.targets))
        original = old_body[start:-1]
        # Previously ternaries chose legacy paths when experiment was None.
        for statement in original:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.IfExp):
                statement.value = statement.value.body
        self.assertEqual([ast.dump(n) for n in original], [ast.dump(n) for n in legacy])
        old_shape = next(n for n in ast.parse(previous("network/renderer.py")).body
                         if isinstance(n, ast.ClassDef) and n.name == "NeROShapeRenderer")
        new_shape = next(n for n in ast.parse((REPO / "network/renderer.py").read_text()).body
                         if isinstance(n, ast.ClassDef) and n.name == "NeROShapeRenderer")
        def method(node, name):
            return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        constructor = method(new_shape, "__init__")
        constructor.args.kwonlyargs.pop()
        constructor.args.kw_defaults.pop()
        constructor.body = [n for n in constructor.body if not (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "isolated_geometry" for t in n.targets))]
        self.assertEqual(ast.dump(method(old_shape, "__init__")), ast.dump(constructor))
        initialize = method(new_shape, "_init_dataset")
        initialize.body = [n.orelse[0] if isinstance(n, ast.If) and
                           ast.dump(n.test) == ast.dump(ast.parse('self.isolated_geometry', mode='eval').body)
                           else n for n in initialize.body]
        self.assertEqual(ast.dump(method(old_shape, "_init_dataset")), ast.dump(initialize))

    def test_training_loop_and_shape_math_are_unchanged(self):
        def source(path):
            return subprocess.check_output(["git", "show", BASELINE + ":" + path], cwd=REPO, text=True)

        def cls(text, name):
            return next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)

        def method(node, name):
            return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)

        old = method(cls(source("train/trainer.py"), "Trainer"), "run")
        new = method(cls((REPO / "train/trainer.py").read_text(), "Trainer"), "run")
        def prefix(node):
            result = []
            for statement in node.body:
                if ((isinstance(statement, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "logs_dir" for t in statement.targets))
                        or (isinstance(statement, ast.If) and ast.dump(statement.test) ==
                            ast.dump(ast.parse('self.experiment is None', mode='eval').body))):
                    break
                result.append(ast.dump(statement, include_attributes=False))
            return result
        self.assertEqual(prefix(old), prefix(new))
        old_shape = cls(source("network/renderer.py"), "NeROShapeRenderer")
        new_shape = cls((REPO / "network/renderer.py").read_text(), "NeROShapeRenderer")
        for node in old_shape.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {"__init__", "_init_dataset"}:
                self.assertEqual(ast.dump(node), ast.dump(method(new_shape, node.name)), node.name)
        # Material I/O hooks are now authorized; material tests characterize those
        # hooks and retain exact AST comparisons for every other method.
        old_material = cls(source("network/renderer.py"), "NeROMaterialRenderer")
        new_material = cls((REPO / "network/renderer.py").read_text(), "NeROMaterialRenderer")
        for node in old_material.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {'__init__', '_init_dataset', 'save_pointcloud', 'nvs'}:
                self.assertEqual(ast.dump(node), ast.dump(method(new_material, node.name)), node.name)

    def test_legacy_split_unpack_and_shape_nvs_signature_are_preserved(self):
        tree = ast.parse((REPO / "network/renderer.py").read_text())
        shape = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NeROShapeRenderer")
        init = next(n for n in shape.body if isinstance(n, ast.FunctionDef) and n.name == "_init_dataset")
        parse = Mock(return_value=object())
        scope = {"parse_database_name": parse, "get_database_split": lambda db: ([1], [2], [3])}
        exec(compile(ast.Module(body=[init], type_ignores=[]), "shape_dataset", "exec"), scope)
        cache = object()
        value = SimpleNamespace(cfg={"database_name": "nerf/scene", "dataset_dir": "/selected"},
                                preprocessing=cache, isolated_geometry=False)
        with self.assertRaises(ValueError):
            scope["_init_dataset"](value)
        self.assertIs(parse.call_args.kwargs["preprocessing"], cache)
        nvs = next(n for n in shape.body if isinstance(n, ast.FunctionDef) and n.name == "nvs")
        self.assertEqual([arg.arg for arg in nvs.args.args], ["self", "pose", "K", "h", "w"])


class GeometrySplitAdaptationTests(unittest.TestCase):
    def setUp(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(np.random.set_state, numpy_state)
        database_tree = ast.parse((REPO / "dataset/database.py").read_text())
        split = next(n for n in database_tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_database_split")
        scope = {"random": random, "BaseDatabase": object}
        exec(compile(ast.Module(body=[split], type_ignores=[]), "baseline_split", "exec"), scope)
        self.split = scope["get_database_split"]
        shape = next(n for n in ast.parse((REPO / "network/renderer.py").read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == "NeROShapeRenderer")
        init = next(n for n in shape.body if isinstance(n, ast.FunctionDef) and n.name == "_init_dataset")
        self.database = SimpleNamespace(get_img_ids=lambda: [str(i) for i in range(32)])
        self.parse = Mock(return_value=self.database)
        self.split_spy = Mock(wraps=self.split)
        self.build = Mock(side_effect=lambda db, ids, is_nerf: {"imgs": np.zeros((len(ids), 3, 2, 2))})
        scope = {"np": np, "parse_database_name": self.parse, "get_database_split": self.split_spy,
                 "build_imgs_info": self.build, "imgs_info_to_torch": lambda info, device: info}
        exec(compile(ast.Module(body=[init], type_ignores=[]), "actual_shape_dataset_init", "exec"), scope)
        self.initialize = scope["_init_dataset"]

    def renderer(self, name, *, isolated=True, preprocessing=None):
        poses = Mock()
        poses.float.return_value.cuda.return_value = poses
        return SimpleNamespace(cfg={"database_name": name, "dataset_dir": "/selected"},
                               preprocessing=preprocessing, isolated_geometry=isolated,
                               is_nerf=name.startswith("nerf/"),
                               _construct_ray_batch=Mock(return_value=({}, poses, 7, 2, 2)),
                               _construct_nerf_ray_batch=Mock(return_value=({}, poses, 7, 2, 2)),
                               _shuffle_train_batch=Mock())

    def test_actual_shape_initialization_preserves_all_split_ids_and_rng(self):
        for name in ["syn/scene", "real/scene/128", "nerf/scene"]:
            with self.subTest(name=name):
                random.seed(999)
                initial = random.getstate()
                numpy_state = np.random.get_state()
                train, validation, nvs = self.split(self.database)
                expected_rng = random.getstate()
                random.setstate(initial)
                cache = object()
                value = self.renderer(name, preprocessing=cache)
                self.split_spy.reset_mock()
                with redirect_stdout(io.StringIO()):
                    self.initialize(value)
                self.split_spy.assert_called_once_with(self.database)
                self.parse.assert_called_with(name, "/selected", preprocessing=cache)
                np.testing.assert_array_equal(value.train_ids, np.asarray(train))
                self.assertEqual(value.test_ids, validation)
                self.assertEqual(value.nvs_ids, nvs)
                self.assertEqual(random.getstate(), expected_rng)
                for actual, expected in zip(np.random.get_state(), numpy_state):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual((value.train_num, value.test_num, value.tbn), (len(train), len(validation), 7))
                self.assertEqual(self.build.call_args_list[-2].args[1].tolist(), train)
                self.assertEqual(self.build.call_args_list[-1].args[1], validation)
                value._shuffle_train_batch.assert_called_once_with()
                (value._construct_nerf_ray_batch if value.is_nerf else value._construct_ray_batch).assert_called_once()

    def test_cache_opt_in_alone_does_not_change_legacy_split_behavior(self):
        for cache in [None, object()]:
            with self.subTest(cache=cache):
                value = self.renderer("nerf/scene", isolated=False, preprocessing=cache)
                expected = self.split(self.database)
                expected_rng = random.getstate()
                self.split_spy.reset_mock()
                with self.assertRaises(ValueError):
                    self.initialize(value)
                self.split_spy.assert_called_once_with(self.database)
                self.assertEqual(random.getstate(), expected_rng)
                self.assertFalse(hasattr(value, "nvs_ids"))
                self.assertEqual(len(expected), 3)


if __name__ == "__main__":
    unittest.main()
