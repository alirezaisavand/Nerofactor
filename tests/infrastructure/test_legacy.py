"""Characterize existing numerical plumbing without importing GPU/science code."""

import abc
import ast
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from infrastructure.config import legacy_merge, load_config


REPO = Path(__file__).resolve().parents[2]


def class_node(path, name):
    return next(n for n in ast.parse((REPO / path).read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)


def defaults(path, name):
    node = class_node(path, name)
    return next(ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "default_cfg" for t in n.targets))


class LegacyCharacterizationTests(unittest.TestCase):
    def test_loading_does_not_import_training_or_gpu_dependencies(self):
        script = """
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'numpy', 'network', 'dataset', 'train', 'open3d', 'nvdiffrast'}
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from infrastructure.config import load_config
from infrastructure.paths import PathResolver
from infrastructure.runs import RunStore
load_config('configs/shape/syn/bell.yaml')
"""
        subprocess.run([sys.executable, "-B", "-c", script], cwd=REPO, check=True, capture_output=True)

    def test_trainer_seed_loss_list_and_nested_defaults_are_preserved(self):
        cfg = load_config(REPO / "configs/shape/nerf/smoothcylinder.yaml")
        effective = cfg.with_defaults(defaults("train/trainer.py", "Trainer"))
        self.assertEqual(effective["random_seed"], 6033)
        self.assertEqual(effective["total_step"], 300000)
        self.assertEqual(effective["lr_cfg"], {})
        self.assertEqual(effective["apply_opacity_loss"], "True,")
        self.assertEqual(effective["loss"], ['nerf_render', 'eikonal', 'std', 'init_sdf_reg',
                                             'occ', 'mask', 'curv', 'opacity'])
        warm = legacy_merge(defaults("train/lr_common_manager.py", "WarmUpCosLR"), effective["lr_cfg"])
        self.assertEqual(warm, {"end_warm": 5000, "end_iter": 300000, "lr": 0.0005})
        opacity = legacy_merge(defaults("network/loss.py", "OpacityLoss"), effective)
        self.assertEqual([opacity[k] for k in ["opacity_loss_weight_begin", "opacity_loss_weight_end",
                                              "opacity_weight_decay_begin", "opacity_weight_decay_end"]],
                         [0, 0.01, 10000, 50000])

    def test_existing_optimizer_and_scheduler_at_boundaries(self):
        # Execute only the original scheduler classes with a scalar math stand-in
        # and recording optimizer. No Trainer/renderer/PyTorch import or step.
        tree = ast.parse((REPO / "train/lr_common_manager.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        scope = {"abc": abc, "np": SimpleNamespace(cos=math.cos, pi=math.pi)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_scheduler", "exec"), scope)

        class Optimizer:
            def __init__(self, parameters, **kwargs):
                self.kwargs = kwargs
                self.param_groups = [{}]

        cfg = load_config(REPO / "configs/material/nerf/smoothcylinder.yaml")
        manager = scope["WarmUpCosLR"](cfg.legacy["lr_cfg"])
        optimizer = manager.construct_optimizer(Optimizer, SimpleNamespace(parameters=lambda: []))
        self.assertEqual(optimizer.kwargs, {"lr": 0.001, "eps": 0.0001, "amsgrad": True})
        for step in [0, 999, 1000, 99999, 100000, 100001, 300000]:
            if step < 1000:
                factor = step / 1000
            else:
                progress = (min(step, 100000) - 1000) / 99000
                factor = (math.cos(math.pi * progress) + 1) * 0.5 * 0.8 + 0.2
            self.assertAlmostEqual(manager(optimizer, step), 0.0005 * factor, places=15)
        trainer = class_node("train/trainer.py", "Trainer")
        clips = [n for n in ast.walk(trainer) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "max_grad_norm" for t in n.targets)]
        self.assertEqual(ast.literal_eval(clips[0].value), 0.7)

    def test_top_level_shader_fields_are_not_silently_forwarded(self):
        cfg = load_config(REPO / "configs/material/nerf/smoothcylinder.yaml")
        renderer_cfg = cfg.with_defaults(defaults("network/renderer.py", "NeROMaterialRenderer"))
        renderer_node = class_node("network/renderer.py", "NeROMaterialRenderer")
        init_shader = next(n for n in renderer_node.body if isinstance(n, ast.FunctionDef) and n.name == "_init_shader")
        shader_defaults = defaults("network/field.py", "MCShadingNetwork")

        def capture_shader(values, trace):
            return legacy_merge(shader_defaults, values)

        scope = {"MCShadingNetwork": capture_shader}
        exec(compile(ast.Module(body=[init_shader], type_ignores=[]), "legacy_shader_setup", "exec"), scope)
        renderer = SimpleNamespace(cfg=renderer_cfg, trace=lambda *args: None)
        scope["_init_shader"](renderer)
        shader = renderer.shader_network
        self.assertEqual(cfg.legacy["reg_energy_loss_lambda"], 0)
        self.assertEqual(shader["reg_energy_loss_lambda"], 0.05)
        self.assertEqual(shader["reg_spec_loss_lambda"], 0.05)
        self.assertEqual(shader["specular_sample_num"], 1024)
        self.assertEqual(shader["change_eps"], 0.02)
        self.assertTrue(shader["anisotropy"])
        self.assertFalse(shader["is_real"])


if __name__ == "__main__":
    unittest.main()
