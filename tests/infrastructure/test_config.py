import os
from pathlib import Path
import tempfile
import unittest

import yaml

from infrastructure.config import ConfigError, legacy_merge, load_config


REPO = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy.yaml"
        self.legacy.write_text("network: shape\nname: sample\nrandom_seed: 6033\nlr_cfg: {}\n")

    def write(self, text):
        path = self.root / "experiment.yaml"
        path.write_text(text)
        return path

    def envelope(self, **updates):
        data = {"schema_version": 1, "stage": "geometry", "legacy_config": "legacy.yaml",
                "inputs": {"dataset": "GlossySynthetic"}}
        data.update(updates)
        return self.write(yaml.safe_dump(data))

    def test_every_existing_legacy_yaml_matches_full_loader(self):
        paths = list((REPO / "configs").rglob("*.yaml"))
        paths = [p for p in paths if "experiments" not in p.parts]
        self.assertEqual(len(paths), 70)
        for path in paths:
            with self.subTest(path=str(path)):
                cfg = load_config(path)
                self.assertEqual(cfg.legacy, yaml.load(path.read_text(), Loader=yaml.FullLoader))
                self.assertEqual(cfg.original, cfg.legacy)
                self.assertEqual(cfg.schema_version, 0)
                self.assertIsNone(cfg.inputs)

    def test_legacy_invalid_boolean_is_preserved_and_reported(self):
        path = REPO / "configs/shape/nerf/smoothcylinder.yaml"
        cfg = load_config(path)
        self.assertEqual(cfg.legacy["apply_opacity_loss"], "True,")
        self.assertTrue(any(d.field == "apply_opacity_loss" and d.level == "error" for d in cfg.diagnostics))
        with self.assertRaises(ConfigError):
            load_config(path, strict=True)

    def test_unused_and_unvalidated_fields_are_retained(self):
        path = self.write("network: material\nanisotropy: false\nreg_energy_loss_lambda: 0\nunused_research_note: keep\n")
        cfg = load_config(path)
        self.assertEqual(cfg.legacy["unused_research_note"], "keep")
        self.assertFalse(cfg.legacy["anisotropy"])
        self.assertTrue(any(d.code == "not-forwarded" for d in cfg.diagnostics))
        self.assertTrue(any(d.field == "unused_research_note" and d.code == "unvalidated-field" for d in cfg.diagnostics))

    def test_full_loader_scalar_alias_and_merge_semantics(self):
        cfg = load_config(self.write("network: shape\nbase: &base {x: 1, y: 2}\nmerged: {<<: *base, x: 3}\nflag: yes\n"))
        self.assertEqual(cfg.legacy["merged"], {"x": 3, "y": 2})
        self.assertIs(cfg.legacy["flag"], True)
        self.assertFalse(any(d.code == "duplicate-key" for d in cfg.diagnostics))

    def test_duplicate_keys_keep_last_value_and_strict_rejects(self):
        path = self.write("network: shape\nrandom_seed: 1\nrandom_seed: 6033\n")
        cfg = load_config(path)
        self.assertEqual(cfg.legacy["random_seed"], 6033)
        self.assertTrue(any(d.code == "duplicate-key" for d in cfg.diagnostics))
        with self.assertRaises(ConfigError):
            load_config(path, strict=True)

    def test_shallow_default_and_override_behavior(self):
        defaults = {"random_seed": 6033, "lr_cfg": {"lr": 0.0005, "end_iter": 300000}}
        override = {"lr_cfg": {"end_warm": 1000}}
        merged = legacy_merge(defaults, override)
        self.assertIs(merged["lr_cfg"], override["lr_cfg"])
        self.assertEqual(merged["lr_cfg"], {"end_warm": 1000})
        self.assertEqual(merged["random_seed"], 6033)
        path = self.envelope(overrides={"lr_cfg": {"end_iter": 100000}})
        cfg = load_config(path, overrides=override)
        self.assertEqual(cfg.legacy["lr_cfg"], {"end_warm": 1000})
        self.assertEqual(cfg.original["overrides"]["lr_cfg"], {"end_iter": 100000})
        override["lr_cfg"]["end_warm"] = 9
        self.assertEqual(cfg.legacy["lr_cfg"]["end_warm"], 1000)

    def test_envelope_reference_is_anchored_and_loading_has_no_side_effects(self):
        path = self.envelope()
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        old = Path.cwd()
        try:
            os.chdir(self.root.parent)
            cfg = load_config(path)
        finally:
            os.chdir(old)
        self.assertEqual(cfg.legacy_source, self.legacy)
        self.assertEqual(cfg.stage, "geometry")
        self.assertEqual(cfg.inputs.dataset, "GlossySynthetic")
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob("*")))

    def test_unknown_envelope_field_fails_strict_and_is_diagnostic_in_compatibility(self):
        path = self.envelope(algorithm="unimplemented")
        with self.assertRaises(ConfigError):
            load_config(path)
        cfg = load_config(path, strict=False)
        self.assertEqual(cfg.original["algorithm"], "unimplemented")
        self.assertTrue(any(d.code == "unknown-field" for d in cfg.diagnostics))
        self.assertNotIn("algorithm", cfg.legacy)

    def test_material_requires_mesh_and_matching_renderer(self):
        with self.assertRaises(ConfigError):
            load_config(self.envelope(stage="material"))
        self.legacy.write_text("network: material\n")
        cfg = load_config(self.envelope(stage="material", inputs={
            "dataset": "GlossyReal", "mesh": {"root": "thesis", "path": "meshes/bear.ply"},
            "checkpoint": {"root": "thesis", "path": "model/bear/model.pth"}}))
        self.assertEqual(cfg.inputs.mesh.root, "thesis")
        self.assertEqual(cfg.inputs.checkpoint.path, "model/bear/model.pth")
        with self.assertRaises(ConfigError):
            load_config(self.envelope())

    def test_invalid_envelope_structures_and_paths(self):
        cases = [
            {"schema_version": True}, {"schema_version": 2}, {"schema_version": 0},
            {"stage": "training"}, {"overrides": []}, {"inputs": []},
            {"inputs": {"dataset": "../data"}}, {"inputs": {"dataset": "/data"}},
            {"inputs": {"dataset": "data", "typo": 1}},
            {"inputs": {"dataset": "data", "mesh": {"root": "thesis", "path": "mesh.ply"}}},
            {"legacy_config": "/etc/example.yaml"}, {"legacy_config": "C:\\example.yaml"},
            {"legacy_config": "$CONFIG"}, {"legacy_config": "missing.yaml"},
        ]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ConfigError):
                load_config(self.envelope(**updates))
        for text in ["[]", "", "key: [", "text"]:
            with self.subTest(text=text), self.assertRaises(ConfigError):
                load_config(self.write(text))

    def test_no_recursive_envelope_includes(self):
        path = self.envelope(legacy_config="experiment.yaml")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_invalid_source_paths_and_encoding_are_configuration_errors(self):
        for value in [None, "bad\x00path", self.root / "missing.yaml"]:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                load_config(value)
        path = self.root / "invalid_encoding.yaml"
        path.write_bytes(b"network: \xff\n")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_invalid_network_values_are_diagnostic_without_crashing(self):
        for network in [[], {}, True, "unknown", None]:
            path = self.write(yaml.safe_dump({"network": network}))
            with self.subTest(network=network):
                self.assertIsNone(load_config(path).stage)
                with self.assertRaises(ConfigError):
                    load_config(path, strict=True)

    def test_stage_override_and_artifact_reference_validation(self):
        with self.assertRaises(ConfigError):
            load_config(self.envelope(), overrides={"network": "material"})
        self.legacy.write_text("network: material\n")
        for mesh in ["mesh.ply", {}, {"root": "thesis", "path": "../mesh.ply"},
                     {"root": "thesis", "path": "mesh.ply", "typo": True}]:
            with self.subTest(mesh=mesh), self.assertRaises(ConfigError):
                load_config(self.envelope(stage="material", inputs={"dataset": "data", "mesh": mesh}))

    def test_invalid_legacy_values_are_not_coerced(self):
        for field, value in [("total_step", "50000"), ("total_step", True),
                             ("random_seed", -1), ("worker_num", -1),
                             ("lr_cfg", []), ("loss", "nerf_render"),
                             ("dataset_dir", "your_dataset_dir"),
                             ("optimizer_type", "sgd"), ("lr_type", "exp_decay")]:
            path = self.write(yaml.safe_dump({"network": "shape", field: value}))
            with self.subTest(field=field, value=value):
                self.assertEqual(load_config(path).legacy[field], value)
                with self.assertRaises(ConfigError):
                    load_config(path, strict=True)
        for lr in [True, 0, -1, float("nan"), float("inf"), "1e-3"]:
            with self.subTest(lr=lr), self.assertRaises(ConfigError):
                load_config(self.write(yaml.safe_dump({"network": "shape", "lr_cfg": {"lr": lr}})), strict=True)

    def test_checked_in_examples_are_strict_and_preserve_referenced_yaml(self):
        for name in ["geometry.example.yaml", "material.example.yaml"]:
            cfg = load_config(REPO / "configs/experiments" / name)
            self.assertEqual(cfg.schema_version, 1)
            self.assertFalse(any(d.level == "error" for d in cfg.diagnostics))
            self.assertEqual(cfg.legacy, yaml.load(cfg.legacy_source.read_text(), Loader=yaml.FullLoader))


if __name__ == "__main__":
    unittest.main()
