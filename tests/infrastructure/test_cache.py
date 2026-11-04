import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from infrastructure.cache import CacheError, PreprocessingCache, PreprocessingSettings
from infrastructure.config import ConfigError, load_config
from infrastructure.paths import PathError, PathResolver
from infrastructure.runs import RunStore


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        self.scene = self.datasets / "family" / "scene"
        self.scene.mkdir(parents=True)
        self.thesis = self.root / "thesis"
        self.thesis.mkdir()
        (self.scene / "image.png").write_bytes(b"fixture image")
        (self.scene / "cache.pkl").write_bytes(b"existing legacy product")
        self.resolver = PathResolver({"datasets": self.datasets, "thesis": self.thesis}, self.root / "outputs")
        self.run = RunStore(self.resolver).create("fixture")
        self.cache = PreprocessingCache(self.resolver, self.run)

    def test_writes_are_isolated_and_existing_products_are_preserved(self):
        import numpy as np
        before = {p: p.read_bytes() for p in self.scene.rglob("*") if p.is_file()}
        state = random.getstate()
        numpy_state = np.random.get_state()
        scene = self.cache.bind("syn/scene", self.scene)
        self.assertEqual(Path(scene.read("cache.pkl")), self.scene / "cache.pkl")
        scene.write("images_raw_8/image.png", lambda p: Path(p).write_bytes(b"derived"))
        self.assertEqual(Path(scene.read("images_raw_8/image.png")).read_bytes(), b"derived")
        self.assertEqual(before, {p: p.read_bytes() for p in self.scene.rglob("*") if p.is_file()})
        self.assertEqual(state, random.getstate())
        for a, b in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(any(p.name.startswith(".image-") for p in scene.path().rglob("*")))

    def test_identity_changes_with_sources_parameters_scene_and_implementation(self):
        original = self.cache.bind("syn/scene", self.scene)
        again = self.cache.bind("syn/scene", self.scene)
        self.assertEqual(original.path(), again.path())
        changed_parameter = self.cache.bind("syn/scene", self.scene, {"resolution": 64})
        self.assertNotEqual(original.path(), changed_parameter.path())
        (self.scene / "image.png").write_bytes(b"different fixture image")
        changed_source = self.cache.bind("syn/scene", self.scene)
        self.assertNotEqual(original.path(), changed_source.path())
        with patch("infrastructure.cache._implementation", return_value={"version": "different"}):
            changed_code = self.cache.bind("syn/scene", self.scene)
        self.assertNotEqual(changed_source.path(), changed_code.path())
        second = self.datasets / "family" / "second"
        second.mkdir()
        self.assertNotEqual(original.path(), self.cache.bind("syn/second", second).path())

    def test_mismatched_or_incomplete_identity_is_not_reused(self):
        scene = self.cache.bind("syn/scene", self.scene)
        marker = scene.path("identity.json")
        marker.write_text("{}")
        with self.assertRaises(CacheError):
            self.cache.bind("syn/scene", self.scene)
        marker.unlink()
        with self.assertRaises(CacheError):
            self.cache.bind("syn/scene", self.scene)

    def test_same_size_edit_with_preserved_mtime_gets_a_new_identity(self):
        scene = self.cache.bind("syn/scene", self.scene)
        source = self.scene / "image.png"
        original = source.stat()
        source.write_bytes(b"changed image")
        self.assertEqual(source.stat().st_size, original.st_size)
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertNotEqual(scene.path(), self.cache.bind("syn/scene", self.scene).path())

    def test_source_mutation_after_binding_is_rejected(self):
        scene = self.cache.bind("syn/scene", self.scene)
        (self.scene / "cache.pkl").write_bytes(b"modified legacy cache")
        with self.assertRaises(CacheError):
            scene.read("cache.pkl")
        (self.scene / "image.png").unlink()
        with self.assertRaises(CacheError):
            scene.source("image.png")

    def test_source_alias_suffix_is_preserved_after_target_validation(self):
        (self.scene / "alias.jpg").symlink_to(self.scene / "image.png")
        scene = self.cache.bind("syn/scene", self.scene)
        self.assertEqual(Path(scene.source("alias.jpg")), self.scene / "alias.jpg")
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        (self.scene / "escape.png").symlink_to(outside)
        with self.assertRaises(PathError):
            self.cache.bind("syn/scene", self.scene)

    def test_unsafe_paths_symlinks_and_source_overlap_are_rejected(self):
        for value in ["../datasets", "/inputs/datasets", "cache/../escape"]:
            with self.subTest(value=value), self.assertRaises(PathError):
                PreprocessingSettings(value)
        scene = self.cache.bind("syn/scene", self.scene)
        for value in ["../source", "/tmp/source", "a/../source"]:
            with self.subTest(value=value), self.assertRaises(PathError):
                scene.write(value, lambda p: Path(p).write_text("bad"))
        scene.path("alias").symlink_to(self.scene, target_is_directory=True)
        with self.assertRaises(PathError):
            scene.write("alias/bad.txt", lambda p: Path(p).write_text("bad"))
        with self.assertRaises(PathError):
            self.cache.bind("syn/scene", self.thesis)
        for name in ["syn/../scene", "syn/..", "syn/scene/../../bad"]:
            with self.subTest(name=name), self.assertRaises((PathError, CacheError)):
                self.cache.bind(name, self.scene)

    def test_failed_writer_leaves_no_reusable_partial_product(self):
        scene = self.cache.bind("syn/scene", self.scene)

        def fail(path):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated interruption")

        with self.assertRaises(OSError):
            scene.write("cache.pkl", fail)
        self.assertFalse(scene.path("cache.pkl").exists())
        self.assertEqual(list(scene.path().glob(".cache-*")), [])
        self.assertEqual(Path(scene.read("cache.pkl")).read_bytes(), b"existing legacy product")

    def test_evaluation_names_distinguish_parameters(self):
        scene = self.cache.bind("nerf/scene", self.scene)
        for scale in [0.5, 1.0]:
            scene.write_evaluation("eval_gt.ply", {"scene_scale": scale}, lambda p: Path(p).write_text("points"))
        self.assertEqual(len(list(scene.path().glob("evaluations/*/eval_gt.ply"))), 2)

    def test_atomic_temporary_names_preserve_png_and_jpeg_encoding(self):
        import numpy as np
        from skimage.io import imsave
        scene = self.cache.bind("syn/scene", self.scene)
        pixels = np.arange(8 * 8 * 3).reshape(8, 8, 3).astype(np.uint8)
        for suffix in ["png", "jpg"]:
            reference = self.root / ("reference." + suffix)
            imsave(str(reference), pixels)
            scene.write("image." + suffix, lambda path: imsave(path, pixels))
            self.assertEqual(reference.read_bytes(), scene.path("image." + suffix).read_bytes())


class ApprovedConfigRootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.approved = self.root / "configs"
        (self.approved / "experiments").mkdir(parents=True)
        (self.approved / "shape").mkdir()
        self.legacy = self.approved / "shape/scene.yaml"
        self.legacy.write_text("network: shape\n")
        self.envelope = self.approved / "experiments/scene.yaml"

    def write(self, reference="../shape/scene.yaml", preprocessing=None):
        obj = {"schema_version": 1, "stage": "geometry", "legacy_config": reference,
               "inputs": {"dataset": "family"}}
        if preprocessing is not None:
            obj["preprocessing"] = preprocessing
        import yaml
        self.envelope.write_text(yaml.safe_dump(obj))
        return self.envelope

    def test_approved_sibling_and_explicit_root_requirement(self):
        path = self.write(preprocessing={"cache": "custom/preprocessing"})
        with self.assertRaises(ConfigError):
            load_config(path)
        cfg = load_config(path, config_root=self.approved)
        self.assertEqual(cfg.legacy_source, self.legacy)
        self.assertEqual(cfg.preprocessing.cache, "custom/preprocessing")
        self.assertEqual(cfg.config_root, self.approved)
        self.assertIsNone(load_config(self.legacy).preprocessing)

    def test_traversal_and_symlink_escape_fail_before_reading_target(self):
        outside = self.root / "outside.yaml"
        outside.write_text("network: shape\n")
        link = self.approved / "shape/link.yaml"
        link.symlink_to(outside)
        for reference in ["../../outside.yaml", "../shape/link.yaml"]:
            path = self.write(reference)
            with patch("infrastructure.config._read_yaml", wraps=__import__(
                    "infrastructure.config", fromlist=["_read_yaml"])._read_yaml) as reader:
                with self.assertRaises(ConfigError):
                    load_config(path, config_root=self.approved)
                self.assertEqual(reader.call_count, 1)
        with self.assertRaises(ConfigError):
            load_config(self.write(), config_root=Path("configs"))

    def test_invalid_preprocessing_fields_or_paths_fail(self):
        for value in [{"cache": "../bad"}, {"cache": "/outputs"}, {"unknown": True}, []]:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                load_config(self.write(preprocessing=value), config_root=self.approved)


if __name__ == "__main__":
    unittest.main()
