import ast
import importlib.util
import json
import io
import pickle
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from infrastructure.cache import PreprocessingCache
from infrastructure.config import load_config
from infrastructure.datasets import load_database
from infrastructure.paths import PathError, PathResolver
from infrastructure.runs import RunStore
from dataset_fixtures import (REPO, baseline_source, create_nerf, create_real, create_synthetic,
                              deny_input_writes, load_module, snapshot)


class DatasetIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_text = baseline_source()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        self.datasets.mkdir()
        self.thesis = self.root / "thesis"
        self.thesis.mkdir()
        self.resolver = PathResolver({"datasets": self.datasets, "thesis": self.thesis}, self.root / "outputs")
        self.run = RunStore(self.resolver).create("fixture")
        self.cache = PreprocessingCache(self.resolver, self.run)
        self.new = load_module((REPO / "dataset/database.py").read_text())
        self.old = load_module(self.baseline_text)
        self.warning_filter = warnings.catch_warnings()
        self.warning_filter.__enter__()
        warnings.simplefilter("ignore", UserWarning)
        self.addCleanup(self.warning_filter.__exit__, None, None, None)
        self.stdout_filter = redirect_stdout(io.StringIO())
        self.stdout_filter.__enter__()
        self.addCleanup(self.stdout_filter.__exit__, None, None, None)

    def compare(self, legacy, isolated):
        self.assertEqual(legacy.img_ids, isolated.img_ids)
        for image_id in legacy.img_ids:
            for getter in ["get_image", "get_K", "get_pose"]:
                a, b = getattr(legacy, getter)(image_id), getattr(isolated, getter)(image_id)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.dtype, b.dtype)

    def test_fresh_real_raw_images_and_colmap_cache_are_isolated(self):
        source = create_real(self.datasets / "real")
        reference_family = self.root / "legacy_real"
        create_real(reference_family)
        before = snapshot(source)
        with deny_input_writes(self.datasets):
            new = self.new.parse_database_name("real/bear/raw_4", str(source.parent), preprocessing=self.cache)
        old = self.old.parse_database_name("real/bear/raw_4", str(reference_family))
        self.compare(old, new)
        self.assertEqual(before, snapshot(source))
        self.assertTrue(new.preprocessing.path("cache.pkl").is_file())
        self.assertEqual((reference_family / "bear/cache.pkl").read_bytes(), new.preprocessing.path("cache.pkl").read_bytes())
        self.assertEqual(len(list(new.preprocessing.path("images_raw_4").glob("*.png"))), 8)
        again = self.new.parse_database_name("real/bear/raw_4", str(source.parent), preprocessing=self.cache)
        self.compare(new, again)
        self.assertEqual(before, snapshot(source))

    def test_existing_legacy_products_and_masks_are_read_only_inputs(self):
        source = create_real(self.datasets / "real", existing=True)
        before = snapshot(source)
        old = self.old.parse_database_name("real/bear/raw_4", str(source.parent))
        # Legacy mkdir on an existing directory is harmless to file contents.
        self.assertEqual(before, snapshot(source))
        with patch.object(self.new, "resize_img", side_effect=AssertionError("must preserve existing processed images")):
            new = self.new.parse_database_name("real/bear/raw_4", str(source.parent), preprocessing=self.cache)
        self.compare(old, new)
        np.testing.assert_array_equal(old.get_mask(1), new.get_mask(1))
        self.assertFalse(new.preprocessing.path("cache.pkl").exists())
        self.assertEqual(before, snapshot(source))

    def test_fresh_cropped_metadata_and_pixels_match_with_the_same_crop_backend(self):
        source = create_real(self.datasets / "real")
        reference_family = self.root / "legacy_real"
        create_real(reference_family)
        before = snapshot(source)
        with deny_input_writes(self.datasets):
            new = self.new.parse_database_name("real/bear/4", str(source.parent), preprocessing=self.cache)
        old = self.old.parse_database_name("real/bear/4", str(reference_family))
        self.compare(old, new)
        self.assertTrue(new.preprocessing.path("images_4/meta_info.pkl").exists())
        self.assertEqual((reference_family / "bear/images_4/meta_info.pkl").read_bytes(),
                         new.preprocessing.path("images_4/meta_info.pkl").read_bytes())
        self.assertEqual(before, snapshot(source))
        with patch.object(self.new, "crop_by_points", side_effect=AssertionError("cache should be used")):
            again = self.new.parse_database_name("real/bear/4", str(source.parent), preprocessing=self.cache)
        self.compare(new, again)

    def test_existing_crop_metadata_and_images_are_preserved_without_regeneration(self):
        source = create_real(self.datasets / "real", existing=True)
        (source / "images_4").mkdir()
        poses, Ks, names, ids = self.new.read_pickle(str(source / "cache.pkl"))
        for image_id in ids:
            Ks[image_id][0, 2] = 19
            self.new.imageio.imwrite(source / "images_4" / names[image_id], np.full((4, 4, 3), 100 + image_id, np.uint8))
        with (source / "images_4/meta_info.pkl").open("wb") as stream:
            pickle.dump([poses, Ks], stream)
        before = snapshot(source)
        old = self.old.parse_database_name("real/bear/4", str(source.parent))
        with deny_input_writes(self.datasets), patch.object(self.new, "crop_by_points", side_effect=AssertionError("preserve existing crop")):
            new = self.new.parse_database_name("real/bear/4", str(source.parent), preprocessing=self.cache)
        self.compare(old, new)
        self.assertEqual(new.get_K(1)[0, 2], 19)
        self.assertEqual(before, snapshot(source))

    def test_custom_method_routing_without_fixing_its_abstract_class_bug(self):
        source = create_real(self.datasets / "custom", scene="object")
        reference_family = self.root / "legacy_custom"
        create_real(reference_family, scene="object")
        before = snapshot(source)
        # Production CustomDatabase lacks get_img_ids. A test-only concrete
        # subclass exercises its preprocessing without repairing that behavior.
        New = type("FixtureCustom", (self.new.CustomDatabase,), {"get_img_ids": lambda db: db.img_ids})
        Old = type("FixtureCustom", (self.old.CustomDatabase,), {"get_img_ids": lambda db: db.img_ids})
        for mode in ["4", "raw_4"]:
            with deny_input_writes(self.datasets):
                new = New("custom/object/" + mode, str(source.parent), preprocessing=self.cache)
            old = Old("custom/object/" + mode, str(reference_family))
            self.compare(old, new)
        self.assertEqual(before, snapshot(source))
        with self.assertRaises(TypeError):
            self.new.CustomDatabase("custom/object/4", str(source.parent), preprocessing=self.cache)

    def test_nerf_pixels_masks_depth_camera_and_splits_are_unchanged(self):
        source = create_nerf(self.datasets / "nerf")
        before = snapshot(source)
        old = self.old.parse_database_name("nerf/scene", str(source.parent))
        new = self.new.parse_database_name("nerf/scene", str(source.parent), preprocessing=self.cache)
        self.compare(old, new)
        for image_id in old.img_ids:
            for a, b in zip(old.get_depth(image_id), new.get_depth(image_id)):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(old.get_mask(image_id), new.get_mask(image_id))
        for split in ["validation", "test"]:
            old_ids = self.old.get_database_split(old, split)
            state = random.getstate()
            new_ids = self.new.get_database_split(new, split)
            for a, b in zip(old_ids, new_ids):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(state, random.getstate())
        self.assertEqual(before, snapshot(source))

    def test_glossy_synthetic_reads_and_view_count_are_unchanged(self):
        source = create_synthetic(self.datasets / "syn")
        before = snapshot(source)
        old = self.old.parse_database_name("syn/scene", str(source.parent))
        new = self.new.parse_database_name("syn/scene", str(source.parent), preprocessing=self.cache)
        self.compare(old, new)
        self.assertEqual(new.img_num, 8)
        for image_id in old.img_ids:
            np.testing.assert_array_equal(old.get_depth(image_id), new.get_depth(image_id))
            np.testing.assert_array_equal(old.get_mask(image_id), new.get_mask(image_id))
        self.assertEqual(before, snapshot(source))
        with self.assertRaises(PathError):
            new.get_image("../escape")

    def test_nerf_point_export_is_redirected_without_changing_point_math(self):
        source = create_nerf(self.datasets / "nerf")
        before = snapshot(source)
        old = self.old.parse_database_name("nerf/scene", str(source.parent))
        new = self.new.parse_database_name("nerf/scene", str(source.parent), preprocessing=self.cache)
        reference = self.root / "reference.ply"
        a = self.old.get_database_eval_points_nerf(old, voxel_size=0, save_path=str(reference))
        state = random.getstate()
        b = self.new.get_database_eval_points_nerf(new, voxel_size=0, save_path="eval_gt.ply")
        self.assertEqual(state, random.getstate())
        np.testing.assert_array_equal(a, b)
        result = next(new.preprocessing.path().glob("evaluations/*/eval_gt.ply"))
        self.assertEqual(reference.read_bytes(), result.read_bytes())
        self.assertEqual(before, snapshot(source))
        with self.assertRaises(PathError):
            self.new.get_database_eval_points_nerf(new, voxel_size=0, save_path="../escape.ply")

    def test_configured_factory_uses_selected_cache_without_rewriting_legacy_fields(self):
        source = create_nerf(self.datasets / "nerf")
        configs = self.root / "configs"
        configs.mkdir()
        (configs / "legacy.yaml").write_text("network: shape\ndatabase_name: nerf/scene\ndataset_dir: data/nerf\n")
        (configs / "experiment.yaml").write_text(json.dumps({
            "schema_version": 1, "stage": "geometry", "legacy_config": "legacy.yaml",
            "inputs": {"dataset": "nerf"}, "preprocessing": {"cache": "custom/cache"}}))
        cfg = load_config(configs / "experiment.yaml", config_root=configs)
        before = snapshot(source)
        with patch.dict("sys.modules", {"dataset.database": self.new}):
            db = load_database(cfg, self.resolver, self.run)
        self.assertIn("custom/cache", str(db.preprocessing.path()))
        self.assertEqual(cfg.legacy["dataset_dir"], "data/nerf")
        self.assertEqual(before, snapshot(source))

    def test_split_and_pure_geometry_helper_asts_are_identical(self):
        names = {"get_database_split", "crop_by_points", "_to_4x4", "_transform_points_col", "_backproject_opengl_raylength"}
        def definitions(source):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name in names}
        self.assertEqual(definitions(self.baseline_text), definitions((REPO / "dataset/database.py").read_text()))

    def test_preprocessing_arithmetic_expressions_are_unchanged(self):
        def expressions(source):
            result = {}
            for cls in ast.parse(source).body:
                if isinstance(cls, ast.ClassDef) and cls.name != "BaseDatabase":
                    for method in cls.body:
                        if isinstance(method, ast.FunctionDef):
                            result[(cls.name, method.name)] = [ast.dump(n, include_attributes=False)
                                for n in ast.walk(method) if isinstance(n, (ast.BinOp, ast.AugAssign))
                                or (isinstance(n, ast.UnaryOp) and not isinstance(n.op, ast.Not))]
            return result
        self.assertEqual(expressions(self.baseline_text), expressions((REPO / "dataset/database.py").read_text()))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Native OpenCV unavailable; fixture backend parity is tested separately")
    def test_native_opencv_raw_resize_parity(self):
        import cv2
        utils = ast.parse((REPO / "utils/base_utils.py").read_text())
        functions = [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name in {"resize_img", "downsample_gaussian_blur"}]
        scope = {"np": np, "cv2": cv2}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "original_resize", "exec"), scope)
        self.old.resize_img = self.new.resize_img = scope["resize_img"]
        source = create_real(self.datasets / "real")
        reference_family = self.root / "legacy_real"
        create_real(reference_family)
        self.compare(self.old.parse_database_name("real/bear/raw_4", str(reference_family)),
                     self.new.parse_database_name("real/bear/raw_4", str(source.parent), preprocessing=self.cache))

    @unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("transforms3d"),
                         "Native OpenCV/transforms3d unavailable; crop stand-in routing is tested separately")
    def test_native_crop_parity(self):
        import cv2
        from transforms3d.euler import euler2mat
        utils = ast.parse((REPO / "utils/base_utils.py").read_text())
        pose_utils = ast.parse((REPO / "utils/pose_utils.py").read_text())
        names = {"downsample_gaussian_blur", "look_at_rotation", "pose_compose", "let_me_look_at_2d", "look_at_crop"}
        functions = [n for tree in [utils, pose_utils] for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {"np": np, "cv2": cv2, "euler2mat": euler2mat}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "original_crop", "exec"), scope)
        self.old.look_at_crop = self.new.look_at_crop = scope["look_at_crop"]
        source = create_real(self.datasets / "real")
        reference_family = self.root / "legacy_real"
        create_real(reference_family)
        self.compare(self.old.parse_database_name("real/bear/4", str(reference_family)),
                     self.new.parse_database_name("real/bear/4", str(source.parent), preprocessing=self.cache))

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "Native Open3D unavailable; point math/path routing is tested separately")
    def test_native_open3d_downsample_and_export_parity(self):
        import open3d
        self.old.o3d = self.new.o3d = open3d
        source = create_nerf(self.datasets / "nerf")
        old = self.old.parse_database_name("nerf/scene", str(source.parent))
        new = self.new.parse_database_name("nerf/scene", str(source.parent), preprocessing=self.cache)
        a = self.old.get_database_eval_points_nerf(old, save_path=str(self.root / "old.ply"))
        b = self.new.get_database_eval_points_nerf(new, save_path="new.ply")
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
