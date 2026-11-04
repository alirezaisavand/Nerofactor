from pathlib import Path
import tempfile
import unittest

from infrastructure.paths import InputRef, InputSelection, PathError, PathResolver


class PathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.datasets = self.root / "data"
        self.thesis = self.root / "thesis"
        self.datasets.mkdir()
        self.thesis.mkdir()
        (self.datasets / "family").mkdir()
        (self.thesis / "mesh.ply").write_text("mesh fixture")
        (self.thesis / "model.pth").write_text("checkpoint fixture")
        self.outputs = self.root / "outputs"
        self.resolver = PathResolver({"datasets": self.datasets, "thesis": self.thesis}, self.outputs)

    def test_explicit_inputs_and_destinations_create_nothing(self):
        selected = self.resolver.resolve_inputs(InputSelection(
            "family", InputRef("thesis", "mesh.ply"), InputRef("thesis", "model.pth")))
        self.assertEqual(selected.dataset, self.datasets / "family")
        self.assertEqual(selected.mesh, self.thesis / "mesh.ply")
        self.assertEqual(selected.checkpoint.read_text(), "checkpoint fixture")
        self.assertEqual(self.resolver.output("run1/cache/crop.png"), self.outputs / "run1/cache/crop.png")
        self.assertFalse(self.outputs.exists())

    def test_invalid_paths_and_unknown_roots(self):
        for value in ["", ".", "..", "../escape", "a/../escape", "/absolute", "a//b",
                      "a/./b", "a/", "C:/file", "a\\b", "$ROOT/file", "~/file", "a\x00b", "a\nb"]:
            with self.subTest(value=value), self.assertRaises(PathError):
                self.resolver.output(value)
        with self.assertRaises(PathError):
            self.resolver.input(InputRef("unknown", "file"))
        with self.assertRaises(PathError):
            self.resolver.input(InputRef("thesis", "missing.ply"))
        with self.assertRaises(PathError):
            self.resolver.input(InputRef("thesis", "mesh.ply"), directory=True)

    def test_root_validation_and_overlap(self):
        roots = {"datasets": self.datasets, "thesis": self.thesis}
        for output in [self.datasets, self.datasets / "cache", self.root, self.thesis / "runs"]:
            with self.subTest(output=output), self.assertRaises(PathError):
                PathResolver(roots, output)
        for output in ["relative", None, self.thesis / "mesh.ply"]:
            with self.subTest(output=output), self.assertRaises(PathError):
                PathResolver(roots, output)
        with self.assertRaises(PathError):
            PathResolver({"datasets": self.datasets}, self.outputs)
        with self.assertRaises(PathError):
            PathResolver({"datasets": self.root / "missing", "thesis": self.thesis}, self.outputs)
        self.assertFalse((self.root / "missing").exists())

    def test_symlink_input_containment(self):
        (self.thesis / "inside.ply").symlink_to(self.thesis / "mesh.ply")
        self.assertEqual(self.resolver.input(InputRef("thesis", "inside.ply")), self.thesis / "mesh.ply")
        (self.root / "outside.ply").write_text("outside")
        (self.thesis / "escape.ply").symlink_to(self.root / "outside.ply")
        with self.assertRaises(PathError):
            self.resolver.input(InputRef("thesis", "escape.ply"))

    def test_writable_symlinks_are_rejected_even_when_pointing_inside(self):
        self.outputs.mkdir()
        (self.outputs / "real").mkdir()
        for name, target in [("inside", self.outputs / "real"), ("outside", self.datasets)]:
            (self.outputs / name).symlink_to(target, target_is_directory=True)
            with self.subTest(name=name), self.assertRaises(PathError):
                self.resolver.output(name + "/result.txt")

    def test_output_root_replacement_is_rejected(self):
        self.outputs.symlink_to(self.datasets, target_is_directory=True)
        with self.assertRaises(PathError):
            self.resolver.output("result.txt")

    def test_additional_prior_geometry_input_root(self):
        geometry = self.root / "prior_geometry"
        geometry.mkdir()
        (geometry / "surface.ply").write_text("prior mesh")
        resolver = PathResolver({"datasets": self.datasets, "thesis": self.thesis,
                                 "geometry": geometry}, self.outputs)
        self.assertEqual(resolver.input(InputRef("geometry", "surface.ply")), geometry / "surface.ply")
        with self.assertRaises(TypeError):
            resolver.input_roots["thesis"] = self.outputs


if __name__ == "__main__":
    unittest.main()
