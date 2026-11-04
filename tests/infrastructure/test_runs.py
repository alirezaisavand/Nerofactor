from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from infrastructure.paths import PathError, PathResolver
from infrastructure.runs import RunError, RunStore, generate_run_id, validate_run_id


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {}
        for name in ["datasets", "thesis"]:
            self.inputs[name] = self.root / name
            self.inputs[name].mkdir()
        self.outputs = self.root / "outputs"
        self.resolver = PathResolver(self.inputs, self.outputs)
        self.store = RunStore(self.resolver)

    def test_generated_ids_are_unique_without_changing_python_rng(self):
        state = random.getstate()
        ids = [generate_run_id("bell-geometry-s6033") for _ in range(50)]
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(random.getstate(), state)
        for value in ids:
            self.assertEqual(validate_run_id(value), value)
            self.assertLessEqual(len(value), 64)

    def test_invalid_ids_fail_before_writing(self):
        for value in ["", ".", "..", "../escape", "/absolute", "Run1", "a/b", "a\\b", " a", "a" * 65, None, 1]:
            with self.subTest(value=value), self.assertRaises(RunError):
                validate_run_id(value)
        with self.assertRaises(RunError):
            self.store.create("../escape")
        with self.assertRaises(RunError):
            generate_run_id("a" * 32)
        self.assertFalse(self.outputs.exists())

    def test_new_run_destinations_and_identity(self):
        run = self.store.create("bell-geometry-s6033")
        self.assertEqual(run.root, self.outputs / run.run_id)
        self.assertEqual(run.checkpoints, run.root / "checkpoints")
        self.assertTrue(run.logs.is_dir())
        self.assertTrue(run.cache.is_dir())
        self.assertIsNone(run.resume_checkpoint)
        identity = json.loads((run.root / ".experiment.json").read_text())
        self.assertEqual(identity, {"schema_version": 1, "run_id": run.run_id})

    def test_collision_does_not_overwrite_or_implicitly_resume(self):
        run = self.store.create("run1")
        marker = (run.root / ".experiment.json").read_bytes()
        (run.checkpoints / "last.pth").write_text("preserve")
        with self.assertRaises(RunError):
            self.store.create("run1")
        self.assertEqual((run.root / ".experiment.json").read_bytes(), marker)
        self.assertEqual((run.checkpoints / "last.pth").read_text(), "preserve")
        (self.outputs / "empty").mkdir()
        (self.outputs / "file").write_text("preserve")
        for name in ["empty", "file"]:
            with self.subTest(name=name), self.assertRaises(RunError):
                self.store.create(name)

    def test_concurrent_collision_has_one_winner(self):
        def create(_):
            try:
                self.store.create("same")
                return True
            except RunError:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(create, range(8))), 1)

    def test_resume_requires_explicit_run_and_checkpoint(self):
        run = self.store.create("run1")
        with self.assertRaises(RunError):
            self.store.resume("run1")
        with self.assertRaises(RunError):
            self.store.resume("run1", "missing.pth")
        (run.checkpoints / "selected.pth").write_text("fixture, never deserialized")
        before = sorted(p.relative_to(self.outputs) for p in self.outputs.rglob("*"))
        resumed = self.store.resume("run1", "selected.pth")
        self.assertEqual(resumed.resume_checkpoint, run.checkpoints / "selected.pth")
        self.assertEqual(before, sorted(p.relative_to(self.outputs) for p in self.outputs.rglob("*")))
        historical = self.inputs["thesis"] / "model.pth"
        historical.write_text("historical input")
        for value in [str(historical), "../../thesis/model.pth", "../logs/file"]:
            with self.subTest(value=value), self.assertRaises(PathError):
                self.store.resume("run1", value)
        self.assertEqual(historical.read_text(), "historical input")

    def test_outputs_and_checkpoint_symlinks_cannot_cross_runs(self):
        first = self.store.create("first")
        second = self.store.create("second")
        (second.checkpoints / "model.pth").write_text("other run")
        (first.checkpoints / "link.pth").symlink_to(second.checkpoints / "model.pth")
        with self.assertRaises(PathError):
            self.store.resume("first", "link.pth")
        for value in ["../second/logs/log.txt", str(second.logs / "log.txt")]:
            with self.subTest(value=value), self.assertRaises(PathError):
                first.output(value)

    def test_resume_rejects_missing_corrupt_mismatched_or_partial_allocations(self):
        with self.assertRaises(RunError):
            self.store.resume("missing", "selected.pth")
        self.assertFalse(self.outputs.exists())
        run = self.store.create("run1")
        marker = run.root / ".experiment.json"
        for value in ["", "[]", '{"schema_version":true,"run_id":"run1"}',
                      '{"schema_version":1,"run_id":"other"}']:
            marker.write_text(value)
            with self.subTest(value=value), self.assertRaises(RunError):
                self.store.resume("run1", "selected.pth")
        marker.write_text('{"schema_version":1,"run_id":"run1"}')
        run.cache.rmdir()
        with self.assertRaises(RunError):
            self.store.resume("run1", "selected.pth")

    def test_mounted_layout_uses_selected_root_and_requires_empty_initialization(self):
        self.outputs.mkdir()
        store = RunStore(self.resolver, layout="mounted")
        run = store.create("mounted-run")
        self.assertEqual(run.root, self.outputs)
        self.assertFalse((self.outputs / "mounted-run").exists())
        (run.checkpoints / "step-1.pth").write_text("fixture")
        for value in ["mounted-run", "different"]:
            with self.subTest(value=value), self.assertRaises(RunError):
                store.create(value)
        self.assertEqual(store.resume("mounted-run", "step-1.pth").root, self.outputs)
        with self.assertRaises(RunError):
            store.resume("different", "step-1.pth")

    def test_mounted_layout_preserves_arbitrary_existing_files(self):
        self.outputs.mkdir()
        foreign = self.outputs / "keep.txt"
        foreign.write_text("preserve")
        with self.assertRaises(RunError):
            RunStore(self.resolver, layout="mounted").create("run1")
        self.assertEqual(foreign.read_text(), "preserve")
        self.assertEqual(list(self.outputs.iterdir()), [foreign])

    def test_partial_allocation_is_retained_and_not_reused(self):
        original = Path.mkdir

        def failing_mkdir(path, *args, **kwargs):
            if path.name == "cache":
                raise OSError("simulated allocation failure")
            return original(path, *args, **kwargs)

        with patch.object(Path, "mkdir", failing_mkdir), self.assertRaises(RunError):
            self.store.create("partial")
        self.assertTrue((self.outputs / "partial").is_dir())
        with self.assertRaises(RunError):
            self.store.create("partial")
        with self.assertRaises(RunError):
            self.store.resume("partial", "selected.pth")


if __name__ == "__main__":
    unittest.main()
