"""Experiment-owned preprocessing storage with versioned source identities."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from uuid import uuid4

from infrastructure.paths import PathError, contained_path, relative_path


class CacheError(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessingSettings:
    cache: str = "cache/datasets"

    def __post_init__(self):
        relative_path(self.cache)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _implementation():
    repo = Path(__file__).resolve().parents[1]
    files = ["dataset/database.py", "infrastructure/cache.py", "utils/base_utils.py",
             "utils/pose_utils.py", "colmap/read_write_model.py", "colmap/plyfile.py"]
    versions = {}
    for name in ["numpy", "opencv-python", "opencv-python-headless", "opencv-contrib-python",
                 "opencv-contrib-python-headless", "scikit-image", "imageio", "open3d",
                 "Pillow", "scipy", "transforms3d"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["loaded_modules"] = {name: getattr(sys.modules.get(name), "__version__", None)
                                  for name in ["numpy", "cv2", "skimage", "imageio", "open3d", "scipy", "transforms3d"]}
    return {"files": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files},
            "python": list(sys.version_info[:3]), "packages": versions}


def _record(path, scene):
    resolved = contained_path(scene, str(path.relative_to(scene)).replace("\\", "/"))
    stat = resolved.stat()
    return {"path": path.relative_to(scene).as_posix(), "target": str(resolved),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


class PreprocessingCache:
    """A factory tied to one declared experiment and its protected input roots.

    Identity uses source names/targets/sizes/mtime/ctime, not full dataset hashes.
    Existing source-side legacy products are preserved as read-only inputs.
    """

    def __init__(self, resolver, run, settings=None):
        self.resolver = resolver
        self.run = run
        self.settings = settings or PreprocessingSettings()
        run_root = run.root.resolve()
        if run_root != resolver.output_root and resolver.output_root not in run_root.parents:
            raise PathError("Experiment is outside the permitted output root")
        resolver.check_output_root()
        self.root = run.output(self.settings.cache)

    def bind(self, database_name, scene, parameters=None):
        parts = database_name.split("/")
        if len(parts) not in (2, 3) or parts[0] not in {"real", "custom", "syn", "nerf"}:
            raise CacheError("Invalid database identity")
        for part in parts:
            if len(relative_path(part).parts) != 1:
                raise CacheError("Database identity components must be single path names")
        scene = Path(scene).resolve()
        datasets = self.resolver.input_roots["datasets"]
        if datasets not in scene.parents or not scene.is_dir():
            raise PathError("Scene must be an existing directory inside the datasets root")
        sources = []
        for path in sorted(scene.rglob("*")):
            if path.is_symlink() and path.is_dir():
                raise PathError("Dataset directory symlinks require explicit materialization/selection")
            if path.is_file() or path.is_symlink():
                sources.append(_record(path, scene))
        identity = {"schema_version": 1, "database_name": database_name,
                    "dataset_root": str(datasets), "scene": str(scene),
                    "parameters": parameters or {}, "implementation": _implementation(),
                    "sources": sources, "legacy_products": "preserve-existing-read-only"}
        key = _digest(identity)
        relative = "{}/{}/{}/{}".format(self.settings.cache, parts[0], parts[1], key)
        return SceneCache(self, scene, relative, identity)


class SceneCache:
    def __init__(self, owner, scene, relative, identity):
        self.owner = owner
        self.scene = scene
        self.relative = relative
        self.identity = identity
        self.sources = {entry["path"]: entry for entry in identity["sources"]}
        root = self.path()
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError:
            try:
                existing = json.loads(self.path("identity.json").read_text())
            except (OSError, ValueError) as exc:
                raise CacheError("Incomplete or unmarked preprocessing cache; not reused") from exc
            if existing != identity:
                raise CacheError("Preprocessing cache identity mismatch; not reused")
        else:
            self.write("identity.json", lambda path: Path(path).write_text(
                json.dumps(identity, sort_keys=True, indent=2, allow_nan=False) + "\n"))

    def path(self, relative=None):
        value = self.relative if relative is None else self.relative + "/" + str(relative_path(relative)).replace("\\", "/")
        return self.owner.run.output(value)

    def source(self, relative):
        path = contained_path(self.scene, relative)
        key = relative_path(relative).as_posix()
        if key in self.sources or path.is_file() or path.is_symlink():
            if not path.is_file():
                raise CacheError("Source disappeared after cache identity was selected: {}".format(key))
            current = _record(self.scene / key, self.scene)
            if self.sources.get(key) != current:
                raise CacheError("Source changed after cache identity was selected: {}".format(key))
        # Validate the target but retain the source filename/suffix seen by the
        # original image decoder, including an in-scene file symlink alias.
        return str(self.scene / relative_path(relative))

    def read(self, relative):
        target = self.path(relative)
        if target.exists():
            return str(target)
        # Match legacy cache-hit semantics without adopting unidentified output
        # namespaces or regenerating already-existing source-side products.
        return self.source(relative)

    def write(self, relative, writer):
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(".{}-{}{}".format(target.stem, uuid4().hex, target.suffix))
        temporary = self.path(temporary.relative_to(self.path()).as_posix())
        try:
            result = writer(str(temporary))
            if result is False:
                raise CacheError("Preprocessing writer reported failure")
            if not temporary.is_file():
                raise CacheError("Preprocessing writer did not create its output")
            temporary.replace(self.path(relative))
            return result
        finally:
            # Only our uniquely named temporary file, never a source product.
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()

    def write_evaluation(self, relative, parameters, writer):
        relative_path(relative)
        return self.write("evaluations/{}/{}".format(_digest(parameters), relative), writer)
