"""Material planning and storage; no scientific imports or model execution."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from infrastructure.cache import source_manifest
from infrastructure.config import ConfigError
from infrastructure.geometry import (CODE_ROOT, GeometryExperiment, GeometryPlan, _canonical,
                                     _signature, _single_name, _write_yaml)
from infrastructure.paths import PathError, contained_path
from infrastructure.runs import RunError, RunStore, validate_run_id


ALGORITHM = "anisotropic-e1e859779301faced46d6476691e20b546c27ae0"
# Reviewed implementations with the same baseline material mathematics. This is
# an explicit origin declaration, not inference from filenames or branch names.
APPROVED_ORIGINS = {"e1e859779301faced46d6476691e20b546c27ae0",
                    "66037fb00a4662693e7bc35694e760429d3d0076"}
KNOWN_LIMITS = [
    "Material shader receives shader_cfg only; top-level shader-like settings remain unforwarded.",
    "Checkpoint shape compatibility does not establish scientific equivalence across historical branches.",
    "Continuation restores network/optimizer/progress without RNG/sampler state; it is not exact.",
]


def file_identity(path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RunError("Input changed while recording identity: {}".format(path))
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "sha256": digest.hexdigest()}


def validate_material(configuration, resolver, initialization_manifest=None):
    """Read-only preflight, preserving raw values and original field scopes."""
    cfg = configuration.legacy
    if configuration.schema_version != 1 or configuration.stage != "material":
        raise ConfigError("This entry point accepts schema-1 material envelopes only")
    if configuration.config_root is None or any(d.level == "error" for d in configuration.diagnostics):
        raise ConfigError("An approved config root and error-free configuration are required")
    if resolver.output_root == CODE_ROOT or CODE_ROOT in resolver.output_root.parents or resolver.output_root in CODE_ROOT.parents:
        raise ConfigError("Material output storage must be separate from the source repository")
    required = {"name", "network", "database_name", "train_dataset_type", "train_dataset_cfg",
                "val_set_list", "loss", "val_metric", "key_metric_name", "lr_type", "lr_cfg"}
    if required - cfg.keys():
        raise ConfigError("Missing material configuration fields: {}".format(", ".join(sorted(required - cfg.keys()))))
    _single_name(cfg["name"], "name")
    if cfg["network"] != "material" or cfg["train_dataset_type"] != "dummy" or cfg.get("multi_gpus", False):
        raise ConfigError("Only the existing single-GPU material/dummy pipeline is supported")
    if cfg["lr_type"] != "warm_up_cos" or cfg.get("optimizer_type", "adam") != "adam":
        raise ConfigError("Use the existing Adam/warm_up_cos implementation")
    losses = {"nerf_render", "mat_reg"}
    if not cfg["loss"] or not set(cfg["loss"]).issubset(losses):
        raise ConfigError("Material integration supports nerf_render and mat_reg losses")
    if "mat_render" not in cfg["val_metric"] or not set(cfg["val_metric"]).issubset(losses | {"mat_render"}):
        raise ConfigError("Material validation requires mat_render and registered material losses")
    if cfg["key_metric_name"] != "psnr":
        raise ConfigError("Only the baseline psnr key metric is supported")
    if not isinstance(cfg["train_dataset_cfg"], dict) or cfg["train_dataset_cfg"].get("database_name") != cfg["database_name"]:
        raise ConfigError("Training and renderer database identities must agree")
    if not isinstance(cfg["val_set_list"], list) or not cfg["val_set_list"]:
        raise ConfigError("At least one explicit validation set is required")
    names = []
    for item in cfg["val_set_list"]:
        if (not isinstance(item, dict) or item.get("type") != "dummy" or not isinstance(item.get("cfg"), dict)
                or item["cfg"].get("database_name") != cfg["database_name"]):
            raise ConfigError("Validation must use the selected material database")
        _single_name(item.get("name"), "validation name")
        names.append(item["name"])
    if len(set(names)) != len(names):
        raise ConfigError("Validation names must be unique")
    selected = resolver.resolve_inputs(configuration.inputs)
    if selected.mesh is None:
        raise ConfigError("Material experiments require inputs.mesh")
    parts = cfg["database_name"].split("/")
    if (len(parts) not in (2, 3) or parts[0] not in {"syn", "nerf", "real"}
            or (parts[0] == "real") != (len(parts) == 3)):
        raise ConfigError("Unsupported material database; custom remains abstract in the baseline")
    for part in parts:
        _single_name(part, "database component")
    if bool(cfg.get("is_nerf", False)) != (parts[0] == "nerf"):
        raise ConfigError("is_nerf and database family disagree; no correction is applied")
    if "shader_cfg" in cfg and not isinstance(cfg["shader_cfg"], dict):
        raise ConfigError("shader_cfg must be a mapping")
    if type(cfg.get("n_lobes", 1)) is not int or cfg.get("n_lobes", 1) not in (1, 2):
        raise ConfigError("Only baseline one/two-lobe renderer layouts are supported")
    scene = contained_path(selected.dataset, parts[1])
    if not scene.is_dir():
        raise ConfigError("Selected material scene is missing")
    required_paths = (["transforms_train.json", "train", "masks_train", "depth"] if parts[0] == "nerf"
                      else ["0-camera.pkl", "0.png", "0-depth.png", "masks_train"] if parts[0] == "syn"
                      else ["images", "object_point_cloud.ply", "images_{}/masks_train".format(parts[2])])
    if parts[0] == "real" and not (scene / "cache.pkl").is_file():
        required_paths.append("colmap/sparse/0")
    for relative in required_paths:
        if not contained_path(scene, relative).exists():
            raise ConfigError("Required material input is missing: {}".format(scene / relative))
    pipeline = deepcopy(cfg)
    pipeline["dataset_dir"], pipeline["mesh"] = str(selected.dataset), str(selected.mesh)
    for dataset_cfg in [pipeline["train_dataset_cfg"], *[v["cfg"] for v in pipeline["val_set_list"]]]:
        if "dataset_dir" in dataset_cfg:
            dataset_cfg["dataset_dir"] = str(selected.dataset)
    files = ["network/field.py", "network/renderer.py", "network/loss.py", "network/metrics.py",
             "train/trainer.py", "train/lr_common_manager.py", "train/train_tools.py", "train/train_valid.py",
             "dataset/database.py", "dataset/train_dataset.py", "dataset/name2dataset.py",
             "utils/base_utils.py", "utils/pose_utils.py", "utils/draw_utils.py", "utils/dataset_utils.py",
             "utils/raw_utils.py", "utils/ref_utils.py", "network/embedder.py", "configs/synthetic_split_128.pkl",
             "raytracing/__init__.py", "raytracing/raytracer.py", "colmap/read_write_model.py", "colmap/plyfile.py",
             "infrastructure/material.py", "infrastructure/material_execution.py", "infrastructure/geometry.py",
             "infrastructure/geometry_execution.py", "infrastructure/config.py", "infrastructure/cache.py",
             "infrastructure/paths.py", "infrastructure/runs.py", "run_material_experiment.py"]
    versions = {}
    for name in ["torch", "numpy", "opencv-python", "open3d", "scikit-image", "tensorboardX", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {"schema_version": 1, "stage": "material", "algorithm": ALGORITHM,
                "scientific_configuration": _canonical(cfg), "resolved_dataset": str(selected.dataset),
                "mesh": file_identity(selected.mesh), "sources": source_manifest(scene),
                "preprocessing": configuration.preprocessing.cache,
                "code": {p: hashlib.sha256((CODE_ROOT / p).read_bytes()).hexdigest() for p in files},
                "packages": versions, "python": list(sys.version_info[:3])}
    if selected.checkpoint is not None:
        identity["initialization_input"] = file_identity(selected.checkpoint)
    if initialization_manifest is not None:
        path = Path(initialization_manifest).resolve()
        if configuration.config_root not in path.parents or not path.is_file():
            raise ConfigError("Initialization manifest must be a file inside the approved config root")
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ConfigError("Invalid initialization manifest") from exc
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "algorithm", "source_commit", "checkpoint_sha256"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or manifest["algorithm"] != ALGORITHM or not isinstance(manifest["source_commit"], str)
                or manifest["source_commit"] not in APPROVED_ORIGINS
                or selected.checkpoint is None
                or manifest["checkpoint_sha256"] != identity["initialization_input"]["sha256"]):
            raise ConfigError("Initialization manifest must bind this checkpoint to a reviewed anisotropic origin")
        identity["initialization_manifest"] = manifest
    return GeometryPlan(configuration, resolver, selected, pipeline, identity,
                        configuration.source.read_bytes(), configuration.legacy_source.read_bytes())


class MaterialExperiment(GeometryExperiment):
    """Reuse the tested stage-independent storage/checkpoint operations only."""

    def __init__(self, plan, run, mode):
        super().__init__(plan, run, mode)
        self.effective["unresolved"] = [
            "Full provenance, deterministic continuation, source-run/mesh coordinate provenance and native validation remain pending.",
            "Shader defaults are captured after construction; top-level shader fields are never forwarded.",
        ]

    @property
    def checkpoint_tag(self):
        return {**super().checkpoint_tag, "stage": "material", "algorithm": ALGORITHM}

    def verify_inputs(self):
        for key, path in [("mesh", self.plan.selected.mesh), ("initialization_input", self.plan.selected.checkpoint)]:
            if path is not None and file_identity(path) != self.plan.identity[key]:
                raise RunError("Material input changed after validation: {}".format(path))

    def check_output_storage(self):
        # Logger/TensorBoard retain their existing interfaces. Reject pre-existing
        # aliases before opening even the console log; storage has one trusted owner.
        self.run.output('logs')
        for path in self.run.root.rglob('*'):
            if path.is_symlink() or path.is_file() and path.stat().st_nlink > 1:
                raise PathError("Material output storage must not contain symlinks or hard-linked files: {}".format(path))

    def restore(self, network, optimizer, loader, total_step):
        if self.mode == "initialize":
            def checked_loader(path):
                self.verify_inputs()
                state = loader(path)
                tag = state.get("experiment") if isinstance(state, dict) else None
                marked = isinstance(tag, dict) and tag.get("stage") == "material" and tag.get("algorithm") == ALGORITHM
                if tag is not None and not marked:
                    raise RunError("Foreign checkpoint algorithm/stage metadata is incompatible")
                if not marked and "initialization_manifest" not in self.plan.identity:
                    raise RunError("Untagged or foreign material checkpoint requires a reviewed initialization manifest")
                return state
            return super().restore(network, optimizer, checked_loader, total_step)
        return super().restore(network, optimizer, loader, total_step)


def check_material_selection(plan, run_id, mode="new", checkpoint=None, layout="collection"):
    if plan.configuration.stage != 'material' or plan.identity.get('stage') != 'material':
        raise RunError("Material allocation requires a validated material plan")
    validate_run_id(run_id)
    if mode not in {"new", "resume", "initialize"} or (mode == "resume") != (checkpoint is not None):
        raise RunError("Choose new, initialize, or resume with an explicit in-run checkpoint")
    if mode == "new" and plan.selected.checkpoint is not None:
        raise RunError("Historical checkpoint selection requires explicit initialize mode")
    if mode == "initialize" and plan.selected.checkpoint is None:
        raise RunError("Initialize mode requires inputs.checkpoint")
    store = RunStore(plan.resolver, layout)
    if mode == "resume":
        run = store.resume(run_id, checkpoint)
        try:
            record = json.loads(run.output("material.json").read_text())
        except (OSError, ValueError) as exc:
            raise RunError("Missing material run identity") from exc
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("schema_version") != 1 or record.get("run_id") != run_id
                or record.get("signature") != _signature(plan.identity)):
            raise RunError("Material continuation config, mesh, inputs, code or dependencies differ")
        return run
    candidate = store._paths(run_id)
    if candidate.root.exists() and (layout == "collection" or any(candidate.root.iterdir())):
        raise RunError("New material run must not overwrite existing output storage")
    return None


def prepare_material(plan, run_id, mode="new", checkpoint=None, layout="collection"):
    selected = check_material_selection(plan, run_id, mode, checkpoint, layout)
    run = selected if selected is not None else RunStore(plan.resolver, layout).create(run_id)
    experiment = MaterialExperiment(plan, run, mode)
    if mode != "resume":
        run.output("config.original.yaml").write_bytes(plan.original_bytes)
        run.output("config.legacy.yaml").write_bytes(plan.legacy_bytes)
        _write_yaml(run.output("config.effective.yaml"), experiment.effective)
        run.output("material.json").write_text(json.dumps({
            "schema_version": 1, "run_id": run_id, "mode": mode, "signature": experiment.signature,
            "compatibility": plan.identity, "limitations": KNOWN_LIMITS,
            "nvs": "Existing material split, image composition and PSNR/SSIM protocol; destinations only relocated.",
            "provenance_incomplete": ["Git diff/command/environment/lifecycle", "source mesh run and coordinates",
                                      "dataset content hashes", "RNG/sampler state", "native numerical validation"],
        }, sort_keys=True, indent=2) + "\n")
    return experiment
