"""Geometry experiment planning and storage, without scientific imports."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
from pathlib import Path
import sys
from uuid import uuid4

import yaml

from infrastructure.cache import PreprocessingCache, source_manifest
from infrastructure.config import ConfigError
from infrastructure.paths import contained_path, relative_path
from infrastructure.runs import RunError, RunStore, validate_run_id


CODE_ROOT = Path(__file__).resolve().parents[1]
KNOWN_LIMITS = [
    "Isolated geometry final NVS is deferred to a separately versioned evaluation protocol.",
    "Resume restores network/optimizer/progress with legacy seeding, without RNG/sampler restoration; it is not exact.",
]
_LOSSES = {"nerf_render", "eikonal", "std", "init_sdf_reg", "occ", "mask", "curv", "opacity", "fg", "bg"}


def _single_name(value, field):
    try:
        if len(relative_path(value).parts) != 1:
            raise ValueError()
    except ValueError as exc:
        raise ConfigError("{} must be a single safe path component".format(field)) from exc


def _canonical(value):
    # FullLoader-compatible types are retained, including tuples and string flags.
    return yaml.dump(value, Dumper=yaml.Dumper, sort_keys=True)


def _signature(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class GeometryPlan:
    configuration: object
    resolver: object
    selected: object
    pipeline_config: dict
    identity: dict
    original_bytes: bytes
    legacy_bytes: bytes


def validate_geometry(configuration, resolver):
    """Validate paths/structure without allocation, imports, seeds, or checkpoints."""
    cfg = configuration.legacy
    if configuration.schema_version != 1 or configuration.stage != "geometry":
        raise ConfigError("This entry point accepts schema-1 geometry envelopes only")
    if configuration.config_root is None or any(d.level == "error" for d in configuration.diagnostics):
        raise ConfigError("An approved config root and error-free configuration are required")
    required = {"name", "network", "database_name", "train_dataset_type", "train_dataset_cfg",
                "val_set_list", "loss", "val_metric", "key_metric_name", "lr_type", "lr_cfg"}
    missing = required - cfg.keys()
    if missing:
        raise ConfigError("Missing geometry configuration fields: {}".format(", ".join(sorted(missing))))
    _single_name(cfg["name"], "name")
    if cfg["network"] != "shape" or cfg["train_dataset_type"] != "dummy" or cfg.get("multi_gpus", False):
        raise ConfigError("Only the existing single-GPU shape/dummy pipeline is supported")
    if cfg["lr_type"] != "warm_up_cos" or cfg.get("optimizer_type", "adam") != "adam":
        raise ConfigError("Use the existing Adam/warm_up_cos implementation")
    if not cfg["loss"] or not set(cfg["loss"]).issubset(_LOSSES):
        raise ConfigError("Unsupported geometry loss selection")
    if not set(cfg["val_metric"]).issubset(_LOSSES | {"shape_render"}) or "shape_render" not in cfg["val_metric"]:
        raise ConfigError("Geometry validation requires shape_render and registered geometry losses")
    if cfg["key_metric_name"] != "psnr":
        raise ConfigError("Only the baseline psnr key metric is registered")
    if not isinstance(cfg["train_dataset_cfg"], dict) or cfg["train_dataset_cfg"].get("database_name") != cfg["database_name"]:
        raise ConfigError("Training and renderer database identities must agree")
    if not isinstance(cfg["val_set_list"], list) or not cfg["val_set_list"]:
        raise ConfigError("At least one explicit validation set is required")
    names = []
    for item in cfg["val_set_list"]:
        if (not isinstance(item, dict) or item.get("type") != "dummy"
                or not isinstance(item.get("cfg"), dict)
                or item["cfg"].get("database_name") != cfg["database_name"]):
            raise ConfigError("Validation must use the same explicitly selected geometry database")
        _single_name(item.get("name"), "validation name")
        names.append(item["name"])
    if len(set(names)) != len(names):
        raise ConfigError("Validation names must be unique")
    selected = resolver.resolve_inputs(configuration.inputs)
    parts = cfg["database_name"].split("/")
    if len(parts) not in (2, 3) or parts[0] not in {"real", "syn", "nerf"}:
        raise ConfigError("Unsupported geometry database (custom remains abstract in the baseline)")
    if (parts[0] == "real") != (len(parts) == 3):
        raise ConfigError("Invalid geometry database name structure")
    for part in parts:
        _single_name(part, "database component")
    if bool(cfg.get("is_nerf", False)) != (parts[0] == "nerf"):
        raise ConfigError("is_nerf and database family disagree; no automatic correction is applied")
    scene = contained_path(selected.dataset, parts[1])
    if not scene.is_dir():
        raise ConfigError("Selected scene directory is missing: {}".format(scene))
    if parts[0] == "nerf":
        required_paths = ["transforms_train.json", "train", "masks_train", "depth"]
    elif parts[0] == "real":
        required_paths = ["images", "object_point_cloud.ply", "images_{}/masks_train".format(parts[2])]
        if not (scene / "cache.pkl").is_file():
            required_paths.append("colmap/sparse/0")
    else:
        required_paths = ["0-camera.pkl", "0.png", "0-depth.png", "masks_train"]
    for relative in required_paths:
        if not contained_path(scene, relative).exists():
            raise ConfigError("Required geometry input is missing: {}".format(scene / relative))
    if not (CODE_ROOT / "assets/bsdf_256_256.bin").is_file():
        raise ConfigError("The baseline BSDF lookup asset is missing")
    pipeline = deepcopy(cfg)
    pipeline["dataset_dir"] = str(selected.dataset)
    for dataset_cfg in [pipeline["train_dataset_cfg"], *[v["cfg"] for v in pipeline["val_set_list"]]]:
        if "dataset_dir" in dataset_cfg:
            dataset_cfg["dataset_dir"] = str(selected.dataset)
    scientific_files = ["train/trainer.py", "train/lr_common_manager.py", "train/train_valid.py",
                        "network/renderer.py", "network/field.py", "network/loss.py", "network/metrics.py",
                        "dataset/database.py", "dataset/train_dataset.py", "configs/synthetic_split_128.pkl",
                        "train/train_tools.py", "utils/base_utils.py", "utils/pose_utils.py", "utils/dataset_utils.py",
                        "utils/ref_utils.py", "utils/raw_utils.py", "network/embedder.py",
                        "colmap/read_write_model.py", "colmap/plyfile.py",
                        "infrastructure/geometry.py", "infrastructure/geometry_execution.py",
                        "infrastructure/config.py", "infrastructure/cache.py", "infrastructure/paths.py",
                        "infrastructure/runs.py", "run_experiment.py",
                        "assets/bsdf_256_256.bin"]
    versions = {}
    for name in ["torch", "numpy", "opencv-python", "scikit-image", "open3d", "tensorboardX", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {"schema_version": 1, "stage": "geometry", "scientific_configuration": _canonical(cfg),
                "resolved_dataset": str(selected.dataset), "sources": source_manifest(scene),
                "preprocessing": configuration.preprocessing.cache,
                "code": {p: hashlib.sha256((CODE_ROOT / p).read_bytes()).hexdigest() for p in scientific_files},
                "packages": versions, "python": list(sys.version_info[:3])}
    if selected.checkpoint is not None:
        stat = selected.checkpoint.stat()
        identity["initialization_input"] = {"path": str(selected.checkpoint), "size": stat.st_size,
                                            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    return GeometryPlan(configuration, resolver, selected, pipeline, identity,
                        configuration.source.read_bytes(), configuration.legacy_source.read_bytes())


def _write_yaml(path, value):
    temporary = path.with_name(".{}-{}.tmp".format(path.name, uuid4().hex))
    try:
        temporary.write_text(yaml.dump(value, Dumper=yaml.Dumper, sort_keys=False))
        temporary.replace(path)
    finally:
        if temporary.is_file():
            temporary.unlink()


class GeometryExperiment:
    """One selected run. No scientific model is constructed by this object."""

    def __init__(self, plan, run, mode):
        self.plan, self.run, self.mode = plan, run, mode
        self.signature = _signature(plan.identity)
        self.preprocessing = PreprocessingCache(plan.resolver, run, plan.configuration.preprocessing)
        self.effective = {"original_legacy_values": deepcopy(plan.configuration.legacy),
                          "pipeline_config_after_path_resolution": deepcopy(plan.pipeline_config),
                          "diagnostics": [vars(d).copy() for d in plan.configuration.diagnostics],
                          "owner_defaults": {},
                          "unresolved": ["renderer, shader, loss, and scheduler defaults are not fully captured; "
                                         "baseline source hashes and original scopes remain authoritative"]}

    @property
    def checkpoint_tag(self):
        return {"schema_version": 1, "run_id": self.run.run_id, "signature": self.signature,
                "continuation": "network_optimizer_step_without_rng"}

    def record_owner(self, name, values):
        self.effective["owner_defaults"][name] = deepcopy(values)
        _write_yaml(self.run.output("config.effective.yaml"), self.effective)

    def restore(self, network, optimizer, loader, total_step):
        if self.mode == "new":
            if any(self.run.checkpoints.iterdir()):
                raise RunError("Fresh runs cannot adopt existing checkpoint files")
            return 0, 0
        source = self.run.resume_checkpoint if self.mode == "resume" else self.plan.selected.checkpoint
        state = loader(str(source))  # Only reached for an explicit execute request.
        if not isinstance(state, dict) or not isinstance(state.get("network_state_dict"), dict):
            raise RunError("Checkpoint must contain network_state_dict")
        if self.mode == "resume":
            tag = state.get("experiment")
            if (not isinstance(tag, dict) or type(tag.get("schema_version")) is not int
                    or tag != self.checkpoint_tag):
                raise RunError("Checkpoint lacks matching same-run geometry compatibility metadata")
            if (type(state.get("step")) is not int or not 0 <= state["step"] <= total_step
                    or not isinstance(state.get("optimizer_state_dict"), dict)
                    or isinstance(state.get("best_para"), bool) or not isinstance(state.get("best_para"), Real)
                    or not math.isfinite(state["best_para"])):
                raise RunError("Invalid checkpoint progress/optimizer/metric state")
            saved_groups = state["optimizer_state_dict"].get("param_groups")
            current_groups = optimizer.state_dict().get("param_groups")
            if (not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups)
                    or any(not isinstance(a, dict) or not isinstance(a.get("params"), list)
                           or len(a["params"]) != len(b["params"])
                           for a, b in zip(saved_groups, current_groups))):
                raise RunError("Checkpoint optimizer parameter groups are incompatible")
        current = network.state_dict()
        if current.keys() != state["network_state_dict"].keys():
            raise RunError("Checkpoint network keys are incompatible")
        for key, value in current.items():
            other = state["network_state_dict"][key]
            for attribute in ("shape", "dtype"):
                if getattr(value, attribute, None) != getattr(other, attribute, None):
                    raise RunError("Checkpoint tensor {} mismatch: {}".format(attribute, key))
        network.load_state_dict(state["network_state_dict"])
        if self.mode == "resume":
            optimizer.load_state_dict(state["optimizer_state_dict"])
            return state["best_para"], state["step"]
        # Historical initialization is weights-only, never a resumed old run.
        return 0, 0

    def save_checkpoint(self, state, filename, saver):
        filename = Path(filename)
        try:
            relative = filename.relative_to(self.run.checkpoints).as_posix()
        except ValueError as exc:
            raise RunError("Checkpoint destination is outside this run") from exc
        destination = contained_path(self.run.checkpoints, relative, writable=True)
        temporary = destination.with_name(".{}-{}.pth".format(destination.stem, uuid4().hex))
        try:
            saver({**state, "experiment": self.checkpoint_tag}, str(temporary))
            temporary.replace(destination)
        finally:
            if temporary.is_file():
                temporary.unlink()


def check_geometry_selection(plan, run_id, mode="new", checkpoint=None, layout="collection"):
    """Check allocation/resume intent without creating files or decoding tensors."""
    validate_run_id(run_id)
    if mode not in {"new", "resume", "initialize"}:
        raise RunError("mode must be new, resume, or initialize")
    if (mode == "resume") != (checkpoint is not None):
        raise RunError("Only resume requires an explicit in-run checkpoint")
    if mode == "new" and plan.selected.checkpoint is not None:
        raise RunError("Selected historical checkpoint requires explicit initialize mode")
    if mode == "initialize" and plan.selected.checkpoint is None:
        raise RunError("Initialize mode requires inputs.checkpoint")
    store = RunStore(plan.resolver, layout)
    if mode == "resume":
        run = store.resume(run_id, checkpoint)
        try:
            record = json.loads(run.output("geometry.json").read_text())
        except (OSError, ValueError) as exc:
            raise RunError("Missing geometry run identity") from exc
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("schema_version") != 1 or record.get("run_id") != run_id
                or record.get("signature") != _signature(plan.identity)):
            raise RunError("Resume configuration, inputs, code, or dependency metadata differ")
        return run
    candidate = store._paths(run_id)
    if layout == "collection" and candidate.root.exists():
        raise RunError("New geometry run already exists")
    if layout == "mounted" and candidate.root.exists() and any(candidate.root.iterdir()):
        raise RunError("Selected mounted output directory is not empty")
    return None


def prepare_geometry(plan, run_id, mode="new", checkpoint=None, layout="collection"):
    """Allocate or explicitly select a run after side-effect-free validation."""
    selected = check_geometry_selection(plan, run_id, mode, checkpoint, layout)
    run = selected if selected is not None else RunStore(plan.resolver, layout).create(run_id)
    experiment = GeometryExperiment(plan, run, mode)
    if mode == "resume":
        return experiment
    run.output("config.original.yaml").write_bytes(plan.original_bytes)
    run.output("config.legacy.yaml").write_bytes(plan.legacy_bytes)
    _write_yaml(run.output("config.effective.yaml"), experiment.effective)
    run.output("geometry.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "mode": mode, "signature": experiment.signature,
        "compatibility": plan.identity, "limitations": KNOWN_LIMITS,
        "final_nvs": {"status": "deferred",
                      "reason": "Requires a separately versioned evaluation protocol; no final NVS is dispatched."},
        "defaults": "partially unresolved until owner initialization; not a full provenance record",
    }, sort_keys=True, indent=2) + "\n")
    return experiment
