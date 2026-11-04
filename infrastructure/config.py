"""Versioned experiment envelopes around unchanged legacy YAML mappings.

No scientific modules are imported, defaults invented, or paths rewritten.
Strict validation covers the envelope and the explicitly checked legacy types;
it is not a proof that every scientific payload field is consumed by training.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Optional, Tuple

import yaml

from infrastructure.paths import InputRef, InputSelection, PathError, relative_path


@dataclass(frozen=True)
class Diagnostic:
    code: str
    field: str
    message: str
    level: str = "warning"


class ConfigError(ValueError):
    def __init__(self, message, diagnostics=()):
        self.diagnostics = tuple(diagnostics)
        super().__init__(message)


def legacy_merge(defaults: Mapping, overrides: Mapping) -> dict:
    """Exactly {**defaults, **cfg}, including shared nested mapping references."""
    return {**defaults, **overrides}


@dataclass(frozen=True)
class Configuration:
    source: Path
    legacy_source: Path
    schema_version: int
    stage: Optional[str]
    original: dict
    legacy: dict
    inputs: Optional[InputSelection]
    diagnostics: Tuple[Diagnostic, ...]

    def with_defaults(self, defaults: Mapping) -> dict:
        """Merge only explicitly supplied owner defaults; never deep-merge."""
        return legacy_merge(defaults, self.legacy)


def _config_path(value):
    try:
        return Path(value).resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ConfigError("Invalid configuration file path: {}".format(exc)) from exc


def _read_yaml(path):
    diagnostics = []

    class LegacyLoader(yaml.FullLoader):
        def construct_mapping(self, node, deep=False):
            # Check explicit duplicates before FullLoader flattens YAML merges.
            keys = set()
            for key, _ in node.value:
                if key.tag == "tag:yaml.org,2002:merge":
                    continue
                value = self.construct_object(key, deep=deep)
                try:
                    if value in keys:
                        diagnostics.append(Diagnostic(
                            "duplicate-key", str(value),
                            "{}:{}: duplicate key; legacy last value is retained".format(
                                path, key.start_mark.line + 1), "error"))
                    keys.add(value)
                except TypeError:
                    pass  # FullLoader reports unsupported unhashable keys.
            return super().construct_mapping(node, deep=deep)

    try:
        with path.open() as stream:
            value = yaml.load(stream, Loader=LegacyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError("Cannot load {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ConfigError("Configuration must be a YAML mapping: {}".format(path))
    return value, diagnostics


_BOOL_FIELDS = {
    "apply_acc_loss", "apply_occ_loss", "apply_opacity_loss", "anisotropy",
    "is_nerf", "fixed_camera", "clip_sample_variance", "multi_gpus", "reg_mat",
    "reg_diffuse_light", "reg_energy_loss", "reg_spec_loss", "reg_change",
    "reg_min_max", "random_azimuth", "human_lights", "human_light", "is_real",
    "sphere_direction", "geometry_init", "use_refscores", "val_geometry",
}
_POSITIVE_INTS = {"total_step", "val_interval", "save_interval", "train_log_step",
                  "novel_view_interval", "diffuse_sample_num", "specular_sample_num"}
_SHADER_ONLY = {"anisotropy", "max_n_exp", "max_alpha_exp", "reg_energy_loss",
                "reg_energy_loss_lambda", "reg_spec_loss", "reg_spec_loss_lambda"}
_STRING_FIELDS = {"name", "network", "database_name", "key_metric_name", "train_dataset_type",
                  "optimizer_type", "lr_type", "mesh"}


def _stage(values):
    network = values.get("network")
    return {"shape": "geometry", "material": "material"}.get(network) if isinstance(network, str) else None


def _legacy_diagnostics(values, stage):
    result = [Diagnostic(
        "legacy-payload", "legacy",
        "Scientific fields are preserved; consumption and full numerical validity "
        "are not certified by this infrastructure schema.", "info")]

    def visit(mapping, prefix="", seen=None):
        seen = set() if seen is None else seen
        if id(mapping) in seen:
            raise ConfigError("Recursive YAML mappings are not supported")
        seen = seen | {id(mapping)}
        for key, value in mapping.items():
            field = "{}.{}".format(prefix, key) if prefix else str(key)
            problem = None
            if not isinstance(key, str):
                problem = "Configuration keys must be strings"
            elif key in _STRING_FIELDS and (not isinstance(value, str) or not value):
                problem = "Expected a nonempty string"
            elif key in _BOOL_FIELDS and type(value) is not bool:
                problem = "Expected a Boolean; value is preserved without coercion"
            elif key in _POSITIVE_INTS and (type(value) is not int or value <= 0):
                problem = "Expected a positive integer"
            elif key in {"random_seed", "worker_num", "end_warm"} and (
                    type(value) is not int or value < 0):
                problem = "Expected a nonnegative integer"
            elif key == "end_iter" and (type(value) is not int or value <= 0):
                problem = "Expected a positive integer"
            elif key == "lr" and (isinstance(value, bool) or not isinstance(value, Real)
                                   or not math.isfinite(value) or value <= 0):
                problem = "Expected a finite positive learning rate"
            elif isinstance(key, str) and (key.endswith("_lambda") or key.endswith("_weight")) and (
                    isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)):
                problem = "Expected a finite numeric coefficient; no value is coerced"
            elif key in {"loss", "val_metric"} and (
                    not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
                problem = "Expected a list of strings"
            elif key in {"lr_cfg", "shader_cfg", "shader_config", "train_dataset_cfg"} and not isinstance(value, dict):
                problem = "Expected a mapping"
            elif key == "dataset_dir" and (not isinstance(value, str) or not value
                                           or value == "your_dataset_dir"):
                problem = "Dataset path is empty, invalid, or a legacy placeholder"
            if problem:
                result.append(Diagnostic("invalid-legacy-value", field, problem, "error"))
            if isinstance(value, dict):
                visit(value, field, seen)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        visit(item, "{}[{}]".format(field, i), seen)

    visit(values)
    for key in values:
        if isinstance(key, str) and key not in (
                _BOOL_FIELDS | _POSITIVE_INTS | _SHADER_ONLY | {
                    "name", "network", "database_name", "dataset_dir", "mesh", "loss",
                    "val_metric", "key_metric_name", "train_dataset_type", "train_dataset_cfg",
                    "val_set_list", "optimizer_type", "lr_type", "lr_cfg", "random_seed",
                    "worker_num", "shader_cfg", "shader_config"}):
            result.append(Diagnostic("unvalidated-field", key,
                                     "Retained as legacy payload; this schema does not certify its type or consumption.", "info"))
    if values.get("network") not in ("shape", "material"):
        result.append(Diagnostic("invalid-network", "network", "Expected the existing shape or material renderer.", "error"))
    if values.get("lr_type") not in (None, "warm_up_cos"):
        result.append(Diagnostic("unsupported-scheduler", "lr_type", "Only warm_up_cos is registered in the current baseline.", "error"))
    if stage == "material":
        for key in sorted(_SHADER_ONLY & values.keys()):
            result.append(Diagnostic(
                "not-forwarded", key,
                "Current material _init_shader forwards shader_cfg, not this "
                "top-level field. It remains in its original scope."))
    if "apply_acc_loss" in values:
        result.append(Diagnostic("unconsumed-field", "apply_acc_loss",
                                 "No active consumer identified in the baseline renderer; retained."))
    if "lr_cfg" in values and isinstance(values["lr_cfg"], dict) and "end_iter" in values["lr_cfg"]:
        result.append(Diagnostic("legacy-horizon", "lr_cfg.end_iter",
                                 "The current scheduler still uses its hardcoded 100000-step clamp."))
    if values.get("optimizer_type") not in (None, "adam"):
        result.append(Diagnostic("unsupported-optimizer", "optimizer_type",
                                 "The current factory supplies Adam-specific arguments; no repair is applied.", "error"))
    return result


def _keys(mapping, allowed, field, diagnostics):
    for key in mapping:
        if key not in allowed:
            diagnostics.append(Diagnostic("unknown-field", "{}.{}".format(field, key),
                                          "Unsupported envelope field; retained in original document only.", "error"))


def _input_ref(value, field, diagnostics):
    if not isinstance(value, dict) or not {"root", "path"}.issubset(value):
        raise ConfigError("{} requires a mapping with root and path".format(field))
    _keys(value, {"root", "path"}, field, diagnostics)
    try:
        return InputRef(value["root"], value["path"])
    except PathError as exc:
        raise ConfigError("{}: {}".format(field, exc)) from exc


def load_config(path, overrides=None, strict=None) -> Configuration:
    """Load legacy YAML or schema 1; loading never allocates runs or rewrites paths.

    Schema 1 is strict by default; unversioned YAML defaults to compatibility
    diagnostics. Both file and API overrides use legacy shallow replacement.
    """
    source = _config_path(path)
    original, diagnostics = _read_yaml(source)
    version = original.get("schema_version", 0)
    if type(version) is not int or version not in (0, 1) or (
            version == 0 and "schema_version" in original):
        raise ConfigError("Only unversioned legacy YAML and schema_version: 1 are supported")
    if strict is not None and type(strict) is not bool:
        raise ConfigError("strict must be True, False, or None")
    strict = version == 1 if strict is None else strict
    selection = None
    legacy_source = source
    if version == 1:
        _keys(original, {"schema_version", "stage", "legacy_config", "inputs", "overrides"}, "config", diagnostics)
        stage = original.get("stage")
        if stage not in ("geometry", "material"):
            raise ConfigError("stage must be geometry or material")
        reference = original.get("legacy_config")
        if (not isinstance(reference, str) or not reference or Path(reference).is_absolute()
                or reference.startswith("~") or any(c in reference for c in "\\:$`")
                or any(ord(c) < 32 for c in reference)):
            raise ConfigError("legacy_config must be a relative file path anchored at the envelope")
        legacy_source = _config_path(source.parent / reference)
        legacy, inherited = _read_yaml(legacy_source)
        diagnostics.extend(inherited)
        if "schema_version" in legacy:
            raise ConfigError("legacy_config must reference an unversioned YAML, not another envelope")
        file_overrides = original.get("overrides", {})
        if not isinstance(file_overrides, dict):
            raise ConfigError("overrides must be a mapping")
        legacy = legacy_merge(legacy, file_overrides)
        inputs = original.get("inputs")
        if not isinstance(inputs, dict) or "dataset" not in inputs:
            raise ConfigError("inputs requires an explicit dataset path relative to datasets root")
        _keys(inputs, {"dataset", "mesh", "checkpoint"}, "inputs", diagnostics)
        try:
            relative_path(inputs["dataset"])
        except PathError as exc:
            raise ConfigError("inputs.dataset: {}".format(exc)) from exc
        mesh = _input_ref(inputs["mesh"], "inputs.mesh", diagnostics) if "mesh" in inputs else None
        checkpoint = _input_ref(inputs["checkpoint"], "inputs.checkpoint", diagnostics) if "checkpoint" in inputs else None
        if stage == "material" and mesh is None:
            raise ConfigError("Material envelopes require an explicit inputs.mesh")
        if stage == "geometry" and mesh is not None:
            raise ConfigError("Geometry envelopes do not consume a mesh input")
        selection = InputSelection(inputs["dataset"], mesh, checkpoint)
    else:
        legacy = deepcopy(original)
        stage = _stage(legacy)
        diagnostics.append(Diagnostic("unresolved-paths", "legacy",
                                      "Original path strings are retained; provide a schema 1 envelope for explicit input selection."))
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("API overrides must be a mapping")
        legacy = legacy_merge(legacy, deepcopy(dict(overrides)))
    if version == 1 and legacy.get("network") != {"geometry": "shape", "material": "material"}[stage]:
        raise ConfigError("Envelope stage and effective legacy network disagree")
    if version == 0:
        stage = _stage(legacy)
    diagnostics.extend(_legacy_diagnostics(legacy, stage))
    if strict and any(item.level == "error" for item in diagnostics):
        raise ConfigError("Strict validation failed: " + "; ".join(
            "{}: {}".format(d.field, d.message) for d in diagnostics if d.level == "error"), diagnostics)
    return Configuration(source, legacy_source, version, stage, deepcopy(original),
                         deepcopy(legacy), selection, tuple(diagnostics))
