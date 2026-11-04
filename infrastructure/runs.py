"""Exclusive run allocation and explicit checkpoint selection (no training)."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Optional
from uuid import uuid4

from infrastructure.paths import PathResolver, contained_path


class RunError(ValueError):
    """An allocation/resume request is unsafe or incomplete."""


def validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise RunError("Run IDs must be 1-64 lowercase letters/digits/_/-, starting with a letter or digit")
    return value


def generate_run_id(label="experiment") -> str:
    validate_run_id(label)
    if len(label) > 31:
        raise RunError("Generated-ID labels must be at most 31 characters")
    return "{}-{}".format(label, uuid4().hex)


_DIRECTORIES = ("checkpoints", "logs", "cache", "meshes", "materials", "renders", "evaluations")
_MARKER = ".experiment.json"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    resume_checkpoint: Optional[Path] = None

    def output(self, relative: str) -> Path:
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise RunError("Run directory was replaced or redirected")
        return contained_path(self.root, relative, writable=True)

    @property
    def checkpoints(self):
        return self.output("checkpoints")

    @property
    def logs(self):
        return self.output("logs")

    @property
    def cache(self):
        return self.output("cache")


class RunStore:
    """Use collection/<id> on hosts, or an exclusively initialized mounted root.

    The marker is allocation identity only, not an experiment provenance record.
    These guards assume no hostile concurrent replacement of filesystem parents.
    """

    def __init__(self, resolver: PathResolver, layout="collection"):
        if layout not in ("collection", "mounted"):
            raise RunError("layout must be collection or mounted")
        self.resolver = resolver
        self.layout = layout

    def _paths(self, run_id):
        validate_run_id(run_id)
        self.resolver.check_output_root()
        root = self.resolver.output(run_id) if self.layout == "collection" else self.resolver.output_root
        return RunPaths(run_id, root)

    def create(self, run_id=None) -> RunPaths:
        paths = self._paths(generate_run_id() if run_id is None else run_id)
        try:
            if self.layout == "collection":
                self.resolver.output_root.mkdir(parents=True, exist_ok=True)
                paths.root.mkdir(exist_ok=False)
            else:
                # A mount target normally exists already. Only empty roots may
                # be initialized; an old run or arbitrary existing files fail.
                paths.root.mkdir(parents=True, exist_ok=True)
                if any(paths.root.iterdir()):
                    raise RunError("Mounted output root is not empty; explicitly resume its run")
            marker = paths.output(_MARKER)
            with marker.open("x") as stream:
                for name in _DIRECTORIES:
                    paths.output(name).mkdir(exist_ok=False)
                json.dump({"schema_version": 1, "run_id": paths.run_id}, stream)
                stream.write("\n")
        except OSError as exc:
            # Preserve partial allocations for inspection; never clean them up
            # or reuse a directory after a failed allocation.
            raise RunError("Run allocation failed without overwriting existing data: {}".format(exc)) from exc
        return paths

    def resume(self, run_id, checkpoint=None) -> RunPaths:
        """Select an existing in-run checkpoint; never infer latest/best."""
        paths = self._paths(run_id)
        if checkpoint is None:
            raise RunError("Resume requires an explicit checkpoint relative to checkpoints/")
        try:
            with paths.output(_MARKER).open() as stream:
                identity = json.load(stream)
        except (OSError, ValueError) as exc:
            raise RunError("Missing or invalid run allocation identity") from exc
        if (not isinstance(identity, dict) or type(identity.get("schema_version")) is not int
                or identity != {"schema_version": 1, "run_id": run_id}):
            raise RunError("Run allocation identity does not match the requested ID/version")
        for name in _DIRECTORIES:
            if not paths.output(name).is_dir():
                raise RunError("Incomplete run allocation: missing {}".format(name))
        selected = contained_path(paths.checkpoints, checkpoint, writable=True)
        if not selected.is_file():
            raise RunError("Selected resume checkpoint does not exist: {}".format(selected))
        return RunPaths(run_id, paths.root, selected)
