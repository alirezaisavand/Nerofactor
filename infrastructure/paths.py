"""Explicit read-only input selections and contained writable destinations."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional
from types import MappingProxyType
import re


class PathError(ValueError):
    """A path violates the declared storage contract."""


def relative_path(value: str) -> Path:
    """Accept portable relative paths without expansion or parent traversal."""
    if (not isinstance(value, str) or not value or value.startswith("~")
            or any(c in value for c in "\\:$`")
            or any(ord(c) < 32 for c in value)
            or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise PathError("Expected a nonempty relative POSIX path without traversal: {!r}".format(value))
    return Path(*value.split("/"))


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def contained_path(root: Path, value: str, writable: bool = False) -> Path:
    """Resolve a child and reject symlink escapes (all child symlinks for writes)."""
    child = relative_path(value)
    try:
        target = (root / child).resolve()
        if not _inside(target, root):
            raise PathError("Path escapes its declared root: {!r}".format(value))
        if writable:
            cursor = root
            for part in child.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise PathError("Writable paths must not use symlinks: {}".format(cursor))
        return target
    except (OSError, RuntimeError) as exc:
        raise PathError("Cannot resolve path {!r}: {}".format(value, exc)) from exc


@dataclass(frozen=True)
class InputRef:
    root: str
    path: str

    def __post_init__(self):
        if not isinstance(self.root, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.root):
            raise PathError("Input root names must be lowercase identifiers")
        relative_path(self.path)


@dataclass(frozen=True)
class InputSelection:
    dataset: str
    mesh: Optional[InputRef] = None
    checkpoint: Optional[InputRef] = None


@dataclass(frozen=True)
class ResolvedInputs:
    dataset: Path
    mesh: Optional[Path]
    checkpoint: Optional[Path]


class PathResolver:
    """Roots are runtime arguments, never inferred from CWD or legacy strings.

    Input directories must already exist. Constructing/resolving paths creates
    nothing. Additional named input roots can expose a prior run read-only.
    """

    def __init__(self, input_roots: Mapping[str, Path], output_root: Path):
        if not isinstance(input_roots, Mapping):
            raise PathError("input_roots must be a mapping of names to absolute paths")
        self.input_roots = {}
        for name, value in input_roots.items():
            InputRef(name, "validation")
            root = self._root(value)
            if not root.is_dir():
                raise PathError("Input root must be an existing directory: {}".format(root))
            self.input_roots[name] = root
        if not {"datasets", "thesis"}.issubset(self.input_roots):
            raise PathError("Declare both datasets and thesis input roots")
        self.input_roots = MappingProxyType(self.input_roots)
        self.output_root = self._root(output_root)
        if self.output_root.exists() and not self.output_root.is_dir():
            raise PathError("Output root is not a directory")
        for root in self.input_roots.values():
            if _inside(self.output_root, root) or _inside(root, self.output_root):
                raise PathError("Input and output roots must not overlap")

    @staticmethod
    def _root(value: Path) -> Path:
        try:
            path = Path(value)
        except TypeError as exc:
            raise PathError("Runtime roots must be filesystem paths") from exc
        if not path.is_absolute():
            raise PathError("Runtime roots must be absolute paths: {}".format(path))
        try:
            return path.resolve()
        except (OSError, RuntimeError) as exc:
            raise PathError("Cannot resolve root: {}".format(path)) from exc

    @classmethod
    def container(cls):
        """Use the future mount contract; fail if input mounts are missing."""
        return cls({"datasets": Path("/inputs/datasets"),
                    "thesis": Path("/inputs/thesis")}, Path("/outputs"))

    def input(self, ref: InputRef, directory: bool = False) -> Path:
        if not isinstance(ref, InputRef):
            raise PathError("Input paths require an explicit InputRef")
        if ref.root not in self.input_roots:
            raise PathError("Undeclared input root: {}".format(ref.root))
        target = contained_path(self.input_roots[ref.root], ref.path)
        valid = target.is_dir() if directory else target.is_file()
        if not valid:
            raise PathError("Required input {} does not exist: {}".format(
                "directory" if directory else "file", target))
        return target

    def resolve_inputs(self, selection: InputSelection) -> ResolvedInputs:
        if not isinstance(selection, InputSelection):
            raise PathError("Resolve an explicit InputSelection; legacy strings are not remapped automatically")
        return ResolvedInputs(
            self.input(InputRef("datasets", selection.dataset), directory=True),
            self.input(selection.mesh) if selection.mesh else None,
            self.input(selection.checkpoint) if selection.checkpoint else None,
        )

    def output(self, value: str) -> Path:
        self.check_output_root()
        return contained_path(self.output_root, value, writable=True)

    def check_output_root(self):
        """Catch root replacement/retargeting between API calls."""
        if self.output_root.is_symlink() or self.output_root.resolve() != self.output_root:
            raise PathError("Output root was replaced or redirected")
        if self.output_root.exists() and not self.output_root.is_dir():
            raise PathError("Output root is not a directory")
