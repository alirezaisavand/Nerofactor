"""Pinned scientific trees and cold-process namespace selection; no model imports."""
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys

from infrastructure.paths import relative_path
from infrastructure.signatures import require

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scientific_sources/manifest-v1.json'
PINS = {'geometry': '79252178e65914ad1acbc14790b1514f276bfea0',
        'material': 'e1e859779301faced46d6476691e20b546c27ae0'}
PREFIXES = ('network', 'dataset', 'train', 'utils', 'raytracing', 'colmap')


def infrastructure_identity():
    paths=list((ROOT/'infrastructure').glob('*.py'))
    paths += [ROOT/n for n in ['run_stage_experiment.py','compare_g_geometry_training.py',
              'docs/protocols/GEOMETRY_G_TWO_UPDATE_V1.md',
              'configs/experiments/geometry_g_bell_v1.yaml','configs/experiments/material_a_bear_v1.yaml']]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def baseline(stage, commit):
    require(stage in PINS and commit == PINS[stage], 'Ambiguous/foreign scientific stage or commit')
    manifest = json.loads(MANIFEST.read_text())
    item = manifest['stages'][stage]
    require(item['commit'] == commit and manifest['policy'] == 'dual-scientific-baseline-v1', 'Source manifest conflict')
    root = ROOT / relative_path(item['root'])
    require(root.resolve() == root and root.is_dir(), 'Unsafe scientific source root')
    for name, record in item['files'].items():
        path = root / relative_path(name)
        require(path.resolve() == path and path.is_file() and digest(path) == record['sha256'] and
                path.stat().st_size == record['bytes'], 'Scientific source changed: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(item['files']), 'Unmanifested scientific source file/cache')
    return root, item


class StageFinder(importlib.abc.MetaPathFinder):
    """Close namespace packages so missing G modules never fall back to A."""
    def __init__(self, root): self.root = root

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split('.')
        if parts[0] not in PREFIXES: return None
        search = [str(self.root)] if len(parts) == 1 else [str(self.root.joinpath(*parts[:-1]))]
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None: raise ModuleNotFoundError('Scientific namespace fallback forbidden: ' + fullname)
        if spec.submodule_search_locations is not None:
            spec.submodule_search_locations = [str(self.root.joinpath(*parts))]
        if spec.origin:
            require(self.root in Path(spec.origin).resolve().parents, 'Foreign scientific import')
        return spec


def activate(stage, commit):
    root, item = baseline(stage, commit)
    require(not any(isinstance(f, StageFinder) for f in sys.meta_path), 'One scientific stage per process')
    require(not any(n.split('.')[0] in PREFIXES for n in sys.modules), 'Scientific namespace already imported; use a fresh process')
    sys.meta_path.insert(0, StageFinder(root))
    sys.path.insert(0, str(root))
    return root, item


def imported_sources(root):
    result = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in PREFIXES: continue
        path = getattr(module, '__file__', None)
        if path:
            p = Path(path).resolve()
            require(root in p.parents, 'Mixed scientific import: ' + name)
            result[name] = {'path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
        else:
            require(all(Path(p).resolve() == root / name.replace('.', '/') for p in module.__path__), 'Mixed namespace path')
    return result


def admit_checkpoint_identity(stage, commit, record):
    """Metadata gate only; never load a checkpoint or optimizer."""
    baseline(stage, commit)
    require(type(record) is dict and record.get('stage') == stage and
            record.get('scientific_commit') == commit and record.get('algorithm') ==
            ('geometry-G' if stage == 'geometry' else 'anisotropic-material-A'), 'Checkpoint origin is not admitted for selected baseline')
    require(record.get('architecture_reviewed') is True, 'Strict architecture review required separately')
    return True
