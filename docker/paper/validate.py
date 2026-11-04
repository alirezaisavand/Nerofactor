"""Read-only definition/artifact checks. Never invokes Docker, pip, Git or models."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HEX = re.compile(r"[a-f0-9]{64}")
COMMIT = re.compile(r"[a-f0-9]{40}")
BASE = re.compile(r"nvidia/cuda:11\.1\.1-devel-ubuntu20\.04@sha256:[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def safe_name(value):
    require(isinstance(value, str) and value and not value.startswith('/'), 'Invalid artifact path')
    require(all(p not in ('', '.', '..', '.git', '.aws', '.ssh') for p in value.split('/')), 'Unsafe artifact path')
    require(not any(c in value for c in '\\:$`\n\r'), 'Unsafe artifact characters')
    return value


def inventory(root):
    require(root.is_dir() and not root.is_symlink(), 'Required directory missing or redirected: ' + str(root))
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink forbidden in build inputs: ' + str(path))
        if path.is_file():
            require(path.stat().st_nlink == 1, 'Hard-linked input must be explicitly materialized: ' + str(path))
            name = safe_name(path.relative_to(root).as_posix())
            require(not any(part.startswith('.env') for part in path.relative_to(root).parts), 'Private env file in context')
            result[name] = digest(path)
        else:
            require(path.is_dir(), 'Special file forbidden in build context')
    return result


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def definition():
    reference = read_json(HERE / 'reference.json')
    require(reference['schema_version'] == 1, 'Unsupported reference schema')
    docker = (HERE / 'Dockerfile').read_text()
    for forbidden in ['apt-get', 'pip install -r requirements.txt', 'git clone', '--upgrade', 'micromamba']:
        require(forbidden not in docker, 'Floating or unrelated installation in Dockerfile: ' + forbidden)
    require('ARG PAPER_BASE_IMAGE\nFROM ${PAPER_BASE_IMAGE}' in docker, 'Base must have no mutable default')
    require(all('--network=none' in line for line in docker.splitlines() if line.startswith('RUN ')), 'Build RUN needs offline network policy')
    require('USER 1000:1000' in docker and 'WORKDIR /workspace' in docker, 'Missing non-root/workspace defaults')
    declarations = {}
    for line in (HERE / 'requirements.reference.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, separator, version = line.partition('==')
            require(separator and name and version, 'Unpinned reference declaration')
            declarations[name] = version
    require(declarations == reference['verified_live_packages'], 'Reference version files disagree')
    for source in reference['extensions'].values():
        require(COMMIT.fullmatch(source['commit']) and HEX.fullmatch(source['setup_py_sha256']), 'Invalid source identity')
    require(reference['image']['base_digest'] is None, 'Review the base status when evidence changes')
    for path in HERE.glob('*.py'):
        # Running this checker on Python 3.8 or newer uses the 3.8 grammar.
        ast.parse(path.read_text(), filename=str(path), feature_version=8)
    print('PASS: definition structure, version declarations and Python 3.8 syntax')
    print('NOT BUILD-READY: base digest, complete package pins/hashes, compiler/Eigen and staged artifacts remain unresolved')
    return reference


def inputs(root, workspace, base, reference):
    require(BASE.fullmatch(base or ''), 'Supply the reviewed historical CUDA base with a real registry digest')
    resolved = read_json(root / 'resolved.json')
    require(resolved.get('schema_version') == 1 and resolved.get('reviewed') is True, 'Inputs require explicit reviewed resolution')
    require(resolved.get('base_image') == base, 'Resolved base and build argument disagree')
    require(COMMIT.fullmatch(resolved.get('workspace_commit', '')), 'Record the approved workspace commit')
    actual = inventory(root)
    actual.pop('resolved.json', None)
    expected = resolved.get('files')
    require(isinstance(expected, dict) and expected and all(HEX.fullmatch(v) for v in expected.values()), 'Missing input file hashes')
    require(all(safe_name(name) for name in expected), 'Unsafe input name')
    require(actual == expected, 'Input files differ from the reviewed complete hash manifest')
    for name in actual:
        require(name == 'python-complete.txt' or name.startswith(('apt/', 'wheels/', 'sources/nvdiffrast/', 'sources/raytracing/')),
                'Unrelated file in build input set: ' + name)
    require(inventory(workspace) == resolved.get('workspace_files'), 'Workspace snapshot/hash manifest mismatch')
    allowed_workspace = {'assets', 'configs', 'infrastructure', 'dataset', 'network', 'train', 'utils',
                         'raytracing', 'colmap', 'run_training.py', 'run_experiment.py',
                         'run_material_experiment.py', 'requirements.txt', 'AGENTS.md', 'README.md'}
    require(all(Path(name).parts[0] in allowed_workspace for name in resolved['workspace_files']),
            'Workspace contains files outside the paper source/assets allowlist')
    packages = resolved.get('python_packages', {})
    required = {normalize(k): v for k, v in reference['verified_live_packages'].items()}
    selected = {normalize(k): v for k, v in packages.items()}
    require(not {'nvdiffrast', 'raytracing'}.intersection(selected), 'Native source-built distributions must not come from the prebuilt wheel set')
    require(all(isinstance(v, str) and v and not any(x in v for x in '*<>=;\n') for v in selected.values()), 'All Python package versions must be exact')
    require(all(selected.get(k) == v for k, v in required.items()), 'A verified live package version changed')
    for name in reference['unresolved_python_packages'][:-1]:
        require(normalize(name) in selected, 'Resolve runtime/build dependency: ' + name)
    require(resolved.get('historical_extras_decision') in ('omit-unused', 'retain-declared-pins'), 'Record the torchvision/audio decision')
    for name, version in reference['historical_declarations_not_verified_live'].items():
        if normalize(name) in selected:
            require(selected[normalize(name)] == version, 'Historical extra must retain its declared pin')
        elif resolved['historical_extras_decision'] == 'retain-declared-pins':
            raise ValueError('Historical extra requested but absent: ' + name)
    lines = [line for line in (root / 'python-complete.txt').read_text().splitlines() if line and not line.startswith('#')]
    locked = {}
    for line in lines:
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([a-f0-9]{64})', line)
        require(match is not None, 'Lock requires one exact wheel and SHA-256 per line, no options/URLs')
        name, version, sha = match.groups()
        name = normalize(name)
        require(name not in locked, 'Duplicate locked package')
        require(sha in [v for k, v in actual.items() if k.startswith('wheels/') and k.endswith('.whl')], 'Locked wheel hash absent')
        locked[name] = version
    require(locked == selected, 'Python lock and resolved package set disagree')
    require(all(Path(k).suffix == '.whl' for k in actual if k.startswith('wheels/')), 'Wheelhouse must contain only wheels')
    apt = resolved.get('system_packages', {})
    require(isinstance(apt, dict) and apt and all(isinstance(v, str) and v for v in apt.values()), 'Resolve exact Debian package versions')
    require({'python3', 'python3-dev', 'python3-pip', 'build-essential', 'libeigen3-dev'}.issubset(apt), 'Required system package roles missing')
    apt_sums = {}
    for line in (root / 'apt/SHA256SUMS').read_text().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  ([A-Za-z0-9_.+%~-]+\.deb)', line)
        require(match is not None, 'Debian checksum file must contain only flat .deb names')
        sha, name = match.groups()
        require(name not in apt_sums, 'Duplicate Debian checksum entry')
        apt_sums['apt/' + name] = sha
    require(apt_sums and apt_sums == {k: v for k, v in actual.items() if k.startswith('apt/') and k.endswith('.deb')}, 'Debian artifact/hash set mismatch')
    toolchain = resolved.get('toolchain', {})
    require(all(isinstance(toolchain.get(k), str) and toolchain[k] for k in ['gcc', 'g++', 'eigen', 'cmake']), 'Recover compiler/Eigen/CMake versions')
    require(all(isinstance(toolchain.get(k), list) and toolchain[k] and all(isinstance(v, str) for v in toolchain[k])
                for k in ['cxx_flags', 'nvcc_flags']), 'Record reviewed exact-source compiler flags')
    require(toolchain.get('nvcc') == reference['runtime']['nvcc'] and toolchain.get('cxx11_abi') == 0, 'CUDA/ABI settings changed')
    for name, source in reference['extensions'].items():
        require(resolved.get('sources', {}).get(name, {}).get('commit') == source['commit'], 'Wrong source revision: ' + name)
        require(digest(root / 'sources' / name / 'setup.py') == source['setup_py_sha256'], 'Source setup hash mismatch: ' + name)
    print('PASS: complete reviewed offline inputs, source pins, hashes and workspace snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--base-image')
    args = parser.parse_args()
    try:
        reference = definition()
        if args.inputs:
            require(args.workspace is not None, '--workspace is required with --inputs')
            inputs(args.inputs, args.workspace, args.base_image, reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Invalid environment definition/inputs: {}\n'.format(exc))


if __name__ == '__main__':
    main()
