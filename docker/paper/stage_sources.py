"""Export verified committed source blobs to a NEW external directory, offline.

Never executes source code, copies working trees, accesses Docker or fetches a URL.
Provenance is written separately from the narrow build context.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from validate import COMMIT, HEX, HERE, digest, inventory, read_json, require, safe_name


PAPER_PATHS = ['infrastructure', 'dataset', 'network', 'train', 'utils', 'raytracing',
               'configs', 'colmap/plyfile.py', 'colmap/read_write_model.py',
               'assets/bsdf_256_256.bin', 'run_training.py', 'run_experiment.py',
               'run_material_experiment.py', 'requirements.txt', 'AGENTS.md', 'README.md']
NVDIFFRAST_PATHS = ['nvdiffrast', 'setup.py', 'README.md', 'LICENSE.txt']
FORBIDDEN_PARTS = {'data', 'datasets', 'final_results', 'paper_runs', 'outputs',
                   '.git', '.aws', '.ssh', '__pycache__', 'build', 'dist'}
SOURCE_SUFFIXES = {'.py', '.pyi', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp',
                   '.inl', '.cmake', '.md', '.rst', '.txt', '.toml', '.cfg', '.ini',
                   '.json', '.yaml', '.yml', '.lib'}


def git(repo, *args):
    result = subprocess.run(['git', '--no-optional-locks', '-c', 'core.hooksPath=/dev/null',
                             '-c', 'core.fsmonitor=false',
                             '-c', 'protocol.allow=never', '-c', 'protocol.file.allow=always',
                             '-c', 'transfer.fsckObjects=true', '-C', str(repo), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, 'Read-only/local Git operation failed: ' + args[0])
    return result.stdout


def outside(path, roots):
    path = path.resolve()
    require(all(path != root and root not in path.parents for root in roots),
            'Staging/report destination is inside a protected or source root')
    require(not path.exists() and not path.is_symlink(), 'Destination/report already exists')
    require(path.parent.is_dir(), 'Create an owned external parent directory first')
    return path


def allowed(name, kind):
    safe_name(name)
    parts = Path(name).parts
    require(not any(p.lower() in FORBIDDEN_PARTS or p.startswith('.env') for p in parts),
            'Data, generated or private path selected: ' + name)
    require(not any(word in Path(name).name.lower() for word in ['credential', 'token']),
            'Private filename selected')
    licence = any(word in Path(name).name.lower() for word in ['license', 'copying', 'notice'])
    require(licence or Path(name).suffix.lower() in SOURCE_SUFFIXES or
            (kind == 'paper' and name == 'assets/bsdf_256_256.bin'),
            'Non-source artifact selected: ' + name)


def stage(kind, repo, commit, destination, report, protected, paths=None):
    reference = read_json(HERE / 'reference.json')
    require(kind in {'paper', 'nvdiffrast', 'raytracing'}, 'Unknown source kind')
    if kind != 'paper':
        expected = reference['extensions'][kind]
        require(commit == expected['commit'], 'Extension commit differs from the approved pin')
    require(COMMIT.fullmatch(commit or ''), 'An exact 40-character paper/source commit is required')
    repo = repo.resolve()
    roots = [p.resolve() for p in protected] + [repo, HERE.parents[1]]
    destination, report = outside(destination, roots), outside(report, roots)
    require(destination not in report.parents and report not in destination.parents and destination != report,
            'Keep the provenance report outside the exported source directory')
    require(git(repo, 'rev-parse', commit + '^{commit}').decode().strip() == commit, 'Commit object mismatch')
    bare = git(repo, 'rev-parse', '--is-bare-repository').decode().strip() == 'true'
    if not bare:
        require(not git(repo, 'status', '--porcelain', '--untracked-files=no'), 'Tracked source/index changes present')
        require(git(repo, 'rev-parse', 'HEAD').decode().strip() == commit, 'Checkout HEAD differs from selected commit')
    if kind == 'paper':
        paths = PAPER_PATHS
    elif kind == 'nvdiffrast':
        paths = NVDIFFRAST_PATHS
    require(isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths),
            'Raytracing needs an explicitly reviewed JSON source-path allowlist')
    for path in paths:
        safe_name(path)
    records = []
    for record in git(repo, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
        if not record:
            continue
        fields, name = record.split(b'\t', 1)
        mode, type_, oid = fields.decode().split()
        name = name.decode('utf-8')
        require(mode != '160000', 'Submodule source needs separate pinned staging review')
        records.append((name, mode, type_, oid))
    # Always retain root licensing/notice files, even if omitted in a supplied list.
    licences = [name for name, _, _, _ in records if '/' not in name and
                any(word in name.lower() for word in ['license', 'copying', 'notice'])]
    paths = list(paths) + licences
    require(all(any(name == p or name.startswith(p + '/') for name, _, _, _ in records)
                for p in paths), 'A selected source path is absent from the pinned tree')
    selected = [item for item in records if any(item[0] == p or item[0].startswith(p + '/') for p in paths)]
    require(selected, 'Empty source selection')
    if kind != 'paper':
        require(any('license' in p.lower() or 'copying' in p.lower() for p in licences),
                'Missing standalone extension licence; stop for review')
    contents, modes, blob_ids = {}, {}, {}
    for name, mode, type_, oid in selected:
        allowed(name, kind)
        require(mode in {'100644', '100755'} and type_ == 'blob', 'Symlink/special source entry forbidden')
        data = git(repo, 'cat-file', 'blob', oid)
        require(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid,
                'Git source blob identity mismatch')
        contents[name], modes[name], blob_ids[name] = data, mode, oid
    if kind != 'paper':
        require('setup.py' in contents and hashlib.sha256(contents['setup.py']).hexdigest() == expected['setup_py_sha256'],
                'Pinned setup.py SHA-256 mismatch')
    destination.mkdir()  # Exclusive: never merge into or overwrite another export.
    for name, data in contents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                     0o755 if modes[name] == '100755' else 0o644)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    files = inventory(destination)
    record = {'schema_version': 1, 'kind': kind, 'commit': commit,
              'git_tree': git(repo, 'rev-parse', commit + '^{tree}').decode().strip(),
              'selected_paths': paths, 'files': files, 'git_blob_ids': blob_ids,
              'git_modes': modes, 'licence_files': licences,
              'export_manifest_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
              'excluded_tracked_files': len(records) - len(selected),
              'limitations': ['export hash covers selected files, not the entire upstream tree',
                              'source identity does not establish rebuilt binary identity',
                              'no dependency installation or setup execution performed']}
    with report.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return record


@contextmanager
def bundle_repository(bundle, expected_sha):
    require(HEX.fullmatch(expected_sha or '') and digest(bundle) == expected_sha, 'Bundle SHA-256 mismatch')
    with tempfile.TemporaryDirectory(prefix='paper-source-bundle-') as directory:
        repo = Path(directory)
        git(repo, 'init', '--bare', '--quiet')
        git(repo, 'bundle', 'verify', str(bundle.resolve()))
        heads = git(repo, 'bundle', 'list-heads', str(bundle.resolve()), 'HEAD').decode().splitlines()
        require(len(heads) == 1, 'Standalone bundle must expose exactly one HEAD')
        git(repo, 'fetch', '--quiet', '--no-tags', str(bundle.resolve()), 'HEAD')
        git(repo, 'fsck', '--strict', '--no-reflogs')
        yield repo, heads[0].split()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['paper', 'nvdiffrast', 'raytracing'], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--repo', type=Path)
    source.add_argument('--bundle', type=Path)
    parser.add_argument('--bundle-sha256')
    parser.add_argument('--paper-commit')
    parser.add_argument('--paths', type=Path, help='Reviewed JSON list of raytracing source paths')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--protected-root', action='append', type=Path, required=True,
                        help='Repeat for original datasets, thesis artifacts and other protected storage')
    args = parser.parse_args()
    try:
        reference = read_json(HERE / 'reference.json')
        commit = args.paper_commit if args.kind == 'paper' else reference['extensions'][args.kind]['commit']
        paths = read_json(args.paths) if args.paths else None
        if args.bundle:
            with bundle_repository(args.bundle, args.bundle_sha256) as (repo, head):
                require(head == commit, 'Bundle HEAD differs from selected pinned commit')
                record = stage(args.kind, repo, commit, args.destination, args.report, args.protected_root, paths)
        else:
            record = stage(args.kind, args.repo, commit, args.destination, args.report, args.protected_root, paths)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Source staging rejected: {}\n'.format(exc))
    print('{} {}: {} files; export manifest {}'.format(args.kind, record['commit'],
          len(record['files']), record['export_manifest_sha256']))


if __name__ == '__main__':
    main()
