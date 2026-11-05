"""Read-only, filtered runtime inventory: stream into the reference system Python.

No scientific imports, pip commands, environment dumps, network or native kernels.
Installed metadata establishes candidate pins, not wheel hashes or a complete lock.
"""

import argparse
import importlib.metadata as md
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig


RUNTIME_ROOTS = ['torch', 'numpy', 'open3d', 'PyMCubes', 'scipy', 'opencv-python',
                 'scikit-image', 'imageio', 'tensorboardX', 'transforms3d',
                 'PyYAML', 'h5py', 'plyfile', 'tqdm', 'matplotlib', 'Pillow',
                 'nvdiffrast', 'raytracing']
BUILD_ROOTS = ['pip', 'setuptools', 'wheel', 'ninja', 'packaging', 'pybind11']
SYSTEM_ROLES = ['python3', 'python3.8', 'python3-dev', 'python3.8-dev', 'python3-pip',
                'build-essential', 'gcc', 'gcc-9', 'g++', 'g++-9', 'libc6', 'libc6-dev',
                'libstdc++6', 'libgcc-s1', 'libgomp1', 'cmake', 'pkg-config',
                'ninja-build', 'libeigen3-dev', 'libgl1', 'libegl1', 'libgl1-mesa-dev',
                'libegl1-mesa-dev', 'libglib2.0-0', 'libsm6', 'libxext6', 'libxrender1']


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def closure(roots, distribution, requirement, environment):
    packages, issues, requested = {}, [], {}
    queue = [(name, set()) for name in roots]
    visited = set()
    while queue:
        name, extras = queue.pop(0)
        name = normalize(name)
        extras = requested.setdefault(name, set()) | extras
        requested[name] = extras
        key = (name, tuple(sorted(extras)))
        if key in visited:
            continue
        visited.add(key)
        try:
            dist = distribution(name)
        except md.PackageNotFoundError:
            issues.append({'package': name, 'issue': 'missing distribution metadata'})
            continue
        record = {'version': dist.version, 'requires_python': dist.metadata.get('Requires-Python'),
                  'requested_extras': sorted(extras), 'active_requirements': []}
        packages[name] = record
        for text in dist.requires or []:
            try:
                dependency = requirement(text)
                if dependency.marker and not any(dependency.marker.evaluate(dict(environment, extra=extra))
                                                 for extra in extras | {''}):
                    continue
                target = normalize(dependency.name)
                edge = {'name': target, 'specifier': str(dependency.specifier),
                        'extras': sorted(dependency.extras)}
                record['active_requirements'].append(edge)
                if dependency.url:
                    issues.append({'package': name, 'dependency': target,
                                   'issue': 'direct URL omitted; recover/review artifact separately'})
                try:
                    installed = distribution(target).version
                    if dependency.specifier and not dependency.specifier.contains(installed, prereleases=True):
                        issues.append({'package': name, 'dependency': target, 'issue': 'installed version violates metadata'})
                except md.PackageNotFoundError:
                    pass
                queue.append((target, set(dependency.extras)))
            except Exception:
                issues.append({'package': name, 'issue': 'unparsed requirement; raw text omitted'})
    return packages, issues


def command(argv):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return {'status': result.returncode, 'output': result.stdout.strip()}
    except OSError:
        return {'status': 'unavailable', 'output': ''}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-historical-extras', action='store_true')
    args = parser.parse_args()
    roots = RUNTIME_ROOTS + BUILD_ROOTS
    if args.include_historical_extras:
        roots += ['torchvision', 'torchaudio']
    # Reuse an already installed parser; never bootstrap packaging/pip.
    try:
        from packaging.requirements import Requirement
        from packaging.markers import default_environment
        parser_source = 'packaging'
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.markers import default_environment
            parser_source = 'pip vendored packaging'
        except ImportError:
            Requirement = None
            parser_source = 'unavailable; closure not resolved'
    if Requirement:
        packages, issues = closure(roots, md.distribution, Requirement, default_environment())
    else:
        packages, issues = {}, [{'issue': 'No installed metadata parser; do not install one during recovery'}]
        for name in roots:
            try:
                packages[normalize(name)] = {'version': md.version(name), 'closure_unresolved': True}
            except md.PackageNotFoundError:
                issues.append({'package': name, 'issue': 'missing distribution metadata'})
    system = {}
    for name in SYSTEM_ROLES:
        system[name] = command(['dpkg-query', '-W',
                               '-f=${Package}=${Version}\nPre-Depends: ${Pre-Depends}\nDepends: ${Depends}\nProvides: ${Provides}\n', name])
    eigen = Path('/usr/include/eigen3/Eigen/src/Core/util/Macros.h')
    macros = eigen.read_text() if eigen.is_file() else ''
    versions = [re.search(r'^#define EIGEN_{}_VERSION\s+(\d+)'.format(p), macros, re.M)
                for p in ['WORLD', 'MAJOR', 'MINOR']]
    tools = {name: command(argv) for name, argv in {
        'gcc': ['gcc', '-dumpfullversion'], 'g++': ['g++', '-dumpfullversion'],
        'nvcc': ['nvcc', '--version'], 'cmake': ['cmake', '--version'],
        'ninja': ['ninja', '--version'], 'glibc': ['getconf', 'GNU_LIBC_VERSION']}.items()}
    data = {'schema_version': 1, 'evidence': 'installed metadata only; not a reviewed build resolution',
            'python': platform.python_version(), 'executable': sys.executable,
            'machine': platform.machine(), 'soabi': sysconfig.get_config_var('SOABI'),
            'metadata_parser': parser_source, 'packages': packages, 'issues': issues,
            'system_roles': system, 'tools': tools,
            'eigen': '.'.join(m.group(1) for m in versions) if all(versions) else None,
            'not_recovered': ['historical base digest', 'artifact hashes', 'Debian dependency closure',
                              'actual extension compiler/ABI flags', 'runtime native compatibility']}
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
