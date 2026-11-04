"""Future image-build provenance; do not execute in definition-only milestone."""

import argparse
import importlib.metadata as md
import json
from pathlib import Path
import platform
import re
import subprocess
import sysconfig

from validate import HERE, digest, read_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-core', action='store_true')
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    reference = read_json(HERE / 'reference.json')
    require(platform.python_version() == reference['runtime']['python'], 'System Python version mismatch')
    # Core library introspection only, in a FUTURE approved build, no CUDA allocation.
    import torch
    require(torch.__version__ == reference['verified_live_packages']['torch'], 'Torch build mismatch')
    require(torch.version.cuda == '11.1' and int(torch._C._GLIBCXX_USE_CXX11_ABI) == 0, 'Torch CUDA/ABI mismatch')
    nvcc = subprocess.check_output(['nvcc', '--version'], text=True)
    require('V11.1.105' in nvcc, 'NVCC mismatch')
    resolved = read_json(Path('/opt/paper-inputs/resolved.json'))
    for name, version in resolved['python_packages'].items():
        require(md.version(name) == version, 'Installed Python package mismatch: ' + name)
    eigen_macros = Path('/usr/include/eigen3/Eigen/src/Core/util/Macros.h').read_text()
    eigen = '.'.join(re.search(r'^#define EIGEN_{}_VERSION\s+(\d+)'.format(part), eigen_macros, re.M).group(1)
                     for part in ['WORLD', 'MAJOR', 'MINOR'])
    require(eigen == resolved['toolchain']['eigen'], 'Eigen header version mismatch')
    if args.check_core:
        return
    require(args.write is not None, 'Select --check-core or --write')
    packages = {name: md.version(name) for name in {dist.metadata['Name'] for dist in md.distributions()}}
    for name, source in reference['extensions'].items():
        require(md.version(name) == source['version'], 'Installed extension distribution mismatch')
    binaries = {}
    dist = md.distribution('raytracing')
    for name in dist.files or []:
        path = Path(dist.locate_file(name))
        if path.name.startswith('_raytracing') and path.suffix == '.so':
            binaries[path.name] = digest(path)
    require(binaries, 'No installed raytracing binary found in package records')
    actual_system = subprocess.check_output(['dpkg-query', '-W', '-f=${Package}=${Version}\n'], text=True)
    installed = dict(line.split('=', 1) for line in actual_system.splitlines() if '=' in line)
    require(all(installed.get(k) == v for k, v in resolved['system_packages'].items()), 'Installed Debian versions differ from resolution')
    for key, command in [('gcc', ['gcc', '-dumpfullversion']), ('g++', ['g++', '-dumpfullversion'])]:
        require(subprocess.check_output(command, text=True).strip() == resolved['toolchain'][key], 'Compiler version mismatch')
    cmake = subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0].split()[-1]
    require(cmake == resolved['toolchain']['cmake'], 'CMake version mismatch')
    data = {'schema_version': 1, 'reference': reference, 'resolved_inputs': resolved,
            'python': platform.python_version(), 'soabi': sysconfig.get_config_var('SOABI'),
            'architecture': platform.machine(), 'libc': platform.libc_ver(),
            'torch': torch.__version__, 'compiled_cuda': torch.version.cuda, 'cxx11_abi': 0,
            'nvcc': nvcc, 'eigen': eigen, 'packages': packages, 'dpkg': installed,
            'torch_build_config': torch.__config__.show(),
            'ninja_binary_version': subprocess.check_output(['ninja', '--version'], text=True).strip(),
            'built_wheels': {p.name: digest(p) for p in Path('/opt/paper-built-wheels').glob('*.whl')},
            'raytracing_binaries': binaries,
            'definition_files': {p.name: digest(p) for p in HERE.iterdir() if p.is_file()},
            'not_claimed': ['runtime image registry digest', 'binary identity with historical extension',
                            'GPU/native acceptance', 'runtime nvdiffrast plugin identity', 'scientific parity']}
    args.write.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
