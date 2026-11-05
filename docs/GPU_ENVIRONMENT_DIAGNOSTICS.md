# Read-only environment recovery commands: 1D.3A

1D.3B-2A follow-up: use the filtered collector and base/source recovery sequence in
[docker/paper/PREREQUISITES.md](../docker/paper/PREREQUISITES.md#safe-read-only-recovery-in-the-regular-terminal).
It adds marker-aware installed dependency closure, missing build-package versions,
Eigen/compiler and Debian role dependency metadata. The container-only information
remains unrecovered in Codex because a read-only inspect still receives socket
permission denied. No access workaround or package installation is authorized.

Follow-up: the recovered reference uses the container's **system `python3`**
(3.8.10), not a Micromamba environment or another `python` executable. Additional
recovery must use that same interpreter. See
[the definition and remaining pins](../docker/paper/README.md). No diagnostics or
container execution is performed in definition-only 1D.3B-1.

Run these in the **regular VS Code Remote SSH terminal on marineris**, where
Docker already works, not through a permissions workaround in Codex. They inspect
the existing image/container and run short diagnostic processes inside the already
running container. They do not build/install, start/stop containers, change
permissions, run training or invoke rendering/JIT builds. If a command is denied,
retain its error; do not add `sudo`, privileged flags or socket changes.

The short collection sequence is groups 1–3 below. Group 4 is optional core-Torch
introspection; source/linkage follow-up is only for identified extension paths.
Use the actual training interpreter in group 2/4 if it is not the container's
`python3`. No shell activation/install command is proposed.

Do not send unfiltered `docker inspect`, image history, `env`/`printenv`,
`pip config`, registry configuration or credential-bearing VCS/index URLs.
The commands deliberately select version/device/mount metadata and known package
fields. Review output before sharing; unknown/private labels, environment
variables, entrypoint commands and mount contents are not requested.

## 1. Engine, image identity, live-container identity and GPU access

```sh
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}'
docker info --format 'default_runtime={{.DefaultRuntime}} runtimes={{range $name, $runtime := .Runtimes}}{{$name}} {{end}}'
docker image inspect nerofactor --format 'id={{.Id}} digests={{json .RepoDigests}} os={{.Os}} arch={{.Architecture}} created={{.Created}}'
docker container inspect intelligent_keldysh --format 'id={{.Id}} image_id={{.Image}} state={{.State.Status}} user={{.Config.User}} workdir={{.Config.WorkingDir}} runtime={{.HostConfig.Runtime}} {{range .HostConfig.DeviceRequests}}gpu_driver={{.Driver}} count={{.Count}} ids={{json .DeviceIDs}} capabilities={{json .Capabilities}} {{end}}'
docker container inspect intelligent_keldysh --format '{{range .Mounts}}{{printf "type=%s destination=%s writable=%t\n" .Type .Destination .RW}}{{end}}'
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
nvidia-container-cli --version
docker exec intelligent_keldysh nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
```

If `compute_cap` is unsupported, repeat just the GPU query without that field and
leave compute capability unresolved. Compare container `.Image` against the tag's
image ID; a changed tag may name another image. Mount destinations/RW flags do not
reveal mount contents or prove the source host directories: verify those privately
against the workspace contract. A live Python installation may be mounted or
changed after image creation. `RepoDigests: []` is possible for a locally built
image; preserve its full image ID and mark registry/base provenance unknown.

## 2. Python/package/source metadata, without importing scientific modules

This prints an allowlist of installed distribution versions, sanitized VCS commit
metadata, static Torch build fields/ABI declaration, and module locations. It
hashes only the selected `_raytracing` shared object, if present. It does not
import `raytracing`, `_raytracing`, nvdiffrast, Torch, Open3D or the research code.
`-B` disables bytecode output, and `/` avoids importing the project accidentally.

```sh
docker exec -i --workdir / intelligent_keldysh python3 -B - <<'PY'
import ast
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig

print('python', platform.python_version(), 'executable', sys.executable)
print('platform', platform.system(), platform.machine(), 'libc', platform.libc_ver())
print('SOABI', sysconfig.get_config_var('SOABI'))
names = ['torch', 'torchvision', 'torchaudio', 'nvdiffrast', 'raytracing',
         'open3d', 'numpy', 'scipy', 'scikit-image', 'imageio', 'Pillow',
         'opencv-python', 'opencv-python-headless', 'opencv-contrib-python',
         'opencv-contrib-python-headless', 'PyMCubes', 'plyfile', 'h5py',
         'transforms3d', 'tensorboardX', 'tensorboard', 'protobuf', 'matplotlib',
         'PyYAML', 'tqdm', 'trimesh', 'xatlas', 'pypng', 'packaging',
         'setuptools', 'pip', 'wheel', 'ninja']
for name in names:
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        print(name, 'NO DISTRIBUTION METADATA')
        continue
    print(name, dist.version)
    direct = dist.read_text('direct_url.json')
    if direct:
        try:
            record = json.loads(direct)
            vcs = record.get('vcs_info', {})
            # Deliberately omit URL, requested branch and local checkout path.
            print('  vcs', {k: vcs[k] for k in ['vcs', 'commit_id'] if k in vcs},
                  'editable', bool(record.get('dir_info', {}).get('editable')))
        except (ValueError, TypeError):
            print('  direct_url metadata could not be parsed; not printed')
for dist in md.distributions():
    name = dist.metadata.get('Name', '')
    if name.lower().startswith('nvidia-'):
        print('cuda-library-distribution', name, dist.version)
for name in ['torch', '_raytracing', 'raytracing', 'nvdiffrast', 'cv2', 'open3d']:
    spec = importlib.util.find_spec(name)
    print('module', name, None if spec is None else spec.origin)
    if name == '_raytracing' and spec and spec.origin and Path(spec.origin).is_file():
        digest = hashlib.sha256()
        with Path(spec.origin).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        print('_raytracing-sha256', digest.hexdigest())
try:
    torch_dist = md.distribution('torch')
except md.PackageNotFoundError:
    torch_dist = None
if torch_dist:
    version_file = Path(torch_dist.locate_file('torch/version.py'))
    if version_file.is_file():
        for node in ast.parse(version_file.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {'__version__', 'cuda', 'hip', 'debug', 'git_version'}:
                    try:
                        print('torch.version-file', target.id, ast.literal_eval(node.value))
                    except ValueError:
                        pass
    cmake = Path(torch_dist.locate_file('torch/share/cmake/Torch/TorchConfig.cmake'))
    if cmake.is_file():
        for line in cmake.read_text().splitlines():
            if '_GLIBCXX_USE_CXX11_ABI=' in line:
                print('torch-ABI-declaration', line.strip())
PY
```

Absence of distribution metadata does not prove absence of an editable/manual
module. Module location and version-file contents do not prove native import or
GPU success. A VCS commit without dirty-source/build evidence is an incomplete pin.
This allowlist is the first pass, not a transitive package lock; recover sanitized
transitive versions and wheel sources/hashes before constructing 1D.3B locks.

## 3. Existing OS/toolchain/native package metadata

```sh
docker exec --workdir / intelligent_keldysh sh -c '
cat /etc/os-release
for tool in nvcc gcc g++ cmake ninja; do
    if command -v "$tool" >/dev/null 2>&1; then
        "$tool" --version
    fi
done
getconf GNU_LIBC_VERSION
'
docker exec intelligent_keldysh dpkg-query -W gcc g++ libc6 libstdc++6 libeigen3-dev libgl1 libegl1 libgomp1 ninja-build build-essential
```

Missing commands/packages or a non-Debian image may make portions fail; record
that instead of installing anything. Compiler metadata does not recover the
compiler/flags used for an already-built extension. Recover the actual CUDA path
and compatible compiler from the existing build records; do not export a new
`CUDA_HOME` or change the current compiler selection during inventory.

## 4. Optional core-Torch build/ABI query

This loads the already-installed core Torch library for introspection only. It
does not import rendering extensions, allocate CUDA tensors, build plugins or
run a model. If it fails, retain the error; package repair is not part of 1D.3A.

```sh
docker exec --workdir / intelligent_keldysh python3 -B -c 'import torch; print("torch", torch.__version__); print("compiled_cuda", torch.version.cuda); print("cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI); print(torch.__config__.show())'
```

Do not use nvdiffrast/plugin import or rendering calls as inventory commands:
some revisions may build on demand. Native viability/kernel checks require the
separate smoke-test approval described in the
[inventory and 1D.3B proposal](GPU_ENVIRONMENT_INVENTORY.md).

## Follow-up only after extension paths are identified

For each known source checkout, read its HEAD and dirty-status/stat privately;
do not print credential-bearing remote URLs or execute `setup.py` to query it.
For the exact installed `_raytracing` binary, `readelf` is a metadata reader:

```sh
docker exec intelligent_keldysh git -C /KNOWN/EXTENSION/CHECKOUT rev-parse HEAD
docker exec intelligent_keldysh git -C /KNOWN/EXTENSION/CHECKOUT status --short
docker exec intelligent_keldysh git -C /KNOWN/EXTENSION/CHECKOUT diff --stat
docker exec intelligent_keldysh readelf -d /KNOWN/PATH/TO/_raytracing.so
docker exec intelligent_keldysh readelf --version-info /KNOWN/PATH/TO/_raytracing.so
```

Replace placeholders with the identified paths; these commands are not useful
until then. Preserve source/patch/archive/wheel hashes in a later reviewed build
record. Do not search credential directories or dump build histories to find
missing provenance. Binary dependency/version requirements do not by themselves
identify a source commit or establish runtime ABI compatibility.

Return groups 1–3 (and group 4 if appropriate), retaining errors and the exact
interpreter used. The next review should resolve the image/container difference,
package/CUDA/compiler/ABI tuple, source revisions and remaining pin gaps before
authorizing any environment construction.
