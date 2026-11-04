# Existing GPU environment inventory: 1D.3A

Follow-up 1D.3B-1: the researcher has recovered the historical system Python
3.8.10 / Torch 1.8.1+cu111 / CUDA 11.1 / ABI 0 reference and exact native source
commits. See [the Docker definition](../docker/paper/README.md) and its classified
reference manifest. The original inventory below remains historical evidence of
what Codex could observe in 1D.3A, not the current reconstruction specification.

Inspection date: 2026-11-04 (America/Denver). Repository:
`/home/isavand/paper/Nerofactor-paper`, branch `paper/infrastructure`, initially
clean at `45f895e64c428092e1b62be67d582b171af73116`.
Only documentation is changed in this milestone. No package installation, image
build, container lifecycle change, native extension import/build, GPU computation,
training, dataset modification, permission change or commit is performed.

## Evidence boundaries and access results

The user identifies host `marineris`, image `nerofactor`, and running container
`intelligent_keldysh`. Their existence, contents and GPU configuration could not
be independently verified through Codex: filtered image/container inspection each
returned Docker socket permission denied at `/var/run/docker.sock`. The read-only
`nvidia-smi` query reported that it could not communicate with the NVIDIA driver.
These are **Codex-context access failures**, not proof that the host or the
regular VS Code terminal's working environment is broken. No escalation, socket
permission/group change, alternate endpoint or security workaround was attempted.

Three evidence classes must remain separate:

1. Versioned repository declarations and code/API requirements: verified by reads.
2. Codex-visible interpreter/toolchain files: verified metadata, not the Docker
   image, not a native import/GPU compatibility test, and not thesis provenance.
3. Working image/container/host runtime: recovery pending using the filtered
   [diagnostic commands](GPU_ENVIRONMENT_DIAGNOSTICS.md). No container version,
   digest, compiler, GPU model, driver or extension revision is inferred.

## Repository and history inventory

| Artifact | Verified finding |
| --- | --- |
| `requirements.txt` | 13 direct declarations; only `open3d==0.16.0` and `pymcubes==0.1.4` pinned |
| `README.md` setup | Installs requirements, then external nvdiffrast and ashawkey/raytracing; no source revisions or wheel hashes |
| Docker/environment specifications | No Dockerfile/Containerfile, Compose/devcontainer, Conda environment, lockfile or project packaging/build recipe found in checkout, including ignored filenames |
| Reachable ref history | No Docker/container/environment recipe, `setup.py`, CMake or native CUDA/C++ source path found across reachable local/remote refs; unreachable objects and external/private build directories were not inspected |
| Historical Torch declaration | `01ce003aae11d923042490e39b4c203e7b9bbc05` removed `torch==1.11.0`; this is a historical clue, not evidence of the working image version |
| Historical branch requirements | Current requirements bytes match `anisotropic`, `origin/aniso_geometry_safe`, and `origin/ablation_anisotropic`; their algorithms/checkpoints remain distinct |
| `raytracing/` | Only `__init__.py` and `raytracer.py`; wrapper eagerly imports externally installed `_raytracing`; no extension source/build recipe/binary is present |
| `run_all_{marineris,mariana,tonga}.sh` | Legacy training launch/log scripts; commented Conda/module/CUDA suggestions are placeholders, not environment specifications; not executed |
| `colmap/build.py`, Windows build helper | Vendored standalone COLMAP download/build tooling, not the research extension build; not executed |
| `assets/bsdf_256_256.bin` | Geometry shader reads it using a relative code-directory path; retain code/assets together under `/workspace` |

`requirements.txt` SHA-256:
`6834b22ad7f6b1ae1594731c4281363610df4dff357ebd46d5fbb92e685dcbcb`.
Local raytracer wrapper SHA-256:
`f670b215c5379d3bdcd9993c836935ea658f5ee9570bbe46bb38096b3f0ac25f`.
Neither hash identifies an external CUDA extension or working image.

If a Dockerfile is recovered elsewhere or from additional history, treat it as
another candidate recipe. It does not identify the current image without digest,
build/dependency evidence and investigation of live-container changes. No
Dockerfile was available to compare during this inspection.

## Dependency and native-extension inventory

| Dependency | Repository declaration / use | Required recovery or compatibility decision |
| --- | --- | --- |
| Python | No historical pin; infrastructure supports 3.9 syntax/APIs; CPU checks used 3.9.12 | Exact CPython patch, interpreter path, SOABI and platform from the training container |
| PyTorch | Not in current requirements; core networks/Adam/autograd/CUDA/global tensor defaults/checkpoints | Exact wheel/build CUDA tag, package source/hash, compiled CUDA/cuDNN, C++ ABI and build config |
| torchvision | No import found in repository Python sources; not declared | Inventory if installed or required transitively; do not add it merely because Torch is present; if retained, use its official matching Torch release |
| CUDA runtime/driver/toolkit | `.cuda()` and CUDA default tensors; native kernels | Distinguish host driver, Torch's compiled/runtime CUDA, toolkit `nvcc`, and toolkit supported host compiler; none implies another's version |
| nvdiffrast | README external install; eager `nvdiffrast.torch` import; geometry uses `dr.texture` for BSDF lookup | Exact upstream/fork commit, source edits, build mode, wheel/plugin hashes, supported Torch/CUDA, architecture flags and runtime/JIT requirements |
| `_raytracing` / distribution `raytracing` | External binary imported by local wrapper; material BVH/intersections | Recover extension source/fork/commit, patches, build recipe/flags/Eigen, binary hash and linkage; import name alone does not establish implementation |
| Open3D | `==0.16.0`; mesh/point-cloud read, in-memory normals, crop/normalization and optional exports | Exact installed version/wheel/hash and native dependencies; CPU mesh operations do not require assuming an Open3D CUDA build |
| PyMCubes / import `mcubes` | `==0.1.4`; eager field import, optional geometry extraction | Matching Python/NumPy/native wheel or pinned compiler build; required to import the shared field module even if extraction is disabled |
| NumPy | Unpinned; all dataset/geometry/image calculations and checkpoint scalars | Exact version/wheel; preserve dtype/API behavior and extension C ABI |
| SciPy | Missing direct requirement; `scipy.spatial.cKDTree` eagerly imported | Add explicit pinned direct dependency in 1D.3B; recover wheel/BLAS native linkage |
| OpenCV | Unpinned `opencv-python`; dataset reads, resize/crop, metric-image padding | Recover exact variant/version; do not silently substitute headless/contrib builds; preserve image operations |
| scikit-image | Unpinned; image I/O and `structural_similarity(..., channel_axis=2)` | Recover version supporting current API; freeze metric and image serialization behavior |
| ImageIO, Pillow | ImageIO imported directly but undeclared; Pillow is an image backend | Explicit version/backend/plugin pins; PNG/JPEG and depth decoding can affect evidence |
| h5py / HDF5 | Unpinned; eager utility import | Recover Python wheel and underlying native HDF5 version/linkage |
| Matplotlib | Missing direct requirement; eagerly imported by validation draw utilities, Agg backend | Pin version and required runtime libraries; keep cache/config outside read-only code/inputs |
| plyfile, transforms3d | Unpinned; eager I/O/pose utilities | Explicit versions; external `plyfile` differs from vendored `colmap/plyfile.py` |
| tensorboardX | Unpinned; logger/SummaryWriter | Freeze version plus relevant protobuf/TensorBoard transitive stack |
| PyYAML, tqdm | Unpinned configuration/progress dependencies | Exact versions; retain `FullLoader` scalar/merge semantics |
| trimesh | Unpinned requirement; standalone extraction/evaluation | Retain in reference inventory; optional tool layer unless a recovered runtime package requires it |
| xatlas | Unpinned requirement; no Python import found | Investigate why installed; do not invent a core material dependency |
| pypng / import `png` | Undeclared, lazy `utils.base_utils.save_depth` helper | Optional utility dependency, not an active training call identified here |
| Blender/COLMAP and ancillary packages | `bpy`, COLMAP executable, optional `requests`, `lxml`, `pandas`, `pyntcloud`, `nerfactor` helper imports | Separate optional preprocessing/relighting/evaluation environment; no standalone tool integration in 1D.3A/B |

The two declared legacy native pins do not select a complete compatible stack.
Open3D 0.16 documents Python 3.7–3.10 packages, so adopting a newer interpreter
without reviewing wheel availability would be a separate decision.
[Open3D 0.16 documentation](https://www.open3d.org/docs/0.16.0/getting_started.html)

### External build evidence, not the working container recipe

Upstream ashawkey/raytracing `setup.py` currently declares a `CUDAExtension`
named `_raytracing`, sources `bvh.cu`, `raytracer.cu`, `bindings.cpp`, Linux
`-O3 -std=c++14` flags, and Eigen discovery/download. It also declares extra
dependencies including a GUI package. None of those source files are in this
repository; the installed binary may be another revision/fork. Recover its actual
recipe before choosing flags or dependencies. Vendor/hash Eigen inputs in the
future build instead of permitting an untracked setup-time download.
[Upstream build recipe](https://raw.githubusercontent.com/ashawkey/raytracing/main/setup.py)

PyTorch 2.4's extension builder supplies C++17 when no standard is explicit and
propagates Torch's C++ ABI setting. A historical explicit C++14 recipe is therefore
a compatibility concern for modern Torch; merely changing compiler flags is not
authorized in this inventory. Build against the selected Torch installation,
supported CUDA host compiler, Python SOABI and recorded architecture targets.
[PyTorch 2.4 extension implementation](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/cpp_extension.py)

Current nvdiffrast installation guidance recommends Python 3.9+, Torch 2.7+ and a
matching CUDA toolkit, and discusses explicit GPU architecture targets. Those are
current upstream recommendations, not requirements of the unobserved historical
installation. Do not install floating HEAD or upgrade Torch to satisfy it; recover
the actual extension revision first. Import success alone does not exercise a
plugin that may compile lazily on first use.
[nvdiffrast installation](https://nvlabs.github.io/nvdiffrast/#installation)

## Codex-visible metadata, separate from the research container

These values were read without importing Torch/Open3D/nvdiffrast or loading native
extensions. They establish file/distribution evidence only.

| Component | Observed in Codex context |
| --- | --- |
| OS/architecture | Ubuntu 22.04.5 LTS, x86_64 |
| Python | CPython 3.9.12 |
| Torch distribution metadata | `2.4.0` |
| Installed `torch/version.py` | `2.4.0+cu121`, compiled CUDA `12.1` |
| Installed `TorchConfig.cmake` | `_GLIBCXX_USE_CXX11_ABI=0`; not independently checked against loaded Torch |
| Toolkit | `nvcc` release 13.3, V13.3.33; CUDA version JSON reports 13.3.0 |
| Compilers | GCC/G++ 11.4.0 (Ubuntu package 11.4.0-1ubuntu1~22.04.3) |
| NumPy / SciPy | 1.21.5 / 1.7.3 |
| scikit-image / ImageIO / Pillow | 0.19.2 / 2.9.0 / 9.0.1 |
| h5py / Matplotlib / plyfile | 3.6.0 / 3.5.1 / 1.1 |
| PyYAML / tqdm | 6.0 / 4.64.0 |
| setuptools / wheel | 61.2.0 / 0.37.1 |
| Missing named distribution metadata | torchvision, torchaudio, nvdiffrast, raytracing, Open3D, PyMCubes, OpenCV variants, transforms3d, tensorboardX, trimesh, xatlas, ninja, pypng |
| Missing module specifications | torchvision, nvdiffrast, `_raytracing`, `cv2`, Open3D |
| Present tools, versions not all queried | docker, nvidia-smi, nvcc, gcc/g++, cmake, readelf; Ninja command absent |

The visible Torch CUDA 12.1 build and default CUDA 13.3 toolkit are a mismatch
for planning an extension rebuild. This does not identify the toolkit in
`nerofactor` and does not prove an existing binary is unusable. No rebuild/import
was attempted. The known CPU infrastructure evidence is not GPU validation.

## Geometry versus material requirements

| Execution path | Actual compute/use | Shared import and runtime constraints |
| --- | --- | --- |
| Config-only validation | Stdlib/PyYAML, metadata and selected input hashes | No model/GPU/extension imports; usable without a native training environment |
| Geometry | SDF/autograd/CUDA, shading/BSDF `dr.texture`; optional marching cubes in validation | Shared renderer eagerly imports raytracing/Open3D; field eagerly imports nvdiffrast/PyMCubes/SciPy even when some calls are unused |
| Material | Read-only mesh, BVH `_raytracing`, anisotropic Monte Carlo shader; unchanged validation and final material NVS | Same renderer/field module imports, dataset/image/pose and logger stack; do not omit nvdiffrast/PyMCubes just because material forward does not directly call them |
| Standalone extraction/evaluation/relighting | Additional rasterization/trimesh/Blender/COLMAP/optional tooling | Deferred; do not claim those paths are isolated or validated by a training image |

The shared import graph favors one pinned core research image for both stages,
with optional tools separated. Geometry final NVS remains deferred; material NVS
retains its existing metrics. Recover CUDA/OpenGL/EGL library needs for the actual
nvdiffrast revision rather than assuming a display server, selecting another
renderer backend, or using privileged execution. CUDA texture use is not evidence
that the standalone OpenGL rasterization paths have been tested.

## Required recovery from image, container and host

- Immutable image ID/repository digest, OS/architecture and intended interpreter;
  compare the container's `.Image` with the current `nerofactor` tag's image ID.
- Live container package versions, editable/VCS dependency commit IDs and native
  binary locations/hashes. Mounts and writable-layer package changes may make
  the running environment differ from its base image; image tags are mutable.
- Exact external extension sources, uncommitted patches, licences, compiler
  flags, build standard, Eigen version/hash, build tools, architecture list and
  built-wheel/plugin hashes. A binary hash alone does not recover source history.
- Torch build CUDA/cuDNN/config/C++ ABI, Python SOABI, CUDA toolkit and GCC/G++/
  glibc/libstdc++ versions; relevant shared-library dependencies/RPATH if needed.
- Host GPU model/UUID/compute capability, driver version and NVIDIA Container
  Toolkit/engine/runtime device configuration. `nvidia-smi` driver support,
  `torch.version.cuda`, and `nvcc --version` answer different questions.
- The original image build context, recipe/base digest and pinned package sources;
  these may not be recoverable from image metadata. No full inspect/history/env
  dump is requested because those can expose credentials or private build values.

The [terminal diagnostic sheet](GPU_ENVIRONMENT_DIAGNOSTICS.md) collects a first
pass without installing or compiling anything. Unknown source checkouts and
unavailable binary linkage must remain explicit rather than assigned guessed pins.

## Recommended pin specification, pending recovery

**Preferred specification: freeze the observed working stack, then reconstruct
it from reviewed sources. It is not build-ready while required pins are pending.**
Do not promote Codex's interpreter/toolchain or a historical requirements line to
the working-image reference. Retain an immutable reference image identifier when
recovered, but recognise that it does not capture modified container layers or
external mounts. No actual lockfile/Dockerfile is created in 1D.3A.

| Pin | Verified declaration/evidence | Recommended 1D.3B decision |
| --- | --- | --- |
| Image/base | Image name supplied; digest unknown | Record observed image ID/digest and reconstructed base digest; no `latest` pin |
| OS/Python | Codex Ubuntu 22.04.5/Python 3.9.12 only | Recover exact container OS and Python patch/SOABI; retain 3.9.12 only as a CPU-characterized candidate |
| Torch | Historical 1.11.0; Codex files 2.4.0+cu121; container unknown | Exact working wheel/build and hash; resolve the discrepancy before selection |
| torchvision | No direct application requirement | Omit from minimal core if unnecessary; otherwise pin the matching observed official pair |
| Toolkit/runtime/driver | Working versions unknown | Pin matching Torch/toolkit runtime versions and compatible host compiler; record required driver capability, never install a host driver into the image |
| Open3D/PyMCubes | Requirements 0.16.0/0.1.4 | Retain as declared candidate pins; compare actual working versions before deciding whether to preserve a discrepancy |
| NumPy/SciPy/image stack | Codex CPU versions above only | Recover exact reference versions and hashes; no automatic NumPy 2 or image-backend migration |
| Remaining Python/native packages | Mostly floating or missing declarations | Explicit direct and transitive version/wheel hashes and build dependencies; use a separate development/test/tool group |
| Extensions/Eigen | Working source revisions and build flags unknown | Exact commits, patch hashes, source archive hashes and wheel hashes; failure to recover is a reconstruction blocker |
| GCC/G++/CMake/Ninja/setuptools/pip | Working versions unknown | Pin toolchain/package versions compatible with selected CUDA and extension recipes |
| C++/Python ABI, SM targets | Working values unknown | Record Torch ABI and CPython SOABI, derive explicit architecture targets from the actual GPU set, preserve tested compiler flags |

Conditional examples for review, **not selected stacks or install commands**:
if the working image really uses Torch 2.4.0 with CUDA 12.1, evaluate a matching
CUDA 12.1 toolkit and optional torchvision 0.19.0, rather than Codex's 13.3 toolkit.
If recovery instead confirms historical Torch 1.11.0+cu113, the official optional
vision pair is 0.12.0 with a CUDA 11.3 toolchain. Neither example supplies missing
extension commits, compiler pins, driver/GPU support or scientific parity.
[Official historical Torch combinations](https://pytorch.org/get-started/previous-versions/)

## Compatibility risk assessment

| Risk | Evidence / implication | Required response before build or native acceptance |
| --- | --- | --- |
| Reference environment unknown | Codex cannot inspect image/container; live changes possible | Obtain filtered runtime evidence; do not claim current Dockerfile/image equivalence |
| Extension provenance missing | Only Python raytracer wrapper versioned | Recover source, patches and build artifacts before declaring reproducibility |
| CUDA/compiler/ABI mismatch | Codex Torch CUDA 12.1 versus nvcc 13.3; actual container unknown | Match selected toolchain/Torch/ABI; do not disable compatibility checks as a workaround |
| New upstream versus historical code | Current nvdiffrast guidance is substantially newer; raytracer upstream explicit C++14 | Pin actual revisions, review build flags separately; do not silently upgrade mathematics or dependencies |
| GPU SM/driver coverage unknown | No successful GPU query in Codex | Confirm hardware/driver and extension architecture coverage before building/testing |
| NumPy/native ABI and image changes | Floating requirements; native mesh/image dependencies | Retain/reference exact versions; compare preprocessing and metric fixtures before any version change |
| Headless/system-library gaps | OpenCV/Open3D/graphics imports may require additional libraries | Recover actual linked/runtime libraries; headless substitutions require review and tests |
| Checkpoint/runtime changes | Legacy CUDA tensor defaults and serialization are version-sensitive | Preserve selected Torch behavior and explicit compatibility gates; no checkpoint conversion or evaluator repair here |
| Read-only runtime writes | Some extension revisions build lazily, libraries cache in user directories | Prebuild where supported; route necessary JIT/config/temp caches under the selected `/outputs` and test with disposable fixtures |

Driver/runtime compatibility has version/platform limits; recover the actual
driver/toolkit rather than assuming a newer toolkit works with any exposed GPU.
[NVIDIA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
NumPy documents that extensions built against NumPy 1.x may fail with NumPy 2;
an unconstrained upgrade is therefore not an infrastructure-only preservation step.
[NumPy ABI troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)

## Proposed Docker architecture and 1D.3B sequence

1. Review recovered evidence and resolve every required pending pin. Keep the
   current installation/image/container untouched. An unrecoverable source recipe
   must be labelled a new reconstruction candidate, not the historical environment.
2. With separate 1D.3B approval, create a digest-pinned CUDA development builder
   and locked Python/toolchain dependency set. Build exact extension wheels against
   the selected Torch/ABI/SM targets; preserve source patches and hashes. Avoid
   install-time downloads of unpinned Eigen/build inputs. No research-data access
   is needed for this build.
3. Produce a pinned runtime image with compatible shared libraries and installed
   wheels; use the matching development image as the runtime only if recovered
   lazy/JIT behavior requires a compiler/toolkit. Optional COLMAP/Blender tools
   live in another target. Do not copy datasets, outputs, secrets or Git metadata
   into the build context; review `.dockerignore` in 1D.3B.
4. Set `WORKDIR /workspace`, retaining code-relative assets/split reads. Mount
   versioned code read-only or bake its reviewed snapshot into the image.
   The storage contract is:

   | Container path | Access | Purpose |
   | --- | --- | --- |
   | `/workspace` | Read-only code or immutable image content | Source/configs/assets, no dataset/output symlink |
   | `/inputs/datasets` | Read-only | Original datasets |
   | `/inputs/thesis` | Read-only | Historical meshes/checkpoints/results |
   | `/inputs/geometry` or another declared input root | Optional, read-only | Selected prior-run geometry, disjoint from writable output |
   | `/outputs` | Writable, one selected experiment | Persistent allocation/checkpoints/logs/cache/renders/debug/tmp |

5. Run with explicit non-root host UID/GID and GPU selection, no Docker socket
   mount or privileged mode. Container runtime/device capability selection is
   distinct from installing CUDA into an image. Set conventional cache/home/temp
   paths to owned `/outputs` storage before scientific imports; preserve existing
   preprocessing identities and shader scope. A read-only root filesystem is a
   proposed validation target, not an implemented or tested claim.
   [NVIDIA device/capability configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
6. Review build/runtime definitions and locks before any build. If access remains
   restricted, an authorized regular-terminal operator can execute separately
   approved builds/tests; Codex must not bypass access controls.

After a separately approved build, native smoke tests should proceed in order:
CPU dependency imports and the existing infrastructure suite with native skips
resolved; minimal isolated CUDA allocation; tiny nvdiffrast texture operation;
raytracer construction/intersection on a deterministic fixture with **more than
eight triangles**; then explicitly budgeted geometry/material initialization and
one validation view, with read-only disposable inputs and an owned output root.
Check that compilation/cache writes stay inside the selected storage and that
the output persists with expected ownership. Test OpenGL only if an approved
path actually requires it. No optimizer steps, full NVS, full experiments or
standalone evaluation are included in the initial smoke stage.

Record import/build/device/mount/parity statuses separately. Preprocessing/native
numerical comparisons need declared reference versions and tolerances; a passing
import or GPU kernel is not reproduced research. 1D.4 retains full experiment
provenance/lifecycle and RNG/sampler restoration. Stop after 1D.3A review now.
