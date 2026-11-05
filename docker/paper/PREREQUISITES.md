# Build prerequisite recovery: Steps 1D.3B-2A / 1D.3B-2B

The architecture, dependency pins and install/preflight boundary are unchanged.
1D.3B-2A prepared recovery/staging tools and fixed the CPU test harness;
1D.3B-2B integrates the supplied installed-metadata inventory without building,
installing or executing scientific GPU code. The 2A tree started clean
on `paper/infrastructure` at `097b70201030129381e562aab6b20c80cdcf03ae`.

## Evidence and unresolved resolution fields

The live versions supplied by the researcher remain in
[reference.json](reference.json): system Python 3.8.10, Torch 1.8.1+cu111, CUDA
11.1/NVCC 11.1.105, ABI 0, and the ten verified Python package pins.
1D.3B-2A could not recover container-only metadata because Docker socket inspection
was denied. In 1D.3B-2B the researcher supplied `runtime_inventory.json`:
[runtime.reference.json](runtime.reference.json) now preserves 102 installed pins,
142 active requirement edges, 21 Debian role pins and compiler/Eigen versions.
[RUNTIME_DEPENDENCIES.md](RUNTIME_DEPENDENCIES.md) documents the independent
consistency checks, capture limitations, candidate base and remaining artifacts.
Host Python 3.9.12 is test evidence only; no access workaround or runtime upgrade
was performed, and installed versions are not verified wheel identities.

| Resolution field | Current evidence / remaining work |
| --- | --- |
| `base_image` | Historical tag known; supplied candidate digest remains registry-unverified and not established historically; final `nerofactor` image ID is not its base digest |
| `workspace_commit` | Exact approved 1A commit above is available; choose/review the intended build source commit explicitly; never select a branch implicitly |
| `workspace_files` | Must be generated from that commit's selected export, not the dirty working tree |
| `python_packages` | 102 installed versions and recorded active closure recovered; 100 prebuilt distribution artifacts and two source-built distributions still need artifact verification |
| `historical_extras_decision` | torchvision 0.9.1+cu111 / torchaudio 0.8.1 remain declarations; explicitly omit unused extras or retain those pins after collecting metadata |
| `system_packages` | 21 exact Debian role pins recovered; five failed queries and versionless `ninja-build` unresolved; complete base-relative dependency/artifact closure missing |
| `toolchain` | GCC/G++ 9.4.0, Eigen 3.3.7, CMake 3.16.3, glibc 2.31, NVCC 11.1.105 recovered; Ninja binary banner differs in format from distribution version; actual source build flags unresolved |
| `sources` | Both exact revisions/setup hashes remain required; nvdiffrast readable/export verified; raytracing unavailable to Codex |
| `files` | Full hashes of acquired wheels, Debian packages, exports and complete Python lock are still required; no artifact bundle or valid `resolved.json` produced |
| `reviewed` | Example stays `false`; only a researcher may mark a completed resolution reviewed |

[artifact_candidates.json](artifact_candidates.json) records publisher-listed
CPython 3.8 Linux x86-64 wheel candidates, not downloaded artifacts or historical
binary identity. NumPy, Open3D and PyMCubes checksums were read from the respective
[NumPy](https://pypi.org/project/numpy/1.24.4/),
[Open3D](https://pypi.org/project/open3d/0.16.0/) and
[PyMCubes](https://pypi.org/project/PyMCubes/0.1.4/) release-file metadata.
The [official Torch CUDA 11.1 index](https://download.pytorch.org/whl/cu111/torch_stable.html)
lists the 1.8.1+cu111 CPython 3.8 Linux wheel; its hash remains unresolved here.
These observations do not authorize substituting newer releases or populate the
reviewed input file manifest. Validate actual artifact bytes and complete closure.

## Safe read-only recovery in the regular terminal

Run from the repository, using the existing working Docker access. The commands
do not install, run a model, import rendering extensions, dump environment values
or alter the running container. Preserve errors; do not use sudo or bypass denied
access. Save output only in a new owned location **outside the build context**.

```sh
docker container inspect intelligent_keldysh --format 'id={{.Id}} image={{.Image}} running={{.State.Running}}'
docker image inspect nerofactor --format 'id={{.Id}} digests={{json .RepoDigests}} os={{.Os}} arch={{.Architecture}} layers={{json .RootFS.Layers}}'
docker image inspect nvidia/cuda:11.1.1-devel-ubuntu20.04 --format 'id={{.Id}} digests={{json .RepoDigests}} os={{.Os}} arch={{.Architecture}} layers={{json .RootFS.Layers}}'
docker exec -i --workdir / intelligent_keldysh python3 -B - < docker/paper/collect_runtime.py
```

The collector queries only installed distribution metadata and known tool/system
roles. An already installed `packaging` parser (or pip's vendored parser) follows
active `Requires-Dist` markers and propagated extras, reporting missing versions
and constraint conflicts. Direct URLs and raw malformed requirement text are
omitted. With no parser available it reports direct versions and explicitly leaves
closure unresolved; it never bootstraps dependencies. It also reads Eigen version
macros and exact GCC/G++/NVCC/CMake/Ninja/glibc and Debian role versions/dependency
metadata. It does not invoke pip, Torch, Open3D or extension setup code.

Collect torchvision/audio metadata as closure roots only if reviewing retention:

```sh
docker exec -i --workdir / intelligent_keldysh python3 -B - --include-historical-extras < docker/paper/collect_runtime.py
```

Inspect the pinned raytracing source separately, without running setup:

```sh
docker exec --workdir / intelligent_keldysh git --no-optional-locks -C /home/Nerofactor/raytracing_git rev-parse HEAD
docker exec --workdir / intelligent_keldysh git --no-optional-locks -C /home/Nerofactor/raytracing_git status --porcelain --untracked-files=no
docker exec --workdir / intelligent_keldysh git --no-optional-locks -C /home/Nerofactor/raytracing_git ls-tree -r --name-only 29329268a86921f1c938b05277937369531209a4
docker exec --workdir / intelligent_keldysh git --no-optional-locks -C /home/Nerofactor/raytracing_git show 29329268a86921f1c938b05277937369531209a4:setup.py
```

Check the exact source's Eigen discovery/download code, build flags, metadata
requirements and any submodules. Preserve the supplied setup hash; do not patch
setup to remove dependencies or permit network downloads in the offline build.
Native source flags are separate from Python's compiler defaults. Existing
`build.ninja` flags, if recovered privately, describe a particular build only;
the selected source and later rebuilt binary still require verification.

**Base identity:** a locally available CUDA tag/digest and matching prefix of
reference-image layer DiffIDs are corroborating evidence, not unique proof of
the original base manifest. A final image ID, empty RepoDigests or the current
registry tag cannot recover an unknown historical base digest. If absent, recover
a trusted original build record/base manifest or stop for explicit review of a
newly identified candidate base. Do not silently promote it to historical fact.
Do not pull/build an image merely to fill this field during this milestone.

## Verified offline source staging

[stage_sources.py](stage_sources.py) exports **Git blobs from exact commit objects**,
not a recursive directory copy or an archive extracted with unchecked paths.
It verifies local HEAD/commit and tracked-clean status, extension setup SHA-256,
selected-path existence, Git blob IDs, source file types and export SHA-256s.
It ignores all untracked files, rejects submodules/symlinks, data/generated/private
paths and non-source binary payloads, and always retains root licences/notices.
Paper staging includes only core code/configs/readers and the required committed
BSDF LUT, excluding presentation images and original research data.

Destinations and reports must be new, external to the repository/source checkouts
and all explicitly protected input roots. The operator must declare **every**
other protected input/source location too. Existing destinations never merge or
overwrite; a partial export after an I/O failure is retained for review and cannot
be reused. Reports record the commit, upstream Git tree, selected file SHA-256s,
Git modes/blob IDs, licences and a canonical exported-file-manifest hash. That
hash covers the selected export, not the complete upstream tree or a built binary.
Original source permissions are unchanged; new source files use recorded Git
executable modes subject to the staging operator's umask.

Use an existing empty owned external parent; keep reports/bundles outside context.
Example variables below must be set to reviewed absolute paths. No host locations
are committed as runtime/config defaults.

```sh
python3 -B docker/paper/stage_sources.py --kind nvdiffrast \
  --repo /home/isavand/nerofactor/nvdiffrast \
  --destination "${PAPER_CONTEXT:?}/inputs/sources/nvdiffrast" \
  --report "${PAPER_EVIDENCE:?}/nvdiffrast-source.json" \
  --protected-root /home/isavand/paper/data --protected-root /home/isavand/paper/final_results

python3 -B docker/paper/stage_sources.py --kind paper \
  --repo /home/isavand/paper/Nerofactor-paper --paper-commit "${PAPER_COMMIT:?exact approved commit}" \
  --destination "${PAPER_CONTEXT:?}/workspace" --report "${PAPER_EVIDENCE:?}/paper-source.json" \
  --protected-root /home/isavand/paper/data --protected-root /home/isavand/paper/final_results
```

Both commands require the selected checkout HEAD to match and tracked files to be
clean. Do not commit automatically to satisfy this condition. Review the pending
changes first, then explicitly select the intended commit. No paper snapshot was
staged from this milestone's dirty working tree. The current paper tree has no
standalone root licence; embedded source notices are retained, and no licence is
invented.

For container-only raytracing, an authorized terminal operator can stream a
**standalone Git bundle** to a new external file, without writing in the container:

```sh
( set -C; docker exec --workdir / intelligent_keldysh git --no-optional-locks \
    -C /home/Nerofactor/raytracing_git bundle create - HEAD \
    > "${PAPER_EVIDENCE:?}/raytracing.bundle" )
sha256sum "${PAPER_EVIDENCE:?}/raytracing.bundle"
```

First require HEAD equals `29329268a86921f1c938b05277937369531209a4` and no tracked
changes. Inspect the committed path listing/setup, then create an explicit JSON
list of build-needed relative paths outside the context (e.g. setup, package,
source/include and README paths **only if present in that exact tree**). No guessed
allowlist is supplied while source access is blocked. Preserve all required
licences/includes/build metadata; source content needed by setup must not be
omitted merely to make a small export. The bundle contains history: keep it
outside the build context and inspect it as source evidence, not build input.

```sh
python3 -B docker/paper/stage_sources.py --kind raytracing \
  --bundle "${PAPER_EVIDENCE:?}/raytracing.bundle" --bundle-sha256 "${PAPER_RAY_BUNDLE_SHA256:?}" \
  --paths "${PAPER_EVIDENCE:?}/raytracing-paths.json" \
  --destination "${PAPER_CONTEXT:?}/inputs/sources/raytracing" \
  --report "${PAPER_EVIDENCE:?}/raytracing-source.json" \
  --protected-root /home/isavand/paper/data --protected-root /home/isavand/paper/final_results
```

The script verifies the bundle digest, prerequisites and Git objects in a fresh
temporary bare repository, disables hooks/network protocols, imports only from
the local bundle, and verifies HEAD against the exact source pin. It never fetches
upstream HEAD. If a local raytracing checkout is accessible, `--repo` may replace
the bundle arguments with the same clean/pinned requirements. Missing source,
submodules, licence or required files are blockers, not permission to bypass.

**Observed staging:** nvdiffrast's exact source was exported to a new owned `/tmp`
directory in this milestone: 56 files, setup hash matches the reference, exported
manifest SHA-256 `9b134cd4d1b3b51b0b740101e510a7fca91e1aff1a0d999520902edcf5a3a251`.
`LICENSE.txt` and package source bytes were preserved; samples/data, `.git` and
the untracked `dockerfile` were excluded. This is a source-only export, not a
complete context, source wheel or runnable environment. Raytracing and paper
exports were exercised using synthetic Git repositories/bundles only.

## Minimum training dependency closure and compatibility gates

| Scope | Required roots / why |
| --- | --- |
| Config/path-only CLI | Python stdlib + PyYAML; no models/native imports |
| Both geometry and material | Torch, NumPy, SciPy, Open3D, nvdiffrast, raytracing, PyMCubes, OpenCV, scikit-image, ImageIO, tensorboardX, transforms3d, PyYAML, h5py, plyfile, tqdm, Matplotlib; both eagerly import the shared renderer/field/metrics/dataset modules |
| Native builds and existing manifest gates | System Python/dev headers, CUDA devel/NVCC, GCC/G++/libstdc++/libc, Eigen, pip/setuptools/wheel, Ninja, packaging, pybind11; CMake/pkg-config where exact source metadata requires them |
| Image/array/logging transitives | Pillow and all marker-active dependencies of the selected pinned distributions, including metadata dependencies of Open3D; do not strip GUI/notebook/transitive metadata to force `pip check` |
| System graphics/runtime | Reviewed GL/EGL development libraries for retained nvdiffrast GL capability, and actual ELF dependencies of OpenCV/Open3D/Torch/HDF5; candidate roles in the collector are not a complete Debian install list |
| Excluded unless required by exact metadata | torchvision/audio decision explicit; Point-NeRF/intermediate NeRO, xatlas/Blender/COLMAP executable tooling and unrelated relighting packages are not active training roots; recovered raytracing metadata requires both trimesh and dearpygui |

This is the **source-established root set**, not a solved artifact lock. The
supplied collector inventory now contains a consistent 102-node Python metadata
closure; independently verify acquired wheel
`METADATA`/markers/Requires-Python/tags offline before selecting artifacts. Source
metadata from both exact extension revisions must agree with installed metadata;
a clean `pip check` is a later build acceptance gate, not evidence obtained here.
The local nvdiffrast setup declares NumPy and Python >=3.6; its unchanged native
plugins compile lazily and require the retained development toolkit at runtime.

For Debian closure, use recovered role Depends/Pre-Depends/Provides and exact
versions against the **digest-pinned base's existing package set**; resolve versioned
alternatives, virtual packages and architecture. Acquire only missing selected
packages and prerequisites into the offline bundle after separate approval.
Base packages may satisfy dependencies but must still be recorded. A broad dump
of the final container is not a minimal base-relative closure; the collector does
not pretend to be an apt solver. Full Debian artifact closure cannot be finalized
without base identity and reference metadata.

Compatibility/availability blockers:

- Python 3.8 is supported by the pinned NumPy/Open3D/PyMCubes wheel listings, and
  [SciPy 1.10.1 declares Python >=3.8, <3.12](https://pypi.org/project/scipy/1.10.1/).
  Available listings do not establish installed binary provenance or every
  transitive wheel's compatibility. NumPy 1.24 removed older aliases; retain
  this historical pin and document any resulting source incompatibility.
- [scikit-image 0.21 runtime requirements](https://scikit-image.org/docs/0.21.x/user_guide/install.html)
  include networkx, Pillow, ImageIO, tifffile, PyWavelets, packaging and lazy_loader.
  Exact live versions and recorded active closure are now preserved in the runtime
  manifest; verify actual wheel metadata/markers. Current releases of
  transitives/build tools cannot be assumed to support Python 3.8.
- [CUDA 11.1.1's Linux support table](https://docs.nvidia.com/cuda/archive/11.1.1/cuda-installation-guide-linux/index.html)
  lists Ubuntu 20.04.1 with GCC 9.3.0 and glibc 2.31. This is compatibility context,
  **not a recovered compiler pin**. Preserve ABI 0 and verify actual source flags;
  do not disable compiler checks, add fast-math or substitute CUDA/Torch versions.
- Torch's CPython 3.8 CUDA wheel is listed, but its bytes/hash, native libraries
  and extension linkage remain unverified. `TORCH_CUDA_ARCH_LIST=8.0` remains the
  proposal; nvdiffrast's exact source clears it during lazy compilation. GPU/JIT
  validation and plugin hashes remain deferred.
- Missing base/Python patch-level Debian artifacts, Eigen artifact hashes and
  raytracing setup behavior are build blockers. No fallback to newer packages, source patches or
  a guessed base is authorized.

## CPU evidence and the separately approved first build

The only existing-test edits replace three Python 3.9-only `ast.unparse` branch
selectors with equivalent `ast.dump(ast.parse(...))` structural comparisons.
All 208 existing assertion calls in the two affected files are structurally
unchanged, including the scientific-parity comparisons. No interpreter or
dependency was installed; actual Python 3.8 is unavailable in Codex.

Run the available checks from the repository:

```sh
python3 -B docker/paper/static_checks.py
python3 -B -m unittest discover -s docker/paper -p 'test_*.py' -v
python3 -B -m unittest discover -s tests/infrastructure -v
```

Current results on Python 3.9.12 after 1D.3B-2B: 28 Docker/prerequisite tests passed (0 skipped),
and 97 infrastructure tests passed (3 skipped: native OpenCV, OpenCV/transforms3d
crop, Open3D). Expected simulated failure traces test logging/cleanup and are not
suite failures. A second infrastructure run with `ast.unparse` removed checks the
missing API without claiming a Python 3.8 interpreter test. Static checks cover
the definition inventory, five offline RUN shell commands, Compose isolation,
and Python 3.8 grammar including infrastructure tests. These checks do not build,
install, import native rendering extensions or run scientific GPU work.

**Do not execute a build in 1D.3B-2A or 1D.3B-2B.** After runtime recovery, source review,
artifact acquisition/hash verification, complete `resolved.json` and separate
build approval, the minimum sequence is:

```sh
python3 -B docker/paper/preflight.py --context "${PAPER_CONTEXT:?}" --base-image "${PAPER_BASE_IMAGE:?}"
# Review/save the printed digest outside context, then explicitly set it below.
# Require a distinct unused candidate tag; never use nerofactor or its reference ID.
case "${PAPER_CANDIDATE_TAG:?}" in
  nerofactor-paper:candidate-*) ;;
  *) printf '%s\n' 'Select a distinct nerofactor-paper:candidate-... tag'; exit 1 ;;
esac
if docker image inspect "${PAPER_CANDIDATE_TAG}" >/dev/null 2>&1; then
  printf '%s\n' 'Candidate tag already exists; refuse overwrite'
  exit 1
fi
test ! -e "${PAPER_EVIDENCE:?}/candidate-image.id"
DOCKER_BUILDKIT=1 docker build --network=none --progress=plain \
  --platform "${PAPER_PLATFORM:?verified build platform}" \
  --build-arg PAPER_BASE_IMAGE="${PAPER_BASE_IMAGE:?}" \
  --build-arg PAPER_PREFLIGHT_SHA256="${PAPER_PREFLIGHT_SHA256:?reviewed preflight digest}" \
  --tag "${PAPER_CANDIDATE_TAG:?distinct unused candidate tag}" \
  --iidfile "${PAPER_EVIDENCE:?}/candidate-image.id" \
  -f "${PAPER_CONTEXT:?}/Dockerfile" "${PAPER_CONTEXT:?}"
docker image inspect "${PAPER_CANDIDATE_TAG:?}" --format 'id={{.Id}} digests={{json .RepoDigests}} os={{.Os}} arch={{.Architecture}}'
```

Keep evidence outside context; retain logs/failures and inspect the recorded
package/source/compiler/ABI identities. These commands are a future plan, not a
request to run now. Container startup, GPU/native smoke tests and scientific
execution need their own approved scope; no first build proves numerical parity.
