# Paper environment reconstruction: definitions and build prerequisites

This is a reviewable reconstruction of the recovered CUDA 11.1 / system Python
3.8 / Torch 1.8 environment. Current **1D.3C** prepares the separate native model
smoke implementation for review; [NATIVE_GPU_SMOKE.md](../../docs/NATIVE_GPU_SMOKE.md)
provides exact future Docker commands, immutable image-ID selection, read-only
code/input overlays, disposable caches and expected/failure results. The supplied
`nerofactor-paper:quick-candidate-1` environment passed basic native checks and the
100-test Python 3.8 suite according to the researcher; Codex has not executed the
new native model smoke tests. The Dockerfile/image is not changed by this step.
1D.3B-2B's supplied runtime metadata and artifact-readiness history remain below.
**Do not build, install packages or run containers.** See
[PREREQUISITES.md](PREREQUISITES.md) for filtered diagnostics, exact source staging,
dependency closure limits, test results and the separately approved first-build
sequence. [RUNTIME_DEPENDENCIES.md](RUNTIME_DEPENDENCIES.md) records the current
recovered pins, independent inventory validation and unverified CUDA base candidate.
The original definition/review history below remains useful context.
Nothing here modifies `nerofactor`,
`intelligent_keldysh`, original data/results, native source checkouts, or scientific
code. No image or GPU compatibility is claimed from static checks.

The branch started clean at `9a95581c274bcc5e37cb01cc83ebbe704077fecd`.
[reference.json](reference.json) records the researcher-supplied verified reference
image ID, runtime/package versions, extension commits/setup hashes, and historical
binary hash. Codex independently read the recovered nvdiffrast checkout's HEAD,
tracked-clean status, package version and setup hash; its untracked `dockerfile`
was neither included nor modified. Raytracing's container source/binary evidence
is supplied by the researcher; Codex did not access that container or replace it
with upstream HEAD when the exact public-source URL was unavailable.

## Files and build-readiness boundary

| File | Purpose |
| --- | --- |
| [Dockerfile](Dockerfile) | Digest-required historical CUDA development base, offline installs, exact source builds, `/workspace`, non-root default |
| [reference.json](reference.json) | Incomplete dependency/source-provenance manifest with evidence classifications and explicit missing hashes |
| [runtime.reference.json](runtime.reference.json) | 102 installed Python pins/metadata, 21 Debian role versions and toolchain evidence; not an artifact lock or reviewed resolution |
| [requirements.reference.txt](requirements.reference.txt) | Verified live version declarations; **not an install file or complete lock** |
| [resolved.example.json](resolved.example.json) | Deliberately incomplete external input-resolution template; `reviewed: false` |
| [validate.py](validate.py) | Read-only definition and optional staged-artifact/hash checks; never builds or copies sources |
| [preflight.py](preflight.py) | Mandatory future host validation of the complete context; writes only two preflight gate files in the new external context |
| [static_checks.py](static_checks.py) | Read-only manifest/YAML/shell/AST checks; imports only stdlib and existing PyYAML |
| [test_definition.py](test_definition.py) | Static regression checks with synthetic temporary context files and shell checksum verification; no installers |
| [collect_runtime.py](collect_runtime.py) | Filtered read-only installed metadata, dependency closure and toolchain recovery; stream into the reference system Python |
| [stage_sources.py](stage_sources.py) | Offline exact-commit source exports with file hashes, licence preservation and separate provenance |
| [test_prerequisites.py](test_prerequisites.py) | Synthetic source/bundle and metadata tests; no Docker, original artifacts or installation |
| [artifact_candidates.json](artifact_candidates.json) | Publisher wheel listings/checksums; not downloaded artifacts or a completed resolution |
| [record.py](record.py) | Future approved image-build provenance/core checks; **not executed in 1D.3B-1** |
| [.dockerignore](.dockerignore) | Generated-context root allowlist and data/secret/history exclusions |
| [compose.yaml](compose.yaml) | Proposed isolated runtime; no automatic build or execution |
| [compose.geometry-input.yaml](compose.geometry-input.yaml) | Optional read-only prior geometry input |
| [runtime.env.example](runtime.env.example) | Empty runtime settings template; no committed host paths or secrets |

The Dockerfile intentionally has no mutable base-image default. The supplied
historical tag is `nvidia/cuda:11.1.1-devel-ubuntu20.04`; its registry digest is
still unknown. The original image ID is not a registry/base digest. Do not invent
a digest or use the old `nerofactor` tag as an output tag. A distinct, reviewed
candidate image must be built in the next milestone, leaving the reference intact.

Installed Python/build pins, GCC/G++ 9.4.0, Eigen 3.3.7, CMake 3.16.3 and 21 Debian
role versions are now recorded. Actual wheels/hashes, complete base-relative
Debian dependency artifacts, source/build provenance and base identity remain
blockers. The supplied candidate digest is not independently registry-verified
or established as historical. The proposal
requires offline, hash-checked wheels and a complete exact Debian artifact bundle
instead of floating `apt-get`/pip resolution. This is an explicit design choice
for review, not a statement that those artifacts have been recovered.

## Historical differences and preserved behavior

| Historical recipe/evidence | Proposed definition |
| --- | --- |
| CUDA 11.1.1 development image / Ubuntu 20.04 | Same declared base family, with required recovered digest; runtime retains compiler for unchanged nvdiffrast JIT |
| System Python 3.8.10, Torch 1.8.1+cu111, ABI 0 | Exact runtime assertions in future build; no Micromamba or Python/Torch upgrade |
| Floating apt/pip installs and repeated unbounded upgrades | Reviewed exact offline Debian/Python inputs and hashes; no `apt upgrade` or automatic resolver fallback |
| Local nvdiffrast copy and external raytracing | Committed source exports at supplied exact revisions, setup hashes verified, no checkout changes |
| Clone/install NeRO, then clone Point-NeRF | No intermediate repositories; reviewed paper source snapshot only |
| torch-scatter wheel copied but not visibly installed | Not included; no source import identified, no inference from a wheel filename |
| Additional LPIPS/Kornia/warmup/GUI/preprocessing packages | Not assumed application dependencies; include only where reviewed source metadata or supported paths require them |
| torchvision/audio historical declarations | Recorded separately; future resolution explicitly chooses omission of unused extras or retaining the declared matching pins |
| `/home`, Bash entrypoint, original relative writable destinations | `/workspace`, explicit Python commands, non-root runtime, isolated persistent `/outputs`; legacy application code remains unchanged |

Original `requirements.txt`, YAML, geometry/material entry points, BRDFs, losses,
optimizer/scheduler, sampling, seeds and evaluation calculations are untouched.
No deterministic/precision/threading flags or alternate rendering backend are
introduced. Package versions are environment inputs, not claims of scientific
equivalence. Preserve original `shader_cfg` scope and geometry final-NVS deferral.

Python 3.8 findings and native API limits are in
[PYTHON38_COMPATIBILITY.md](PYTHON38_COMPATIBILITY.md).

## Native sources, Eigen, compiler and lazy plugins

Both exact source commits and setup hashes are in the reference manifest. Future
source exports must preserve licences and source bytes, omit `.git`, build outputs,
untracked files and local Dockerfiles, and carry a full file SHA-256 inventory.
The staging operator verifies Git identity before export; setup hash alone does
not prove the rest of a tree. Source revisions likewise do not prove rebuilt
binary identity. The historical `_raytracing` hash is a reference measurement,
not an enforced expected result for a newly compiled binary.

Review the exact raytracing `setup.py` before collecting artifacts. Its Eigen
search/download behavior and declared transitive dependencies must be satisfied
offline, not patched away. The proposal uses reviewed `libeigen3-dev` headers at
`/usr/include/eigen3`; `record.py` checks their version against the resolution.
The supplied inventory now identifies Eigen 3.3.7 and GCC/G++ 9.4.0; verify their
actual artifacts/build selection rather than guessing from the Ubuntu release.
Lock required graphics/runtime libraries and the complete Debian
closure. Preserve recovered C++/NVCC flags, optimizations and ABI 0; do not disable
compiler compatibility checks or add fast-math to make an extension compile.

Raytracing builds against the exact installed Torch, Python and CUDA toolchain.
The proposed build target `TORCH_CUDA_ARCH_LIST=8.0` targets the reported A100 and
requires native acceptance later. nvdiffrast 0.3.3 at the recovered revision sets
that variable to an empty string inside `_get_plugin`, then calls Torch's extension
loader lazily. Its Python/source wheel installation does **not** precompile all
plugins. Retain the development toolkit/Ninja and installed package source files,
route `TORCH_EXTENSIONS_DIR` to the experiment cache, and record actual runtime
plugin hashes/architecture in later validation. No source patch or warm-up is
included here, and no promise is made of GPU-free precompilation of this revision.

The GL plugin links GL/EGL; include reviewed development libraries if retaining
that capability, but do not exercise standalone GL evaluation here. The training
path's CUDA texture operation does not validate OpenGL. Source build dependencies
(including any declared GUI/trimesh extras) must appear in the resolved closure
if needed for `pip check`; they are not silently stripped from recovered setup code.

## Safe future local staging, only after separate review

1D.3B-1/1A performed no original source staging. In 1D.3B-2A the automated
[verified procedure](PREREQUISITES.md#verified-offline-source-staging) supersedes
the manual archive outline below and has exported the pinned nvdiffrast source
only into new temporary storage. No packages were downloaded. Use a **new empty directory outside
this Git repository, both original extension checkouts, data, and results**.
Never build with the repository root, `/home/isavand/nerofactor`, or an original
source checkout as Docker context. A proposed generated layout is:

```text
<external-context>/
  Dockerfile                  # copy the reviewed definition
  .dockerignore               # copy docker/paper/.dockerignore to this root
  PREFLIGHT.base-image        # generated only by successful host preflight
  PREFLIGHT.sha256            # generated full context checksum manifest
  definition/                 # exact inventory below; no other files
    Dockerfile
    reference.json
    requirements.reference.txt
    validate.py
    record.py
    preflight.py
  workspace/                  # approved committed paper source/assets snapshot
  inputs/
    resolved.json             # reviewed completed resolution + full hash maps
    python-complete.txt       # exact names/versions/wheel hashes; one wheel per package
    wheels/*.whl              # all reviewed runtime/build/transitive wheels
    apt/*.deb                 # exact complete Debian package/dependency closure
    apt/SHA256SUMS            # every .deb, flat names, final newline
    sources/nvdiffrast/...    # committed package/source + setup/README/licence
    sources/raytracing/...    # committed build-needed source/includes/licences
```

`validate.DEFINITION_FILES` is the canonical definition inventory. Copy each of
those files byte-for-byte from the approved `docker/paper/` definition. The
reference requirements file is required by the validator even though it is not
an install lockfile. `.dockerignore` explicitly retains every inventory member;
static regression checks run validation against a minimal staged definition and
verify that removing any member fails clearly.

1. Verify local source HEAD equals the supplied commit and tracked/index changes
   are absent, using `git --no-optional-locks`. A clean tree does not identify the
   installed binary, but ensures an unmodified export. Ignore/exclude the known
   untracked nvdiffrast `dockerfile`; never copy entire original directories.
2. Export the **explicit commit**, not the mutable working tree, using Git archive.
   For nvdiffrast the build-needed archive paths are `nvdiffrast`, `setup.py`,
   `README.md`, `LICENSE.txt`. Extract only into the new owned staging directory.
   Preserve all native package source files and licences; samples/data are not
   required for this build. Archive hashes and per-file hashes must be recorded.
3. If raytracing is available only in the original running container, an authorized
   regular-terminal operator can stream a read-only `git --no-optional-locks -C
   /home/Nerofactor/raytracing_git archive --format=tar
   29329268a86921f1c938b05277937369531209a4` to external staging. Inspect the committed
   archive paths first, then retain setup, source, headers, licences and metadata
   needed to build. Do not use `docker cp` of an entire checkout, include credentials,
   run setup to inspect it, change the container, or bypass socket restrictions.
   If unavailable, stop source preparation and mark it missing. Do not substitute
   HEAD/a fork/a new commit or a prebuilt binary without separate review.
4. Export the approved paper commit into `workspace` using the allowlist enforced
   by `validate.py` (core source directories, configs, assets, entry points and
   optional README/AGENTS/requirements). Do not export data/results/checkpoints,
   `.git`, credentials, unrelated repositories or local changes. The current
   scientific source is preserved; commit/review any intended definitions first
   instead of silently snapshotting an unreviewed dirty tree.
5. Recover missing versions and package/wheel sources without exposing credentials.
   Collect **only reviewed exact artifacts** and hashes outside Git. This milestone
   neither installs nor downloads any. Resolve all required transitive/build
   dependencies, including those declared by the exact extension setup files.
   Wheel availability, Debian dependency closure and compatibility are not assumed.
6. Complete `inputs/resolved.json` from the deliberately invalid template. Required
   fields: reviewed resolution flag; digest-qualified base; approved workspace
   commit and all workspace relative-file hashes; exact Python package versions;
   explicit historical-extras decision; exact Debian package versions; compiler/
   Eigen/CMake versions, reviewed source C++/NVCC flags and NVCC/ABI; exact source commit declarations; and `files`
   mapping **every file beneath inputs except resolved.json** to SHA-256.
   Verify exported setup hashes against the reference manifest independently.
   `reviewed: true` records a researcher review, not an automatically inferred fact.
7. `python-complete.txt` has one line per selected package, in this form:
   `NAME==EXACT_VERSION --hash=sha256:ACTUAL_WHEEL_HASH`. No URLs, wildcards,
   editable installs or index options. Match its package set to `python_packages`.
   The file is absent until resolved, not a fabricated lockfile in this change.
   Keep nvdiffrast/raytracing out of this prebuilt package set: those two wheels
   must be built from the exact exported sources, then installed explicitly.
   `apt/SHA256SUMS` uses `SHA256` followed by two spaces and a flat `.deb` filename.
   All staged files must be regular files with no symlinks/hardlinks; materialize
   exports into new files and validate the complete sets before any installer runs.
8. **Mandatory preflight before any future build:** use an available, trusted host
   Python 3.8 or newer (stdlib only). Do not assume Python is installed in the CUDA
   base, install a bootstrap interpreter, or skip validation if host Python is
   unavailable. Stop and recover an approved host preflight environment instead.
   Run the reviewed script against a fresh completed external context:

```sh
python3 -B docker/paper/preflight.py \
  --context /ABSOLUTE/EXTERNAL/CONTEXT \
  --base-image 'nvidia/cuda:11.1.1-devel-ubuntu20.04@sha256:ACTUAL_VERIFIED_DIGEST'
```

The placeholder deliberately fails validation. Preflight first requires the exact
reviewed definition files and context Dockerfile/ignore rules, then executes the
complete existing `validate.inputs` checks (including reviewed resolution, source
pins, all artifact hashes and workspace inventory). It rejects extra context files
and existing gate files. Only on success does it create `PREFLIGHT.base-image` and
`PREFLIGHT.sha256`, covering **every context file** except the checksum manifest
itself, including `resolved.json`, definitions and the base-selection file. It does
not change any staged package/source/workspace bytes or install anything.

Save/review the printed `PAPER_PREFLIGHT_SHA256` **outside the context**. During a
separately approved build, explicitly supply that digest as the required build
argument, alongside the matching digest-qualified `PAPER_BASE_IMAGE`. Never
substitute an arbitrary digest, accept a receipt from an untrusted source, or
generate one without the full preflight. A checksum is an integrity record, not
proof of researcher review or authenticated proof that validation ran. Keep the
context exclusively controlled and unchanged from preflight through Docker's
snapshot; if inputs/definitions change, prepare and validate a new context.

The first Docker RUN uses only the base's Bash/coreutils/find tools: it checks the
externally supplied receipt hash, regular-file policy, exact file count, all file
hashes and matching base selection. Missing/filtered/extra or changed files fail
before any installer runs. These utilities must exist in the recovered base; if
absent, the build fails closed rather than installing a validator dependency.
Only after this boundary may `dpkg --unpack`/configuration execute. The subsequent
Python validator remains an independent recheck after Python becomes available.

No image build/run command is prescribed for execution in this milestone. The
checker validates file identities and recorded selections; it does not authenticate
an operator's source declaration, resolve Debian dependencies, inspect wheel
platform/metadata compatibility or establish native numerical behavior. Offline
`dpkg`, hashed `pip --no-deps` and `pip check` fail rather than fetch replacements.

## Runtime contract

The image uses baked paper code/assets under `/workspace`. Default UID/GID is
1000:1000; Compose requires explicit host UID/GID, GPU, **new** image selection
and existing bind sources. Do not run as host UID 0. Non-root support and device
access still need acceptance tests; no host permission changes are proposed.

| Path | Access |
| --- | --- |
| `/workspace` | Read-only image code/assets |
| `/inputs/datasets` | Read-only original datasets |
| `/inputs/thesis` | Read-only historical meshes/checkpoints/results |
| `/inputs/geometry` | Optional read-only prior geometry input override |
| `/outputs` | Only selected experiment's persistent writable bind |

Copy/fill the empty environment example **outside Git** after approval. Compose
does not build, adopt a reference image/container, silently create missing host
directories, expose the Docker socket, request privileged capabilities or default
to all GPUs. Read-only root filesystem and disabled networking are proposals to
validate later, not proven runtime behavior. Use optional geometry override only
when that root is explicitly declared through the material CLI.

The Python entrypoint defaults to help. Supply explicit geometry/material CLI
arguments with schema-1 config, roots and `--layout mounted`; training additionally
requires explicit `--execute` and remains separately authorized. **Do not create
cache/home/temp directories inside an empty `/outputs` before run allocation**:
the mounted allocator must exclusively initialize it. ENV points caches there,
but no startup hook writes files. Existing runtime infrastructure makes the cache/
temp directories after allocation and before scientific imports. No scientific
RNG, optimizer, schedule, sampler or shader option is set by the image.

This runtime layout targets the isolated CLIs. The legacy entry point is retained
unchanged; it is not automatically made write-isolated and may fail when trying
to write its original relative destinations under read-only code. Use the
preserved original environment as the historical reference, not a silent legacy
path rewrite inside this definition.

`record.py` in a future image build verifies core version/ABI/NVCC/Eigen, records
actual package and system versions, input/definition/wheel hashes and the rebuilt
raytracer hash. It does not import the rendering extensions or run CUDA kernels.
Final image ID/digest and lazy plugin/GPU identity must be captured externally
after the later build/runtime checks; they cannot be truthfully known here.

## Static validation, blockers and Step 1D.3B-2

Run the static checker and synthetic regression fixtures:

```sh
python3 -B docker/paper/static_checks.py
python3 -B -m unittest discover -s docker/paper -p 'test_*.py' -v
```

The checker parses declarations, the complete staged definition allowlist,
pre-install gate ordering, Compose policy, Docker RUN shell grammar and Python
3.8 source grammar. The 14 regression tests create only synthetic temporary
artifacts (not real wheels, Debian packages or extension exports), validate their
manifests and execute only the first checksum-verification shell command. They
check missing definition files, unreviewed/changed inputs, receipt anchoring,
base selection, collisions, file-set changes and symlink rejection. Neither
command stages original sources, downloads, invokes Docker/pip or runs models.
The additional prerequisite tests exercise offline synthetic Git exports/bundles
and sanitized dependency metadata. See the compatibility report and
[current prerequisite results](PREREQUISITES.md#cpu-evidence-and-the-separately-approved-first-build).
1D.3B-2A now authorizes the CPU infrastructure suite and its narrowly scoped test
harness fix; actual Python 3.8 execution remains unavailable.

Before building: verify/review the base digest; recover remaining system closure
pins and all actual artifact hashes; inspect/stage the exact raytracing tree;
review recovered Eigen/compiler
selection, full dependency closure, context/hash coverage and Docker/Compose
support for offline RUN, strict bind sources and GPU device reservations. Complete
actual Python 3.8 test execution after the test-only compatibility fix. The
Dockerfile is proposed and its static checks are not a successful build or complete
reproducibility lock.

Proposed 1D.3B-2, requiring new approval: review/finalize external inputs; build a
distinct candidate without touching the old image/container; record build output
and exact installed identities; run dependency/core imports, native CPU tests and
read-only mount/cache/ownership checks with disposable fixtures. Then separately
budget a tiny CUDA allocation, CUDA nvdiffrast texture/plugin compile and BVH trace
on a fixture with more than eight triangles. Preserve the lazy build behavior;
capture compiler flags, architecture and plugin hashes. Follow with bounded
geometry/material initialization and one validation view only after prerequisites
pass. No optimizer step, full NVS/training or standalone evaluator belongs in the
first native smoke phase. Failures must be reported before proposing scientific,
compiler-flag or source patches. Full provenance/RNG lifecycle remains 1D.4.
