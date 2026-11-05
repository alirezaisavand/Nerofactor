# Incremental infrastructure implementation plan

Status: **1D.4A authorizes baseline evidence/preparation documentation only**.
The researcher reports successful geometry/material native model smoke tests in
`nerofactor-paper:quick-candidate-1` and all 109 Python 3.8 infrastructure tests
passing. Original launch/test reports and immutable identities have not been
supplied/checked in here; these are reported successes, not Codex GPU measurements.
See [BASELINE_PREPARATION.md](BASELINE_PREPARATION.md) for evidence and blockers.
No inference runner, GPU evaluation/training, checkpoint deserialization,
scientific/configuration change, image modification or branch merge is authorized.

The starting implementation is `anisotropic` at
`e1e859779301faced46d6476691e20b546c27ae0`, on `paper/infrastructure`.
[BRANCH_COMPARISON.md](BRANCH_COMPARISON.md) documents alternative behavior;
[BASELINE_INVENTORY.md](BASELINE_INVENTORY.md) separates artifact evidence and
historical reports from independent reproduction. Follow [AGENTS.md](../AGENTS.md).

## Architecture and storage contract

Keep source code, original inputs, historical results, and future outputs in
separate locations. Do not solve legacy relative paths by combining originals
and outputs behind a writable `data` directory or symlink.

| Host location | Proposed container location | Access and purpose |
| --- | --- | --- |
| `/home/isavand/paper/Nerofactor-paper` | `/workspace` or image-baked code | Versioned source; read-only if mounted for execution |
| `/home/isavand/paper/data` | `/inputs/datasets` | Read-only original datasets |
| `/home/isavand/paper/final_results` | `/inputs/thesis` | Read-only historical meshes, checkpoints, logs, and comparison artifacts |
| `/home/isavand/paper/paper_runs/<experiment-id>` | `/outputs` | The selected experiment's persistent writable output tree |

For example, material geometry can explicitly identify
`/inputs/thesis/meshes-pretrained/nerf/smoothcylinder.ply`; a fresh geometry run
must instead be referenced by its recorded artifact identity. Do not silently
substitute meshes, stages, datasets, or coordinate frames because names match.
Mount only the selected experiment for writes so other runs are not exposed as
writable storage. Input mounts must fail clearly when a required host path is
missing, without silently creating an empty replacement directory.

Proposed per-experiment host layout (nothing created in 1D.1):

```text
paper_runs/<experiment-id>/       # mounted at /outputs
  config.original.yaml
  config.resolved.yaml
  manifest.json
  checkpoints/
  logs/
  cache/
  meshes/
  materials/
  renders/
  evaluations/<evaluation-id>/
```

Use unique experiment IDs containing a descriptive scene/stage/variant/seed
label plus an unambiguous uniqueness component. Validate path-safe IDs and create
run directories exclusively; timestamps or scene names alone are insufficient.
Reject reuse unless explicit resume is requested. Resume must name the run and
checkpoint and validate configuration, stage, inputs, architecture, and progress.
Historical checkpoints are input artifacts, never implicit writable resumes.

## 1D.1: Baseline preservation and documentation

Deliver root `AGENTS.md`, the three documents in `docs/`, and necessary ignore
rules. Record exact branch tips, scene names, artifact coverage, known report
sources, and uncertainties. Use lightweight metadata; do not hash entire large
datasets or deserialize checkpoints. Preserve every original artifact and all
historical refs. Verify the diff contains no scientific source/config changes.

The inventory provides preservation documentation, not a new backup, content
hash manifest, verified software environment, or reproduced baseline.

## 1D.2: Configuration, path, and storage refactor

This milestone is split into 1D.2A (standalone foundation) and 1D.2B (later
integration). The full proposal below remains the target for the combined work;
it does not describe capabilities already connected to the research pipeline.

### 1D.2A implemented boundary

`infrastructure/config.py`, `paths.py`, and `runs.py` provide a versioned envelope
around existing YAML, diagnostics with strict validation, explicit runtime roots,
contained input/output selections, exclusive run allocation, and explicit
checkpoint selection. New examples and CPU-only tests exercise these APIs.
No original YAML or scientific entry point uses them yet. See
[CONFIGURATION_AND_PATHS.md](CONFIGURATION_AND_PATHS.md) for conventions,
preserved behavior, compatibility limits, and the proposed 1D.2B integration.

In 1D.2A, path destinations existed only as an API. Resume still validates
allocation identity and checkpoint location/existence,
not scientific state compatibility. Configuration defaults are supplied explicitly
by their owning code; this foundation does not introduce a new numerical preset.

### 1D.2B-1 implemented boundary

Dataset constructors accept explicit preprocessing storage; a new configuration
factory passes it without rewriting legacy numerical settings. Camera pickles,
processed images/crop metadata, and the dataset NeRF point export route to a
versioned experiment cache. Existing source-side products are preserved read-only.
Schema-1 references are confined to an explicitly approved configuration root.
Existing training/evaluation entry points still select legacy execution.

See [DATASET_CACHE_ISOLATION.md](DATASET_CACHE_ISOLATION.md) for the writer
inventory, metadata identity policy, fixture/native test distinction, retained
custom-loader defect, and proposed 1D.2B-2 scope. Full pipeline write isolation
and native-backend numerical parity are not established by this boundary.

### 1D.2B-2A implemented boundary

`run_experiment.py` is the separate geometry-only schema-1 entry point.
Side-effect-free preflight checks paths, required structure, allocation intent,
and resume metadata before scientific imports. Explicit execution threads one
dataset selection and preprocessing cache through training, validation, and the
shape renderer. Checkpoints, optimizer state, text/TensorBoard/console logs,
validation images, and conventional runtime caches/temporary files use the
selected experiment. Existing `run_training.py --cfg ...` remains legacy.

Original envelope/reference bytes, path-resolved settings, partial owning-class
defaults, and a conservative compatibility identity are saved. New runs reject
collisions. Resume requires an explicit same-run checkpoint and matching
config/input metadata/code/dependency identity; historical initialization loads
weights only into a new run. RNG/sampler state is not added or restored, so no
exact-resume claim is made. Full provenance remains 1D.4.

CPU fakes exercise the shared loop/evaluator/logger/metrics; no native training
was executed. At this boundary, shape construction retained the two-versus-three
split-unpack error and the trainer retained the final NVS signature mismatch.
The approved follow-up below adapts isolated mode only. Native execution and GPU
parity remain unvalidated. See [GEOMETRY_TRAINING.md](GEOMETRY_TRAINING.md) for usage,
writer inventory, evidence, limitations, and proposed material scope.

### 1D.2B-2A.1 implemented boundary

The isolated trainer explicitly selects three-list split adaptation in the shape
renderer. It retains the helper's exact training, validation and novel-view IDs,
calls the unchanged helper once, and preserves RNG effects. Final geometry NVS
dispatch is deferred and recorded in run metadata/console output until a
separately versioned evaluation protocol is reviewed. Legacy entry-point behavior,
validation/checkpoint selection, and all scientific computations are preserved.
Focused CPU initialization/split/loop tests complement the full infrastructure
suite; no real run, native training, Docker or GPU experiment is executed.

### 1D.2B-2B implemented boundary

`run_material_experiment.py` adds schema-1 material preflight/explicit execution
without changing the legacy or geometry CLIs. Explicit runtime input roots select
historical or prior-experiment meshes; output and input roots must be disjoint.
Mesh contents are hashed/read only, and only the runtime mesh/dataset path fields
are replaced. Training/validation/rendering share the preprocessing cache. Material
checkpoints, logs, TensorBoard, validation/NVS images and metrics, optional debug
point clouds, runtime caches and temporaries use the owned run. Existing output
aliases are rejected before opening logs. Material NVS calculations/splits and
shader configuration scope are preserved; geometry NVS remains deferred.

Same-run continuation validates identity and explicit checkpoint selection;
historical initialization loads compatible weights only. Untagged historical
weights require a reviewed hash-bound baseline-origin declaration, because shape
checks cannot distinguish mathematically different same-shaped branch variants.
Original settings, actual renderer/shader configurations and mesh identity are
recorded separately. Full provenance and RNG/sampler restoration remain 1D.4.
CPU fakes and source comparisons establish I/O/control-flow preservation only.
See [MATERIAL_TRAINING.md](MATERIAL_TRAINING.md) for usage, writers, tests, limits
and proposed environment/native checks. No Docker/native execution is performed.

### Proposed implementation

The following list is the broader target; it does not override the implemented
boundaries or authorize subsequent milestones.

1. Introduce a versioned configuration schema and explicit path resolver.
   Separate geometry/material stage, dataset identity/root, cache root, mesh
   input, checkpoint input, and output directory. Resolve roots independently of
   the process working directory. Store original and resolved configs.
2. Describe legacy A/G/B presets using exact source snapshots and effective
   settings. Preserve A's existing numerical implementation first. G/B require
   separate reviewed implementation/parity work before claiming executable
   support in one maintained codebase; do not merge them wholesale or advertise
   a configuration flag whose corresponding behavior is unavailable.
3. Expose optimizer/scheduler/loss/sampling/seed fields with their legacy values
   and semantics. Preserve A's optimizer arguments, gradient clip, fixed/clamped
   cosine horizon, rotation representation, thresholds, loss formulas, and
   random-number behavior. Do not reinterpret a configured horizon as a fix.
4. Specify default resolution, nested merge semantics, supported combinations,
   and unknown-key/type checks. A legacy adapter must record historical inputs
   without silently normalizing string flags or discarding unused settings.
   A strict new schema should reject placeholders and invalid types with an
   actionable error. Any corrected interpretation needs a separate version and
   authorization. Moving a path is distinct from changing a scientific setting.
5. Redirect every generated file: dataset caches, resized/cropped images,
   evaluation point clouds, COLMAP working products when applicable, checkpoints,
   TensorBoard/text logs, NVS renders, meshes, materials, and relighting outputs.
   Keep source images/cameras/reference geometry read-only.
6. Key derived caches by input identity, preprocessing parameters, and version.
   Preserve preprocessing values, order, and RNG consumption; never regenerate
   into originals. Existing caches are historical inputs only when explicitly
   selected and their provenance limitations are recorded.
7. Add unique-ID output allocation and explicit resume selection. Keep restart
   semantics distinct from warm-starting a new run with inherited weights.
   Cross-run geometry references must record mesh coordinate conventions and
   source run/checkpoint when known; hashing of selected artifacts follows the
   provenance implementation in 1D.4.

### Tests and acceptance criteria

- Focused configuration fixtures verify schema versions, resolved values,
  defaults/overrides, invalid types, missing inputs, and unsupported combinations.
  Snapshot the numerical settings separately from relocated paths.
- Characterize legacy optimizer arguments and learning-rate values at step 0,
  warm-up boundaries, 99,999, 100,000, and configured/actual endpoints before
  adapting their plumbing. Test clipping and loss/sampling setting preservation.
  No unapproved formula or schedule correction is accepted as infrastructure work.
- Fixed input IDs and seeds reproduce legacy split membership and required RNG
  effects. Using a new local RNG must not silently change the legacy trajectory.
- Temporary fixtures verify path containment, cache/output isolation, missing
  path errors, collision refusal, and explicit resume validation. Spy/mock input
  writes without changing host permissions; real read-only mount tests belong
  to 1D.3. Cache relocation must preserve preprocessing results.
- Static checks find no remaining uncontrolled writers under input roots across
  training, extraction, evaluation, and optional utility entry points in scope.
- Source review confirms no geometry/material formula changes, historic-branch
  merges, original-artifact modifications, or unrelated dependency upgrades.
  GPU-dependent parity checks remain explicitly pending if execution is unavailable.

Deliver a small reviewed diff with configuration migration examples and a list
of supported versus deferred legacy behaviors. Stop for review before 1D.3.

## 1D.3: Docker and development environment

### 1D.3A inventory boundary

Read-only source/history/dependency inspection, filtered Docker/GPU queries and
Codex-visible metadata are recorded in
[GPU_ENVIRONMENT_INVENTORY.md](GPU_ENVIRONMENT_INVENTORY.md). Docker inspection
still fails at the socket and NVIDIA-SMI cannot reach the driver in Codex; these
are not proof of host failure. No security bypass or environment modification is
attempted. The working `nerofactor` image / `intelligent_keldysh` container remain
user-reported, with exact versions/digests/revisions pending recovery using the
[regular-terminal diagnostic sheet](GPU_ENVIRONMENT_DIAGNOSTICS.md).

No Dockerfile/build recipe or CUDA extension sources are available in the checkout
or reachable ref history; only two current requirements pins are present. Codex
package files show Torch 2.4.0+cu121 while nvcc reports 13.3, which is not evidence
for a compatible working-container stack. Prefer recovering/pinning the actual
reference tuple over selecting versions from that mixed context or floating
upstream HEAD. Required pending pins explicitly block a reproducible build claim.

### 1D.3B-1 definition boundary

The researcher supplied the recovered Python 3.8.10 / Torch 1.8.1+cu111 / CUDA
11.1 / ABI 0 environment, native source commits/setup hashes and reference binary
identity. These supersede the previously unknown reference tuple; Codex's separate
Python 3.9/Torch 2.4/CUDA 13.3 metadata is not the reconstruction target.
[docker/paper/README.md](../docker/paper/README.md) provides the proposed Dockerfile,
runtime layouts, source staging procedure, incomplete dependency manifest,
provenance definitions, historical differences and remaining build gates.

The definition requires a real base digest and complete reviewed offline package
artifacts rather than guessing missing pins. It removes intermediate repositories,
retains the exact native sources and nvdiffrast lazy compilation, and isolates
read-only inputs from non-root persistent outputs. No source staging, package
installation, build, container execution or native computation is performed.
Python 3.8 source grammar/static policy checks pass; actual 3.8 runtime remains
unverified. In 1D.3B-1, existing tests' `ast.unparse` calls were recorded as a
blocker. 1D.3B-2A authorizes replacing only those test branch selectors with
equivalent AST comparisons; no interpreter upgrade or scientific patch is applied.

Review correction 1D.3B-1A includes every validator dependency in the generated
definition inventory. Before any Debian installation, a mandatory host-side
Python preflight validates the full reviewed inputs/workspace and definition
bytes. Its externally supplied receipt SHA-256 gates an offline shell-only
verification inside the historical base; no base Python is assumed. The Python
recheck after installation remains defence in depth. Static synthetic regression
fixtures exercise this boundary; builds/installations remain separately approved.

### 1D.3B-2A prerequisite boundary

[Build prerequisites](../docker/paper/PREREQUISITES.md) records filtered runtime
recovery, exact-commit offline source staging with licence/file-hash provenance,
the source-established training dependency roots and missing closure evidence.
Only local nvdiffrast source identity/export can be independently verified here;
Docker socket denial leaves reference-container pins/base/raytracing recovery
pending. Publisher-listed wheel candidates are not acquired/installed artifacts.
Run static and CPU infrastructure fixtures, retaining native skips separately.
First builds, package acquisition/installation, GPU/native validation and actual
experiments remain separate approvals; no environment redesign is included.

### 1D.3B-2B recovered-dependency boundary

[Recovered dependency pins](../docker/paper/RUNTIME_DEPENDENCIES.md) records the
supplied inventory's hash and independent 102-distribution/142-edge consistency
checks, all installed pins, 21 Debian role versions and GCC/G++/Eigen/toolchain
evidence. Installed metadata is separate from publisher checksum candidates and
locally byte-verified artifacts. Five failed Debian role queries and versionless
`ninja-build` are unresolved, not guessed. The supplied CUDA manifest digest
remains registry-unverified and is not established as the historical base.
Preserve the accepted Docker/preflight architecture and unreviewed resolution;
only document exact pins, remaining artifact/source gates and a small separately
approved acquisition plan. Static/CPU checks do not establish native/GPU parity.

### Proposed 1D.3B-2 implementation

The proposal below requires a new approval; none of its build/install/native
execution steps is performed by 1D.3A or 1D.3B-1.

1. Recover a known working historical environment when accessible. Identify
   Python, PyTorch/CUDA, driver requirements, compiler/toolchain, GPU architecture,
   and extension ABI compatibility before selecting pinned versions. Do not
   assume the current host environment produced the thesis results.
2. Add a GPU-capable Docker build, dependency locks, pinned base-image digest,
   pinned `nvdiffrast` and raytracing source revisions, and reproducible extension
   build instructions. Include missing direct runtime dependencies. Separate
   optional Blender/COLMAP tooling from the core training environment as needed.
3. Add runtime definitions using the exact mount contract above. Run under the
   intended host UID/GID, select the GPU explicitly, and route caches/build
   temporaries to declared writable locations. Persist outputs independently of
   container lifecycle; exclude datasets, outputs, Git metadata, and secrets from
   image build context.
4. Document commands for preflight, build, execution, and failure diagnosis.
   Access failures are reported, not bypassed. Never change socket permissions,
   group membership, drivers, or security controls as a workaround.

### Tests and acceptance criteria

- Parse/validate runtime definitions and verify image context exclusions before
  container execution. Dependencies and extension revisions must not float.
- When authorized and accessible, build a clean image and import PyTorch,
  `nvdiffrast`, and `_raytracing`; run minimal GPU tensor/rasterizer/raytracer
  checks without starting research training.
- Verify container-visible GPU identity and compatibility, record actual
  versions/digests, and distinguish build success from GPU-runtime success.
- Inspect read-only input mounts and test rejected writes with disposable
  read-only fixtures, never destructive write probes against original data.
  Verify a disposable output persists across container removal and has the
  intended ownership without changing host permissions.
- Demonstrate that one run cannot write another run's storage and no Docker
  socket or privileged security workaround is required by the runtime definition.
- If Docker/GPU access is unavailable, mark execution checks blocked and retain
  static evidence. Do not declare GPU compatibility validated.

Deliver environment definitions and recorded validation limits. Stop for review
before 1D.4. No exact CUDA/PyTorch combination is promised by the 1D.1 inventory.

## 1D.4: Experiment management and validation

The [1D.3C smoke protocol](NATIVE_GPU_SMOKE.md) is now implemented separately
from training: approved configuration/path/cache/run infrastructure, OS read-only
input checks, actual renderer initialization and one configured training-mode
forward at step 0, without optimizer/backward/checkpoint/evaluation dispatch.
Its native backend was not executed by Codex in the implementation milestone.
The researcher now reports both model smoke tests and all 109 Python 3.8 tests
passed. Raw launch reports/immutable identities remain unarchived here; numerical
checkpoint parity and thesis reproduction remain unmeasured. Disposable cache/
logs and phase/failure reports must precede any later approved launch. No image
rebuild or scientific behavior change is included, and full provenance remains 1D.4.

### 1D.4A preparation boundary

[BASELINE_PREPARATION.md](BASELINE_PREPARATION.md) and its
[metadata snapshot](evidence/1d4a-metadata.json) record read-only inventory and
rank bell geometry first, with bear material blocked by absent weights.
[Checkpoint-inference protocol v1](protocols/CHECKPOINT_INFERENCE_V1.md) proposes
one validation view, two routing/reference arms, immutable identities, split and
preprocessing preservation, existing full-image PSNR/SSIM, provisional tolerances
and repeatability gates. It is not an evaluator implementation or launch approval.
Trust/origin, selected artifact hashes, exact target-interpreter splits, reference
outputs and archived native reports remain prerequisites. Historical scores are
preserved, not recomputed or used as invented expected outputs. No checkpoint or
input pickle is deserialized in 1D.4A, and no GPU/Docker operation is performed.

### Proposed implementation beyond 1D.4A (requires separate approval)

1. Extend run allocation into a manifest/state lifecycle: created, running,
   completed, failed, interrupted, and explicitly resumed. Persist commands,
   resolved configs, source commit/diff, versions, timestamps, exit codes, input
   identities, and any missing provenance fields. Capture all run outcomes.
2. Hash selected meshes/checkpoints, config/split files, and environment
   definitions; define the dataset content-manifest coverage and cost. Store
   manifests outside input roots. Authorize large dataset hashing separately
   rather than implying metadata counts are content checksums.
3. Log raw and weighted losses, effective coefficients, learning rates, selected
   validation metric/checkpoint, gradient information needed for diagnosis, and
   evaluation outputs in machine-readable form alongside human-readable logs.
   Checkpoint writes must be atomic and preserve previous valid state on failure.
4. Save/restore applicable Python, NumPy, CPU/CUDA RNG and sampler state with
   training progress and optimizer state. Record worker-seeding and deterministic
   settings. Legacy checkpoints missing state must be labeled non-exact resumes;
   do not fabricate their unavailable random history.
5. Version evaluation protocols, including split IDs, masks, units, coordinate
   transforms, color space, image resolution, point sampling, metric reduction,
   and aggregation. Separate historical reports, newly recomputed historical
   artifacts, and newly trained results. Use unique evaluation IDs and refuse
   overwrites of previous results.
6. Prepare a concrete validation-run specification identifying scene, stage,
   legacy preset, input artifact, seed, step budget, GPU/resource limits,
   evaluation protocol, comparison target, and tolerances. Execute training or
   expensive evaluation only when authorized in that milestone's task scope.

### Tests and acceptance criteria

- Verify manifest completeness, unique-ID/collision behavior, failure recording,
  atomic checkpoint recovery, explicit resume compatibility checks, and selected
  artifact hashes using small fixtures.
- Demonstrate stable split IDs and compare uninterrupted versus resumed state in
  a controlled small test. Document tolerance and nondeterministic kernels;
  seeded execution alone does not justify bitwise-reproducibility claims.
- Checkpoint inference and approved bounded validation runs compare against a
  specified legacy reference. Missing material checkpoints or incompatible
  heads must block the relevant claim, not trigger an invented comparison.
- Metric fixtures verify exactly the declared protocol, including full-image
  masked versus foreground-only PSNR/SSIM and geometry coordinate conventions.
  Review evaluation repairs as behavior changes; preserve old protocol labels.
- A run's outputs are sufficient to identify inputs, code, settings, environment,
  and outcome without consulting shell history. No original/historical artifact
  is modified and no failed or excluded trial is silently discarded.
- Publish separate statuses for static tests, CPU tests, GPU extension tests,
  inference parity, and training validation. Pending or blocked work stays visible.

## Deferred scientific decisions and access limitations

Step 2 mathematical review must resolve angular encoding, distribution/PDF
consistency, tangent semantics, material parameterization, geometry coordinate
conventions, effective losses, split definitions, and evaluator defects before
introducing corrected methods. Infrastructure approval is not approval to fix
these findings or reinterpret historical claims.

Docker previously failed inside Codex; no Docker access was probed in 1D.1 and
no permissions were changed. A preceding NVIDIA-SMI query could not communicate
with the driver. Those Codex access failures are historical, separate from the
researcher's newer reported native successes. No new access attempt or permission change is made in
1D.4A. Archive actual launch/image/GPU identities before claiming independently
verified runtime evidence; source inspection alone is not execution evidence.
