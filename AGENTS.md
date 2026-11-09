# Research and development rules

## Purpose and methodology

This repository develops NeRO-based geometry reconstruction and anisotropic
material estimation from calibrated multiview images. Geometry training learns
an implicit surface; mesh extraction supplies geometry to material training.
Rendering, novel-view synthesis (NVS), and geometry evaluation assess different
parts of the method and must retain their own protocols and provenance.

The research workflow is: state a hypothesis, identify a reference implementation
and its exact configuration, define controlled changes and evaluation splits,
record all runs, and distinguish historical reports from independent evidence.
An infrastructure refactor must preserve the computation being studied.

## Workspace and repository conventions

The host is `marineris`. The maintained development branch is
`paper/infrastructure`, initialized at the `anisotropic` baseline.

**Authoritative scientific selection (1D.5-RESET):** Stage I geometry uses
`aniso_geometry_safe` at `79252178e65914ad1acbc14790b1514f276bfea0`; Stage II
material uses `anisotropic` at `e1e859779301faced46d6476691e20b546c27ae0`.
Infrastructure ancestry is not scientific-stage authority. Use the exact separate
snapshots and fresh-process namespace selection, never an A geometry/G material
substitution. This researcher decision does not identify old producing revisions.

| Location | Role |
| --- | --- |
| `/home/isavand/paper/Nerofactor-paper` | Versioned code, configurations, documentation, and small reviewed fixtures |
| `/home/isavand/paper/data` | Original input datasets; preserve without modification |
| `/home/isavand/paper/final_results` | Historical thesis results, meshes, checkpoints, logs, and comparison artifacts; preserve without modification |
| `/home/isavand/paper/paper_runs` | Future experiment outputs; not yet created in milestone 1D.1 |

Keep code separate from datasets and generated storage. Do not introduce a
writable repository `data` symlink that combines original inputs with outputs.
Proposed container paths are `/inputs/datasets` and `/inputs/thesis` read-only,
and `/outputs` for the selected experiment's writable storage. These mounts are
planned, not implemented by this document.

Use existing Python/PyTorch/NumPy style, `snake_case` names, and focused diffs.
Avoid unrelated formatting, dependency upgrades, or configuration rewrites.
Keep host-specific settings and secrets outside committed files. Ignore rules
reduce accidental Git additions; they are not filesystem protection.

## Historical branches

The inspected baseline refs and exact hashes are recorded in
[BASELINE_INVENTORY.md](docs/BASELINE_INVENTORY.md).

| Branch | Historical role and limitation |
| --- | --- |
| `anisotropic` | Starting implementation for infrastructure development; contains ablation-labeled settings whose effective scope needs verification and is not automatically a canonical full-method preset |
| `aniso_geometry_safe` | Alternative material parameterization, geometry/tangent experiments, optimization settings, and dataset/evaluation interfaces; the name does not establish mathematical correctness |
| `ablation_anisotropic` | Single-angle ablation plus other configuration and numerical differences; not automatically a controlled one-variable comparison |

Only `anisotropic` has a local historical branch at inspection; the other two
are available as `origin/...` remote-tracking refs. Inspect historical files with
`git show` and compare them without changing branches or merging. Do not treat
any pair of these implementations, checkpoints, or reported metrics as
equivalent. See [BRANCH_COMPARISON.md](docs/BRANCH_COMPARISON.md).

## Scientific integrity

- Never invent measurements, claim an unexecuted validation passed, or label an
  old log as an independently reproduced result.
- Identify historical reported results, newly measured results, and hypotheses
  separately. Preserve original precision, source paths, and uncertainty.
- Record failed, interrupted, and excluded runs and the reasons for exclusions.
  Do not select seeds, views, checkpoints, masks, or metrics after inspecting
  test outcomes to improve a reported comparison.
- Separate validation-based checkpoint selection from final reporting. Record
  exact image IDs, masking, color space, resolution, coordinate frames, scale,
  point sampling, metric definitions, and aggregation rules.
- Do not infer training steps from mesh filenames or producing commits from
  directory names. Mark missing provenance as unknown.
- Treat cache generation, preprocessing, random-number consumption, numerical
  tolerances, and dependency versions as potentially scientifically relevant.

## Prohibited destructive operations

- Do not move, delete, overwrite, regenerate in place, or change permissions on
  original datasets or anything in `final_results`.
- Do not run cleanup commands against research storage, force-add large or
  ignored artifacts, rewrite historical Git history, or merge historical
  branches as part of infrastructure cleanup.
- Do not overwrite another experiment, implicitly resume by scene name, or write
  generated caches into an input dataset.
- Do not start training, expensive evaluation, or Docker containers unless the
  current task explicitly authorizes that execution.
- Do not change Docker socket permissions, group membership, system permissions,
  drivers, or security settings, or bypass restricted access. Report limitations.
- Do not expose credentials in code, configuration, logs, documentation, or tool
  output. Inspect only the task-relevant environment and files.

## Changes to numerical behavior

Infrastructure work must not silently change algorithms, optimizer defaults,
schedules, losses, coefficients, sampling, formulas, tensor precision, epsilon
values, clipping, or seeds. Preserve the selected legacy behavior before
introducing alternatives. A suspicious implementation is evidence to document,
not permission to fix it during cleanup.

For a proposed numerical change, identify the exact legacy commit and execution
path, explain the mathematical or experimental rationale, isolate the change
from storage refactoring, obtain authorization within the task scope, and supply
focused regression evidence with stated tolerances. Give corrected behavior a
distinct configuration/protocol version. Do not retrofit corrected measurements
into historical result tables or silently convert old checkpoints.

## Reproducibility requirements for future experiments

Each experiment must have a unique ID and separate persistent output directory.
Resuming must explicitly name the experiment and checkpoint and verify compatible
configuration and inputs. A material run must identify its exact geometry mesh,
source run/checkpoint when known, coordinate convention, and content hash.

Record Git commit and any diff, original and resolved configuration, command,
dataset identity and split IDs, preprocessing/cache parameters, seed policy,
optimizer parameters, scheduler semantics, all loss weights, sampling settings,
environment/dependency versions, image digest when applicable, GPU/driver details,
start/end times, exit status, and evaluation protocol. Future checkpoints should
include training progress, optimizer state, and relevant RNG/sampler state.
Distinguish seeded execution from demonstrated deterministic execution.

Milestone 1D.1 uses lightweight names/counts/sizes and small-file inspection.
Do not hash entire large datasets here. Later provenance work must hash selected
meshes/checkpoints and define a dataset content-manifest policy; record the scope
and any missing hashes rather than implying full integrity verification.

## Review, testing, and milestone boundaries

For every change, report its purpose, changed files, evidence, limitations, and
whether numerical behavior could change. Verify branch and working-tree state
before editing, preserve user changes, and keep reviews small enough to assess.

For documentation-only work, validate source/ref/path claims, Markdown links,
whitespace, ignore behavior, and the exact changed-file scope. Do not run training
to validate documentation. For later implementation, add focused tests for
configuration semantics, path containment, write isolation, split IDs,
optimizer/scheduler parity, and explicit resume. Report CPU checks separately
from GPU tests; blocked GPU checks are not passes. Do not import training entry
points merely to inspect metadata, because existing loaders can write inputs.

Current authorized task: **bounded native A anisotropic material smoke
preparation**, reviewed from `71257544d9038929197d45b3dafb2434ed2d731f`.
Read [MATERIAL_TRAINING_SMOKE.md](docs/MATERIAL_TRAINING_SMOKE.md).
Prepare exactly two original A optimizer returns using `TwoUpdateTrace`; preserve
the pinned A scientific snapshot and original bear YAML, including effective
shader regularization scopes, warm-up, clipping, sampling and all mathematics.
The explicitly selected/hash-pinned historical bear mesh is approved only for
this isolated diagnostic. Ordinary research runs still require an attributable
G-to-A mesh handoff. No checkpoint initialization, resume, validation, saving,
final NVS, material extraction or full training is authorized by this task.
Record actual anisotropic/RGB/regularizer calls and per-group gradients, effective
settings, Adam state, sampling and source identities with the existing helpers.
CPU toy autograd/explicit stand-ins and static checks only from Codex. Native
Docker/GPU execution is researcher-side after patch review; no scientific edits,
original input modifications, installations, image/permission changes, commit
or push. Short diagnostic success is not publication/full-training acceptance.

Earlier authorized milestone: **1D.6A, G geometry validation and checkpoint
compatibility**, reviewed from `596c070664ffd9cfff50805939026cff9ce21883`.
Read [G lifecycle audit and commands](docs/G_GEOMETRY_LIFECYCLE.md) and
[G training acceptance](docs/evidence/G_GEOMETRY_TRAINING_ACCEPTANCE.md).
The verified native G R/R-repeat/C evidence admits only the fresh two-update
prefix. Use evidence policy v2; preserve v1 and all original reports/protocols.

Prepare independent hash/source-gated G compatibility, optional historical
optimizer-load diagnostic (no update), original G camera-0 validation, and
manual G-produced save/load roundtrip protocols. Use only the exact G snapshot;
no scientific method replacement or A scientific import. Historical pickle
loading remains explicitly trusted and researcher-side only in an isolated
runtime. Strict network keys/shapes/dtypes and loading must precede validation.
An optimizer library load can override G settings and is not G resume admission.
Manual serialization after two G updates is not the natural 1,000-step schedule.
Extraction is source audit/design only; original `model.pth` selection cannot
be silently replaced by `model_best.pth`. Random/two-update diagnostic geometry
must never be accepted as research geometry or a material handoff.

Codex may inspect read-only bytes/metadata, implement isolated adapters and CPU
toy/explicit stand-in tests, and run CPU/static checks. No native model,
historical checkpoint/camera deserialization, GPU/Docker execution, extraction,
1,000-step/full training, scientific edits, installs, environment/permission
changes, commit or push. Stop for review before researcher native execution.
Earlier source-preservation and scientific-integrity rules continue to apply.

Earlier authorized milestone: **1D.5-RESET, authoritative dual scientific
baselines**, reviewed from `bed325fe511ec52a61f5793d0ee0acd91dad26b9`.
Read [DUAL_SCIENTIFIC_BASELINES.md](docs/DUAL_SCIENTIFIC_BASELINES.md), the pinned
[source manifest](scientific_sources/manifest-v1.json),
[evidence supersession v1](docs/evidence/dual-baseline-supersession-v1.json) and
[G two-update protocol](docs/protocols/GEOMETRY_G_TWO_UPDATE_V1.md).

Authorize source audit, exact immutable G/A Git-blob copies, stage-specific
source/import selection, filesystem-only I/O adaptation, metadata/provenance,
CPU fixtures and static checks. Root legacy scientific code/configuration,
original branch histories, artifacts/reports/checkpoints/meshes/logs and existing
A protocols remain unchanged. Source snapshots must match every pinned byte;
never fix suspicious science inside them. Any required scientific change is a
blocker for separate review and a newly versioned scientific source/protocol.

Selected execution uses `run_stage_experiment.py`: G's actual geometry trainer,
renderer/dependencies, two-way unshuffled splits, original losses, Adam defaults,
configured cosine schedule and **no clipping**; A's actual anisotropic material
implementation, defaults/regularizer scopes, clipping, optimizer/scheduler and
validation/NVS behavior. One scientific namespace per cold process; reject
ambiguous sources, module fallbacks, custom overrides or unadmitted checkpoint
initialization/resume before native execution. Record infrastructure and
scientific commits/hashes separately. The old experiment CLIs reject --execute;
legacy A diagnostic code is historical-only, not selected G acceptance.

Preserve and supersede prior evidence by versioned records, not rewriting it.
A two-update R is invalid for G acceptance; prepared A R-repeat/C remain
unexecuted/superseded. Exact camera-0 PSNR and montage identity remain valid A
evaluation observations. The original unshuffled training split is strongly
supported but unproven producing-run attribution, and the historical checkpoint
is not automatically G-produced or eligible for G resume. Future G-to-A meshes
require explicit read-only file/hash, producing G run/source evidence and
coordinate/scale/normal/orientation/preprocessing conventions; no substitution.

Stop after implementation/CPU/static checks for review. New G R, independent
R-repeat and C must be executed afresh by the researcher under the frozen G
protocol, reviewed for compatibility/repeatability before research progression.
No Codex Docker/GPU/native model/inference/training, original camera/checkpoint
deserialization, installs, builds, permission/environment changes or automatic
commit/push. CPU toy autograd/Adam and original source bodies with explicit
stand-ins are allowed; they are not native scientific acceptance. Earlier A
preservation rules below describe historical paths and do not override the
new selected G geometry authority.

Earlier authorized milestone: **1D.4B-3D, finalize historical bell camera-0
evaluation evidence**. Inspect the existing primary/control/comparison reports
and their recorded hashes without regeneration. Write documentation and a small
versioned success manifest, preserve original evidence and all image-58 code/
reports, and run CPU/static validation only. No scientific source/configuration,
checkpoint/dataset, Docker/image/environment/permission changes or native GPU/
training execution; no automatic commit or push.

The exact camera-0 PSNR, five-decimal SSIM match and step-pair equality may be
recorded as verified within the inspected files and tested implementation. Camera
selection accounts for the observed metric discrepancy in that scope; it does
not establish producing-run training membership. The unshuffled two-way training
split remains a strongly supported hypothesis. Current A validation 58 must not
be labeled the historical geometry evaluation split. Preserve native versus host
JPEG-decoder measurements separately and never fit a tolerance or rewrite an
original report. See [camera-0 success](docs/evidence/BELL_CAMERA0_DIAGNOSTIC_SUCCESS.md)
and the updated [attribution report](docs/evidence/BELL_EVALUATION_PROVENANCE.md).

Earlier authorized milestone: **1D.4B-3C, historical bell camera-0 diagnostic
preparation**, on the existing `paper/infrastructure` worktree. Preserve the
uncommitted 1D.5A implementation/protocol and all original image-58 code/evidence;
shared AGENTS/static-check changes may add this scope only. No commit/push.

Prepare a separate explicit source-camera "0" adapter using the pinned admitted
bell checkpoint, original maintained A renderer/mathematics and existing strict
hash/state checks. Keep A train/test/NVS tables unchanged; do not substitute G or
infer source ID from validation iterator index. An external selection facade
may supply camera-0 information to original forward/test_step methods without
editing renderer internals or weights. Preserve preprocessing, resolution,
backgrounds, sampling, SDF gradients, conversion and metric definitions. Compare
fresh evaluation steps 149999 and 150000 with stored checkpoint step still 150000.

Record the researcher's visual historical-0/current-58 identification as supplied
evidence. The unshuffled two-way producing split and explanation of the metric
gap remain hypotheses. Freeze the [camera-0 protocol](docs/protocols/BELL_CAMERA0_DIAGNOSTIC_V1.md):
reuse unchanged inference RGB bounds for the step pair; historical score/JPEG
comparisons are descriptive with no invented acceptance threshold, resizing,
alignment or compression compensation. Closeness is never producing-run proof.
See [BELL_CAMERA0_DIAGNOSTIC.md](docs/BELL_CAMERA0_DIAGNOSTIC.md).

Codex may prepare the isolated runner, CPU fixtures/static checks and documentation
only. **No historical checkpoint/camera-pickle deserialization, native model
construction, Docker/GPU execution, native inference/training, optimization,
package/image/permission changes or original-artifact modification from Codex.**
Researcher commands must use explicit trust/runtime gates, existing admitted image,
read-only source/datasets/thesis, no network, one selected GPU, non-root/fresh
outputs and resource limits. Stop for review before native execution. Document
any scientific-semantic blocker rather than repairing it or fitting the result.

Earlier preparation retained for review: **1D.5A, bounded bell geometry training smoke
implementation and CPU/static validation for review**, based on
`ef7e6b270137c1315c1ef01f303af910ee1dcbaf`. Read the admitted inference evidence,
historical attribution audit and existing geometry/training implementation first.
Implement only separate researcher-side entry points, a scoped observer/stop,
evidence/comparison helpers, disposable CPU fixtures and documentation. Keep the
actual Trainer.run body, renderer, datasets/loaders, losses, optimizer/scheduler,
scientific source and legacy configurations unchanged. No loop replacement,
operator simplification, sample reduction or numerical repair is authorized.

The fixed budget is exactly **two successful Adam returns at steps 0 and 1**.
Preserve full total_step=300000, intervals, seed 6033, actual A splits/loader/RNG
order, clipping 0.7 and native Adam construction. Step 0 LR is zero: moment/counter
state advances without parameter changes; step 1 LR is about 1e-7 and parameters
must change. Do not add a third update or modify warm-up to get two nonzero-LR
updates. The proposed external CPython observer must stop immediately after
Adam return 2, reject incompatible code/traces, retain failures and close owned
workers/writer/progress. If preserving these operations is impossible, document
the blocker and stop; never silently modify the scientific loop.

Use independent legacy R/R-repeat and schema-1 C configuration routes, fresh
unique run directories and no checkpoint/optimizer restoration. Prepare read-only
source/dataset mounts and an **empty read-only thesis placeholder**, so historical
weights/results are not exposed. Record finite loss/gradient/parameter/Adam-state,
actual aggregate/clipping/LR, parameter changes, sampling/RNG/runtime identities,
source/config/input hashes and failure/cleanup evidence. No checkpoint/validation
is expected before the preserved natural 1000-step intervals. Do not force them.

Freeze the [two-update protocol](docs/protocols/GEOMETRY_TWO_UPDATE_V1.md) before
native execution. Require R-repeat repeatability before C interpretation; keep
strict tensor/state hashes and training-unit scalar bounds distinct from RGB
inference tolerances. Hash/norm agreement for this short instrumented prefix
cannot establish full training, historical thesis or material/BRDF equivalence.
See [GEOMETRY_TRAINING_SMOKE.md](docs/GEOMETRY_TRAINING_SMOKE.md) for the design,
report schema, expected commands, scientific limits and native validation gaps.

Codex may execute **CPU toy autograd/Adam fixtures and static/configuration
checks only**, with temporary inputs and explicit renderer/loader/I/O stand-ins.
Never construct the native scientific model, decode original camera/checkpoint
payloads or run native training from Codex. **No Docker/GPU commands, native
training/inference/rendering, package/image/environment/permission changes, mesh
extraction or historical artifact modifications.** Prepared researcher commands
are for code review, not execution approval. Preserve earlier real inference and
blocked/provenance evidence; do not invent measurements or solve producing-run
uncertainty through this smoke. Stop after implementation/CPU/static checks;
do not commit or push automatically or start full/material training.

The researcher approved archiving the original small smoke reports/configuration
and manifest files at `/home/isavand/paper/evidence/1d3c` after the originally
requested `paper_runs/evidence/1d3c` destination was inaccessible. Verify the
approved destination is writable, preserve original bytes and source paths/
hashes, refuse overwrites, and exclude caches/generated datasets. Do not change
ownership/permissions or Docker settings. That earlier milestone additionally
authorized only the dual-baseline reset implementation and CPU/static work above;
earlier A evidence and preparations remain preserved under supersession policy.
No scientific code/YAML,
image/build, package/environment or original artifact changes are allowed.

Use **verified**, **inferred** and **unresolved** classifications in
[BELL_CHECKPOINT_ADMISSION.md](docs/evidence/BELL_CHECKPOINT_ADMISSION.md).
Distinguish file-backed smoke evidence, independently corroborated runtime
identity and researcher-reported 109/109 Python 3.8 test results. An image ID in
a report is caller-declared until independently checked; source-file hash matches
do not establish the launch's full Git commit or clean status. Report Docker
socket limitations separately without bypassing them.

Current A-based inference regression and historical G-based thesis reproduction
have separate admission decisions. Preserve the exact A three-way seeded split
and G two-way unshuffled split/RNG effects; verified present source-defined IDs
are not verified producing-run IDs. Checkpoint hashing, matching YAML, same-shaped
weights or smoke success cannot establish origin/compatibility/reproducibility.
No historical checkpoint loading is allowed from Codex in this implementation
milestone. Saved step/metric are now observed in the admitted JSON report, while
architecture compatibility is structurally verified against the real reference.
Producing commit/configuration/split/selection remain unresolved. Inference
adapters are implemented and the admitted researcher execution passed. The
separate two-update training preparation above is authorized; no scientific
changes or new native execution are authorized in this follow-up.
Do not commit or push automatically. The earlier
[baseline preparation](docs/BASELINE_PREPARATION.md) and
[inference protocol](docs/protocols/CHECKPOINT_INFERENCE_V1.md) remain proposals,
not execution approval.

Keep installed evidence, publisher checksums and byte-verified artifacts distinct.
Registry presence/tag agreement does not establish historical base identity;
do not mark an incomplete resolution reviewed or change the accepted Docker image.
Preserve the accepted Docker architecture, full pre-install validation gate,
dependency pins and historical runtime. Use filtered read-only diagnostics,
registry metadata queries, dependency/source assessment and CPU/static fixtures.
Preserve infrastructure test-harness behavior and every existing assertion; do
not introduce unrelated fixes or stage originals in this documentation milestone.
Never run setup to inspect it, copy a working tree wholesale, invent missing pins,
or promote publisher artifact metadata to verified downloaded/historical identity.
The earlier 1D.3C scope was the separate smoke CLI/helper, CPU tests,
the AST compatibility fix and supporting documentation; it is not extended here.
Use the recovered system Python 3.8.10 / CUDA 11.1 / Torch
1.8.1+cu111 / ABI 0 reference; no Micromamba or scientific upgrades. Pin nvdiffrast
and raytracing to the exact recovered revisions/setup hashes. Record unknown
package/system versions, hashes and base digest as unresolved; never invent them
or label an incomplete manifest a complete lock. Do not copy original checkouts
into Git or change their contents. CPU infrastructure fixtures must use temporary
datasets and explicit model/device fakes; never run native training or rendering.
Do not install packages, build images, start/stop containers, run GPU code or
download package artifacts. Read-only access may fail; do not bypass restrictions.
Preserve the existing image/container and all scientific/infrastructure behavior.
Do not change host configuration/permissions or bypass access controls. Separate
reference declarations, local observations, Codex metadata and proposals.
Inspect only task-relevant fields;
never publish credentials, private environment values, tokens or unfiltered
container/build configuration. If access fails, report it and supply filtered
read-only commands for the already-authorized regular terminal.

The following integration rules remain in force for later authorized work.
Geometry behavior must remain unchanged, including its split adaptation and
final-NVS deferral.
Preserve the original split helper, call count, train/validation/NVS IDs and RNG effects.
For geometry, record final NVS as deferred to a separately versioned evaluation
protocol; do not introduce cameras, renders, or metrics for that protocol. Keep the legacy
`run_training.py` path and legacy defaults, YAML, training-loop computations,
network definitions, evaluation protocols, losses, optimizers, scheduler,
sampling, seeds, precision, and arithmetic unchanged. Do not migrate legacy
configurations or repair scientific defects.

The new envelopes are accepted by separate `run_experiment.py` geometry and
`run_material_experiment.py` material entry points; existing entry points remain
legacy. `run_native_smoke.py` is a separate explicit preflight/one-forward harness;
see [NATIVE_GPU_SMOKE.md](docs/NATIVE_GPU_SMOKE.md) for mount gates and limits.
Preserve shallow legacy merges and the scope of shader fields; top-level
shader-like fields must not be silently forwarded into `shader_cfg`. Diagnostics
are evidence, not authorization to correct behavior. Run allocation and the
shared trainer loop are tested only in disposable CPU fixtures with explicit
model/device fakes; do not allocate a real experiment or execute native training
for this task. Configuration validation must not import scientific modules,
construct models, load tensor checkpoints, or consume scientific RNG state.
Fresh runs must refuse collisions; same-run continuation requires an explicitly
named checkpoint and matching identity. Historical initialization is explicitly
weights-only in a new run. Neither continuation nor old checkpoints establish
exact resumption without RNG/sampler state.
Dataset construction without `preprocessing` remains legacy compatibility mode.
New mode must write derived products only inside the selected experiment cache,
preserving existing source-side products as read-only inputs. Cache identities
must distinguish dataset, scene/mode/parameters, and implementation versions.
Schema-1 loads require an explicitly approved configuration root; references must
not escape it through traversal or symlinks. Record native-backend skips and
pre-existing custom-loader/evaluator defects instead of silently fixing them.
The split-unpack adaptation requires the explicit renderer keyword
`isolated_geometry=True`, supplied by the experiment trainer. Preprocessing-cache
opt-in alone must not select it. The legacy split-unpack and final NVS dispatch
behavior remains unchanged. Isolated execution records the NVS deferral in
`geometry.json` and the console log. See
[GEOMETRY_TRAINING.md](docs/GEOMETRY_TRAINING.md). CPU routing/interface tests do
not prove native geometry training is runnable or numerically equivalent on GPU.

Material runs must select `inputs.mesh` through an explicitly declared input
root. Keep mesh bytes and coordinate conventions unchanged; computed normals
remain in-memory legacy behavior. Never forward top-level regularization,
anisotropy, lobe, or other shader-like fields into `shader_cfg`. Record actual
renderer and shader settings separately. Material NVS retains its original split,
rendering, metric, image composition and append/overwrite behavior within the
explicitly selected run. Debug point export is routed when called, not enabled
as a new training option.

Before material checkpoint state loading, verify stage/algorithm declaration and
network keys/shapes/dtypes. Untagged historical initialization requires a reviewed
manifest under the approved config root binding checkpoint SHA-256 to an approved
source commit. Same tensor shapes do not establish branch/mathematical equivalence;
reject foreign origins and conflicting tags. Declarations are evidence supplied
by the researcher, not independently recovered producing history. A future
scientific variant must receive a distinct algorithm identity. See
[MATERIAL_TRAINING.md](docs/MATERIAL_TRAINING.md) for manifest requirements,
writer inventory, preserved behavior, and remaining provenance/native limits.

The environment evidence and proposed recovery commands are in
[GPU_ENVIRONMENT_INVENTORY.md](docs/GPU_ENVIRONMENT_INVENTORY.md) and
[GPU_ENVIRONMENT_DIAGNOSTICS.md](docs/GPU_ENVIRONMENT_DIAGNOSTICS.md).
See [docker/paper/README.md](docker/paper/README.md) for the proposed definition,
unresolved build inputs and Python 3.8 assessment. Test-harness compatibility only
is authorized; no interpreter upgrade or scientific compatibility fix is allowed.
See [docker/paper/PREREQUISITES.md](docker/paper/PREREQUISITES.md) for recovery,
verified source staging and remaining gates, and
[docker/paper/RUNTIME_DEPENDENCIES.md](docker/paper/RUNTIME_DEPENDENCIES.md) for
the recovered pins and unverified CUDA base candidate. Before any future build, review
the full exact dependency/source/artifact resolution and use a distinct candidate
image tag. Preserve source licences, compiler flags and scientific settings.
Do not begin first builds/environment installation, native GPU smoke tests,
extraction/evaluation implementation or work beyond current 1D.5A preparation,
or scientific repairs without separate user approval. Do not run Docker builds/
runs, GPU tests, real training or expensive experiments in this milestone.
Do not commit or push automatically. A later explicit user instruction
updates this boundary. See [CONFIGURATION_AND_PATHS.md](docs/CONFIGURATION_AND_PATHS.md)
for implemented APIs and limitations, and [INFRASTRUCTURE_PLAN.md](docs/INFRASTRUCTURE_PLAN.md)
for later milestones and acceptance criteria. The writer inventory, opt-in API,
preservation assessment, and limitations are in
[DATASET_CACHE_ISOLATION.md](docs/DATASET_CACHE_ISOLATION.md).
