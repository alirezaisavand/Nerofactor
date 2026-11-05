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

Milestones through 1D.3B-2B have been reviewed and approved.
Current authorized milestone: **1D.3C, native GPU smoke implementation for review**.
Prepare separate geometry/material initialization and one ordinary forward using
existing schema-1 configurations, real read-only inputs and disposable isolated
outputs. Do not execute the native backend/GPU/Docker commands in this milestone;
only static checks, structural preflights and explicit CPU fixtures are authorized.
No training loop, optimizer construction/update, backward, checkpoint I/O,
validation/NVS or image/package changes. Keep original model equations, sampling,
counts, defaults, seeds, splits and shader scopes unchanged. Record that the smoke
omits DataLoader iteration and is not training-loop RNG or scientific parity.
The researcher reports quick-candidate-1 passed basic native checks and 100 tests
on Python 3.8; attribute that evidence and do not claim these new model tests ran.
Preserve the existing Python 3.8 parser-created AST literal compatibility fix and
all original source-comparison assertions. Stop with the implementation ready for
review; native launches need a later explicit approval.
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
Its scope is the separate smoke CLI/helper, CPU tests, the AST compatibility fix
and supporting documentation. Use the recovered system Python 3.8.10 / CUDA 11.1 / Torch
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
extraction/evaluation refactoring, 1D.4, or scientific repairs without separate
user approval. Do not run
Docker builds/runs, GPU tests, real training, or expensive experiments in this milestone.
Do not commit or push automatically. A later explicit user instruction
updates this boundary. See [CONFIGURATION_AND_PATHS.md](docs/CONFIGURATION_AND_PATHS.md)
for implemented APIs and limitations, and [INFRASTRUCTURE_PLAN.md](docs/INFRASTRUCTURE_PLAN.md)
for later milestones and acceptance criteria. The writer inventory, opt-in API,
preservation assessment, and limitations are in
[DATASET_CACHE_ISOLATION.md](docs/DATASET_CACHE_ISOLATION.md).
