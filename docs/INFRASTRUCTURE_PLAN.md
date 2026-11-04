# Incremental infrastructure implementation plan

Status: **1D.1 is approved; 1D.2A is implemented as standalone configuration and
path infrastructure**. Entry-point/writer integration in 1D.2B and milestones
1D.3 and 1D.4 still require review and authorization. No branch merge, container
execution, training, artifact relocation, or numerical correction is included.

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

In particular, path destinations exist as an API; dataset writers have not been
redirected. Resume validates allocation identity and checkpoint location/existence,
not scientific state compatibility. Configuration defaults are supplied explicitly
by their owning code; this foundation does not introduce a new numerical preset.

### Proposed implementation

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

### Proposed implementation

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

### Proposed implementation

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
with the driver. GPU identity, usable runtime, and version compatibility remain
unverified. This plan preserves those as separate environment blockers rather
than treating source inspection as execution evidence.
