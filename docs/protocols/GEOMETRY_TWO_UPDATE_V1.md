# Bell two-update geometry training protocol v1

Protocol ID: **`bell-two-update-training-v1`**. Milestone **1D.5A**, review
baseline `ef7e6b270137c1315c1ef01f303af910ee1dcbaf`. Implementation and commands
are prepared for review. **No native training/GPU execution is authorized or
performed by Codex.** This is fresh maintained-A infrastructure validation, not
historical thesis training reproduction, G reproduction or a convergence test.
See [the execution/audit guide](../GEOMETRY_TRAINING_SMOKE.md).

## Fixed scientific question and budget

Can the actual maintained `Trainer.run()` initialize its native geometry/data/
loss/Adam path, compute finite losses and gradients, and return from **exactly
two Adam calls**, with valid optimizer state and a real parameter change?

Use original `configs/shape/syn/bell.yaml`, the existing geometry schema-1
example, seed 6033, the approved isolated A three-list split shim and original
scientific modules. No checkpoint/optimizer state is loaded. No scientific
configuration is shortened: `total_step=300000`, validation/save intervals 1000,
train-log interval 20, workers 8, 512 training rays and all native sample/loss/
precision settings remain unchanged. No first-interval validation or checkpoint
is expected; no forced save or early validation is allowed. Mesh extraction,
material/NVS execution and full training are outside this protocol.

**Zero-LR first call:** `WarmUpCosLR({})` constructs Adam at LR 0.001, then sets
step 0 LR to **0.0**, step 1 LR to **`5e-4 * (1/5000)`** (approximately `1e-7`).
Preserve Python's original float arithmetic, not a rounded literal substitution.
The first Adam call advances moments/per-parameter counters but must leave model
parameters unchanged. The second must change at least one parameter. These are
two successful optimizer calls and **one nonzero-LR parameter update**, not two
parameter-changing steps. A requirement for two nonzero-LR updates would need a
separate protocol revision; never execute a third call to obtain that result.

Adam remains the actual native class with initial `lr=1e-3`, `eps=1e-4`,
`amsgrad=True`, native betas `(0.9,0.999)` and zero weight decay. Preserve the
scheduler's warm-up length 5000, retained `end_iter=300000` attribute and actual
hard-coded cosine endpoint **100000**, alpha 0.2. Preserve the original loss
aggregation/insertion order, `.backward()`, L2 clipping at **0.7** and
`optimizer.step()` calls. Do not move loss fields, alter gradients/precision,
change sample counts, restore RNG or add deterministic/TF32 overrides.

## Arms and independent configuration resolution

1. **R:** direct legacy `yaml.FullLoader` resolution with explicit dataset path
   substitution; it never calls C's schema loader/geometry validator.
2. **R-repeat:** another fresh process/output directory with R's route.
3. **C:** actual schema-1 loader, PathResolver and geometry validation with the
   unchanged example envelope, empty overrides, no checkpoint or mesh selection.

Both routes pass their independently resolved mappings to the same unchanged
Trainer, with the approved isolated geometry/cache/storage context. They share
scientific definitions and observers; this is not an independent algorithmic
implementation. R also uses the explicit A split-unpacking compatibility shim,
so it is not an unmodified legacy-entry-point run. No historical G run is implied.

The source and effective trainer/renderer/shader/loss scopes, dataset identities,
ordered split, cache settings, native image/runtime/backend/thread settings,
loader/scheduler/optimizer identities and initial parameter/buffer metadata must
match. The full A split is frozen: train 122, validation `["58"]`, NVS
`["11","105","114","17","76"]`, with complete ordered lists in the report.
View 58's historical held-out status remains unknown; it is G training under the
inspected G split. Training does not evaluate that view during this budget.

## Exact bounded stop and observations

The separate harness invokes the **original Trainer.run method**, not a copied
loop. A scoped CPython `sys.settrace` observer selects only that owned frame and
locates its existing statements by AST. Pinned source hashes and code filename/
first-line checks guard those boundaries. It never wraps or replaces Adam,
losses, clipping, dataset methods or native renderer operators. It refuses an
existing debugger/coverage trace or a reused controller; it has no fallback
that could launch the unbounded loop.

Observe the actual dataset/network initialization, loop entry, forward,
backward, clip, optimizer and first following statement boundaries. The actual
Adam call must return before counting completion. After return 2, record its
outcome and raise a private `BaseException` stop **before any next loop work**.
Never fetch a third batch or call Adam a third time. Steps are exactly 0 and 1;
returning without this stop is failure. An exception or diagnostic failure aborts
immediately, preserving the count of already observed successful Adam returns.

Observations copy detached tensors to CPU and do not mutate scientific tensors,
change gradient mode or draw scientific RNG. Per-loss observational means use
Torch's mean on detached values; the aggregate is the **actual original loop's
loss tensor**, not a reconstructed replacement. Parameter/gradient/Adam moment
statistics use float64 reductions on detached CPU copies; those diagnostic
reductions are not fed back into training. Report the original clip function's
returned norm separately from diagnostic CPU64 pre/post norms. Enforce finite
loss elements, aggregate, gradients, parameters and moments. Abort if post-clip
CPU64 norm exceeds 0.70001 (rounding slack around the unmodified 0.7 operator).

Per-parameter Adam counters increment only for that parameter's non-None gradient;
newly active parameters can legitimately have counter 1 at global call 2. Do not
force all counters to 2 or treat inactive/frozen heads as missing training.

Cleanup restores the previous trace, shuts down the owned training DataLoader
iterator's workers, closes the progress bar and TensorBoard writer, and records
cleanup errors. An error in cleanup is not a pass. External termination may leave
`running`/partial reports and stale last-persisted counts; preserve them as
incomplete and never resume/relabel them. One fresh process owns one arm.

## RNG, batching and unavoidable limitations

Use the actual Trainer constructor's Torch/NumPy/Python seed calls in original
order. Keep native dataset/renderer construction, the two actual A split calls,
full-resolution ray-pool construction and initial shuffle. The local split
verifier makes no additional global shuffle draws.

Retain real train/validation DataLoader constructors: batch 1, 8 workers, training
RandomSampler and original collate. Native DummyDataset length is 99999999;
starting its iterator can allocate a correspondingly large permutation and
consume main Torch RNG despite yielding an empty dummy dictionary. **Do not
replace or optimize this away.** Native worker creation, prefetch, sampler draws,
train batch slicing/shuffle conditions and forward/backward order stay intact.

Record RNG hashes before batch fetch, before forward and after Adam for each
step. Record selected real ray-pool slice hashes (`dirs`, `rgbs`, `idxs`, `masks`
when present), pool size/cursors and all effective sample counts. Dummy dataset
indices are discarded by original `__getitem__`; worker RNG states and the full
sampler permutation are not exported. Do not inject index reporting or traverse
an extra loader iteration merely for provenance.

Observers synchronize GPU-to-CPU reads, add memory/time overhead and change wall
clock/prefetch timing. Both arms share this instrumentation. CPU tests verify no
scientific RNG draws/tensor mutations by observations; they cannot prove an
unobserved full run's native trajectory. Exception cleanup is early-stop resource
cleanup, not normal full-loop completion. Partial constructor failures may lack a
fully assigned writer/iterator to close; process isolation/resource limits remain
necessary. Legacy CUDA default-tensor failure behavior is not silently repaired.

## Frozen repeatability and comparison rules

Run **R → R-repeat → CPU repeatability gate → C → final CPU comparison**, each in
a fresh output directory/process on the same selected GPU/image. If R/R-repeat
fails, stop before C; final comparison independently rechecks repeatability.
The comparator delays opening C until the reference repeat passes.

This first protocol is deliberately a **strict state-repeatability screen**:
exact parameter, gradient and Adam-moment byte hashes, shapes/dtypes, active-key
sets, counters, ordered sampling hashes/cursors and pre/post RNG identities
must match. Initial seeded parameter/buffer/optimizer metadata and all source/
config/input/runtime identities must match exactly. Hash mismatch fails even
when norms are close: norms alone cannot establish vector equality. Full tensors
are not exported to make a more permissive elementwise comparison. If native
training is not byte-repeatable, report the discrepancy and request a separately
reviewed protocol with the necessary vector evidence; do not widen bounds, retry
seeds, average results or turn on deterministic kernels after seeing outcomes.

Predeclared scalar diagnostic bounds are `abs(other-R) <= atol + rtol*abs(R)`:

| Quantity | atol | rtol | Scope/rationale |
| --- | --- | --- | --- |
| Actual aggregate and named weighted loss means/recorders | 1e-7 | 1e-5 | Dimensionless legacy loss reductions; float32-scale diagnostic allowance |
| Gradient and clip-return norm/statistics | 1e-7 | 1e-5 | Gradient units; no inference RGB tolerance substitution |
| Per-update parameter norm/mean/max statistics | 1e-8 | 1e-6 | Weight units; initial state identity remains exact |
| Parameter-update L2/max magnitude | 1e-12 | 1e-4 | Updates are about 1e-7; an RGB absolute bound of 1e-4 would obscure them |
| Adam moment norm/mean/max statistics | 1e-12 | 1e-5 | Small moment units; epsilon/AMSGrad behavior remains native |

These are conservative, precommitted **screening thresholds**, not measured
native error bounds or scientific equivalence guarantees. They allow tiny
reporting-reduction differences; they do not relax tensor-byte equality. LR,
clipping threshold, steps and discrete metadata are exact. Durations and memory
measurements are diagnostic and are not numerical equivalence criteria.
An exact-hash failure can reflect native nondeterminism; it does not by itself
establish a scientific algorithm defect. Preserve it for review without repairs.

Reports contain every scalar difference, limit and pass flag, identity and exact
state mismatches. No PSNR/SSIM/RGB threshold or benchmark is used here. Passing
only certifies this instrumented two-call prefix under the admitted environment.
It establishes no full-training, later scheduler/loss transition, validation,
checkpoint-selection/resume, historical thesis or material/BRDF equivalence.

## Report schema and failures

[TrainingProbe](../../infrastructure/training_probe.py) populates and
[the JSON validator](../../infrastructure/training_comparison.py) checks schema 1:

| Field | Required contents |
| --- | --- |
| `protocol`, `schema_version`, `arm`, `run_id`, `route` | Version above, R/R-repeat/C, unique ID, actual independent route |
| `status`, `phase`, timestamps, `argv` | Running/failed/complete; last observable phase, original invocation, failure type/message when available |
| `source`, declared image/commit, `runtime` | Protected historical/current source hashes, new harness/protocol hashes, caller declarations, actual native package/backend/ABI/GPU/thread fields |
| `execution_mode`, `checkpoint_input` | Fresh initialization, null checkpoint; no historical checkpoint mount |
| `legacy_total_step`, `requested_optimizer_updates`, `optimizer_calls_completed` | 300000 unchanged, budget 2, actual observed completed returns |
| `used_inputs`, `preprocessing`, `splits` | Hash-bound admitted 375-file identity superset (includes unused validation depth), actual cache identity and ordered A lists |
| Effective trainer/renderer/shader/losses, `scheduler`, `loader`, initial optimizer/parameters/buffers | Actual defaults/scopes (including each registered loss object's cfg in order), native construction metadata and exact initial tensor hashes |
| `updates[0:2]` | Steps 0/1, LR, named loss means and actual aggregate, pre/post clipping, parameter snapshots/deltas, Adam state metadata/counters, selected-ray/RNG identities, GPU memory and duration |
| `partial_update`, `error_type`, `error`, `cleanup_errors` | Last partial diagnostics and failure/cleanup records; no invented completion |
| `output_hashes`, `limitations` | Owned config/geometry/log/event byte identities; caches/temp are isolated but not exported; incomplete provenance limitations |

No tensor values, full gradient/optimizer payloads or model checkpoint files are
exported. State records contain tensor name/shape/dtype/hash/numel/L2/mean/max-abs,
not data arrays. Update records add changed-element counts and L2/max deltas.
The original config bytes and actual owner defaults are saved by approved
GeometryExperiment storage. A successful final phase is
`bounded_stop_after_second_optimizer_return`, not full Trainer completion.

CPU comparison outputs `training-comparison.json`: report path/SHA identities,
frozen tolerances, repeatability/candidate results and compact scalar rows
`[path,reference,other,absolute_difference,limit,passed]`. Results are
`repeatability_passed` / `bounded_training_regression_passed` (exit 0),
`repeatability_failed` / `candidate_mismatch` (exit 1), or `blocked` (exit 2).
Candidate failure retains the completed reference-repeat diagnostics. Missing,
nonfinite, invalid, tampered, partial or incompatible reports cannot pass.
