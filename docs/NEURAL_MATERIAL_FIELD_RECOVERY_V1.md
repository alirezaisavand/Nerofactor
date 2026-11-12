# Constant neural material field: 3D-2 Phase B

Phase B implements the approved design. **Scientific validation and recovery
have not been executed.** The [Phase A plan](NEURAL_MATERIAL_FIELD_IMPLEMENTATION_PLAN_V1.md),
[source audit](A_NEURAL_MATERIAL_FIELD_AUDIT_V1.md) and
[draft protocol](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1_DRAFT.md) remain
unchanged. No final protocol or freeze is created here.

The implementation branch is `feat/neural-material-field-implementation-v1`,
based on Phase A `a1f15d1030cb954deddd18fb408de39a452fd441` and accepted
3D-1 `ccb03d0ec5adc814775538b9cc7427cc57480a98`. A separate worktree protects
the original workspace and its pre-existing research document.

## Representations and source fidelity

| Arm | Trainable representation | Material/shader interface |
| --- | --- | --- |
| D0 | Original seven-element leaf tensor | Original physical decoder and `ObservationBatch` operation order |
| N0 | One seven-element `nn.Parameter`; no linear layer | Identical physical computation, encoding, broadcasting and original batch adapter as D0 |
| N1 | Original A trunk and five heads, 1,485,584 trainable scalars in84 named tensors | Constant anchor; original material decoding; actual normalized two-channel orientation passed directly to the shader |

The seven physical diagnostic values are `[mx,my,theta,kR,kG,kB,metallic]`.
Six scalar latent coordinates use their original sigmoids/transforms; D0/N0
theta remains unconstrained. N1 theta is reconstructed only for diagnostics
and physical metrics using the accepted normalized half-atan2 convention.
The rendering path never reconstructs N1 encoding from that scalar angle.

[Field implementation](../experiments/neural_material_field_v1.py) admits all
pinned A source files through the existing stage-source manifest and verifies
`network/field.py` SHA-256
`b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.
Material A is `e1e859779301faced46d6476691e20b546c27ae0`;
geometry G remains `79252178e65914ad1acbc14790b1514f276bfea0`.
An AST whitelist compiles the exact original `Embedder`, `get_embedder`,
activation helpers, `make_predictor`, `MaterialFeatsNetwork` and
`predict_anisotropic_components` definitions in a private CPU namespace.
The full native module, CUDA renderer and `MCShadingNetwork` constructor are
never imported/constructed. There is no monkeypatch of scientific globals.
Source byte and individual AST hashes accompany N1 identities.

The original positional encoder has51 channels. The trunk contains eight
weight-normalized256-wide linear layers, with a307-wide skip input. Each
head is `259→256→256→256→d`, with the original hidden ReLUs and final sigmoid.
Head construction order is mx(1), my(1), metallic(1), rotation(2), kd(3):
eight raw channels for seven physical degrees of freedom. All28 original
linear modules retain deprecated `nn.utils.weight_norm` g/v/b parameterization;
its warning is expected. There is no illumination/geometry network.

Both the trunk input and the three raw coordinates concatenated into every
head are fixed at `(0,0,0)`. One anchor result is broadcast across arbitrary
finite CPU float32 query positions. This is the approved constant-field
restriction, not native spatial training or an architecture-only intervention.

## Deterministic fixed calibration

Original fresh seeds101,103,105 are reused across M1/M2/M3. Network draws use
private CPU Torch seeds10101,10103,10105, restoring caller RNG exactly. All
original nonzero random weights remain trainable. Each head retains its original
modules/hooks once and computes:

`sigmoid(target_logit + (original_pre_sigmoid(w,anchor) - fixed_initial_logit))`.

The fixed center and target are detached immutable buffers, not parameters or
per-update recalibrations. Six scalar target logits equal D0's original logits.
For rotation, float32 `e=(cos(2theta),sin(2theta))`, `q=(1+.5*e)/2`,
`log(q/(1-q))` supplies the two target logits. Original `2*q-1` and
`F.normalize(...,eps=1e-6)` then produce the actual shader encoding.
Ground truth is never an initializer input. The fixed radius is not a penalty.

Field-only tests find six scalar values bitwise equal for all three seeds and
Q within original `2e-7+2e-6*abs(Q0)` bounds. Seed101's scalar theta differs
from D0 by `5.960464477539063e-08` radians in the current CPU runtime; bitwise
scalar theta equality is not claimed. This observation is not rendered-output
admission. Initial centered radius, actual encoding norm and raw sigmoid
rotation outputs are distinct diagnostic quantities. Zero/near-zero actual
encodings fail the existing gate without a fallback direction or repair.

## Harness and available entry point

[Recovery harness](../experiments/neural_material_recovery_v1.py) reuses the
accepted M1/M2/M3 reference targets, original24 fitting observations and84
external observations. It verifies102 inherited source/evidence identities
in the current accepted chain; no target or quadrature is regenerated.
Actual training tapes3000–3003 and external tapes1000–1003 are verified by
array identities, including mutation checks during future execution.

The only CLI command currently exposed is read-only:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m experiments.neural_material_recovery_v1 --describe
```

Run from the implementation checkout. It verifies existing sources/evidence and
prints provisional settings and unresolved gates. It allocates no run, model,
shader or optimizer. `--execute` is deliberately unsupported.

Future `execute_reviewed(review,root,run_id)` is an API guarded before allocation
by explicit Phase D authorization, hash-selected final protocol/freeze and actual
Phase C report identities. Required Phase C evidence must describe this exact
implementation and certify both shader admission and runtime feasibility.
These files do not exist as Phase B products; this API does not authorize itself.

The implemented future ordering is all27 initial forwards/nine D0/N0/N1 gates,
then all nine D0 histories, all N0 histories, then N1. Each D0 must reproduce
the accepted1000-update prefix exactly, including saved losses, gradients,
parameters, optimizer moments/counters and snapshots250/1000. N0 is checked
against paired D0 logged trajectories and all snapshot states. No old runner
is invoked and no prefix is repeated. A rolling SHA-256 chain covers every
D0/N0 transition's loss, gradient, parameters and Adam state, so unlogged updates
also participate in exact trajectory admission without extra model queries.
Full trajectory gates remain unexecuted.

The candidate budget is5000 continuous original Adam returns, lr=.01,
betas=(.9,.999), eps=1e-8, weight_decay=0; no clipping, schedule, regularizer
or LM. Original A2/atan2 shader, float32 sample arithmetic,512 diffuse/1024
specular samples, four-tape component means then linear-RGB MSE are retained.
N1 is not a native-A loss or native training integration.

Snapshots are post250/1000/5000 own numeric states. The post250 objective reuses
the ordinary251 forward;1000 uses one counted no-grad fit forward;5000 uses
one terminal fit forward/backward without a step. Logged material/gradient
information comes from existing forwards; intermediate values are explicitly
pre-update. One final external evaluation uses own material truth and original
metrics, including component residual compensation and seen/unseen relighting.
No held-out selection or extra intermediate external evaluation is implemented.

## Work, resources and failure semantics

Per-history caps match the Phase A draft:5000 optimizer returns,5003 fit shader
forwards, one external forward,5002 backwards,480624 shaded points,
246079488 diffuse directions and492158976 specular directions. Attempts and
completions are separate. Two constant-field probes add two field queries;
the single N1 initialization calibration is separately counted, not a shader
call. Caps apply to failed attempts too. D0/N0 synthetic fixture work is not
scientific-run evidence.

Provisional resource limits remain one CPU thread, hidden GPUs,28800 whole
seconds,900 accumulated active seconds/history,10 seconds endpoint reserve,
4GiB peak RSS and8GiB address-space backstop. Initial all-model admission is
charged to whole-process setup; optimization, checkpoint I/O and final evaluation
are charged to each history. These budgets have not been shown feasible for N1.
Runtime ownership restores timers, signal handlers, thread count, limits and
caller RNG; unexpected scientific global RNG consumption is an integrity failure.

Source/target/tape/runtime or reproduction mismatches stop globally. Isolated
nonfinite forward/loss/gradient/Adam states become `failed_numerical`; timed
histories are `censored`, not successful, restarted or substituted. Integrity
errors are `failed_integrity`; partial outputs remain. A numerical exception
during final external evaluation stops execution globally with its diagnostic;
it is not an admitted completed result. D0 failure prevents neural execution.
All statuses must remain in future reporting; no retries or budget extensions.

## Output and pickle-free state format

An explicitly selected existing absolute, unaliased, user-owned EMPTY directory
outside code/input roots is required, with validated unique run ID. Collisions,
symlink escapes, traversal and overwrites are rejected using existing guards.
`identity.json`, streamed strict `report.json` and `checkpoints/` are owned output
files. Allocation failures preserve partial directories and close owned handles.
No historical checkpoint, dataset or prior experiment is an output destination.

Each own checkpoint is an exclusive `.npz` numeric archive plus `.json` manifest:
named parameter arrays, named Adam step/first/second moments, immutable
anchor/calibration buffers and numeric CPU Torch/NumPy RNG state. The manifest
records roles, shapes, dtypes, device, per-array SHA-256, whole archive hash/size,
source/config identity, completed count and effective optimizer options.
Full N1 weights/moments stay outside Git; summary reports contain hashes.

`restore_checkpoint` requires explicit own manifest path/hash and verifies the
archive BEFORE `np.load(...,allow_pickle=False)`. Keys, shapes, dtypes, buffer
values, moments, counters and optimizer options must match before mutation.
Failure-diagnostic archives may preserve nonfinite numeric arrays with explicit
flags and strict finite JSON metadata, and are never resumable. RNG is recorded
for provenance and is not restored by this state-inspection helper. Experiment
execution itself has no resume route. No historical pickle is deserialized.

## Phase B checks and pending validation

Focused tests use field-only original components, synthetic algebraic shader
fixtures and at most a few seven-vector Adam transitions. They do not instantiate
the real scientific shader, perform sampling, render, optimize N1, rerun any
prior experiment or claim physical recovery. Seven-direction reachability uses
a7×8 Jacobian of original final-layer biases, not a shader Jacobian over1.49M
weights. Whole-head/trunk detached paths fail; predicted ReLU/weight-normalized
constant-input structural zeros are distinguished from missing gradients.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest \
  tests.scientific.test_neural_material_field_v1 \
  tests.scientific.test_neural_material_recovery_v1 -v
```

Phase B result: **42 passed,0 failed,0 skipped** (16 field tests and26 harness
tests;9.129 seconds). Runtime: Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121
executing on CPU with one thread and GPUs hidden. Earlier development checks
caught a wrong fixture component-magnitude assumption and an overly broad
Git-directory guard; both were corrected in the new files before this final
run. Static AST parsing, local documentation links and whitespace checks pass.
SHA-256 preservation checks confirm667 pre-existing files unchanged, excluding
only the explicitly justified AGENTS milestone-status update. Full shader
forwards, quadrature/reference generation and N1 optimizer updates: **zero**.

No final protocol/freeze, standalone saved-data analysis tool or recovery result
is created in Phase B. The Phase A analysis-tool proposal is deferred until
there is approved evidence to analyze. No scientific thresholds were changed.

Next request should authorize **bounded Phase C only**, with an exact work cap:
real A2/atan2 initial frame, samples, masks, per-tape and mean components, Q,
original arithmetic envelope, exact replay and finite backward admission for
all seeds/material associations; gauge fixtures; target-independent N1 CPU and
storage feasibility. Use original tolerances with no post-result calibration
search. The real shader may fail these gates despite passing Phase B tests.
Only after that review may a final protocol/freeze and separately authorized
Phase D27-history scientific execution be considered. No recovery improvement,
population reliability or full native-A equivalence is established here.
