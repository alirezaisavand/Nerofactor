# E.2a v1: paired early-step LR protocol, preparation only

Authority: researcher E.2a instruction, based on E.1 commit
`06f837af623df76e194f60d475a1deb94fa4045a`, branch
`feat/neural-material-early-step-e2a`. **Execution is not authorized.**
The previous [E.2 draft](NEURAL_MATERIAL_ABLATION_E2_DRAFT.md) remains unchanged;
this explicit five-case/100-update amendment supersedes its proposed four cases.

## Question and matrix

Does reducing only Adam LR change the early saturation/gradient/displacement
trajectory of the original N1? This is a ten-update diagnostic, not a recovery
experiment or evidence that a smaller LR improves 5000-update outcomes.

| Case | Role | Independent fresh arms |
| --- | --- | --- |
| M1/101 | Complete saturation | LR .01, LR .001 |
| M1/103 | Successful recovery control | LR .01, LR .001 |
| M2/103 | Complete saturation in another material | LR .01, LR .001 |
| M3/101 | Roughness-floor failure | LR .01, LR .001 |
| M3/103 | Photometric/orientation near miss | LR .01, LR .001 |

Run in table order, baseline then intervention for each case. Reconstruct both
models independently from the original initial coordinates/seed 10000+seed;
require identical full parameter, calibration-buffer, source and initialization
catalogs before either arm. No trained-state initialization, continuation, retry,
replacement case or alternate LR. A failure stops the entire experiment; later
arms remain explicitly `not_started`.

## Scientific invariants

Reuse the original `neural_material_field_v1` constructor and
`neural_material_recovery_v1` MaterialBatch, MSE, backward, Adam and evaluation
functions. Do not edit them. Preserve float32 CPU operation order, eight-layer
feature trunk, five centered-logit heads, weight normalization, sigmoid bounds,
roughness transform, orientation normalization and A2/atan2 shader. Actual
orientation is the two-channel encoding passed into `shade`, not diagnostic
theta. Preserve 512 diffuse/1024 specular samples, fixed fit tapes 3000–3003,
evaluation tapes 1000–1003, observation ordering, independent existing targets,
four-tape mean linear RGB then original MSE, physical/gauge criteria and all Adam
flags. The original betas (.9,.999), eps 1e-8 and zero weight decay are unchanged;
only the intervention LR differs. No scheduler, clipping or regularization.

The freeze retains the full old configuration as `inherited_configuration_context`
for provenance. Its old three-arm/5000-step schedule is **not** the E.2a schedule;
only the explicit N1 matrix and ten-return bounds here are prospective authority.
The exact accepted runtime is Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121,
CPU float32, one computation thread and the frozen arithmetic probe identity.
The CUDA-tagged installed Torch package does not authorize GPU execution.

## Call and state accounting

| Per arm | Forwards | Backwards | Successful optimizer returns |
| --- | ---: | ---: | ---: |
| Pre-update states 0–9 | 10 | 10 | 10 |
| Terminal fitting state 10 | 1 | 1 | 0 |
| External endpoint at state 10 | 1 | 0 | 0 |
| Total | 12 | 11 | 10 |
| Ten arms | 120 | 110 | 100 |

Fitting calls have 24 observations x four tapes = 96 points. The single external
call has 84 x four = 336 points. Per arm: 1392 points, 712704 diffuse and
1425408 specular directions. Original constructor calibration remains separately
identified as `calibration_forwards=1` per reconstructed arm; it is not an extra
fitting/shader query. No initial admission rerender, prefix query or loss
recomputation is added. Count attempts before calls and completions after returns;
retain native counters if a post-return finite/state check fails.

Historical log label **1 measures state 0**, label **10 measures state 9**.
Before baseline updates 1 and 10, require exact saved loss, original N1 gradient
norm (original group reduction/order) and physical-before-update values. Fail
globally on mismatch, before that update. Baseline completes before intervention;
intervention state-0 loss/norm/physical signature must also equal baseline.
Source, target, tape and full initial-model identities are independently gated.
Real scientific parity remains pending E.2b; saved-data tests do not establish it.

## Passive observation contract

Hooks return `None` and retain actual autograd nodes. Capture the actual input of
each final Sigmoid **after centered calibration**, its output, retained loss
gradients, boundary distance and float64 analytic q(1-q) diagnostic. Capture all
22 named ReLUs' inputs/outputs and retained gradients, activity/zeros, shared
feature output/gradient, and all six parameter-gradient groups. These diagnostics
do not change the float32 forward/backward computations.

Wrap only the shader call boundary to retain the actual mx/my/albedo/metallic/
encoding tensors and their loss gradients; delegate exactly once. Capture
rotation raw-channel gradients via its sigmoid nodes. Record centered radius and
theta as diagnostics; they need no retained loss gradient. Never classify a
disconnected diagnostic-theta path as missing physical orientation gradients.

Record states 0–10 with actual loss, physical values, Q/recovery error, physical
boundary distances, orientation-identifiability diagnostics, finite/zero/nonzero
gradient summaries, exact parameter hashes, Adam options and before/after
moment/counter statistics. Record the first **observed** sigmoid-boundary and
all-zero-gradient states; null means not observed during this horizon, not absent
later. Do not interpolate old sparse logs. Hook removal and graph release are
mandatory on success/error. No silent gradient replacement, detach-before-
backward, tensor substitution, extra graph query or extra optimizer return.

Measure displacement from real pre/post float32 parameters. Report per-tensor/
group numeric and bitwise changed counts, float64 L2/max changes and exact hashes.
Adam momentum-derived proposals are not displacement evidence: a zero current
gradient can still move weights, and nonzero proposals can round away.

## Numeric evidence and output bounds

All new output is a fresh owned, unaliased `neural-e2b-*` child of the approved
`/home/isavand/paper/evidence`, outside Git and historical inputs. No overwrite or
implicit selection/resume. Exclusive JSON/NPZ writes, SHA-256 catalogs, record
hash chaining, finite/dtype/shape checks and `allow_pickle=False` reads are
required. Initial full state plus uint32 XOR deltas reconstruct all ten float32
transitions bit for bit, including signed zeros. Keep final full parameters,
buffers, Adam moments/counters and CPU/NumPy RNG; reserve one failure state.
These diagnostic states do not authorize resumption. Never dump full parameter
gradients or all Adam moments at every step.

The frozen 84 parameter tensors contain 1485584 float32 elements (5942336
bytes); 12 immutable buffers occupy 104 bytes. Shape-derived caps:

| Capture | Count per arm | Cap each |
| --- | ---: | ---: |
| Initial parameters/buffers/RNG | 1 | 6 MiB |
| Bitwise parameter XOR deltas | 10 | 6 MiB |
| Final and reserved failure state | 2 | 18 MiB |
| Actual-graph signal/gradient arrays | 11 | 1 MiB |
| JSON records, including numeric manifests | at most 64 | 2 MiB |

Per-arm reserve: **241 MiB**. Ten arms plus 4 MiB root metadata:
**2531262464 bytes (2414 MiB)**; minimum free capacity **5062524928 bytes**
(twice that bound). Enforce total bytes, per-archive bytes and JSON inventory;
check capacity before allocation. Full signal payload is 98528 bytes/pass,
130 arrays maximum plus 133120 bytes header allowance. It includes retained
intermediates and gradients, not the renderer's entire autograd graph.

Memory planning uses twice the accepted prior peak RSS 1229254656 bytes, plus
16 parameter-byte sets for paired states/transient checkpoint/XOR copies,
98528 signal bytes and 128 MiB bounded JSON/serialization workspace: **2687902944
bytes**, below 4 GiB. This is conservative planning, **not a measured new
instrumentation benchmark or mathematical upper bound for arbitrary Torch
internals**. The 4 GiB RSS / 8 GiB address-space caps, timer and watchdog remain
mandatory. Raw graphs retain their existing renderer allocations; hooks retain
small graph nodes only until capture/release. No stacked histories are retained.

## Time, interruption and authorization

Candidate bounds: 120 active wall seconds/arm, ten seconds reserved before an
optimizer call, 1800 seconds whole process; no retries/extensions. Prior accepted
continuation took 3808.4619465004653 wall seconds for 34000 native optimizer
returns (including the separately authorized replay), approximately 1.12 seconds
per ten returns under a simple linear projection. That is scale context only; new
instrumentation/fsync time is unmeasured. These remain defensible **candidate**
limits, not a promise of native completion. A resource failure censors/stops;
it does not permit a larger budget. Hidden GPUs and OMP/MKL/OpenBLAS thread
variables must be set before process startup. Restore owned timers, handlers and
resource state on exit; reject unexpected global RNG consumption.

Future E.2b must supply a separately approved hash-selected JSON review with
exact phase `E.2b`, positive scientific authorization, freeze SHA, exact clean
reviewed branch/commit, fresh output root/run ID and external watchdog
`{wall_seconds:1800, kill_grace_seconds:10}`. The E.2a freeze itself always says
`execution_authorized:false`; no positive review is created here. The runner CLI
only describes preparation. The scientific factory and generic arm driver both
gate scientific use; all scientific imports are below complete admission.

Future reviewed invocation must be externally wrapped in
`timeout --signal=TERM --kill-after=10s 1800s` with CPU/thread environment set.
The review records the watchdog contract; it is not OS attestation that an
external supervisor was used. E.2b review must check the actual invocation.
TERM/time/resource failure preserves partial results and one bounded failure
snapshot if possible, removes hooks and stops all further arms. Abrupt KILL/disk
failure can prevent final serialization; exclusive partial files and last valid
record hashes remain evidence, never a successful run. Do not retry/recover.

## E.2a validation and immutable freeze

Only saved-JSON/hash admission and unrelated tiny toy-network/Adam **engineering
fixtures** are permitted. No real N1 construction/query, known-plane shader,
scientific optimizer, new targets or CUDA. Focused tests cover passive equality,
all groups, actual encoding gradients, ReLUs, derivatives, actual displacement,
state indexing, exact parity logic, malformed evidence, isolation, interruption,
caps and authorization rejection. A scientific-import blocker protects fixtures.

Freeze source/tests/protocol identities, inherited hashes, saved parity rows,
initial identities, runtime, call/resource/storage contracts and actual test
inventory. Validation JSON is a separately hashed DAG input; freeze excludes
itself and the human readiness summary to avoid circular hashes. Creation uses
exclusive `x` mode. Verification independently hashes the selected freeze and
recomputes its full source/evidence contract without Torch/model imports:

```bash
python3 -m experiments.neural_material_early_step_e2a --describe
python3 -m experiments.freeze_neural_material_early_step_e2a --verify-sha256 SELECTED_SHA256
```

The next action is researcher review of the E.2a commit. Scientific parity,
native overhead, actual saturation onset and paired LR outcomes remain
unverified until separately authorized E.2b. No optimization improvement is
claimed, and no 5000-update continuation is prepared or authorized.
