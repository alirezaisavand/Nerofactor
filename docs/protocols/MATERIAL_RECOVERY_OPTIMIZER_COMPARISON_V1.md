# Material recovery optimizer comparison v1

Protocol `material-recovery-optimizer-comparison-v1`, milestone **3C-4**.
Frozen before the single authorized execution. Base
`a64e7e77503fb3ed1a9fb26969d2bf2ad330fcea`; feature branch
`feat/material-recovery-optimizer-comparison-v1`. No result-dependent revisions,
retries, alternative policies or later experiments are authorized.

## Question and controlled factors

Test optimizer-family limitations in the seven-scalar, known-flat-material case.
Both arms start at the exact five accepted update1000 parameter vectors, seeds
101–105. Adam retains its historical step/moments; LM has no equivalent history.
They share parameters and objective, **not optimizer state**. Iteration ceilings
and equal wall caps do not establish equal computational work.

Preserve unchanged A2/atan2, Beckmann/Kurt a=0 algebra, sigmoid/angle transforms,
known geometry/lighting, the24 original fitting observations and independent
targets, original four fixed training tapes0–3,512 diffuse/1024 specular samples,
and four evaluation tapes1000–1003. Average predictions over the tapes **before**
linear-RGB MSE. No regularization, clipping, scheduler, noise, new initialization,
shader/decoder/sampling changes, target regeneration or held-out selection.
Reuse admitted convergence and local identifiability gates; never rerun them.

## Artifact/state admission and mandatory initial gates

Read only these exact accepted JSON artifacts and their bound source/protocols:

| Artifact | SHA-256 |
| --- | --- |
| `docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json` | `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451` |
| `docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json` | `6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f` |
| `docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json` | `224cb9a0cb27bbaacc61b8d575efbcd2c37854b716fc51bc78b439020239f24f` |
| `docs/protocols/MATERIAL_RECOVERY_CONTINUATION_V1.md` | `e6bbde21188d7178669369ea551cfeb663633b60e0c09ce775154d3e473b0693` |

Verify complete states, seeds/counters, source/protocol identities, scene,
targets/tapes and original optimizer options. Require the original recorded
Python3.9.12, NumPy1.21.5 and Torch2.4.0+cu121 versions, CPU/default float32,
one Torch/BLAS/OpenMP thread and hidden GPUs. Version matching does not prove
original binary identity. No CUDA query, network, native model or neural training.

For all five starts, admit the typed/hash-bound `latest_state`: one CPU float32
seven-vector, physical values, scalar step1000 and float32 seven-vector moments.
Every saved JSON number must round-trip exactly as float32. Verify dtype, shape,
device, byte hash and physical transform; require nonnegative second moments.
Restore Adam via `load_state_dict` and demand exact snapshot equality including
all effective flags. No earlier updates are replayed. LM receives an independent
clone of exactly the same coordinates, with no Adam state or fallback optimizer.

Before **any** optimizer update, evaluate each of the ten starting states, once,
using original `ObservationBatch`. Demand bitwise-equal arm predictions,
coordinates, physical values and normalized residuals, equal arm MSE and **exact
equality** with the saved3C-3 endpoint MSE. Any discrepancy is a fatal mandatory
gate failure; preserve evidence and stop. These real initial checks are the first
phase of the one authorized execution, not a preliminary experiment. Tiny new
fixtures validate their logic before execution.

## Arm A: Adam continuation

At most750 additional original Adam returns, ending at historical counter1750.
lr=.01, betas(.9,.999), eps1e-8, weight_decay0, amsgrad/maximize/capturable/
differentiable false and foreach/fused None. Use unchanged original float32
MSE/backward, no clipping/regularization/schedule. Count successful returns and
require the Adam counter equals1000+returns. Interrupted/ambiguous return
boundaries fail closed. Stop at ceiling or time allowance, never restart Adam.

Log the first return and every tenth thereafter, with **pre-update** objective
and gradient, post-update coordinates, cumulative work and elapsed active time.
The final endpoint is always recorded even when a time stop is between logs.
Preserve complete final Adam state. No iterate is selected by evaluation data.

## Arm B: damped Gauss–Newton/LM

Let the float32 residual be
`r=vec(four_tape_mean_linear_RGB-target_float32)/sqrt(72)` for24 RGB observations.
The unmodified original fitting objective is `mean((prediction-target)^2)`.
Every fit forward checks its equality to `r.double().dot(r.double())` with
absolute1e-12+relative5e-7 allowance for float32 normalization/summation roundoff.
This is an algebra/rounding check, not a changed optimization target. Original
float32 MSE is used for actual reduction and acceptance. Starting MSE equality
has **no tolerance**.

Construct the72x7 Jacobian using exactly seven AD directional JVPs per complete
outer iteration, one for each standard-basis direction. Use explicit
double-reverse AD: for each direction one primal shader forward, then a
`create_graph=True` VJP against a zero72-vector requiring gradients, followed by
a second reverse sweep with the direction. This is an exact local AD JVP,
**not** a finite-difference or silently selected fallback implementation.
Require every primal residual to equal the cached current residual bitwise and
all derivatives to be finite. Failure stops the scientific run. New synthetic
analytical and tiny A2 fixtures must verify dimensions, derivative agreement
against bounded central differences (steps1e-4 and5e-5), and reverse-gradient
consistency first. Tiny real-shader fixture counts8/16 are test-only; the
experiment always retains512/1024. FD fixture bounds are rtol.03/atol2e-4 for
tiny A2 and rtol.01/atol3e-4 for the synthetic nonlinear residual; no scientific
derivative tolerance or estimator tuning is introduced.

Promote J and r to float64 for the small solve:

```
H=J.T@J; g=J.T@r
D=diag(diag(H)+1e-8)
(H+lambda*D)*delta=-g
candidate_float32=(z_float64+delta).float32
represented_delta=candidate_float32.float64-z.float64
predicted_reduction=-2*g.dot(represented_delta)-represented_delta.T@H@represented_delta
```

The predicted reduction uses the **actually representable** float32 proposal,
and the full squared-residual objective (no missing factor2). No step clipping,
angle repair, sigmoid/axis change or alternative solve is permitted.

- Initial lambda1e-2; clamp lambda to[1e-8,1e8]. At most80 outer iterations and
  four trials per outer. Reuse that outer's single J/H/g for all its trials.
- Accept only strict training-MSE decrease and finite ratio
  `(old_MSE-new_MSE)/predicted_reduction > .1`.
- On acceptance: divide lambda by3 for ratio>.75; multiply by2 for ratio<.25;
  otherwise retain it. Equality at.75/.25 retains lambda; then apply clamps.
- On rejection: multiply lambda by10 before the next trial, including carrying
  the increased lambda to the next outer after a fourth rejection.
- Zero/negative/nonfinite predicted reduction is rejected **without** a trial
  shader forward. Nonfinite/invalid proposal coordinates, invalid numerical
  shader outputs, singular/unsolvable systems and nonfinite ratios are rejected.
  Preserve reasons and attempted coordinates; explicit JSON nulls/indices mark
  nonfinite numbers, not substitute parameters.
- A rejection **at** lambda1e8 ends that seed/arm as `damping_saturated`.
  Reaching1e8 by multiplication allows the next trial at that value. An accepted
  step at the maximum can continue under the usual policy.
- No gradient-norm, held-out or success-based early stopping. No second Jacobian
  in an outer, extra trial, alternate optimizer, adaptive tape or policy retry.

Preserve each outer linearization point, residual-Jacobian hash, normal matrix,
singular values/right singular vectors, gradient, damping, proposal coordinates,
float64/represented deltas, actual/predicted reduction, ratio, mask changes,
angle-branch crossings and decisions. Sensitivity belongs to its recorded
linearization point, not automatically to the final accepted successor. These
are local derivatives of the finite-pattern objective, not proof of global rank
or agreement with continuous derivatives.

## Equal time allowances and exact accounting

Use one CPU process, global external timeout+internal alarm1800 seconds. Each
arm/seed has120 seconds **accumulated active wall time**, including its own
initial gate forward, optimization, logging/serialization and final measurement.
Exclude time waiting while another arm runs. Initial gates for all ten arms
are completed first; then seed101 Adam/LM, seed102 Adam/LM, through105.
Common artifact admission, target/tape setup and optimizer restoration are
outside per-arm clocks but inside the whole-process cap. Report actual whole,
per-arm active wall and per-arm CPU times. Fixed sequential order and shared
report serialization can affect elapsed times; do not claim randomized timing.

Symmetrically reserve the final10 seconds of each120-second allowance for one
endpoint fit forward/backward plus one independent evaluation. Optimization
checks its110-second cumulative active boundary before each forward/trial/step;
do not accept proposals or Adam steps after that check fails. No work allowance
is extended. A120-second active timer and global1800-second deadline remain
armed. Signals interrupt at Python/native-return boundaries, not inside arbitrary
native instructions; observed times and partial operations must be reported.
An incomplete Jacobian/trial is retained as interrupted. If the hard arm limit
prevents endpoint completion, retain its state and classify
`incomplete_budget_limited`, with missing metrics explicitly absent. Continue
other arms only for expected per-arm budget exhaustion. Mandatory integrity,
current-state numerical or derivative failures stop the whole run. SIGTERM
always terminates the whole experiment. No automatic retry/extension.

| Maximum across all five seeds | Adam | LM |
| --- | --- | --- |
| Successful parameter steps | 3,750 | 400 |
| Outer/Jacobian constructions | 0 | 400 |
| Directional JVPs | 0 | 2,800 |
| Reverse sweeps including five endpoint gradients | 3,755 | 5,605 |
| Linear solve/trial attempts | 0 | 1,600 |
| Full forwards including initial, endpoint, evaluation | 3,765 | 4,415 |

One completed JVP costs one full forward and **two** reverse sweeps; do not
count it as a free derivative or equate LM iterations to Adam returns. Record
begun/completed forward, reverse, JVP, Jacobian and solve counts, accepted/rejected
steps, fit/evaluation points and completed-forward sample directions. Maximum
combined points787,680 (784,320 fit+3,360 evaluation),806,584,320 specular and
403,292,160 diffuse primal directions. Derivative sweeps do not redraw directions;
their cost is recorded separately. Interrupted/invalid attempts have separate
attempt counts; uncompleted forwards do not establish their actual generated
direction count. No new independent-reference evaluations.

## Endpoint evaluation and success

After the last accepted state/Adam return, measure original training-tape MSE
and latent gradient without updating parameters or state. Save all fit RGB
components. Evaluate all84 observations once per arm using original evaluation
tapes and the unchanged `evaluate_start` metric/split/success helper, with only
forward accounting wrapped. Report RMSE/MAE/relativeL2/max error for diffuse,
specular and total, on fit, held-out, relight, relight-seen and relight-unseen
views; save predictions/hashes and all physical/latent parameters. Held-out data
never participates in damping, step selection or stopping.

Unchanged all-required success: relative Q<=.05; identifiable gauge-equivalent
major-axis error<=5 degrees; maximum RGB albedo error<=.02; metallic error<=.03;
held-out total relativeL2<=.01. Use original physical gauges/denominator floors.
Report every one of the ten outcomes and failed criteria. Inspect saved
diffuse/specular residual cancellation and local sensitivity without new model
calls. A small gradient, damping saturation or good radiance fit does not prove
global optimality, unique material recovery or continuous-objective correctness.

## Files, tests and exact execution

- Implementation: `experiments/material_recovery_optimizer_comparison_v1.py`.
- New tests: `tests/scientific/test_material_recovery_optimizer_comparison_v1.py`.
- Exclusive-create evidence: `docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json`.
- Post-run report: `docs/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1_RESULTS.md`.
- Offline figures from saved evidence only, using existing plotting tools when
  available: `docs/figures/material-recovery-optimizer-time-v1.svg` and
  `docs/figures/material-recovery-optimizer-work-v1.svg`. Show all five paired
  loss curves versus measured active wall time and completed full forwards;
  separately report reverse/JVP counts rather than inventing a FLOP equivalence.

Run only NEW focused tests and essential static checks, with python3. Fixtures
must cover exact uninterrupted Adam restoration; residual/JVP/solve dimensions
and math; damping/ratio/rounding/saturation; invalid proposals; four-trial/step
caps; starting mismatches; metrics parity; budgets, collision and cleanup.
Protected scientific code/shader/old sources/protocols/evidence remain byte-
identical. The new protocol/code hashes are saved before scientific execution.
Keep the original workspace's user edit and use the separate feature worktree.

From `/home/isavand/paper/Nerofactor-3c4-optimizer-comparison`, run **once only**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_recovery_optimizer_comparison_v1 \
  --output /home/isavand/paper/Nerofactor-3c4-optimizer-comparison/docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json
```

Restore owned CPU thread settings/signal handlers, verify both global RNGs and
tapes unchanged, preserve partial evidence on failure, and never overwrite an
existing output. Interpret H1 optimizer limitations, H2 compensation, H3 finite-
objective limitations and H4 numerical/branch issues separately. Optimizer failure
alone cannot establish H3. Commit/push only this feature branch after reporting;
do not merge, change approved science or start the recommended next experiment.
