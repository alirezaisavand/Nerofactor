# Controlled material recovery continuation v1

Protocol: `material-recovery-continuation-v1`, milestone **3C-3**. Frozen before
the single authorized execution. Base `15d0a07e65e5d3f18f72d68819b62e386886c492`;
branch `feat/material-recovery-continuation-v1`. Do not revise this protocol
in response to observations. This is seven-scalar CPU inverse rendering, not
neural training, historical reconstruction or a new shader comparison.

## Hypothesis and controlled change

Test whether the five unsuccessful 3C-1 recoveries were mainly update-budget
limited. The **only experimental change** is the endpoint, from 250 to 1,000
total Adam returns. Continue seeds 101,102,103,104,105 in that order, exactly
750 additional returns each (3,750 total). Never repeat their first 250 updates,
generate new starts, restart Adam, select an iterate or initialization by
held-out scores, or silently resume this experiment after interruption.

## Admission and exact restoration

Admit the unchanged `docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json`, SHA-256
`f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451`.
Verify original scientific/source/protocol identities, passed convergence and
seven-parameter identifiability gates, complete seeds and update counters.
Reuse all accepted independent targets and gates; do not regenerate references
or execute previous experiments or test suites.

Also admit the unchanged 3C-2 diagnostic JSON, SHA-256
`6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f`,
and its source/protocol hashes. Reuse its exact terminal fit objectives and
latent gradients to describe update 250, without replaying those states.
Update-250 held-out/relighting/component scores come from original 3C-1.

Require matching original Python 3.9.12, NumPy 1.21.5 and Torch 2.4.0+cu121
versions, CPU, default float32, one thread. The CUDA-linked distribution is
used on CPU with GPUs hidden; no CUDA operations or queries are authorized.
Version equality does not establish original installed binary/library hashes.

Restore each `last_unconstrained` as one CPU float32 tensor of shape [7],
requiring every JSON number to round-trip exactly through float32. Recompute
physical values using the unchanged original transform and require exact equality
with `final` and `last_physical`. Require completed update 250 and status complete.

Restore Adam's scalar CPU float32 `step=250` and float32 [7] `exp_avg` and
`exp_avg_sq`. Require exactly these fields, finite values, nonnegative second
moments and exact float32 JSON round trips. Construct Adam with the original
effective flags, use `load_state_dict`, then require exact tensor shape, dtype,
device and value equality plus exact effective optimizer configuration.
Record byte hashes, values, dtypes/shapes and settings for all five restorations
**before any shader forward or update**. Any failed restoration stops the run.

The original JSON has numerical values, not dtype tags. CPU float32 moments and
step representation are established from the admitted original runner's float32
parameter construction, unchanged Adam flags/default dtype, and matching Torch
version. This is source-supported reconstruction of representation, not a claim
that dtype metadata was saved. Focused synthetic tests must demonstrate bitwise
equality to an uninterrupted JSON-round-tripped Adam continuation. No real-data
prefix replay is allowed. Old `.grad` tensors are unnecessary: original
`zero_grad()` occurs before every forward. No scheduler/grad-scaler/sampler state
exists here; original global RNG remained unchanged and all tapes are explicit.

## Unchanged computation

- Accepted `MaterialShaderV1('A2','atan2')`, same source identity, float32 math,
  parameter transforms, angle encoding and midpoint diffuse heights.
- Original flat scene, known lighting/visibility, 24 fitting observations and
  original independent float64 targets cast identically to float32 for MSE.
- Original seed-major fixed training tapes 0..3, 512 diffuse/1,024 specular
  samples. Verify full draw hashes and observation ordering before execution.
- Average four tape predictions **before** linear-RGB MSE. Reuse original
  `ObservationBatch` and physical parameter transform without changes.
- Adam lr .01, betas (.9,.999), eps 1e-8, weight decay 0; amsgrad/maximize/
  capturable/differentiable false, foreach/fused None. No regularization,
  scheduler, clipping, noise, decoder/sampling changes or target changes.

Run updates 251..1,000 inclusive. Count only successful `optimizer.step()`
returns; require Adam counter and experiment counter agree at every return.
Fail immediately on nonfinite predictions, objective, gradient, parameter or
optimizer state. Log update 251 and each tenth update thereafter; record loss
and gradient at **update-1**, then post-update physical/unconstrained values and
the latest Adam state. Preserve failure/partial state, never retry automatically.

After update 1,000, perform one extra fitting forward/backward **without a step**
to record the actual endpoint objective and latent gradient. This measurement
does not change any parameter or Adam state. Evaluate all 84 observations once
per start with the unchanged original tapes 1000..1003, using original evaluation
and success helpers. No new reference integrals. Save all predictions and hashes,
all five successes/failures and diffuse/specular/total metrics on fit, held-out,
relight, relight-seen and relight-unseen views. Fit metrics from evaluation tapes
and the training-tape objective are separately labeled.

Success requires **all** original bounds: relative Q error <= .05,
identifiable gauge-equivalent orientation error <= 5 degrees, maximum RGB
albedo absolute error <= .02, metallic absolute error <= .03 and held-out total
relative L2 <= .01. Preserve the original gauge, axis and denominator rules.
Report all terminal parameters, sorted roughness errors, Q/orientation/albedo/
metallic errors, endpoint gradient, trajectory and update250-to1000 comparisons.
No threshold, hyperparameter or report-selection changes after results.

## Fixed budget, files and execution

One CPU process, one thread, **1,800 seconds maximum wall time**, external timeout
plus internal alarm. No automatic retry or extension. Maximum work:

| Operation | Cap |
| --- | --- |
| Additional optimizer returns | 5 x 750 = 3,750 |
| Fitting forwards for updates | 3,750 x 96 = 360,000 points |
| Endpoint fitting gradients, no update | 5 x 96 = 480 points |
| Total fit directions | 369,131,520 specular + 184,565,760 diffuse |
| Endpoint evaluation | 5 x 336 = 1,680 points |
| New reference directions/prefix replays | zero |

Implementation: `experiments/material_recovery_continuation_v1.py`.
New tests only: `tests/scientific/test_material_recovery_continuation_v1.py`.
Exclusive-create output `docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json`;
never overwrite or append a second attempt. It includes original hashes, reused
gates, frozen protocol/code hashes, source/tape/runtime identities, all restoration
records, trajectories, endpoint states, comparisons, counters, times and failures.
Write the post-run analysis separately to
`docs/MATERIAL_RECOVERY_CONTINUATION_V1_RESULTS.md`; never change old evidence.
Restore owned thread settings and signal handlers and verify global CPU RNGs
and fixed tape bytes are unchanged. Reuse the isolated feature worktree to
preserve the existing user edit in the original repository.

From `/home/isavand/paper/Nerofactor-3c3-continuation`, execute **once only**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_recovery_continuation_v1 \
  --input /home/isavand/paper/Nerofactor-3c3-continuation/docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json \
  --diagnostic /home/isavand/paper/Nerofactor-3c3-continuation/docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json \
  --output /home/isavand/paper/Nerofactor-3c3-continuation/docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json
```

Interpretation: additional successful recovery would support insufficient duration
in this bounded known-material case; remaining failures do not alone establish
global ambiguity, a shader defect or an irreducible estimator error. Finite-tape
forward/gradient and branch artifacts documented in 3C-2 remain relevant. No
native reconstruction, production improvement or next experiment is authorized.
