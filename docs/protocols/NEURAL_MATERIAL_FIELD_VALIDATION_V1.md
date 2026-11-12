# 3D-2 Phase C: bounded numerical validation v1

This protocol authorizes one bounded validation invocation, not recovery or a
final Phase D freeze. Base: `a11efd615d18ac69fa1c26a3692af511f1c78302`.
Preserve every Phase A/B file and all previous scientific evidence. No retry,
threshold/seed/radius change, calibration search, reference generation or GPU.

Before execution, run the new synthetic unit tests and `--check-plan`. Record
the exact runner/test/protocol hashes, Phase B blob identities, accepted102-file
chain, original G/A identities and current Git state in the owned run report.
These are validation provenance, not a final recovery protocol freeze.

## Fixed matrix and ordering

1. Source/field audit for seeds101,103,105: independently parse the pinned A
   definitions; compare original component parameters/structure/embedding and
   calibration centers. Probe N1 at origin, the original lexicographic nine
   plane points,96 repeats/truncations, then repeat the nine points. Compare
   D0 initialization. Seven field-only VJPs per seed into the eight original
   final biases establish local rank7 without a dense full-network Jacobian.
2. Initial admission, material-major M1/M2/M3, seed-major101/103/105, arm-major
   D0/N0/N1:27 real A2/atan2 forwards and27 original RGB-MSE backwards. Exactly
   original24 observations/four tapes3000–3003,512/1024 samples, known plane
   and lights. Truth only selects accepted targets, never initialization.
3. Deterministic replay: seed101 for each material, all three arms, fresh
   constructions and the exact same inputs:9 forwards/backwards, no updates.
   Require exact model, sample/component/loss and parameter-gradient identities.
4. Gauge probes for each of the three fresh D0 initial seeds, M1 target only:
   theta+pi with original tapes; swapped axes/theta+pi/2 with original tapes;
   same swapped material with the inherited signed quarter-turn coupled tape.
   Nine forwards, no backwards or updates. Baselines reuse saved M1 initial
   captures; no extra baseline rendering. Uncoupled sample/RGB differences are
   reported, not forced to agree. Periodic/coupled physical, frame, sample,
   mask, component and arithmetic gates retain the original tolerances.
5. Storage/optimizer fixture: one fresh model of each arm at seed101. One
   field-only sum-of-squared-physical-outputs backward and ONE original Adam
   step each, with no fitting targets or shader call. Serialize each own state,
   deliberately clear only that disposable model's parameters/optimizer state,
   restore via its explicit hash-pinned pickle-free manifest and require exact
   state equality. No synthetic-update state enters admission or recovery.

On a mandatory failure, preserve raw arrays and partial summaries immediately
and stop before later fixtures. Never substitute an initialization or rerun the
diagnostic after changing settings. Unexecuted matrix entries remain explicit.

## Exact planned work (also upper bounds for this invocation)

| Operation | Attempts/completions if all fixtures finish |
| --- | ---: |
| Full scientific shader forwards |45|
| Full renderer backwards |36|
| Field-only derivative probes |24 (21 final-bias VJPs +3 synthetic backwards)|
| Synthetic Adam steps |3 (one per arm); zero recovery updates|
| Explicit material field queries |63|
| D0 / N0 / N1 constructions |25 /13 /16|
| N1 fixed calibration passes |16, part of construction|
| Independent original component constructions/forward checks |3 /3|
| Shaded points / diffuse directions / specular directions |4320 /2211840 /4423680|
| LM, quadrature, historical optimizers, external recovery evaluations |0|

Separate attempted and completed counters precede/follow each operation, with
unique operation labels. Failed operations consume attempts; never retry them.
N1 construction includes one calibration pass. The source audit's three
independent original component checks are separately counted. No hidden shader
calls in unit tests, state identity generation or timing warmups.

Ceilings:300 process-wall seconds (external timeout includes Python imports),
4GiB peak RSS, one CPU thread, GPUs hidden; existing8GiB address backstop.
45/36/24 planned scientific operations are strictly below64/64/200 authorized
ceilings. All timing, capture, hashing and serialization share the300-second
window. No extension. Restore owned timers/handlers/limits/thread/RNG state.

## Unchanged gates and diagnostics

Six N1 physical scalars exact; radius calibration rho=.5 via original float32
operations, actual normalized two-channel encoding, no scalar re-encoding for
N1 shading. Q `2e-7+2e-6*abs(Q0)`; frame2e-6; h `2e-6+2e-5*abs(h0)`;
wi `4e-6+2e-5*abs(wi0)`; angular error<=2e-5; mask XOR0; diffuse directions
and diffuse components exact. Every-tape AND mean specular/total RGB
`2e-6+2e-5*abs(D0)`. D0/N0 all sample/PDF/component/loss/gradient values exact.
Original gamma75 arithmetic envelope and exact mean/MSE replay are mandatory;
old scalar loss bound remains warning-only. Original source/runtime rounding
assumptions must pass. All finite outputs and whole head/trunk connectivity
are mandatory; expected structural zeros are reported. Retain gradients at
the actual shader encoding input without changing its values or graph.

PDF differences for N1 or gauge comparisons are measured with worst indices,
actual values and relative differences, **without inventing a PDF equality
gate**: the inherited3D-1 protocol explicitly treats them diagnostically.
Normal/finite PDF outputs and exact D0/N0/replay PDF identity remain required.
Uncoupled Monte Carlo gauge differences are not BRDF inequivalence evidence.
Zero and near-zero encodings must be rejected by the existing decoder contract.
No normal flips, clamping additions, angle canonicalization or sample repair.

For every comparison report maximum absolute/relative errors, denominator
convention, worst-case indices and material/seed. Preserve full numeric return
arrays and gradients in hash-catalogued NPZ (`allow_pickle=False` when reading)
outside Git. Include initialization identities, raw rotation/radius, field and
shader-input norms, structural gradient zeros and all failure reasons.

## Feasibility and interpretation

Time source/construct, integrated forward/MSE, backward/gradient diagnostics,
synthetic Adam, checkpoint serialize/hash/restore separately. Estimate5000-step
cost conservatively using the largest observed N1 forward/backward/synthetic
Adam times plus observed admission/endpoint/checkpoint overhead. Report median
and maximum, costs of each arm, state bytes, checkpoint bytes and peak RSS.
This is a bounded prediction, not proof of completion within900 active seconds.
Logging/prefix-gate/OS variability remain uncertainty. If the estimate exceeds
the provisional900-second cap, declare feasibility blocked and recommend budget
review; do not alter it. Scientific numerical admission also requires every
prescribed fixture to finish and pass within the300-second validation ceiling.

Review Phase B gate implementations read-only. In particular, do not label
1000-step D0 reproduction or5000-step N0 parity as passed. Preserve the original
five physical recovery criteria, no learned geometry/lighting and no implicit
resume. Any clear implementation defect is reported/proposed for review rather
than corrected by modifying Phase B/scientific computation during this run.

Expected outputs: owned identity/report, input NPZ, per-fixture raw NPZ and
catalogues, own synthetic checkpoint manifests, versioned bounded summary JSON
and results document. No final Phase D protocol/freeze, full recovery or merge.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m experiments.validate_neural_material_recovery_v1 --check-plan

# One invocation only, after preflight; ROOT must be a fresh owned directory.
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5s 300s \
python3 -B -m experiments.validate_neural_material_recovery_v1 \
  --execute-once --output-root "$ROOT" --run-id "$RUN_ID"
```
