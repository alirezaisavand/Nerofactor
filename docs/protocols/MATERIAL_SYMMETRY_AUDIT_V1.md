# Material symmetry and sampling audit v1

Protocol `material-symmetry-audit-v1`, milestone **3C-5**. Freeze this document
before the single authorized diagnostic; do not revise bounds after observing
results. Base `f4f5f4d57fc9f57c4d6387663ac9ef19b40144c0`, branch
`feat/anisotropy-symmetry-audit-v1`. This is forward-only: no optimizer creation,
parameter update, gradients, fitting, new target, dense integral or new sweep.

## Fixed scientific question and artifacts

Test the physical gauges `(mx,my,theta)`, `(my,mx,theta+pi/2)` and
`(mx,my,theta+pi)` separately from their finite common-tape estimates. A changed
sample correspondence is not automatically a changed physical distribution.
No representation replaces a prior result, and no favorable representation or
initialization is selected.

Admit the following exact accepted artifacts and every bound source/protocol
identity before numerical evaluation:

| Artifact | SHA-256 |
| --- | --- |
| `docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json` | `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451` |
| `docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json` | `6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f` |
| `docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json` | `224cb9a0cb27bbaacc61b8d575efbcd2c37854b716fc51bc78b439020239f24f` |
| `docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json` | `66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f` |
| `docs/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1_RESULTS.md` | `699fe50ebdfe2cd087147814787e8ee23b1267e3753c9ad92d5961382579d5f2` |

Reuse the existing read-only admission chain, never its optimizer restoration
or execution functions. Verify the accepted A2 shader, original observation/
metric helpers, independent reference kernel, pinned G/A manifests, all original
protocols and the comparison implementation hashes. Reuse the original converged
float64 targets and identifiability evidence without regenerating them.

Require the accepted CPU Python3.9.12, NumPy1.21.5 and Torch2.4.0+cu121 versions,
default float32, one Torch/BLAS/OpenMP thread and hidden GPUs. Do not query CUDA.
Version agreement is not historical binary identity. Preserve all accepted
artifacts and the original workspace user edit in a separate feature worktree.

## Exactly six states, three representations

Ordered states: original3C-1 nominal truth
`(.20,.55,.31,.30,.45,.60,.25)`, followed by final LM seeds101,102,103,104,105
read directly from3C-4. Validate each LM physical tensor's CPU/float32 dtype,
shape[7], exact round-trip values and saved SHA-256. Never reconstruct an
optimization trajectory or create a new initialization.

Parameter order is `mx,my,theta,k_R,k_G,k_B,metallic`. Form P0/P1/P2 in float64
from each admitted nominal state. P1 exchanges only the first two widths and
adds pi/2 to theta. P2 retains widths and adds pi. Do not wrap angles, canonicalize
axes or change albedo/metallic. Convert each resulting seven-vector once to the
original CPU float32 shader input. Record both nominal and actual float32 values,
double-angle encoding, decoder coefficients, actual T/B/N and frame-derived Q.
Truth uses direct float32 physical constants, not an inverse-sigmoid roundtrip.

## Mandatory physical equivalence before all shader forwards

Compute independently in float64:
`Q=R(theta) diag(mx^2,my^2) R(theta).T`, its ordered eigenvalues/eigenvectors,
major-axis orientation modulo pi, and the accepted independent Cartesian
Beckmann/Kurt alpha=0 kernel (`audits/anisotropic_brdf.py`). Use original RGB
`F0=.04*(1-metallic)+metallic*albedo`. This is the ideal-domain specular kernel,
not a new dense hemisphere integral or the clamped source estimator.
Also compare the ideal reflected-direction PDF at **common world directions**.

Fixed incoming raw vectors (normalize only to define these analytical probes):
`(.2,.4,1),(-.4,.1,1),(.6,-.3,1),(-.2,-.7,1),(.8,.2,.45)`.
Fixed outgoing raw vectors: `(.3,-.1,1),(-.5,.35,1),(.1,.6,.7)`.
Save the normalized directions and every kernel/PDF value. Fifteen pairs and
three F0 channels give45 scalar kernels per physical representation.

Nominal gates, unchanged after freeze:

- Q/eigenvalue elementwise `abs(delta)<=5e-14+5e-12*abs(P0)`.
- Kernel/ideal PDF elementwise `abs(delta)<=1e-13+1e-10*abs(P0)`.
- Identifiable major-axis angular error<=1e-11 radians. If either anisotropy
  gap is below.1, report orientation null/reason, never a fictitious zero score.

Separately evaluate physical Q/kernel/PDF at actual float32 parameters promoted
to float64. Rounded Q/eigenvalues must agree with rounded P0 within
`2e-7+2e-6*abs(P0)`; rounded kernel/orientation differences are descriptive,
not tested against the strict nominal-float64 tolerance. Record nominal-to-
rounded Q differences. Actual captured T/B must reconstruct the rounded Q
within the same rounded bound, with T0=+Y and B0=-X in the flat n=+Z frame.

All18 nominal and rounded physical checks must pass **before any** shader
evaluation. Failure marks the affected comparison invalid and stops the whole
run with partial evidence; never reinterpret failed physical equivalence as
finite sampling. Maximum pointwise work:1,620 kernel scalars and540 ideal PDF
scalars (nominal+rounded), no dense grids or new reference targets.

## Original finite shader and common tapes

Use only `MaterialShaderV1('A2','atan2')` with its original512 diffuse and1024
specular samples. Use the accepted `ObservationBatch` without rewriting its
averaging: original known flat geometry, normals, views, lights and observation
order; original training tapes0–3 and evaluation tapes1000–1003, including full
draw byte hashes. Preserve original `is_train=True` draw policy in these batches.
Every shade call occurs under `torch.no_grad()` with fixed CPU float32 inputs.

For each state/representation evaluate exactly:

1.24 fit observations x4 tapes=96 points, original training tapes.
2.All84 observations x4 tapes=336 points, original evaluation tapes.

An observer captures the **same returned** component radiances, sample directions,
half-vectors, PDFs, frames and acceptance masks. It does not change shader inputs,
operators or outputs, draw extra numbers, flip frames or resample a direction.
Mean RGB comes from original `ObservationBatch`; capture per-tape components
without changing the mean's operation ordering.

Saved-prediction policy:

- LM P0 fit/evaluation means from3C-4 have the identical admitted original
  batching path and exact physical input values. Verify stored array hashes
  and bitwise equality against the observer call, then reuse those saved values.
  A mismatch is fatal, with no fallback/retry.
-3C-2 truth predictions were stacked before averaging; exact arithmetic-order
  identity with original `ObservationBatch` is not assumed. Use one explicitly
  counted baseline forward and report its discrepancy to the old mean.
- P0 calls are still required to obtain missing raw sample/frame arrays and LM
  per-tape RGB. **No extra baseline forward** is performed. Reusing a mean does
  not claim avoidance of these new, necessary sampling-observation calls.
- P1/P2 use new declared forwards only, never another state's cached values.

## Measurements and frozen numerical comparison bounds

For all18 representations, save all per-tape/mean component RGB values, shapes,
dtypes/hashes, original/effective material values, encodings/decoder coefficients,
sample/frame hashes and deterministic first-four-sample excerpts per tape.
Save full per-tape/per-observation valid counts. Require finite sampled fields,
nonnegative PDFs, sampled vector lengths within2e-6 of one, and agreement between
captured masks and the original valid counts. Confirm returned diffuse+specular
equals total with atol2e-7/rtol2e-6 (float32 addition), without changing any value.

Against P0, report P1/P2 RMSE/MAE/relativeL2/max difference for diffuse/specular/
total, both each tape and the original four-tape mean, separately on fit,
held-out and relighting sets. Use the original metric helper and denominator
floor1e-4. Retain raw differences; an operational RGB roundoff-compatible bound
is elementwise `2e-6+2e-5*abs(P0)`. Count exceedances; exceeding it is an observed
finite-estimator difference, **not** a fatal integrity failure or significance
test for a population Monte Carlo expectation.

Compute fitting MSE exactly as before: Torch float32
`mean((four_tape_mean[:,total]-target_float32)^2)`. Also record each tape's MSE,
without confusing their average with the objective. Verify LM P0 objectives
equal saved3C-4 values exactly. Report signed/absolute/relative P1/P2 differences
with explicit relative denominator floor1e-12. The operational objective-roundoff
bound is `1e-12+5e-7*abs(L(P0))`; do not tune it after results. No comparison
bound changes the fitting objective or permits choosing a representation.

Report the unchanged target metrics on the full evaluation set for all three
components and each of fit/held-out/relight, including individual tapes. No
evaluation score influences sampling, thresholds or parameter selection.

Compare sampled directions/half-vectors/PDFs at identical **random indices**,
with global and each-tape errors and direction-angle summaries. This is sample
correspondence, not equality of physical PDFs at identical world events. The
independent common-direction PDF above supplies the latter check. Record mask
XOR and both transition directions per tape/observation; compare actual frame
vectors and their signed2x2 alignment to P0. Do not automatically align, sort,
negate or flip samples/frames to make them agree. Arccos angle diagnostics clip
only the computed diagnostic cosine to[-1,1]; original vectors are untouched.

## Budget, failure and outputs

One diagnostic, **1,800-second** external process timeout plus internal alarm,
one CPU thread and GPUs hidden. At most36 shader calls:6 states x3 representations
x2 batches. Exactly7,776 points,7,962,624 specular+3,981,312 diffuse primal
directions if complete. Count attempted/completed calls separately on failure.
No gradients or parameter updates, no retries/extensions/new seeds/increased
samples/additional sweep or follow-up experiment.

Memory: process soft address-space ceiling8GiB (or a stricter inherited limit),
leaving the inherited hard limit unchanged; restore the original soft limit on
exit. Check peak resident memory<=4GiB before/after forwards and analytical
blocks, and record peak RSS. These RSS checks are cooperative at Python/native
return boundaries; the address-space cap supplies an additional allocation bound.
Retain only the current phase's P0 raw arrays plus one comparison at a time;
save summaries/hashes rather than millions of raw sampling vectors. JSON RGB,
counts and physical probes remain complete. Any mandatory identity, physical,
finite/shape/component/mask or resource failure stops and preserves partial JSON.
No unreviewed scientific repair or retry. Restore owned threads, signals and
memory limits, verify both global CPU RNGs and all common tape bytes unchanged.

Implementation `experiments/material_symmetry_audit_v1.py`; new tests only
`tests/scientific/test_material_symmetry_audit_v1.py`. Exclusive-create
`docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json`, never overwrite. Post-run
analysis goes to `docs/MATERIAL_SYMMETRY_AUDIT_V1_RESULTS.md`; preserve this frozen
protocol and every older artifact. Use python3 for every Python command.

From `/home/isavand/paper/Nerofactor-3c5-symmetry-audit`, execute **once only**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_symmetry_audit_v1 \
  --output /home/isavand/paper/Nerofactor-3c5-symmetry-audit/docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json
```

Report every state/P0/P1/P2 side by side. Distinguish H1 physical invariance,
H2 finite-estimator non-invariance, H3 fixed-objective preference and H4 possible
relationship to LM groups. Neither similar losses nor a successful forward
proves identical physical Q, optimizer-basin causation or statistical bias.
Do not run a causal optimization study here. Commit/push this feature branch,
without merging, source corrections or starting the proposed next experiment.
