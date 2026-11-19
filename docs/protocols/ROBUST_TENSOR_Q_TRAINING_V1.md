# Robust transported tensor regularization v1

The paired batch tests whether attenuating high-residual surface pairs improves
roughness-boundary recovery. The primary endpoint is update 10,000 in both
arms. All evaluations use the admitted heterogeneous references.

| Arm | Matched control | Existing distance weight | Frozen delta |
| --- | --- | --- | ---: |
| Donut-R1 | Heterogeneous Donut D6 | Squared reference-distance ratio | 0.030024479708747484 |
| Bell-R1 | Heterogeneous Bell B2 | Same ratio, capped at four Bell medians and scaled to the Donut median | 0.008516476041898608 |

Apply the penalty after weighting. For each sampled pair let `s = w * ||Q0 − R Q1 Rᵀ||²_F`
where R is the existing minimal normal transport. The new objective uses
`0.05 * mean(rho(s))` in place of `0.05 * mean(s)`.
Here `rho(s) = 2s / (sqrt(1 + s/delta²) + 1)`. The implementation evaluates
`s * (2delta / (sqrt(s + delta²) + delta))` to avoid overflow from either
`2s` or `s/delta²`. The multiplier on a pair's original gradient is
`delta / sqrt(s + delta²)`. It is one at agreement, one over square root
of two at sqrt(s) = delta, and decays as the discrepancy grows. The value
approaches a linear penalty in the weighted residual norm at large discrepancy.
Changing this weighting of residuals can attenuate large smooth-region errors
as well as true boundary errors.

## Frozen fitting-only scale

At the accepted controls' fixed update-10,000 endpoints, use their archived
prediction-only squared transported differences on the existing reserved
geometry edges. Include an edge when both endpoints have a valid fitting-camera
surface observation within 0.01 native units and with a shading normal within
30 degrees of that endpoint normal. The two supporting observations can come
from different fitting cameras. Compute the 90th percentile of the resulting
`sqrt(w * transported_square)` using linear interpolation and require at least
32 usable edges per scene. Use the actual Donut D6/Bell B2 weights in this
calculation. The selected counts are 695 for Donut and 647 for Bell. These
residual- and support-based scales are proxies for typical local variation, and
some selected edges may straddle real boundaries. The calibration reads positions,
normals, validity, geometry-only edge information and predictions. Oracle
materials, boundary labels, held-out observations and recovery errors play no
part in the scale selection. The input identities and selected pair IDs are in
[the calibration evidence](../evidence/ROBUST_TENSOR_Q_V1_CALIBRATION.json).
The host launcher replays this calculation before native dispatch.

## Matched execution and measurement

Each arm uses its control's admitted dataset and camera split, historical
fresh initialization identity, full chart material field, corrected native
shader and visibility, free learned light, legacy priors, seed, original Adam
and schedule, photometric RNG, geometry-only sampler stream and normal transport.
The Donut domain uses smooth interpolated normals and Bell uses its existing
faceted-normal domain. Compatibility, initial-state and sampler gates must pass
as in the controls. Keep initial evaluation, eight-draw evaluation, checkpoint
roundtrip and maps on the existing schedule, with training checkpoints at
2,000, 4,000, 6,000, 8,000 and 10,000 updates. Run Donut-R1 first, then Bell-R1.

Per-update logs record the robust loss, quadratic counterfactual loss, their
ratio, mean and minimum gradient multiplier, and fraction of pairs above
delta. At update 1 and every 2,000 updates, measure the robust and quadratic
network-gradient norms, group norms and cosines relative to each other and RGB
on the same sampled batch and state. These measurements add one reverse sweep
at each of the six existing diagnostic updates per arm. They use the same
forward graph and do not write parameter gradients or use RNG. The optimizer
uses the robust objective. Record native timings, memory, coverage and Bell's
existing shortest-edge diagnostics.

The freeze binds the existing datasets, exact control recipes and the new
sources. The total planned and enforced counters include 20,000 Adam returns,
24,768 shader calls, 19,217,227,776 shadow rays, 2,560,000 sampled training
pairs, 72 diagnostic reverse sweeps, and 12 surface-boundary predictor calls
on Bell at the existing 2,474 probes per side-call. Use one existing corrected
image, GPU 0, four CPU threads, 64 GiB RAM without swap, 128 PIDs, the existing
25% CUDA allocator fraction and 8 GiB output cap per arm. Run in one foreground
7,500-second watchdog with 30 seconds of kill grace. Each arm has a 3,600-second
host limit and a 3,540-second inner limit with 20 seconds of inner kill grace.
Source, input and root mounts are read-only. Run non-root and networkless,
with a new batch root. Stop and preserve partials on failure. No retries,
coefficient sweeps, historical reruns, new references or training extensions.

## Evaluation and interpretation

Compare Donut-R1 against Donut D6 and D2, and Bell-R1 against Bell B2 and D2.
Use the existing signed width-jump and projected Q-jump tests, boundary-band
and smooth-interior physical errors, near-isotropic cohorts, fitting-view and
highlight-support strata, reserved-edge Q consistency, width saturation,
orientation eligibility, scalar material, illumination and linear RGB metrics.
Report observed-triangle-area weighting alongside observation weighting.
Use matched orientation-eligible observations for comparisons sensitive to
changes in eligibility.

Plot signed transition profiles using fixed boundary-distance magnitudes
`[0, .005, .01, .02, .035]` in native units, positive for region 0 and negative
for region 1. Group by region even at zero distance, use lower-inclusive bins
with the final upper endpoint included, and report empty bins explicitly.
Compute per-bin mean predicted and oracle sorted widths and signed width errors
separately for fitting and held-out observations. Use the paired jump probes
to measure abruptness independently of these aggregate profiles. Region,
boundary and reference-material labels are evaluation-only.

Improved signed jumps and Q projection alongside retained smooth-interior
accuracy and RGB would support the robust method. Interpret boundary changes
alongside attenuation, same-batch gradient scales and reserved-edge variation.
This design cannot establish an advantage over a separately gradient-matched
weaker quadratic penalty. Material-field expressivity and area-consistent
mesh discretization remain separate investigations.
