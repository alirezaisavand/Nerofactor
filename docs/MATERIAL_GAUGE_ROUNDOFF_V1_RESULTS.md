# 3C-9b: initial gauge roundoff and objective arithmetic

The single bounded diagnostic completed all six authorized forwards. All four
native baselines exactly reproduced 3C-9, including seed102's failed gate.
The failure is explained by a pre-existing RGB discrepancy plus float32
objective arithmetic. Replacing only the final MSE mean by an offline float64
mean would pass for the saved predictions. A separate encoding-only intervention
also passes for102, but does not remove all numerical differences. Neither result
changes the failed3C-9 evidence, accepted shader or original objective/gate.

## Execution, identities and scope

Base `f9043aaecb709cebde597eab970458c72ad0f2c5`, feature branch
`feat/material-gauge-roundoff-v1`, separate worktree
`/home/isavand/paper/Nerofactor-3c9b-gauge-roundoff`.
The [protocol](protocols/MATERIAL_GAUGE_ROUNDOFF_V1.md) and
[pre-execution freeze](protocols/MATERIAL_GAUGE_ROUNDOFF_V1.freeze.json) were
written before the ONE scientific invocation. Frozen SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Runner | `e0c68f771c52b8483150e7ccc9914d860c43ef890b95f293ad0783f95d0395f4` |
| New tests | `e3df23867354861e1c3e04e9d8e35c172f633ef201d041e312d4605453c924c2` |
| Protocol | `2a55539fc481c6c27c1b708fc2ee5fc3f38b05ebde7132db2ec86f7174ab93d4` |
| Primary JSON | `4753e1e6d971b033d24d686e146ab0d22dfd4941eee3d03df9780dc0fce37c5b` |
| Offline summary | `510775832a45f9ec1956f2cc6ab5fc27ce0a1f9d5a88a9ff53e8944a4dbf934c` |

[Primary evidence](evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json) stores the exact
baselines, captured array shapes/hashes, complete per-tape/mean RGB and numerical
comparisons. [Offline summary](evidence/MATERIAL_GAUGE_ROUNDOFF_V1_OFFLINE_SUMMARY.json)
derives a finer accumulation ladder and H1–H5 verdicts solely from that JSON;
zero additional shader/derivative calls. The primary report was not rewritten.

All48 chain-derived identities were admitted and rechecked after execution:
3C-1–3C-8 sources/protocols/targets/evidence, failed3C-5 files, and complete
failed3C-9 files/freeze. Frozen G/A and earlier experimental shader/sampling files
were preserved. The original user document/worktree was preserved. No datasets,
checkpoints or original results were loaded or changed.

Actual resources: **3.942658579s wall, 3.938773302s CPU, 563,556,352 bytes peak
RSS**. Six attempts/six completions,576 sampled points,294912 diffuse/589824
specular directions. No derivatives, backward, optimizers or updates. Runtime
matched Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121 on CPU, one thread, hidden
GPUs. Thread, profiler, memory and signal state were restored; global Torch and
NumPy RNG identities and original/copied tapes remained unchanged.

Only continuation update1000 states101/102 were used. Physical values and latent
coordinates matched3C-9 exactly; Q gauge relative differences were3.5258411e-8
and3.7387575e-8, within the original physical-equivalence bounds. Albedo and
metallic were identical. Both frame signs were+1. Coupled azimuth construction
matched the admitted helper and hashes exactly; maximum periodic float32 draw
rounding was2.9802322387695312e-8. No radial draws hit the source1e-6 clamp.
P1-U was read only as an archived reference, never rendered.

## Exact native replay and controlled results

All native mean RGB bytes, scalar MSEs, mask identities and active tapes matched
their archived3C-9 baselines. All six calls used unchanged ObservationBatch and
A2/atan2 arithmetic,24 original observations/four original tapes,512/1024 samples,
original geometry/lighting/targets and mean-before-MSE convention.

| Seed | P0 MSE | Native P1-C MSE | P1-C-E MSE |
| --- | ---: | ---: | ---: |
| 101 | 8.299173714476638e-6 | 8.299176442960743e-6 | 8.299176442960743e-6 |
| 102 | 8.895775863493327e-6 | 8.895770406525116e-6 | 8.895771316019818e-6 |

| Seed/arm | Absolute gap to P0 | Original bound | Result |
| --- | ---: | ---: | --- |
| 101/native | 2.7284841053187847e-12 | 5.1495868572383185e-12 | Pass |
| 102/native | 5.4569682106375694e-12 | 5.447887931746663e-12 | **Fail, reproduced** |
| 101/encoding intervention | 2.7284841053187847e-12 | 5.1495868572383185e-12 | Diagnostic pass |
| 102/encoding intervention | 4.547473508864641e-12 | 5.447887931746663e-12 | Diagnostic pass |

The native102 excess is exactly the prior9.080278890906499e-15. Float32 loss
spacing at these losses is9.094947017729282e-13: native101 differs by3 loss ULPs,
native102 by6. Its gate permits5.990016127776 ULPs. The encoding intervention
changes102's gap to5 loss ULPs. These quantized comparisons explain the tiny
distance past the gate without turning its historical failure into success.

## Earliest divergence and propagation

For BOTH states the float32 pi/2 scalar differs from mathematical pi/2 by
**4.371139006309477e-8 radians**. Adding that rounded scalar to the original
float32 theta is exact for these two states: subsequent addition error is0.
This constant conversion is the earliest identified representation-rounding
operation. Intended width exchange and quarter-turn frame changes are not
themselves numerical defects.

Native E1 differs from ideal `-E0` by one float32 encoding ULP (max5.9604645e-8).
There is one differing encoding component for101 and two for102. The original
normalization expands102's aligned error to1.1920929e-7 (two ULPs). Copies of
actual inputs, head normalization, decoder normalization, angle, sine/cosine,
tangent/bitangent, source samples, lights and weighted RGB are hash-recorded.

Gauge-aligned maximum errors; h/wi are compared in the same WORLD coordinates:

| Stage | 101/native | 102/native | 102/encoding intervention |
| --- | ---: | ---: | ---: |
| Encoding versus -E0 | 5.9604645e-8 | 5.9604645e-8 | 0 |
| Normalized encoding | 5.9604645e-8 | 1.1920929e-7 | 0 |
| Decoded cos correspondence | 0 | 5.9604645e-8 | 0 |
| Decoded sin correspondence | 2.9802322e-8 | 2.9802322e-8 | 0 |
| Tangent/bitangent numeric error | 2.9802322e-8 | 5.9604645e-8 | 0 |
| Local half-vector coeff_x | 2.9802322e-7 | 2.9802322e-7 | 2.9802322e-7 |
| Local half-vector coeff_y | 6.1094761e-7 | 6.4074993e-7 | 6.4074993e-7 |
| World half vector | 6.1839819e-7 | 6.7800283e-7 | 6.5565109e-7 |
| Reflected direction | 1.1464581e-6 | 1.2516975e-6 | 1.2218952e-6 |
| Incident RGB lighting | 1.6689301e-6 | 1.5497208e-6 | 1.5497208e-6 |
| Per-sample Fresnel RGB | 3.1071901e-4 | 2.8002262e-4 | 4.3678284e-4 |
| Weighted specular RGB | 5.0663948e-7 | 4.7683716e-7 | 4.6193600e-7 |
| Per-tape specular RGB | 2.9802322e-8 | 2.9802322e-8 | 2.9802322e-8 |
| Averaged specular RGB | 1.4901161e-8 | 1.4901161e-8 | 2.9802322e-8 |
| Averaged total RGB | 5.9604645e-8 | 2.9802322e-8 | 2.9802322e-8 |

Native maximum h/wi angular differences were6.2939737e-7/1.1508246e-6 radians
for101 and6.8017766e-7/1.2516975e-6 for102. After encoding intervention102 they
remain6.5797407e-7/1.2218951e-6. Large ULP distances near zero do not imply
large angular or radiometric error; per-tape RMSE/relative/ULP/count statistics
for every raw field and aligned stage are preserved in the JSON.

Diffuse directions, diffuse weighted RGB, per-tape diffuse RGB and mean diffuse
RGB are bitwise unchanged in all paired arms. All mask XOR counts are0; no
reflection-boundary samples occur at the predeclared1e-5 indicator. The mismatch
is therefore present in specular RGB BEFORE total-RGB averaging or MSE reduction.
The large Fresnel maximum is a per-sample observation, not evidence that it
dominates final RGB/MSE; its exact masked/spatial localization was not retained.

Auxiliary PDFs differ (native max errors1.08905 for101 and.125992 for102), with
875/1048 source-clamped wo-dot-h samples near zero. These are new observations,
not the43 historical3C-6 PDF-bound exceptions. The latter remain preserved.
The active source AST still does not read sampled pdf in optimized RGB; there
is no demonstrated PDF cause of this gate failure. PDF pre-clamp locals were
overwritten before return and were not captured or fabricated.

## What the encoding intervention establishes

P1-C-E changed ONLY the input encoding to the direct negation of captured P0's
float32 E0. The physical width/albedo/metallic inputs, samples, lighting, batch
and shader were unchanged. Independent ideal Q equivalence errors were at most
2.7755576e-17; actual rounded frame Q max errors were7.7092160e-9 for101 and
numerically0 for102. Zero-sign differences make102's Q/bitangent byte identity
false even though every numerical element is equal; no normal was flipped.

- **101:** normalization becomes exactly antisymmetric, but decoded coefficients,
  frames, samples, per-tape RGB, mean RGB and objective are bitwise unchanged
  from native P1-C. A one-ULP input encoding discrepancy does not necessarily
  survive decoder quantization.
- **102:** the aligned frame becomes numerically exact, sample vectors and RGB
  change, and the objective gap falls from6 to5 loss ULPs. This is a demonstrated
  one-factor effect of encoding construction under this unchanged finite shader.
  The total-mean RGB RMSE nevertheless increases from7.9511655e-9 to8.2369268e-9
  (max remains2.9802322e-8); mean-specular RMSE also increases. Improved objective
  agreement is not uniform improved RGB or physical accuracy.

Even with102's exact numeric frame correspondence, local half-vector coefficient
errors are unchanged and world h/wi differences persist. This localizes remaining
differences to source sampling/local coefficients/world arithmetic, beyond frame
encoding alone. It does not separately isolate float32 copied-v rounding from
atan2/trigonometric/summation rounding; no additional interventions were allowed.
Neither encoding construction nor a different accumulation precision is adopted
as a training change.

## Offline residual and objective decomposition

Original float32 MSE is exactly reconstructed from saved arrays before any new
shader call. For every observed P0/native/intervention residual, float32
subtraction equals subtraction of the stored float32 operands in float64:
**all72 residuals have zero subtraction-rounding error** in these states.

Signed differences below are candidate minus P0, not absolute values:

| Offline objective calculation, native C versus P0 | Seed101 | Seed102 |
| --- | ---: | ---: |
| Original float32 residual/square/mean | +2.7284841053e-12 | -5.4569682106e-12 |
| Only final mean in float64; squares remain float32 | +2.9751471350e-12 | -5.2662072243e-12 |
| Float32 residual promoted before square/mean | +2.9940324901e-12 | -5.3037450971e-12 |
| Stored RGB promoted before subtraction/square/mean | +2.9940324901e-12 | -5.3037450971e-12 |

For102, changing ONLY final-mean precision OFFLINE is sufficient to pass both
the fixed historical bound and the same formula evaluated on hypothetical P0.
The original failed MSE and original fitting objective remain unchanged.

For float32 residuals promoted to float64, the exact dyadic algebraic identity
`mean(r1²-r0²)=mean(2*r0*dr+dr²)` holds. Independently rounded float64 forms
differ by9.4115e-22/3.7646e-22, within the frozen float64 error bounds. They must
not be called bitwise equal. First-order/quadratic contributions respectively:

| Seed | mean(2*r0*dr) | mean(dr²) |
| --- | ---: | ---: |
| 101 | +2.9939677280586985e-12 | +6.47630097698008e-17 |
| 102 | -5.30380831846126e-12 | +6.32210333467103e-17 |

Thus residual correlation with the tiny RGB perturbation, rather than its
quadratic magnitude, controls the loss difference. For102 the largest individual
terms include observation16/view4/light0/G (-4.3475964e-12),
observation12/view3/light0/G (-2.9298169e-12) and observation18/view4/light2/G
(-2.5092860e-12). Other terms cancel them. All72 signed contributions,24
observation totals and top10 channels are in the evidence; no views were selected.

The short signed decomposition is:

| Contribution to native C−P0 objective | Seed101 | Seed102 |
| --- | ---: | ---: |
| Difference already in stored RGB, objective in float64 | +2.9940324901e-12 | -5.3037450971e-12 |
| Residual subtraction rounding | 0 | 0 |
| Squaring rounding | -1.8885355112e-14 | +3.7537872726e-14 |
| Final mean rounding | -2.4666302970e-13 | -1.9076098631e-13 |
| Original difference | +2.7284841053e-12 | -5.4569682106e-12 |

The RGB discrepancy accounts for most of the native102 difference; final mean
rounding is the incremental effect that crosses this narrow bound. It would be
incorrect to claim the RGB values were identical or that reduction caused all
the discrepancy.

## Earlier RGB accumulation effects

Native weighted sample sum/division and four-tape mean replay exactly from
captured values. Float64 alternatives below use only STORED float32 sample
contributions; none is a higher-precision shader evaluation.

For native102, float64 averaging of stored per-tape totals gives an objective
difference -7.5016340423e-12, which is LARGER than the original gap. Float64 sample
sum/component addition/mean together give -3.4588559078e-12. Higher precision at
one stage therefore need not monotonically improve cross-arm agreement.

The [offline summary](evidence/MATERIAL_GAUGE_ROUNDOFF_V1_OFFLINE_SUMMARY.json)
splits these stages using stored diffuse/specular components; their original
float32 addition reproduces per-tape total RGB exactly. A fixed telescoping
replacement ladder gives these signed contributions:

| Offline ladder contribution | Seed101/native | Seed102/native | Seed102/encoding intervention |
| --- | ---: | ---: | ---: |
| Captured sample arithmetic before sums, with subsequent sums/addition/mean in float64 | -1.0239786754e-12 | -3.4588559078e-12 | -3.8758642139e-12 |
| Sample sum/division rounding | -4.1060217394e-13 | -1.3217283069e-12 | -2.0764261079e-12 |
| Per-tape diffuse+specular addition rounding | +1.6590944452e-14 | -2.7210498276e-12 | -1.8324663855e-12 |
| Four-tape mean rounding | +4.4120223950e-12 | +2.1978889452e-12 | +2.7679009495e-12 |
| Residual subtraction rounding | 0 | 0 | 0 |
| Square rounding | -1.8885355112e-14 | +3.7537872726e-14 | +3.0918167942e-14 |
| Final mean rounding | -2.4666302970e-13 | -1.9076098631e-13 | +4.3846408099e-13 |
| Total | +2.7284841053e-12 | -5.4569682106e-12 | -4.5474735089e-12 |

These are order-dependent OFFLINE effects with substantial cancellation, not
independent causal percentages. The per-arm float32 sample-sum versus offline
float64 component errors are also recorded; diffuse sum errors are equal across
each pair and do not alone explain the cross-arm discrepancy. No earlier
per-sample transcendental operation was rerun in a new precision.

## Scientific verdicts and limits

| Hypothesis | Verdict | Scope |
| --- | --- | --- |
| H1 forward mismatch | **Confirmed** | Both native specular/total RGB differ before MSE |
| H2 objective amplification | **Confirmed** | Exact replay and residual identity explain measured MSE; final reduction tips102 past the bound |
| H3 independent encoding contributes | **Confirmed for102** | Single encoding intervention changes downstream results;101 unchanged downstream |
| H4 remaining sampling/accumulation effects | **Supported but not isolated** | Local coefficient/sample differences persist; recorded offline accumulation effects are nonzero; individual sampling/transcendental causes not intervened upon |
| H5 original gate is stricter than observed correspondence | **Confirmed for these states** | Native102 is physically equivalent within admitted bounds but fails the scalar gate; this does not authorize changing it |

No physical BRDF defect is established by remaining finite numerical differences.
No optimizer trajectory, Jacobian, material recovery, generalization or causal
historical solution-group result is produced here. Only two states and fixed
tapes were tested. No assertion of float32 equality for all representations or
future decoder branches follows.

The read-only hook observes actual return locals. Pre-clamp overwritten PDF
locals and exact internal arithmetic of every individual transcendental/sum
kernel are not traced. Sample arrays were retained in memory for all comparisons;
JSON stores their hashes/shapes/statistics, complete per-tape/mean RGB and small
parameter/encoding/frame arrays, not millions of raw sample values. This limits
further sample-wise localization without separately approved work.

## Future initial-equivalence criterion — recommendation only

Retain the exact original3C-9 failure and original scalar threshold as a
warning-level diagnostic in a FUTURE protocol, without retroactive passing.
Separate physical/gauge admission from byte-identical baseline replay:

1. Exact source/input/tape/observation identities and reproducible P0 controls;
   physical Q tolerance, albedo/metallic equality, chart/sign correctness.
2. Independent sample/frame/mask checks; diffuse RGB bitwise equality where
   required; separate specular AND total RGB bounds inherited or justified from
   numerical analysis. Total RGB compensation cannot hide component errors.
3. An objective discrepancy explained by recorded RGB perturbation plus an
   explicit arithmetic envelope, rather than only a fraction of a tiny MSE.

The error model was declared in the protocol BEFORE execution: with n=72,
u=2^-24 and gamma_k=k*u/(1-k*u), a conservative residual perturbation envelope
uses `epsilon_i=abs(delta_RGB_i)+u*(abs(P0_i-target_i)+abs(P1_i-target_i))`, with
underflow allowance if necessary. Its loss term is
`mean(2*abs(r0)*epsilon+epsilon²)`; a candidate square/mean roundoff envelope is
`gamma_(n+3)*(mean(r0²)+mean(r1²))`, pending reduction operation-count verification.
This derives from perturbation/roundoff bounds, not from choosing a constant
slightly above102's observed gap. It is NOT implemented as a new acceptance gate.

Such a worst-case envelope can be too loose and could hide source/configuration,
estimator or component-compensation defects unless the earlier independent checks
remain mandatory. Actual discrepancy and original warning must always be saved.
Do not silently promote the encoding intervention or float64 objective to a new
training implementation.

The evidence supports proposing a **new separately reviewed/versioned3C-9
trajectory attempt with unchanged native float32 shader/objective and original
LM**, after the initial-equivalence policy and arithmetic assumptions are
approved. All five initial states must still pass its frozen mandatory checks
before any optimization. Seeds103–105 and trajectory behavior remain untested.
No new trajectory attempt, criterion implementation, threshold fitting or
additional diagnostic was executed.

## Validation and changed scope

**25 new focused CPU tests passed,0 failed,0 skipped**, final pre-freeze run1.505s.
Zero full scientific shader calls in tests. New tests cover exact chain/state/
tape identities, ideal encoding/Q, argument-only intervention, synthetic exact
baseline rejection, mean order, residual identities/contributions, dtype/shape/
hashes, ULP/vector comparisons, resource/no-grad/call budgets, capture cleanup,
source preservation and strict partial JSON. Original suites/experiments were
not rerun. Static syntax, links, JSON/freeze hashes, whitespace and previous-file
preservation checks passed.

Changed scope: one new runner, one new test module, frozen protocol/manifest,
primary numerical evidence, offline-only summary, this results report and
AGENTS milestone authorization/completion. No original scientific source,
shader, sampler, YAML, dataset, checkpoint or accepted evidence changed.
