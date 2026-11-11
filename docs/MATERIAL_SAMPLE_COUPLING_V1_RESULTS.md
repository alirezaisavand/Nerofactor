# Common-world-sample coupling v1 — results

**Coupling restores the fixed-state RGB/objective symmetry within the frozen
bounds for all six materials. The strict PDF correspondence subcheck fails for
43 grazing samples; this residual is preserved, with no tolerance adjustment.**

The one approved 3C-6 execution completed its full 24 calls. `status=complete`
means the diagnostic ran with all mandatory integrity gates passed; it does not
mean every scientific comparison passed. Half-vectors and reflected directions
pass their declared bounds, masks have zero XOR, and diffuse RGB is bitwise
unchanged. Maximum fitting-objective difference falls from the archived
1.14e-7–2.81e-7 range to 6.821210263296962e-13, inside each frozen bound.

These fixed-state forward observations support random-sample correspondence as
the mechanism for the tested 3C-5 radiance/objective discrepancy. They do not
establish why the optimizer reached its two groups, correct an optimizer,
improve recovery, prove global source-PDF symmetry or prove unbiasedness.

## Frozen experiment and artifact identities

Base `1d16e9e147f6cdf270fbc0ed19e7add36ad17281`; branch
`feat/anisotropy-sample-coupling-v1`, separate worktree. No accepted scientific source, shader,
old protocol, previous evidence or original workspace content was changed.
The failed first 3C-5 artifact and failure manifest remain preserved byte-exact.

| Artifact | SHA-256 |
| --- | --- |
| [Frozen protocol](protocols/MATERIAL_SAMPLE_COUPLING_V1.md) | ca968222f7f67c2164b238b9da35145ad9ad8a4d785ea675282971680ad4611f |
| [Runner](../experiments/material_sample_coupling_v1.py) | 598bc1f55b40d83c89f7768a1d0a2008d70e8f5fdce08b7a58757028a015a49c |
| [Focused tests](../tests/scientific/test_material_sample_coupling_v1.py) | 6b46f801b28836d8d1f5a9e25ffbd8f94c02ac77027c0d75d5a98d9a4e59b2c3 |
| [New evidence](evidence/MATERIAL_SAMPLE_COUPLING_V1.json) | c8e9ddc1d0ab3c5d028bf9925062b6e2d27b9b6bc017800e7a2b00d92aebd87d |
| [Accepted 3C-5 attempt2](evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json) | a7ab47837718eccb69e935c676796ada39aac367b021ce08d79533cb729299a1 |
| [Accepted 3C-5 results](MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2_RESULTS.md) | ab785097f3a2f7e648bc51f7330dbda99853e54aa0c1dda375759b370864c6a8 |
| [Unchanged A2 shader](../experiments/material_shader_v1.py) | de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f |

The new JSON is 7,002,201 bytes, schema version 1,
`material-sample-coupling-v1`. Its accepted identities additionally bind the
original recovery/diagnostics/continuation/optimizer-comparison evidence,
source helpers, old protocols, both frozen G/A snapshot verification and original
combined/per-array tape hashes. It records the exact command, source base and
new source/test/protocol hashes. New source was uncommitted but hash-frozen at
execution; the publication commit contains these same frozen bytes.

One CPU execution, **8.12332270 s wall / 8.42306116 s CPU**
for execute/setup. Report timestamps span 8.922359 s including
admission/final serialization; imports preceding report creation were not timed
separately but were covered by the external timeout. Peak RSS was
552,263,680 bytes (526.67968750 MiB), below the
4 GiB cap. The 8 GiB soft address limit and inherited hard limit policy were
preserved/restored. No retry, optimizer, backward, new independent target,
reference quadrature, GPU query, training, Docker or installation occurred.

Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121; CPU, one Torch/BLAS/OpenMP
thread with GPUs hidden. Version equality is not historical binary identity.
Exactly **24 attempted/completed shader calls**, **5,184 points**,
**5,308,416 specular** and **2,654,208 diffuse** directions. P0 was recaptured
once per state/batch for raw sampling arrays, and every original serialized
field matched the accepted P0 record exactly, including each-tape prediction
hashes, samples/frames/masks and metrics. P1-U was reused from the admitted
3C-5 JSON; no P1-U or P2 arm was rerun.

## Analytical coupling and actual frame signs

The source inversion, despite its GGX method name, gives Beckmann local slopes
`q=sqrt(-log(max(u,1e-6)))*(mx*cos(alpha),my*sin(alpha))`, `alpha=2*pi*v`.
With world columns `F=[T,B]`, `h=(F*q+n)/sqrt(1+q.q)`. Let
`F1.T@F0=[[0,s],[-s,0]]`, verified from actual float32 decoded/source frames.
For exchanged widths, `alpha_c=alpha-s*pi/2` gives
`q_c=(s*q_y,-s*q_x)=(F1.T@F0)*q`; hence the same world h in exact arithmetic.
Use `u_c=u`, `v_c=(v-s/4) mod 1`. The original u clamp is retained, and the
source elevation denominator clamp is inactive algebraically for these widths.

The circle translation preserves Lebesgue-uniform azimuth measure and the
sample index order. This exact-real statement does not assume that float32
rounding is bijective. See the frozen protocol for the full derivation,
predeclared bounds and proof. No new sampling method is installed.

| State | Actual P0 principal angle (rad) | Actual P1 angle (rad) | s | v shift (turns) | max matrix error |
| --- | --- | --- | --- | --- | --- |
| truth | 0.309999992 | -1.260796306 | -1 | +0.25 | 2.838e-08 |
| LM-101 | -1.254158970 | 0.316637385 | 1 | -0.25 | 2.832e-08 |
| LM-102 | -1.254158857 | 0.316637470 | 1 | -0.25 | 6.246e-08 |
| LM-103 | -1.254135493 | 0.316660720 | 1 | -0.25 | 1.133e-07 |
| LM-104 | 0.305577035 | -1.265219217 | -1 | +0.25 | 7.478e-08 |
| LM-105 | 0.305575488 | -1.265220885 | -1 | +0.25 | 4.635e-08 |

Truth and LM104–105 use `s=-1`, M approximately `[[0,-1],[1,0]]`;
LM101–103 use `s=+1`, M approximately `[[0,1],[-1,0]]`. Both batches independently
match the accepted actual frame vectors. No fixed positive-frame assumption,
normal flip, frame alignment, sample sorting or parameter canonicalization was
used. Full F0/F1 vectors and both batch mappings are in JSON.

Each diagnostic tape owns cloned u/v/diffuse-azimuth storage. Only v receives the
quarter-turn. Original and copied radial/diffuse hashes remain equal; complete
accepted original tape hashes, global NumPy/Torch RNG state and all original
tape bytes remain unchanged after the run. Copied tapes also retain their hashes.
No rounded v reached endpoint 1 in the scientific run; pre-execution fixtures
cover that case and both signs, explicitly reducing periodic 1 to 0.

Maximum measured circular float32 v-rounding error: 2.980232238770e-08 turns.

## Original fitting objective: all six states

The unchanged Torch float32 objective averages four tape predictions **before**
MSE against the original float32 fitting target. P1-U and P1-C have exactly the
same material values/encoding/frame; only copied specular azimuth differs.
P1-U is archived, not recomputed. Positive differences increase MSE.

| State | L(P0) | L(P1-U) | L(P1-C) | P1-U minus P0 | P1-C minus P0 | Coupled relative change |
| --- | --- | --- | --- | --- | --- | --- |
| truth | 3.070781247061e-07 | 1.929867323724e-07 | 3.070782668146e-07 | -1.140913923336e-07 | 1.421085471520e-13 | 0.000046278% |
| LM-101 | 1.270369978101e-07 | 4.074891819528e-07 | 1.270370830753e-07 | 2.804521841426e-07 | 8.526512829121e-14 | 0.000067118% |
| LM-102 | 1.270366141171e-07 | 4.074614423644e-07 | 1.270371399187e-07 | 2.804248282473e-07 | 5.258016244625e-13 | 0.000413898% |
| LM-103 | 1.270423410915e-07 | 4.076077573245e-07 | 1.270416589705e-07 | 2.805654162330e-07 | -6.821210263297e-13 | -0.000536924% |
| LM-104 | 1.853012321362e-07 | 3.708114491019e-07 | 1.853014026665e-07 | 1.855102169657e-07 | 1.705302565824e-13 | 0.000092029% |
| LM-105 | 1.853015589859e-07 | 3.706104507728e-07 | 1.853020421549e-07 | 1.853088917869e-07 | 4.831690603169e-13 | 0.000260747% |

All six P1-U differences exceed the inherited
`1e-12+5e-7*abs(L(P0))` bound; all six P1-C differences lie inside it.
Maximum coupled absolute difference is **6.821210263296962e-13**, versus bounds
1.0635e-12–1.1536e-12. Relative differences alone can exceed 5e-7 when the
objective is tiny; the frozen **absolute plus relative** rule is unchanged.
No representation, initialization or checkpoint was selected based on loss or
held-out results. No weights or physical material values were optimized.

## World-space half-vectors, reflection and masks

All 12 state/batch comparisons pass h and wi component/angular bounds. Sample
pairs total 2,654,208 (the budget counts both arms, 5,308,416 specular directions).
Actual float32 trigonometry, quarter-shift rounding and frame decoding produce
small differences rather than bitwise-identical world vectors.

| State | Batch | h RMSE | h max abs | wi RMSE | wi max abs | Mask XOR | Valid samples / arm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truth | fit | 4.244754585825e-08 | 4.991888999939e-07 | 7.977974047684e-08 | 9.536743164062e-07 | 0 | 91250 |
| truth | evaluation | 4.239518778334e-08 | 5.513429641724e-07 | 7.756854760806e-08 | 1.013278961182e-06 | 0 | 307990 |
| LM-101 | fit | 4.034158208004e-08 | 6.034970283508e-07 | 7.614919632594e-08 | 1.192092895508e-06 | 0 | 91436 |
| LM-101 | evaluation | 4.046064864021e-08 | 6.034970283508e-07 | 7.433819489344e-08 | 1.102685928345e-06 | 0 | 308328 |
| LM-102 | fit | 4.004572357073e-08 | 5.960464477539e-07 | 7.585744578967e-08 | 1.177191734314e-06 | 0 | 91436 |
| LM-102 | evaluation | 4.015826262901e-08 | 5.960464477539e-07 | 7.390628070531e-08 | 1.132488250732e-06 | 0 | 308328 |
| LM-103 | fit | 4.753966685075e-08 | 5.587935447693e-07 | 8.793438231797e-08 | 1.132488250732e-06 | 0 | 91436 |
| LM-103 | evaluation | 4.751670932556e-08 | 5.587935447693e-07 | 8.555713739228e-08 | 1.013278961182e-06 | 0 | 308326 |
| LM-104 | fit | 4.659446026224e-08 | 5.960464477539e-07 | 8.631490077740e-08 | 9.834766387939e-07 | 0 | 90585 |
| LM-104 | evaluation | 4.649052930060e-08 | 5.960464477539e-07 | 8.392274305535e-08 | 1.013278961182e-06 | 0 | 305406 |
| LM-105 | fit | 4.690023743050e-08 | 5.215406417847e-07 | 8.629427572781e-08 | 8.884817361832e-07 | 0 | 90587 |
| LM-105 | evaluation | 4.683874026913e-08 | 5.215406417847e-07 | 8.402955997887e-08 | 9.834766387939e-07 | 0 | 305410 |

Global maximum h error: **6.034970283508e-07**, angular
6.385240787683e-07 rad. Maximum reflected-direction error:
**1.192092895508e-06**, angular 1.266952753223e-06 rad. All vector sample and
angular exceedance counts are zero at the predeclared bounds. Every mask,
per-tape/per-observation valid count and packed mask hash matches its P0.
Twenty-three pairs fall within the frozen reflection-boundary band
`abs(wi.n)<=2e-5`; none flips acceptance and none exceeds the PDF bound.
Rejected counts equal total-minus-valid and are saved for both arms.

The JSON includes h/wi/PDF MAE, relative L2, RMSE/max, per-tape metrics and
exceedance counts, stable atan2 angular summaries and mask transitions. Complete
sampling arrays are intentionally summarized by hashes/ranges/excerpts, not
archived; their hashes cannot be recomputed offline from the saved excerpts.

## Actual source PDF: preserved failed subchecks

The read-only PDF observer called the unchanged source method once per forward,
returned that same tensor and restored the owned dynamic class. Reconstructed
source operations matched the observed PDF **bitwise**. The reported errors
compare source PDFs at the coupled, approximately corresponding world h;
no ideal PDF replaces the source implementation.

| State | Batch | PDF RMSE | PDF max abs | PDF relative L2 | max elementwise relative | Exceeding PDF bound | Grazing pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truth | fit | 3.414379268460e-03 | 1.053039550781e+00 | 1.955202908121e-06 | 0.140938% | 2 | 6 |
| truth | evaluation | 6.835531898616e-03 | 2.247314453125e+00 | 3.023660891555e-07 | 0.136150% | 5 | 68 |
| LM-101 | fit | 1.934745591353e-03 | 5.802612304688e-01 | 1.222018329433e-06 | 0.066719% | 2 | 7 |
| LM-101 | evaluation | 9.534468539758e-03 | 4.224121093750e+00 | 4.268808215908e-07 | 0.208288% | 5 | 57 |
| LM-102 | fit | 7.459831291506e-03 | 2.162109375000e+00 | 4.711755958635e-06 | 0.100162% | 2 | 7 |
| LM-102 | evaluation | 2.021613577048e-02 | 1.084814453125e+01 | 9.051244507298e-07 | 0.295034% | 4 | 57 |
| LM-103 | fit | 1.741395855493e-02 | 5.457031250000e+00 | 1.099862504509e-05 | 0.112617% | 1 | 7 |
| LM-103 | evaluation | 3.368935579943e-02 | 1.609082031250e+01 | 1.508351860282e-06 | 0.526859% | 5 | 56 |
| LM-104 | fit | 1.729498160494e-03 | 4.753417968750e-01 | 7.215872315592e-07 | 0.035018% | 1 | 8 |
| LM-104 | evaluation | 2.218064610747e-01 | 1.300761718750e+02 | 8.749037062124e-06 | 0.552474% | 6 | 89 |
| LM-105 | fit | 8.404381680563e-04 | 7.031250000000e-02 | 3.506507813425e-07 | 0.011239% | 0 | 8 |
| LM-105 | evaluation | 5.384204317142e-02 | 3.110644531250e+01 | 2.123772643669e-06 | 0.323324% | 10 | 89 |

**43 of 2,654,208 sample pairs exceed the frozen PDF bound**
`1e-5+2e-4*abs(P0)` (about 0.00162%). All 43 are among the 459 pairs with
`abs(wo.h)<=1e-3`; none is in a clamped category, and no clamp-status flip occurs.
The maximum elementwise relative PDF difference is **0.5524737674714157%**;
maximum absolute difference is **130.076171875**. These maxima need not occur
at the same sample. Aggregate relative L2 reaches 1.0998625045088723e-5.
This is a **failed global PDF-bound subcheck**, not silently relabeled a pass.

| Frozen diagnostic category (overlaps allowed) | Paired samples | PDF exceedances |
| --- | --- | --- |
| cos_squared_clamped | 0 | 0 |
| exponent_clamped | 0 | 0 |
| dot_clamped | 34449 | 0 |
| denominator_clamped | 34451 | 0 |
| any_clamp | 34451 | 0 |
| grazing_wo_dot_h | 459 | 43 |
| near_reflection_boundary | 23 | 0 |
| outside_clamps_grazing_boundary | 2619497 | 0 |

In both arms, cos² and exponent clamps are inactive in this sample set.
There are 34,449 pairs with a raw dot clamp and 34,451 with a denominator clamp;
all these PDF comparisons stay inside the bound. The 2,619,497 pairs outside
clamp/grazing/reflection-boundary categories also have zero PDF exceedances.
The category rows overlap and should not be summed as a partition.

The localization is consistent with sub-micro world-vector rounding being
amplified by the `1/(wo.h)` reflected-direction PDF near a small positive dot.
It rules out an observed **clamp-branch change** or reflection-mask flip for the
43 outliers. It does not isolate each floating-point operation causally or
prove that no other arithmetic detail contributes. No precision rerun or extra
sample probe was authorized/performed. Raw PDF intermediate ranges, hashes,
clamp counts and grouped errors are retained, but full intermediate arrays and
individual outlier indices are not archived; finer retrospective localization
is limited by that deliberate bounded-output design.

## Radiance, held-out observations and unseen relighting

Diffuse directions and every per-tape diffuse RGB value are bitwise unchanged.
All coupled averaged RGB comparisons for diffuse/specular/total, fit/held-out/
relight, have **zero** frozen RGB-bound exceedances. Across both averaged and
individual-tape values, maximum absolute RGB difference is
**5.960464477539063e-8**. Residuals are in the specular component; no diffuse,
color-space, lighting, target or loss adjustment was made.

| State | Fit total RGB RMSE to P0 | Held-out RMSE to P0 | Relight RMSE to P0 | Max individual-tape component abs |
| --- | --- | --- | --- | --- |
| truth | 5.268356063862e-09 | 1.058234556830e-08 | 2.377795944006e-09 | 5.960464477539e-08 |
| LM-101 | 2.634178031931e-09 | 4.118463385050e-09 | 1.388333723880e-09 | 5.960464477539e-08 |
| LM-102 | 1.068205295384e-08 | 8.780593439770e-10 | 1.930399705364e-09 | 5.960464477539e-08 |
| LM-103 | 4.728494277760e-09 | 4.583604342066e-09 | 2.977644107809e-09 | 5.960464477539e-08 |
| LM-104 | 4.666949429355e-09 | 4.211024681184e-09 | 1.995855478982e-09 | 5.960464477539e-08 |
| LM-105 | 5.483478845611e-09 | 4.809329095244e-09 | 2.180464928525e-09 | 5.960464477539e-08 |

Independent-evaluation-tape **total-radiance relative L2 errors to the original
independent targets** follow. The fit column below uses evaluation tapes
1000–1003 on the fit observations, rather than fitting tapes 0–3. These scores
are reported without choosing a representation or claiming improved recovery.
P1-U entries come from accepted archived predictions. P1-C returns to P0 within
roundoff; it does not make the original finite target error zero.

| State | Arm | Fit observations | Held-out | Relight |
| --- | --- | --- | --- | --- |
| truth | P0 | 2.467316101423e-03 | 1.855076670496e-03 | 2.941697045397e-03 |
| truth | P1-U (archived) | 1.609307312293e-03 | 1.910881310274e-03 | 3.035682920160e-03 |
| truth | P1-C | 2.467312386936e-03 | 1.855073924125e-03 | 2.941694115846e-03 |
| LM-101 | P0 | 1.817402779511e-03 | 2.296680550716e-03 | 3.846063671858e-03 |
| LM-101 | P1-U (archived) | 2.451078166390e-03 | 2.542523943840e-03 | 3.204990318116e-03 |
| LM-101 | P1-C | 1.817399112500e-03 | 2.296681646706e-03 | 3.846065628022e-03 |
| LM-102 | P0 | 1.817430917968e-03 | 2.296607196506e-03 | 3.846003608234e-03 |
| LM-102 | P1-U (archived) | 2.451182723775e-03 | 2.542371262141e-03 | 3.204699486302e-03 |
| LM-102 | P1-C | 1.817428055671e-03 | 2.296607199241e-03 | 3.846005331425e-03 |
| LM-103 | P0 | 1.817392930738e-03 | 2.297992032358e-03 | 3.846594069685e-03 |
| LM-103 | P1-U (archived) | 2.450898069934e-03 | 2.543811916516e-03 | 3.206857750842e-03 |
| LM-103 | P1-C | 1.817390830223e-03 | 2.297988052421e-03 | 3.846596028778e-03 |
| LM-104 | P0 | 2.797299237176e-03 | 2.974204922875e-03 | 4.192125654371e-03 |
| LM-104 | P1-U (archived) | 2.632253330203e-03 | 3.115543600971e-03 | 5.029399208564e-03 |
| LM-104 | P1-C | 2.797298984209e-03 | 2.974202447067e-03 | 4.192135897341e-03 |
| LM-105 | P0 | 2.797087146052e-03 | 2.972529923246e-03 | 4.188741096807e-03 |
| LM-105 | P1-U (archived) | 2.631621096179e-03 | 3.113930032395e-03 | 5.027108443507e-03 |
| LM-105 | P1-C | 2.797086836676e-03 | 2.972533255472e-03 | 4.188754318637e-03 |

Component errors are preserved separately. The following coupled evaluation
means give diffuse/specular/total relative L2 to the original targets for the
held-out and relighting observations; all P0 and archived P1-U component metrics,
plus RMSE/MAE/max and individual-tape values, are retained in JSON.

| State | Split | Diffuse relative L2 | Specular relative L2 | Total relative L2 |
| --- | --- | --- | --- | --- |
| truth | held_out | 1.595063957904e-05 | 9.042159000828e-03 | 1.855073924125e-03 |
| truth | relight | 6.574172326382e-05 | 1.354797620286e-02 | 2.941694115846e-03 |
| LM-101 | held_out | 6.215097405992e-03 | 2.535650350145e-02 | 2.296681646706e-03 |
| LM-101 | relight | 6.213201231052e-03 | 2.733156842095e-02 | 3.846065628022e-03 |
| LM-102 | held_out | 6.214477598686e-03 | 2.535582883122e-02 | 2.296607199241e-03 |
| LM-102 | relight | 6.212531709324e-03 | 2.733101946659e-02 | 3.846005331425e-03 |
| LM-103 | held_out | 6.225891754521e-03 | 2.538995884791e-02 | 2.297988052421e-03 |
| LM-103 | relight | 6.224233892500e-03 | 2.735774693900e-02 | 3.846596028778e-03 |
| LM-104 | held_out | 3.478620965321e-03 | 2.098882300013e-02 | 2.974202447067e-03 |
| LM-104 | relight | 3.477133297433e-03 | 2.123082619741e-02 | 4.192135897341e-03 |
| LM-105 | held_out | 3.472481025912e-03 | 2.096705320156e-02 | 2.972533255472e-03 |
| LM-105 | relight | 3.471043873893e-03 | 2.120762878052e-02 | 4.188754318637e-03 |

## Hypothesis assessment and scope

| Hypothesis | Assessment |
| --- | --- |
| H1: measure-preserving sample correspondence | Established analytically for the source ideal inversion and declared signed frames. Both signs and wrapping pass independent float64/tiny-A2 fixtures. Quantization is separately recorded. |
| H2: finite-sample correspondence | h/wi bounds and all masks pass for every state/batch. **PDF subcheck fails for 43 grazing pairs**; do not claim exact or uniform source-PDF correspondence. |
| H3: objective symmetry | Passed at the original frozen absolute-plus-relative bounds for all six states. RGB symmetry likewise restored within bounds. |
| H4: numerical residuals | Measured h/wi float32 differences; PDF bound failures localized to grazing, unclamped pairs, with no clamp/mask flips. Denominator sensitivity is a supported explanation, not an operation-level causal proof. |
| H5: LM solution groups | Unresolved. No optimization or derivative experiment was performed; a demonstrated forward mechanism does not identify optimizer-basin causation. |

The groups remain different physical Q tensors. Coupling preserves each
state's own Q and does not turn LM104–105 into LM101–103 or into truth.
Accepted 3C-4 material-recovery outcomes remain **3/5**. No new method,
unbiasedness result, global angular/PDF guarantee, shader fix, native material
reconstruction or publication-quality recovery claim follows from this audit.

## Tests, verification and preservation

**19 new focused tests passed, 0 failed, 0 skipped**, test-body time 1.701 s.
Only the new file was run; no historical scientific suite/experiment was rerun.
The first development test run exposed the float64 analogue of a modulo endpoint
rounding to 1 in an analytical fixture. The fixture now explicitly reduces that
periodic endpoint to 0, consistently with the documented float32 copy policy;
this was before protocol/source freeze and before the single scientific run.
No run threshold or shader calculation was changed after observing outcomes.

The fixtures cover Q symmetry, both actual decoded signs and full archived batch
frames with no shader forward, independent float64 half-vector equality,
uniform-grid interval measures, wraps/endpoints, radial/diffuse preservation,
disjoint copy storage, tiny real A2 h/wi/PDF/masks, full successful strict JSON,
exact P0 hashes, source PDF observation/cleanup, failed admission, resource/call
limits, collision rejection, failure persistence and no optimizer/backward/RNG.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p 'test_material_sample_coupling_v1.py' -v
```

Both new Python files pass Python 3.8 AST parsing; actual runtime tests used
Python 3.9.12. Markdown links, strict successful JSON, frozen identities and
Git whitespace/scope checks were inspected. A preservation manifest checked
586 pre-existing tracked files plus the original workspace document, and the
three newly frozen code/test/protocol files: **590 hashes unchanged** after
execution. AGENTS.md is intentionally updated for authorization/completion.

Independent post-run NumPy/stdlib-only saved-array verification checked all
six states, exact P0/archived P1-U identities, P1-C material/frame parity,
RGB array hashes, **7,200 target/difference scalar metrics**, objectives,
roundoff classifications, original/copy identity records, mask balances and
all declared work/cleanup flags. It invoked no Torch, shader, kernel, RNG or
optimizer. NumPy four-tape means matched recorded Torch means exactly; maximum
objective reduction difference versus float64 arithmetic was
2.4761635649028004e-14. Full raw h/wi/PDF arrays cannot be rechecked offline
from the intentionally saved hashes/excerpts, and the report does not claim it.

## Next single controlled experiment — not implemented

Request a bounded **gradient/Jacobian gauge-equivariance diagnostic** before any
optimizer change: at these same six materials and original batches, compare
P0 derivatives with P1-C derivatives under the exact width-permutation chain
rule and locally fixed derived tape coupling. Preregister how decoder branch,
mask boundaries and grazing-PDF cases are classified; preserve all targets,
samples, states and the accepted shader. This would test whether forward
roundoff agreement extends to the derivatives actually used in inverse fitting.
It would not itself prove the causal origin of the two LM solution groups.

That proposal needs separate approval because the present milestone forbids
backward/gradient calculations. No derivative implementation, optimization,
additional sweep or native/GPU experiment was begun. Commit/push only this
feature branch, do not merge, and stop for review.
