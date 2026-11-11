# Material symmetry audit v1 — approved attempt2 results

**The approved fresh audit completed. Physical symmetry passed, while axis
exchange changed the finite-sample fitting objective for all six states.**
At truth, P1 lowers MSE by 37.15%; at LM101–103 it raises MSE by 220.74–220.84%; at
LM104–105 it raises MSE by 100.00–100.11%. P2 (theta+pi) remains within the frozen
roundoff bounds in every case. These are fixed-state forward measurements,
not a new fitting run, selected representation or proof of optimizer-basin
causation. Accepted 3C-4 recovery outcomes remain 3/5, with no thresholds changed.

The first failed attempt remains preserved and labeled failed. This second
attempt was explicitly approved after its reporting defect was diagnosed;
it was not an automatic retry.

## Reporting repair, identity and execution

Base `13fe645af355f66400d435f9e25f5fcc63870912`; same feature branch
`feat/anisotropy-symmetry-audit-v1`. Only the analytical predicate's **output
type** changes (`numpy.bool_` to native `bool`); its truth value is unchanged.
An audit-local writer validates the full strict JSON payload before modifying
the prior snapshot. This does not claim atomic filesystem writes. All scientific
function bodies, shader, constants, transforms and bounds are unchanged.
Metadata/output names now identify attempt2; the disposable collision fixture
uses that new path. The first implementation is preserved in Git at the base.

The new regression reproduces the old serialization failure, verifies repaired
value equality, JSON-roundtrips nominal/rounded anisotropic/isotropic P0/P1/P2
records and a complete tiny sampling record, and checks previous-snapshot
preservation on invalid payloads. The original 21 focused assertions remain;
**27 current focused tests passed, 0 failed/skipped**, test-body time 1.771 s.
No previous scientific experiment or historical test suite was rerun.

| Artifact | SHA-256 |
| --- | --- |
| [Frozen attempt2 protocol](protocols/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.md) | 0fe9f41b299f848c661529a6c36d5b2d061ef1fb5f028873f42960021807dbac |
| [Repaired runner](../experiments/material_symmetry_audit_v1.py) | 53dce44200eb3278cf99561da0eb1d099e8c9da5bd15b7cc341149c86628fde9 |
| [Successful attempt2 evidence](evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json) | a7ab47837718eccb69e935c676796ada39aac367b021ce08d79533cb729299a1 |
| [Original scientific protocol](protocols/MATERIAL_SYMMETRY_AUDIT_V1.md) | 8e510d566d443ed44f3e0428c429837ccd70b6bcfba0951efd5ce6bcf1d4572a |
| [Preserved raw failed attempt](evidence/MATERIAL_SYMMETRY_AUDIT_V1.json) | 055aa0a2c13457dc198bb888c45ff15062fec90856de31220e09ada196568a55 |
| [Preserved failure manifest](evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json) | 47e879568455b593117304a89707a4216620fe6ca0e0a8985e4c98686f579b46 |

New evidence: 7,726,878 bytes. One approved execution, exactly
**36 completed shader calls**, 7,776 points,7,962,624 specular and 3,981,312 diffuse
primal directions. All 18 physical gates passed before the first shader call.
Pointwise analytical work was 1,620 kernel scalars and 540 ideal-PDF scalars,
with no dense integral or new targets. Numerical execution/setup took
**11.793675328s wall / 11.789069004s CPU**. The report timestamp
interval including admission/final serialization was 12.437149 s; library imports
before report creation were not separately measured but were covered by the
external 1,800-second timeout. Peak RSS was 570,937,344 bytes
(544.488281MiB), below 4 GiB, with the 8 GiB soft address cap.

CPU Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121, one thread, GPUs hidden. No CUDA
query, optimizer, parameter update, backward call, neural training, checkpoint
loading, Docker or package installation. Both global CPU RNGs and all tape bytes
remained unchanged; owned thread/signal settings and memory limits were restored.
No mandatory check failed and no extra attempt/sweep was performed. The exact
command and settings are frozen in the attempt2 protocol and JSON `argv`.
Version equality does not establish historical installed-binary identity.

## Physical equivalence and rounding

P0 is original, P1 swaps widths and adds pi/2, P2 retains widths and adds pi.
Albedo/metallic and all observations/lighting remain fixed. Mathematical transforms
are computed in float64, then rounded once to original float32 shader inputs;
angles are not canonicalized. Truth is the direct original physical material,
not its inverse-sigmoid roundtrip. The following are relative Frobenius/norm
errors against each state's own P0; they are **not recovery errors to truth**.

| State | Representation | Nominal Q relative difference | Independent kernel relative difference | Rounded-input Q relative difference |
| --- | --- | --- | --- | --- |
| truth | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| truth | P1 | 0.00000e+00 | 0.00000e+00 | 1.93360e-08 |
| truth | P2 | 6.43200e-17 | 3.25927e-17 | 3.38441e-08 |
| LM-101 | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| LM-101 | P1 | 4.58542e-17 | 7.91061e-17 | 1.68220e-08 |
| LM-101 | P2 | 2.10130e-16 | 1.60850e-16 | 3.84435e-08 |
| LM-102 | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| LM-102 | P1 | 6.48476e-17 | 8.98814e-17 | 1.68220e-08 |
| LM-102 | P2 | 2.15075e-16 | 1.41171e-16 | 3.84435e-08 |
| LM-103 | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| LM-103 | P1 | 1.02525e-16 | 7.28904e-17 | 9.13077e-08 |
| LM-103 | P2 | 2.15059e-16 | 2.16766e-16 | 1.05730e-07 |
| LM-104 | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| LM-104 | P1 | 7.23956e-17 | 2.89284e-16 | 5.39895e-08 |
| LM-104 | P2 | 1.72336e-16 | 7.06914e-17 | 1.07979e-07 |
| LM-105 | P0 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 |
| LM-105 | P1 | 5.91163e-17 | 1.03194e-16 | 5.39897e-08 |
| LM-105 | P2 | 1.25405e-16 | 1.31587e-16 | 3.92608e-08 |

All nominal Q/eigenvalue/orientation, independent ideal-kernel and common-world
ideal-PDF checks passed at the frozen bounds. Maximum relative nominal Q error
is 2.1507517662425826e-16, kernel error 2.892835008947909e-16 and ideal PDF error
3.012379562449256e-16. These kernel/PDF results cover the declared 15 incoming/
outgoing pairs and three RGB F0 channels, not a new hemisphere integration or
an exhaustive test of the clamped source PDF. Q symmetry also follows directly
from the declared rotation/tensor algebra.

Rounded-input Q errors reach 1.0797895708180565e-7; these float32 representation
errors are separate from nominal equivalence. Actual decoded T/B reconstruct
rounded Q within the original rounded-frame bounds. Complete original/transformed
seven-vectors, encodings, decoder coefficients, eigenvalues/orientations and
kernel/PDF values are in JSON. No float32 discrepancy was silently repaired.

## Fitting objectives: every state and representation

Original float32 fitting MSE is computed **after** averaging the same four fitting
tapes 0–3. It is not the mean of four tape-wise MSEs. All five LM P0 values match
accepted 3C-4 exactly. Objective differences use each state's own P0; the existing
absolute 1e-12 + relative 5e-7 roundoff bound is unchanged.

| State | L(P0) | L(P1) | L(P2) | Signed P1-P0 | P1 relative change % | Signed P2-P0 |
| --- | --- | --- | --- | --- | --- | --- |
| truth | 3.070781247061e-07 | 1.929867323724e-07 | 3.070786362969e-07 | -1.140913923e-07 | -37.153865 | 5.115907697e-13 |
| LM-101 | 1.270369978101e-07 | 4.074891819528e-07 | 1.270369978101e-07 | 2.804521841e-07 | 220.764178 | 0.000000000e+00 |
| LM-102 | 1.270366141171e-07 | 4.074614423644e-07 | 1.270366141171e-07 | 2.804248282e-07 | 220.743311 | 0.000000000e+00 |
| LM-103 | 1.270423410915e-07 | 4.076077573245e-07 | 1.270423410915e-07 | 2.805654162e-07 | 220.844022 | 0.000000000e+00 |
| LM-104 | 1.853012321362e-07 | 3.708114491019e-07 | 1.853011042385e-07 | 1.855102170e-07 | 100.112781 | -1.278976924e-13 |
| LM-105 | 1.853015589859e-07 | 3.706104507728e-07 | 1.853020989984e-07 | 1.853088918e-07 | 100.003957 | 5.400124792e-13 |

All six P1 objective differences exceed the frozen roundoff bound. The maximum
absolute P2 objective difference is 5.400124791776761e-13; all P2 differences are
inside the bound. A small-loss relative change alone is not the classification:
P2 can have a few-parts-per-million relative change while its absolute difference
remains compatible with the predeclared numerical bound. No representation was
chosen based on the lower training, held-out or relighting value.

The P0 means for the five LM states were reused only after array-hash and bitwise
identity checks. One baseline capture per state/batch was still necessary for
previously missing per-tape RGB and raw sampling arrays; no extra baseline was
run. Truth's older stacked-before-mean path was not assumed identical. Its one
counted baseline in fact matched the old mean bitwise in both batches.

## Radiance differences and per-tape behavior

Diffusive directions and diffuse RGB are **bitwise unchanged** for P1/P2 in this
case. Differences are concentrated in specular radiance. Four-tape mean total
RGB differences against P0 follow; the fitting column uses training tapes 0–3,
while held-out/relighting use independent evaluation tapes 1000–1003.

| State | Transform | Fit RMSE | Fit maximum absolute | Held-out RMSE | Relighting RMSE |
| --- | --- | --- | --- | --- | --- |
| truth | P1 | 8.067163417e-04 | 3.332495689e-03 | 4.433558789e-04 | 3.279210366e-04 |
| truth | P2 | 4.477241729e-09 | 2.980232239e-08 | 7.079140749e-09 | 2.323126661e-09 |
| LM-101 | P1 | 8.244588177e-04 | 3.401577473e-03 | 4.514895137e-04 | 3.340093537e-04 |
| LM-101 | P2 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 |
| LM-102 | P1 | 8.244571856e-04 | 3.401577473e-03 | 4.514875615e-04 | 3.340087322e-04 |
| LM-102 | P2 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 |
| LM-103 | P1 | 8.245038834e-04 | 3.401815891e-03 | 4.515193652e-04 | 3.340269377e-04 |
| LM-103 | P2 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 | 0.000000000e+00 |
| LM-104 | P1 | 8.337142258e-04 | 3.514885902e-03 | 4.698317269e-04 | 3.429270439e-04 |
| LM-104 | P2 | 4.728494278e-09 | 1.490116119e-08 | 7.359478369e-09 | 1.562518054e-09 |
| LM-105 | P1 | 8.336809770e-04 | 3.514707088e-03 | 4.698030989e-04 | 3.429118634e-04 |
| LM-105 | P2 | 5.483478846e-09 | 2.980232239e-08 | 3.196184257e-09 | 1.520843396e-09 |

Across all components, P1's largest four-tape mean absolute RGB difference is
0.0035149455070495605; largest individual-tape difference is 0.011120200157165527.
P2's largest absolute RGB difference is 5.960464477539063e-8 for both individual
and averaged tapes, with zero averaged RGB bound exceedances. LM101–103 P2
predictions are bitwise identical to P0. P1 has 3,876 averaged component/RGB values
beyond the bound across the declared training/evaluation comparisons; these are
finite-pattern observations, not statistical significance or population bias.

Individual-tape total-RGB difference RMSEs for P1 (ordered by the four declared
tapes within each batch) demonstrate that averaging and per-tape effects differ:

| State | Fit tapes 0–3 RMSE | Held-out tapes 1000–1003 RMSE | Relight tapes 1000–1003 RMSE |
| --- | --- | --- | --- |
| truth | 1.2796866e-03, 1.3815728e-03, 1.2567251e-03, 1.0729363e-03 | 1.4573863e-03, 2.0001484e-03, 1.1924869e-03, 1.3628250e-03 | 6.0320548e-04, 7.0462145e-04, 4.9834538e-04, 6.9085279e-04 |
| LM-101 | 1.3056532e-03, 1.4096193e-03, 1.2886857e-03, 1.0970764e-03 | 1.4896430e-03, 2.0355424e-03, 1.2126623e-03, 1.3897978e-03 | 6.2379498e-04, 7.1744533e-04, 5.0848826e-04, 7.0765782e-04 |
| LM-102 | 1.3056505e-03, 1.4096196e-03, 1.2886850e-03, 1.0970755e-03 | 1.4896361e-03, 2.0355462e-03, 1.2126623e-03, 1.3898012e-03 | 6.2379417e-04, 7.1744434e-04, 5.0848745e-04, 7.0765751e-04 |
| LM-103 | 1.3057151e-03, 1.4096993e-03, 1.2887667e-03, 1.0971567e-03 | 1.4897626e-03, 2.0356450e-03, 1.2127320e-03, 1.3898789e-03 | 6.2384886e-04, 7.1748113e-04, 5.0853354e-04, 7.0770160e-04 |
| LM-104 | 1.3160459e-03, 1.4370143e-03, 1.2871301e-03, 1.1153554e-03 | 1.5397163e-03, 2.0867400e-03, 1.2558879e-03, 1.4253420e-03 | 6.0933816e-04, 7.2964562e-04, 5.2935926e-04, 7.0914604e-04 |
| LM-105 | 1.3159966e-03, 1.4369350e-03, 1.2870718e-03, 1.1152896e-03 | 1.5396172e-03, 2.0866494e-03, 1.2558037e-03, 1.4252614e-03 | 6.0931428e-04, 7.2961354e-04, 5.2931630e-04, 7.0911874e-04 |

Full RMSE/MAE/relativeL2/max differences and target errors for diffuse/specular/
total, individual tapes and four-tape means, all RGB arrays and hashes are in
JSON. No tape, observation or failed criterion was excluded.

## Unchanged target metrics and generalization

Total linear-radiance target errors below are **percent relativeL2**, from the
original fine reference and independent evaluation tapes. These are local known-
material radiance errors, not image PSNR or historical scene scores.

| State | Representation | Fit target error % | Held-out target error % | Relighting target error % |
| --- | --- | --- | --- | --- |
| truth | P0 | 0.2467316 | 0.1855077 | 0.2941697 |
| truth | P1 | 0.1609307 | 0.1910881 | 0.3035683 |
| truth | P2 | 0.2467312 | 0.1855076 | 0.2941691 |
| LM-101 | P0 | 0.1817403 | 0.2296681 | 0.3846064 |
| LM-101 | P1 | 0.2451078 | 0.2542524 | 0.3204990 |
| LM-101 | P2 | 0.1817403 | 0.2296681 | 0.3846064 |
| LM-102 | P0 | 0.1817431 | 0.2296607 | 0.3846004 |
| LM-102 | P1 | 0.2451183 | 0.2542371 | 0.3204699 |
| LM-102 | P2 | 0.1817431 | 0.2296607 | 0.3846004 |
| LM-103 | P0 | 0.1817393 | 0.2297992 | 0.3846594 |
| LM-103 | P1 | 0.2450898 | 0.2543812 | 0.3206858 |
| LM-103 | P2 | 0.1817393 | 0.2297992 | 0.3846594 |
| LM-104 | P0 | 0.2797299 | 0.2974205 | 0.4192126 |
| LM-104 | P1 | 0.2632253 | 0.3115544 | 0.5029399 |
| LM-104 | P2 | 0.2797301 | 0.2974203 | 0.4192126 |
| LM-105 | P0 | 0.2797087 | 0.2972530 | 0.4188741 |
| LM-105 | P1 | 0.2631621 | 0.3113930 | 0.5027108 |
| LM-105 | P2 | 0.2797087 | 0.2972526 | 0.4188751 |

At truth, the better P1 fitting objective does not improve every evaluation
metric. For LM101–103, P1 worsens fitting and held-out errors while improving
relighting; for LM104–105 it worsens fitting objective and held-out/relighting
errors. These mixed observations forbid choosing a representation as an implicit
scientific improvement. The audit has no parameter fitting and does not change
the accepted recovery table.

## Sampling correspondence, masks, PDFs and actual frames

P1 changes the world incident directions/half-vectors generated by the **same
uniform-number indices**. Such sample-pair differences do not establish a
changed physical PDF. The ideal PDF at common world directions passed physical
invariance above. Raw source PDFs below are evaluated at different sampled
world directions; their absolute differences are not direct PDF-invariance
errors. High/clamped PDF values also make raw absolute errors scale-sensitive.

| State | P1 fit incident-direction RMSE | P1 half-vector RMSE | P1 indexwise source-PDF RMSE | Fit mask flips /98,304 | Evaluation mask flips /344,064 |
| --- | --- | --- | --- | --- | --- |
| truth | 0.4862334 | 0.2931505 | 2431.883915 | 11855 | 57311 |
| LM-101 | 0.4855844 | 0.2926808 | 2262.320538 | 11773 | 57029 |
| LM-102 | 0.4855844 | 0.2926809 | 2262.329964 | 11773 | 57029 |
| LM-103 | 0.4855981 | 0.2926909 | 2275.372363 | 11774 | 57033 |
| LM-104 | 0.4976601 | 0.3015607 | 3338.386451 | 12951 | 61053 |
| LM-105 | 0.4976452 | 0.3015497 | 3338.385091 | 12949 | 61047 |

| State | Fit P0/P1 accepted samples | Evaluation P0/P1 accepted samples | P1 first-frame alignment [T1·T0,T1·B0;B1·T0,B1·B0] |
| --- | --- | --- | --- |
| truth | 91250 / 91393 | 307990 / 308185 | [[0.0, -1.0], [1.0, 0.0]] |
| LM-101 | 91436 / 91297 | 308328 / 308145 | [[-0.0, 1.0], [-1.0, -0.0]] |
| LM-102 | 91436 / 91297 | 308328 / 308145 | [[0.0, 1.0], [-1.0, 0.0]] |
| LM-103 | 91436 / 91296 | 308326 / 308141 | [[0.0, 1.0], [-1.0, 0.0]] |
| LM-104 | 90585 / 90700 | 305406 / 305671 | [[0.0, -1.0], [1.0, 0.0]] |
| LM-105 | 90587 / 90700 | 305410 / 305673 | [[-0.0, -1.0], [1.0, -0.0]] |

P2 has zero mask flips in every batch; its source sample arrays are identical
for LM101–103 and differ only slightly for the other states. P1 T/B exchange
includes different signed quarter turns selected by the atan2 principal frame.
Those signs are observed unchanged, not aligned or flipped after rendering.
Their physical tensor remains equivalent despite altered random-index mapping.
No normal or tangent repair is applied.

Source widths and theta, double-angle encodings and actual decoder coefficients
are retained for all 18 representations. Samples have finite valid shapes and
unit-vector checks passed. Full valid counts, both mask-transition directions,
per-observation XORs, frame hashes, sample hashes and fixed excerpts are saved.
Millions of raw sample vectors were intentionally not saved; their full hashes
cannot be independently recomputed offline from excerpts. Arccos can report tiny
nonzero angles for bitwise-identical vectors because of diagnostic cosine
rounding; zero array differences/hashes, not that tiny angle, establish identity.

## Scientific interpretation and relationship to the LM groups

| Hypothesis | Assessment |
| --- | --- |
| H1: physical BRDF/Q invariance | **Confirmed in the declared algebra/probes.** Nominal Q/eigen/orientation and independent ideal kernel/PDF agree under P1/P2; rounded physical differences are separately bounded. No physical-model symmetry defect is demonstrated. |
| H2: finite-estimator non-invariance | **Confirmed for P1 with common unmodified tapes.** Different world samples, masks and finite specular predictions occur despite physical equivalence. This can occur for finite Monte Carlo/quadrature estimates and is not by itself an incorrect BRDF or proof of inconsistent expectation. P2 is roundoff-compatible in this audit. |
| H3: objective preference | **Confirmed.** Truth favors P1 by 37.15%, while all five LM endpoints have higher P1 MSE than P0, by 100.00–220.84%. This is a fixed-tape objective preference, not a data-selection policy or population-bias estimate. |
| H4: contribution to solution groups | **Plausible, causally unresolved.** Groups 101–103 and 104–105 react differently in magnitude and signed frame mapping; both currently favor P0. No optimizer was rerun with controlled gauge/sample coupling, so basin causation or removal is unproved. |

The two LM groups have different physical Q tensors. Applying P1 to an endpoint
preserves **that endpoint's** Q; it does not turn the 104/105 Q errors near 8.9%
into the 101–103 errors near 1.23%. Similar objective values or gauge changes must
not be used to reclassify those different materials as physically identical.
Accepted 3C-4 thresholds and 3/5 successes remain unchanged.

A finite common-random-number estimator need not map identical tape indices to
identical world samples after an axis exchange. This distinction plausibly
contributes to representation-dependent fitting landscapes, but no expectation,
large-sample convergence, global optimum or causal training effect was measured.
No unsupported claim of a shader bug, native-reconstruction benefit or repaired
scientific algorithm is made.

## Next single controlled experiment — not implemented

Prepare a separate **forward-only common-world-sample coupling check**. At the
same truth and saved LM states, compare the existing unchanged-tape P1 results
against an analytically derived coupling of the same base uniform draws under
axis exchange. Derive the required signed quarter-turn azimuth mapping from the
actual principal T/B frames and frozen Beckmann inversion before execution;
keep radial draws, targets, states, counts and physical shader fixed. Use only
copied diagnostic tapes, never alter accepted originals or silently canonicalize
the production sampler. Freeze all mappings/budgets/bounds and report every case.

If coupling restores sample/mask/radiance correspondence, it supports sample
correspondence as the mechanism; residual effects would motivate further clamp/
precision inspection. It would still not prove why the original optimizer found
its two groups. New sampling coupling is outside the current authorization:
no such experiment, implementation or optimization was performed here.

## Validation and preservation

The new tests include the exact prior NumPy-Boolean collision, strict full-record
JSON coverage and preservation of the last snapshot on serialization errors.
AST checks confirm unchanged scientific function bodies and the same orientation
predicate wrapped only for reporting. All 271 protected previous source/protocol/
evidence files, including the raw invalid first output, failure manifest,
original protocol and stopped report, retain their original bytes. The original
workspace's user-edited file contents were preserved. The original scientific
snapshots and accepted A2 shader/helpers remain unchanged.

Independent NumPy-only post-run checks validated all 18 records, nominal/rounded
input transforms, RGB and constant-frame hashes, all 7,200 target/difference scalar
metrics, mask count balances/transitions and every objective difference. Source
Torch and offline NumPy four-tape means matched exactly here. Maximum objective
reduction roundoff versus independent float64 arithmetic was
2.4761635649028004e-14, within the frozen operational bound. No shader/reference
kernel/optimizer was called during this offline verification. The new evidence
is strict valid JSON; the original failed output intentionally remains invalid.

Tests/source pass Python 3.8 AST parsing; runtime validation used Python 3.9.12.
Version labels are not historical binary hashes. These six fixed materials,
four fixed tapes and 15 ideal direction pairs do not establish population
statistics, global angular invariance of the clamped source implementation or
causal optimization effects. Full raw sample arrays were summarized, not archived.
No old scientific experiment, neural/GPU training, package installation or
additional sweep occurred.

Focused tests (already completed):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p 'test_material_symmetry*.py' -v
```

The reporting repair and successful fresh evidence are published on the existing
feature branch without merging. Stop here; no third attempt or further experiment
is authorized.
