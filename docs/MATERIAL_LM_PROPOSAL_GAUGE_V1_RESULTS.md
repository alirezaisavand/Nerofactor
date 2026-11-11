# Offline LM proposal gauge-consistency v1 — results

**All fifteen proposals are counterfactual saturation-endpoint diagnostics and
round to zero float32 updates. Coupled mathematical proposals fail the frozen
consistency criterion for all five seeds, despite much smaller absolute
mismatches than the uncoupled control.**

LM102 has a **77.170479%** mathematical step discrepancy, absolute L2
**6.105772966775855e-15**. Its P0 step norm is only 7.91205788151339e-15.
Every represented proposal is exactly zero, every proposed physical material is
unchanged from its arm's input, and every original represented-step predicted
reduction is **-0.0**. Agreement of these zero represented steps is not evidence
of convergence or useful moving-step consistency.

The one scientific execution completed exactly 15 solves with no rendering,
new derivative, optimizer loop, applied parameter update, candidate objective,
accept/reject decision or damping variation. No failed numerical criterion was
relaxed. This measures proposal sensitivity, not recovery improvement or the
cause of the historical LM solution groups.

## Frozen provenance and reproducibility

Base `31dd7979256795abd8e0043099c649518aabad35`; branch
`feat/material-lm-proposal-gauge-audit-v1`, separate worktree. Implementation,
new tests and protocol were hash-frozen before execution. The publication
commit contains those same bytes. All earlier sources/evidence and original
workspace content remain unchanged, including the failed 3C-5 raw artifact.

| Artifact | SHA-256 |
| --- | --- |
| [Frozen protocol](protocols/MATERIAL_LM_PROPOSAL_GAUGE_V1.md) | e881753a8b973a9c4a3735147e6a543201d84e0518015c54e930a0afecfe14b1 |
| [Runner](../experiments/material_lm_proposal_gauge_v1.py) | 0a4cfae0c5fc561773b691076bc8e78650e02540f148473ad575eaf73521e94d |
| [Focused tests](../tests/scientific/test_material_lm_proposal_gauge_v1.py) | f6ab71b57aca8ce0eecabf44859151ed4c07c9857a601a226cef66e685f6ce31 |
| [Full proposal evidence](evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json) | 54920234b87edf17eeba688e0ffab83cdbd71f9246a4ca7654fb68f12bd5663e |
| [Historical LM evidence](evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json) | 66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f |
| [Archived derivatives](evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json) | 58a47eca9978e32385d94c1b3c1f087b2be0dff15fb095fe95b3e5601bc34070 |

New JSON: **1,339,003 bytes**, schema version 1,
`material-lm-proposal-gauge-v1`. Full r/J/H/g/d/A matrices, scalar/vector metrics,
all 15 deltas, candidates/physical values/Q, spectra, residual checks and exact
input/protocol/source identities are archived. Provenance ledgers include full
ordered controller decisions and JSON pointers. No binary/checkpoint is loaded.

Measured execute/setup time **1.52139967 s wall /
1.52329700 s CPU**; report timestamps span
1.915530 s including admission/final serialization. Imports
before report creation were covered by the external 300-second timeout but not
separately timed. Peak RSS **452,378,624 bytes
(431.421875 MiB)**, below 2 GiB. The owned 2 GiB RLIMIT_DATA
allocation cap and inherited hard limit policy were restored. RSS enforcement
is cooperative, not a cgroup hard resident limit; virtual library mappings are
not equated with resident memory. No host/container permissions were changed.

Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121; CPU, one Torch/BLAS/OpenMP
thread, hidden GPUs. Torch executes only the original small parameter transform
without gradients and inspects/restores owned thread/RNG settings. No shader
module is imported. Version labels do not prove historical binary identity.

Exactly **15 attempted/completed normal-equation solves**, 15 candidate arrays,
15 proposed physical-transform evaluations and ten input reconstruction checks.
Each case has three SVD summaries (normalized J, H, A), 45 total decompositions;
these are not extra normal-equation solves. There are **zero** shader forwards,
AD evaluations, optimizer updates and candidate objective evaluations. No
verification solve, inverse, pseudoinverse, least-squares fallback or retry.

## Historical damping: exact endpoint association

Damping provenance was inspected **before implementation or numerical proposal
execution**. The original source initializes .01, caps to [1e-8,1e8], updates
accepted trials by the frozen ratio policy, and multiplies rejected-trial
lambda by ten unless already at the maximum. A rejection at maximum terminates
`damping_saturated`. The saved outcome damping equals final `lambda_after` at
`z_after`; it is not the last accepted trial's used damping.

All five runs are complete, not budget-limited, and terminate at saturation.
Every ordered historical outer/trial was checked for source-policy and state/
MSE continuity without rerunning a solve or optimizer. Terminal z and physical
values match the admitted 3C-7 P0 endpoint exactly. Final trajectory and endpoint
records agree, and the final rejected iterations leave z unchanged.

| Seed | Outcome index | Terminal outer | Terminal trial | Used in all three new arms | Last accepted outer/trial | Last accepted used lambda (not substituted) |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | 14 | 1 | 100000000.0 | 11/1 | 1.524157902759e+01 |
| 102 | 3 | 17 | 3 | 100000000.0 | 14/4 | 3.763352846318e-02 |
| 103 | 5 | 11 | 4 | 100000000.0 | 9/3 | 1.371742112483e+02 |
| 104 | 7 | 9 | 2 | 100000000.0 | 5/1 | 1.234567901235e-04 |
| 105 | 9 | 14 | 4 | 100000000.0 | 11/3 | 5.080526342529e-03 |

Terminal iteration pointers (zero-based JSON indices) are:

| Seed | Terminal iteration pointer | Terminal trial pointer | Original termination |
| --- | --- | --- | --- |
| 101 | /outcomes/1/iterations/13 | /outcomes/1/iterations/13/trials/0 | damping_saturated |
| 102 | /outcomes/3/iterations/16 | /outcomes/3/iterations/16/trials/2 | damping_saturated |
| 103 | /outcomes/5/iterations/10 | /outcomes/5/iterations/10/trials/3 | damping_saturated |
| 104 | /outcomes/7/iterations/8 | /outcomes/7/iterations/8/trials/1 | damping_saturated |
| 105 | /outcomes/9/iterations/13 | /outcomes/9/iterations/13/trials/3 | damping_saturated |

Each terminal rejected trial recorded zero represented delta and -0.0 predicted
reduction in the historical log. The new experiment independently computes its
15 proposals from archived **chain-derived** derivatives and does not substitute
those historical proposals. All use terminal lambda=100000000 identically
across P0/P1-U/P1-C. No value is invented, reset to .01, taken from an earlier
accepted step or chosen using a new proposal's quality.

Because every source run already stopped, all new proposals are **counterfactual
hypothetical next systems**, not historically executable resumed iterations.
This saturation limits what the experiment can establish about moving updates.

## Exact equation and the admitted Jacobian distinction

Read archived physical fitting component Jacobians (72x7), original float32
transformation diagonals and exact saved z. All five endpoints reconstruct
physical values bitwise through the original source function; P1-U/P1-C share
identical coordinates/materials, and D1 permutes exactly to D0. Truth is not an
experimental case because its inverse-coordinate reconstruction was inexact.

The equation is the unchanged 3C-4 normal-equation formulation:

```
r32 = vec(mean_total_RGB32 - target32) / float32(sqrt(72))
r = float64(r32)
Jz = float64(Jphysical32) * float64(recorded_dp_dz32)
J = Jz / sqrt(72)                         # float64 chain analysis
H = J.T @ J
g = J.T @ r
d = diag(H) + 1e-8
A = H + lambda * diag(d)
delta64 = solve(A, -g)
candidate32 = float32(float64(z32) + delta64)
represented64 = float64(candidate32) - float64(z32)
predicted = -2*dot(g, represented64) - dot(represented64, H@represented64)
```

No identity damping, extra regularizer, trust-region rule or objective change
is introduced. Every archived latent matrix is verified against both its hash
and the physical-J/transform product before normalization. Mean-before-MSE and
original target conversion hashes are preserved. `g` is the normal vector;
the archived full MSE gradient and `2g` are saved separately. Maximum observed
L2 difference between `2g` and the archived full latent MSE gradient is
**5.82433377345189e-12**, consistent with the declared arithmetic distinction.

The historical optimizer obtained latent JVPs directly through the float32
normalized residual, then promoted their outputs to double. This audit uses the
3C-7 **chain-derived float64 latent Jacobian**, analytically divided by sqrt(72),
plus the original float32 residual values. NumPy solves the float64 system,
while the source used Torch. Formula/normalization conventions match, but these
are **not bitwise historical latent-AD Jacobians or reproduced 3C-4 steps**.
The limitation is retained rather than hidden behind a purported exact replay.

Map P1 steps by the seven-coordinate width permutation P.T, separately for
mathematical and represented deltas. Remove the admitted actual float32 base
angle offset for affine candidate-coordinate comparisons. For physical/Q
comparisons use the ideal pi/2 gauge and retain its pre-existing float32 phase
error; no angle canonicalization, frame repair or fitted alignment occurs.

## All fifteen mathematical and represented proposals

Step order is **[mx logit,my logit,theta,kR logit,kG logit,kB logit,metallic logit]**.
Values below are the raw float64 solutions in each arm's own coordinates.
The full JSON preserves exact values/hashes, original z/p, full normalized
residual/J and every normal/damped matrix. Candidates are never applied.

| Seed | Arm | delta mx-logit | delta my-logit | delta theta | delta kR-logit | delta kG-logit | delta kB-logit | delta metallic-logit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | P0 | -6.50963750e-14 | -1.00821687e-13 | -1.06310917e-15 | -8.74437375e-16 | -5.33429936e-16 | 1.42942759e-14 | -3.11612567e-14 |
| 101 | P1-U | -3.41302507e-10 | 1.41815212e-10 | -1.75214924e-10 | 8.58610600e-12 | 1.03037094e-11 | 1.37700175e-11 | -9.30685867e-11 |
| 101 | P1-C | -1.01200034e-13 | -6.48326465e-14 | -8.64611140e-16 | -8.74772988e-16 | -5.55512039e-16 | 1.43049527e-14 | -3.11595193e-14 |
| 102 | P0 | 4.50412873e-15 | 5.96579313e-15 | 3.99428812e-16 | 5.38431353e-16 | -1.70096008e-18 | -1.20262600e-15 | 2.19705091e-15 |
| 102 | P1-U | -3.41201617e-10 | 1.41882849e-10 | -1.75212167e-10 | 8.58766983e-12 | 1.03036638e-11 | 1.37553495e-11 | -9.30362351e-11 |
| 102 | P1-C | 1.01123316e-14 | 8.82221006e-15 | 5.64206851e-16 | 5.33727088e-17 | -4.10621601e-16 | -1.20164158e-15 | 3.20263798e-15 |
| 103 | P0 | -2.00715701e-13 | -4.57994727e-13 | -2.01601899e-13 | 6.63299772e-15 | 1.59980021e-14 | 2.19770672e-14 | -1.00117499e-13 |
| 103 | P1-U | -3.41629813e-10 | 1.41659597e-10 | -1.75434018e-10 | 8.59446280e-12 | 1.03204767e-11 | 1.37802982e-11 | -9.31393316e-11 |
| 103 | P1-C | -4.59051551e-13 | -2.00826735e-13 | -2.01336074e-13 | 6.83890107e-15 | 1.59975307e-14 | 2.19766819e-14 | -1.00380066e-13 |
| 104 | P0 | -2.16647116e-13 | -1.73729215e-13 | 2.65615019e-14 | 1.61847580e-15 | 2.53047746e-15 | 1.38611478e-16 | -7.92923155e-15 |
| 104 | P1-U | -1.57514538e-10 | 3.96494314e-10 | 1.88082614e-10 | -8.87751138e-12 | -1.06750501e-11 | -1.42928663e-11 | 8.62598888e-11 |
| 104 | P1-C | -1.74294605e-13 | -2.16975040e-13 | 2.51961449e-14 | 1.63680986e-15 | 2.56510816e-15 | 1.88498545e-16 | -8.05173083e-15 |
| 105 | P0 | -1.28569483e-14 | 1.28148155e-15 | 1.32846140e-15 | -2.74305170e-15 | -4.79084927e-16 | -8.91173097e-16 | -4.17335654e-15 |
| 105 | P1-U | -1.57360312e-10 | 3.96696773e-10 | 1.88040067e-10 | -8.88119674e-12 | -1.06771698e-11 | -1.42939660e-11 | 8.62706531e-11 |
| 105 | P1-C | -2.49231865e-16 | -1.21718030e-14 | 2.91414197e-15 | -2.51583081e-15 | -4.69398663e-16 | -8.90486311e-16 | -4.40908719e-15 |

| Seed | Arm | Mathematical step L2 | Unrounded model reduction | Represented model reduction | Damped condition |
| --- | --- | --- | --- | --- | --- |
| 101 | P0 | 1.248201912544e-13 | 3.498407714530e-22 | -0.0 | 375.103322 |
| 101 | P1-U | 4.199172389542e-10 | 2.295152612905e-15 | -0.0 | 385.322444 |
| 101 | P1-C | 1.249883503309e-13 | 3.499271622145e-22 | -0.0 | 375.103291 |
| 102 | P0 | 7.912057881513e-15 | 2.295601142905e-24 | -0.0 | 375.106392 |
| 102 | P1-U | 4.198493229780e-10 | 2.294304280235e-15 | -0.0 | 385.325585 |
| 102 | P1-C | 1.386655187420e-14 | 3.577349345718e-24 | -0.0 | 375.106376 |
| 103 | P0 | 5.490862641978e-13 | 3.534658509745e-21 | -0.0 | 374.990552 |
| 103 | P1-U | 4.202388296899e-10 | 2.298967892459e-15 | -0.0 | 385.196772 |
| 103 | P1-C | 5.499615826676e-13 | 3.541769120875e-21 | -0.0 | 374.990521 |
| 104 | P0 | 2.790971890723e-13 | 4.947059818506e-22 | -0.0 | 409.834869 |
| 104 | P1-U | 4.745855677917e-10 | 2.583627555896e-15 | -0.0 | 397.083662 |
| 104 | P1-C | 2.795813798460e-13 | 4.959913531348e-22 | -0.0 | 409.834916 |
| 105 | P0 | 1.395252929068e-14 | 8.342002970529e-24 | -0.0 | 409.921423 |
| 105 | P1-U | 4.746888291120e-10 | 2.583141156340e-15 | -0.0 | 397.168136 |
| 105 | P1-C | 1.354584378945e-14 | 7.768865569574e-24 | -0.0 | 409.921425 |

Every case has seven unchanged represented coordinates: **all 15 candidate32
arrays equal their own original z32 exactly**, and all 15 original-source
physical transforms return the original physical material. Represented deltas
are zero and represented model reductions are signed **-0.0**. Zero-step cosine
is null with reason. This is float32 stagnation under the selected saturated
lambda, not evidence of convergence or an accepted objective improvement.

The proposed physical Q tensors also equal their own input Q exactly. Between
P0 and P1, only the already admitted float32 phase discrepancy remains; no
material moved toward truth. No actual reduction or new acceptance/rejection
is measured (all corresponding fields are null).

## Coupled versus original: failed mathematical criteria

The frozen mathematical bound is elementwise
`1e-16+2e-4*abs(delta_P0)`, with relative norm/cosine floor 1e-16.
**All five coupled mathematical-step comparisons fail.** Tiny absolute values
are reported alongside relative differences; this result is not relabeled a
pass because represented arrays happen to be zero.

| Seed | P0 delta norm | Coupled difference L2 | Max element error | Relative L2 discrepancy | Cosine | Frozen criterion pass |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 1.248201912544e-13 | 5.026987749890e-16 | 3.783479205310e-16 | 0.40273835% | 0.999992807 | False |
| 102 | 7.912057881513e-15 | 6.105772966776e-15 | 4.318081329848e-15 | 77.17047901% | 0.991685281 | False |
| 103 | 5.490862641978e-13 | 1.145079832947e-15 | 1.056824285424e-15 | 0.20854279% | 0.999999098 | False |
| 104 | 2.790971890723e-13 | 1.520010007465e-15 | 1.365356939249e-15 | 0.54461674% | 0.999986698 | False |
| 105 | 1.395252929068e-14 | 2.331133434816e-15 | 1.585680571860e-15 | 16.70760467% | 0.986061312 | False |

For LM102, P0 norm is **7.91205788151339e-15**, coupled norm is
**1.386655187420304e-14**, and difference is **6.105772966775855e-15**:
**77.17047901080274%** relative, cosine **0.9916852813220781**. Its discrepancy
is largely a magnitude change along a similar direction. All mathematical P0
step norms exceed the declared 1e-16 denominator floor, so these percentages
are not produced by flooring the denominator. Their absolute physical relevance
is nevertheless limited because all such steps vanish on float32 conversion.

The represented-step criterion (2e-7 absolute +2e-5 relative) passes for all five
coupled comparisons with exactly zero difference, as does the affine candidate
comparison after removing the actual angle offset. This is a degenerate equality
of zero updates, not corroboration of a meaningful moving-step relation.

## Uncoupled control: larger mathematical changes, same zero updates

| Seed | Uncoupled difference L2 | Relative L2 factor | Cosine | Frozen mathematical pass |
| --- | --- | --- | --- | --- |
| 101 | 4.198495023280e-10 | 3.363634505833e+03 | 0.542778485 | False |
| 102 | 4.198533311522e-10 | 5.306499743047e+04 | -0.506583608 | False |
| 103 | 4.198267231060e-10 | 7.645915596148e+02 | 0.750816883 | False |
| 104 | 4.746999801526e-10 | 1.700841136131e+03 | -0.409692717 | False |
| 105 | 4.747001418209e-10 | 3.402251533978e+04 | -0.810794834 | False |

All five uncoupled mathematical comparisons fail, with difference L2 about
**4.20e-10–4.75e-10** versus coupled **5.03e-16–6.11e-15**. Uncoupled changes
are much larger mathematically and can reverse direction relative to P0.
Yet all five uncoupled represented comparisons also equal zero and pass; their
physical candidates are unchanged. No arm is selected as a better reconstruction
or update based on these calculations.

## Physical candidate correspondence and Q

Mapped physical and Q bounds are the predeclared 2e-7 absolute +2e-6 relative.
All five coupled and all five uncoupled physical/Q comparisons pass. Q values
are computed in float64 analysis using the original pure Q metric on admitted
float32 physical parameters. The original Q is in the shared unrotated tangent
coordinates; no world transform is introduced.

| Seed | Coupled Q difference Frobenius | Coupled Q relative difference | Uncoupled Q relative difference |
| --- | --- | --- | --- |
| 101 | 5.091182832596e-09 | 1.682200760524e-08 | 1.682200760524e-08 |
| 102 | 5.091193083598e-09 | 1.682202495769e-08 | 1.682202495769e-08 |
| 103 | 2.763643194235e-08 | 9.130771019476e-08 | 9.130771019476e-08 |
| 104 | 1.792576067464e-08 | 5.398947854599e-08 | 5.398947854599e-08 |
| 105 | 1.792415966687e-08 | 5.398971167203e-08 | 5.398971167203e-08 |

The Q differences, at most about 9.13e-8 relative, are exactly the existing
rounded-representation discrepancy because no candidate changes its arm's
material. This comparison therefore **cannot distinguish coupling from the
uncoupled control at the tested damping** and is not a recovery improvement.

## Predicted decrease and numerical sensitivity

All 15 mathematical/unrounded model reductions are positive; the original
**represented-step** reductions are all -0.0. All unrounded coupled and
uncoupled reduction-consistency comparisons fail the separate frozen
`1e-30+2e-4*abs(reference)` bound; all represented reduction comparisons agree
at zero. Relative represented-reduction metrics have zero references and are
explicitly flagged below floor, not treated as meaningful relative agreement.

Every mathematical reduction, including the uncoupled values near 2.6e-15,
is below the predeclared old operational objective-precision floor
`1e-12+5e-7*abs(archived_MSE)`. That is an interpretation warning, **not** a new
optimizer acceptance rule. Positive unrounded predictions do not guarantee
improvement, and no candidate is rendered to measure actual reduction. No
scientific case produces a negative or nonfinite reduction; synthetic fixtures
cover those diagnostic paths.

All normalized J, H and damped A systems have numerical rank seven at the frozen
relative threshold 1e-12. Damped condition numbers span roughly **374.99–409.92**;
there is no failed or unstable solve. Maximum relative matvec backward residual
is **1.0426667962354042e-17**, well inside 1e-11. Verification uses matrix-vector
products only; **no additional solve** was performed.

For a saved-array sensitivity identity, let A1/g1/delta1 be mapped to P0
coordinates. Then, up to the recorded solve residuals,

```
A0 * (delta1-delta0) = -(g1-g0) - (A1-A0)*delta1
```

This post-run check uses only archived solved arrays and matvecs, not another
proposal, decomposition, inverse or solve. The following terms localize the
forcing of the **coupled** proposal difference:

| Seed | norm(g1-g0) | norm((A1-A0)*delta1) | Relative A difference | Identity residual L2 |
| --- | --- | --- | --- | --- |
| 101 | 1.230305193327e-11 | 5.916784699893e-17 | 2.018698177521e-09 | 8.588180139212e-25 |
| 102 | 2.708820516768e-10 | 1.279438680005e-17 | 1.837974643600e-08 | 4.463342047136e-26 |
| 103 | 7.441449099132e-11 | 9.609441599374e-16 | 7.415790025667e-09 | 1.919080937634e-24 |
| 104 | 3.439207560577e-11 | 9.080089005293e-17 | 2.186052355810e-08 | 1.907406235103e-25 |
| 105 | 8.460900054085e-11 | 1.462113381066e-17 | 4.902861909340e-09 | 8.075352610465e-26 |

At LM102, the gradient forcing is **2.708820516767958e-10**, versus
**1.279438680004995e-17** from the matrix-change term. The coupled A difference
is only about 1.84e-8 relatively, while the proposal differs by 77.17% relatively.
This is consistent with the near-stationary residual sensitivity documented in
3C-7 and anisotropic system scaling, not a bad linear solve or a large absolute
motion. Large lambda suppresses absolute steps; tiny reference steps make
relative proposal percentages sensitive. No different damping was tested, and
there is no inference about usable proposal agreement at ordinary damping.

## Scientific verdicts

| Hypothesis | Classification and evidence |
| --- | --- |
| H1: coupled LM proposal consistency | **Contradicted for the frozen mathematical-step criterion:** 0/5 pass. Zero represented-step agreement is confirmed but degenerate under saturation; it does not rescue the failed mathematical comparisons. |
| H2: uncoupled discrepancy | **Confirmed for mathematical proposals:** 5/5 fail with much larger absolute changes, including direction changes. **Supported but limited** for practical implications because all represented steps remain zero. |
| H3: near-stationary sensitivity | **Confirmed in the recorded proposal systems:** small RHS differences produce significant relative step differences, especially LM102. Float32 rounding erases every step at the admitted saturation damping; this is not convergence. |
| H4: physical candidate correspondence | **Supported but limited:** all Q bounds pass, but both coupled and uncoupled candidates are unchanged inputs. Moving-candidate correspondence and recovery remain untested. |
| H5: historical optimization implications | **Unresolved:** no historically executable next iteration, optimizer trajectory, applied step or actual reduction is observed. Neither sample correspondence nor rounding is established as the cause of the two LM solution groups. |

Accepted 3C-4 recovery outcomes remain **3/5**, including the physically distinct
LM104–105 endpoints. This experiment does not improve them or attribute their
origin. The strongest practical limitation is universal terminal saturation:
these results cannot establish whether coupling restores useful proposal
consistency away from this stopped, float32-stagnant regime. The chain-derived
Jacobian/NumPy-solver distinction is an additional limit to historical claims.

## Tests, static validation and preservation

**19 new focused tests passed, 0 failed, 0 skipped**, final test-body time
**0.454 s**. Only the new test file was executed. The first development run
caught a signed-zero mismatch from grouping vector multiplication before the
dot product; the implementation was corrected to the original scalar-dot
formula **before freeze and before the one scientific run**. No experimental
threshold or damping was changed after results.

Tests include full logged damping admission and corrupt/missing provenance,
original normalized equations and float32 residual parity, physical/latent hash
checks, independent exact-gauge systems with unequal sensitivities, near-zero
residuals and small/large damping, pure original-proposal parity on synthetic
fixtures, float32 candidate/represented steps, signed zero/nonfinite outcomes,
singular/ill-conditioned handling without fallback, 15-solve cap, strict JSON,
partial failure/owned cleanup and no shader/AD/optimizer/RNG execution. Synthetic
fixture solves are pre-execution tests, not extra scientific proposals.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p 'test_material_lm_proposal_gauge_v1.py' -v
```

Both new Python files pass Python 3.8 AST parsing; actual tests/execution used
Python 3.9.12. Strict evidence JSON, Markdown links, source/evidence hashes,
changed-file scope and Git whitespace were checked. **600 preservation hashes**
cover 596 previous tracked files excluding intentionally updated AGENTS.md,
the original workspace document and three newly frozen code/test/protocol files.
No previous experiment, scientific source, configuration, dataset, checkpoint,
mesh or historical result was modified or regenerated.

Independent post-run NumPy/stdlib-only verification checked 381 array records,
250 comparison fields, original archived inputs/latent conversion, every system,
all candidate/represented values, predictions, spectrum-consistency identities,
solve matvec residuals, Q, damping pointers/ledgers, work counts and cleanup flags.
It used **no Torch, parameter transform, renderer, AD, decomposition, inverse,
normal-equation solve or optimizer**. No extra numerical solve was hidden in
verification. All global CPU RNGs and admitted input arrays stayed unchanged.

## Next substantive experiment — recommended, not implemented

Propose a preregistered **paired known-material LM trajectory experiment from
the original 3C-4 starting states**, using the original policy for genuinely new
runs and admitted original targets. Compare equivalent P0/P1-U/P1-C starts;
reuse accepted P0 evidence where exact identities permit. Before execution,
freeze a branch-local coupling policy and stop at the first unhandled decoder
branch crossing rather than silently changing the tape map. Preserve failed
starts and use fixed termination/rounding rules, without damping sweeps or
held-out selection. An original initial damping in such a newly authorized run
must not be mislabeled a reset/resume of these saturated endpoints.

That would address trajectory/recovery implications in a moving regime instead
of adding more terminal counterfactuals. It requires separate approval for
rendering, AD and optimization. No such implementation, extra damping value,
new trajectory or further experiment was begun here. Commit/push this feature
branch, do not merge, and stop for review.
