# Gradient and Jacobian gauge-equivariance v1 — results

**Coupling restores the tested RGB Jacobians and MSE gradients within the frozen
absolute-plus-relative bounds. Relative gradient accuracy near stationarity is
limited: LM102 has a 46.60% physical-gradient difference and cosine 0.892788,
despite passing the predeclared absolute bound.**

All six materials, all 30 declared state/representation/batch cases and every
mandatory integrity/construction gate completed. Coupled physical Jacobians
pass all 36 component comparisons and coupled physical gradients pass all six
checks; the uncoupled control fails all 12 specular/total Jacobian checks and
all six gradient checks. Diffuse derivatives are unchanged. These verdicts do
not imply uniformly small **relative** gradient error, global differentiability,
an optimizer fix, improved material recovery or causal attribution of LM groups.

All 36 finite-difference column/step checks meet the frozen numerical bounds,
but **24 span changed reflection masks**. No decoder branch is crossed.
Passing a finite-difference bound across pieces is not a certificate of a smooth
ordinary derivative. The source PDF is not an active input to the RGB objective;
the 43 accepted grazing PDF exceptions remain preserved, not reclassified.

## Evidence, exact work and reproducibility

Base `f5a3d714fd57bfef02baf6ca47b15ab69927e7ea`; feature branch
`feat/anisotropy-gradient-equivariance-v1`, separate worktree. Source/test/protocol
hashes were frozen before the one scientific execution and remain unchanged.
The publication commit records the same code bytes. No earlier accepted source,
shader, protocol, evidence or original workspace content was changed.

| Artifact | SHA-256 |
| --- | --- |
| [Frozen protocol](protocols/MATERIAL_GRADIENT_EQUIVARIANCE_V1.md) | 352452eb1699a8203efa6a77a5e9c42101614b1a8104447844eb9dd5960204c3 |
| [Runner](../experiments/material_gradient_equivariance_v1.py) | b53bd46bf766d0e2203bb74cfc09bddac1680bc7b75a9c8c23972bccb556495c |
| [New focused tests](../tests/scientific/test_material_gradient_equivariance_v1.py) | b21c59fba283bd8fe2f4caa1fc7a3aa452ef2292781903f27eba7520cfb2b15f |
| [Complete numerical evidence](evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json) | 58a47eca9978e32385d94c1b3c1f087b2be0dff15fb095fe95b3e5601bc34070 |
| [Accepted coupling evidence](evidence/MATERIAL_SAMPLE_COUPLING_V1.json) | c8e9ddc1d0ab3c5d028bf9925062b6e2d27b9b6bc017800e7a2b00d92aebd87d |
| [Unchanged A2 shader](../experiments/material_shader_v1.py) | de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f |

The new evidence is **23,732,557 bytes**, strict JSON schema version 1,
protocol `material-gradient-equivariance-v1`. It preserves complete physical and
valid optimizer-coordinate Jacobian arrays, prediction arrays, gradients,
metrics, spectra/near-null vectors, packed reflection masks and hashes. All
source/evidence/protocol/sampler identities are bound through the accepted chain,
including failed 3C-5 files. No checkpoint or binary pickle is loaded.

One process on CPU, one Torch/BLAS/OpenMP thread, GPUs hidden. Original numerical
runtime: Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121, default float32 shader/AD.
Float64 is used only for analysis of returned float32 derivatives, coordinate
scaling, finite-difference quotients, norms and SVD. Version matching does not
establish historical installed-binary identity.

Measured execute/setup time: **196.66970316 s wall /
196.65610673 s CPU**. Report timestamps span
198.048494 s including admission/final serialization; imports
before report creation were covered by the external timeout but not separately
timed. Peak RSS **884,097,024 bytes (843.140625 MiB)**,
below 4 GiB; the 8 GiB soft address limit and inherited hard limit policy were
restored. No retry/extension, optimizer, parameter update, GPU query, training,
new target/reference integration, Docker or package installation occurred.

| Scientific work | Completed |
| --- | --- |
| Original baseline forwards | 30 |
| Directional AD primal forwards | 210 |
| Direct MSE-gradient forwards | 18 |
| Declared finite-difference forwards | 72 |
| Total forwards / shader points | 330 / 54,720 |
| Fit / evaluation points | 22,464 / 32,256 |
| Diffuse / specular sampled directions | 28,016,640 / 56,033,280 |
| Complete seven-direction Jacobians | 30 |
| JVP double-reverse sweeps | 420 |
| Direct MSE reverse sweeps | 18 |
| Small original-coordinate transform VJPs | 10, with no rendering |
| Total counted reverse sweeps | 448 |

All original and coupled tape hashes and both global CPU RNG states remained
unchanged; owned threads/signals/memory settings were restored. Thirty baselines
matched their accepted P0/P1-U/P1-C forward identities exactly. Every JVP and
direct-gradient primal also matched its accepted current baseline, including
all per-tape component predictions and reflection masks. No implicit baseline
substitution or finite-difference fallback occurred.

The 90 physical component matrices contain **90,720 float32 derivative values**.
Twenty-five valid latent-coordinate cases yield 75 derived component matrices,
**75,600 values**, without extra renderer calls. Total-RGB Jacobians are 72x7
for fitting and 252x7 for evaluation. Full layout and column identities are in
JSON; no individual-tape objective replaces the mean-before-MSE objective.

## Parameter transformation and locally fixed sampling contract

Physical p is a column `[mx,my,theta,kR,kG,kB,metallic]`; P swaps only its first
two coordinates. Axis exchange is `p1=P*p0+(pi/2)*e_theta`, with differential
`dp1=P*dp0`. For locally corresponding outputs, independent chain-rule derivation
therefore gives **J0=J1C P** and **g0=P.T g1C**. Output rows are observation/RGB;
parameter columns are never transposed or aligned using numerical agreement.

Actual archived/source decoded frame signs are independently checked before
rendering: truth/LM104/LM105 have s=-1; LM101/LM102/LM103 have s=+1.
`u_c=u`, `v_c=(v-s/4) mod 1` uses exactly the accepted copied-tape implementation
and hash identities. The original radial and diffuse azimuth draws are unchanged.
Copies own disjoint storage. The sign and tapes remain **fixed constants**
inside every JVP, gradient and declared perturbation; no derivative through a
sign, wrap, RNG or tape-construction path is taken.

| State | Actual frame sign s | P0 branch margin (rad) | P1 branch margin (rad) |
| --- | --- | --- | --- |
| truth | -1 | 1.2607963244107108 | 0.3099999864909311 |
| LM-101 | 1 | 0.3166373531010489 | 1.25415895978478 |
| LM-102 | 1 | 0.31663747231033845 | 1.2541588405754904 |
| LM-103 | 1 | 0.316660805544581 | 1.254135596748215 |
| LM-104 | -1 | 1.2652192870762686 | 0.305577083430018 |
| LM-105 | -1 | 1.2652208367970328 | 0.30557553370925383 |

For latent coordinates p=psi(z), use the original sigmoid transforms and their
observed float32 AD diagonal D, so `Jz=Jp D`, `gz=D.T gp`. Both exact physical
reconstruction and the same width-logit/angle translation are checked first.
The ten transform VJPs compute D at two representations for five saved LM
endpoints. This coordinate analysis reuses physical Jacobians; it is not a
second uncounted latent-rendering experiment.

### Truth inverse roundtrip is unavailable for the latent comparison

Direct physical truth remains
`[.20000000298023224,.550000011920929,.3100000023841858,
.30000001192092896,.44999998807907104,.6000000238418579,.25]` in float32.
The existing inverse-sigmoid candidate reconstructs width mx as
`.19999998807907104` and blue albedo as `.5999999642372131`.
Maximum physical mismatch is **5.960464477539063e-8** in both representations.
Those candidate values/coordinates are recorded but **not rendered** or
substituted. Truth latent derivative comparisons are explicitly unavailable.
All five saved LM vectors and both equivalent representations reconstruct their
admitted physical inputs bitwise; their latent comparisons are valid.

## Coupled physical Jacobians and parameter sensitivities

All physical component/column comparisons pass the fixed elementwise bound
`2e-6+2e-4*abs(J0)`. The largest coupled elementwise error is
**1.1920928955078125e-7**; largest component relative Frobenius error is
**8.837216439918196e-8**. Diffuse matrices agree exactly. The errors below use the
same width-permuted J1C columns, not a fitted alignment.

| State | Batch | Diffuse relative J error | Specular relative J error | Total relative J error | Max component element error |
| --- | --- | --- | --- | --- | --- |
| truth | fit | 0.000000000000e+00 | 5.214491730275e-08 | 1.817753482592e-08 | 2.980232238770e-08 |
| truth | evaluation | 0.000000000000e+00 | 6.148834200862e-08 | 1.849885277700e-08 | 4.470348358154e-08 |
| LM-101 | fit | 0.000000000000e+00 | 6.000537725629e-08 | 1.424133439049e-08 | 2.980232238770e-08 |
| LM-101 | evaluation | 0.000000000000e+00 | 5.585794748936e-08 | 1.971746251438e-08 | 2.980232238770e-08 |
| LM-102 | fit | 0.000000000000e+00 | 5.570009602117e-08 | 3.357918214902e-08 | 5.960464477539e-08 |
| LM-102 | evaluation | 0.000000000000e+00 | 5.616185790465e-08 | 1.755974485478e-08 | 2.980232238770e-08 |
| LM-103 | fit | 0.000000000000e+00 | 6.829074677280e-08 | 2.717521894724e-08 | 4.470348358154e-08 |
| LM-103 | evaluation | 0.000000000000e+00 | 8.837216439918e-08 | 3.722640360660e-08 | 5.960464477539e-08 |
| LM-104 | fit | 0.000000000000e+00 | 5.624434501767e-08 | 3.099769877318e-08 | 5.960464477539e-08 |
| LM-104 | evaluation | 0.000000000000e+00 | 7.071085712961e-08 | 2.857316011382e-08 | 5.960464477539e-08 |
| LM-105 | fit | 0.000000000000e+00 | 8.556347116296e-08 | 2.395021147111e-08 | 5.960464477539e-08 |
| LM-105 | evaluation | 0.000000000000e+00 | 7.766681855946e-08 | 4.499420393160e-08 | 1.192092895508e-07 |

| Physical parameter | Maximum fit total-J column element error | Maximum evaluation error |
| --- | --- | --- |
| mx | 1.490116119385e-08 | 1.490116119385e-08 |
| my | 1.490116119385e-08 | 1.862645149231e-08 |
| theta | 1.210719347000e-08 | 1.303851604462e-08 |
| kR | 5.960464477539e-08 | 1.192092895508e-07 |
| kG | 5.960464477539e-08 | 1.192092895508e-07 |
| kB | 5.960464477539e-08 | 1.192092895508e-07 |
| metallic | 4.470348358154e-08 | 4.470348358154e-08 |

Per-column absolute/relative Frobenius errors, raw norms and denominator-floor
flags are retained for every component and case. Relative-J metrics use floor
1e-8. Zero-reference diffuse width/orientation columns are explicitly marked;
they are not scored as independently identifiable directions.

The fitting uncoupled control changes the specular Jacobian by about
**2.0704–2.1286%** in relative Frobenius norm and the total Jacobian by
**0.7410–0.7662%**. All six uncoupled diffuse matrices pass exactly, while all
six specular and six total matrices fail. Uncoupled evaluation Jacobians were
not authorized and were not computed.

## Fitting gradients: absolute pass and near-zero relative limitations

All 18 direct original-MSE gradients agree with the residual-Jacobian
reconstruction at the mandatory frozen bound. Maximum construction elementwise
error is **2.1017479809759976e-11**. The source objective is exactly the original
float32 MSE **after** four-tape averaging. No auxiliary loss enters it.

Coupled physical gradients pass all six frozen **elementwise**
`2e-9+2e-4*abs(g0)` checks. Maximum element error is
**1.6467538443976082e-9** and maximum difference L2 is
**2.373455918114176e-9**. The following table makes the tiny denominators visible;
passing that absolute-plus-relative criterion does not establish uniformly
small relative errors near a stationary point.

| State | norm(g0) | norm(Delta g) | max abs element | Relative L2 error | Cosine | Frozen bound pass |
| --- | --- | --- | --- | --- | --- | --- |
| truth | 8.518085054673e-05 | 8.335138814038e-10 | 8.221832104027e-10 | 0.000979% | 1.000000000 | True |
| LM-101 | 5.720140672609e-08 | 1.006239762028e-10 | 8.913048077375e-11 | 0.175912% | 0.999998813 | True |
| LM-102 | 5.093639442414e-09 | 2.373455918114e-09 | 1.646753844398e-09 | 46.596465% | 0.892788374 | True |
| LM-103 | 1.183495988718e-07 | 7.076700380103e-10 | 6.970140020712e-10 | 0.597949% | 0.999983324 | True |
| LM-104 | 1.527669217821e-08 | 2.551239137534e-10 | 1.909938873723e-10 | 1.670021% | 0.999910704 | True |
| LM-105 | 1.025304365025e-08 | 7.799417467198e-10 | 7.667040335946e-10 | 7.606929% | 0.999216054 | True |

At **LM102**, norm(g0)=5.093639442413695e-9, difference L2=2.373455918114176e-9,
relative error **46.596464962770895%**, cosine **0.8927883740622288**. Its largest
element difference is 1.6467538443976082e-9, below the frozen 2e-9 absolute term.
The declared relative denominator floor is only 1e-10; none of the six gradient
norms is below it. The floor is a metric convention, not a guarantee that these
small directions are numerically robust. LM104 and LM105 also show 1.67% and
7.61% relative error. No threshold or floor is changed to hide these effects.
Cosines in the raw report use recorded vector arithmetic and clipping to [-1,1];
last-bit cosine rounding does not establish exact directional equality.

The uncoupled control changes every physical gradient well beyond the bound:

| State | Uncoupled gradient difference L2 | Relative L2 (factor) | Cosine | Frozen bound pass |
| --- | --- | --- | --- | --- |
| truth | 8.273377256108e-05 | 9.712719705198e-01 | 0.238959364 | False |
| LM-101 | 8.506012038107e-05 | 1.487028470967e+03 | 0.685923461 | False |
| LM-102 | 8.506133768783e-05 | 1.669952077478e+04 | -0.542347123 | False |
| LM-103 | 8.506902634876e-05 | 7.187943783477e+02 | 0.962102009 | False |
| LM-104 | 8.634683616903e-05 | 5.652194543280e+03 | -0.553094053 | False |
| LM-105 | 8.634547133080e-05 | 8.421447745291e+03 | 0.525689039 | False |

Large relative control errors at endpoints reflect a roughly 8.5e-5 changed
gradient compared with a very small original g0; they are not hidden as a
normalization artifact. Some control directions are opposed. This measures
fixed-state finite-objective sensitivities, not an observed optimizer trajectory.

### Saved-array explanation of the residual coupled gradient differences

Read-only NumPy analysis decomposes the **recorded** reconstructed difference:

```
g1_mapped - g0 = 2/N * (J1_mapped-J0).T * r0
+                2/N * J1_mapped.T * (r1-r0)
```

Here r uses the original float32 subtraction and N=72. Direct autograd differs
from the float64-analysis reconstruction by a small additional reduction
remainder, also retained below. This calculation uses saved matrices/residuals
only: no new forward, reverse, target or scientific run.

| State | Jacobian-difference term L2 | Residual-difference term L2 | Direct-AD reconstruction remainder L2 |
| --- | --- | --- | --- |
| truth | 1.605790649523e-12 | 8.353469716613e-10 | 7.038476722557e-12 |
| LM-101 | 8.280933128686e-13 | 9.862738364083e-11 | 2.832046217146e-12 |
| LM-102 | 1.274352068826e-12 | 2.373186913240e-09 | 1.944592767682e-12 |
| LM-103 | 8.110949155886e-13 | 7.084068115757e-10 | 3.152652572283e-12 |
| LM-104 | 1.061647993435e-12 | 2.542206072287e-10 | 2.751904321630e-12 |
| LM-105 | 1.253013775554e-12 | 7.812294633373e-10 | 3.235853653040e-12 |

For LM102, the residual term is **2.3731869132399898e-9**, versus
1.2743520688260104e-12 from the Jacobian difference and 1.944592767682163e-12
from the direct-AD reconstruction remainder. Thus the saved-array accounting
attributes most of the gradient difference to last-bit prediction/residual
differences at this near-stationary point. This does not identify which source
operation generated those RGB bits or prove a global optimizer effect. No
higher-precision shader or altered reduction was implemented/tested.

## Original optimizer-coordinate comparisons

All five valid LM coordinate pairs pass the same frozen bounds: 30 coupled
component-J comparisons and five coupled gradient comparisons. Maximum latent
Jacobian element difference is **2.9540583312837043e-8**, and maximum latent
gradient element difference is **4.04386213540876e-10**. The uncoupled latent
control fails all ten specular/total and all five gradient comparisons; its
five diffuse comparisons pass. Truth is unavailable for the reason above.

| State | Original optimizer g0 norm | Latent difference L2 | Relative L2 error | Cosine | Frozen bound pass |
| --- | --- | --- | --- | --- | --- |
| LM-101 | 1.354944002114e-08 | 2.506262926180e-11 | 0.184972% | 0.999998619 | True |
| LM-102 | 1.193851880026e-09 | 5.419257200574e-10 | 45.393045% | 0.899526723 | True |
| LM-103 | 2.806898355751e-08 | 1.485956249016e-10 | 0.529394% | 0.999986606 | True |
| LM-104 | 3.582171164554e-09 | 6.902213652434e-11 | 1.926824% | 0.999851260 | True |
| LM-105 | 2.203321557149e-09 | 1.688807493431e-10 | 7.664825% | 0.999015058 | True |

LM102 again has a large relative effect: **45.39304490985919%**, with cosine
0.8995267226263213. Transforming to optimizer coordinates does not remove the
near-zero-gradient precision issue. The recorded exact reconstruction and
original transform D distinguish this statement from an inverse-roundtrip
physical substitution or restoration of Adam/LM state.

## Conditioning and numerical near-null directions

All tested total-RGB physical Jacobians have numerical rank **7**; diffuse
matrices have rank **3** and specular matrices rank **6**, at the fixed
`max(1e-8,1e-4*sigma_max)` rule, in both fit/evaluation batches and all arms.
This is consistent with component parameter dependencies; a component-only
null direction does not imply the full RGB problem is locally rank deficient.
Full singular values and near-null right vectors are archived for every matrix.

| State | Fit P0 sigma_max | Fit P0 sigma_min | P0 condition | P1-U condition | P1-C condition | P0 latent condition |
| --- | --- | --- | --- | --- | --- | --- |
| truth | 2.579813782500e+00 | 7.074301572727e-02 | 36.467399 | 36.140880 | 36.467399 | unavailable |
| LM-101 | 2.574945556436e+00 | 7.287598669006e-02 | 35.333251 | 35.766866 | 35.333254 | 43.820157 |
| LM-102 | 2.574945986907e+00 | 7.287595847843e-02 | 35.333271 | 35.766892 | 35.333274 | 43.820249 |
| LM-103 | 2.574937537103e+00 | 7.287801658971e-02 | 35.332157 | 35.764602 | 35.332164 | 43.816233 |
| LM-104 | 2.575767547050e+00 | 7.203027448413e-02 | 35.759513 | 35.039172 | 35.759508 | 44.601091 |
| LM-105 | 2.575772815123e+00 | 7.202946634961e-02 | 35.759987 | 35.039993 | 35.759991 | 44.604977 |

Total fitting physical conditions span about **35.33–36.47**; coupled values
agree closely, while uncoupled sampling measurably changes their spectra.
Total evaluation conditions span **28.52–31.00**. Valid original optimizer
fitting conditions span **43.82–44.60**, illustrating coordinate dependence.
There is no full-total numerical near-null direction at the declared rule.
These are local Jacobian properties at six states, not a global identifiability,
Hessian convexity, optimizer-basin or recovery-success proof. Where a component smallest singular value falls below the absolute floor,
its condition number is null with an explicit reason, not JSON infinity.
Numerical rank uses the separately declared absolute-plus-relative SVD rule.

## Frozen central-difference checks and piecewise masks

Only truth, LM101 and LM104, fitting observations, P0/P1-C, two width coordinates
and angle were checked. Perturbations use the corresponding transformed columns
(mx goes to my, my to mx), fixed tapes, and exactly h1=1e-4/h2=5e-5 before the
original float32 input conversion. Actual represented spans are saved, and the
quotient denominator remains **2h**. No step sweep, extra state or sign update
was performed.

All **108 component FD-to-AD checks**, **54 component step-stability checks**
and **54 component FD gauge comparisons** meet their frozen numerical bounds.
Maximum FD-to-AD element error: **9.514093399047852e-4**; maximum h2-to-h1
change: **8.940696716308594e-4**; maximum matched FD gauge difference:
**5.960464477539062e-4**. These are judged against a predeclared 1e-3 absolute
plus 2% relative FD bound, much looser than the AD gauge bound. Halving h does
not consistently reduce error; the two-step evidence is not a convergence proof.

| State | P0 coordinate | P0 h1 max FD error | P0 h2 max error | P1-C h1 max error | P1-C h2 max error | P0 +/- mask XOR [h1,h2] | P1-C mask XOR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truth | mx | 2.395510673523e-04 | 4.480928182602e-04 | 2.395473420620e-04 | 4.879944026470e-04 | [0, 0] | [0, 0] |
| truth | my | 1.941658556461e-04 | 5.240589380264e-04 | 1.818090677261e-04 | 5.240440368652e-04 | [5, 3] | [5, 3] |
| truth | theta | 2.251537516713e-04 | 4.927352420054e-04 | 2.251532860100e-04 | 4.927344853058e-04 | [1, 0] | [1, 0] |
| LM-101 | mx | 4.787296056747e-04 | 9.514093399048e-04 | 4.787221550941e-04 | 9.514093399048e-04 | [3, 1] | [3, 1] |
| LM-101 | my | 3.588460385799e-04 | 7.393900305033e-04 | 3.588497638702e-04 | 7.393881678581e-04 | [1, 1] | [1, 1] |
| LM-101 | theta | 4.088365822099e-04 | 5.715619772673e-04 | 4.088354762644e-04 | 5.715615116060e-04 | [0, 0] | [0, 0] |
| LM-104 | mx | 1.442264765501e-04 | 5.212202668190e-04 | 1.826211810112e-04 | 5.753263831139e-04 | [3, 0] | [3, 0] |
| LM-104 | my | 4.530847072601e-04 | 4.334673285484e-04 | 4.530847072601e-04 | 4.998967051506e-04 | [5, 3] | [5, 3] |
| LM-104 | theta | 2.488298341632e-04 | 5.922280251980e-04 | 2.488326281309e-04 | 7.644870784134e-04 | [2, 1] | [2, 1] |

Of 36 central column/step checks, **24 have at least one mask transition**:
6/12 at truth, 8/12 at LM101 and 10/12 at LM104. Their secants cross pieces;
ordinary smooth derivative agreement is therefore not certified by their
passing numerical errors. The other 12 have no observed mask transition.
All 36 corresponding P0/P1-C perturbed-forward mask comparisons agree exactly.

No FD pair crosses an ideal decoder branch. The smallest original branch
margin across the six states is **0.30557553370925383 rad**, well above either
orientation step. Margins/branch cells and unchanged actual float32 base frames
are recorded separately; the angle is never canonicalized for rendering.

Machine-readable mask labels `plus_to_base`, `minus_to_base`, `plus_to_minus`
mean **first array compared with second**, consistent with the inherited helper.
Their `valid_to_rejected`/`rejected_to_valid` counts run from the **second
(reference) array to the first (current)**: base->plus, base->minus, minus->plus.
This explicit convention avoids mistaking comparison labels for temporal
transition direction. XOR counts are symmetric. Full packed masks and hashes
allow independent verification without native rendering.

## PDF evidence and numerical interpretation

Active AST inspection finds **zero reads** of the `pdf` argument in the original
`compute_radiance`. The A2 approved total-count estimator retains this property.
A tiny new AD fixture confirms RGB has no graph dependency on the returned
sampled PDF. The optimized objective contains only total RGB MSE, without the
shader's auxiliary energy/specular/regularization-like returns.

The 43 previous near-grazing PDF-bound exceptions remain accepted evidence of
imperfect PDF correspondence; no new PDF invariance claim or PDF correction is
made. Baseline/FD grazing and reflection-boundary counts are recorded, but this
audit does not compute sample-resolved PDF gradients or establish operation-level
causality from such aggregate associations. The measured near-stationary RGB
gradient difference is accounted for by saved RGB residual changes as above,
not inferred from a PDF quantity that is absent from the objective.

## Verdicts and remaining limits

| Hypothesis | Verdict |
| --- | --- |
| H1: local gradient/Jacobian gauge-equivariance | Passes every frozen coupled physical/valid-latent absolute-plus-relative bound. Jacobians are very close. Relative gradients near stationarity can remain inaccurate (LM102 about 46.6% physical / 45.4% latent); no uniform relative-precision claim. |
| H2: uncoupled derivative discrepancy | Confirmed for all six specular/total fitting Jacobians and gradients; diffuse control is unchanged. Finite fixed-tape sensitivities differ in magnitude, direction and spectra. |
| H3: numerical instability | Observed in the relative direction of tiny gradients and mask-changing finite-difference secants; no tested decoder branch crossing. All FD numerical bounds pass, without certifying smoothness across masks. No global differentiability claim. |
| H4: optimization implications | Unresolved. A mechanism that can affect directions/conditioning is measured, but no optimizer, trajectory or update is executed; causal origin of the two LM groups is not established. |

Accepted material-recovery outcomes remain **3/5** from 3C-4; no fitted material
or threshold changes. This diagnostic does not establish unbiased gradients,
global invariance, an optimizer fix, native reconstruction or improved recovery.
Only these six states, eight original tapes, known flat geometry/lighting and
the declared observation sets are covered. Evaluation derivatives are kept
separate from fitting and never select a parameter or validation choice.

## Tests, static verification and preservation

**25 new focused tests passed, 0 failed, 0 skipped**, final test-body time
**3.603 s**. No historical suite or previous scientific experiment was rerun.
The tests include independent analytical permutation/gradient fixtures, both
signed frames, real tiny A2 float32 AD, direct/J-derived MSE and composed original
optimizer-coordinate gradients, local fixed tapes, FD stability/branch/masks,
partial AD failure without fallback, exact primal identity, strict successful
JSON and prior-snapshot preservation, budgets, failure cleanup and source pins.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p 'test_material_gradient_equivariance_v1.py' -v
```

The two new Python files pass Python 3.8 AST parsing; execution was Python 3.9.12,
not a Python 3.8 runtime test. Strict evidence JSON, Markdown links and Git diff
scope/whitespace were checked. **595 preservation hashes** cover all 591 old
tracked files other than the intentionally updated AGENTS.md, the original
workspace document, and the three newly frozen code/test/protocol files.
The raw failed 3C-5 output, valid failure manifest, accepted shader, both pinned
scientific trees and every earlier evidence file retain their bytes.

NumPy/stdlib-only post-run verification rechecked array values/shapes/hashes,
full physical/latent matrix relationships, spectra/ranks/near-null directions,
gradient construction and comparisons, original/coupled forward identities,
FD columns/represented perturbations/packed-mask transitions/branch decisions,
all work counts and cleanup flags. It executed no Torch, shader, AD, RNG or
optimizer. Earlier evidence was admitted unchanged; no previous experiment
was rerun.
The new verification read 834 array records (including repeated references),
checked 10,480 comparison fields, and recomputed stored spectra for consistency.
NumPy four-tape means matched the recorded Torch means bitwise; maximum objective
reduction difference against float64 arithmetic was 2.4761635649028004e-14.

## Next single controlled experiment — not implemented

Request a bounded **proposal-only original-LM gauge audit**, using these archived
physical/valid latent Jacobians, residuals and gradients. Freeze an explicitly
admitted original damping value and original diagonal-scaling/linear-solve
formula. Compare P0, P1-U and P1-C proposed latent steps after the same width
permutation, exposing how near-zero residual roundoff and conditioning propagate
to a proposed update. Do not apply proposals or render new states. No damping
sweep, optimizer rerun or parameter selection would be needed for that narrow
question. The new protocol must explicitly distinguish these chain-derived
latent matrices from the original float32 latent-JVP rounding path; it must not
claim bitwise historical-step identity. Unknown damping provenance must be
resolved before admission.

That would test one optimizer implication, not prove the historical trajectory
cause. It requires separate approval: no proposal/linear solve, optimizer,
new experiment implementation or further scientific execution was performed
here. Commit/push this feature branch only, do not merge, and stop for review.
