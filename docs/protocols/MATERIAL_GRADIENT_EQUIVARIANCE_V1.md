# Gradient and Jacobian gauge-equivariance audit v1

Protocol `material-gradient-equivariance-v1`, milestone **3C-7**. Freeze source,
tests and this document before **one** bounded CPU execution. Base
`f5a3d714fd57bfef02baf6ca47b15ab69927e7ea`, new branch
`feat/anisotropy-gradient-equivariance-v1`, separate worktree. No optimizer,
parameter update, estimator correction, new material/target/reference integration,
GPU/neural training, package installation, Docker, retry or budget extension.
No source or evidence from an earlier milestone may change.

## Independent chain-rule derivation and numerical layout

Let p be a **column** vector ordered
`[mx,my,theta,kR,kG,kB,metallic]`. The axis exchange is
`T(p)=P*p+(pi/2)*e_theta`; P is the 7x7 identity with its first two rows
exchanged. Consequently P is symmetric, involutory, and `dT=P`.
Within a neighborhood with fixed frame-sign branch and fixed coupled tape,
the physical correspondence is `f0(p)=f1C(T(p))`. Applying the ordinary
chain rule to each scalar RGB output gives

```
J0 = J1C @ P           # rows = outputs, columns = physical coordinates
g0 = P.T @ g1C         # gradients are column covectors in these coordinates
g = (2/N) * J_total.T @ (mean_over_four_tapes(total_RGB)-target)
```

This relation is a hypothesis for the actual float32 piecewise shader, not
assumed true by construction. No matrices/columns may be aligned using their
numerical errors. The only allowed map is the declared width permutation.
Keep diffuse, specular and total derivatives separate. Internal Jacobian layout
is `[observation, component(diffuse,specular,total), RGB, parameter]`.
Seven directional ADs produce the complete `[n*9,7]` matrix; each component is
then flattened observation-major/RGB-major to `[n*3,7]`. Thus total RGB is
**72x7** for fitting and **252x7** for evaluation. No individual-tape objective
or sRGB derivative replaces the original four-tape mean linear RGB.

For original optimizer coordinates z, `p=psi(z)` with
`m=.04+.96*sigmoid(z)`, theta unchanged, and albedo/metallic sigmoid. Its
diagonal derivative D has width entries `.96*sigma*(1-sigma)`, angle entry 1,
and albedo/metallic entries `sigma*(1-sigma)`. Then
`Jz=Jp*D`, `gz=D.T*gp`. The corresponding latent transform also swaps the two
width logits and translates theta. When `D1*P=P*D0`, the same permutation
relations apply to Jz and gz. These spectra are in a different Euclidean
coordinate metric, not an intrinsic material identifiability score.

For each LM endpoint, read the exact saved 3C-4 float32 unconstrained vector,
verify shape/device/dtype/value hash, and require original `physical(z)` to
equal the admitted physical vector **bitwise**. For P1, swap width logits and
set theta to the same once-rounded admitted physical theta+pi/2. Require exact
reconstruction again. Never restore an optimizer. Obtain the transform diagonal
from one small original-transform VJP with all-ones cotangent, at most 12 such
reverse sweeps total; no renderer call is involved. Apply this observed float32
diagonal to the physical derivatives in float64 **analysis only**.

Truth remains the direct original physical material. Check the existing pure
`inverse_coordinates` candidate at original nominal truth, separately recording
its float32 roundtrip. If reconstruction is inexact, retain its values/errors
but mark latent derivatives unavailable; never render that different physical
point or silently substitute it. Missing truth latent results do not invalidate
the direct physical audit. A corrupt/non-exact saved LM reconstruction is a
mandatory failure. New focused fixtures compare the analytic chain with direct
AD through the composed original transform and tiny shader.

## Accepted source, evidence, states and fixed tapes

Admit the unchanged complete chain through the accepted read-only helpers,
never their runners/optimizer restorers/reference generators. Key additional
identities:

| Artifact | SHA-256 |
| --- | --- |
| `docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json` | c8e9ddc1d0ab3c5d028bf9925062b6e2d27b9b6bc017800e7a2b00d92aebd87d |
| `docs/MATERIAL_SAMPLE_COUPLING_V1_RESULTS.md` | 9b9fd3a20be59e295606323d2c91f534a8bab631eebb55c0ebffcc805ce24dde |
| `docs/protocols/MATERIAL_SAMPLE_COUPLING_V1.md` | ca968222f7f67c2164b238b9da35145ad9ad8a4d785ea675282971680ad4611f |
| `experiments/material_sample_coupling_v1.py` | 598bc1f55b40d83c89f7768a1d0a2008d70e8f5fdce08b7a58757028a015a49c |
| `experiments/material_shader_v1.py` | de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f |
| Reused 3C-4 directional AD helper | 4abdf70b1a8c3b309f32561e31bcdb605392d0858bbe2015bf98c1a8f065cb6e |
| Reused 3C-2 pure inverse-coordinate helper | 687a84cf141a4d005e30be9b0d8bd20599bca1fc93f812c54679fe92c79d5d2a |

The hash-bound inherited manifests include the original recovery/diagnostics/
continuation/optimizer-comparison/symmetry evidence and protocols, shader,
sampler helpers, and both pinned scientific snapshots. A is
`e1e859779301faced46d6476691e20b546c27ae0`; G remains
`79252178e65914ad1acbc14790b1514f276bfea0`. Retain the first failed/truncated
3C-5 raw JSON and separate valid failure manifest unchanged; never parse/repair
the truncated file. Record accepted identities plus new code/test/protocol hashes.

Exactly truth followed by final LM101,102,103,104,105 from 3C-4, same as 3C-6.
Original P0 and once-rounded P1 physical vectors are taken from admitted 3C-6.
Check nominal Q symmetry and actual decoded/source T/B matrices **before**
shader work. Recompute the signed-quarter matrix at both original batch sizes
and require agreement with archived 3C-6 mappings. Do not choose signs using
gradient/radiance errors.

Reuse `u_c=u, v_c=(v-s/4) mod 1` exactly via the hash-pinned 3C-6 copy helper.
Require full copy metadata/hashes to equal archived 3C-6, including periodic
endpoint treatment. All original u/v/diffuse tape bytes and disjoint copied
storage remain preserved. Tapes are **locally fixed constants** throughout AD
and finite differences: no differentiation through sign/wrapping, no sign
reselection on perturbation, and no seed changes. Record the ideal decoder
branch cell `floor((theta+pi/2)/pi)` and distance to boundaries
`theta=(k+1/2)*pi`, separately from actual admitted float32 frames/encoding.
No angle canonicalization is applied to renderer inputs.

Use unchanged `MaterialShaderV1('A2','atan2')`, 512 diffuse/1024 specular samples,
known flat geometry/lighting, original 24 fitting observations, full 84 evaluation
observations, and original independent targets. Fit tapes 0–3 and evaluation
tapes 1000–1003 retain all admitted combined/per-array hashes. No target/reference
convergence gate is rerun. The existing scene, target hash and original fitting
target float32 hash must match. The objective remains Torch float32
`mean((four_tape_mean[:,total]-target_float32)^2)`, with no extra terms.

## AD construction, exact primals and measurement bounds

Fitting: P0, P1-U, P1-C for all six states. Evaluation: P0, P1-C only.
One no-grad baseline per combination verifies original mean/per-tape prediction
values/hashes, material/encoding/tape identities and mask summary against the
accepted 3C-5 or 3C-6 record. Preserve actual baseline evidence before a failed
gate. No fallback baseline, re-render or permissive mismatch is allowed.

For each Jacobian reuse the exact 3C-4 `directional_jvp` double-reverse procedure
with the seven physical-coordinate unit vectors. Every AD primal must equal the
accepted current baseline **bitwise**, including all components, individual
tapes and masks. Preserve each completed column in partial JSON until the full
matrix is available. An unsupported AD operator, wrong shape/dtype or nonfinite
derivative is a fatal construction failure, never a gauge failure or reason
to substitute finite differences.

For all 18 fitting combinations, use a separate direct original-MSE forward and
one autograd VJP. Compare its physical gradient to `2*J_total.T*residual/72`,
using the original float32 residual, with mandatory elementwise construction
bound `2e-9+5e-5*abs(reconstructed)`. Save both gradients and actual errors.

Freeze the following operational **elementwise absolute-plus-relative** bounds
before the scientific run:

| Comparison | Absolute | Relative | Relative metric denominator floor |
| --- | --- | --- | --- |
| Coupled physical/latent Jacobian | 2e-6 | 2e-4 | 1e-8 |
| Coupled physical/latent MSE gradient | 2e-9 | 2e-4 | 1e-10 |
| Direct MSE versus J reconstruction (mandatory) | 2e-9 | 5e-5 | 1e-10 |
| Central difference versus AD / h2 versus h1 / FD gauge | 1e-3 | 2e-2 | 1e-8 |

Reference magnitude is always P0 (or AD/h1 for their checks). Tolerances cover
float32 trig/frame/reduction accumulation; new independent analytical and tiny
real A2 fixtures validate them before execution. They are not statistical
confidence limits. The FD absolute scale acknowledges float32 output rounding
divided by a 1e-4 or 2e-4 central denominator. Absolute floors can dominate small
derivatives; retain raw norms, maximum/column errors and denominator flags.
No threshold may be changed after any scientific result is observed.

Report full component/parameter Jacobians, Frobenius absolute/relative errors,
max/RMSE and each-column errors. For gradients retain magnitude differences,
relative L2 and cosine only when both norms exceed 1e-10; otherwise null with
reason. Apply the same stated bounds in optimizer coordinates where valid.
Report float64-analysis singular values, full condition number only when the
smallest singular value exceeds 1e-8, numerical rank/near-null right vectors
at `max(1e-8,1e-4*sigma_max)`, and retained-subspace condition. Do not emit
infinity/NaN or interpret zero-reference columns as well-measured directions.
Conditioning is coordinate- and component-dependent. No held-out result selects
parameters, bounds, representations or finite-difference steps.

## Fixed finite differences and piecewise behavior

Exactly truth, LM101, LM104; P0 and P1-C on the **fitting** observations only.
Exactly coordinates mx,my,theta and central steps `h1=1e-4`, `h2=5e-5`.
For the transformed arm use coordinate P*j (mx maps to my, my to mx, angle to
angle). Construct plus/minus coordinates in float64, round once to original
float32 physical inputs; record represented positive/negative steps/span and
use the declared denominator **2h**, not a silently altered effective step.
These are temporary declared perturbations, not new fitted materials.

Keep the already selected coupled tape unchanged at every perturbation.
Record plus/minus mean RGB, full packed masks/hash/counts and branch cells;
retain each forward snapshot before the other side completes. Compare the
central columns to AD, h1 to h2 and corresponding P0/P1-C perturbations. Count
mask XORs/transitions per tape/observation against baseline and between signs.
If a branch crosses, classify its quotient as not an ordinary smooth derivative.
If masks change, retain numerical errors and identify a secant crossing pieces;
do not certify smoothness merely because numerical errors pass a loose FD bound.
Never smooth masks, change h, repair angles or replace AD with these checks.

Source review must distinguish active RGB from diagnostics: the original
`compute_radiance` has **zero active reads** of its `pdf` argument. A2 retains
the approved total-count specular denominator; sampled PDF and auxiliary
regularization-like returns are not terms in this fitting objective. Inspect
and record the active AST. New tiny-AD fixture checks RGB has no graph dependence
on returned sampled PDF. The 43 accepted grazing PDF exceptions remain failed
PDF comparisons; they are not by themselves evidence of RGB-gradient failure.
Record observed grazing and reflection-boundary counts per baseline/FD snapshot,
but do not infer per-operation or per-sample derivative causality from aggregates.

## Exact work and resource accounting

| Work | Fit calls / points | Evaluation calls / points | Total |
| --- | --- | --- | --- |
| Baselines | 18 / 1,728 | 12 / 4,032 | 30 calls / 5,760 points |
| Seven directional AD primals | 126 / 12,096 | 84 / 28,224 | 210 calls / 40,320 points |
| Direct MSE gradients | 18 / 1,728 | 0 | 18 calls / 1,728 points |
| Fixed central differences | 72 / 6,912 | 0 | 72 calls / 6,912 points |
| Maximum complete run | 234 / 22,464 | 96 / 32,256 | **330 calls / 54,720 points** |

Maximum sampled directions: **56,033,280 specular**, **28,016,640 diffuse**.
Count attempted/completed forwards and categories separately. At most 30
Jacobians, 210 directional JVPs, **420 double-reverse sweeps** plus **18 direct
MSE reverse sweeps**; additionally at most **12 tiny original-coordinate
transform VJPs** with no shader call. Count these extra sweeps explicitly.
Reverse calls are never labeled free or conflated with ordinary forwards.

One CPU process, one Torch/BLAS/OpenMP thread, hidden GPUs, original Python
3.9.12/NumPy 1.21.5/Torch 2.4.0+cu121, default float32 shader/AD. No CUDA query.
External process timeout and internal alarm at 1,800 seconds; no retry/extension.
Peak RSS 4 GiB checked cooperatively before/after native returns; soft address
cap 8 GiB or stricter inherited value, hard limit unchanged. Restore owned
signals/threads/limits and verify global Torch/NumPy RNG plus original and
copied tapes unchanged. Do not retain unused sample graphs between AD calls.

Mandatory source/data/shape/AD-construction/nonfinite/primal/runtime/resource
failures stop immediately, preserving valid partial evidence and incomplete
case/column/FD records. **Finite scientific comparison failures continue** through
the fixed plan, preserving failed comparisons without threshold adjustment.
No optimizer or parameter update is constructed. Computational limits take
precedence over completing the plan.

## Files, tests, execution and interpretation

New files only: `experiments/material_gradient_equivariance_v1.py`,
`tests/scientific/test_material_gradient_equivariance_v1.py`, this protocol,
exclusive-create `docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json`, and
post-run `docs/MATERIAL_GRADIENT_EQUIVARIANCE_V1_RESULTS.md`. Update AGENTS.md
for authorization/completion; preserve every old source/evidence/protocol byte.
JSON contains complete physical/valid latent Jacobian values, masks, predictions,
gradients, numerical metrics and hashes. Validate strict serialization of the
entire payload before touching a previous snapshot, using the accepted repaired
writer. Refuse any output collision. Never repair/reuse a failed evidence path.

Run only the **new** focused tests and necessary static/identity checks. Fixtures
cover Q and seven-coordinate chain rules, both frame signs, float32 tiny-A2 AD,
direct versus reconstructed MSE gradients, composed optimizer-coordinate AD,
fixed/disjoint tape use, truth roundtrip invalidation, FD stability/branch/masks,
exact primal failures, unsupported AD without fallback, partial records,
strict JSON, resource/call limits, cleanup and original source preservation.
No old experiment, gate integration or historical test suite is rerun.

From `/home/isavand/paper/Nerofactor-3c7-gradient-audit`, execute **once**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_gradient_equivariance_v1 \
  --output /home/isavand/paper/Nerofactor-3c7-gradient-audit/docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json
```

Use python3 for all Python commands. Report every state and H1 local derivative
gauge-equivariance, H2 uncoupled control, H3 numerical/branch/mask effects, and
H4 **unresolved** optimization implications independently. Do not claim a new
optimizer fix, global differentiability/unbiased gradients, recovery improvement
or causal explanation of LM groups. Recommend one next controlled experiment
without implementing it. Commit/push only this feature branch, do not merge,
and stop for review.

## Frozen implementation and tape hashes

Recorded after the final new focused tests and before scientific execution.
Do not edit these files after this freeze.

| File | SHA-256 |
| --- | --- |
| `experiments/material_gradient_equivariance_v1.py` | `b53bd46bf766d0e2203bb74cfc09bddac1680bc7b75a9c8c23972bccb556495c` |
| `tests/scientific/test_material_gradient_equivariance_v1.py` | `b21c59fba283bd8fe2f4caa1fc7a3aa452ef2292781903f27eba7520cfb2b15f` |

All individual u/v/diffuse array hashes and state-specific coupled hashes are
bound by the accepted 3C-6 JSON SHA-256 above and independently checked before
AD. Original combined ordered tape identities are also listed here.

| Batch | Seed | Combined original SHA-256 |
| --- | --- | --- |
| fit | 0 | `83bc9de75b5b32cc68c8f9663720f3cacbbad436c7f72e81a248f20d3b668f53` |
| fit | 1 | `f4689a08ea7c685f5e3341caa985a2823b42762aa291dad403c26607527c9f1c` |
| fit | 2 | `f52a8b1245bc7dd09819a4bfc202cfcbac63169804f96c419d82570db21c3f3f` |
| fit | 3 | `912987fb679a9ee40f171e44a637107a570a7cff5b152d6509b29d3a1c5efdb3` |
| evaluation | 1000 | `e9fb759ec21b9f4351754cb2071117e73f115d9116dd0955139fc845b74adc3e` |
| evaluation | 1001 | `5c9ce66939dde1b3404ff3bcd31f9163ab1dfa85d8561d76ed4c358a584a59e5` |
| evaluation | 1002 | `cb3ae520dafd76c9bfdf4d47a4ad7cef46f2ea576c70e0ffb8f0b8cee5cc1afa` |
| evaluation | 1003 | `cdff3cdf5d322e5fb3c0275c56f33e1f5ca7f7baf69815d9576a33c4932e716a` |
