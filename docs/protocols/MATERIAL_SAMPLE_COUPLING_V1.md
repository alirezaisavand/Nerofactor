# Common-world-sample coupling diagnostic v1

Protocol `material-sample-coupling-v1`, milestone **3C-6**. Freeze before the
single authorized CPU execution. Base `1d16e9e147f6cdf270fbc0ed19e7add36ad17281`;
branch `feat/anisotropy-sample-coupling-v1`, separate worktree. No optimizer,
parameter update, backward, shader correction, reference integration or new
target. No automatic retry, changed bounds, representation selection or merge.

## Algebra and signed-frame decision

Read the admitted A source `sample_aniso_ggx_directions`,
`rotate_tangent_bitangent`, `compute_pdf_aniso_ggx` in
`scientific_sources/material_a/network/field.py`, commit
`e1e859779301faced46d6476691e20b546c27ae0`, SHA-256
`b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.
The retained name says GGX, but this inversion uses Beckmann slopes. This
diagnostic does not replace it. A2 uses the previously approved total-count
specular estimator and midpoint cosine diffuse construction, with atan2 decoder;
none of these accepted operations is edited.

For `alpha=2*pi*v`, the source gives

```
phi = atan2(my*sin(alpha), mx*cos(alpha))
D = cos(phi)^2/mx^2 + sin(phi)^2/my^2
tan(theta_h) = sqrt(-log(max(u, 1e-6))) / sqrt(max(D, 1e-6))
```

For positive admitted widths at most 1, ideal `D>=1`, so its epsilon clamp is
inactive algebraically. The local slope simplifies to
`q=sqrt(-log(max(u,1e-6)))*(mx*cos(alpha), my*sin(alpha))` and
`h=(F*q+n)/sqrt(1+q.q)`, where columns of `F` are world T/B. The fixed normal
is +Z; the source flat unrotated basis is T=+Y, B=-X. The source rotation is
`T'=c*T+s*B, B'=-s*T+c*B`. The accepted decoder uses
`theta_principal=atan2(sin(2*theta),cos(2*theta))/2`, so a nominal +pi/2 change
does **not** fix the sign of the actual principal frame rotation.

For each of the six states, use the original float32 encoding, accepted decoder
and source flat basis at the actual batch sizes (96 and 336 points), without a
sampler/forward call. Require the frames to equal the admitted archived first
frames and to be constant over these constant-material flat batches. Form
`M=F1.T@F0` in float64. Require orthonormality and exactly one signed-quarter-turn
interpretation, within absolute `2e-6` and zero relative tolerance:

```
M = [[0, s], [-s, 0]], s in {-1,+1}
F1 @ M = F0
```

Choose `s` from the sign of M[0,1], then verify the whole matrix/reconstruction;
never choose the sign using radiance errors. Require the same sign in both
batches. With exchanged widths, set `alpha_c=alpha-s*pi/2`. Then

```
(my*cos(alpha_c), mx*sin(alpha_c))
    = (s*my*sin(alpha), -s*mx*cos(alpha))
    = M @ (mx*cos(alpha), my*sin(alpha))
```

Thus `u_c=u`, `v_c=(v-s/4) mod 1` preserves the world slope, elevation and half
vector in exact arithmetic. Reflection, source PDF and radiance should then
correspond, subject to the original float32 arithmetic and clamp/mask boundaries.
This proof does not declare the actual clamped source PDF globally consistent.

Circle translation preserves Lebesgue measure: for any measurable azimuth set
E, its preimage is E+s/4 modulo 1 with equal measure; density remains one and
the Jacobian is one except at the measure-zero cut. No rejection, sorting,
reindexing, resampling or sample-count change occurs. This is an exact-real
measure statement, not a claim of a bijection of all quantized float32 values.

Diagnostic tape construction: retain original order; clone all three tensors
into disjoint storage. Compute the quarter shift/remainder in float64, round v
once to float32, then reduce a rounded endpoint 1 to periodic 0 using modulo
again. Record endpoint counts and circular rounding error. Do not clip, jitter,
flip frames, alter u or alter diffuse azimuth. The analogous periodic endpoint
is reduced in float64 boundary fixtures. Source u-clamping at 1e-6 remains
unchanged and its active count is recorded.

## Admission and fixed inputs

Verify all eight requested artifacts and original source/protocol identities
through the existing **read-only** admission chain. Never call its execution,
optimizer restoration, target generation, dense reference or old-test helpers.
The accepted 3C-5 success JSON SHA-256 is
`a7ab47837718eccb69e935c676796ada39aac367b021ce08d79533cb729299a1`.
Its results SHA-256 is
`ab785097f3a2f7e648bc51f7330dbda99853e54aa0c1dda375759b370864c6a8`.
Original scientific protocol SHA-256:
`8e510d566d443ed44f3e0428c429837ccd70b6bcfba0951efd5ce6bcf1d4572a`;
attempt2 protocol SHA-256:
`0fe9f41b299f848c661529a6c36d5b2d061ef1fb5f028873f42960021807dbac`.
The accepted A2 shader SHA-256 is
`de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f`;
the reused capture/report helper SHA-256 is
`53dce44200eb3278cf99561da0eb1d099e8c9da5bd15b7cc341149c86628fde9`.
The runner's explicit INPUTS table and inherited accepted identities bind the
remaining hashes, including the failed raw 3C-5 output and its failure manifest.
Never try to parse or repair that truncated raw JSON. Verify both frozen G/A
snapshot manifests; neither source pin changes. Record new runner/protocol
hashes and accepted identities in the new evidence before any forward.

Exactly truth followed by saved final LM 101,102,103,104,105; parameter order
`mx,my,theta,k_R,k_G,k_B,metallic`. Reuse truth and exact LM float32 physical
values from admitted evidence. P0 is original; P1-U and P1-C both use the same
float64 width swap +pi/2 followed by one float32 material conversion. No angle
canonicalization. P1-C differs from P1-U only in copied specular v.
Verify nominal float64 Q symmetry before all shader forwards, with inherited
`atol=5e-14, rtol=5e-12`. Reuse the accepted physical-kernel/convergence and
identifiability gates; no repeated kernel probes or reference quadratures.

Use unchanged `MaterialShaderV1('A2','atan2')`, 512 diffuse/1024 specular,
original scene/known lighting/cameras, 24 fit observations and all 84 evaluation
observations (24 fit,24 held-out,36 relight), original float64 independent targets
and original float32 fit targets. Use unchanged `ObservationBatch` and
`is_train=True` with fitting tapes 0–3 and evaluation tapes 1000–1003. Private
reconstruction of the original tapes must match every original byte hash; no
new seed or RNG-policy change. Complete original seed/array hashes are inherited
from accepted evidence and copied into the new report; diagnostic copied v gets
its own separate identity. Original u and diffuse azimuth must remain bitwise
equal throughout, and original/copy storage must be disjoint.

## Calls, measurements and predeclared bounds

Exactly one P0 and one P1-C forward per state/batch: at most **24 calls**, **5,184
points**, **5,308,416 specular** and **2,654,208 diffuse** directions. No P1-U or
P2 forwards. Recapture P0 because accepted evidence lacks full raw samples;
require **every original serialized P0 field** to equal its archived counterpart,
including mean/per-tape values, hashes, metrics, physical settings, frames,
sample hashes, masks and identities. Any mismatch stops immediately before that
state/batch's coupled call. Reuse archived P1-U predictions/metrics without
rerendering. Save full P0/P1-C per-tape/mean component RGB, original target errors
and archived P1-U comparisons. No outcome is selected.

Capture the original return arrays using the unchanged accepted observer. Add a
read-only observer on this diagnostic's owned dynamic shader class PDF method:
invoke the exact saved source method once with identical arguments, save its
inputs/return, return that same tensor, and restore the owned class in finally.
No accepted module/file, sampler or operator changes. Reconstruct its PDF
operations only for diagnostics and require bitwise equality with the observed
PDF. This is not an additional shader call or substitute PDF. Save shape/hash,
range and clamp counts for raw cos(theta)^2, exponent, wo.h and denominator;
compare the actual returned PDFs at coupled approximately corresponding h.

Freeze these bounds **before** the scientific run; no tuning on full outcomes:

| Quantity | Absolute tolerance | Relative tolerance/definition |
| --- | --- | --- |
| Float64 analytical fixture half-vector | 5e-14 | 0 |
| Actual frame quarter-turn/orthonormality | 2e-6 | 0 |
| Half-vector components | 2e-6 | 2e-5 times abs(P0) |
| Reflected-direction components | 4e-6 | 2e-5 times abs(P0) |
| Vector angular deviation | 2e-5 radians | atan2(norm(cross),dot), unmodified vectors |
| Source PDF | 1e-5 | 2e-4 times abs(P0) |
| Mean linear RGB | 2e-6 | 2e-5 times abs(P0), inherited 3C-5 |
| Fit MSE difference | 1e-12 | 5e-7 times abs(L(P0)), inherited 3C-5 |

These operational float32 bounds are motivated by small accumulated trig/frame
roundoff and potentially ill-conditioned PDF division, not statistical confidence
limits. The analytical tests use exact-world frames independently of float32
decoding. Do not silently relax an operational bound for a large relative error.

For h, wi, actual PDF save RMSE, MAE, relative L2, maximum absolute error, all-tape
and each-tape errors, sample exceedance counts; for vectors add stable angular
statistics. PDF elementwise relative errors use floor 1e-12; aggregate original
radiance/error helper uses L2 denominator floor 1e-4. Include both PDF measures.
Record masks, both transition directions, XOR per tape/observation, valid/rejected
counts and boundary samples `abs(wi.n)<=2e-5`. Define grazing
`abs(wo.h)<=1e-3`. Report PDF errors on overlapping source-clamp categories
(cos²<1e-6, exponent outside [-40,40], dot outside [0,1], denominator<1e-6),
grazing/boundary samples and their unclamped nongrazing complement. Record
clamp-status flips. Empty categories are null with explicit reasons, never NaN.

Require bitwise equal diffuse directions and per-tape diffuse RGB. Compare
diffuse/specular/total RGB to P0, both individual tapes and four-tape mean,
separately fit/held-out/relight. Report original metrics to unchanged targets
for all arms. The fitting objective is exactly original Torch float32
`mean((mean_over_four_tapes(total_RGB)-target_float32)^2)`, not mean tape MSE.
Record L(P0), L(P1-U), L(P1-C), signed/absolute/relative differences with original
objective denominator floor 1e-12. No regularization or other objective added.

Operational bounds on measured correspondences are **scientific outcomes**, not
integrity gates: a finite mismatch is retained as a failed H2/H3 subcheck and all
six declared states remain reported. Fatal gates are missing/corrupt identities,
Q/frame derivation, exact P0 mismatch, source-PDF observation/reconstruction,
nonfinite/wrong shapes, component/mask invariants, diffuse/radial/tape mutation,
unowned output, runtime/resources or unexpected global RNG consumption. A fatal
gate preserves partial strict JSON and stops, with no repair/retry.

## Resource limits, tests, outputs and interpretation

One CPU process/thread, original Python 3.9.12/NumPy 1.21.5/Torch 2.4.0+cu121
versions, default float32, hidden GPUs; no CUDA query, optimizer or autograd.
One 1,800-second external timeout plus internal process alarm covering admission.
Peak RSS maximum 4 GiB checked cooperatively at return boundaries; soft address
space cap 8 GiB or stricter inherited limit, hard limit unchanged. Restore owned
threads/signals/memory settings and verify global Torch/NumPy RNG and original
tapes unchanged. Keep only current batch's P0/P1-C raw sampling arrays; archive
complete RGB but sampling hashes/ranges/excerpts and comparisons instead of
millions of raw vectors. Document this offline verification limitation.

Only new focused tests in `tests/scientific/test_material_sample_coupling_v1.py`
and static checks. Fixtures cover both signs, exact algebra, wrapping/quantization,
uniform interval measures, disjoint tapes, real tiny A2 correspondence, strict
successful-record JSON, source observation/cleanup, admission failures, exact
P0 mismatch, budgets and no optimizer/autograd/RNG consumption. Do not rerun old
scientific suites/experiments. Use python3 for every Python command.

Implementation `experiments/material_sample_coupling_v1.py`. Exclusive-create
`docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json`; serialize the whole strict JSON
payload (`allow_nan=False`) before seeking/writing, reusing the repaired accepted
writer. Refuse output collisions. Post-run
`docs/MATERIAL_SAMPLE_COUPLING_V1_RESULTS.md` records all six outcomes and
failures. Preserve all old bytes and original workspace user edits.

From `/home/isavand/paper/Nerofactor-3c6-sample-coupling`, execute **once only**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_sample_coupling_v1 \
  --output /home/isavand/paper/Nerofactor-3c6-sample-coupling/docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json
```

Assess H1 analytical measure-preserving correspondence, H2 finite h/wi/mask/PDF
correspondence, H3 original objective symmetry, H4 measured clamp/rounding/boundary
residuals and H5 unresolved causation of LM groups independently. Do not infer
unbiasedness in expectation, global source-PDF correctness, an optimizer fix or
improved recovery from these finite fixed states. Recommend one next experiment
without implementing/executing it. Commit and push only the feature branch,
without merging, then stop for review.

## Frozen implementation and original tape identities

The following source/test hashes were recorded after the new focused tests
and before the one scientific execution; neither file may change afterward.

| File | SHA-256 |
| --- | --- |
| `experiments/material_sample_coupling_v1.py` | `598bc1f55b40d83c89f7768a1d0a2008d70e8f5fdce08b7a58757028a015a49c` |
| `tests/scientific/test_material_sample_coupling_v1.py` | `6b46f801b28836d8d1f5a9e25ffbd8f94c02ac77027c0d75d5a98d9a4e59b2c3` |

Original combined ordered u/v/diffuse-azimuth byte hashes follow. Each
individual array hash is additionally bound inside the admitted accepted JSON
and checked during reconstruction. No coupled hash is claimed before generation.

| Batch | Seed | Original combined SHA-256 |
| --- | --- | --- |
| fit | 0 | `83bc9de75b5b32cc68c8f9663720f3cacbbad436c7f72e81a248f20d3b668f53` |
| fit | 1 | `f4689a08ea7c685f5e3341caa985a2823b42762aa291dad403c26607527c9f1c` |
| fit | 2 | `f52a8b1245bc7dd09819a4bfc202cfcbac63169804f96c419d82570db21c3f3f` |
| fit | 3 | `912987fb679a9ee40f171e44a637107a570a7cff5b152d6509b29d3a1c5efdb3` |
| evaluation | 1000 | `e9fb759ec21b9f4351754cb2071117e73f115d9116dd0955139fc845b74adc3e` |
| evaluation | 1001 | `5c9ce66939dde1b3404ff3bcd31f9163ab1dfa85d8561d76ed4c358a584a59e5` |
| evaluation | 1002 | `cb3ae520dafd76c9bfdf4d47a4ad7cef46f2ea576c70e0ffb8f0b8cee5cc1afa` |
| evaluation | 1003 | `cdff3cdf5d322e5fb3c0275c56f33e1f5ca7f7baf69815d9576a33c4932e716a` |
