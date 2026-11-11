# Gauge roundoff v1 — frozen 3C-9b diagnostic

Base `f9043aaecb709cebde597eab970458c72ad0f2c5`, branch
`feat/material-gauge-roundoff-v1`, separate worktree. One new CPU forward-only
diagnostic, with at most six full shader calls. No LM/Adam, parameter update,
derivative/backward, reference quadrature, new targets/materials/seeds/counts,
accepted shader/source changes, Docker, GPU, installs, retries or next experiment.

Freeze runner, new tests and this protocol in
`MATERIAL_GAUGE_ROUNDOFF_V1.freeze.json` before the one execution. Exclusively
create `docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json`; refuse a collision.
Preserve the original user worktree and all prior source/protocol/evidence bytes.

## Admission and fixed states

Read-only admission through 3C-9's helpers verifies the complete 3C-1–3C-8 chain,
including failed 3C-5 artifacts, source/sampler/shader/target hashes and pinned
G/A snapshots. Do not invoke old runners or regenerate references. Additional
hard-pinned 3C-9 identities:

| Artifact | SHA-256 |
| --- | --- |
| Failed raw JSON | `88b7e83ec7f79b3e9f476c9aa5ccd199e5394342ca83205394188f800cfb8f9a` |
| Failure analysis JSON | `218ebd3e03f8e14accf8286789409dc93d1e704a434ea7856cd219c4dfc3c503` |
| Results | `c07e24dbc6cc35e1c5960714003d48e6503bbbb814210dc53747c99345bf8f0b` |
| Protocol | `47e083d21f66f8ffe75e01072e512219264d3161974ea9397819231ab8455ba7` |
| Freeze | `9d2e6845cd14d5991c6a5c4c655c14bf406f97c49b9a15572ffceb235d6f4db7` |
| Runner | `27383436180896391f78e2bab3c3404ee029c6bd614506884b59a4ad0610c0d3` |
| Tests | `b4e4d6e20d24fbcffad04b9f15c377fc109a9e2a1ca0d2ce825636fb27863662` |

Verify the full frozen 3C-9 execution identities and its exact failed state,
six initial forwards, zero derivatives/solves/updates. Preserve its raw fields
and original gate outcomes; this diagnostic cannot repair or rerun that attempt.

Exactly continuation seeds101/102, update1000, hash/dtype/shape/value-checked
original seven-vectors. P1 swaps original width logits and once-float32 theta+pi/2.
Require latent/physical records equal their archived 3C-9 counterparts; Q within
original elementwise `2e-7+2e-6*abs(Q0)`, identical albedo/metallic. Same original
24 fitting observations, ordering, targets, fixed training tapes0–3 and coupled
tape hashes/signs. Exact original frame mapping and copied-tape metadata must
match 3C-9. P1-U is an archived reference only; never render it.

Keep A2/atan2, original seven-coordinate transforms, known geometry/lighting,
visibility, 512 diffuse and1024 specular samples, CPU float32 operations. Preserve
original independent targets' identities and float32 fitting conversion. Runtime
must equal Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121, CPU/one thread.

## Phase 1: offline analysis before any shade call

Read exact saved float32 predictions `[24,3,3]`, components diffuse/specular/total.
Require all JSON numbers exactly representable and their original hash. Recreate
the original MSE with Torch `(prediction-target)**2` then `mean`, exactly equal
the stored scalar. All analyses below retain original observations/channels.

For P0 and P1-C compute separately:

1. Original float32 residual, square and mean.
2. Float32 square values averaged in float64 (final mean only changed offline).
3. Float32 residuals promoted before float64 square/mean.
4. Stored float32 predictions/target promoted BEFORE residual subtraction.

For float32 residuals r0/r1 promoted to float64, dr=r1-r0, record
`mean(r1²)-mean(r0²)` and `mean(2*r0*dr+dr²)`. Their independently rounded
float64 reductions need not be bitwise equal: predeclare absolute error bound
`64*eps64*(mean(r1²)+mean(r0²)+mean(abs(2*r0*dr+dr²)))`.
Also verify the algebra EXACTLY with stdlib Fraction on the same dyadic values;
never describe cancellation-rounded float64 equality as exact bitwise equality.

Record first/quadratic terms, all72 per-observation/channel contributions
divided by72, all24 observation totals, and top10 by absolute contribution
(deterministic index ordering). Original target is the same recorded float32
array, not the original float64 quadrature target for this arithmetic comparison.

Decompose signed objective difference by a telescoping precision ladder:
stored-RGB/f64 residual+square+mean difference; plus float32-subtraction rounding;
plus float32-square rounding; plus float32-final-mean rounding. These are offline
arithmetic contributions, not alternative executed shader variants. Evaluate
the ORIGINAL gate `abs(LC-L0)<=1e-12+5e-7*abs(L0)` for each hypothetical scalar
calculation, both with its own P0 scalar and the fixed historical bound. Report
all; never change the accepted original objective or gate.

## Phase 2: exactly six diagnostic calls

Execute four baseline calls FIRST, then two encoding interventions:

1. 101/P0, original tapes.
2. 101/P1-C, original admitted coupled tapes.
3. 102/P0, original tapes.
4. 102/P1-C, original admitted coupled tapes.
5. 101/P1-C-E, same P1-C state/tape, encoding -E0.
6. 102/P1-C-E, same P1-C state/tape, encoding -E0.

Every call goes through unchanged production `ObservationBatch.__call__`, with
exact original shade bytecode. A local invocation proxy passes every baseline
argument unchanged; for P1-C-E it replaces ONLY argument7 (orientation encoding).
No duplicate shader call for capture/replay. Whole execute scope uses no_grad;
all inputs and copies must be CPU/non-grad tensors. Code has no optimizer/AD call.

At actual Python return events, a temporary `sys.setprofile` hook observes exact
registered code objects: decode_atan2, MaterialShaderV1.shade, original sampler,
approved A2 compute_radiance, and source PDF. It copies existing tensors; it does
not substitute functions, compile modified code, patch Torch globals, reorder
arithmetic or reconstruct the active shader. Require exactly one return for each
registered function and restore the prior profiler even on failure. Fail closed
if another profiler is installed. Tests invoke only tiny synthetic functions,
never a full scientific shader.

Capture actual fields (N=96, seed-major four tapes x24 observations):

| Stage | Shapes/layout |
| --- | --- |
| mx/my/metallic; albedo | [96,1]; [96,3] |
| Input double-angle encoding; head normalization; decoder normalization | [96,2] each |
| Decoded principal angle, cos, sin | [96,1] each |
| Tangent/bitangent/normal | [96,3] each |
| Original u/v; diffuse azimuth | [96,1024]; [96,1,1] |
| Clamped xi1, xi2, source azimuth/elevation and local half coefficients | [96,1024] |
| Half vectors; reflected directions | [96,1024,3] |
| Diffuse directions | [96,512,3] |
| PDF, cos(theta_h) intermediates, dot(wo,h), denominator | [96,1024,1] |
| Incident lighting; Fresnel | [96,1536,3]; [96,1024,3] |
| Active diffuse/specular weighted RGB before sum | [96,512,3]; [96,1024,3] |
| Active specular f_s, cos_in and denominator | [96,1024,3]; [96,1024,1] |
| Actual source reflection mask | [96,1024] bool |
| Per-tape components/RGB | [4,24,3,3] |
| Original four-tape mean | [24,3,3] |

The PDF return locals exp_term and dot_wo_h are ALREADY clamped; overwritten
pre-clamp values are not accessible with this instrumentation. Do not fabricate
them or claim their source-line origin was directly traced. Record accessible
grazing indicators at predeclared1e-5 threshold, with explicit field meaning.
Preserve the historical43 PDF exceptions; this is not a new PDF acceptance test.
Verify the original AST still does not read pdf in active compute_radiance RGB.

All native baseline mean RGB bytes, scalar MSE, mask identities and active tape
values must exactly match 3C-9. No approximate substitute. Validate every required
array's shape/dtype/finite values. Save each obtained call record before identity
decisions so a failed baseline leaves usable partial evidence. Mandatory identity
or construction failure stops the entire run without retry.

For P1-C-E, E0 is the SAME P0 float32 cos/sin encoding. Negate it directly,
reshape/expand to96 rows. No recomputed theta encoding, normalization replacement,
parameter update or mask intervention. Before either controlled call, independent
float64 atan2/Q interpretation must establish ideal physical equivalence under
negated encoding (atol5e-14,rtol5e-12), with widths swapped and other physical
values equal. After capture, actual rounded world-frame Q must agree with P0
within original2e-7+2e-6 relative-element bounds. Verify other active material
inputs equal native P1-C. Record ideal versus rounded physical interpretation.

## Numerical comparisons and offline accumulation

Capture full arrays in memory for paired comparisons. JSON retains typed shapes,
hashes for EVERY captured array, full actual encodings/frames/material inputs,
full per-tape/mean RGB, and numerical summaries; it deliberately does not commit
millions of sample values. This limitation must be stated in the results.

Compare every captured field directly (with gauge-change caveats) and relevant
fields under explicit signed correspondence: E1 versus -E0; c1 versus -s*sin0,
sin1 versus s*cos0; T1 versus s*B0 and B1 versus -s*T0; local cx1 versus s*cy0
and cy1 versus -s*cx0. Compare WORLD h/wi/light/RGB directly. Do not label raw
quarter-turn frame differences as numerical error. Record raw-normalized head
and decoder encodings separately. Earliest numerical difference is measured
against the exact intended representation relation, not an intended width swap.

Report bitwise identity (including signed zero distinction), differing counts,
max absolute, RMSE, relative L2 (floor1e-30), per-tape statistics, ordered float32
ULP distance and error/reference-spacing where applicable. Collapse signed zeros
only for ULP distance, not bitwise equality. Near-zero/gauge differences can
make ULP/relative errors misleading; these are diagnostics, not new tolerances.
Use atan2(norm(cross),dot) for vector angles without repairing vectors; mask XOR
counts separately. Stage relationships and differences do not alone prove causes.

Offline only, replay original weighted-sample sum/count and each original
contiguous component-wise Torch four-tape mean exactly. Then compare:
float64 sums of STORED float32 weighted contributions, float64 per-tape RGB mean,
float64 residual/square/mean variants above. A changed hypothetical accumulation
is not a higher-precision shader. No extra forward is allowed for these analyses.

The controlled E arm isolates changing encoding construction in this fixed call.
Its downstream effects can establish a one-factor contribution, not that every
remaining error is sampling rather than BRDF arithmetic. No PDF causation without
an active dependency. H1 RGB mismatch, H2 objective amplification, H3 encoding,
H4 remaining sampling/accumulation, H5 gate interpretation require separate
confirmed/supported-but-not-isolated/contradicted/unresolved verdicts.

## Resources, freeze and reporting

One thread, hidden GPUs, maximum600 seconds whole execution,4 GiB peak RSS,
8 GiB address soft cap as backstop. Enforce exact call order; attempts and
completions separately. At most6 calls/576 points/294912 diffuse/589824 specular
directions. No derivatives or optimizer work. Restore thread/RNG/profiler/signal/
memory state; unexpected integrity failures invalidate the report. Check input
hashes after execution. Strict JSON validates BEFORE replacing owned partial
snapshots; any error preserves available data and stops. No automatic retry.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest tests.scientific.test_material_gauge_roundoff_v1 -v

# ONE invocation after source/test/protocol freeze; timeout includes imports:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 600s \
python3 -B -m experiments.material_gauge_roundoff_v1 \
  --output "$PWD/docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json"
```

Only new focused tests: exact state/tapes/evidence, algebra, small fixtures,
baseline identity rejection, one-factor invocation, no-grad/call/resource caps,
capture cleanup, strict serialization and immutable source checks. No historical
suite or experiment replay. Then freeze hashes of runner/tests/protocol and run
the single scientific invocation. Publish evidence/results and commit/push only
the feature branch; no merge or next experiment.

## Predeclared interpretation of a future criterion (unimplemented)

Do not fit a new constant to seed102. Retain original gate as historical outcome
and potentially a future warning. A proposed future protocol must first require
physical Q equivalence, correct chart/sign/tape/input identities, equal masks,
unchanged diffuse RGB where required and separate specular/total RGB component
checks under independently justified bounds (e.g. inherited forward-correspondence
bounds), so compensating component errors cannot hide in total RGB.

Objective discrepancy should then be explained by measured RGB discrepancy and
an arithmetic envelope rather than a fixed fraction of near-zero loss alone.
For an analytical candidate bound, n=72, u=2^-24, gamma_k=k*u/(1-k*u). For normal
finite residual arithmetic, use `epsilon_i = abs(delta_RGB_i) +
u*(abs(P0_i-target_i)+abs(P1_i-target_i))` (plus underflow allowance if needed).
The residual perturbation contribution is bounded by
`mean(2*abs(r0)*epsilon + epsilon²)`; square/mean roundoff can be conservatively
bounded by `gamma_(n+3)*(mean(r0²)+mean(r1²))`, AFTER the future protocol verifies
the relevant reduction operation-count assumptions. This is a predeclared
analytic error model, not an implemented new acceptance test, tolerance or
retroactive success. Such worst-case envelopes may be too loose: without strict
component/sample checks they could hide real estimator or configuration errors.
The results may recommend a separately reviewed trajectory protocol but cannot
implement or execute it.
