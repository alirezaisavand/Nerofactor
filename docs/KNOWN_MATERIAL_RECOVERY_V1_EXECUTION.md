# 3C-1 execution specification — frozen before the single run

Approval: the researcher explicitly authorized the first flat, noise-free v1
experiment, its success criteria, five initializations, CPU execution and
commit/push on `feat/known-material-recovery-v1`. Base:
`179e36eb01e976d216a45fd323cc764308b49bd0`.
The [approved protocol](protocols/KNOWN_MATERIAL_RECOVERY_V1.md) is preserved
byte-for-byte, including its earlier proposal-status wording. The later user
approval supplies execution authorization; it does not revise that protocol.
This file resolves implementation details **before** execution. Its hash and
the hashes of all numerical code enter the one-run report. Do not edit it after
the run; put interpretation in the separate results document.

## Fixed scope and ordering

Only the flat case, p=(0,0,0), n=+Z, original base tangent+Y/bitangent-X,
truth `(mx,my,theta,kR,kG,kB,m)=(.20,.55,.31,.30,.45,.60,.25)`.
A2/atan2 with diffuse512/specular1024, no priors/noise/scheduler/clipping.
No frozen-source edits, old experiments, native datasets/checkpoints, neural
networks, GPU execution, Docker, dependency installation or other scene/variant.

World direction and light definitions are exactly the approved protocol.
Observation ordering is fit24, held-out24, relight36, with views outermost and
lights innermost. Train views are polar20/50 crossed with azimuth0/120/240;
held-out views polar35/65 crossed with azimuth60/180/300. Light0 is constant;
lights1..3 use polar40/azimuth0,120,240/power8; lights4..6 use polar55/
azimuth60,180,300/power16. Floor .1, amplitude .9, known white illumination.

Array layout: observation, component(diffuse/specular/total), RGB. All 84 target
observations use independent float64 quadrature; the fitted CPU source shader
uses float32. Fit targets are explicitly rounded to float32 for its MSE;
report their hash and retain the original float64 arrays. No gamma conversion
or display clipping enters the objective or comparisons.

## Independent reference and mandatory gates

Use the two prescribed 256x512 and512x1024 Gauss-Legendre-cosine/midpoint-
azimuth grids. The NumPy reference never calls the shader, sampler or source
radiance method. For the ideal Cartesian Beckmann/Kurt a=0 kernel, Fresnel is
affine in F0: `F=s+(1-s)*F0`, `s=(1-dot(wo,h))^5`. Integrate its two scalar
coefficients against each light, then combine RGB F0. Cache only identical
per-view kernels across lights/channels. Diffuse is integrated independently
as d/pi. This algebraic reuse changes work counts, not the integral or grids.
Count actual per-view directional kernel evaluations and precomputed lights.

Require all target channels, components and totals to meet unchanged
`abs(fine-coarse) <= 2e-6 + 5e-5*abs(fine)`. Totals additionally require their
components to converge. Any failure stops before shader/optimizer construction.
Keep both grids and every failed value; no target substitution or grid retry.

Compute independent reference Jacobians only for the 24 training observations,
using all seven physical variables and both central step sizes1e-4/5e-5 times
scales(.96,.96,pi,1,1,1,1). Every plus/minus observation must first pass the same
reference convergence gate. Preserve all28 perturbation evaluations, both-grid
Jacobians, physical parameter order and scale-normalized columns:
`J_normalized = J_physical * parameter_scale`.

At each step size, the fine-grid SVD must have `s_min/s_max >= 1e-4` for seven
parameters. Record full singular spectra/right vectors/null directions and all
cross-step/grid discrepancies. Operationalize the protocol's stable-difference
requirement conservatively without a tuned numeric tolerance: the spectral norm
of each observed step/grid Jacobian discrepancy must be smaller than the
smallest retained fine-grid singular value. This is a matrix-perturbation rank
margin check, not a guarantee against all integration/derivative error. Failure
stops; no threshold adjustment or redesigned views/lights. Local rank cannot
prove global uniqueness or optimization success.

All five initial vectors are recorded before either gate. Gate failure leaves
them `not_started`, with null final parameters/metrics; do not fabricate recovery
or held-out errors. No optimizer or shader is constructed until both gates pass.

## Exact bounded fit and evaluation

For each private RandomState seed101..105 draw in this order: two widths
uniform[.15,.75], three albedos[.20,.80], one metallic[.10,.90], then theta
[-pi/2,pi/2). Convert widths/albedos/metallic to their sigmoid logits and theta
to its unchanged scalar coordinate. The optimizer vector order remains
(mx_logit,my_logit,theta,kR_logit,kG_logit,kB_logit,m_logit). Save the original
float64 physical values, actual float32 logits and decoded float32 start.
Use exactly the approved parameterization and encode theta as cos(2theta),
sin(2theta). No angle canonicalization or branch-cut repair in optimization.

Adam is CPU-only on one seven-scalar tensor: lr.01, betas(.9,.999), eps1e-8,
weight_decay0, default remaining runtime flags. At most250 successful returns
per start,1,250 total. No schedule, clipping, regularizer, neural optimizer,
validation-based selection, retry or extra update.

Generate fit tapes from private NumPy RandomState seeds0..3 with arrays u[24,1024],
v[24,1024], then azimuth[24,1,1], converted to float32. Final evaluation tapes
use seeds1000..1003 and corresponding84-row arrays. Preserve shapes/seed/array
hashes and verify tapes were not mutated. Reuse fixed fit tapes across every
start/update. Seed-major point batching is only vectorization of independent
observations; average four linear radiance estimates **before** MSE. No global
RNG is seeded or consumed. Invalid rounded uniforms fail rather than being
resampled or clipped by infrastructure.

Each update computes the current full-fit loss/gradient, then calls original
Torch Adam once. Log update1 and every10 returned updates: pre-update loss and
gradient norm, before/after parameter norms, post-update physical parameters,
time, finite state and successful counts. A log labeled completed250 has its
loss evaluated at249, and post-update parameters at250. No extra post-update
forward on fit tapes is added beyond the declared fitting budget. Final-state
fit/held-out/relighting metrics all use the separate evaluation tapes, clearly
labeled; no final249 estimate is reported as an evaluation of250.

Evaluate all84 observations at each final250 state. Report components and raw
predictions, fit24/held-out24/relight36 errors, plus relighting split into18 seen-
view and18 unseen-view observations. Parameter/Q/axis/orientation/albedo/metallic
errors use approved gauge-invariant metrics. Retain all five starts and pairwise
Q distances, raw-parameter median/ranges labeled gauge-dependent, optimizer state
and stop reasons. No favorable-start-only report. Preserve numerical failures;
stop the experiment on a fatal invalid/nonfinite/runtime condition rather than
retrying or silently completing remaining starts.

Success is exactly: relative Q<=.05; identifiable orientation<=5 degrees;
max RGB albedo absolute error<=.02; metallic absolute error<=.03;
held-out relative radiance L2<=.01. No added relighting success threshold.
Always report continuous errors even when criteria fail. Forward correctness,
local identifiability, bounded optimization and held-out generalization remain
separate conclusions.

## Resource limits, outputs and commands

Preserve protocol maxima: target55,050,240 and Jacobian440,401,920 reference
node evaluations (before exact algebraic caching); fit120,000 shader points /
122,880,000 specular +61,440,000 diffuse directions; final evaluation1,680
points /2,580,480 directions. Record actual vectorized call counts separately.
No other sample budget or scene sweep. Fresh explicit output with exclusive
creation; never overwrite old evidence or resume a prior attempt.

One process with Torch/BLAS/OpenMP thread counts1 and GPUs hidden. Internal
wall checks and SIGALRM plus external timeout enforce1,800 seconds; termination
cleanup does no additional numerical experiment work. Preserve partial JSON if
stopped. Restore owned signal handlers/thread count and verify unchanged global
CPU RNG states. CPU time/wall time include gates, optimization and report writes;
external cap also includes interpreter/import startup. No automatic extension.

Run only NEW unit fixtures first:

```bash
cd /home/isavand/paper/Nerofactor-paper
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p test_known_material_recovery_v1.py -v
```

Then execute **once**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
  python3 -B -m experiments.run_known_material_recovery_v1 \
  --output /home/isavand/paper/Nerofactor-paper/docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json
```

No numerical override flags are implemented. Gate stop returns status2; failure
retains diagnostics; complete means the fixed protocol finished, **not** that
all recovery criteria passed. Interpret the one-run evidence read-only afterward.
Commit/push only the feature branch; do not merge or begin another experiment.
