# E.16 — Observation-guided native full-field material recovery, v1

Preparation only until one explicit approval binds the final commit/freeze,
image, inputs, work/resource limits and fresh output. Preserve E.11–E.15, all
frozen G/A/variant implementations, checkpoints, failed gates and raw evidence.
No Bell regeneration, E.13 rerun, new lighting/material family or unrestricted
training. This is a sparse curved-surface neural-method test, not a full object
reconstruction or a new pointwise observability experiment.

## Paired hypotheses and data

For each of the original and stronger material fields run two neural histories:

- **A:** fresh corrected native full material field, photometric optimization.
- **B:** identical initial weights; uniformly distill observation-derived world Q
  at all84 coordinates, then the same photometric optimization.

Use the exact E.15 acquisition:84 fixed physical points,220 admitted point/view
correspondences,137 fitting /83 withheld views, camera order[58,44,9,0], L2 and
its fixed45-degree world rotation. No replacement points or views. Known geometry,
normals, color, metallic and illumination are fixed in both histories. The same
single frozen initialization seed6033 is used across all four models; this is
not a multi-seed robustness or population experiment.

Only E.15 **free-ratio /two-light /continuous** fitting-selected endpoints are
teachers: original panel03 and stronger panel11. Require accepted report SHA256
`78245f0ab91c2d18b83c87e36556c79e618859ccfeb7f8cd956988a98c38178a`,
plus exact endpoint/trajectory hashes and fitting-only argmin identities. Do not
use oracle-fixed-ratio estimates, GT parameter errors or held-view scores to
select a teacher. All84 estimates are used uniformly, including inaccurate,
failed-to-converge and quadrature-flagged cases. No new physical optimization.

Training inputs expose only positions, known normals/frames/color/metallic,
fitting RGB and teacher Q. The teacher extractor never reads the archive's truth,
held-view prediction or physical-error fields. Evaluation truth is used only by
snapshots; their scores do not affect training, matching, stopping or selection.
Fitting observations count points with more views more often; distillation is
uniform per physical point. This difference is explicit, not a confidence weight.

## Actual native field and physical conventions

Use pinned A `e1e859779301faced46d6476691e20b546c27ae0` with the unchanged E.10
`A_corrected_core_v1`. Instantiate the real MCShadingNetwork and its original
MaterialFeatsNetwork and five predictors. Native material capacity must remain
**1,485,584 parameters /84 parameter tensors** in both histories. Query actual
Bell positions, with the original positional encoding and shared features; no
constant anchor, calibration, reduced network or head-only restriction.

All shader parameters retain requires_grad=True and enter Adam. Shared features
and mx/my/rotation predictors receive the three-coordinate losses. Known color,
metallic and fixed lighting bypass their learned predictions; those head/light
weights must remain bitwise unchanged by absence of gradients. Their unused
outputs can change as shared features learn; they are not physical inputs.

Preserve the native [.04,1] width transformations, normalized double-angle encoding,
atan2 decoder, X/Y tangent construction and literal original trace-normal negation.
Do not sort/relabel network heads during training. For native rotated axes a,b:

`Q_world = mx²*a*a^T + my²*b*b^T`.

World Q is unchanged by theta+pi or by swapping widths with theta+pi/2. Distill
the full tensor, not a chosen angle or axis ordering. Verify accepted/native base
frames agree within2e−6 before training; do not repair or flip them. Compatibility
teacher angles are explicitly projected from the accepted frame into the native
frame. The differentiable control chart is float64; widths/encodings passed into
the actual shader remain float32. Independent snapshot comparisons map native
axes back into the accepted reference chart, disclosing frame roundoff.

The native shader uses512 diffuse /1024 specular samples, random azimuths,
known distant environment and zero indirect bounce; human light and all material
priors are off. Its actual anisotropic shading/lighting methods run, and the
linear auxiliary radiance is used directly, without inverse/clipped sRGB. This
is a separate paired adapter; legacy Trainer/entry points and shader sources
are unchanged. The separate energy-budget family is not selected.

## Bounded native/reference admission before any Adam update

Instantiate all four models and verify exact full-state and initial prediction
parity before training. Use the existing corrected image/BVH, original geometry
initializer and original trace() method. Original mesh bytes are read-only.
Vertex normals may be computed in memory by the unchanged initializer; native
tracing still uses its original face-normal convention. This does not create a
new admitted research G-to-A handoff.

At each teacher field evaluate all137 fitting correspondences under both lights,
using32 independent native draws. Compare total, diffuse and specular against
the saved E.15 deterministic teacher predictions and the accepted known diffuse
component. No new target images are generated. Each repeat has a distinct seed;
the two lights share random numbers at fixed material inputs.

Also compare native pathwise derivatives in(theta,u,v) to exact analytic
derivatives of the independent finite quadrature. Use only camera58 at indices
floor(linspace(0,83,6))=[0,16,33,49,66,83], selected by fixed geometric ordering,
not responses. Both materials/lights and every RGB channel are retained. Reuse
saved256²/512² nodes and incident visibility for the independent calculation.
Native gradients include the original hard visibility behavior; do not add a
boundary correction or assume these gradients are unbiased.

For repeats x, deterministic reference r and grid proxy g=abs(r512−r256), compute
mean m, sample standard deviation s(ddof1), and empirical SE=s/sqrt(32). Require:

`abs(m−r) <= a + 6*SE + g + 64*eps_float32`, componentwise,

and `norm(m−r) <= b*norm(r) + norm(6*SE+g+64*eps_float32)`.

Forward budgets: a=.001 unit linear RGB, b=.02. Derivative budgets: a=.001
RGB per declared coordinate, b=.10. Additionally require
`norm(SE) <= a*sqrt(number_of_components) + b*norm(r)` so excessive variance
cannot automatically pass by widening the comparison. These are prospective
practical error budgets; they are not tuned to this execution. Six empirical
SEs are descriptive, not a rigorous simultaneous confidence guarantee. The
deterministic grid proxy is separate from stochastic variance and is not an
absolute-accuracy proof. Record every mean, sample, reference, gradient and error.

**Any failed admission stops before distillation or photometric updates.** Preserve
partial files and failure reports. No extra samples, threshold changes, alternate
renderer, bias repair or automatic retry. A pass establishes bounded compatibility
on these controls, not global gradient correctness or a globally watertight BVH.

## Uniform observation-guided initialization and photometric training

Arm B performs exactly200 full-field Adam returns minimizing

`mean_points ||Q_pred−Q_teacher||_F² / ||Q_teacher||_F²`.

The denominator is teacher-only, positive and checked. No GT tensor, confidence
weight, ratio constraint, orientation target, learned light or additional prior.
Evaluate and retain the last distillation state even if it degrades true recovery.
No best-distillation checkpoint is selected.

Use a **fresh** Adam state for B photometric training after distillation; A also
starts photometric Adam fresh. Both receive exactly1000 photometric updates at
the primary endpoint, minimizing the same mean squared **linear-RGB** residual
over137 fitting correspondences×two lights×RGB. Field inputs and native samples
are identical in definition; outputs differ as intended after the intervention.

Both optimizers use LR.001, betas(.9,.999), eps1e−8, AMSGrad, weight_decay0 and
gradient norm clip.7, with no scheduler or early stopping. These are explicit new
paired-experiment settings, not a claim to preserve the legacy Trainer's eps1e−4
or warmup/cosine schedule. The legacy path remains unchanged. All full-field
parameters stay in the optimizer; no head freezing or teacher filtering.

Training repeat seed is800000+zero-based photometric update. Both lights,
histories and material fields use the same seed at matching update indices.
The original native sampling methods consume their original draw shapes/order;
no redraw or valid-only normalization is introduced. Each call forks/restores
CPU/CUDA RNG. Evaluation seeds900000–900007 are independent of fitting seeds;
compatibility seeds700000–700031 are separate. No simulated camera noise.

Record each objective, unclipped group gradients, clipping norm, Adam returns and
state, periodic weights/moments and partial progress. Unexpected missing gradients,
nonfinite values, invalid encodings or scientific-identity/resource failures stop
the execution. Expected inactive color/metal/light parameters are verified unchanged.

## Matched-update and approximate compute comparison

Primary: A1000 versus B200-distillation+1000-photo. Report the extra teacher and
distillation costs, rather than claiming equal training cost.

Secondary: after B finishes, continue A from its exact1000-step Adam state until
its measured active training time reaches B's photo time +distillation time
+72.43427120149136 seconds. The last term conservatively charges the **entire**
accepted E.15 experiment to each guided field because per-panel timing is absent.
Report exact inherited teacher-panel direction/Jacobian counts separately; do not
pretend deterministic BRDF contractions and native traced samples are equal FLOPs.

Cap baseline continuation at1000 additional updates (2000 total). The stopping
rule uses time only, never GT/withheld/teacher-error quality. GPU synchronization
includes completed Adam work in active timing. Evaluation, logging and checkpoint
I/O are excluded from active time and reported as overhead. If capped below the
target, label the comparison unmatched; do not extend. Runtime drift, conservative
teacher charging and unequal kernels make this approximate, not exact compute
equivalence or a speedup result. No learning-rate or budget tuning after outcomes.

## Fixed evaluation, diagnostics and interpretation

Capture initial, B post-distillation, primary A/B photo1000, and A compute-endpoint
states for each material. All84 points remain in every table. Record native Q,
world/local relative Q error, sorted width errors and orientation modulo physical
equivalences. Keep the existing Q≤.05 and eligible orientation≤5-degree descriptive
counts. Eligibility remains truth major−minor≥.1; predicted near-isotropy is
explicit, not excluded to improve a score. Known color/metal are not recovered.

For each state use8 common independent native draws over all220 correspondences
and both lights. Save linear components and float64 summary means; also compute
independent deterministic256²/512² predictions at the same neural material. This
separates material error, native-estimator discrepancy and repeat variance. It is
evaluation-only and never changes training. Preserve per-point and pooled errors,
fitting versus83 withheld views, all-start teacher failures and native/independent
differences. There is no spatial-position holdout or unseen-light test: both
lights are observed, and only views are held out. No such generalization claim.

Report actual oracle-round-to-nearest8-bit agreement for the native mean and for
individual draws, with clipping counts. This is explicitly different from legacy
PNG truncation. No native output, reference or boundary code is repaired. Compare
post-distillation with final states to expose any physical degradation.

Native single-draw MSE has an estimator-variance term as well as squared bias.
Even a mean-unbiased renderer can induce a different noisy objective. Record
single-draw versus mean prediction losses; do not attribute every failure to
neural representation or teacher quality. Hard visibility can also bias pathwise
gradients. The compatibility gate and endpoint comparisons delimit these issues,
without silently implementing a new estimator or loss.

The optional **diagnostic only** uses saved E.15 fitting MSE, near-fit multistart
counts(RMSE within1e−4 of the best), Q disagreement and last-pretrial normal
eigenvalues. It never reads truth/held-view errors to form confidence. Eigenvalues
are not a calibrated covariance. No uncertainty weighting or extra ablation is
implemented in the primary experiment. Offline correlation with evaluation truth
is descriptive and never feeds back into training or selection.

All single-seed and sparse-surface limitations remain. Success means improved
actual neural Q recovery at fixed endpoints, not only lower RGB loss or faster
convergence. If supported, propose a full float-linear multiview/multilight Bell
benchmark and controlled native estimator/initialization ablations. If not, use
the recorded teacher, distillation and estimator diagnostics to identify one
method-level change. Do not start another study under this authorization.

## Exact ceilings, storage and execution

- At most6000 photometric +400 distillation Adam returns; no prior experiment rerun.
- At most12000 photometric /128 compatibility /192 evaluation native shader calls.
- At most6000 native photometric backwards,400 field-only backwards and384
  compatibility reverse sweeps. No field optimization before compatibility pass.
- At most **2,616,999,936 new shadow rays** through the unchanged corrected BVH.
  Native material-dependent samples require new visibility; quadrature masks
  cannot be substituted. No new primary/outgoing camera-ray selection.
- At most6412 full-field queries,869,007,360 independent BRDF point-directions,
  72,089,600 coefficient /55,050,240 environment point-directions, using cached
  quadrature geometry. No reference images/dataset regeneration or new lights.

Use existing image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
corrected tracing binary
`800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`,
A100GPU0,4 CPU threads,64GiB host RAM/no swap,pids128,25% CUDA allocator and2GiB
owned outputs. One foreground3600s watchdog/30s kill grace, inner3480s/20s;
estimated5–20 minutes, unmeasured. E.13's accepted2000-update core run took403.6s
at512 training rays; E.16's paired batch has274 shaded observations per update
but extra checks/known-light work. This estimate is not a performance guarantee.

Read-only source, benchmark, mesh, all legacy/evidence inputs and root filesystem;
non-root, no network, builds, installs or permission changes. A fresh hash-bound
output under `/outputs/neural` contains snapshots, own numerical checkpoints,
Adam moments, confidence, progress and the final report. Nothing writes into
datasets, scientific sources, the repository or historical evidence. Partial
checkpoints are preserved; no resume or retry is implied. Cleanup success requires
normal resource-context exit and verified RNG/thread/default-type restoration.
If Docker is unavailable to Codex, use the identical approved researcher-side
foreground tmux launcher without bypassing restrictions.

The native test is still unexecuted during preparation. Focused CPU tests use
small stand-ins and admitted method bodies, never the full native field or GPU.
Archive full test evidence, commit the reviewed preparation for its clean-source
gate, request one explicit execution approval, then complete that single run's
file-only analysis, tables/figures and results commit/push. No unrelated preparation
phase, source repair or full training expansion is authorized.
