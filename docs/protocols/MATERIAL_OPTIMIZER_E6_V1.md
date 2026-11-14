# E.6 v1: physically scaled LM versus original direct Adam

**Preparation only; explicit researcher execution approval is required.**
Accepted base `438bb28d0f2fad046a50cf1941c0e44de399ffc9`, branch
`feat/neural-material-early-step-e2a`. Preserve the original scientific sources,
all E.1–E.5 evidence and their protocols. Use `python3` commands.

## Hypothesis and unchanged experiment

[E.5](../MATERIAL_OBJECTIVE_E5_RESULTS.md) found a lower objective at truth,
nonstationary neural endpoints and correlated width/albedo/metallic directions.
Test whether physical curvature and controlled steps improve **physical**
recovery beyond direct Adam. LM itself is established and is not a novelty claim.

Cases are fixed: **M2/103 and M3/101**, original N1 LR=.001 **state 100** from
accepted E.3. M3 is the contrasting case with good held-out RGB but inaccurate
Q and near-isotropic predicted orientation. Freeze exact physical values,
original head logits, signal archives, producing report paths and SHA-256.
No new initializations or selected best checkpoints.

Use unchanged `ObservationBatch`, A2/atan2, 512 diffuse/1024 specular samples,
known flat geometry/lighting, original 24 fitting observations and the original
84-observation evaluation table. Fit tapes 3000–3003 and evaluation tapes
1000–1003, same original private NumPy draws, are shared across both methods.
Reuse accepted independent fine targets/reference gates for M2/M3. No new
target generation, independent tape study, neural field or GPU execution.

Minimize **only original float32 total linear-RGB fitting MSE** after averaging
four tapes. Component targets, physical truth, Q, success criteria and held-out
metrics are reporting information only. They do not enter step selection,
damping, stopping, bounds or loss. No regularization, component supervision,
oracle initialization or threshold changes.

## Initial admission and derivative reuse

Before either optimizer runs, replay each endpoint's exact archived incoming
shader encoding and require its original fitting loss exactly. Decode captured
mx/my/kd/metallic logits plus saved theta using the original direct `physical`
function; require **bitwise equality** of the seven physical values. Fresh
separate LM and Adam forwards must produce identical RGB and objective. Both
initial gates for both cases pass before any update. Retain E.5 canonical
encoding bridge bounds: RGB max≤1e-6 and MSE delta≤`1e-10+1e-4*saved_MSE`.

Reuse E.5's passed M2 physical derivative admission, with frozen implementation,
runtime, target, samples and coordinates unchanged. M2's new Jacobians are
needed by the optimizer; do not repeat E.5's separate finite-difference study.
For M3, after the first algorithmic Jacobian and before its first proposal,
check its weakest scaled direction with central differences at both fixed
steps 1e-3 and 5e-4: four additional forwards, common samples, no extra Jacobian.
Require discrepancy L2≤`1e-4+.05*||J*v||` at each step. Save both errors and
direction. Invalid perturbations or disagreement stop the experiment; do not
adapt h, repair domains or retry. These M3 admission calls are charged to LM
in the work-matched comparison. All Jacobian primals must match cached current
RGB bitwise. Seven physical columns reuse E.5's double-reverse JVP helper.

## One fixed physical LM policy

Physical order is `[mx,my,theta,kR,kG,kB,metallic]`, with inherited scales
`S=[.96,.96,pi,1,1,1,1]`. For residual r=(predicted−target)/sqrt(72), compute
J=d(predicted)/d(normalized physical coordinates)/sqrt(72), H=JᵀJ and g=Jᵀr
in float64 from original float32 derivatives/residuals. Use **only total RGB**.
This differs from the older latent-coordinate LM and is separately versioned.

Solve `(H+lambda*I)d=−g` by float64 Cholesky and two solves. Verify the linear
solve residual within `1e-10*(||A||*||d||+||g||)` (floor 1e-30); fail closed on
nonfinite/failed solves. No pseudoinverse, extra jitter or alternate solver.
Initial lambda=.01, minimum 1e-8, maximum 1e8, matching the earlier declared
range. The identity damping is in **physically scaled coordinates**, not an
unscaled latent or diagonal-component loss.

Cap the normalized step L2 at **.1**. If it would cross a parameter bound,
shorten it to .99 of the first boundary distance. Widths are [.04,1], RGB and
metallic [0,1], with the accepted float32 representation of .04; theta remains
unbounded. Cast the proposed physical state once to float32, validate it and
compute actual represented normalized displacement `(candidate−current)/S`.
No post-hoc clipping, material repair, sorting or normal/BRDF change.

Predicted full-MSE reduction is `−2*gᵀd−dᵀH*d`, using that represented step.
Query a trial only if predicted reduction is positive and the state changes.
Accept iff rho=(old_MSE−trial_MSE)/prediction > .1 and actual reduction exceeds
`16*float32_eps*max(old_MSE,trial_MSE,1e-12)`. The numerical floor avoids
accepting roundoff-only gains. Accepted rho>.75 divides lambda by 3; rho<.25
multiplies it by 2; otherwise unchanged. Rejection multiplies lambda by 10;
clamp lambda to the fixed range. No parameter or held-out error is consulted.

Target **20 accepted updates** per case. At most four trials per Jacobian;
four rejections stop that arm as `censored_algorithm`, with no next trial
batch, reset or experiment retry. There are at most 20 Jacobians/80 solves and
trial attempts per case. Rejected, nonpositive-prediction and zero-represented
steps consume trial/solve budgets, even when no renderer query is made. All
queried failures consume attempted work. No gradient-based early success test.

Preserve native axis labels/theta throughout rendering. Axis interchange with
theta±pi/2 and theta+pi are ideal physical gauges, but the accepted finite-tape
sampler can depend on representation. Do not canonicalize query inputs or
eigenframes. Record decoder-branch crossings and measured acceptance; no hidden
chart repair. Q/orientation errors use original gauge-aware definitions,
including null orientation when either anisotropy gap is below .1. Isotropic
orientation sensitivity is allowed to vanish; damping handles rank loss without
adding an angle prior. Synthetic covariance tests concern the linear optimizer
when its input Jacobians transform covariantly, not claimed shader symmetry.

## Adam control and two predeclared comparisons

One fresh original direct-coordinate Adam history per case: original bounded
sigmoid transform, unbounded theta, LR=.01, betas(.9,.999), eps1e-8, decay0,
unchanged effective options, no scheduler/clipping/regularizer. Fresh moments
are deliberate; this is not restoration of neural Adam or an E.5 rerun. The
original physical family, initial state, renderer and objective match LM.

1. **Accepted-update comparison:** LM at exactly 20 accepted steps versus
   Adam at exactly 20 returns. Capture Adam state 20 using its ordinary next
   forward; no extra fit measurement. This comparison is available only if
   both reach 20. Never relabel outer attempts/rejections as accepted updates.
2. **Renderer-work comparison:** predeclare half-unit cost
   `C=2*(fit forwards+fit reverse sweeps)+7*external forwards`. Each fit/reverse
   covers 96 tape-observation points; each external covers 336. Jacobian
   construction costs its **seven actual forwards plus fourteen reverse
   sweeps**, not another double-counted Jacobian charge. All initial/terminal
   calls, M3 derivative checks, queried rejected trials and independent endpoint
   evaluations count. Also report Jacobians/solves/nonqueried trials separately.

Run LM first to its fixed accepted target or algorithmic stop, including one
final fit forward/physical-gradient reverse and one external evaluation. Its
actual cost fixes Adam length by a **score-independent mechanical rule**:
choose largest integer n≤250 with
`C_Adam(n)=4*n+4+7*(1+[n>20]) <= C_LM`.
This includes Adam's n fitting forward/reverse pairs, final fitting forward/
physical-gradient reverse, state-20 external evaluation when distinct, and one
final external evaluation. If n=20 reuse that one external record. There is no
held-out selection, favorable checkpoint or discretionary budget extension.

For successful LM target runs n>20; actual costs differ by only 1 or 3 half
units (0.5 or 1.5 fit forward/reverse units). Censored arms may have a larger
discrete gap, explicitly reported, and no matched-20 comparison. Both final
states are fixed by this policy. The E.5 50-step Adam result is context only,
not substituted for the new computation-matched control.

Equal C is a **declared query proxy**, not equal FLOPs or wall time. Forward
and reverse costs differ, JVPs retain derivative graphs, and 7×7 solves/I/O
are not renderer queries. Report F/R/Jacobian/trial/solve vectors, CPU and wall
time, common admission overhead and actual remaining cost gap. Arm clocks
include own diagnostic/output work; initial forwards occur in shared admission
and are charged to their arms. Do not claim runtime superiority from C or
accepted-update counts alone.

## Maximum work and bounded runtime

| Per case | Fit forwards | Reverse sweeps | External forwards | Updates |
| --- | ---: | ---: | ---: | --- |
| M2 LM | 222 | 281 | 1 | ≤20 accepted |
| M3 LM, including four FD calls | 226 | 281 | 1 | ≤20 accepted |
| M2 Adam | 249 | 249 | ≤2 | ≤248 returns |
| M3 Adam | 251 | 251 | ≤2 | ≤250 returns |

Additionally, two shared archived-encoding replay forwards. Aggregate maxima:
**956 shader forwards, 1062 reverse sweeps, 498 Adam returns, 40 accepted LM
steps, 40 Jacobians, 160 solves/trial attempts**. Counts may be smaller due to
acceptance policy/censoring; bounds are not completion targets. There are at
most 950 fitting-size calls and six external-size calls, 93,216 tape-observation
forward points, 47,726,592 diffuse and 95,453,184 specular forward directions.
Reverse work is separately bounded, not reported as new direction sampling.
Zero neural construction/query/training or new targets.

One CPU thread, GPUs hidden, accepted Python/NumPy/Torch arithmetic runtime;
1800-second whole foreground watchdog, 10-second kill grace; 120-second common
admission and 120 seconds per optimizer arm; 4 GiB RSS/8 GiB address space;
1 GiB output cap/2 GiB free disk. One fresh `material-e6-<id>` output under
`/home/isavand/paper/evidence`. No retry, revised tolerances, extended budget,
new arms, targets, sampling, Docker, installations or scientific-source edits.

## Evidence, tests and integration interpretation

Use existing Store/NPZ (`allow_pickle=False`), runtime cleanup and report/seal
patterns. Persist all accepted/rejected trials, actual/predicted improvement,
lambda, physical steps, gradients, singular values and weakest-direction step
fractions. Bind every saved Jacobian and final physical/Adam snapshot directly
into the sealed report, avoiding E.5's final-snapshot binding omission. Preserve
partial states/counters on failure. A `censored_algorithm` result is retained,
not called recovery success. Actual scientific completion requires both arms,
all initial gates, identity/tape/counter checks, normal resource/RNG cleanup,
seal and successful external process exit. Cleanup is acknowledged after exit.

Only new synthetic tests: scaling/JVP/FD correctness, bounds/Cholesky solves,
ideal gauge permutation, damping/rejection/acceptance, identical-interface
starts, exact stopping, work matching, incomplete work and execution gates.
Block real shader/neural imports. Preserve all historical validations. Freeze
sources/tests/protocol and actual JUnit/console/exit evidence before requesting
one execution authorization. CLI only prepares/verifies/describes; no automatic
execution approval is created. Later approval may permit the preparation commit
for the clean-checkout gate, one supervised execution, offline figures/report,
final commit and one ordinary push attempt.

After approved execution, report all fixed endpoints, physical Q/axis/angle/
RGB-albedo/metallic errors, original success flags, held-out/relighting and
component compensation, costs/conditioning/acceptance and numerical failures.
Weak-direction displacement fractions are diagnostics, not supervised losses.
Compare both cases; no population or global-identifiability claim.

A successful direct optimizer would motivate **physical-output corrections
followed by neural-field projection**, or Jacobian-informed head updates. A
neural Jacobian maps weights to physical values, may be rank deficient or
ill-conditioned, and projection can change other spatial points. Physical
gauge, nonlinear bounds, finite-tape coupling and projection accuracy require
new controls; no integration is implemented here. LM success alone is not a
novel neural method. If gains are absent, retain the negative result rather
than extend the run. Preserve observation/lighting design, neural conditioning,
bounded mappings and geometry-to-material uncertainty in the research backlog.
