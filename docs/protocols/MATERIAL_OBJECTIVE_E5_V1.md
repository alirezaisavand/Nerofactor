# E.5 v1: physical identifiability and fixed-objective diagnosis

**Preparation only. New scientific execution requires explicit researcher
approval of this protocol and its freeze hash.** Branch
`feat/neural-material-early-step-e2a`, accepted base
`a832f020d4d9618295ccfe818131a0f8193eb248`. Preserve all accepted sources,
protocols, targets and evidence. Use `python3` only.

## Question, comparison points and interpretation

Diagnose M2/103 without another neural recovery experiment. Three primary
physical materials are fixed before execution: original M2 truth, original N1
LR=.001 state-100 endpoint from accepted E.3, and AS1 state-100 endpoint from
accepted E.4. Their exact float32 endpoint values and source identities are in
the freeze. Original N1/AS1 Q errors are .4246977180/.6280131958 and held-out
RGB relative errors .0433210268/.0609149235. Include existing M1/103 and M3/101
component-compensation results only as saved-data context, with no new runs.

Preparation also derives each endpoint's physical gradient from the already
saved shader-input gradients: sum repeated mx/my/kd/metallic contributions and
contract the encoding gradient with `2*(-e_y,e_x)` for a fixed-radius angle
displacement. This is a saved-data chain calculation, not a new autodiff call
or proof of a smooth descent path through sampler discontinuities. Keep its
actual-encoding provenance distinct from the proposed new canonical-encoding
AD/finite-difference admission.

Hypotheses, none assumed true:

1. Truth is favored but endpoint physical gradients remain nonzero: incomplete
   optimization is supported; this alone does not attribute failure exclusively
   to neural architecture rather than optimizer/coordinate geometry.
2. Truth/endpoints change preference across sample quartets: finite-sampling
   sensitivity contributes. A fixed finite-sample optimum need not equal truth.
3. Small local singular values, correlated parameter sensitivities and
   cancelling diffuse/specular residuals suggest weak local constraints.
   They cannot establish global intrinsic non-identifiability.

Direct-probe improvement is local evidence only. Better RGB with worse Q is
retained. No endpoint selection by held-out error, adaptive optimizer or
post-result threshold change is permitted. No novelty is claimed.

## Unchanged physical rendering and references

Use original `run_known_material_recovery_v1.ObservationBatch`,
`MaterialShaderV1('A2','atan2')`, 512 diffuse/1024 specular samples, known flat
normal +Z and accepted view/illumination definitions. Preserve the ideal
Beckmann model and actual finite estimator, its clamps, masks and gradients.
Physical order is `[mx,my,theta,k_R,k_G,k_B,metallic]`. The Q definition remains
`R(theta) diag(mx²,my²) R(theta)^T`, slope covariance Q/2. No BRDF, target,
sampling distribution, regularizer or neural-field change is allowed.

Reuse admitted independent coarse/fine radiance targets from
`docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json`, SHA-256
`3abc274191d0e84c456355a7c351ac1aa94a2c976c1f789979ddb550a4383e89`.
Reuse its successful reference convergence/rank gates, not their execution.
Record the accepted coarse/fine discrepancy in subsequent interpretation; it
is a convergence diagnostic, not a rigorous error bound or sampling confidence
interval. **Zero new reference generation.**

There are 24 fitting, 24 held-out and 36 relighting observations, in their
original ordered 84-observation table. Fit uses original float32 total-RGB MSE
against float32 targets after averaging four tapes. External component metrics
retain original float64 metric arithmetic against accepted fine references.
Report fitting MSE and separate external fit/held-out/relight errors; they use
different fixed tapes and must not be conflated.

## Sampling and explicit gauge/encoding controls

Predeclare three fitting quartets: accepted 3000–3003, independent 4000–4003,
independent 5000–5003. Each uses the original private NumPy `RandomState`
generation and draw order for 24 points per tape. Existing evaluation quartet
1000–1003 uses 84 points per tape. All arrays, dtypes and hashes are frozen
before scientific calls; no replacement seeds or sample increase. Comparisons
between points and finite differences use common random numbers **within**
each quartet. Record all four individual tape predictions and their losses.
Quartet MSE is loss of the mean, not mean single-tape loss. Three quartets are
too few for population uncertainty claims; report range and paired differences.

The physical adapter uses the **existing direct control's** cos/sin
double-angle encoding. The original N1/AS1 encodings can differ in final
float32 bits. First replay each saved endpoint using its exact archived shader
encoding, without constructing N1/AS1: exactly reproduce its saved fitting
loss or stop. Then measure the canonical direct-encoding bridge separately.
Require maximum fitting RGB delta≤1e-6 and MSE delta≤`1e-10+1e-4*saved_MSE`.
These predeclared provisional diagnostic bounds guard against a material
substitution; they do not claim bitwise equivalence or a universal rounding
bound. Preserve actual differences. No tolerance adjustment after execution.

Q is invariant under theta+pi and axis exchange plus pi/2, but the inherited
fixed-tape estimator is gauge-sensitive. Do **not** canonicalize renderer
inputs. Include one explicitly labelled auxiliary point, truth with widths
exchanged and theta decreased by pi/2: three fitting queries and one external
query, no derivatives or optimization. Record its float32 Q mapping error.
This tests gauge-induced sampling differences rather than treating equivalent
physical descriptions as new truths. Align small/large-width Jacobian columns
only in offline reporting. The unchanged .1 near-isotropy rule gives null,
never zero/passed, orientation scores.

## Physical derivatives and local sensitivity

At each primary point and each fitting quartet (nine combinations), measure
one base forward and actual physical MSE gradient. Seven scaled columns use
the accepted double-reverse directional-JVP method: seven forwards and fourteen
reverse sweeps per combination. Each primal must match the base output
bitwise. No neural field or neural Jacobian is constructed.

Use the inherited physical scales `S=[.96,.96,pi,1,1,1,1]`, for normalized
coordinates `p+S*d`. Retain full Jacobians of diffuse, specular and total
linear RGB (24×3×3×7). Report scaled physical gradients and
`-||S*grad_p||²`, with finite-difference admission, as a local pathwise descent
diagnostic; no arbitrary claim of practical stationarity from its magnitude.
Check `2*J_total^T*r/72` against the scaled autodiff MSE gradient within
`1e-7+1e-3*||scaled_gradient||` in L2.

Numerical derivatives use fixed central differences at **both** normalized
steps 1e-3 and 5e-4: 28 forwards per point/quartet. All signs/columns use the
same samples. Retain the two full FD matrices, not just flags. For every
seven-column full-component Jacobian, require each step's column discrepancy
L2≤`1e-4+.05*||J_column||`. This is a float32 piecewise-sampler check, not a
reference-accuracy criterion. Horizon/clamp discontinuities can cause a genuine
derivative-admission failure: retain it, stop before the direct probe, and
request scientific review. Do not change h, smooth masks, switch derivative
methods, retry or select the favorable step.

For total RGB, form J/√72; for separate diffuse/specular components stack
both and divide by √72 (sum of component MSE objectives, not their average).
Report singular values, effective rank at the inherited 1e-4 relative ratio,
weak scaled right-singular direction, column cosines and local Gauss–Newton
matrix `2*J^T*J`. Report diffuse/specular spectra separately. Zero columns are
explicit and zero-matrix rank is zero. Singular-vector signs have no meaning;
directions in nearly repeated singular spaces are unstable. Scaling affects
condition numbers. GN is not the full Hessian or a global inverse covariance.
No extra confidence or identifiability claim is inferred from this local audit.

## One direct-parameter control

Only after all nine derivative admissions pass, start **one** 50-update direct
probe from original N1 M2/103 state 100. Use the exact captured original
mx/my/kd/metallic logits and saved physical theta. The original
`run_known_material_recovery_v1.physical` must reproduce all seven saved
physical values **bitwise before any update**. No approximate logit inversion,
ground-truth initialization, neural model or old optimizer-state restoration.

Use the accepted direct control's transform and fresh Adam: LR=.01,
betas (.9,.999), eps 1e-8, weight decay 0; no scheduler, clipping or regularizer.
This is explicitly a new direct-coordinate local probe, not neural-state
continuation. It changes coordinates and optimizer history, so it cannot
identify neural architecture as the sole causal factor. Use accepted fitting
tapes 3000–3003 and the identical float32 objective. Initial fitting loss must
equal the earlier canonical N1 loss exactly. Exactly 50 Adam returns, no early
success selection. Record each pre-update loss/latent gradient, actual float32
movement, physical parameters and original recovery errors. Measure one final
fitting forward/physical gradient and one external forward. Preserve failures.

## Complete execution budget and limits

| Work | Forwards | Reverse sweeps | Adam returns |
| --- | ---: | ---: | ---: |
| Exact archived-encoding endpoint replays | 2 | 0 | 0 |
| Nine base physical measurements | 9 | 9 | 0 |
| Nine seven-column directional Jacobians | 63 | 126 | 0 |
| Nine two-step central-difference checks | 252 | 0 | 0 |
| Three auxiliary truth-gauge fits | 3 | 0 | 0 |
| External truth/N1/AS1/auxiliary/final-probe evaluations | 5 | 0 | 0 |
| Direct optimization and final fitting gradient | 51 | 51 | 50 |
| **Total** | **385** | **186** | **50** |

380 fitting-sized calls (96 tape-observation points each), five external calls
(336 each): 38,160 point/tape combinations; 19,537,920 diffuse and 39,075,840
specular directions. No new neural construction/calibration/forward/backward,
GPU work, reference generation or full recovery history. Counters separate
attempts from successful completions and fail before exceeding any phase cap.

Reuse the accepted foreground supervisor/watchdog pattern: one launch,
1,800-second external timeout, 10-second kill grace, one CPU thread, hidden
GPUs, accepted Python/NumPy/Torch arithmetic runtime, 4 GiB RSS, 8 GiB address
space, 1 GiB output cap and 2 GiB free disk. Each sensitivity combination,
replay/gauge/external stage and direct probe has a 120-second active cap. No
retry, budget extension or settings change after results. No packages/Docker.

Outputs go to a fresh `material-e5-<unique-id>` directory under the already
approved `/home/isavand/paper/evidence`. Original raw evidence is read-only.
Store small reports plus numeric NPZ arrays (`allow_pickle=False`), original
and effective identities, counters, per-tape predictions, Jacobians, gradients,
component metrics and direct trajectory/final Adam state. Preserve partial
records and explicit failing stage on any interruption. Completion requires
all counters, terminal source/input/tape checks, normal RNG/resource cleanup,
report+seal and successful foreground process exit. Cleanup is acknowledged
only after the limits context exits normally.

## Preparation, authorization and later analysis

Focused tests use synthetic algebra/fake renderers and block original
shader/neural imports. Verify physical-transform derivatives, scaled JVP/FD
agreement, gauge/isotropic interpretation, private fixed samples, accounting,
failure gates and observation-only captures. No historical suite or scientific
experiment is repeated. Hash the actual JUnit/console, exit code and test
inventory, then create the versioned freeze. CLI permits preparation only:

```bash
python3 -m pytest -q -p no:cacheprovider tests/scientific/test_material_objective_e5_v1.py
python3 -m experiments.material_objective_e5_v1 --describe
python3 -m experiments.material_objective_e5_v1 --prepare
python3 -m experiments.material_objective_e5_v1 --verify-sha256 <freeze-sha256>
```

Request one explicit execution approval binding freeze, fresh output, full
work and resource limits. That later approval may authorize committing frozen
preparation for the clean-checkout gate, one supervised launch, offline
finalization, final commit and one ordinary push attempt. **No affirmative
execution authorization is created during this preparation.**

After successful approved execution, saved-data analysis will compare all
points/quartets, ensemble and single-tape loss variation, physical/photometric
changes, gradient/singular-vector correlations and component cancellation.
Make objective, spectral/direction, component and direct-trajectory figures.
Keep all original success thresholds; no formal population inference from this
single case. Separate optimization, estimator sensitivity and weak local
constraints rather than forcing a single explanation. Rank physical
preconditioning/trust regions, gauge-aware updates, compensation controls,
illumination/observation design and hybrid direct/neural optimization by what
the measured mechanism supports; implement none automatically. Preserve the
broader E.4 research backlog. Final decision is PROCEED / FIX AND PROCEED /
BLOCKED with limitations and the smallest next controlled proposal.
