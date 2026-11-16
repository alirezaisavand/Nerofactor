# E.15 — Single-/two-light physical recovery, v1

Preparation only until one explicit approval binds the final commit, freeze,
input identities, image, work/resource limits and fresh output. Preserve E.11–E.14,
their source freezes, checkpoints, failed criteria and raw evidence. This does
not authorize a dataset, native neural training, light search or retries.

## Scientific question and controls

Test whether the E.14 angular-information gain translates into physical recovery,
and whether fixing the weak log-ratio makes the remaining coordinates recoverable.
Use all 84 original points, known geometry/frames, color and metallic, the two
fixed lights, independent additive Kurt BRDF and accepted outgoing visibility.
Evaluate original and E.11F stronger materials. E.14's failed sensitivity screen
and E.13's failed physical recovery remain unchanged.

For each material and observation representation run the four paired conditions:

1. L2, optimize theta and log geometric mean; ratio fixed to oracle truth.
2. Both lights, the same constrained optimization.
3. L2, optimize theta, log mean and log ratio jointly.
4. Both lights, the same free optimization.

There are 16 panels: two materials × two observation representations × two ratio
conditions × two lighting conditions. Each panel has three starts at every point:
4,032 independent point/start fits, with no neural field or spatial regularizer.
Do not treat these as 4,032 independent population samples. Primary summaries
compare the same 84 physical points, paired across conditions, at start 0. Retain
all three starts; a secondary per-point multistart result selects only the smallest
final **fitting** objective, with first-index exact ties. Never use truth or held-out
scores to select starts, points, models, stopping or acceptance.

Truth-fixed ratio is an oracle intervention. The inverse and target references use
the same mathematical BRDF and finite-grid quadrature: this is a same-kernel inverse
diagnostic, not independent validation of practical neural material recovery or
the physical correctness of the BRDF. There is no neural-network training here.

## Input identities and geometric train/held-out split

Admit the E.14 native report SHA256
`215eb11f9dd0a2fff9074cf6bab5eca6be832402a1860e3dcf27138f8458bab5`,
both numeric grid archives and comparison hashes, plus the existing E.11D/E/F/G
chain and all consumed artifacts. All NPZ loads use allow_pickle=False. Original
G/A snapshots and the corrected image identity remain unchanged; no camera or
checkpoint deserialization is needed.

Keep camera order [58,44,9,0] and all 220 accepted point/view correspondences.
For each point with at least two valid views, hold out its last valid view in
that fixed order; otherwise retain its sole view for fitting and record that
held-out-view evidence is missing. No image/material response selects this split.
The result is **137 fitting and 83 held-out correspondences**, shared exactly by
all conditions. Held cameras: 9 at 23 points, 0 at 60 points. The single-view-only
point (source pixel 6208) stays in all-point results with an explicit missing
held-out-view flag. Freeze both Boolean masks.

Training-view-count strata are 1:36 points, 2:43, 3:5. Report all 84 and these
geometry-defined strata, plus the same 83 held-out-eligible points. The E.14
full-acquisition conditioning is context, not the exact E.15 training Jacobian
after withholding a view. Do not discard the single-training-view stratum.

For L2-only fits the rotated light is unseen. Report L2 held-view reconstruction,
rotated-light held-view reconstruction, and rotated-light transfer across all
220 correspondences separately. For two-light fits only withheld **views** are
independent: rotated-light fitting-view observations are training data, not
held-out relighting. No third light or independent new rendering is introduced.
A two-light unseen-illumination claim is therefore unavailable in this study.

## Physical coordinates and starts

Use theta in radians modulo pi, u=log(sqrt(mx*my)), v=log(mx/my), with canonical
major mx≥minor my and v≥0. Widths are exp(u+v/2), exp(u−v/2), each in [.04,1].
Theta proposals are wrapped modulo pi, an exact physical equivalence. Invalid
width/ratio proposals are rejected, never clipped, axis-swapped or repaired.
Preserve literal reference axis=cos(theta)T+sin(theta)B and bitangent=cross(n,axis).
Do not assume finite-precision frames are perfectly orthonormal or replace that
cross product by an idealized rotated B.

The three deterministic global starts (theta,u,v) are:

- (−pi/3, log(.22), log(1.5))
- (0, log(.30), log(2))
- (+pi/3, log(.40), log(3))

These values do not depend on a point's truth or observed responses. Every
material/light/representation shares theta/u starts; free-ratio starts are also
identical across both materials. In constrained fits replace only v with the
explicit oracle ratio, held unchanged throughout. Consequently fixed/free arms
have matched unknown-coordinate starts, not identical physical widths; disclose
this intended oracle intervention. Validate all starts before scientific execution.
No random initialization or simulated noise is generated.

## Exact forward model and sufficient numerical checks

Reuse saved 512×512 incident nodes and packed visibility for fitting, without
subsampling, interpolated BRDF tables, new geometry queries or reference targets.
Precompute material-independent coefficients from known geometry/color/metal and
the two known lights. This is an algebraic factorization of the unchanged finite
sum, not a surrogate. For h=normalize(wi+wo), store

`x=h·T/(h·n), y=h·B/(h·n), bx=h·cross(n,T)/(h·n), by=h·cross(n,B)/(h·n)`.

Let p=cos(theta)x+sin(theta)y, q=cos(theta)bx+sin(theta)by,
A=exp(−2u−v)p², B=exp(−2u+v)q². The material-dependent factor is
`Dfactor=exp(−A−B−2u)`; the precomputed RGB coefficient is the original
`F*Li*cosine*visibility*dOmega/(4*pi*(h·wo)*(h·n)^4)`.
Diffuse allocation is known and independently integrated once. The exact analytic
log-factor derivatives are:

- theta: `−2*(exp(−2u−v)*p*p_theta + exp(−2u+v)*q*q_theta)`
- u: `2*(A+B)−2`
- v: `A−B`

No shader, native neural model, BVH extension or optimizer state is loaded. This
new physical solver does not change any original shader or network. The retained
corrected BVH binary is hashed only as part of image provenance.

Synthetic CPU fixtures compare factorized radiance with the unmodified independent
BRDF, including a slightly imperfect frame, and derivatives with both central
differences and autograd of that BRDF. Before any real optimization require:

1. Full 512-grid original/stronger truth forward parity with saved E.14 targets at
   all 220 correspondences/lights (absolute2e−12, relative2e−10).
2. Analytic derivatives versus central differences with h=1e−5 at global start0
   for all 84 points/all 220 observations, all three coordinates and both lights:
   absolute2e−8 + relative5e−5. This does not use truth to initialize a fit.
3. Finite, valid starts and positive incident/outgoing/half-vector hemispheres.

Failure stops the run with partial evidence. No modified gate, replacement grid,
alternate solver, reference regeneration or automatic retry is allowed.

## Observation objectives and one optimizer policy

Continuous fits minimize mean squared linear-RGB residual to preserved E.14
512-grid truth. Deterministic 8-bit fits minimize squared distance in **linear**
radiance to the exact inverse-sRGB code intervals. Codes0/255 have infinite
external bounds; other bounds are inverse-sRGB half-code values. The residual
and derivative are zero inside the interval. There is no center-value regression,
fictitious Gaussian likelihood, added target noise or adaptive sigma. Zero
interval loss is compatible with physically distinct materials. Closed interval
boundaries are the censored-loss convention; evaluate actual round-to-nearest,
ties-to-even codes separately, including numerical boundary ties.

Each point/start uses float64 damped Gauss–Newton/LM with at most **20 outer
iterations, three trials per outer**, unchanged across all panels. Normalize by
the number of fitting scalar channels: one versus two lights does not silently
double the objective scale. No priors or regularization are added. For H=J'J/N,
g=J'r/N solve `(H+lambda*diag(max(diag(H),1e−12))) delta=−g` over the unknown
coordinates, with Euclidean step norm at most .35 in the declared units.

Lambda starts .001, bounded [1e−10,1e10]. Accept only strictly reduced fitting loss
and actual/predicted reduction ratio>.1. Accepted ratio>.75 divides damping by3;
ratio<.25 doubles it; other accepted damping is unchanged. Rejection multiplies
it by10, including out-of-domain proposals. Three failed trials consume the
outer iteration; the next outer may use its increased damping. No hidden restart
or tuning. Stop a point/start only on fitting-gradient infinity norm≤1e−10,
accepted step norm≤1e−8, or the outer limit. Record these as algorithmic stopping
conditions, not proof of physical convergence. Preserve rejected/failed trials,
all states, gradient norms and normal-matrix eigenvalues.

The maximum accepted LM steps is80,640 across all point/starts; maximum trial
attempts241,920. Actual work is reported. Both lights may be computed in the same
forward contraction, but L2-only fitting/acceptance/stopping uses L2 alone. No
held-out output enters those calculations. Equal outer/trial ceilings do not
imply equal rendering cost or unknown dimension; report both limitations.

## Endpoints, ambiguity and scientific interpretation

Save every endpoint and trajectory, all initializations, optimizer damping and
termination flags. Evaluate each endpoint once at both512² and256² over all
original220 correspondences and both lights. Retain original .003 absolute/.002
relative two-grid checks; a failure makes rendering-based interpretation
numerically unresolved, not permission to change quadrature or drop the endpoint.

Freeze the following reporting definitions:

- World/local Q relative Frobenius error, both widths' absolute errors, absolute
  u/v errors, orientation error modulo pi in the canonical major-axis chart.
- Original orientation eligibility: truth major−minor≥.1, with truth orientation
  defined; predicted near-isotropy (gap≤1e−6) is explicit, not a favorable exclusion.
- Existing Q≤.05 and eligible orientation≤5-degree descriptive pass counts; keep
  Q over all points. Fixed ratio is known, not recovered. No new u/v pass threshold.
- Fitting and independent held-view linear RGB RMSE/relative L2, actual8-bit code
  equality/errors, and the clearly labeled L2-only unseen-light transfer above.
- Primary start0, every other start, and secondary fitting-selected multistart
  endpoints. No held-out selection. Report all geometry-defined strata.
- All pairwise multistart endpoint prediction distances and physical differences;
  describe a pair as near-photometric only at prediction RMSE≤1e−4, reporting its
  target residuals as well. Distinguish near-matching predictions that both miss
  the data from actual alternative fitting solutions. Identical fitting codes
  are reported exactly; a small RGB loss does not establish uniqueness.

Compare paired errors and pass counts without a population-significance claim.
Do not use the frozen per-view .003 response gate to replace stacked inversion.
Low-loss wrong parameters/varied multistart solutions support ambiguity in this
bounded acquisition; high fitting residual versus an admitted truth solution
supports optimization/numerical failure. Neither is a global proof. Use saved
E.14 sensitivities as contextual evidence while acknowledging the changed fitting
view mask. Color/metallic and neural/spatial coupling are excluded.

## Work, resources and evidence

Exact new ceilings, counted before calls, include rejected queried proposals:

- Fit:137,908,715,520 BRDF point-directions (16 panels×3 starts×137 observations
  ×512²×80 maximum forward/Jacobian evaluations).
- All endpoints on both grids:3,460,300,800 point-directions.
- Truth parity and derivative admission:519,045,120 point-directions.
- Total new BRDF ceiling: **141,888,061,440** point-directions. Analytic derivative
  contractions are recorded with their forward; no hidden reverse passes.
- Material-independent coefficient preparation:72,089,600 point-directions;
  environment:55,050,240 light-point-directions, reused across all fits.
- Zero new geometry rays, reference datasets, neural calls or Adam updates.

Use the existing image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A100GPU0,4 CPU threads,64GiB host RAM/no swap,pids128,25% CUDA allocator,
1GiB outputs. Coefficient storage is about5.37GiB on GPU, plus streamed temporary
buffers; avoid retaining autograd graphs. One foreground **3600s watchdog/30s
kill grace**, inner3480s/20s. Runtime estimated10–45 minutes with algebraic reuse,
not yet measured; the hard cap is not extended if this estimate is wrong. This
ceiling is substantial but bounded and uses every point, three starts and full
admitted quadrature. No downsampled fitting grid is silently substituted to save
time. A watchdog stop remains incomplete evidence with no retry authorization.

Read-only source/all original inputs/root, non-root, no network/build/install.
All writes go into the fresh frozen `/outputs/recovery` run. Reuse the established
foreground launcher, source/hash admission and owned-resource cleanup. Partial
progress and unexecuted panels remain explicit; no resume is implied. Native
cleanup success is acknowledged only after normal cleanup-context exit.

After the one authorized execution, file-only analysis verifies receipts/hashes
and every endpoint, produces paired tables/figures, and commits/pushes small
evidence. Large arrays/trajectories stay outside Git. Decide what actually
improves: free three-parameter recovery, oracle-constrained recovery, neither,
or numerical/optimization failure. Recommend one bounded neural-method experiment
supported by these outcomes; do not execute it, a v4 dataset or E.13 rerun here.
