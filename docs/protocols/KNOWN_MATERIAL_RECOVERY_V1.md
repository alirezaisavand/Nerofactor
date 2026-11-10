# Known-material recovery protocol v1 — design for review

Protocol: `known-material-recovery-v1`. Status: **proposed, not executed or
execution-authorized**. Prepared in milestone3C-0 from
`1067d3d5370eaaeabac7eedd650d4d913f87f820` on
`feat/material-recovery-preparation-v1`.

This document specifies future experiments, not options in a training launcher.
No observation generator, optimizer loop, synthetic dataset or experiment command
is implemented here. Approval of this design must precede implementation and
execution of the first small inverse experiment. No historical geometry is used.

## Questions and separate decisions

1. Forward correctness: do sample estimates/derivatives agree with independently
   computed integrals/derivatives in their valid domains?
2. Inverse identifiability: do the chosen observations constrain the physical
   material parameters, up to gauge symmetries?
3. Optimization: can a declared bounded optimizer reach the identifiable solution
   from fixed initializations, without tuning after seeing held-out results?
4. Generalization: do recovered parameters predict unseen views and known lights?

These are distinct tests. Passing shader fixtures does not establish recovery,
optimizer convergence, physical energy conservation or native reconstruction.
The fitted law remains A's Beckmann/Kurt a=0 model with its original Fresnel and
material definitions. It is not claimed to represent every real material.

## Known-input scene configurations

All observations are local outgoing linear RGB radiances at known surface
points, under a distant known environment. Directions point away from the
surface. Length unit is an arbitrary declared unit; radiance is relative to
unit white incident illumination. No camera exposure/white-balance fitting,
visibility inference, interreflection, mesh, SDF, learned lighting or ray tracing.
Visibility is identically one on the local incoming hemisphere. Curved-patch
observations are local surface evaluations, not rasterized self-occluding images.

| Scene | Known points/normals | Ground-truth parameters | Purpose |
| --- | --- | --- | --- |
| Flat anisotropic / first | p=(0,0,0), n=(0,0,1) | mx=.20, my=.55, theta=.31; RGB k=(.30,.45,.60), m=.25 | Smallest seven-scalar recovery problem |
| Curved anisotropic / later | Unit-sphere patch points p=n: (0,0,1), (+.25,0,sqrt(.9375)), (-.25,0,sqrt(.9375)), (0,+.25,sqrt(.9375)) | mx=.30, my=.70, theta=.47; k=(.25,.50,.65), m=.40, constant relative to the known local T/B frame | Normal variation with exact known tangent coordinates |
| Isotropic control / later | Same flat geometry | mx=my=.40; theta arbitrary; k=(.30,.45,.60), m=.25 | Orientation must be treated as unidentifiable |

At each point T0=normalize(n cross world+X), B0=n cross T0, with the source's
world+Y fallback if needed. The curved patch deliberately avoids that switch.
The physical frame is T=cos(theta)T0+sin(theta)B0,
B=-sin(theta)T0+cos(theta)B0. The benchmark knows this convention exactly;
it does not solve global tangent continuity or infer a UV field.

Parameters are physical slope widths, not already squared roughness. Design
ranges are mx,my in [.04,1], RGB k and metallic in [0,1], theta modulo pi.
Initial ordinary recovery stays in the interior. Later separately labeled
stress cases are: near-isotropy (.40,.41), narrow widths (.04,.10), broad
widths (.90,1), near-metallic m=.99, near-dielectric m=.01, dark achromatic
k=(.02,.02,.02), and outgoing80 degrees. These are proposed one-factor stress
cases, not permission for a sweep now. Narrow/grazing reference convergence and
low-signal identifiability must be checked rather than presumed.

## Fixed view and illumination identities

World direction(polar,azimuth)=(sin(polar)cos(azimuth),
sin(polar)sin(azimuth),cos(polar)); angles in degrees. Observers are distant;
these directions are known and shared across points. The proposed curved
normals keep all listed directions in the outgoing hemisphere. Every future
record must save actual world directions, point/normal/TBN and observation ID.

| IDs | Polar angles | Azimuth angles | Count |
| --- | --- | --- | --- |
| v_train_0..5 | 20,50 | 0,120,240, Cartesian product in that order | 6 |
| v_hold_0..5 | 35,65 | 60,180,300, Cartesian product in that order | 6 |

Training light l0 is constant RGB white, L=1. Each other light is RGB white
with L(w)=.1+.9*max(dot(w,axis),0)^p. Unit axes use the same world-angle formula.
No light amplitude, color or orientation is optimized.

| Light IDs | Axis polar | Axis azimuths | Exponent | Role |
| --- | --- | --- | --- | --- |
| l1,l2,l3 | 40 | 0,120,240 | 8 | Training and known-light held-out views |
| l4,l5,l6 | 55 | 60,180,300 | 16 | Unseen relighting only |

Exact split membership, with no random shuffle:

- Fit: six v_train crossed with l0..l3 = **24 observations**.
- Held-out views: six v_hold crossed with l0..l3 = **24 observations**.
- Relighting: all twelve views crossed with l4..l6 = **36 observations**.

Total per point:84 RGB observations. None of the 60 held-out observations is
used for loss selection, early stopping, hyperparameter tuning or picking a
favorable initialization. Do not silently move a view/light between sets.

## Independent ground truth and avoiding the inverse crime

Generate proposed target radiances with independent float64 Cartesian Beckmann
NDF and a=0 Kurt kernel, in the declared known frame. Integrate over incoming
hemisphere directions with Gauss-Legendre cosine nodes and uniform midpoint
azimuth nodes. Compute Lambertian d/pi separately; add the specular integral.
Use analytic diffuse moments when available as additional checks. This reference
must not call the shader's sampler, compute_radiance or its estimator adapter.
The accepted NumPy Cartesian reference provides the flat-surface algebra;
a future generator must independently implement/verify known-frame world mapping
for curved points before any curved recovery is approved.

Use exactly **256x512 and 512x1024** reference grids for each required RGB
observation. Predeclared convergence is per-channel
abs(fine-coarse) <= 2e-6 + 5e-5*abs(fine), requiring both components and total.
This is an operational two-grid check, not a certified integration error bound.
Preserve both values and discrepancy. Failed reference convergence blocks that
observation/case and its recovery claim; no adaptive grid increase, tolerance
tuning or shader-generated replacement target is allowed under v1.
A different reference strategy/resolution requires reviewed v2.

Use the converged fine reference as the provisional target, with the two-grid
difference reported as uncertainty. The fitted shader instead uses finite
512/1024 patterns and explicit paired uniforms. This deliberately avoids
synthesizing and evaluating observations solely with the fitted approximation.
Fixed-pattern errors can impose a radiance/parameter error floor; do not assign
all resulting parameter differences to optimizer failure.

First establish a noise-free result. Only a later separately approved phase may
add independent additive Gaussian linear-radiance noise sigma=.005, private
seed2718, preserving both clean/noisy targets and without clipping noisy values.
Noisy values can be negative measurements; clipping would introduce a different
observation model. Do not introduce noise during the first experiment.

## Gauge, identifiability and degeneracies

Physical equivalences are theta modulo pi, and
(mx,my,theta) equivalent to (my,mx,theta+pi/2). Raw head/angle differences are
not appropriate success criteria. Define in declared T0/B0 coordinates:

Q=R(theta) diag(mx^2,my^2) R(theta)^T.

The gap g=abs(mx^2-my^2)/(mx^2+my^2) determines whether orientation can be
scored; predeclare threshold **g>=.1**. Major-axis orientation is theta when
mx>=my, otherwise theta+pi/2. Its error is
abs(.5*atan2(sin(2*delta),cos(2*delta))). Both truth and prediction must exceed
the gap threshold. Otherwise report null orientation and its reason; Q/axis
errors and missing-orientation counts remain mandatory. A predicted isotropic
collapse against anisotropic truth cannot be counted as successful orientation
recovery. At exact isotropy, theta is intrinsically unidentifiable.
Finite sample patterns may still produce theta-dependent numerical residuals;
fitting that artifact is not physical orientation recovery.

Before future fitting, inspect the training-observation Jacobian of the
**independent** forward integral with respect to seven physical scalars:
mx,my,theta,k_R,k_G,k_B,m. Use central steps1e-4 and5e-5 times parameter scales
(.96,.96,pi,1,1,1,1), with stable finite differences. Normalize columns by these
scales for the conditioning report. At both steps require local singular-value
ratio s_min/s_max>=1e-4 for all seven parameters; report the whole spectrum,
null directions and cross-step discrepancy. This threshold is a proposed
local-identifiability gate, not proof of global uniqueness. The reference grid
must itself converge before using its derivatives. At isotropy drop the theta
column and explicitly test six-dimensional identifiability instead.

If the gate fails, stop/report the case without inventing a stronger light set
under this version. A revised design needs review. Limited azimuth coverage,
constant illumination, normal-only views, isotropy and diffuse/specular tradeoffs
can conceal parameters. Dark achromatic albedo/metallic can be ambiguous even
when radiance agrees. Check alternative physically inequivalent solutions and
parameter stability before declaring global recovery. Anisotropy axis swapping
and scalar-encoding periodicity are gauges, not separate solutions.

## Proposed first small inverse experiment — not executed

Scope: **only the flat anisotropic case**, noise-free, A2 with atan2 decoder,
known lighting, one material constant across observations. Do not optimize a
network, geometry, normal, light, camera, visibility or historical artifact.
The new shader is ready for local autograd tests; a bounded direct-parameter
fitting runner and reference generator still require separate implementation
approval. Passing this milestone is not that approval.

Proposed parameterization: mx,my=.04+.96*sigmoid(a_x,a_y), RGB k=sigmoid(c),
m=sigmoid(b), plus scalar theta encoded as (cos(2theta),sin(2theta)). This is
seven scalar degrees of freedom, avoiding the zero-norm encoding. It changes
no original neural head. First objective: mean squared **linear RGB** residual
on the 24 fit observations, with no spatial, energy or L_spec priors. This is
a proposed controlled loss, not a change to original A training.

Proposed bounded optimizer: Adam lr=.01, betas(.9,.999), eps1e-8, weight_decay0,
no scheduler/clipping, exactly up to250 updates per initialization; five private
initialization seeds101..105. Draw physical widths uniformly [.15,.75], RGB k
[.20,.80], m[.10,.90], theta[-pi/2,pi/2), then invert sigmoid for corresponding
unconstrained coordinates. Preserve actual vectors. These settings are new
analytical-benchmark proposals and must never be described as original A's
training optimizer/schedule. No optimizer was constructed in 3C-0.

Fix four private NumPy RandomState tapes seeds0..3 in original u/v/azimuth
order; shader input values are rounded float32. Use the same tapes for every
initialization, arm and update, without redrawing or consuming global RNG.
Average the four forward estimates before residual calculation. Store tape
hashes. This objective is deterministic; four tapes do not measure the bias of
its fixed radial quadrature. Evaluation uses separately declared seeds1000..1003
so it does not merely reuse fit noise; never choose a favorable seed afterward.

Evaluate the final declared update, not a held-out-selected best iterate.
Log fit loss, gradient/parameter norms, finite state, time and budgets every10
updates; stop immediately on nonfinite output, invalid encoding/input, reference
failure or budget exhaustion. Preserve failures and final available states.
No retry, learning-rate search or extra update allowance in v1. A flat loss
curve alone is not evidence of optimizer convergence or identifiability.

Proposed budgets for this first experiment:

| Phase | Maximum work |
| --- | --- |
| Target integration,84 observations | 55,050,240 direction/kernel evaluations across the two fixed grids |
| Local Jacobian,24 fit observations, two FD steps/seven columns | 440,401,920 direction/kernel evaluations upper bound; cache only mathematically identical terms |
| Fitting | 5 initializations x250 updates =1,250 scalar-parameter optimizer calls |
| Shader points during fitting | 24 observations x4 tapes x1,250 =120,000 point evaluations |
| Fitting sample directions | 122,880,000 specular +61,440,000 diffuse |
| Final all-split evaluation | 84 observations x4 new tapes x5 starts =1,680 point evaluations,2,580,480 directions |
| Entire first experiment | Single CPU process, one Torch thread,1,800-second wall cap; no automatic extension |

The Jacobian upper bound counts 24x7x2 signs x2 step sizes x655,360 nodes.
It may dominate runtime; if it cannot finish inside the cap, report incomplete
without launching optimization or weakening the gate. A separate proposed
cheaper derivative reference would require review. No such computation ran here.
Future experiment manifests must also record literal commands, Git/source hashes,
configuration/tape/target hashes, runtime versions, all initialized/final physical
parameters, stop reason and actual counters. Use a fresh explicit output path,
refuse collisions, preserve all initializations and leave accepted evidence alone.

## Metrics and proposed reporting criteria

| Quantity | Required metric |
| --- | --- |
| Roughness recovery | Sorted major/minor width errors and relative errors, after gauge handling |
| Physical anisotropy | Absolute/relative Frobenius Q error, with declared T0/B0 coordinates |
| Orientation | Major-axis angular error modulo pi only for identifiable cases; missing count/reason otherwise |
| Albedo | RGB absolute error, RMSE and relative norm error, denominator floor1e-4 |
| Metallic | Absolute error, and boundary/near-boundary indication |
| Fit/held-out radiance | Linear RGB RMSE/MAE, relative L2 error floor1e-4, max absolute error, component errors |
| Relighting | Same radiance metrics on unseen l4..l6, separately for seen and unseen views |
| Stability | All five physical parameter/Q estimates, pairwise Q distances, median/range; no best-start-only table |
| Resources | CPU/wall time, target/reference nodes, shader calls and sample counts, updates, failures |

For review, propose a noise-free recovery target: relative Q error<=.05,
identifiable orientation error<=5 degrees, max RGB albedo error<=.02,
metallic error<=.03, held-out relative radiance L2 error<=.01. These are proposed
scientific criteria, not verified attainable tolerances or code-enforced options.
Report continuous metrics and forward/reference uncertainty even when a threshold
is missed. Criterion approval must precede generation/fitting; do not tune them
after results. No image PSNR claim is needed for a few local radiance samples;
never report these as thesis NVS or real reconstruction scores.

Forward-error budgets must be assessed separately from inverse parameter error.
Small radiance error with a singular Jacobian is not parameter recovery; high
parameter error with low residual can expose ambiguity or estimator bias rather
than an implementation fault. Failure to converge within250 updates is a bounded
optimization result, not permission to add steps silently.

## Later comparisons and unresolved decisions

After reviewing the first experiment, propose A0/A1/A2 with decoder held fixed,
then original/atan2 decoding with estimator held fixed, all sharing observations,
starts and tapes. Do not silently combine decoder and estimator changes into a
single attributed gain. Curved/isotropic scenes, stress cases, noise, alternate
quadrature, regularizers and GPU acceleration each remain later approved work.

Unresolved: approval of the first experiment and numerical criteria; adequacy
of the local/global identifiability design and runtime budget; branch-cut/mask
finite-sampling gradient effects; independent curved-frame reference mapping;
policy for learned near-zero encodings; corrected energy-prior semantics; and
the purpose/scaling/detach of L_spec. Energy-conserving model redesign and
historical geometry research admission are outside this protocol.
