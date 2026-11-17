# Sphere albedo and anisotropy-smoothness audit

Decision: **FIX AND PROCEED with a bounded comparison, not a100000-step restart.**
The current evidence demonstrates material–illumination compensation and a weak,
chart-dependent spatial prior. It does not identify a unique cause of albedo
darkening or establish physical material recovery.

## Inspected evidence

Run `/home/isavand/paper/evidence/sphere-core-v2-ec8cee6b6d234133b0c7e1168fff02e1`
is marked incomplete after41427 Adam returns; its last completed checkpoint is
40000. Twenty immutable step checkpoints and20 validation-camera36 archives
were hash-verified. No checkpoint or prior artifact was modified. An interrupted
cleanup is visible in the log; no launcher exit-code receipt exists. Full
cleanup is not verified. The new launcher refuses to start while that exact
container is still running; it never stops unrelated containers.

Checkpoint metadata/sidecars/hashes were inspected. A CPU
`torch.load(weights_only=True,map_location='cpu')` of the verified40000 checkpoint
was blocked by the saved NumPy scalar metadata. No unsafe fallback or allowlist
was used, no optimizer was restored, and no historical neural graph was replayed.
Saved arrays already provide direct material/component evidence without loading
the network. New component-gradient observations require the separately approved
bounded native execution; they cannot be reconstructed from old aggregate logs.

Offline analysis is in `/home/isavand/paper/evidence/sphere-regularization-audit-v1`;
small tables/figures and an external-artifact manifest are checked in under
`docs/results/sphere_regularization_v1`. All numerical NPZ reads disable pickle.

| Completed updates | Mean red kd | Mean metallic | Mean red kd(1−m) | Effective red diffuse incident light | Mean red diffuse RGB, linear | Mean red specular RGB, linear |
|---:|---:|---:|---:|---:|---:|---:|
|2000|.99749|.82071|.17881|.45293|.08418|.14436|
|10000|.07362|.84459|.01161|5.23603|.06646|.15276|
|20000|.07250|.84679|.01129|5.45427|.06712|.15837|
|40000|.07059|.84828|.01089|5.63878|.06694|.15902|

The incident-light diagnostic is `linear_diffuse / [kd(1−m)]` where allocation
exceeds1e-8, per pixel/channel, before taking its mean. For this corrected
cosine-sampled diffuse estimator it measures effective sampled incoming light,
including visibility/learned indirect light. It is not the full unoccluded HDR
environment. Display diffuse/specular-light maps are clipped and cannot recover
HDR magnitude. Red-channel division is well supported; nearly zero green/blue
allocations make those ratios unreliable, and their sample counts must be retained.

The red allocation falls about16.4× while effective illumination rises12.5×;
diffuse RGB changes much less and specular RGB remains dominant. This is observed
compensation, consistent with the unknown-light inverse-rendering ambiguity.
It is not proof of which loss caused the change. Raw numeric kd changes show
that the darkening is not simply a PNG gamma/display artifact.

The diffuse-light chromaticity prior receives `to_sRGB(mean light).clamp(0,1)`,
not unclipped HDR illumination. At40000,93.1% of visible pixels have all three
diffuse-light display channels clipped to1. Beyond that clamp this penalty has
zero direct lighting gradient; even below it, a grey illumination scale has
zero chromaticity penalty. It cannot fix the global radiometric scale gauge.
This is an inherited modeling limitation, not a silently repaired loss.
Where red is below the clamp and green/blue are already clipped, this prior has
a negative derivative with respect to red incident illumination: increasing red
light makes the clipped light more white. RGB can counterbalance that change by
lowering red material response. This is a verified local mechanism consistent
with the observations, not a measured historical neural-gradient attribution.
The kd spatial L1 term also shrinks under a global kd scaling. Neither spatial
smoothness nor a one-sided upper energy bound fixes the illumination/albedo gauge.

## Effective priors and exact code mapping

The active shader mode is **legacy_v1**, not unconditional_v1 or disabled_v1.
Shader energy/specular coefficients are both.05; the historical sphere YAML's
top-level.01 declarations were ignored. The renderer diffuse-light coefficient
is.01, spatial coefficient.05, Gaussian displacement sigma.02 and random azimuth
enabled. They remain unchanged in the proposed comparison. Top-level max_n_exp /
max_alpha_exp and shader shading_feature_num / num_viewdir_freqs remain diagnosed
as unused; no setting is silently forwarded into a new scope.

PBR-NeRF source: researcher PDF `/home/isavand/paper/data/PBR-NeRF.pdf`, SHA-256
`b39249084436b76f7f34791950ec7fa78545387676ead49822353a9995b0594d`, inspected
at PDF pages4–5 and15. Official
[loss.py](https://github.com/s3anwu/pbrnerf/blob/9d50aab977bcb50ad03db728f2d4957cba13b95e/code/model/loss.py)
is pinned to `9d50aab977bcb50ad03db728f2d4957cba13b95e`, content SHA-256
`2aa6c45477253758cd2f16261aaf100fd9eb7e1774c9e12e12b79813f242ea50`.
The code is reviewed as a reference; it is not copied into the shader.

| Quantity | PBR-NeRF | Current corrected A with legacy priors |
|---|---|---|
|Diffuse BRDF vs allocation|Eq3: b(1−m)/π|Variable f_d is allocation d=kd(1−m); cosine-sampled diffuse integrates the π factor analytically|
|BRDF family|Disney microfacet with spherical-Gaussian NDF and geometry term|Additive Kurt a=0,ks=1 with anisotropic Beckmann NDF; no substituted Disney G term|
|Energy, Eq7|Uniform/Fibonacci incident hemisphere;2π/N times sum of f_r cos_i|Half-vector proposal p_h=D cos_h, reflection Jacobian4|wo·h|; RGB uses M total proposals, but legacy prior divides specular sum by valid count|
|Specular, Eq8|softmax(detach D/T) along incident samples, then diffuse BRDF; official reduction divides N and3|softmax(detach D) over sampled half vectors, invalid reflections masked, then divide valid count and3; allocation d rather than d/π|
|Smoothness, supplementary Eq23|Spatial norms of roughness/metallic gradients, weighted by exp(−image-gradient norm)|Off-surface tangent-plane perturbation; scalar L1 changes plus raw double-angle orientation distance; no surface projection, bilateral edge gate or normal transport|

For corrected A, define `w_j=F_j max(n·wi_j,0)/cos_h_j`, with invalid samples
contributing zero. The reflected-direction density is
`p_i=D(h)cos_h/[4|wo·h|]`. The implied specular kernel is
`F D(h)/[4|wo·h|]`, so `mean_M(w_j)` estimates its directional-hemispherical
reflectance. Rejected samples stay in M. The legacy proxy instead uses
`sum(w_j)/max(valid_count,1)`, estimating a conditional integral. This can
overestimate reflectance and penalize otherwise subunit materials. Channel
aggregation is a sum followed by point mean, matching the current implementation;
there is no extra RGB averaging for the energy term.

The independently implemented all-proposal proxy is a **new analytical candidate**
in `energy_proxy_candidate`; it is not enabled in the smoothness experiment.
Existing unconditional_v1 already changes the proxy but also changes the
specular-prior definition and gating. Switching the whole mode is not a clean
one-term repair, and neither mode is silently substituted.

The legacy specular prior uses detached NDF softmax weights. Holding those weights,
validity and count fixed, `L_spec,c = d_c * a/(3 N_valid)` for retained softmax
mass a. Thus its direct physical derivatives decrease kd and/or increase metallic;
there is no direct derivative through roughness/orientation/NDF. Shared neural
features can still transmit indirect changes to other heads. No direct lighting
gradient exists for that prior. The actual historical network-gradient magnitudes
and opposition to RGB are **unavailable**, not inferred from scalar losses.

For direction-independent diffuse BRDF with all uniform incident samples valid,
the normalized softmax in paper Eq8 cancels in the final directional sum,
leaving an allocation penalty scaled by1/N (and the official channel mean).
The sphere's half-vector proposal/mask/valid-count dependence changes that
functional and its scale; it is not an importance-correct implementation of the
same incident-hemisphere expression. Do not copy PBR-NeRF's weights: distribution,
BRDF, diffuse π convention, valid-count denominator, channel aggregation, RGB
loss and sample counts all differ.

## What the saved losses establish

Energy penalty is zero at every saved validation view; none of its saved pixels
exceeds the current proxy bound. Weighted validation specular-prior diagnostics
fall from2.29e-6 at2000 to1.39e-7 at40000. These are reconstructed validation
diagnostics, not unsaved training-minibatch sublosses. Actual aggregate weighted
training losses are plotted separately; the original RGB Charbonnier schedule,
floor sqrt(.001), spatial prior and diffuse-light chromaticity penalty are retained.
Magnitude alone cannot establish neural-gradient dominance or rule out effects
from unseen training views. Turning off energy to fix this darkening is not
supported by these saved validation samples.

Independent CPU quadrature used six raster-spaced materials at each of four
saved steps, two predeclared Gauss-Legendre-mu/midpoint-azimuth grids128×256 and
256×512. The bound is `.001+.002*abs(fine)`. **17/24 pass;7/24 remain unresolved**,
including near-grazing cases. Do not revise thresholds or claim absolute accuracy
from two-grid agreement. On passing checks, saved red energy proxy minus true
reflectance ranges+.00499…+.16559, consistent with conditional normalization and
finite1024-sample variation. The expected conditional limit and independent
acceptance mass are reported per probe. No new geometry rays, references for
training, GPU work, native forward/backward or model construction occurred.

## Anisotropy smoothness correction

The historical orientation multiplier is
`1e-4 + 9e-7*(cos(pi*step/100000)+1)/2`, ranging only.0001009→.0001, before
spatial coefficient.05. The extra.001 factor makes its dynamic range much smaller
than the apparent1e-3→1e-4 intent. Raw double-angle distance handles180° but not
90° width swapping or independent tangent-chart changes; unconstrained displaced
points need not lie on the mesh. Near isotropy, orientation is unidentifiable.

New selectable **surface_world_q_v1** uses
`Q=mx² ttᵀ+my² bbᵀ` in world coordinates, with t,b from the frozen corrected
chart/atan2 convention. Compare `Q_i−R_ji Q_j R_jiᵀ`, where R is the minimal
normal-aligning rotation. Apply a fixed.02 scene-unit reference length divided by
the mesh-surface path length, square it for Q's squared Frobenius difference,
and use coefficient.05. Scalars kd/metallic use the same surface pairs and
distance-normalized L1 term. This is a new spatial objective, not a claim that
the original losses are mathematically equivalent.

Pairs share actual manifold mesh edges; centroids lie on their source triangles.
Distance follows both centroids through the common edge midpoint. Reject paths
outside[1e-6,.02] and dihedrals>15°. No Euclidean nearest-surface shortcuts, mesh
repair, normal flips or averaging across disconnected sheets. CPU inspection
found4 zero-area faces; only their undefined loss neighborhoods are excluded.
The original225888-vertex/451776-face rendering mesh is unchanged.677655 admissible
pairs are available before selecting512 deterministic, raster-independent pairs.

Boundary weights are fixed visible **training-image** RGB contrasts from the first
original training camera, after bounded point-to-camera hit/normal checks. They
are detached and never inferred from trainable predictions or material truth.
They can mistake illumination edges for material boundaries, and unseen regions
are not covered by this one-camera gate. Zero-weight/invalid pairs are reported
without replacement. This is a bounded first test, not full-surface admission.

The candidate respects180° director symmetry,90° width swapping, reference-frame
changes and gradients through both widths/orientation. Q loses orientation
dependence continuously at isotropy. It does **not** provide a globally smooth
nonzero tangent director on a sphere: curvature, holonomy and genuine isotropic
regions/defects remain. A lower Q smoothness loss can also result from isotropizing
or shrinking widths; report eigenvalue/gap distributions alongside seam metrics.

New checkpoint visualizations preserve original RGB/style and add raw chart
orientation, world unoriented-axis projectors, six fixed-scale Q images and
numeric Q/projector/gap arrays. Diagonal Q scale0…1; off-diagonals−.5….5 shifted
by+.5. Gap<.1 is grey in the world-orientation DISPLAY only; it is not a training
exclusion. A sign-changing axis image is not by itself a BRDF seam. Saved screen
neighbor diagnostics do find physical Q jumps, but do not establish their cause
or label them all as chart artifacts without source-face correspondences.

## Recommendation

Run only the [matched4000-update surface comparison](protocols/SPHERE_SURFACE_COMPARISON_V1.md)
after explicit approval, as two forks of the same state4000 checkpoint ending
at8000. The saved curve places the principal darkening between4000 and8000;
retraining0…4000 would miss that interval and duplicate completed work. Restore
exact weights/Adam, but declare the common seed6033 RNG/sampler reset because
their historical states were not saved. This is not exact historical continuation.
Keep energy/specular modes/weights fixed; record exact
component and group gradients at three states. Compare world-Q against a chart
control on IDENTICAL surface pairs, avoiding neighborhood/sampler confounding.
The interrupted legacy run remains historical context, not a matched control.

Retain a candidate only if it reduces gauge-invariant seams without unacceptable
RGB loss or trivial isotropization. Even success cannot establish true kd/Q
recovery with unknown illumination and no material ground truth. Do not restart
100000 steps, silently disable priors, tune illumination scales or launch Bell/E.17.
Calibrated/known illumination or independent reflectance evidence is needed for
a trustworthy physical-albedo claim; smoothing alone cannot remove that gauge.
