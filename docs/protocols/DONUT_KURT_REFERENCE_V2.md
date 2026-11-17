# Donut v2: slope-space reflectance and conditional completion

Preparation only. This is one proposed bounded execution, not an authorized
rerun. The original v1 failed batch, malformed report, authorization and freeze
remain unchanged. Scientific baseline: reviewed3a4288032b9f81f26e8f223318894020dc5d8699.
The new protocol is `donut-kurt-reference-v2`.

## Unchanged acquisition and saved evidence

Reuse the oracle donut, existing128×128 camera calibrations, smooth shading
normals, principal director and world-Q. Color[.10804546,1,.03044616],metallic.8,
major/minor widths.35/.08, additive Kurt a=0/ks=1, Schlick metallic F0,
L2_six_cardinal, zero indirect/no point light, fitting0/50/1/45 and withheld28/34
are unchanged. No normal flip, material substitution, pixel removal, energy
clipping or threshold adjustment is permitted.

The prior source is the exact failed batch
`/home/isavand/paper/evidence/donut-kurt-reference-v1-5df0234193664e9b93cae546542ca7a2`.
All eleven original file hashes are bound by the reviewed offline-analysis
manifest, including its invalid terminal JSON. Camera0 NPZ SHA-256:
`8e49c382adeba827163f42b55f1b79d2d3c43fe1fdc9530dc4307fc97b4ab99c`.
Read numerical archives with allow_pickle=False. Do not repair or overwrite
the original report. It never became an admitted dataset.

## Exact change of variables

Let n be the known unit smooth normal, a the unit major director, b=n×a,
wo=(wx,wy,c) in that frame with c>0, and mx,my the Beckmann widths. Use scaled
Gaussian polar slopes sx=mx*r*cos(phi),sy=my*r*sin(phi), r≥0. Then

```
B = mx^2*cos(phi)^2 + my^2*sin(phi)^2
A = mx*wx*cos(phi) + my*wy*sin(phi)
ch = 1/sqrt(1+B*r^2)
h = (sx*a + sy*b + n)*ch
oh = wo.h = (c+A*r)*ch
wi = 2*oh*h - wo
ci = n.wi = 2*(c+A*r)/(1+B*r^2) - c
```

The existing physical specular BRDF is F(oh)*D(h)/(4*oh), with no added Smith
factor or hidden co denominator. Reflection gives dOmega_i=4*oh*dOmega_h on
the valid support. Slope solid angle is dOmega_h=ch^3*dsx*dsy, slope-plane
Jacobian dsx*dsy=mx*my*r*dr*dphi, and
D=exp(-r^2)/(pi*mx*my*ch^4). Thus

```
rho_spec = integral F(oh)*(ci/ch)*exp(-r^2)*r/pi dr dphi
rho_diff = kd*(1-metallic)                     # analytic Lambert integral
F0 = .04*(1-metallic) + metallic*kd
F(oh) = F0 + (1-F0)*(1-oh)^5
```

The condition ci>0 is exactly c+2*A*r-c*B*r²>0. Its valid radial interval is
0≤r<rmax, where

```
rmax = (A+sqrt(A²+c²*B))/(c*B)       if A>=0
rmax = c/(sqrt(A²+c²*B)-A)          if A<0  # stable equivalent positive root
```

Positive oh follows on this interval. Map u=1-exp(-r²), with
umax=-expm1(-rmax²), r=sqrt(-log1p(-u)). The probability measure becomes
du*dphi/(2*pi); the integrand is F(oh)*ci/ch. **Do not divide by valid probability.**
There is no finite radial-tail cutoff or accepted-sample normalization. Where
umax rounds to1 at float64 precision, the quadrature covers the entire radial
CDF to that precision; no user-selected radius is introduced.

Use float64 Gauss–Legendre quadrature in u∈(0,umax), and four azimuth quadrants.
Shift phi's origin by atan2(my*wy,mx*wx), determined only by the known view and
material, so the A=0 boundaries align with quadrant edges. This is analytic
domain alignment, not response-driven adaptation. Normal-view offset is0.
Coarse:64 radial nodes ×64 nodes/quadrant ×4 =16,384 nodes/point. Fine:
128×128×4=65,536. Total81,920/point. No other order, refinement, retry or tuning.

RGB reflectance follows two scalar Fresnel moments I0 and I5, combined as
F0*I0+(1-F0)*I5. This is exact linearity in RGB, not a BRDF approximation.
All nodes must satisfy finite positive ci,oh. Fresnel's cosine receives the same
unit-roundoff clamp as the old independent incident kernel; reflectance and
weights are never clamped or normalized. The implementation imports no Torch,
native shader, model or geometry and uses deterministic NumPy float64 on CPU.

## Validation first, before any new rays

Synthetic preparation fixtures verify the reflection Jacobian against finite
differences; exact radial boundaries including negative A and synthetic grazing;
quadrature probability weights; analytic diffuse; an independent1D isotropic
normal-view integral; direct incident-hemisphere controls at ordinary synthetic
angles; director-sign/axis-swap invariance; frame conversion and work accounting.
These are not measurements of the five real failing observations.

The execution first checks original input/report hashes, then evaluates ONLY
the five saved camera0 witnesses: pixel IDs12635,4037,12581,4825,4027. Both slope
orders must agree under the original bounds: maximum RGB-channel difference
≤0.003 AND relative vector L2≤0.002. Save both arrays and old hemisphere values
before testing. Failure stops without GPU initialization or new rays.

Next use eight fixed ordinary-angle camera0 observations, n·wo≥.25 (view angle
≤75.52°), selected by equally spaced ranks in original pixel ordering, without
using material responses. Their IDs are4151,5713,6812,8217,9771,10784,11835,13508.
Require: slope coarse/fine agreement; agreement of the preserved incident256²/
512² reflectance at these points; and new fine slope versus preserved incident
fine agreement. Apply the same0.003/0.002 bounds to each check. There are no new
incident BRDF queries or visibility rays for these crosschecks. Failure stops.
Two-order agreement is evidence of convergence, not a proof of absolute accuracy.

Save the exact quadrature nodes/weights and dedicated grazing/ordinary JSON and NPZ evidence. Cache these13 new slope
results. Recompute slope reflectance for the other4,523 valid camera0 observations,
so all4,536 receive the same new reference method without duplicate evaluation.
Require the same full-view two-order convergence bounds before any new rays.

## Conditional dataset completion

Only after all preceding gates pass, copy camera0 RGB/components, masks,
positions, normals, material/Q labels, triangle/barycentric identities and other
numeric fields unchanged. Replace its reflectance/coarse_reflectance arrays
with the validated slope result; retain old hemisphere arrays as explicitly
labeled diagnostics and store analytic diffuse/specular reflectance. Verify
exact dtype/shape/C-order byte equality of every reused array. Copy its three RGB/component PNG previews
byte-for-byte. Bind original NPZ and all original artifacts in the new report.
Do not retrace camera0, regenerate its RGB, or infer producing-history identity.

Generate ONLY cameras50,1,45,28,34 with the unchanged v1 primary-ray and independent
256²/512² RGB integration, visibility conventions, corrected BVH and image gates.
Both RGB grids must pass0.003/0.002. Compute all valid points' unoccluded reflectance
using the same64/128 slope orders and the same full-view convergence bounds.
Keep incident-grid energy values as diagnostics from the already evaluated BRDF
values; they do not substitute for the new reflectance rule. Preserve failure
arrays/reports before stopping. Do not remove additional grazing observations.

Save material/reference numerical maps, linear diffuse/specular/total images,
highlight masks/diagnostics and RGB/component contact sheet. Report reflectance
above1 without clipping; corrected-core is not generally energy conserving.
Completion is `complete_pending_researcher_admission`, not automatic approval.
Terminal identities, exact work counters and cleanup must pass before completion.

## Exact workload, limits and execution

The13 validation points require1,064,960 slope nodes, reused within camera0's
371,589,120 total nodes. Maximum whole job:

- 7,082,475,520 slope nodes, separately counted attempts/returns.
- 81,920 NEW primary rays,26,843,545,600 NEW shadow directions and the same
  number of independent incident BRDF point-directions;40 NEW CPU crosschecks.
- ZERO new camera0 RGB/geometry rays; ZERO native Material A models, optimizer
  updates, checkpoint loads, Blender calls, image builds or training.

Existing immutable image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A100 GPU0 for conditional new views,4 CPU threads,64GiB RAM/no swap,128 PIDs,
25% CUDA allocator,512MiB outputs.7200s foreground native watchdog/30s grace,
7080s inner limit; outer supervisor7260s/30s. Estimated5–40 minutes, unmeasured
for the full CPU slope work; partial v1 camera0 took about61 seconds. No budget
extension or fallback execution strategy is authorized after failure.

Use a fresh hash-bound output, read-only source/oracle/calibration/previous
batch, non-root, no network, no installation. The existing owned-container
cleanup and foreground launcher are reused. Existing Codex Docker restrictions
must not be bypassed. A single explicit approval must bind final source commit,
v2 freeze, old artifact identities, limits and fresh output BEFORE execution.

Inside a normal tmux shell, the eventual approved command uses the existing
launcher with DONUT_PHASE=reference, DONUT_FREEZE_NAME=DONUT_KURT_REFERENCE_V2.freeze.json
and DONUT_PREVIOUS pointing to the ORIGINAL failed batch (not a latest directory).
No affirmative authorization is created during preparation. After successful
generation and offline verification, request separate researcher dataset admission
and D1/D2 training approval. The training body, initialization, priors, schedule,
sample counts and budgets are unchanged; only reference-protocol admission accepts
the new validated dataset. No training is covered by this reference authorization.
