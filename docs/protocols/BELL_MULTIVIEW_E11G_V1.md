# E.11G v1 — physically visible angular diversity at fixed Bell points

**Preparation only until one explicit execution approval.** Base
`3c6cf4f511ab825203e190ba078a75d59411c000`. Preserve all E.11D/E/F evidence,
failed candidate decisions, frozen G/A sources and the original v3 benchmark.
No full images, dataset regeneration, new lighting, neural models or training.

## Hypothesis and single controlled intervention

Test whether multiple physically valid outgoing directions separate the nearly
parallel orientation/width RGB responses found in E.11F. Keep camera58's original
84 **world-space surface points**, normals, frames, original and stronger material
definitions, fixed L2 environment and independent Cartesian BRDF. This is not a
nearest-pixel/nearest-surface correspondence experiment. Evaluate each exact
point towards additional existing camera centers, retaining its subpixel image
projection rather than substituting the nearest rendered image pixel.

Retain the entire original panel, including probes with no added visible view.
No view/point selection may use rendered material responses or recovery errors.

## Geometry-only view selection, frozen before rendering

Use the admitted primitive numeric `geometry/calibrations-safe.npz`:
SHA256 `4d8f4e13d7fd5fffec8d258bfc7c5599ef6d81525547d678ba088e76b1656c9c`.
No camera pickle is loaded. Poses mean world-to-camera R|t, camera center=-Rᵀt,
wo=(center-point)/distance. World units remain original (meters unverified).
Intrinsics are the unchanged128×128 cameras, already scaled128/800; projection
is continuous, no new pixel-center offset or image resizing.

Starting with58, greedily add **three** distinct camera IDs from0–127. A camera
is eligible if at least42/84 points have n·wo>0, positive camera z and projected
coordinates within[0,128)². Score it by the minimum, over already selected cameras,
of its median angular separation across **all84 original points**. Maximize the
score rounded to1e−10 radians; ties choose lowest ID. Do not use visibility
outcomes or BRDF responses for selection or substitute an occluded selection.

This deterministic geometric calculation selects **[58,44,9,0]**. The extra
cameras have42,42,62 geometrically admissible correspondences, respectively;
their greedy minimum median separations are85.70767°,81.50759°,51.91374°.
All128 candidate counts and selections are saved in the freeze. These are
pre-render geometric measurements, not evidence of observable material recovery.

The base view keeps E.11E's actual saved ray-derived wo. Its camera-center
direction must agree within the existing2e−6 geometry roundoff bound. No base
RGB/legacy counterfactual is rerendered. Camera58's84 correspondences retain
their accepted primary-ray visibility. Added views may be grazing or occluded;
report this, rather than changing the selections after seeing the result.

## Minimal new geometry work

Exactly **146** point-to-camera segment queries, once per positive-facing,
in-frame added correspondence, using the admitted corrected CUDA BVH. No extra
primary rays and **zero new incident-shadow queries**. Reuse E.11E's exact
256²/512² local nodes, solid-angle weights, packed incident visibility and all
input/source identities. Opaque geometry makes incident visibility independent
of material and outgoing direction; this reuse is valid for this experiment.

Use the same float32 tracing conversion and1e−5 offset along the outgoing ray.
Record nearest depth, unoffset camera distance, floating origin/direction
conventions, n·wo and subpixel projection. A camera is unobstructed only when
the nearest hit is beyond `distance−1e−5+2e−4`. A hit before
`distance−1e−5−2e−4` is occluded; the intervening endpoint band is unresolved.
The2e−4 segment guard is the already admitted native depth/position tolerance,
not a new BRDF response threshold. No retries, offset adjustment, normal flips,
mesh repair, threshold relaxation or missing-view replacement. Nonfinite or
negative trace depths fail execution. Keep invalid/unqueried payloads as NaN
with explicit geometric/queried/valid/occluded/unresolved masks.

## Paired material measurements

For both original and exact E.11F-strengthened materials, use unchanged truth,
orientation±15°, major±10%, minor±10% and matched isotropic control. At new
outgoing directions compute the mathematically common canonical isotropic case
once and share it after the E.11F equivalence gate. At58 reuse all eight accepted
columns byte-for-byte from E.11E/E.11F.

Add four direct physical counterfactuals to each material at every valid view:
u=log(sqrt(mx*my)), v=log(mx/my); u±.1 multiplies both widths by exp(±.1),
while v±.1 multiplies mx by exp(±.05), my by exp(∓.05). Hold all other parameters
fixed. These are actual BRDF evaluations, not relabelled old secants. Reject
widths outside the unchanged[.04,1] domain; no clipping. The new contrasts use
the **same** .003 /4*local grid difference /2-native-sRGB-code response gates.
Keep original radiometry and quantization; no exposure, denoising or reweighting.

Every measured case/view retains max RGB≤.003 and relative L2≤.002 reference
convergence. Zero visible correspondences are unresolved, not a passing empty
case. A failed reference-convergence panel is inconclusive: preserve it and do
not add grids or reinterpret failed responses. Both deterministic grids can
share physical/numerical errors, so agreement is not absolute-accuracy proof.

## Matched comparisons and interpretation

Record per-probe valid view count1–4. Compare base and stacked responses on the
**same points**, both stratified by exact count and over all points with at least
two views. Never compare a favorable multiview subset with the unfiltered base.
Report individual-view response gates, any-view unions, joint orientation+both-
width coverage, quantization and invalid/occluded/missing correspondence counts.
Any-view coverage is an opportunity measure, not an inferred inverse solution.

For each point construct3V×3 matrices with RGB rows per view. First basis:
half signed contrasts for θ/mx/my (legacy perturbation-scaled convention).
Second basis: θ derivative secant divided byπ/6 radians and actual u/v central
secants divided by.2. Keep the bases separate: condition numbers depend on
parameter scaling. Normalize stacked matrices by1/sqrt(V), preventing repeated
identical views from manufacturing stronger singular values. Report smallest
singular values, full spectra, condition distributions, column norms and the
existing numerical rank convention, not rank as a recovery verdict.

Record the spectral norm of fine-minus-coarse matrices as a deterministic
grid-difference proxy and max(0,smin−proxy). This is not a statistical bound or
an absolute Jacobian error guarantee. Tiny singular directions near that proxy
remain unresolved even when numerical rank is full. Finite perturbations are
not exact infinitesimal derivatives. Color/metal stay known: this does not prove
seven-parameter identifiability, neural optimization success or full-scene recovery.

For interpretation, on the matched ≥2-view panel predeclare a **descriptive**
material-improvement screen: median RMS-normalized smin at least doubles and
median conditioning ratio is≤.5, with both original width response coverage
floors≥25%. Report both bases and both materials and all failures; no best-arm
selection. This new exploratory screen is not a retrospective E.11D/E/F gate
or automatic full-benchmark admission. If conditional evidence is favorable,
recommend the smallest cross-view design; otherwise distinguish limited angular
coverage from persistent BRDF/parameter compensation and recommend one next test.

## Fixed bounds, outputs and authorization

Work ceiling: **1,320,550,400** new BRDF point-directions,
`(8*84 +23*146)*(256²+512²)`. Actual work decreases only with recorded invalid/
occluded outgoing views. Eight new u/v columns per base probe;23 unique
material-case evaluations per admitted extra-view correspondence. The actual
count must equal `(8*84+23*number_of_valid_extra_correspondences)*327680`.
Exactly27,525,120 environment point-directions, shared across cases/views.
No original base rendering, primary/incident-shadow rays, models or optimization.

Existing immutable image:
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`;
corrected binary SHA256
`800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.
A100 GPU0,4 CPU threads,64GiB host memory/no swap,25% Torch allocator,256MiB
outputs,4-point×16384-direction chunks. One foreground900s watchdog/30s grace,
inner840s/20s. Estimated20–120 seconds, unmeasured; no extension or retry.
Non-root, no network, read-only source/v3/E.11D/E.11E/E.11F and root filesystem.

Save frozen view selection, numeric outgoing geometry/masks, both grid captures
including NaNs for unmeasured slots, original-ID maps, comparison metrics,
counters, source/input/image/binary identities, runtime/memory, console/exit and
completion/cleanup status. Preserve partial captures with completed-probe masks.
Only owned container resources may be cleaned up; successful cleanup must return
before complete status. File-only analysis adds no scientific queries.

One explicit approval must bind the final commit/freeze, these limits and a fresh
batch; it covers this experiment and offline analysis/figures/evidence commit/push.
If Codex Docker access is restricted, provide the identical approved foreground
tmux command without bypassing permissions. No full dataset or training is
authorized, regardless of the outcome. Preserve all E.11D/E/F conclusions.
