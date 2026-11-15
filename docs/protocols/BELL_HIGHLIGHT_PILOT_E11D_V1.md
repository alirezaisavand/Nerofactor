# E.11D v1 — paired independent Bell lighting pilot

**Preparation only. One explicit commit/freeze/input/output-bound approval is
required before new rendering.** Do not regenerate128 views or run any material
model, optimizer, reconstruction or image build. Preserve v3 as a potential
weak-cue control. Base: `929414f6a18d97e3f15f1ec35aeb4e381509f8ef`.

## Fixed observations and independent equations

Use saved primary hits, unit normals, labels and view directions from the original
v3 package, generation SHA256
`5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7`.
The original mesh SHA256 remains
`4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01`.
All original world coordinates, winding, camera matrices, pixel centers and
128×128 resolution are unchanged. No new primary rays, rescaling, calibration
pickle execution, surface selection by highlight strength or normal repair.

Choose cameras **0,58,105**, spanning an existing training view, validation view
and held view. These three established diagnostic views are selected for different
camera elevations, not their future counterfactual scores. All foreground pixels
receive truth rendering. For paired material probes, divide the image into8×8
cells of16×16 pixels. In every nonempty cell, select at most three foreground
pixels at the first/middle/last evenly spaced integer ranks of C-order indices.
No importance weighting or selection by material/lighting/response. This gives
90/84/72 probes,246 total (at most192/view), across12,157 foreground observations.
The probe panel is spatially stratified, not an unbiased area sample; report
per-view scores and counts, not population identifiability estimates.

Independent reference is the additive Kurt a=0 law:
`f = k(1-m)/pi + F*D/(4*wo.h)`, Schlick F0=.04(1−m)+mk,
Cartesian anisotropic Beckmann
`D=exp(-((h.t/mx)^2+(h.b/my)^2)/(n.h)^2)/(pi*mx*my*(n.h)^4)`.
Use only valid upper-hemisphere terms and the same original visibility policy.
This law remains not globally energy conserving. No native A/A2/corrected shader
is imported. The explicit-axis independent kernel is checked against the accepted
Cartesian reference on synthetic inputs, including a common rotation and misses.

Retain exact16,384-direction fine /4,096-direction nested coarse cosine/isotropic
GGX MIS tapes, samples, PDFs and normalization from v3. All lighting/material
counterfactuals share identical world directions and the **same** corrected-BVH
visibility decisions. One shadow query set per surface point, float32 offset
`p+1e-5*wi`, depth≥10 is a miss, zero indirect/human light. No accepted-ray redraw.
Reuse the admitted corrected image/backend; no build or E.11C repetition.

## Fixed lighting candidates: equal total incident RGB radiance

For `L(w)=ambient+sum RGB_j*max(w.axis_j,0)^p_j`, its complete-sphere mean is
`ambient+sum RGB_j/[2*(p_j+1)]`. Fix this mean per channel to v3's
(.1398901098901,.1464835164835,.1673626373626), rather than simply increasing exposure.
This controls global incident energy, **not** irradiance at every point or shadowed
surface. Angular and spectral placement still intentionally differ. Material,
geometry and outgoing views do not.

| Candidate | Ambient | Axes (unit normalized) | Power | RGB per lobe |
| --- | --- | --- | --- | --- |
| L0_v3 | (.08,.08,.08) | +Z,+X | 12,6 | (1,.8,.6),(.3,.5,.9) |
| L1_three_oblique | (.01,.01,.01) | (1,1,1),(-1,-1,1),(1,-1,-1) /sqrt3 | 48 | (4.2430769231,4.4584615385,5.1405128205) |
| L2_six_cardinal | (.01,.01,.01) | ±X,±Y,±Z | 64 | (2.8142857143,2.9571428571,3.4095238095) |

Full precision definitions are in the freeze. L1 tests compact asymmetric
illumination; L2 tests wider directional coverage with more, narrower peaks.
Symmetry in L2 may itself reduce orientation sensitivity at particular normals.
Lower ambient raises contrast, but compact lobes can concentrate irradiance,
clip PNG observations and increase integration error. Equal sphere means do not
guarantee equal local brightness, material sensitivity or estimator variance.
No tone mapping, clipping, per-candidate exposure or amplitude retuning.

## Paired physical counterfactuals (truth is unchanged)

At the246 probes render eight cases under each of the three environments:

1. Original smooth v3 ground-truth material.
2. Original tangent principal direction rotated +15° about the original normal.
3. The same direction rotated −15°.
4–5. Original major-axis slope width multiplied by1.1 / .9.
6–7. Original minor-axis slope width multiplied by1.1 / .9.
8. Matched isotropic widths `sqrt(mx*my)`, preserving the tangent-Q determinant.

Keep color/metallic, surface point, normals, lighting, tapes and visibility fixed.
Major/minor names in perturbations refer to the original axes: if a perturbation
crosses their ordering, do not resort axes or change orientation. All widths must
remain within[.04,1]; undefined probe orientation stops rather than fabricating a
chart. The current admitted probes have defined orientation. Isotropic matching
does not preserve all-direction reflectance; its change is the intended contrast.

## Frozen checks, diagnostic criteria and ranking

Mandatory source/input identities and v3 full-image replay must pass. Recomputed
L0 fine/coarse truth RGB must match the saved v3 arrays with max abs≤2e−10, a
float64 algebra-equivalence gate separate from scientific convergence tolerances.
Nonfinite/negative radiance, changed inputs, query/work/resource failure or an
incomplete return count stops the run and preserves failure/partial evidence.

For each candidate, each truth image and each paired-case probe set must satisfy
the inherited convergence bounds: foreground max RGB difference≤.003 and relative
L2≤.002. This is deterministic two-grid agreement, not a variance confidence bound.
Record all failures. RGB>1 makes the affected candidate ineligible for the native
PNG benchmark: retain raw arrays, omit invalid PNGs and execute only the remaining
already-authorized candidate panel. No threshold changes or replacement draws.

Measure diffuse/specular luminance, S/D, S/(D+S), per-view highlight coverage,
p99−p50/ratio contrast, linear ranges and original8-bit sRGB quantization as in
the saved-data report. Highlight=Yspec≥.01 AND≥3*per-view median(Yspec). These
descriptive cutoffs are not physical identifiability criteria.

For orientation/each width, record both the central difference and the maximum
absolute response relative to truth over the two signed perturbations. For the
isotropic case use its absolute truth difference. A probe's response is
**operationally resolved** only if all three hold:

- Maximum absolute linear RGB channel response≥.003.
- The response≥4 times the larger truth/counterfactual fine/coarse max-channel
  difference at that point (a conservative grid-sensitivity guard, not a CI).
- At least one original sRGB channel changes by≥2 integer codes.

Report fractions also at the quantization-only gate so its separate effect is
visible. Do not infer sensitivity from width gaps alone. Save the complete paired
RGB/component arrays; no gradients, optimized estimates or learned estimator.

Reuse the already computed weighted samples to report per-component contribution
concentration: `(sum x)^2/sum(x^2)` and `max(x)/sum(x)`, where x is a sample's
nonnegative diffuse/specular luminance contribution. Save truth concentration
maps. Compact illumination can reduce effective contribution counts and raise
grid error even with improved contrast. These fixed-QMC concentration diagnostics
are **not** variance estimates; independent-tape statistical uncertainty remains
unmeasured and is not claimed by this pilot. No extra radiance queries are used.

Report singular values of the per-probe RGB Jacobian approximation with columns
half the ±15° /±10% response differences. Numerical rank uses
`smin > max(1e-6*smax,1e-8)`. This is **conditional three-column, one-view RGB**
conditioning at fixed color/metallic, not proof of seven-parameter, spatial-neural
or multiview identifiability. Different-camera probes are not the same surface
point; do not pool them as if they were. Even passing these tests leaves joint
color/metal compensation and full material ambiguity unresolved.

A lighting candidate may be recommended for a separately frozen v4 only when:

- Every truth/paired reference converges and is in the PNG domain.
- Every camera has resolved fractions≥.25 for orientation, both widths and the
  anisotropic/isotropic contrast.
- Orientation resolved fraction does not decrease versus L0 in any camera;
  the probe-count-weighted gain over L0 is≥.10 absolute.
- Highlight coverage≥.05 in at least two of three views.

Among eligible candidates maximize worst-view orientation resolved fraction,
then mean of the two width resolved fractions, then minimize worst relative
convergence error. Exact ties use L1 before L2. Preserve all candidates, failures
and raw numbers. No result means automatic128-view generation or training.

## Work, resources and outputs

Exact planned work from admitted hit/probe counts:199,180,288 proposed shadow
directions,227,393,536 BRDF point-direction evaluations,682,180,608 light-weighted
point-direction evaluations. Valid traced queries/returns are separately counted.
No new primary tracing, native model/shader/optimizer, autograd or training.
Conservative enforced ceilings:805,306,368 shadow directions;871,366,656 BRDF
point-directions;2,614,099,968 light-weighted point-directions. These ceilings do
not authorize additional views/candidates/probes. No new random tapes.

One A100 GPU0; corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
binary SHA256 `800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.
Four CPU threads,64GiB host RAM/no swap, CUDA allocator fraction.25,512MiB output
cap. One foreground1800s watchdog/30s kill grace; inner1740s/20s grace. Estimated
1–10 minutes, **unmeasured**. No retry, budget extension or alternative backend.

Fresh directory and authorization path are in the preparation manifest. Run the
existing-style foreground launcher `experiments/run_bell_highlight_e11d.sh` only
after explicit approval binds its final commit/freeze and directory. Source,
original v3 package and authorization are read-only; non-root, no network, no
image build. If Docker/GPU access is restricted, supply the identical approved
researcher-side command without bypassing security restrictions.

Save nine full truth fine/coarse NPZ captures, paired component archives within
each, source pixel IDs, images only where valid, per-view JSON diagnostics,
terminal `pilot.json`, output hashes, runtime/backend identity, work counters,
cleanup result, console and exit code. On Python failure save incomplete arrays
with an explicit completed-pixel mask; zero uncompleted slots are not radiance
measurements. Hard-kill may prevent finalization: preserve host receipt/partial
files and do not infer completion. File-only analysis after execution may create
figures/ranking without additional scientific queries.

## If lighting alone fails

Do not silently change this pilot. Propose a **separate** material factor under
the selected fixed lighting: original versus major*=1.25,minor*=.6 and
metallic=.65+.05sin(pi*qx)cos(pi*qy), unchanged color and orientation/Q chart.
This intentionally changes both width ratio and F0/diffuse allocation and would
need a factorial separation if either is claimed causal. It is not authorized or
implemented here. Prefer separate roughness-only and metallic-only arms over
attributing a joint change to one factor. Higher resolution is deferred: current
saved highlights are mostly multi-pixel; the pilot must first show compact cues
are poorly sampled before a separately approved resolution comparison.
