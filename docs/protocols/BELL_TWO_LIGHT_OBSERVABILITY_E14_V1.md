# E.14 — fixed two-light anisotropic observability, v1

Preparation only until a separate approval binds the final commit/freeze and fresh
output. This implements the unchanged [E.12 proposal](BELL_TWO_LIGHT_INFORMATION_E12_V1.md).
Preserve E.11D–G and E.13, including their failed criteria, artifacts and checkpoints.
E.13 motivates this acquisition test; no E.13 outcome selects lights, points or gates.
No full benchmark, neural/material training, new geometry queries, retries or tuning.

## Exact intervention and input admission

Reuse E.11G's camera58-derived84 world points, faceted normals/tangent conventions,
original and E.11F stronger materials, outgoing cameras[58,44,9,0] and exact masks.
There are220 admitted point/view observations:84 base +136 added. The10 occluded
extra correspondences, geometric exclusions, and all84 points remain recorded.
One point retains only its base view; the exact1/2/3/4-view strata and the matched
83 points with at least two views remain separate. No selection uses radiance.

The first light is unchanged L2_six_cardinal. The only added light rotates all
six lobe axes45 degrees about normalized world(1,1,1), using Rodrigues' formula.
Ambient, power64, RGB amplitudes and each lobe's whole-sphere radiance integral
2pi*RGB/65 remain unchanged. This preserves total incident radiance per capture,
not irradiance at every surface normal or a fixed total two-capture photon budget.
There is no exposure compensation or search. Freeze the actual float64 axes.

Reuse E.11E's256²/512² midpoint solid-angle nodes/weights and packed incident
visibility, and E.11G's accepted outgoing visibility/directions. Geometry is opaque;
visibility is independent of material and distant-light radiance. No tracer or
scientific model is imported/constructed, no checkpoint/camera pickle is loaded,
and no primary/shadow/outgoing rays are queried. The corrected image/backend
identity is inherited and its installed binary is hashed, without loading it.

Admission validates the exact native E.11G report SHA
`db1c0b3d5ce630501645d07612f940e46df6bfa77e551d839882be8503205e09`,
its output hashes, numeric geometry classifications, IDs and cached controls,
plus the existing E.11D/E/F identity chain. All NPZ readers use allow_pickle=False.
Frozen G/A sources stay unchanged. Old freezes remain version-bound to their own
source checkouts; E.14 hashes the actual source closure it executes.

## Rendering and work

Only the rotated light is newly integrated. Retain all12 E.11G cases for each
original/stronger material, including theta±15°, major/minor±10%, isotropy,
log-geometric-mean±.1 and log-ratio±.1. The canonical equal-geometric-mean isotropic
material is verified algebraically, rendered once at the NEW light for each
admitted point/view, then shared. Do not reuse an old-light isotropic radiance as
if it were invariant to lighting. The old L2 captures are copied exactly, never
rerendered. The new23-case count is12 original +11 stronger.

New exact work:220*23*(256²+512²)=1,658,060,800 BRDF point-directions, including
72,089,600 new-light canonical-isotropic point-directions. Compute27,525,120
world environment point-directions, shared across cases/views. Preserve the
existing independent Cartesian Kurt-a0 additive BRDF, four-point batches,
16384-direction chunks, horizon/normal/visibility conventions and reduction.
No random directions or simulated measurement noise are generated.

Save both lights/materials/grids in light-major archives with shape
(material2,light2,view4,probe84,case12,component3,RGB3). Components remain
(diffuse,specular,total); invalid correspondences are NaN, with original masks.
Keep partial arrays/completed-probe masks on catchable failure, progress counters,
source/input identities, runtime, memory and cleanup status. A hard watchdog can
leave only completed-grid/progress evidence; never infer missing work completed.

## Unchanged response and convergence criteria

For every material/light/view/case use the original maximum RGB difference≤.003
and relative L2≤.002 between grids. The response conjunction remains amplitude
≥.003, amplitude≥4*local grid discrepancy, and≥2 original oracle sRGB codes.
Preserve per-light results, matched points and exact-count strata. Code contrasts
reuse the original round-to-nearest oracle convention, not E.13's legacy prediction
PNG truncation. Record out-of-domain radiance/clipping; no hidden exposure/repair.
A failed convergence or PNG-domain result is inconclusive for reliable observability,
not retrospective acceptance of E.11D L2. Do not add grids or change gates.

Compare full admitted L2 multiview observations with the SAME views/points at two
lights. Normalize matrices by1/sqrt(L*V), where one observation is one RGB triplet.
Do not normalize each parameter column to manufacture a condition improvement.
Report all singular values, RMS column norms/cosines and conditions, weakest
vectors and fine-minus-coarse operator norms. Retain legacy half-contrast and
physical(theta radians,log mean,log ratio) bases separately. The existing screen
remains median paired RMS smin gain≥2, median paired condition ratio≤.5 and both
legacy width coverage floors≥25% on the matched83 points. Its baseline here is
single-LIGHT multiview, not E.11G's single camera. Report every material/basis,
all84 points and failed screens. No new log-ratio or Fisher pass cutoff is added.
This descriptive screen never automatically admits a dataset or recovery claim.

## Noise-aware diagnostics, no change to scientific thresholds

Use the original central secants: theta denominator pi/6 radians, and actual
log-mean/log-ratio denominators.2. Color/metallic remain known. These are finite
conditional responses, not exact infinitesimal derivatives or full seven-parameter
identifiability. Keep original/rotated lights separately and combined.

- **Ideal linear:** unweighted J and J-transpose-J spectra, without a calibrated
  Fisher or statistical-uncertainty claim for a noiseless inverse problem.
- **Hypothetical Gaussian linear:** independent additive unit-radiance RGB noise
  sigma=.001, unchanged across all channels/views/lights. F=sum J-transpose-J/sigma².
  It is not measured sensor noise, native estimator noise or noise added to targets.
- **Quantized observations under that SAME Gaussian model:** exact inverse-sRGB
  half-code boundaries; codes0/255 extend to minus/plus infinity. Compute P and
  dP/dmu with log-CDF/survival differences and expm1, then sum(dP²/P) in log space.
  No probability floor, information clamp, threshold tuning or sigma adjustment.
  Natural IEEE underflows are counted. Analytic/synthetic controls verify total
  probability, score derivative, binary-threshold/fine-bin limits, derivative
  finite differences and quantization's data-processing inequality.
- **Actual deterministic8-bit oracle:** no analogue noise is injected. Report
  truth codes, linear-radiance bin bounds/widths, grid-induced code differences,
  existing finite code contrasts, and which perturbations leave every code
  unchanged. Do not assign these noiseless code steps a Fisher interpretation.

For noisy models report total information and information per RGB observation,
spectra and weakest directions. In the JSON, condition is the weighted-J condition;
gram_condition is its square, the information-matrix condition. Divide the total
information by L*V for quantity-normalized comparisons. Total-data inverse-
information axis scales are reported only above machine-precision rank. These
are conditional local-linearized uncertainty diagnostics, not confidence intervals
or global nonlinear Cramer–Rao guarantees. A duplicate-light synthetic control
must double total information while preserving per-observation information,
conditioning and noiseless feasible strips.

Report sensor-bin line sections through the true parameters along each declared
coordinate and the ideal weak direction. They intersect all linearized code-strip
constraints while holding the other directions fixed; they are NOT joint/marginal
feasible ranges and do not impose global material-domain constraints. Saturated
bins may be unbounded. This reports quantization ambiguity without a new inverse
optimization or fictitious Gaussian noise on noiseless measurements.

Grid errors remain correlated deterministic approximations. Do not add them to
sigma². Compare RMS operator differences between fine/coarse response matrices,
including corresponding Gaussian/quantized weights, against weak singular values.
Positive smin-minus-proxy is not a rigorous absolute-accuracy guarantee. Report
all numerical limitations. No extra real-data sigma sweep is authorized.

## Resource and execution bounds

Existing image sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331;
no rebuild/install. A100 GPU0,4 CPU threads,64GiB host RAM/no swap,pids128,
25% Torch allocator fraction,256MiB outputs. One foreground900s watchdog/30s kill
grace, inner840s/20s. Expected roughly40–120s based on the old multiview work,
not a measured guarantee for the new integrations/information analysis. Stop and
preserve on resource failure; no retries or extension. Native SciPy1.10.1/Torch1.8.1
CUDA11.1 and the approved image remain required. CPU fixture results do not admit
native execution. Read-only code/original inputs/root, non-root, no network.

One approval must name the final commit/freeze and fresh batch directory. The
reviewed foreground launcher writes receipts and cleans only its CID/label-bound
container. If Codex Docker access is restricted, provide the identical researcher-
side tmux command without bypassing it. No full images, new dataset, neural/material
training or optimizer experiment is authorized by this study.

After the one execution, compare all materials and strata using the frozen gates
and noise models, preserve failed outcomes, and commit/push small evidence and
figures. If informative, propose one fixed multiview/multilight benchmark with
held-out views and a genuinely distinct relighting condition (a90° axis rotation
of six identical cardinal lobes can be a symmetry, not an unseen light). Otherwise
recommend an identifiable parameter subset or a different constrained acquisition,
rather than an open-ended illumination search. E.13 remains separate empirical
motivation and its physical-recovery failures are unchanged.
