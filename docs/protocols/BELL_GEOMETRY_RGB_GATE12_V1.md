# Gate 12 — Authenticated Bell geometry-to-rendering sensitivity

Preparation only. One separate approval must bind the final committed source,
freeze SHA-256, corrected image and fresh output before the launcher is used.
No checkpoint deserialization, SDF query, new primary ray, material update or
segmentation inference belongs to this experiment.

## Admitted derivative and immutable controls

The researcher admits the authenticated O1 step-300k candidate **only as the
perfect-synthetic-mask-assisted `G_background_opacity_oracle_v1` derivative**.
The [separate handoff](../results/publication_gate12/O1-controlled-handoff.json)
binds its producing checkpoint, training/evaluation receipts, extraction policy,
mesh and accepted Gate 11 verification. It does not rewrite any old receipt or
represent O1 as unchanged original G, a realistic SAM2 result, or an automatically
authorized material-training run. O0's failed admission remains failed.

Three immutable meshes are used: admitted Bell oracle, admitted O1 derivative,
and numerically valid domain-aware O0 as an **unadmitted diagnostic only**. No
mesh writes, trimming, component selection, remeshing or normal flipping occur.
The checkpoint is identified in the handoff but is not mounted or deserialized.

Use the original six Gate 10 cameras in their frozen order: validation 58,
held views 11/105/114/17/76, each 128×128. Reuse saved full-raster rays and
closest hits from O1 Gate 11 and O0 Gate 10.2; do not repeat tracing or quality
admission. These are development diagnostics, with no new fitting-view rendering.
All 122 fitting cameras remain reserved for subsequent matched recovery.

## Physical definition and support

Use **original Bell v3 illumination**, not E.14 L2: ambient RGB .08 and the two
original lobes (+Z, power12, RGB1/.8/.6; +X, power6, RGB.3/.5/.9). Keep exposure1,
linear RGB, zero indirect, no point light and the original additive Kurt/Beckmann
BRDF. No energy repair or new BRDF is introduced. The original additive model's
energy-conservation limitations remain limitations.

Oracle inputs are the existing position-dependent analytic material labels,
actual archived shading normals and calibrated normalized outgoing directions.
Reconstruct their physical Q from sorted squared widths and the archived major
director; replay the independent NumPy label equations within 2e-12. Exact
isotropy has no eligible orientation: a tangent basis can represent its same Q,
but no orientation accuracy is invented. Keep the archive orientation eligibility.

For each reconstructed hit, use **exactly** the accepted same-ray correspondence:
both surfaces positive-facing, displacement≤.02 and signed-normal angle≤30°.
Replay its masks from the saved audit; do not loosen eligibility for rendering.
On valid pairs, evaluate the **unchanged original analytic Bell field** at the
reconstructed position and actual shading normal. In particular its ambient
definition is projected into that normal's tangent plane. This makes the primary
material field consistent across cameras, rather than copying a different oracle
ray's labels onto the same reconstructed point. Parameters of the analytic field
remain fixed; local widths/color/metal can change with its coordinates and normal.
This is an explicit analytic model on admitted near-surface support, not a measured
label on the reconstruction. The archived-label versus audit-normal roundoff is
recorded, with the archived actual material normal authoritative.

Reconstructed hits without a valid correspondence, including all reconstruction-
only hits, have **undefined material and RGB**. Do not extrapolate labels to them.
Reconstructed misses have known zero radiance. Preserve every excluded observation,
overlapping facing/distance/normal reason, excess and missing mask, undefined
oracle energy and full-raster denominator. Report both per-arm common support
and the additional joint oracle/O0/O1 valid subset. Never use the favorable joint
subset as a replacement for the full coverage accounting.

## Independent rendering and controlled substitutions

Reuse the accepted independent Cartesian Kurt kernel, float64 midpoint
solid-angle hemisphere nodes and `dOmega=2*pi/N` sum with cosine. Corrected native
BVH is only a visibility service. Each immutable geometry gets its own BVH;
visibility is recomputed with the original float32 direction, origin+1e-5*direction
and depth≥10 miss convention. No native material model/shader defines truth.

Integrate five cases on each valid reconstructed point:

1. **Primary:** the same fixed global analytic material field at the reconstructed
   position/normal, with its own geometry's incident visibility.
2. **Oracle visibility:** the identical BRDF, world directions and light samples,
   replacing only binary visibility with oracle tracing from the corresponding
   oracle point. Primary minus this case isolates the defined visibility-function
   substitution; that function includes the change in origin and mesh. It does
   not isolate open-boundary causality or pure position versus mesh separately.
3. **Unoccluded:** the same primary BRDF with visibility1. Its oracle comparison
   combines the analytic field's coordinate and normal-projection effects, not
   a pure normal intervention.
4. **Transported source material:** keep the matched original oracle color, metal
   and sorted widths, with Q/director transported by the unique minimum normal
   rotation into the reconstructed plane; use its own geometry visibility.
   Primary minus this control includes coordinate-dependent material resampling
   and projection-versus-transport changes. Scalar changes are included, so it
   is not a Q-only or position-only effect. This **camera-conditioned diagnostic
   is never the primary field** and is never passed to a training objective.
5. **Transported source unoccluded:** reuse case4's exact BRDF samples with
   visibility1. Compared with the original oracle unoccluded material, this
   isolates the normal/minimum-frame-rotation effect for fixed scalar values,
   widths and outgoing direction under a distant environment. Position alone
   has no unoccluded effect for this fixed material.

Oracle cases1/2/4 coincide; oracle cases3/5 are unoccluded. Save diffuse, specular
and total RGB. Save source/target points, source Q, transported Q and primary Q.
Separately evaluate Q at the target position in the source plane, giving a
coordinate-Q shift; compare its transported Q to target-normal projection to
quantify the additional projection-versus-transport shift. These are physical
tensor diagnostics, not interchangeable RGB causal contributions.

Each reconstructed pair uses the **same world directions** for its own and
oracle-visibility queries. Save per-point counts for visible, disagreement and
both disagreement directions, plus light/cosine-weighted disagreement. Normalize
by each observation's own grid count before pooling so refined pixels are not
overweighted. Blocker intersections near the radial omission band are a diagnostic
association, not proof that an open edge caused an error. Report error cohorts
with any observed incident disagreement and any blocker in that radial band;
neither cohort is a new training weight or an outcome-based acceptance gate.

Also perform material-free visibility checks at **all undefined reconstructed
hits** on both baseline grids: query both meshes at the same reconstructed origin
and directions in its normal hemisphere. Record counts and their grid change.
Such an origin can lie inside the oracle. Label this as geometry-only visibility,
not a physical oracle correspondence or an RGB score. These checks assign no
materials and have zero BRDF work.

## Convergence and prospective bounded refinement

Use 256²/512² baseline grids. Preserve the established .003 maximum absolute
channel and .002 pooled relative-L2 limits. Apply both limits separately to all
five cases' diffuse/specular/total components, so a total-RGB cancellation cannot
hide a component failure. Fresh oracle total radiance must also agree with the
admitted v3 MIS reference under the original limits; keep the original reference
bytes unchanged. Report oracle replay discrepancy and deterministic-grid changes;
two-grid agreement is not proof of absolute integration accuracy.

If either gate fails, choose an unrefined source pixel deterministically across
all three arms. Absolute failures take priority, ranked by the largest component
channel change. Otherwise rank by summed squared coarse/fine component change
in an arm with a failing pooled gate. Break ties by pixel ID, then oracle/O0/O1.
Evaluate that pixel wherever radiance is defined at1024² and2048², preserving
baseline captures. At most32 distinct refined pixels per camera and128 over
the whole panel. A refined absolute failure, unresolved pooled failure, exhausted
budget, nonfinite value, identity failure or resource failure stops the batch.
No automatic extra grid, altered threshold, retry or partial success.

The file-only verifier replays capture inventory, selection and all scientific
decisions exactly. Only the scalar norm reduction permits the established Gate3
allowance (absolute1e-15 / relative1e-12); both independently computed gates must
pass. Saved CPU/GPU float64 material-transport arrays permit only2e-12 absolute
replay roundoff, with every nonzero event recorded. Identities, counts, masks,
grid assignments, component sums and scientific flags remain exact.

## Evaluation and interpretation

Report per camera, validation and held split: linear RGB relative L2, RMSE,
signed luminance, diffuse/specular errors, distributions/tails and the coarse/fine
uncertainty in the **between-geometry difference**. Include common foreground,
known full raster, missing, excess, unmatched, background and joint valid scopes.
Undefined support prevents reporting a completed all-raster RGB score; retain
its oracle energy and the explicitly labeled measured-error lower bound only.
This numerical accounting is not an attainable inverse-rendering bound.

Report grazing(cosine<.1), normal error>15°, accepted/rejected large depth shifts,
and near-open-boundary(distance to saved boundary-edge midpoints≤.02) cohorts.
Boundary-edge proximity is a proxy, not exact distance to every edge segment.
All normal/distance correspondence failures and all large-shift exclusions remain
counted even when radiance cannot be assigned. Use identical figure scales and
magenta for undefined pixels, display clipping counts and unclipped numeric arrays.

O1 versus O0 measures whole-geometry changes. Only the declared substitutions
support narrower statements. State explicitly whether O1's support benefit
coexists with worse common-valid RGB or specular tails. Numerical uncertainty
must be reported relative to the measured effect; no arbitrary new success
threshold is introduced. Gate12 requires researcher interpretation before a
matched recovery launch. If numerical gates fail, preserve and report the specific
failure rather than recommending geometry repair from the observed outcome.

## Execution envelope

The committed freeze derives exact conservative work caps from the saved point
inventory: five baseline BRDF/visibility slots per oracle point, plus five slots
per allowed refined pixel. Undefined-hit visibility adds two baseline slots per
undefined reconstructed observation and no BRDF slots. No primary rays,
checkpoints, SDF queries, native models or optimizer returns are authorized.
The final numerical ceilings are45,018,316,800 shadow directions,
42,678,681,600 BRDF point-directions and25,607,208,960 environment point-directions.

Use the corrected image/BVH bound in the freeze, A100 GPU0,4 CPU threads,
64GiB RAM/no swap,128PIDs,25% CUDA allocator and2GiB owned outputs. Scientific
foreground watchdog7200s/30s grace, inner7080s/20s grace; file-only verification
at most600s. Overall proposed foreground wrapper7800s/30s grace. Estimate20–110min,
unmeasured; the ceilings, not this estimate, bind execution. Non-root, read-only
source/inputs/root, no network, no build/install and no retries. Prior active
geometry/material containers or a busy GPU cause refusal before launch.

After separate approval, create the exact `authorization(f, committed_HEAD,
freeze_sha)` record beside the fresh output, then run the existing foreground
launcher with `BELL_RGB_SOURCE` and `BELL_RGB_AUTH`. Preparation and geometry
admission do **not** authorize it. Large native arrays stay outside Git; commit
the verified small evidence, figures and scientific conclusions after execution.
