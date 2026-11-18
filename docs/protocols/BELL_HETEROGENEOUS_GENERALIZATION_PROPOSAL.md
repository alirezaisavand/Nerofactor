# Bell heterogeneous generalization proposal

Bell is the second geometry for a later generalization batch. It has
faceted geometric normals from A's BVH convention, a different silhouette
and surface topology from the smoothly shaded Donut, and an admitted
128-view geometry archive. Retain the Bell E.11 v3 mesh, camera
calibrations, scaled 128×128 intrinsics, half-pixel convention,
saved primary intersections and final negated BVH normals. Reuse
saved geometry through explicit hash admission. The historical Bell
material, RGB, relights and results remain independent evidence.

Use a six-view panel matched in size to Donut. Select fitting cameras
from the admitted E.11 v3 training IDs. Start at the lowest ID and
repeatedly maximize the minimum angular distance to the selected
camera centers. Use lower ID as tie break. Add two held-out cameras
from the existing disjoint validation/NVS IDs by the same rule. Camera
centers are `−Rᵀt` from the safe numeric world-to-camera poses, with
directions from the original acquisition origin. This geometry-only
selection yields fitting IDs **0, 4, 19, 2** and held-out IDs **105,
114**. The six saved hit counts are 4,395, 4,365, 4,392, 4,421, 3,607
and 3,475. The safe calibration SHA-256 is
`4d8f4e13d7fd5fffec8d258bfc7c5599ef6d81525547d678ba088e76b1656c9c`.
The Bell generation and admission hashes are
`5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7`
and `f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.

Define the new material in normalized Bell mesh coordinates, with
smooth width modulation within geometry-defined regions and a width
family step across oblique planes. Use the same width families as
Donut and a smooth, constant-limit nearly isotropic patch. Keep base
color and metallicity continuous across the step. Project a fixed
world grain vector into the admitted tangent plane. At projection
degeneracies taper anisotropy to exact isotropy with zero anisotropic
deviator and mark orientation ineligible. This keeps the physical
tensor defined on a faceted surface and avoids assuming a global
nonvanishing tangent director. Freeze plane locations, widths and
patch parameters from oracle geometry before rendering.

Use the same L2 distant illumination, zero indirect light and
independent axis-kernel RGB and slope-space reflectance methods as the
heterogeneous Donut. Preserve the .003/.002 deterministic convergence
bounds. Saved Bell primary hits can be reused after validating the
geometry gate and original binary identity. Retrace shadow visibility
for new incident grids when required. No historical Bell RGB or
material labels represent the proposed heterogeneous material.
Report area/visibility coverage, region and boundary strata and the
same train/held-out metrics as for Donut. A subsequent bounded Bell
generation and dataset admission are needed before training.
