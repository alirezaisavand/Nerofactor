# Prospective Gate 13 relighting: materials and geometry together

Planning only; no rendering/integration/optimization authorization. Freeze exact
Gate 13 endpoint export hashes and work/resource limits after its completed
verification. Do not select a different illumination after inspecting outcomes.

Use the already defined E.14/Gate 6 analytic **L2 rotated 45 degrees about
normalized (1,1,1)**, from `benchmarks.material_information_e14.rotated_light`
applied to `benchmarks.donut_recovery_v1.ENVIRONMENT`. This is intentionally a
sharper, directional light than original Bell v3; it was proposed before seeing
either Gate 13 recovery. Preserve ambient, lobe powers and RGB, exposure 1,
zero indirect and all original held cameras 11/105/114/17/76. It is a new Bell
evaluation illumination, not a replacement training light or illumination search.

One combined batch, four conditions:

1. Analytic true material on oracle geometry.
2. The same analytic material field evaluated at actual O1 hits/normals on O1.
3. Frozen K1-Oracle step-10k physical exports on oracle.
4. Frozen K1-O1 step-10k physical exports on O1.

Reuse saved ray hits and corresponding common support; recompute incident
visibility independently per immutable mesh. If maps already provide all physical
exports, no pickle load or neural query is necessary. Undefined O1 material
support remains undefined: do not invent source labels outside valid transfers.
Keep full-raster missing/excess counts and separately state which zero-background
contributions can be scored. Never infer unseen predictions from archive zeros.

Use the validated independent float64 Kurt kernel, shared incident nodes,
256²/512² baseline grids and unchanged .003 max absolute/.002 pooled relative
convergence. Prospectively retain the accepted bounded 1024²/2048² refinement
policy (32 observations/view, 128 total), preserving original and refined arrays.
The final executable freeze must count all four BRDF conditions, all visibility
directions, selection/stopping rules and failure/timeout paths before approval.
No new integration occurs while preparing Gate 13.

Report diffuse/specular/RGB on identical supports, per camera and pooled held
views, reference uncertainty versus between-condition differences, weak/highlight,
grazing, normal and displacement cohorts. Compare learned-versus-true material
on each geometry separately; the fixed-true geometry comparison is part of this
same experiment, not a redundant standalone geometry gate. Geometry and material
effects interact: do not sum these residuals as an additive causal decomposition
or call a true-material discrepancy a lower bound on optimized RGB.

Do not change weights, fit lighting/exposure, select checkpoints, repair geometry
or tune light parameters. Preserve the oracle-assisted support and O1 provenance
qualifications. Retain unknown-light D2, matched Chart/Ambient, anisotropy-aware G
and independent cross-object confirmation as subsequent distinct research tasks.
