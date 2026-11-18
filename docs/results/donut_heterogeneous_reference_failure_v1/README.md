# Heterogeneous Donut v1 reference generation stopped at camera 50

The approved launch exited 1 after camera 50's RGB convergence gate.
Camera 0 passed. Camera 50's unoccluded slope reflectance and its
ordinary-angle cross-checks passed. The launch saved both camera
archives before stopping. Cameras 1, 45, 28 and 34 remain unrendered.
The package is ineligible for admission, and training remains pending.

## Saved-data replay

| Check at camera 50 | Maximum channel difference | Relative L2 | Decision |
| --- | ---: | ---: | --- |
| 256² versus 512² RGB | 0.00698958 | 0.00242652 | Fail |
| Order 64 versus 128 slope reflectance | 0.00000251 | 0.00000185 | Pass |
| Ordinary-angle incident reflectance, 256² versus 512² | 0.00000310 | 0.00000356 | Pass |
| Ordinary-angle slope versus fine incident reflectance | 0.00000076 | 0.00000089 | Pass |

Replaying the saved arrays reproduces every decision with the unchanged
0.003 maximum channel and 0.002 relative L2 limits. The archived geometry
and analytic material labels independently match for both saved views.

Only pixel **(21, 57)**, index 7,317 on triangle 13,099, exceeds the
absolute limit. Its view cosine is 0.00552 (about 89.68° from the
normal) and its sorted widths are 0.26813 and 0.10483. The point lies
outside the boundary band and the weak-anisotropy patch. It contributes
96.38% of camera 50's squared fine/coarse RGB difference. The largest
channel discrepancy is almost entirely specular. For localization
only, the other pixels have maximum difference 0.000693 and relative
L2 difference 0.000462. The witness remains required for admission.

At the witness, the independent incident-grid **unoccluded**
reflectance differs by up to 0.004451 between grids. The two
slope-space reflectance values differ by about 5.1 × 10⁻⁷. This supports
grazing-angle under-resolution in incident angular quadrature. Shadow
visibility cannot explain the unoccluded discrepancy. The fine-grid
visible count is 258,802 of 262,144, identical to the homogeneous
reference at the same pixel. That count cannot establish that all
visibility decisions or their radiance-weighted contributions agree.

## Smallest follow-up

Under separate approval, run the unchanged RGB integrand at 1024² and
2048² on the witness and two saved controls. Use camera 50's next
largest-residual pixel (19, 62) and its first ordinary probe at index
6,577. Use the archived 256² and 512² estimates as context and save
separate diffuse, specular, unoccluded-reflectance and visibility
diagnostics. Two grids over three fixed points cost 15,728,640 shadow
queries and the same BRDF point-directions. This would test whether
higher angular resolution stabilizes the grazing result before any
revised full-panel generation. If the two grids still disagree, a
visibility-weighted exact-support slope quadrature for specular
radiance is a numerical candidate for review.

Preserve the field, geometry, views, all grazing points and both
convergence thresholds. Keep camera 0's passing arrays and all failed
evidence intact. A revised generation needs its own freeze and approval.
No rendering, optimizer work or checkpoint loads were performed for
this diagnosis.

The original receipt records 2,350,120,960 shadow and BRDF point-directions
and 587,530,240 slope nodes, with zero primary BVH queries. It does not
attest successful cleanup after the exception. The original exit and
receipt are preserved with all input and output hashes used here.
[Machine-readable replay](analysis.json)
