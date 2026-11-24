# Dense Bell reference v1 stopped on camera 127

The authorized run stopped after camera 127 used all 32 permitted
high-resolution refinements. It completed no new reference view. Camera
selection, geometry and the six admitted sparse references remain valid.
The dense package has no admission.

## What the saved arrays establish

| Camera 127 RGB convergence | 256² vs. 512² | After replaying the 32 patches | Required maximum |
| --- | ---: | ---: | ---: |
| Maximum absolute channel difference | .00427679 | .00147993 | .003 |
| Pooled relative L2 | .00306919 | .00208507 | .002 |
| Observations over the absolute bound | 3 | 0 | 0 |

The original deterministic selector replayed all 32 patches, including
the pooled-only selections. Its next pixel is raster index 7623
(column 71, row 59 in zero-based C order). Unoccluded reflectance and
both ordinary-angle checks passed throughout. The 32 largest remaining
unrefined RGB contributors have outgoing normal cosines .9758–.9996.
Their specular-residual L2 is 11.22 times their diffuse-residual L2.
This localizes the remaining convergence issue to nearly head-on
specular observations. The saved arrays do not isolate BRDF quadrature
from shadow-visibility integration as the cause of that discrepancy.

An optimistic calculation holding the assembled fine-RGB norm fixed
and setting each successive candidate's residual to zero first crosses
.002 after 10 more patches. That is a budget illustration. The actual
next-grid values and convergence of the remaining cameras are unknown.

The file-only audit verified the launch identity, all 58 recorded
output files, the exact sparse copies, archived geometry, native work
counters and cleanup receipt. The run consumed 1,249,443,840 shadow
directions, the same number of independent BRDF point-directions, and
270,417,920 attempted and returned slope nodes. It made zero new primary
rays or material optimizer updates. [Exact replay](analysis.json)

## Proposed separately reviewed continuation

Reuse the hash-verified camera 127 baseline and 32 patches in their
original order, and preserve the six sparse references and all 18 camera
identities. Continue with the same ranking and 1024²/2048² integration.
Allow up to 64 total patched observations per view, keeping the original
128-patch panel cap inclusive of the 32 saved patches. This gives at most
96 additional patches. The eleven unstarted views still use the original
baseline grids and reflectance checks. Preserve both convergence
thresholds and every observation. Stop on a failed mandatory gate or
cap exhaustion and keep the original failed batch read-only.

The upper bound for that continuation is 15,099,822,080 new shadow
directions and the same number of new independent BRDF evaluations,
3,649,126,400 slope nodes attempted and returned, and zero primary
rays. Reused work plus new work fits within the original panel-wide
scientific work ceiling. Freeze the amended cap, source, inputs and a
fresh output directory before seeking an explicit execution approval.
If a new gate fails, the next numerical alternative is to test a more
efficient visibility-weighted specular quadrature on a bounded set of
saved points. Training remains gated on complete reference verification
and separate researcher admission.
