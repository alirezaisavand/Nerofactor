# Heterogeneous Donut Kurt reference v1

This benchmark tests material recovery when real roughness boundaries coexist
with smooth variation and nearly isotropic surfaces. It uses the admitted
Donut oracle geometry, smooth shading normals, NeRF camera calibration and
six views at 128×128 from the existing reference. Fitting cameras are
0, 50, 1 and 45. Held-out cameras are 28 and 34. Reuse archived primary
intersections, masks, triangle IDs, barycentrics and ordered camera rays.
Their input hashes bind the original admission. No primary ray tracing is
needed. The new package contains fresh material labels and integrated
radiance under the unchanged six-lobe L2 environment, zero indirect
illumination and no point light.

## Material field

Use native world coordinates centered at zero with torus axis +Z. Let
`u` be azimuth and `r` cylindrical radius. Set `R=(min(r)+max(r))/2`
on the oracle vertices and `v=atan2(z,r-R)`.
Switch width families where `cos(2u)` changes sign. The four interfaces
cross the tube's meridians. Use region ID 0 for the nonnegative side
and 1 for the negative side.

| Material quantity | Definition |
| --- | --- |
| Width family 0 | major 0.42, minor 0.17 |
| Width family 1 | major 0.30, minor 0.112 |
| Width variation in both families | add `0.035 sin(u+v)` and `0.016 cos(2u−v)`, respectively |
| Principal director | project the azimuthal unit tangent into the saved shading plane, rotate about the shading normal by `0.40 sin(u+v) + 0.28 cos(2u−v)` radians |
| Linear base color | `(0.20+0.05 sin(u−v), 0.58+0.10 cos(v), 0.13+0.04 cos(u+v))` |
| Metallicity | `0.68+0.08 sin(u−v)` |

The weak-anisotropy patch lies inside region 0. Blend both widths toward
0.223 and 0.214 with the product of two clamped cubic smoothsteps,
`smoothstep((cos(u)−0.90)/0.075)` and
`smoothstep((cos(v−0.5)−0.35)/0.45)`. The principal director remains
continuous in the physical shading plane up to its sign equivalence.
Both scalar materials are continuous across every width boundary. The
width jumps are 0.12 and 0.058. Widths stay within the supported interval
[0.04, 1]. The definition uses geometry and fixed analytic functions
without depending on any method's chart or tensor representation.

Export each point's director, world Q, sorted widths, linear base color,
metallicity, region ID and weak-patch blend weight. The boundary distance
is the cylindrical azimuthal arc length to the nearest width seam,
`r abs(asin(cos(2u)))/2`. The boundary band is 0.035 native units.
The nearly isotropic stratum uses the squared-width gap
`(major²−minor²)/(major²+minor²) < 0.1`. Treat the director as axial
and require both truth and prediction to pass the gap criterion for
orientation error. Report eligibility counts.

## Generation and verification

Use the existing independent explicit-axis Kurt/Beckmann kernel and 256²
and 512² float64 solid-angle midpoint grids for RGB. Use the admitted E.11C
BVH for shadow visibility with the original Donut 1e−5 ray offset and
depth ≥10 miss rule. Reuse the analytic Lambertian allocation and
exact-support Beckmann slope quadrature at orders 64 and 128 for
unoccluded directional reflectance. Check both RGB and slope reflectance
on every valid observation at the unchanged 0.003 maximum channel and
0.002 relative L2 bounds. For each camera, cross-check slope and
independent incident reflectance at eight fixed ordered observations with
view cosine ≥0.25. Those values come from directions already evaluated
for RGB. Preserve every original grazing observation and keep zero RGB on
below-horizon geometry hits.

Save fine and coarse unclipped linear diffuse, specular and total RGB,
both reflectance solutions, visibility counts, highlight masks and mesh
triangle areas. Store sRGB previews with explicit display-only upper
clipping and channel counts. A mandatory failure ends the one launch
and preserves arrays already written. The file-only verifier hashes
outputs, replays labels, masks, registration, radiometric component sums,
preview conversion and convergence. Scientific validity is limited to
the verified acquisition and the stated deterministic quadrature checks.

The exact panel contains 22,630 shader-valid observations. Generation
charges 7,415,398,400 shadow queries and the same number of independent
BRDF point-directions, plus 1,853,849,600 slope quadrature nodes.
There are zero primary BVH queries, model calls, checkpoint loads or
optimizer updates. Use the reviewed corrected image on A100 GPU 0 with
four CPU threads, 64 GiB RAM with no swap, 128 PIDs, a 25% CUDA allocator
limit and 512 MiB of outputs. Use a 7,200-second foreground watchdog
with 30-second kill grace and an inner 7,080-second limit with 20-second
grace. Run non-root and networkless with read-only source and inputs.
No retries, changed tolerances or adaptive material selection.

A hash-bound generation approval covers this dataset only. Researcher
dataset admission follows saved-data verification. The four training
arms require a separate admitted-dataset authorization.

## Matched comparison after admission

| Arm | Existing method retained |
| --- | --- |
| D2 | Corrected-core chart predictor and original legacy priors |
| D6 | D2 plus the accepted 0.05 normal-transported raw-Q penalty |
| D8-C | D8 chart predictor with matched width-0.25 isotropic start, tensor shader and sampler |
| D8-T | D8 ambient predictor with the same physical start, tensor shader and sampler |

Retain each method's accepted seed, optimizer schedule, full network,
direct-light learning, loss weights, visibility and 10,000 Adam returns.
Preserve D8's common Gaussian tapes, D6 neighborhood stream and
identical D8 initialization/legacy scalar perturbation policy. Report
representation and initial width differences between D2/D6 and the
D8 pair. As in the homogeneous experiment, D6 and the D8 arms sample
neighborhoods based on geometry alone, including pairs across true
material boundaries. Region labels, GT material values, boundary
locations and held-out RGB stay outside the optimization objective and
neighborhood selection. Retain the existing no-gradient reference
compatibility diagnostic, with each probe's admitted heterogeneous
material. Save 0, 2,000, 4,000, 6,000, 8,000 and 10,000 update maps
and keep 10,000 as the primary endpoint.

Score relative world-Q Frobenius error, sorted width MAE and relative
error, axial orientation, eligibility, anisotropy gap, saturation and
scalar errors directly against each point's reference labels. Report
fitting and held-out linear RGB, illumination and gradients/time.
Break results out by boundary band, smooth interiors, near-isotropy
and fitting-view geometry/highlight support. Use the established 0.01
native-unit and 30° normal compatibility coverage test and the
admitted highlight convention. Report cohort counts and visible area.
For an area-weighted score, divide each observed triangle's area
among its observations within a split, then aggregate. Report observed
triangle area coverage separately. This is a visible-surface area proxy
and leaves unsampled surface error undefined.

For boundary preservation, compare normal-transported Q jumps and
signed sorted-width jumps in opposite-region four-neighbor image pairs.
Use same or edge-neighbor mesh faces, position separation ≤0.02,
and normal separation ≤30°. Use GT region IDs to identify endpoint
pairs during evaluation only. Compare jump magnitude and signed
agreement against labels, and report errors on each side by distance
to the interface. Oversmoothing means attenuation of the true jumps
with associated boundary-band recovery errors. It cannot be inferred
from a small global smoothness residual.
