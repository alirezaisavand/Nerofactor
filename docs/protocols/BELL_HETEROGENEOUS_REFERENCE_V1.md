# Heterogeneous Bell reference v1

Use the admitted E.11 v3 mesh, numeric calibrations and primary hit arrays.
Recompute the geometry-only selection in the
[generalization proposal](BELL_HETEROGENEOUS_GENERALIZATION_PROPOSAL.md).
The ordered fitting panel is 0, 4, 19, 2 and the held-out panel is 105, 114.
All cameras retain 128 × 128 resolution, original world units, world-to-camera
pose, scaled intrinsics, pixel center (+0.5) and original translation.
The six views provide 17,573 fitting and 7,082 held-out observations.

Bind the original generation and researcher admission, completion seal, exact
mesh and calibration, six ray archives and six matching masks. Replay the
geometry checks and independent pinhole ray construction on saved arrays.
Retain the archived normalized, negated BVH face normal at each hit.
The archive lacks face IDs. Attribute a saved point to its original PLY
face using the plane distance, three signed edge distances and agreement
with the raw BVH normal. Candidate inclusion allows 1e−6 native units
and raw normal dot at least (1-2	imes10^{-6}). Lowest plane distance
then original face index resolves tolerance-level ties, which are recorded.
Positions, hit masks and normals remain intact. This attribution uses
zero new primary rays. The original source-silhouette IoUs are admitted
from the v3 receipt, and its saved masks are checked against the hit arrays.

## Geometry-defined material

Let (u=(x,y,z)=2(p-mathrm{midpoint})/(mathrm{max}-mathrm{min}))
using the Bell mesh bounds componentwise. Region 1 is where
((x+0.30z+0.10)(y-0.25z-0.08)<0), and region 0 covers the rest.
The starting width pairs are ((0.42,0.17)) in region 0 and
((0.30,0.112)) in region 1. Both regions add
((0.035sin(pi(x+0.4y)),0.016cos(pi(z-0.35y)))). The two oblique
planes create real width steps where the anisotropy taper is inactive.
The boundary band is 0.035 native units from either plane, using the
world-space plane distances.

The isotropic cap blends both widths toward 0.22 by cubic smoothstep
of ((x+0.35z-0.58)/0.28), clamped outside ([0,1]).
The principal director projects normalized world grain
((0.35,0.80,0.48)) into the archived tangent plane. A second smoothstep
from projection norm 0.05 to 0.20 scales the squared-width anisotropic
deviator at fixed trace. Below that interval the tensor is exactly
isotropic. A unit tangent from the existing X/Y basis convention is
stored there to evaluate the isotropic kernel. Orientation is undefined
at exact isotropy and eligible for error measurement only when the
squared-width gap ((lambda_{max}-lambda_{min})/
(lambda_{max}+lambda_{min})) reaches 0.1.
The cap can also attenuate the geometric width steps, so later boundary
scores use the actual oracle jumps at the paired points.

Linear color is
((0.20+0.05sin(pi(x-y)), 0.58+0.10cos(pi z),+0.13+0.04cos(pi(x+y)))).
Metallicity is (0.68+0.08sin(pi(y-z))).
These are continuous across both width boundaries. The only field
inputs are mesh bounds, position and the actual local shading normal.
The normal changes at facet edges, so the world director is defined
per tangent plane with the physical tensor as its reference.

## Independent integration and bounded refinement

Use the heterogeneous Donut L2 illumination with zero indirect
light and no point light. Reuse the exact float64 midpoint solid-angle
incident quadrature, corrected BVH shadow query and independent
axis-form Kurt/Beckmann kernel. Integrate diffuse and specular components
at 256² and 512² directions for all 24,655 retained hits. Independently
integrate unoccluded reflectance in Beckmann slope space at orders
128 and 256, retaining the 8 prescribed ordinary-angle incident
crosschecks per view. Reflectance is descriptive under the additive
BRDF. Preserve unclipped linear data, with display-only PNG upper
clipping and recorded counts.

Reuse the Donut v2 adaptive RGB rule with a fresh baseline for each
view. Check the baseline reflectance controls before refinement.
If the full-view RGB gate fails, rank absolute channel offenders by
their largest channel residual. For a pooled-only failure, rank
unrefined pixels by their squared RGB residual. Lower pixel index
breaks ties. Evaluate each chosen pixel once at 1024² and 2048²,
retain its original and improved components and visibility, and
rerun the full-view gate. Stop on a refined absolute failure,
a 32-pixel per-view or 128-pixel panel budget exhaustion, or any
remaining final gate failure. There is no retry or observation
removal. Unchanged bounds are 0.003 maximum channel difference and
0.002 pooled relative L2, including the existing reflectance controls.
Two-grid agreement is a numerical acceptance check for this reference.

| Work | Maximum |
| --- | ---: |
| New primary rays and CPU intersection witnesses | 0 |
| Baseline shadow directions and BRDF point-directions, each | 8,078,950,400 |
| Shadow directions and BRDF point-directions with refinement, each | 8,750,039,040 |
| Slope nodes, attempted and returned | 2,019,737,600 each |
| Native material model calls and Adam returns | 0 |

One authorized foreground run uses the existing corrected image on
A100 GPU 0, four CPU threads, 64 GiB RAM without swap, 128 PIDs, 25%
CUDA allocator cap and 512 MiB output guard. The foreground timeout
is 7,200 seconds with 30 seconds of kill grace, and the inner timeout
is 7,080 seconds with 20 seconds of kill grace. Mount source and inputs
read-only, use a non-root UID and disable networking. The freeze binds
the source/input identities and a fresh output. Keep all prior datasets
and any incomplete new evidence.

## Geometry transport and admission

Reuse D6's geometry-only rule for prospective spatial arms. Pair
centroids across actual two-face manifold edges, with edge-midpoint
surface distance in ([10^{-6},0.02]) native units and dihedral at
most 15°. Both normals are negated PLY face normals, consistent
with the admitted A shading convention. Use the existing minimal
normal rotation and squared distance normalization. Reserve the
same 1,024 evenly indexed diagnostics and report the remaining
domain in faces and area, including fits with no nearby observation.
Boundary and material labels stay outside neighborhood selection.
The 362,520 eligible pairs cover 99.87% of the mesh by triangle area.
After reserving diagnostics the training domain covers 248,612 of
248,908 faces and 99.87% of area. Fitting views observe 9.43% by
unique triangle area and held-out views 3.84%. These are preflight
domain statistics. Eligible distances span 8.66e−6 to 6.51e−3
native units, with maximum dihedral 14.995°. The short-edge tail
can amplify the unchanged squared distance normalization. Training
will measure actual sampled coverage and gradient scales.
Area-weighted recovery allocates each observed triangle's area
across its observations within a split. No physical error is
available for unobserved triangles.

The offline verifier rechecks the terminal receipt, hashes, baseline
and refinement order, exact assembled arrays, component sums, PNGs,
highlight mask, convergence decisions and charged work. Only after
successful replay can the researcher admit the new benchmark by
hash. The existing weak-cue Bell admission applies to the earlier
benchmark; it supplies the geometry for this reference.

After separate admission, bind D2, D6, D8-C and D8-T to the six-view
linear acquisition with fresh accepted initialization and 10,000
updates per arm. Preserve their priors, optimizers and checkpoints at
2,000-step intervals. Recompute the Bell sampler inventory and retain
the matched tensor renderer, objective and tapes in the D8 pair.
Score exactly isotropic points for RGB, Q and width, and exclude
undefined orientation from angular error. Report common orientation
subsets, boundary attenuation, surface and highlight coverage and
Bell/Donut endpoint comparisons. Training requires its own execution
approval after dataset admission.
