# Heterogeneous Bell dense acquisition

Extend the admitted heterogeneous Bell benchmark from four to sixteen fitting
cameras. Preserve the geometry, calibration, 128 × 128 resolution, half-pixel
convention, shading normals, heterogeneous material field, analytic L2
illumination, additive Kurt/Beckmann BRDF and linear radiometry. Indirect
illumination remains zero and the acquisition uses no point light.
[Admitted sparse reference](../results/bell_heterogeneous_reference_v1/README.md) ·
[Accepted known-light controls](../results/bell_known_l2_recovery_v1/README.md)

## Selected views and geometric preflight

Start with fitting cameras 0, 4, 19, 2. From the original 122 training IDs,
greedily select the camera whose unit direction from the acquisition origin
maximizes its minimum angular distance to the fitting cameras already chosen.
Compute camera centers as `−Rᵀt` from the admitted numeric world-to-camera poses.
Sort candidate IDs before scoring to break exact ties in favor of the lower ID.
The validation camera 58 and original NVS cameras 11, 105, 114, 17, 76 are
excluded throughout. Hold out cameras 105 and 114 in their original order.
No material, RGB, highlight or recovery measurements enter selection.

The added cameras, in order, are **127, 71, 73, 56, 75, 95, 93, 110, 91, 7, 5, 3**.
Their respective hit counts are 3,301, 4,025, 4,087, 4,195, 4,022, 3,813,
3,822, 3,556, 3,891, 4,375, 4,418 and 4,341. The angle to the nearest already
selected fitting camera varies from 63.54° for the first to 25.53° for the last.
File-only preflight replays the saved primary-ray geometry gates, hit masks
and normalized negated BVH normals. It attributes saved hits to their
original PLY faces with the accepted plane/edge/normal checks.

| Geometric measure | Sparse | Dense |
| --- | ---: | ---: |
| Fitting observations | 17,573 | 65,419 |
| Fraction of mesh area on unique fitted triangles | 9.43% | 27.79% |
| Held-out observations with zero nearby fitting cameras | 586 | 8 |
| Held-out observations with at least two nearby fitting cameras | 2,671 | 7,054 |

For each fixed held-out observation, count fitting cameras with a sample
within .01 native units and normal agreement within 30°. A camera counts
as a highlight observation if any compatible neighbor meets the existing
reference highlight rule. Angular spread is the maximum pairwise angle
between outgoing directions from the held-out point to those camera centers.
It is a nearby-sample coverage proxy. It does not claim that the held-out
point itself is visible in every counted view. Save pointwise sparse and
dense coverage, and report angular gains on the common subset with at
least two sparse fitting directions. Highlight coverage is measured after
generation.

## Reference generation and independent replay

The sparse reference NPZs and diffuse, specular and RGB previews for all
six original cameras are copied byte-for-byte into the new package. Bind
the admitted generation, researcher admission, offline verification and
original artifact hashes. Keep the original archives read-only. Only the
12 appended views enter rendering or quadrature.

Every new view uses the existing independent float64 incident 256²/512²
solid-angle quadrature and corrected BVH for shadow visibility. Reuse
Beckmann slope quadrature at orders 64 and 128 for unoccluded reflectance,
with eight ordinary-angle incident crosschecks per view. Preserve the
0.003 absolute and 0.002 pooled-relative convergence bounds. Reuse the
accepted deterministic refinement selection, including pooled-only
failures, with at most one 1024²/2048² refinement per pixel, 32 points
per new view and 128 across the new panel. Preserve baseline and refined
arrays, component sums, visibility counts, quadrature grids and preview
clip counts. Stop on a failed mandatory gate or exhausted budget.
The offline replay verifies the full new artifact inventory, reused hashes,
input identities, selected pixels, work counters, final masks and decisions.
Only the previously reviewed one-ULP whitelist applies to non-gating
highlight summary scalars. A passing replay remains pending researcher
admission.

| New reference work | Maximum |
| --- | ---: |
| Primary rays and CPU primary intersection witnesses | 0 |
| Shadow directions and independent BRDF point-directions, each | 16,349,265,920 |
| Slope nodes attempted and returned, each | 3,919,544,320 |
| Neural/native material calls and optimizer updates | 0 |

Use the existing corrected image on A100 GPU 0. Use four CPU threads,
64 GiB host RAM without swap, 128 PIDs, a 25% CUDA allocator fraction and
a 512 MiB output budget. The launcher uses a foreground 7,200-second timeout
with 30 seconds of kill grace and an inner 7,080-second timeout with 20
seconds of kill grace. Preserve the read-only root, input and source
mounts, non-root UID, disabled networking and fresh output directory.
This is one launch with no retries. Reference generation requires a
commit/freeze/path-bound approval. Neither that approval nor a passing
replay authorizes training.

## Matched recovery after separate admission

K1-Dense, K2-Dense and K3-Dense inherit K1, K2 and K3 respectively at
the accepted source and use the same 10,000-update endpoint. Only the
ordered fitting acquisition changes. Retain the exact initial material
fields, known L2 and zero indirect, renderer, losses and coefficients,
mesh domain, boundary probes, sampler policies, Adam schedule and 512
observations per update. Training observations contain positions, normals,
outgoing directions and linear RGB only. The D8 pair uses its shared
renderer, draws, matching photometric sample indices and physical
initialization. The same train-only ordering and sampled-index sequence
must be verified for both dense tensor arms. Freeze the eventual native
adapter against the verified and separately admitted dataset before
requesting a single three-history execution.

The new fitting pool contains 3.72 times as many observations. Each
arm still consumes 5,120,000 fitting slots, equivalent to 78.26 pool
passes versus 291.36 on the sparse acquisition. The existing 512-wide
shuffle drops the incomplete tail at each reshuffle. The dense pool
has 127 complete batches and a 395-observation tail per permutation
versus 34 batches and 165 observations for sparse. Report actual
exposure separately from evaluation cost. The six evaluation states
at 0, 2,000, 4,000, 6,000, 8,000 and 10,000 updates use 7,152 evaluation
shader calls per dense arm, including the unchanged held-out views.

Score endpoints both by observation and observed-triangle area. Report
physical Q, sorted width errors, orientation on common eligible
observations, anisotropy gap, scalar materials, fixed-probe boundary
jumps and linear RGB. Compute fitting errors on the full designated pools
and additionally on the original four fitting views for a matched
comparison. Keep both the original sparse geometry/highlight strata and
newly computed dense strata on the exact same held-out observation
indices. Pay particular attention to formerly unsupported and still
weak-highlight regions. Increased angular coverage, added observations
and lower per-observation exposure are simultaneous consequences of the
acquisition intervention under a fixed optimizer budget.
