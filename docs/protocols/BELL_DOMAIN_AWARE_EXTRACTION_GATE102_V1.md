# Gate 10.2 — Domain-aware raw-SDF extraction

Status: preparation for one separately authorized evaluation. The accepted 300,000-update
training history and successful Gate 10.1 diagnosis remain unchanged. Neither the
original failed mesh nor a Gate 10.2 candidate is admitted by this preparation.

## Inputs and extraction policy

Use Gate 10.1's hash-bound `raw-SDF-grid.npy` and `outside-sphere.npy`, captured
at the original 512³ nodes over [-1,1]³. The freeze binds both original files,
the accepted diagnostic receipt, acquisition, source and corrected BVH/image.
The raw grid is the original learned SDF before the sphere sentinel replacement.
Read it without querying a network. Make one writable C-contiguous float32 copy
and call the original PyMCubes 0.1.4 implementation once at level zero. Verify
that the copy's values remain unchanged. Save unfiltered voxel vertices, world
vertices and original face indices/dtypes. Convert coordinates by the original
`voxel / 511 * 2 - 1` rule.

Keep a face only when every possible generating cell has all eight corners
inside the captured domain mask. For triangle voxel-coordinate bounds [a,b],
the integer cell origin on each axis lies between `ceil(b-1)` and `floor(a)`,
intersected with [0,510]. This identifies one cell ordinarily and multiple
possible cells for exact grid-face/edge coincidences. Require every such cell
to be supported. Use exact comparisons, without a tolerance, centroid shortcut
or clipping of triangles. Save the number of possible/supported generators and
separate mixed-generator, boundary-cell and exterior-cell rejection reasons.
An impossible generating-cell inventory is a numerical failure.

This conservative rule needs neither positive exterior values nor an artificial
closing surface. Convexity of the sphere makes a cell with all corners inside
fully supported. It intentionally omits boundary-crossing cells, including
learned surfaces in their supported portions, and preserves the remaining open
boundaries and disconnected components. It does not claim that an SDF learned
outside the original domain is supported geometry.

The existing mask contains 69,861,728 inside nodes and 133,432,831 cells:
69,248,221 with all corners inside, 1,230,074 with mixed corners and 62,954,536
with all corners outside. The inventory's `fully_outside` label describes corner
classification; it does not prove that a curved sphere cannot intersect that
cell's interior. All such cells are conservatively unsupported either way.
The grid spacing is 2/511 = 0.0039138943. Any omitted supported-domain face due
to the conservative cell rule lies in a boundary band no deeper than the cell
diagonal, sqrt(3)*2/511 = 0.0067790638 scene units. These are domain counts and
a geometric bound, not surface-area coverage or a prediction of quality.
Save actual supported faces, areas, components and raster coverage at execution.
No oracle geometry or held-out outcome selects faces or components.

## Exact native-precision cleanup

Convert retained world coordinates once to float32, the actual downstream
Open3D/BVH coordinate precision. Weld only numerically equal coordinates under
exact float32 equality, with deterministic lexicographic indexing. Compute
cross products in float64 on these float32 coordinates; remove only faces with
exactly zero area, including repeated-index and coincident-coordinate cases.
Keep every positive-area face, however small, in its original winding and order.
Compact unused vertices and use native int32 indices. Verify each final triangle
equals its original retained triangle rounded to float32.

Save raw-vertex→candidate and candidate-face→raw mappings, exact-zero removed
face IDs, coordinate change and merge counts. Record edge-connected components,
per-component area/count, boundary edges, self/nonmanifold edges and inconsistent
two-face winding before and after cleanup. Exact welding can join index seams
at identical positions; it cannot bridge separated positions. Components made
entirely of exact degeneracies have no positive-area support and their removal
is explicit. There is no area threshold, component selection, smoothing, hole
filling, remeshing, normal flip or Trimesh tolerance-based processing.

Write a separate float32 binary PLY and require exact readback through both the
existing parser and original Open3D interface. The corrected native BVH and
independent float64 closest-hit checks are mandatory. A tiny positive-area face
that cannot produce a valid native normal remains a failure, not permission
to delete it. Compare the historical invalid mesh only using safe array-based
statistics and its positive-area indexed topology. Never submit that invalid
mesh to downstream BVH operations.

## Original geometry audit

Reuse the unchanged Gate 10 implementation for all six cameras 58, 11, 105,
114, 17 and 76, including original calibrated rays, signed normal convention,
oracle replay, closest-hit checks and full-raster exclusion inventories.
The original per-view quality requirements remain silhouette IoU >=0.90 and
valid oracle correspondence fraction >=0.80. Correspondence uses distance
<=0.02 and signed normal difference <=30 degrees, with the unchanged tight
and grazing cohorts. Original denominators include all oracle foreground;
boundary-support losses cannot improve the decision by being excluded.

Add evaluation-only conservative-domain flags at oracle hit positions and
report missing hits inside/outside this domain. These flags help distinguish
domain limitations from interior errors, but do not change rays, support,
correspondence acceptance or scientific thresholds. All mismatch and invalidity
counts remain explicit. Record depth, surface displacement and signed normals.

Only after the raster audit, construct the unchanged pinned G model and load
the exact accepted endpoint once, on CPU, solely for original consistency queries.
Checkpoint SHA-256:
`27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd`.
Require the accepted tensor catalog and fingerprint before querying. Use the
original deterministic area/rim/vertex/raster witness selection, at most 512
points. Query raw SDF and first derivatives, then inward/outward values, at most
1,536 SDF evaluations and 512 first-derivative points. Retain the original
1/511 distance-proxy/offset bound, gradient minimum 1e-6, sign margin 1e-7,
minimum 128 eligible interior and 32 oriented interior samples, and zero
permitted consistency, unsupported-rim or signed-orientation failures.
No sentinel values substitute for raw SDF. Verify neural state unchanged.

Execution completion and geometry admission are separate. Preserve a complete
audit even when original quality criteria fail. A numerical/identity/work
failure stops immediately. Create an authenticated, derivation-bound G→A handoff
candidate only after all original quality criteria and offline replay pass;
researcher admission and material execution remain separate decisions.

## One bounded execution

The committed freeze binds the fresh evidence directory, all input hashes,
source identities, policies, limits and original quality definitions. One
explicit commit/freeze/output-bound approval covers the saved-grid CPU extraction,
corrected GPU BVH audit, conditional single CPU checkpoint deserialization,
bounded SDF consistency queries and subsequent file-only verification/reporting.
It does not authorize training, a second extraction, an alternative policy or
a retry. The launcher refuses an existing output, dirty source, wrong image,
occupied GPU or active prior geometry/diagnostic container.

- Corrected image: `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`.
- Authoritative G: `79252178e65914ad1acbc14790b1514f276bfea0`, unchanged.
- One A100 80GB PCIe GPU0; four CPU threads, 128 GiB RAM/no swap, 128 PIDs,
  75% CUDA allocator fraction, 8 GiB owned output cap.
- One MCubes call over the saved 512³ grid, maximum 4,000,000 raw/candidate faces;
  zero extraction-network queries. At most one accepted endpoint CPU load.
- Exactly 196,608 primary BVH rays and 384 independent CPU ray witnesses when
  the full audit completes; no incident/shadow rays, shader calls or updates.
- Foreground 1,800-second watchdog/30-second grace; inner 1,740-second timeout.
  Estimated 1–10 minutes, unmeasured for this policy. Stop without extension.
- Non-root, read-only source/inputs/root, no network/build/install. Only owned
  outputs and the existing runtime's local cache are writable. No image rebuild.

After approval, bind `authorization(freeze, accepted_commit, freeze_sha256)` to
the external authorization JSON and use the existing foreground workflow:

```bash
env \
  BELL_DOMAIN_SOURCE=/home/isavand/paper/Nerofactor-donut-recovery \
  BELL_DOMAIN_AUTH=/home/isavand/paper/evidence/APPROVED-BATCH-authorization.json \
  bash /home/isavand/paper/Nerofactor-donut-recovery/experiments/run_bell_geometry_gate102.sh
```

The actual output and authorization identities come from the freeze and explicit
approval, not the illustrative placeholder. No authorization file is created by
preparation. The launcher records image identity, exit status and owned-container
removal; the offline verifier replays domain, cleanup, topology, raster and SDF
decisions and checks artifact identities without further extraction/model calls.

## Interpretation and backlog

Removing sentinel crossings changes extraction policy, not the accepted learned
weights. A raw zero crossing alone does not establish accurate learned geometry.
Open-boundary support loss, interior surface error, normal error and independent
ray/SDF failures must be distinguished using the saved evidence. Do not infer
isotropic-appearance causality from these measurements alone.

If quality fails, preserve the candidate and recommend the smallest intervention
supported by the failure. Do not silently relax thresholds or modify the surface.
Retain foreground-only mask loss, background-NeRF interaction, effective loss
schedules and anisotropy-aware geometry as future investigations. Retain the
15k/30k/50k/300k checkpoint geometry comparison as a separate, no-new-optimization
follow-up; intermediate held-out results cannot replace the fixed 300k endpoint.
