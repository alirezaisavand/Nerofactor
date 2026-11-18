# Heterogeneous Donut reference convergence repair v2

The v1 run stopped when camera 50's 256²/512² RGB comparison exceeded
both unchanged convergence limits, 0.003 maximum channel difference and
0.002 relative L2. Pixel (21, 57) accounts for 96.4% of the squared
difference and shows the same under-resolution in unoccluded incident
reflectance. The heterogeneous material, six 128×128 cameras, geometry,
shading domain, L2 illumination, Kurt BRDF, visibility convention and
reflectance synthesis remain the v1 definitions.
[V1 failure replay](../results/donut_heterogeneous_reference_failure_v1/README.md)

## Mandatory diagnostic and conditional completion

In the single authorized launch, first evaluate camera 50 pixels
7,317, 7,955 and 6,577 (witness, next-largest residual and first
ordinary control) using the unchanged incident midpoint kernel at
1024² and 2048². Save fine and coarse diffuse, specular and combined
RGB, unoccluded incident reflectance and visibility counts for each
point. Require the same maximum channel and relative L2 limits on the
combined three-point RGB panel before proceeding. Component and
incident-reflectance convergence measurements in that diagnostic are
recorded separately. The full-view unoccluded slope and ordinary-angle
controls retain the v1 requirements.

On a diagnostic pass, byte-copy camera 0's arrays and previews.
Its highlight mask and complete-view convergence are replayed. Use
camera 50's v1 archive as the baseline for that camera. The other four
views (1, 45, 28 and 34) first receive the original 256² and 512² RGB
and order-64 and order-128 slope integrations. Complete baseline arrays
are saved before selection. A baseline reflectance-control failure
stops execution and preserves measurements.

For each view, test the current assembled RGB arrays over all
shader-valid observations. If any channel exceeds 0.003 at an unrefined
point, refine the point with the largest such difference. When only the
pooled 0.002 relative L2 gate fails, refine the unrefined point with the
largest squared sum of RGB differences. Ties in either rank use the
lowest source pixel ID. Evaluate the chosen point once at 1024² and
2048², retaining both estimates. A selection from the diagnostic panel
reuses its recorded high-grid pair. Replace the coarse/fine RGB and
incident-reflectance pair and visibility counts at that point alone.
Slope reflectance and all material and geometry labels stay fixed.
Then recompute the two tests over the entire valid view. Stop refining
a view as soon as both pass. An already-refined absolute failure,
unresolved pooled failure with no remaining candidates, or a work cap
stops the batch. The final view must also pass its original slope and
ordinary-angle cross-checks.

Allow at most 32 refined points per view and 128 over the panel,
including reused diagnostic points selected for the final assembly.
The ceiling charges up to 128 additional high-grid points plus the
three mandatory diagnostic points. No observations are discarded,
including grazing or below-horizon hits, and valid masks are unchanged.
If the diagnostic fails or RGB remains unresolved after the allowance,
preserve the failure. A subsequent numerical alternative is an
independent visibility-weighted exact-support slope quadrature for
specular RGB. That alternative requires a separate freeze and approval.

## Evidence and launch limits

Each final view archive retains its linear RGB components, unoccluded
reflectance arrays, visibility counts, geometry and heterogeneous
material labels. Camera 0 and camera 50's baseline archives and the
shared surface-area array retain their v1 byte identities. Separate
baseline and patch archives preserve both original and higher-grid
estimates. Per-view quadrature sidecars record pixel order, coarse and
fine angular grid side lengths, coarse visibility counts and ordered
refined pixel IDs. A grid of zero marks an original shader-invalid
geometry hit. A coarse count of −1 marks a valid reused v1 point
whose 256² count was never archived. No extra queries fill that gap.
Recompute final highlight masks and encode sRGB previews with the
original explicit display-only clipping convention.

One exact-source approval binds the failed evidence and the fresh
output batch. Runtime limits remain 7,200 seconds in the foreground
with 30 seconds kill grace, and 7,080 seconds internally with 20
seconds grace. Use the reviewed corrected image on A100 GPU 0 with
four CPU threads, 64 GiB RAM with no swap, 128 PIDs, a 25% CUDA
allocator limit and 512 MiB output cap. Source, root filesystem and
inputs are read-only. The runtime is non-root and networkless.

| New work | Maximum |
| --- | ---: |
| Three diagnostic points at 1024² and 2048² | 15,728,640 shadow and BRDF directions each |
| Four remaining baseline views | 5,065,277,440 shadow and BRDF directions each |
| Additional high-grid refinement | 671,088,640 shadow and BRDF directions each |
| Total | 5,752,094,720 shadow and BRDF directions each |
| Slope integrations on the four remaining baseline views | 1,266,319,360 nodes |
| Primary BVH queries, model calls and optimizer returns | 0 |

The old v1 batch remains intact. On completion the saved-data verifier
rehashes inputs and outputs, checks source geometry and labels,
reconstructs the decision sequence and resolution assignments, and
replays convergence, component sums, highlights, RGB previews and the
component sheet. The technical decision stays pending researcher
admission. The D2, D6, D8-C and D8-T protocol and Bell proposal remain
available for later review. Neither training nor Bell generation
falls within this authorization.
