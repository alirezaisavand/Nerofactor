# Donut fixed-material geometry RGB floor

**Proposed. Requires a separate implementation freeze and explicit execution approval.** Measure how much the images change when the validated historical reconstruction replaces the oracle geometry. Use the unchanged heterogeneous Donut material field, known analytic L2 environment, zero indirect illumination and no point light. This comparison precedes any reconstructed-geometry material optimization. The historical reconstruction remains provenance-unverified. [Completed geometry gate](../results/publication_gate2/README.md)

## Paired geometry and material inputs

Bind both meshes, camera calibration, admitted heterogeneous Donut v2 package and all 18 Gate 2 products through the [Gate 2 manifest](../results/publication_gate2/artifact-manifest.json). The Gate 2 execution receipt SHA-256 is `5c99f7f93fa4f87fb3a8a5630ed2846f217fb9d1512131d0b24a622de0b978e2` and the verified geometry metrics SHA-256 is `9c24b45f5e4810506ac24866fd4550385ac7622633350f3ed3ff7dea15297667`. Use the same row-major 128 × 128 cameras, fitting 0/50/1/45 and held out 28/34. Reuse the saved Gate 2 first-hit positions, masks and normals. Verify compatibility with the admitted original camera directions and retain the original float64 outgoing directions for BRDF evaluation. No new primary rays or geometry edits are required.

- **Oracle geometry.** Use the admitted heterogeneous Donut labels at the oracle hit and the accepted smooth normals. Use the established zero contribution for the nine oracle hits below or on the shading horizon. Render all 22,630 positive-facing oracle hits.
- **Historical reconstruction.** Use the 22,520 valid Gate 2 transfers at the reconstructed hits. Color, metallicity and sorted widths come from the unchanged oracle field at the accepted local correspondence. Use its recorded transported principal director and Q, with the reconstructed shading normal. Check that the renderer's director and widths reconstruct the transported tensor. Keep the 61 rejected transfers as undefined radiance and keep their reasons. All 20 valid reconstruction-only transfers are included.

This pair fixes the physical material field and illumination. It measures the combined effect of geometry, normal frame, visibility and sampling different material points where a reconstructed surface lands on another part of the oracle field. Gate 2 records sampling and normal-frame changes separately in Q. Causal separation of their RGB effects would require a separate intervention.

## Independent integration and gates

Reuse the accepted float64 incident-hemisphere midpoint integrator and Kurt BRDF. Build the corrected FP64-predicate BVH on each immutable original mesh. Recompute binary incident visibility for every integrated direction on the corresponding mesh, using the admitted original offset and hit convention. Use the original local quadrature rules at 256² and 512² per point on both arms. Verify finite, nonnegative unmodified linear RGB and the diffuse-plus-specular component sum. Require each mesh and camera to meet the unchanged `0.003` maximum channel-difference and `0.002` pooled relative-L2 limits. Check the converged fresh oracle-geometry images against the admitted independently integrated oracle images under the same bounds. Save raw linear components, visibility counts and grid assignments. Display previews may clip with the clipped-pixel counts recorded. [Accepted independent reference integration](DONUT_HETEROGENEOUS_REFERENCE_V2.md)

If a per-view integration fails convergence, reuse the accepted error-ranked refinement rule. While either mesh exceeds the absolute bound, rank its unrefined pixels by largest channel difference across the failing meshes. When absolute gates pass and a pooled gate fails, rank by largest squared RGB residual among meshes with a failing pooled gate. Break ties by lower source pixel ID and then oracle before reconstruction. Evaluate the selected camera/pixel at 1024² and 2048² on both meshes wherever the radiance is defined and the hit is positive-facing. Keep both baseline and refined estimates. Allow at most 32 distinct refined pixels per camera and 128 over the panel. Stop on any unresolved numerical failure or exhausted work cap. Compute the coarse-to-fine change in the *between-mesh RGB difference* on the same known support and report it in absolute units and normalized by oracle radiance energy. This quantifies integration uncertainty even when the between-mesh difference is small.

## Report and interpretation

Report per camera and split, with linear RGB relative L2, absolute RMSE and signed luminance difference. Include separate diffuse and specular errors and fixed-scale RGB and error maps. Report common valid foreground, oracle-only pixels with a known reconstruction miss, valid reconstruction-only pixels with a known oracle miss, and the full raster's known support with its denominator. Include background as zero on a true miss. Keep rejected transfers explicitly undefined and show the fraction of the raster and oracle radiance energy they represent. For an all-raster squared-error lower bound, retain the full oracle denominator and count only measured error in the numerator. Do not report a single completed full-raster RGB score when pixels remain undefined.

Keep the fitting/held-out split unchanged and report boundary-band, grazing, degenerate-adjacent and large same-ray depth-shift cohorts. In particular, preserve the ten `|Δdepth| > .05` fitting pixels with their Gate 2 status. Nine are transfer-invalid and one camera-1 pixel remains a valid transfer with a distant same-ray oracle hit. Report how much error is localized to valid outliers and how much oracle radiance is unaccounted for at the invalid pixels. [Saved depth outliers](../results/publication_gate2/depth-outliers.csv)

## Bounded execution proposal

| Work | Ceiling |
| --- | ---: |
| Baseline point-directions per shadow or BRDF calculation | 14,794,752,000 |
| Additional point-directions per shadow or BRDF calculation | 1,342,177,280 |
| Total shadow queries and independent BRDF point-directions | 16,136,929,280 each |
| New primary rays, checkpoints, native models and optimizer updates | 0 |

Use the reviewed corrected image and A100 GPU 0 with four CPU threads, 64 GiB host RAM with no swap, 128 PIDs, the existing 25% CUDA allocator fraction and 512 MiB outputs. Proposed foreground timeout is 1,800 seconds with 30 seconds of kill grace, with a 1,740-second inner limit and 20-second inner grace. This runtime is unmeasured for this pair. Require one hash-bound approval, one fresh output directory, non-root, networkless execution and read-only inputs. Preserve partial arrays and the decision witness on failure. No automatic retries, changes to quadrature gates, mesh modification, new dataset or training. Finalize and inspect the RGB floor and its coverage limits before deciding whether a matched recovery protocol is justified.
