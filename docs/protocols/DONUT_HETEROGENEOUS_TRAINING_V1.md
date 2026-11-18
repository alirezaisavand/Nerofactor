# Matched heterogeneous Donut recovery v1

Train **D2, D6, D8-C and D8-T**, in that order, from their accepted
fresh initial states. Use the exact admitted heterogeneous Donut v2
reference. The researcher admission applies only to that dataset.
A separate source-bound authorization is required for training.

The fit set is cameras 0, 50, 1 and 45, with 15,286 observations.
Camera 28 and 34 contribute the 7,344 held-out observations. All four
arms learn the direct illumination and keep zero indirect light,
corrected A visibility, full network, seed 6033, original native
optimizer, scheduler, sample counts, and legacy material settings.
The photometric objective is the existing linear-RGB Charbonnier loss.
The existing loader is used with the heterogeneous input adapter.

| Arm | Accepted initialization and objective |
| --- | --- |
| D2 | Original corrected-core chart field, legacy priors and no added surface term |
| D6 | Same corrected-core initialization, legacy priors and transported raw world-Q term at 0.05 |
| D8-C | Chart parameterization with D8 width-0.25 isotropic start, D8 tensor renderer, scalar/energy/specular/white and transported raw Q |
| D8-T | Ambient parameterization with matched D8 physical start, renderer, loss and common world Gaussian tapes |

The D8 pair compares tensor parameterizations with their shared renderer
and objective. Differences between D6 and D8 extend to the representation,
initial width, sampler and chart-specific legacy term. D6 remains the
homogeneous reconstruction baseline and D8-T the ambient candidate.
D11's log loss stays outside this comparison.

## Admission and isolation

Check generation, verification and researcher admission bytes and all
39 reference outputs before model construction. Fresh initial state
hashes must agree with the accepted historical records, including the
D8 parameter and buffer signatures. The reference compatibility gate
uses the original 24 fitting-camera-0 probes and 32 native draws. It
renders the admitted pointwise materials under known L2 only inside the
existing no-gradient evaluation scope. Keep its accepted tolerances.
Require D8's paired initial physical tensor, RGB, scalar and shared
state checks. Verify all D6 and D8 neighborhood arrays and six sampler
snapshots against the accepted D6 stream.

The optimizer sees only positions, shading normals, outgoing directions
and observed linear RGB. The mesh sampler sees geometry and its frozen
RNG. Material labels, highlights, region IDs, boundary distances and
held-out RGB stay outside optimization. Native validation and saved-data
analysis score them as diagnostic targets. Boundary labels never mask or
reweight the geometry-only spatial term.

There are exactly 10,000 updates per completed arm. Save fresh maps at
initialization and at updates 2,000 through 10,000 every 2,000, with one
checkpoint at each post-initialization state. Keep 10,000 as primary.
Use the existing 8-draw evaluation, per-view arrays and visualization
encoding. The first two Adam displacement checks and five local
checkpoint round trips per arm retain the accepted behavior. Historical
initializers are JSON state identities. Checkpoint deserialization,
where used, is limited to strict round trips of checkpoints written by
that arm. There is no historical weight or optimizer loading.

A100 GPU 0 uses the reviewed corrected image, four CPU threads,
64 GiB host RAM without swap, 128 PIDs, 25% CUDA allocator cap and
8 GiB output cap per arm. The four arms run sequentially in a fresh,
non-root, networkless batch with read-only source, root and inputs.
Use a foreground watchdog of 14,760 seconds with 30 seconds of kill
grace. Each arm has a 3,600-second outer limit and 3,540-second native
limit with 20 seconds of kill grace. Stop the batch on a mandatory
failure and preserve all original and partial data.

| Work in the completed batch | Count |
| --- | ---: |
| Adam returns | 40,000 |
| Shader calls | 49,156 |
| Shadow-ray slots | 38,136,004,608 |
| Component reverse sweeps | 114 |
| Sampled surface pairs | 3,840,000 |
| New primary BVH rays and reference evaluations | 0 |

The adapted training runner shares the historical native optimization
implementation. Historical source-bound dataset generation and replay
remain reproducible at their original commits. All prior result,
freeze and authorization files are preserved.

## Evaluation

Replay physical labels from the admitted arrays and use the same
per-observation Q Frobenius ratio, sorted widths, linear RGB and
scalar-color/metallic error definitions. Report orientation modulo
director sign on points where both truth and prediction have gap
at least 0.1. Record eligibility, saturation, mean, median and p95
where applicable. Score all fitting and held-out observations,
the 0.035 native-unit boundary band, smooth interiors outside the
boundary band and weak patch, and nearly isotropic observations.
Retain both raw observation and observed-triangle area-weighted
statistics. Area uses the fixed face-area share per observation
within each split. Unsampled triangle error is undefined.

Count registered fitting views and views with an admitted highlight
within 0.01 native units and normal separation at most 30°. Fitting
points exclude their own camera from those counts. For boundary
preservation, use the fixed opposite-region right/down pixel pairs
on same or edge-adjacent manifold triangles, with the 0.02 separation
and 30° normal checks. Compute width jumps in the region-0 minus
region-1 direction. Compare signed and relative transported
world-Q jumps against the oracle. A predicted jump with small
projection onto the reference jump indicates attenuation or wrong
jump direction. Report errors on either side as boundary cohorts
using the same held-view masks.

The coverage preflight finds 63 fitting and 29 held-out adjacent
boundary pairs. These are repeated view observations. Both splits
have near-isotropy coverage, concentrated in camera 28 for the
held-out split. Also score learned illumination against L2 under
solid-angle weighting, report the original scalar material errors
and count operational time and native resources. The analysis is
file-only and does not select an endpoint using held-out performance.
The Bell proposal remains pending a subsequent stage.
