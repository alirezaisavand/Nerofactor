# Gate 10 v4 completed training and failed geometry admission

**BLOCKED for geometry admission and downstream G→A.** Authenticated pinned G
completed300000 original Adam returns, but its frozen mesh gate rejected10
exactly zero-area faces. The stop is scientifically valid. Training, extraction
and failed evaluation evidence are preserved. A separately authorized
extraction/export diagnosis can reuse the completed endpoint.
[Verified numerical summary](results-v4/summary.json)

## Verified execution and provenance

| Quantity | Saved and independently checked evidence |
| --- | --- |
| Training source | G79252178e65914ad1acbc14790b1514f276bfea0 |
| Infrastructure commit | 6638d3233e439b2c1ec9279eb46c2d514ba2509a |
| Freeze | 6c77069cc577735a63638674094e6e7ce23666f57a12e72a3ea91a6044683a8e |
| Dataset | Admitted Bell v3,128²,122 fitting /1 validation /5 held |
| Seed and endpoint | 6033, exactly300000 original Adam returns |
| Natural validation / saves | 300 /300, with prescribed intermediate checkpoints verified |
| Training wall time | 39469.19s,10.964h |
| Training cleanup / host exit | Verified true /0, owned container absent |
| Evaluation host exit / native cleanup | 1 /false, owned container absent |
| Endpoint checkpoint SHA-256 | 27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd |
| Checkpoint size | 29967467 bytes |
| Extracted PLY | 556329 vertices,1112676 triangles |
| Extraction calls | All512 tiles,134217728 original field nodes |
| New evaluation ray queries | 0, stopped before oracle/reconstruction tracing |

The offline verifier checks clean frozen input/source identities, original
authorization, training completion and cleanup receipt, first1000 CPU checkpoint
gate, all retained checkpoint hashes, exact endpoint receipt/network identity,
evaluation failure boundary, training log counts and runtime extension hashes.
No checkpoint was deserialized during this analysis. The manifest binds all
regular files in the original batch by absolute parent, relative path, size and
SHA-256, including the original checkpoint and mesh. Large native files remain
outside Git. Copies of the small analysis outputs match their external originals.
[Complete artifact manifest](results-v4/artifact-manifest.json)

Original evidence is at
/home/isavand/paper/evidence/bell-geometry-gate10-v4-5eba8365e19e4ab9aa2320491429cffb.
Final offline analysis is at
/home/isavand/paper/evidence/bell-geometry-gate10-v4-analysis-v3.
Earlier file-only analysis-v1 and analysis-v2 outputs are retained. No accepted run, frozen protocol,
source snapshot or failed launch record was rewritten.

## Exact mesh defect

All mesh coordinates and triangle-area calculations are finite. Ten triangles
have exactly zero area in independent float64 arithmetic on the saved float32
PLY coordinates, representing0.0008987% of faces. All10 contain coincident
coordinates. Six repeat a vertex index, including one triangle with a single
index repeated three times. The remaining four have distinct indices whose
coordinates coincide. Zero area is therefore a measurable invalidity, independent
of a small-area tolerance. The smallest positive triangle area is3.07674e−15.
No triangle was deleted, collapsed or repaired.
[All face IDs, vertices and incident-face diagnostics](results-v4/degenerate-faces.json)

Native extraction recorded556332 raw vertices and1112676 faces. The saved PLY
contains556329 vertices and the same face count. Three vertices disappeared
within the subsequent default Trimesh constructor/export pipeline. The failed
run used PyMCubes0.1.4, Trimesh4.12.2 and Open3D0.16.0. The native pipeline did
not save raw pre-constructor arrays, processed pre-serialization arrays or the
dense SDF grid. It is consequently unresolved whether each offending triangle
was already invalid at marching cubes, collapsed during default vertex
processing, or lost distinct coordinates during float32 PLY serialization.

A focused CPU fixture demonstrates that distinct positive-area float64 vertices
can collapse to zero-area coordinates upon float32 serialization. This proves a
possible numerical mechanism, without attributing the actual triangles to that
mechanism. Another fixture verifies repeated-index versus coincident-coordinate
accounting. Both pass. [JUnit evidence](results-v4/tests.junit.xml)

## Unit-sphere investigation from existing geometry

Using one extraction grid spacing h=2/511 as a descriptive radial band,395786
faces have all three vertices within h of unit radius. These faces account
for36.9475% of exported positive surface area. Four zero-area faces touch this
band and six occur farther inside, including radius approximately.391 and.953.
The degenerate-face failure therefore spans interior and near-boundary regions.

The saved point projections show a broad near-spherical lower region beyond
much of the oracle vertex distribution. Original extract_fields sets every
radius≥1 node to+1. A learned negative interior can produce an artificial cap at
that boundary. The observations warrant checking that mechanism, but a cap
classification requires the raw SDF and sign-crossing witnesses, which were not
reached. Neither the radial band nor these vertex projections measures
silhouette IoU, signed normals, ray visibility or correspondence quality.
[Saved mesh/oracle projections and defect locations](results-v4/mesh-defect-locations.png)

## Loss trajectories and optimization behavior

Training logs contain15000 every20-step rows, and validation logs contain300
every1000-step rows. The scalar logs use native rounded precision. In particular,
logged curvature0.00000 cannot establish zero curvature loss. Recorded
full-precision phase witnesses show its weighted contribution around4.5–4.8e−7.

The original inverse-variance phase is visible in active gradient-bearing
parameter counts increasing from135 to136 at return15001. Occlusion is zero
before its source-defined20k activation, and the first recorded active witness
has weighted loss.02137. At300k the witness records RGB.0321106,
occlusion.0082355, eikonal5.13719e−5, mask1.81608e−6 and opacity7.97560e−5.
The sampled preclip gradient norm is.31697 at the final return, with native final
LR approximately2.5e−5. These finite observations support executed optimization
and do not establish convergence or geometry correctness.
[Weighted training trajectories](results-v4/training-trajectories.png)

The final native camera58 validation reports PSNR33.72639dB and SSIM.939866
on the original32² downsampled display-domain raster. Median validation PSNR is
36.510dB over11k–20k and33.792dB over21k–50k. It is34.089dB over201k–300k.
The change overlaps scheduled occlusion, opacity, mask and alpha-annealing
phases and continuing optimization. It does not isolate any individual
coefficient's effect. The fixed300k endpoint is retained regardless of earlier
validation scores.
[All native validation measurements](results-v4/validation.csv)

The montage reuses the original saved JPEGs at1k/15k/30k/50k/100k/200k/300k.
Its first row of native panels is GT RGB, predicted RGB and the renderer's normal
display, identified from ShapeRenderMetrics source. JPEG compression,32²
resolution and native display encoding limit interpretation. These panels were
not rerendered and their normal colors do not measure signed normal accuracy.
[Stored validation progression](results-v4/saved-validation-progression.png)

## Scientific interpretation and missing evaluation

| Possible limitation | Available evidence and remaining uncertainty |
| --- | --- |
| Acquisition | Admitted images/calibrations and split inventories verify. G loaded only122 fitting and1 validation view. The independent full-raster replay was unexecuted. |
| Numerical mesh validity | Ten exact degeneracies in the saved artifact are confirmed. The responsible extraction/processing/serialization stage remains unresolved. |
| Extraction support | A large near-unit-sphere region warrants raw-SDF cap checks. Source sentinel behavior is known, but learned raw SDF was not captured here. |
| Optimization | All original300k returns and schedules ran, with finite logged losses and sampled gradient witnesses. Validation trajectories alone do not establish geometry convergence. |
| Appearance approximation | Pinned G has two scalar roughness encodings without a physical principal director. Its effect on these particular geometry errors is unisolated. |
| Silhouette objective | The original mask expression has zero background contribution. This may permit extra support, but the current evidence does not establish a causal effect. |

Silhouette IoU, depth/displacement distributions, signed normal errors,
closest-hit visibility, correspondence coverage/full-raster exclusions, raw-SDF
zero crossings and geometry-quality admission remain **unverified**. No partial
surface score or successful G→A handoff is claimed. Training cleanup passed;
evaluation cleanup is not certified by its native receipt, while host container
removal is confirmed. No material or external baseline training occurred.

## Smallest next scientific action

Keep the completed300k checkpoint and all existing intermediates. Prepare one
separately authorized **evaluation-only extraction/export diagnostic** from this
exact checkpoint, preserving the original512 grid, sphere sentinel, level and
source. Capture the dense SDF field and all stages: raw marching-cubes vertices/
faces, coordinates returned by original extract_geometry, post-Trimesh arrays
and PLY readback, including dtypes and index mappings. Compare exact areas and
coordinate coincidences at each stage. Capture the existing raw-SDF sign
witnesses near the sphere using the frozen definitions.

This localizes whether a precision-preserving export can retain the original
surface or whether the raw extraction itself requires a separately reviewed
mesh policy. Any derived mesh must have explicit provenance and its own approval.
An invalid-face filter or precision change is not applied in this task. The
full training history can be reused without repeating or extending optimization.

After diagnosis, obtain researcher review of the narrowly justified export or
mesh policy and its new evaluation freeze. Preserve the original failed gate
and require all remaining numerical and geometry admission checks before G→A.
The existing material, ShinyNeRF and anisotropy-aware geometry backlog remains
pending authenticated geometry acceptance.
