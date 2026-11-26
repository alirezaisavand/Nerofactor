# Gate 10.1 results, successful diagnosis and substantive extraction review

**Execution completed successfully. Geometry admission remains blocked for
researcher review.** The batch exited0 in26.52seconds, verified native cleanup,
and recorded host container absence. It localized all ten degenerate triangles
and identified a broad sentinel-induced surface in the extraction field.
The frozen substantive-geometry stop correctly withheld the correction/audit
branch and G→A handoff.
[Verified numerical summary](results-v1/summary.json)

## Provenance and completed work

Execution used preparation commit1852244c140c1e825003f44743865ece36aede35,
freeze b1ad431e11a2bd59e97734de566eb767f7d9cd1339d08afabb80ae2c67608933,
the frozen corrected image, pinned original G and the exact accepted300k
checkpoint27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd.
The original masked extraction reproduced the failed PLY's exact SHA-256.

| Work | Observed |
| --- | --- |
| Checkpoint CPU deserializations | 1 |
| Original extraction tiles / nodes | 512 /134217728 |
| Marching-cubes calls | 1 |
| Additional diagnostic SDF points | 3840 |
| Additional input-gradient points | 640 |
| Optimizer returns / scientific shader calls | 0 /0 |
| Native BVH rays / CPU closest-hit checks | 0 /0 |
| Cleanup and network-state preservation | Verified |
| Conditional derived mesh / handoff exported | Neither |
| Original geometry-quality audit | Unexecuted following the frozen review stop |

Original evidence is at
/home/isavand/paper/evidence/bell-geometry-gate101-ea426f2b86d44aa3ae8be40fd6ab25c0.
Final offline analysis is at
/home/isavand/paper/evidence/bell-geometry-gate101-analysis-v3.
Earlier offline partial/analysis outputs are retained.

The file-only analysis verifies every native output identity, source/input
bindings, checkpoint/network identity receipts, all stage metrics and vertex
mapping, full-grid sentinel relation, original-grid crossing counts, radial
classifications and raw-SDF support decisions. The native receipt SHA-256 is
2a8c748f687bb3db1c53ebd364aba90275af01068efb6064f6b62eed4c34bcb0.
It performs no additional checkpoint deserialization, model query or extraction.
Both original G/A manifests remain unchanged. Small tracked outputs match the
external analysis byte-for-byte. Dense grids, raw arrays and binaries remain
outside Git with sizes and hashes.
[Artifact manifest](results-v1/artifact-manifest.json)

## The ten degeneracies have three origins

| Stage | Total zero-area faces | Newly introduced |
| --- | --- | --- |
| Raw marching cubes | 4 | 4 |
| Original world-coordinate conversion | 4 | 0 |
| Default Trimesh processing | 6 | 2 |
| Float32 PLY serialization | 10 | 4 |

All counts use exact float64 area arithmetic on each saved stage's coordinates.
No small-area threshold, normal flip or face repair was applied.
[Per-face attribution and stage areas](results-v1/face-localization.csv)

Four faces229038/229053/229055/233117 are already degenerate in raw
marching-cubes output. Their distinct indices meet at coincident raw coordinates.
They share the integer grid node(165,256,298), whose learned raw SDF is exactly0.
The sphere sentinel is inactive at that node. These are raw isosurface-corner
degeneracies, inherited by later stages.
[Exact zero-valued grid-corner evidence](results-v1/exact-zero-grid-corners.json)

Default processing introduces degeneracy in faces773844/777213. It merges three
vertices overall, with maximum measured world-coordinate shift1.64958e−9.
The captured face-occurrence mapping ties every raw vertex and triangle to the
processed geometry, establishing processing-stage attribution.

Faces458754/462320/701575/705281 first become zero-area during float32 PLY
serialization. Their processed float64 coordinates retain positive area, while
the exported float32 coordinates coincide. The largest coordinate shift from
serialization is2.98023e−8.

Double-precision export roundtrips exactly through Open3D, but still has six
zero-area faces inherited from earlier stages. Converting that export to the
current BVH's float32 input recreates all ten. Precision-preserving export alone
therefore does not resolve this numerical pipeline.
[Stage and lower-surface diagnostic figure](results-v1/defect-stages-and-surface-support.png)

## The broad lower sphere is unsupported by the learned zero level

At197236 original grid edges, the inside/outside sphere mask changes while
both raw learned-SDF endpoints are negative. The masked field introduces
negative-to-positive crossings on those edges by substituting+1 outside the
sphere. Counts by axis are42941,110993 and43302.

This is direct evidence of artificial sign changes in the actual extraction
field. It localizes the sentinel's numerical effect independently of export.
It does not attribute the learned negative extension to any particular loss,
appearance approximation or optimization behavior.
[Raw and sentinel-masked dense field slices](results-v1/raw-versus-sentinel-fields.png)

The frozen sphere-direction groups have deliberately different spatial coverage,
so they are reported independently.

| Group | Sentinel-crossing evidence | Raw radial crossing | No sampled crossing | Ambiguous |
| --- | --- | --- | --- | --- |
| Global256 directions | 42 | 0 | 213 | 1 |
| Lower-region128 directions | 118 | 0 | 6 | 4 |

Sentinel evidence requires all five raw radial values to remain below−1e−7.
The global result is42/256,16.4%. The deliberately lower-focused group is118/128,
92.2%. Their pooled fraction is not a whole-sphere or mesh-area estimate.
The median lower-group raw SDF at radius1 is−.17050. Gradient-based support
at the384 inward radial locations is5/384. That local-gradient test and
the radial sign-crossing classification describe different directions.
[Radial measurements and masked-field change](results-v1/sphere-crossing-evidence.png)

Among256 distinct area-quantile lower near-sphere face witnesses, only2 satisfy
the frozen raw zero-level/gradient/sign rule. The other254 are unsupported.
Their median raw SDF is−.19758, median gradient length.99302, and median
first-order zero-level distance proxy.19617 scene units. The allowed proxy
bound is1/511≈.001957. The median proxy is about100 times that bound.
This is substantial unsupported surface, beyond the ten zero-area primitives.

These measurements support a substantial sentinel-induced lower region.
They do not establish that every face in Gate10's36.95% near-sphere area band
has the same cause. Raw-field slices also show learned negative regions extending
beyond the intended domain. Eliminating artificial closure would expose those
learned-field/domain limitations, which still require geometry evaluation.

## Frozen stop and remaining scientific decision

The batch correctly completed its bounded diagnosis and stopped for substantive
review. The ten-zero-face policy produced an in-memory positive-area view for
area sampling. Its conditional export/audit path was not entered.
No corrected mesh, altered extraction field, sphere trimming, geometry training
or material training occurred.

Silhouette IoU, depth/displacement, signed normals, visibility/correspondence
coverage, full-raster exclusions and the complete original SDF/mesh gate remain
unverified. Removing the ten numerical degeneracies would leave the broad
sentinel-supported surface. Geometry admission remains blocked.

The next useful decision is a separately reviewed **domain-aware raw-SDF
extraction policy** using the already captured field. A prospective policy should
evaluate raw zero crossings only in the original supported domain, specify
boundary-cell handling explicitly, retain open boundaries and components, and
avoid manufacturing positive exterior values or selecting surfaces using oracle
geometry. This changes extraction behavior while preserving the learned network.
Its numerical cleanup, visibility compatibility and original quality criteria
must be frozen before execution.

The captured arrays can support that next extraction without another checkpoint
load, SDF query or optimization history. They do not supply correctness outside
the trained geometry domain. Whether the remaining learned surface is accurate
must be measured through the original full-raster audit.

Retain the15k/30k/50k/300k geometry comparison as an evaluation-only follow-up
after the extraction policy is reviewed. It can investigate schedule-related
geometry evolution without replacing the fixed publication endpoint.
