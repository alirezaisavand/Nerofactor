# Gate 10.1 preparation

**PROCEED to one separately authorized evaluation-only diagnostic.**
Authenticated300k G training is reused. Geometry admission remains blocked.
The original failed mesh and all checkpoints/freezes are retained.
[Execution protocol](../../protocols/BELL_GEOMETRY_EXTRACTION_GATE101_V1.md)

The single run will record the original masked and unmasked512³ fields and
all marching-cubes, coordinate-transform, Trimesh and serialization stages.
Face IDs, exact degeneracy, dtype/coordinate changes and vertex mapping
will localize the ten invalid serialized triangles. Exact historical PLY replay
is mandatory. A double-precision diagnostic PLY tests precision preservation,
including its conversion to the native float32 BVH input.

Raw-SDF diagnosis uses384 deterministic sphere directions, including128
lower-cap directions, five radial positions per direction, gradient/sign
witnesses, and256 area-stratified lower near-sphere mesh centroids.
These are defined from world coordinates and reconstructed geometry alone.
Raw-negative endpoints across sentinel grid edges and unsupported learned
zero-level samples expose extraction support problems independently of export.

Any diagnosed sentinel-induced crossing or unsupported lower-surface sample
stops for researcher review. That outcome completes the bounded diagnosis while
leaving the geometry unadmitted. No broad cap trimming or oracle-guided
modification is authorized.

The only conditional numerical correction removes exactly the ten confirmed
zero-area serialized faces, keeping all vertices and positive-area triangles.
If the broad raw-SDF diagnosis permits this correction, the run verifies
Open3D/native BVH compatibility and performs the remaining six-camera Gate10
audit with the original thresholds and full exclusion accounting. The
existing G→A handoff validator is reused only after every original gate passes.
The derivation and face mapping remain explicit in the new artifact.

## Prepared work and evidence

| Item | Maximum |
| --- | --- |
| Accepted checkpoint CPU loads | 1 |
| Original extraction / marching cubes | 1 /1 |
| Original grid SDF points | 134217728 |
| Additional diagnostic SDF / gradient points | 3840 /640 |
| Conditional original SDF / gradient points | 1536 /512 |
| Conditional primary BVH rays / CPU witnesses | 196608 /384 |
| Optimizer / scientific shader calls | 0 /0 |
| Foreground watchdog / grace | 1800s /30s |
| GPU / CPU / RAM / outputs | A100 GPU0 /.75 allocator,4 CPU,128GiB /8GiB |

A cold native process selects pinned G and the admitted corrected tracing
extension. It strictly loads only the accepted checkpoint, verifies its
network identity, preserves weights throughout, restores CPU/default trace state
and records cleanup separately from diagnostic/scientific admission outcomes.
Read-only checkpoint/source/oracle mounts and a fresh output are required.

Focused CPU tests cover coordinate-precision collapse, exact-zero-only filtering,
vertex merges and unused vertices, binary PLY encodings, raw/Sentinel crossing
classification, original threshold support, import blocking, mount-independent
freeze validation, the actual capture observer with synthetic functions,
and ownership of the substantive-geometry stop over the conditional audit.
These fixtures perform no checkpoint load, SDF query or actual extraction.
Native container execution remains untested.

The exact source/input identities, frozen output path, test inventory, JUnit
identity and final counts are in preparation.json and the protocol freeze.
No execution authorization or scientific output directory is created during
preparation. The expected runtime is1–10 minutes, unmeasured.

## Next research decisions

A sentinel-induced lower surface needs an explicitly reviewed extraction-policy
or learned-SDF decision. The current numerical filter cannot resolve it.
If the diagnostic instead isolates a purely numerical defect and all existing
quality gates pass, submit the derived mesh and authenticated handoff for review
before material training.

Retain the15k/30k/50k/300k geometry comparison as a separate high-value
evaluation-only follow-up. It can investigate phase scheduling and geometry
evolution without new optimization. It cannot replace the publication endpoint.
Material/illumination recovery and the ShinyNeRF comparison remain downstream.
