# Gate 10.1, authenticated endpoint extraction diagnosis and conditional recovery

**PROCEED to one separately authorized evaluation-only batch.** Accepted training
at300000 updates remains the publication history. This task prepares one original
512³ extraction with complete stage captures, prospective raw-SDF witnesses,
and a conditional exactly-zero-face correction and original geometry audit.
Checkpoint deserialization, extraction and SDF queries require a new explicit
commit/freeze/output-bound approval.

## Immutable inputs and runtime

Use checkpoint27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd
from Gate10 v4. Preserve failed mesh
99c093250840be41dac95b73de38db9e7da92dbbc5235215ccccd27fc581d149,
all intermediate checkpoints and earlier evidence. Host preflight checks the
accepted receipt/checkpoint/mesh hashes without loading pickle. Native execution
permits one isolated CPU deserialization of this exact checkpoint, strict native
network loading and exact accepted tensor identity. Optimizer restoration is
absent. Original G79252178e65914ad1acbc14790b1514f276bfea0 remains immutable.

Retain admitted Bell v3 calibration, six original evaluation views, original
level0, bounds[−1,1]³, resolution512, float32 field,64-axis chunks, original
radius≥1 sentinel+1, PyMCubes.1.4, Trimesh4.12.2 and Open3D.16.0.
Use the existing corrected image and predicate-FP64 BVH identity. Oracle geometry
enters only the conditional quality audit and cannot determine SDF samples,
correction, trimming or checkpoint selection.

## Extraction-stage capture

Invoke original extract_geometry once. A read-only Python trace captures
extract_fields' returned dense field and marching_cubes' arrays immediately
after its original assignment, before world-coordinate conversion. The SDF
callback records each raw unmasked float32 tile and the exact Torch sphere mask
before the original function overwrites sentinel values. Verify every dense
masked tile equals where(outside,1,raw). These recordings add no model queries.

Save all of the following as numeric NPY/PLY with identities.

- Full raw learned SDF grid, outside-sphere mask and actual dense extraction field.
- Raw marching-cubes voxel vertices/faces and original returned world arrays.
- Post-default-Trimesh arrays and original-default PLY readback.
- Returned-to-processed vertex mapping, including explicitly unmapped unused
  vertices, coordinate shifts and unchanged face-occurrence correspondence.
- Double-precision diagnostic PLY with exact Open3D readback.
- Exact area, repeated-index and coincident-coordinate face IDs at every stage.

Preserve dtypes and byte identities. Original face order must be traceable
through processing. The original-default PLY must reproduce the failed mesh's
exact SHA-256. Any mismatch, nonfinite array, changed face inventory, unexpected
mapping stops the run. Stage-local degeneracy counts are diagnostic measurements;
the corrective policy remains restricted to the ten known serialized faces. No alternate output is
selected to make replay succeed.

For each known bad serialized face, report the first stage where its exact area
becomes zero. Distinguish raw marching-cubes degeneracy, voxel-to-world conversion,
default vertex processing and float32 export. Default processing can move/merge
vertices; report the measured shifts. This experiment measures the source of
each defect rather than inferring it from the final PLY.

## Prospective raw-SDF diagnosis

Let h=2/511, the original extraction spacing. Use no oracle geometry or material
labels to select these diagnostic points.

Define256 deterministic equal-area Fibonacci directions over the unit sphere.
Add128 equal-area directions restricted to world-Y∈[−1,−.65], the lower region.
Use midpoint Y and golden-angle azimuth, with float32 directions saved exactly.
Query radii1+h×[−1,−.5,0,.5,1] for each direction,1920 raw values.
The half-spacing inner/outer samples must straddle the original Torch sphere mask.
At all384 radius1−h/2 points, record raw SDF, input gradient and ±h/2 probes along
the normalized gradient, for another1152 value evaluations.

Independently select256 area-quantile positive-area triangle centroids from
world-Y<−.65 and abs(radius−1)≤h, using the replayed mesh. Retain duplicates if
area quantiles select one triangle twice. Record the eligible population and
selected face IDs. No replacement domain is used if this region is absent.
At these points evaluate raw SDF, input gradient and the same ±h/2 probes,
at most768 further value evaluations.

The original numeric support rule is unchanged: gradient norm≥1e−6,
abs(SDF)/gradient_norm≤h/2, inward SDF<−1e−7 and outward SDF>1e−7.
Report supported raw zero-level samples, unsupported/ambiguous surface samples,
and the original offset/mask exclusions.

If learned raw values at all five frozen radial positions are<−1e−7, the sentinel inserts a
sign change absent in the learned radial values. Record it as sentinel-induced
crossing evidence. Negative-inside/positive-outside raw values indicate a raw
radial crossing. All five positive values indicate no sampled crossing at the sentinel.
Other cases are ambiguous. The radial classification and gradient support
measure different local properties and are retained independently.

Also count original dense-grid edges straddling the sphere whose raw endpoints
are both negative while masked endpoints cross zero. This supplies direct
original-grid evidence independent of mesh export. The raw field also preserves
learned geometry away from the ten defective triangles. Queries slightly outside
the extraction box are diagnostic evaluations of the unchanged SDF network,
with domain[−1−h,1+h]³. They provide no training or extrapolation-accuracy claim.

**Substantive-geometry stop.** Any diagnosed sentinel-induced radial crossing,
unsupported selected lower-surface sample, or missing lower-surface sample domain
requires researcher review before correction/audit. Preserve all stage and
witness arrays and report the conditional audit as unexecuted. This prevents
automatic removal of a broad sphere cap or oracle-guided trimming. An ambiguous
radial direction without an extracted-surface sample is a diagnostic observation;
it does not manufacture a surface correspondence.

## Prospectively fixed numerical correction

The only eligible corrected surface is the historical replayed float32 PLY with
exactly these ten zero-area faces removed, preserving all vertices and all
positive-area face coordinates, winding and order.

229038,229053,229055,233117,458754,462320,701575,705281,773844,777213.

Require the exact zero-area inventory to match. No small-area tolerance is used.
Save candidate-to-original face indices and its own PLY/hash. Removing only
zero-area primitives preserves the positive-area triangle support exactly.
Index incidence, boundary/nonmanifold edges and winding diagnostics are reported
before/after. The native BVH receives a finite, strictly positive-area candidate.

The float64 PLY is a diagnostic export with verified Open3D parity. The current
BVH consumes float32 coordinates, so precision preservation alone cannot be
accepted if the native cast reintroduces degeneracy. Record its zero-area count
after the native float32 cast. It is not a second candidate selected by results.

The correction is reached only after the broad raw-SDF diagnosis permits it.
A larger correction, different processing strategy, broad trimming, hole filling,
normal flip, extraction-domain change or new level requires researcher review.

## Conditional completion of original Gate 10 audit

Use the unchanged Gate10 inspect_view, paired_surface, surface_witnesses and
sdf_consistency definitions. Verify candidate PLY and Open3D coordinates/indices
exactly before native BVH construction. Evaluate all six original128² rasters
against oracle and independent FP64 closest-hit references. The mapped retained
positive-area triangles are identical to those in the failed original artifact,
providing explicit visibility-support preservation. Native and independent
closest-hit checks measure the corrected candidate's numerical behavior.

Retain exact oracle hit-mask replay, original2e−4 position/depth/normal and2e−6
unit-normal limits, per-view IoU≥.90 and eligible correspondence coverage≥.80,
positive facing, distance≤.02, signed normal angle≤30°, and the original tighter
.005 and grazing cohorts. Preserve every raster observation/exclusion. Run the
original≤512-point raw-SDF gate with its original thresholds and zero allowed
failures. Broad diagnosis adds evidence without loosening that gate.

If original numerical checks fail, stop and preserve partial records. If
geometry-quality gates fail, report their failure without another candidate,
threshold change or extraction. On complete quality success, the offline verifier
may produce a separate derived-mesh handoff through the unchanged G→A validator.
It records the failed parent mesh, removed face IDs, mapping identity, checkpoint,
new producing diagnostic receipt and source. Researcher review remains pending
before any material execution.

## Work, resources and receipts

At most one CPU checkpoint load, one marching-cubes call,512 original extraction
tiles /134217728 raw grid SDF points. Additional diagnostic maxima are3840 SDF
points /640 first input-gradient points. Original conditional audit maxima
remain1536 SDF points /512 gradient points,196608 BVH primary rays and384 CPU
closest-hit witnesses. No native shader call, renderer RGB query, optimizer,
reference generation or training is authorized. Array recording and mask/grid
arithmetic add no SDF evaluations. All attempted work is bounded before issue.

One foreground non-root Docker invocation, existing immutable corrected image,
A100 GPU0,4 CPU threads/cores,128GiB host RAM/no swap,128 PIDs,.75 CUDA allocator,
8GiB owned outputs,1800s outer/1740s inner watchdogs and30s termination grace.
Estimate1–10 minutes, unmeasured. Dense files total about1.2GiB before mesh stages.
Root/source/checkpoint/oracle mounts are read-only, network disabled.
Retain TORCH_EXTENSIONS_DIR under owned outputs. No workers or training DataLoaders
are constructed. Confirm prior container-removal receipts and reject active GPU
experiments before allocation. Stop on mandatory failure without retry/extension.

Keep an atomic incomplete/failed/complete diagnostic receipt, actual exit code,
host container absence, cleanup outcome, source/input/output/binary identities
and each conditional path's executed or missing work. Completion of the bounded
diagnosis can legitimately end with substantive review required and no admitted
mesh. The offline verifier replays stage and witness decisions and only creates
a handoff when the complete original audit passes.

## Subsequent research

Retain a separately approved15k/30k/50k/300k extraction comparison to study the
relationship between original schedules and geometry evolution. It requires
checkpoint deserialization and extraction but zero optimization. Its observations
cannot replace the fixed300k endpoint. Material training and external baseline
execution remain outside Gate10.1.
