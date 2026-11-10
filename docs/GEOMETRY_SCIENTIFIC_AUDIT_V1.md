# Milestone 3A-1 — independent geometry evidence/coordinate audit v1

Protocol `bell-geometry-scientific-audit-v1`. Implemented from accepted base
`aec29d2063b9c1a88113380018239722dd2682b6` on
`feat/geometry-scientific-audit-v1`. The later `d3f3bec4` revision is not this
baseline. Original scientific snapshots, artifacts and accepted evidence remain
unchanged. No training, model inference, native validation, SDF evaluation,
Docker execution or infrastructure smoke repetition is part of this audit.

## Inputs, identity and output contract

`audit_geometry_scientific.py --inputs <absolute JSON> --inputs-sha256 <sha256>
--output-dir <new absolute directory>` runs NumPy/CPU only. There are no threshold,
resolution, camera substitution, transform, mesh repair, normal flip or retry
options. Each attempt must name a new output directory. No implicit file search,
latest/best selection, input writes or permission changes occur.

The version1 selection JSON contains exactly `schema_version: 1`, this `protocol`,
`artifacts` and `views`. Every artifact reference has explicit absolute `path` and
`sha256`. `artifacts` selects the committed pinned `source_manifest`, accepted
`extraction` report, immutable PLY `mesh` and accepted `rays` report. `views` selects
exactly three NPZs, in order, with explicit `role`/`source_camera_id`:
train14, validation58, NVS11. Every selection/file hash must match; report links,
source commits/files, extraction settings, completion flags, mesh identity, split
and view/archive bindings must agree. Fixed accepted identities:

| Artifact | SHA-256 |
| --- | --- |
| Scientific source manifest | `39b9baecce2aab6fc62ec57bd6f45e453d8e9d68215f6af3f25b8ed118c5a3f6` |
| G extraction report | `caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f` |
| G r128 mesh | `eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5` |
| A saved ray report | `88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510` |
| train14 NPZ | `3e786f597a11055962cbe1a8073c6d1c8ebc320d48cf26c8c9ec3961828a1449` |
| validation58 NPZ | `b588f8ef8d0ee61e07e10344c2350b287054a69e6a12994ea2eb89de69f3e5dd` |
| NVS11 NPZ | `57eff4adad35ec1eaf9ba7a8b70ae333f26aa281792178074ac1cfb80353407c` |

Sources selected by the explicit pinned manifest are transitively enumerated
with absolute paths and SHA-256s in the audit report. These are committed code,
not inferred original producing sources. No reference to a camera pickle or
checkpoint in the admitted JSON is followed or deserialized. Checkpoint SHA
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc` is only a
manifest identity assertion; its file is never opened by this tool.

All numerical loading uses `np.load(..., allow_pickle=False)`. V1 accepts the
saved NPZ layout only; its numerical NPY members are header-checked before array
allocation. Standalone NPY, object/structured dtypes, duplicate/extra/missing
members, altered shapes/precision and oversized/unsupported archives fail closed.
The exact12 members are origins, directions, positions, raw_normals, raw_depth,
normals, depth, hit, mask, K, pose, rgb. Camera/RGB shapes and all640000 rays per
original800×800 view are checked. Float arrays must be float32; hit/mask bool.
Miss positions/normals are unspecified original BVH payloads and may be nonfinite;
only hit geometry is interpreted. All depths, camera/ray inputs and RGB must be
finite. RGB remains original source RGB/255, without decoding images anew.

The PLY reader supports the admitted little-endian XYZ float32/triangular int32
layout without a geometry library. It measures finite coordinates, valid indices,
nonzero triangle area, bounds and signed volume, preserving original order and
winding. No Trimesh/Open3D processing or new mesh export. This is not independent
marching-cubes/SDF reconstruction or a nearest-surface/3D accuracy test.

Outputs are only a new directory containing `audit.json`. Output overlaps with
code, conventional original data/final_results storage or selected original
artifact directories are rejected. Sources/inputs open read-only; byte identities
are checked before loading and again at completion. Bad selection hash/schema,
missing paths or unsafe/colliding output fail before allocation; once selection
and output are admissible, artifact/numerical failures produce a versioned failed
report with blocker reasons. Partial files are retained after interruption and
must not be reused. No old evidence is overwritten or relabeled.

## Independent mathematics and frozen tolerances

The reference uses NumPy float64 column-vector algebra, independently of
`bell_material_rays.summarize()` and all scientific execution methods. Saved
float32 arrays are not changed. No native ray tracer/model/scientific namespace
is imported. All pixels/rays are checked, not a sample.

Pinned G geometry `79252178e65914ad1acbc14790b1514f276bfea0` and A material
`e1e859779301faced46d6476691e20b546c27ae0` synthetic loaders read the original
3×4 pose at scale_factor1 and K as float32. Pose is world-to-camera:
`X_camera = R X_world + t`; camera center is `-R.T t`. Pinhole rays use positive
camera Z and row-major pixels `(column+0.5,row+0.5,1)`. G builds camera rays with
K inverse, then applies R transpose and unit normalization in _process_ray_batch.
A applies the equivalent row-vector transform and unit normalization before BVH.
This audit independently solves for the center, solves K against all pixel centers,
rotates/normalizes rays and projects saved directions back into the image.

G's validation normally uses quarter-scale200×200 RGB/K; the saved A ray evidence
uses full800×800 calibration. V1 does not downsample or mix these calibrations.
It does not provide a newly observed G camera execution or open original camera
pickles to establish independent calibration identity. All three saved A camera
IDs also belong to the **current G training split**; they are not G held-out views
or verified historical training-membership claims.

G extraction maps XYZ voxel coordinates by `v/(128-1)*2-1`, isovalue0,
bbox[-1,1]³, outside-unit-sphere samples+1. SDF scale is1; no export transform is
introduced. The scene/SDF units have no independently established physical length
unit. Original producing dataset version/normalization remains unknown.

| Required check | Frozen bound/contract |
| --- | --- |
| Rotation R R.T vs I; determinant vs +1 | maximum absolute error≤1e-5 each |
| K last row [0,0,1], lower-left K[1,0] | error≤1e-6; positive fx/fy; condition number≤1e6 |
| Saved origins, independently solved center, -R.T t and R C+t | error≤1e-5 scene units |
| Saved vs calibrated unit ray components | error≤2e-6 |
| Saved ray/final normal unit lengths | error≤2e-6 |
| Saved rays projected to half-pixel centers | error≤0.005 pixels; nonzero homogeneous depth magnitude>1e-12 |
| Hit mask/raw depth | exact hit=(depth<10), raw_depth=depth, positive hit depths; misses≥10 |
| Hit count | at least1 |
| Intersections vs saved unit rays AND independently calibrated camera rays | max coordinate error≤2e-4 scene units |
| Hit positions outside mesh AABB | escape≤2e-4; this does not prove surface membership |
| Existing A normal negation/normalization | component error≤2e-6 |
| Raw BVH opposite-view/final view-facing fractions | ≥0.99 each; record every back/grazing count |
| Silhouette IoU | ≥0.90, exact TP/FP/FN/TN, no mask resizing/compensation |
| Recomputed vs reported counts/IoU | counts exact; IoU error≤1e-12 |
| PLY vs extraction serialized bounds/signed volume | absolute difference≤1e-12 |

Raw BVH lengths must be finite/positive; their measured unit-length deviations
are reported without assuming the raw magnitude contract is unit-length. A's
final normals are compared against its **already applied** negative raw unit
normal. The audit performs no mesh or input normal flip. Facing statistics are
observations, not proof of physical outward SDF gradients. The0.99 bound permits
outliers, which remain visible in counts/minimum cosines; do not claim all normals
are correct from an aggregate pass. Thresholds are frozen before original-evidence
execution and have no CLI overrides. Failures require review, not retuning/retries.

## Status meanings and unresolved scientific claims

Every check records pass/fail/unresolved, measured values, tolerance/contract and
blocker reasons. Root pass means **only the bounded saved-evidence coordinate
checks passed**. Empty hits fail and leave hit-geometry statistics unresolved.
Other scientific categories remain separately explicit:

- Historical checkpoint origin/configuration/split: unresolved.
- G architecture compatibility: not newly assessed by this CPU audit. The bound
  extraction report's prior accepted exact structural match is recorded, not
  revoked and not inferred anew from mesh/ray consistency.
- Extraction identity: manifest/mesh/source binding only, not new extraction.
- Coordinate consistency: bounded saved-camera/ray algebra only.
- Original calibration identity/native G camera execution: not independently
  verified from A dumps; original pickle is deliberately unopened.
- Scientific3D geometry accuracy/convergence: unresolved; no reference geometry,
  distance metric or ground-truth depth is consumed. Silhouette agreement is2D.
- **3A-2 SDF sign/gradient orientation: unresolved and unexecuted.** Signed volume,
  winding and camera-facing normals cannot promote it to verified.
- Research G→A admission: unresolved, with research_handoff_admitted=false.
  Historical r128 is diagnostic, not an attributable new G run or canonical512.

See [dual baselines](DUAL_SCIENTIFIC_BASELINES.md),
[extraction protocol](G_MESH_EXTRACTION_DIAGNOSTIC.md),
[ray protocol](BELL_MATERIAL_INTEGRATION_DIAGNOSTIC.md), and
[historical attribution](evidence/BELL_EVALUATION_PROVENANCE.md).

## Small researcher-side CPU commands on marineris

Use the reviewed feature checkout and an existing CPU Python/NumPy environment;
no installation, Docker, model/native validation or permission change. Verify
accepted ancestry and exclude the rejected revision before running:

```bash
set -eu
S_SOURCE=/home/isavand/paper/Nerofactor-paper
git -C "$S_SOURCE" merge-base --is-ancestor aec29d2063b9c1a88113380018239722dd2682b6 HEAD
if git -C "$S_SOURCE" merge-base --is-ancestor d3f3bec4b814933e493158c5d29ed582afc776dd HEAD; then
  echo 'Rejected baseline ancestry; stop for review' >&2
  exit 2
fi
S_BATCH=$(mktemp -d /home/isavand/paper/evidence/geometry-scientific-v1.XXXXXXXX)
cat > "$S_BATCH/inputs.json" <<'JSON'
{
  "schema_version": 1,
  "protocol": "bell-geometry-scientific-audit-v1",
  "artifacts": {
    "source_manifest": {"path": "/home/isavand/paper/Nerofactor-paper/scientific_sources/manifest-v1.json", "sha256": "39b9baecce2aab6fc62ec57bd6f45e453d8e9d68215f6af3f25b8ed118c5a3f6"},
    "extraction": {"path": "/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run/G-mesh-diagnostic.json", "sha256": "caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f"},
    "mesh": {"path": "/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run/meshes/bell_shape-best-150000-r128.ply", "sha256": "eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5"},
    "rays": {"path": "/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run/bell-material-diagnostic.json", "sha256": "88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510"}
  },
  "views": [
    {"role": "train", "source_camera_id": "14", "path": "/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run/evaluations/train-14.npz", "sha256": "3e786f597a11055962cbe1a8073c6d1c8ebc320d48cf26c8c9ec3961828a1449"},
    {"role": "validation", "source_camera_id": "58", "path": "/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run/evaluations/validation-58.npz", "sha256": "b588f8ef8d0ee61e07e10344c2350b287054a69e6a12994ea2eb89de69f3e5dd"},
    {"role": "nvs", "source_camera_id": "11", "path": "/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run/evaluations/nvs-11.npz", "sha256": "57eff4adad35ec1eaf9ba7a8b70ae333f26aa281792178074ac1cfb80353407c"}
  ]
}
JSON
S_SHA=$(sha256sum "$S_BATCH/inputs.json")
S_SHA=${S_SHA%% *}
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -B "$S_SOURCE/audit_geometry_scientific.py" \
  --inputs "$S_BATCH/inputs.json" --inputs-sha256 "$S_SHA" --output-dir "$S_BATCH/result"
sha256sum "$S_BATCH/result/audit.json"
printf 'Retained CPU audit: %s\n' "$S_BATCH"
```

Exit0: bounded checks pass, scientific unresolved categories still present.
Exit2: admission/check failure; inspect blockers and retain the fresh directory.
If interrupted without a report, do not assume completion or reuse the directory.
The shell's set-e stops before the last display on failure; `S_BATCH/result/audit.json`
still contains a report for failures after admissible selection/output allocation.

## Implementation validation

Focused synthetic CPU suite: **16 passed,0 failed,0 skipped**, using host
Python3.9.12/NumPy1.21.5. Fixtures independently construct a small rectangular
camera grid and inward cube; fixture hashes/image size are patched only in tests,
with no runtime bypass option. Tests cover deterministic results/input and RNG
preservation, wrong proper rotation, altered K, reversed normals, bad intersections,
mask disagreement, empty hits, incompatible IDs, incomplete reports, wrong SHA,
object/extra/missing/malformed NPZ and bad shapes, output aliases/collisions,
forbidden threshold overrides, allow_pickle=False and denied scientific imports.

```bash
python3 -B -m unittest discover -s tests/infrastructure -p test_geometry_scientific_audit.py -v
python3 -B docker/paper/static_checks.py
```

Static definition/isolation checks and5 Docker RUN shell parses pass;
Python3.8 grammar passes for185 files. Local links, documented CPU Bash syntax
and whitespace pass. Both pinned scientific closures (205 files) pass their
manifest checks;384 baseline scientific/configuration/evidence/infrastructure/
policy files match aec29d exactly. Accepted reports, PLY and all three original
NPZs are accessible and their opaque SHA-256s match the explicit commands.
Real JSON/source/hash admission passes without numerical archive loading.

The original-evidence numerical audit was **not executed** during preparation;
its actual residuals will be measured by the explicit CPU command above. No old
infrastructure smokes/full test suite, native validation, training, model inference,
SDF test, Docker/GPU, original pickle loading, package/permission change, automatic
commit/push/merge or existing-evidence edit was performed. The static checker's
existing incomplete build-manifest notice is unchanged and unrelated to this
NumPy-only audit. Python3.8 runtime itself was not executed here.
