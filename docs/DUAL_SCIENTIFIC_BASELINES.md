# Authoritative dual scientific baselines: 1D.5-RESET

The researcher selected **G for Stage I geometry** and **A for Stage II material**.
This is an authoritative future baseline decision, **not recovered attribution of
old checkpoints**. Work started clean on `paper/infrastructure` at reviewed commit
`bed325fe511ec52a61f5793d0ee0acd91dad26b9`. No branches are merged or rewritten.

| Stage | Scientific source | Exact commit | Maintained selection |
| --- | --- | --- | --- |
| Geometry | `aniso_geometry_safe` (G) | `79252178e65914ad1acbc14790b1514f276bfea0` | pristine `scientific_sources/geometry_g` |
| Material | `anisotropic` (A) | `e1e859779301faced46d6476691e20b546c27ae0` | pristine `scientific_sources/material_a` |

## Audit scope, exact differences and classification

The [source manifest](../scientific_sources/manifest-v1.json) records every source
path, commit, size and SHA-256: **99 G files / 967,155 bytes**, **106 A files /
976,792 bytes**, copied exactly from local Git objects. The snapshots retain all
network/dataset/train/utils/raytracing modules; both reachable COLMAP readers;
original YAMLs/split pickle; BSDF asset; original training/extraction/relighting/
evaluation entry sources and README. Root scientific code/configs are unchanged.
No dataset, weights or external checkout is copied. License/header limitations
are documented in [scientific_sources/README.md](../scientific_sources/README.md).

The [machine-readable audit](evidence/dual-baseline-source-audit-v1.json) compares
all these files against the reviewed infrastructure commit. It records both
source hashes, changed qualified functions/classes with source ranges/AST hashes,
and imports (including optional local dependencies), not just Trainer. Complete
text differences are retained in the [G patch](audits/geometry-to-infrastructure-v1.patch)
and [A patch](audits/material-to-infrastructure-v1.patch). There are **29 differing
G files** and **5 differing A files** in this closure; class-level AST differences
include changed children and do not imply every method changed. Whole-file
bytes/patches also capture top-level/default/config changes.

| Reachable area | G versus previous A-based geometry | Classification / consequence |
| --- | --- | --- |
| `network/field.py:SDFNetwork, SingleVarianceNetwork, NeRFNetwork`, ray construction/sampling and `extract_geometry` | Core definitions match inspected A; no license to replace the complete dependency module | Source equality where recorded; preserve all actual G source anyway |
| `network/field.py:AppShadingNetwork.predict_specular_lights/predict_diffuse_lights` | G uses `F.normalize` default epsilon; A explicitly supplies `1e-6` for sphere-light normalization | **Scientific behavioral difference**, potentially active depending on configuration; equal tensor architecture or camera-0 montage is insufficient to infer equality |
| `network/renderer.py:NeROShapeRenderer.__init__/_init_dataset` | G directly consumes two-list G helper. Current infrastructure adds storage hooks and A three-list adaptation | G's original consumer retained; current cache/output hooks are infrastructure/interface changes, not substitutes for G split behavior |
| `dataset/database.py:get_database_split` | G validation seeds Python 6033 without shuffle, val first ID, remaining train; A seeds/shuffles, five NVS plus one validation | **Scientific behavioral difference**: bell G train 1–127, validation 0, no NVS; actual input ordering and both original validation-loader/renderer calls retained |
| `dataset/database.py:NeRFSyntheticDatabase.__init__/get_depth` | G vertical focal is focal; A multiplies by H/W. G depth is random placeholder with alpha mask; A reads scaled image depth and thresholds it | **Scientific behavioral differences/RNG and evaluation risks**; G source retained, placeholder depth is not validated geometry GT; NeRF is outside first bell selector |
| `dataset/database.py:GlossyRealDatabase/CustomDatabase`, Synthetic readers; `utils/*`, COLMAP readers | Current infrastructure mainly redirects derived products and explicit dataset roots. Underlying original G read/crop/resize/normalize routines are preserved in the snapshot | **Infrastructure-only** paths and **interface adaptations** audited separately; source cache/pickle compatibility and native backend behavior remain runtime risks |
| `train/trainer.py:__init__/_init_dataset/_init_network/run` | Same seed sequence and registry-driven reduction pattern, but different split population and below optimizer/clipping behavior. Current infrastructure adds routing/state handling and skips incompatible A final geometry NVS | Use actual G constructor and loop/registries. No A loss/optimizer/renderer substitution or A split shim; G has no final material-style NVS dispatch |
| `train/lr_common_manager.py:LearningRateManager.construct_optimizer` | G passes only lr `1e-3`; Torch 1.8 defaults eps `1e-8`, AMSGrad false. A specifies eps `1e-4`, AMSGrad true | **Scientific behavioral difference** in Adam moments/updates; new G checkpoints cannot be admitted by key shapes alone |
| `train/lr_common_manager.py:WarmUpCosLR` | G alpha `.05`, configured end 300000 and no A horizon clamp; A alpha `.2` and hard-coded/clamped end 100000 | **Scientific behavioral difference**. Initial two LR values coincide, so CPU source/schedule checks beyond warm-up are also necessary |
| `train/trainer.py:Trainer.run` backward/optimizer block | G backward immediately followed by optimizer step; A clips gradient norm at `.7` | **Scientific behavioral difference**; no clipping in new G path. External observer records but never modifies gradients |
| `network/loss.py:OpacityLoss.default_cfg` and registry | G begin `.001`, transition start 20000, end `.01` at 50000; A begin zero/start10000. All registered G loss factories/defaults retained | **Scientific behavioral difference**, including early prefix. RGB/eikonal/std/init-SDF/occlusion/mask/curvature implementations and scope stay G; full list/config captured |
| `network/metrics.py`, `train/train_valid.py`, `train/train_tools.py` | Shape PSNR/SSIM computation/evaluator/selection rules agree in inspected sources; current output-path hooks differ. Original G validation uses source0, original zero-based call step and step+1 checkpoint progress | Numerical rules remain G; output paths only adapted externally. No forced validation or save within two updates |
| `extract_mesh.py`, `network/field.py:extract_geometry` | Exact original extractor source retained: bbox [-1,1], isosurface0, default resolution512, original marching cubes/trimesh export; original script implicitly loads model.pth | No new extraction execution wrapper or checkpoint substitution. Standalone extraction remains **blocked for target acceptance** until explicit source/checkpoint/output admission is reviewed |
| `eval_real_shape.py`, `eval_synthetic_shape.py`, `get_database_eval_points` | G uses triangle-mesh vertices for real GT and synthetic evaluation point generation/cache; A has changed helpers/point populations | **Scientific evaluation differences**. Not executed or silently repaired; optional eval-point writers must use owned view under any later approved evaluator |
| G material/PCA/extra sampler definitions imported in whole modules | Substantially different G material heads, frames and PDFs exist but are not the selected Stage II implementation | **Scientific differences, inactive for selected Stage I shape registry**; keep G dependency bytes, reject `network: material` under geometry stage |

| Material A versus reviewed infrastructure | Exact paths/functions | Classification / preservation |
| --- | --- | --- |
| Shader/BRDF, tangent frame, angle half-angle decoding, roughness/metallic/rotation heads/ranges, sample distributions/PDF/clamps, light and regularizers | `network/field.py:MCShadingNetwork` and helpers; `network/base_brdf.py`, `embedder.py`, `mlp_brdf.py`, `seq_brdf.py`, `utils/raw_utils.py/ref_utils.py` | Byte-identical original A versus reviewed root implementation. New Stage II imports the exact A snapshot, never G's material classes |
| Registered losses, Adam and cosine/clipping | `network/loss.py`, `train/lr_common_manager.py`, `Trainer.run` | A loss/LR modules are byte-identical. Current Trainer has I/O/state/NVS routing hooks but retains A numerical block. New tree executes original A: eps1e-4/AMSGrad, alpha.2/hard endpoint100000, clipping.7 |
| Renderer construction, geometry reading, shader scope, datasets | `NeROMaterialRenderer.__init__/_init_dataset/_init_geometry/_init_shader` | Reviewed root adds optional storage hooks; original A geometry/shading retained. A reads triangles into ray tracer, computes missing vertex normals in memory, writes no mesh conversion. Top-level shader-looking flags **remain unforwarded**; only `shader_cfg` plus derived is_real reaches shader |
| Validation, NVS and optional point cloud writes | `network/metrics.py`, `NeROMaterialRenderer.nvs/save_pointcloud`, `Trainer` | Original A metric definitions/splits/composition/append behavior and NVS dispatch retained; existing changes concern destinations. New source isolates destinations via filesystem view, not a rewritten evaluator |
| Dataset/cache interfaces | `dataset/database.py`, `dataset/train_dataset.py` | Optional current cache arguments/routes are **infrastructure/interface adaptations**; exact A snapshot avoids asserting module API similarity is mathematical parity. Native cache/image-library/mesh admission remains a compatibility gate |
| Legacy YAML/defaults and cameras | All source YAML and dataset methods, particularly real normalization and NeRF poses/depth | Exact A bytes, no G parameterization/defaults imported. Bear mesh must already share the documented A normalized scene frame; no corrective scaling/axis flip is performed |

All remaining text/config differences are in the complete patches and function
catalog. Scientific equality is asserted only for verified identical bytes/AST
regions, never from matching signatures. This audit does not establish native G
execution, numerical equivalence across stages, scientific correctness of either
baseline or producing provenance of historical assets.

## Stage selection and isolation implementation

Use **`run_stage_experiment.py`**, with mandatory stage and exact scientific
commit. R/R-repeat load G bell YAML directly and independently; C uses the new
schema-1 envelope. Arbitrary source/stage/config/mesh/checkpoint combinations fail
before model imports. Current selector scope is **G bell geometry** and **A bear
material**; all other snapshot presets are preserved for later reviewed selectors.
No custom numerical override, historical initialization or resume is enabled in
this reset. New checkpoint eligibility additionally needs strict stage-specific
architecture/state/provenance review, not merely a user-supplied origin label.

`stage_sources.py` verifies every snapshot file and closes all scientific
namespace packages to one selected tree. It rejects preloaded scientific modules,
missing-module fallbacks, second stage activation and unexpected source files.
Stage switching requires a new process. Shared infrastructure handles YAML,
paths, filesystem views, run allocation, diagnostics and orchestration; original
G/A trainer/model/loss/optimizer/scheduler/loader/metric methods execute unchanged.
The I/O subclass only rebinds log/checkpoint attributes after original constructor
and rejects implicit restore. Scientific source and infrastructure hashes/commits
are recorded separately; declarations are not signed clean-tree attestations.

`stage_workspace.py` mirrors only the selected scene into a fresh owned cache
using read-only input links. New derived products use original stage arithmetic
in writable directories; original cached products remain read-only inputs. Cache
identity includes actual stage/config/source and input metadata/hashes, with no
reuse across runs. Source symlinks and output/source overlaps fail. This filesystem
view is deliberate input linking, not a writable data symlink combining original
inputs and output storage. It is safe only after verified read-only native mounts.

Writer destinations are confined to the selected output: original model paths
are rebound to `checkpoints/`, Logger to `logs/`, original `data/train_vis` to
`renders/validation/`, `data/meshes` to `meshes/`, NVS `nvs_imgs` to `renders/nvs/`
and relative `logs` to `logs/nvs/`. Optional `cloud.ply`/other relative debug files
remain in `cache/stage-workspace/` and are output-hashed. Dataset cache.pkl,
poses.pkl, resized/cropped products and eval_pts remain within the owned scene
view or existing RO input links. Runtime temp/HOME/JIT caches are redirected to
owned storage. Checkpoints retain original scientific state format; context is
stored alongside in the run manifest, not fabricated as historical payload tags.
Standalone export/eval/relight scripts are pinned/audited but **not newly enabled**.

Earlier `run_experiment.py`/`run_material_experiment.py --execute` now fail closed
with the RESET selection instruction; their metadata-only preflights remain.
The original legacy `run_training.py`, old A diagnostic/inference/smoke code and
protocols are unchanged historical paths and are not selected-baseline execution
or evidence admission. No old result is relabeled as G. No current native command
has been run by Codex.

## Evidence policy and required replacement order

[Supersession matrix v1](evidence/dual-baseline-supersession-v1.json) separates
accepted-for-scope, historical/diagnostic-only, target-invalidated,
awaiting-review and unexecuted statuses. It covers both native model smokes,
image58 regression, A architecture match, camera0 exact-score/montage diagnostic,
actual A two-update R, prepared/unexecuted A R-repeat/C, material CPU contracts,
historical checkpoint/meshes and fresh G repetitions. Every available original
report remains unchanged. `evidence_policy.py` refuses acceptance outside each
explicit stage/scope; CPU-contract acceptance is not native training acceptance.

Actual prior A R:
`evidence:bell-training-two-updates.ukS2dZOL/r/training-smoke.json`, SHA-256
`009001b40f92c6337ef4db3260b5d88ba675ea1ff677cf52756858d18b04e071`,
complete/two calls at reported infrastructure commit `bed325f…`. It is valid
A-specific historical evidence and **invalid for target G training acceptance**.
Do not reuse it as R; do not complete A R-repeat/C to satisfy a G gate.

1. Review this source audit, isolation, G protocol and CPU/static tests first.
2. Execute the first fresh **G bell R** using the exact bounded command in the
   [G protocol](protocols/GEOMETRY_G_TWO_UPDATE_V1.md#first-researcher-side-g-run-commands-for-review-not-execution-approval).
   It exercises native G dataset/model initialization and two forward/backward/
   optimizer calls; prior A model smoke is insufficient. Preserve failed native
   G preflight attempts too; do not change scientific code to pass.
3. After R review, run independent **G R-repeat**, gate repeatability, then a
   fresh **G C** through schema-1 and compare. Review all evidence before proceeding.
4. Separately verify a **real G architecture/checkpoint-inference** path if loading
   the historical bell checkpoint is proposed. Existing A structure/camera0
   evidence remains true but does not approve G loading or resumed training.
5. Review natural validation/checkpoint intervals and explicitly versioned G
   extraction/mesh identity before any full-run approval. Two updates produce no
   trained mesh; no existing mesh is silently substituted.
6. Review G-to-A handoff, then rerun **A material initialization/forward** and
   bounded training/validation/NVS checks through the new source isolation. Prior
   A material CPU contracts remain narrowly accepted; native source path, caches,
   and handoff need replacement/review. Full training requires separate approval.

No target-native G result or end-to-end G-to-A acceptance exists yet. The selected
historical checkpoint is not declared G-produced or eligible for G resume.

## Geometry-to-material handoff

Material `inputs.mesh` resolves through an explicitly named read-only input root.
A separate `--mesh-handoff /inputs/geometry/handoff.json` must bind file SHA-256,
geometry run ID, algorithm `geometry-G`, pinned G commit, geometry-run manifest
reference/hash, original checkpoint/run context when available, coordinates,
scale, normals/orientation and material preprocessing. The referenced **complete
G run** must explicitly list the selected mesh in `mesh_outputs`; a two-update
report without an exported mesh is insufficient. Historical/unattributed meshes
cannot be asserted to satisfy this handoff. Coordinate/normal assertions require
reviewed producing evidence, not just metadata schema acceptance.
The handoff and producing manifest must name the same `database_name` as material
training. A G bell run cannot supply an A bear mesh. The current bounded geometry
selector covers bell; bear geometry selection/export needs a later reviewed
extension using its preserved G preset before a new G-bear/A-bear handoff exists.

Example schema (placeholders must be replaced with actual reviewed evidence):

```json
{
  "schema_version": 1,
  "geometry_algorithm": "geometry-G",
  "scientific_commit": "79252178e65914ad1acbc14790b1514f276bfea0",
  "geometry_run_id": "actual-reviewed-G-run-id",
  "geometry_run_manifest": {"root": "geometry", "path": "geometry-run.json"},
  "geometry_run_manifest_sha256": "actual-sha256",
  "geometry_checkpoint": {"path": "explicit-if-available", "sha256": "actual-sha256"},
  "mesh_sha256": "actual-sha256",
  "database_name": "real/bear/black_800",
  "coordinates": "actual G scene-world frame, camera convention and normalization transform",
  "scale": "actual units and numeric scale/transform; no guessed conversion",
  "normals": "actual exporter vertex-normal policy",
  "orientation": "actual winding/handedness and outward-normal convention",
  "material_preprocessing": "A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write"
}
```

Synthetic G bell reads float32 original poses at scale_factor1 and trains in its
scene frame; real datasets apply their original point/pose normalization. The
extractor samples normalized [-1,1] bounds and exports trimesh without a new
coordinate conversion. A material reads triangles as given, builds the ray tracer
from those coordinates and computes missing normals in memory. Never assume a
bear mesh and a bell mesh, differently normalized datasets or reconstructed and
pretrained meshes share a frame. No scale, axis flip, winding repair or normal
conversion is introduced to make a test pass.

## Outstanding compatibility and native gates

- G has not been natively initialized or trained by this reset. Its exact
  AppShadingNetwork epsilon, losses/defaults, original import/native ABI closure
  and absence of clipping must survive native checks; source parity is not a pass.
- Source filesystem views preserve calculation bodies but need native crop/cache/
  image-library verification; existing original cache products may have unknown
  producing versions. No generic cache compatibility is inferred from filenames.
- Historical checkpoint architecture match was against A. Even a future G tensor
  match cannot identify origin, restore missing RNG state or approve optimizer
  continuation. New execution deliberately supports only fresh initialization.
- Current material handoff is blocked by absence of a reviewed G run/export
  manifest binding a mesh. Metadata validation cannot infer geometry provenance.
- Standalone extraction/relighting/evaluation, alternative scene selectors, G
  full-training interval coverage and full stage-aware resume are not integrated
  or authorized. Optional source defects and placeholder-depth semantics are
  documented, not repaired. Scientific changes require separate review.
- Root snapshots and old protocols remain immutable historical records. New
  source versions/protocols/acceptance need explicit supersession; never rewrite
  old evidence, checkpoints, montages or logs to satisfy the selected baseline.

Linux's original fork DataLoader context is retained and recorded; non-fork
contexts are rejected rather than silently changing worker/import semantics.
The material loop has an external resource-local observer for worker/writer/
progress cleanup and captures effective trainer/renderer/shader/loss/optimizer/
scheduler settings after original initialization. It does not replace the loop.
Constructor failures may not expose fully assigned resources; isolated-process
limits remain necessary. New cache identities include Python/package versions
and declared immutable image identity as well as stage source/config/input hashes.

The machine audit also binds original G/A requirements declarations and the
existing reviewed native-extension manifest. Both stages retain the recovered
nvdiffrast 0.3.3 revision `729261dc64c4241ea36efda84fbf532cc8b425b8` and raytracing
0.1.0 revision `29329268a86921f1c938b05277937369531209a4` runtime proposal, Torch
1.8.1+cu111/CUDA11.1/ABI0 and original native package family. Source hashes and
historical binary hashes remain distinct from unverified new JIT/build identity.
No extension checkout, package, compiler, driver, image or container was modified.

## CPU/static validation for this reset

Complete suite: **216 tests, 213 passed, 0 failed, 3 skipped**, including **19 new
reset tests**. Existing skips remain OpenCV/transforms3d crop parity, Open3D
downsample/export parity and OpenCV RAW resize parity. No new reset case skipped.
CPU fixtures exercise actual pinned G Trainer.run AST with explicit fake data/
renderer/I/O and real CPU autograd/Adam; they verify >0.7 gradients remain
unclipped, exactly two returns, zero/nonzero-LR parameter behavior, state/cursor/
RNG records, R-repeat/C gates, failures, resource/trace/environment cleanup and
no implicit checkpoint loading. Separate checks exercise the actual original G
split, loss schedule and G/A optimizer/scheduler source, strict namespace
isolation, config/mesh/handoff rejection, cache/output containment and supersession.
Whole-orchestration CPU stand-ins are explicitly labeled and never native evidence.

Every one of the **205 snapshot files** matched its pinned Git blob. Root
scientific code and existing legacy YAMLs/protocols/evidence stayed byte-identical.
The existing material source-preservation test now normalizes only the explicit
two-line old geometry CLI rejection; all original scientific assertions remain.
Snapshot Git attributes preserve original bytes/line endings and allow historical
whitespace in exact source copies/audit patches; no source is reformatted.

Host runtime: Python **3.9.12**, Torch **2.4.0+cu121 CPU operations only**, NumPy
**1.21.5**. This is not a Python3.8/Torch1.8 native execution. Static checks passed
Python3.8 grammar for **160 files**, Docker definition/isolation policy, five RUN
shell parses, JSON, documentation links/Bash syntax and Git diff whitespace.
The static tool's pre-existing artifact-readiness warning is not a new judgement
on the working candidate image. No Docker engine or GPU execution occurred.

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/nerofactor-reset-matplotlib \
  python3 -B -m unittest discover -s tests/infrastructure -v
python3 -B docker/paper/static_checks.py
```

Original experimental evidence was hash-checked without regeneration; the new
policy changes only its target admissibility. No historical checkpoint/camera
pickle was deserialized, no training/evaluation of real inputs was performed,
and no commit or push was made. Stop here for source/protocol/CPU review before
any proposed researcher-side native G run or next research milestone.
