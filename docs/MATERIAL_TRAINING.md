# Isolated material training: 1D.2B-2B

1D.3C follow-up: [NATIVE_GPU_SMOKE.md](NATIVE_GPU_SMOKE.md) documents the separate
review-only initialization/one-forward harness with the explicit read-only mesh.
It leaves this pipeline, BRDF/shader behavior, defaults and NVS unchanged. No new
native GPU model test was run by Codex.

Review baseline: `66037fb00a4662693e7bc35694e760429d3d0076` on
`paper/infrastructure`. The separate `run_material_experiment.py` entry point
connects schema-1 material envelopes to the existing Stage II pipeline.
`run_training.py` remains legacy. The geometry CLI, shape renderer, geometry
validation and geometry final-NVS deferral remain unchanged. No historical branch
is merged and no original scientific YAML is rewritten.

This milestone executes only CPU fixtures/fakes. No native material training,
GPU work, full NVS evaluation, Docker container, real experiment allocation,
input modification, permission change, commit or push is performed.

## Configuration and explicit input roots

[material.example.yaml](../configs/experiments/material.example.yaml) wraps the
original bear configuration without changing scientific values:

```yaml
schema_version: 1
stage: material
legacy_config: ../material/real/bear.yaml
inputs:
  dataset: GlossyReal
  mesh:
    root: thesis
    path: meshes-pretrained/real/bear.ply
preprocessing:
  cache: cache/datasets
overrides: {}
```

Preflight from the repository root, with absolute roots supplied by the caller:

```sh
python3 -B run_material_experiment.py \
  --cfg configs/experiments/material.example.yaml \
  --config-root "$PWD/configs" \
  --datasets-root "$NERO_DATASETS_ROOT" \
  --thesis-root "$NERO_THESIS_ROOT" \
  --output-root "$NERO_OUTPUT_ROOT" \
  --run-id bear-material-seed6033-review-001 \
  --validate-only
```

`--validate-only` and `--execute` are mutually exclusive required actions. Neither
action is implicit. Preflight resolves and validates structure/paths, prints
diagnostics, reads scene metadata and small code hashes, and streams SHA-256 for
the selected mesh/checkpoint only. It imports no scientific/model modules,
constructs no model, decodes no tensor checkpoint and allocates no run. Dataset
contents are not fully hashed, and image/mesh validity and native dependencies
are not established by path/hash checks. Explicit `--execute` enters the existing
trainer after output/runtime setup; implementing this action is not authorization
to start a native experiment in this milestone.

The selected `inputs.mesh` must exist. The runtime pipeline receives its canonical
path and the selected dataset family path; original legacy strings remain in the
saved payload. The renderer reads mesh data, constructs its existing ray tracer,
and computes missing vertex normals in memory as before. It does not save or
rewrite the mesh. Training/validation/renderer receive the same dataset root and
`PreprocessingCache`. Original masks/cameras/images/reference geometry and
existing legacy preprocessing products remain read-only inputs. New derived
products are written to the selected run cache with the prior identity policy.

For prior geometry output, expose that run through an explicit read-only root:

```yaml
inputs:
  dataset: GlossyReal
  mesh:
    root: geometry
    path: meshes/surface.ply
```

Add `--input-root geometry=/absolute/prior/geometry/run` at runtime. The reference
is illustrative; extraction is not added here, so select an existing mesh rather
than expecting geometry training to export it. Additional input-root names must
be distinct; all input roots must be disjoint from the declared writable root.
If the prior run is in the host run collection, select a separate new output
directory with `--layout mounted`, rather than declaring the whole collection
writable. New mounted roots must be empty; collection layout exclusively creates
`<output-root>/<run-id>`. Output storage overlapping the source repository is
rejected. No host roots are committed as defaults.

Future mounts remain `/inputs/datasets` and `/inputs/thesis` read-only, an explicit
read-only geometry root if needed, and `/outputs` writable with `--layout mounted`.
Path checks do not create mounts or change permissions. Operating-system
read-only enforcement remains part of separately approved 1D.3 work.

## Run and checkpoint selection

- New material run: explicit unique `--run-id`, no checkpoint request. Existing
  storage is rejected, including empty collection run directories. Unexpected
  fresh-run checkpoint files fail instead of triggering implicit loading.
- Same-run continuation: same config, roots and ID, plus explicit
  `--resume-checkpoint model.pth` or another in-run checkpoint filename. Allocation
  and material compatibility identity must match. No latest/best discovery is
  introduced. Tensor decoding happens only during explicit execution.
- Historical weights-only initialization: declare `inputs.checkpoint: {root:
  thesis, path: relative/checkpoint.pth}` and pass `--initialize-from-input` with
  a new ID. Optimizer, progress and best metric restart. Network keys, shapes
  and dtypes must agree exactly before loading. No partial key matching,
  architecture conversion, or silent head remapping is implemented.

New checkpoints retain the original scientific state payload and add a tag with
material stage, algorithm identity, run ID and conservative compatibility
signature. Same-run continuation checks that tag, network layout, optimizer
group structure, finite best metric and step bounds. Saves atomically replace
only owned checkpoint destinations. A tagged baseline material checkpoint may
initialize another compatible material run through explicit weights-only intent.
Conflicting stage/algorithm tags are rejected even when a manifest is supplied.

Tensor shapes alone cannot identify producing mathematics: the ablation branch
shares heads while changing angle interpretation, and the safe-geometry branch
also changes heads and material behavior. An **untagged historical checkpoint**
therefore requires `--initialization-manifest /absolute/approved/config/root/file.json`:

```json
{
  "schema_version": 1,
  "algorithm": "anisotropic-e1e859779301faced46d6476691e20b546c27ae0",
  "source_commit": "e1e859779301faced46d6476691e20b546c27ae0",
  "checkpoint_sha256": "REPLACE_WITH_VERIFIED_CHECKPOINT_SHA256"
}
```

The manifest must be a JSON file inside the explicit approved config root, with
exactly these fields. Its hash must match the selected checkpoint. Accepted origin
declarations are the inventoried anisotropic baseline
`e1e859779301faced46d6476691e20b546c27ae0` and reviewed infrastructure baseline
`66037fb00a4662693e7bc35694e760429d3d0076`, whose material mathematics are retained.
Other origins require separate compatibility review; branch labels and filenames
are insufficient. Source tests check the retained baseline shader implementation.
The declaration must be supported by research records. It is a researcher-supplied
attestation bound to bytes, not independently recovered proof of producing history.
No manifest is fabricated or written into historical storage by this milestone.

Preflight can validate a supplied manifest/hash but does not inspect tensor tags;
an untagged checkpoint without a manifest fails at explicit state loading. Keep
the same initialization manifest/config when continuing an initialized run.

**Continuation is not exact resume.** Legacy constructor seeding and its order
are retained. Neither legacy nor new tagged checkpoints restore RNG/sampler/worker
state. Run signatures cover original settings/scopes, selected input paths,
mesh/checkpoint identity, scene metadata, relevant source hashes and selected
Python/package metadata. Changing these, including total steps or code, rejects
continuation. No cross-version resume migration is supplied.

## Preserved scientific behavior and configuration scope

The original `MCShadingNetwork` code is unchanged: BRDF equations, material heads
and ranges, two-component angle prediction/interpretation, tangent/bitangent
construction, importance sampling, PDFs and regularization calculations remain
the anisotropic baseline. Ray construction, split generation and its RNG effects,
tensor precision, loss formulas/order/coefficients, Adam arguments, clipping,
and the hard-coded cosine scheduler are unchanged. Adam still receives `lr=1e-3`,
`eps=1e-4`, `amsgrad=True`; clipping is `0.7`, seed default `6033`, and cosine
clamp/alpha remain 100,000/`0.2`.

`_init_shader` still forwards **only `shader_cfg`**, then sets its `is_real` from
the database family. Top-level `reg_energy_loss_lambda: 0.01` in the bear YAML
does not override shader default `0.05`; the same applies to specular regularization.
Nested `shader_cfg` values do override those defaults. Top-level lobe/anisotropy
fields are not moved into the shader. Loader diagnostics remain visible in the
console and effective record. Original and renderer-merged values are separate
from the actual merged shader values captured after construction.

The existing renderer `n_lobes` branch and shader head conventions warrant
separate mathematical/native review; they are retained rather than reconciled.
Supporting a wrapper is not a claim that every historical setting is scientifically
valid. Unknown payload fields retain bounded diagnostics rather than being
silently consumed or repaired. Supported routing is single-GPU material/dummy,
Adam/warm_up_cos, baseline `nerf_render`/`mat_reg`, `mat_render`/PSNR, and existing
`syn`, `real`, `nerf` families. The abstract custom-loader defect remains outside
scope. Structural preflight is not full scientific validation.

Material final NVS still runs on the current network at loop completion; it does
not reload the best checkpoint. It retains the original five held-out novel-view
IDs, order, `test_step(..., is_nvs=True)`, PSNR/SSIM definitions, precision, PNG
composition and per-view/average metric text. Only destinations change. Explicit
same-run continuation retains legacy append behavior for NVS metrics and replaces
that run's same-named NVS images. Unique evaluation lifecycles/protocol manifests
remain 1D.4; no new evaluation algorithm is introduced here.

## Writer inventory and containment

| Reachable owner | Legacy destination | Isolated material destination / status |
| --- | --- | --- |
| Material `_init_geometry` | Reads configured mesh; normals computed in memory | Reads selected `inputs.mesh`; no mesh writer or coordinate transform added |
| Dataset parse/crop/raw/image helpers | Scene camera pickle, processed images/crop metadata | New products in owned `preprocessing.cache`; existing source products read-only |
| `Trainer._save_model` | `data/model/<name>/{model,model_best}.pth` | `checkpoints/`; same network/optimizer/step/best payload plus compatibility tag, atomic writes |
| `Logger.log`, TensorBoard | Model directory text/event files | `logs/{train,val}.txt`, `logs/events.out.tfevents.*` |
| `MaterialRenderMetrics` | `data/train_vis/<model>-<val>/*.jpg` | `renders/validation/<model>-<val>/*.jpg`; final filename checked |
| Trainer final NVS -> material `nvs` | `logs/<name>/log.txt`, `nvs_imgs/<name>/test_<id>.png` | `logs/nvs/log.txt`, `renders/nvs/test_<id>.png`; final filenames checked; metrics/images unchanged |
| Material `save_pointcloud` | Repository-relative `cloud.ply` | `debug/cloud.ply` when explicitly called; existing caller is commented, no new debug option/invocation |
| Console/failure trace | Process stdout/stderr | Also `logs/console.log`; writer closed and runtime environment restored on failure |
| Conventional library caches/temp files | User/global defaults | Approved execution environment redirects XDG, Matplotlib, Torch/extension/Inductor, Triton, Numba, CUDA and temp paths into the run; bytecode writes disabled |
| Original/effective config and identity records | Infrastructure only | Run-root `config.original.yaml`, `config.legacy.yaml`, `config.effective.yaml`, `material.json`, allocation marker |

The audit follows trainer/evaluator/logger, material renderer/metrics, shader/loss,
dataset and utility call sites. Shader/BRDF/loss and active raytracing calls add no
project-owned file writer. Commented segmentation/mask/debug routines are not
enabled. Generic COLMAP, plot/point exporters, standalone extraction, relighting,
masked evaluation and geometry evaluation tools retain their legacy destinations;
they are not invoked by this entry point and are not newly isolated here.

Before opening even the console log, material execution rejects pre-existing
symlinks and hard-linked files anywhere in its run tree. This covers existing
Logger/TensorBoard destinations without changing those classes or geometry mode.
PathResolver/RunPaths additionally reject output escapes and aliases, including
final image/log/cloud filenames. The selected input identities are rechecked
before model construction and historical initialization.

Storage assumes one trusted owner and no hostile concurrent filesystem replacement.
It is not an OS sandbox. Libraries previously imported by an embedding caller may
retain earlier global caches; use the separate CLI/import boundary. Native
third-party writes still need actual environment/mount validation. Unsafe output
storage is rejected before file logging; ordinary execution failures are captured
inside owned logs. Original tensor-default exception behavior is retained; no
claim of recovering all model/library globals for in-process retries is made.

## Provenance and validation evidence

`material.json` records execution mode, stage/algorithm identity, mesh SHA-256 and
path/size/times, selected initialization input/hash/declaration when applicable,
compatibility signature, existing NVS policy and limitations. Original envelope
and referenced YAML bytes are saved separately. `config.effective.yaml` records
path-resolved values, diagnostics and successfully constructed trainer/renderer/
shader settings. Failure during construction may leave partial snapshots.

Step 1D.4 still needs full Git diff/command/environment/lifecycle manifests,
dataset content-hash coverage, source mesh run/coordinate provenance, complete
default/seed/optimizer/sampling records and RNG/sampler restoration. Mesh names
do not establish producing history, coordinate compatibility, or scientific
correctness. Historical reports remain separate from independently measured work.

Run the CPU infrastructure suite:

```sh
python3 -B -m unittest discover -s tests/infrastructure -v
```

The complete suite recorded **100 discovered, 97 passed, 0 failed and 3 skipped**,
including 15 material tests. Python 3.9.12, NumPy 1.21.5, scikit-image 0.19.2,
ImageIO 2.9.0 and PyYAML 6.0 were used. New fixtures execute the
real material constructor, geometry/dataset/shader initialization, validation
metric, NVS and point-export methods, plus the shared trainer/evaluator/logger/
loss/scheduler code, through AST-loaded definitions. Image I/O, PSNR/SSIM, pickle,
NumPy and draw/padding logic operate on tiny fixtures; shader/model/raytracer/
device/optimizer/TensorBoard/native mesh/point-cloud boundaries are explicit fakes.
No anisotropic rendering mathematics is executed by those fakes.

The committed bear material example passed read-only `--validate-only` preflight
against the declared host roots. No run was allocated; `paper_runs` remained
absent. This is metadata/path/hash evidence, not native mesh/model execution.

Tests compare exact legacy/new validation JPG, NVS PNG and metric-log bytes,
split IDs, settings and optimizer/checkpoint selection; enforce input bytes/mtime
snapshots and write guards; and exercise mesh-root selection, scope preservation,
fresh/explicit resume/initialization, foreign or incompatible checkpoints,
output aliases and failure cleanup. Source/AST comparisons preserve all scientific
methods and geometry behavior. Existing native OpenCV/Open3D/transforms3d checks
remain skips where dependencies are absent. Fake success is not native GPU parity,
full NVS validation, deterministic continuation or reproduced research evidence.

## Proposed next environment and native validation, requiring approval

First review/pin the historically appropriate Python/PyTorch/CUDA and native
extension environment, without assuming the current host produced thesis results.
Then separately authorize dependency/import and minimal raytracer/rasterizer checks,
followed by one-scene material initialization and a single validation view with
explicit mesh identity, reference/tolerances and a fixed resource budget. Exercise
read-only mounts with disposable fixtures before any research input is used.
Full NVS/training needs its own budget and approval.

Docker access previously failed inside Codex; it is not retried or bypassed here.
No socket/group/system permission changes are proposed as a workaround. Native
OpenCV/Open3D/transforms3d, CUDA/nvdiffrast/raytracing behavior, device memory and
ABI compatibility remain unvalidated. Native raytracer construction also requires
more than eight triangles; the tiny I/O fixtures deliberately fake that boundary
and do not establish native mesh validity. No additional native defect is repaired in
this milestone. Keep standalone extraction/evaluation and scientific corrections
in separate reviews.
