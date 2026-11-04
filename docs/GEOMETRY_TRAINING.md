# Isolated geometry integration: 1D.2B-2A

This milestone adds a separate geometry entry point and optional I/O hooks. The
review baseline is `52783b0f23dccdf3e46a707faf090c95b311873a` on
`paper/infrastructure`. Existing `python run_training.py --cfg ...` keeps legacy
configuration, paths, and implicit legacy checkpoint behavior. No original YAML
is migrated. Material training and standalone extraction/evaluation are deferred.

**Native geometry execution remains blocked by pre-existing interfaces.**
`NeROShapeRenderer._init_dataset` unpacks two lists from `get_database_split`,
which returns three; construction raises `ValueError` when it reaches that call.
The trainer's final `.nvs(log_dir_str=..., imgs_dir=...)` call also disagrees with
the shape renderer's `.nvs(pose, K, h, w)` signature. Neither defect is repaired,
and no NVS call is removed or replaced. Source tests preserve these facts.
Infrastructure approval does not authorize deciding the intended split or NVS
protocol. Those decisions need separate review before native training is useful.

No native training, GPU test, Docker container, expensive evaluation, real run
allocation, original-data write, or historical-result write was performed here.

## Entry point and configuration

Use a schema-1 `stage: geometry` envelope, an explicitly approved configuration
root, and explicit absolute runtime roots. For example, the committed
[geometry envelope](../configs/experiments/geometry.example.yaml) references the
unchanged synthetic bell configuration:

```yaml
schema_version: 1
stage: geometry
legacy_config: ../shape/syn/bell.yaml
inputs:
  dataset: GlossySynthetic
preprocessing:
  cache: cache/datasets
overrides: {}
```

The dataset selection denotes the family directory containing the selected scene.
The referenced YAML supplies `database_name`, train/validation dataset settings,
losses, optimizer/scheduler declarations, and other scientific settings. Only
dataset path strings are replaced in the runtime copy. The original payload and
its scalar types remain recorded, including unused fields. No deep merge,
shader-field forwarding, or automatic correction is introduced.

For a read-only preflight from the repository root, set runtime environment
variables to the chosen absolute host roots and run:

```sh
python3 -B run_experiment.py \
  --cfg configs/experiments/geometry.example.yaml \
  --config-root "$PWD/configs" \
  --datasets-root "$NERO_DATASETS_ROOT" \
  --thesis-root "$NERO_THESIS_ROOT" \
  --output-root "$NERO_OUTPUT_ROOT" \
  --run-id bell-shape-seed6033-review-001 \
  --validate-only
```

Root variables are supplied by the caller; defaults contain no host paths.
Preflight checks configuration structure, source scene/required paths, output
containment, allocation collisions, and selected resume identity. It prints
diagnostics and known limitations. It creates no directories, imports no model
or scientific modules, decodes no tensor checkpoints, and neither seeds nor draws
from scientific RNGs. It reads small implementation/asset hashes and scene file
metadata, not entire dataset contents. It does not decode images/cameras or
establish native dependency availability, mathematical validity, or trainability.

Both `--validate-only` and `--execute` are explicit, mutually exclusive actions;
omitting both is an error. `--execute` allocates/selects the run and enters the
existing geometry trainer after runtime output setup. That action is implemented
but was not run on native models in this milestone and is not authorization to
start an experiment. Known native defects above still apply.

Host layout defaults to `collection`: output root contains `<run-id>/`. For a
single experiment mounted at `/outputs`, specify `--layout mounted`, with roots
`/inputs/datasets`, `/inputs/thesis`, `/outputs`. Mounted layout exclusively
initializes an empty selected root; it adds no run-ID subdirectory. This code
does not create mounts or enforce read-only access at the operating-system level.

## Supported geometry boundary

Preflight requires the existing shape renderer, dummy datasets, Adam,
`warm_up_cos`, an explicit nonempty validation list, registered geometry losses,
`shape_render`, and key metric `psnr`. Training, renderer, and validation database
identities must agree. `is_nerf` must agree with the selected family; disagreement
is rejected rather than corrected. Single-GPU geometry families `syn`, `real`,
and `nerf` are routed; the existing abstract `custom` loader is rejected. This
structural support does not establish native execution for any family.

Every validation dataset and the renderer receive the same selected absolute
dataset root and the same `PreprocessingCache` factory. Existing source caches,
processed images, masks, cameras, and reference geometry remain read-only inputs.
New derived products use the prior versioned cache semantics. No split membership,
preprocessing arithmetic, resolution, or camera convention is changed.

Runtime execution keeps the repository working directory for existing relative
BSDF-asset and optional test-split reads. Audited output writers instead receive
absolute paths into the run. Before scientific imports, the separate CLI routes
TMPDIR/TMP/TEMP, Python temporary files, XDG cache/config/data, Matplotlib,
Torch model/extension/Inductor, Triton, Numba, and CUDA runtime caches into the
run, and disables bytecode writes. It restores environment/CWD on exit and
captures console output and execution exceptions in `logs/console.log`.

These settings require the separate process/import boundary. Libraries already
initialized by an embedding caller may retain earlier global cache locations.
Path checks are not an OS sandbox or protection against hostile concurrent
filesystem replacement. Runtime mounts and native extension checks remain 1D.3.

## Run creation, continuation, and historical initialization

- New run: supply a unique explicit `--run-id`; no checkpoint selection. Existing
  runs and nonempty mounted roots are rejected. Unexpected checkpoint files in
  a fresh allocation also fail instead of triggering implicit load.
- Same-run continuation: supply the original envelope/roots/run ID and
  `--resume-checkpoint model.pth` (or another explicit filename relative to that
  run's `checkpoints/`). There is no latest/best discovery. The allocation marker,
  geometry identity, and selected checkpoint location must match. Preflight checks
  these files without decoding tensors. On explicit execution, checkpoint tags,
  network keys/shapes/dtypes, optimizer group structure, step bounds, and finite
  best metric are checked before state loading.
- Historical initialization: declare an existing `inputs.checkpoint`, such as
  `{root: thesis, path: model/shape/bell_shape/model.pth}`, and explicitly select
  `--initialize-from-input` with a new run ID. This loads compatible network
  weights only. Historical optimizer state, best metric, and step are not adopted.
  This example path is illustrative; select an inventoried existing artifact.
  A checkpoint declaration without initialization intent is rejected for new runs.

For schema-1 continuation, the checkpoint stores the original network/optimizer/
step/best payload plus a geometry-run compatibility tag. Saves use a unique
temporary file and atomic replacement inside the selected checkpoint directory.
Historical checkpoints without that tag cannot masquerade as same-run resumes.

Compatibility identity covers the canonical original legacy settings, selected
dataset path and scene metadata, preprocessing destination, relevant code/asset/
split hashes, Python version, and selected installed-package metadata. Any change
to this identity rejects continuation, including changing `total_step`, moving
the dataset mount, or changing relevant code. Historical initialization inputs
record path/size/times; they are not content-hashed here. Scene metadata is a
lightweight fingerprint, not a dataset content checksum or complete environment
manifest. Full provenance is deferred to 1D.4.

**Continuation is not exact resume.** Legacy seeding still runs at construction;
neither old nor newly tagged checkpoints save/restore Python, NumPy, CPU/CUDA RNG,
sampler, or worker state. Network/optimizer/progress continuation cannot reproduce
the uninterrupted stochastic trajectory from that checkpoint. This limitation
must accompany any future use, independently of configuration compatibility.

## Outputs, provenance, and writer inventory

```text
<selected-run>/
  .experiment.json                 # existing exclusive allocation marker
  geometry.json                    # compatibility identity and known limitations
  config.original.yaml             # exact envelope source bytes
  config.legacy.yaml               # exact referenced YAML source bytes
  config.effective.yaml            # original values, path resolution, diagnostics,
                                  # partial owner snapshots and unresolved defaults
  checkpoints/model.pth
  checkpoints/model_best.pth
  logs/{train.txt,val.txt,console.log,events.out.tfevents.*}
  logs/nvs/                        # unchanged final NVS call still fails natively
  renders/validation/<model>-<validation-name>/<step>-index-<i>.jpg
  renders/nvs/                     # destination only; no new NVS protocol
  cache/datasets/...               # or selected preprocessing.cache
  cache/runtime/...
  tmp/...
```

`config.effective.yaml` captures the merged trainer config and merged renderer
config when those owners are constructed successfully. Shader/loss/scheduler
defaults and their effective scopes are explicitly marked partially unresolved;
construction failure can leave only the trainer snapshot. The original config,
diagnostics, and source hashes remain authoritative. This is not the full Git
diff/command/environment/lifecycle/evaluation manifest planned for 1D.4.

| Reachable owner or inspected call site | Legacy destination | Isolated geometry behavior |
| --- | --- | --- |
| Dataset constructors and `save_pickle`/image writers | Scene `cache.pkl`, processed images and crop metadata | Prior `PreprocessingCache` routes new products into selected run; source products read-only |
| `Trainer.__init__`, `_save_model`, `_load_model` | `data/model/<name>`, implicit `model.pth` load | Explicit run checkpoint directory, atomic saves, explicit selection |
| `Logger.log`, `SummaryWriter` via `_init_logger` | Text/TensorBoard files under model directory | Selected `logs/` |
| `ValidationEvaluator` -> `ShapeRenderMetrics` | `data/train_vis/<model>-<val>/*.jpg` | `renders/validation/<model>-<val>/`; image composition, encoding arguments and metrics unchanged |
| Shape renderer forward/render/validation, field/loss helpers | No active file export in this path | Geometry extraction returns vertices/triangles in memory; no new export added |
| Trainer final NVS path creation and delegation | `logs/<name>`, `nvs_imgs/<name>` | `logs/nvs`, `renders/nvs`; signature defect unchanged, so native NVS output not demonstrated |
| Scientific imports, extension builds, runtime temporary files | User/global library defaults | Conventional cache/temp variables redirected before imports; no permission changes |
| Infrastructure records, checkpoint temporaries, console tee | New infrastructure only | Contained run destinations |

The audit includes trainer, evaluator, logger, shape metrics/renderer, dataset,
field/loss and imported utility call sites. `train_tools.save_model`, generic
`save_h5`/`save_depth`/plot/point-cloud exporters and COLMAP writers are not called
by the audited geometry-training path. Dataset evaluation point export already
supports the cache API but is not invoked by shape validation here.

Remaining legacy writers outside the integrated mode include material metrics,
material renderer `save_pointcloud` (`cloud.ply`) and NVS, standalone extraction,
relighting, geometry evaluation logs/point clouds, masked-evaluation renders, and
COLMAP preprocessing/export tools. They are unchanged and must not be assumed
isolated. No active project-owned writer reachable in the supported isolated
geometry path was identified outside the selected run; third-party behavior and
native execution still require authorized environment validation.

## Numerical compatibility and CPU evidence

Network definitions, tensor precision, ray sampling, BRDFs, geometry losses,
coefficient schedules, train/validation splits and metric definitions are
unchanged. Adam construction still uses `lr=1e-3`, `eps=1e-4`, `amsgrad=True`
and its existing library defaults; gradient clipping remains `0.7`. Trainer seed
default is still `6033`. The cosine scheduler still uses its hard-coded 100,000
step clamp and alpha `0.2`, independently of configured `end_iter`. No seed,
optimizer, loss, sampling, or scheduler option is reinterpreted as a bug fix.

```sh
python3 -B -m unittest discover -s tests/infrastructure -v
```

Recorded on 2026-11-04: **82 tests discovered: 79 passed, 0 failed, 3 skipped**,
including 17 new geometry integration/compatibility tests. Python 3.9.12,
NumPy 1.21.5, scikit-image 0.19.2, ImageIO 2.9.0, and PyYAML 6.0 were used.
The committed bell example also passed `--validate-only` against the declared
host roots; `/home/isavand/paper/paper_runs` remained absent. This was a metadata
preflight, with no model import, checkpoint decoding, or run allocation.

The new geometry tests execute the real shared trainer loop, evaluator, logger,
shape metrics, losses, scheduler, and split helper through AST-loaded definitions,
using explicit CPU model/device/DataLoader/optimizer/TensorBoard fakes. They check
input-write guards and bytes/mtime snapshots, root/cache propagation, checkpoint
selection, settings and event parity, identical fixture validation logs/JPEGs,
collisions, unsafe names/paths, partial/default snapshots, failure logging,
environment restoration, RNG preservation during setup, and import-free CLI
validation. Fake renderer construction and fake NVS intentionally bypass the
native interface failures; dedicated source tests confirm those failures persist.

Native OpenCV/Open3D/transforms3d fixture checks remain skipped where dependencies
are missing. CPU fake success is evidence for I/O/control-flow plumbing, not
native model correctness, GPU numerical parity, end-to-end trainability, exact
resume, or scientifically reproduced results. No dependencies were installed.
Docker access previously failed inside Codex; it was not retried or bypassed.

## Proposed next material scope, requiring separate approval

Inspect material initialization/metrics/NVS/debug writers, then add a distinct
material adapter with an explicit selected read-only mesh and dataset/cache
propagation. Route material checkpoints, logs, validation/NVS images and point
clouds through owned run paths. Preserve shader field scopes, BRDFs, losses,
sampling, tangent/coordinate conventions, optimizer and scheduler behavior.
Define material architecture/mesh/config compatibility and weights-only
initialization separately from continuation. Add temporary CPU differential tests
and writer guards before any native experiment. Keep extraction/evaluation
refactoring, native geometry repairs, full provenance, Docker and training
execution in separately authorized work.
