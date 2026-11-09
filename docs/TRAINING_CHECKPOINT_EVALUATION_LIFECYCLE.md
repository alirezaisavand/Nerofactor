# Milestone 2 — bounded G/A training/checkpoint/evaluation lifecycle

Implementation preparation from `d72b2b84c2e7d704ee2f6642eca9b2ab0366350b`.
Native execution remains researcher-side after review. Milestone2 is **not yet
accepted**: it requires successful native save, fresh-process restore/update and
held-out evaluation evidence for both stages. Short checkpoints and metrics are
functional evidence, not convergence, thesis reproduction or paper-quality results.

## Original lifecycle audit

Authoritative snapshots: geometryG `79252178e65914ad1acbc14790b1514f276bfea0`,
materialA `e1e859779301faced46d6476691e20b546c27ae0`. All scientific files/configs
remain immutable; no legacy entry point is changed. Full source-file hashes are
checked against `scientific_sources/manifest-v1.json` before native imports.

| Behavior | G geometry bell | A anisotropic material bell |
| --- | --- | --- |
| Original YAML | configs/shape/syn/bell.yaml | configs/material/syn/bell.yaml |
| Total updates |300000|100000|
| Seed/loader |Torch, NumPy, Python6033 in constructor;8 workers, shuffled dummy train loader|Same|
| Split |Unshuffled train1…127, validation0; no G three-way NVS split|Python6033 shuffle0…127, first5 NVS, next1 validation58, remaining122 train; first NVS11|
| Adam construction |lr1e-3, default eps1e-8, betas(.9,.999), AMSGrad false, weight decay0|lr1e-3, eps1e-4, AMSGrad true, same betas/weight decay|
| Scheduler |Warm-up5000, base5e-4, configured end300000, cosine floor0.05|Warm-up1000, base5e-4, hard-coded cosine end100000, floor0.2|
| LR at zero-based steps0/1/2 |0,1e-7,≈2e-7|0,5e-7,1e-6|
| Gradient clipping |None|Original norm clip0.7|
| Validation/save cadence |Every1000 completed updates and final validation|Every2000 completed updates and final validation|
| Logging cadence |20 updates|10 updates|
| Validation resolution |Original800 RGB/cameras, original quarter preprocessing →200×200;512 test rays/chunk|800×800,1024 test rays/chunk|
| Final NVS |Trainer.run has no final NVS call; shape nvs(pose,K,h,w) is a separate interface|After final update, nvs(log_dir_str,imgs_dir) traverses all5 current NVS views|

Both original `_save_model()` implementations write exactly four top-level keys:
`step`, `best_para`, `network_state_dict`, `optimizer_state_dict`. The network
state includes parameters/buffers; Adam state includes per-parameter counters and
moments (A also AMSGrad maxima) and parameter groups. No RNG states, scheduler
object/state, configuration, source/image identity, loader iterator/sampler state,
training ray-pool shuffle or cursor is included. Adam parameter IDs are positional,
not named provenance. Source/config equality and a separate named-state manifest
are therefore necessary for admitted restoration. Old same-shaped checkpoints
are not automatically attributable or research-compatible.

Both `_load_model()` methods implicitly inspect `self.pth_fn`, normally relative
`data/model/<name>/model.pth`, using full pickle `torch.load()` with original device
mapping, network `load_state_dict` (strict default), optimizer `load_state_dict`.
They return `(best_para,start_step)` or `(0,0)` when absent. `model_best.pth` is
never implicitly selected for resume. `Trainer.run()` starts its range at saved
`step`; an update with zero-based index s saves **s+1**, while validation/evaluator
receives **s** and logs use s+1. LR is recomputed from that loop index, not a stored
scheduler counter; the restored optimizer LR initially reflects the last update.

Validation occurs before scheduled regular saving. Original `ValidationEvaluator`
sets eval mode, adds `eval=True` and step to each item, runs original forward in
`torch.no_grad`, then invokes all configured losses/metrics. It concatenates
outputs and takes mean PSNR as key metric. The **last** validation dataset wins
checkpoint selection. Strict `val_para > best_para` saves `model_best.pth` with
step s+1; then validation logging; then cadence `model.pth` save. Best starts0.
No ties are saved. A final NVS uses the last in-memory network, without reloading
best. This milestone manually saves early; it does not execute1000/2000 updates,
best selection, final full NVS or claim to have tested natural scheduling.

## Evaluation semantics and known limitations

G uses original validation camera0 and original `test_step(index,step)` and
`ShapeRenderMetrics`, including SDF gradient enablement inside rendering. It must
not use inference_mode or substitute A's split. Its synthetic preprocessing,
mask/depth nearest-neighbor resizing, original RGB interpolation/cameras and
gradient-dependent rendering remain unchanged. G test_step calls zero_grad;
gradient buffers are not an optimizer-state restoration claim. Original PSNR/SSIM
are full-image: the returned gt_mask is not used to restrict them.

A validation calls original renderer forward→`test_step(0)` for source58 through
the unchanged `ValidationEvaluator` and `MaterialRenderMetrics`, including original
loss calls. Next, a separate external dispatch facade calls original
`test_step(0,is_nvs=True)` for source11, using the same CUDA-default tensor context
as original nvs(). It then runs the same original evaluator/metric implementation.
The full five-view nvs loop is not invoked or rewritten. Both IDs must be excluded
from actual original training IDs. Original initialization reads all128 images
for an otherwise-unused image-info build; the optimization ray pool uses only
train_ids. Loading held-out input metadata is distinct from optimizing on it.

A's test_step preallocates zeros for every output pixel and writes RGB/ground truth
and channels **only at mesh hits**. Missed rays are black in both returned images.
Metrics nevertheless average the whole image. They are not foreground-only scores,
nor comparison against unmodified RGB at misses. This can inflate scores with a
large black background; no mask/background correction is applied. Original alpha
is allocated but never filled. Rotation is not exposed by original test_step;
record its absence, never synthesize it. The bounded evaluation exports all actual
tensor channels, including mx/my/metallic/kd and original encoded tangent/normal vectors,
without renaming or forcing arbitrary channels into RGB range.

A shading already applies its original linear_to_srgb. A tangent/bitangent/normal
outputs are visualization encodings `(unit_vector+1)/2`, not raw BVH vectors;
G normal output is also encoded and additionally masks inner samples. Do not
interpret these arrays as unencoded directions or apply another gamma/normal flip.

Original metric RGB conversion is `clip(float_RGB*255,0,255).astype(uint8)`: truncation,
not rounding, no added gamma. Raw float arrays and their actual ranges are kept;
finite values outside[0,1] are reported, not silently changed before the original
metric conversion. Full-color PNGs contain exactly the original converted bytes.
Material/SDF raw channels remain lossless .npy, including HDR values and the
original channel encodings.
Original montage writers retain their JPEG/composition behavior in owned outputs.

Original PSNR casts uint8 RGB to float32, computes per-channel MSE then channel
mean, and uses10log10(255²/MSE). Independent comparison uses float64 reduction over
the **same** quantized pixels, with frozen absolute bound0.001dB. SSIM is independently
re-invoked using the same skimage implementation, win_size11/channel_axis2/
data_range255, bound1e-12. This is not an independent SSIM implementation. Both
positive-infinite PSNR values are an exact-match case, encoded as a tagged JSON
metric; a finite/infinite mismatch fails. Bounds are fixed before native results;
do not retune them post hoc. No foreground or new quality metric is introduced.

## Diagnostic actions and admission

`run_lifecycle_diagnostic.py` has separate `save`, `restore`, `evaluate` actions,
each stage explicit, each cold process, fresh unique outputs and exact pinned
source. `--validate-only` is metadata-only: no native imports/allocation, historical
camera/checkpoint pickle or RNG changes. Execute requires trust/manual-save and
isolation acknowledgements. Checkpoint SHA, explicit producer report/hash, source,
configuration, mesh, splits and file identities must pass before unsafe loading.

Save uses the existing accepted two-update transport/probes, actual original
Trainer.run and Adam, full defaults/sample counts. It stops immediately after
returns0/1, closes owned resources, then explicitly invokes original `_save_model`
for `checkpoints/model.pth`, stored step2/best0. Best0 (original integer type) is initialized progress,
not a measured best validation score. No model_best is written or selected.
Original bytes are not augmented. The separate versioned manifest records file
SHA/size, exact named weight/buffer bytes, every Adam moment, counter values/types,
group values/types/order and scheduler attributes.

Restore requires the completed save manifest and its exact checkpoint, selected
explicitly in a read-only prior-output mount. No fallback/latest lookup. A fresh
process reconstructs the real original dataset/model/optimizer and seeds as before.
CPU-mapped pickle inspection checks exact fields/progress, finite catalog, all
moment shapes/dtypes/hashes/counters, positional named mapping and hyperparameter
compatibility before modifying live state. Then the original `_load_model()`
restores strictly; exact state equality is required before the original loop
executes one genuine update at index2 and stops after its optimizer returns.
Loaded LR and actual original step2 scheduler LR are recorded. Counter/moment
progress, finite losses/gradients and nonzero parameter changes are required.
Original G has no clipping; A keeps its original anisotropic calls and clipping.

Original checkpoints cannot restore RNG/loader/ray cursors. No continuation sidecar
is implemented here. Original reconstruction resets the ray pool/cursor; the real
post-reload selected sampling hashes, cursor0 and post-update cursor, RNG hashes,
and comparison to saved-prefix sampling are recorded. No sampled ray, cursor or
seed is patched to manufacture equivalence. The evidence can establish exact
checkpoint state restoration and a functional post-reload update, **not bitwise
equivalence to uninterrupted training**. Full reproducible continuation remains
a separately scoped future protocol requiring actual trajectory proof.

Evaluate requires a completed save checkpoint and producer manifest, strictly
loads network state only, never restores optimizer state and never enters run().
Original fresh optimizer is retained unchanged for before/after checks. G evaluates
camera0 at200²; A evaluates58 then11 at800², at zero-based step1 corresponding to
stored2. Source IDs, camera arrays, masks/hits, float channels, prediction/GT PNGs,
original/independent metrics and output hashes are retained. Model weights/buffers,
optimizer and ray cursor must remain unchanged; RNG effects are recorded. Scoped
guards block genuine training/save/load/NVS/optimizer/backward bodies during eval;
G's original SDF autograd.grad path is retained. Original evaluator workers and
progress resources are closed on success/failure.

## Evidence and manifest

Milestone1's existing A report and ray report were independently inspected as
files and SHA-verified, without regenerating them. See
[milestone1 acceptance manifest](evidence/milestone1-bell-ga-acceptance-v1.json).
The inspected A report records exactly2 Adam returns, LR0/5e-7,753356 changed
elements after return2, eight active groups and successful cleanup. Source and
mesh identities agree. This accepts only the already-reviewed short prefix.

The [manifest schema](protocols/training-lifecycle-manifest-v1.schema.json) versions
`lifecycle.json`: original/effective config, source and declared infrastructure/
image identities, actual runtime/CUDA/extension binary hashes, dataset hashes,
separate G/A splits, mesh/provenance, checkpoint/named optimizer hashes, progress/
LR/RNG/sampling evidence, evaluation/range/masking definitions, output hashes,
status/errors/timestamps and owned cleanup. Jsonschema is already present in the
inventoried environment; no dependency/image changes or network schema refs.

Historical r128 mesh admission remains the separate fixed-hash diagnostic exception
from milestone1. It is not canonical512, new attributable G training evidence or
an ordinary research G→A handoff. `stage_workspace.handoff` and ordinary launchers
are unchanged. Historical producing checkpoint revision/split remain unresolved.

All originals/code/selected checkpoints and prior reports are read-only; only a
fresh owned `/outputs` is writable. Existing workspace/cache/log routing, cold
stage import selection and cleanup are reused. Failed/partial runs are preserved.
Timeout/OOM can prevent graceful cleanup or a final report; retained terminal logs,
launch identity and exit-code files are evidence, not a successful lifecycle.

## Researcher-side commands after patch review

Start a new tmux session on marineris:

```bash
tmux new-session -s "bell-lifecycle-$(python3 -B -c 'import uuid; print(uuid.uuid4().hex[:12])')"
```

In that shell define the launcher below. `L_INFRA` explicitly pins this reviewed
baseline plus the reviewed worktree patch; all actual implementation-file hashes
are recorded. If the researcher commits the reviewed implementation first, replace
that one literal with the **reviewed new commit**, then retain the equality checks.
Do not silently follow a moving branch or change source/image between actions.
Native source selection independently requires exact G/A scientific commits.

```bash
set -eu
L_SOURCE=/home/isavand/paper/Nerofactor-paper
L_INFRA=d72b2b84c2e7d704ee2f6642eca9b2ab0366350b
L_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
L_GEOMETRY=/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run
L_ACCEPTED_A=/home/isavand/paper/evidence/bell-G-to-A.VgChrrec/run
L_ACCEPTED_RAYS=/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run
L_UID=$(id -u)
L_GID=$(id -g)
test "$L_UID" -ne 0
L_GPU=0
lifecycle() {
  L_STAGE=$1
  L_ACTION=$2
  L_PRIOR=${3-}
  test "$(git -C "$L_SOURCE" branch --show-current)" = paper/infrastructure
  test "$(git -C "$L_SOURCE" rev-parse HEAD)" = "$L_INFRA"
  test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$L_IMAGE"
  L_ARGS=()
  L_MOUNTS=()
  case "$L_STAGE" in
    geometry) L_SCIENCE=79252178e65914ad1acbc14790b1514f276bfea0 ;;
    material)
      L_SCIENCE=e1e859779301faced46d6476691e20b546c27ae0
      L_MOUNTS+=(--mount "type=bind,src=$L_GEOMETRY,dst=/inputs/geometry,readonly")
      L_MOUNTS+=(--mount "type=bind,src=$L_ACCEPTED_A,dst=/inputs/accepted-a,readonly")
      L_MOUNTS+=(--mount "type=bind,src=$L_ACCEPTED_RAYS,dst=/inputs/accepted-rays,readonly")
      L_ARGS+=(--geometry-root /inputs/geometry --mesh /inputs/geometry/meshes/bell_shape-best-150000-r128.ply)
      L_ARGS+=(--mesh-sha256 eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5)
      L_ARGS+=(--extraction-report /inputs/geometry/G-mesh-diagnostic.json)
      L_ARGS+=(--extraction-sha256 caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f)
      L_ARGS+=(--accepted-material-report /inputs/accepted-a/bell-material-diagnostic.json)
      L_ARGS+=(--accepted-ray-report /inputs/accepted-rays/bell-material-diagnostic.json)
      ;;
    *) return 2 ;;
  esac
  case "$L_ACTION" in
    save) test -z "$L_PRIOR" ;;
    restore|evaluate)
      test -n "$L_PRIOR"
      L_CHECKPOINT_SHA=$(sha256sum "$L_PRIOR/checkpoints/model.pth")
      L_CHECKPOINT_SHA=${L_CHECKPOINT_SHA%% *}
      L_PRODUCER_SHA=$(sha256sum "$L_PRIOR/lifecycle.json")
      L_PRODUCER_SHA=${L_PRODUCER_SHA%% *}
      L_MOUNTS+=(--mount "type=bind,src=$L_PRIOR,dst=/inputs/lifecycle,readonly")
      L_ARGS+=(--checkpoint-root /inputs/lifecycle --checkpoint /inputs/lifecycle/checkpoints/model.pth)
      L_ARGS+=(--checkpoint-sha256 "$L_CHECKPOINT_SHA" --producer-report /inputs/lifecycle/lifecycle.json --producer-sha256 "$L_PRODUCER_SHA")
      ;;
    *) return 2 ;;
  esac
  L_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-lifecycle.XXXXXXXX)
  mkdir "$L_BATCH/run" "$L_BATCH/empty-thesis"
  L_RUN_ID=$(python3 -B -c 'import uuid; print("bell-lifecycle-"+uuid.uuid4().hex)')
  printf 'stage=%s\naction=%s\ninfrastructure=%s\nscientific=%s\nimage=%s\ngpu=%s\nrun_id=%s\n' \
    "$L_STAGE" "$L_ACTION" "$L_INFRA" "$L_SCIENCE" "$L_IMAGE" "$L_GPU" "$L_RUN_ID" > "$L_BATCH/launch.txt"
  git -C "$L_SOURCE" diff HEAD --binary > "$L_BATCH/reviewed-worktree.patch"
  set +e
  docker run --rm --pull=never --init --user "$L_UID:$L_GID" \
    --gpus "device=$L_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$L_SOURCE,dst=/workspace,readonly" \
    --mount type=bind,src=/home/isavand/paper/data,dst=/inputs/datasets,readonly \
    --mount "type=bind,src=$L_BATCH/empty-thesis,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$L_BATCH/run,dst=/outputs" \
    "${L_MOUNTS[@]}" --workdir /workspace --entrypoint /usr/bin/timeout \
    "$L_IMAGE" --signal=TERM --kill-after=30s 1800 \
    /usr/bin/python3 -I -B /workspace/run_lifecycle_diagnostic.py \
    --stage "$L_STAGE" --action "$L_ACTION" --scientific-commit "$L_SCIENCE" \
    --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
    "${L_ARGS[@]}" --run-id "$L_RUN_ID" --execute \
    --acknowledge-trusted-state-and-diagnostic-save --confirm-isolated-runtime \
    --image-id "$L_IMAGE" --infrastructure-commit "$L_INFRA" \
    > "$L_BATCH/terminal.log" 2>&1
  L_EXIT=$?
  set -e
  printf '%s\n' "$L_EXIT" > "$L_BATCH/exit-code.txt"
  cat "$L_BATCH/terminal.log"
  printf 'Retained lifecycle evidence: %s\n' "$L_BATCH"
  test "$L_EXIT" -eq 0
  sha256sum "$L_BATCH/run/lifecycle.json"
  L_LAST_OUTPUT=$L_BATCH/run
}
```

Execute these six **separate actions**, reviewing each result before the next.
Each function invocation starts a new isolated container/process and allocates a
new owned directory. Do not paste the complete sequence as an unattended batch.
Both evaluation actions use the saved step2 checkpoint, not unsaved post-reload
step3 weights. No historical model is substituted.

```bash
lifecycle geometry save
L_G_SAVE=$L_LAST_OUTPUT
```

```bash
lifecycle geometry restore "$L_G_SAVE"
```

```bash
lifecycle geometry evaluate "$L_G_SAVE"
```

```bash
lifecycle material save
L_A_SAVE=$L_LAST_OUTPUT
```

```bash
lifecycle material restore "$L_A_SAVE"
```

```bash
lifecycle material evaluate "$L_A_SAVE"
```

`--validate-only` may replace execute/acknowledgements for metadata preflight; it
never loads camera pickle/checkpoint tensors or starts a model. Code/data/mesh/
accepted evidence/prior state remain read-only; thesis mount is deliberately empty.
Each action has1800s timeout+30s grace,64GiB/no extra swap,4CPUs,128PIDs,2GiB shm,
one selected A100 and no network. There is no GPU-memory cap. Full original A
ray-pool construction/rendering may exhaust the budget: retain failure evidence,
never reduce resolution/sample counts/defaults to obtain a passing result.

## Acceptance and remaining work

For each stage require completed schema-valid save manifest, exactly2 original
returns, unmodified original checkpoint format with stored2/best0 and exact file
hash; a fresh-process restore action with exact weights/buffers, Adam moments/
counter values **and types**, groups and scheduler identity; exactly1 successful
original post-reload update at index2, expected original LR, finite losses/active
gradients, nonzero parameter changes and advanced Adam state. G's scalar LR at2
is exactly the original expression `5e-4*(2/5000)` (typically
`2.0000000000000002e-7`), not a rewritten floating-point operation order.

Require G source0/currentG split and A source58 then11/currentA split, actual original
test_step/evaluator/metric calls,200²/800² outputs, finite returned channels,
lossless float/PNG files, explicit miss handling/encoding and frozen metric
comparison success. Model/optimizer/ray cursor must be unchanged by evaluation.
Check extension binary/runtime/source/input/output hashes, no unexpected checkpoints,
owned cleanup and retained failure records. Loaded extension modules are hashed
without triggering extra compilation; this is not a complete system-library hash
closure. Validate a report together with the exact referenced files, not status alone.

Native evidence is pending. RNG, loader and ray cursor continuation is not restored;
natural cadence/best selection/full NVS, convergence, historical producing identity,
research-grade G→A mesh provenance and bitwise uninterrupted training remain outside
this bounded acceptance. No scientific source/configuration or original evidence
is changed, and no native GPU/Docker execution is performed by Codex.

## CPU/static validation

Host Python3.9.12/Torch2.4.0 CPU fixtures:15 focused lifecycle tests pass. Full
infrastructure suite:283 total,279 passed,4 skipped,0 failed. Skips remain the
existing unavailable native OpenCV/transforms3d crop, Open3D point export, OpenCV
RAW resize, and native CPU PyMCubes/Trimesh export checks. New fixtures execute
unchanged original G/A trainer, scheduler, save/load, A test_step and metric
bodies with explicit toy models/BVH/shaders/loaders. They check finite state,
full typed optimizer round trips, actual bounded post-load updates, failed
restoration/no partial loading, wrong hashes/trust, leakage, original background/
channel behavior, independent metric scaling, shared-decorator collision and
owned cleanup. They are not native geometry/BRDF/GPU or Python3.8 acceptance.

Static definition/mount checks and5 Docker RUN shell parses pass; Python3.8 grammar
passes for182 source/definition files. All documented Bash blocks parse; local
Markdown links and whitespace pass. Real G/A metadata preflight passes with
native imports blocked, no original pickle loading and no run allocation.
377 baseline scientific/configuration/evidence/infrastructure/policy files match
the reviewed baseline, including the ordinary research handoff, all previous
smoke/mesh/metric implementations, snapshots/YAML and Dockerfile. Accepted A/ray
JSON and extracted PLY hashes remain unchanged. No Docker/GPU, original model/
camera/checkpoint deserialization, package/image/permission change or automatic
commit/push occurred. Synthetic CPU checkpoints are disposable trusted fixtures.
The static checker's existing incomplete build-manifest warning is separate
from the already working reviewed image; this task makes no new build-readiness
claim. Native save/restore/evaluation, resource requirements and convergence remain
unverified until the separately reviewed researcher-side evidence is accepted.
