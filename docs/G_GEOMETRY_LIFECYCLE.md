# G geometry lifecycle: 1D.6A

Implementation and CPU/static preparation only. **No native compatibility,
validation, optimizer load, roundtrip or extraction has been executed by Codex.**
The [accepted two-update evidence](evidence/G_GEOMETRY_TRAINING_ACCEPTANCE.md)
has a narrower scope. Original A evidence remains separately labeled/unchanged.

G is `aniso_geometry_safe@79252178e65914ad1acbc14790b1514f276bfea0`;
material A remains `anisotropic@e1e859779301faced46d6476691e20b546c27ae0`.
The exact snapshots are verified by the immutable
[source manifest](../scientific_sources/manifest-v1.json), SHA
`39b9baecce2aab6fc62ec57bd6f45e453d8e9d68215f6af3f25b8ed118c5a3f6`.
The [source audit](evidence/g-validation-checkpoint-source-audit-v1.json)
records full file hashes, byte counts, function line ranges and AST hashes for
trainer, validation, tools, metrics, renderer, field, extraction and dataset
dependencies. No scientific file, YAML, old protocol or source snapshot is edited.

## Exact lifecycle audit

| Scope | Original G behavior | New acceptance boundary |
| --- | --- | --- |
| Initialization | Trainer shallow defaults merge; Torch/NumPy/Python seed6033; dataset then network; original loaders/losses/metrics/Adam factory | Exact original source, effective scopes, full preprocessing; storage redirection only |
| Direct validation | `ValidationEvaluator` calls eval, loader/to_cuda, sets eval/step, original forward under no_grad, then original losses/metrics, concatenates arrays and mean-PSNR key | Strictly admitted checkpoint; actual source0 reads; network/cursor unchanged; zero updates/checkpoint writes |
| Serialization | `_save_model` writes step, best_para, network_state_dict, optimizer_state_dict | Separate manual G-produced save/load test; no historical optimizer implicitly restored |
| Natural training | Train/backward/Adam, optional train log, validation, strict best selection/save, val log, periodic save, progress update | Not tested here; separate future approval to reach interval1000 |
| Extraction | Explicit original `model.pth`, stored step, strict load, SDF grid/marching cubes, Trimesh PLY | Audit/design only, separate checkpoint admission and future reviewed mesh protocol |

Original G references are under `scientific_sources/geometry_g/`:

- `train/trainer.py:40–101`: original initialization; `:103–188`: original loop.
  Bell val/save intervals are both1000 and total_step300000. Validation runs when
  `(step+1)%val_interval==0` or final step, receives zero-based `step`, and selects
  using the **last** validation set's key. Only `val_para > best_para` saves best
  at stored `step+1`. Val logging follows at `step+1`; periodic `model.pth` save
  is separate and later. Loop999 therefore validates999 and saves1000; the
  historical hypothesis is validation149999/save150000.
- `train/trainer.py:190–223`: `_load_model` implicitly checks its `model.pth`
  attribute, uses full `torch.load` without map_location, default strict network
  load and optimizer restoration, returning best/start. No RNG, producing source
  or split identity is saved. The new historical diagnostics never call this
  natural restore. Only the manual roundtrip calls it on its own newly saved file.
  `_log_data` averages arrays/tensors; original Logger emits text at five decimal
  places and TensorBoard under its storage directory. Direct validation JSON
  does not pretend that natural scheduling/logging occurred.
- `train/train_valid.py`: original evaluator uses no_grad around forward; SDF
  gradient in `network/field.py` explicitly enables gradients and retains/creates
  the graph. Replacing this with inference_mode is invalid. Original renderer
  forward switches default tensor type to CUDAFloat and back on normal return;
  failures terminate the isolated process and remain failed evidence.
- `dataset/database.py:574–590`, `dataset/train_dataset.py`: two-way unshuffled
  split seeds Python6033, training ordered1–127, validation0; DummyDataset yields
  index0. Both actual split calls and original getters are externally observed.
  Full original 800² preprocessing/training-table construction remains; original
  test_step downsampling produces200², not a new low-resolution shortcut.
- `network/renderer.py:496–555`: test_step obtains source-indexed depth/mask,
  original downsample/intrinsics/linear-color path and512-ray rendering chunks,
  perturb0/anneal0/is_trainFalse. It returns RGB, GT/depth/mask and diagnostics
  then zero_grad. `val_geometry=False` stays unchanged; no validation mesh writer.
- `network/metrics.py`: original uint8 conversion, float32 full-image PSNR and
  win11 SSIM. Original montage goes to
  `data/train_vis/bell_shape-val/149999-index-0.jpg`, redirected by owned worktree
  symlink to `/outputs/renders/validation/bell_shape-val/…`. No new scoring mask.

## Checkpoint identities and independent gates

| Artifact | SHA-256 | Status |
| --- | --- | --- |
| `thesis:model/bell_shape/model_best.pth` | `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc` | Selected diagnostic input; G compatibility unexecuted |
| `thesis:model/bell_shape/model.pth` | `838c6fec3e18999554bc04951b6643acc1c5cc24c95cfbeda7f81ba6830e732a` | Opaque-byte inventory only; not substituted or loaded |

Both are29,966,819 bytes at inspection. Exact stat identities and source audit
are in the JSON audit. Historical best stored150000/best23.787122565451323 and
137 tensors were previously inspected in an authorized runtime. G's expected
tensor catalog must be constructed/compared afresh, not assumed equal to A's.

The new entry point `run_g_geometry_diagnostic.py` has four independent actions:

1. [Compatibility](protocols/G_CHECKPOINT_COMPATIBILITY_V1.md): pinned input,
   strict schema/key/shape/dtype/finite gate and strict load into original G
   extraction-mode network; no forward, optimizer restore or resume.
2. Optional **optimizer** diagnostic, separately acknowledged and dependent on
   reviewed compatibility: inspect mappings/moments, load disposable optimizer,
   record actual overriding of G settings, no update. Library load success does
   not prove semantic or producing-run compatibility. No state repair.
3. [Camera-0 validation](protocols/G_CAMERA0_VALIDATION_V1.md): reviewed SHA-bound
   compatibility, original G dataset/evaluator/losses/metrics, exactly one view
   at149999, unchanged state. Report historical comparisons descriptively.
4. [Manual roundtrip](protocols/G_MANUAL_CHECKPOINT_ROUNDTRIP_V1.md): separate
   fresh two-update G prefix, original save/load with populated Adam state,
   step2/best0.0; no post-load update and no natural cadence claim.

`--validate-only` performs paths, config, evidence and opaque hash checks; it
does not allocate, import Torch/scientific modules, load checkpoint pickle or
construct a model. Unique run directories reject collisions. Only native
`--execute` with explicit trust/action/runtime gates can do native work, in one
cold G namespace with no scientific fallback to A. Every native scope records
the effective trainer/renderer/shader configuration where available; compatibility
uses training=False and validation uses full original training initialization.

## Filesystem writers and evidence

All original scene files link to read-only mounts in an owned per-run filesystem
view. Original missing derived caches are created only inside that view; existing
source caches retain read-only input semantics and cannot be overwritten.
Identity records include source files/config/source/package identities; fresh
outputs prevent cross-run cache reuse. Dataset and thesis mounts remain read-only.
Camera pickles are researcher-trusted and only read by original G in native runs;
Codex does not deserialize them. No checkpoint path is inferred/resumed.

Original Trainer constructor's `data/model/bell_shape` directory is owned inside
the workspace; I/O attributes subsequently route logs/checkpoints to run-owned
directories. Validation does not call `_init_logger` or Trainer.run; the original
metric montage writer and all optional extraction writers remain distinct.
The external read observer forbids Trainer.run/Adam.step/backward/save in
read-only scopes, records original calls without patching methods, and closes
owned validation workers/progress. Original SDF autograd.grad is allowed.

Wrapper writers are `/outputs/G-diagnostic.json`, original YAML, cache identity,
runtime HOME/tmp/JIT caches, console log, arrays, metrics and lossless diagnostic
PNGs. Original montage writes under `/outputs/renders/validation`. Compatibility
and optimizer modes do not load scene images; their workspace contains only
read-only assets/configs. Manual roundtrip alone writes one owned checkpoint.
Outer terminal/exit records live beside the fresh run. Original/other-run writes
are blocked by mounts, output containment and collision gates. No remaining
reachable enabled validation writer is intentionally outside owned outputs;
native write-isolation/failure-path behavior remains to be verified.

Every report is diagnostic-only, with protocol/source/file/runtime identities,
checkpoint SHA, effective scopes, actual read provenance where applicable,
state/output hashes, failure phase and explicit exclusions. Timeout/SIGKILL can
leave `status=running`; retain outer terminal/exit and treat it as incomplete,
never accepted. A source/image declaration is distinguished from independently
inspected identity. No report proves historical producer, split or final status.

## Mesh extraction audit and future design (no execution)

Original G `extract_mesh.py` selects **`model.pth`** by name, constructs original
shape network(training=False), uses stored step and default strict network load,
eval/CUDA/defaultCUDAFloat, bbox `[-1,-1,-1]` to `[1,1,1]`, no_grad and SDF query.
CLI default resolution512, isovalue0. `extract_fields` chunks64 per axis, dense
float32 field, **norm(points)>=1 -> outside_val=+1** (unit-sphere constraint).
Original PyMCubes marching cubes returns triangles; vertices map by
`v/(resolution-1)*(bbox_max-bbox_min)+bbox_min`. No extra scene normalization,
rescale, axis permutation, or explicit winding flip is authorized.

Trimesh(vertices,triangles) uses its original default processing before PLY export
to `data/meshes/<name>-<stored-step>.ply`. Explicit normal computation/outward
orientation is not specified by the G code; winding/normal/export details depend
on pinned PyMCubes and Trimesh implementations. Record those versions and
processing behavior before a material handoff, rather than assert an unverified
normal convention. Preserve raw arrays and final mesh hashes in a future design.

Future bounded diagnostic proposal: admit the **alternate model.pth** identity,
metadata and G compatibility separately, keep original extraction functions,
explicit coarse128 resolution (distinct from original512), threshold0/unit box/
sphere mask, fresh owned PLY plus raw vertex/triangle arrays and settings/hash
report. Resolution512 entails~134m samples/512MiB field;128~2.1m/8MiB. A future
protocol must freeze time/GPU/CPU/RAM budgets and inspect winding/scale/processing
with the actual pinned runtime. No native extraction adapter is authorized here.

Best-checkpoint extraction would be a separate explicitly selected protocol;
never silently supply best to the legacy model.pth CLI. Random/two-update smoke
weights are diagnostic-only and never valid research meshes. G-to-A handoff
requires explicit file/hash, admitted producing G run/source/checkpoint,
scene coordinates/scale, winding/normals and unchanged A preprocessing. A
historically unattributed bell mesh or short-prefix checkpoint does not satisfy it.

## First researcher-side commands (review before execution)

No Docker command below was executed by Codex. Use the reviewed checkout after
review, preserve its commit/diff, and the already installed immutable image;
no pull/build/install. These commands expose one GPU for original G construction.
Compatibility is first; review its report before running validation or optimizer.
Do not run commands concurrently into the same output.

```bash
set -eu
G_SOURCE=/home/isavand/paper/Nerofactor-paper
G_DATA=/home/isavand/paper/data
G_THESIS=/home/isavand/paper/final_results
G_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$G_IMAGE"
G_INFRA=$(git -C "$G_SOURCE" rev-parse HEAD)
G_UID=$(id -u)
G_GID=$(id -g)
test "$G_UID" -ne 0
G_GPU=0
G_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-G-lifecycle.XXXXXXXX)
mkdir "$G_BATCH/compatibility"
printf 'infrastructure=%s\nscientific=%s\nimage=%s\ngpu=%s\n' \
  "$G_INFRA" 79252178e65914ad1acbc14790b1514f276bfea0 "$G_IMAGE" "$G_GPU" > "$G_BATCH/launch.txt"
git -C "$G_SOURCE" diff HEAD --binary > "$G_BATCH/reviewed-worktree.patch"

# Each invocation gets a new output directory. Arguments are explicit CLI flags.
g_diagnostic() {
  G_DEST=$1
  G_SECONDS=$2
  shift 2
  docker run --rm --pull=never --init --user "$G_UID:$G_GID" \
    --gpus "device=$G_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$G_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$G_DATA,dst=/inputs/datasets,readonly" \
    --mount "type=bind,src=$G_THESIS,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$G_BATCH/compatibility,dst=/inputs/compatibility,readonly" \
    --mount "type=bind,src=$G_DEST,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$G_IMAGE" --signal=TERM --kill-after=30s "$G_SECONDS" \
    /usr/bin/python3 -I -B /workspace/run_g_geometry_diagnostic.py \
    --scientific-commit 79252178e65914ad1acbc14790b1514f276bfea0 \
    --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
    --execute --confirm-isolated-runtime --image-id "$G_IMAGE" --infrastructure-commit "$G_INFRA" "$@"
}

set +e
g_diagnostic "$G_BATCH/compatibility" 300 --action compatibility --run-id g-bell-compatibility \
  --acknowledge-trusted-checkpoint > "$G_BATCH/compatibility-terminal.log" 2>&1
G_EXIT=$?
set -e
printf '%s\n' "$G_EXIT" > "$G_BATCH/compatibility-exit-code.txt"
cat "$G_BATCH/compatibility-terminal.log"
test "$G_EXIT" -eq 0
sha256sum "$G_BATCH/compatibility/G-diagnostic.json"
printf 'Review compatibility evidence in %s before proceeding.\n' "$G_BATCH"
```

The compatibility output is also mounted read-only under `/inputs/compatibility`
in that first invocation, but the action does not read that alias; it writes only
the separately writable `/outputs`. Later actions consume the completed report
only through the read-only alias. Both original dataset/thesis stay read-only.
No GPU-memory cap is claimed. Full original validation initialization can use
substantial host/GPU memory; preserve out-of-memory/timeout failures without
reducing scientific sizes.

After compatibility review, in the same shell and unchanged reviewed source:

```bash
G_COMPAT_SHA=$(sha256sum "$G_BATCH/compatibility/G-diagnostic.json")
G_COMPAT_SHA=${G_COMPAT_SHA%% *}
mkdir "$G_BATCH/validation"
set +e
g_diagnostic "$G_BATCH/validation" 900 --action validation --run-id g-bell-camera0 \
  --acknowledge-trusted-checkpoint \
  --compatibility-report /inputs/compatibility/G-diagnostic.json --compatibility-sha256 "$G_COMPAT_SHA" \
  > "$G_BATCH/validation-terminal.log" 2>&1
G_EXIT=$?
set -e
printf '%s\n' "$G_EXIT" > "$G_BATCH/validation-exit-code.txt"
cat "$G_BATCH/validation-terminal.log"
test "$G_EXIT" -eq 0
sha256sum "$G_BATCH/validation/G-diagnostic.json"
```

Expected compatibility result: complete with both full tensor catalogs, no
mismatches and strict load, no forward/update/checkpoint. Any mismatch is a
blocker, not a license to load partially. Expected validation evidence includes
original getter/split provenance, source0/step149999, full RGB/mask arrays,
lossless PNGs, original JPEG, metrics, historical differences and exact output
hashes. No expected G PSNR/SSIM value is asserted before execution.

The optional optimizer diagnostic requires separate review/acknowledgement:
new directory `$G_BATCH/optimizer`, same function/budget300, `--action optimizer
--run-id g-bell-optimizer --acknowledge-trusted-checkpoint
--acknowledge-optimizer-diagnostic --compatibility-report
/inputs/compatibility/G-diagnostic.json --compatibility-sha256 "$G_COMPAT_SHA"`.
Capture outer terminal/exit exactly as above. No optimizer.step is allowed.
Manual roundtrip instead requires a new batch/output and **empty read-only
thesis** mount, action roundtrip/manual acknowledgement, budget900; follow its
independent protocol, not this historical-thesis function.

Failure diagnostics: examine phase/error in G-diagnostic.json and console.log,
outer exit124/137 or device/JIT/import errors. Hash/source/catalog/split guards
must stop, not fall back. Never install/modify science/retune a score to recover.
Compatibility or validation success does not authorize natural1000 training,
G historical continuation, extraction, material handoff or full experiments.

## CPU/static validation recorded in 1D.6A

Host Python3.9.12/Torch2.4.0+cu121, **CPU operations only**, with disposable
synthetic checkpoints/datasets and explicit native operator/I/O stand-ins:

- `python -B -m unittest discover -s tests/infrastructure -v`: **238 tests,
  235 passed, 0 failures, 3 skips**; includes22 new G lifecycle tests. Existing
  skips are native OpenCV/transforms3d crop, Open3D downsample/export and OpenCV
  RAW resize because those host packages are unavailable.
- Original G forward/test_step, SDF gradient, validation dataset/evaluator and
  metric source bodies execute in CPU fixtures; observed and unobserved metrics
  agree exactly. Original G save/load methods roundtrip populated toy Adam state;
  historical optimizer load fixtures expose eps/AMSGrad overrides without updates.
  These are **not native G model/preprocessing or Torch1.8 acceptance**.
- `python -B docker/paper/static_checks.py`: passed definition/mount/pre-install
  checks, five Docker RUN shell parses and Python3.8 grammar for165 files.
  Its definition-artifact readiness warning concerns the incomplete checked-in
  build inventory, not a new assessment of the already working researcher image.
- New documented shell blocks parse with `bash -n`; old330 versioned scientific,
  configuration, protocol and evidence files match the reviewed baseline bytes.
  Both historical checkpoint opaque hashes/sizes, original evidence reports and
  16 declared outputs are unchanged. New JSON/Markdown links and whitespace pass.

Python3.8 runtime, Torch1.8 native construction, real G validation/read provenance,
historical optimizer semantics and manual native roundtrip remain pending the
separate researcher protocols. No historical checkpoint/camera was deserialized,
and no native scientific/GPU/Docker workload, extraction or installation ran here.
