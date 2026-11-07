# Historical bell camera-0 diagnostic preparation: 1D.4B-3C

**Implemented for review, not executed natively by Codex.** This separate runner
selects source camera **`"0"`** explicitly using the admitted historical bell
checkpoint. It retains maintained A science and original A split tables. The
existing camera-58 regression, reports and code remain unchanged.

Work began on `paper/infrastructure` at
`ef7e6b270137c1315c1ef01f303af910ee1dcbaf` with the uncommitted 1D.5A preparation
already present. That work is retained; this diagnostic neither executes it nor
changes its harness/protocol. Shared AGENTS/static-check updates add this scope.

The [frozen protocol](protocols/BELL_CAMERA0_DIAGNOSTIC_V1.md) contains the
source-level evaluation/step review, exact numerical policy and expected files.
[Admission evidence](evidence/bell-camera0-diagnostic-admission.json) records the
new researcher visual identification and pins the camera-0 files/montage. The
historical two-way split (validation 0, training 1–127) remains a hypothesis, not
producing-run attribution. Camera 58's higher current score is not by itself
proof of historical training membership or an explanation of the difference.

## Minimal adapter and checked invariants

[run_bell_camera_diagnostic.py](../run_bell_camera_diagnostic.py) exposes explicit
`evaluate` and CPU-only `compare` actions. `evaluate` requires source ID string
`0`, step 149999 or 150000, a unique run ID, trusted-checkpoint/execution/runtime
acknowledgements and image/source declarations. It has no split/branch, seed,
checkpoint, sample-count, resize, background or tolerance override.

[The adapter](../infrastructure/bell_camera_diagnostic.py) owns only its separate
camera selection table. It delegates to the original forward/test_step and
original downstream methods and uses the original camera preparation functions.
The real model's training/test/NVS lists and validation-info object are checked
unchanged. Selected getters record actual ID 0 for RGB/pose/K/mask/depth;
validation index 0 maps to this explicit table, while the real A validation table
still maps to 58. Root/global hooks must be absent; original child/weight_norm
hooks remain active. Original no-grad/default-CUDA handling and SDF gradient
requirements are retained, without an inference-mode shortcut.

Existing hash-bound metadata/architecture evidence and the original strict loader
are reused. Checkpoint SHA-256 remains
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`;
stored step 150000 is not rewritten when evaluating at 149999. Loaded weights/
buffers are fingerprinted before/after inference. Torch 1.8 deserializes the
trusted full pickle container on CPU inside the prepared isolated runtime;
optimizer state is not restored. Codex never performs this load or constructs the
native model. Invalid hashes/paths/config/source fail before native execution.

The source review finds identical active step branches at 149999 and 150000:
freeze/occlusion/early-SDF thresholds do not change, validation annealing/perturb
arguments remain fixed at zero, and the AppShadingNetwork accepts/forwards step
without using it in the light calculation. Output filenames/step metadata differ.
Actual GPU equality remains a test, not an observation. The paired comparison
reports any differences without deciding they were caused by step rather than
native nondeterminism; no same-step repeat is included in this two-render budget.

[The CPU comparison](../infrastructure/bell_camera_comparison.py) retains the
old RGB thresholds only for the same-camera step control. Historical PSNR/SSIM
comparison is descriptive, including full versus five-decimal values, with no
new acceptance tolerance. Historical GT is compared using the frozen JPEG crop,
without resizing or compression compensation. Its decoded panel is never used
as the scoring GT. Numerical/JPEG proximity cannot authenticate producing code.

## Exact researcher-side commands (for review only)

Freeze/review the source and protocol first. This sequence performs **two**
independent single-view runs, camera 0 at 149999 then at 150000, and one CPU-only
saved-output comparison. No new GPU/training execution is authorized here.
Parent logs/returned exit codes are outside the initially empty run roots.

```bash
set -eu
CAM_SOURCE=/home/isavand/paper/Nerofactor-paper
CAM_DATA=/home/isavand/paper/data
CAM_THESIS=/home/isavand/paper/final_results
CAM_METADATA=/home/isavand/paper/evidence/bell-metadata.xIvZ9k0z/checkpoint-metadata.json
CAM_REFERENCE=/home/isavand/paper/evidence/bell-reference.eWPhSaW8/reference-signature.json
CAM_ARCHITECTURE=/home/isavand/paper/evidence/bell-comparison.UXWN3DqC/architecture-comparison.json
CAM_IMAGE=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$CAM_IMAGE" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
CAM_UID=$(id -u)
CAM_GID=$(id -g)
test "$CAM_UID" -ne 0
CAM_GPU=0
CAM_COMMIT=$(git -C "$CAM_SOURCE" rev-parse HEAD)
CAM_TOKEN=$(python3 -B -c 'import uuid; print(uuid.uuid4().hex)')
CAM_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-camera0.XXXXXXXX)
mkdir "$CAM_BATCH/primary" "$CAM_BATCH/step-control" "$CAM_BATCH/comparison"
printf 'source=%s\nimage=%s\ngpu=%s\nuid=%s\ngid=%s\n' \
  "$CAM_COMMIT" "$CAM_IMAGE" "$CAM_GPU" "$CAM_UID" "$CAM_GID" > "$CAM_BATCH/launch.txt"

camera0_run() {
  CAM_NAME=$1
  CAM_STEP=$2
  set +e
  docker run --rm --pull=never --init --user "$CAM_UID:$CAM_GID" \
    --gpus "device=$CAM_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$CAM_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$CAM_DATA,dst=/inputs/datasets,readonly" \
    --mount "type=bind,src=$CAM_THESIS,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$CAM_METADATA,dst=/inputs/metadata/checkpoint-metadata.json,readonly" \
    --mount "type=bind,src=$CAM_REFERENCE,dst=/inputs/reference/reference-signature.json,readonly" \
    --mount "type=bind,src=$CAM_ARCHITECTURE,dst=/inputs/comparison/architecture-comparison.json,readonly" \
    --mount "type=bind,src=$CAM_BATCH/$CAM_NAME,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$CAM_IMAGE" --signal=TERM --kill-after=30s 900 \
    /usr/bin/python3 -I -B /workspace/run_bell_camera_diagnostic.py evaluate \
    --source-id 0 --eval-step "$CAM_STEP" --run-id "$CAM_NAME-$CAM_TOKEN" \
    --image-id "$CAM_IMAGE" --source-commit "$CAM_COMMIT" --execute \
    --acknowledge-trusted-checkpoint --confirm-isolated-runtime \
    > "$CAM_BATCH/$CAM_NAME-terminal.log" 2>&1
  CAM_STATUS=$?
  set -e
  printf '%s\n' "$CAM_STATUS" > "$CAM_BATCH/$CAM_NAME-exit-code.txt"
  cat "$CAM_BATCH/$CAM_NAME-terminal.log"
  return "$CAM_STATUS"
}

camera0_run primary 149999
camera0_run step-control 150000
CAM_PRIMARY_SHA=$(sha256sum "$CAM_BATCH/primary/camera-diagnostic.json" | cut -d ' ' -f 1)
CAM_CONTROL_SHA=$(sha256sum "$CAM_BATCH/step-control/camera-diagnostic.json" | cut -d ' ' -f 1)
set +e
docker run --rm --pull=never --init --runtime=runc --user "$CAM_UID:$CAM_GID" \
  --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true \
  --memory=2g --memory-swap=2g --cpus=1 --pids-limit=64 --ulimit core=0 \
  --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void --env NVIDIA_DRIVER_CAPABILITIES= \
  --env HOME=/outputs --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$CAM_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$CAM_BATCH/primary,dst=/inputs/primary,readonly" \
  --mount "type=bind,src=$CAM_BATCH/step-control,dst=/inputs/control,readonly" \
  --mount "type=bind,src=$CAM_BATCH/comparison,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout \
  "$CAM_IMAGE" --signal=TERM --kill-after=10s 120 \
  /usr/bin/python3 -I -B /workspace/run_bell_camera_diagnostic.py compare \
  --primary /inputs/primary/camera-diagnostic.json --primary-sha256 "$CAM_PRIMARY_SHA" \
  --control /inputs/control/camera-diagnostic.json --control-sha256 "$CAM_CONTROL_SHA" \
  --output-root /outputs > "$CAM_BATCH/comparison-terminal.log" 2>&1
CAM_STATUS=$?
set -e
printf '%s\n' "$CAM_STATUS" > "$CAM_BATCH/comparison-exit-code.txt"
cat "$CAM_BATCH/comparison-terminal.log"
test "$CAM_STATUS" -eq 0
printf 'Preserved camera-0 diagnostic evidence: %s\n' "$CAM_BATCH"
```

Per native process: 900 seconds plus 30-second termination grace, four CPUs,
64 GiB host memory/no extra swap, one explicitly selected GPU; no GPU memory cap
is claimed. Each process preserves the original full initialization, not a
reduced-data convenience path. CPU comparison: 120 seconds plus 10-second grace,
one CPU, 2 GiB, no exposed GPU and no dataset/checkpoint mount. All source/input
mounts and root filesystems are read-only; only the chosen output is writable.
No source/configuration, Docker settings or host permissions are changed.

Expected native exit 0 means one valid camera-0 diagnostic completed, **not**
historical metric reproduction. Expected comparison result is
`within_frozen_step_comparison_bounds` if the pair meets unchanged RGB criteria;
`step_pair_mismatch` exits 1 and reports differences, invalid evidence exits 2.
No historical metric/JPEG closeness is required for completion. Preserve all
failed/partial outputs and outer exit records; abrupt shell termination may omit
an exit record. Never fabricate an exit code, infer a signal from truncation,
repeat/tune runs, change the camera/step/checkpoint or widen thresholds to finish.

## Evidence files and limitations

Each run records source ID 0 versus iterator index 0, old A validation ID 58,
untouched A split/table identity, observed getters, checkpoint stored step versus
evaluation step, strict architecture/state checks, unchanged weight/buffer hashes,
input identities, effective settings, runtime/backend/package identities and RNG.
Lossless prediction/GT/mask arrays and PNGs, original validation montage, metrics,
JPEG-derived historical panel and signed GT difference are separately named.
Owned caches/temp/logs are isolated; no checkpoint or historical output is written.

The new admission classifies the researcher visual identity and two-way split
hypothesis explicitly. It does not establish historical training membership,
full producing settings, original preprocessing/library versions or GPU RNG state.
Native GPU behavior, historical metric agreement and JPEG differences remain
unexecuted here. An unexpected original evaluator/scientific failure is reported
as a blocker, never repaired silently. Existing image-58 evidence stays the
accepted A regression and is not overwritten or relabeled by this diagnostic.

## CPU/static validation

The complete infrastructure suite ran **197 tests: 194 passed, 0 failed,
3 skipped**, including **14 new diagnostic tests**. Existing skips cover the
OpenCV/transforms3d crop, Open3D point-cloud and OpenCV RAW-resize fixtures whose
optional dependencies are unavailable. No new diagnostic test was skipped.

CPU fixtures exercise original forward/test_step/camera-preparation method bodies
with explicit CPU stand-ins, internal SDF gradients under no-grad, child hooks,
getter IDs, unchanged split tables/weights/RNG, both steps, strict report/artifact
validation, numeric mismatches, JPEG diagnostics, output collision/path checks,
hash gates and failure cleanup. They do not establish native renderer execution.

Validation used Python 3.9.12, CPU Torch 2.4.0+cu121 operations, NumPy 1.21.5,
scikit-image 0.19.2 and Pillow 9.0.1. The static checker accepted all **85 Python
files under Python 3.8 grammar** and checked Docker definition shell/mount rules;
this is not execution under Python 3.8/Torch 1.8.1. CLI help, documented Bash
syntax, local documentation links and Git whitespace checks passed.

Scientific source, legacy configurations, image-58 code/evidence and prior 1D.5A
harness/protocol files remain unchanged. Shared AGENTS/static-check edits only
add this milestone. Codex neither deserialized historical camera/checkpoint
payloads nor executed Docker/GPU commands, inference or training. Native results,
historical metric agreement and decoded-JPEG differences remain pending review
and separately authorized researcher execution.
