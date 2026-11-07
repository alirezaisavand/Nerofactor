# Bounded bell geometry training preparation: 1D.5A

**Prepared for review; no native training or Docker/GPU command executed by
Codex.** This adds a separate fresh two-Adam-call entry point around the unchanged
maintained Trainer and geometry implementation. Review baseline:
`ef7e6b270137c1315c1ef01f303af910ee1dcbaf`, clean `paper/infrastructure` before edits.
[The frozen protocol](protocols/GEOMETRY_TWO_UPDATE_V1.md) defines the exact stop,
scientific settings, comparison thresholds, evidence schema and limitations.

The previous [bell inference result](evidence/BELL_INFERENCE_REGRESSION_SUCCESS.md)
verified zero R/R-repeat/C differences for a checkpoint-based A validation view.
It did not execute Adam/backward or establish historical training provenance.
[Historical attribution](evidence/BELL_EVALUATION_PROVENANCE.md) remains separate:
32.8498 dB is not a historically verified thesis score, producing branch/split are
unknown, and A view 58 would be G training under the inspected G split.

## Audited native path and selected harness

| Existing implementation | Preserved operation / observation |
| --- | --- |
| `train/trainer.py:119–130` | Shallow default merge; Torch then NumPy then Python seed 6033; owned output selection and effective trainer recording |
| `Trainer._init_dataset`, `dataset/train_dataset.py` | Real dummy datasets, train/val loaders, batch 1/workers 8, original shuffle/collate/length; validation dataset's A split call |
| `Trainer._init_network`, `network/renderer.py:250–325` | Actual shape constructor with cache and explicit A split shim; native model modules, renderer's split call, 800x800 image preparation, full train-ray pool/shuffle; real loss objects |
| `train/lr_common_manager.py` | Actual Adam construction (`lr=1e-3`, eps 1e-4, AMSGrad, native betas); original warm-up/cosine callable, hard-coded endpoint 100000 |
| `Trainer.run:137–185` | Fresh restore returns (0,0) through the existing new-run storage mode, actual loader iterator, step 0/1, original train mode/LR/zero-grad/forward/loss mean aggregation/backward/0.7 clip/Adam |
| `Trainer.run:186–216` | Original log/save/validation intervals unchanged; neither 1000-step interval is reached. No checkpoint save, validation dispatch or final NVS is expected |
| `network/renderer.py:578–608` | Original 512-ray slice/cursor, possible reshuffle branch, camera/ray conversion, native sampling/rendering and losses; no mathematical edits |
| Historical `anisotropic:train/trainer.py` / G trainer | Inspected read-only; A clipping differs from G, split rules differ. New smoke is maintained A, not a G/ancestor reproduction |

Exact source hashes are checked against the committed structural/inference
admission manifests before execution and again afterward. The actual Trainer,
loaders, scientific modules, losses, scheduler and optimizer files are unchanged.
AST-derived trace boundaries reference their existing statements; no source is
rewritten and the training loop is not reimplemented.

**LR nuance:** first Adam call at step 0 uses zero LR. It must still execute
backward/clipping/Adam and advance moment/counter state, with no parameter
change. Step 1's LR is about 1e-7 and must change parameters. Calling this two
parameter-changing updates would be incorrect; never add a third call to hide
that difference or alter warm-up to make both calls nonzero.

The external observer at the first line after the second optimizer return raises
a private stop and closes owned resources. It refuses an existing trace/debugger,
wrong code, wrong configuration, invalid progress or reused controller. If
statement layout/runtime compatibility or native scientific execution fails,
record the blocker; no fallback to unbounded training or algorithm repair exists.
This observer/stop design is explicit implementation **for code review**, not an
already executed native validation or permission to launch it.

## Entry points and outputs

- [run_geometry_training_smoke.py](../run_geometry_training_smoke.py): explicit
  `--execute`, arm, unique ID, source/image declarations and isolated-runtime
  acknowledgement. Fixed budget two; no step/seed/LR/sampling/checkpoint/resume
  override. Native backend accepts only reviewed Python 3.8/Torch 1.8.1+cu111,
  CUDA 11.1 and one GPU. R independently resolves legacy FullLoader YAML; C uses
  the existing schema-1 geometry envelope and PathResolver.
- [bounded_training.py](../infrastructure/bounded_training.py): single-frame
  tracing, counted stop, worker/progress/writer cleanup; no native math operators.
- [training_probe.py](../infrastructure/training_probe.py): detached diagnostic
  copies, finite checks, actual loss/gradient/parameter/Adam/RNG/sampling evidence.
- [geometry_training_smoke.py](../infrastructure/geometry_training_smoke.py):
  pinned sources/inputs, config routes, fresh storage, runtime isolation and report.
- [compare_geometry_training_smoke.py](../compare_geometry_training_smoke.py) /
  [training_comparison.py](../infrastructure/training_comparison.py): CPU JSON-only
  schema/hash validation and gated reference/candidate comparisons. No Torch,
  model or checkpoint loading in this comparison path.

Each fresh mounted `/outputs` gets `training-smoke.json`, original/legacy/effective
configs, existing `geometry.json`, `logs/console.log` and native TensorBoard event
metadata. Runtime/home/JIT/preprocessing/temp products stay under that output.
No training tensor arrays or checkpoint files are exported. `checkpoints/` remains
empty. No generated original-data products are modified. Initial scientific/
config/source validation, source hashing and input identity checks precede native
imports, model construction and fresh run allocation.

Inputs use `/inputs/datasets` read-only; **historical results/checkpoints are not
mounted**. The required `/inputs/thesis` resolver root is a separately created,
empty read-only placeholder. This makes accidental historical initialization
unavailable, in addition to the fixed new-run storage mode. The original bell
configuration and original effective defaults remain authoritative; host paths
are provided only by the launch command.

## Researcher-side bounded commands (not executed by Codex)

Review/freeze the implementation and protocol first. These commands preserve the
existing image, source and input mounts, select one GPU, disable networking and
use one writable output per arm. They stop on any failed arm/comparison. Do not
turn them into a retry/seed-selection loop. The first R is already the two-call
native smoke; R-repeat and C are gated follow-ups, not full training.

```bash
set -eu
TRAIN_SOURCE=/home/isavand/paper/Nerofactor-paper
TRAIN_DATA=/home/isavand/paper/data
TRAIN_IMAGE=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$TRAIN_IMAGE" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
TRAIN_UID=$(id -u)
TRAIN_GID=$(id -g)
test "$TRAIN_UID" -ne 0
TRAIN_GPU=0
TRAIN_COMMIT=$(git -C "$TRAIN_SOURCE" rev-parse HEAD)
TRAIN_TOKEN=$(python3 -B -c 'import uuid; print(uuid.uuid4().hex)')
TRAIN_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-training-two-updates.XXXXXXXX)
mkdir "$TRAIN_BATCH/empty-thesis" "$TRAIN_BATCH/r" "$TRAIN_BATCH/r-repeat" "$TRAIN_BATCH/c"
printf 'source=%s\nimage=%s\ngpu=%s\nuid=%s\ngid=%s\n' \
  "$TRAIN_COMMIT" "$TRAIN_IMAGE" "$TRAIN_GPU" "$TRAIN_UID" "$TRAIN_GID" > "$TRAIN_BATCH/launch.txt"

training_arm() {
  TRAIN_ARM=$1
  TRAIN_NAME=$2
  set +e
  docker run --rm --pull=never --init --user "$TRAIN_UID:$TRAIN_GID" \
    --gpus "device=$TRAIN_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=256 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$TRAIN_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$TRAIN_DATA,dst=/inputs/datasets,readonly" \
    --mount "type=bind,src=$TRAIN_BATCH/empty-thesis,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$TRAIN_BATCH/$TRAIN_NAME,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$TRAIN_IMAGE" --signal=TERM --kill-after=30s 900 \
    /usr/bin/python3 -I -B /workspace/run_geometry_training_smoke.py \
    --arm "$TRAIN_ARM" --run-id "$TRAIN_NAME-$TRAIN_TOKEN" \
    --source-commit "$TRAIN_COMMIT" --image-id "$TRAIN_IMAGE" --execute --confirm-isolated-runtime \
    > "$TRAIN_BATCH/$TRAIN_NAME-terminal.log" 2>&1
  TRAIN_COMMAND_STATUS=$?
  set -e
  printf '%s\n' "$TRAIN_COMMAND_STATUS" > "$TRAIN_BATCH/$TRAIN_NAME-exit-code.txt"
  cat "$TRAIN_BATCH/$TRAIN_NAME-terminal.log"
  return "$TRAIN_COMMAND_STATUS"
}

training_compare() {
  TRAIN_COMPARE_NAME=$1
  mkdir "$TRAIN_BATCH/$TRAIN_COMPARE_NAME"
  TRAIN_R_SHA=$(sha256sum "$TRAIN_BATCH/r/training-smoke.json" | cut -d ' ' -f 1)
  TRAIN_REPEAT_SHA=$(sha256sum "$TRAIN_BATCH/r-repeat/training-smoke.json" | cut -d ' ' -f 1)
  TRAIN_C_MOUNTS=()
  TRAIN_C_ARGS=()
  if test "${2:-}" = with-candidate; then
    TRAIN_C_SHA=$(sha256sum "$TRAIN_BATCH/c/training-smoke.json" | cut -d ' ' -f 1)
    TRAIN_C_MOUNTS=(--mount "type=bind,src=$TRAIN_BATCH/c,dst=/inputs/C,readonly")
    TRAIN_C_ARGS=(--candidate /inputs/C/training-smoke.json --candidate-sha256 "$TRAIN_C_SHA")
  fi
  set +e
  docker run --rm --pull=never --init --runtime=runc --user "$TRAIN_UID:$TRAIN_GID" \
    --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true \
    --memory=2g --memory-swap=2g --cpus=1 --pids-limit=64 --ulimit core=0 \
    --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void --env NVIDIA_DRIVER_CAPABILITIES= \
    --env HOME=/outputs \
    --mount "type=bind,src=$TRAIN_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$TRAIN_BATCH/r,dst=/inputs/R,readonly" \
    --mount "type=bind,src=$TRAIN_BATCH/r-repeat,dst=/inputs/R-repeat,readonly" \
    "${TRAIN_C_MOUNTS[@]}" \
    --mount "type=bind,src=$TRAIN_BATCH/$TRAIN_COMPARE_NAME,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$TRAIN_IMAGE" --signal=TERM --kill-after=10s 120 \
    /usr/bin/python3 -I -B /workspace/compare_geometry_training_smoke.py \
    --reference /inputs/R/training-smoke.json --reference-sha256 "$TRAIN_R_SHA" \
    --repeat /inputs/R-repeat/training-smoke.json --repeat-sha256 "$TRAIN_REPEAT_SHA" \
    "${TRAIN_C_ARGS[@]}" --output-root /outputs \
    > "$TRAIN_BATCH/$TRAIN_COMPARE_NAME-terminal.log" 2>&1
  TRAIN_COMMAND_STATUS=$?
  set -e
  printf '%s\n' "$TRAIN_COMMAND_STATUS" > "$TRAIN_BATCH/$TRAIN_COMPARE_NAME-exit-code.txt"
  cat "$TRAIN_BATCH/$TRAIN_COMPARE_NAME-terminal.log"
  return "$TRAIN_COMMAND_STATUS"
}

training_arm R r
training_arm R-repeat r-repeat
training_compare repeatability
training_arm C c
training_compare final with-candidate
printf 'Preserved bounded-training evidence: %s\n' "$TRAIN_BATCH"
```

Each native arm is limited to 900 seconds plus a 30-second termination grace,
64 GiB host memory/no extra swap, four CPUs and one GPU. No GPU memory cap is
claimed. Eight loader workers remain native despite the CPU quota. CPU comparison
has 120 seconds plus 10-second grace, 2 GiB, one CPU, no exposed GPU and no dataset/
historical-checkpoint mount. Do not pre-create console files inside fresh arm
roots; allocation refuses any collision. Timeouts/OOM/signals preserve partial
output as incomplete. Resource limits are not scientific-config overrides.
Host-side terminal logs and returned exit codes are saved in the parent bundle,
outside the initially empty arm roots. Abrupt parent-shell termination can omit
an exit-code record; never fabricate it or infer a signal from a truncated log.

Expected native CLI exit 0: `bounded_training_smoke_passed`, complete phase
`bounded_stop_after_second_optimizer_return`, two optimizer returns, steps 0/1,
zero parameter changes at step 0 and positive changes at step 1, finite actual
loss/gradient/state records, closed owned resources and empty checkpoints.
Expected gates are `repeatability_passed` then
`bounded_training_regression_passed`. These are expected outcomes to test, not
observed GPU results. Any other result stops the sequence for inspection/review.

## Failure diagnostics and review boundary

Reports retain the phase, exception type/message and last completed/partial update
when possible; console logs include native traceback. Failures before allocation
print a diagnostic without creating a run. A nonfinite loss blocks backward;
nonfinite gradients block clipping/Adam; invalid clip/Adam/parameter state aborts
without any next update. A failure after Adam can leave an already completed call
with invalid state: record it, do not roll back or call it a successful smoke.
Cleanup errors make the run failed. External termination may bypass Python
cleanup/report finalization; container process/resource isolation is still
required. Never substitute another checkpoint, reduce memory by changing samples,
skip a loss/operator or silently fix a renderer defect to obtain a pass.

The new stop mechanism, observational overhead and strict hash-repeatability
policy require code review before researcher-side execution. No native training
claim follows from CPU toy fixtures or the previously passed inference tests.
No material integration, full training, historical reproduction, Docker setup,
image build/package installation, standalone extraction or new scientific change
is implemented in this milestone.

## CPU/static validation performed

Executed in Codex with Python **3.9.12**, NumPy **1.21.5** and existing CPU-test
Torch **2.4.0+cu121**, with `CUDA_VISIBLE_DEVICES=''`. All fixture tensors and
operations explicitly use CPU; no native scientific renderer or original dataset
payload is loaded. No package or interpreter was installed/upgraded.

```text
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/nerofactor-two-update-mpl python3 -B -m unittest discover -s tests/infrastructure -v
Ran 183 tests: 180 passed, 0 failed, 3 skipped.
python3 -B docker/paper/static_checks.py
PASS: definition/mount policy, five Docker RUN shell parses, Python 3.8 grammar for 80 files.
```

All **28 new fixtures** passed. They exercise the actual Trainer class/loop AST
with explicit CPU renderer/dataset/loader/logger/loss stand-ins and real CPU
Torch autograd, clipping and Adam. Coverage includes independent config routes,
source/RO-input guards, exactly two calls, correct zero/nonzero-LR behavior,
finite loss/gradient/parameter rejection, clipping, per-active-parameter counters,
original scheduler endpoints, source sampling/RNG records, parameter changes,
no implicit load/resume, unavailable trace refusal, error/cleanup paths, owned
outputs/immutable fixture inputs, run/report collisions and gated comparisons.
The real native loop is imported/called directly by the researcher-side backend;
AST fixture execution is a test technique, not a production loop replacement.

The three existing skips are native OpenCV/transforms3d crop parity, native
Open3D point downsample/export parity and native OpenCV RAW resize parity.
Unavailable backends are not native passes. New full-suite execution under Python
3.8/Torch 1.8, real DataLoader worker/prefetch behavior, GPU backward/operator
compatibility and native state-repeatability remain **unexecuted**. CPU Torch
2.4 fixture results do not certify Torch 1.8 numerical or optimizer equivalence.

Both CLI help paths work with `python3 -I -B`. Configuration/source checks import
no scientific model. Documented commands were parsed with `bash -n` only; local
links, protected source/config hashes, code scope and whitespace were checked.
The existing Docker static checker's conservative build-artifact-readiness
warning is unrelated to this bounded test and does not recertify/rebuild the
working admitted image. No Docker access/security bypass was attempted.

Scientific source, existing entry points, original YAML, datasets, checkpoints,
historical results and admitted evidence manifests remain unchanged. No native
training/GPU execution, image change, automatic commit or push occurred. Review
this implementation and fixed protocol before any researcher-side GPU command.
