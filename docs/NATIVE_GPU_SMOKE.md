# Native model smoke protocol: Step 1D.3C

**Current evidence update (1D.4A):** the researcher reports that both actual
geometry and material model smoke tests passed using
`nerofactor-paper:quick-candidate-1`, and all **109 infrastructure tests passed
on Python 3.8**. Original smoke/test logs, immutable image ID and launch-specific
source/input identities have not been supplied/checked in here. These are
reported successes, not GPU launches independently observed by Codex. See
[BASELINE_PREPARATION.md](BASELINE_PREPARATION.md) for the evidence ledger and
[proposed inference protocol](protocols/CHECKPOINT_INFERENCE_V1.md) for the next
review gate. Smoke success does not establish checkpoint inference parity or
historical result reproduction.

The remainder records the 1D.3C implementation and its original CPU/static checks.
No native GPU smoke or Docker run was executed by Codex during that milestone,
and none is authorized by the current documentation-only 1D.4A task. Preserve
the image; do not rebuild/install to use this code.

Review baseline: `ae1856529ab83865cea6f95b9820e4cd50e13fff` on
`paper/infrastructure`. A pre-existing one-line change in
`tests/infrastructure/test_material_training.py` replaces a hand-created
`ast.Constant('cloud.ply')` with `ast.parse("'cloud.ply'", mode='eval').body`.
It is retained to match parser-created AST fields under Python 3.8 while keeping
all original parity assertions. The earlier removal of `ast.unparse` stays intact.

## What is exercised and preserved

[run_native_smoke.py](../run_native_smoke.py) accepts the existing schema-1
geometry/material envelopes, approved config root, PathResolver input roots and
unique RunStore directory. Validation is separate from execution. It rejects
checkpoint inputs, resume/initialization options, output collisions and output
storage overlapping code. Legacy entry points and scientific YAML are unchanged.

The explicit execution protocol `native-training-forward-v1`:

1. Check OS `statvfs(ST_RDONLY)` for code, configuration roots/files, every declared
   input root, selected dataset/scene and selected mesh. No input write probes or
   chmod operations. Require writable output storage before allocating a fresh run.
2. Allocate exclusive disposable output storage and reuse PreprocessingCache and
   the approved runtime/cache environment, including HOME inside the selected run.
   Scientific imports occur only after this boundary.
3. Call the existing `Trainer` constructor for its unchanged shallow defaults and
   seed policy, then `_init_dataset()` for the same train/validation dataset and
   cache propagation. DataLoader objects are constructed but never iterated;
   no workers are started or enormous dummy-sampler permutation allocated.
4. Construct the actual existing renderer, exactly as isolated training does:
   geometry receives `isolated_geometry=True` and the cache; material receives the
   selected read-only mesh, cache and run output paths. Transfer with `.cuda()`;
   capture renderer/shader effective settings and original split membership.
5. Obtain the existing empty `DummyDataset[0]` item, add `step=0`, select normal
   training mode and call the renderer **once**. Keep autograd enabled because
   geometry computes SDF derivatives internally. Check nonempty N-by-3 primary
   RGB on CUDA and finiteness of returned tensors, then exit.

The renderer owns actual scene pixels, cameras, masks, mesh and training rays.
This is one ordinary configured **training-forward batch**, not a one-pixel
replacement or full validation view. No ray/image/sample-count reduction, new
camera, mesh transform, precision change, seed override, formula patch or shader
scope correction is introduced. Existing defaults normally use 512 training
rays, and the bear material config retains 512 diffuse/1024 specular samples.
Original configuration-specific values take precedence unchanged.

No `Trainer.run`, `_init_network`, optimizer construction/update, training loss
class aggregation, `.backward()`, scheduler invocation, checkpoint loading/saving,
evaluation/NVS, extraction or debug point export is called. Renderer-internal
loss/regularization expressions and SDF derivative calculations remain part of
the unchanged forward. RunStore creates an **empty checkpoints directory**;
the harness verifies it remains empty, and never serializes model tensors.

This separate protocol omits DataLoader iteration and other training-loop work;
it does **not** claim the training loop's exact RNG trajectory or optimizer-step
parity. It preserves the model sampling implementations, configured counts and
constructor seeding, without selecting favorable rays or retrying seeds. Fresh
random weights and finite outputs do not measure reconstruction quality. A failed
native formula/interface is reported for review, never repaired by the smoke tool.

**Initialization is eager:** material construction decodes the scene and traces
training rays against the real mesh; geometry also builds its ordinary ray pool.
It may require substantial CPU/GPU memory before the single forward. The first
use of nvdiffrast can compile plugins in the disposable experiment cache. No
claim is made that initialization costs one ray or that each scene fits a budget.

## Side-effect-free host preflight

These checks were run successfully against the declared host inputs in this
milestone; they do not allocate an output directory, check mount flags, decode a
mesh, deserialize checkpoints or import Torch/models:

```sh
python3 -B run_native_smoke.py \
  --cfg configs/experiments/geometry.example.yaml \
  --config-root /home/isavand/paper/Nerofactor-paper/configs \
  --datasets-root /home/isavand/paper/data --thesis-root /home/isavand/paper/final_results \
  --output-root /tmp/nerofactor-1d3c-review-planned --run-id geometry-review --validate-only

python3 -B run_native_smoke.py \
  --cfg configs/experiments/material.example.yaml \
  --config-root /home/isavand/paper/Nerofactor-paper/configs \
  --datasets-root /home/isavand/paper/data --thesis-root /home/isavand/paper/final_results \
  --output-root /tmp/nerofactor-1d3c-review-planned --run-id material-review --validate-only
```

Expected final line: `Smoke structure validated; no run/model/GPU allocated, no
mount or native compatibility claimed.` Legacy scope/type diagnostics remain
visible; material top-level regularization is not moved into `shader_cfg`, and
the scheduler's hard-coded horizon is unchanged (the smoke does not run it).

## Exact Docker execution commands, only after launch approval

**Do not execute this section during implementation/review.** Use the regular
terminal's existing Docker/GPU access; never bypass socket/device restrictions.
These commands use an inspected immutable ID from the accepted candidate tag.
The new reviewed repository is mounted over `/workspace` read-only because the
existing image does not contain this new entry point. This code overlay is
explicitly recorded by source/config hashes; success would validate this overlay
with the candidate environment, not silently attest to the original baked code.

One container and new owned output directory per stage; run them sequentially.
The following budget is proposed for review: 900 seconds plus a 30-second forced
termination interval, 64 GiB host RAM, 4 CPUs, 128 processes and 2 GiB `/dev/shm`.
GPU memory is not capped by the harness. A resource failure is a diagnostic,
not permission to reduce scientific settings or change the image. The image must
already provide `/usr/bin/timeout`; if missing, report it without installing.

```sh
SMOKE_IMAGE=nerofactor-paper:quick-candidate-1
SMOKE_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$SMOKE_IMAGE")
SMOKE_CODE=/home/isavand/paper/Nerofactor-paper
SMOKE_DATA=/home/isavand/paper/data
SMOKE_THESIS=/home/isavand/paper/final_results
SMOKE_GPU=0
SMOKE_UID=$(id -u)
SMOKE_GID=$(id -g)
test "$SMOKE_UID" -ne 0
SMOKE_ID=$(python3 -B -c 'import uuid; print("native-smoke-" + uuid.uuid4().hex)')
SMOKE_ROOT=$(mktemp -d /tmp/nerofactor-native-smoke.XXXXXXXX)
mkdir "$SMOKE_ROOT/geometry" "$SMOKE_ROOT/material"
```

Geometry uses the unchanged existing bell envelope and real GlossySynthetic data:

```sh
docker run --rm --pull=never --init --name "${SMOKE_ID}-geometry" \
  --user "${SMOKE_UID}:${SMOKE_GID}" --gpus "device=${SMOKE_GPU}" \
  --read-only --network=none --cap-drop=ALL --security-opt=no-new-privileges:true \
  --memory=64g --cpus=4 --pids-limit=128 --shm-size=2g --workdir=/workspace \
  --mount "type=bind,source=${SMOKE_CODE},target=/workspace,readonly" \
  --mount "type=bind,source=${SMOKE_DATA},target=/inputs/datasets,readonly" \
  --mount "type=bind,source=${SMOKE_THESIS},target=/inputs/thesis,readonly" \
  --mount "type=bind,source=${SMOKE_ROOT}/geometry,target=/outputs" \
  --entrypoint /usr/bin/timeout "$SMOKE_IMAGE_ID" --signal=TERM --kill-after=30s 900 \
  /usr/bin/python3 -B /workspace/run_native_smoke.py \
  --cfg /workspace/configs/experiments/geometry.example.yaml --config-root /workspace/configs \
  --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
  --layout mounted --run-id "${SMOKE_ID}-geometry" --image-id "$SMOKE_IMAGE_ID" --execute
```

Material uses the unchanged existing bear envelope, real GlossyReal images and
`/inputs/thesis/meshes-pretrained/real/bear.ply` as an explicit read-only input:

```sh
docker run --rm --pull=never --init --name "${SMOKE_ID}-material" \
  --user "${SMOKE_UID}:${SMOKE_GID}" --gpus "device=${SMOKE_GPU}" \
  --read-only --network=none --cap-drop=ALL --security-opt=no-new-privileges:true \
  --memory=64g --cpus=4 --pids-limit=128 --shm-size=2g --workdir=/workspace \
  --mount "type=bind,source=${SMOKE_CODE},target=/workspace,readonly" \
  --mount "type=bind,source=${SMOKE_DATA},target=/inputs/datasets,readonly" \
  --mount "type=bind,source=${SMOKE_THESIS},target=/inputs/thesis,readonly" \
  --mount "type=bind,source=${SMOKE_ROOT}/material,target=/outputs" \
  --entrypoint /usr/bin/timeout "$SMOKE_IMAGE_ID" --signal=TERM --kill-after=30s 900 \
  /usr/bin/python3 -B /workspace/run_native_smoke.py \
  --cfg /workspace/configs/experiments/material.example.yaml --config-root /workspace/configs \
  --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
  --layout mounted --run-id "${SMOKE_ID}-material" --image-id "$SMOKE_IMAGE_ID" --execute
```

No implicit pull, package install, writable code/input mounts, Docker socket mount
or reference-container change. Inspect exit status and geometry output before
starting material. Keep disposable output evidence until review; no cleanup or
automatic deletion command is supplied. A rerun needs a new ID and empty output
directory. Never resume/reuse a partial run or overwrite a checkpoint.

For an explicitly selected prior-run mesh, prepare a reviewed schema-1 envelope
whose `inputs.mesh.root` is `geometry`, add a read-only bind to `/inputs/geometry`
and `--input-root geometry=/inputs/geometry`; its original input remains immutable.
Do not write a path change into historical configs or substitute a mesh silently.

## Expected results and failure diagnostics

A successful native process exits zero and reports these phases in order:
`runtime_ready`, `native_imports`, `initializing`, `initialized`, `forward`, `complete`, followed by
`Smoke passed: /outputs/smoke.json`. Both `status: passed` and the process exit
code matter; the examples above are **expected outputs, not observed GPU results**.

`smoke.json` records protocol/stage, declared image ID, smoke code hashes,
compatibility signature, inspected read-only paths, seed, train/validation/NVS
IDs, configured ray count and available rays, one forward at step 0, primary/output
tensor shapes/dtypes/devices/finiteness, Torch/CUDA/GPU identity and limitations.
Original/effective configurations and mesh identity are preserved in the existing
run records. Logs/cache/temp products stay under the disposable output root.
No output tensor/model checkpoint is saved, no metric is used for selection and
no PSNR/SSIM/reconstruction-quality result is inferred.

| Failure | Evidence and action |
| --- | --- |
| Config/path/collision error | CLI nonzero, often before run allocation; select valid real inputs/new output, never enable implicit resume |
| Writable input/code filesystem | Fail before allocation or model import; correct the declared mounts in a separately reviewed launch, never chmod originals |
| Missing/incompatible CUDA/extension | Native import/construction trace in `logs/console.log`, phase in `smoke.json`; stop, retain evidence, do not upgrade/install/patch automatically |
| Mesh BVH failure | Native raytracer needs more than eight triangles; verify the selected real mesh identity/topology read-only, never rewrite it |
| Initialization OOM or timeout | Eager image/ray/BVH construction can be large; exit 124 for normal timeout, other signals/OOM may yield 137; retain partial logs and review budget/scene selection explicitly |
| Forward shape/key/math/nonfinite failure | Last phase `forward`, failing output/error name where available; no equation, head, tangent frame, sampling, coefficient or seed correction in this milestone |
| Unexpected checkpoint file | Smoke fails instead of claiming success; inspect why a writer ran, preserve its disposable evidence |

Caught Python failures save `status: failed` and the last phase, retain a console
trace, and restore CWD/HOME/runtime environment in CPU-tested control flow. A hard
kill/OOM may leave `status: running` or no report; the external exit code/logs are
then authoritative. Native renderer failure may leave its original tensor-default
global state; this is a **single-use subprocess** protocol, not an in-process retry
or a scientific exception-behavior repair.

## Review validation

CPU fixtures test routing and one-forward dispatch for both stages, input byte/
mtime preservation, shader scope, no optimizer-producing trainer methods, no
checkpoint adoption, empty checkpoint output, exclusive allocation, OS mount-gate
rejection, finite/shape validation, failure logging, environment restoration and
import/RNG-free structural validation. Fakes report `injected_cpu_fixture`, never
native GPU success. No GPU/device/extension smoke was executed here.

```sh
MPLCONFIGDIR=/tmp/nerofactor-smoke-cpu-matplotlib python3 -B -m unittest discover -s tests/infrastructure -v
python3 -B -m unittest discover -s docker/paper -p 'test_*.py' -v
python3 -B docker/paper/static_checks.py
```

Results on Codex's available Python 3.9.12: **109 infrastructure tests discovered,
106 passed, 0 failed, 3 native-backend checks skipped**, including nine new smoke
tests. Docker/prerequisite tests: **28 passed, 0 failed/skipped**. Static checks
parse 61 Python files using the Python 3.8 grammar, five offline Docker RUN shell
commands and the documented Docker command syntax. Both real-input host
structural preflights passed without allocating a run. The three host skips are
native OpenCV, OpenCV/transforms3d crop and Open3D; the researcher's reported
100/100 Python 3.8 container result is separate evidence.

Actual Python 3.8/native model execution of this new harness, read-only GPU runtime
mount behavior, JIT latency, memory bounds and scientific numerical parity remain
unverified by Codex. Full training, checkpoint continuation, validation/NVS metrics,
standalone extraction/evaluation, image changes and Step 1D.4 remain outside scope.
