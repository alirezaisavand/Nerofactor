# Bell checkpoint inference regression preparation: 1D.4B-3A

**Implementation for review; no real inference run was executed by Codex.** This
is A-based current-code routing regression, not thesis reproduction, NVS or a
historically verified held-out score. View `"58"` is a training view under the
inspected G split; producing revision/configuration/split and final-version
status remain unresolved. No G reproduction is authorized.

Review baseline: `d775cf1fb51afbfa9645cbf4a975fa7b9acd96f0`, clean
`paper/infrastructure` before edits. Existing scientific code/YAML are unchanged.

## Successful structural evidence, independently inspected

[The versioned evidence manifest](evidence/bell-structural-match.json) records
original paths/hashes, discovered comparison reports, recomputed strict results,
reference source/runtime provenance and limitations. One comparison report was
found; selection is explicit, never latest/glob-first:

| Original evidence | SHA-256 |
| --- | --- |
| `evidence/bell-comparison.UXWN3DqC/architecture-comparison.json` | `4a2413bb174a94507dbb771477652b356e5e9269c891f8c1d4ec5fde9fc4886c` |
| `evidence/bell-reference.eWPhSaW8/reference-signature.json` | `768ce796e7d3356b17cdf76042666c71d73462442be2a99a02480214a047b0f8` |
| `evidence/bell-metadata.xIvZ9k0z/checkpoint-metadata.json` | `6108d45e49f502e48b78f71a8dd329b8f37b95fd249a04fb2a386688d2eb6720` |

Paths in this table are relative to `/home/isavand/paper`. Original files remain
untouched. The comparison recomputes to `exact_structural_match`: 137 tensors on
each side; zero missing/unexpected keys and zero shape/dtype mismatches, including
buffer/weight-normalization coverage. All reference source hashes and the prior
comparison-tool hashes match the inspected checkout. The reference records
Python 3.8.10, Torch 1.8.1+cu111, CUDA 11.1 and NVIDIA A100 80GB PCIe; source commit
is caller-declared as the review baseline. Content hashes bind bytes, not an
independent attestation of the Docker launch or producing checkpoint history.

Historical input remains only
`/inputs/thesis/model/bell_shape/model_best.pth`, SHA-256
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
Saved step is **150000** and `best_para` is **23.787122565451323**. These are saved
fields, not newly measured scores or an inferred 300000-step final model.

## Two adapters and one bounded scientific execution path

[run_bell_inference.py](../run_bell_inference.py) requires explicit arm, unique
run ID, image/source identity, execution, trusted-checkpoint and isolated-runtime
acknowledgements. It has no checkpoint/scene/seed/sample/tolerance override,
resume or automatic checkpoint discovery. The three arms are separate processes:

- **R / R-repeat:** `resolve_reference` reads the fixed legacy bell YAML directly
  with `yaml.FullLoader`; it never calls the schema-1 configuration adapter.
  Dataset root substitution is explicit and independent.
- **C:** `resolve_candidate` uses the approved schema-1 envelope, `load_config`,
  PathResolver and geometry validation. Scientific overrides are rejected.

Both then use the same pinned scientific implementation and approved storage/
PreprocessingCache, so this tests configuration/routing, not independent rendering
implementations. The numerical configuration mappings and actual owner defaults
must agree exactly. Raw legacy bytes are saved; C additionally saves the envelope.
The report records path-resolved values and effective trainer/renderer/shader
settings separately. Shallow merges and nested shader scopes are preserved.

The [execution helper](../infrastructure/bell_inference.py):

1. Revalidates all three evidence hashes and strict equality; checks every pinned
   reference source/config/asset hash, read-only input/source/rootfs mounts,
   non-root/no-network runtime and the selected checkpoint hash before scientific
   imports or run allocation. RunStore mounted allocation rejects nonempty roots;
   each arm owns its caches, temporaries, logs, images and arrays.
2. Imports native code only after those gates. It uses `Trainer.default_cfg`
   without constructing a Trainer or optimizer. The original shallow merge and
   seed order are used: Torch, NumPy, Python, with effective seed 6033. Backend
   determinism/TF32 flags are recorded, not changed to force agreement.
3. Constructs the existing training DummyDataset and validation DummyDataset in
   their original order, then the actual geometry renderer and `.cuda()`. No
   DataLoader objects/workers/iteration are used. The renderer's ordinary training
   ray-pool construction/shuffle is preserved; no ray count/resolution reduction
   or shortened sampler is introduced. Thus eager initialization remains costly.
4. Explicitly passes `isolated_geometry=True`: the approved A split-unpacking
   shim consumes train/validation/NVS lists. It calls no alternative split helper.
   Actual database order must be strings `"0"`–`"127"`; actual ordered lists must
   equal A's seeded shuffle, with validation `"58"`. A separate local Random
   verifies membership without changing global RNG. There are exactly two real
   split calls (validation dataset, renderer), as in dataset/renderer setup.
5. Hashes the used camera/RGB/mask/depth input set before dataset reads, checks it
   after initialization and again after evaluation. This covers 128 camera
   pickles, 123 RGBs, 123 masks and the validation depth: **375 selected files**,
   not entire dataset families. Original preprocessing math/cache semantics stay
   unchanged; synthetic bell requires no relocated resize/crop products. Full
   scene metadata and cache implementation/version identity are also recorded.
6. Compares the live model's full state signature against the pinned reference,
   hashes the same open checkpoint descriptor before pickle loading, loads the
   complete trusted container on CPU, verifies exact state key sets/shapes/dtypes
   and stored step, then calls `load_state_dict(..., strict=True)`. No optimizer
   is constructed/restored; its serialized payload is only unpickled into CPU
   memory and discarded. The descriptor hash and saved metric are rechecked.
   No prefixes/casts/partial loading or checkpoint substitution are allowed.
7. Calls `model.eval()`, obtains existing validation item index 0, sets `eval=True`
   and **step 150000 exactly**, captures RNG hashes, then calls the model once
   under the original `torch.no_grad()` validation boundary. SDF.gradient's own
   `torch.enable_grad()` remains untouched. No `inference_mode`, derivative
   suppression, extra view, final NVS, mesh extraction or backward is used.
8. Uses the unchanged `ShapeRenderMetrics` for the validation montage and existing
   full-image PSNR/SSIM. Saves float32 RGB/GT NPY with `allow_pickle=False`, mask
   NPY, lossless uint8 prediction/GT PNGs, original diagnostic JPG, `metrics.json`,
   tensor-finiteness diagnostics and output hashes. Source/evidence/new-adapter
   identities are rechecked. Checkpoint directories must remain empty.

Runtime input paths are the same in every arm. Original source/inputs are
read-only; HOME/library caches/JIT/temp/console outputs use the owned run. Failure
phase/type and console traceback are retained; failed directories are never
resumed or reused. A failed original renderer/evaluator is a blocker for review,
not authorization to fix mathematics or invent alternative outputs. Single-use
processes avoid claiming recovery of the legacy global tensor-default state
following a renderer exception.

No full training-loop RNG trajectory, historical RNG restoration or first
optimizer-step parity is claimed. The seed and dataset/renderer construction
policy above is explicit and common to all arms, without training-loader
iteration. Stored step semantics are not corrected by subtracting one.

Reports include command arguments, package/driver/backend/thread/ABI settings and
additional hashes for the imported Trainer/defaults and training helper modules.
Image/commit values remain caller declarations; the Docker command independently
selects the known image ID. Complete native-library/extension/JIT binary hashes,
image build-chain attestation and historical producing provenance are explicitly
unresolved, not inferred from package versions or source revisions.

## Repeatability and numerical comparison

[compare_bell_inference.py](../compare_bell_inference.py) and its
[CPU helper](../infrastructure/bell_comparison.py) import no Torch or scientific
model. They validate explicit report SHA-256 values, schema/arm/status/count,
architecture identities, output containment and every declared artifact hash.
NPY loading always uses `allow_pickle=False`; malformed/object arrays are not
unpickled. RGB must be finite float32, full-view H-by-W-by-3. The comparison must
use the same NumPy/scikit-image versions; metrics are recomputed from lossless
arrays and checked exactly against the stored metrics and metric artifact.

Metrics preserve the existing conversion (multiply 255, clip, truncate uint8),
float32 mean-square per channel then mean channels, `10*log10(255**2/MSE)` and
SSIM `win_size=11`, `channel_axis=2`, `data_range=255`. There is no extra masking,
foreground-only reduction, gamma transform, resize or JPEG-derived comparison.
Positive infinite PSNR is explicitly tagged; both exact-match infinities have
zero delta, while only one infinite fails. Nonfinite/unsupported metrics block.

Before numbers are interpreted, exact parity is required for source/evidence/
input/cache/config/split/runtime identities, GT/mask values and pre/post inference
RNG-state hashes. RNG draw divergence is an identity failure requiring review;
it is not hidden by favorable PSNR. Each arm must have a distinct run ID.
These identity guards make the protocol's RNG/chunk/config requirements explicit;
they do not add a new numerical tolerance.

The precommitted [protocol thresholds](protocols/CHECKPOINT_INFERENCE_V1.md)
remain unchanged: float `abs(C-R) <= 1e-4 + 1e-4*abs(R)` elementwise; uint8 maximum
channel error <=1; PSNR delta <=0.01 dB; SSIM delta <=1e-4. No CLI changes these.
Difference arithmetic uses float64 on the saved float32 values; original render
arrays and metric calculations are unchanged. Reports include all max/mean
absolute/relative diagnostics, error/violation counts, uint8 differences,
per-arm metrics/deltas and full signed-difference/violation-mask NPY files with
hashes. Relative errors at zero reference are counted separately, without silent
epsilon clipping. Diagnostic JPG byte equality is not a metric requirement.

Run R, R-repeat, then a CPU repeatability gate. **Only after that passes run C**
and the final comparison, which independently rechecks repeatability. C's report
is not even read/interpreted by the comparator when repeatability fails. Results
are `repeatability_passed` / `regression_passed` (exit 0), `repeatability_failed` /
`candidate_mismatch` (exit 1), or `blocked` on invalid/missing identities/data
(exit 2). Partial comparison artifacts and failure diagnostics are preserved.
No numerical threshold or seed may be adjusted after viewing results without
separate review and a protocol revision.

## Exact researcher-side commands (not executed by Codex)

Review and freeze the implementation snapshot before execution. Use the existing
image, ordinary terminal Docker access and the same explicitly selected GPU for
all arms. No image rebuild, installation or permission change is included.
Commands abort on failure; do not turn them into a retry/seed-selection loop.

```bash
set -eu
BELL_SOURCE=/home/isavand/paper/Nerofactor-paper
BELL_DATA=/home/isavand/paper/data
BELL_THESIS=/home/isavand/paper/final_results
BELL_METADATA=/home/isavand/paper/evidence/bell-metadata.xIvZ9k0z/checkpoint-metadata.json
BELL_REFERENCE=/home/isavand/paper/evidence/bell-reference.eWPhSaW8/reference-signature.json
BELL_ARCHITECTURE=/home/isavand/paper/evidence/bell-comparison.UXWN3DqC/architecture-comparison.json
BELL_IMAGE=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$BELL_IMAGE" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
BELL_UID=$(id -u)
BELL_GID=$(id -g)
test "$BELL_UID" -ne 0
BELL_GPU=0
BELL_COMMIT=$(git -C "$BELL_SOURCE" rev-parse HEAD)
BELL_TOKEN=$(python3 -B -c 'import uuid; print(uuid.uuid4().hex)')
BELL_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-regression.XXXXXXXX)
mkdir "$BELL_BATCH/r" "$BELL_BATCH/r-repeat" "$BELL_BATCH/c"

bell_arm() {
  BELL_ARM=$1
  BELL_NAME=$2
  docker run --rm --pull=never --init --user "$BELL_UID:$BELL_GID" \
    --gpus "device=$BELL_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$BELL_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$BELL_DATA,dst=/inputs/datasets,readonly" \
    --mount "type=bind,src=$BELL_THESIS,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$BELL_METADATA,dst=/inputs/metadata/checkpoint-metadata.json,readonly" \
    --mount "type=bind,src=$BELL_REFERENCE,dst=/inputs/reference/reference-signature.json,readonly" \
    --mount "type=bind,src=$BELL_ARCHITECTURE,dst=/inputs/comparison/architecture-comparison.json,readonly" \
    --mount "type=bind,src=$BELL_BATCH/$BELL_NAME,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$BELL_IMAGE" --signal=TERM --kill-after=30s 900 \
    /usr/bin/python3 -I -B /workspace/run_bell_inference.py \
    --arm "$BELL_ARM" --run-id "$BELL_NAME-$BELL_TOKEN" \
    --image-id "$BELL_IMAGE" --source-commit "$BELL_COMMIT" --execute \
    --acknowledge-trusted-checkpoint --confirm-isolated-runtime
}

bell_compare() {
  BELL_COMPARE_NAME=$1
  mkdir "$BELL_BATCH/$BELL_COMPARE_NAME"
  BELL_R_SHA=$(sha256sum "$BELL_BATCH/r/inference.json" | cut -d ' ' -f 1)
  BELL_REPEAT_SHA=$(sha256sum "$BELL_BATCH/r-repeat/inference.json" | cut -d ' ' -f 1)
  BELL_C_MOUNTS=()
  BELL_C_ARGS=()
  if test "${2:-}" = with-candidate; then
    BELL_C_SHA=$(sha256sum "$BELL_BATCH/c/inference.json" | cut -d ' ' -f 1)
    BELL_C_MOUNTS=(--mount "type=bind,src=$BELL_BATCH/c,dst=/inputs/C,readonly")
    BELL_C_ARGS=(--candidate /inputs/C/inference.json --candidate-sha256 "$BELL_C_SHA")
  fi
  docker run --rm --pull=never --init --runtime=runc --user "$BELL_UID:$BELL_GID" \
    --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true \
    --memory=2g --memory-swap=2g --cpus=1 --pids-limit=64 --ulimit core=0 \
    --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void --env NVIDIA_DRIVER_CAPABILITIES= \
    --env HOME=/outputs --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --mount "type=bind,src=$BELL_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$BELL_BATCH/r,dst=/inputs/R,readonly" \
    --mount "type=bind,src=$BELL_BATCH/r-repeat,dst=/inputs/R-repeat,readonly" \
    "${BELL_C_MOUNTS[@]}" \
    --mount "type=bind,src=$BELL_BATCH/$BELL_COMPARE_NAME,dst=/outputs" \
    --workdir /workspace --entrypoint /usr/bin/timeout \
    "$BELL_IMAGE" --signal=TERM --kill-after=10s 120 \
    /usr/bin/python3 -I -B /workspace/compare_bell_inference.py \
    --reference /inputs/R/inference.json --reference-sha256 "$BELL_R_SHA" \
    --repeat /inputs/R-repeat/inference.json --repeat-sha256 "$BELL_REPEAT_SHA" \
    "${BELL_C_ARGS[@]}" --output-root /outputs
}

bell_arm R r
bell_arm R-repeat r-repeat
bell_compare repeatability
bell_arm C c
bell_compare final with-candidate
printf 'Preserved evidence bundle: %s\n' "$BELL_BATCH"
```

Per arm: 900 seconds plus 30-second termination grace, four CPUs, 64 GiB host
memory/no extra swap and one GPU. No GPU memory cap is claimed. Per CPU comparison:
120 seconds plus ten-second grace, one CPU, 2 GiB, no exposed GPU and no checkpoint
mount. No other arm's output is writable from a native arm. The CPU comparator
reads reports read-only and writes a different fresh comparison directory.

Do not tee/create files inside a fresh arm/output root before allocation. Missing
runtime/inputs, timeout/OOM, malformed native/evaluator output, nonfinite RGB,
identity mismatch or tolerance failure stops the sequence. Preserve exit status,
full command, logs, partial reports and failed directories. A running/incomplete
report or missing exit-0 completion never counts as success. Do not switch
checkpoint, crop/resolution, sample count, seed, device or threshold to finish.

## Review boundary

This implementation has CPU/static evidence only. Native one-view validation,
its original optional diagnostic layout, numerical repeatability and R/C parity
remain unexecuted here. Structural compatibility removed an architecture blocker;
it did not verify any image or historical metric. Review this implementation and
command before researcher-side execution. Any newly exposed scientific/evaluator
defect is a separately reported blocker, not an infrastructure cleanup fix.

## CPU and static validation

Executed in Codex with Python **3.9.12**, NumPy **1.21.5** and scikit-image
**0.19.2**:

```text
MPLCONFIGDIR=/tmp/nerofactor-bell-regression-matplotlib python3 -B -m unittest discover -s tests/infrastructure -v
Ran 155 tests: 152 passed, 0 failed, 3 skipped.
python3 -B docker/paper/static_checks.py
PASS: definition policy, five Docker RUN shell parses, Python 3.8 grammar for 72 files.
```

All **16 new CPU fixtures** passed. Coverage includes independent configuration
parity, shallow defaults/seed order, full ordered split/RNG preservation, strict
weights-only loading gates, checkpoint-hash failure before allocation/import,
375-file opaque input identities without unpickling, untouched fixture inputs,
output containment/collisions, metric equivalence, invalid/tampered reports,
repeatability/candidate gating, numerical/identity mismatches, complete difference
artifacts and partial-failure evidence/runtime cleanup.

Three existing skips are native OpenCV/transforms3d crop parity, native Open3D
point downsample/export parity and native OpenCV RAW resize parity; required
libraries are unavailable in Codex. Fixture routing tests still execute. These
skips are not native parity passes. Full Python 3.8 execution of this new suite
and the image's NumPy 1.24.4/scikit-image 0.21.0 metric runtime remain untested here;
3.8 grammar checks do not establish runtime compatibility.

Original evidence hashes/sizes, strict structural recomputation, pinned reference
and comparator source hashes were independently rechecked. Both CLI help paths
work with `python3 -I -B`; documented Bash parses without execution. Local links,
whitespace and final changed-file scope were checked. Scientific source, original
YAML, datasets, historical results and original reports are unchanged. No Docker
access was attempted, and no historical checkpoint was opened/deserialized by
Codex. The existing Docker static checker still prints its conservative legacy
artifact-build-readiness warning; this task neither builds nor re-certifies the
researcher's working image.
