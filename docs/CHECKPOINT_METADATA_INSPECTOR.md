# Isolated bell checkpoint metadata inspector: 1D.4B-1A

Implementation and execution command **for review; not run on the historical
checkpoint by Codex**. No Docker or bubblewrap command was executed in this
milestone. The researcher confirms they trained the selected checkpoint and
authorizes CPU-only metadata inspection; producing revision and final-version
status remain unknown. That trust does not establish architectural compatibility
or historical reproducibility.

[inspect_bell_checkpoint.py](../inspect_bell_checkpoint.py) is standalone and
imports only the standard library until all runtime/hash gates pass. It imports
no renderer, dataset, configuration loader or CUDA-dependent scientific module.
It uses the reviewed image's Torch **1.8.1+cu111**, rejecting other versions in
the production entry point. Unit fixtures separately exercise available local
Torch versions; those results are not pinned-image runtime validation.

## Scope and gates

The only accepted checkpoint path is `/inputs/checkpoint/model_best.pth`, with
hard-coded SHA-256
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
The only output root is `/outputs`. There is no hash override, alternate checkpoint,
resume, model-load or fixture mode in the CLI. Explicit `--execute`,
`--acknowledge-trusted-checkpoint` and `--confirm-isolated-runtime` are required;
they acknowledge the existing authorization/runtime contract, not new trust
evidence recovered from the checkpoint.

Before Torch import, runtime guards require Linux/non-root execution, explicit
read-only file mounts for the inspector and checkpoint, a read-only rootfs,
a separately mounted writable output root, only loopback interfaces, no visible
NVIDIA/DRI/KFD device nodes, and explicit CPU-only device environment values.
Output must be fresh/empty. The hash is checked on the read-only, non-symlink
file descriptor later passed to Torch, avoiding a reopen between verification
and loading. Its identity and hash are rechecked after inspection. Report creation
is exclusive and precedes Torch import; collisions never overwrite evidence.

These are defense-in-depth observations, **not a sandbox or independent proof of
Docker policy/image authenticity**. The researcher must use the isolated command
below; flags alone cannot turn an ordinary process into an isolated runtime.
The inspector does not create namespaces or try to circumvent Codex restrictions.

`torch.load(stream, map_location='cpu')` uses the PyTorch 1.8 API. **Pickle
deserialization occurs and can execute code.** The entire checkpoint container,
including any serialized optimizer tensors, is deserialized into CPU memory.
The inspector never restores optimizer/scheduler state or traverses optimizer
payload contents for its report. It has no `weights_only` argument/fallback,
model construction, `load_state_dict`, forward/backward, inference or GPU call.
This is appropriate only for the exact explicitly trusted artifact in isolation.

## Report contract

`/outputs/checkpoint-metadata.json` records:

- Protocol/schema/status/phase/timestamps, verified input hash/filesystem identity,
  inspector source hash, observed runtime guards and Python/Torch versions.
  Image ID is caller-declared, not queried/authenticated inside the process.
- Top-level checkpoint keys/types, and every `network_state_dict` tensor's name,
  shape, dtype and CPU device. Both parameters and buffers are catalogued; no
  tensor values, reductions, samples or tensor hashes are exported.
- Stored nonnegative integer `step` and finite scalar `best_para`, when present.
  Python/NumPy numeric scalars are supported. Tensor-valued scalar fields are
  explicitly unsupported rather than read via `.item()`; absent/invalid/nonfinite
  values receive a status, never a fabricated value.
- Bounded primitive fields from an explicit provenance-key allowlist at the top
  level and within `provenance`, `metadata`, `experiment`. Nested configurations,
  arbitrary object representations, tensor-valued provenance, unknown payloads
  and optimizer contents are omitted. The allowlist/limits are recorded as a
  limitation; missing output fields do not prove the artifact lacks provenance.

The accepted network container is a nonempty dict/OrderedDict of ordinary CPU
Tensor/Parameter objects. Unsupported containers, oversized keys/catalogues or
nonstandard/non-CPU network entries fail clearly by phase/type. No key rewriting,
dtype conversion, partial model loading or compatibility repair occurs.
`metadata_extracted` means successful metadata extraction only; A/G compatibility,
stored step interpretation, producing revision, split and final-version status
still require review. A container key named `step` is recorded as saved, not
reinterpreted from logs or the YAML endpoint.

## Exact researcher-side Docker command (do not run from Codex)

Run only after reviewing this implementation. This uses the existing image and
ordinary Docker access in the researcher's terminal, without rebuilding or
changing runtime permissions. If image inspection/runtime access fails, stop.
The expected image ID below is the caller-declared ID preserved in the original
smoke reports; this command checks it through the local engine before use.
A different ID requires review, not an automatic retag/build or fallback.

```bash
set -eu
INSPECT_SOURCE=/home/isavand/paper/Nerofactor-paper
INSPECT_CHECKPOINT=/home/isavand/paper/final_results/model/bell_shape/model_best.pth
INSPECT_IMAGE_ID=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$INSPECT_IMAGE_ID" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
INSPECT_UID=$(id -u)
INSPECT_GID=$(id -g)
test "$INSPECT_UID" -ne 0
test -f "$INSPECT_SOURCE/inspect_bell_checkpoint.py"
test -f "$INSPECT_CHECKPOINT"
INSPECT_OUTPUT=$(mktemp -d /home/isavand/paper/evidence/bell-metadata.XXXXXXXX)

docker run --rm --pull=never --init --runtime=runc \
  --user "$INSPECT_UID:$INSPECT_GID" \
  --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --memory=2g --memory-swap=2g --cpus=1 --pids-limit=64 \
  --ulimit nofile=256:256 --ulimit core=0 --ipc=private --shm-size=64m \
  --env NVIDIA_VISIBLE_DEVICES=void --env NVIDIA_DRIVER_CAPABILITIES= \
  --env CUDA_VISIBLE_DEVICES= --env HOME=/outputs \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$INSPECT_SOURCE/inspect_bell_checkpoint.py,dst=/workspace/inspect_bell_checkpoint.py,readonly" \
  --mount "type=bind,src=$INSPECT_CHECKPOINT,dst=/inputs/checkpoint/model_best.pth,readonly" \
  --mount "type=bind,src=$INSPECT_OUTPUT,dst=/outputs" \
  --workdir /outputs --entrypoint /usr/bin/timeout \
  "$INSPECT_IMAGE_ID" --signal=TERM --kill-after=10s 180 \
  /usr/bin/python3 -I -B /workspace/inspect_bell_checkpoint.py \
  --checkpoint /inputs/checkpoint/model_best.pth --output-root /outputs \
  --image-id "$INSPECT_IMAGE_ID" --execute \
  --acknowledge-trusted-checkpoint --confirm-isolated-runtime

printf 'Report: %s\n' "$INSPECT_OUTPUT/checkpoint-metadata.json"
```

There is deliberately **no `--gpus`, device bind, Docker socket, dataset, mesh,
host HOME, entire research-result directory or whole-repository mount**. Explicit
`runc` and `NVIDIA_VISIBLE_DEVICES=void` prevent dependence on a host's default
NVIDIA runtime. If `runc` or `/usr/bin/timeout` is unavailable, report that blocker;
do not install packages/change Docker settings. Only the standalone source file
and selected checkpoint are mounted read-only; one fresh host directory is writable.
CPU thread environment settings constrain this metadata process only. Python
isolated mode excludes caller Python paths/user-site packages; bytecode writes
are disabled and conventional HOME/cache/temp paths are redirected under
`/outputs/runtime`. No original datasets/checkpoints/results are writable.

Proposed budget: one CPU, 2 GiB host memory without additional swap, 180 seconds
plus ten-second termination grace, one deserialization, no automatic retries.
Keep failed output directories; reruns require a different empty directory.
Do not create/tee a log inside the output directory before startup: the freshness
guard will reject it. Console success is one line plus a report with
`status=metadata_extracted`, `phase=complete`, and process exit 0. Preserve the
exact command and external exit status separately when archiving the result.

Failures before report allocation produce stderr phase/type and nonzero exit;
they do not load the checkpoint. Later failures retain a failed report with
phase/error type and no partial tensor catalogue; arbitrary exception text is
not serialized because it can contain payload data. A hard kill/OOM may leave
an incomplete/running report: that is not success. Invalid acknowledgements or
fixed-path arguments exit 2. A runtime/hash/import/load/recheck failure exits 1.
No failure triggers a different checkpoint, runtime or deserialization strategy.

Review the extracted metadata before publishing/committing it or proceeding to
architecture admission. The earlier
[compatibility report](evidence/BELL_CHECKPOINT_COMPATIBILITY.md) remains an
accurate blocked snapshot; it is not overwritten with a synthetic or unexecuted
inspection result.

## CPU/static evidence from implementation

Focused host tests: **16 passed**, including a real disposable synthetic Torch
checkpoint with Python/NumPy scalar fields, CPU parameters/buffers and optimizer
payload deliberately excluded from output. The production main path is tested
with explicit fake runtime/loader boundaries; no fixture mode is exposed by CLI.
No test opens the historical artifact or invokes a model/GPU API.

Full infrastructure suite: **125 discovered, 122 passed, 0 failed, 3 skipped** on
Python 3.9.12. Skips remain the existing native OpenCV resize, OpenCV/transforms3d
crop and Open3D export/downsample checks. Focused Python **3.8.19** checks:
**15 tests passed; one real-Torch fixture class skipped** because that interpreter
has no Torch. Real synthetic serialization on the host uses local Torch 2.4.0;
it does not establish pinned-image PyTorch 1.8 execution compatibility.

Tests cover required acknowledgements, fixed selectors/hash, pre-import gate
failures, mount modes/non-root/network/device/freshness checks, collisions,
read-only descriptor metadata/content stability, mutation rejection, CPU mapping,
tensor shape/dtype preservation, scalar/provenance limits, version rejection and
failure-data omission. Unit guards are explicit temporary/mocked fixtures, not a
claim that Codex can create the previously blocked namespace.

The **28 Docker-definition/prerequisite CPU tests passed**. Static checks passed
definition/staging/pre-install policy, five Docker shell parses and Python 3.8
grammar for 63 files. Local Markdown links, whitespace, fixed hash/API/import
audits and shell syntax for the proposed command passed. No Docker engine call,
bubblewrap retry, historical checkpoint deserialization, GPU work, scientific
change, package/image modification, commit or push was performed.
