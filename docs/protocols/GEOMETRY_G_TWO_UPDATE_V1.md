# Bell G geometry two-update protocol v1

Protocol identity: **`bell-G-two-update-training-v1`**. Status: implementation and
CPU/static fixtures prepared for review; **no native G execution accepted**.
This supersedes `bell-two-update-training-v1` for selected geometry-baseline
acceptance only. Preserve that A protocol and its real R observation unchanged.

## Scientific identity and bounded computation

Stage I source is G `79252178e65914ad1acbc14790b1514f276bfea0`, exact files under
`scientific_sources/geometry_g`, recorded by `scientific_sources/manifest-v1.json`.
A fresh process closes network/dataset/train/utils/raytracing/colmap namespaces
to this tree. Missing modules fail; no maintained-A fallback. Stage II selects
A `e1e859779301faced46d6476691e20b546c27ae0` in a separate process/tree.

Use G's original `configs/shape/syn/bell.yaml`, byte-for-byte. Preserve full
`total_step=300000`, worker count 8, train-log interval 20, validation/save
intervals 1000, seed 6033, registered eight losses and their original scopes.
Keep original ray pool, batch 512, sample counts, tensor precision, model/shader
initialization, original forward/backward, gradients and original loss reduction.
No historical weights or optimizer/RNG restoration, mesh input, sample reduction,
early validation, forced checkpoint or replacement training loop is permitted.

G `get_database_split(...,'validation')` seeds global Python RNG to 6033, copies
numeric input IDs, does **not shuffle**, and returns two lists. For bell IDs
`"0"`–`"127"`, validation is `["0"]`, training is ordered `"1"`–`"127"`;
there is no NVS partition. G's validation DummyDataset and actual renderer both
call the original G helper; no A three-list shim is supplied. Preserve both
original calls and their RNG resets. G `test` split reads its original pickle
without the validation seed reset; it is not used by this bounded smoke.

G constructs actual Torch Adam with `lr=1e-3` only. In the reviewed Torch 1.8.1
runtime the defaults are betas `(0.9,0.999)`, eps `1e-8`, weight decay 0 and
AMSGrad false. Its original WarmUpCosLR uses warm-up 5000, LR `5e-4`, alpha `.05`
and configured horizon 300000, without A's 100000 clamp. These expectations were
checked against G source and CPU execution of its factory/scheduler, not copied
from A. At steps 0 and 1, the actual scheduled LRs are 0 and
`5e-4*(1/5000)`; those first two LR values happen to equal A's but cannot establish
scheduler equivalence. Preserve G's **absence of gradient clipping**. G opacity
loss begins at `.001`, transitioning after 20000 toward `.01` at 50000;
AppShadingNetwork also retains G's own normalization epsilon behavior.

Run the actual pristine G `Trainer.run`. A separate CPython line observer
identifies original forward/backward/optimizer boundaries and aborts immediately
at the first line after **exactly two successful optimizer returns**, steps 0/1.
The observer rejects changed/reentrant code, existing debugger/coverage traces,
wrong progress, state or stage. No method body or optimizer is substituted.
Owned workers, progress and TensorBoard writer are closed on success/failure;
the original Linux fork loader context is retained. A non-fork process context
is rejected because worker import isolation under spawn needs separate review;
the framework never sets a different multiprocessing start method.
original failures and partial-update evidence remain failures. At zero first LR,
moments/counters advance while parameters remain unchanged; second-call parameter
change is required. CPU toy autograd exercises these verified G rules, not the
native geometry model. Later schedules, validation, saving and convergence are
outside this prefix; source/CPU checks cover their identity, not native behavior.

## Input/output and configuration arms

Mount source and original datasets read-only, empty thesis placeholder read-only,
one explicit A100 80GB PCIe, no network, non-root, read-only root filesystem.
Use existing immutable image
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`.
Do not rebuild it. Python 3.8/Torch 1.8.1+cu111/CUDA 11.1 are required.

A unique empty output mount is allocated exclusively. Original source files are
linked into an experiment-owned filesystem view after read-only mount checks.
Missing derived cache paths are writable in this view; existing input/cache
artifacts remain read-only. No resizing/cropping/depth calculation is replaced.
Selected scene source content hashes, metadata, full config and scientific/
infrastructure hashes identify the cache; no cross-run cache reuse is allowed.
All relative G writers use the owned working directory. Original Trainer
constructor/seed executes unchanged; an I/O-only subclass then rebinds model/log/
checkpoint destination attributes. It rejects implicit checkpoint continuation.
No optimizer/network/dataset/scheduler/loss/loop override occurs in this subclass.

- **R:** independently load pinned G legacy YAML via FullLoader; never call the
  schema loader. Full original scientific config remains intact.
- **R-repeat:** new independent process and newly initialized weights, same R
  route, fresh ID/output. No old A R or G checkpoint may be reused.
- **C:** schema-1 envelope `configs/experiments/geometry_g_bell_v1.yaml`, approved
  config root `/workspace`, selecting the exact pinned G YAML and no overrides.
  That broader read-only config root is explicit because the scientific snapshot
  is outside the legacy `configs/` folder. Escaping the approved root fails.

Only dataset/mesh/output ownership paths may change. Record raw YAML, original
and runtime/effective trainer/renderer/shader/loss configs, ordered IDs, source
manifest, actual imported module origins, scientific commit separately from
infrastructure commit, image/runtime/backend/thread/package identities, input
hashes, initial parameter/buffer/Adam identities, losses, gradients, actual LR,
sampling cursor/ray hashes, moments/counters, parameter changes and RNG records.
`stage-run.json` is complete only after successful stop, cleanup, input/source
rechecks and output hashes. No `.pth` is expected before the unchanged interval.
Keep partial failures; never overwrite, retry into a run, or relabel a failure.

## Fresh repeatability and compatibility gates

The first G protocol deliberately uses **exact equality** of recorded tensor/
state hashes and scientific scalars; no A or RGB numerical tolerance is imported.
Tensor records contain full shape/dtype/hash and detached CPU64 norms/statistics;
no norm-only agreement can hide element differences. Require contiguous update
records, exact G origin/config/split/Adam/scheduler/no-clipping identities, fresh
state, finite losses/gradients/state and correct ray cursor/optimizer progress.
Elapsed time and memory instrumentation are recorded but excluded from equality.

First compare R versus R-repeat. Only after this gate passes may C be loaded/
interpreted. Then compare R versus C. Both scientific inputs, effective settings,
initial weights/buffers, sample/RNG records, losses, gradients, parameter changes
and optimizer state must agree. Reject A reports, unknown provenance, source
changes, architecture/state mismatches, clipping or checkpoint initialization.
Any mismatch blocks acceptance. Do not tune seeds or relax bounds after viewing
results: a different policy needs a separately reviewed version, preserving all
attempts. These strict first gates may expose native nondeterminism; a failed
hash comparison is not automatically proof of a scientific defect. A two-update
match still cannot establish full training, historical thesis reproduction,
checkpoint-selection equivalence or G-produced origin of the historical bell
checkpoint. No native acceptance exists until all replacement evidence is reviewed.

## First researcher-side G run (commands for review, not execution approval)

```bash
set -eu
G_SOURCE=/home/isavand/paper/Nerofactor-paper
G_DATA=/home/isavand/paper/data
G_IMAGE=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$G_IMAGE" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
G_INFRA=$(git -C "$G_SOURCE" rev-parse HEAD)
G_UID=$(id -u)
G_GID=$(id -g)
test "$G_UID" -ne 0
G_GPU=0
G_TOKEN=$(python3 -B -c 'import uuid; print(uuid.uuid4().hex)')
G_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-G-two-update.XXXXXXXX)
mkdir "$G_BATCH/empty-thesis" "$G_BATCH/r"
printf 'infrastructure=%s\nscientific=%s\nimage=%s\ngpu=%s\n' \
  "$G_INFRA" 79252178e65914ad1acbc14790b1514f276bfea0 "$G_IMAGE" "$G_GPU" > "$G_BATCH/launch.txt"
set +e
docker run --rm --pull=never --init --user "$G_UID:$G_GID" \
  --gpus "device=$G_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
  --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
  --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$G_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$G_DATA,dst=/inputs/datasets,readonly" \
  --mount "type=bind,src=$G_BATCH/empty-thesis,dst=/inputs/thesis,readonly" \
  --mount "type=bind,src=$G_BATCH/r,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout \
  "$G_IMAGE" --signal=TERM --kill-after=30s 900 \
  /usr/bin/python3 -I -B /workspace/run_stage_experiment.py \
  --stage geometry --scientific-commit 79252178e65914ad1acbc14790b1514f276bfea0 \
  --arm R --run-id "g-bell-r-$G_TOKEN" --datasets-root /inputs/datasets \
  --thesis-root /inputs/thesis --output-root /outputs --execute \
  --confirm-isolated-runtime --image-id "$G_IMAGE" --infrastructure-commit "$G_INFRA" \
  > "$G_BATCH/r-terminal.log" 2>&1
G_EXIT=$?
set -e
printf '%s\n' "$G_EXIT" > "$G_BATCH/r-exit-code.txt"
cat "$G_BATCH/r-terminal.log"
test "$G_EXIT" -eq 0
sha256sum "$G_BATCH/r/stage-run.json"
printf 'G reference evidence: %s\n' "$G_BATCH"
```

Budget: one independent G process, two successful optimizer calls, 900 seconds
plus 30-second termination grace, four CPUs/64 GiB host RAM/no extra swap, one
selected GPU (no GPU-memory cap claimed). Unexpected scientific/native behavior
must stop with a failure report, never a source repair. Preserve outer exit/
terminal records even if the inner report remains incomplete. Review R before
scheduling R-repeat/C. This command does not mount historical weights/results.

After separate approval, R-repeat uses a **new empty output**, unique ID and
`--arm R-repeat` with otherwise identical mounts/environment. C uses a third
fresh output/ID and `--arm C --cfg /workspace/configs/experiments/geometry_g_bell_v1.yaml
--config-root /workspace`. Do not rerun into R or use the old A output.
The CPU-only comparer is `compare_g_geometry_training.py`: select explicit
`--reference`, `--repeat`, optional `--candidate`, their required `--*-sha256`
values and a fresh `--output-root`. It writes `G-training-comparison.json` and
requires repeatability before candidate interpretation. Run it in the same
reviewed source version, without GPU/checkpoint access; do not borrow the A
comparer or source declarations. Native full training, G checkpoint inference/
architecture, natural interval tests, extraction and G-to-A material validation
remain separately reviewed later gates.
