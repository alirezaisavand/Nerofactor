# Historical G bell mesh → pinned A material diagnostic

This preparation adds two independent researcher-side actions. No native action
has been executed by Codex. This is diagnostic-only: the historical checkpoint's
producing revision/split remain unresolved; resolution128 is not canonical512,
and neither action admits an attributable research G→A handoff. Existing evidence
and ordinary `run_stage_experiment.py` handoff checks remain unchanged.

## Admitted inputs and source audit

- G: `79252178e65914ad1acbc14790b1514f276bfea0`.
- A: `e1e859779301faced46d6476691e20b546c27ae0`.
- Original A YAML: `scientific_sources/material_a/configs/material/syn/bell.yaml`.
- Mesh: `meshes/bell_shape-best-150000-r128.ply`, SHA-256
  `eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5`.
- Inspected extraction report:
  `/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run/G-mesh-diagnostic.json`,
  SHA-256 `caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f`.
  Protocol `bell-historical-best-G-mesh-r128-v1`; run
  `bell-g-best-mesh-2f4a25e1828f47109433f42a56d53f4a`; complete.
- The report and actual PLY hash were independently inspected, without loading
  the historical checkpoint or camera pickles. The report binds the admitted
  best-checkpoint SHA, strict G loading and unchanged network, source snapshot,
  bbox±1/level0/r128, no transforms, and serialized triangle winding preservation.
  It reports30806 vertices/61612 nondegenerate triangles, no boundary/nonmanifold
  edges, consistent winding, signed volume−0.8496040975861767. The researcher's
  one-component observation has not been independently recomputed here.

A `NeROMaterialRenderer._init_geometry()` reads the PLY with Open3D, creates
`RayTracer(vertices,triangles)`, then computes missing vertex normals **in memory**.
There is no mesh normalization, rescaling, triangle reversal or file write.
The ray tracer receives the original triangles before vertex-normal computation.
Its Python wrapper casts camera rays to contiguous float32, moves them to CUDA
and calls `_raytracing` to return intersection positions, face normals and depth.
Vertex normals are not supplied to the BVH. A `trace()` then reshapes depth,
performs exactly `normals = -normals`, applies `F.normalize`, and uses depth<10
for hits. The existing warning is retained; it does not authorize a normal flip.

The extension's recovered source pin is `29329268a86921f1c938b05277937369531209a4`.
The CUDA implementation is external to the pinned A snapshot and its original
checkout is unavailable in this session. Consequently this audit cannot establish
its cross-product implementation from local C++ source. Native reports record
actual extension binary SHA; original BVH normals and A's resulting normals are
observed directly. Installed/source identities are distinct from historical
binary-build identity. Negative signed volume alone is not proof of physically
appropriate rendering normals.

A material bell uses `syn/bell`, original full800×800 RGB/cameras/masks, scale1,
original half-pixel rays and camera-to-world direction conversion. No geometry
quarter-resolution preprocessing is substituted. `build_imgs_info()` already
requires original `masks_train/<id>.png`; missing/mismatched masks block execution,
with no depth-derived, resized or invented replacement. Dataset depth maps are
not needed for the frozen silhouette test. A's split helper seeds Python6033,
shuffles ordered0…127, selects first5 NVS, next1 validation58, remaining122 train.
These are **current A** split labels, not historical held-out membership claims.

Bell disables human lighting and `is_real=False`; its human-light predictor is
absent. Original diffuse/specular samples512/1024,512 training rays, anisotropic
BRDF/tangent/sampling/losses are unchanged. Top-level energy/specular0.01 are
**not forwarded** to `shader_cfg`; effective shader defaults remain0.05. Actual
trainer/renderer/shader settings, losses and optimizer state are recorded for
training. Ray-only execution constructs no shader, renderer model, Trainer or
optimizer: an external facade binds the original mesh/ray methods unchanged.

## Frozen protocols and gates

`historical-G-bell-to-A-rays-v1` traces exactly three full views: first ordered
training ID14, validation58, first NVS ID11, each recorded with actual source camera
ID and full ordered split lists. Original `_construct_ray_batch(...,is_train=False)`
returns all rays, including misses, without shuffling a training pool. The original
split call retains its seed/shuffle effects; ray observation adds no RNG draws.
This bounded camera check does not establish all-view geometry/evaluation quality.

A scoped CPython return observer copies raw BVH outputs and original A trace
outputs. It never wraps/replaces the tracer, changes rays, interpolates normals,
or changes winding. Original Open3D vertices and **ordered triangles** must equal
the serialized PLY. Hit positions/normals/depths must be finite, depths positive
and<10; miss payload positions/normals may be unspecified and are not physically
interpreted. Cameras/ray directions and all returned depths must be finite.

Frozen provisional diagnostic bounds, declared before any native observation:

| Check per view | Bound |
| --- | --- |
| Nonempty ray hits | At least1 |
| Original A normal negation/normalization maximum absolute difference | ≤2e-6 |
| Final normal length error | ≤2e-6 |
| Intersection vs origin+depth×direction max coordinate difference | ≤2e-4 SDF units |
| Final normals facing view (dot(normal,−ray)>0) | ≥0.99 of hits |
| Raw normals opposite view | ≥0.99 of hits |
| Hit silhouette vs original mask IoU | ≥0.90 |

These gate physical camera/normal usability, not paper-quality reconstruction.
Report hit/miss counts/rates, depth range, dot statistics and mask intersection/
union. Record all check outcomes before a failing gate is raised. A failure blocks
training: preserve evidence and investigate camera scale/coordinates, winding,
BVH semantics, resolution error or masks. Do not fit bounds, flip normals, align
meshes, resize inputs or change rendering mathematics to obtain a pass. A changed
protocol requires separate review/versioning.

`historical-G-bell-to-A-two-update-v1` additionally requires an explicit completed
ray report **and its SHA**, all three passing views and intact array artifacts,
matching mesh/extraction/source/config/infrastructure/runtime/image/input identities.
It executes the actual A `Trainer.run()` using the existing `MaterialProbe` and
`TwoUpdateTrace`, stopping immediately after successful original Adam return2,
before validation/checkpoints/NVS. Original seed6033, Adam eps1e-4/AMSGrad/betas,
clip0.7 and warm-up/cosine schedule are unchanged. LR0 on step0 advances moments
without weight changes; LR5e-7 on step1 must change weights. Original anisotropic,
RGB and material regularization calls must execute both passes, with finite losses,
active predictor/shared/illumination gradients and original pre/post-clip checks.
Absent human lighting is expected, not an unexplained missing gradient group;
other inactive lighting is classified from actual original path/mask samples.
No historical weights, optimizer restoration or implicit resume are involved.
Training starts independently, with the original constructor/seed/RNG order;
ray-only execution does not consume the training process's RNG state.

## Outputs, isolation and failure behavior

Each action uses a fresh empty mounted output root and unique run ID. Outputs:
`bell-material-diagnostic.json`, untouched `config.original.yaml`, existing
`cache/stage-cache-identity.json`, owned runtime/preprocessing caches and console
logs. Ray diagnostics additionally save `evaluations/<role>-<sourceID>.npz` with
full rays, raw/final normals, positions/depth/hit flags, RGB, mask, K and pose
(no object arrays/pickle), and hashes. These arrays are diagnostic artifacts,
not NVS images or publication metrics. The report contains original configuration,
explicit dataset/mesh path-only runtime overrides, source/image/binary hashes,
mesh/extraction and ray gate identities, runtime, inputs and limits. Two-update
reports also contain all existing optimizer/gradient/parameter/call diagnostics.

Source, dataset, extraction mesh/report and prior ray evidence are mounted read-only.
Only fresh `/outputs` is writable; all original relative-path writers land in the
existing owned stage-workspace. The historical checkpoint need not be loaded or
exposed for either action. The runtime rejects root/network/writable input mounts,
foreign source/image/GPU, reused outputs and mixed scientific namespaces.
Owned workers, progress/writer and profile/trace state are cleaned via the existing
helpers; default tensor type/HOME/CWD/cache/log handling are restored. Failed runs
and partial outputs are retained, never deleted/reused. The900s process timeout
bounds native execution; hard timeout/OOM can leave only partial logs/reports and
cannot guarantee graceful worker cleanup. No GPU-memory bound is claimed.

## Researcher commands, inside tmux

Review the patch before either native action. Start a fresh tmux session:

```bash
tmux new-session -s "bell-ga-$(python3 -B -c 'import uuid; print(uuid.uuid4().hex[:12])')"
```

In that shell, define the shared launcher and run **rays only**. No builds/pulls
or package changes. All existing evidence remains read-only.

```bash
set -eu
B_SOURCE=/home/isavand/paper/Nerofactor-paper
B_GEOMETRY=/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run
B_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$B_IMAGE"
test "$(git -C "$B_SOURCE" branch --show-current)" = paper/infrastructure
B_INFRA=$(git -C "$B_SOURCE" rev-parse HEAD)
B_UID=$(id -u)
B_GID=$(id -g)
test "$B_UID" -ne 0
B_GPU=0
bell_diagnostic() {
  B_ACTION=$1
  shift
  B_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-G-to-A.XXXXXXXX)
  mkdir "$B_BATCH/run" "$B_BATCH/empty-thesis"
  B_RUN_ID=$(python3 -B -c 'import uuid; print("bell-ga-"+uuid.uuid4().hex)')
  printf 'action=%s\ninfrastructure=%s\nimage=%s\ngpu=%s\nrun_id=%s\n' \
    "$B_ACTION" "$B_INFRA" "$B_IMAGE" "$B_GPU" "$B_RUN_ID" > "$B_BATCH/launch.txt"
  git -C "$B_SOURCE" diff HEAD --binary > "$B_BATCH/reviewed-worktree.patch"
  set +e
  docker run --rm --pull=never --init --user "$B_UID:$B_GID" \
    --gpus "device=$B_GPU" --network=none --read-only --cap-drop=ALL \
    --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
    --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
    --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$B_SOURCE,dst=/workspace,readonly" \
    --mount type=bind,src=/home/isavand/paper/data,dst=/inputs/datasets,readonly \
    --mount "type=bind,src=$B_BATCH/empty-thesis,dst=/inputs/thesis,readonly" \
    --mount "type=bind,src=$B_GEOMETRY,dst=/inputs/geometry,readonly" \
    --mount "type=bind,src=$B_BATCH/run,dst=/outputs" \
    "$@" --workdir /workspace --entrypoint /usr/bin/timeout \
    "$B_IMAGE" --signal=TERM --kill-after=30s 900 \
    /usr/bin/python3 -I -B /workspace/run_bell_material_diagnostic.py \
    --action "$B_ACTION" --scientific-commit e1e859779301faced46d6476691e20b546c27ae0 \
    --datasets-root /inputs/datasets --thesis-root /inputs/thesis \
    --geometry-root /inputs/geometry --output-root /outputs \
    --mesh /inputs/geometry/meshes/bell_shape-best-150000-r128.ply \
    --mesh-sha256 eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5 \
    --extraction-report /inputs/geometry/G-mesh-diagnostic.json \
    --extraction-sha256 caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f \
    --run-id "$B_RUN_ID" --execute --acknowledge-diagnostic-mesh-and-trusted-cameras \
    --confirm-isolated-runtime --image-id "$B_IMAGE" --infrastructure-commit "$B_INFRA" \
    "${B_RAY_ARGS[@]}" > "$B_BATCH/terminal.log" 2>&1
  B_EXIT=$?
  set -e
  printf '%s\n' "$B_EXIT" > "$B_BATCH/exit-code.txt"
  cat "$B_BATCH/terminal.log"
  printf 'Retained evidence: %s\n' "$B_BATCH"
  test "$B_EXIT" -eq 0
  sha256sum "$B_BATCH/run/bell-material-diagnostic.json"
}
B_RAY_ARGS=()
bell_diagnostic rays
B_RAY_DIR=$B_BATCH/run
printf 'Review ray evidence before the separate training command: %s\n' "$B_RAY_DIR"
```

Stop and review the ray report/arrays. Only after all ray gates pass and the result
is reviewed, run this separate command in the same tmux shell. It mounts previous
ray evidence read-only and allocates entirely new outputs; it never reuses rays'
output root. If the checkout/protocol/image/runtime/dataset changed, admission
fails: do not bypass it.

```bash
B_RAY_SHA=$(sha256sum "$B_RAY_DIR/bell-material-diagnostic.json")
B_RAY_SHA=${B_RAY_SHA%% *}
B_RAY_ARGS=(--ray-report /inputs/ray-evidence/bell-material-diagnostic.json --ray-sha256 "$B_RAY_SHA")
bell_diagnostic two-update --mount "type=bind,src=$B_RAY_DIR,dst=/inputs/ray-evidence,readonly"
```

`--validate-only` replaces `--execute` for metadata-only preflight; it never
allocates outputs, imports science, loads camera pickle or consumes scientific
RNG state. Native camera pickle loading occurs only after explicit trust and
isolated-runtime acknowledgements; neither action deserializes checkpoints.

Acceptance: exact hash/report/source/config; no transforms or triangle changes;
three actual ordered source IDs and frozen ray gates pass; raw/final normal arrays
support the interpretation; separate training report has exactly2 successful Adam
returns, original calls/losses/gradients, LR0/5e-7 and nonzero second-update changes;
cleanup succeeds, no checkpoints/validation/NVS and no research-handoff claim.
Failed/missing geometry, masks, binary/runtime differences, resource exhaustion,
profile incompatibility or frozen gate failure block progression. Do not launch
full experiments from this diagnostic's success alone.

## CPU/static validation and limits

Host Python3.9 CPU checks:10 new focused tests pass; full infrastructure suite
268 tests total,264 passed,4 skipped,0 failed. Existing skips cover unavailable
OpenCV/transforms3d crop, Open3D point export, OpenCV RAW resize, and native CPU
PyMCubes/Trimesh export. Fixtures use disposable files, toy autograd/BVH/light
operators and unchanged original A method/loop bodies; they are not GPU acceptance.
The existing bear tests, shared-decorator regression, bounded-stop, finite checks
and owned-cleanup tests still pass. The minimal shared probe generalization leaves
its default bear settings/groups unchanged and only selects bell's expected
synthetic settings/absent human network in the separate subclass.

`python -B docker/paper/static_checks.py` passes definition checks,5 Docker RUN
shell parses and Python3.8 grammar for176 files. Documented Bash blocks parse;
real hash-gated ray preflight passes with native imports explicitly blocked and
creates no output directory.319 baseline scientific/configuration/evidence/
protocol/policy files match their reviewed bytes, including unchanged ordinary
stage execution/handoff, snapshots, tracer, base probe, Dockerfile and YAML.

These host checks do not verify native Python3.8 runtime, CUDA/BVH/shader backward,
physical normals or silhouettes, full-scene memory/time requirements or actual
native gradients. The static checker's existing incomplete-build-manifest warning
is separate from the admitted working image; no build/readiness/image claim is
added. No Docker, native model, camera/checkpoint loading or GPU execution occurred.
Researcher-side ray results must pass the frozen gates and be reviewed before the
separate bounded training command. No native success is asserted in this milestone.
