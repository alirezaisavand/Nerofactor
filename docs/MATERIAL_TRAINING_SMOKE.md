# Bounded A anisotropic material smoke

Protocol `bear-A-two-update-material-smoke-v1`, prepared from infrastructure
`71257544d9038929197d45b3dafb2434ed2d731f`. Native execution is pending review.
The existing material launcher executes the full original loop; the existing
`TwoUpdateTrace` already supplies the required stop and resource cleanup.
This diagnostic reuses that tracer, `TrainingProbe` reductions/clipping/Adam
accounting, stage source selection, schema-1 paths, owned workspace and reports.
No scientific source, equation, legacy YAML or prior evidence is changed.

## Frozen scientific selection and bounds

- Exact A `e1e859779301faced46d6476691e20b546c27ae0`, original
  `scientific_sources/material_a/configs/material/real/bear.yaml`, raw_1024 bear.
  The smoke envelope changes only dataset/mesh/output ownership; no overrides.
- Original Trainer.run, dataset selection/preprocessing, seed6033, Adam
  lr1e-3/eps1e-4/AMSGrad/betas0.9,0.999/weight_decay0, clipping0.7, all sampling,
  losses, tangent/rotation/BRDF/PDF equations stay unchanged. Original512 rays,
  diffuse512/specular1024 samples and full configured100000 steps/intervals remain.
- Bear's configured warm-up is **1000**, not geometry's5000. The two original
  optimizer returns at loop steps0 and1 use LR0 and `5e-4*(1/1000)` =5e-7.
  First Adam moments/counters must advance without weight changes; second return
  must change weights. Do not add a third update or alter LR to get a change.
- CPython tracing verifies the original source boundary and stops at the first
  line after optimizer return2, before any following loop statement. Validation,
  checkpoint saving, extraction and final NVS are additionally forbidden by the
  scoped call observer. Unsupported trace/profile/source conditions fail closed.
  No checkpoints are loaded, initialized or resumed. Natural2000-step validation/
  save, final NVS, full optimization and research reconstruction remain untested.

## Historical mesh admission

Explicitly approved diagnostic input:
`/home/isavand/paper/final_results/meshes-pretrained/real/bear.ply`, SHA-256
`abfead88fc664463b83b273c41f90cecb6e02ed578efb6d7037c8b4494625d24`.
Read-only inspection verified11,047,097 bytes, binary-little-endian VCGLIB PLY,
684,931 finite XYZ vertices, 217,517 triangular faces with valid indices and
217,517 nondegenerate triangles. Bounds are approximately
`[-0.984381,-0.992304,-0.441181]` to `[0.992256,0.992279,0.648253]`.
The reader accepts only this simple reviewed XYZ/triangle layout; malformed,
empty, nonfinite, collapsed, truncated, aliased or hash-mismatched inputs fail.

This establishes structural input compatibility, **not native Open3D/BVH execution,
camera alignment or producing provenance**. Native original initialization/ray-pool
construction remains a gate; record loaded mesh counts/bounds/normals/tracer class
and original normal warnings. Never change mesh contents, axes, scale, winding,
normals or placement to make the smoke pass. Producing revision/geometry protocol
are unknown; the mesh is historical/diagnostic, not an admitted G research handoff.

`run_stage_experiment.py` and ordinary material research execution still require
the original attributable G-to-A handoff. Only the separate smoke CLI selects
the bounded exception, pinned historical mesh and fixed source/configuration.
It requires explicit `--mesh` and `--mesh-sha256` and an execution acknowledgement.
No new ordinary-run bypass or historical checkpoint exception is introduced.

## Evidence and acceptance

The source-hashed Python-call observer records actual entered/completed original
MCShadingNetwork anisotropic forward, mixed anisotropic shading, sampling,
tangent-frame construction, radiance, anisotropic regularization, renderer RGB
loss, NeRFRenderLoss and MaterialRegLoss during **both** training forwards.
RGB/regularizer tensors must retain autograd connections. Original backward
then supplies finite parameter gradients; no gradient hooks or extra backward,
sampling, optimizer/scheduler calls or scientific RNG draws are introduced.

All parameter gradients are summarized before/after original clipping. Require
present, finite and group-nonzero gradients for mx/my/rotation/metallic/kd heads
and shared features. Illumination outer/inner/human groups use actual original
hit/near/human-mask observations: no contributing samples permits explicitly
`expected_inactive` (zero/absent gradients), while an active group with missing/
zero gradients fails. An inactive classification with nonzero gradients also
fails. Per-parameter hashes/statistics remain available, including zeros within
an active group; no head is silently omitted and no seed/batch is retried to pass.
These masks identify eligibility, not full physical sensitivity. If actual native
zero gradients have another legitimate cause, retain failure for review rather
than broaden acceptance after observing it.

Record original YAML, effective trainer, renderer and shader scopes separately.
Top-level energy/specular lambdas are0.01; they are **not** forwarded into
`shader_cfg`. Effective shader energy/specular lambdas stay **0.05** with both
enabled and `reg_change=True`; original reg_lambda1=0.05, perturbation/change
defaults, diffuse-light penalty0.1 and all other settings remain unchanged.
The original RGB cosine-weight schedule also remains unchanged.

`stage-run.json` retains the existing report layout with diagnostic protocol,
mesh identity and `diagnostic_only=true`, complete original splits, actual
sampling/RNG identities, all losses and aggregate, required call counts,
pre/post-clipping gradients, optimizer moments/counters/settings, parameter
changes, effective configurations, runtime/image/source identities and cleanup.
No moment tensors or raw input dataset copies are exported. Full tensor hashes
and detached CPU statistics are diagnostics and never feed back into computation.

The existing runtime redirects HOME/tmp/JIT caches/TensorBoard/console and original
relative output writers into one fresh `/outputs` filesystem view. All original
scene files, cached input products and the historical mesh stay read-only; newly
derived products are owned caches. No cross-run reuse or automatic resume.
Workers, progress and writer are closed on bounded stop or failure; preserve
partial update count and failure report/console. No checkpoints or validation/NVS
images may appear. Timeout/kill can leave incomplete inner reports: preserve the
outer exit and log, and do not count those as successes. Diagnostic outputs are
separate from publication-quality paper runs; two updates prove no full experiment.

## Researcher command on marineris (for review, not executed by Codex)

Use the reviewed checkout; all new files must be available in it. No build, pull,
package installation or image modification. `--validate-only` can replace
`--execute` plus execution acknowledgements for filesystem/config/mesh inspection
without model imports, RNG setup, output allocation or training.

```bash
set -eu
M_SOURCE=/home/isavand/paper/Nerofactor-paper
M_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$M_IMAGE"
test "$(git -C "$M_SOURCE" branch --show-current)" = paper/infrastructure
M_INFRA=$(git -C "$M_SOURCE" rev-parse HEAD)
M_UID=$(id -u)
M_GID=$(id -g)
test "$M_UID" -ne 0
M_GPU=0
M_RUN_ID=$(python3 -B -c 'import uuid; print("bear-a-smoke-"+uuid.uuid4().hex)')
M_BATCH=$(mktemp -d /home/isavand/paper/evidence/bear-A-two-update.XXXXXXXX)
mkdir "$M_BATCH/run"
printf 'infrastructure=%s\nscientific=%s\nimage=%s\ngpu=%s\nrun_id=%s\n' \
  "$M_INFRA" e1e859779301faced46d6476691e20b546c27ae0 "$M_IMAGE" "$M_GPU" "$M_RUN_ID" > "$M_BATCH/launch.txt"
git -C "$M_SOURCE" diff HEAD --binary > "$M_BATCH/reviewed-worktree.patch"
set +e
docker run --rm --pull=never --init --user "$M_UID:$M_GID" \
  --gpus "device=$M_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
  --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
  --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$M_SOURCE,dst=/workspace,readonly" \
  --mount type=bind,src=/home/isavand/paper/data,dst=/inputs/datasets,readonly \
  --mount type=bind,src=/home/isavand/paper/final_results,dst=/inputs/thesis,readonly \
  --mount "type=bind,src=$M_BATCH/run,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout \
  "$M_IMAGE" --signal=TERM --kill-after=30s 900 \
  /usr/bin/python3 -I -B /workspace/run_material_training_smoke.py \
  --scientific-commit e1e859779301faced46d6476691e20b546c27ae0 \
  --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
  --mesh /inputs/thesis/meshes-pretrained/real/bear.ply \
  --mesh-sha256 abfead88fc664463b83b273c41f90cecb6e02ed578efb6d7037c8b4494625d24 \
  --run-id "$M_RUN_ID" --execute --acknowledge-historical-diagnostic-mesh \
  --confirm-isolated-runtime --image-id "$M_IMAGE" --infrastructure-commit "$M_INFRA" \
  > "$M_BATCH/terminal.log" 2>&1
M_EXIT=$?
set -e
printf '%s\n' "$M_EXIT" > "$M_BATCH/exit-code.txt"
cat "$M_BATCH/terminal.log"
test "$M_EXIT" -eq 0
sha256sum "$M_BATCH/run/stage-run.json"
printf 'Review diagnostic evidence: %s\n' "$M_BATCH"
```

Budget900 seconds +30-second termination grace, four CPUs,64GiB/no extra swap,
128PIDs,2GiB shm, one explicitly selected A10080GB PCIe; no GPU-memory cap claimed.
Original full scene preprocessing may dominate runtime/memory. Preserve timeout,
OOM, native import/geometry failures and diagnostics; never reduce sizes/samples,
alter mesh or change gradients to obtain a passing report.

Acceptance checklist: exit0 and complete bounded-stop report; exactly two original
Adam returns; required original anisotropic/RGB/regularizer calls in each; finite
losses and active-group gradients before/after original clipping; expected inactive
lighting explicitly explained; first LR0/moments advanced/weights unchanged;
second LR5e-7/weights changed; original effective regularizers0.05 preserved;
exact A/config/mesh/source identities and unchanged read-only inputs; cleanup
succeeded, no checkpoints/validation/NVS, no research-handoff/full-training claim.

## CPU/static checks

Host Python3.9.12, CPU Torch2.4.0: **247 tests total;244 passed,3 skipped,0 failed**,
including9 focused material smoke tests. Reused tests cover generic exact stopping,
trace rejection, clipping, nonfinite loss/gradient/state, collisions, source
snapshot preservation and owned cleanup. New fixtures reuse loader/logger helpers
and execute unchanged A Trainer/RGB/regularizer/dispatch/light-mask source bodies
with explicit toy shader/BVH/operators; they do not establish native BRDF parity.
The3 existing skips concern unavailable OpenCV/transforms3d crop, Open3D export,
and OpenCV RAW resize. Run `python -B -m unittest discover -s tests/infrastructure -v`.

`python -B docker/paper/static_checks.py` passes definition/mount checks,5 RUN
shell parses and Python3.8 grammar for168 files; the existing build-readiness
warning concerns the incomplete checked-in artifact inventory, not the already
working researcher image. New command shell syntax is checked without execution.
Original335 scientific/configuration/evidence/protocol files, reused tracer,
base probe/workspace, ordinary CLI and Dockerfile match the review baseline.
Native Python3.8/CUDA/BVH/backward, resource requirements and actual light-path
gradient activity remain unverified until the reviewed command is executed.
