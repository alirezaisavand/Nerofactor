# Historical bell best checkpoint -> G mesh diagnostic v1

Protocol **`bell-historical-best-G-mesh-r128-v1`**, prepared from infrastructure
`d4e80bde47fbcb0d2a30fe4bf462d7a28e8fb9d1`. Implementation/CPU preparation only;
no native mesh has been extracted by Codex. This is a separate best-checkpoint
selection protocol, not a change to the original CLI or a new G training run.
The researcher reports completed G/A two-update smokes and strict G checkpoint
compatibility. No accepted evidence or scientific snapshot is rewritten here.

## Source and checkpoint audit

Pinned G: `79252178e65914ad1acbc14790b1514f276bfea0`. Pinned material A remains
`e1e859779301faced46d6476691e20b546c27ae0`; no material code executes here.
The immutable source manifest and existing
[G source audit](evidence/g-validation-checkpoint-source-audit-v1.json) remain
authoritative. Relevant exact G file SHA-256 values:

| G file | SHA-256 |
| --- | --- |
| `extract_mesh.py` | `63b1cf824e564b67910a160ff080338dc9d9546f441408fd937ca4473d3d39a9` |
| `network/field.py` | `06fb6606c580dede3e7e12ddfad2a2fed9368fa4adfcadf7cf7cef73e940617f` |
| `network/renderer.py` | `38c280ed812c4fb22b980caf842f0cd302acbee3bde9c449972dabd23485584b` |
| `configs/shape/syn/bell.yaml` | `026f9a683c0cec3bee7e7ff65c07316c61c2d32a8338a18aa839345e28fefd8f` |

Original `extract_mesh.py` loads `data/model/<name>/model.pth` with `torch.load`
without map_location, obtains its stored step, loads network state with default
strict behavior, then `eval().cuda()`. It constructs the full original
NeROShapeRenderer with `training=False`, so dataset initialization is omitted.
It sets default CUDAFloat tensors, creates bbox±ones, invokes original
extract_geometry under no_grad, constructs `trimesh.Trimesh(vertices,triangles)`
with defaults, then exports `data/meshes/<name>-<step>.ply`. CLI default is512.

The original G SDF is the renderer's weight-normalized network: 3D input,
frequency6 positional encoding, hidden256,8 configured layers, skip index4,
257 output channels, scale1.0 and Softplus(beta100) hidden activations. Query
`sdf(x)` takes only `forward(x)[...,:1]`; forward uses its original scale and
skip concatenation divided by sqrt2. The constructor's `sdf_activation` argument
does not introduce an output transformation in this implementation. Do not
repair/reinterpret it or substitute an SDF-only architecture for strict loading.

Original extract_fields uses linspace/meshgrid in XYZ order,64 samples per axis
per chunk, a dense float32 field and **norm(points)>=1 -> +1** outside the sphere.
Original PyMCubes marching_cubes uses isovalue0; vertices map by
`v/(resolution-1)*(bbox_max-bbox_min)+bbox_min`. No axis swap, scale fit, smoothing,
component filtering, winding flip, normal fix or extra geometric processing.

## Explicit best-checkpoint admission

Selected file: `thesis:model/bell_shape/model_best.pth`, SHA-256
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`, stored150000.
It is **never** supplied under the original `model.pth` name or substituted into
the old CLI. Its producing revision/configuration/split and new-G-run attribution
remain unresolved; G compatibility is an implementation observation.

Independently inspected existing compatibility report:
`/home/isavand/paper/evidence/bell-G-lifecycle.dnc5Hmhd/compatibility/G-diagnostic.json`,
100,210 bytes, SHA-256
`73e405b99a3041bb6bc7195e220bd128b82ea3ee54d5a4f5f4df14476b20435a`.
It reports complete strict G compatibility at declared infrastructure71257544,
137/137 tensors, no missing/unexpected/shape/dtype mismatches, step150000 and
best23.787122565451323, with exact pinned G source files and original YAML hash.
Source/report hashes and contents are checked, not a new model load by Codex.

The new CLI requires explicit checkpoint path/hash, that exact report/hash, G
commit and resolution128. Preflight reads YAML/JSON and opaque checkpoint bytes
only: no Torch/scientific imports, RNG setup, allocation or checkpoint loading.
The old lifecycle live-infrastructure gate remains unchanged; this distinct
protocol binds the older report's fixed SHA/scientific identity and rechecks
the actual architecture and loaded tensor bytes against it in the new runtime.

Native loading requires explicit trust and isolation. Torch1.8
`torch.load(..., map_location='cpu')` still deserializes the **whole trusted pickle
container**, including optimizer payload, but restores **network weights only**.
Require exact metadata and all keys/shapes/dtypes/finite values before strict
loading; loaded bytes must match the admitted report. No partial load, optimizer
construction/restoration, trainer, seeding override, rendering or training.
Original extraction has no explicit seed; original construction RNG is consumed
but all saved parameters/buffers are strictly loaded before any SDF query.

## Bounded extraction, inspection and artifacts

Resolution128 is diagnostic, **not canonical512**. It evaluates128³ points in
eight original64³ chunks, retaining bbox±1/isovalue0/outside+1 and exact SDF and
marching-cubes code. A read-only profiler verifies original call counts, no_grad,
finite float32 SDF/field and fixed query budget; forbidden renderer/dataset/
backward/optimizer bodies are identified using inspect.unwrap. No extra SDF
queries or science method replacements are introduced. The original model's
weights/buffers must remain byte-identical before/after extraction.

Raw and processed mesh arrays must be finite, nonempty, within the original box,
integer-indexed and entirely nondegenerate (strict positive triangle area in a
detached float64 diagnostic calculation). Failure never triggers a mesh repair.
Preserve original default Trimesh construction and export. **Export occurs before
lazy normal inspection**, since inspecting/cache-populating normals first could
affect exporter behavior. Reopen only the newly generated PLY with process=False
for inspection; validate serialized geometry, unchanged triangle ordering/winding,
and report coordinate quantization. No modified readback is re-exported.

Record raw/processed/serialized bounds/counts/areas, boundary/nonmanifold edges,
two-face edge winding consistency and signed volume. These diagnose topology;
they do not prove physical/global outward orientation, particularly for open or
multiple shells. Record actual PLY header/normal properties and Trimesh lazy
face/vertex normal lengths/zero counts after export. Do not flip or normalize
the mesh itself. Normal arrays are separate diagnostic outputs; export bytes
and processed topology must stay unchanged during inspection.

Coordinate convention is unmodified checkpoint G SDF XYZ, queried on[-1,1]³;
physical units and producing camera normalization are unverified. Explicit
external rescaling/reorientation is false. Record actual Trimesh/PyMCubes versions,
module hashes, image ID declaration and standard GPU/Torch/CUDA/ABI runtime
identity; preserve original library defaults rather than invent a normal rule.

Fresh `/outputs` contains:

- `G-mesh-diagnostic.json`: status/phase, checkpoint selection/stat/hash, original
  YAML/effective scopes, source/config/runtime/compatibility identities, settings,
  call budget, network immutability, mesh diagnostics and output hashes.
- `meshes/bell_shape-best-150000-r128.ply`, plus numeric NPY snapshots of raw and
  processed vertices/faces and diagnostic face/vertex normals (allow_pickle=False).
- Original YAML, console log and owned HOME/tmp/JIT caches. Input assets/configs
  are read-only links; no dataset is instantiated/read. No checkpoints are saved.

Use existing PathResolver/RunStore/runtime logging/cache isolation and cold G
namespace selection. Restore default CPUFloat/profile and release model/cache
resources on extraction failure. Partial outputs/failure logs are retained;
timeouts/kills may leave a running inner report, never a successful admission.
All original source, dataset, thesis and compatibility mounts stay read-only.

The manifest has `research_geometry_admitted=false` and
`research_handoff_admitted=false`, no attributable new G training run or
`mesh_outputs` handoff assertion. The ordinary research handoff validator is
unchanged and rejects this diagnostic report. Same-scene G-to-A integration and
full material training need a later reviewed scope after native mesh inspection.

## Exact researcher command: enter tmux, then run

No command below was run by Codex. In the regular marineris SSH terminal:

```bash
tmux new-session -s "bell-g-mesh-$(python3 -B -c 'import uuid; print(uuid.uuid4().hex[:12])')"
```

Inside that new tmux session, paste the complete block. Use the reviewed source
and installed image; no build/pull/install. Every attempt gets a new evidence
directory and run ID. Do not reuse any previous smoke/compatibility output.

```bash
set -eu
E_SOURCE=/home/isavand/paper/Nerofactor-paper
E_PROOF=/home/isavand/paper/evidence/bell-G-lifecycle.dnc5Hmhd/compatibility
E_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$E_IMAGE"
test "$(git -C "$E_SOURCE" branch --show-current)" = paper/infrastructure
E_INFRA=$(git -C "$E_SOURCE" rev-parse HEAD)
E_UID=$(id -u)
E_GID=$(id -g)
test "$E_UID" -ne 0
E_GPU=0
E_RUN_ID=$(python3 -B -c 'import uuid; print("bell-g-best-mesh-"+uuid.uuid4().hex)')
E_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-G-best-mesh.XXXXXXXX)
mkdir "$E_BATCH/run"
printf 'infrastructure=%s\nscientific=%s\nimage=%s\ngpu=%s\nrun_id=%s\n' \
  "$E_INFRA" 79252178e65914ad1acbc14790b1514f276bfea0 "$E_IMAGE" "$E_GPU" "$E_RUN_ID" > "$E_BATCH/launch.txt"
git -C "$E_SOURCE" diff HEAD --binary > "$E_BATCH/reviewed-worktree.patch"
set +e
docker run --rm --pull=never --init --user "$E_UID:$E_GID" \
  --gpus "device=$E_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
  --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
  --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E_SOURCE,dst=/workspace,readonly" \
  --mount type=bind,src=/home/isavand/paper/data,dst=/inputs/datasets,readonly \
  --mount type=bind,src=/home/isavand/paper/final_results,dst=/inputs/thesis,readonly \
  --mount "type=bind,src=$E_PROOF,dst=/inputs/compatibility,readonly" \
  --mount "type=bind,src=$E_BATCH/run,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout \
  "$E_IMAGE" --signal=TERM --kill-after=30s 900 \
  /usr/bin/python3 -I -B /workspace/run_g_mesh_diagnostic.py \
  --scientific-commit 79252178e65914ad1acbc14790b1514f276bfea0 \
  --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
  --checkpoint /inputs/thesis/model/bell_shape/model_best.pth \
  --checkpoint-sha256 d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc \
  --compatibility-report /inputs/compatibility/G-diagnostic.json \
  --compatibility-sha256 73e405b99a3041bb6bc7195e220bd128b82ea3ee54d5a4f5f4df14476b20435a \
  --resolution 128 --run-id "$E_RUN_ID" --execute --acknowledge-trusted-checkpoint \
  --confirm-isolated-runtime --image-id "$E_IMAGE" --infrastructure-commit "$E_INFRA" \
  > "$E_BATCH/terminal.log" 2>&1
E_EXIT=$?
set -e
printf '%s\n' "$E_EXIT" > "$E_BATCH/exit-code.txt"
cat "$E_BATCH/terminal.log"
test "$E_EXIT" -eq 0
sha256sum "$E_BATCH/run/G-mesh-diagnostic.json" "$E_BATCH/run/meshes/bell_shape-best-150000-r128.ply"
printf 'Review native diagnostic evidence: %s\n' "$E_BATCH"
```

Budget900 seconds plus30-second TERM/KILL grace, four CPUs,64GiB/no extra swap,
128PIDs,2GiB shm, one selected A10080GB PCIe. No GPU-memory cap is claimed.
Control resolution and query budget are fixed, not a full512 extraction. Existing
source/cache/JIT/runtime failures, OOM, invalid triangles, zero surfaces or PLY
precision collapse must be retained and reviewed, never repaired automatically.
For metadata-only validation replace execute/acknowledgements with --validate-only.

Success means a complete manifest, strict admitted weights and unchanged network,
eight original G field/SDF query chunks, finite nonempty nondegenerate raw/processed/
serialized geometry and an independently inspectable hash-pinned PLY. Inspect
normals/topology/shape before proposing the next same-scene integration milestone.
It does not establish canonical512 quality, producing-run attribution, training
reproducibility or research-grade G-to-A handoff authorization.

## CPU/static preparation results

Host Python3.9.12, CPU operations only: **258 tests,254 passed,4 skipped,0 failed**.
Ten new extraction tests cover source/checkpoint/report/resolution/trust gates,
immutable input/output isolation, collision/failure logging, rejection as research
handoff, original G field/query/mapping bodies with an analytic CPU SDF and
explicit marching-cubes stand-in, mesh validity/winding and export/normal order.
One new optional real CPU PyMCubes/Trimesh export/readback test is skipped because
those host packages are unavailable; the3 existing OpenCV/Open3D skips remain.
No packages were installed. CPU fixtures are not native SDF-model acceptance.

Commands: `python -B -m unittest discover -s tests/infrastructure -v` and
`python -B docker/paper/static_checks.py`. Static checks pass the existing
definition/mount policies,5 RUN shell parses and Python3.8 grammar for172 files.
The checker's existing build-readiness warning concerns the uncompleted checked-in
artifact inventory, not a new assessment of the working researcher image.
The exact tmux/Docker shell blocks parse without execution. Real host-path
--validate-only succeeds without allocation, model imports or pickle loading.
Native Python3.8/CUDA/SDF/PyMCubes/Trimesh output and normal/topology results remain
pending researcher execution/review; no existing success is inferred for this mesh.
