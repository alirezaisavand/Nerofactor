# Milestone 3A-2 — bounded G SDF/mesh/A-normal audit v1

Protocol `bell-geometry-sdf-mesh-audit-v1`, prepared from
`3fe197bbdc37957c187c8589fffdb123aaac692d` on
`feat/geometry-sdf-mesh-audit-v1`. Implementation and NEW analytical CPU tests only;
no native execution by Codex. This does not repeat3A-1, lifecycle, extraction,
training, rendering or a prior smoke. All original evidence/source bytes remain
unchanged; no research handoff is admitted.

## Admission, source and runtime

Pinned G `79252178e65914ad1acbc14790b1514f276bfea0` and A
`e1e859779301faced46d6476691e20b546c27ae0` are independently verified by the
committed source manifest. Explicit inputs JSON/hash selects that manifest,
accepted extraction JSON, PLY, ray report, all three NPZs, historical checkpoint
and accepted3A-1 report. Every path/hash is explicit; no latest/best discovery.
Identity helpers from3A-1 are reused, **never its full audit/check_view**.

The real accepted3A-1 report at
`/home/isavand/paper/evidence/geometry-scientific-v1.0w4FOvmw/result/audit.json`
was read as JSON and independently hashed during preparation:
`9067af609b5289440ce07d9ca47370f830952c9dfa64bd72d24aa12e80f5c327`.
It reports pass/blockers[]; its implementation hashes match the accepted checkout.
An explicit SHA remains required at execution. No hash is inferred from its name.
The old report was produced from another worktree: paths in it are preserved but
not followed in the container. Current explicit mounted files must have identical
content hashes, source closures and source-camera bindings. A relocation of the
same bytes is not a scientific input substitution.

Fixed mesh/extraction/checkpoint/ray-report hashes respectively:
`eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5`,
`caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f`,
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`,
`88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510`.
Original3A-1 NPZ identities/800² layouts are retained. All numerical NPZ reads use
allow_pickle=False with existing header/shape/resource checks. No original images,
GT masks or camera pickles are loaded. Source dataset classes are never instantiated.
Saved IDs14/58/11 retain their current A train/validation/NVS labels. All three
belong to the pinned G unshuffled training split; these labels do not establish
historical held-out evaluation. Saved normals are actual BVH/final-normal arrays,
not the renderer's encoded normal-visualization channels. The A operation remains
the original normalized `-raw_normal`; this audit never flips either stored array.

Only acknowledged isolated execution imports Torch/scientific modules. One original
G NeROShapeRenderer(cfg,training=False) registers its full original architecture;
strict original full-network loading replaces constructor values before any SDF
query. The model is then eval().cuda(); there is no later reinitialization. The
original SDFNetwork/sdf function is used, not a reimplementation or SDF-only proxy.
Trusted Torch1.8 CPU-mapped full pickle deserialization still includes the optimizer
payload; **no optimizer is constructed or restored**. Metadata/catalog/finite
checks, all137 tensors and loaded weight/buffer hashes must match accepted G
extraction state. Before/after model state must be byte-identical, with no parameter
grad accumulation. Model/network/channels are never modified to improve agreement.

Reuse the reviewed immutable image, non-root/read-only/no-network/fork guards,
RunStore and owned HOME/temp/JIT/log cleanup. The storage adapter's mandatory
`datasets` label is code storage here, not a dataset selection; no dataset mount
or API is used. Every selected input file itself is checked read-only. A scoped
observer blocks original renderer/data/training/serialization/extraction helpers,
Adam construction/step/restoration and original second-order SDFNetwork.gradient.
It counts actual original sdf/autograd calls, rejects create_graph=True, and
restores profiling on failure. No Docker or filesystem access restriction bypass.

## Frozen sample and evaluation budget

- At most1920 face centroids (below the authorized2048 cap): face-area rank
  quartiles, evenly spaced in radius order within each quartile; explicitly retain
  minimum-area and maximum-radius faces, replacing final slots when necessary.
- At most256 evenly spaced vertex indices, retaining maximum-radius vertex.
- Every final back-facing saved hit from camera14/58/11: expected0/2/1. These
  counts must match accepted3A-1; unexpected counts stop, never silently drop hits.
- At most32 deterministic positive-view-cosine controls per camera, evenly spaced
  in row-major hit order. All selected ray/face/vertex indices are recorded.
- Base points≤2275; normalized-gradient offsets±h/4,±h/2 on eligible gradients;
  original face-normal offsets±h/4,±h/2 for faces; original A final-normal offsets
  ±h/4,±h/2 for hits; toward/away-camera offsets±h/2 for hits.
- Worst case:2275+9100+7680+396+198=**19649 SDF point evaluations**, cap20000.
  Low-gradient exclusions reduce this count. Batch size128 fixed; no CLI overrides.

h=2/127 is the original r128 extraction grid spacing. CPU geometric calculations
are float64; actual queried points are explicit float32 tensors on the selected
GPU. Face-centroid rounding and actual float32 query coordinates are recorded.
Base values and gradients use original sdf(x) and torch.autograd.grad with
create_graph=False/retain_graph=False. Offset queries use no_grad, no second
derivatives. Finite raw SDF/gradients are mandatory; gradient length is measured,
**never presumed one**. No seed changes, camera regeneration or neural rendering.
Small batches may use different CUDA kernels than the original dense extraction;
record actual TF32/runtime flags, without changing them or asserting bitwise grid
query equivalence.

Raw network SDF is kept even outside the unit sphere. Separately report the
masked extraction field(+1 there), using **original torch.norm(float32)>=1**
mask precision/operator for each query, not a float64 sphere approximation.
Finite network derivatives outside that sphere are not derivatives of its forced
masked extraction field. Report all radii, near-sphere counts(within h), face
areas, tiny-triangle counts, exclusions and sign changes at both offset scales.
No boundary or tiny-triangle population is hidden by component filtering/repair.

## Measurements, interpretation and predeclared criteria

Hard execution gates: exact identities/source/prior admission, finite model/SDF/
gradients, unchanged model, exact anomaly counts, fixed sample/query budgets,
float32/batch contract and complete cleanup. A gate failure produces failed action
evidence; it does not trigger a repair, new threshold or automatic retry.

Scientific criteria below are **local sampled diagnostic conventions**, not
publication-quality/canonical512 accuracy or research-admission thresholds:

- Near-zero gradients: norm≤1e-6. Retain raw values; exclude directional/proxy
  interpretations explicitly. |SDF|/norm is a first-order proxy, not exact distance.
- Eligible interior sampled zero-level criterion: proxy≤h/2 and sign crossing
  at±h/2 along the normalized gradient. Both probes and the base must be inside
  the original extraction sphere; otherwise exclude/mark unresolved. Robust
  sign magnitude must exceed1e-7. Report raw/proxy/h quantiles by sample kind.
- Face orientation: calculate original cross-product normals independently, signed
  gradient-vs-face and gradient-vs-negated-face cosines/angles. Uniform sampled
  relation requires all eligible cosines≥cos30° or≤−cos30°; explicitly name which
  signed relation is observed. Mixed relations remain unresolved, not repaired.
- A normals: retain signed gradient-vs-final and gradient-vs-matched-original-face
  comparisons at the actual saved hit. Do not select the favorable sign. Where
  CPU geometry establishes a unique first hit from a camera outside mesh AABB,
  ray-side probes straddle robust opposite SDF signs, no second mesh intersection
  is crossed, and neither probe is sphere-masked, the **observed camera-side sign**
  defines the local exterior SDF polarity. Report that sign explicitly. Only then
  assess A's final normal against that exterior-signed gradient(cos≥cos30° on
  eligible controls). This is a measured polarity interpretation, not a normal
  flip or an assumption that SDF must be positive outside. Exclusions remain in
  per-ray records. Anomalies have their own classifications.

For every selected ray, a bounded CPU two-sided float64 Moller-Trumbore test
checks the unchanged original triangles. Relative parallel cutoff1e-12×edge-length
product preserves tiny triangles; barycentric tolerance1e-9; positive t>1e-8.
Record sorted first8 positive candidates plus closest saved-depth candidate,
positions, cross normals, areas, barycentrics and candidate counts. Depth match
bound2e-4, first-hit tie5e-5, near-edge barycentric≤1e-5 and tiny area≤1e-10 are
fixed. Near-edge/tiny-face/tied cases are **ambiguous**, not automatically blamed
on floating point. Later-than-first hits, mismatched face normals or consistent
first-hit geometry can be **supported_by_evidence observations**; unmatched cases
remain unresolved. No exact cause, self-intersection absence or global outward
orientation is guessed from these measurements or signed volume.

The three anomalies retain exact source ID, row/column, flattened ray index,
saved origin/direction/position/depth, raw/final normals and viewing cosine, CPU
candidates and SDF/gradient comparisons. A supported observation is not a proven
failure mechanism. Classification and causal uncertainty are separate fields.

Silhouette characterization uses only existing hit/mask arrays, no new observation.
Reference boundary = foreground pixels adjacent to 4-neighbor background; outside
image is background. Inclusive Chebyshev distance to those pixel centers defines
1/2/4-pixel bands, formed by deterministic 8-neighbor dilation. Count FP/FN in
each cumulative band and deeper than4. Empty masks have no boundary; no artificial
alignment, dilation of the reference for scoring, or regenerated GT is introduced.
The bands describe localization of disagreements, not an accuracy threshold.

## Outputs and statuses

`sdf-mesh-audit.json`: exact artifact/source/checkpoint/runtime identities, original/
effective model configuration, fixed indices/settings, actual query accounting,
per-group distributions, per-ray/anomaly observations, output hashes, timestamps,
errors and cleanup. `measurements.npz`: bounded numerical base/probe coordinates,
raw SDF, gradients/lengths/proxies, actual extraction masks/masked-field values,
face IDs/normals/areas and probe-to-base/direction/offset mappings. Direction codes
0=gradient,1=original face,2=A final,3=toward camera. Numeric exclusions may be NaN
in this numeric-only NPZ; JSON uses null/exclusion counts, never fabricated values.
Owned caches/console log and allocation marker follow existing conventions.

Separate pass/fail/unresolved scientific statuses are evidence_identity,
sdf_mesh_zero_level_consistency, sdf_gradient_quality,
face_winding_vs_sdf_orientation, A_shading_normal_consistency,
three_anomalous_hits, silhouette_disagreement_characterization,
historical_checkpoint_provenance, research_geometry_admission. Completed action
status does not mean all scientific criteria pass. **The last two always remain
unresolved**, with research_handoff_admitted=false. No full-convergence, producing
branch/split, physical units, global SDF outwardness, canonical512 quality or
ordinary research G→A admission follows from this bounded diagnostic.

## Researcher-side one-shot command (not executed by Codex)

Use the reviewed new feature checkout with all implementation files present.
Start a fresh tmux session if desired, then run this single bounded action.
The3A-1 hash below was independently observed from the actual file, not invented.

Run inside an existing tmux session on marineris (select an available A100 explicitly
with `SDFA_GPU`). This creates only fresh selection/output/log files. The command
uses original accepted artifact paths, no implicit discovery, no actual dataset
mount and no writable historical artifacts. The source mount must contain this
reviewed implementation; a dirty checkout is recorded, never concealed by its
HEAD identity. No commit is created by these commands.

```bash
set -euo pipefail
SDFA_SOURCE=/home/isavand/paper/Nerofactor-paper
SDFA_GPU=0
SDFA_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
SDFA_G=/home/isavand/paper/evidence/bell-G-best-mesh.djdTf3F5/run
SDFA_A=/home/isavand/paper/evidence/bell-G-to-A.TlwUeOsY/run
SDFA_COORD=/home/isavand/paper/evidence/geometry-scientific-v1.0w4FOvmw/result
SDFA_CHECKPOINT=/home/isavand/paper/final_results/model/bell_shape
SDFA_COMMIT=$(git -C "$SDFA_SOURCE" rev-parse HEAD)
test "$(git -C "$SDFA_SOURCE" branch --show-current)" = feat/geometry-sdf-mesh-audit-v1
test "$(git -C "$SDFA_SOURCE" merge-base HEAD 3fe197bbdc37957c187c8589fffdb123aaac692d)" = 3fe197bbdc37957c187c8589fffdb123aaac692d
test "$(id -u)" != 0
SDFA_RUN_ID="bell-sdf-$(cat /proc/sys/kernel/random/uuid)"
SDFA_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-sdf-audit.XXXXXXXX)
mkdir "$SDFA_BATCH/selection" "$SDFA_BATCH/run"
git -C "$SDFA_SOURCE" status --porcelain=v1 > "$SDFA_BATCH/checkout-status.txt"
git -C "$SDFA_SOURCE" diff --binary HEAD -- > "$SDFA_BATCH/checkout.diff"
cat > "$SDFA_BATCH/selection/inputs.json" <<'JSON'
{
  "schema_version": 1,
  "protocol": "bell-geometry-sdf-mesh-audit-v1",
  "artifacts": {
    "source_manifest": {"path": "/workspace/scientific_sources/manifest-v1.json", "sha256": "39b9baecce2aab6fc62ec57bd6f45e453d8e9d68215f6af3f25b8ed118c5a3f6"},
    "extraction": {"path": "/inputs/g-mesh/G-mesh-diagnostic.json", "sha256": "caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f"},
    "mesh": {"path": "/inputs/g-mesh/meshes/bell_shape-best-150000-r128.ply", "sha256": "eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5"},
    "rays": {"path": "/inputs/rays/bell-material-diagnostic.json", "sha256": "88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510"},
    "checkpoint": {"path": "/inputs/checkpoint/model_best.pth", "sha256": "d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc"},
    "coordinate_audit": {"path": "/inputs/coordinate/audit.json", "sha256": "9067af609b5289440ce07d9ca47370f830952c9dfa64bd72d24aa12e80f5c327"}
  },
  "views": [
    {"role": "train", "source_camera_id": "14", "path": "/inputs/rays/evaluations/train-14.npz", "sha256": "3e786f597a11055962cbe1a8073c6d1c8ebc320d48cf26c8c9ec3961828a1449"},
    {"role": "validation", "source_camera_id": "58", "path": "/inputs/rays/evaluations/validation-58.npz", "sha256": "b588f8ef8d0ee61e07e10344c2350b287054a69e6a12994ea2eb89de69f3e5dd"},
    {"role": "nvs", "source_camera_id": "11", "path": "/inputs/rays/evaluations/nvs-11.npz", "sha256": "57eff4adad35ec1eaf9ba7a8b70ae333f26aa281792178074ac1cfb80353407c"}
  ]
}
JSON
SDFA_SELECTION_SHA=$(sha256sum "$SDFA_BATCH/selection/inputs.json")
SDFA_SELECTION_SHA=${SDFA_SELECTION_SHA%% *}
printf '%s\n' "$SDFA_RUN_ID" > "$SDFA_BATCH/run-id.txt"
printf '%s\n' "$SDFA_COMMIT" > "$SDFA_BATCH/checkout-head.txt"
printf '%s\n' "$SDFA_IMAGE" > "$SDFA_BATCH/image-id.txt"
set +e
timeout --signal=TERM --kill-after=30s 960s docker run --rm --init \
  --name "$SDFA_RUN_ID" --network none --read-only \
  --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges:true \
  --gpus "device=$SDFA_GPU" --memory 16g --memory-swap 16g --cpus 4 --pids-limit 128 \
  --shm-size 256m --ulimit nofile=1024:1024 --ulimit core=0 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$SDFA_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$SDFA_G,dst=/inputs/g-mesh,readonly" \
  --mount "type=bind,src=$SDFA_A,dst=/inputs/rays,readonly" \
  --mount "type=bind,src=$SDFA_COORD,dst=/inputs/coordinate,readonly" \
  --mount "type=bind,src=$SDFA_CHECKPOINT,dst=/inputs/checkpoint,readonly" \
  --mount "type=bind,src=$SDFA_BATCH/selection,dst=/inputs/selection,readonly" \
  --mount "type=bind,src=$SDFA_BATCH/run,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$SDFA_IMAGE" \
  --signal=TERM --kill-after=20s 900s python -B /workspace/run_geometry_sdf_mesh_audit.py \
  --inputs /inputs/selection/inputs.json --inputs-sha256 "$SDFA_SELECTION_SHA" \
  --output-root /outputs --run-id "$SDFA_RUN_ID" \
  --geometry-commit 79252178e65914ad1acbc14790b1514f276bfea0 \
  --material-commit e1e859779301faced46d6476691e20b546c27ae0 \
  --image-id "$SDFA_IMAGE" --infrastructure-commit "$SDFA_COMMIT" \
  --acknowledge-trusted-checkpoint --confirm-isolated-runtime --execute \
  2>&1 | tee "$SDFA_BATCH/docker-console.log"
SDFA_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$SDFA_RC" > "$SDFA_BATCH/exit-code.txt"
printf 'Evidence: %s; exit: %s\n' "$SDFA_BATCH" "$SDFA_RC"
test "$SDFA_RC" -eq 0
sha256sum "$SDFA_BATCH/run/sdf-mesh-audit.json" "$SDFA_BATCH/run/measurements.npz"
```

The inner900s timeout bounds the action; outer960s bounds Docker client waiting.
Hard termination/OOM can leave incomplete evidence without final JSON/cleanup;
that is not success. Inspect the named container read-only if interrupted; never
reuse its output, automatically retry or alter thresholds. No image build or
container execution was performed during implementation. If access is denied,
report it and use the regular authorized terminal; no permissions bypass.

Expected successful *action* output: `Bounded SDF audit complete`, action
status=complete, evidence_identity=pass, strict full weights unchanged, matching
query observer counts≤20000, no optimizer/dataset/render action, unchanged
measurement RNG and complete owned cleanup. Review each independent scientific
status, anomalies and exclusions; a completed action may report scientific failure
or ambiguity. Never reinterpret this as provenance/research admission.

Before that single native action, metadata-only admission is available by replacing
`--execute` with `--validate-only` in the same command and using a **different fresh**
batch/output. It opens/hash-checks opaque checkpoint/NPZ bytes but deserializes
neither and constructs no model. This is optional, not a second scientific run.

## Implementation validation and limits

Only the new analytical test module and essential static checks are authorized:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/infrastructure -p test_geometry_sdf_mesh_audit.py -v
PYTHONDONTWRITEBYTECODE=1 python docker/paper/static_checks.py
```

The tests use disposable fixtures, analytical plane/sphere functions on CPU and
metadata-only pinned-source verification. Reused3A-1 fixture builders do not run
any3A-1 audit/test. No original numerical NPZs/checkpoints were evaluated here.
Native Python3.8/Torch1.8/CUDA runtime behavior, actual SDF alignment, anomaly
mechanisms and masked-sphere boundary quality require separately reviewed native
evidence. A finite/local consistent SDF is not independently measured3D accuracy.

Observed implementation checks: **25 new tests passed, 0 failed, 0 skipped**
(CPython3.9.12, Torch2.4.0+cu121 used strictly on CPU, NumPy1.21.5).
Analytical tests do not establish Torch1.8 native equivalence. Static checks passed:
Python3.8 grammar for188 source/definition files, Docker/Compose definition policy
and five Docker RUN shell parses; both documentation Bash blocks parse without
execution. Frozen G/A source closures and accepted3A-1 implementation remain
byte-identical to the accepted base. No full or historical test suite was run.
The existing static definition validator still reports unresolved artifact/build
pins in the generic reconstruction definition; it does not inspect or invalidate
the separately accepted immutable working image. No build readiness, actual image
identity inspection, native SDF result or new scientific success is claimed here.
