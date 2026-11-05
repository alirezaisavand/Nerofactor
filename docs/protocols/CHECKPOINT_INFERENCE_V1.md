# Proposed checkpoint-inference comparison protocol v1

Protocol ID: **`checkpoint-inference-validation-v1`**. Specification revision 1,
**proposed, not implemented or executed** in 1D.4A. Approval of this document does
not authorize GPU launches. Baseline readiness and evidence classifications are
in [BASELINE_PREPARATION.md](../BASELINE_PREPARATION.md).

## Question, boundaries and reference

The first question is whether isolated configuration/path/cache routing preserves
one existing **validation-view** inference result from the same admitted weights,
scientific implementation, effective settings and inputs. This is a routing/
inference regression comparison, not independent reproduction of training,
checkpoint selection, mesh extraction or a thesis score.

Proposed arms, each in a fresh single-use process/output directory:

- **Reference R:** a later reviewed thin adapter to the direct renderer using the
  admitted original legacy YAML and explicit input paths. Use the unchanged
  scientific definitions at the pinned source snapshot. It must independently
  record effective settings and pass the same explicit read-only cache and mesh.
- **Candidate C:** the later isolated inference adapter resolving the schema-1
  envelope with PathResolver/PreprocessingCache and the same admitted checkpoint.

Both use the same exact pinned source snapshot and native image for this first
regression. R must not simply call C's config adapter; compare separately resolved
settings and preserve a diff of both adapters. Sharing scientific definitions
means this is not an independent algorithmic implementation. A stronger historical
comparison requires an attributable producing snapshot and a separate reviewed
reference. Historical directory names or same-shaped weights do not supply that.

Do **not** launch either training entry point: existing initialize/resume paths
also construct optimizers and enter training. `run_native_smoke.py` explicitly
rejects checkpoints and is not this evaluator. R cannot run the historical
`run_training.py` unchanged: geometry has known split-unpacking/final-NVS
interfaces. Only the approved isolated three-list split adaptation is permitted
in the future direct-renderer reference. Record that compatibility shim and its
source hash; never describe it as unmodified legacy execution. Geometry final
NVS stays deferred. Do not repair unrelated renderer/evaluator defects if exposed.

## Admission manifest and immutable inputs

Before deserialization or GPU execution, freeze one manifest per candidate with:

| Field | Required identity / current candidate |
| --- | --- |
| Protocol / status | ID above, spec revision, review decision; missing fields block execution |
| Source | Full Git commit, clean status or full patch/hash, adapter/source hashes, exact command and exit status |
| Environment | Immutable image ID, image tag as annotation only, available image manifest digest, runtime package versions, Python, Torch/CUDA, driver/GPU, ABI, extension revisions and binary/JIT hashes; unknown build provenance stays explicit |
| Configuration | Raw schema-1 envelope and legacy YAML bytes and SHA-256; separate effective trainer/renderer/shader settings; only dataset/mesh/output paths may differ between arms |
| Checkpoint | Explicit read-only root/path, size, SHA-256, trust/ownership declaration, producing commit/config/algorithm when known, selected checkpoint policy; no latest/best auto-discovery or fallback |
| Stored state | After separate trust approval: record available `step`, `best_para`, key/shape/dtype architecture signature and provenance tags. Strict network-only load; no optimizer/scheduler/RNG restore or training continuation |
| Mesh (material) | Explicit root/path, SHA-256, size, producing run/frame/normalization history when known; no transform, crop, simplification or save |
| Dataset | Root alias, family, database string, source filename/order manifest and content hashes for every input actually read, including camera metadata, masks, normalization cloud and reused legacy products |
| Splits | Exact typed and ordered database IDs, train/validation/NVS lists and ID-to-source-filename mapping; hash canonical split JSON, record helper source hash, seed and calls/order |
| Preprocessing | Versioned cache identity, implementation/package hashes/versions, effective parameters, source-product vs generated-product use, cached camera/image hashes and decoding backend |
| Execution | Fresh comparison/run ID, arm, timestamps, resource budget, RNG policy/state hashes, native report phase/exit/failure and paths to raw outputs |

A content hash binds bytes, not trust or producing history. Torch 1.8 checkpoint
loading uses pickle; never deserialize an untrusted checkpoint merely to inspect
its keys. In this milestone no checkpoint is hashed or deserialized. Later
trusted loading must be separately authorized, read only from the explicit input,
check stage/algorithm and exact key/shape/dtype compatibility before state load,
and reject incompatible/foreign/ambiguous architecture without key stripping,
partial loads, dtype conversions or silently ignored fields. A reviewed
hash-bound origin declaration is required for untagged material artifacts, as in
[MATERIAL_TRAINING.md](../MATERIAL_TRAINING.md). Same shapes do not prove branch
compatibility; bell geometry needs an equivalent origin review too.

Unknown producing history may permit a reviewer-authorized **current-code routing
regression** with explicitly limited provenance, but must block claims of thesis
checkpoint reproduction. Missing or ambiguous architecture blocks any inference.
Do not infer the stored step from a mesh filename, log row, tensor shapes or
configured `total_step`. For this protocol feed the exact admitted stored step
into inference in both arms and record its zero/one-based uncertainty. Missing
step or ambiguous effective configuration requires review before proceeding;
never substitute step 0 or the configuration endpoint. Preserve any legacy
step-dependent behavior rather than correcting presumed off-by-one semantics.

## Candidate inputs and split freeze

**Bell geometry first:** select `model/bell_shape/model_best.pth` under thesis;
alternate `model.pth` requires a separately named manifest. Raw YAML
`configs/shape/syn/bell.yaml`, envelope `configs/experiments/geometry.example.yaml`,
`GlossySynthetic/bell`, database `syn/bell`. Source-derived proposed validation ID
is string `"58"` at index 0; proposed NVS strings are `"11", "105", "114", "17",
"76"`. Full initial/train/validation/NVS lists are in the
[metadata snapshot](../evidence/1d4a-metadata.json). Validate ordered equality using
the unchanged helper under Python 3.8 and the actual admitted database before
rendering. The proposal was derived with an isolated Python 3.9 Random instance
from camera filename metadata; it neither executes the loader nor proves an old
training split. The helper itself seeds global Python RNG with 6033 and shuffles
once per call; preserve and record its actual call order/effects in each arm.

**Bear material next, currently blocked:** YAML `configs/material/real/bear.yaml`,
envelope `configs/experiments/material.example.yaml`, database
`real/bear/raw_1024`, mesh `thesis:meshes-pretrained/real/bear.ply`. Checkpoint
path/hash **missing**; do not use fresh smoke weights or geometry weights instead.
Exact registered camera IDs/order, full split and validation ID are **unresolved**
because `cache.pkl` was not loaded. Later trusted metadata recovery must establish
ordered cache IDs and mapping to COLMAP/image filenames. Do not assume IDs are
contiguous or equal to filenames; do not select an appealing view after seeing
scores. The helper's existing single validation ID becomes the first view.
The PBRNeRF report's `[40,3,69,61,24]` is historical NVS evidence only, not an
admitted current split. Preserve original ID types in the later manifest.

Use `split_type='validation'` only: five NVS views, one validation view, all
remaining views training, in helper order. `split_type='test'` reads a separate
pickle-based split and is **outside v1**. Do not reinterpret its lists or substitute
its IDs. NVS cameras and material NVS metric outputs are not exercised by this
minimal protocol; a later NVS extension needs its own reviewed version/identity.

## Preprocessing, model and RNG preservation

Both arms use the same admitted input root and cache semantics, with separate
writable caches under their owned outputs. Existing source-side products remain
read-only inputs. For bear this includes `cache.pkl` and existing resized JPGs;
do not silently force regeneration or consume the extra PNG files as new views.
Record every reused product and its hash. A test of rebuilding caches from raw
COLMAP/images is a separate experiment, not a fallback after mismatches.

Preserve the actual synthetic loader: camera float32 casts, scale factor 1,
source RGB first three channels, existing mask reads, depth conversion
`uint16 / 65535 * 15`, depth threshold `< 14.5`, original pose/intrinsics.
Geometry validation defaults are `test_downsample_ratio=True`,
`downsample_ratio=0.25`, `test_ray_num=512`; preserve its Gaussian/image resize,
intrinsic update, nearest-neighbor depth/mask resize and all render sample counts.
Do not render full resolution or shrink rays as an infrastructure convenience.

Preserve bear `raw_1024` behavior: scale from the first registered image to
maximum length 1024 with existing integer dimension rules, image resize helper
and intrinsic row scaling; camera normalization from object point cloud and
hard-coded forward/up vectors. No additional crop, background compositing,
masking, color-space conversion or mesh alignment. Material validation uses its
existing `test_ray_num` default **1024** and shader counts **512 diffuse / 1024
specular** from the current YAML. Record effective values at construction; no
shader-like top-level field is forwarded into `shader_cfg`. Top-level `0.01`
energy/specular declarations and effective shader `0.05` defaults remain distinct.
Record `is_real` and renderer/shader lobe scopes separately; do not reconcile them.

Use the existing constructor seed policy (`random_seed` default 6033), model
construction, split calls and configured sampling implementations, same CUDA
device class, float32/default tensor behavior and inference chunk order. Model
construction still consumes RNG even when weights are subsequently loaded.
Freeze this order for both adapters; record Python/NumPy/Torch CPU/CUDA RNG state
hashes just before inference. Do not restore a hypothetical historical RNG state,
choose a new seed, introduce deterministic kernels/TF32 toggles or change backend
flags to obtain agreement. Record actual backend settings, without overriding
legacy defaults. A mismatch in states/order must be explained before interpreting
numerical differences. No DataLoader shuffling or training batches are needed.

Select normal model evaluation mode and the existing validation dispatch:
`model({'index': 0, 'eval': True, 'step': admitted_step})` with the existing tensor/
device handling. It invokes geometry `test_step(index, step)` or material
`test_step(index, is_nvs=False)`. Preserve the existing validation no-grad boundary
and geometry's internal SDF gradient enabling; do not wrap the model in
`inference_mode` or disable required derivatives. No train_step, loss backward,
optimizer, scheduler, training loop, checkpoint write, mesh export or final NVS.
If an original validation path fails, record it and stop; no scientific repair
is authorized by this protocol.

## Outputs and metric definitions

One complete existing validation view per admitted stage/arm. Preserve the native
primary float RGB (`ray_rgb` geometry, `rgb_pr` material), corresponding GT, masks,
shape/dtype/finiteness, image dimensions and relevant depth/normal/auxiliary output
metadata. Store primary/GT arrays losslessly, e.g. non-pickled NPY, plus separate
lossless uint8 PNGs and the unchanged validation montage JPG if generated. Hash
all outputs. Never infer numerical equality from compressed montage JPGs.

Use existing `network/metrics.py` definitions, not masked standalone evaluators:

- RGB conversion: multiply float RGB by 255, clip to `[0,255]`, cast to `uint8`
  (**truncate**, not round, gamma-convert or rescale by data extrema).
- PSNR: reshape RGB into N-by-3, cast to float32, mean squared difference per
  channel then mean channels; `10*log10(255**2/MSE)` on the entire converted image.
- SSIM: scikit-image `structural_similarity`, `win_size=11`, `channel_axis=2`,
  `data_range=255`; retain the installed historical version and other defaults.
- No extra foreground-only reduction or mask multiplication. Material
  `test_step` already fills non-hit pixels with zero in prediction and GT; retain
  that behavior and record hit coverage rather than reporting foreground PSNR.
- The validation set has one view, so the mean is that per-view value. Keep
  per-view full precision, do not average RGB before PSNR or round per-view
  metrics. Exact-match infinite PSNR is recorded with an explicit JSON status,
  not silently discarded to obtain a finite mean.

RGB PSNR/SSIM are appearance metrics, **not geometry reconstruction error**. No
Chamfer distance, F-score, LPIPS, reference mesh alignment or point resampling is
part of v1. `eval_synthetic_shape.py` has unresolved helper interfaces and generic
cloud provenance is unknown. PBRNeRF masked JSON, NeRO material NVS logs,
README tables and geometry validation scores retain separate protocol labels.
No cross-method score comparison or expected historical score is implied.

## Proposed numerical acceptance (must be reviewed before runs)

These are **tentative regression thresholds**, not observed error bounds or
historical quality targets. No pass/fail inference result exists yet.

| Quantity | Proposed requirement |
| --- | --- |
| Checkpoint, mesh, used input/config/source/environment identities | Exact content identities for shared inputs/scientific code; allow only recorded routing-path differences |
| Ordered split IDs, image dimensions, effective scientific settings, chunk order | Exact equality; any mismatch blocks metric comparison |
| Float RGB C vs R | All finite; elementwise `abs(C-R) <= 1e-4 + 1e-4*abs(R)` across the full view |
| Converted uint8 RGB C vs R | Maximum absolute channel difference <= 1 (subtract using signed/wide type); report mismatch count and MAE, never demand compressed JPG byte equality |
| Full-image PSNR C vs R | Absolute difference <= 0.01 dB; both exact-match infinities are a separately marked exact case; only one infinite is a mismatch |
| Full-image SSIM C vs R | Absolute difference <= 1e-4 |

Record max/mean absolute and relative float error, allclose violation count,
uint8 disagreement count, per-arm metrics and deltas, even on failure. Auxiliary
outputs are diagnostic in v1; no unreviewed depth/normal tolerance becomes a
scientific acceptance criterion.

Proposed minimum execution sequence per admitted stage: one fresh R process,
one fresh repeat R process to characterize repeatability, then one fresh C
process, all with the same frozen seed/order/inputs. Apply the same thresholds
to R-repeat; if native repeatability exceeds them, stop and report the measured
spread. Do not widen tolerances, retry seeds or average away a mismatch after
seeing C. New bounds or different GPU/environment comparisons require prior
review and a protocol revision. There is no claim of bitwise determinism or
training/resume equivalence.

## Isolation, expected result bundle and failures

Use an already approved immutable candidate image; no image modification/build.
Read-only code at `/workspace`, datasets `/inputs/datasets`, historical artifacts
`/inputs/thesis`, optional explicitly declared prior-run `/inputs/geometry`.
Only a fresh arm-specific directory is mounted writable at `/outputs` for each
process, with non-root UID/GID, one explicitly selected GPU, read-only rootfs,
no network, no Docker socket and no mounts of unrelated experiments. Reuse the
approved [smoke mount/cache safeguards](../NATIVE_GPU_SMOKE.md). Redirect all
cache/HOME/JIT/temp/log writes; reject collisions and output escapes. No input
write probes or permission changes. Checkpoint input mounts never become output
or resume locations.

Proposed review budget: one selected view per arm, 900 seconds per process plus
30-second termination grace, 64 GiB host-memory limit, four CPUs, sequential
processes on one selected GPU (three processes per admitted stage). Eager scene
initialization may exceed that budget; report failure without reducing scientific
samples/resolution. No automatic extension or full NVS/training fallback. Final
budget/GPU selection and exact command require separate execution approval.

Future bundle (names are a proposed contract, not currently generated):

- `comparison.json`: protocol/status/admission decisions, arm linkage, thresholds,
  per-view metrics/deltas, repeatability, pass/fail and reason.
- Per-arm manifest plus original/effective configs, `inputs.json`, `splits.json`,
  environment/source/command identities, RNG state hashes and report timestamps.
- `arrays/{prediction,ground_truth}.npy`, masks/hit coverage where available;
  `renders/{prediction,ground_truth}.png`, original validation montage if emitted.
- `metrics.json`, `logs/console.log`, process exit status and phase/resource summary;
  output hashes and an unchanged-input identity check. No checkpoint files.

Missing input/trust/config/split rejects before model work; incompatible state
rejects before loading network weights. Native exception, timeout/OOM, nonfinite
output, metric error and identity/tolerance mismatch all remain failed or blocked
records with partial logs. Missing/partial reports never count as success. Keep
failed output directories; never resume them or overwrite an earlier arm.
Historical results are immutable references and are not replaced by this bundle.
