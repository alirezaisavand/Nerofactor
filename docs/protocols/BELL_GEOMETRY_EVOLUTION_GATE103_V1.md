# Gate 10.3 — Frozen geometry evolution and opacity diagnosis

Status: prepared for separate exact commit/freeze/output authorization. No checkpoint
deserialization, additional extraction, native rendering or training is authorized
by preparation. The original G source and Gate 1–10.2 evidence remain unchanged.
This study never admits a new checkpoint or creates a G→A handoff.

## Four authenticated checkpoints, one publication endpoint

Use only the following files from the completed Gate 10 v4 training history.
Hashes are checked against its producing training receipt; payload schema/step and
tensor catalog can be checked only after explicitly authorized CPU deserialization.

| Stored step | File | SHA-256 |
| --- | --- | --- |
| 15,000 | model-step15000.pth | 8022a6caafd4a70e8b1c27c2b5eec6658f6c4c5b816db14c14aab8f754691114 |
| 30,000 | model-step30000.pth | 04a103499ebc5c23b701615749bb60bc403c1b8ab9cc075d8ac50be8cec62b59 |
| 50,000 | model-step50000.pth | 50ce75c0477946115010fc01fb27649c5a2978056e28ab08f2b25ab34c512ff1 |
| 300,000 | model.pth | 27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd |

Exactly one CPU load of each file is allowed, four total. Never restore optimizer
state. Require the original strict G schema, expected stored step, network tensor
catalog and finite tensors; record actual intermediate tensor fingerprints after
loading rather than inventing them during preparation. Verify weights/buffers
unchanged and parameter gradients absent at the end of every checkpoint audit.

**300k remains the fixed original publication endpoint**, irrespective of earlier
held-view scores. Reuse its accepted Gate 10.2 candidate, six-camera rasters,
topology, SDF witnesses and quality decisions byte-for-byte. Its new checkpoint
load serves only the fitting-view neural-opacity diagnostic, not another extraction
or repetition of the accepted six-camera audit.

## Domain-aware extraction and geometry metrics

For 15k/30k/50k, call pinned G's original `extract_fields` on [-1,1]³ at 512³,
64³ tiles and original CUDA float32 node construction. Capture each tile's raw
SDF before the function applies its historical sentinel; compare the generated
domain mask with the accepted mask and check the returned masked/raw relation.
Discard the masked field. Marching cubes runs exactly once on each captured raw
grid, level zero, through PyMCubes 0.1.4. No artificial positive exterior values
enter marching cubes. Preserve all three raw grids and unfiltered mesh arrays.

Reuse Gate 10.2's all-eight-corners/all-consistent-generators domain policy,
exact float32 equality welding, exact-zero-area-only removal, original winding,
component preservation and all mapping/topology outputs. No oracle selection,
small-area threshold, hole filling or smoothing. Open3D/PLY parity and the
corrected native BVH contracts remain mandatory. Empty/invalid extraction or
exhausted work stops the batch without an alternative policy or retry.

Run the original six-camera geometry audit at the earlier three checkpoints
with the same calibrated 128² rays, signed normals, 384 independent FP64 ray
witnesses per checkpoint and original <=512-point SDF consistency witnesses.
Retain every foreground/support failure. The unchanged per-view criteria remain
IoU >=0.90 and valid oracle correspondence >=0.80, distance <=0.02 and signed
normal difference <=30°. Low geometry-quality scores and failed SDF *quality*
decisions are diagnostic outcomes and do not stop the remaining checkpoints.
Numerical, identity, invalid-native-normal or resource failures do stop execution.

Report precision and recall separately, all FP/FN/shared-miss counts, depth and
signed-normal distributions/tails, correspondence exclusions, grazing cohorts,
normalization/closest-hit checks, components, open boundaries and domain support.
Do not silently interpret high foreground correspondence as good silhouettes.

From saved grids, report negative-node fractions inside/outside the accepted
domain, the fixed original lower cube band y<−0.65, and raw-negative edges
crossing the domain boundary. The sign margin remains 1e-7. These are grid/cube
statistics, not valid exterior geometry or surface-area estimates. Reuse the
accepted 300k raw grid for these same file-only statistics without another model
query. Join geometry evolution with the original weighted-loss and validation
trajectories; rounded scalar zeros do not prove zero loss or gradient.

The checkpoint label is completed updates; rendering uses original loop index
s=step−1. Thus 15k precedes trainable inverse variance at s=15000. Occlusion and
opacity scheduling start at s=20000, mask weight reaches 1 at s=30000, and alpha
annealing/occlusion/opacity reach their final phases at s=50000. The saved 50k
checkpoint is just before the first s=50000 update, not a substituted 50001
checkpoint. Curvature's effective weight remains .001 despite nominal ramp
fields. All exact schedules and LR remain the
[accepted original definitions](BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.md).
Temporal association cannot isolate a coefficient's causal effect.

## Fitting-view inventory and neural accumulation

Trace all **122 original fitting cameras**, 128×128, for all four meshes.
Use their existing admitted camera-ray archives and oracle hit/depth/normal
arrays, verify calibration, and recompute candidate hits only. Keep the original
validation camera 58 and held views 11/105/114/17/76 outside fitting selection.
Four independent fitting ray witnesses at each of 16 prospectively selected
fitting cameras give 64 extra CPU checks per checkpoint. Full fitting mesh
silhouettes are evaluated, not a favorable camera subset.

For the more expensive neural diagnostic select 16 fitting cameras by max-min
world camera-direction coverage: first original training ID, then farthest
minimum angular separation, with numeric camera-ID ties. The fixed result is
14,10,29,12,127,81,66,83,30,90,7,93,5,0,113,99. No material response or recovered
geometry score enters this selection.

Each camera has four 64-index strata: whole foreground, whole background,
four-pixel inner silhouette band and four-pixel outer band. Bands are Chebyshev
bands from admitted masks. Select evenly spaced C-order indices in each stratum,
take a sorted unique union, preserve stratum membership and report absent strata
without replacement. All selected real strata have 64 entries. Deduplication
leaves **4,055 unique rays per checkpoint per draw**. This panel is not a
probability sample; per-stratum results and unweighted panel summaries cannot
be extrapolated to full-raster neural IoU. Only mesh silhouettes use full rasters.

Call the original `_construct_ray_batch`, `_process_ray_batch` and `render`
methods without changing their implementations. Unused zero RGB placeholders
serve only the camera batch constructor; no RGB target or loss is evaluated.
Preserve the original CPU ray-pool construction and then move the selected batch
to CUDA for original ray processing; do not substitute GPU calibration inversion.
Verify derived rays against exactly those used for mesh tracing within 2e-6.
Use `is_nerf=False`, original **training-forward** policy (`is_train=True`,
original perturb=1 via overwrite −1, source `get_anneal_val(s)`, s=step−1) and
two fixed draws, seed `6033+1000*camera_ID+draw`, reset per camera/draw and shared
across checkpoints. This avoids the different source validation alpha policy
being mistaken for training-volume fit. It does not reproduce historical
training RNG states, which were not saved. Original curvature and occlusion
auxiliaries still execute; their work is included. No Trainer, backward,
optimizer construction/update, optimizer restoration or checkpoint writer runs.

Switching the PyTorch module's training flag for this forward-only measurement
does not authorize optimization. Preserve original CUDA factory context for
unqualified tensor constructors and restore CPU defaults on every exit.
Save raw `fg_acc`, total accumulation, native RGB/foreground/background outputs,
seeds, loop index, calibrated rays and paired mesh/oracle labels. Native G RGB
is display-encoded; do not call these outputs a new linear reference dataset.
The allowed accumulation roundoff envelope is [-1e-4,1+1e-4], justified by the
source's 160-term float32 cumprod with +1e-7 factors. No values are clipped.

Report continuous foreground/background accumulation, per-stratum confusion
at primary threshold .5 and fixed descriptive thresholds .1/.9, mesh–volume
disagreement, stable errors across both draws and sampling-induced threshold
flips. Two draws show sensitivity, not a converged Monte Carlo uncertainty
interval. Different learned SDF importance locations remain expected even with
common primary random tapes. No tuning follows any diagnostic result.

## Runtime and hard work ceilings

Use the accepted corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`
and authoritative G `79252178e65914ad1acbc14790b1514f276bfea0`.
Copy the hash-bound accepted G-training nvdiffrast cache into the owned executable
output cache. Invoke only the original runtime loader; its cache validation or
same-source rebuild is bounded by the same resources, MAX_JOBS=4, with no image
build or installation. Require the accepted plugin binary SHA-256 **before any
scientific query/render**, then again at completion. A changed binary is a stop,
not a silently accepted environment update. Inputs and original cache are read-only.

| Work | Maximum |
| --- | --- |
| CPU checkpoint deserializations | 4, exactly the files above, once each |
| Dense raw grids / original tiles / SDF nodes | 3 / 1,536 / 402,653,184 |
| MCubes calls | 3; none at 300k |
| Raw/candidate faces per extraction | 4,000,000 |
| Earlier six-camera BVH rays | 589,824 |
| All-checkpoint fitting BVH rays | 7,995,392 |
| Total native BVH rays | 8,585,216 |
| Independent CPU ray witnesses | 1,408 |
| Extra raw-SDF consistency / first-derivative points | 4,608 / 1,536 |
| Original neural render calls | 128 |
| Neural rays | planned 32,440; ceiling 32,768 |
| Neural final ray-sample slots | 5,242,880 |
| Neural SDF-forward point slots | 35,127,296 |
| Neural gradient point slots | 10,485,760 |
| Original occlusion SDF-probe origins | 196,608, 64+16 samples each |
| Validation-only secondary probes | 0; source training branch preserved |
| Appearance calls / point slots | 128 / 5,242,880 |
| Optimizer returns / checkpoint writes / Material A calls | 0 / 0 / 0 |

The SDF-forward ceiling includes original coarse sampling (64+3×16 per ray),
core SDF+gradient and curvature gradient (3×160 per ray), plus bounded 80-point
occlusion probes. These are primitive point slots, not independent visibility
queries or total FLOPs. A profile observer counts actual original entries and
fails closed outside the declared phase/caps; no renderer arithmetic is replaced.

One A100 GPU0, four CPU threads, 128 GiB host RAM/no swap, 128 PIDs, 75% CUDA
allocator, 8 GiB output cap; foreground 3,600s watchdog/30s grace and inner 3,540s.
Estimate 5–30 minutes, unmeasured. No timeout extension. Source/root/inputs are
read-only, non-root, no network. Preserve all partials on failure, without retry.
The host refuses a dirty checkout, wrong image, occupied GPU, active old geometry
containers or an existing output. Source/input/receipt hashes are verified before
and after execution and during file-only replay.

## Interpretation and future intervention

Full fitting mesh failure supports training-view geometry fitting failure;
better fitting than held-view geometry indicates a generalization gap but does
not isolate a cause. Neural accumulation matching masks while the zero mesh
fails supports a volume/zero-level mismatch on measured rays. Stable errors in
both indicate a broader fitting failure. Background-NeRF RGB can compensate
without constraining foreground occupancy to the observed mask. Domain cohorts
and open boundaries remain separate from all these cases. No trimming or repair
uses any diagnostic label.

If supported, the prospective isolated **G_background_mask_v1** ablation is one
fresh 300k history, seed6033, exact original initialization/acquisition/schedules,
changing only the mask loss by adding
`lambda_bg * mean((1-mask) * fg_acc)` with **lambda_bg=1**. Multiply the full sum
by the existing mask schedule. Preserve the original full-batch mean and absolute
foreground expression exactly; do not rebalance foreground/background means.
On physical acc∈[0,1] this equals the requested
`mean(mask*(1-fg_acc)+lambda_bg*(1-mask)*fg_acc)`. Retaining the exact L1 foreground
also preserves the original subgradient at acc=1 and roundoff behavior outside
the ideal interval. No live G variant or training launcher for this loss is
implemented or authorized here. Source analysis/synthetic derivatives do not
establish that the old mask expression caused the observed mesh errors.

Background silhouette supervision is an established technique, not a novelty
claim; the released [NeuS implementation](https://github.com/Totoro97/NeuS/blob/main/exp_runner.py)
already uses opacity-versus-mask BCE. The proposed L1-compatible ablation is not
that BCE and changes no other objective. Extraction/BVH/serialization repairs
remain implementation correctness work. Anisotropy-aware directional geometry
appearance and ambient-tensor integration are separate, unimplemented method
candidates, not established contributions from this diagnostic.

Bell validation/held-view outcomes now inform method design and are explicitly
**development evidence**. Freeze any intervention before independent confirmation
on another sufficiently dense scene or a prospectively reserved camera acquisition
outside these 128 views. No new reference acquisition is generated here. Do not
claim a three/four-checkpoint temporal association is a controlled loss ablation.
