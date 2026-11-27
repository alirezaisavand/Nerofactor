# Gate 11 — Isolated oracle-mask background-opacity diagnostic

Prepared for one separate exact-source/freeze/output authorization. This task
does not authorize training, checkpoint payload loading, SDF queries, extraction,
GPU rendering, SAM2 inference or R0/R1 optimization. Preserve O0, Gates 1–10.3,
all source snapshots, rejected geometries and historical authorizations.

## O1 versus authenticated O0

O0 is the accepted original 300k history at
`bell-geometry-gate10-v4-5eba8365e19e4ab9aa2320491429cffb/train`, authoritative G
`79252178e65914ad1acbc14790b1514f276bfea0`. Its 300k geometry admission remains
blocked. Reuse its existing Gate 10.3 checkpoint evolution and opacity analysis;
never rerun O0 or select a different O0 endpoint.

O1 is `G_background_opacity_oracle_v1`, an explicitly selected derivative, not a
change to frozen G. The derivative inherits the original renderer and delegates
the complete original `train_step`. It retains a view of the **pre-call** CPU
mask slice, since the original method may replace its shuffled dictionary before
returning. It then changes only:

```text
foreground = outputs['loss_mask']  # exact original F.l1_loss(m, m*fg_acc, mean)
background = mean((1 - same_original_mask_slice) * outputs['fg_acc'])
outputs['loss_mask'] = foreground + 1.0 * background
```

The original `MaskLoss` applies its exact cosine coefficient .3→1 over loop
indices 0..30000 to the combined scalar. Retain the full original ray denominator,
original L1 subgradient and behavior outside ideal opacity [0,1]. No clipping,
BCE substitution or separate foreground/background balancing. Coefficient one
is a prospectively specified symmetric oracle-mask diagnostic, not a tuned optimum.
No module, parameter, buffer, forward render or other objective is added.

The source dispatch dictionary selects the derivative only inside the new
training context and restores the original entry afterward. Original Trainer.run,
architecture, SDF initialization, curvature, entropy opacity, occlusion, RGB,
background NeRF, appearance approximation, sample counts, optimizer and all
schedules stay unchanged. Evaluation loads the structurally identical weights
into the original architecture; its training-only mask derivative is inactive.
Producing receipts identify O1 explicitly. Do not mislabel its checkpoints as
original G or promote them through an unchanged-original-G handoff declaration.

## Acquisition, initialization and lifecycle

Use identical admitted Bell v3 RGB, synthetic oracle-hit masks and safe NPZ camera
calibration at 128². Preserve the original 122 fitting / validation58 / five held
11,105,114,17,76 split and ordering. The training mount contains only the same
123 RGBs/masks and validation depth; no fitting depth, oracle mesh, physical
labels or held-camera files. The existing acquisition guards and two original
Python split-seed resets remain in place.

Original Trainer seeds Python/NumPy/Torch with 6033. Before the first forward,
require network fingerprint
`9c7b468700061fa7f45fa132fd20bb03e982d52184fdbeed4263fd4200027dd4`
to match authenticated O0 exactly. No weights are loaded for initialization.
The derivative has no extra initialization, random draws, parameters or heads.
Preserve the original CPU shuffle policy, eight spawned DummyDataset workers,
camera transform and native RNG policy. O0 did not save full per-update tapes;
do not claim observed bitwise sampling parity for the whole history. Geometry-
dependent branches may consume different GPU random counts as the fits diverge.

Preserve 512 training rays, 64 coarse+64 importance+32 background sample slots,
original Adam/no clipping and warmup-cosine LR. The complete original
[effective loss/LR schedules](BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.md) apply;
none is truncated or compressed. Exactly 300,000 original Adam returns, fresh
weights, no resume/retry/extension. Retain all 300 natural validation/save boundaries,
every-20-step logging, native best, fixed endpoint and full 1k/15k/30k/50k/100k/200k
checkpoints. Only the fixed 300k endpoint determines primary quality eligibility.

The natural 1k save is the established lifecycle gate, not a geometry-quality
gate. It checks finite sampled losses/gradients, first-two Adam state advancement
and zero/nonzero displacement, owned checkpoint roundtrip and cleanup/data access.
Continue automatically after this gate under the single approved batch.

Record foreground/background/total raw mask terms and weighted parts every20
updates plus the original phase probes: 15,006 rows. The callback receives detached
tensors and contributes no objective term, random draw or extra reverse pass.
The original logged `loss_mask` remains the actually optimized combined term.

## Four-checkpoint evaluation

After training completion and cleanup, evaluate 15k/30k/50k/300k in one separate
isolated process. CPU-deserialize each newly produced, receipt-bound checkpoint
once, discard optimizer payload without restoring it, and require strict finite
network catalog/step identities. Together with the natural 1k roundtrip this is
**five owned CPU loads**. No historical checkpoint load is authorized.

Reuse the Gate 10.2 domain-aware rule: original raw 512³ float32 nodes at [-1,1]³,
zero level, fully-inside generating cells, exact float32 coordinate welding and
exact-zero-area removal only. Preserve open boundaries, disconnected components,
winding and all index mappings. No artificial positive exterior values enter
marching cubes; no GT trimming, hole fill, smoothing, remeshing or normal flip.

At each checkpoint, measure all 122 fitting silhouettes and the original six
validation/held rasters, with the accepted corrected BVH and independent FP64
witnesses. Preserve signed normals, same-ray distances, full exclusions and
all original criteria: per-camera IoU≥.90, correspondence≥.80 with displacement
≤.02 and signed-normal angle≤30°, plus the unchanged numerical/SDF contracts.
Raw-SDF zero/gradient/crossing checks and negative-extension/domain/topology
statistics remain identical. All quality failures are retained; earlier quality
failures do not prevent completing the later fixed checkpoints.

Use the unchanged 16 fitting cameras, 4,055 unique rays/draw and two fixed draws
from Gate 10.3. Original CPU camera-prefix methods enforce the accepted 2e-6
archive match before CUDA transfer. Keep original shader/SDF arithmetic and TF32
policy; no repeat of the resolved camera precision intervention is needed.
Use original training-forward sampling, loop index step−1, seed6033+1000*camera+draw.
No backward/optimizer, extra architecture or Material A execution in this phase.

Replay decisions, masks and identities exactly. The accepted adjacent-float64
normal-angle summary rule applies to fitting and six-view summary replay only;
log every roundoff event. It does not alter native scientific thresholds.
All other replay mismatches fail. No capture is replaced by a reference value.

## Bounded execution envelope

Same corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A10080GB GPU0, four CPUs/threads, original eight spawned training workers,
128GiB host RAM/no swap, 256 PIDs, .75 CUDA allocator, 16GiB owned outputs per
phase. Train hard watchdog129600s/inner129540s (36h), evaluation7200s/7140s (2h),
30s termination grace; 1k lifecycle boundary≤3600s. Cooperative batch-time checks
use 138000s. Per-phase watchdogs enforce the scientific execution limits.
Historical O0 duration was about11h; O1 duration is unmeasured. No extension.

| Work | Maximum |
| --- | --- |
| Original Adam returns / training rays | 300,000 / 153,600,000 |
| Training ray-sample slots | 24,576,000,000 |
| Original occlusion origins after20k | 573,440,000, original64+16 probe samples |
| Natural validation calls / rays / slots | 300 / 307,200 / 49,152,000 |
| Periodic saves / best saves | 300 / at most300 |
| Owned CPU checkpoint loads | 1 natural lifecycle +4 evaluation |
| Dense grids / tiles / SDF nodes | 4 / 2,048 / 536,870,912 |
| Marching-cubes calls | 4; at most4,000,000 faces each |
| Six-view + fitting BVH rays | 786,432 +7,995,392 =8,781,824 |
| Independent FP64 witnesses | 1,792 |
| Consistency SDF / gradient points | 6,144 / 2,048 |
| Neural renders / rays | 128 / planned32,440, ceiling32,768 |
| Neural SDF / gradient point slots | 35,127,296 /10,485,760 |
| Additional mask reverse passes / SAM2 / Material A / external training | 0 /0 /0 /0 |

All remaining primitive evaluation/appearance caps are in the adjacent freeze.
Reuse the accepted hash-bound nvdiffrast cache through the same original runtime
loader/byte-identity contract, owned executable output cache and MAX_JOBS4.
No image build, install, new plugin or synthetic warmup. Source/root/inputs
read-only, non-root, no network. Reject dirty source, old active experiments,
wrong image/GPU, occupied GPU, collisions or missing hash-bound inputs.
Identity, numerical, lifecycle or resource failures stop without retry.

## Primary comparison and interpretation

Compare fixed300k O1 against archived O0. Report split-pooled precision/recall/IoU
and per-camera gates, depth/normal/displacement distributions and tails, correspondence,
neural foreground/background opacity and stable-vs-sampling errors, domain support,
negative extension, topology, open boundaries and runtime. Explicitly report loss
of foreground recall/depth/normal accuracy even if silhouette precision improves.
Retain intermediate comparisons as diagnostic only. No held-view tuning or
checkpoint replacement. Quality eligibility creates a review record only; geometry
admission and an O1-identified handoff require researcher review.

The original foreground-only term was intentional for incomplete high-precision
SAM2 masks. O1 uses perfect synthetic masks solely as a diagnostic; success cannot
establish imperfect-mask robustness, new methodological novelty, or causation by
any other loss. The separate realistic acquisition/R0-R1 protocol is not executed
under O1 approval. Bell is development evidence; freeze an intervention before
another dense scene or prospectively reserved acquisition confirms it.
