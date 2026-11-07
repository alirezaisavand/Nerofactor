# Bell explicit camera-0 diagnostic protocol v1

Protocol ID: **`bell-camera0-validation-diagnostic-v1`**, milestone **1D.4B-3C**.
**Preparation for review; no Docker/GPU inference executed by Codex.** This is a
source-camera diagnostic using the maintained A implementation, not historical
thesis reproduction, G execution or producing-run attribution.

## New evidence, hypothesis and fixed inputs

The researcher visually identifies camera **0** in historical montage
`train_vis/bell_shape-val/149999-index-0.jpg`, and camera **58** in the current
admitted R ground truth/prediction. Record this as **researcher visual evidence**;
no independent producing-run/source/split chain is established by the claim.
An unshuffled two-way producing split (validation 0, training 1–127) is a leading
hypothesis. It is consistent with the inspected G rule and other historical
first-ID rules, not proof of G origin or the reason for a metric difference.

The [new admission manifest](../evidence/bell-camera0-diagnostic-admission.json)
binds that statement, original evidence pointers and the actual source-0 file
identities. It does not rewrite the
[successful image-58 evidence](../evidence/BELL_INFERENCE_REGRESSION_SUCCESS.md).

- Exactly `thesis:model/bell_shape/model_best.pth`, SHA-256
  `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
  Saved step **150000** and `best_para=23.787122565451323` remain unchanged.
- Same original `configs/shape/syn/bell.yaml`, independently resolved direct
  FullLoader route, actual effective trainer/renderer/shader defaults and
  preprocessing/cache infrastructure. No scientific override is exposed.
- Source camera string **`"0"`**, explicitly selected through database getters
  for RGB, camera pose/intrinsics, training mask and depth/mask. The original
  database class/getter source is pinned and numeric ID ordering is verified.
  The selected source is not inferred from validation iterator position 0.
- Same 375-file admitted initialization/input superset, plus camera-0 depth:
  **376 file identities**. Camera 0 RGB/mask/camera metadata already belong to
  that superset; the new admission separately pins all four camera-0 inputs.
- Historical montage SHA-256
  `2800003a665d47f757b1ec30b98d225bcfe19751b2608f1caa75587f8bcb3a44`,
  RGB JPEG **800 × 1000**; frozen GT panel box **`[0,0,200,200]`** (x0,y0,x1,y1).
  This uses the inspected montage layout plus the researcher identification.
- Existing admitted immutable image
  `sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`,
  Python 3.8 / Torch 1.8.1+cu111 / CUDA 11.1, same selected A100 GPU class.
  Caller image/commit declarations are not a complete launch/build attestation.

## Two separately bounded evaluations

1. **Primary:** source camera `"0"`, evaluation step **149999**, one full existing
   200 × 200 validation rendering in a fresh single-use process/output root.
2. **Step control:** same source camera, checkpoint, configuration, inputs,
   initialization, RNG policy and runtime, evaluation step **150000**, another
   fresh process/output root. Never continue after the primary process's RNG
   state, modify the saved checkpoint's step, or reuse its output directory.

These two renders are the complete proposed native budget. There is no training,
optimization, checkpoint write, mesh extraction, full NVS or camera/seed/score
search. Any extra repeat requires separate review. If the pair differs, native
nondeterminism is an alternative to step sensitivity; without a same-step repeat,
nonzero differences cannot establish causality. Source review and raw differences
must remain separate from observed repeatability claims.

## Evaluation adapter and scientific boundary

Keep the existing original A initialization prefix: Trainer defaults and seed
6033 in original Torch/NumPy/Python order, train/validation DummyDataset setup,
the two actual A split calls, original shape constructor, full A ray-pool build/
shuffle, and `.cuda()`. No DataLoader iteration occurs in this inference protocol.
Retain original A train/validation/NVS lists, including validation **58**, exactly.
Camera 0 is an A training ID under that partition; this diagnostic does not label
it held out under A or establish its historical membership.

Verify the actual live 137-tensor architecture against the admitted reference,
checkpoint bytes before and after trusted CPU `torch.load`, stored step/metric,
and exact network keys/shapes/dtypes, then use the existing strict state loader.
Torch 1.8 still deserializes the full trusted pickle container; unused optimizer
payload is discarded, never restored or used. No model/optimizer checkpoint is
loaded from Codex. Record loaded network/buffer hashes and verify identical
values after the diagnostic, without freezing gradients needed by the SDF.

A separate parameter-free **external adapter** owns its own `["0"]` evaluation
ID table and image information built with the original `build_imgs_info` and
`imgs_info_to_torch`. It never assigns the real renderer's test/train/NVS tables,
changes its class, patches its functions or loads state into a differently named
network. A read-only getter facade records the actual source IDs. Expected
selected reads are image/pose/K/mask/depth, all **`"0"`**. Verify original table
identity/membership and getter mapping after evaluation.

The adapter invokes the original renderer **forward and test_step functions**
on this selection facade. All downstream render/camera/network methods remain
bound to the real renderer. It explicitly delegates original `zero_grad` behavior.
Original child module calls and weight-normalization hooks remain active; original
root/global forward/backward hooks must be absent or the adapter refuses the run.
This restriction prevents silently changing custom root-hook behavior when using
the external wrapper. No registered parameters/child model are added to it.

Use original `eval`/`torch.no_grad()` validation boundaries, original CUDA-default
forward handling and internal SDF `torch.enable_grad`. No `inference_mode`,
derivative suppression, new pose or scientific operation replacement. Camera-0
image information is built at full original resolution and downsampled **inside
the original test_step**: ratio 0.25, original Gaussian/linear RGB resize,
intrinsic scaling and nearest-neighbor depth/mask resize. Preserve 512-ray
chunks, all sample counts, backgrounds, masks, shader defaults and math.
Camera selection must consume no scientific RNG; before/after RNG hashes record
the original render's actual draw consumption, including auxiliary calculations.

## Source-level step review

Pinned maintained sources: `network/renderer.py` and `network/field.py`, with
identities in the original structural admission. Relevant current locations:

| Active path | 149999 versus 150000 |
| --- | --- |
| `forward:1046–1068`, `test_step:520–579` | Step forwarded unchanged; validation calls `render(..., perturb=0, cos_anneal_ratio=0, is_train=False)`; no training annealing helper |
| `compute_sdf_alpha:811–842` | `step < freeze_inv_s_step` (15000 from bell YAML) is false in both cases |
| `compute_occ_loss:852–880` | `step < occ_loss_step` (20000) is false in both cases; auxiliary occlusion/RNG work is retained |
| `render_core:919–1044` | Early SDF initialization branch `step < 1000` is false in both; curvature schedule references are commented out, not active arithmetic |
| `compute_validation_info:790–809` | Passes step to the same geometry shader; no new schedule |
| `AppShadingNetwork.forward:626–696` / `predict_specular_lights:580–608` | Step is forwarded, then not read in specular-light calculations; material-stage shader schedules are a different path |
| `ShapeRenderMetrics` | Step is used in output filenames; original full-image metric formulas do not depend on step |

**Source expectation:** no active mathematical branch/expression distinguishes
these two evaluation steps in the inspected geometry path. This is not a measured
GPU equality result. CPU tests inspect the actual step predicates and exercise
original forward/test_step/SDF-gradient bodies with stand-ins. Unreviewed source
changes invalidate the pinned source gates; never rely on this table after them.
No claim of general step independence at other steps or configurations is made.

## Frozen numerical and JPEG comparisons

Preserve original RGB conversion and metrics: multiply by 255, clip, truncate to
uint8; float32 pixel MSE per channel then mean channels for PSNR; full-image SSIM
with win_size=11, channel_axis=2, data_range=255 and the admitted package versions.
No extra masking, background substitution, gamma conversion or alternate resize.
Save primary/GT float32 RGB and int32 mask losslessly, plus original-conversion
PNGs and original validation montage. The checkpoint/data and observed state
identities must be equal between the two steps; GT/mask values must be exact.

For the **same-camera step pair only**, reuse unchanged existing inference
thresholds: float `abs(control-primary) <= 1e-4 + 1e-4*abs(primary)` elementwise,
uint8 max error <=1, PSNR delta <=0.01 dB, SSIM delta <=1e-4. Compare all input/
source/config/runtime/initial weight and before/after RNG identities. Record
bitwise RGB equality, every RGB difference/violation, metric deltas and auxiliary
output fingerprints. Auxiliary hash differences are diagnostics, not a new
unreviewed depth/normal tolerance. No threshold/CLI option may be adjusted after
seeing results. Passing only bounds this observed pair, not historical origin or
universal determinism.

For the **historical score comparison**, report signed/absolute differences from
stored full-precision PSNR 23.787122565451323, logged PSNR **23.78712**, and
logged SSIM **0.87727**. Record whether the new values have the same five-decimal
printed display. Historical SSIM has only five logged decimals. There is **no
new historical acceptance tolerance** and no score-fitting loop. Completion of
the diagnostic does not require historical metric agreement. The original
image-58 values 32.84983236325549 / 0.946790605392639 remain a distinct observation.

For **GT versus historical JPEG panel**, use the fixed box without resizing,
alignment, recompression, denoising or any attempt to remove compression error.
Decode the admitted JPEG once; save the decoded panel losslessly with a JPEG-
derived label and the int16 signed difference from the new true GT uint8 image.
Report max/mean absolute difference, RMSE, p50/p95/p99, exact pixel equality and
Pillow/libjpeg versions. JPEG is lossy; no pixel-exact identity or pass/fail
threshold is inferred. Never use that decoded historical panel as the scoring
GT. Unknown historical preprocessing/layout/version effects remain limitations.

## Expected evidence and failures

Each fresh root contains `camera-diagnostic.json`, original YAML, owned console
log, lossless prediction/GT/mask NPYs, prediction/GT PNGs, original renderer
montage, decoded historical panel PNG, signed JPEG-GT difference NPY, and
`metrics.json`. Reports include source ID versus iterator mapping, getter reads,
stored/evaluation steps, actual effective settings, input hashes, strict state
fingerprints, runtime, RNG, auxiliary metadata and output hashes. Cache/home/JIT/
temp writes remain under that root; checkpoint directories stay empty.

Reports start `running`, retain current phase and become `complete` only after
one valid view and unchanged state/source/input checks. Errors preserve type,
message, console traceback and any partial output. External termination can leave
incomplete reports, never successes. No checkpoint fallback/resume/retry exists.
The CPU comparison loads only explicit SHA-bound complete reports and saved
non-pickled arrays. It writes a separate `step-comparison.json`, full signed RGB
difference and float violation mask; it never loads a checkpoint or model.

Neither metric closeness, JPEG similarity, successful execution nor current
structural compatibility proves producing commit/configuration, historical split,
training membership, thesis benchmark reproduction or that the metric gap is
caused by evaluating a historically trained-on image. Preserve the hypothesis
as such even if the camera-0 numbers closely match the historical log.
