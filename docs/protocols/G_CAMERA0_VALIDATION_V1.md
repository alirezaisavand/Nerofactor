# Original G camera-0 validation v1

Protocol ID: `bell-G-camera0-validation-v1`. Prepared, not natively executed in
1D.6A. This is a selected-G diagnostic of an historically unattributed checkpoint,
not historical G producing-run reproduction or validation-split attribution.

Require the [compatibility protocol](G_CHECKPOINT_COMPATIBILITY_V1.md)'s completed,
explicit SHA-bound report from this exact implementation, original G source
`79252178e65914ad1acbc14790b1514f276bfea0`, original bell YAML without scientific
overrides, and exact admitted `model_best.pth` hash. Recheck schema/catalog and
strictly load the network only. No optimizer state load, update or checkpoint
write; no Trainer.run, manual validation schedule or extraction.

Call original G `_init_dataset` then `_init_network`, preserving constructor
seed/order, loaders, preprocessing and renderer initialization. Do not reduce
image/ray/sample sizes or alter training tables. Original source input is 800²;
G validation's original quarter-scale Gaussian/linear-color/intrinsics path
produces 200² RGB, with original nearest mask/depth resize. Original test rays
are chunked in 512s. G's two original split calls both seed Python random 6033
without shuffle; ordered train IDs are `1`–`127`, validation `0`, no NVS table.

Use original validation dataset/loader, original `ValidationEvaluator`, registered
losses, `ShapeRenderMetrics` and key metric. Exactly one index-0 forward at
zero-based step **149999**; stored checkpoint step remains **150000**. Observe
actual original getter calls externally: source image, pose, intrinsics, mask
and evaluation depth must identify database camera **"0"**, rather than assuming
iterator index. Record database-instance mappings, both split calls, image-info
IDs/shapes/dtypes and original camera/cache pickle read paths. Original loaders
read trusted researcher camera pickles only in the isolated native runtime.

Preserve original renderer `forward`/`test_step` and SDF `enable_grad` with
`autograd.grad(create_graph=True, retain_graph=True)`. Keep original evaluator
`no_grad`; do not use `inference_mode`, model-method replacement or A scientific
imports. Original test step calls `zero_grad`, not optimizer.step; evaluation
must leave network weights/buffers and training-ray cursor unchanged. Fresh
optimizer state must stay empty. Record before/after RNG, not an invented
RNG-equality requirement: the original DataLoader iteration can consume RNG.

PSNR uses original RGB-to-uint8 conversion, float32 pixel MSE then channel mean,
`10*log10(255²/MSE)`; SSIM uses original win_size11/channel_axis2/data_range255,
full image. No newly masked score, resize, background conversion or metric
substitution. Record all original evaluator aggregates and actual key metric.
Save float32 predicted/GT RGB `.npy`, int32 mask, lossless RGB PNGs, original
G diagnostic JPEG montage and `metrics.json`, plus output hashes.

Historical comparisons are **descriptive**, with no equality or fitted numerical
tolerance: PSNR 23.787122565451323, logged SSIM 0.87727. Decode only the admitted
historical JPEG SHA
`2800003a665d47f757b1ec30b98d225bcfe19751b2608f1caa75587f8bcb3a44`, 800×1000,
top-left GT panel `(0,0,200,200)`. Save decoded panel and signed GT difference,
max/mean/RMS/percentiles, decoder identity and montage byte comparison. Do not
resize/align/recompress to force agreement. The shared comparison helper reads
historical metadata and JPEG only; it does not import A scientific modules.

## Acceptance and bounds

Complete `G-diagnostic.json` must prove exact source/config/input/checkpoint,
strict loading, actual camera 0, original G calculations, finite full outputs,
unchanged network, no updates/state restoration/checkpoint files, and successful
resource cleanup. A differing historical score is a diagnostic observation, not
a reason to edit G or borrow A. No repeatability/long-training claim follows.

Use a new isolated process/output, reviewed image, no network, all original and
prior-report mounts read-only, one selected GPU; four CPUs, 64 GiB/no extra swap,
128 PIDs, 2 GiB shm, **900 seconds + 30 seconds** termination grace. Exact
[researcher commands](../G_GEOMETRY_LIFECYCLE.md) require prior compatibility
review. Retain outer terminal/exit and failed inner reports. Stop on blockers.
Do not execute native workloads from Codex or change previous A evidence.
