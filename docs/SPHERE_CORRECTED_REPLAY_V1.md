# Corrected sphere reconstruction

This is one fresh `A_corrected_core_v1` reconstruction of the existing NeRF
synthetic sphere, using pinned Material A at
`e1e859779301faced46d6476691e20b546c27ae0`. It is a separate historical-mesh
replay, not an admitted research G→A handoff. The stage launcher's Bell/Bear
selection and mesh-admission rules are unchanged. Native execution requires one
explicit approval of the final source/freeze, inputs, budget and fresh output.

## Inputs inspected

- Dataset: `/home/isavand/nero/data/nerf_synthetic/sphere`. All 201 active
  files (train JSON, 100 RGBA images, 100 masks) match the copy under
  `/home/isavand/paper/data/nerf_synthetic/sphere` by SHA-256.
- Mesh: `/home/isavand/nero/final_results/meshes/sphere_shape-300000.ply`,
  8,583,965 bytes, SHA-256
  `3e512585ad62aafdbdfdde785f64d1be2454a891fde2f730c027f5b02ed60eba`.
- Historical NVS: `/home/isavand/nero/final_results/nvs_imgs/sphere_material`;
  five 1080×2160 PNGs, original vertical GT/prediction panels.
- Historical log: `/home/isavand/nero/final_results/logs/sphere_material/log.txt`.
  Reported average PSNR 35.08154, SSIM 0.95358. These are historical reports,
  not newly reproduced results.
- Historical validation: `train_vis/sphere_material-val/99999-index-0.jpg`.
  The filename is a nominal step label, not producing-checkpoint attribution.

The original and pinned copies of `configs/material/nerf/sphere.yaml` are byte
identical (SHA-256 `06d7de5307054ddf775f55b40874d28cf63a75ee47a4b63f4b4b1dc6c943b4c7`).
No historical material checkpoint or producing configuration was located in the
specified historical tree. The NVS log contains no step identity. Final NVS
comparisons therefore label the historical training step **unknown**, and never
claim an equal-budget causal comparison. Source RGBA-to-white GT conversion
agrees within one uint8 level on nonzero historical GT pixels; the final comparison
checks complete GT arrays and keeps separate GT panels/scores if they differ.

## Preserved computation

The original `Trainer.run()` executes unmodified. Original NeRF loader,
white-alpha composition, 1080×1080 images, OpenGL camera-to-world poses with
translation divided by two, integer pixel rays, normal negation/normalization,
and `is_nerf=True` remain intact. The loader reads only `transforms_train.json`.
`transforms_test.json` exists but references a missing test directory; it is not
used or substituted into the split. Python seed6033 yields validation database
ID36, NVS IDs39/2/94/62/23 in that order, and the remaining94 training IDs.

Keep full feature/material/learned outer and inner illumination networks; no
known environment, Bell lighting setting, head-only restriction or reduced image
resolution. Preserve 512 training rays, 512 diffuse/1024 specular samples,
seed6033, eight loader workers and original sampling/draw order. Fresh weights
only: no historical checkpoint loading and no implicit resume.

Original Adam/AMSGrad uses eps1e-4, default betas(.9,.999), zero weight decay;
clip norm .7. Warmup1000 starts at LR0 then5e-7, peaks at5e-4; retain the
hard-coded100000 cosine horizon and alpha.2. Original Charbonnier RGB loss
sqrt(sum-channel-squared-error+.001), including original render-loss coefficient
schedule (1→5→1 over0/5000/10000), and material/diffuse-light priors remain.

The original YAML's top-level energy/specular .01 declarations **never reached
the shader**. The replay records these ignored values and retains actual shader
defaults .05/.05, `reg_change=True`, epsilon.02 Gaussian spatial prior,
reg_lambda1=.05 and random azimuth. `regularization_version=legacy_v1` delegates
the historical regularization function without changing its conditionals or RNG.
Renderer diffuse-light coefficient remains .01. Unused top-level max_n_exp /
max_alpha_exp and shader shading_feature_num / num_viewdir_freqs are explicitly
diagnosed, not promoted to effective controls. Full architecture is unchanged.

Intentional differences are the accepted corrected-core estimator/BRDF,
atan2 orientation and tangent construction, and admitted FP64-predicate BVH.
Legacy priors act on these corrected quantities; this is not numerical identity
to historical A. Invalid orientation, nonfinite loss/gradient, invalid outgoing
normal domains or resource failure stops the run: no normal flips, fallback,
automatic repair, tuning or retry. The historical backend identity is unknown.

## Outputs and progression

Each2000 completed original optimizer returns: validation ID36, original
validation metric/JPEG montage and best-checkpoint selection, regular model.pth,
immutable step-N.pth with optimizer state, and supplemental lossless maps/NPZ.
Original zero-based montage filename is1999 at2000 completed returns.

Each10000: all five original NVS IDs, original vertical PNGs/logged PSNR/SSIM,
plus the same richer maps/arrays. Extra NVS calls at10000…90000 restore Python,
NumPy, CPU/CUDA RNG, model modes and default tensor type. The original final NVS
call supplies100000 without duplication. Original validation RNG behavior is
preserved. No extra field/shader calls are made merely to export material maps.

`renders/step-N/{validation,nvs}/camera-ID` contains GT/predicted RGB, albedo,
metallic, raw mx/my, canonical max/min widths, major-axis orientation, normals,
diffuse/specular RGB, mask, `materials.npz` and identity/hash metadata. NPZ retains
raw double-angle rotation, raw head logits, linear components, world positions
and shading normals, and original encoded tangent/bitangent/normal maps. No
material ground truth is available for this existing sphere: maps are estimates,
not evidence of physical parameter recovery.

Display conventions are fixed: raw unit coefficients/widths use0…1; RGB uses
original uint8 clipping with no extra gamma; diffuse/specular display uses
original shader sRGB conversion, with linear values retained numerically.
Orientation hue spans a major-width angle moduloπ in the corrected local tangent
chart; exactly isotropic pixels are grey. The chart has the accepted tangent
reference seam. Raw head widths remain unsorted in training and the archive.
Background is the original mesh-hit-masked black, including masked GT.

Final file-only processing writes full-resolution GT/original/corrected sheets,
native-definition per-image PSNR/SSIM, comparison caveats and training curves.
If GT differs, the sheet contains both GTs and reports each prediction against
its own preserved GT. Historical log averages and training-step uncertainty stay
separate. No resized/cropped substitutes or unmatched-step equivalence claims.

## Bounded execution

Use [the working config](../configs/replays/sphere_corrected_core_v1.yaml),
[entry point](../run_sphere_replay.py) and
[foreground launcher](../experiments/run_sphere_replay.sh). The freeze binds source,
both pinned snapshots, active inputs, historical comparison files and output ID.
All source/input mounts are read-only, root read-only, non-root, no network.
Image `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`;
corrected binary `800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.

Proposed caps: one A100GPU0, four CPU threads, original eight loader workers,
128GiB host RAM/no swap,256 PIDs,64GiB owned outputs; **24h foreground watchdog**
with30s kill grace and85800s inner timeout. Cooperative stop reserves120s for
cleanup/final inventory. No new GPU allocator cap or scientific precision change.
100000 original Adam returns max;50 validation+50 NVS images, full1080².
Worst-case shading direction slots:78,643,200,000 training +179,159,040,000
evaluation=257,802,240,000. Actual hit-only work is lower. Runtime is unmeasured
for this sphere; full native preprocessing and frequent high-resolution rendering
may exhaust the cap. Preserve last completed checkpoints/images on timeout;
no extension, automatic retry or claim of completed100000 steps. Checkpoints
contain original optimizer state but not a complete RNG/sampler resume capsule;
future continuation needs its own explicit review.

Relative original writers are confined to a new owned CWD. Only assets/BSDF lookup
inputs are symlinked read-only; train_vis links to owned output. Active writers:
Trainer checkpoints; Logger text/TensorBoard; MaterialRenderMetrics JPEG; original
NVS PNG/log; supplemental maps/NPZ/manifests/JSONL; final comparisons/curves.
NeRF loader does not create resize/pickle caches. Optional original cloud.ply
debug writer is inactive; even if called its relative destination is owned CWD.
No extraction, standalone eval pointcloud generation or historical pth loading.

Preparation runs only focused synthetic CPU tests and static checks. No native
model, training, rendering or CUDA is launched before the explicit approval.
Docker socket access is restricted in Codex; use the exact approved researcher
tmux command, without changing permissions or Docker settings. The one-time
command/authorization will bind the final commit and freeze after review.
