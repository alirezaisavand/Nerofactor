# Bounded sphere surface-regularizer comparison v1

Status: prepared, not scientifically executed. One explicit exact-source
researcher approval is required. No restart/resume in the interrupted output.
Use two matched new forks of its state4000, not an exact historical continuation.

Two cold-process arms, fixed order:

1. `surface_chart_control_v1`: distance-normalized raw-width / double-angle
   spatial prior on fixed mesh pairs, preserving the historical small orientation
   schedule exactly.
2. `surface_world_q_v1`: replace ONLY that anisotropy part by normal-transported
   world-Q Frobenius smoothness on identical pairs. Keep the scalar kd/metallic
   spatial part, bilateral weights and energy/specular losses identical.

Both are explicit new spatial candidates. The chart control is not an exact
rerun of legacy off-surface smoothness. No third arm, coefficient search, energy
repair, learned-light restriction, parameterization change or GT-material loss.

## Fixed science

- Construct original A full material/learned-light network with seed6033, then
  strictly restore the same selected state4000 weights and Adam in each fork. Pin material
  source `e1e859779301faced46d6476691e20b546c27ae0` and unchanged corrected-core.
- Original sphere YAML/effective settings, original NeRF data/mesh/cameras,
  1080²,94 training /validation36 /five untouched NVS IDs; use v2's explicit
  unit-outgoing boundary. No image resizing, remeshing, normal flip or split change.
- Original Adam/AMSGrad eps1e-4, betas(.9,.999), clipping.7 and original100000
  horizon/warmup1000. Keep total_step=100000 for the original computation, but
  stop externally after exactly4000 NEW original returns, at absolute8000 and
  its completed checkpoint/validation. Do not repeat0…4000 or restore implicitly.
- Checkpoint: `sphere-core-v2-ec8cee6b6d234133b0c7e1168fff02e1/sphere/checkpoints/step-004000.pth`
  under the evidence root, SHA-256
  `610b42044d43d64fd3bf5c8d309fe8acd9f266b05e85acccd92acdd222f173ea`.
  Explicit researcher trust/CPU pickle-deserialization approval is bound to this
  own checkpoint; loading occurs ONLY in the isolated approved runtime with
  map_location=cpu, strict network load and exact Adam-state equality. No other
  checkpoint is admitted. No final NVS or full training.
- Checkpoint lacks historical RNG/sampler position. Both forks retain the common
  seed6033 construction/shuffle state and start the sampler at its initial index.
  Do not describe this as exact historical resume; restoration of weights/Adam
  is exact, the historical sample trajectory is unavailable.
- Original Charbonnier RGB/loss schedule; shader energy/specular coefficients.05
  and renderer diffuse-light.01. Keep `regularization_version=legacy_v1` for
  the frozen energy/specular proxy. The external spatial candidate is recorded
  separately and behavior-tagged in checkpoint buffers/sidecars.
- All restored named parameters/buffers, Adam values, sample tapes, splits, RNG and
  surface-pair descriptors must match before arm2's first update. No truth-based
  initialization or copying from arm1's endpoint. Original Gaussian-neighbor RNG requests are consumed/discarded
  identically so subsequent photometric sampling keeps its original sequence.

## Surface loss

Select512 evenly spaced indices from the deterministic sorted manifold-edge
pair inventory, never by response/material/error. Four zero-area original faces
are excluded only from the new loss domain, not removed from the rendering mesh.
Two centroid samples per edge, raw mesh normal negation as in A.trace, path
length through the edge midpoint in[1e-6,.02], dihedral≤15°. Coefficient.05;
reference length.02 in unchanged scene units. For each selected pair,

`s=.02/d_surface`, `w=visible_pair*exp(-||I_train(p_i)-I_train(p_j)||/.1)`.

Weights use only the first original training camera; exactly1024 camera visibility
queries/arm. Preserve missing/occluded pairs with zero weight, no replacement.
Require at least one positive weight; otherwise stop before updates. No validation
or material truth enters the gate. RGB values are loader values; no new color
transform. Visibility position tolerance2e-4 and normal dot>1−1e-4, with positive
facing and in-frame projection.

Use `Z=max(1,sum(w))`. Common scalar term is
`.05 sum(w*s*(mean_RGB|kd_i-kd_j|+|m_i-m_j|))/Z`.

Chart term is `.05 sum(w*(s*(|mx_i-mx_j|+|my_i-my_j|)
+s²*decay(step)*||rotation_i-rotation_j||²))/Z`.
World-Q term is `.05 sum(w*s²*||Q_i-R_ji Q_j R_jiᵀ||_F²)/Z`.

No normalization by learned anisotropy or threshold-based teacher selection.
The reference length ties units to the historical.02 displacement; squared tensor
and scalar L1 scales differ and are deliberately a declared new objective.

## Observations and acceptance

Exactly4000 NEW Adam returns/arm,8000 new batch maximum, absolute4000→8000.
Validation maps/checkpoints at6000/8000, fixed camera36 and fixed display scales. Save original images and
numeric components plus raw orientation, world-Q and unoriented projectors.
Keep all parameter/eigenvalue/gap distributions; no favorable seed/point removal.

Actual weighted scalar components logged every10 updates. At zero-based steps
4000/5999/7999, record gradients of six components (RGB, tensor/chart spatial,
scalar spatial, energy, specular, diffuse-light) before original backward. Extra
AD uses retain_graph, never modifies .grad or RNG, and records both full-vector
and shared-feature/material-head/lighting group norms/cosines. A zero gradient
has undefined direction, reported null. These probes cannot recover old histories.

Report matched absolute8000-step RGB PSNR/SSIM, transported-Q neighbor distributions,
raw-chart versus world-Q seams, albedo/metallicity/diffuse allocation/illumination
compensation and compute overhead. No material GT exists. A useful candidate
should reduce transported-Q p95 by≥20% while losing at most.2dB PSNR/.005SSIM;
flag any≥20% decline in anisotropy gap or mean tensor trace as a possible
isotropization/shrinkage explanation, not physical recovery. These are new
comparison interpretation rules, not retroactive acceptance of the prior run.
No automatic full training even if the candidate meets them.

## Work, resource and failure bounds

At most8000 original RGB backwards/Adam returns, plus36 component reverse sweeps;
8192000 surface-pair material predictor point evaluations;2048 extra geometry
visibility rays;6291456000 training and7166361600 validation shading-direction
slots. No independent-reference generation, relighting, NVS or GPU quadrature.

Use existing image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`
and admitted predicate_fp64 binary. One A100GPU0,4 CPU threads,8 loader workers,
128GiB RAM/no swap,256 PIDs;8GiB cooperative output cap/arm. Whole foreground
3600s+30s grace, host arm1740s+30s, inner1680s+20s; cooperative1600s leaves cleanup
time. Runtime estimated10–30min, unmeasured. No new GPU allocator cap.

Sources/dataset/mesh/authorization/root read-only, non-root,no network, no image
build/install. Arm2 sees only arm1's completed parity report read-only. Each arm
has its own writable mount and exclusively created output root. Host launcher
refuses a running previous sphere container; cleanup only removes the currently
owned labeled container. Preserve partial data; stop batch on first failure,
no automatic retries, extensions, other arms, thresholds or setting changes.

The seven unresolved CPU reflectance quadrature checks remain unresolved. They
do not admit an energy-prior repair. This comparison keeps that prior unchanged
and asks only about surface smoothness; no claim of complete energy calibration.

Use `experiments/run_sphere_surface_comparison.sh` only after an authorization
binds the final commit/freeze and fresh batch. Codex does not invoke Docker/GPU
under preparation approval. After execution, analyze saved artifacts and report
all outcomes, including candidate failure or worse albedo, without retuning.
