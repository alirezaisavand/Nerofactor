# E.13: first native three-arm material recovery, v1

Prepared scientific execution definition; **execution requires separate explicit
researcher approval bound to the final commit and freeze SHA-256**. This finalizes
[Bell's proposal](BELL_THREE_ARM_RECOVERY_AFTER_E12B_V1.md). Do not run the two-light
study, repeat E.12B, generate references, tune, resume, retry, or extend budgets.

## Question and immutable inputs

Compare numerical/estimator/core corrections, image reconstruction, and physical
recovery separately on the admitted **weak-cue** Bell v3 known-light oracle.
This is one scene/one seed per arm, not a population estimate or the final
informative paper benchmark. E.11D–G remain unchanged; no strong-identifiability
claim follows from this experiment.

Fixed order: A_original, A_estimator_v1, A_corrected_core_v1. Each uses its existing
`configs/experiments/material_a_bell_{original,estimator,core}_known_v1.yaml` and
its existing legacy YAML. Pinned material A: e1e859779301faced46d6476691e20b546c27ae0.
No frozen G/A, variant, YAML, or earlier evidence edits. The derivative source
identity is distinct from its pinned A ancestor. The separate energy-budget
material model is excluded.

Benchmark root is recorded in the freeze. Admission SHA-256:
`f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.
The freeze enumerates every consumed label, ray capture and reference with size
and SHA-256; the admission independently checks all512 dataset-file identities,
mesh and generation report. Verify at entry and exit. Original inputs are RO.
Only the newly generated, admitted calibration pickles are read by the original
loader; no historical checkpoints or camera pickles are loaded. Numeric evidence
uses allow_pickle=False. Labels define an oracle on this mesh, not independently
measured geometry. World units remain original mesh units, not verified meters.

Same122 ordered training views; validation58; held views11,105,114,17,76. The native
split must equal the admitted ordered arrays. Training never sees held-out truth.
One endpoint per arm, no truth/held-out-based checkpoint selection.

## Unchanged native training

Exactly2000 original optimizer returns at zero-based steps0–1999 per arm; at most
6000 across the batch. Original Trainer.run, model, losses, Adam, clipping and
scheduler execute. Observe statement boundaries without replacing operators.
Original seed6033,512 rays/update,512 diffuse/1024 specular samples, sRGB
Charbonnier RGB loss, Adam/AMSGrad eps1e−4,betas(.9,.999),weight decay0,clipping.7,
1000 warm-up steps and hard-coded100000 cosine endpoint remain unchanged.
Step0 LR0; step1 LR5e−7; step1000 LR.0005. No scheduler/gradient modifications.
Native ray-pool exhaustion/shuffling is preserved, including original RNG draws.

All arms use existing known_env_v1/zero_bounce_v1, no human light, reg_mat=false,
reg_diffuse_light=false. Original v3 environment: ambient[.08,.08,.08], +Z power12
RGB[1,.8,.6], +X power6 RGB[.3,.5,.9]. This is not L2. Shader regularizer declarations
remain distinct; the renderer's reg_mat gate disables them in every arm. Record
original YAML, effective Trainer/renderer/shader/loss/scheduler separately.

Legitimate differences: A preserves original estimator/half-angle/tangent/sampling
method bodies under known lighting; B changes the specular estimator while retaining
legacy sampling/frame behavior; C uses the existing corrected core. Different
variant signature buffers are required. No energy-budget variant or new fix.
Initial named-parameter records and ordered splits must match exactly across arms
before B/C updates; do not copy parameters to force agreement. Same seed does not
imply identical subsequent samples when algorithms consume RNG differently.
Record initial buffers, initial/endpoint RNG hashes and native runtime details.

First-two-update forward/backward, RGB, group-gradient, zero-LR displacement and
Adam checks execute inside each history using the E.12B observer. No separate
smoke or extra C updates. Later updates require finite loss and gradients, active
material/shared groups, expected inactive lighting groups, original LR/clip/cursor
and Adam counts. Record every loss/LR/native pre-clip norm and group norms; retain
full parameter/Adam/RNG summaries at100/500/1000/1500/2000. Saturated zero components
are observed, not repaired. Unexpected missing active gradients/all-zero aggregate,
nonfinite state, identity or resource failure stops the batch.

## Fixed endpoint and strict checkpoint

The external observer stops immediately after the2000th original optimizer return,
before the original final logging/validation/save/NVS dispatch. No scientific loop
body changes. The independent trajectory retains update2000 loss, which the native
TensorBoard logger has not yet written at the stop boundary. This is an explicit
endpoint-storage/evaluation protocol amendment to the earlier proposal's request
to retain natural final dispatch, avoiding duplicate renders and best-view selection.
Both the pinned validation RGB buffer and Trainer NVS keyword are valid; earlier
suspicions about those interfaces were rejected by complete source inspection.

Save exactly one endpoint-2000.pth with network, original optimizer state, step2000,
evaluation_step1999, source/config/variant/admission identities and best_para=null.
A null best metric means unmeasured, not zero or an invented historical score.
Hash and load this own trusted artifact on CPU, check all tensor keys/shapes/dtypes/
contents, then strictly load the same variant into the existing model and verify
parameter equality. Native variant guards remain active. Optimizer state is NOT
restored. This PyTorch1.8-compatible load uses pickle; only the current run's own
trusted endpoint is admitted inside the isolated runtime. No historical loads,
partial/cross-variant weights, extra model constructors or implicit resumes.
This checkpoint is an endpoint artifact, not an exact-resume claim (full ray-pool,
worker and RNG restoration is not implemented).

## Physical evaluation, frozen before execution

Export at all128 label archives' exact physical positions, cast to native float32,
with their original faceted shading normals. These are515698 pixel/view
observations, not unique surface points. Batch size4096; at most256 predictor and
256 pure tangent-basis calls per arm. These operations draw no RNG, perform no
BRDF integration and must preserve the training-endpoint RNG identity.

The independent CPU evaluator forms nominal principal axes from the predicted
double-angle atan2/2 and native unrotated tangent basis. Q_world=mx²uuᵀ+my²vvᵀ;
Q_local is its projection into the saved GT(T,B) basis. This nominal definition
does not repair the original renderer's actual half-angle frame. Export actual
rendering axes using the original decoder algebra for A/B and existing core
decoder for C, with no sampling calls, normalization, flips or replacement values.

Report per view and train/validation/held aggregates:

- Absolute Frobenius and relative Frobenius Q error, world and local separately.
  Relative denominator max(||Q_GT||F,1e−15).
- Major/minor widths sorted descending; absolute width errors (not squared widths).
- Major-axis orientation acos(|u_pred·u_GT|), degrees, modulo sign and axis exchange.
  Eligible only where saved orientation_defined and width gap≥.1. A predicted
  width gap≤1e−6 has undefined direction; retain it as an eligible angular failure,
  and report conditional angular statistics plus the missing count.
- Maximum absolute linear-RGB albedo error per observation and absolute metallic
  error. No gamma conversion of material coefficients.
- Actual-frame validity: max entrywise |FᵀF−I|≤1e−5 for F=(actual_T,actual_B,n_GT).
  Also require |det(F)−1|≤1e−5 (right-handed). Material widths must be(0,1], color/metallic[0,1], finite double-angle norm within
  1e−5 of1. No repair. Save invalid outcomes, report counts, then stop before
  remaining evaluation/arms if any material/frame is invalid in any arm.

Means are observation-weighted, p95 NumPy linear quantiles and maxima; split
aggregates never mix train and held. Orientation summaries state eligibility.
Nominal errors remain descriptive even for invalid frames; physical pass fractions
count invalid observations as failures in the original denominators.
Legacy descriptive targets: Q_relative≤.05 (world/local separately), eligible
orientation≤5°, max albedo error≤.02, metallic error≤.03. Report fraction passing
and numerator/denominator per criterion, not a post-hoc best-view aggregate.
These flat-material criteria are not validated object-level success guarantees.

## Bounded native source-light and relighting evaluation

Evaluate validation58 and held11,105,114,17,76 once at the endpoint for source
light and once for admitted unseen+90°Y environment:12 images per arm,128×128.
Reuse original test_step, camera selection, pixel centers, trace normal negation,
shader, sampling and test_ray_num1024. Explicit source IDs, not iterator indices.
Native shade evaluation uses step=None, exactly as original test_step; checkpoint
step2000/last training step1999 are separate metadata. No new camera/lighting.
Evaluation continues the native training-endpoint RNG without reseeding, in the
listed camera order, source light first then relight. Different variant draw
consumption may yield different evaluation samples despite the same initial seed.
Record both endpoint RNG identities; this single panel does not estimate MC
variance or justify strong conclusions from small photometric differences.
Relighting uses the existing shader.environment context (also used by the native
relight interface); only the known environment changes. No additional probes.

Capture actual returned compute_radiance(total,diffuse,specular) without querying
again or inverting sRGB. Native RGB must match captured-linear transfer within
2e−6; preserve the actual RGB for scoring. Native camera hit masks must exactly
match admitted captures; position≤2e−4, normals and ray directions≤2e−6. These
geometry bounds inherit the admitted tracing contracts; no per-result tuning.

Store float32 linear components, actual native sRGB, GT, masks and lossless PNGs.
Native-equivalent metrics use original color_map_backward uint8 truncation (not
rounding), original full-frame PSNR and SSIM(win11,channel_axis2,data_range255).
Source GT is the native hit-masked source image; relight GT is the independent
saved relight PNG. Also report independent fixed-GT-mask float-linear RMSE,
relative L2, diffuse/specular errors, float-sRGB RMSE, clipping counts. Full-frame
SSIM/PSNR include background; never substitute them for foreground recovery.
Pixel errors use the fixed admitted hit mask, checked against native geometry.

Report per-image metrics, equal-view mean PSNR/SSIM and pooled foreground relative
linear L2 separately for validation/held and each illumination. Held source-light
relative L2≤.01 is a descriptive target, not selection. Preserve all outcomes,
including poor reconstruction/physical recovery and diffuse/specular compensation.
Finite-sample predictions are not convergence studies; independent references
and their accepted numerical limits are reused without regeneration.

## Resources, guards and evidence

Corrected image sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331;
corrected backend SHA256800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5.
Same backend in A/B/C. GPU0/A10080GBPCIe,4 CPUs/threads,64GiB host RAM/no swap,
pids128, no new CUDA allocator cap (observe peak). Read-only root/source/oracle/
reference/admission/thesis mounts, non-root, no network or image build/install.
Each arm runs in a fresh process/container with only its own output writable;
B/C receive A's hash-bound initialization packet read-only.

Whole batch: foreground7200s watchdog,30s kill grace. Each Docker arm2100s+30s;
inner timeout2040s+20s. No extension/retry. Owned output≤4GiB/arm, checked every5s
and on finalization; redirected host console logs are outside that byte counter.
The launcher removes only CID/label-bound owned containers on exit. Failed/partial
runs are never reused. Existing outputs/evidence are not modified. Runtime is
unmeasured for this full panel; the E.12B smoke is not a throughput guarantee.

Upper work limits, reported separately:

|Work|Per arm|Panel maximum|
|---|---:|---:|
|Original training forwards/backwards/Adam returns|2000 each|6000 each|
|Training sample-direction slots|1572864000|4718592000|
|Constructor primary rays(122 views)|1998848|5996544|
|Evaluation primary rays(12 views)|196608|589824|
|Evaluation shader calls|≤192|≤576|
|Evaluation direction slots|≤301989888|≤905969664|
|Material predictor/pure basis batches|≤256 each|≤768 each|
|New own-endpoint serializations/strict loads|1 each|3 each|

Direction slots bound shader work, not exact native BVH queries. Some invalid
incident rays are not traced. No new reference integration, optimization arms,
checkpoint-selection queries, meshes, histories or relighting definitions.

Run phases, partial updates, optimizer returns, loss trajectories and owned-resource
cleanup are recorded incrementally. Complete requires all2000 returns, first-two
gates, strict checkpoint, full export,12 image evaluations, terminal source/input
identities and cleanup. Completion seal binds recovery.json; output hashes bind
all small/large owned results without adding raw weights to Git. A failure stops
remaining arms and records their unexecuted status through launch receipts.

## Execution after approval

The freeze records exact fresh batch/run IDs. Generate an external authorization
binding final commit, freeze, image, admission, GPU0, limits and run IDs only after
explicit approval. Researcher command (fill approved SOURCE/BATCH/AUTH literally):

```bash
# Inside an attached tmux terminal; keep the foreground supervisor alive.
E13_SOURCE=/home/isavand/paper/Nerofactor-3d2-e2a \
E13_BATCH=<approved-fresh-batch> E13_AUTH=<approved-authorization.json> \
timeout --foreground --signal=TERM --kill-after=30s 7200s \
bash /home/isavand/paper/Nerofactor-3d2-e2a/experiments/run_bell_material_recovery_e13.sh
```

No launch is authorized by this template. If Docker is restricted in Codex, use
this identical approved researcher-side command without permission bypass.
After execution, validate receipts/hashes/counters, compare all arms and preserve
failures. Loss/gradient trajectories and physical-versus-photometric endpoints
can motivate a later duration study;2000 updates do not establish convergence.
Do not launch that study, the separate two-light experiment or full paper training.
