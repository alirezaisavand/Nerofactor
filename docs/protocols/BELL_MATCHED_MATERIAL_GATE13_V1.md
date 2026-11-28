# Gate 13: matched material recovery on authenticated Bell geometry

Status: **PROCEED to researcher review; preparation only.** This protocol does
not authorize a model construction, checkpoint load, BRDF integration, ray query,
optimizer return, relighting render or segmentation inference.

## Question and provenance

Compare two fresh K1-style histories on original weak-cue Bell v3, changing the
immutable geometry only. This is a new acquisition-specific pair, not a rerun of
heterogeneous Bell K1-Dense. K1 remains our developed reconstruction recipe.
O1 is our authenticated **perfect-synthetic-mask-assisted background-opacity G
derivative**, not unchanged original G and not a realistic SAM2-trained result.
O0 remains unadmitted and is not a training arm.

The derivative handoff is the accepted
[O1 record](../results/publication_gate12/O1-controlled-handoff.json), bound by
SHA-256 `46ae8c7bb6a23956185afb1a2c6b97e5b769702946baa7005500952a1a9e8e14`.
O1 mesh SHA-256 is
`0becad75d5f5404b84f5a154870eaf66e71f00ba5b843776f50dd6b1989d7ac3`;
its producing 300k checkpoint hash is
`b9a5fb4c461de77ec9f387466e05373f959df65baebd5151f02278d1648c46e7`.
That checkpoint is provenance only: it will not be deserialized or queried.

Original Bell v3 generation SHA-256 is
`5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7`;
researcher admission SHA-256 is
`f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.
[Input inventory](../results/publication_gate13/input-inventory.json) records
544 accepted file identities, calibration, mesh, all 128 saved ray/label/RGB
archives, O1 hits/correspondences, Gate 12 compatibility/visibility captures,
initial-state JSON, and admission/verification records. All data remain read-only.

## Frozen observation pool

Use all original 122 fitting IDs in their original order, validation 58, and held
views 11/105/114/17/76. Never move validation or held views into training.
Resolution remains 128². Reuse saved calibrated rays and Gate 11 O1 closest hits.
Do not retrace primaries to find a more favorable pool.

Replay the already accepted same-ray correspondence: both hits positive-facing,
Euclidean displacement at most .02, signed normal angle at most 30 degrees.
Additionally require strictly positive outgoing dot products for the actual
archive-label oracle normal and actual O1 shading normal. This removes zero
additional observations. No material parameter, RGB residual, reference energy,
highlight or held-out error participates in selection.

This is explicitly an **oracle-assisted correspondence restriction**. It is not
an unassisted end-to-end reconstruction claim. The common fitting pool contains
483,062 observations; all fitting/validation/held evaluation pools total 506,669.
Across the 128 rasters there are 515,698 oracle hits and 515,318 O1 hits; preserve
718 missing, 338 excess and 8,649 O1-hit observations outside the admitted common
pool. Reasons overlap and their masks must not be summed as disjoint counts.

Training sees only position, normal, outgoing direction and original oracle
linear RGB. **Never replace targets by Gate 12 reconstructed-geometry RGB.**
Analytic material arrays and oracle-source transported labels are evaluation-only.
Coverage is frozen before native work. Evaluation export zeros outside the common
pool are storage placeholders, not material predictions. Report known-support
scores and undefined-support counts; do not manufacture a complete-raster score.

## Paired scientific recipe

Order: **K1-Oracle**, then **K1-O1**, fresh seed 6033 weights in both. Reuse pinned
A `e1e859779301faced46d6476691e20b546c27ae0`, unchanged
`A_corrected_core_v1`, full 1,485,584 trainable material parameters, chart width
and orientation heads, corrected BVH `predicate_fp64`. Do not alter frozen sources.
The accepted K1 recipe is bound to
`BELL_DENSE_KNOWN_L2_RECOVERY_V2.freeze.json` hash
`1c7887d33f8f8ba3fb28cc48d795636ee2955eedbeebc7ead56bdef26736faf7`.
Reuse its Gate 4 acquisition overlay; replace only scene-specific data,
illumination identity and saved-map inventory. Remove old off-image diagnostic
queries, not any training objective. The new source overlay has exact-site checks
and its generated Python SHA-256 is frozen.

The scene illumination is **original Bell v3**, not L2: ambient RGB .08; a +Z
power-12 lobe with RGB [1,.8,.6] and +X power-6 lobe with RGB [.3,.5,.9]. Keep
linear radiometry, exposure 1, zero indirect, no human lights. The native corrected
known-environment path is used for compatibility, training, every evaluation and
illumination exports. A private Bell-v3 version of the existing light observer
retains all learned-module/optimizer guards without modifying the L2 observer.

Both histories retain original K1 linear Charbonnier RGB (epsilon .001),
`legacy_v1` scalar/chart-neighborhood, energy and specular priors, and diffuse-white
coefficient .01. No surface-Q or new prior. Known light makes diffuse-white a
reported light-only constant with zero material derivative; verify that property
and keep its scalar contribution. Preserve additive Kurt energy limitations.

Use original Adam/AMSGrad: initial LR .001, eps 1e-4, betas .9/.999, no decay,
gradient clip .7. Reuse original WarmUpCosLR, warmup 1k, scheduled LR .0005 and
100k cosine horizon; **do not compress it to the 10k endpoint**. Exactly 10,000
returns per arm. First LR zero, second nonzero displacement; check original Adam
moments and finite gradients within these histories, without a separate smoke.

Use 512 fitting rays, 512 diffuse and 1,024 specular samples. Retain original
diffuse random-azimuth and Gaussian neighborhood draws, including their order.
The full original material/light initial state must match historical JSON anchor
SHA `0407da2d23f545004a5505e475d497ba2c41760988048d2ff09654e18b5ab6a9`
and each other. This is a JSON fingerprint, not loading a historical checkpoint.

Both use the exact frozen NumPy shuffle/drop-partial-batch index stream. Preserve
every original random request; do not reseed per training update. Log full CUDA
RNG-state digests and request shapes/dtypes/device before and after every draw;
hash actual sampled values at updates 1/2000/4000/6000/8000/10000. O1 compares
each ledger row to the immutable completed Oracle row before its pending optimizer
step. Any mismatch stops the pair. Geometry-dependent visibility may differ; the
underlying proposal tapes may not. Evaluations restore RNG states as before.
There are 5,120,000 training exposures, approximately 10.599 passes over the pool.

Identical weights do not imply identical Q/RGB at different positions/normals.
At step 0 quantify scalar, sorted-width, transported-Q, orientation and RGB
differences without matching them by oracle supervision. The geometry and
corrected shader remain the intended experimental intervention.

## Native admission, checkpoints and outputs

Before any update, require source/freeze/data/initialization/image checks and the
accepted 32-draw native/reference gate at 24 fixed camera-58 common observations,
using the corresponding Oracle/O1 Gate 12 256²/512² independent captures. Preserve
the accepted .001 absolute/.02 relative plus six-SE compatibility definition.
The underlying reference captures passed .003 absolute/.002 relative integration
convergence. There is no new reference integration in Gate 13. Fail without retry.

Save fixed all-128-camera native maps at 0/2k/4k/6k/8k/10k: eight matched draws,
GT/reconstruction, base color, metallic, widths, raw and world directors/Q,
diffuse/specular, numerical arrays, learned-module-frozen illumination maps and
linear errors. Keep raw eight-draw components as well as mean/SE; capturing them
does not add shader work. Display-only clipping is documented; no metric clipping.
Save own checkpoints at 2k/4k/6k/8k/10k, CPU roundtrip with strict state identity,
no optimizer restoration. Separate approval must expressly cover these ten owned
pickle loads. No historical pickle load is needed.

Keep every training loss, active group gradient norm, energy/specular/white
component-gradient interaction at the six existing diagnostic boundaries, Adam
checks, output hashes, cleanup and timing. Log sampling-observer and draw-export
host time separately; synchronization makes these observed costs, not isolated
GPU overhead. Do not change the scientific sampling to reduce bookkeeping cost.

## Evaluation and interpretation

Primary comparison is fixed step 10,000. Earlier states explain trajectories,
not checkpoint selection. Report fitting, validation and held views separately,
both individual per-camera distributions and observation-pooled split summaries.
These are image-sample-weighted, not uniform surface-area estimates.

Receiving truth is the unchanged analytic Bell field evaluated at each actual
position and shading normal. Report Q Frobenius relative error, sorted-width MAE,
relative errors and signed biases, log mean/ratio, base-color and metallic errors,
Q<=.05 counts and tails. Preserve K1 orientation eligibility: normalized squared
width gap at least .1 in prediction and truth; compute director error modulo pi.
Use the intersection of the two arms' eligible observations for paired angles;
report individual eligibility too. Source archive's absolute width-gap mask is
not silently substituted for this K1 definition. Isotropic Q stays valid even
where orientation is undefined.

Separately compare predictions transported by minimal signed-normal rotation to
the same-ray source oracle material. Transport receiving truth identically and
verify primary-error invariance. Report coordinate-only true-Q shifts and changes
from normal projection versus parallel transport; these are not learned errors.
Receiving and source scalar/width errors are distinct targets. No normal flipping
or arbitrary orientation fallback is allowed for predictions.

Report linear RGB RMSE/relative L2, diffuse/specular errors, raw arrays and paired
eight-draw uncertainty. Stochastic SE does not bound deterministic diffuse bias
or absolute estimator bias; compare with accepted compatibility uncertainty.
Keep all full-raster exclusions, excess and missing support. Cohorts: normal
error >15 degrees, grazing min cosine <.1, displacement >.01, near open-boundary
midpoints within .02, and weak receiving-field anisotropy. Incident-visibility
disagreement is available only on six accepted Gate 12 cameras; do not invent a
122-view incident-visibility diagnostic. Retain continuous associations and raw
paired arrays; no cohort enters optimization or selection.

Compare original-light RGB improvements with physical-error deterioration.
Gate 12 fixed-true-material discrepancy is neither an attainable nor irreducible
optimization bound. Original v3's weak anisotropy cues can produce ambiguity even
on oracle geometry. Associations with normals, position or visibility are not
individual causal effects, and their interacting residuals are not additive.

## Exact proposed execution envelope

Fresh output:
`/home/isavand/paper/evidence/bell-material-gate13-f1a471f430a443f4927cbdc5fa2dd430`.
Image `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`.
One A100 GPU 0, four CPU threads, 64 GiB host RAM/no swap, 128 PIDs, .25 CUDA
allocator fraction, 8 GiB cooperative owned-output bound per arm. Inputs/source
and root read-only, non-root, no network/build/install. The completed Oracle arm
is additionally mounted read-only while O1 runs. Visibility is recomputed on
each immutable geometry for the native incident directions; no new primary rays,
mesh changes, SDF queries, G loads or geometry training.

Per arm: **60,960 shader calls; 45,221,191,680 shadow-ray slots; 30 component
reverse sweeps; 10,000 Adam returns**. Pair maxima: 121,920 shader calls,
90,442,383,360 shadow slots, 60 component reverse sweeps, 20,000 Adam returns.
Eight-draw all-view evaluation is included (1,061 chunks per complete view panel).
No independent BRDF point-directions or new reference images are authorized here.
The v3 training wrappers directly use saved hits, rather than the original
Trainer's camera preprocessing; this is the accepted controlled-recovery adapter.

Per-arm host foreground watchdog 9,000 s +30 s kill grace, inner 8,940 s +20 s;
whole foreground watchdog 18,600 s +30 s, including sequential arms and file-only
analysis. Estimated 1–5 hours, unmeasured on this 128-view package; the caps, not
this estimate, govern. Preserve partials on timeout. Do not retry/resume/extend,
change coefficient/gates, generate references, launch SAM or relighting.

After exact commit/freeze/output/image/owned-load approval, the foreground workflow
is a tmux session invoking `timeout --foreground --signal=TERM --kill-after=30s
18600s env BELL_MATERIAL_SOURCE=... BELL_MATERIAL_AUTH=... bash
experiments/run_bell_material_geometry_gate13.sh`. Do not detach scientific work
from its watchdog. Authorization JSON is generated from `m.authorization` only
after explicit researcher approval; preparation does not write an approval file.
The launcher verifies the image, native gates and each complete receipt before
advancing. Final analysis uses saved files only and never reopens checkpoints.

## Decision rules

Execution/numerical failure: preserve evidence and stop, **BLOCKED** pending the
specific fix and new authorization. Completed comparison: report degradation,
tradeoffs or equivalence without requiring a favorable result. Advancing to
relighting requires review of numerical reliability and material outputs, not
passing an invented recovery threshold. Geometry admission is not reinterpreted.
The [prospective relighting comparison](BELL_GATE13_RELIGHTING_PROSPECTIVE.md)
and [limited SAM readiness](../results/publication_gate13/SAM_READINESS.md) are
separate future execution scopes.
