# Gate 10.3 imperfect segmentation supervision clarification

Updated 2026-11-27. This is an interpretation and future-design amendment. The
existing Gate 10.3 evaluation protocol, implementation, input inventory and
Gate 10.2 evidence are preserved byte-for-byte. No geometry optimization,
checkpoint load, SDF query, native rendering or segmentation inference is part
of this amendment. The candidate hierarchy below supersedes the earlier claim
that a conventional binary background loss is necessarily the preferred method.

## Original scientific rationale

The researcher clarifies that the thesis intentionally introduced foreground
mask, curvature and opacity losses to mitigate geometry degeneration on
anisotropic objects under NeRO's approximate geometry-stage reflectance model.
Masks came from SAM2 treating ordered multi-view images as video, with interior
object prompts and a cautious threshold favoring foreground precision over recall.
This is researcher-supplied method context; it is not independently recovered
producing-run metadata for every historical scene.

The loss `L1(m, m*fg_acc)` supervises the labeled foreground only. Its zero direct
background derivative is consistent with that design: an omitted appendage or
thin feature must not become a definite free-space label. A conventional binary
silhouette target can erase correct geometry when masks undersegment. Curvature,
opacity, RGB and background-NeRF interactions remain relevant and are held fixed
in the proposed isolated supervision comparisons.

Gate 10.2's 2,807 extra silhouette hits and four failed IoU gates remain valid
observations against the defined synthetic oracle. They neither identify a bug
in the intentional foreground loss nor show that binary supervision would improve
geometry. Small foreground loss and high foreground correspondence do not certify
silhouette precision. See the preserved [Gate 10.2 report](../publication_gate102/RESULTS_V2.md).

## Exact authenticated Bell v3 mask provenance

The [machine-readable audit](mask-provenance-v1.json) binds the producing Gate 10
v4 training receipt, its freeze, the Bell v3 generating receipt and inspected
source hashes. Reproduce the read-only audit to a fresh output with:

```bash
python3 audits/bell_mask_provenance_gate103.py --output /tmp/bell-mask-audit-new.json
```

| Artifact or path | Verified semantics and use |
| --- | --- |
| Bell v3 `benchmark/datasets/GlossySynthetic/bell/masks_train/<id>.png` | 128 binary uint8 128×128 masks; each equals the saved oracle-mesh primary-ray `hit` array in C order, with `hit = depth < 10`. No SAM2 step. |
| Gate 10 v4 `training/bell/masks_train/<id>.png` | All 123 staged masks match the training freeze and generated PNG hashes. The receipt records 122 fitting mask reads and one validation mask read. No held-view mask is staged. |
| Pinned G `GlossySyntheticDatabase.get_mask` | Reads grayscale, divides by 255, thresholds at .5; the acquisition adapter changes address/split scope, retaining this mask reader. All binary values survive exactly. |
| Original `paper/data/GlossySynthetic/bell/masks_train/<id>.png` | 128 hash-verified 800×800 binary masks. The generator subsamples at `floor((i+.5)*800/128)`, thresholds at 127 and compares to independent mesh hits for camera/mesh admission. These pixels are not the exported v3 supervision masks. Their original producing algorithm is unresolved. |

The generator writes `255*hit` after geometry and radiance admission; the saved
v3 mask is not a resized SAM2 confidence map, RGB threshold, or depth-PNG
reconstruction. The audit compares all 128 ray arrays and masks, and every staged
mask, without retracing rays. Original camera-admission masks are near but not
identical to ray-hit masks; for example, generation camera 0 recorded IoU
0.978196684079037. That comparison does not establish their historical producer.

Bell v3 therefore gives a perfect discrete reference-mask control relative to
its specified oracle mesh, cameras and raster. It is not independently measured
physical truth, continuous antialiased coverage, or an example of SAM2 robustness.
Synthetic mask availability must be disclosed when interpreting the original
training. An oracle mesh, its depth and held views remain excluded from realistic
method supervision.

## Recovered SAM2 procedure and confidence availability

The inspected `/home/isavand/sam2/sam2.py` is an untracked local script, currently
configured for `nerf_synthetic/icosphere/train/jpg_converted`. Its content hash
is recorded in the audit. SAM2 checkout HEAD is
`2b90b9f5ceec907a1c18123530e92e794ad901a4`; that tracked revision does not version
the custom script or establish the original runtime revision.

| Procedure | Current code evidence | Historical limit |
| --- | --- | --- |
| Model | Declares SAM2.1 Hiera large checkpoint and matching YAML, CUDA video predictor | Declared filenames do not prove the weight bytes used in old runs |
| Image order | Numerically sorts JPEG stems and initializes a video state | Exact original per-scene ordering needs its own manifest |
| Prompt | Object 1, frame 0, positive point `(950,570)`; no active negative point | Current scene-specific coordinates are not Bell prompts |
| Propagation | Calls `propagate_in_video` with default forward direction | No executed reverse-consensus run is recorded |
| Threshold | Prompt preview uses `logit > 0`; exported propagation mask uses `logit > 1` | Commented `>9` block is an alternative, not evidence of execution |
| Export | Saves binary 0/255 PNGs and a visualization; no logits, state, per-mask quality or prompt manifest is serialized | Many score fields existed transiently but are unrecoverable from a binary PNG |

`click.py` collects mouse coordinates and prints them; it does not save a prompt
manifest. `convert_jpg.py` converts PNG RGB to JPEG and skips depth/back images;
it supplies no confidence archive. The video predictor retains `pred_masks` and
object score logits in memory and yields full-resolution mask scores. Its builder
also enables point-mask binarization for memory encoding and hole filling up to
eight low-resolution pixels. These details matter for a future reproducible
segmentation recipe; they are not evidence that old outputs preserve raw scores.
The script's distance-transform weighting helpers are defined but never called
in the active export, and would be spatial heuristics rather than SAM2 confidence.

A compiled artifact `sam_codes/__pycache__/sam2.cpython-310.pyc` remains without
its source. Read-only strings reveal a historical teapot input path and a
`masks_smoothcylinder` output label. It was not imported or executed, and does not
authenticate run settings or provide score maps.

The saved [search inventory](sam2-artifact-search-v1.json) covers SAM2, sam_codes,
nero/data, paper/data, nerofactor_real, seg_masks and projected_masks. The masks
directories contain PNGs; no candidate score/logit/state archive was found by the
recorded filename/extension search. All 69 sampled PNGs (first/middle/last in each
selected mask directory) contain only 0/255 values; this is not an all-file scan
of PNG contents. The old `nerofactor/data` path is absent.
This is a scoped search, not a claim that no external archive exists. **Original
confidence information has not been recovered.** Re-running segmentation would
create a new, explicitly versioned acquisition, not recover original logits.

Pixel mask logits, image-level mask quality predictions and object-presence
scores are different quantities. None is a calibrated pixelwise probability of
geometric foreground or background. Applying a sigmoid does not establish
calibration. Very negative logits may still correspond to a missed object part
under positive-only prompting.

## Candidate hierarchy and three-state objective

1. Preserve the existing foreground-only loss as the intentional baseline.
2. Use a conventional binary loss with perfect synthetic masks only as an isolated
   diagnostic control. The existing proposed additive L1 background term is one
   such control, not a claim that BCE and L1 are interchangeable.
3. Consider confidence-aware trimap supervision from imperfect masks as a realistic
   opacity intervention.
4. Consider multi-view confidence-aware free-space SDF supervision from imperfect
   fitting-camera observations as a realistic direct-geometry intervention.

For physical accumulation `a` in [0,1], the requested formulation is

```text
mean(m_fg * (1 - a) + lambda_bg * m_bg * a)
```

The two masks are disjoint. Confident foreground has `(m_fg,m_bg)=(1,0)`, confident
background `(0,1)`, and uncertainty `(0,0)`. Do not set `m_bg = 1-m_fg` for an
imperfect mask. With N original sampled raster rays, derivatives in the interior
are `-1/N`, `lambda_bg/N` and zero respectively. These derivatives explain target
semantics, not measured gradients of a trained model.

For an isolated background addition, use the original accepted foreground mask
`m_orig` in both arms, with the actual source expression retained exactly:

```text
w_mask(s) * mean(abs(m_orig - m_orig*a) + lambda_bg*m_bg*a)
```

Keep `m_fg=m_orig`, the original full-ray denominator, L1 behavior/subgradient at
one, rounding behavior outside [0,1], and the original cosine mask schedule
(.3→1 over s=0..30000, then one). Use the same schedule on the added opacity term.
Fix curvature, entropy opacity, RGB, lighting, sampling, initialization and all
other schedules. `lambda_bg=0` recovers the original term. `lambda_bg=1` is a
transparent symmetric perfect-mask diagnostic coefficient, not a tuned optimum
for imperfect supervision. Do not clip accumulation or separately normalize the
foreground and background to change their relative influence.

Prospective trimap construction should retain raw logits and record exact image
hashes, resolution, camera mapping, prompts, checkpoint hash, postprocessing and
propagation direction. Use dual score thresholds `t_bg < t_fg`, with a dead zone;
combine a low-score requirement with corroboration for background. Keep pixels
near any plausible silhouette or prompt disagreement uncertain. Morphological
uncertainty margins can exclude a fixed dilation around the union of plausible
foreground, rather than turning every excluded pixel into background. An erosion
of the original foreground would change the foreground objective and belongs in
a separate, explicitly matched mask-acquisition comparison.

Consensus across independently chosen prompts and forward/reverse propagation
can reduce some failures, but those predictions are correlated and agreement
cannot establish correctness. Require conservative negative evidence or manual
image-only checks for background; retain missed parts, thin features and holes
as uncertain when unsupported. If only binary masks remain, morphology alone
cannot certify their exterior as background: keep the baseline, or acquire new
scores under a prospective recipe. Freeze thresholds/margins and prompt rules
using fitting images alone, never oracle silhouette/depth, held views or geometry
recovery scores. Any later GT mask error audit is diagnostic and cannot feed back
into those choices.

## Prospective multi-view free-space SDF constraint

Use fitting cameras only, with the same imperfect-mask evidence as the opacity
candidate. For a point x in the supported learned domain, project it into valid,
in-frame positive-depth camera observations. Build a conservative confidence
weight from confident background, boundary clearance and angularly separated
camera agreement. Unknown, out-of-frame, behind-camera and conflicting evidence
abstains. Foreground projection alone never establishes that x is occupied: x
may lie in front of the visible surface.

A candidate rule requires at least two suitably separated fitting cameras with
corroborated background at the projection; all other observations remain unknown.
A confident-foreground conflict may conservatively veto a constraint, but must
not create an occupancy label. This abstention trades free-space coverage for
precision. View count and angular coverage are geometric evidence checks, not
independent probability trials. Camera count, baseline, margins and support
thresholds need prospective definitions; none is selected from Gate 10.3 errors.

With pinned G's outside-positive SDF convention, a possible soft objective is

```text
L_fs = mean_x(q(x) * huber(relu(delta - sdf(x))))
L_total = L_original + w_fs(s) * lambda_fs * L_fs
```

Here `q=0` for uncertainty and is bounded for admitted free space. The expectation
and its denominator must be declared, with sampling weights if sampling is
nonuniform. `delta >= 0` is an exterior margin in scene units; a positive margin
must respect the image-boundary and calibration uncertainty. A Huber penalty
limits influence from contradictory labels without imposing a hard sign edit.
SDF distance scale and sign must be checked before implementation; `lambda_fs`
has different units from opacity's coefficient and cannot be copied blindly.
Specify its scale from fitting-only synthetic/gradient checks before optimization.
Keep the original foreground term and schedule intact; predeclare the additional
SDF schedule and point-query budget separately.

Sample only supported in-domain volumes or fitting-camera ray segments. A
confident background ray asserts free space within that modeled interval, not
an oracle depth, a complete visual hull or valid geometry outside the learned
domain. Never use held cameras, oracle mesh intersections, GT depths, diagnostic
error maps or model-dependent favorable selections to create the constraint.
Do not initialize, clip, trim or carve the SDF from a visual hull. Retain conflicts
and report coverage, vetoes and uncertainty. A soft loss still inherits label
errors, so multi-view agreement and abstention are essential design choices.

Direct SDF supervision can act on erroneous zero-level geometry that contributes
little rendered opacity; opacity supervision more directly targets the quantity
already observed on a background ray. Neither guarantees correct normals,
watertightness, unseen surfaces or physically valid reflectance.

## How the unchanged Gate 10.3 observations guide the choice

There are no new Gate 10.3 native results in this amendment. Apply the following
interpretation after the already prepared study, without changing its evaluation.

| Frozen observation | Motivation for future intervention | Remaining uncertainty |
| --- | --- | --- |
| Fitting mesh false positives and stable high background `fg_acc` across both draws | Confidence-aware background-opacity addition is the lower-cost first realistic candidate | No causal evidence that it will fix zero-level geometry; imperfect background labels still need admission |
| Mesh false positives while background opacity is low on the same sampled rays | A volume/zero-level mismatch motivates direct fitting-camera free-space SDF supervision | Limited panel and NeuS sampling can hide occupancy; no new sign query is authorized here |
| Strong draw-dependent opacity flips | Resolve sampling sensitivity before assigning a stable mechanism | Two draws do not estimate converged uncertainty |
| Good fitting geometry but poor validation/held-view geometry | Investigate generalization and camera coverage | Binary fitting masks alone need not improve generalization |
| Predominantly domain-boundary/open-surface errors | Keep representation support and free-space supervision questions separate | A mask term cannot establish an unsupported closure |

Retain the fixed 15k/30k/50k/300k checkpoints and 300k publication endpoint, all
122 fitting mesh views, the fixed 16-camera two-draw opacity panel, original
thresholds and exclusion counts. Temporal overlap with loss phases is descriptive.
Gate 10.2 already shows excess support; it does not supply the missing fitting
opacity evidence needed to choose between these mechanisms. Neither candidate
is a demonstrated improvement until a matched future optimization comparison.

## Cost-effective comparison and mask accuracy attribution

| Arm | Supervision acquisition | Objective change | Valid comparison |
| --- | --- | --- | --- |
| O0 | Existing synthetic oracle masks | Existing foreground-only loss | Reuse authenticated 300k history; no rerun |
| O1 | Same oracle masks | Add conventional binary background opacity only | O1−O0 diagnoses the objective with perfect masks; oracle-assisted control only |
| R0 | One prospectively frozen imperfect SAM2 acquisition | Existing foreground-only loss | Required same-mask baseline for realistic method claims |
| R1 | Exactly R0 foreground and same fixed confident-background evidence | Add trimap background opacity only | R1−R0 isolates objective improvement under imperfect masks |
| R2 | Exactly R0 foreground and the same image evidence, fused across fitting views | Add soft free-space SDF only | R2−R0 tests direct geometry supervision; R2−R1 is a method comparison unless confidence support is also matched |

First reuse completed audits and finish Gate 10.3 without new optimization. Then
freeze an image-only imperfect acquisition and a paired R0 plus one realistic
candidate chosen by the mechanism table. This needs two fresh histories, not a
grid search or a rerun of O0. O1 is an optional third diagnostic history. Test the
other realistic mechanism only if a prospectively declared follow-up justifies
the cost; do not present post hoc best-arm selection as confirmation. A full
mechanism comparison adds the fourth new history. All are future proposals.

For strict attribution, use fresh identical initialization/seed6033, fitting
cameras, RGB, original schedules and 300k updates, with zero changes to appearance,
curvature, entropy opacity or background NeRF. Record extra SDF queries and
segmentation cost separately; equal updates need not mean equal compute. The
completed original run took 39,469 seconds (about 11 hours). Two/three comparable
histories suggest roughly 22/33 GPU-hours before added method/evaluation cost;
these are planning estimates, not measured runtimes of the candidates. No launch,
runtime extension or new execution approval is implied.

To attribute a difference specifically to opacity versus SDF, freeze a common
multi-view-supported background-ray/segment inventory, confidence weights and
uncertainty vetoes for both R1 and R2 before either fit. Only the loss target and
declared SDF point sampling may differ. If R2 alone adds multi-view confidence
fusion, report the comparison as a combined-method change; do not attribute its
gain solely to direct SDF supervision. This control prevents better confidence
selection from being mistaken for a geometry-objective improvement.

Comparing R1 directly with O0 changes both masks and objective and cannot isolate
either. O0−R0 measures sensitivity of the same original objective to acquisition,
not a method gain. A later objective-by-mask factorial is needed for a quantified
interaction; the small initial comparison does not establish that interaction.
If comparing different imperfect-mask qualities, keep each mask acquisition
fixed across its paired objectives rather than improve masks for one arm only.

Oracle masks are permitted as labeled O controls and evaluation truth only. For
R arms, exclude them from prompts, threshold selection, uncertainty margins,
multi-view voting, SDF sampling, checkpoints and tuning. Bell validation/held views
are already development evidence. Freeze the resulting method before independent
confirmation on another dense scene or a prospectively reserved acquisition
outside these 128 views. That confirmation remains separate from Gate 10.3.
