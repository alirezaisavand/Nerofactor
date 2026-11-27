# Gate 11 — Prospective image-only SAM2 acquisition and same-mask R0/R1

Status: protocol/source-and-weight readiness only. No segmentation inference,
weight deserialization, geometry optimization or installation occurred. O1
authorization cannot authorize this acquisition or either R history.

## Runtime and historical limits

The existing `/home/isavand/anaconda3/envs/sam2/bin/python` is Python3.10.18 with
Torch2.6.0+cu124, torchvision0.21.0+cu124, NumPy2.1.2, Hydra1.3.2, iopath0.1.10
and Pillow11.0.0. It resolves SAM2 to `/home/isavand/sam2/sam2`. Source HEAD is
`2b90b9f5ceec907a1c18123530e92e794ad901a4`; the custom prompting script is untracked
and separately hashed. Config, model, predictor, builder, utility and extension
identities appear in [readiness metadata](../results/publication_gate11/sam2-readiness.json).

The existing SAM2.1 Hiera large model is898,083,611bytes, SHA-256
`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
The matching YAML and compiled `_C.so` are present. Package metadata meets the
local project's declared version requirements, so no installation is presently
justified. Model loading, extension compatibility, actual device memory and
end-to-end inference remain untested. Official builder uses CPU
`torch.load(...,weights_only=True)`; future approval must bind that one model.
Keep this environment separate from the accepted G Python3.8/Torch1.8 runtime.

Reuse the prior scoped historical search: original confidence maps were not
recovered. Current local `sam2.py` uses numerically sorted JPEGs, a scene-specific
interior positive prompt, forward propagation and exported logits>1 binary PNGs;
it saves no raw logits/prompt manifest. It is not a reusable authenticated Bell
acquisition recipe. Do not rerun it unchanged or infer confidence from old binary
PNGs. New segmentation would be a new acquisition, not recovery of old logits.

## Image-only inventory and prompts

The [prospective definition](../results/publication_gate11/realistic-mask-definition.json)
lists all122 fitting RGB images and hashes, ascending numerical camera IDs,
frame-to-camera mapping and two manually recorded prompt sets. Cameras58 and
11/105/114/17/76 are excluded. Prompts were placed by visual inspection of fitting
RGB0/127 alone, at their128×128 coordinates. No mask, depth, mesh, material or
held-view information was used for placement. This is image annotation, not an
automatic central-point assumption. Every positive/negative point and object ID1
must be retained exactly in the acquisition manifest; no prompt search is allowed.

Physically stage fitting RGBs and the prompt manifest only. Exclude oracle masks,
geometry, depths, physical labels and validation/held images. Convert each fitting
PNG RGB to a new JPEG at quality100/subsampling0 for the official JPEG video
loader; preserve original PNG and derived JPEG hashes, dimensions and image-ID
mapping. No alpha channel may become segmentation supervision. Camera order is
fixed before outcomes, not rearranged to improve propagation.

Create four fresh predictor states in this fixed order: P1-forward, P1-reverse,
P2-forward, P2-reverse. Each gets the two declared endpoint anchor annotations.
Forward starts at frame0; reverse starts at frame121. Reset state completely
between passes; never seed reverse propagation from a completed forward state.
Use the official non-VOS-optimized video predictor, default model postprocessing,
eval mode, seed6033 and float32 without an extra autocast/TF32 override. Preserve
effective builder overrides, including dynamic multimask stability, binarized
point-mask memory and fill_hole_area8, in the receipt.

## Preserve raw scores and distinguish decision scores

For every model frame call, capture the returned `track_step` low-resolution
`pred_masks` **before** predictor hole filling, along with object-presence logits.
A read-only return observer must copy outputs without changing them. Preserve
raw shape/dtype and an interpolation-derived original-resolution raw map using
the original bilinear/align_corners=False convention. Separately save postprocessed
low-resolution scores and original-resolution scores yielded by official video
propagation. The latter are the frozen **decision scores** used below; do not
mislabel hole-filled scores as raw decoder outputs. Record all anchor calls and
propagated outputs, including repeated frame visits; no silent last-write overwrite.

Save float32 numeric arrays with `allow_pickle=False` compatibility, object IDs,
presence/stability/quality information where supplied, effective postprocessing,
prompts, image order/hashes, model/source/runtime identities, seeds, and all calls.
Keep raw logits in logit units. Neither sigmoid logits nor object/mask-quality
scores are calibrated probabilities of geometric foreground/background.

## Frozen conservative three-state rule

Use the four official full-resolution decision maps `z0..z3` in the order above.
Coefficients are prospective fixed heuristics in logit/pixel units, not calibrated
probabilities or fitted thresholds:

```text
m_fg = z0 > +1                       # preserve P1-forward conservative foreground
plausible = any(z_i >= -1)
margin = Chebyshev dilation(plausible, 4 original-resolution pixels)
m_bg = all(z_i < -1) AND NOT margin AND NOT m_fg
if any object-presence logit <= 0: m_bg = 0 for that frame
uncertain = NOT (m_fg OR m_bg)
```

The pure rule is in `benchmarks/bell_mask_trimap_gate11.py`. Four-pixel margin is
fixed at this128² acquisition; a different resolution requires a new definition.
Foreground is never eroded or replaced by consensus. All uncertain/disagreeing
pixels remain unsupervised. Corroborated negative prompts plus four-pass consensus
and margin are conservative evidence, not a correctness guarantee: the passes
are correlated, and all can miss a genuine part. Object absence abstains rather
than declaring the whole image background. Save coverage/conflict/uncertainty
counts for all frames; never drop an inconvenient frame or fill uncertain regions.

Before admitting labels for R histories, review overlays against fitting RGB only.
A documented image-only veto can only add uncertainty to proposed background;
it cannot change foreground or tune thresholds. Freeze all vetoes and label bytes
before either R fit. If the acquisition fails (nonfinite scores, missing frame,
empty object foreground, wrong model or broken extension), preserve and stop for
review, without automatic prompts, fallback model, threshold changes or retry.

## Same-mask R0/R1 and future execution

R0 uses exactly `m_fg` and original foreground-only objective. R1 uses identical
RGBs, `m_fg`, initialization/seed, known cameras, architecture, all other losses,
optimizer, sampling policy and300k budget, adding only

```text
w_mask(s) * mean(m_bg * fg_acc)
```

to the exact original foreground L1 expression. Coefficient1 is frozen for this
first isolated pair; preserve full-ray denominator and original mask schedule.
The admitted `m_bg` is calculated and frozen for both histories, though R0 does
not optimize it. Do not improve segmentation for one arm or use GT/held data
for thresholds, margins, vetoes, loss support, checkpoint selection or tuning.
R1−R0 isolates the objective under a common imperfect acquisition. O1−O0 measures
the perfect-mask diagnostic; R1−O0 confounds acquisition and objective. A quantified
mask-quality interaction would require another prospective factorial, not a
post-hoc interpretation of these comparisons.

After label bytes and image-only admission are frozen, a separate GT-mask audit
may report foreground/background precision, recall and uncertainty as diagnostic
evaluation only. It must retain all mistakes and cannot admit/reject an acquisition,
drop frames, select a prompt pass, change thresholds/vetoes or trigger an automatic
retry. This prevents evaluation access from becoming an accuracy oracle for the
proposed realistic acquisition.

Before future execution, bind the model, all SAM2 sources, prompts, derived JPEG
recipe, score observer and trimap implementation to a dedicated acquisition
freeze. Prepare one bounded approval for four video passes/122 frames each, with
explicit encoder/anchor call counts, dtype, memory/output and time limits after
reviewing that implementation. The current readiness report is not that launch
approval. Then obtain separate dataset-label admission and a paired R0/R1 training
approval. No SAM2, R0 or R1 execution belongs to O1's authorization.

Keep fitting-only, confidence-aware direct free-space SDF as a separate candidate,
not a hidden part of R1. All Bell views are development evidence; reserve another
dense scene or independent prospective acquisition for confirmation after freezing
a resulting method.
