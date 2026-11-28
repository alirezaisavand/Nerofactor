# SAM2 readiness after Gate 12 — planning only

The accepted angular order and both endpoint prompt sets remain frozen. No
segmentation, weight deserialization, model import or R0/R1 optimization was
performed. Gate12's successful rendering does not validate imperfect masks.

The next inference preparation should implement the existing
[V2 addendum](../../protocols/BELL_SAM2_ANGULAR_ORDER_GATE12_V2.md) directly:

1. Stage only the122 fitting RGBs in the accepted frame/camera order, plus exact
   image-only prompt records. Convert with the existing quality100/subsampling0
   JPEG recipe; save source/derived hashes and dimensions. Exclude all oracle
   masks, labels, meshes, depth and validation/held images.
2. Reuse the existing separate SAM2 Python3.10/Torch2.6 runtime and pinned
   SAM2.1 Hiera-large source/config/weights. Actual extension loading and device
   memory remain untested. The next approval must explicitly cover only that
   known model's deserialization; Gate12 permits none.
3. Construct four fresh states in the accepted P1/P2 forward/reverse order.
   Retain two endpoint annotations per state:8 planned anchor API calls and
   488 unique propagation yields. Count decoder, image-encoder and memory-encoder
   primitives separately. `init_state` warms frame0, the predictor caches only
   its most recent image feature, and preflight encodes conditioning memory.
   Consequently488 yielded frames is not an encoder-work ceiling.
4. Capture `track_step` outputs before predictor hole filling: clone raw masks
   and object-presence logits, preserving their original shape/dtype and all
   repeated visits. Separately save the original postprocessed decision maps
   and original-resolution interpolation. Do not mistake the already hole-filled
   `_run_single_frame_inference` return for the raw decoder output.
5. Keep default reviewed postprocessing and all original foreground/background
   thresholds, four-pixel margin, consensus and abstention rules unchanged.
   Scores are not calibrated probabilities. Do not search prompts, orderings,
   thresholds or margins using masks, geometry or held-out results.

Before inference, freeze that focused staging/observer implementation with
source/model hashes, primitive work caps, memory/output/time limits and focused
CPU fixtures, then obtain separate exact execution approval. Preserve any failure
without automatic model/prompt fallback or retry. Image-only label review and
admission must precede the same-mask R0/R1 comparison; its common masks and
original foreground objective/schedule remain unchanged.

Confidence-aware free-space SDF remains a separate intervention. Bell is
development evidence; imperfect-mask robustness needs independent confirmation.
