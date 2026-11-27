# Bell realistic SAM2 acquisition — prospective angular ordering addendum

This preparation advances the original Gate 11 fitting-image-only protocol.
It does not authorize inference, checkpoint loading, installations or R0/R1.
Preserve the [original protocol](BELL_IMPERFECT_MASK_ACQUISITION_GATE11_V1.md),
its ascending-ID manifest, original image-only prompts and conservative trimap.

Numerical IDs are not a video trajectory. Audit only the122 fitting camera
calibration rows; cameras58/11/105/114/17/76 never enter the angular calculation.
Compute world camera centers as `-R.T@t`, then normalize relative to the fixed
acquisition coordinate origin. No object mesh, silhouette, material label, held
RGB, model response or segmentation outcome defines the path or origin.

Freeze a path starting at camera0 and ending at127 so the already recorded two
endpoint annotations and four fresh P1/P2 forward/reverse states remain usable.
Choose successive unused cameras by minimum great-circle angular distance,
breaking ties by camera ID. Apply deterministic first-improvement2-opt to reduce
total angular path length, with endpoints fixed, improvement threshold1e-12
radians and at most1000 swaps. Fail on noncompletion rather than selecting a
different path using inferred masks. Every fitting camera appears exactly once.
Forward/reverse passes use this same path in opposite traversal directions.

The [saved audit](../results/publication_gate12/sam2-ordering-v2.json) records both
orders' adjacent-angle distributions, complete frame/camera mapping, used rows,
calibration hash, endpoint prompts and exclusions. Geometric proximity reduces
abrupt viewpoint jumps but does not certify visual overlap, prompt quality or
calibrated confidence. Do not optimize ordering after viewing SAM2 results.

Keep both original P1/P2 prompt sets at their exact camera-local pixel coordinates.
No re-annotation is needed; no masks or GT are used to relocate prompts. Retain
all original raw-before-postprocessing score capture, decision-logit handling,
object-presence abstention, dual thresholds(+1/−1), four-pixel uncertainty margin,
four-pass corroboration and image-only background veto rules unchanged.

Before inference, prepare the actual fitting-only staging/observer implementation
and a separately reviewed acquisition freeze using this V2 ordering. Bind the
existing SAM2 runtime, model weights, exact sources, JPEG conversion and prompts,
and prospectively bound encoder/anchor/propagation calls, resource/time/output
limits. No hidden pilot or alternative prompt search. Runtime compatibility and
inference memory remain untested. Only after explicit inference approval and
image-only label admission may same-mask R0/R1 histories be authorized.

R0 and R1 must share the exact admitted imperfect foreground/background/uncertain
labels; R1 alone adds confident-background opacity at the original coefficient
schedule. Oracle masks are diagnostic evaluation only after labels are frozen,
not sources for prompts, order, thresholds, margins, loss support or tuning.
Raw SAM2 scores are not calibrated probabilities. Bell is development evidence;
reserve another dense scene or a prospective independent acquisition for confirmation.
