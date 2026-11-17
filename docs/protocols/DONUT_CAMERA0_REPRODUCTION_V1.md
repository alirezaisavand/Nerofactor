# Donut camera0 source-scene reproduction v1

Prepared for one separately authorized CPU rendering execution. No full dataset,
material training, BRDF edits, fitted exposure or automatic retry is permitted.
Accepted inspection:
`/home/isavand/paper/evidence/donut-blend-inspection-v2/DONUT_GROUND_TRUTH_REPORT.md`.
The research infrastructure base is a51ca60bebaed84faf5ae15ae14029eb689bb089.

## Source and exact scientific definition

Use the unchanged donut.blend (SHA-256
fcf8956740d6543782c190ec1c90e1bbef6f3ae36dcc9c3c0aa9b7fd7b742d57),
original oracle torus, Material.002, known HDR and finite point light. Do not
substitute the reconstructed mesh or halve coordinates. Use source frame1 and
BlenderNeRF Camera, verified equal to stored dataset camera0 /train/0001.png.
No pose/intrinsic fitting, resizing, cropping or registration.

Blender4.5.16 LTS, build3f6797307da5, the verified portable distribution. Exact
binary, relevant shared libraries, OpenImageIO and OCIO assets are hash-bound
in the freeze, together with the scene/HDR/calibration/original PNG/mask and
accepted inspection records. The exact historical rendering build is unknown.

Preserve saved Cycles CPU settings:1080²/100%,4096 maximum samples,adaptive
threshold.01,seed0,OpenImageDenoise HIGH with original inputs/prefilter,12 total
bounces,4 diffuse/glossy bounces,indirect clamp10,Blackman-Harris1.5px filter.
Keep the original Principled MULTI_GGX inputs, light parameters and world mapping.
Keep sRGB display,AgX/Look None,exposure0,gamma1,dither1 and opaque film. Runtime
thread count8 is a declared scheduling/resource override; no sampling change.

## One render and outputs

Exactly ONE bpy.ops.render.render(write_still=True) call, with a FINISHED return
required. The original RGB compositor path is Render Layers.Image → Reroute →
Composite. Preserve it. Redirect the original compositor depth File Output into
the owned run directory, retaining its computation/format; never write its old
Windows destination or save the source scene.

Add only an output capture from that SAME RGB Reroute to an EXR File Output:
32-bit RGB,float,ZIP lossless,Linear Rec.709,save_as_render=False. This captures
the denoised scene-linear RGB before AgX. It does not evaluate a second scene or
change material/lighting calculations. Main output remains original16-bit RGBA
PNG with the source display transform.

Then load the saved EXR and save one additional AgX PNG with the original scene
color settings. This is image encoding only, not a second rendering. Compare
that round-trip with the original reproduced PNG to check the linear export.
Do not switch AgX to Raw for the main image, manually fit a transfer curve or
replace returned radiance values.

Expected files:

- reproduced.png — original-style16-bit AgX RGBA.
- HDR/scene-linear-0001.exr — same-render linear RGB,32-bit float.
- source-depth/Image0001.png — redirected original depth branch.
- linear-to-AgX.png — image-only round-trip.
- comparison-inputs.npz — lossless uint16 originals/reproduction/round-trip,
  float32 EXR and aligned original/render masks,read with allow_pickle=False.
- render-receipt.json /execution-receipt.json /console.log, copied freeze/auth.
- comparison.json,RGB-errors.npz,preview PNGs andcamera0-comparison.png.

Physical settings are recorded before/after and must remain identical. All
original input/source hashes must still pass after rendering. Partial/failed
records are preserved; zero process exit alone is not a completed-render receipt.

## Frozen comparison definitions

Decode PNG samples explicitly as uint16 through Blender's bundled OpenImageIO.
Do not pass them through Pillow/imageio's8-bit RGBA conversion. Compare top-row-
first RGB arrays at original resolution; normalize by65535 only for metrics.

Report full-image, fixed original foreground, common foreground, and common
interior metrics. The original mask is fixed at>127. The same-render geometric
mask is redirected mapped depth<65535 (object depth is below15; background clamps
to1). Report mask IoU and original-only/render-only pixel counts. No favorable
mask replacement. Interior is common mask eroded by an11×11 window so SSIM's
11×11 window remains inside the common object. Full/fixed metrics remain visible
even if an interior score is better.

Metrics are display-RGB MSE,PSNR(data range1),SSIM(win11,channel_axis2),MAE,
signed RGB bias,absolute-error quantiles/max and fraction within2 uint16 LSB.
SSIM regional scores are means of the full SSIM map; boundary dependence is
reported separately through the common interior. Exact equality has a separate
flag and null PSNR rather than JSON Infinity. No score cutoff alone proves the
producing scene/build. Do not revise definitions after results.

EXR→AgX round-trip should differ by at most2/65535 per component, allowing
quantization/dither/float effects; report actual maximum and differing counts.
Failure is retained as an image-formation/export issue, not tuned away. Source
versus reproduced data are classified exact or non-exact and quantified; there
is no predeclared PSNR threshold that establishes historical attribution.

The four-panel sheet is original AgX RGB /reproduced AgX RGB /linear EXR through
fixed standard-sRGB preview (clip0…1,no AgX,no auto-exposure) /max-channel absolute
display-RGB error. Error color scale is fixed0…0.1. Full numerical error maps
and unclipped EXR remain available. Visual previews are8-bit display artifacts,
not the metric inputs.

## Attribution and next benchmark

Camera/hash/settings mismatches stop before rendering. If EXR round-trip fails,
do not interpret residual historical RGB error as a BRDF or camera failure.
Compare mask disagreement and interior/boundary errors to diagnose alignment;
do not fit a better camera. Remaining interior discrepancies can include the
historical build/device/denoiser, adaptive sampling and changed saved scene
state. One frame cannot uniquely separate these causes or establish all-view
provenance. No additional render or alternative setting is authorized.

The subsequent minimal Kurt control should reuse the oracle torus, camera panel
and exported world director, while explicitly specifying a new supported-range
Beckmann material and independently integrated linear targets. Original GGX
widths are not exact Beckmann labels (the narrow GGX width is below.04). Retain
both HDR and finite point light, use original world units or consistently convert
positions/radii/power. Keep the original Blender images as a model-mismatch
control; do not relabel them as Kurt GT. Begin with a few fitting/held-out views,
not100-frame regeneration, and evaluate physical Q/directors separately from RGB.

## Execution bounds and authorization

One fresh foreground supervised process; at most1 frame/4096 samples/1080².
Maximum primary sample slots4,777,574,400; adaptive sampling can stop earlier.
Up to12 path bounces plus original direct-light/shadow/denoising work; no claim
that the primary-slot count bounds every secondary ray. Exact secondary work is
not known beforehand.

CPU only,8 rendering threads and affinity restricted to at most8 allowed CPUs;
CUDA_VISIBLE_DEVICES empty,original CPU denoiser.16GiB monitored RSS,32GiB address
space,512MiB owned outputs. Foreground supervisor3600s,TERM then30s kill grace;
process-local CPU ceiling28800 CPU seconds. Poll resource use every2s. Use an
additional GNU timeout3660s/40s grace around the supervisor. Estimated7–40min,
unmeasured on this CPU; historical first-image metadata records about404s.

Source scene and all original assets are read only in the code path. Native
execution uses the normal Codex filesystem restrictions and requires no Docker,
root, network, package installation or permissions changes. Output/cache/config
paths are within the fresh run root. No full dataset or training. A one-time
researcher authorization binds final commit/freeze/output/limits; do not create
an affirmative authorization or execute before approval.
