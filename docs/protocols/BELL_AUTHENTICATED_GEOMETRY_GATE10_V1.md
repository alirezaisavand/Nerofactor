# Authenticated Bell geometry, publication Gate 10 v1

**PROCEED to exact execution review.** This protocol prepares one fresh geometry
run from authoritative G revision 79252178e65914ad1acbc14790b1514f276bfea0.
The publication checkpoint is prospectively fixed at **300,000 original Adam
returns**. Gate 10 preparation authorizes source review and focused CPU fixtures.
Execution requires separate approval of the final infrastructure commit, freeze
SHA-256, corrected image and fresh evidence path in the adjacent freeze.

## Acquisition and minimal adaptations

Use admitted E.11 Bell v3 at 128×128, 122 fitting views, validation camera 58,
and five held cameras 11/105/114/17/76 in the original frozen ordering. All source
RGB PNG and silhouette-mask bytes are preserved. Training sees 123 camera records,
123 PNGs, 123 masks and camera 58's depth PNG. The physical staging tree contains
no held-camera images, training depth, oracle mesh or material labels.

The adapter overrides acquisition dispatch in the fresh G process. It uses safe
float32 NPZ calibration rows with the original projection matrices, pixel centers,
scale and units. It replaces the stock first-camera split with the admitted
122/1 split and rejects unauthorized view/depth access. It preserves the original
split helper's two Python seed resets. No legacy calibration pickle is loaded.
The original non-NeRF RGB preprocessing divides the admitted sRGB PNG by 255.
Original learned appearance converts predictions to sRGB. The G training
objective therefore uses the original display-domain convention. Later linear
material fitting needs its own declared acquisition adapter.

Original Trainer.run, architecture, losses, optimizer, sampling, RNG seed 6033
and all coefficient schedules remain byte-identical. A subclass redirects owned
checkpoint/log paths and observes natural saves. A trace observes original loop
boundaries without invoking initialization or optimizer methods. These lifecycle
and storage adaptations do not change scientific calls. Validation uses only 58.
Its depth is accessed by original test_step for diagnostics, outside fitting.
The original validation-best file may be written and preserved as provenance.
Extraction always selects the fixed endpoint model.pth.

## Exact effective schedules

Here s is the original **zero-based** Trainer loop index. The checkpoint step is
s+1. Coefficients below multiply the original reductions, whose exact algebra
remains in network/loss.py and network/renderer.py.

| Component | Effective definition and activation |
| --- | --- |
| RGB | Charbonnier, mean of sqrt(sum channel squared residual + .001). Weight 1 at s=0, cosine rise to 5 at 5000, cosine fall to 1 at 10000, then 1 through 299999. |
| Eikonal | .1 × mean of (norm(SDF gradient)−1)² at inner samples. Both anneal bounds are zero, so .1 from the first update. |
| Initial SDF | Only s<1000. Small radius .1 and large radius 1.05 hinge constructions, with the source's exact reductions. Weight 3 through s<500, cosine decay to zero approaching 1000, absent thereafter. |
| Standard deviation | Recorded reciprocal inverse variance. apply_std_loss is false, so this recorder contributes zero to the objective. |
| Inverse-variance freezing | inv_s is detached for s<15000 and trainable from s=15000. Clamped to [1e−6,1e6]. |
| Occlusion | Renderer returns zero before s=20000. Thereafter L1 predicted versus SDF intersection opacity, eligible inner radius<.999, abs(SDF)<.01, front-facing gradient, at most 2048 samples. Weight .01 through 20000, cosine rise to 1 at 50000, then 1. Intersection settings sn0=64/sn1=16. |
| Mask | L1(mask, mask × accumulated inner opacity). Cosine weight .3 at zero to 1 at 30000, then 1. Background pixels supply zero to this particular mask expression. |
| Curvature | Random tangent displacement epsilon=1e−4, acos(clamped normalized-gradient dot)/pi. Both effective weight endpoints are .001, hence constant .001 at every step. The advertised 20k–50k weight ramp has zero amplitude. Renderer curvature_reduce_start/step fields are unused commented code. |
| Opacity | Binary cross entropy of clamped foreground accumulation with itself, differentiated through both arguments. Cosine weight .001 through 20000 to .01 at 50000, then .01. The similarly named renderer opacity_loss_weight does not multiply this output again. |
| Other computed RGB terms | loss_fg and loss_bg are produced by the renderer but fg/bg loss objects are absent from the selected Bell loss list, so they do not enter the training objective. |
| NeuS alpha cosine | min(1,s/50000), independently of the preceding loss weights. |

Adam uses native betas (.9,.999), epsilon 1e−8, weight decay 0, no AMSGrad and no
gradient clipping. Its constructor LR is .001. Before each update the native
WarmUpCosLR overrides it with .0005×s/5000 for s<5000, and
.0005×[.05+.95×(1+cos(pi×(s−5000)/295000))/2] thereafter.
The first return advances Adam moments at LR zero, the second has LR 1e−7.
The final training step is 299999, approaching LR .000025.
The full budget reaches every original activation and retains 250k updates after
the final 50k phase. No schedule is compressed.

The researcher's experience that reasonable geometry may emerge near 30k is
planning context. The original 300k configuration and archived historical Bell
training/validation logs also support a full-schedule reference. Endpoint choice
uses source/configuration and historical schedule evidence, without inspecting
new held-out outcomes.

## Lifecycle, intermediates and stop conditions

One approval covers sequential foreground training and endpoint extraction/audit.
At the natural step-1000 validation/save boundary, require finite losses, sampled
finite active gradients, expected first-two Adam advancement/displacement,
original data access, finite native validation and isolated CPU verification of
this run's own checkpoint schema, weight bytes and Adam layout. The gate concerns
execution and lifecycle. No silhouette or geometry accuracy threshold is applied
at step 1000, and no intermediate mesh is extracted.

Preserve full natural checkpoints at 1000, 15000, **30000**, 50000, 100000 and
200000, alongside the original endpoint and native best. Preserve all 300
validation logs/images and original every-20-step weighted training logs. These
provide evolution evidence without extra model queries or checkpoint selection.
The running checkpoint remains at its native overwrite path.

Stop on identity, unauthorized access, nonfinite loss/witness gradient, invalid
checkpoint, lifecycle, timeout, output or resource failure. Preserve partial
logs and checkpoints. No resume, retries, extensions, mesh repair, normal flip,
threshold revision, extra extraction, material training or external training
belongs to this launch.

## Extraction and geometry evaluation

After complete training and verified owned-resource cleanup, an isolated
evaluation container loads only the run's hash-bound step-300000 checkpoint on
CPU, strictly restores network tensors, and discards optimizer state. Training
never restores a checkpoint. These and the step-1000 diagnostic are the two
authorized pickle loads, limited to newly produced researcher-owned files.

Run original extract_geometry once, resolution 512, bounds [−1,1]³, level zero,
64-axis chunks, float32 field and original +1 sentinel at radius≥1. Preserve
original default Trimesh construction/export. Require PyMCubes .1.4 and the
Open3D triangle/coordinate order expected by original A. No remeshing, winding
repair or oracle-guided mesh selection is allowed.

On validation 58 and all five held views, independently reconstruct calibrated
rays and require agreement within 2e−6 with admitted rays. Replay oracle and
trace reconstructed geometry through admitted FP64-predicate BVH. Require exact
oracle hit-mask replay and accepted 2e−4 depth/position/normal, 2e−6 unit-normal
tolerances. Compare 16 deterministic foreground and 16 background rays per view
and mesh with independent FP64 closest-triangle references. Preserve shared-edge
ties in the existing witness convention.

Every raster retains oracle-only, reconstruction-only, common and shared-miss
counts. Common-surface correspondences require positive-facing A shading normals,
same-ray displacement≤.02 and signed normal angle≤30°. Also record the tighter
existing .005 displacement support and grazing cosine<.1. This is an explicit
same-ray visible-surface correspondence, with geometric separation and normal
tests. It does not assert a unique intrinsic material correspondence.

Prospective handoff requires each camera IoU≥.90 and eligible correspondence
coverage≥.80 of its oracle foreground. The .02 radius covers about five extraction
voxel spacings. The minimum coverage limits constrain the scope of a later paired
geometry study. They are proposed Gate 10 criteria subject to execution review.
Every excluded observation remains recorded, including overlapping failure masks.

Raw-SDF consistency uses up to 512 deterministic samples comprising 224
area-quantile triangle centroids, up to 64 additional centroids near the extraction
sphere, 128 evenly indexed vertices, and 16 reconstructed hits per camera.
Deduplicate selected faces. At each sample query raw SDF and its first input
gradient and two offsets ±1/511 along that gradient. Minimum gradient norm1e−6,
robust sign epsilon1e−7, abs(SDF)/gradient_norm≤1/511 and signed crossings are
required on interior eligible samples, with at least128 interior witnesses.
Oriented eligible samples must include at least32 samples and agree with A's
negated face normals within30°. Preserve all near-sphere samples and exclusions.
Near-sphere samples lacking raw-SDF crossing support fail the gate, which exposes
artificial sentinel caps instead of accepting them. Zero witness failures are
allowed. This finite sampling establishes a bounded check, with unsampled surface
behavior remaining uncertain.

The oracle has vertices slightly outside the original unit extraction sphere.
This is a documented mismatch between original G's extraction support and the
benchmark. The full run measures its effect. A gate failure preserves the original
scientific result and withholds downstream handoff.

## Execution envelope and evidence

The exact fresh output, input byte sizes/hashes, source hashes, splits, corrected
image/binary, policies and limits live in the adjacent JSON freeze. The single
launcher verifies clean committed source and exact authorization before staging.
It checks GPU0 availability and conflicting experiment processes.

Use existing corrected image sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331,
one A100 80GB GPU0, four CPU threads/cores, eight native data-loader workers,
128GiB host RAM/no swap, 256 PIDs, .75 CUDA allocator fraction and a cooperative
16GiB output cap per phase. Train watchdog129600s (36h), evaluation7200s (2h),
inner limits60s shorter, TERM/30s KILL grace. The first natural1000 boundary must
complete within3600s. Historical Bell event/log duration is about11h on an
unverified historical environment. Planning estimate for the new run is12–36h,
with no native timing measurement yet.

Maximum300000 training rays batches of512, 24,576,000,000 final ray-sample slots,
573,440,000 eligible occlusion-probe ray slots, 300 validations of1024 rays and
49,152,000 validation final slots. Original nested SDF importance, curvature,
background and occlusion/intersection queries remain within their source-fixed
per-ray branches. The budget does not equate final sample slots with total SDF
calls. Extraction evaluates exactly134,217,728 grid nodes in512 tiles. Audit traces
196608 primary BVH rays,384 CPU closest-hit witnesses and≤1536 SDF witness point
evaluations/≤512 first input-gradient points. Maximum4M extracted triangles.
No scientific output or authorization directory is allocated during preparation.

Run separate non-root, read-only-source/input/root, no-network containers
sequentially under the same foreground host supervisor. The training mount omits
all held assets and oracle geometry. Evaluation mounts its completed train outputs
read-only and receives the oracle archive only then. Retain console logs, actual
exit codes, container-absence receipts, source/input/checkpoint/mesh hashes,
effective configs, parameter initialization identity, activation-boundary training
witnesses, calibration parity, full raster arrays and numerical witness records.
Success is recorded only following normal cleanup. The offline verifier rechecks
identities and saved raster decisions and writes the unchanged G→A validator's
hash-bound handoff only if all frozen quality gates pass.

Subsequent researcher review remains necessary before material execution.
Authorization must be generated from the exact committed freeze using
benchmarks.bell_geometry_gate10_v1.authorization and stored outside Git.
After that explicit approval the foreground entry point is
experiments/run_bell_geometry_gate10_v1.sh with BELL_GEOMETRY_SOURCE and
BELL_GEOMETRY_AUTH set to their absolute approved paths.
