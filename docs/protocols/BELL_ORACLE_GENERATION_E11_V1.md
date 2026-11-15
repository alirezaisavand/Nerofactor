# E.11 v1 — first Bell oracle dataset generation

**Frozen preparation; real generation and native training await separate approvals.**
This implements the Bell portion of the [E.10 proposal](PAPER_ANISOTROPIC_BENCHMARK_E10_V1.md),
without rewriting it. Accepted base is `fd53e6eb29cc7a7764795943e632e1535f2d1296`.
The accompanying `.freeze.json` pins generation code, tests, inputs, E.10 configurations,
protocols and scientific manifest. The authorization binds its SHA256 and the final
committed source, not an uncommitted implementation or an old freeze.

## Oracle, calibration and geometry admission

Use exactly Bell mesh SHA256
`4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01`:
124456 vertices,248908 triangles, unchanged coordinates and inward serialized
winding (signed volume−.8547482). It defines a **new oracle surface**, not measured
GT geometry, an attributable G training result or the earlier diagnostic r128 mesh.
Frozen G/A remain `79252178e65914ad1acbc14790b1514f276bfea0` /
`e1e859779301faced46d6476691e20b546c27ae0`.

Every original camera is checked against the E.10 inventory; all128 original masks
have new frozen byte identities. The data-only decoder accepts only the observed
protocol4, two-NumPy-floating-matrix dialect. It interprets bounded opcodes as data;
it **never** invokes pickle.load, payload globals, reducers or BUILD methods.
Unexpected dialect/shape/type/byte count/nonfinite values fail. No checkpoint is
needed. New numeric calibration pickles are written only in the generated scene,
for compatibility with A; accompany them with an allow_pickle=False NPZ archive.
Original camera files stay read-only and unchanged.

Original `[R|t]` is world-to-camera; center is−Rᵀt. Original A CPU float32 pinhole
arithmetic uses normalize(([u+.5,v+.5,1]K⁻ᵀ)R). Set K128[:2]=K800[:2]*(128/800),
keep poses unchanged. Preliminary data-only inspection found valid shapes and
rotation error≤1.20e−7; that is **not** real camera/mesh admission. World units
remain unverified (do not label them meters).

After approval, use only the pinned A RayTracer wrapper / installed `_raytracing`
for BVH queries. Do not import a scientific renderer, model, shader or optimizer.
Store raw face normals and original A's final normalize(−raw); never repair/flip
winding. Reference integration promotes these normals to float64 and unit-normalizes
them; this removes float32 length error, without changing orientation. All128
primary views must pass:

| Check | Frozen bound |
|---|---|
|Calibration orthogonality/determinant; independent CPU pinhole ray error|2e−6|
|Hit positions versus origin+unit_direction*depth|2e−4 world units|
|Final normal length|2e−6|
|Final normal dot view|strictly positive for **every** hit|
|Reprojected primary pixel-center error|2e−4 pixels|
|IoU with original mask, nearest sample at the new pixel centers|≥.90 for every camera|
|Independent CPU versus BVH hit status|exact|
|Independent CPU/BVH hit depth and raw normal error|≤2e−4 each|

Independent float64 Möller–Trumbore checks run64 fixed rays: cameras
0,11,17,58,76,105,114,127 × pixels(y,x)
(64,64),(48,64),(80,64),(64,48),(64,80),(48,48),(80,80),(0,0).
At least16 must hit. No edge exclusion, alternative rays or relaxed bound after
failure. All geometry gates precede radiance generation. Save rays/hits/normals,
errors and source mask IoUs; a BVH-only self-consistency check is insufficient.

## Ground-truth field, lighting and independent integration

Use exactly E.10's smooth additive-family field. With q=(p−bbox_center)/half_max_extent,
a=.23+.03q_x, b=.35+.03q_y, g=normalize([1,2,3]), u=g−(g·n)n,
Q_world=a²(I−nnᵀ)+b²uuᵀ. RGB=(.35+.08q_x,.50+.06q_y,.20+.04q_z),
metallic=.25+.05sin(pi*q_x)cos(pi*q_y). Geometry is never transformed.
Save world/local Q, sorted widths, major-axis theta, double-angle encoding,
RGB/metallic, actual normal and native X/Y tangent basis. Undefined theta remains
NaN with a separate validity mask. Eligibility for angular reporting requires
major−minor≥.1; retain all pixels/Q/axes and report counts, not a favorable subset.
Independent NumPy reconstructs every saved Q/material field before admission.

Physical BRDF is d/pi+F*D/(4*wo·h), d=k(1−m), F0=.04(1−m)+mk, Schlick Fresnel;
D is the Cartesian anisotropic Beckmann density from Q. Below-horizon terms are
zero. This is the **additive** family, not globally guaranteed passive. Record
unoccluded unit-environment directional energy and channels>1 descriptively;
require finite/nonnegative energy and valid parameter domains. Do not call a
finite energy measurement a proof of conservation. A_energy_budget_v1 is excluded.

Reference code imports neither corrected A nor experimental A2. It evaluates
f*cos/pdf explicitly in float64. Integration has equal counts from cosine-hemisphere
and **isotropic GGX half-vector** proposals(alpha=.3); GGX is only a reference
proposal, not the physical NDF. Mixture PDF is .5*cos/pi+.5*D_GGX(h)*n·h/(4*|wo·h|).
Rejected specular draws contribute zero and retain the total draw denominator.
The independent proposal is deliberately different from A's anisotropic Beckmann
sampler/cancellation. Analytical/local-matrix and separate uniform-solid-angle
quadrature fixtures check it on synthetic inputs.

Frozen scrambled Sobol tapes: SciPy-generated seeds7300/7301,8192 float64 pairs
per proposal. Fine=16384 directions per point; coarse=first2048 from **each**
proposal (4096 total). Reuse tape bits at all points; no scrambling/replacement
based on results. Archive tape bytes/hash. This is deterministic two-grid
operational convergence, **not** an independent statistical confidence bound.
Visibility uses the original float32 offset p+1e−5*direction and depth≥10 miss
policy. Indirect/human light is zero. No accepted-ray redraw or secondary bounce.

Known distant world environment:
ambient(.08,.08,.08)+max(w·Z,0)^12*(1,.8,.6)+max(w·X,0)^6*(.3,.5,.9).
For unseen relighting rotate light axes actively +90° about Y: Z→X, X→−Z.
Reuse the same geometry/visibility directions, not new shadow queries. Produce
relights at validation58 and held cameras11,105,114,17,76.

For **every** original-light and relight image: foreground fine/coarse RGB maximum
absolute difference≤.003 and relative L2≤.002. No percentile exclusions, changed
resolution, denoising, post-result tolerance tuning or automatic retry. Save fine,
coarse, diffuse/specular components, differences and incomplete arrays before a
failure. Reject nonfinite/negative/out-of-[0,1] final RGB rather than introduce
exposure/tone mapping or hide clipping. Such a rejection may reveal a needed
separately reviewed dataset revision, not permission to tune this execution.

## Images, splits and records

Render all128 camera views newly at128×128. Historical RGB images are never
loaded as targets. Save float64 linear components as the primary reference.
Native PNG: original A sRGB transfer with float32 epsilon, round-to-nearest uint8;
black background. Save binary masks and uint16 round(distance/15*65535) depth,
miss=65535, where depth is unit-ray **distance**, not camera Z. No antialiasing,
resampling of newly rendered observations or exposure fitting. Native fitting
therefore includes PNG quantization; report that separately from float64 GT.

A's newly designed seeded6033 split is122 training, validation58 and held views
11,105,114,17,76, with exact ordered IDs saved. These are **not** historical G
training/evaluation split claims. Relighting is excluded from fitting/selection.

Output contains datasets/GlossySynthetic/bell/{ID.png,ID-depth.png,ID-camera.pkl,
masks_train/ID.png}, geometry/{mesh.ply,calibrations-safe.npz,rays/,labels/,
labels.json,generation.json,oracle-admission.candidate.json}, references/ and
completion.json. Bind all outputs by hashes and all native dataset files also by
size/mtime. Record sources, independent implementation/tapes, original artifacts,
image identity, Python/Torch/CUDA/ABI/GPU/backend identity, actual work, RSS, GPU
memory, elapsed time, failure diagnostics and cleanup outcome. Runtime checks
read-only mounts/non-root/no network/one A100; source bytes are checked again at
termination. The host verifies clean committed HEAD because linked-worktree .git
administration is deliberately not mounted into Docker.

The candidate admission has researcher_reviewed_generation=false. Only after
successful offline verification and explicit review, including the subsequent
native-smoke authorization, may a **new** oracle-admission.json set that field
true, binding the unchanged generation/labels hashes. Never rewrite the candidate
or equate hash checks with independent physical correctness.

## Frozen cost and execution boundary

One selected A10080GB, four CPU threads,64GiB host memory, no swap, CUDA allocator
fraction .25,4GiB generated-output cap,≥8GiB initially free. Point batches16;
2,097,152 primary rays and at most34,359,738,368 proposed shadow directions
(actual traced upper-hemisphere rays separately counted); at most35,970,351,104
BRDF direction evaluations including the six relight pairs.64 CPU triangle queries,
128 fine/coarse image pairs and6 relight pairs; zero models, shader-native calls,
optimizer returns, checkpoint loads, training updates or new ground-truth geometry.
Conservative estimated runtime30–120 minutes; **unmeasured**, not a benchmark.
Foreground outer watchdog7200s +30s kill grace, inner7140s +20s grace, no retries.
If unable to access Docker/device, report the restriction without bypassing it.

The [foreground launcher](../../benchmarks/run_bell_e11.sh) requires a separate
hash-bound approved authorization JSON and a fresh host directory; it cannot
create affirmative authorization. It removes only its own label-bound container
on exit. Authorization keys: protocol,approved,freeze_sha256,commit,image_id,
output_directory=`/outputs/benchmark`,host_output_directory and exact work_limits
from the freeze. Before approval, run **only** the synthetic tests or data-only
input inspection documented below. Full generation is not authorized by this file.

After a complete dataset, run the offline verifier (no models/rays), inspect its
failures, then request separate dataset admission + two-update native approval.
Use E.10's exact existing smoke command with the admitted paths/source commit;
no additional checkpoint/extraction/relight shader queries are implicit in it.
No2000-update experiment starts automatically.
