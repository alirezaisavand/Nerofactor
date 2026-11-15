# E.11C v1 — isolated predicate build and native BVH regression

Preparation from `d3ea1a3542b13635afcea54e65b58a38d1c75773`, branch
`feat/neural-material-early-step-e2a`. Build and CUDA are **not authorized by this
file**. One explicit approval must bind the final commit, companion freeze hash,
source/patch/input identities, selected GPU and evidence root below. That approval
covers compilation, derivative assembly, both native probes and offline analysis,
without a second review between build and probe. No generation or training.

## Source and derivative

Upstream raytracing `29329268a86921f1c938b05277937369531209a4`; reviewed patch
`docker/paper/patches/raytracing-triangle-fp64-v1.patch`, SHA256
`73fc52bb2d9de3d88fb12516ee5dad6e3a5edd978f68140ccc2ec0ff2f1a5b9e`.
Stage only12 required upstream files(src/include/setup/two licences) by exact
Git blob identity. Retain original and patched copies. Only triangle.cuh differs;
no other predicate, BVH, AABB, normal or material source change. Preserve the
original accepted sources, masks, reports, meshes and freezes. Patch staging
executes no source/setup code. Actual source hashes are frozen before approval.

Pinned local base image:
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`.
The E.11 historical backend in this image is SHA256
`9cff02fdd0752c17a613874c0f71e47803a666234e5e5046cf194c5646102180`;
this is not the older historical-install binary identity from environment recovery.

Compile once in a no-network, non-root, read-only-root **base-image container
without GPU exposure**. Inputs/source/workspace read-only; only owned outputs
writable. Require Python3.8.10/Torch1.8.1+cu111/CUDA11.1/ABI0, GCC/G++9.4.0,
NVCC11.1.105, installed Eigen3.3.7, TORCH_CUDA_ARCH_LIST8.0, MAX_JOBS4. Invoke
original setup.py **build_ext**, with build-lib=/outputs/corrected and a separate
build-temp. No pip install/resolution, network fallback or package upgrades.
Record command/output, compiler paths/versions/hashes, Eigen, original/patched
source hashes, build.ninja, binary SHA and elapsed time. Original site-packages
extension is verified unchanged before/after.

Assemble a new uniquely tagged image by COPY only, retaining the historical base
layers and original extension. Tag `nerofactor-paper:ray-predicate-fp64-e11c-ffbde4d0e261`;
refuse an existing tag. The base tag is checked against its immutable ID before
assembly; post-build root-layer prefix equality and repeated base-ID checks
reject tag substitution. No RUN/setup/install in image assembly. Store the new
immutable image ID and layer ancestry. Place source/licences/manifest/patch/build
record and corrected binary under /opt/e11c. Never replace the old .so or image.

## Native correctness panel

Both backends run in separate fresh processes **inside the same derivative
image**, selecting the exact .so path/hash before importing the pinned A
RayTracer wrapper. No renderer, model, shader, optimizer or checkpoint is used.
This exercises _raytracing.create_raytracer, full BVH construction/traversal and
trace kernel, not a stand-alone triangle function. All inputs are immutable.

Bell uses original mesh SHA256
`4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01`.
36 cases: exact archived5×5 camera1 neighborhood(25, including ray13621), three
fractions(.25,.5,.75) of the front181142/shared181143 edge with dimensionless
barycentric offsets−1e−4,0,+1e−4(9), plus two1e−5-offset visibility rays along
±the saved direction from the independently established front point(2).
All final ray inputs are frozen float32 arrays. No camera convention changes.

Reuse accepted25 float64 references; compute only11 additional Bell and8 cube
references on the exact quantized ray inputs using independent float64
Möller–Trumbore. The inward-wound12-triangle cube has central/shared-edge,
edge-side, exterior-miss, away-miss, inside-exit, side-hit and near-parallel-miss
controls. No regenerated RGB, labels, material truth or reference integration.
Inside-exit controls can correctly have back-facing normals; they do not change
the outside-camera admission criterion of the Bell generation protocol.

Frozen checks: hit/miss equals independent reference exactly; depth and position
absolute errors≤2e−4, normal component error≤2e−4, normalized-normal length error
≤2e−6. These reuse E.11's CPU/native contracts, not post-result fitted bounds.
Require finite nonnegative depths and finite hit positions/normals. No barycentric
epsilon, pixel removal or normal flip beyond original A's normalize(−raw).

Known ray13621 must reach original PLY181142 at approximately2.75133775, facing
approximately+.95854; reference normal≈(.39438296,.48206510,−.78235243).
The extension does not return a primitive index: original-index compatibility is
checked through independently known closest point/depth and triangle normal,
not falsely described as an exposed BVH primitive ID. Original reference must
reproduce its archived incorrect3.60320640 depth within2e−4; its CPU mismatches
remain visible. Its passing *reproduction check* does not mean correct geometry.
Corrected backend must match **all44** CPU references. Retain false-positive,
false-negative, depth/position/normal failures and raw arrays before rejection.

## Representative timing and budget

Reuse the32768 saved primary rays(camera0+1), full unchanged mesh and BVH.
Per backend:5 warm-ups,20 timed API traces, same input bits/order; record each
synchronized wall latency and CUDA-event latency. Include allocation/API costs
in wall figures. Exactly819244 ray queries/27 trace calls per successful backend,
1,638,488 total. No extra uncounted retries or warm-ups. BVH construction/setup
and cold-process elapsed time are recorded separately from trace timing.

Report corrected/original wall-median ratio and absolute distribution. Ratio>2
is a predeclared descriptive substantial-penalty flag, **not** a correctness
failure or license to tune predicates. Sequential same-image timing does not
establish statistical significance, thermal-order independence, or end-to-end
reference-generation feasibility. No global watertightness claim from44 cases.

Whole foreground watchdog1800s+30s kill grace; build900s+20s; COPY-image
assembly120s+20s; each probe180s+20s. Four CPU threads; build16GiB host memory,
each GPU probe8GiB host memory, no swap. One selected A100 GPU0; CUDA allocator
fraction.1. Owned output cap2GiB; compiler outputs/image binaries stay outside
Git. Corrected process has its own output/cache tree. Narrow read-only code,
source/panel/auth/mesh mounts; no network, non-root, dropped capabilities.
Cleanup removes only containers created under recorded CIDs with this owner
label. Preserve failed containers' logs/status and partial outputs; no retry or
alternative strategy if build/native gates fail. Docker-access denial is a
runtime blocker, never authorization to bypass restrictions.

## Evidence and conditional continuation

Fresh root:
`/home/isavand/paper/evidence/e11c-ray-backend-ffbde4d0e261491bb097d01fe072883b`.
Preparation lives in inputs/; execution/ must be absent before the one launch.
Authorization binds this root/final commit/freeze/base image/exact limits and
covers **build+both probes+offline analysis**. The foreground launcher is
`docker/paper/e11c/run.sh`; required explicit environment variables are
E11C_SOURCE,E11C_ROOT,E11C_AUTH,E11C_GPU. It cannot fabricate approval.

Outputs: execution/build/out/{compile.log,build.json,corrected/,build/},
image-context/,image.json,image-build.log and stage exit codes;
original/out/ and corrected/out/ with probe.json, completion receipts, Bell/cube
raw NPZ and final timing-batch NPZ. Offline analysis checks hashes, recomputes
comparisons from actual captured arrays, and writes comparison.json. New sources,
inputs, compiled binary and immutable image identities all remain traceable.

Only if both reproduction and corrected correctness gates pass: record native
acceptance, integrate explicit corrected-sidecar loading before scientific
imports in Bell generation and native material execution, preserve selectable
historical loading, and pin the **same corrected backend** for all A/B/C arms.
Then prepare a fresh Bell generation implementation freeze and output directory,
without changing fields, BRDF, cameras, normals, tapes or numerical admission
bounds. Those generation/training executions need separate approval. If tests
fail, analyze saved captures for predicate/traversal/AABB mechanisms and stop;
no full generation and no invented corrected image admission.
