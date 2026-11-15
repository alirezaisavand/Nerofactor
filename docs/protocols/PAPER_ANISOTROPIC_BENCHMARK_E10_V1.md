# E.10 v1 — practical oracle-object benchmark proposal

**Prepared, not executed or finally frozen.** No new images, reference targets,
model training or recovery measurements exist. The native implementation and
configurations are executable candidates; generated-dataset admission and native
GPU validation remain blockers. Do not treat this proposal as authorization.

## Actual assets and scientific purpose

The [read-only inventory](../evidence/PAPER_BENCHMARK_E10_ASSET_INVENTORY_V1.json)
records exact mesh hashes, bounds/finite/index/area metadata, all256 small camera
file identities, image0 identities/dimensions, masks and ordered split IDs.
No camera/checkpoint pickle was deserialized; entire image datasets were not
hashed. Pretrained reference surfaces can define a *new synthetic oracle*;
this does not establish original GT, producing revision or a new G training run.

| Candidate | Existing source | Mesh identity | Status |
| --- | --- | --- | --- |
|Bell, first object|`data/GlossySynthetic/bell`,128 images/cameras/masks at800x800|`final_results/meshes-pretrained/syn/bell.ply`,124456 vertices/248908 triangles; SHA256 `4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01`|Inputs accessible; no verified BRDF/lighting labels|
|Cat, reserve second object|`data/GlossySynthetic/cat`,128 images/cameras/masks at800x800|`final_results/meshes-pretrained/syn/cat.ply`,530122 vertices/1060260 triangles; SHA256 `292962778d949e444b3e8b97951c3eb0435f66415120d915f146b7563884d920`|Higher geometry cost; defer until Bell passes|

Serialized signed volumes are −.8547482 and−.5805701. There are no PLY normal
properties. These numbers do not establish outward shading normals, watertightness
or camera alignment. Do not flip/repair/rescale either input. Verify original
A BVH normals and its existing negation, intersections/camera centers/silhouettes
and source-calibration identities before new GT production. Camera units in
meters remain unknown; use the unchanged native scene frame/scale_factor1.

Existing RGB/depth/masks are calibration/geometry references, **not** images of
the new prescribed material under the new known light. Relabeling them as such
is forbidden. All generated data must have a fresh directory under the writable
evidence root, separate from `/data` and `/final_results`.

## Proposed known-material dataset

Start with Bell's unchanged mesh as the benchmark-defined oracle. Use the128
ordered camera IDs0..127 after separately authorized, isolated trust/calibration
verification. Native non-NeRF convention is world-to-camera [R|t], center−Rᵀt,
world ray normalize(([u+.5,v+.5,1]K⁻ᵀ)R). Do not substitute the alternate NeRF
ray constructor. The source expects a3x4 pose/K payload; inventory did not inspect
its numeric values. A compact safe NPZ calibration archive should accompany any
trusted legacy camera decoding in the future generator.

New pilot images:128x128; scale K consistently by128/800 for the new image grid,
with the same +.5 pixel convention and unchanged poses. This is new data,
not resizing original observations. Initial full-panel resolution and reference
quality still need a generation protocol/runtime review. Preserve continuous
linear RGB and normals/material labels separately; encode native input PNGs
with original linear-to-sRGB, no exposure fitting or hidden tone mapping.
Background is black and native training uses original hit rays. Preserve masks,
depth and original source-depth encoding contract; write them only in the new root.

Define q=(p−bbox_center)/(max_bbox_extent/2), affecting **only the material field**,
not geometry/cameras. For the given oracle normal n, let P=I−nnᵀ,
g=normalize([1,2,3]), u=P*g, a=.23+.03*q_x, b=.35+.03*q_y. Define

`Q_world(p,n)=a²*P+b²*u*uᵀ`.

It has positive tangent eigenvalues a² and a²+b²||u||², with widths in the
original allowed domain; normals' faceted discontinuities remain geometry
controls. Material RGB is `[.35+.08*q_x,.50+.06*q_y,.20+.04*q_z]`, metallic
`.25+.05*sin(pi*q_x)*cos(pi*q_y)`. The physical tensor is independent of the
arbitrary X/Y shader chart. Derive local Q, sorted widths and orientation relative
to actual native T/B; at u=0 the material is isotropic and orientation is
unidentifiable. Store that fact, never a fabricated angular success or an
arbitrary physically meaningful direction. This field is a dataset definition,
not a new trainable parameterization/optimizer.

Known distant environment, implemented in all paired native adapters:
ambient(.08,.08,.08), plus max(w·Z,0)^12*(1,.8,.6) and
max(w·X,0)^6*(.3,.5,.9), in linear RGB. Human light off, indirect zero, original
BVH visibility and ray offset preserved. This is **explicit single-bounce
image formation**, not learned-indirect relighting. New held-light captures
must use a separately pinned rotated environment outside training.

Independent GT must evaluate Cartesian Beckmann D using the prescribed tensor,
not call the corrected native Monte Carlo estimator to produce its own targets.
For additive-family data use d/pi+F*D/(4*wo·h); for convex-budget-family data use
d/pi+(1−d)*F*D/(4*wo·h). Record which family generated every image/label archive.
Independent directional integration/convergence and energy/domain checks must
pass before references are admitted. Do not loosen grids/bounds or substitute
materials after inspecting recovery. No such references were generated here.
The [machine-readable draft](PAPER_ANISOTROPIC_BENCHMARK_E10_V1.json) makes the
current preparation/missing-artifact status explicit.

## Splits, metrics and fair arms

Mirror A's actual seeded6033 shuffle over ordered0..127 using a private RNG for
planning. New benchmark validation=`58`; held-view/NVS=`11,105,114,17,76`;
the remaining122 IDs fit. Complete ordered arrays are in the inventory. These
are **new designed splits**, not historical G thesis evaluation attribution.
Unseen illumination is evaluated independently at frozen cameras after fitting.

First same-family correction panel: `A_original`, `A_estimator_v1`,
`A_corrected_core_v1`, on identical additive-law targets and initial seed6033.
Known geometry/light,128x128 images,512 train rays,512 diffuse/1024 specular
samples and2000 original Adam returns per arm. Native original Charbonnier on
sRGB, original initialization, eps1e-4/AMSGrad, clipping.7 and1000-step warm-up /
hard100000 cosine semantics remain. Priors and diffuse-light whiteness are off
for **all** arms. This is not E.9's linear-MSE flat-field experiment. Save/validate
at2000; no best-seed/checkpoint selection. Queries, elapsed time, memory, optimizer
returns and final NVS must be reported separately; equal updates are not equal
actual GPU cost. The files below implement this fixed update count, not a budget
that still silently runs100000 steps:

| Arm | CLI selector | Envelope under `configs/experiments/` |
| --- | --- | --- |
|A|`A_original`|`material_a_bell_original_known_v1.yaml`|
|B|`A_estimator_v1`|`material_a_bell_estimator_known_v1.yaml`|
|C|`A_corrected_core_v1`|`material_a_bell_core_known_v1.yaml`|
|D, separately reviewed physical family|`A_energy_budget_v1`|`material_a_bell_energy_budget_known_v1.yaml`|

D has a proven ideal energy bound and focused numerical checks, but changes
material coupling. Its default admission requires **budget-family GT**, not the
additive dataset. Compare C/D on common fixed-parameter forward fixtures first
(as the unit tests do); any inverse cross-family comparison must identify
model mismatch and decide common BRDF/radiance metrics separately. Do not pool
parameter-recovery scores across differing material semantics or silently give D
an easier target. No fourth inverse arm/sweep is authorized here.

Report Q relative/absolute error in a common declared frame, sorted widths,
identifiable orientation, base color/metallic and all original physical pass
criteria; held-view linear/sRGB RGB, unseen relight, diffuse/specular components
and their compensation. Preserve native validation/NVS definitions for continuity.
Also require independent GT-mask/full-frame scoring: native evaluation composes
GT/prediction using model hit masks, so it alone cannot certify later G geometry
accuracy. Geometry/normals need independent GT surface/angle/silhouette metrics. Original-A
nominal Q must be accompanied by actual decoded-frame validity; nominal parameter
recovery does not certify that its buggy rendered BRDF matches the intended law.
For truly near-isotropic GT, keep all Q/axis/color/metallic results and counts;
report orientation as unidentifiable. Preserve original all_passed flags, whose
orientation requirement cannot pass there; do not treat that flag alone as a
physical failure or drop those pixels to inflate success. Any eligibility-based
aggregate needs an explicitly reviewed benchmark definition. No new evaluation
mathematical definition is silently substituted into A.

Only after oracle recovery is established: replicate fixed seeds and the reserve
Cat scene, then replace oracle geometry with an attributable, canonical-resolution
G run and audited normals. Keep geometry error and estimated illumination as
separate factors. Historical pretrained/r128 meshes do not satisfy the research
G→A provenance policy. Real-object claims additionally need calibrated lighting,
geometry and independent anisotropic material measurements where feasible.

## Admission and the smallest native launch

`--oracle-benchmark /inputs/geometry/oracle-admission.json` is a distinct policy.
Its protocol is `paper-anisotropic-oracle-dataset-v1`, schema1, complete status,
scene/mesh hash, explicit coordinates/normals/image formation/light/BRDF/split
labels and researcher-reviewed generation/known-material declarations. It binds
`generation_report` and `material_labels` through explicit InputRefs + SHA256,
and exact `dataset_files` records from the selected scene. The generation report
must independently identify complete matching dataset/mesh/labels and, for known
light, the exact environment/zero-bounce model. Hashes establish identity, not
physical correctness; generation review remains explicit evidence. No completed
manifest is fabricated while targets/labels/generation are absent.

Once that package exists and ONE scientific launch is separately approved, first
run only `--material-smoke` for C: original initialization +two original Adam
returns, then stop before any validation/checkpoint/NVS. Use the following inside
a foreground tmux terminal. **Do not execute now.** It deliberately stops before
Docker if the generated package is absent. Source and all inputs are read-only.

```bash
set -euo pipefail
E10_SOURCE=/home/isavand/paper/Nerofactor-3d2-e2a
E10_BENCH=/home/isavand/paper/evidence/paper-benchmark-bell-v1
E10_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
E10_GPU=0
E10_COMMIT=$(git -C "$E10_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E10_SOURCE" status --porcelain)"
test "$(git -C "$E10_SOURCE" branch --show-current)" = feat/neural-material-early-step-e2a
test -s "$E10_BENCH/geometry/oracle-admission.json"
test -d "$E10_BENCH/datasets/GlossySynthetic/bell"
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$E10_IMAGE"
E10_RUN=$(python3 -B -c 'import uuid; print("bell-e10-core-smoke-"+uuid.uuid4().hex)')
E10_BATCH=$(mktemp -d /home/isavand/paper/evidence/bell-e10-smoke.XXXXXXXX)
mkdir "$E10_BATCH/run"
set +e
docker run --rm --pull=never --init --user "$(id -u):$(id -g)" \
  --gpus "device=$E10_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g \
  --cpus=4 --pids-limit=128 --shm-size=2g --ulimit core=0 \
  --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E10_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$E10_BENCH/datasets,dst=/inputs/datasets,readonly" \
  --mount "type=bind,src=$E10_BENCH/geometry,dst=/inputs/geometry,readonly" \
  --mount type=bind,src=/home/isavand/paper/final_results,dst=/inputs/thesis,readonly \
  --mount "type=bind,src=$E10_BATCH/run,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$E10_IMAGE" \
  --signal=TERM --kill-after=30s 900s /usr/bin/python3 -I -B /workspace/run_stage_experiment.py \
  --stage material --scientific-commit e1e859779301faced46d6476691e20b546c27ae0 \
  --arm C --material-variant A_corrected_core_v1 --material-smoke \
  --cfg /workspace/configs/experiments/material_a_bell_core_known_v1.yaml --config-root /workspace \
  --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
  --input-root geometry=/inputs/geometry --oracle-benchmark /inputs/geometry/oracle-admission.json \
  --run-id "$E10_RUN" --execute --confirm-isolated-runtime \
  --image-id "$E10_IMAGE" --infrastructure-commit "$E10_COMMIT" \
  > "$E10_BATCH/console.log" 2>&1
E10_EXIT=$?
set -e
printf '%s\n' "$E10_EXIT" > "$E10_BATCH/exit-code.txt"
cat "$E10_BATCH/console.log"
test "$E10_EXIT" -eq 0
sha256sum "$E10_BATCH/run/stage-run.json"
```

This reuses the existing reviewed image identity/limits and does not rebuild it.
CPU fixtures/Python3.8 grammar are insufficient to certify its Torch/CUDA ABI
execution. A future authorization must bind the final E.10 commit/implementation
hashes, admitted input hashes, scene/arm, exactly2 optimizer returns and fixed
sampling,900-second/30-second grace,64GiB host memory/four CPUs/one selected A100,
fresh run/evidence paths and no retries. Preserve any failure. Full2000-step
A/B/C work and any target generation need distinct concrete budgets; neither is
an automatic continuation from the smoke.

## Remaining blockers and scientific priority

1. No new known-material dataset, safe calibration archive, independent generation
   report or labels exist. Source calibration trust/numerical inspection, normal/
   silhouette admission and reference feasibility precede a generation freeze.
2. No native corrected-GPU constructor/backward, benchmark checkpoint roundtrip,
   full NVS or relighting image execution has been performed. No performance or
   recovery improvement is claimed.
3. Core's additive model has energy counterexamples. The separate budget model
   is passive under its stated assumptions, but has distinct spectral coupling
   semantics and no empirical/native-GPU admission. Review the intended physical
   material interpretation before a paper-primary choice.
4. Zero-bounce known lighting is a controlled initial assumption, not a solution
   to learned illumination/indirect-light ambiguity. Expand those factors only
   after same-family oracle recovery is demonstrated.

Prioritize these validity gates and a small Bell correction comparison over
another spatial optimizer study or broad scene/seed sweep. The E.9 opportunity
backlog remains intact; standard estimator repairs/convex mixing are not novelty.
