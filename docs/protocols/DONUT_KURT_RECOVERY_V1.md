# Controlled donut physical material recovery v1

Status: implementation and CPU fixtures only. **No reference rendering, native
construction, optimizer update or training is authorized yet.** Each execution
needs the researcher's separate exact-source approval. Accepted camera0
reproduction at e0e9a2a3b5c48db7987e43508e0a93599f425bd6 is not repeated.

## Fixed new acquisition

Use the previously exported oracle `oracle-geometry-and-frames.npz` and
`native-chart-orientation-reference.npz` from
`/home/isavand/paper/evidence/donut-blend-inspection-v2`, and original
`/home/isavand/nero/data/nerf_synthetic/donut/transforms_train.json`.
Exact identities and the proposed fresh output paths are in the companion freeze.
Only primitive numerical arrays are loaded, with allow_pickle=False. Neither
Blender nor historical checkpoint/camera pickle loading is involved.

Material: **linear color[0.10804546,1.0,0.03044616], metallic0.8, major/minor
Beckmann widths0.35/0.08**, along the exported Blender world principal director.
These are NEW additive Kurt(a=0,ks=1)/Schlick-metallic material targets. They are
not converted Principled MULTI_GGX labels. Preserve original geometry and winding;
the reconstructed donut mesh is not used in this first matched-geometry test.

Use the already specified E.11D **L2_six_cardinal** distant environment: ambient
RGB0.01, six world cardinal power64 lobes, exact RGB amplitudes returned by
`experiments.bell_highlight_e11d.candidates()`. The freeze records all numerical
values. There is NO point emitter or indirect radiance. This intentionally
simplified acquisition differs from the historical Blender images and their GI,
HDR, finite light, AgX and denoising. There is no lighting search in this task.

Camera panel is frozen before radiance measurements. Start at source camera0,
then repeatedly maximize minimum angular separation of camera-center directions
about the known world origin; ties use smallest source index. Ordered panel:
**0,50,1,45,28,34**. First four are fitting views;28/34 are withheld. Their added
minimum angular separations are124.27°,76.80°,75.07°,56.10°,55.03°. This uses
geometry only, without material-response selection or replacement views.

Render128×128, scale focal and principal point directly from the original FOV.
Preserve A's NeRF integer-index ray convention (no+0.5 pixel shift), c2w with−Z
forward/+Y up, world Z up. Camera translations and geometry both use A's0.5
Blender-world scale. Normalize outgoing shader vectors explicitly. No fitted
registration, resizing of old RGB, or reinterpretation of old masks occurs.

## Geometry, normal and independent reference gates

Use the admitted E.11C fp64-predicate BVH/image without rebuilding:
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`.
Primary rays use immutable oracle vertices. Independently crosscheck eight
geometrically spaced hits/view against float64 closest-triangle intersection
(48 total; depth error≤2e-4). Position versus o+depth*d must agree within2e-5.

Attribute every hit to a source triangle among32 nearest triangle centroids.
Require plane residual≤2e-6 and barycentric weights in[−2e-4,1.0002]; otherwise
stop without a fallback mesh/projected point. This accommodates float32 hit
arithmetic, not geometry changes. Interpolate exported corner smooth normals.
Project corner principal directors into that normal's tangent plane and
interpolate their unoriented projectors; require dominant eigenvalue gap>0.5.
Use the resulting principal line to construct world-Q. Director signs and axis
chart choices must not affect Q. Face/shading normal hemispheres must agree.

Oracle normals already point outward. They are not sent through original A's
negation intended for inward reconstructed meshes. Secondary BVH normals have
no lighting effect because indirect radiance is zero. Save raw BVH normals,
source face normals and actual smooth shading normals separately; never flip.

Preserve ALL primary hits in the geometric mask and labels. The existing
one-sided BRDF has zero radiance where smooth n·wo≤0; label these explicitly
with shader_valid=False. Such pixels remain in the saved geometry/zero RGB
image, with counts reported; do not silently change normals or invent a hit.
Native shading queries and physical observation metrics use the declared
upper-hemisphere domain; excluded counts remain visible.

Reuse the independent Cartesian `kernel_axes` Kurt reference and E.11E midpoint
solid-angle grids256² and512². Weights are2π/N, z=cos(theta) in(0,1), full azimuth,
SUM(f*Li*cos*dOmega); never an additional sin(theta) or extra mean. The analytic
solid-angle/cosine controls must pass. Query visibility using exactly the
existing native origin offset1e-5 along float32 wi and miss criterion depth≥10.
No secondary illumination is synthesized. Use all quadrature directions without
random tapes, material-based probe selection or per-pixel adaptive refinement.

For EVERY view, require both independent radiance and unoccluded
directional-hemispherical reflectance to converge: max RGB difference≤0.003
AND relative L2≤0.002 between grids. Preserve coarse/fine arrays before deciding.
All radiance/reflectance must be finite and nonnegative. Report reflectance>1
honestly: additive corrected-core is not guaranteed energy conserving, and a
soft native proxy does not establish conservation. Do not renormalize the BRDF,
clip radiance, revise bounds or automatically add samples after failure.

Each view NPZ retains unclipped float64 diffuse/specular/total RGB, coarse
components, reflectance, masks/validity, positions, depth, outgoing views,
triangle/barycentric identity, smooth/face/BVH normals, world director/Q and
explicit color/metal/width labels. PNGs are standard-sRGB clipped previews ONLY.
Save an RGB/diffuse/specular contact sheet and frozen E.11D saved-component
highlight contrast/coverage diagnostics, without new counterfactual rendering
or selection of favorable cameras. Reference admission must review actual
highlight visibility and resolution; no such success is claimed before rendering.
Generation report binds source, calibration, material, environment, counters and
all output hashes. No corrected native shader produces these reference images.

## Matched D1/D2 recovery

The small adapter uses the full original A MCShadingNetwork material field
(1,485,584 material parameters), frozen corrected-core BRDF/sampling/decoder,
and original A Charbonnier and prior implementations. The historical Trainer/
dataset loader is not run: it consumes tone-mapped8-bit data and reconstructed
face normals rather than these linear oracle observations/smooth normals. This
explicit new observation interface preserves the full native material model;
it is not a replay of historical donut training.

D1 uses the analytic L2 environment. D2 uses original A's directional outer_light
MLP through corrected-core's existing environment callback. Both retain exactly
zero inner/human radiance. D1 freezes light parameters; D2 trains outer_light
only. All six material groups/shared features remain trainable in BOTH arms.
No GT material term, teacher, learned geometry, head restriction or new prior.
The declared shader config remains known_env_v1/zero_bounce_v1; reports separately
identify D2's effective learned callback, avoiding ambiguous runtime semantics.

Fresh seed6033, strict complete initial-state hash equality, common fitting data
and dedicated seed6033 shuffled512-ray observation batches. Preserve original
512 diffuse/1024 specular samples and random-azimuth policy. Use original Adam
(initial LR.001,betas.9/.999,eps1e-4,AMSGrad,no weight decay) and WarmUpCosLR
(warmup1000,default scheduled LR.0005,hard-coded cosine horizon100000,alpha.2).
Exactly10000 returns per arm; first LR0, second LR5e-7. Clip gradients at0.7.
No resume, model selection, early favorable checkpoint, schedule tuning or extension.

RGB loss is original Charbonnier(sum channel squares+0.001), now on unclipped
linear RGB instead of old sRGB. Hold actual effective legacy_v1 priors identical:
reg_change=True,gaussian0.02,spatial coefficient0.05,energy/specular0.05 each,
original orientation decay and diffuse-white coefficient0.01. Historical top-level
0.01 energy/specular declarations were ignored; they are not silently enabled.
The optional surface-world-Q candidate is NOT used. Existing energy/specular
proxy limitations remain explicit and may bias recovered materials.

Before any Adam update, verify known-light native/reference agreement at24
geometrically spaced valid camera0 pixels,32 fixed draws: reuse E.16's0.001
absolute/0.02 pooled-relative bounds, six empirical SEs, retained two-grid error
and64 float32-epsilon guard; require adequate SE precision. These are bounded
compatibility diagnostics, not rigorous simultaneous confidence intervals.
Failure stops that arm/batch. It does not trigger more draws or fitted correction.
Fitting cannot consume held-out observations or truth labels. Truth enters only
this compatibility gate and declared evaluation diagnostics.

Require finite native outputs/losses/active gradients; all material groups must
be connected, as must D2's outer_light. Observe first two Adam state counters,
finite moments and zero/nonzero warmup displacements. Save trajectory losses,
kd/metal/diffuse allocation,widths and gradient groups every update. At step1
and each2000-return boundary, record four component-gradient probes (photo,
material,legacy energy,specular), including group norms and cosine interactions.
No probe modifies gradients, optimizer, RNG, or loss coefficients.

Save own checkpoints and ALL six fixed camera/material maps at0/2000/.../10000.
Checkpoint files are written only at2000/.../10000, contain weights/Adam/RNG,
sample order/cursor and strict arm/variant/freeze/dataset identities. Isolated
CPU pickle loading is limited to each process's own new checkpoint for exact
strict roundtrip; no historical checkpoint is loaded or optimizer restored.

Use eight fixed native draws for RGB/components at every endpoint, preserving
the training RNG/modes/default tensor type. Save raw numerical predictions,
standard errors, world-Q/projectors/raw orientation, sorted/raw widths, kd,
metal,diffuse/specular components, GT and RGB previews with consistent scales.
Save linear illumination maps/known references and weighted spherical errors.
Report Q-relative error mean/median/p95 and Q≤.05 counts, orientation mod physical
equivalence (predicted anisotropy gap≥.1 eligibility; no ineligible successes),
both sorted-width errors, color and metallic errors, near-bound counts and
green kd<0.5 collapse counts. RGB errors are linear RMSE/relative L2 on the
same fixed domain, with stochastic SEs reported separately. Do not infer physical
recovery from RGB alone. Physical metrics are repeated view observations, not
an area-uniform global-surface score. Smoothness is a diagnostic on1024 fixed
edge-connected mesh pairs, using the existing normal-transported Q computation
and comparison with oracle Q variation; it is NEVER an added loss.

D2 has unresolved material–illumination scale/color ambiguity. Known geometry
and exact BRDF do not prove seven-parameter identifiability. Compare D1/D2 to
diagnose compensation and prior/optimization effects, without claiming unique
absolute recovery for the unanchored learned-light problem.

## Separate execution bounds and sequence

Reference: six128² images,98,304 primary rays, at most32,212,254,720 shadow and
the same number of independent BRDF point-directions,48 CPU checks. Actual
integration scales with valid foreground count and will be reported. No native
model/optimizer calls. Runtime7200s/30s grace (inner7080s), four CPUs,64GiB RAM,
25% CUDA allocator fraction,512MiB outputs,one A100 GPU0. Estimated5–60 minutes,
unmeasured for this torus/material; convergence at grazing points remains unknown.

Training proposal, separately approved after reference admission:20000 Adam
returns total, at most38496 native shader calls and30,226,513,920 shadow rays,
48 component-gradient reverse sweeps,24 smoothness field calls and12 illumination
maps. Two arms,3600s/arm (inner3540s),7500s whole foreground watchdog/30s grace,
four CPUs,64GiB RAM,25% CUDA allocator,8GiB outputs/arm. Estimated10–60 minutes
total, unmeasured. Exact limits are machine-readable in the freeze. If time or
scientific validity fails, retain partial checkpoints/maps and stop all remaining
work. No retries, extensions, hidden lighting normalization or extra arms.

Use the existing immutable corrected image, non-root, no network, read-only
source/oracle/calibration/admitted dataset, separately writable fresh outputs.
Only owned labeled containers may be cleaned up. Codex's previous Docker denial
does not authorize bypasses; provide researcher-side foreground commands when
needed. No image rebuild, install, sphere/Bell/E.17 launch or Blender rerender.

The launcher requires an exact approved authorization, fresh output identity and
clean source commit. Reference success does not automatically create affirmative
dataset admission or launch training. The separate training authorization must
bind generation.json SHA-256 and researcher_admitted_dataset=True.
Launch from a regular tmux terminal, retaining the foreground watchdog:

```bash
# Supply the hash-bound approval file only after researcher authorization.
env DONUT_SOURCE=/home/isavand/paper/Nerofactor-donut-recovery \
  DONUT_ORACLE=/home/isavand/paper/evidence/donut-blend-inspection-v2 \
  DONUT_CAMERAS=/home/isavand/nero/data/nerf_synthetic/donut/transforms_train.json \
  DONUT_BATCH='<fresh frozen reference output>' DONUT_AUTH='<approved authorization.json>' \
  DONUT_GPU=0 DONUT_PHASE=reference \
  timeout --foreground --signal=TERM --kill-after=30s 7260s \
  bash /home/isavand/paper/Nerofactor-donut-recovery/experiments/run_donut_recovery_v1.sh
```

For the later training authorization use DONUT_PHASE=training, the separate
frozen training batch, DONUT_BENCHMARK pointing explicitly to the admitted
reference directory, and7500s whole watchdog. No training command is authorized
by preparation or reference approval alone. The next immediate step is ONE
reference-generation authorization; empirical results must precede conclusions.
