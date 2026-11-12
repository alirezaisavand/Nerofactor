# 3D-2 Phase A: original A material-field and evidence audit

**Audit/design only.** No new model was constructed, no shader or quadrature was
executed, and no optimizer/gradient step was performed. This document records
source inspection and offline checks of already saved 3D-1 numbers. The proposed
experiment is [a draft](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1_DRAFT.md), not
a frozen or executed protocol. See the [implementation plan](NEURAL_MATERIAL_FIELD_IMPLEMENTATION_PLAN_V1.md).

## Repository and authority

The completed reference worktree is `/home/isavand/paper/Nerofactor-3d1-multi-material`,
branch `feat/multi-material-recovery-v1`, clean at
`ccb03d0ec5adc814775538b9cc7427cc57480a98`. The original workspace
`/home/isavand/paper/Nerofactor-paper` is a different, clean worktree at
`72629688bee84c0b06f4fa3c19b207917d2dd450` on `feat/known-material-recovery-v1`;
it was not switched, reset or edited. Earlier milestone worktrees were retained.

A separate documentation worktree was created from the completed 3D-1 commit:
`/home/isavand/paper/Nerofactor-3d2-design`, branch
`feat/neural-material-field-design-v1`. This keeps proposed design changes off
the accepted scientific branch. No commit/push or protocol freeze is performed
in Phase A. All 664 original tracked files and the existing original-workspace
scientific-audit document are covered by a 665-file preservation inventory.
AGENTS.md, old protocols and all scientific/evidence files remain unchanged.

| Authority | Commit | Independent source inspection |
| --- | --- | --- |
| G geometry, `aniso_geometry_safe` | `79252178e65914ad1acbc14790b1514f276bfea0` | All 99 snapshot files match manifest SHA-256/size and original Git blob bytes |
| A material, `anisotropic` | `e1e859779301faced46d6476691e20b546c27ae0` | All 106 snapshot files match manifest SHA-256/size and original Git blob bytes |

The two roots are distinct. Geometry G is an admitted source identity, not an
executed dependency of a known-plane material experiment. No historical producing
checkpoint identity or new G-to-A research handoff is inferred from these checks.

## 3D-1 evidence: what was checked

All 83 inherited source/evidence identities in the primary report were verified.
Adding the 3D-1 frozen implementation/test/protocol, offline analyzer, primary
report and seven figure identities gives 96 directly checked identity entries.
The analysis, freeze and results document were also hashed and compared with the
accepted clean Git baseline. Important file identities are:

| File under `docs/` | SHA-256 |
| --- | --- |
| `evidence/MULTI_MATERIAL_RECOVERY_V1.json` | `3abc274191d0e84c456355a7c351ac1aa94a2c976c1f789979ddb550a4383e89` |
| `evidence/MULTI_MATERIAL_RECOVERY_V1_ANALYSIS.json` | `eb0c7d5f5f0d803c1a01dbf31739efc1cd9dbb280dfab2dd772401b5bf8688ee` |
| `protocols/MULTI_MATERIAL_RECOVERY_V1.md` | `cbdbaf9ee7a421b03738674afeae19214969c62b72d03c3072727e569404019a` |
| `protocols/MULTI_MATERIAL_RECOVERY_V1.freeze.json` | `f8c8de54cff14d614165dfe2fc46a1ec6e88ff3e685a6ebb9e34a2f02bc2e6db` |
| `MULTI_MATERIAL_RECOVERY_V1_RESULTS.md` | `465270c9bcae9eacbf91dec079134b7f75fe716765f31af65ee3c1a8a6981ead` |

Independent **saved-data** checks recomputed three target-array identities and
two-grid convergence gates, checked all 84 recorded perturbations, reconstructed
all three two-step/two-grid Jacobian arrays from their stored finite differences,
and recomputed the original singular-ratio/stability conditions. All passed.
Eighteen saved Adam snapshots at 250/1000 were checked for float32 shape/value
hashes and step counts. The 27 saved terminal physical states and RGB prediction
arrays were rescored against their own truths and original split membership.
The resulting counts are again 14 simultaneous physical passes, 25 held-out RGB
passes and 11 RGB-pass/Q-fail endpoints. This was not a render, a new reference
evaluation or a historical optimizer rerun.

| Material | Reference rank | Fine-grid condition | P0/U/C recovery, out of 3 | Median Q error |
| --- | --- | --- | --- | --- |
| M1 | 7 | 66.59263209 | 1 / 2 / 1 | 0.06227862 |
| M2 | 7 | 19.19138545 | 2 / 1 / 2 | 0.02722988 |
| M3 | 7 | 38.82267671 | 2 / 1 / 2 | 0.02674521 |

The report records nine completed continuous Adam histories, 27 completed LM
arms, 353894400 reference direction evaluations, unchanged RNG/tapes and restored
runtime state. Their counters, snapshot consistency and recorded flags were
inspected. Phase A has not independently repeated the execution, timed it,
reconstructed live RNG history or established cross-machine determinism.
Local rank remains local; the stored results are not evidence of neural-field,
learned-geometry or unknown-lighting recovery.

### A consequential baseline limitation

The nine **Adam update1000** Q errors, independently recomputed from saved
physical parameters, are:

| Material | Seed 101 | Seed 103 | Seed 105 |
| --- | --- | --- | --- |
| M1 | 0.69821795 | 2.04836022 | 0.77365868 |
| M2 | 0.25722477 | 0.33418630 | 0.05668149 |
| M3 | 0.39023026 | 0.37966142 | 0.91585332 |

All exceed the original 0.05 Q criterion. Adam-only simultaneous recovery at
that checkpoint therefore cannot pass; separate Adam held-out evaluations were
not archived and are not invented here. The published 14/27 successes concern
the subsequent LM endpoints. Comparing a 1000-update neural Adam run directly
with those Adam-plus-LM endpoints would confound representation and optimization.
Equal failure at 1000 would not establish neural recovery reliability either.
The proposed fixed longer Adam comparison is a new, unverified design choice.

## Active native A path and exact source locations

All locations below refer to **the material A snapshot**, never G or the
repository's unpinned `network/` directory.

| Path/function | Lines | Verified behavior |
| --- | --- | --- |
| [renderer.py: `NeROMaterialRenderer.__init__`](../scientific_sources/material_a/network/renderer.py#L1103) | 1103–1110 | Initializes geometry, datasets and shader |
| [renderer.py: `_init_shader`](../scientific_sources/material_a/network/renderer.py#L1251) | 1251–1253 | Constructs `MCShadingNetwork(shader_cfg, trace_callback)`; only adds `is_real` to that scope |
| [renderer.py: `train_step` / `shade`](../scientific_sources/material_a/network/renderer.py#L1638) | 1618–1621, 1638–1668 | Native GPU batches feed points/views/normals into shader; native RGB/material/light regularization is separate from the analytical benchmark |
| [field.py: `MCShadingNetwork.__init__`](../scientific_sources/material_a/network/field.py#L772) | 772–793 | Instantiates shared `MaterialFeatsNetwork` and five anisotropic heads |
| [field.py: `forward` / `anisotropic_forward`](../scientific_sources/material_a/network/field.py#L1603) | 1603–1619 | `anisotropy=True` selects `predict_anisotropic_components`, not the legacy isotropic helper |
| [field.py: `MaterialFeatsNetwork`](../scientific_sources/material_a/network/field.py#L709) | 709–738 | Spatial positional encoding, eight weight-normalized linear layers and a concatenated input skip |
| [field.py: `Embedder` / `get_embedder`](../scientific_sources/material_a/network/field.py#L16) | 16–63 | Includes raw coordinates and sine/cosine frequencies; no learned encoding weights |
| [field.py: `make_predictor`](../scientific_sources/material_a/network/field.py#L319) | 319–356 | Four weight-normalized linear layers, three ReLUs and default final sigmoid |
| [field.py: `predict_anisotropic_components`](../scientific_sources/material_a/network/field.py#L1198) | 1198–1219 | Concatenates shared features **and raw points** for each head; decodes widths/color/metallicity/orientation |
| [field.py: `shade_anisotropic_mixed`](../scientific_sources/material_a/network/field.py#L1540) | 1540–1601 | Samples directions, queries learned illumination, computes linear radiance, then native display/sRGB outputs |

### Architecture and material decoding

`get_embedder(8,3)` produces 51 coordinates: raw xyz plus sin/cos for frequencies
1,2,4,...,128 on each axis. `module0` is `51→256→256→256→256`, with ReLU after
every linear. Concatenate its output with the **51 encoded input channels**;
`module1` is `307→256→256→256→256`, with ReLU after its first three linears and
no activation after its last linear. The feature output is 256 channels.

Each predictor receives `[features, raw_xyz]`, dimension 259, and has topology
`259→256→256→256→d`. All four linears use the original `nn.utils.weight_norm`;
the three hidden activations are ReLU and the final activation is sigmoid.
There is no dropout or batch normalization in these material components.

| Predictor | Output channels | Physical interpretation |
| --- | --- | --- |
| `mx_predictor` | 1 | `mx=.04+.96*sigmoid(raw_x)` |
| `my_predictor` | 1 | `my=.04+.96*sigmoid(raw_y)` |
| `metallic_predictor` | 1 | `metallic=sigmoid(raw_m)` |
| `kd_predictor` | 3 | Base color `kd=sigmoid(raw_rgb)`, not already multiplied by `(1-metallic)` |
| `rotation_predictor` | 2 | `q=sigmoid(raw_rot)`, `v=2*q-1`, `rotation=normalize(v,eps=1e-6)` |

Thus **eight predictor channels encode seven physical degrees of freedom**.
The orientation radius is redundant physically but changes gradients and
conditioning. It is not equivalent to a single unconstrained scalar theta.
Widths are the active anisotropic slope widths in [.04,1], not squared widths;
do not import the isotropic export helper's squared-roughness/square-root
convention. Bounds at infinite logits are limiting values; finite-gradient
and nonzero-encoding gates are still needed.

Static parameter counting, including each weight-normalized linear's `weight_v`,
`weight_g` and bias, gives 488960 shared-feature parameters and 996624 predictor
parameters: **1485584 total**, excluding all illumination networks. This count
has not been verified by constructing a live N1 model in Phase A.

### Tangent frame, encoding and shader identity

At [field.py:897–931](../scientific_sources/material_a/network/field.py#L897),
the sources=None basis is `T=normalize(n×X)`, with Y fallback, and
`B=normalize(n×T)`. For n=+Z, T=+Y and B=−X. Rotation is
`T'=cos(theta)*T+sin(theta)*B`, `B'=-sin(theta)*T+cos(theta)*B`
([1304–1307](../scientific_sources/material_a/network/field.py#L1304)).
Double-angle encoding is `(cos(2theta),sin(2theta))`; theta+pi and swapping
widths with theta+pi/2 preserve the physical roughness tensor, not necessarily
an uncoupled finite-sample rendering realization.

The **native frozen A** sampler reconstructs half-angle coefficients with
square roots/clamps/sign at [1330–1335](../scientific_sources/material_a/network/field.py#L1330).
The **accepted experimental 3D-1** shader instead explicitly uses A2/atan2:
[material_shader_v1.py:46–52](../experiments/material_shader_v1.py#L46),
[107–121](../experiments/material_shader_v1.py#L107),
[149–177](../experiments/material_shader_v1.py#L149).
It uses half-atan2 and preserves its existing zero/near-zero encoding rejection.
These are distinct, versioned decoders. Proposed 3D-2 keeps the accepted
experimental decoder unchanged for **all** arms; it does not switch N1 back to
the historical half-angle implementation or claim native A rendering parity.

A2 already contains approved experimental corrections relative to native A:
cosine diffuse sampling with midpoint radial heights and total-count specular
normalization. Their exact scope is in
[anisotropic_estimator_v1.py:26–67](../experiments/anisotropic_estimator_v1.py#L26)
and [material_shader_v1.py:115–120](../experiments/material_shader_v1.py#L115).
Native [field.py:1464](../scientific_sources/material_a/network/field.py#L1464)
uses valid-count normalization. Preserve these identities rather than describing
A2 as untouched native A algebra. No further correction is proposed in 3D-2.

Despite names such as `sample_aniso_ggx_directions`, the active sampling/PDF
uses the anisotropic exponential-in-tan-squared formulation
([1285–1300](../scientific_sources/material_a/network/field.py#L1285),
[1342–1355](../scientific_sources/material_a/network/field.py#L1342)). Names of
other GGX/Smith helpers do not select the active anisotropic model. Do not
substitute GGX/Smith. Retain the accepted Fresnel, clamps, masks, PDFs and
known-input illumination interface. Native kd is the base color; native
`diffuse_term` actually returns `kd*(1-metallic)` at
[1517–1538](../scientific_sources/material_a/network/field.py#L1517).

### Inactive helpers and native dependencies

[mlp_brdf.py](../scientific_sources/material_a/network/mlp_brdf.py) defines a
generic `MLPNetwork`, but a source-wide search finds no material execution-path
use. Its `SequentialNetwork`/`Network` parents build a ModuleList, not the active
eight-layer shared trunk/five heads. Do not choose a generic `[3,64,64,7]` MLP
and label it original A. `SequentialNetwork.build` uses `torch.randn` while its
module imports only `torch.nn`; that latent inactive issue is documented, not
repaired or executed here.

The native field module imports nvdiffrast, mcubes, Open3D and SciPy; full shader
construction calls `.cuda()` for sampling buffers at 822/828. The native renderer
constructs geometry/raytracing and native learned lighting. These are unnecessary
for constant known-plane material recovery. CPU feasibility therefore depends
on a future reviewed extraction of exact CPU-capable material definitions, not
calling the native renderer or substituting a smaller network. The existing
experimental algebra already uses a reviewed source-AST admission pattern.

Native RGB training uses Charbonnier/L1 and material/light regularization
([renderer.py:1623–1664](../scientific_sources/material_a/network/renderer.py#L1623));
the controlled benchmark uses linear-RGB MSE and no regularizers. Native shader
energy/specular coefficients default to .05 in field.py:766–769; top-level YAML
.01 values are not automatically forwarded by `_init_shader`. No such losses or
native warm-up/cosine schedule should leak into N1. Component reuse is not reuse
of the entire native scientific training protocol.

## Design consequences and remaining uncertainty

1. Constant input must replace both spatial paths: trunk xyz encoding and raw
   xyz concatenated into every head. Replacing only one does not make a constant field.
2. N0 is seven trainable unconstrained coordinates (six logits plus raw theta)
   wrapped in a module. It is an identity/control representation, not evidence
   of meaningful hidden-network recovery. A `Linear(1,7)` is not an equivalent
   seven-parameter control if both weights and biases are trained.
3. Faithful N1 retains eight head channels and A's material decoding, then uses
   unchanged A2/atan2 shading. It tests architecture **plus output encoding**;
   differences cannot be attributed to architecture alone.
4. Exact six-scalar initialization and deterministic orientation matching are
   different requirements. Inverse sigmoid and two-channel normalization do not
   guarantee bitwise theta/render parity. Proposed bounded gauge-physical
   orientation matching needs approval and Phase C validation; it is not verified.
5. Original LM accepts only a seven-vector and 72×7 Jacobian
   ([optimizer comparison:153–164](../experiments/material_recovery_optimizer_comparison_v1.py#L153)).
   N0 fits this interface. For 1485584 N1 parameters, a dense float64 normal
   matrix would require **17655678568448 bytes** (~17.66 TB), before other state;
   its 72×P Jacobian alone is 855696384 bytes. A constant material's residual
   dependence has at most seven physical tangent directions, so neural weights
   are not individually identifiable even when physical parameters are locally
   full rank. Output-space LM or a new matrix-free/dual solver changes the
   optimizer/protocol; neither is part of the recommended primary comparison.
6. N1 initialization/gradient behavior, runtime, parameter count in a live model,
   CPU compatibility and all proposed 3D-2 admission gates remain **unexecuted**.
   No 5000-update direct or neural recovery outcome is predicted from this audit.

Source SHA-256 references: field.py `b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`;
renderer.py `dd82c61cb91a3d54e801f1fb32238e3b03a87e1ceb4c50f24f1b359e50cfd5e9`;
mlp_brdf.py `639e9f50a5ad7d335d50dc19ae2de0fe86d83235daf0796184b59cd55912ba82`;
experimental shader `de79ef3b0199b5f6e372feed6a9f16492ad966802b814e2cc9aab6700d35014f`;
recovery metrics `316e79b5e06615c9fa46e714f35f515b204c1f175c6c56636a6c4d9235ccabc4`.
