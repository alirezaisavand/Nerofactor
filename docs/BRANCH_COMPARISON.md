# Historical branch comparison

Inspection date: 2026-11-03. This is a static comparison of locally available
Git objects, not a numerical equivalence test. No fetch, merge, training,
evaluation, or algorithm modification was performed for milestone 1D.1.

## References and comparison scope

| Label | Inspected ref | Exact commit |
| --- | --- | --- |
| A | `anisotropic` and `origin/anisotropic` | `e1e859779301faced46d6476691e20b546c27ae0` |
| G | `origin/aniso_geometry_safe` | `79252178e65914ad1acbc14790b1514f276bfea0` |
| B | `origin/ablation_anisotropic` | `2519ab9b480f414ba2ae96dd860d8b868c584054` |

`paper/infrastructure` starts at A. G and B are remote-tracking snapshots; their
current upstream freshness has not been checked. Do not infer equivalence or
correctness from the words `safe` or `ablation`.

| Comparison | Common ancestor | Commits exclusive to A / other ref | Changed paths |
| --- | --- | --- | --- |
| A versus G | `92b9e4ba8b66830dcda1025abb181cd75edc043c` | 359 / 6 | 43 |
| A versus B | `12a52ab215747b7d93df8d392babfafdf0492dc0` | 21 / 8 | 7 |

These are endpoint tree comparisons, not a claim that the other branch contains
all of A plus the listed exclusive commits. Reproduce the inspection with
`git diff anisotropic..origin/aniso_geometry_safe -- <path>` and
`git diff anisotropic..origin/ablation_anisotropic -- <path>`.

## Algorithmic differences

| Area and source | A | G | B |
| --- | --- | --- | --- |
| Material heads, `network/field.py:MCShadingNetwork` | Predicts `mx`, `my`, metallic, `kd`, and a rotation pair; remaps roughness to `[0.04, 1]` | Predicts `mx`, `my`, `alpha`, `F0`, `kd`, `ks`, and rotation; no corresponding `[0.04, 1]` remap; adds second-lobe heads | Same material-head code as A |
| Rotation representation, `predict_anisotropic_components` and `sample_aniso_ggx_directions` | Remaps rotation output to `[-1, 1]`, normalizes it, interprets it as cosine/sine of twice the angle, and reconstructs a half-angle using square roots and a sign | Normalizes the predictor pair without A's `[-1, 1]` remap and uses its components directly as cosine/sine | Keeps A's predictor transformation but uses the pair directly as cosine/sine instead of reconstructing the half-angle |
| Tangent construction, `network/field.py` | Normal/reference-vector construction, optional `sources` argument, fallback threshold `1e-6`, explicit normalization epsilon `1e-6` | Removes `sources` argument from this helper; retains normal/reference-vector construction with different normalization defaults | Changes fallback threshold from `1e-6` to `0.1` |
| Mesh frame infrastructure, `network/renderer.py` and `network/field.py` | No G-style PCA/adjacency/KD-tree frame pipeline | Computes mesh normals, adjacency, PCA vertex frames, and triangle KD-tree; adds interpolation helpers and passes mesh/frame data to shading | Same renderer tree as A |
| Active frame selection in G | Not applicable | The inspected primary sampler calls `compute_tangent_bitangent_flat(normals)` and then rotates that frame; the mesh-frame interpolation calls in that sampler are commented out | Not applicable |
| Distribution/PDF numerics, `compute_Dh`, `compute_pdf_aniso_ggx` | Clamps grazing-angle denominators, exponent range, and view/half-vector dot products; uses explicit epsilons | Removes or changes those clamps and uses different epsilon/denominator treatment; adds alternative sampler/PDF helpers | Same implementations as A outside the rotation/threshold differences |
| Radiance and regularization, `network/field.py` | Metallic-based material computation and associated regularization inputs | Different `alpha`/`F0`/`ks` radiance and regularization inputs, plus multi-lobe paths | Same code as A for these functions; configuration weights differ |
| Output interfaces, `network/renderer.py`, `network/metrics.py` | Exposes metallic and `f_d` among material outputs | Exposes `ks`, `F0`, `alpha`, and `f_d_sum`; different visualization/output expectations | Same interfaces as A |

Mesh-frame infrastructure existing in G does not demonstrate that the inspected
experiments used interpolated PCA frames. Different material heads also preclude
assuming checkpoint interchangeability between A and G. B having the same head
shapes as A does not establish semantic checkpoint compatibility.

## Optimization-related differences

| Setting and source | A | G | B |
| --- | --- | --- | --- |
| Optimizer construction, `train/lr_common_manager.py` | Passes `lr=1e-3`, `eps=1e-4`, `amsgrad=True` | Passes only `lr=1e-3`; remaining settings depend on the selected optimizer/library defaults | Same as A |
| Warm-up/cosine defaults | `lr=5e-4`, warm-up 5,000, configured `end_iter=300000`; alpha `0.2` | Same declared learning-rate/warm-up/end defaults; alpha `0.05` | Same as A |
| Actual cosine horizon | Hardcoded 100,000; clamps the step at that horizon instead of using configured `end_iter` | Uses configured `end_iter` and does not apply A's horizon clamp | Same as A |
| Gradient clipping, `train/trainer.py` | Clips parameter gradient norm at `0.7` | No corresponding clipping call | Same as A |
| Opacity loss schedule, `network/loss.py` | Begin weight `0`, end weight `0.01`; schedule begins at step 10,000 and ends at 50,000 | Begin weight `0.001`, end weight `0.01`; schedule begins at 20,000 and ends at 50,000 | Same as A |
| Shader regularization defaults | `change_eps=0.02`; energy/specular coefficients default to `0.05` | `change_eps=0.05`; coefficients default to `0.01` | Same defaults as A |

The construction learning rate is subsequently set by the scheduler during
training. Record both construction arguments and the actual per-step schedule.
The common factory also offers SGD in the trainer while A supplies Adam-specific
arguments: document and validate supported combinations rather than repairing
that path during infrastructure cleanup.

## Configuration-related differences

Examples below are endpoint values, not a complete resolved-config manifest.
Nested shader defaults and per-scene overrides must be traced together before
declaring a legacy preset reproduced.

| Configuration | A | G | B |
| --- | --- | --- | --- |
| `configs/material/real/bear.yaml` | Pretrained mesh; 1,024 specular samples; validation/save every 2,000 | Reconstructed mesh; 256 samples; validation 1,000/save 500 | Same as A |
| `configs/material/real/bunny.yaml` | Pretrained mesh; 1,024 samples; explicit energy/specular coefficients `0.01`; `n_lobes: 2` | Reconstructed mesh; 256 samples; adds `fixed_camera: true`; removes those explicit overrides | Same as A |
| `configs/material/syn/angel.yaml` | Top-level energy/specular flags false with zero coefficients; 1,024 specular samples | Removes those top-level declarations, uses reconstructed mesh and 256 samples; dataset path becomes a placeholder | Top-level flags true with coefficients `0.01`; otherwise retains A's sampling settings |
| `configs/material/nerf/smoothcylinder.yaml` | Energy/specular flags true but coefficients zero; 1,024 samples; 50,000 steps; validation/save 2,000 | File absent | Coefficients `0.01`; 512 samples; 20,000 steps; validation/save 5,000 |
| `configs/material/nerf/donut.yaml` | 100,000 steps; validation/save 2,000 | File absent | 80,000 steps; validation/save 5,000 |
| `configs/shape/nerf/smoothcylinder.yaml` | Includes opacity loss; 300,000 steps; validation/save 1,000 | Removes opacity from loss list; 100,000 steps; validation/save 5,000 | Same as A |
| `configs/shape/real/bunny.yaml` | Includes opacity, curvature, and mask losses; validation 1,000 | Removes those three losses; validation 5,000; dataset path becomes a placeholder | Same as A |

G also lacks A's material configs for cube, sphere, and tetra and shape configs
for cube and donut. It changes additional real/synthetic configs, mesh choices,
and validation/save intervals. B lacks A's material tetra config. File absence
does not establish that an experiment was never performed.

The A tip records disabled/zero top-level regularizer declarations for angel and
smoothcylinder. **These declarations do not establish the effective shader
weights.** Follow-up CPU-only characterization in 1D.2A confirms that A's
`NeROMaterialRenderer._init_shader` forwards only `shader_cfg` plus derived
`is_real`; it does not forward those top-level regularizer fields. In a fresh
shader, the default energy/specular coefficients remain `0.05` unless overridden
inside `shader_cfg`. B has the same renderer setup. The exact producing code for
historical runs is still unknown. No field is relocated by this finding.

The B tip also changes multiple controls beyond angle encoding, so comparing
tip-level metrics is not automatically an isolated angle ablation.

## Dataset/evaluation-related differences

| Area | A and B | G | Consequence |
| --- | --- | --- | --- |
| `dataset/database.py:get_database_split` | Seeds global Python RNG with 6033, shuffles loaded IDs, reserves five NVS IDs and one validation ID, returns three lists | Validation uses first loaded ID without shuffling, trains on the remaining IDs, returns two lists | Different train/validation membership and return contracts |
| Loaded NeRF metadata | `NeRFSyntheticDatabase` loads only `transforms_train.json` | Also loads the train split in the inspected implementation | NVS naming is not evidence that `transforms_test.json` was used |
| NeRF intrinsics | Vertical focal value is `focal * (H/W)` | Vertical focal value is `focal` | Requires camera-convention verification, particularly for non-square images |
| NeRF depth | Reads `depth/ImageNNNN.png`, scales by `15/65535`, masks values below `14.5` | Returns scaled `torch.randn(800, 800)` values and uses image alpha as mask | G's synthetic depth is a placeholder, not interchangeable geometry ground truth |
| Dataset GT helper | Defines `get_database_eval_points_nerf` but no `get_database_eval_points` | Defines GlossySynthetic `get_database_eval_points`, which can create `eval_pts.ply` in the dataset | A/B synthetic evaluator imports an absent helper; G can write inputs |
| Real geometry evaluation | Reads point clouds | Reads triangle meshes and evaluates their vertices | Different point populations and density weighting |
| Material NVS | End-of-training NVS method and logs present | Corresponding material NVS method/call removed | Output coverage and metrics differ |
| Evaluation scripts | A has masked NVS and two NeRF shape evaluators | These three scripts absent | Evaluation availability is not equivalent |
| B versus A outputs | A uses `logs/<name>` and `nvs_imgs/<name>` | Not applicable | B uses `logs_ablation/<name>` and `nvs_imgs_ablation/<name>`; checkpoint path remains shared by name |

The A/B masked evaluator has its JSON write commented out while printing a saved
path. Its metric computation masks both images and then averages over the full
image; that is distinct from foreground-pixel-only PSNR. Preserve and name the
legacy protocol before considering a different definition. Existing historical
JSON files do not prove that this exact tip produced them.

## Obsolete or unconfirmed utility code

This category identifies review candidates, not authorization to delete code.

- G adds `click.py`, an interactive Matplotlib point-selection utility using
  `004.jpg` in its example entry point. Its role in reported experiments is
  unconfirmed.
- A/B contain machine-named runners (`run_all_mariana.sh`,
  `run_all_marineris.sh`, `run_all_tonga.sh`); G lacks them. They select different
  scene lists and are historical launch aids, not provenance manifests.
- Commented alternative tangent and segmentation paths, debug prints, and G's
  extra sampler/interpolation helpers need call-site review. A helper being
  defined is not evidence of execution. Do not activate it during cleanup.
- Legacy PointNeRF absolute paths/options in A's drums material configuration
  appear unrelated to the inspected NeRO entry point; their intended use is
  unconfirmed. Preserve them pending a configuration audit.

## Mathematical verification required before selecting or correcting behavior

1. Derive the angular periodicity and half-angle reconstruction for anisotropic
   axes, including sign ambiguity, zero crossings, predictor ranges, and gradients.
   A, G, and B use different mappings; no equivalence is established.
2. Check tangent-frame handedness, orthogonality, normal alignment, reference-axis
   switching, mesh-frame sign ambiguity, and interpolation continuity. Separate
   active normal-based frames from unused mesh-frame alternatives.
3. Match the implemented normal-distribution function, random sampler, and PDF.
   Some helper names say GGX while the primary formulas contain exponential
   slope terms; names alone do not identify the intended model. Verify PDF
   normalization, half-vector-to-direction Jacobian, hemisphere treatment,
   mixture weighting, and grazing-angle stability.
4. Verify material-head ranges, metallic versus explicit Fresnel/specular
   parameterization, lobe combination, energy accounting, regularizer scaling,
   and linear/sRGB transitions. Similar rendered images cannot establish model
   or checkpoint equivalence.
5. Trace each loss and its effective coefficient/schedule through defaults and
   overrides, including disabled losses and zero coefficients. Verify the cosine
   schedule beyond warm-up and its nominal endpoint before changing the horizon.
6. Establish pose direction, scene scaling, depth-as-ray-length versus camera-Z,
   intrinsics, masks, and mesh coordinate conventions before comparing geometry
   errors. Random placeholder depth must not be accepted as ground truth.
7. Define benchmark splits and metrics independently of checkpoint selection.
   Check point sampling/density and full-image masked versus foreground-only
   reductions; preserve historical protocol labels even if corrections follow.

These are documented investigation items for Step 2 scientific review. They are
not fixes included in 1D.1 or implicit permission to alter behavior in 1D.2.
Historical artifact coverage and reported values are recorded separately in
[BASELINE_INVENTORY.md](BASELINE_INVENTORY.md); implementation sequencing is in
[INFRASTRUCTURE_PLAN.md](INFRASTRUCTURE_PLAN.md).
