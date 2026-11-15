# E.10 — versioned native material-A corrections

**FIX AND PROCEED.** The corrections are selectable in the actual stage-aware A
renderer/trainer, not only the flat analytical shader. Frozen G/A and accepted
experiments/evidence remain unchanged. This is implementation and focused CPU
verification; no native model, GPU, reconstruction, dataset-reference production
or recovery experiment was executed. Native admission and empirical recovery
remain pending. See the [benchmark proposal](protocols/PAPER_ANISOTROPIC_BENCHMARK_E10_V1.md).

Base: `c5db1693625044e80bbd518bfc6fc0aa59bcb520`, branch
`feat/neural-material-early-step-e2a`. G remains
`79252178e65914ad1acbc14790b1514f276bfea0`; A remains
`e1e859779301faced46d6476691e20b546c27ae0`.

## Scientific identities and native dispatch

| Explicit variant | Estimator / decoder / numerical law | Physical allocation |
| --- | --- | --- |
| `A_original` | Exact original A methods and defaults | Original additive law |
| `A_estimator_v1` | Total-M specular; midpoint-cosine diffuse; original decoder, sampler guards and legacy regularizer diagnostics | Original additive law |
| `A_corrected_core_v1` | Estimator fixes plus atan2, domain rejection, consistent Gaussian slopes/PDF and finite-gradient safeguards | Original additive law; NOT generally energy conserving |
| `A_energy_budget_v1` | Same corrected core numerics | Separate reciprocal convex-budget model, below; never selected implicitly |

`run_stage_experiment.py --material-variant ...` is explicit; the default remains
original A. Geometry rejects material variants. The existing cold-process
StageFinder still selects pinned A datasets/cameras, networks, trainer, losses,
Adam, clipping, scheduler, validation/NVS and ray tracer. Only the material entry
in `name2renderer` is replaced after namespace admission. New candidate YAMLs carry the reserved `stage_variant_dummy_v1` dataset entry
guard. The stage launcher alone resolves it to the exact original `dummy` class;
legacy Trainer dispatch rejects it before network construction. No split, loader
or sampling policy changes. Legacy entry/source files remain unchanged.
Module-global original
`MCShadingNetwork` and original renderer class remain unchanged. Code is in
[scientific_variants/material_a_v1.py](../scientific_variants/material_a_v1.py);
metadata-only validation/selection is in
[material_variants.py](../infrastructure/material_variants.py).

The subclass inherits all native material/illumination network definitions. It
calls the native BVH/visibility/direct/indirect/human-light interfaces, original
Fresnel and metallic/base-color transforms, then original linear-to-sRGB. It does
not import the A2 experimental shader or substitute its known-flat-geometry
lighting callback into ordinary native training. With `learned_native`, original
lighting is unchanged. Known environment and zero-bounce overrides are explicit
**benchmark factors**, not BRDF repairs; they are described separately below.

The original-A known-light comparator uses the *identical original method
objects* for material prediction, frames, both samplers, NDF/PDF, radiance,
shading and regularization. Only its lighting interface and safety/provenance
wrapper differ. This enables fair known-light comparison without labeling a
corrected sampler as original A.

## Confirmed repairs and their limits

Let d=k(1−m), F0=.04(1−m)+mk, c_i=n·wi, c_h=n·h. Keep Beckmann slope widths
in [.04,1], Kurt a=0, ks=1 in core, Schlick Fresnel and the original RGB loss.

- **Specular count:** core/B estimate is `(1/M) sum_j L_j F_j max(c_i,0)/c_h`,
  including zero contributions from rejected reflections. No acceptance-count
  gain. B retains original sampling/light directions and count; core also applies
  explicitly versioned numerical repairs. All drawn samples still enter native
  lighting/visibility queries; there is no valid-only redraw or ray-count change.
- **Diffuse:** original golden-angle azimuths, bases, sample counts and one
  training azimuth shift per point remain. New heights z=(j+.5)/N give
  cos(theta)=sqrt(z), sin(theta)=sqrt(1−z). No extra RNG draw. Constant light
  cannot distinguish the old defect; directional-light moments can. Midpoint
  integrates z exactly, but z² has deterministic error −1/(12N²); finite
  quadrature bias is not estimated by azimuth-seed variance.
- **Orientation:** theta=.5 atan2(r_y,r_x), normalized double-angle vector.
  Raw `2*rotation_raw−1` is checked *before* original epsilon normalization can
  hide its small norm. Norm<1e-6, zero, nonfinite or overflow stops the pending
  update before Adam; no arbitrary angle, dropped point, new draw or retry.
  Negative-axis frames stay orthonormal. The half-angle frame sign jumps across
  its chart cut; physical periodicity is not bitwise finite-tape invariance.
- **Sampler/PDF:** core uses elliptical Gaussian Cartesian slopes, algebraically
  the same continuous Beckmann distribution as A's spherical inversion. Two
  uniform requests, shape/order/count unchanged. Signed reflection replaces the
  invisible-facet collapse to −wo; invalid contributions remain zero. Reflected
  h/wi are unit within arithmetic error. The source ring atom from clamping all
  U<1e-6 is removed: only exact U=0 maps to float32's smallest normal positive
  value. Torch already draws below1; explicit float64 tapes use E.8R's single
  nextafter mapping when valid values round to float32 1. Invalid uniforms fail.
- **Density:** normalized ideal Beckmann D and `D*c_h/(4*abs(wo·h))` replace the
  old denominator floor and lifted exponential tail. Exact zero Jacobians are
  reported as zero density (measure-zero), not epsilon-clipped into a continuous
  PDF. Upper-reflection mass is acceptance probability, not necessarily1.
  Cancelling BRDF/PDF weights are retained on their intended valid domain.
- **Domain/safety:** nonunit/degenerate normals or views, lower-hemisphere outgoing
  views, invalid widths/colors/metallicity, negative/nonfinite incident light,
  and undefined explicit tangent sources fail. Unit-direction tolerance1e-5 is
  a validation bound, not a normal repair. The original X/Y reference switch
  remains, including its seam; no global tangent continuity or normal flip is
  claimed. Finite-gradient hooks reject nonfinite backwards without modifying
  finite derivatives. Native default-tensor-type restoration now also occurs
  on corrected-renderer failure.

Numerical core changes are not a claim of bitwise A2 finite-tape parity. The
accepted A2 construction/estimator/atan2 formulas transfer; core additionally
removes documented sampling/density approximations. At regular inputs, new tests
compare the production methods to independent Cartesian integration/PDFs and
actual shader autograd, with synthetic native lighting/BVH stand-ins. Mask
transitions and finite-precision uniform atoms still qualify unbiased-gradient
claims; local finite differences do not prove global estimator-gradient validity.

## Explicit regularizer versions

All corrected shader coefficients live **only** in `shader_cfg`; duplicate
historical top-level energy/specular/anisotropy declarations are rejected rather
than silently forwarded. Counts, flags and coefficients must be explicit and
well typed. Explicit geometry_type,reg_min_max,max_n_exp,max_alpha_exp are
rejected in corrected configs because they have no active anisotropic consumer;
their inherited historical defaults are not advertised as working controls.
Original A continues to ignore its old top-level declarations and
use its historical nested/default .05 behavior; no old YAML was migrated.

| `regularization_version` | Meaning |
| --- | --- |
| `legacy_v1` | Conditional `f_s_sum`, detached-softmax `L_spec`, original reg_change gate and inverse-count scale; required by estimator-only isolation |
| `disabled_v1` | Renderer reg_mat=false and all material-prior flags off, explicitly; production defaults for controlled data-only core comparisons |
| `unconditional_v1` | Original spatial-prior algebra, plus independent energy/allocation gates and explicit new semantics |

In unconditional mode, `f_s_sum=mean_all_M(F*c_i/c_h*mask)` includes the energy
budget when selected, independent of incident illumination. The energy loss is
lambda_E * sum_RGB(relu(d+f_s_sum−1)) per point. Native loss aggregation averages
points as before. It trains roughness/orientation through sampled weights as
well as color/metallic; no indirect illumination-network constraint is implied.
It is a noisy view-conditioned reflectance prior, not all-direction conservation.

The old specular prior's intent is ambiguous. Its **replacement candidate** is
explicitly a diffuse-allocation prior:
`L_alloc_RGB = d_RGB * stop_gradient(f_s_sum_RGB) / 3`, loss
lambda_S * sum_RGB(L_alloc). No extra Nvalid or M division. Repeating identical
samples leaves its value unchanged; its direct gradients reach d (kd/metallic),
not detached roughness/orientation or illumination. Its scale/meaning is *not*
the historical specular penalty. Default paper controls disable it. The separate
[prior-candidate config](../configs/material_corrected/bell_core_prior_candidate_v1.yaml)
enables it and the energy prior even with reg_change=false; coefficients are
uncalibrated, not a recommended recovery intervention.

Original spatial perturbations, Gaussian/constant options and the unusual
orientation decay `1e-4+9e-7*(1+cos(pi*step/100000))/2` are preserved, not repaired
as an incidental cleanup. Renderer reg_mat still gates all material priors;
renderer diffuse-light whiteness remains separate. No optimizer/scheduler,
clipping, network initialization or geometry loss was changed.

## Separate physical model: reciprocal convex energy budget

The [accepted counterexample](evidence/ANISOTROPIC_BRDF_ANALYTICAL_V1.json) remains
true for core: m=.001,k=.999,mx=.6,my=.9 gives ideal normal total1.0177790177.
Sampler correction alone cannot fix additive allocation. A Fresnel multiplier
on diffuse alone is not a proof. Kurt's [primary paper](https://cgg.mff.cuni.cz/~jaroslav/papers/2010-anisobrdf/2010-anisobrdf.pdf)
distinguishes its normalized Beckmann model, a=0 fitting kernel and alternative
couplings; blindly inserting Smith G or Eq9 would change the chosen model.

Here is an independent bound for its **a=0 specular component**. Under projected
NDF measure q(h)=D(h)c_h, pair h±=±u+t*n with equal density, t=c_h>0. Let
co=n·wo, A=co*(2t−1/t), B=2*wo_tangent·u. With unit Fresnel, the pair's reflected
integrand average is `J=((A+B)_+ +(A−B)_+)/2`.

- A≥|B|: J=A≤co≤1, since 2t−1/t≤1.
- A≤−|B|: J=0.
- Otherwise J=A/2+|B|/2≤co*(t−1/(2t))+sqrt(1−co²)*sqrt(1−t²).
  For t≤1/sqrt(2), the first term is nonpositive, so J≤1. For larger t,
  Cauchy–Schwarz bounds the expression by1/(2t)≤1.

Thus J≤1. The normalized, tangent-even Beckmann measure integrates to1;
Fresnel in[0,1] cannot increase the bound, so rho_s≤1 for all unit upper views
and allowed widths. Reciprocal fs follows from wi·h=wo·h and even D.

Define the **separate** model
`f_budget = d/pi + (1−d)*fs_Kurt(F0,mx,my,orientation)` per RGB channel.
Then `rho_budget=d+(1−d)*rho_s≤1`, and reciprocity/nonnegativity are preserved.
This is a direction-independent convex reflectance budget, not output clipping,
measured-rho normalization, a fitted safety factor or a replacement NDF. The
same d/F0 numeric maps and Beckmann widths remain explicit, but specular
allocation becomes ks=1−d. In particular, high diffuse allocation suppresses
specular amplitude. It is an abstract passive mixture, **not an exact layered
plastic model**; combined-interface reflectance is no longer an unconstrained
F0 amplitude. This modeling limitation must accompany parameter interpretation.

The production energy variant applies the budget to every specular radiance and
reflectance-proxy contribution; diffuse is unchanged. Export includes both
budgets. Core never applies this factor. Independent unit tests check the pairing
bound, reciprocity and18 directional integrals on two fixed grids; native-path
fixtures check the isolated allocation change and finite backward. The theorem
assumes exact unit frames/normalized continuous D. Float32/sample discretization
is an approximation; an individual finite MC estimate may exceed1. There is no
RGB clamp to conceal it. This establishes model-level admissibility under stated
assumptions, not recovery, measured-material fidelity or native GPU acceptance.

## Native training, extraction, relighting and checkpoints

The CLI still uses schema1, PathResolver, RunStore, stage-owned cache/log/event/
render/checkpoint destinations and explicit read-only mesh selection. Ordinary
runs still require the strict same-scene G handoff. A separate `--oracle-benchmark`
admission is only for reviewed, independently generated known-material datasets;
it checks dataset/mesh/labels/generation hashes, scene, target family and known
lighting/bounce consistency. It never creates G provenance from reference assets.
Existing historical images without labels/generation evidence fail that gate.

`--material-smoke` reuses TwoUpdateTrace and original Trainer.run: exactly two
successful Adam returns, stopping before validation/checkpoint/NVS. The corrected
probe uses actual selected method objects and the reviewed inspect.unwrap
forbidden-call guard. Original warm-up gives LR0 then5e-7; original clipping.7,
eps1e-4, AMSGrad and scheduler remain. Known-light networks are explicitly
expected inactive; learned-native mode retains original light-path accounting.
No CPU fixture is described as native execution evidence.

`predict_materials()` now exports linear mx,my,metallic,albedo,double-angle
rotation and 2x2 local-frame Q; energy mode additionally exports budgets. No
isotropic roughness square root or gamma encoding of physical coefficients.
For the original known-light comparator, Q is the nominal predicted parameter
tensor; it does not certify the defective original decoded frame. Report actual
frame validity/BRDF behavior separately. Core/energy require valid unit frames.
This is the selected native renderer API; old standalone `extract_materials.py`
and its isotropic Blender pipeline remain unchanged and must not be used as a
corrected exporter. World Q additionally requires the declared geometric T/B.

`renderer.relight(..., environment, environment_identity)` uses original geometry,
visibility and sampler, with explicit outer-radiance callback and identity in
outputs. `learned_native` retains learned secondary light: changing outer light
does not solve indirect relighting. The first benchmark instead explicitly uses
`known_env_v1/zero_bounce_v1`, human_lights=false, retaining BVH shadows while
zeroing secondary bounces. Both original and corrected arms use that same factor.
The known analytic environment is implemented, not a placeholder configuration.

One persistent32-byte `_a_variant_signature` binds base/derivative source identity,
variant and effective shader config. Neural definitions/learned capacity are
unchanged. Parent renderer loading checks the tag **before** weights; only strict
loading is accepted. Old-A, another variant, different shader settings or missing
tag is rejected. No historical checkpoint was loaded or repurposed. Current
stage execution remains fresh-only; scientific optimizer-resume admission is
not added. Original ordinary-A checkpoints and methods remain unchanged.

## Evidence, readiness and publication priorities

The [verification package](evidence/NATIVE_MATERIAL_E10_VERIFICATION_V1.json)
contains exact source/test/config hashes, actual process exit, full JUnit,
inventory, numerical fixture properties, static checks and preservation evidence.
Final focused result: **55 passed,0 failed,0 errors,0 skipped**, process exit0;
52 new cases and3 existing compatibility contracts. One expected deprecation
warning concerns default-tensor-type restoration on Torch2.4. The full native/
N1 import blocker stayed intact. An early synthetic seam fixture expected90°
while varying the normal along a180° sign-switch path; its test geometry was
corrected without changing scientific thresholds or hiding the seam. Preliminary
records are preserved separately from the final passing run.

| New synthetic numerical check | Observation |
| --- | --- |
| Actual corrected-shader local gradient versus finite difference | max absolute error7.88495e-5, within fixed rtol.04/atol3e-4 |
| mx=.3,my=.7,normal view,constant F=.04 | MC .0272332672; independent integral .0272545019 |
| mx=.6,my=.9,80° view,constant F=.04 | MC .0161423907; independent integral .0161799086 |
| Independent two-grid differences in those broad-lobe tests | 1.28e-16 /4.51e-17; fixed convergence bound2e-5 |
| Largest budget-family reflectance among18 integrated cases | .9980667488, within the fixed energy/convergence bound |

These are small unit-fixture observations, not reconstruction, recovery or
independent dataset-reference production. The existing B0/B1/B3 experiment and
its unconverged cases remain unchanged. The new estimates use32768 fixed draws
per case and retain the original audit's .004 absolute fixture comparison bound.
No old ablation or recovery panel was rerun.

Tests run on host Python3.9/Torch2.4 **CPU**, not native Python3.8/Torch1.8.
Python3.8 grammar is checked; `torch.linalg.norm` exists in the
[official1.8.1 source](https://raw.githubusercontent.com/pytorch/pytorch/v1.8.1/torch/linalg/__init__.py).
API inspection and fixtures do not substitute for the native gate.

Use core for controlled implementation-repair comparisons. For claims requiring
passive physical materials, review the separately bounded model/parameter meaning
and independent matched-family targets before choosing it as a paper primary.
Do not tune priors or call this an empirical reconstruction improvement. A2's
estimator/decoder results support the repairs, but E.9 did not establish a spatial
optimizer advantage and none is introduced here. Old ablation, gauge, recovery
and energy evidence is reused unchanged; no full ablation is repeated.

Rank remaining research work: (1) admit oracle geometry/normals and independent
known-light targets; native correctness gate; (2) gauge-aware, spatially coherent
material recovery with fair full/head controls; (3) physically motivated parameter
conditioning; (4) sensitivity-aware illumination/observations; (5) geometry-to-
material uncertainty; (6) diffuse/specular disentanglement versus independently
measured material data. Repairs, atan2 and standard convex mixing are not novelty.
Any method claim needs recovery, relighting, seed/scene replication and cost.

No new scientific execution is authorized by this report. The smallest next
execution proposal is the explicitly bounded native smoke **after** a generated
oracle package or attributable G input is admitted; new image/reference generation
needs its own concrete frozen implementation/budget approval first. No placeholder
manifest is offered as complete evidence.
