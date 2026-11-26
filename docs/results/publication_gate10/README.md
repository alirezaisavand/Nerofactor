# Gate 10 preparation and method readiness

**PROCEED to one separately authorized full G run.** Gate 9 is accepted at
98fc5c8f857dfb10b938f5295596d6e6c50ffd6f. Preparation uses authoritative G
79252178e65914ad1acbc14790b1514f276bfea0 and admitted Bell v3 with its
122/1/5 split. The new protocol binds all inputs and a fresh evidence root.
Training, extraction and GPU validation remain unexecuted.
[Full protocol and schedule table](../../protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.md)

The fixed publication endpoint is 300k. The 1k natural boundary tests execution,
data access, validation/checkpoint integrity and cleanup behavior. It does not
admit early geometry quality. Native checkpoints at15k/30k/50k/100k/200k
are retained, including the roughly30k emergence period reported by the researcher.
The native 1k checkpoint is also retained. These require only file copies of
already scheduled saves. There are no additional training or rendering calls.

## Scientific source inspection

The effective G schedule has important interactions. Early RGB weight peaks
at5k, initialization penalties end at1k, inverse variance unfreezes at15k,
the mask weight finishes increasing at30k, occlusion and opacity weights finish
increasing at50k, and the NeuS alpha cosine finishes annealing at50k. Curvature
weight is constant .001, despite configured ramp endpoints. Foreground/background
auxiliary losses are computed but are absent from the selected loss registry.
The source coefficients, exact reductions and LR equations are in the protocol.

The researcher's isotropic-appearance warning is supported by the lack of a
physical tangent director, but the pinned source includes a partial modification.
AppShadingNetwork predicts roughness_x and roughness_y and concatenates two
integrated spherical encodings. Each is rotationally symmetric about the same
reflection vector. Its lookup and optional human light use
sqrt(roughness_x×roughness_y+1e−6). There is no tangent-angle-dependent physical
anisotropic NDF in this geometry appearance path. This makes directional
anisotropic highlights a plausible source of biased shape/normal estimates.
The existing schedules and geometry regularizers may limit that compensation,
but this run cannot isolate their individual causal effects.
[Source](../../../scientific_sources/geometry_g/network/field.py)

The source's learned direct/indirect appearance, background NeRF and sphere
initialization are preserved. Bell v3's true zero-indirect, anisotropic material
and L2 illumination belong to acquisition provenance and evaluation. They are
not substituted into G's learned appearance. Numerical failure is assessed from
finite losses/gradients, checkpoint identity, SDF and intersection witnesses.
Acquisition failures are assessed from safe calibration, ray replay and split
access. Optimization/schedule effects may be examined through saved weighted loss
trajectories and checkpoints. Residual correlations with reference anisotropy
cannot establish that the appearance approximation caused an error.

## Endpoint and historical evidence

The pinned bell.yaml explicitly selects300000 updates, validation/save1000.
Historical Bell train.txt has15000 logging rows and val.txt300, consistent with
every20/every1000 through300k. The historical event stream spans roughly11h.
Those logs inform a bounded runtime estimate only. Their source, producing
configuration and hardware remain unauthenticated. No historical checkpoint is
loaded for Gate10. The new36h training plus2h extraction/audit caps cover a
prospectively fixed endpoint without compressing scientific phases.

Original G extraction clamps the exterior of the unit sphere to+1. The oracle
Bell has65 vertices at radius≥1, with maximum radius approximately1.0002103.
The prior historical low-resolution extraction audit found unsupported SDF
witnesses. Gate10 explicitly tests raw SDF near the sphere boundary and preserves
all misses/correspondence exclusions. This support mismatch may prevent a handoff.
It is reported as a limitation and never repaired using oracle geometry.
[Historical diagnosis](../../GEOMETRY_SDF_MESH_AUDIT_V1.md)

## Prepared implementation and validation

The acquisition adapter stages only123 PNGs,123 masks,one validation depth and
numeric calibration. All fitting/validation dispatches are scoped and held
access is rejected. The original G source manifest is verified before and after
work. Separate train and evaluate containers preserve a complete source/seed/
configuration/acquisition/checkpoint/mesh chain. Only the new run's own1000 and
300000 checkpoints can be deserialized. A finite-checking callback observes
original extraction tiles without changing values or call ordering.

The endpoint audit retains six full rasters and excludes no difficult pixels
from accounting. Same-ray surface matching retains signed normals, geometric
distance, grazing and support exclusions. Raw-SDF witnesses separately assess
mesh zero-level, face convention and sphere caps. A future handoff is written only
after both quality and lifecycle checks, using the existing unchanged
infrastructure.stage_workspace.handoff validator. Its scientific scope is sampled
and visible-surface validation, with explicit incomplete or failed outcomes.

Focused CPU tests cover real admitted file staging and calibration parity,
forbidden data access, exact numerical loss/LR coefficients from original source
AST, signed-normal rejection, silhouette/support accounting, raw-SDF crossing/
normal failures, incomplete-cleanup receipt rejection and blocked scientific
imports. Final test exit/counts, JUnit identity and source validation are saved
in preparation.json. Native GPU behavior and actual full-run runtime remain
unmeasured.

## Downstream experiment after geometry acceptance

Use exactly this accepted mesh for chart-based K1/D2 and ambient project recipes,
with the same Bell camera inventory and observation radiometry. First prepare one
bounded known-material RGB geometry-sensitivity comparison with separately
recomputed visibility on oracle and reconstructed meshes, then a matched recovery
pair per selected recipe. Use identical initial weights, observation/index
inventories, known versus learned illumination status, updates and evaluation
supports within each geometry pair. Measure normal-induced Q transport separately
from surface correspondence changes. Choose the control before execution.

Existing dense heterogeneous Bell K1/Chart/Ambient runs use a different16-view
material acquisition. They supply context but cannot serve as an exact geometry
control for the full122-view Bell v3 acquisition. The future oracle control must
match the new acquisition and the selected material/illumination recipe.
Original A's eager all-camera access requires a scoped material acquisition
adapter before downstream training. Gate10 does not weaken that requirement.

The same122 fitting images, monitoring58 and five held images can support the
official ShinyNeRF comparison. Reuse Gate9's camera-ray conversion audit and enforce
the same split instead of its loader's validation/test alias. Freeze native
schedule, appearance metrics and tangent-eligibility conventions separately.
Compare appearance and geometry with declared mask/radiometry support. Native
ASG concentration/eccentricity retain their units. Calibrated physical relighting
requires an explicit incident-light interface that its audited release lacks.
[Gate9 feasibility and official-source references](../publication_gate9/README.md)

## Publication opportunities retained

| Priority | Experiment or methodological opportunity | Relevant source and controls |
| --- | --- | --- |
| Immediate | Authenticated original G, then matched geometry sensitivity and material/illumination recovery | Gate10 full schedule and handoff. Shared reconstructed geometry for all project representations. |
| Next | Cross-object full G→A and official anisotropic external comparison | Same acquisition/split, geometry and tangent conventions. Freeze before any scientific run. |
| Retained | Exact known-versus-learned Final-Ambient comparison | Isolate illumination status while holding all other recipe details fixed. Gate9 D2/D8 transfer provides separate evidence. |
| Retained | NeRO original100k schedule sensitivity | Preserve accepted10k evidence and full source behavior. Additional cost requires its own approval. |
| Retained | Additive-BRDF energy conservation | Audit the accepted physical model and published alternatives separately. No silent BRDF replacement or novelty claim for correctness fixes. |
| Retained | Observability-aware material recovery and richer acquisitions | Use fitting-only information, independent validation and higher-cue scenes/lights. Preserve prior failed sensitivity decisions. |
| Future method | Anisotropy-aware geometry appearance | AppShadingNetwork.predict_specular_lights, forward and predict_materials in G field.py. Explore tangent-oriented directional convolution or anisotropic directional bases instead of reflection-axis symmetric encodings. |
| Future method | Ambient tensor integration in geometry | Couple a bounded projected ambient tensor to directional appearance and SDF-derived normals, with finite gradients near isotropy and a physical tangent convention. Reuse Final-Ambient's matrix operations only under a separately selected geometry derivative. |
| Future controls | Shape/appearance compensation and schedule interactions | Freeze original G as reference, compare matched capacity and radiometric conventions, distinguish gradients through SDF normals from appearance changes, and use independent geometry/held-light evaluation. |

Anisotropy-aware geometry would require new visibility/appearance approximations,
normal/tensor coupling tests, near-isotropic stability checks and matched schedule
controls. These are prospective interventions. The current experiment establishes
the authenticated original G reference first. K1, D2 and ambient prediction retain
their accepted project contribution scopes.
