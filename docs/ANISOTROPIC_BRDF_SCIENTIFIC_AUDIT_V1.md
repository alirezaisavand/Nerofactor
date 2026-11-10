# Milestone3B-1: scientific anisotropic BRDF audit

Base `959db370cd18a09a20b2a0c9bb24a20a2cdba22a`; feature branch
`feat/anisotropic-brdf-scientific-audit`. Frozen material A remains
`e1e859779301faced46d6476691e20b546c27ae0`, geometry G remains
`79252178e65914ad1acbc14790b1514f276bfea0`. This milestone performs offline saved
array analysis and analytical CPU checks only. No scientific source/configuration,
model, dataset, mesh, checkpoint or accepted evidence was changed; no native
experiment, historical suite or previous diagnostic was repeated.

## Deliverables and evidence scope

- [3A-2 close-out](evidence/GEOMETRY_SDF_MESH_CLOSEOUT_V1.md) and
  [complete failed-sample localization](evidence/GEOMETRY_SDF_MESH_CLOSEOUT_V1.json).
- [Equation-to-code and thesis/paper mapping](ANISOTROPIC_BRDF_EQUATION_MAP_V1.md),
  including actual roughness, theta, tangent frames, PDFs, estimator weights,
  regularization scopes, illumination and RGB assumptions.
- [Independent numerical witnesses](evidence/ANISOTROPIC_BRDF_ANALYTICAL_V1.json)
  and [source/PDF identities](evidence/BRDF_REFERENCE_DOCUMENTS_V1.json).
- Pure offline/reference code in `audits/`; new tests in `tests/scientific/`.
  These are not imported by any renderer/training entry point and implement
  no production correction or new experiment-launch/provenance framework.

The researcher supplied three PDFs. Thesis SHA-256 is
`1b0404df159381e7beb643979c2c159a23c83c6c2f191d9faff5fc41997b62ea`;
Kurt `9335210d2fcd34eb96c72f47d4f6caeb89f2d1cedc1703dec1a9d7dfe8280398`;
NeRO `129be771b22030f02f8f023bf7716c48e0bd8c5e0f49f249d9f4c806b072b54a`.
They were read without modification, with text extracted into temporary storage.
Thesis Eq2.11 was additionally inspected visually on PDF26/printed16 to resolve
its denominator; this is not an OCR-only attribution. Original PDFs are not
committed. Publication links and exact printed/PDF page mappings are in the map.

Verified frozen `network/field.py` SHA-256:
`b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.
The complete manifest-checked source closure, not a similarly named live module,
is used for the CPU method checks. This does not identify historical producing
revisions/checkpoint splits or validate historical thesis experiments.

## Scientific close-out of3A-2

The completed action has2,119 eligible samples,156 exclusions and74 failures.
73 failures cluster within h=2/127 of the extraction sphere and have negative
raw SDF; none of the failed face samples is a tiny triangle under the frozen
1e-10 area label. Masked-field cap generation is a strongly supported mechanism,
not proven provenance of individual marching-cubes cells. The remaining interior
point passes its first-order residual proxy and crosses at±h/4, but fails the
frozen±h/2 bracket because of additional sampled sign reversal. The criterion is
not a universal exact-distance or surface-thickness test. Keep all failures.

The single failing non-anomalous A normal control is also interior; it cannot be
ascribed solely to the extraction sphere. All eligible centroid gradient/face
cosines are negative: angular deviations, not mixed orientation signs, made the
uniform30° criterion unresolved. Geometry remains diagnostic/unadmitted. BRDF
analysis therefore uses analytical normals and lighting, without changing or
selectively filtering the historical mesh to improve material results.

## Ranked findings

"Confirmed" below means algebra/source inspection plus the stated CPU witness,
not evidence that a particular learned object exhibits the worst case. A passing
test suite includes assertions that these defects exist. It is not BRDF admission.

| Rank / class | Finding | Evidence and implication |
| --- | --- | --- |
| 1 — confirmed estimator defect | Specular division by Nvalid instead of total M | Ideal sample/PDF cancellation is correct, but final estimator is conditional. At mx=.6,my=.9, P(normal)=.82151718 vs P(oblique)=.56305787. Reciprocal base pair=.22821601 both ways, conditional effective pair=.27779822 vs .40531537. More samples do not remove this bias |
| 2 — confirmed diffuse integration defect | Constructed directions are not cosine-weighted | Source radial map is ci²=2*arcsin(z)/pi; generating density qd=(ci/2)*cos(pi*ci²/2), not ci/pi. Under L=ci², source limit=.36338023*d instead of .5*d; constant light hides the issue. The continuum effective diffuse kernel is nonreciprocal |
| 3 — confirmed implementation defect at admissible encoding | Half-angle branch collapses a unit orientation's frame | rhat=(−1,0) gives c=.001,s=0, hence tangent/bitangent lengths=.001 and non-unit h. Near-zero mapped head outputs are also allowed; rhat=(0,0) yields lengths sqrt(.5). Sampling/PDF and analytic cancellation assumptions then fail, even though values can be finite |
| 4 — confirmed physical counterexample | Combined reflectance can exceed1 in allowed parameter range | At mx=.6,my=.9,k=.999,m=.001, normal unit-illumination total=1.02207599, oblique=1.05043763 in the source continuum limit. Even the unconditioned normal total=1.01777902: fixing Nvalid alone does not establish total energy conservation. A soft regularizer is not a hard guarantee |
| 5 — confirmed written-equation inconsistency | Thesis2.11 differs from3.8/Kurt | The visually inspected2.11 denominator4*ci*co^a is not the correct stated half-vector expression and is generally nonreciprocal at a=0. Code matches the latter specular kernel before estimator normalization. Document an erratum; do not change science to match the misprint |
| 6 — numerical/gauge fragility | Reference-cross tangent switch and angle guards | A tested normal crossing the1e-6 X-reference threshold causes a90° tangent change. Ordinary points are locally continuous. A global smooth sphere frame cannot be obtained by merely hiding a fallback seam. Zero/parallel optional source vectors give zero bases |
| 7 — numerical approximation | PDF/normal clamping and sample atoms | Float32 floors keep tested grazing values finite but perturb exact density/orthonormality. U clipping and invisible-facet collapse create atoms not represented by a continuous PDF. Do not mistake upper-hemisphere PDF submass P<1 for a normalization bug |
| 8 — verified configuration/modeling discrepancy | Effective priors differ from declared top-level settings | Bell declares .01; shader actually uses .05 for energy and specular priors. Both are nested inside reg_change. Preserve that scope; forwarding those top-level values would be a scientific change |
| 9 — sample-count-sensitive modeling prior | Specular penalty is detached-softmax weighted and divided again | Its inverse-M behavior is also present in thesis3.30; actual code adds masking, Nvalid and RGB reduction. It directly trains diffuse allocation, not roughness through detached D; increasing M weakens its effective scale |
| 10 — documented modeling assumptions | a=0, ks=1, metallic diffuse scaling, learned direct/indirect illumination | a=0 is explicitly stated in thesis3.4. No active Smith G is required by this chosen model. Beckmann despite GGX method names is consistent with thesis/Kurt. Direction-only Bell outer light approximates finite-distance illumination; learned illumination/albedo can absorb estimator errors |

Nonnegative/finite behavior holds on the tested valid parameter/direction grid;
this is weaker than reciprocity or energy conservation. Ideal NDF normalization,
isotropic limit, theta periodicity and swapped-axis equivalence pass independently
and against pure source algebra away from identified decoding singularities.
No testing here proves global finite derivatives at every guard or every learned
material field. Half-vector reconstruction of antipodal directions is undefined
physically even when source normalization returns a finite zero vector.

## Focused CPU tests and bounds

Observed: **25 passed,0 failed,0 skipped**, using CPython3.9.12,
NumPy1.21.5 and Torch2.4.0+cu121 **on CPU only**. No CUDA availability check,
GPU allocation, model construction, weight loading or optimization is performed.
The Torch version is the available CPU-test runtime, not claimed to be the
reviewed native Torch1.8 environment. Original A constructor/imports are never
executed: the test copies only enumerated/hash-verified pure method AST bodies,
without rewriting their algebra, into a CPU-only stand-in. Network predictors
are deterministic constants where the output transform is checked. `torch.rand`
is replaced with explicit private-NumPy fixture draws. The independent oracle
uses Cartesian Gaussian slopes, 1-D acceptance-circle integration and hemisphere
quadrature, rather than duplicating the original angular inversion as a reference.

Coverage: projected NDF normalization, source NDF values; reciprocity/nonnegativity;
sample/PDF agreement and reflected submass; total-count MC consistency and
accepted-count bias; directional energy counterexample; diffuse moments/density;
theta periodicity/swapped-axis/isotropic checks; exact half-angle degeneration;
pole continuity, zero/parallel normals, antipodal/grazing and zero-acceptance
cases; scope/roughness/normalization differences; ignored PDF argument; specular
prior sample scaling; saved-array hash binding, mask eligibility, finite checks,
missing/duplicate probes and original-preserving close-out.

Fixed analytical tolerances: NDF/projected/reflected-mass quadrature abs2e-5;
source Cartesian-sample comparison abs16 float32 epsilons (CPU transcendental
roundoff), PDF rel1e-4/abs5e-6; independent Monte Carlo integral abs.004 with
131,072 fixed fixture draws. The MC/source conditional identity uses abs2e-6.
Ideal reciprocity uses rel2e-12/abs1e-12. These synthetic bounds do not replace or
tune any frozen3A-2/native acceptance thresholds. Existing statuses are unchanged.

```bash
cd /home/isavand/paper/Nerofactor-paper
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/scientific -v
# Pure NumPy witnesses only; choose a fresh output filename.
PYTHONDONTWRITEBYTECODE=1 python3 -B -m audits.anisotropic_brdf \
  --output /tmp/anisotropic-brdf-analytical-review.json
```

## Proposed changes — review required, none implemented

| Priority / proposal | Motivation and expected effect | Comparison and ablation | Computational cost |
| --- | --- | --- | --- |
| P1: fixed-total-count specular estimator | Remove view/roughness-dependent acceptance gain; expected lower radiance where P<1, unbiased ideal kernel integration | Exact A vs only M denominator; fixed uniforms/normals/lights/heads, report bias/variance/reciprocity per view, never change material heads simultaneously | Same rays/light queries; one scalar denominator change, no claimed measured speedup |
| P1: genuine cosine diffuse sampling | Integrate declared Lambertian coefficient under directional lighting | Exact A vs only radial construction changed: use existing sphere height z, ci=sqrt(z), radius=sqrt(1−z), same azimuth pattern/count/RNG. Separate from specular correction, then2x2 combination | Same512 diffuse rays; O(1) scalar transformation per direction, no new network/BVH calls |
| P2: robust double-angle decoding and explicit zero-vector policy | Preserve orthonormality at(−1,0) and identify undefined orientations; expect fewer gauge/branch artifacts | Current sqrt/sign vs reviewed atan2-based decode, holding head outputs/checkpoints fixed; sweep endpoints and tiny norms, gradients and isotropic/axis-swap tests. Zero policy must be stated, not silently chosen | A few trig operations per point; no additional samples. Any new head regularizer requires separate coefficient ablation |
| P3: physically validated diffuse/specular coupling | Current additive allocation has energy counterexamples even without count bias | Compare corrected-estimator existing coefficients against a separately versioned Fresnel/coupled or explicit energy-budget parameterization. Independent directional energy/reciprocity grids first; a Fresnel factor alone is not assumed sufficient | Typically O(1) extra per direction; energy verification is an offline quadrature cost, not new training rays. Retraining/changed material meaning would need approval |
| P3: explicit, sample-count-calibrated priors | Separate physical constraints from a weak M-dependent penalty and mistaken config scopes | Retain effective .05 baseline; vary M=256/512/1024 with paired fixed draws; independently vary regularizer normalization, detached weighting, and coefficient. Do not combine silently with top-level .01 forwarding | Same rays for fixed M; changing M scales illumination/BVH cost linearly, and must be reported |
| P4: gauge-aware anisotropy tensor/transport prior | Potential method hypothesis: learn/smooth physical anisotropy rather than chart-dependent theta labels; avoid quotient/axis-swap artifacts | Compare decoded axes with a tangent SPD covariance represented through double-angle entries, plus transported/world-frame regularization. Separate decoder-only, prior-only and combined arms; test rotating charts and axis swaps at equal physical fields | O(1) small matrix arithmetic per point; actual curved-surface transport needs neighbor geometry/adjacency or extra queries. Benchmark before claiming negligible cost |
| P5: explicit near-field light model comparison | Bell direction-only direct light can confound BRDF recovery | Hold corrected BRDF fixed; direction-only vs existing sphere-position option under controlled near/far illumination; separate reconstruction and relighting metrics | Larger input representation/possible network cost, same ray count; architecture/checkpoint compatibility must be reviewed |

P1/P2 are correctness/robustness work, not novelty claims. Tensor/transport and
physically constrained disentanglement are **potential publishable directions**
only after novelty review, controlled evaluations and cost measurements; Kurt's
known formulas, standard importance weighting or atan2 alone are not contributions.
No publication claim, improvement measurement or new production variant is made.

## Recommended next controlled experiment

Approve a small **CPU analytical2x2 estimator experiment**, before any new full
reconstruction. Arms: frozen A; denominator-only; cosine-only; both. Preserve
original A as the immutable reference and put any approved variant in an explicitly
versioned experimental implementation, never edit the snapshot. Use common saved
uniforms, same512/1024 sample counts, a=0, original F0/d mappings and no learned
illumination, geometry, optimizer or training. Fix all heads and flat unit normals.

Predeclare mx/my pairs(.04,.04),(.1,.6),(.3,.7),(.6,.9),(1,1); outgoing angles
0/30/60/80/89 degrees; tangent theta=.31 for the regular-frame comparison;
separately report exact singular-encoding cases rather than hide or repair them.
Test unit light, ci² light and a nonnegative azimuthal lobe. Use private seeds0–15
for paired fixture draws, without changing historical scientific RNG policy.
Independent hemisphere references256×512 and512×1024 must agree within2e-5;
otherwise mark that integral unresolved. Compare bias with3 standard errors plus
reference error; record variance, acceptance, reciprocity and energy, not only RGB.
No post-result threshold/grid selection. No native model/BVH calls are needed.

Expected discriminating outcomes: only fixed-count corrects specular acceptance
gain; only cosine construction corrects the diffuse second moment; both should
recover their intended component integrals in the valid frame. Combined energy
and theta singularities may still fail and remain separate blockers. After these
controlled results and decoder/energy protocol review, propose a separately
approved same-scene material experiment with admitted geometry and explicit
training/evaluation splits. Historical r128 is not such an admission. No GPU
training, old-checkpoint score reinterpretation or silent thesis correction is
authorized by this recommendation.
