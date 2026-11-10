# Material-A equation-to-code map v1

Scope: exact material A `e1e859779301faced46d6476691e20b546c27ae0`.
The frozen source is authoritative for this audit, not comments, method names,
thesis prose or another branch. No formula or configuration is changed.
Source files are under `scientific_sources/material_a/`.

Reference PDFs were provided by the researcher under
`/home/isavand/paper/evidence/thesis-docs`; exact file hashes/page counts are in
[evidence/BRDF_REFERENCE_DOCUMENTS_V1.json](evidence/BRDF_REFERENCE_DOCUMENTS_V1.json).
Thesis page references below give **printed page / PDF page**, accounting for its
10 front-matter pages. Local Kurt PDF numbering is1–9. The public primary reference
is [Kurt et al., 2010](https://cgg.mff.cuni.cz/~jaroslav/papers/2010-anisobrdf/2010-anisobrdf.pdf).
The provided local PDF is individually hashed; no binary equivalence to that URL
is asserted. NeRO's primary paper is [Liu et al., 2023](https://arxiv.org/abs/2305.17398).

## Notation, actual outputs and domains

All directions point away from a point on the surface. Unit normal n, incident
wi, outgoing wo; ci=n·wi, co=n·wo. A physically valid BRDF comparison uses ci>0,
co>0 and h=normalize(wi+wo). Tangent-frame components of h are hx,hy,ch. Symbols
mx,my denote **Beckmann slope widths**, and a denotes the optional Kurt exponent;
a is distinct from a microfacet roughness parameter often also called alpha.
Metallic is m, predicted RGB albedo is k, d=k(1−m), F0=.04(1−m)+mk.

| Quantity | Actual code location | Implemented behavior |
| --- | --- | --- |
| Material heads | `network/field.py:1198–1219`, `make_predictor:319` | Sigmoid heads; mx=.04+.96*sx, my=.04+.96*sy; m,k in(0,1); no active alpha/ks/source head |
| Roughness | `compute_Dh:1221`, sampler:1355 | mx,my enter squared **inside** the slope exponent; no extra squaring of head outputs; do not transfer isotropic roughness comments here |
| Theta encoding | `predict_anisotropic_components:1213–1214` | r=2*rraw−1; rhat=r/max(norm(r),1e-6), not r/(norm(r)+eps) |
| TBN | `compute_tangent_bitangent_flat:897–948` | T=normalize(n×X); if norm<1e-6 use n×Y; B=normalize(n×T); no global continuity guarantee |
| Rotation | `sample_aniso_ggx_directions:1327–1337`, rotation:1304 | Double-angle half identities with sqrt floors1e-6 and sign(sin2theta), then T'=c*T+s*B, B'=−s*T+c*B; no post-rotation orthonormalization |
| NDF | `compute_Dh:1221–1254` | Beckmann exponential, despite methods named aniso_ggx; equation below |
| Specular sampler | `sample_aniso_ggx_directions:1339–1375` | Two fresh torch.rand(N,M) arrays on **every** call, including evaluation; below-horizon draws retained, later masked |
| Direction PDF | `compute_pdf_aniso_ggx:1261–1302` | q/(4*pi*mx*my*ch^3*(wo·h)), denominator floored1e-6, dot clamped[0,1] |
| Fresnel | `fresnel_schlick_directions:1077–1082`, `fresnel_schlick:1074` | Reconstruct H=normalize(wo+wi), F=F0+(1−F0)*(1−clamp(wo·H))^5 |
| Diffuse samples | `sample_diffuse_directions:950–969`, `utils/base_utils.py:887–900` | Golden-angle upper-sphere points, **arcsin elevation remapped**, then square roots; not a cosine distribution |
| Diffuse weight | `diffuse_term:1517–1538`, `compute_radiance:1441–1443` | mean(Ld)*d; no angular Fresnel coupling; pi cancellation would be correct under a cosine proposal |
| Specular weight | `compute_radiance:1457–1465` | sum(Ls*F*max(ci,0)/max(ch,1e-6))/max(Nvalid,1), Nvalid=count(ci>0); computed PDF argument unused |
| Linear/color space | `shade_anisotropic_mixed:1573–1577`, `utils/raw_utils.py:5–18` | Add diffuse+specular in linear space, then linear_to_srgb. Energy tests must use linear values, not gamma-coded RGB |
| Illumination | `get_lights:1044–1072`, `predict_outer_lights:1020–1042` | BVH trace from p+1e-5*direction; learned direct field on misses, learned positive indirect field on hits, depth≤eps suppressed |
| Mesh normals | `network/renderer.py:1273–1283` | Original BVH normal negated then normalized. The audit never changes it |
| Shader scope | `network/renderer.py:1251–1253`, field defaults:748–773 | Only shader_cfg forwarded (+is_real), not top-level energy/specular coefficients |

## NDF, sampling density and ideal specular kernel

Independently in Cartesian coordinates, for ch>0:

\[
 D(h)=\frac{\exp[-h_x^2/(m_x^2c_h^2)-h_y^2/(m_y^2c_h^2)]}
 {\pi m_xm_yc_h^4},\qquad\int_{H^+}D(h)c_h\,d\omega_h=1.
\]

The original clamps cos²(theta_h) below1e-6 in the exponent, log(q) below−40
and the complete denominator below1e-6. It does not return a globally normalized,
unclamped mathematical density for arbitrary grazing/invalid inputs. Tests compare
the interior domain separately from those guards. The slope widths are standard
Gaussian-slope scale parameters: each independent slope has variance m_axis²/2.
mx=my gives Beckmann, **not GGX** and not the complete original NeRO isotropic
Cook–Torrance model.

For uniform U,V, the independent Cartesian sampling form is
(sx,sy)=(mx*sqrt(−log U)*cos(2*pi*V), my*sqrt(−log U)*sin(2*pi*V)),
h=normalize(sx*T'+sy*B'+n). It agrees with the source's spherical inversion:
phi_h=atan2(my*sin(2*pi*V),mx*cos(2*pi*V)),
tan(theta_h)=sqrt(−log U)/sqrt(cos²(phi_h)/mx²+sin²(phi_h)/my²).
Full-range atan2 resolves azimuth quadrants; its use is appropriate.

With an orthonormal frame and valid unit directions, the reflection map
wi=2*(wo·h)*h−wo has Jacobian4*|wo·h|. Thus p_i=D(h)*ch/(4*(wo·h))
on valid upper reflections. **Its integral over the upper hemisphere is the
acceptance probability P(wo), not necessarily1.** Invalid draws are part of the
original measure; a fixed-total-count estimator should assign them zero weight.
A subdensity alone is not a PDF normalization defect.

The source clamps U below1e-6 (a small ring atom), clamps negative wo·h to0
(an atom at −wo for invisible facets), then normalizes wi. These measures and the
clipped PDF are not exactly the ideal continuous reference. They are characterized
as numerical approximations unless a quantitative downstream defect is established.
In particular, a non-unit rotated frame breaks the preceding change-of-variables
assumptions. Returned PDF is never used to correct the radiance weight.

The algebraic specular kernel **before** accepted-count normalization is

\[
 f_s^0(w_i,w_o)=\frac{F(w_o\cdot h)D(h)}{4(w_o\cdot h)},\quad
 \frac{f_s^0 c_i}{p_i}=F\frac{c_i}{c_h}.
\]

This is Kurt's a=0, ks=1 case, not a standard Cook–Torrance kernel with a missing
Smith G accidentally to be restored. It is reciprocal on its valid domain.
The source instead computes

\[
 \widehat L_s=\frac{\sum_{j=1}^M 1_{c_{ij}>0} L_j F_j c_{ij}/c_{hj}}
 {\max(N_\mathrm{valid},1)}.
\]

For independent draws, conditional on Nvalid>0 its expectation is the conditional
mean. Unconditional expectation equals that mean times1−(1−P)^M. As M grows it
tends to the intended integral divided by P(wo). This view/roughness-dependent
factor changes the effective kernel and breaks reciprocity in general. It is not
fixed by more samples or by relabeling the existing PDF as conditional: the
corresponding factor P is absent from the weights.

## Actual diffuse sampling measure

For N source samples, sample_sphere(N,0) uses z_j=j/N, elevation=arcsin(z_j),
and golden-angle azimuth (from index N+j). Initialization remaps radial input
e_j=1−2*arcsin(z_j)/pi; sample_diffuse_directions uses
ci_j=sqrt(1−e_j+1e-7). Direction length²=1+2e-7 for an orthonormal base.
Training adds one random azimuth shift per point, not independent radial draws;
evaluation does not randomize that shift. Source specular sampling remains random
in evaluation, independently of this is_train flag.

In the dense-pattern limit, ignoring the explicit epsilon, ci²=2*arcsin(z)/pi.
Consequently the generating density per solid angle is

\[
 q_d(w_i)=\frac{c_i}{2}\cos(\pi c_i^2/2),\quad
 \int q_d\,d\omega=1,\quad q_d\ne c_i/\pi.
\]

The source mean(d*L) therefore does not integrate Lambertian d/pi for general
lighting. Its continuum effective kernel is d*q_d/ci =
(d/2)*cos(pi*ci²/2), which depends only on the incoming direction and is not
reciprocal. Constant lighting hides this error because both proposal averages
are1; under L=ci² the intended cosine integral is d/2, versus
source limit d*(1−2/pi)≈0.36338023*d. Finite512 source directions reproduce that
bias. This is an inherited sampler/construction issue in the frozen A code;
its origin in any earlier producing revision was not investigated or inferred.

## Thesis and primary-paper comparison

| Written source | Agreement or disagreement with frozen A |
| --- | --- |
| Thesis(2.8–2.10), printed15/PDF25; Kurt(4–7), PDF2 | Same ideal Beckmann q,D and isotropic limit; source adds numerical guards |
| Thesis(2.11), printed16/PDF26 | **Visually verified discrepancy:** printed denominator is4*ci*co^a, missing wo·h and different exponent grouping from its own(3.8). For a=0 it is not reciprocal in general. Do not silently reinterpret it |
| Thesis(3.8–3.9), printed26/PDF36; Kurt(10–12), PDF2–3 | Correct ideal specular form with thesis-declared a=0; F0 agrees. Source also multiplies diffuse by(1−m), absent from written3.8; no separate ks head |
| Thesis3.4 cosine-sampling statement; NeRO(15), PDF7 | Intended mean diffuse weight assumes ci/pi; actual A direction construction differs as derived above. Missing pi in that *weight* alone is not an error |
| Thesis(3.10–3.14), printed27/PDF37; Kurt(16–21), PDF7 | Ideal sample/PDF algebra agrees on the valid domain; source clips and filters. The weight cancellation is appropriate there; source Nvalid normalization disagrees with fixed-N MC(thesis2.13/Kurt15) |
| Thesis(3.15–3.18), printed28–29/PDF38–39 | Pointwise reference-cross frame agrees for valid normals. Fallback cannot guarantee a globally smooth frame; tested pole switch has a90° tangent jump |
| Thesis(3.19–3.22), printed29–30/PDF39–40 | Earlier single-angle formulation is background/alternative; not the actual A double-angle head |
| Thesis(3.23–3.25), printed31/PDF41 | Double-angle intent agrees. Code divides by max(norm,eps), not norm+eps. Both need a zero-vector policy. sign(0)=0 at rhat=(−1,0) collapses the half-angle frame; code sqrt floors additionally perturb unit length. Smoothness/orthonormality claims are not unconditional |
| Thesis(3.26–3.28), printed32/PDF42 | Rotation squared difference appears; scalar absolute differences equal scalar L2 norms, but source RGB channelwise mean absolute variation is not the stated RGB vector L2. The tiny double-scaled rotation decay is additional implementation detail |
| Thesis(3.29), printed33/PDF43 | Written2*pi/N sum(fr*ci) presumes uniform hemisphere sampling. Source penalizes relu(d+conditional_mean(F*ci/ch)−1). Importance cancellation can legitimately change form; Nvalid and diffuse proposal still invalidate literal equivalence. A finite-weight soft penalty never guarantees all-direction conservation |
| Thesis(3.30), printed34/PDF44 | Detach(D), temperature1 and softmax over sample axis agree. Source further masks invalid samples, divides by3*Nvalid, sums RGB then averages points. The inverse-sample-count scale is already present in the written formula, not proof of a coding transcription defect |
| NeRO(16), PDF7; Appendix A.1, PDF18 | Original isotropic GGX/G/roughness convention is a different BRDF; do not restore its formulas into A. Explicit A a=0 is a documented modeling choice |

## Regularization and illumination assumptions

Bell declares top-level reg_energy_loss_lambda=reg_spec_loss_lambda=.01; actual
shader defaults are **.05 each** because only nested shader_cfg is forwarded.
Both penalties run only inside reg_change, under renderer reg_mat. Bell actually
uses512 diffuse and1,024 specular directions per surface point; shader default
specular256 is overridden. The regularizer's D is detached, so its smoothmax
weights do not directly train mx/my/rotation; gradients primarily affect k,m
through d (mask/count operations are discrete). With all M samples valid,
L_spec per RGB channel=d/(3*M); summed RGB/point penalty scales as d/M for equal
channels. Increasing M therefore weakens this prior at fixed lambda.

Actual spatial regularization uses mean-channel(|delta_k|+|delta_mx|+|delta_my|+
|delta_m|+rotation_squared_difference*decay)*.05. Rotation decay is
1e-4+9e-7*(1+cos(pi*step/100000))/2, **not** a sweep up to the named max_weight1e-3.
This is a verified effective coefficient, not authorization to alter its intent.
Perturbations use a random tangent direction times a signed Gaussian radial
scalar(std=.02). Top-level unused fields must never be silently forwarded.

Bell selects direction-only outer illumination, not the optional sphere-position
representation; finite-distance lighting may therefore be approximated. Indirect
light is a learned positive radiance field at secondary hits, not a new multi-bounce
path solution. Light/albedo ambiguity, energy penalties and RGB fitting can absorb
BRDF estimator errors. All energy/counterexample tests here use linear unit
illumination and fixed analytical normals; they do not measure a trained object,
validate r128 geometry, explain a historical PSNR, or establish producing provenance.
