# E.12 — one paired two-light observability study (not executed)

This is a separate scientific proposal. Native smoke approval does **not**
authorize new reference rendering. Preserve E.11D–G and v3 unchanged. No search,
additional light candidates, threshold tuning, full images or neural training.

## Fixed factor and reusable data

Retain the exact E.11G84 physical probe points, cameras[58,44,9,0], their observed
valid masks, original/stronger materials, frames, independent BRDF and all12
frozen perturbations. Add exactly one known illumination: rotate each L2 lobe
axis45° about normalized world(1,1,1) using Rodrigues' formula. Keep ambient,
power64, RGB amplitudes and integral energy unchanged. This rotation is fixed
before measuring its response; it is not selected from a family of candidates.
Original L2 captures are controls and must not be rendered again.

Reuse E.11E's exact256²/512² incident nodes/weights and packed material-independent
visibility, plus E.11G's136 added unobstructed correspondences. Lighting rotation
does not change geometry visibility. Zero new primary, shadow or outgoing segment
queries are needed; there are **220 total admitted point/view observations**
(84 base +136 added). The10 occluded correspondences and single-only point remain
reported; do not replace them. All84 points stay in summary tables.

Each valid point/view needs23 distinct case evaluations at the new light (two
12-case material controls sharing isotropy). Exact new ceiling:
`220*23*(256²+512²) = 1,658,060,800` BRDF point-directions. Compute27,525,120
environment point-directions, sharing incident light across views/cases. Preserve
the existing four-point/16384-direction chunks and numeric implementation. This
definition is not a launch freeze: source hashes, resource/output bindings and
ONE separate approval are still required before rendering.

## Frozen legacy metrics remain unchanged

For each light/material/view/case preserve .003 max-RGB and .002 relative-L2
reference-convergence requirements, .003/4*local-grid-error/2-code local response
gates and all E.11G coverage/conditioning definitions. Retain individual-light
results; then compare one-light versus two-light stacks on the **same points and
view masks**, stratified by valid view count. Normalize descriptive spectra by
sqrt(number of observations) and report actual observation counts. No favorable
view/light subsets, exposure matching or retroactive acceptance of E.11D L2.

## Separate noise-aware information analysis

Use declared θ/u/v coordinates (radians, log geometric mean, log ratio), fixed
color/metal and E.11G's exact finite perturbations. This remains a conditional
three-parameter analysis, not full seven-parameter recovery or an infinitesimal
Jacobian proof. The original median-condition/smin/25%-width screen is retained;
the following information diagnostics add no replacement pass threshold.

1. **Ideal float-linear observations:** report the unweighted matrix spectra and
   response magnitudes, with fine-minus-coarse operator norms as deterministic
   numerical-resolution proxies. A noiseless inverse problem has no finite
   Gaussian Fisher information; do not label unweighted JᵀJ a calibrated FIM.
2. **Explicit noise model, sensitivity analysis only:** assume independent
   additive Gaussian error in linear RGB, fixed σ=0.001 in unit radiance. This
   is a predeclared hypothetical observation model, not measured camera noise,
   native Monte Carlo variance or added target noise. For float observations,
   F=sum JᵀJ/σ² across valid views/lights. Report its spectrum, weakest directions,
   condition and local uncertainty scales where invertible. Also report F per
   observation to distinguish angular separation from merely more measurements.
3. **8-bit observations under that same analogue-noise model:** use exact sRGB
   transfer/round-to-nearest bins. For a channel with mean μ and linear-domain
   bin bounds[a_k,b_k), let P_k=Φ((b_k−μ)/σ)−Φ((a_k−μ)/σ), and
   dP_k/dμ=[φ((a_k−μ)/σ)−φ((b_k−μ)/σ)]/σ. Use
   I_channel=sum_k (dP_k/dμ)²/P_k, then F8=sum Jᵀdiag(I_channel)J. Code0/255
   have infinite external bounds to account for saturation; report actual
   clipping separately. Verify probability normalization, derivatives and
   limiting controls with stable tail arithmetic; no arbitrary probability
   floor or post-result σ adjustment. Quantization cannot increase information
   relative to the corresponding analogue observation; test this analytically.
4. **Actual deterministic8-bit oracle:** there is no injected analogue noise in
   the saved benchmark. Quantization yields intervals, not differentiable exact
   radiance. Report bin widths, original finite code contrasts and feasible
   intervals; do not assign a fictitious Gaussian Fisher interpretation to
   noiseless code steps. Keep this distinct from the hypothetical noisy model.

Reference-grid differences are correlated deterministic approximations, **not**
independent Gaussian noise. Do not simply add their squares to σ² and call the
result measured uncertainty. Compare the inferred weak directions with those
proxies and declare numerical limitations. Local Cramér–Rao scales are conditional
model-based diagnostics, not global confidence intervals or recovery guarantees.

Report every original/stronger arm: minor/major/log-ratio sensitivity, smallest
singular/information eigenvalues, matched spatial coverage, quantization loss,
noise-assumption sensitivity limits and failures. If lighting diversity improves
the remaining signals, propose the smallest cross-view/two-light benchmark with
separate observed lights and held-out relighting. Otherwise report the limitation
without proposing another uncontrolled pilot sequence. No execution is authorized
by this document, dataset admission, or the separate native smoke approval.
