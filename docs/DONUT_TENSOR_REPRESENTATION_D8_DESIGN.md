# Donut D8 tensor representation design

Replace the chart-dependent width and angle heads with a world-space tensor
predictor in a separately selectable variant. The shared field, base-color
and metallic heads, corrected material law, shadow tracing and lighting
remain the comparison reference. D8 is a design at this stage.

## Bounded tangent tensor

Predict the six independent entries of a symmetric ambient matrix H(x).
For the unit shading normal n, let P = I − nnᵀ and X = PHP. The accepted
width range is [0.04, 1], so put l = 0.04², u = 1, a = (u + l)/2 and
b = (u − l)/2. Use the matrix map

```text
Q = a P + b X (I + X²)^(-1/2).
```

The two tangent eigenvalues are a + b z/sqrt(1 + z²), with z the eigenvalues
of X restricted to the tangent plane. They lie strictly inside (l,u) in
real arithmetic. The normal eigenvalue is zero. The map covers the
interior of the allowed width range, including exactly isotropic states.
At a repeated tangent eigenvalue the matrix function remains smooth.
Its scalar derivative is b/(1 + z²)^(3/2), so large logits can still make
learning slow. Track that saturation and finite-precision bounds.

For efficient computation, any orthonormal tangent basis U may implement the
same map on a 2×2 matrix; its output must transform by conjugation under
U → UO. For SPD M, use its principal square root
`(M + sqrt(det M) I) / sqrt(tr M + 2 sqrt(det M))` and linear solves on
this SPD matrix. This avoids learned axes, eigenvector derivatives and
divided differences at equal eigenvalues. Export sorted widths and
principal directions offline. The latter are reported only where the
existing anisotropy gap makes a direction meaningful. Check transport
and covariance in world space. A smooth tensor on a curved surface can
still have isotropic points or physical discontinuities.

## Beckmann evaluation and sampling

Q has tangent eigenvalues mx² and my². It is twice the slope covariance.
For c = n·h > 0, the slope is s = Ph/c and A = Q + nnᵀ is SPD. Its
determinant is the tangent determinant of Q. The slope density and
Beckmann normal distribution can be evaluated without angles:

```text
p_s(s) = exp(-sᵀ A^(-1) s) / (π sqrt(det A))
D(h)   = p_s(Ph/c) / c^4
p_h(h) = D(h) c
p_i(wi | wo) = p_h(h) / (4 |wo·h|)
```

The slope-to-solid-angle Jacobian is c⁻³. In the physical principal frame
the exponent is −sx²/mx² − sy²/my² and sqrt(det A) = mx my. For the
accepted additive Kurt a=0 specialization, retain
`fs = F D(h)/(4 |wo·h|)` on supported upper-hemisphere directions, and
retain its diffuse allocation. Swapping incident/outgoing directions
preserves h and |wo·h|. Leave out-of-domain proposals at zero contribution
with the original proposal denominator, without conditioning on accepted
directions. The current radiance estimator weight F (wi·n)/c follows on
valid support. Preserve the distinction between that estimator and the
legacy regularizer's reduction. [Kurt et al. (2010)](https://cgg.mff.cuni.cz/~jaroslav/papers/2010-anisobrdf/2010-anisobrdf.pdf),
[Walter et al. (2007)](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf).

One chart-independent proposal is `s = A^(1/2) P g / sqrt(2)`, with
`g ~ N(0,I_3)`, followed by `h = normalize(n+s)`. It has the same
continuous slope law. A 2×2 principal square root with a temporary tangent
basis uses two Gaussian coordinates and the existing radial law. Basis
changes preserve its density, while a fixed finite 2D tape co-rotates
only if its Gaussian inputs are transformed too. For a neural comparison,
share a world Gaussian tape and the same tensor sampler/evaluator in both
arms. Also hold diffuse tapes, sample counts, support and visibility fixed.
Common tapes can establish samplewise parity when Q agrees. Independent
quadrature should separately check BRDF values and finite-sample behavior.

## Fair neural comparison

Use the same shared-trunk weights, scalar heads, light, photometric
observations and transport regularizer within the paired experiment.
Compare a chart decoder and tensor decoder through the same evaluator.
Freeze and verify a shared physical starting field, for example a constant
interior isotropic width specified before fitting. In the tensor arm set
H = z I, using `z = y/sqrt(1 - y²)` with `y = (width² - a)/b`.
In the chart arm initialize both widths equally and use a valid unit
angle encoding. Verify initial Q and RGB with common tapes. At exact
isotropy the chart angle derivative vanishes. The tensor arm has tangent
traceless gradients available. Report initial physical Jacobians and
gradient norms, and use a separately predeclared weak-anisotropy stress
test to examine start dependence. Preserve comparable head depths and
hidden sizes. Count shared and anisotropy-head parameters and time.
The six ambient outputs add two channels versus the four chart outputs;
normal projection removes three pointwise directions in ambient tensor
space. Equal output-field capacity does not imply matched optimization.

The current legacy perturbation prior compares chart widths and angle.
Applying it only to the chart arm would confound the paired comparison.
Use one reviewed policy in both arms, for example replace only that
anisotropy component in both with the same physical Q comparison while
retaining the legacy k_d/metallic, energy, specular and white terms and
their draw policy. The detached legacy specular softmax can use the same
tensor D(h) values and reduction. Verify its forward value and gradient
scope. This paired bridge control is distinct from historical D6,
which remains a contextual result.

Before training, require independent fixed-Q BRDF/reciprocity parity,
basis-rotation and axis-swap equivalence, positive bounds, near-isotropy
gradient checks, grazing/support tests, slope covariance and reflected
density tests and draw-matched estimator parity. Benchmark overhead on
the same number of samples. Use Q, width and held-out RGB errors to
measure gains, with orientation eligibility and gap distributions to
detect collapse. This proposal adds no new illumination or BRDF family.
