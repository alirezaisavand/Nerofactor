# Donut D8 paired tensor representation

D8 compares two parametrizations of the same physical Q through the
same tensor renderer and proposal. D8-C decodes the existing chart
widths and double-angle head. D8-T predicts six symmetric ambient
components with three two-output heads. It projects onto the shading
plane and applies the bounded matrix function from the
[tensor design](../DONUT_TENSOR_REPRESENTATION_D8_DESIGN.md).
The pinned A and corrected-core sources are unchanged.

## Tensor math and shading

The computational frame uses a least-aligned normal axis. The tangent
2×2 tensor transforms by conjugation if this frame changes. Neither the
ambient field nor its world-Gaussian slope proposal is defined by that
computational frame. The projected ambient logits are evaluated in
float64 using the scalar function z/sqrt(1+z²). A rationalized divided
difference avoids subtracting saturated same-sign values. At repeated
tangent eigenvalues it gives the finite derivative of the scalar map.
The analytic square root and inverse use a 2×2 SPD determinant in
float64. The square root is cast to the native direction dtype for
sampling. The allowed squared widths are [0.04²,1]. Extreme logits
can still saturate in finite precision.

The specular proposal projects a standard world Gaussian into the
tangent plane, transforms it by sqrt(Q/2), and normalizes n+s.
For valid upper half-vectors, use D(h) = p_s(s)/(n·h)^4 and the
reflected density D(h)(n·h)/(4|wo·h|). For unit h and cosine ≤1/32,
the upper bound on D is below the float64 subnormal range even at
the maximum permitted width. Mask this region before dividing by the
cosine in the direct NDF evaluator. Use signed reflection and zero
radiance for invalid incident directions, retaining all proposals in
the radiance denominator. Native visibility, learned direct lighting,
zero indirect, original Fresnel, diffuse allocation and diffuse
sampling come from the existing shader. The tensor normal distribution
supplies the detached pre-mask softmax for the legacy specular prior.
The energy proxy still divides by the accepted specular count and the
specular prior by three times that count, each with the old minimum
of one. The legacy RGB channel and point reductions remain intact.

During each photometric forward, consume the original two uniform RNG
requests and use a separate device generator for the world Gaussian.
The training seed is 603300000 plus the zero-based update. Evaluation
uses 1600000 plus the existing camera, chunk and draw seed. The
known-light gate uses seeds 900000–900031. The initial paired probe
uses seed 120008. This preserves the inherited default RNG position
for diffuse azimuth and the perturbation prior. The Gaussian tape is
matched across D8 arms. It changes the finite specular samples compared
with D6, which remains historical context.

## Initialization, capacity and objective

Start both arms from the original seed 6033 and verify the fresh
uncustomized A state against the accepted D2 initialization identity.
Use the same shared trunk, scalar material heads and learned light.
Set the last weight-normalized head gains to zero. D8-C sets both widths
to nominal 0.25 with a valid chart angle. D8-T adds the isotropic
inverse-map bias as a fixed identity offset and initializes the six
residual outputs to zero. The shared field is isotropic at launch.
The chart angle derivative vanishes at this state, while the ambient
head can express all three tangent variations. The hidden widths and
depths are identical. The two added head outputs add 516 trainable
weights, biases and gains, for 1,486,100 material parameters versus
1,485,584 in D8-C. The optimizer retains all shared/scalar/anisotropy
parameters and the original outer-light MLP.

Retain the original Gaussian neighbor draws and its color/metal
penalty, with the original coefficient and reductions. Remove both
raw-width and chart-angle differences from this legacy perturbation
term in both arms. Retain the legacy energy and specular terms and
diffuse-white unchanged. Add the D6 world-Q penalty at coefficient
0.05 with the original distance normalization and normal transport.
Use identical domain, smooth normals, training and reserved edges,
128 pairs per update and seed 26061. Verify the exact D6 domain and
all six count snapshots. Ground truth is used for evaluation and
compatibility probes, never as material supervision.

The admitted Donut fitting cameras, linear Charbonnier objective,
512 photometric rays, 512 diffuse and 1,024 specular samples, original
Adam/AMSGrad and warmup/cosine schedule, clipping and all other legacy
photometric settings are common. D8-C and then D8-T each run for
exactly 10,000 updates. A mandatory failure stops the batch.

## Admission, resource limits and outputs

Focused CPU tests check the independent Cartesian Kurt BRDF,
reciprocity and invalid support, half-vector normalization,
sample/PDF correspondence and covariance, sign and axis-swap
equivalence, computational/world rotation covariance, finite
differences at, near and away from isotropy, near-bound and extreme
logits, retained legacy reductions, RNG position and synthetic paired
forward/backward. A separate CPU timing uses fixed analytic proposals
and measures numerical overhead without rendering a dataset.

Before any updates in each arm, verify the known-light 24-point,
32-draw gate against admitted independent D1 targets using the original
compatibility bounds. Each process also constructs the counterpart and
checks bitwise shared trunk, scalar and lighting state, exact scalar
predictions and common-tape RGB components on the fixed 24 fitting
probes under the initial learned light. The max initial and paired Q
error is 2e−8, and the max RGB component error is 1e−6. Save the captures,
then discard the counterpart. These are launch gates, pending native
execution.

Use the existing corrected BVH image, A100 GPU 0, four CPU threads,
64 GiB host RAM with no swap, 128 PIDs, a 25% CUDA allocator fraction,
and 8 GiB owned output per arm. Foreground watchdog 7,500 s plus
30 s kill grace. Each arm uses the 3,600 s host and 3,540 s inner limits
with 30 s and 20 s grace, respectively. Keep source, roots and inputs
read-only, run non-root and disable the network. Each arm is limited
to 19,250 shader calls and 15,113,330,688 shadow rays including the
paired gate. Zero new primary rays or reference integrations. Each
arm draws 1,280,000 spatial pairs, calls the spatial predictor 20,000
times on 2,560,000 points, and records 30 component gradient sweeps.
Preserve partials on a mandatory failure. Approval covers one batch.

Save evaluation at 0 and every 2,000 updates and five own checkpoints
with explicit D8 variant tags. Export raw world `Q_tensor_direct`,
sorted widths, diagnostic principal axes, eligible orientation, RGB
and components and illumination. Analytic principal axes are decoded
under no-grad for displays and orientation errors. Legacy display
frames must reconstruct the raw Q to 1e−6 absolute. As in D6, the
anisotropy gap floor for orientation scoring is 0.1. Save the original
reserved-edge diagnostics and timing of training forwards, the spatial
loss and total run.

Compare D8-T against D8-C at update 10,000. Report overall and held-out
Q medians, tails and cutoff counts, orientation eligibility/tails,
both widths and log-anisotropy errors, camera 34 coverage strata,
raw transported-Q variation on reserved edges, width saturation and
the anisotropy gap, held-out RGB and light recovery and computational
overhead. Use D6 to contextualize the results. Its initialization,
finite specular tape and legacy spatial objective differed. The paired
comparison uses fixed final endpoints and the same illumination and
BRDF family throughout.
