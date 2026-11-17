# Donut D3 calibrated illumination recovery

D3 tests whether calibrated global illumination scale and color improve roughness
recovery, using the same admitted Kurt/Beckmann acquisition and full material
field as D2. D1 and D2 are the accepted historical controls recorded at
`cce13b5795f1818434abfdfc096be7f87469ccad`. Use the predeclared update-10,000
endpoint in all three arms. D3 requires separate execution approval.

## Calibration and matched settings

The original Material A directional MLP predicts positive linear RGB radiance
`L(w)`. For each channel, D3 supplies

```text
m = sum_i w_i L(direction_i)
L_calibrated(w) = L(w) * mean_calibrated / m
```

Use the existing 128×256 equirectangular midpoint export grid. Normalized
`sin(theta)` weights represent the solid-angle element, with theta measured
from world +Z and phi about +Z. This constrains the discrete spherical mean.
The same scale applies to all directions in an update or evaluation. Autograd
includes the denominator, allowing angular redistribution and canceling a uniform
gain in each channel. Require finite nonnegative node radiance, strictly
positive finite denominators and scales, and a connected training graph.

The fixed calibrated means are **[0.13995771766028667, 0.14655455327313444,
0.1674445384474473]**, computed in float64 from the saved D1 known-light map
using those quadrature weights. The calibration source map is hash-bound in the
freeze. Only these three values enter the D3 lighting callback. The analytic L2
definition is used by the same pretraining native/reference compatibility control
as D2 and as the numerical comparison for illumination exports. It supplies no
directional targets to D3 fitting. The material labels likewise enter evaluation
and the existing compatibility control.

The callback normalizes the original MLP radiance for unoccluded directions
and preserves the existing visibility queries. Reuse one differentiable denominator per update and
one no-grad denominator for each of six evaluations. Export the normalized
quadrature radiance using that same denominator and verify each channel's
realized mean to an absolute tolerance of 2e-6. Record raw mean, scale and
realized mean at all endpoints. PNG preview clipping is display only.

Use fresh material and lighting parameters and require the full initial state
hash to equal D2's saved `initial.json` (only the small identity file is
mounted). Keep seed6033, fitting cameras0/50/1/45, held-out cameras28/34,
512-ray batches, corrected-core BRDF, 512/1024 diffuse/specular samples, direct-only
light, original Adam/AMSGrad, WarmUpCosLR, clip0.7 and the identical legacy
energy, specular, spatial and diffuse-white priors. Visible intersections and
radiance targets are the same admitted reference. Optimization uses the original
linear-RGB Charbonnier objective plus the same weighted priors. The normalization
changes the effective starting environment, while the parameter initialization
is identical. Both known-geometry and zero indirect assumptions still apply.

## Work and verification

Save the existing camera maps, numerical materials, Q, RGB/components and
illumination diagnostics at updates0/2000/4000/6000/8000/10000. Write the five
usual own-run checkpoints and perform the existing strict CPU roundtrip. At
update1 and each 2,000-update boundary, measure the weighted diffuse-white term
separately with the existing gradient diagnostics alongside photo, material,
energy and specular. Record group norms and cosine interactions. Charge up to
30 component reverse sweeps. Diagnostic gradients do not enter the update.

One GPU0 A100 arm has 10,000 Adam returns and unchanged per-arm shader/shadow
ceilings (19,248 and 15,113,256,960). The normalization adds exactly 10,006
grid evaluations and 327,876,608 directional MLP inputs, with no extra rays.
Timing records CUDA grid forward/reduction and host setup including synchronization
separately for training/evaluation. The denominator graph also adds backward
work during the existing gradient calculations. Compare whole arm runtime and
observed peak memory with D2, without interpreting that difference as isolated
backward cost.

Use the same admitted image, one GPU, 4 CPU threads, 64 GiB host memory with
no swap, 128 PIDs, 25% CUDA allocator fraction and 8 GiB owned outputs.
Foreground arm timeout 3,600s, inner 3,540s, outer 3,720s, with 30s host
and 20s inner kill grace. Use the launcher in calibrated mode and its new
dedicated freeze. Source and inputs are read-only, root is read-only and
execution is non-root with no network. Verify the existing dataset admission
and completed D2 initialization identity before GPU work. Stop on numerical,
identity or resource failure, preserving the partial history and with no retry.

## Evaluation and coverage

Compare all six matched camera panels and both splits at each endpoint, using
the existing masks and eight native evaluation draws. Report width errors,
log geometric-mean width and log anisotropy ratio errors, world-Q and qualified
orientation distributions, color, metallic, linear RGB, environment relative
L2, fitted/held-out differences and the existing normal-transported surface-Q
consistency diagnostic. Report the diffuse-white loss and gradient interactions
alongside the other probes. Report the actual minor-width bias and unresolved
orientation tails rather than inferring recovery solely from threshold counts
or the known calibrated mean.

The file-only camera34 check counts how many fitting cameras supply a valid
sample within radii 0.005, 0.01 and 0.02 native units with normals at most
30° apart. These radii bracket the median camera34 nominal pixel footprint
0.00893. Also count samples on the same triangle and stratify by camera34
normal-to-view angle. These are sampling coverage proxies. They do not
establish exact point visibility, conditioning or causality. The saved D1/D2
analysis already shows much higher errors without nearby fitting samples.
Apply these same bins to D3 for interpretation only. Never change camera
selection, losses or endpoint selection based on the comparison.

Future removal of the diffuse-white prior and a surface-Q smoothness loss are
separate interventions. Neither is part of D3.
