# Donut D9 paired initialization sensitivity

D9-C and D9-T use the D8 tensor renderer, Gaussian tapes, fitting data,
world-Q surface loss, legacy scalar/energy/specular/white terms, full
network, original Adam and scheduler. Run D9-C first and then D9-T for
10,000 updates each. The starting tensor differs from D8's isotropic
0.25 field. Scene data, frozen G/A and the accepted D8 tensor math and
shader remain available at their original identities.

## Geometry-only starting field

Set a nominal width scale near 0.5. Along the two most separated initial
axes the widths are 0.52 and 0.48. These constants and the existing
world-X chart axis were chosen independently of target material labels,
target orientation and held-out images. Use the same fixed ambient
anchor for both arms:

```text
l = 0.04², u = 1
a = (u+l)/2, b = (u-l)/2
F(z) = a + b z / sqrt(1+z²)
z_hi = F^-1(0.52²), z_lo = F^-1(0.48²)
H0 = z_hi I + (z_lo-z_hi) e_x e_x^T
Qt0 = F(U^T H0 U)       # U is an orthonormal shading tangent frame
Qw0 = U Qt0 U^T
```

The major axis is normal × world X when its norm is nonzero. The
tangent directions at a world-X normal are isotropic, because the
projected rank-one term vanishes. The A chart uses its existing world-Y
fallback below a cross-product norm of 1e−6. In that region the gap
vanishes quadratically and any difference from the analytic director
is below the native Q gate. The maximum starting gap is .0799,
below the existing .1 threshold for scoring orientation. Orientation
is therefore ineligible at initialization even where the chart
rotation has a nonzero derivative.

D9-C uses D8-C's hidden subnets with zero final gains and a constant
valid zero-angle encoding. Its two width subnets output residual
logits, initially zero. Add each residual to the inverse-sigmoid of
the corresponding anchor width scaled to [.04,1]. The residual
angles still use the chart and contribute through the usual tensor
rotation. D9-T adds its six ambient residual outputs, initially zero,
to H0 before projection and bounding. Its predictor uses ambient
coordinates throughout; the computational tangent frame only
evaluates a covariant matrix function.

The anchor offsets are fixed throughout optimization. The chart
offset varies with the shading normal. Thus D8 versus D9 measures
sensitivity to these initialized parameterizations. It cannot separate
the width scale, added weak anisotropy and geometry-dependent coordinate
centering. The paired training comparison within D9 holds the physical
start fixed.

For each width residual at zero, the width derivative is
`0.96 sigma(1-sigma)`, approximately .24. A chart rotation's
tangent-Q derivative has Frobenius norm `sqrt(2)|mx²-my²|` per radian.
This is nonzero away from the isotropic directions and vanishes
exactly at them. In the ambient arm the diagonal tangent derivatives
are F′(z) = b/(1+z²)^(3/2), and the off-diagonal uses its continuous
divided difference. At an isotropic normal all three tangent
variations still have nonzero local derivatives. The pointwise
ambient projection has a three-dimensional nullspace.

## Launch and evaluation

Keep the D8 known-light 24-point/32-draw native gate. Use the same
24-point paired learned-light gate with tolerances 2e−8 for raw Q
and 1e−6 for RGB components. Require bitwise shared scalar and light
state and exact scalar predictions before updating. Verify saved full
initial exports as part of the offline comparison. Use the same D6
neighborhood arrays, sampler history, reserved diagnostic edges and
snapshot limits.

Use the reviewed corrected image and A100 GPU 0. Bounds are 19,250
shader calls, 15,113,330,688 shadow-ray slots, 30 component reverse
sweeps and 1,280,000 surface pairs per arm. Use four CPU threads,
64 GiB host RAM without swap, 128 PIDs, 25% CUDA allocator, 8 GiB
output per arm, a 7,500 s foreground watchdog and 30 s grace.
The host/inner arm limits are 3,600/3,540 s. The approved launcher
keeps source and inputs read-only and runs non-root without network.
There are no new references or primary-ray geometry queries.

Save five checkpoints and six complete evaluation sets at the old
2,000-update interval. Keep 10,000 as the primary endpoint, comparing
each D9 arm with its D8 counterpart, the other D9 arm, and D6 as
historical context. Report overall/held-out Q, cutoff counts,
orientation eligibility and joint-eligible tails, bias and MAE of both
widths, gap and saturation distributions, held-out RGB, both light
error measures, camera 28 component/radial errors, camera 34 coverage
strata, reserved-edge transported Q and computational costs. The
camera 28 archive diagnosis defines a post-hoc localization region;
its measurements play no role in this launch or its thresholds.
[Camera 28 offline report](../DONUT_CAMERA28_RING_D9_V1.md).
