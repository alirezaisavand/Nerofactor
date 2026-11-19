# Localized ambient material-field expressivity v2

The four-arm supervised comparison tests whether local coordinates or more
fitting observations help the existing ambient tensor field retain sharp
roughness boundaries. Carry over the V1 analytic targets, initialization
seeds 6033 and 6034, held triangles, independent points and boundary probes
on Donut and Bell. Training sees positions, normals and oracle world Q.
There are zero RGB renders and zero illumination updates.

| Arm | Ambient predictor | Fitting points |
| --- | --- | ---: |
| E1 | Existing global D8-T | 4,096 |
| E2 | Existing global D8-T | 16,384 |
| E3 | Global D8-T plus localized residual | 4,096 |
| E4 | Global D8-T plus localized residual | 16,384 |

The dense panel begins with the exact 4,096 V1 observations in their
original order. Append 12,288 draws with seed 74174 using the V1
area-proportional face and uniform barycentric sampler restricted to the
reconstructed, frozen fitting triangles. Compare the inherited panels to
the V1 array hashes. The additional draws cannot use held triangles.
Both localized arms use the same grid within each scene.

## Local residual and geometric budget

Add a six-component residual to the unmodified D8-T ambient logits in the
order xx, yy, zz, xy, xz, yz. Each of two vertex grids supplies one
trilinearly interpolated six-component value in native mesh coordinates.
Sum the grids and the global logits, then apply the exact existing bounded,
projected tangent-tensor decoder. The grids start at exactly zero, so
the global and localized fields begin with the same width-0.25 tangent-plane
tensor and identical global parameters. The existing global network and
scalar heads retain their V1 initialization and behavior.

Compute the leave-one-out Euclidean nearest-neighbor median of the sparse
fitting positions. Let L be the longest mesh bounding-box extent. Select
the largest fine long-axis count in [8, 12, 16, 24, 32, 48, 64] such that
its pitch L/count is at least three sparse medians and the total number of
trainable grid scalars is at most 8% of the 1,087,244 active global
parameters. The coarse grid uses half that count. On each level, ceil the
number of cells along the other axes to cover the mesh, retaining the same
pitch in all axes. Each vertex has six scalars. This deterministic rule
uses geometry and fitting positions only, with no training comparisons
or oracle-boundary and held-error inputs.

| Scene | Sparse / dense nearest-neighbor median | Coarse / fine XYZ cells | Added scalars | Localized total |
| --- | ---: | --- | ---: | ---: |
| Donut | .00976 / .00476 | 16×16×5 / 32×32×10 | 82,278 | 1,169,522 |
| Bell | .01681 / .00825 | 10×12×10 / 19×24×19 | 69,438 | 1,156,682 |

Each local prediction makes two trilinear lookups, drawing on at most
eight vertices per lookup. Report active parameter counts, wall time,
optimization and evaluation cost separately. Density effects are isolated
within either model. Localized-versus-global effects combine local support
and added capacity because all four arms retain the same global model.
A parameter-matched comparison with a larger global field would be needed
to attribute a gain specifically to locality.

## Matched fitting and evaluation

Use the V1 256-point shuffled-pass batches, world-Q mean squared Frobenius
loss, and Adam with LR .001, betas .9/.999, epsilon 1e-8 and zero weight decay.
Fit every history for 2,000 updates from the common initial physical tensor.
At the same density, the global and localized arms receive identical
minibatch streams. Fixed update counts expose the sparse samples to 125
passes and the dense samples to 31.25 passes. Record scores at 0, 250, 500,
1,000, 1,500 and 2,000. Save the per-point predictions at 1,000 and 2,000.
E1 at 1,000 must match the corresponding archived V1 D8-T predictions
exactly. The 2,000 endpoint is primary.

Report fitting and independent Q distributions, sorted width MAE,
orientation modulo director sign, and near-isotropic and exactly isotropic
subsets. At both retained checkpoints, compare orientation on the
intersection of the oracle and all four predictions that meet the V1
normalized squared-width gap of .1, reporting the size of that intersection.
Score the same fixed signed width jumps and projected transported Q jumps
on the V1 boundary probes. The held transition profiles use fixed
distance magnitudes [0, .005, .01, .02, .035] in native units, positive for
region 0 and negative for region 1. Boundary and region labels enter only
evaluation.

At update 2,000, compute the nearest fitting position to each held point
under the current density, then stratify relative Q error on all points,
smooth interior, boundary band and near-isotropic points. Use normalized
Euclidean distances with fixed bin edges at 0, .5, 1, 2 and 4 times the
scene's sparse leave-one-out median. Report the open upper bin, empty
bins, counts and Spearman association. Euclidean spacing is a proxy for
local sampling availability. It can cross nearby surfaces, and the
associations are descriptive. Compare E2−E1, E4−E3, E3−E1 and E4−E2 at
matched updates and report their interaction.

Consider a subsequent photometric test only if E4 versus E2 improves
median projected Q jump by at least 10 percentage points, reduces both
signed width-jump MAEs, and increases median smooth-interior relative Q
by no more than 5%, for both seeds on both scenes. This is a practical
screen, not a statistical proof. Differences between fitting and held
errors are part of the outcome. This design cannot establish an
irreducible field capacity limit from one optimizer budget.

## Bounded execution

One CPU-only run contains 16 histories and up to 32,000 Adam updates with
four CPU threads, hidden CUDA devices, 12 GiB address space, and 192 MiB
owned outputs. Use a 2,400-second foreground watchdog and 20 seconds of
kill grace. Frozen counters include every fitting point, evaluation point
and localized point-level lookup, with exact zero-render and zero-light
counts. Preserve partial reports on failure, with no retry, extra
architecture, or setting adjustment. Only saved-data analysis follows.
