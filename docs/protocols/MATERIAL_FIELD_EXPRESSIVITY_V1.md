# Supervised chart and ambient material-field diagnostic

Fit the existing heterogeneous Donut and Bell analytic anisotropy fields
directly in world Q. The task supplies oracle Q at fitting coordinates and
has no RGB rendering, lighting model, appearance loss, visibility queries or
spatial regularizer. Its results characterize supervised field fitting.
Any implications for inverse rendering require separate evidence.

## Models and optimization

Use the frozen Material A positional trunk and anisotropy heads with the
existing D8-C chart or D8-T ambient tensor decoder. The CPU adapter admits
the existing isolated source definitions. Both fields begin at the D8
physical tangent-isotropic width 0.25 with identical shared weights.
The chart uses four head outputs and the ambient field uses six, with the
existing hidden widths. Count the trainable parameters. Scalar heads keep
their common initial values and remain frozen and outside the loss.
The chart retains its raw-orientation check. At exact isotropy, changes to
the chart angle alone have zero first derivative on Q, while the ambient
field can change tensor components directly. The experiment measures this
initialization and all subsequent optimization as part of the chosen setup.

Use two initialization seeds, 6033 and 6034, for each scene and method,
yielding eight histories. Both arms at a given seed receive the same
sequence of 256-point minibatches from repeated full shuffled passes
through 4,096 fitting samples. Optimize the mean squared Frobenius distance
between predicted world Q and oracle world Q with Adam at learning rate
0.001, betas 0.9 and 0.999, epsilon 1e-8 and zero weight decay. Use 1,000
updates per history, with evaluations at 0, 250, 500 and 1,000. There is
no held-out selection or early stopping. If the raw chart guard raises, keep
that history incomplete and continue with the remaining independent histories.
Any other mandatory error stops the experiment.

## Samples and comparisons

Reserve entire triangles for evaluation, including all boundary probe
triangles. Use a fixed permutation of the other faces to fill the 25% held
triangle quota. Use independent seeded area-proportional face and uniform
barycentric draws for 4,096 fitting and 8,192 held samples per scene.
Donut uses smoothly interpolated shading normals. Bell uses its facewise
shading normal convention. Boundary brackets are the geometry-only
opposite-region edge-linked centroid pairs on Donut and the existing
same-face plane crossings on Bell. Their fixed panels contain 192 and 256
pairs, respectively. The expected held panels contain 183 near-isotropic and
1,400 boundary-band points on Donut, and 564 near-isotropic and 827 boundary-band
points on Bell. Report these cohorts separately, including exactly isotropic
observations where present. This split interleaves held triangles across the
surfaces and tests spatial interpolation. It does not measure extrapolation
to another object or a completely unseen region.

Report fitting and held median and p95 relative Q, both sorted-width MAEs,
joint eligibility and axial director errors, gap behavior, signed
width jumps, and projected transported Q jumps. Use the existing 0.1 gap
criterion for orientation in both prediction and oracle. Compare learning
curves, raw gradients and seed variation before interpreting residual errors.
Good fitting accuracy with worse held accuracy indicates limited spatial
generalization under the tested split. Persistently high fitting error at the
fixed budget can reflect optimization difficulty, approximation limits or both.
The comparison cannot prove an irreducible capacity limit or exact recovery of
arbitrarily sharp discontinuities.

## Bounds and outputs

Use one separate CPU-only launch with four threads, hidden CUDA devices,
12 GiB address space and 128 MiB of output. The foreground watchdog is
1,800 seconds with 20 seconds of grace. The maxima are 8,000 Adam updates,
2,048,000 fitting point slots and 440,320 evaluation point slots including
initial parity. Save the frozen sample arrays, endpoint predictions, loss and
gradient traces, per-cohort scores and hashes in the fresh output directory.
Use no inverse-rendering checkpoint. The geometry and analytic target
definitions are the inherited benchmark inputs.
