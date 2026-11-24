# Final Dense Bell chart and ambient tensor comparison

Use the admitted Dense Bell v2 reference with 65,419 fitting observations
from 16 cameras and 7,082 held-out observations from cameras 105 and 114.
Retain the accepted K1-Dense run as the absolute reference.
[Reference and coverage](../results/bell_dense_reference_v2/README.md) ·
[K1-Dense and prior matched comparison](../results/bell_dense_known_l2_recovery_v2/README.md)

## Pair definition

Run Final-Chart and Final-Ambient for 10,000 Adam returns each, in that
order. Both use the shared validated D8 tensor BRDF evaluator, Gaussian
specular proposal, diffuse sampling, known analytic L2, zero indirect
illumination and corrected visibility backend. Use the original shared
trunk and scalar heads, 512 fitting observations per update, seed 6033,
unmodified learning-rate schedule, Adam/AMSGrad settings and gradient
clipping. The full material predictors have the same depth and hidden
widths. Final-Ambient has 1,486,100 trainable material parameters versus
1,485,584 for Final-Chart, a difference of 516 weights, biases and gains
from the two additional outputs. Neither arm trains light parameters.

Use the validated D9 geometry-only initialization in both arms. It
sets a common projected world-X anchor with widths between .48 and .52.
The two representations decode that same physical field through their
own coordinates. Widths coincide at world-X normals. The chart rotation
derivative can vanish there. Away from that degeneracy the weak starting
anisotropy gives the chart a nonzero angle derivative. The initialization
and its constants come from the accepted D9 protocol, with zero oracle
material supervision. The geometry-only CPU check across all 72,501
observations measured a maximum absolute initial world-Q difference of
2.50e−16. The native gate independently matches the historical initial
weight and buffer hashes, paired Q to 2e−8, RGB components to 1e−6, and
scalar predictions and shared state exactly.
[Weak initialization derivation](DONUT_KURT_PAIRED_TENSOR_D9_V1.md)

Use a common K1-derived linear Charbonnier objective. Keep the compatible
legacy scalar perturbation for albedo and metallicity, the accepted energy
and detached-NDF specular priors, and the .01 diffuse-white contribution.
Remove the chart-coordinate width and angle perturbations from both arms
using the accepted scalar-only perturbation variant. Omit the D6 spatial
penalty in both arms. Reserved edges and the 2,474 fixed boundary probes
remain evaluation-only. The versioned trainer overlay applies five exact
edits to the hash-pinned existing trainer. The original source remains
available byte-for-byte. The overlay selects the validated initialization,
disables its formerly coupled spatial path, and adds the pre-update
diagnostics and their work and objective reporting.

## Admission and execution

Use the existing 24-point, 32-draw independent-reference render gate and
the 24-point paired native RGB/Q gate. Additionally use 16 fixed incident
and Gaussian directions per point in a separate seeded float64 diagnostic.
Compare both arms' diffuse and specular BRDF, proposed half and reflected
directions, NDF and reflected PDF. Independently recompute the PDF from
the direct NDF and half-vector Jacobian. Require matching numerical outputs
within absolute 1e−8 and relative 1e−6 and finite, nonzero diagnostic
gradients in each final anisotropy and scalar bias group. Coordinate
gradients may differ between the parameterizations. The new diagnostic
makes two predictor calls and reverse sweeps per arm process, leaves
parameter gradients untouched and adds zero native shader or visibility
calls. Retain the native first-update gradient and warm-up gates, and
record actual index arrays before the corresponding optimizer work.
Their expected common hash is the verified K1-Dense index hash.

| Work | Maximum per arm |
| --- | ---: |
| Adam returns | 10,000 |
| Native shader calls | 17,186 |
| Shadow-ray slots | 13,210,927,104 |
| Retained component reverse sweeps | 30 |
| Initial diagnostic predictor calls and reverse sweeps | 2 each |
| Boundary predictor calls and points | 12 calls / 29,688 points |
| Spatial training pairs and new primary rays | 0 |

Use the existing corrected image on A100 GPU 0, four CPU threads, 64 GiB
host RAM without swap, 128 PIDs, a 25% CUDA allocator cap and 8 GiB
output per arm. The foreground watchdog is 7,500 seconds with 30 seconds
of kill grace. Arm limits remain 3,600 seconds on the host and 3,540
seconds inside the container, with 30/20 seconds of kill grace.
Keep original inputs/source and container root read-only, run non-root
and disable network access. Stop the batch on the first failed gate.
One explicit execution approval binds the source, freeze and fresh output.
No retries, extensions, reference generation or K1-Dense rerun.

## Method decision and publication readiness

Use step 10,000 as the primary endpoint. Compare held-out Q median and
p95, sorted width errors and signed bias, eligible and common-subset
orientation, anisotropy gap, fixed-probe signed jumps, linear RGB and
compute. Use all six saved states for the recovery trajectories and
predeclared 500-update fitting log windows for when differences emerge.
Record the first exact numeric log difference separately. Attribute
squared relative world-Q error to spectral width and eigenframe terms
using the symmetric float64 matrices, with a check against the saved Q
error. Report scalar errors separately. This diagnostic does not replace
the physical outcome metrics or infer photometric causation.

Promote ambient prediction to final validation if the matched comparison
improves physical recovery without a material deterioration in RGB,
boundaries, or compute and supports a competitive recipe against
K1-Dense. Retain the original method if the evidence favors it.
If the comparison remains unresolved, state the limiting evidence.
Use the frozen endpoint and report conflicting metrics. The K1-Dense
comparison still differs in its physical start, evaluator and sampling,
and some legacy prior terms. Only the final pair directly isolates the
predictor. A proposed method freeze must name the winning representation,
initialization, evaluator, proposal, objective and admitted output hashes.

Publication validation still needs repeated seeds and another scene or
acquisition, matched oracle versus reconstructed geometry, held-out
illumination under fixed radiometry, and an external baseline with
matched views, masks and compute accounting. Keep this one-seed control
separate from those future validations.

The focused CPU checks passed for BRDF, density and gradients, the
new admission and dispatch, and the saved K1-Dense width/eigenframe
diagnostics. Two legacy Donut runtime tests failed on pre-existing
exception-message expectations while exercising the unchanged guard.
Those legacy files are preserved.
