# Donut diffuse-white ablation D4/D5

Complete a matched 2×2 factorial with accepted D2/D3 and two fresh arms.
The D2/D3 evidence is fixed at the result commits cce13b5795f1818434abfdfc096be7f87469ccad
and e6fd936f1a57a3ae03a3fe06c4996f42d2011920. D1 remains a known-light
reference outside the factorial.

| Learned directional illumination | Diffuse-white objective term on | Diffuse-white objective term removed |
| --- | --- | --- |
| Free mean | D2 | D5 |
| Fixed calibrated RGB mean | D3 | D4 |

The training objective for D4/D5 is the existing RGB loss plus the unchanged
legacy material prior. Compute the original diffuse-white loss with its
0.01 coefficient on every update and log it with the actual optimization
loss. At updates 1, 2,000, 4,000, 6,000, 8,000 and 10,000, use the existing
component-gradient diagnostics to measure that term separately. Its
counterfactual gradient is diagnostic only, and never accumulates into
the optimizer gradient. Keep the RGB, material, energy and specular probes.

D4 retains D3's differentiable spherical normalization at the frozen
128×256 grid and the same three calibrated RGB means. D5 calls the same
unconstrained directional MLP as D2. Both arms use the admitted six-camera
reference, four fitting views, two held-out views, identical fresh material
and light state, seed6033, original 512-ray batches, shader samples,
full field, corrected-core BRDF, existing BVH, original Adam/AMSGrad
and WarmUpCosLR, clipping and zero indirect light. Enforce the accepted
initial state hash before updates in both arms. No GT material or held-out
observations enter training. Neither an energy nor a smoothness term is added.

The calibrated-mean record is now captured separately from per-camera
component means. This corrects `realized_mean_RGB` to a three-element list
while preserving the same calibration computation and acceptance gate.
The D3 receipt and saved maps retain their original bytes.

## Work and launch

One dedicated freeze binds the admitted reference, D2 initial-state identity,
D2 and D3 receipts, shared code and the new arm-specific work limits. Use the
existing launcher with `DONUT_PHASE=ablation` and its dedicated freeze. Run
D4 then D5 as separate cold processes. Each must supply a complete receipt
before advancement. Save all six evaluation states and the five own-run
checkpoints per arm at the existing 2,000-update cadence.

The authorized maximum is 20,000 completed Adam updates, 38,496 shader calls,
30,226,513,920 shadow rays and 60 component reverse-sweep slots for both
arms combined. D4 alone adds 10,006 calibration calls (327,876,608 MLP
direction inputs). D5 has no calibration work. Use the same reviewed
image, A100 GPU0, four CPU threads, 64 GiB host memory with no swap,
128 PIDs, 25% CUDA allocator fraction and 8 GiB owned outputs per arm.
Use a 7,500s foreground watchdog and 30s kill grace, 3,600s per-arm
host limit and 3,540s inner limit with 20s grace. Inputs, source and root
are read-only, the process is non-root and the container is offline.
Stop on failure and preserve receipts and partial artifacts. Do not retry
or extend. Another explicit approval is required for any subsequent launch.

## Fixed comparisons

Use the 10,000-update endpoints as primary. Compare D4–D3 (prior removal with
calibration), D5–D2 (prior removal without calibration) and D4–D5
(calibration without the prior). The interaction on each reported metric is
`(D4 - D3) - (D5 - D2)`. Positive interaction on an error means prior removal
increased error more, or reduced it less, in the calibrated condition.
A positive interaction on success counts has the opposite quality meaning.
Report differences on the same physical observations and original masks.
Each cell has one fixed seed, so interactions are descriptive.

Reuse the existing saved-map, illumination and trajectory analysis. Report
absolute/signed widths, log mean and ratio errors, Q and eligible orientation
distributions, color and metallic, fitting/held-out linear RGB, spherical
illumination means and relative error, component gradients and calibration
overhead. Apply the already frozen camera 34 coverage labels, same-triangle
strata and view-angle bins. Report the existing normal-transported surface-Q
diagnostic. Plot earlier endpoints for interpretation with no checkpoint
selection or tuning. The absence of a recorded diffuse-white gradient probe
in D2 is a missing comparison rather than a zero derivative. No learned
surface-Q ablation is part of this batch.
