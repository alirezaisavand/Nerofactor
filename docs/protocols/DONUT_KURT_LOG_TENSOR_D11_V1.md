# Donut D11 transported log-Euclidean Q regularization

D11 uses the D8-T ambient-tensor model and replaces its added transported
raw world-Q loss. Keep the D8-T seed 6033, exact fresh state including
buffers, width 0.25 isotropic initialization, corrected tensor shader,
learned directional light, legacy scalar/energy/specular/white losses,
Gaussian tapes, original optimizer and scheduler. Use all original
fitting observations. The normal-transport calculation, sampler and
distance scaling are shared with D8-T. No reference values or held-out
observations enter the spatial training term. The 485-pixel camera 28
ring is used only for evaluation.

## Tensor metric

For tangent basis U, unit shading normal n and physical tangent matrix
q, set Q = U q Uᵀ, A = Q + nnᵀ and B = log(A) = U log(q) Uᵀ. The
normal eigenvalue is one and its logarithm is zero. Use the minimal
rotation R that takes the second normal to the first. The D11 loss is

```text
λ mean_edges [(0.02 / distance)² ||B0 − R B1 Rᵀ||²_F]
```

For positive definite symmetric q = mI + S with trace-free S, let
r² = (S₀₀² + S₀₁²) / m². Evaluate

```text
log(q) = (log(det(q))/2) I + [atanh(r)/(m r)] S
```

Evaluate the ratio in float64 using log1p and use its analytic series
1 + r²/3 + r⁴/5 + r⁶/7 for r² ≤ 1e−6 before dividing by m. The unused
square-root branch receives a positive safe argument. This keeps the
gradient finite at repeated eigenvalues without an eigendecomposition or
eigenvector derivatives. The log is evaluated on native squared widths
in [0.04², 1]. The embedded matrices use the same training transport dtype
as D8-T. The logarithm is covariant under a tangent basis rotation and
normal transport. Axis swap and director sign leave physical q identical.
A proportional width change by factor c at both axes produces squared
log distance 8 log²(c) under exact transport, regardless of width scale.

## One fitting-only coefficient

Set λ to 0.05 times the ratio of raw-to-log pooled RMS Frobenius
gradients with respect to the two physical tangent Q matrices at each
pair. Use fixed right/down pairs of valid archived D8-T fitting pixels
in cameras 0, 50, 1 and 45 at updates 2,000 through 10,000 every 2,000.
Both pixels must lie on the same face or faces sharing an admitted D6
training edge, with normal separation ≤15° and chord distance within
[1e−6, 0.02]. Apply the D6 distance factor in both losses. This uses
55,075 pair observations across checkpoints. Read only geometry,
predicted Q, masks and fitting work logs. The procedure gives raw/log
unweighted RMS gradients of approximately 0.13938/97.77155 and freezes
λ at approximately **7.12784e−5**. The exact coefficient and all input
hashes are in the [coefficient report](../evidence/donut_log_tensor_d11_preparation_v1/coefficient.json).

The fit-neighbor chord distance is a scale proxy for the D6 mesh
centroid distance. Matching output-space derivatives does not imply
equal parameter gradients or equal weighted objective values. Report
actual gradients on the existing six native probes. The log metric
itself shifts emphasis to proportional changes in narrower widths.
Neither its scale nor its definition will be adjusted after the run.

## Native execution and measurements

Run one new D11 arm with exactly 10,000 Adam returns and the D8-T
initialization and sampling streams. Before any update, verify state
identity and byte equality of the full 24-point paired initial capture
against D8-T. Keep the D8-T known-light compatibility gate. Keep the
original D6 training domain, all 1,024 reserved edges and the exact
sampler snapshots. The two immutable D8-T initialization control files
contain JSON and arrays, without historical checkpoint deserialization.

Use the reviewed corrected image, one A100 GPU 0, four CPU threads,
64 GiB host RAM without swap, 128 PIDs, 25% CUDA allocator, and 8 GiB
outputs. Use a 3,720 s foreground watchdog with 30 s kill grace, a
3,600 s arm limit and a 3,540 s inner limit with 20 s grace. Work caps
are 19,250 shader calls, 15,113,330,688 shadow-ray slots, 30 component
reverse sweeps and 1,280,000 neighborhood pairs. Require separate exact
execution approval. Stop on any mandatory failure and preserve partial
evidence. There is no retry, new reference, historical rerun, sweep or
extension.

Save checkpoints and the existing visualization/array exports at
2,000-update intervals, plus the initial maps. Compare the 10,000
endpoint against D8-T and historical D6. Report absolute and relative
width errors, Q and eligible orientation distributions, gap and
saturation, RGB, scalar materials and lighting. Use the established
camera 34 strata and camera 28 ring. Report raw and log transported
squared residuals on the reserved edges. D11 evaluates both directly.
For D8-T and D6, recover log residuals from archived widths and raw-Q
squared residuals. In a common transported tangent plane, squared
Frobenius distance decomposes into the trace difference, a spectral-radius
difference and a nonnegative relative-axis term. The log replaces
trace/radius values by their log-spectrum counterparts and rescales the
relative-axis term by the product of the log divided differences. This
eliminates the need to infer ill-conditioned directors. Record any
roundoff correction to the historical squared residual term. Reject
archive discrepancies exceeding 1e−6 absolute in raw squared units.

Use the recorded network-gradient probes, loss trajectories, CUDA
forward time and overall wall time to quantify intervention and
overhead. D6 is historical context because its representation differs.
This single seed and smooth material field cannot establish performance
across shapes or at true material boundaries. Additional geometries and
heterogeneous anisotropy take priority after this one arm.
