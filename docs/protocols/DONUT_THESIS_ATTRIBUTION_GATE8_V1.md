# Publication Gate 8 — Original thesis Material A on Donut

## Design

Fit one fresh A-Original-Oracle history and reuse the accepted Gate 4 K1-Oracle history. Use the pinned original Material A source at `e1e859779301faced46d6476691e20b546c27ae0`. The comparison estimates the combined change from the original anisotropic implementation to the corrected K1 recipe within the controlled acquisition. The historical thesis images and historical joint lighting/geometry procedures are outside this experiment. [Attribution and external baseline assessment](../results/publication_gate8/README.md)

Reuse Gate 4 oracle positions, normals, directions, exact masks and RGB targets. Cameras 0, 50, 1 and 45 supply 15,191 fitting observations. Cameras 28 and 34 supply 7,306 held-out observations. Verify the identical 5,120,000-index byte stream and order. The known analytic L2 adapter uses the accepted corrected visibility backend on oracle geometry, with zero indirect and no point light or learned-light gradients. The only optimizer inputs are fitting positions, normals, directions and observed RGB. No GT widths, axes, region labels or held-out observations enter the objective. [Gate 4 acquisition and exclusion criteria](DONUT_MATCHED_GEOMETRY_RECOVERY_V1.md)

Construct the original full chart field at seed 6033. Verify all initialized parameters and buffers against the archived K1 hash with only the behavior-signature buffer substituted in a hash-only view. The original checkpoint retains its own distinct behavior signature. The sampler and original shader keep their own numerical arithmetic. Identical raw heads and sampling counts do not guarantee identical realized tangent frames, physical Q or RGB. Report initial head and frame differences without altering the original decoding or reinitializing either method.

Keep K1's linear Charbonnier, effective legacy spatial/material/energy/specular/white objective, original Adam/AMSGrad and warm-up/cosine schedule, and 0.7 clipping. K1 and original both use 512 diffuse and 1,024 specular proposals per query. Keep the original random request order and the original azimuth, elevation, specular PDF, half-angle, support and denominator arithmetic. The same seeds and tape schedule control the random draws. The actual sampled directions differ because the proposals differ. The legacy priors use original computed proxies in A and corrected computed proxies in K1. [E10 correction inventory](../NATIVE_MATERIAL_A_CORRECTIONS_E10.md)

## Native gates and interpretation of invalid frames

The gate on the same 24 oracle-material inputs and 32 repeats used by Gate 4 records discrepancy, numerical precision and convergence against the unchanged independent target. The corrected-law agreement threshold is reported for original A as a scientific result. A discrepancy alone is expected to be possible in this attribution task. Admission requires finite, nonnegative and exactly additive original radiance draws, state and input identities, correct known-light routing and the existing first-update backward/Adam checks. Stop on numerical, identity, gradient, budget or resource failure. Preserve partial outputs and receipts.

Capture raw pre-display radiance from the original reduction and retain the exact native arithmetic and derivatives. At steps 0, 2,000, 4,000, 6,000, 8,000 and 10,000, export eight-draw L2 RGB/components, raw widths, color, metallic, orientation heads and decoded sampling frame, using the same seed and camera partition as Gate 4. The renderer never normalizes a bad original frame for export. Record the raw frame and its validity mask, with separate reasons for near-zero raw orientation, non-unit or incorrectly oriented frame, normal mismatch and out-of-domain widths. Use 2e-6 for frame checks and the existing 1e-6 raw-orientation threshold as an export eligibility diagnostic. Raw widths and scalar estimates can still be compared where meaningful. An invalid frame makes its physical Q and orientation unavailable. It remains in RGB coverage and invalidity accounting, and flags incomplete physical export if applicable. No fallback decoding or conversion of Q from nominal angle. For valid original frames use Q formed by the actual original sampling axes and widths. Under both methods use Gate 5's gap rule for orientation and compare on the intersection of eligible points. [Gate 5 metrics](DONUT_REPRESENTATION_GATE5_V1.md)

## Frozen endpoints and relighting

Evaluate physical Q median/p95, sorted widths and signed bias, common-eligible orientation, scalar errors, boundary jumps, camera/split differences and difficult cohorts. Include both common-valid physical metrics and full RGB coverage, with missing denominators visible. Preserve L2 trajectories and both the signed errors and positive minor-width tails. The primary endpoint is 10,000 optimizer returns.

After those returns, render the live original endpoint at Gate 6's frozen 45°-rotated L2 on cameras 28 and 34. Match the archived K1 native light, 32 draws, camera/raster order, 512-point batch sizes and seed schedule. Freeze the model with no gradient calls during relighting and verify identical model hash before/after. Report components, relative L2, linear RMSE, fixed-unit-peak PSNR and finite-draw standard error. Use the 7,306 common points, 38 extra positive-facing pixels and shared full-raster definition. Reuse the admitted independent reference, all masks and the archived K1 native predictions. An independent corrected-law rendering of original A predictions would change the model being tested and would require a separate intervention. [Gate 6 rendering and scopes](DONUT_RELIGHTING_GATE6_V1.md)

## Exact envelope and verification

| Work | Bound |
| --- | ---: |
| Optimizer returns | 10,000 |
| L2 shader calls and shadow slots, including diagnostic/trajectory work | 12,288 / 9,524,158,464 |
| Unseen-light shader calls and shadow slots | 480 / 360,972,288 |
| Gradient diagnostic reverse sweeps | 30 |
| Own-checkpoint isolated CPU roundtrips | 5 |
| Historical checkpoint deserialization, primary rays, new reference integrations | 0 |

Reuse corrected image `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`, one A100 GPU 0, four CPU threads, 64 GiB host memory without swap, 128 PIDs, 25% CUDA allocator and 8 GiB cooperative owned-output cap. Foreground 3,600-second watchdog with 30-second grace, inner 3,540 seconds with 20-second grace. Run read-only source/root/inputs, non-root, with networking disabled. Verify Docker image identity, immutable inputs, closed prior containers, complete work/cleanup and artifacts through the existing foreground workflow. Checkpoints are freshly generated and round-tripped only by their own training process. The authorization must bind the commit, freeze, image, arm and unused output path. One attempt only.

The launcher is [run_donut_thesis_attribution_gate8_v1.sh](../../experiments/run_donut_thesis_attribution_gate8_v1.sh). The file-only verifier reads sealed captures and reports, including per-camera masked comparisons and figures. It performs no shader evaluation or deserialization. [Offline verifier](../../experiments/analyze_donut_thesis_attribution_gate8_v1.py)
