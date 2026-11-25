# Publication Gate 5 — Matched Donut representation comparison

## Question and histories

Test whether the representation-level differences from the three-seed dense Bell comparison recur on heterogeneous Donut at seed 6033. Add one fresh Final-Chart-Oracle history. Use the completed Gate 4 Ambient-Oracle as the matched comparator and the completed K1-Oracle as a separate recipe reference. No historical training, reference integration or data generation is part of this protocol.

Bind the approved Gate 4 freeze, receipt manifest, native comparator captures, and original Bell Final-Chart recipe through the Gate 5 freeze. Reuse the exact Gate 4 tensor trainer source overlay. The only arm recipe changes relative to Ambient-Oracle are the selectable predictor and its expected full initial-state hash. The chart field uses 1,485,584 trainable material parameters and the ambient field uses 1,486,100. This difference follows from the output parameterization. Both use the same shared trunk.

## Frozen acquisition and method

| Item | Definition |
|---|---|
| Observations | Gate 4 oracle common support, 15,191 fitting and 7,306 held out |
| Fitting / held out | Original camera order 0, 50, 1, 45 / 28, 34 |
| Targets | Original admitted oracle linear RGB on those exact pixels |
| Lighting | Known analytic L2, zero indirect, corrected visibility BVH on the oracle mesh |
| Initialization | Seed 6033, shared base weights and scalar heads, D9 projected world-X weak start, major width 0.52 and minimum minor width 0.48 |
| Rendering | Identical corrected Kurt tensor evaluator, common world-Gaussian specular samples and inherited diffuse sampling, 512 diffuse and 1,024 specular samples |
| Objective | Linear RGB plus the compatible scalar material, energy, specular and diffuse-white legacy priors. Both arms omit chart-specific spatial anisotropy and the additional D6 surface-Q loss. |
| Optimizer | Original Gate 4 Adam/AMSGrad, epsilon 0.0001, warm-up and cosine schedule, gradient clip 0.7, 512 fitting observations per return |
| Progress | Fixed 10,000 returns, archives at 0/2,000/4,000/6,000/8,000/10,000, evaluation uses eight draws |
| Evidence | No historical checkpoint loading. Existing own-checkpoint roundtrips and original artifact validation remain in place. |

The frozen observation/target SHA-256 is `c3634a728c082605e36e5bb4e60263f1f26bfac8c86b442b30eea28167c34575`. The frozen fitting-index stream SHA-256 is `0c8b943813ee391218aa74d00f620c086cca509753a579f9d04c62cea588ce36`, with 5,120,000 little-endian uint32 indices in acquisition order. Check the stream from actual chart fitting accesses at completion.

## Admission gates

The foreground preflight admits the original Gate 4 inputs and manifest, checks both completed receipt identities and frozen work, and verifies the chart initialization anchor and archived Ambient-Oracle capture hashes. The native path checks immutable input file identities before constructing the field. It then uses the existing native paired-state gate, physical BRDF/proposal-density and gradient gates, and independent known-light compatibility gate. Replay both representations' archived Gate 4 initial arrays on the existing 24-point panel before any Adam return. Check field inventories, shape, dtype and finite values. Require exact scalar predictions, reference RGB, coarse RGB and pixel identities. Use absolute differences of at most `2e-8` for the paired world Q and `1e-6` for paired RGB components and the native compatibility draws. Apply the existing tensor physical gate tolerances of `atol=1e-8, rtol=1e-6` to BRDF, half/reflected samples, NDF and PDF. These replay allowances account for floating-point differences across processes and do not change native acceptance criteria. Report maximum drifts. The offline verifier also checks saved initial Q, scalar and RGB fields over all cameras on the complete observation support.

Any identity, parity, scientific validity, work, or resource failure stops the experiment and preserves available evidence. Use a fresh output once. A failed launch consumes the authorization. No retries, extensions, or hyperparameter changes.

## Execution envelope

A separate affirmative authorization must bind the final commit, exact Gate 5 freeze SHA-256, arm, corrected Docker image, one launch, and the fresh host output specified by the freeze. Use `experiments/run_donut_representation_gate5_v1.sh` in a foreground tmux terminal with `DONUT_REPRESENTATION_SOURCE` and `DONUT_REPRESENTATION_AUTH` set to the canonical checkout and bound authorization file. The launcher requires a clean checkout and no conflicting native Donut, Bell or geometry container. Use the existing corrected image `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331` on A100 GPU 0, four CPU threads and cores, 64 GiB host RAM with no swap, PID cap 128, 25% CUDA allocator fraction and 8 GiB output cap. The foreground Docker watchdog is 3,600 seconds with 30 seconds kill grace. The inner limit is 3,540 seconds with 20 seconds kill grace. The container is non-root, network-isolated and read-only except for its fresh output and runtime scratch. Maximum work is 10,000 Adam returns, 12,290 shader calls, 9,524,232,192 shadow-ray slots, 30 component reverse sweeps, and two explicit field probes and two derivative sweeps in tensor admission. There are zero new references and no independent historical model execution.

## Evaluation and interpretation

Run `python3 -B experiments/analyze_donut_representation_gate5_v1.py --freeze-sha256 <freeze-sha256> --output <fresh-absolute-analysis-directory>` after execution. Verify all output hashes, work and cleanup receipts, learning-rate streams, known-light route counts, initial-state identities and numerical captures. Use the step-10000 comparison for decisions. Plot fixed saved trajectories, retaining all physical errors and every observation on the common raster support. Score widths in sorted physical units, orientation on the frozen eligibility masks and the intersection for each comparison, relative world Q, scalar errors, RGB and signed width jumps with normal-transported Q. Report both splits and all cameras, including region, boundary-band, near-isotropic, grazing, nearby fitting and highlight cohorts. Point counts refer to image observations and may include repeated surface locations.

The chart/ambient pair isolates the predictor representation including its parameterization and small induced difference in parameter count. K1 uses a different initializer, renderer sampling path and prior implementation. The one-seed Donut result complements the three-seed Bell panel. It cannot establish cross-seed Donut robustness or a universal material-recovery advantage. Gate 4's historical reconstruction and its unresolved extraction provenance do not enter this oracle-only comparison.
