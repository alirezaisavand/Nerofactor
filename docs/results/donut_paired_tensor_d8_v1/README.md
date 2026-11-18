# Donut D8 results package

The [results report](../../DONUT_KURT_PAIRED_TENSOR_D8_V1_RESULTS.md) compares
the two representations at the fixed 10,000-update endpoint and gives D6 as
historical context.

- [Pooled metrics](endpoint-metrics.csv) and [per-camera metrics](per-camera-metrics.csv)
  preserve all six recorded states per arm.
- [Camera 34 coverage](camera34-coverage.json) retains the frozen radius,
  triangle and view-angle strata. [Analysis](analysis.json) records the joint
  orientation-eligible subset, direct tensor export checks, native gates,
  illumination, spatial diagnostics, gradient probes and timing.
- [Artifact manifest](artifact-manifest.json) identifies the receipts,
  checkpoints and saved evaluation directories. The original D8 directories
  contain the full raw world-Q, width, orientation, RGB, material and
  illumination outputs at every evaluation.

Use the [file-only replay](../../../experiments/analyze_donut_paired_tensor_d8.py)
to regenerate the analysis and figures:

```bash
python3 -B experiments/analyze_donut_paired_tensor_d8.py --output /tmp/donut-d8-results-new
```
