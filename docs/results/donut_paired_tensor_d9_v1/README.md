# Donut D9 results package

The [results report](../../DONUT_KURT_PAIRED_TENSOR_D9_V1_RESULTS.md)
compares the D9 pair with both D8 arms and historical D6.

- [Pooled](endpoint-metrics.csv) and [per-camera metrics](per-camera-metrics.csv)
  cover the six evaluation states of all five runs.
- [Camera 34 coverage](camera34-coverage.json) retains the archived strata.
  [Camera 28 component errors](camera28-radial-errors.json) use the old
  radius quartiles and the same 485-pixel D8-T excess region for all arms.
- [Analysis](analysis.json) records matched initialization, shared sample
  counts, the joint orientation-eligible subset, raw Q export checks,
  illumination, gradient probes and resource use.
- [Artifact manifest](artifact-manifest.json) links the original checkpoints
  and full native render, material and environment exports.

The [file-only replay](../../../experiments/analyze_donut_paired_tensor_d9.py)
regenerates the comparison in a fresh directory:

```bash
python3 -B experiments/analyze_donut_paired_tensor_d9.py --output /tmp/donut-d9-results-new
```
