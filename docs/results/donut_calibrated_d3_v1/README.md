# Donut D3 results package

See the [scientific comparison](../../DONUT_KURT_CALIBRATED_D3_V1_RESULTS.md).

- `endpoint-metrics.csv` compares all/fitting/held-out observations at
  six matched endpoints for D1, D2 and D3.
- `per-camera-metrics.csv` contains the 108 individual camera/endpoints.
- `analysis.json` records illumination errors and means, component gradients,
  execution counts, overhead and the recovered metadata discrepancy.
- `coverage/` preserves the existing camera 34 labels and adds D3 outcomes.
- `artifact-manifest.json` binds the original execution artifacts, checkpoints,
  and accepted reference and control receipts to their SHA-256 hashes.
- Eight figures show RGB reconstruction, material maps, progression,
  recovery curves, final error distributions, loss gradients, illumination,
  and camera 34 coverage.

The [file-only analysis script](../../../experiments/analyze_donut_calibrated_d3.py)
recreates the package in a new directory from its exact frozen evidence.
The figures use display clipping only and all metrics use linear arrays.
Full-resolution saved maps and checkpoints remain in the original batch.
`*_abs_*` columns are absolute errors, and `*_predicted_mean` columns are
predicted material values. Orientation distributions use the frozen anisotropy
eligibility rule, except for the original camera 34 strata's raw axial diagnostic.

```bash
python3 -B experiments/analyze_donut_calibrated_d3.py --output /tmp/donut-d3-analysis-new
```
