# Donut D6 results package

See the [scientific report](../../DONUT_KURT_SURFACE_Q_D6_V1_RESULTS.md) for the
fixed endpoint comparison and its scope.

- [Pooled metrics](endpoint-metrics.csv) and [per-camera metrics](per-camera-metrics.csv)
  include six endpoint evaluations for D2, D4 and D6. RGB errors use linear
  values. Orientation errors apply the frozen anisotropy-gap criterion.
- [Analysis](analysis.json) preserves the sampler replay, timings,
  illumination summaries, independent diagnostic neighborhoods and original
  component-gradient probes.
- [Camera 34 strata](camera34-coverage.json) retain the established coverage
  assignments and raw axial orientation diagnostic.
- [Artifact manifest](artifact-manifest.json) indexes the original receipts,
  logs, reference inputs and checkpoints. Full maps at every saved endpoint
  are available under each listed arm directory. The original receipt gives
  each output identity.
- [Six-view progression](D6-six-view-progression.png) and
  [camera 34 material progression](D6-camera34-material-progression.png)
  show all scheduled D6 evaluations.

The [file-only replay](../../../experiments/analyze_donut_surface_q_d6.py)
regenerates the summaries and figures from the frozen data in a new directory.

```bash
python3 -B experiments/analyze_donut_surface_q_d6.py --output /tmp/donut-d6-results-new
```
