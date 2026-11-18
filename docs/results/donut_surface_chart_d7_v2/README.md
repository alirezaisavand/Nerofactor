# Donut D7 v2 results package

The [results report](../../DONUT_KURT_SURFACE_CHART_D7_V2_RESULTS.md) relates
the chart control to D2 and D6 at the fixed 10,000-update endpoint.

- [Pooled metrics](endpoint-metrics.csv) and [per-camera metrics](per-camera-metrics.csv)
  preserve all six saved states per arm.
- [Camera 34 coverage](camera34-coverage.json) uses the frozen radius, triangle
  and view-angle strata. [Analysis](analysis.json) includes illumination,
  reserved-edge consistency, surface coverage, resource use and gradient probes.
- [Artifact manifest](artifact-manifest.json) lists the receipts, interrupted
  evidence, checkpoints and paths to the full saved maps.
- [Camera 34 material progression](D7-camera34-material-progression.png) shows
  RGB, albedo, widths, world orientation and the diffuse/specular components
  across all six saved D7 evaluations.

The [file-only replay](../../../experiments/analyze_donut_surface_chart_d7.py)
regenerates the metrics and figures in a new directory:

```bash
python3 -B experiments/analyze_donut_surface_chart_d7.py --output /tmp/donut-d7-results-new
```
