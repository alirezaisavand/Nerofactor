# Donut D4/D5 factorial results package

The [scientific report](../../DONUT_KURT_DIFFUSE_WHITE_ABLATION_V1_RESULTS.md)
explains the continuous interactions and the evidence limits.

- [Pooled trajectories](endpoint-metrics.csv) and [per-camera metrics](per-camera-metrics.csv)
  cover all arms and all six saved endpoint evaluations.
- [Factorial effects](factorial-effects.csv) tabulates D4−D3, D5−D2, D4−D5
  and the difference of the two removal effects. RGB and illumination
  relative errors are stored as fractions. Orientation values are in degrees.
- [Analysis](analysis.json) records compute, calibration, the illumination
  diagnostics and original component-gradient probes.
- [Camera 34 strata](camera34-coverage.json) use the exact earlier coverage
  assignments for all five arms. They retain the original raw axial angle
  diagnostic. The main orientation summaries apply the frozen eligibility rule.
- [Artifact manifest](artifact-manifest.json) gives paths and identities for
  the run receipts, logs and checkpoints. Every full saved map and image
  at updates 0, 2k, 4k, 6k, 8k and 10k is indexed by the original receipt.
- [D4 RGB progression](progression-D4-all-cameras.png) and
  [D5 RGB progression](progression-D5-all-cameras.png) show all six cameras.
  The [D4 material progression](progression-D4-camera0.png) and
  [D5 material progression](progression-D5-camera0.png) show material,
  orientation and component maps for the fixed camera 0.

The [file-only analysis](../../../experiments/analyze_donut_diffuse_white_ablation.py)
recreates the figures and numerical results from the frozen source/data
and completed receipts.

```bash
python3 -B experiments/analyze_donut_diffuse_white_ablation.py --output /tmp/donut-factorial-analysis-new
```
