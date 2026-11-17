# Donut D1/D2 results

See [scientific report](../../DONUT_KURT_RECOVERY_V2_RESULTS.md).

- `analysis.json`: provenance, verification, all endpoint/split summaries,
  component-gradient probes and numerical replay differences.
- `endpoint-metrics.csv`: all/fitting/held-out summaries at all six endpoints.
  `predicted_*_mean` denotes predicted values; `*_width_mean` / `metallic_mean`
  denote mean absolute errors. Orientation columns are degrees.
- `artifact-manifest.json`: absolute external evidence/checkpoint identities.
- `finalization-validation.json`: final file/static validation identities.
- Six PNGs: RGB comparisons, camera0 material maps/progression, recovery curves,
  training losses/gradients, diffuse/specular components and illumination.

Raw maps, ten checkpoints and full training logs remain in the original batch.
RGB figures use saved data only. Error color caps are visual, not metric bounds;
HDR environment previews explicitly clip above one. Raw linear values remain
unchanged. Camera0 is the first fixed fitting camera, not a selected best case;
the RGB sheet includes all six cameras and the problematic held-out camera34.
World-director RGB encodes projector off-diagonal entries shifted by0.5, not a
globally defined scalar angle. Backgrounds and display-only caps are visible.

To regenerate into a **new** directory (no renderer/model queries):

```bash
python3 -B experiments/analyze_donut_recovery_v2.py --output /tmp/donut-recovery-analysis-new
```

The script refuses an existing output directory and checks exact source/input
identities first. It requires NumPy, Pillow and Matplotlib already installed.
