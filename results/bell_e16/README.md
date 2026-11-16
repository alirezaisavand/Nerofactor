# E.16 recorded native neural recovery

See the [scientific results](../../docs/BELL_NEURAL_OBSERVATION_E16_RESULTS_V1.md)
and [evidence manifest](../../docs/evidence/BELL_NEURAL_OBSERVATION_E16_RESULTS_V1.json).
All tables/figures are exact copies of the final offline analysis; their source
paths, sizes and SHA256 values are recorded. Raw weights, Adam moments, renderer
captures and trajectories remain in the original external evidence directory.

- `endpoints.csv`: all12 captured states, physical and photometric metrics.
- `point_errors.csv`: all84 points at every captured state and both teachers;
  no quality-based exclusions (1,176 rows).
- `gradients.csv`: all six material/arm/phase summaries, including zero-gradient
  and clipping counts, not a replacement for external per-step logs.
- `physical_recovery.png`, `paired_Q.png`, `photometry_and_variance.png`:
  physical distributions, matched-endpoint comparisons and native fitting noise.

The [analysis code](../../experiments/analyze_bell_neural_material_e16.py) reads
only hash-admitted numeric evidence (allow_pickle=False); it does not import
Torch or execute a model, shader or optimizer. To regenerate into a **new**
directory from the repository root:

```bash
MPLCONFIGDIR=/tmp/e16-review-mpl python3 -B experiments/analyze_bell_neural_material_e16.py \
  --batch /home/isavand/paper/evidence/bell-neural-e16-05d9ed3a61a34a72b0752672968d5cae \
  --authorization /home/isavand/paper/evidence/bell-neural-e16-preparation-bde518df05af496ca3e987da97926209/execution-authorization-v1.json \
  --output /tmp/bell-e16-offline-review-new
```

The source evidence must be available and unchanged. This is an offline command,
not permission to rerun E.16. Its exact scalar results are reproducible; generated
figure bytes can depend on Matplotlib/font versions. The original copied figure
identities, rather than a cross-environment byte-equivalence claim, are preserved.
