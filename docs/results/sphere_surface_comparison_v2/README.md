# Sphere surface-comparison results

See [the scientific report](../../SPHERE_SURFACE_COMPARISON_RESULTS_V2.md).
`results.json` contains all numerical observations, gradient probes, limitations
and external artifact locations/sizes/SHA-256 hashes. No checkpoint or large NPZ
is copied into Git. `artifact-manifest.json` binds copies to the original offline
analysis. Presentation-v2 uses taller figures; original figures are preserved.

To regenerate file-only analysis into a new directory, without model execution:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/sphere-surface-analysis-matplotlib \
timeout --signal=TERM --kill-after=10s 300s python3 -B \
  experiments/analyze_sphere_surface_comparison.py \
  --batch /home/isavand/paper/evidence/sphere-surface-v2-712adf6d038d49cc87807855ec360fd6 \
  --output /ABSOLUTE/NEW/ANALYSIS
MPLCONFIGDIR=/tmp/sphere-surface-analysis-matplotlib python3 -B \
  experiments/visualize_sphere_surface_components.py \
  --report /ABSOLUTE/NEW/ANALYSIS/results.json \
  --output /ABSOLUTE/NEW/ANALYSIS/RGB-component-comparison.png
```

NPZ loading uses allow_pickle=False. Checkpoints are hashed only. The source
freeze/input gates must still pass; this does not launch either training arm.
Raw orientation/projector/Q displays are not ground-truth error maps. No new
research thresholds or training interventions are inferred from the figures.
