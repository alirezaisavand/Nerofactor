# E.6 physically informed optimizer results

Read the [scientific report](../../MATERIAL_OPTIMIZER_E6_RESULTS.md) and
[frozen protocol](../../protocols/MATERIAL_OPTIMIZER_E6_V1.md).

- `comparisons.csv`: two predeclared work-matched comparisons; neither has an
  available matched-20 LM endpoint.
- `endpoints.csv` / `comparison.json`: all physical/component/total-RGB errors,
  success flags, timings and work, including separately labelled unmatched
  Adam-20 context. No replacement LM checkpoint was selected.
- `trials.csv`: all 35 trials, including rejections and nonqueried zero steps.
- `trajectories.csv`: accepted-state physical/MSE histories at the cost of
  their measurements, including Jacobian/rejection/evaluation work. No new
  intermediate held-out renderings or ground-truth optimization occurred.
- `conditioning.csv`: all 24 local physical Jacobian spectra/gradient norms.
- `figures/`: four scientific figures, each PNG and PDF.
- `integrity.json`: independent saved-array checks and evidence limitations.
- `evidence_manifest.json`: raw paths, sizes and hashes; raw archives remain
  outside Git. `package_manifest.json` verifies exact copied-file agreement.
- `analyze_e6.py`: offline verification/regeneration, blocking Torch and all
  experiment module imports.

From the repository root, with original admitted evidence and existing
NumPy/Matplotlib, regenerate into a **fresh** directory:

```bash
python3 docs/results/material_optimizer_e6_v1/analyze_e6.py \
  --evidence-parent /home/isavand/paper/evidence \
  --output-dir /home/isavand/paper/evidence/material-e6-analysis-REPLACE_WITH_FRESH_ID
```

This recomputes metrics, algebraic acceptance checks and plots from saved data;
it does not query the renderer, execute an optimizer or regenerate a reference.
Float32 loss checks use a gamma-72 reduction envelope. Existing pure metric
bodies are extracted through a hash-checked AST namespace. PDF/figure bytes may
vary with library versions/timestamps; packaged originals retain exact hashes.
