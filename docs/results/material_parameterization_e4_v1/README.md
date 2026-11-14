# E.4 AS1 pilot: saved-data results

See the [scientific report](../../MATERIAL_PARAMETERIZATION_E4_RESULTS.md) for
interpretation, all paired endpoints, limitations and the negative recovery
conclusion. Frozen preparation is a separate earlier commit; these analysis
files do not change that freeze or authorize another execution.

- `comparison.json` and `endpoints.csv`: full five-case paired state-100
  results, physical vectors/errors, component metrics and success decisions.
- `trajectories.csv`: 1,010 unique measured state rows (five cases × two methods
  × states 0–100). `candidate_measurements.csv` keeps all 512 AS1 fitting rows,
  including repeated state-10 measurements.
- `gradient_routes.csv`: 1,024 AS1 width-head chain measurements.
- `parameter_movement.csv`: 1,000 actual weight movements, 500 per method.
- `conditioning.csv`: 30 analytical output-coordinate Q-Jacobian measurements;
  not a full neural or observation Jacobian, and fitted states differ.
- `integrity.json`: independently recomputed saved-data checks.
- `figures/`: three static scientific figures, each PNG and PDF.
- `evidence_manifest.json`: raw paths, sizes, hashes and source identities;
  no raw checkpoint/signal arrays are copied to Git.
- `package_manifest.json`: exact-byte agreement with original analysis files.
- `analyze_e4.py`: offline regeneration, blocking scientific module imports.

Regenerate only from the admitted original saved evidence into a **new** output
directory. This does not rerender, reconstruct the neural field, regenerate
targets or update parameters. NumPy and Matplotlib must already be available.
From the repository root:

```bash
python3 docs/results/material_parameterization_e4_v1/analyze_e4.py \
  --evidence-parent /home/isavand/paper/evidence \
  --output-dir /home/isavand/paper/evidence/material-e4-analysis-REPLACE_WITH_FRESH_ID
```

The analyzer uses the accepted pure numeric metric function bodies in a private
AST namespace, rather than importing scientific experiment modules. It verifies
frozen source/input hashes, accepted control files against their Git baseline,
report/seal/receipt identities, trajectory chains and saved arrays before
writing. Numeric comparisons can be regenerated; PDF byte identities may vary
with timestamps/Matplotlib versions. The packaged originals retain their exact
recorded hashes.
