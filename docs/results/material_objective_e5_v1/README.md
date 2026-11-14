# E.5 saved-data results

See the [scientific report](../../MATERIAL_OBJECTIVE_E5_RESULTS.md) and
[frozen protocol](../../protocols/MATERIAL_OBJECTIVE_E5_V1.md).

- `comparison.json`: full objective, sensitivity and external measurements,
  direct-control effects, unchanged recovery flags and sampling comparisons.
- `objectives.csv`: 12 fixed point/quartet fits plus the final direct fit.
- `sensitivity.csv`: nine scaled local Jacobian spectra and weak directions.
- `external.csv`: five physical points × three original evaluation splits.
- `direct_trajectory.csv`: states 0–50, correctly pairing pre-update loss with
  the same physical state. No intermediate held-out results were generated.
- `figures/`: four scientific figures, each lossless PNG and vector PDF.
- `integrity.json`: independent saved-array verification and the unsealed
  final-Adam-snapshot provenance limitation.
- `evidence_manifest.json`: original raw locations, sizes, hashes and launch
  sidecars; no raw archives copied into Git.
- `package_manifest.json`: exact-byte comparison with original analysis files.
- `analyze_e5.py`: analysis/regeneration with scientific imports blocked.

Offline regeneration requires the admitted original evidence and already
installed NumPy/Matplotlib. Choose a **new** output directory; originals must
not be overwritten. From the repository root:

```bash
python3 docs/results/material_objective_e5_v1/analyze_e5.py \
  --evidence-parent /home/isavand/paper/evidence \
  --output-dir /home/isavand/paper/evidence/material-e5-analysis-REPLACE_WITH_FRESH_ID
```

No Torch, model construction, shader, reference generation, optimizer update or
scientific derivative call occurs. Existing pure metric bodies are extracted
into a private AST namespace after source-hash verification. Float32 reduction
checks use a gamma-72 arithmetic envelope; scientific admission tolerances
remain unchanged. Figure/PDF bytes can vary with timestamps and library
versions; packaged originals retain their recorded hashes.
