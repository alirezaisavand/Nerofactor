# E.11E completed reference follow-up

[Scientific results](../../docs/BELL_QUADRATURE_E11E_RESULTS_V1.md),
[identity manifest](../../docs/evidence/BELL_QUADRATURE_E11E_RESULTS_V1.json),
[unchanged scientific protocol](../../docs/protocols/BELL_QUADRATURE_E11E_V1.md).

Three CSVs and three derived SVGs are tracked; native nodes/visibility/components,
receipts and original analyses remain at the explicit external manifest paths.
Copied artifacts agree byte-for-byte. Sparse maps show actual84 probe positions
without interpolation, with common per-factor scales and the unchanged local
response gates. They are not full images or dataset admission. The source study
converges on this panel; original E.11D remains unaccepted.

To reproduce **file-only** verification and figures in a new output:

```bash
cd /home/isavand/paper/Nerofactor-3d2-e2a
MPLCONFIGDIR=/tmp/e11e-analysis-mpl python3 -B -m experiments.analyze_bell_quadrature_e11e \
  --batch /home/isavand/paper/evidence/bell-quadrature-e11e-4f228ac5e56c4803b7b33772680c7a7f \
  --benchmark /home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark \
  --pilot /home/isavand/paper/evidence/bell-highlight-e11d-08fb9def2696483ba97f8828fa86da6c/pilot \
  --output /home/isavand/paper/evidence/bell-e11e-offline-review-NEW-UNUSED-ID
```

This imports no Torch/native science and executes no BRDF or ray tracer. Existing
output directories are rejected; original evidence is never overwritten. Earlier
analysis-v1 is preserved; final v2 adds explicit coarse-grid response summaries.
Different NumPy/Matplotlib versions may alter diagnostic/figure bytes within
documented replay precision, but identities/counts/scientific decisions must
match exactly. No next rendering, full dataset or training is authorized.
