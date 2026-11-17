# Verified donut reference v2

See [results](../../DONUT_KURT_REFERENCE_V2_RESULTS.md) and the frozen
[definition](../../protocols/DONUT_KURT_REFERENCE_V2.md).

- verification.json: file-only checks and full-precision numerical results.
- view-metrics.csv: all six views, including fitting/held-out membership.
- reference-RGB-diffuse-specular.png: exact copied native preview contact sheet.
- convergence.png/.pdf: saved-array comparisons against unchanged absolute bounds.
- artifact-manifest.json: external raw data and tracked figure/report identities.

No raw NPZ reference archives are copied into Git. `researcher_admitted_dataset`
remains false until the researcher explicitly approves this verified package.

The verifier can run again against the original immutable evidence, selecting a
NEW small-results destination. It performs no rendering/integration/training:

```bash
cd /home/isavand/paper/Nerofactor-donut-recovery
python3 -B -m experiments.finalize_donut_reference_v2 \
  --batch /home/isavand/paper/evidence/donut-kurt-reference-v2-262b3cfcf73340a0867829c48c0198f7 \
  --prior /home/isavand/paper/evidence/donut-kurt-reference-v1-5df0234193664e9b93cae546542ca7a2 \
  --destination "$PWD/docs/results/donut_kurt_reference_v2_reverification"
```

Do not overwrite the current results package or rerun scientific generation to
recreate these figures. Native generation has completed; training remains gated.
