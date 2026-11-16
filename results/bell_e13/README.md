# Native Bell E.13 results

The [scientific report](../../docs/BELL_THREE_ARM_RECOVERY_E13_RESULTS_V1.md)
separates successful native execution from unsuccessful physical recovery.
Execution commit: c372859dfa38789528517859a49e7f0f132cbdab. Each arm completed
2000 original Adam updates; no model selection by held-out truth.

|File|Contents|
|---|---|
|endpoints.csv|All three fixed endpoints, held physical/image metrics, resource observations|
|per_view_materials.csv|All128 views/arm, split labels, mean errors and validity counts|
|per_view_images.csv|All36 source-light/relighting image evaluations|
|training_50_update_bins.csv|Nonoverlapping50-update means; raw loss removes only the recorded original scalar weight|
|training.svg|Weighted loss, excess above the original Charbonnier floor, native gradient norms|
|physical_recovery.svg|Held-observation error distributions and unchanged descriptive targets|
|observations.svg|All six prescribed source/relighting views, GT/A/B/C, no exposure adjustment|
|analysis.json|File-only acceptance, endpoints, provenance and486 raw-output location/size/SHA-256 records|
|completion.json|Original offline completion seal and file identities|

The copied CSV/SVG/JSON bytes match the final external offline analysis package:
`/home/isavand/paper/evidence/bell-material-e13-v2-be28c65f719543e09065d728bd2bad71-analysis-v3`.
Its source run is the adjacent directory without the `-analysis-v3` suffix.
Raw checkpoints, full native reports and numeric exports stay outside Git. The
[publication manifest](../../docs/evidence/BELL_THREE_ARM_E13_RESULTS_V1.json)
binds original and tracked copies, raw checkpoint identities and focused test evidence.

The preserved failed file-only analysis-v1 encountered a host ImageIO API issue;
the tested Pillow reader replaces that API without changing decoded PNG values.
Analysis-v2 passed all numerical checks. Analysis-v3 only adjusts figure captions/
spacing, repeats file-only verification, and yields byte-identical CSVs to v2.
Neither analysis runs a model, restores an optimizer, deserializes a checkpoint,
integrates a BRDF or generates a new reference. Both original native v1 failure
and successful v2 run remain unchanged.

Regeneration uses
[experiments/analyze_bell_material_recovery_e13.py](../../experiments/analyze_bell_material_recovery_e13.py).
Supply a new explicit output directory; existing outputs are rejected:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/e13-offline-matplotlib \
python3 -B experiments/analyze_bell_material_recovery_e13.py \
  --batch /home/isavand/paper/evidence/bell-material-e13-v2-be28c65f719543e09065d728bd2bad71 \
  --output /home/isavand/paper/evidence/bell-e13-offline-review-CHOOSE-A-FRESH-ID
```

SVG metadata/IDs may differ when regenerated; the numeric CSVs are checked against
saved arrays. New physical criteria, model queries or scientific experiments are
not authorized by regeneration. The weak-cue, single-scene/single-seed panel does
not establish identifiability or a general ranking between estimator and core.
