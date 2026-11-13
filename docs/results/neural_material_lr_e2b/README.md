# E.2b paper results

See the [publication-oriented report](../../NEURAL_MATERIAL_LR_E2B_RESULTS.md).
The scientific execution source is commit
`a1314e192ed87b70bcf5f78c223e319d8e3ce56f`; a later archival commit does not change
that producing-code identity.

## Contents

- `endpoints.csv`: all ten original paired endpoint rows, with physical values,
  errors, original success flags, saturation onsets and parameter motion.
- `comparison.json`: the small original machine-readable scientific summary.
- `trajectories.csv`, `sigmoid_boundaries.csv`, `adam_displacements.csv` and
  `gradient_routes.csv`: original recorded-data summaries supporting the figures
  and gradient-path findings. These are copied without numerical edits.
- `figures/`: three original analysis figures, each as a PNG preview and vector
  PDF for paper use. Every copy is byte-identical to the external evidence.
- `analysis_original.py`: byte-identical snapshot of the executed offline audit
  and analysis. It is an archival recipe, not a scientific-training entrypoint.
- `regenerate_figures.py`: portable CSV-only renderer, using the exact plotting
  block from the original analysis. It blocks Torch/scientific imports and needs
  only NumPy/Matplotlib plus the included tables.
- `evidence_manifest.json`: absolute locations, sizes and SHA-256 identities for
  original raw/analysis/failed-launch evidence and mappings for tracked copies.

## Figure regeneration

From the repository root, choose a **fresh** output directory outside Git:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 docs/results/neural_material_lr_e2b/regenerate_figures.py \
  --output-dir /tmp/e2b-paper-figures-new
```

The script does not overwrite existing directories or read model checkpoints.
It performs no neural-field, shader, optimizer, or reference queries. The same
plotting environment reproduces the supplied PNGs exactly; other library/font
versions can change rendering. Regenerated PDFs may have different creation-time
metadata, so byte identity is required for the archived copies, not promised for
every regeneration. Numerical tables and scientific definitions remain unchanged.

## Full original audit

`analysis_original.py` retains the exact historical host paths and accepted
commit guard used for the completed analysis. Do not execute it in this tracked
directory: its outputs are exclusive and it requires a fresh external folder.
For a full read-only reanalysis, retain the raw inputs identified by the manifest,
use a separate source checkout at the accepted scientific execution commit, and
copy the archived script into a fresh external output folder. Adjust only its
`REPO` path to that accepted checkout (and `ROOT` only if the hash-identical raw
evidence was mirrored). Preserve and record such path-only adaptations. The
script independently audits numeric arrays with `allow_pickle=False` and blocks
scientific model imports. No full audit or experiment rerun is needed to use the
paper figures and endpoint table.

The original experiment/analysis and earlier failed launch remain outside Git and
unchanged. No raw NPZ/checkpoint or large archive is included. The current result
does not authorize the proposed 100-update continuation.
