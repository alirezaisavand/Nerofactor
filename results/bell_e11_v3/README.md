# Bell E.11 v3 saved-evidence review

[Results and limitations](../../docs/BELL_ORACLE_E11_V3_RESULTS.md) and
[identity manifest](../../docs/evidence/BELL_ORACLE_E11_V3_DATASET_REVIEW_V1.json)
describe the single authorized generation. Only the small CSV, derived SVG
figures and review code are tracked. The original approximately246MiB batch and
full saved-array review remain under the explicit evidence locations in the manifest.

`review_saved_evidence.py` performs data-only inspection: no models, renderer,
reference integration, CUDA, training or checkpoint loading. It requires the
explicit completed file-only verifier result and the admitted generation SHA-256.
It independently checks original inputs/calibrations, saved rays and normal/mask
gates, completed counts, component sums and material/energy domains. The preceding
verifier additionally checks every output hash, camera/depth/PNG encoding,
independent material-label reconstruction and convergence. Run the pair with
**new** writable destinations; existing reports are never overwritten:

```bash
cd /home/isavand/paper/Nerofactor-3d2-e2a
REVIEW=$(mktemp -d /home/isavand/paper/evidence/bell-e11-v3-offline.XXXXXXXX)
python3 -B -m benchmarks.verify_bell_e11 \
  --root /home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark \
  --report-sha256 5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7 \
  --output "$REVIEW/verification.json"
PYTHONPATH=. MPLCONFIGDIR="$REVIEW/matplotlib" python3 -B \
  results/bell_e11_v3/review_saved_evidence.py \
  --batch /home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240 \
  --verification "$REVIEW/verification.json" --output "$REVIEW/analysis"
```

The figures display saved observations at native128×128, without exposure changes.
Relight differences are max absolute linear-channel differences; colorbar0–.35.
Material-label plots use fixed physical ranges(.04–1 slope widths,0–1 metallic).
NaN/background slots are excluded visually. Copied artifacts match the review
copies byte-for-byte after SVG whitespace/CSV line-ending normalization. Raw
Matplotlib outputs and the original CSV remain unchanged under `analysis/`;
normalized copies are separately stored under `analysis/git-figures-v1/`.
Figure metadata dates are omitted; Matplotlib versions can
affect regenerated SVG bytes without affecting underlying numerical evidence.
The CSV has one row per source-camera ID; empty relight columns mean no relight
was generated for that camera, not a zero error.

The review was executed on host Python3.9.12/NumPy1.21.5/Pillow9.0.1/Matplotlib3.5.1.
It does not repeat scientific generation or constitute native training admission.
