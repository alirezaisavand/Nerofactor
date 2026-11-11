# Material symmetry audit v1 — approved attempt2

Protocol identity `material-symmetry-audit-v1-attempt2`. Base
`13fe645af355f66400d435f9e25f5fcc63870912`, branch
`feat/anisotropy-symmetry-audit-v1`. The researcher explicitly approved the
reporting-only Boolean repair, serialization regression tests and **one fresh
versioned attempt** after the stopped3C-5 run. Freeze this addendum before that
attempt. No automatic third attempt, extra sweep, tuning or optimization.

## Unchanged scientific protocol

The complete scientific specification is inherited **without modification**
from [MATERIAL_SYMMETRY_AUDIT_V1.md](MATERIAL_SYMMETRY_AUDIT_V1.md), SHA-256
`8e510d566d443ed44f3e0428c429837ccd70b6bcfba0951efd5ce6bcf1d4572a`.
This addendum changes reporting and output identity only. In particular:

- Exactly truth and final LM101–105, P0/P1/P2; original float64 transforms and
  float32 shader inputs; unchanged Q/kernel/eigen/orientation gates and directions.
- A2/atan2,512/1024 diffuse/specular samples; original fitting tapes0–3 and
  evaluation tapes1000–1003; unchanged observation order, geometry, lighting,
  independently generated targets and mean-before-MSE calculation.
- Exactly36 completed shader calls/7,776 points on success,7,962,624 specular
  and3,981,312 diffuse directions;1,620 pointwise kernel and540 ideal PDF scalars.
- Same saved-mean admission, original target metrics, per-tape sampling/RGB/mask
  diagnostics and roundoff bounds. No selected preferred representation or outcome.
- One CPU thread, hidden GPUs, maximum1,800-second process wall time; peak-RSS
  bound4GiB and soft address-space cap8GiB or the stricter inherited limit.
- All mandatory physical checks precede shader evaluation. Stop and preserve
  evidence on any failed integrity, physical, shape/finite/mask/component/resource
  check. Never relax a threshold or repair numerical behavior after observations.
- No optimizer, parameter update, backward/gradient call, neural/GPU training,
  new dense integral/target, new seed/sample pattern/count, source/decoder change,
  prior experiment rerun or follow-up experiment.

All scientific function bodies are unchanged from the first attempt. The one
exception is wrapping the **unchanged** analytical orientation predicate in
native `bool(...)`, which preserves its truth value for JSON output. New fixtures
check nominal/rounded anisotropic/isotropic P0/P1/P2 serialization and compare
the old Boolean result with the repaired result. The original audit tests retain
their assertions; only the disposable output-path fixture follows attempt2.

## Reporting repair and preserved failure

The shared scientific/infrastructure helpers remain untouched. The audit-local
report writer constructs strict JSON (`allow_nan=False`, indent2) completely
before seeking or writing. An unencodable payload leaves the previous snapshot
byte-identical; new tests cover NumPy Boolean and NaN rejection as well as normal
truncation of a previously longer snapshot. This is prevalidation, not a claim of
atomic filesystem writes or crash durability.

The first attempt and its frozen code remain preserved in Git at the base above:

| Preserved artifact | SHA-256 |
| --- | --- |
| Raw `docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json` | `055aa0a2c13457dc198bb888c45ff15062fec90856de31220e09ada196568a55` |
| Valid `docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json` | `47e879568455b593117304a89707a4216620fe6ca0e0a8985e4c98686f579b46` |
| First-attempt implementation at base commit | `65a5b9e134df648ac6633336c3d59ef8a65fa9015d4deefd8c30b490f1fc68fc` |

Do not overwrite, normalize, delete or reinterpret the truncated raw output,
failure manifest, first protocol or stopped-attempt results. The new source
explicitly requires a distinct output path and records its own hash, this
protocol's hash and the prior failed attempt identity. Collision rejection remains
mandatory. All prior accepted scientific sources/protocols/evidence are read-only.

The initial attempt computed the beginning of the truth/P0 analytical checks but
performed zero shader calls. Repeating those checks is explicitly part of this
new approved attempt, not an unreported retry or a completed previous experiment.
All first-attempt results remain distinct from any newly obtained measurements.

## Files, verification and the one execution

Implementation: `experiments/material_symmetry_audit_v1.py` (reporting repair).
Focused tests: `tests/scientific/test_material_symmetry_audit_v1.py` and new
`tests/scientific/test_material_symmetry_reporting_attempt2.py`. Run only these
current focused tests and essential static/identity checks; no earlier suites.
Verify protected old bytes and frozen source/test/protocol hashes before launch.

New exclusive-create evidence:
`docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json`.
New post-run analysis:
`docs/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2_RESULTS.md`.
Original stopped-attempt files remain unchanged. Record every scientific
measurement/failure, time, memory, RNG/tape status and cleanup outcome. Report
H1–H4 separately; any causal attribution of LM basins remains constrained by the
forward-only design.

From `/home/isavand/paper/Nerofactor-3c5-symmetry-audit`, run **once only**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
python3 -B -m experiments.material_symmetry_audit_v1 \
  --output /home/isavand/paper/Nerofactor-3c5-symmetry-audit/docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json
```

Commit/push the reporting repair and new evidence to the same feature branch,
without merging. Stop after this attempt and report all outcomes honestly.
Any additional repair/rerun or next controlled experiment requires new approval.
