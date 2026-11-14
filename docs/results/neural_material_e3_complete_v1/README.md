# Complete E.3 panel from two linked executions

All ten arms have state-100 endpoints. **Zero passes full physical recovery.**
The original six completed endpoints are preserved exactly; four endpoints come
from the separately authorized recovery. The original execution remains failed.
See the [scientific report](../../NEURAL_MATERIAL_E3_COMPLETE_RESULTS.md).

Files:

- `endpoints.csv`, `comparison.json`: all ten state-10/state-100 comparisons,
  full physical/component errors, success checks and endpoint source run.
- `trajectories.csv`: 910 unique saved fitting states (10–100 for ten arms),
  with a source-run column. Three repeated state-10 measurements agree exactly
  and are de-duplicated only in these presentation rows.
- `gradient_routes.csv`: 4,550 actual head/state records. Exact sigmoid
  endpoints and physical roughness-floor contact are different observations.
- `adam_displacements.csv`: 900 actual float32 update summaries/probes across
  the two runs. No claim of complete per-step weight/XOR reconstruction.
- `integrity.json`: checkpoint bridge/parity checks, eight new full numeric
  snapshots, twelve new signal archives, re-evaluated saved physical/radiance
  metrics and unchanged original success predicates.
- `evidence_manifest.json`: every new raw file and external process/launch/
  authorization/log sidecar with absolute path, byte size and SHA-256; links
  the immutable original inventory and report.
- `package_manifest.json`: exact byte/hash agreement between original analysis
  outputs and tracked copies, source identity and offline admission-failure
  record. Four figures have PNG and vector PDF copies.

Raw scientific evidence stays outside Git:

| Evidence | Location | Status |
| --- | --- | --- |
| Original E.3 | `/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae` | failed; six complete arms, one partial, three parity-only |
| Recovery E.3R | `/home/isavand/paper/evidence/neural-e3r-39baf93f9caa4d89b3fc2ddc549eb205` | successful; exactly 349 new Adam returns |
| Combined offline analysis | `/home/isavand/paper/evidence/neural-e3r-39baf93f9caa4d89b3fc2ddc549eb205-combined-analysis-v2` | complete saved-data analysis; zero model queries |

The original raw inventory is 125 files / 388,436,575 bytes; recovery is
55 files / 161,035,990 bytes. External sidecars are listed separately. No raw
neural weights, Adam archives or generated caches are tracked here.

Regenerate the compact outputs with a fresh external directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 \
  docs/results/neural_material_e3_complete_v1/analyze_complete_e3.py \
  --evidence-parent /home/isavand/paper/evidence \
  --output-dir /home/isavand/paper/evidence/neural-e3-complete-offline-UNIQUE
```

The analyzer blocks Torch and all scientific experiment imports. It performs
only file/hash checks, numerical-array reads (`allow_pickle=False`), saved-data
metric reductions and plotting. It privately compiles the unchanged analytical
radiance/success functions and the approved representation-aware diagnostic
adapter; no model/shader/optimizer is imported or queried. Original references
are reused, never regenerated. The first offline attempt hit the generic
small-report size limit on the accepted 19,679,861-byte reference JSON. Its
failure is preserved in `-combined-analysis-v1`. The completed analyzer admits
that one legacy reference by its exact known size and pinned SHA-256, without
changing the generic reader or any scientific input. This was not another
scientific execution.

Combined work is 900 E.3 Adam returns, 924 forwards and 914 backwards, plus 14
constructor-calibration passes. Four extra fitting forward/backwards relative
to the original single-launch plan are explicitly counted restoration repeats.
The original and recovery report/seal/exit records retain their separate status.
