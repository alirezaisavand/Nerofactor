# Gate 10.3 — Offline replay acceptance v3

This is an offline comparator correction and results finalization, covered by the
approved v2 study's file-only verification scope. It authorizes no new checkpoint
load, SDF query, extraction, GPU operation, training or scientific threshold change.
Keep v1/v2 freezes, launchers, authorizations, native arrays, receipts and the
original failed verifier unchanged. Write separate versioned offline evidence.

The native v2 receipt is complete, the host exit is zero and container absence
and native cleanup pass. The researcher reported the original offline verifier
stopping at `Fitting physical/eligibility replay differs`. File-only diagnosis of
all 488 fitting views finds exactly two differences:

| Step | Camera | Summary field | Captured | Host replay | Absolute difference |
| --- | --- | --- | --- | --- | --- |
| 30k | 61 | Normal-angle p95 | 20.87159936365109 | 20.871599363651086 | 3.552713678800501e-15 degrees |
| 50k | 98 | Normal-angle mean | 5.776839143688098 | 5.776839143688097 | 8.881784197001252e-16 degrees |

Each is one adjacent IEEE float64 value. The final summaries include inverse
trigonometric evaluation and reduction/quantile arithmetic in different NumPy/
CPU runtimes. These discrepancies are consistent with floating-point roundoff;
the exact contributing low-level instruction is not identified. No eligibility
mask, count, classification or scientific pass/fail decision differs. The original
six-camera and SDF witness replays are exact at all four checkpoints.

The new explicit rule is `normal-angle-summary-float64-adjacent-v1`:

- Require exact types, keys, None/empty cases, identities, counts, classifications,
  scientific decisions and every field outside the normal-angle summaries.
- Require finite values, including when both sides otherwise compare equal.
- For only `signed_normal_error_degrees.{median,p95,mean,maximum}`, accept exact
  equality or one adjacent finite float64 value, tested with `nextafter`.
- Reject two ULPs, larger differences, missing/extra keys and any eligibility-mask
  or dtype difference. Never substitute a reference value for a captured value.
- Log every accepted roundoff discrepancy. Preserve all native decision thresholds
  and original captured statistics. This is a replay rule, not a scientific error
  tolerance or retroactive modification of the native experiment.

The focused tests exercise the actual paired-surface metric path, both recorded
one-ULP examples, two-ULP and substantive differences, strict other fields, changed
counts/decisions/types, missing/extra values and nonfinite summaries.

Execute only the file-only verifier, followed by saved-array analysis:

```bash
CUDA_VISIBLE_DEVICES='' python3 -B experiments/bell_geometry_gate103_offline_v3.py \
  --output /home/isavand/paper/evidence/bell-geometry-gate103-v2-ad4a430c57b149f58a6b936ede1f017d-offline-v3
CUDA_VISIBLE_DEVICES='' python3 -B experiments/analyze_bell_geometry_gate103_v3.py
```

These commands use write-once outputs and are recorded for reproducibility, not
an instruction to repeat completed analysis. The verifier first authenticates
the original source/freeze/approval/input and completion receipts; replays every
existing extraction map, topology, raster, SDF, fitting and neural witness; and
checks work, source, state-preservation and cleanup evidence. No original model
or renderer is constructed. Plots are computed solely from saved arrays, with
fixed camera IDs 14/58/105, common color scales and explicit undefined support.
Raw negative-grid statistics are replayed from existing arrays without model calls.

The completed study is a diagnostic. Geometry admission remains blocked by the
fixed 300k silhouette criterion. Earlier checkpoints never replace the publication
endpoint. Future objective interventions require separate preparation/approval.
