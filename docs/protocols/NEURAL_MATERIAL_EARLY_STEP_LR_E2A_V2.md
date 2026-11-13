# E.2a v2: engineering review corrections

Review baseline: `e0b29b89b97e4c4b226a379ad66e3d634d6040a8` on
`feat/neural-material-early-step-e2a`. This is preparation only. No scientific
execution or affirmative E.2b review is authorized.

This amendment retains the [v1 scientific protocol](NEURAL_MATERIAL_EARLY_STEP_LR_E2A.md).
The five cases, two learning rates, initialization, architecture, A2/atan2,
sample counts/tapes, targets, optimizer flags, measurements, physical criteria,
work totals and resource ceilings are unchanged. Scientific-loop/arm-driver AST
comparisons and frozen-field equality checks protect these invariants.

## Historical package and replacement authority

Preserve the v1 [freeze](NEURAL_MATERIAL_EARLY_STEP_LR_E2A.freeze.json) with SHA-256
`69af5785e37a0843e9a4d854ffd441ff390f5a51b98fc78044621f0f83183ed6` and
[validation](../evidence/NEURAL_MATERIAL_E2A_VALIDATION.json) with SHA-256
`9303e05691b6febc679f819ffc2f0ee8033830c76cf8561ec775944ce4e59eaf`.
Do not rewrite those files or reinterpret their missing process-exit evidence.
The historical readiness document and original protocol also remain unchanged.

The corrected preparation authority is exclusively
`NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json`, protocol
`neural-material-early-step-lr-e2a-v2`, with separately selected
`NEURAL_MATERIAL_E2A_VALIDATION_V2.json`. The future execution gate reads only
this v2 path, verifies its selected hash, sources, tests and protocol amendments,
and rejects the v1 hash. The new freeze also binds the historical package hashes.
Execution remains false until a separately reviewed E.2b authorization binds
the new freeze, exact clean commit/branch, fresh output root/run ID and watchdog.

## Atomic work admission

`Work.attempt` validates the overall and phase-specific caps before modifying
any counter. Invalid phases, exhausted training/terminal/external allowances or
exhausted total allowances reject without recording a nonexistent call.
Attempts are incremented immediately before admitted calls; completion counts
remain updated only after returns. There remain exactly ten ordered paired arms,
100 Adam returns, 120 forwards and 110 backwards; no instrumentation query added.

## Completion lifecycle

The reviewed source already placed its final cleanup marker after the `with`
statement. V2 makes the distinction testable and explicit rather than relying on
that one marker. `finish_execution` owns the lifecycle while the unchanged
scientific loop is a closure:

1. `work_counters_verified` becomes true only after all ten arms, per-arm native
   counters, measurement counters, ten returns/arm and hook-removal flags pass.
2. `cleanup_verified` and `runtime_cleanup_status:passed` are set only after
   normal context-manager exit. A cleanup exception yields `status:failed`,
   `failure_stage:runtime_cleanup`, cleanup false, scientific completion false
   and unperformed terminal checks explicitly `not_verified`.
3. Terminal integrity re-admits the hash-selected corrected freeze. This
   rechecks its selected scientific/infrastructure sources, all accepted
   reference/report identities and historical artifact inventories. Then verify
   the original in-memory tape manifest against the frozen manifest and use the
   original tape-array hashing routine against that same frozen manifest.
   Recheck every arm and aggregate work counts. These are file/hash/array checks;
   no field, shader, rerender, optimizer, target generation or loss call occurs.
4. Only after cleanup and all three terminal categories pass may
   `scientific_work_completed` become true. Before finalization, the overall
   status is `work_and_cleanup_verified`, not `complete`.
5. Evidence finalization is separately acknowledged after durable report write
   and bounded seal write/readback. Only then does the returned outcome become
   `complete`. A failed scientific run may have finalized **failure evidence**;
   that does not make its scientific outcome successful.

If work is interrupted or fails, cleanup on exceptional context exit is not
claimed verified merely because a `finally` block ran. Report it conservatively
as `not_verified`; retain the original failure/stage, partial arms/records and
unperformed checks. A failed terminal check records its reason and leaves
subsequent checks unverified. Cleanup may be verified while terminal integrity
fails; work counters may be verified while cleanup fails. These states are
deliberately distinct. No automatic repair, resume, retry or budget extension.

## Report finalization and bounds

`result.json` is an exclusive immutable snapshot written and fsynced before any
finalization acknowledgement. It deliberately contains `evidence_finalized:false`
and does not claim an overall `complete` status. After its write returns,
`finalization/seal.json` selects the exact report path/hash/size and records the
complete/failed/censored outcome. Verify that seal by selected hash/readback.
The returned in-memory result acknowledges report/seal identities and evidence
finalization only after these operations return normally.

Consumers must require **all** of: successful external process exit, valid seal,
matching report hash, successful cleanup, all terminal checks passed and complete
work counters. A report without a seal, an unacknowledged/partial seal, or a
nonzero/missing process exit is incomplete evidence. In particular, even a
parseable seal left by a failed fsync or a late process failure cannot override
the missing/nonzero process status. Reporting exceptions propagate and the
in-memory outcome is failed with finalization false and a separate failure
reason. Previously written records are preserved without overwrite.

The seal is at most 4096 bytes; the review is at most 16384 bytes; the report
remains at most 2097152 bytes. Together these fit inside the existing 4 MiB root
metadata reserve. The nested seal does not consume a third root-level JSON slot.
No archive or total-output allowance increases: 241 MiB/arm, 2414 MiB total,
minimum free 5062524928 bytes. CPU/one thread/hidden GPUs, 4 GiB RSS/8 GiB AS
scientific runtime backstops, 120 active seconds/arm, 1800 whole-process external
watchdog and ten-second grace remain unchanged. Terminal metadata/reporting
after normal runtime restoration remain covered by that external watchdog;
they do not start a new scientific runtime or extend its budget.

## Correction validation

Run only focused engineering fixtures under the session scientific-import
blocker, using `python3`. Capture the actual subprocess return code, full JUnit
XML and SHA-256, stdout/stderr and hashes, exact named inventory and all counts
in the new validation JSON. The freezer independently checks these fields,
requires exit 0 and zero failures/errors/skips, and binds the corrected source
inventory. Preserve v1 validation verbatim.

New regressions cover rejected-call counter immutability; normal/failed cleanup;
interruption; corrupted terminal source/reference/tape identities; counter and
matrix mismatch; report/seal failure; and rejection of nonzero exits/corrupt
test evidence. Existing passive-hook, actual-displacement, cleanup, parity and
authorization fixtures remain. Synthetic metadata arms are not real experiment
results. The real N1, original shader, scientific optimizer and references must
not be constructed/executed during this correction.

After preparation, independently select and verify the **new** freeze SHA:

```bash
python3 -m experiments.freeze_neural_material_early_step_e2a --verify-sha256 NEW_V2_SHA256
python3 -m experiments.neural_material_early_step_e2a --describe
```

Stop for independent researcher review of the correction commit. Native parity,
hook overhead, runtime feasibility and LR outcomes remain pending E.2b. This
engineering correction supplies no evidence of improved material recovery.
