# E.2a correction readiness, v2

Based on researcher-reviewed `e0b29b89b97e4c4b226a379ad66e3d634d6040a8`, branch
`feat/neural-material-early-step-e2a`. The next action is independent researcher
review of the correction commit. No scientific execution is authorized.

## Changes

| File | Purpose |
| --- | --- |
| `experiments/neural_material_early_step_e2a.py` | Transactional call admission, explicit cleanup/completion lifecycle, terminal integrity and bounded report seal |
| `experiments/freeze_neural_material_early_step_e2a.py` | Select corrected v2 authority, verify full process/JUnit evidence, bind historical package and enforce unchanged scientific/resource contracts |
| `tests/scientific/test_neural_material_early_step_e2a.py` | Existing import-blocked fixtures plus cleanup/finalization, rejected calls, terminal identity/counter and evidence regressions |
| [Protocol amendment](protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.md) | Corrected lifecycle, output interpretation and unchanged invariants |
| [Validation v2](evidence/NEURAL_MATERIAL_E2A_VALIDATION_V2.json) | Actual test exit, complete XML/stdout/stderr, hashes, inventory and source identities |
| [Freeze v2](protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json) | Immutable corrected preparation contract; execution false |
| This readiness document | Concise correction results and remaining gates |

`Work.attempt` now validates every applicable cap before modifying counters.
Rejected forwards record zero nonexistent attempts. Completion independently
checks ten ordered arms and their exact counters. Cleanup is acknowledged only
after normal `limits()` exit. The terminal gate re-admits selected frozen
sources/reference/evidence files, verifies immutable tapes and rechecks counters
without executing the field/shader. Scientific completion remains false if any
required verification fails or is not reached.

The persisted report remains explicitly unfinalized. A separate bounded seal
selects its exact hash after durable writing; only after seal write/readback does
the returned result acknowledge finalization. Consumers additionally require
successful external process exit. Cleanup failure, interrupted work, integrity
failure and report/seal failure preserve distinct failed/censored/incomplete
metadata. Finalized failure evidence never means successful scientific work.

## Focused validation

**55 passed; 0 failed; 0 errors; 0 skipped. Actual subprocess exit: 0.**
Pytest reported 6.75 seconds. The versioned validation embeds the complete JUnit
XML, stdout/stderr, exact command and named inventory. Engineering-only raw
process evidence also remains at `/tmp/neural-e2a-v2-tests-9jf53os7`; the versioned
embedded records do not depend on that temporary path remaining available.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -p no:cacheprovider -q tests/scientific/test_neural_material_early_step_e2a.py --junitxml=/tmp/neural-e2a-v2-tests-9jf53os7/junit.xml
```

JUnit SHA-256:
`25b77bf473c08a8eed41872ae6c14058bac9b5f70f91afc9defcf5e14b58380a`.
Validation JSON SHA-256:
`45ebcfc39b1592e90655cf833769400f56ba3d0f2aba64a6086349103f16347a`.

The freezer independently validates exit 0, XML hash, case inventory/counts and
process-output hashes; flags alone are insufficient. New regressions include
successful/failed cleanup, interruption, source/reference/tape mismatch, exact
matrix/counters, report/seal failure and rejected-phase counter immutability.

## Preservation and scope

Verified all 348 old frozen files, the expanded 357-file inherited identity
inventory, accepted reference/report identities and all 131 original plus 45
continuation artifact hashes. No checkpoint or pickle deserialization. The
original scientific G/A, N1, A2/atan2 and recovery sources remain unchanged.
AST comparisons additionally establish that the existing arm driver, scientific
`Core` and five-case training loop are unchanged by the reporting refactor.

The new freeze enforces equality with v1 for five cases, both LRs, ten returns
per arm, 100 total returns, 120 forwards, 110 backwards, initial states, physical
measurements/criteria, targets/tapes, optimizer flags and all work/resource
settings. Existing 4 MiB root metadata allowance includes the bounded review,
report and new seal; no resource ceiling increases.

The v1 freeze, validation, readiness and original protocol remain byte-for-byte
identical to the reviewed commit. Their hashes are preserved in the corrected
package. They are historical preparation evidence, not execution authority for
the modified sources. No historical report was rewritten as successful.

**Zero scientific model calls, shader calls, optimizer updates or reference
generation occurred.** The test session blocks scientific-module imports;
subprocess import-blocker fixtures also pass. Only unrelated toy networks/Adam
and synthetic metadata callbacks execute. Metadata admission/freezing completes
without importing Torch. No E.2b execution review was created; no CUDA work ran.

## Corrected freeze and remaining gates

New freeze SHA-256:
`f9d8f28a46641c8f3aa777118b7f106f120918e7c60e6605387fcb7c46d984ad`.

```bash
sha256sum docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json
python3 -m experiments.freeze_neural_material_early_step_e2a --verify-sha256 f9d8f28a46641c8f3aa777118b7f106f120918e7c60e6605387fcb7c46d984ad
python3 -m experiments.neural_material_early_step_e2a --describe
```

Execution remains false. E.2b requires independent acceptance of this correction
commit and a separate affirmative review binding the v2 hash, exact clean branch/
commit, fresh output root and external watchdog. Real loss/gradient parity,
native passive-hook equality, instrumentation/runtime feasibility and paired LR
outcomes remain unverified. No real LR experiment or longer recovery study has
been run, and no optimization improvement is claimed.
