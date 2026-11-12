# D.2 reporting integration and equivalence validation

**READY FOR FREEZE REVIEW**, with watchdog, memory and capacity caveats below.
This is an integration review decision, not recovery execution admission. The
public v2 API always rejects execution. No final freeze or authorization exists.
No material-shader forwards, recovery updates or historical optimizer replays
were run. Ten newly authorized field-only engineering Adam updates were run.

Branch: `feat/neural-material-reporting-integration-v2`, based on
`40b7b301d02835d1b324ed54db6a6c2649f0e049`. All699 baseline tracked files,
322 frozen source hashes, original freeze, original Phase D report and both
D.1 benchmark directories remain unchanged. Identities, test and measured
results are in [the versioned evidence](evidence/NEURAL_MATERIAL_REPORTING_INTEGRATION_V2.json).
Large journals, own numeric checkpoint, source epochs and lifecycle fixtures:
`/home/isavand/paper/evidence/3d2-reporting-integration-v2-aed448730b1c`.

## Integration and scientific equivalence

`experiments/neural_material_recovery_logged_v2.py` derives a private execution
graph from the SHA-admitted v1 source. Only report/row dictionary allocations
are wrapped. Removing those two wrappers gives exactly the original execute
AST. Training uses the original **identical code object**, with unchanged
scientific globals. The four replaced original globals are the report output,
closed review gate, private train-function binding and wrapped execute function.
There is no global monkeypatch. Construction, calibration, shader/renderer
arithmetic, four-tape mean, MSE, backward/gradient checks, continuous Adam,
update ordering, prefix/control checks, evaluation, failure handlers, clocks,
settings, checkpoints and cleanup retain their original implementations.

The production ceiling remains5000, snapshots250/1000/5000,900 active seconds,
28800 whole seconds,4GiB RSS and8GiB AS. No seeds, physical criteria, targets,
tapes, model/calibration or optimizer settings changed. V2 source admission
checks the original runner and all original frozen hashes. Source equivalence
checks every original module global, not just selected numerical helpers.

`infrastructure/scientific_reporting_adapter_v2.py` instruments nested report
dictionaries/lists, preserving row aliases, scalar types and absent-versus-empty
fields. Every mutation is queued in order; intermediate counter/digest changes
are **not coalesced**. Only new values are copied, never previous histories at
persistence. Deep-copying a work counter copies that subtree, not its owner.
Unsupported deletion/reordering operations fail closed.

| Original mutation | V2 representation |
| --- | --- |
| Run, sources, configuration, tapes, initialization | Initialization event followed by associated/global metadata deltas |
| Construct/admit row; admission result | Associated outcome append, original status transitions and gate payload |
| Every work-counter/update/digest change | Ordered typed-path metadata mutations |
| Logged loss, physical state, gradients, times, counters | Exactly one ordered history event, retaining the full original row |
| Snapshot and later objective/material annotations | Associated checkpoint event then nested metadata changes |
| Fit/external evaluation and scientific outcome | Original nested evaluation/metadata payloads and status |
| Failures, censoring, partial checkpoint references | Explicit failure/censor classification plus original diagnostic metadata |
| Unstarted rows | Original admitted/constructed status and zero counters, never promoted |
| Each original persist call | Sequential v2 boundary with mutation count and preceding journal sequence |
| Close | One final report materialization and separately selected final-manifest hash |

D.1 remains unchanged. Its SHA-pinned writer/verifier definitions are compiled
in a private v2 namespace, with v2 envelopes, boundary validation and one narrow
schema extension: v1 creates an empty history field after an admitted row was
persisted. V2 permits that initialization, never erasure of existing history.
Ordinary mutations still use the original transactional validator. Batches have
at most128 changes, records at most8MiB, summaries at most4096bytes. The measured
maximum pending queue was152 changes. Summaries inspect at most27 statuses.
The full structured report remains in memory for original control checks;
a separate journal reconstruction state duplicates it, with bounded matrix size.

No whole-report serialization, history traversal, journal reconstruction or
full-chain rehash occurs per progress persistence. Close compares live and
journal states, finalizes/hashes once, writes `report.json` once and records
`reporting-result.json`. Independent full scan happens after execution, not in
the loop. Its cost must still be included when scheduling external commands.

## Synthetic lifecycle and integrity evidence

New focused tests run the **actual v1 and wrapped v2 orchestration** with explicit
inert scientific stand-ins. Their private test loop scales250/1000/5000 to1/2/3;
it does not run a scientific trajectory or validate real shader mathematics.
Real shader construction, Adam construction and autograd backward are forbidden
by mocks. Production bytecode, milestones and helper identities are audited
separately. Every original report field and every persistence-boundary digest
matches exactly, including synthetic optimizer histories, work, snapshots and
D0/N0 rolling transition hashes.

Covered: complete27 scheduling; D0 prefix mismatch; N0 trajectory mismatch and
censor before N1; N1 numerical/censor failures with remaining N1 scheduling;
whole watchdog; initialization/before-step/checkpoint/evaluation failures;
unstarted rows; partial writes, poisoning, truncation, missing/reordered/duplicate
records and boundary IDs; altered identities/final hashes; failed finalization;
missing/corrupt checkpoint metadata; capacity, path, symlink and allocation
rejection; unchanged RNG and closed old/missing/unauthorized v2 execution.
Missing metadata/archive hashes are checked by the read-only final verifier.
The large reporting-only benchmarks use explicitly inert snapshot references;
only those benchmarks disable checkpoint-artifact checks, and say so in their
measurements. Full lifecycle fixtures use actual synthetic checkpoint metadata.

The initial19-test run passed. Expanding the suite exposed one test using a
`status` event for a `failed` classification; the validator correctly rejected
it. The test was corrected to the required `failure` kind. That failed22-test
run and its artifacts are preserved separately from the final test results.
The final expanded suite passed **23 tests, zero failures/errors/skips**.
No old test suite or successful scientific experiment was repeated.

## Measured reporting performance

Each case was run once with the actual v2 adapter and rich original-schema
synthetic payloads. Complete:27x5000 logical updates. Censored:15x5000 plus one
4860-update N0, eleven unstarted. These are dictionary mutations, not optimizer
updates. Both cases reconstructed exactly.

| Measurement | Completed27 | Censored |
| --- | ---: | ---: |
| Whole engineering case, seconds |193.435470|114.920111|
| Mean reporting/bookkeeping per synthetic history, seconds |3.832487|3.841632|
| Max per-history reporting/bookkeeping, seconds |4.097056|4.013637|
| Append/validate/encode/write, total seconds |85.790304|51.087165|
| Bounded summary updates, total seconds |2.766284|1.655696|
| All persist calls, seconds |89.105076|53.120741|
| Persist calls |13556|8020|
| Finalization/materialization, seconds |9.001826|4.572368|
| Independent verification/reconstruction, seconds |72.493098|42.258108|
| Extra independent journal hash scan, seconds |0.127372|0.076103|
| Peak process RSS, bytes |1059221504|732811264|
| Logical output storage, bytes |316202144|188734305|

Reporting/bookkeeping includes mutation capture and per-step synthetic counters;
persist totals are a subset. Case wall time also includes setup, original-save
samples, finalization and scan. Storage includes R0 sample reports and measurement
artifacts; it is neither journal-only bytes nor physical disk traffic.

R0 uses original full-report persistence on identical report contents at
prespecified history indices0/13/26, costing0.077008/0.937602/2.156108seconds
per save. Censor indices0/13/15 cost0.076898/0.928793/1.051916seconds.
Those are actual samples. Multiplying by501 gives a **projection**, not a
measured R0 full-run trajectory. Exact full-path R0/v2 equivalence is separately
established by the smaller lifecycle fixtures. Do not call these ratios shader
or recovery speedups. Per-save v2 costs stay small as previous histories grow.

## Single target-independent N1 fixture

The archived pre-execution plan declared exactly10 field-only updates, one
attempt,120seconds and2GiB RSS. Initial model identity exactly matched frozen
M1/101/N1:84 tensors,1485584 parameters, original seed101/weight10101, calibration,
float32 CPU and original Adam options. All ten attempts returned, no additional
field queries or retries. Wall time1.873792seconds, peak RSS537333760bytes.
Torch/NumPy global RNGs were unchanged. No original target/tape was consumed.

| Operation | Mean seconds | Maximum seconds |
| --- | ---: | ---: |
| Field forward |.008074|.010179|
| Field-only backward |.003561|.008833|
| Original Adam return |.006707|.012200|
| Gradient observer |.008155|.008470|

The fixed engineering objective fell from7.05864477 to.34083754. All gradients
were connected and finite. **All groups became zero-gradient from step3**;
all84 parameter tensors nevertheless changed over the attempt. This saturation
is retained and limits throughput interpretation. It is not parameter recovery.
The original own-state checkpoint took.088751seconds to serialize and.118282 to
hash/read/verify without pickle. No restoration timing was collected. An
intervening unexecuted draft sentence proposing restoration is preserved in the
source epoch; the earlier archived pre-execution plan excluded restoration,
matching the executed fixture. No repeat was used to fill this measurement.
The benchmark source epoch and delivery source differ only in the later
synthetic snapshot-template correction; exact source hashes are recorded.
Subsequent integration changes strengthened source and admission reporting,
without changing append, loop or finalization algorithms used in timing.

## Feasibility and review boundaries

The D.1 observational control residual382.18seconds plus the largest new
reporting/bookkeeping cost4.10 suggests roughly386.28seconds/history. It is a
conditional estimate, not a new measured control history. Historical C/C.1 real
N1 maximum forward/backward timings are.057176/.042688seconds. Replacing the old
synthetic Adam estimate with the new measured maximum and adding the maximum
observer cost conservatively yields about603seconds for5000updates before
small endpoint costs. The historical backward timing already includes its
gradient observer, so adding another observer explicitly double-counts that
component; this is a conservative sensitivity calculation, not a confidence
bound. With reporting and three observed checkpoint writes the estimate is
about607seconds; a20% compute sensitivity remains below900seconds. No budgets
or scientific tolerances were changed.

Combining18 projected controls and9 projected N1 histories suggests roughly
12417seconds plus initial admissions, source checks and final verification,
below28800seconds under those assumptions. Actual long-run N1 shader allocation,
nonfinite-state checks, saturation, OS noise and sustained throughput remain
unmeasured. The new ten-step field-only test does not establish them.

Peak benchmark RSS is about.99GiB and the field fixture.50GiB, separately measured.
They do not establish simultaneous27-model/shader/report memory use. Preserve
4GiB RSS/8GiB AS gates. The current1GiB free-space floor rejects clearly
insufficient storage but is not full-run capacity sizing: derive a conservative
journal/report/checkpoint/failure budget during freeze review.

Censor finalization4.57seconds leaves measured room within10seconds. The largest
complete finalization9.00seconds leaves only about1second before considering
failure checkpoint/cleanup and timing variability. **This thin watchdog margin
must be reviewed explicitly before another execution is authorized.** Independent
verification takes longer and is not hidden in active loop costs. Do not quietly
extend budgets, drop records, or defer required failure evidence to claim fit.
READY FOR FREEZE REVIEW means the integration/evidence can be reviewed; it does
not establish that an end-to-end neural experiment is ready or will succeed.

Next milestone: review source/mutation equivalence, capacity, memory and
failure-time handling; decide whether a separately versioned freeze needs an
engineering revision. Then, only with a new final freeze and explicit run-specific
researcher authorization, consider a future recovery execution. No D.3 is started.

## Inspection and executed commands

Read-only source audit:

```bash
python3 -B -m experiments.neural_material_recovery_logged_v2 --describe
```

New synthetic tests (no historical suites):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m unittest tests.scientific.test_neural_material_reporting_integration_v2 -v
```

The already executed engineering commands used this prefix and the explicit
fresh destinations recorded in the evidence manifest:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 120s \
python3 -B -m experiments.benchmark_neural_material_integration_v2 \
  --case field --output /home/isavand/paper/evidence/3d2-reporting-integration-v2-aed448730b1c/n1-field-only
```

Reporting cases used the same environment, a620-second external backstop and
600-second internal engineering cap, `--case complete`/`--case censored`, with
corresponding fresh `complete`/`censored` destinations under that parent.
Existing paths must fail; these commands are records, not approval to rerun
scientific or field-only fixtures. No CLI enables scientific recovery.
