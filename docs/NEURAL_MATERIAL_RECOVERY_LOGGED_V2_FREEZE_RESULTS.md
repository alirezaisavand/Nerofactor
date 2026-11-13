# D.3 logged-v2 final freeze and execution readiness

**READY FOR SEPARATE EXECUTION AUTHORIZATION.** No scientific execution or
positive research authorization was issued. No real shader, recovery forward/
backward or recovery Adam update was run. No Phase D output was allocated.
The source freeze is a separate artifact, not execution approval.

Branch: `feat/neural-material-logged-v2-freeze`.
Base: `7580126b8d30fd7ad6621dc673e473d49aebfe2d`.

- [Final protocol](protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.md)
- [Final freeze](protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.freeze.json)
- [Readiness measurements and gates](evidence/NEURAL_MATERIAL_RECOVERY_LOGGED_V2_READINESS.json)

Freeze SHA256: `dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af`.
Readiness SHA256: `e579bd36a1cbf450604d2ce6db1e609cf95541b3f1c6861663aba9f2e7d29f85`.
The freeze binds348 sorted explicit source/test/protocol/evidence identities,
all27 original initial model identities and the unchanged scientific contract.
Readiness is a separately selected DAG dependency; neither it nor the freeze
contains a self hash. This results document is deliberately excluded.

The first candidate (SHA256`48ba946ba8422326949a4d5c57176dcb9ecbecfd260834662f6465f88cb9314d`)
was invalidated before delivery or execution after a durability review found
that first-commit identity/directory entries also needed fsync. It, its readiness
record/results and source epoch are preserved under `superseded-candidate/`.
The revised candidate above was regenerated after the fix, stricter complete
Adam catalog checks and the22-test pass. No original freeze was overwritten.

## Scientific equivalence and evidence preservation

All706 baseline files and1508 prior external artifacts were verified unchanged,
including original C.2 freeze
`730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`,
original incomplete D report
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`,
and D.1/D.2 code and measurements. No old worktree/source/result was edited.
Both G79252178 and Ae1e85977 remain separately pinned. No historical producing
revision or paper-quality reconstruction claim is added.

New execution derives from the unchanged D.2 graph. Original training bytecode
and every scientific binding remain identical. Only report/row wrapping and
output/review bindings differ. Model construction/calibration, A2/atan2, loss,
backward, Adam options/order, sampling/tapes, thresholds, clocks and source
failure policies are unchanged. Exactly27 fresh histories,5000 updates and
250/1000/5000 snapshots remain mandatory. No historical state warm start.

The original C/C.1 saved-array verifier was rerun read-only, without shader,
model or optimizer construction. Its direct aggregate equality rejected one
metadata path: `completion_versioned/path` named the historical C.2 worktree.
Recomputation in this new worktree changed that path only. The old failed
comparison is preserved. The new adapter hashes BOTH repository-local completion
files to the same admitted SHA256, permits exactly that path relocation, and
requires every other aggregate field to match. All original saved numerical,
replay, gauge, state, count and feasibility checks remain unchanged. A regression
test rejects a numerical-count change. The old aggregate/helper remains intact.
Fresh freeze preparation and verification use that explicit adapter.

## Watchdog concern resolved by preservation, not longer grace

`RuntimeOutput` appends the same ordered events at original persistence points,
then fsyncs journal bytes (including identity/directory creation on first commit) and atomically fsyncs a bounded commit anchor. It
records the actual future approved-review hash and frozen source/config/run
identity. Checkpoint/archive writes are fsynced before references become
committed evidence. The source-derived journal/derived-artifact caps fail closed
through infrastructure/storage exceptions; no data is truncated to fit.

Shutdown checks the last committed boundary and writes at most4096-byte
`execution-end.json`. It never serializes the report, computes a full report
hash or scans the chain. Previous9-second whole-report finalization has been
removed from the execution path. The targeted32-progress-sample cases measured:

| Synthetic case | Mean progress persist | Bounded shutdown |
| --- | ---: | ---: |
| Completed | 3.088 ms | .605 ms |
| Censored | 3.093 ms | .621 ms |

These are targeted engineering measurements, not recovery speedups or a claim
that arbitrary I/O will finish within10seconds. Old D.2 expensive benchmarks
were not rerun. The initial targeted audit and two source-change remeasurements
are all preserved; no favorable run was selected or scientific setting tuned.

Real SIGTERM/SIGKILL injection covered synthetic optimization, journal writing,
terminal sealing and offline materialization. Disk-full, partial writes,
corruption, duplicates, ordering, missing seals and interrupted outputs were
also checked. A kill can leave a valid committed prefix and no seal; it cannot
be silently presented as scientific completion. No design assumes `finally`
runs on SIGKILL. Fsync semantics depend on the filesystem/storage honoring them;
media corruption is detected, not repaired. The external28800/10 watchdog and
original900/10 active/reserve limits are unchanged.

A separate read-only finalizer verifies explicit freeze/anchor/journal/seal
hashes, source/config identity, all events/boundaries, referenced checkpoint
metadata/archives, strict architecture catalogs and scientific completion gates.
Report/final manifest are atomically written/fsynced to a separate fresh output.
No weights are loaded into a model or optimizer. Complete27/5000/work/initial/
control/endpoint/runtime evidence and the exact approved-review record are
required. A merely valid journal prefix cannot become complete, even when its
status says complete. Failed/censored statuses remain unchanged. Prefix-only
diagnosis is explicit, records any trailing bytes and never repairs/resumes.

## Conservative storage and combined memory

Measured checkpoint archive+manifest pairs: D0/N0 each11501bytes; N1 at least
18031522bytes. Full N1 weights, two moments, buffers and RNG data are counted,
not merely seven material scalars. New provenance inventory allowance is added.
Three scheduled states plus a failure-only state per history,28 overbounded
triple-audit diagnostic archives, journal, report+temporary, controls and an
equal-size safety reserve produce these frozen requirements:

| Storage component | Bytes |
| --- | ---: |
| Scheduled states | 978923016 |
| Failure-only states | 326307672 |
| Diagnostic archive budget | 8469595680 |
| Enforced journal cap | 1028540948 |
| Report plus atomic temporary | 380035256 |
| Control metadata | 524288 |
| Subtotal | 11183926860 |
| Equal safety reserve | 11183926860 |
| Required free capacity | 22367853720 |

This is22.37GB decimal (about20.83GiB), derived from measurements, shapes,
source cardinalities and explicit enforced caps. It supersedes the preliminary
1GiB admission floor. Current approved-parent capacity passes; no files or
permissions were changed to make space. No future run was allocated.

Memory accounting is 2911871696bytes, below4294967296bytes, combining observed
old D and full D.2 reporting peaks, all27 parameter/gradient/two-moment/buffer
bytes, checkpoint temporaries and the observed N1 field footprint. This
intentionally double-counts common bases. The bounded touched-array/two-report
fixture measured 1041563648bytes peak RSS under a2GiB engineering cap and8GiB
AS backstop. It made zero real model/shader/backward/Adam calls. Full real
shader/autograd/allocator concurrency and peak virtual size remain unmeasured;
this is a conservative planning estimate, not a real-run guarantee. Original
4GiB RSS/8GiB AS execution guards remain mandatory and cannot be enlarged.

## Runtime feasibility and uncertainty

Observed inputs are historical D0/N0 active times, D.1 full-report costs,
D.2 integrated reporting/field timings, C/C.1 real-shader initial timings and
D.3 targeted durable-commit costs. No new sustained neural training was timed.
The D.2 field objective saturated at update3 and remains weak evidence for
long-run N1 behavior. Independent42–72-second journal verification is separate
saved-data analysis, not moved scientific work.

Conditional projections, including approximately.320seconds added durable
commit cost per history:

- D0/N0: 386.599seconds/history.
- N1: 607.454seconds/history.
- N1 with20% compute sensitivity: 727.988seconds.
- N1 with50% compute sensitivity: 908.788seconds, exceeding900 and therefore
  requiring censoring if realized, not a larger budget.
- 27-history active sum: 12425.867seconds, excluding startup, admission,
  source checks, OS variation and final analysis; not an observed experiment.

These projections support a bounded attempt with existing guards; they do not
prove material recovery or universal resource fit. No threshold, learning rate,
seed, sample count, model/calibration or resource budget was changed.

## Tests, remaining risks and next action

The final revised suite passed22 tests with zero failures/errors/skips, including
all11 reused actual-harness synthetic lifecycle scenarios. Earlier20-test and six admission-test passes, the17-test shutdown-hash failure
and18-test pass remain preserved. The22-test revision adds first-commit
identity/directory fsync ordering and complete Adam-moment/counter catalogs.
No historical test suite, scientific trajectory or reporting-scale benchmark
was repeated. AST/JSON/hash/Markdown/whitespace checks and freeze round-trip
validation are additional static/saved-data checks, not scientific admission
runs. Large outputs remain outside Git at:

`/home/isavand/paper/evidence/3d2-logged-v2-readiness-cb936925df90`.

Remaining execution risks are explicitly bounded rather than hidden: storage
blocking/power failure can lose uncommitted progress; real simultaneous memory
and sustained N1 timing remain unmeasured; recovery may fail despite successful
execution. Such outcomes must be preserved and reported. Default execution is
closed. Old v1 authorization cannot select v2. No positive v2 research review
exists or was created in D.3.

Next action: researcher review of this exact source/freeze/readiness contract,
then a separate explicit v2 run-specific scientific execution instruction if
accepted. Stop here; do not run the27 histories or automatically authorize them.

Read-only verification (not recovery):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B -m experiments.freeze_neural_material_recovery_logged_v2 \
  --freeze-sha256 dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af
```
