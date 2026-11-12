# D.1: offline performance audit and reporting prototype

Full-report rewriting is a demonstrated growing CPU bottleneck. Its measured
cost and size dependence closely match the original slowdown, strongly
supporting it as a major contributor. **The exact historical time share and
exclusive causal attribution remain unmeasured.** The standalone append-only
prototype is ready for review; live integration and a revised scientific freeze
are not ready or authorized. No scientific experiment was run.

**A. Repository and original evidence.** New branch
`feat/neural-material-reporting-audit-v1`, based on
`de5277686653033d5655f71bcaa3d70627c12945`. All 693 baseline tracked files,
322 frozen source hashes, 99 original output files and eight control files
remain unchanged. The original Phase D report is
`/home/isavand/paper/evidence/3d2-recovery-v1-3e3001d85600/report.json`, SHA-256
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`.
The scientific freeze remains
`730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`.
Original G/A implementations, BRDF, calibration, optimizer, seeds, targets,
sampling, success thresholds and 900/28800-second limits are untouched.

The [versioned audit](evidence/NEURAL_MATERIAL_REPORTING_AUDIT_V1.json), SHA-256
`44253370392a110b571ccd67fa6c9ff128af3a916e20a5c33eaba0ac933e37dc`,
contains every original timing point, work/status record, checkpoint-neighbor
comparison, all benchmark measurements, source epochs, artifact hashes and
explicit feasibility formulas. Large synthetic logs and arrays remain outside
Git. The original failed/incomplete Phase D result has not been rewritten.

**B. Root-cause investigation.** The exact persistence call resolves from
`OwnedOutput.persist` through `previous.coupling.accepted.persist_report` to
`experiments/material_symmetry_audit_v1.py`. It performs:

```python
payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
stream.seek(0)
stream.write(payload)
stream.truncate()
stream.flush()
```

Every save traverses and pretty-prints the entire report, including all prior
trajectories. Training saves the first and every tenth update inside the active
history clock; there are also boundary/failure saves. There is no `fsync`, atomic
replacement, per-save hash recomputation or independent schema-validation pass
in this function. Encoding checks finite JSON values. It does not copy tensors
or deep-copy the report itself. Scientific observations are converted upstream.

With a byte-identical copy of the actual final 25,744,940-byte report, three
measured saves averaged **1.109708 s**: JSON encoding **1.083282 s (97.62%)**,
write/truncate/flush **0.026425 s**. CPU means were 1.083245 s and 0.026423 s,
respectively. Thus cached physical-file traffic is not the principal measured
cost; these numbers do not measure physical disk commit latency or original
historical persistence calls.

**C. Saved execution timings.** All 16 optimizer-started histories and 11
optimizer-unstarted records are retained in the audit. Absolute per-history
start times and historical file sizes were not saved. Timings below are actual
saved active seconds, not reconstructed persistence-call timings.

| Stage/material/seed | Updates | Active seconds | Early / late seconds per update |
| --- | ---: | ---: | --- |
| D0 M1/101 | 5000 | 386.978 | .074715 / .079885 |
| D0 M1/103 | 5000 | 423.126 | .081947 / .087083 |
| D0 M1/105 | 5000 | 455.495 | .088361 / .093437 |
| D0 M2/101 | 5000 | 489.724 | .095343 / .100387 |
| D0 M2/103 | 5000 | 524.061 | .102212 / .107230 |
| D0 M2/105 | 5000 | 558.143 | .109021 / .113999 |
| D0 M3/101 | 5000 | 592.721 | .115824 / .120984 |
| D0 M3/103 | 5000 | 626.216 | .122590 / .127706 |
| D0 M3/105 | 5000 | 660.802 | .129503 / .134559 |
| N0 M1/101 | 5000 | 699.846 | .137085 / .142429 |
| N0 M1/103 | 5000 | 737.273 | .143801 / .151936 |
| N0 M1/105 | 5000 | 780.646 | .153209 / .158506 |
| N0 M2/101 | 5000 | 816.988 | .160416 / .165833 |
| N0 M2/103 | 5000 | 851.760 | .167697 / .172602 |
| N0 M2/105 | 5000 | 884.037 | .174382 / .177860 |
| N0 M3/101, censored | 4860 | 890.830 | .180579 / .185625 |

Early spans update1 to1000; late spans4000 to the last saved update. All
100-update windows, throughput in both units, middle windows, CPU seconds,
snapshot counts and work counters are in JSON. Within-history late/early
slowdown is about 2.0–6.9%; the first-to-last completed-history active-time ratio
is about 2.28. Across complete histories, prior accumulated log-record count and
seconds/update have Pearson r=.999610. This is observational: order, material,
arm, accumulated metadata and transient load are confounded.

The fitted increase is approximately 35.8 s per additional completed history
at 501 logs/history. A timestamp is recorded before that log's persistence;
the next interval includes its encoding/write. Checkpoint-crossing intervals
are compared against immediate neighbors in JSON; update1000 additionally
contains the original prefix forward. They do not show a growing checkpoint
spike sufficient to explain hundreds of extra seconds. Do not attribute the
entire interval difference to checkpoint I/O.

**D. Benchmarks and provenance.** Two bounded offline runs completed:

- Initial prototype: `/home/isavand/paper/evidence/3d2-reporting-audit-v1-7b5812abf650`,
  audit SHA `576f1d05e2a95ee44b7c8edc52efe169c4dda481bc25fa006ce28108c66cdb23`;
  135.23 s external wall time.
- Final prototype: `/home/isavand/paper/evidence/3d2-reporting-audit-v1-87f191295282`,
  audit SHA `209432cd8eed55180735f4de3bbd557757b7369f9786b7718eaef0d523f89fd0`;
  135.89 s external wall time, 135.40 s internal wall time.

Both used one thread, hidden GPUs, an internal 180-second cap and external
240-second watchdog. The initial source epoch is preserved separately. Review
then added final-manifest binding of the identity file/durability declaration
and strict bool/int checks; all four cases were repeated for the final source.
Neither run contains shader calls, backwards, recovery Adam updates or target
generation. Inert tensor fixtures measure detached observer costs only; they do
not construct a real field, shader or recovery optimizer.

R0 and R1 receive the same hashed synthetic fixtures and three ten-update-spaced
persistence events per scale. Fixture values are payload labels, not optimizer
updates. N1 uses realistic 84-tensor identity/state and gradient-group schemas;
its trajectories are synthetic projections, not measured training. Censor-scale
synthetic pretty JSON is 23.761 MB, 92.3% of the actual 25.745 MB report; omitted
endpoint/state fields and shorter synthetic numbers explain the difference.
The planned-scale fixture is 47.380 MB and must not be called an exact future
report-size prediction. Separate processes measure independent peak RSS.

| Scale | Pretty MB | R0 seconds/save | R1 milliseconds/save | Save-window speedup |
| --- | ---: | ---: | ---: | ---: |
| Early | .727 | .027939 | 1.066 | 26.2× |
| Middle | 19.719 | .846919 | 1.143 | 741.0× |
| Censor scale | 23.761 | 1.027286 | 1.124 | 913.9× |
| Planned27, synthetic | 47.380 | 2.095685 | 1.294 | 1619.9× |

These are **per-save reporting speedups, not scientific experiment speedups**.
The initial epoch gave26.4/756.4/906.6/1642.1×; neither is selectively discarded.
All R0/R1 canonical final reports match exactly. R0 encoding and write/flush are
separately instrumented; R1 append validation/encoding/hash/write is combined,
and summary persistence is separately timed in each sample. Raw CPU/time/byte
samples and file counts are retained. Original persistence-call durations and
historical report-growth bytes were not recorded and have not been fabricated.

**E. Other contributors.** Static inspection and synthetic measurements give:

| Operation | Evidence and classification |
| --- | --- |
| Full-report pretty JSON | Directly measured, size-dependent CPU bottleneck; magnitude matches saved timing progression |
| D0/N0 gradient summary | Inert seven-value fixture34.2 microseconds/call; measured proxy, about.17 s/5000 if representative |
| Rolling vector transition digest | Original observer AST on inert seven-value state74.3 microseconds/call; about.37 s/5000 proxy, not an optimizer execution |
| Resource query | About1.12 microseconds/call in synthetic repeats; cannot explain the progressive hundreds of seconds |
| Numeric checkpoint writing | Synthetic vector.000622 s; 1.49M-parameter/three-array N1 shape.036328 s,17.889 MB; cached write proxy, no real model state |
| N1 gradient summaries | Inert84-tensor fixture6.79 ms/call; meaningful possible cost, about34 s/5000 if representative; not N1 training throughput |
| N1 parameter/moment hashes | About11.67 ms/proxy; frozen harness performs large N1 state hashing at checkpoints, not rolling each update |
| Scalar conversion, finite-state checks, allocation | Present in source, partially included in observers; not isolated across the scientific loop |
| Model/output metadata construction | Small repeated dictionaries plus serialized observations; not separately profiled |
| Full-report copies or validation in old persistence | Unsupported by source: the old save encodes the existing object; it does not deep-copy or run an additional validator |
| Thread/GPU change | No evidence of a change; accepted runtime stayed single-thread CPU |
| Neural architecture causing N0 slowdown | Unsupported: N0 has seven coordinates, and no N1 recovery ran |

The observer ASTs are executed only on inert constant tensors and stand-in
state/options dictionaries. Their timings are engineering proxies, not measured
production-loop decompositions. The benchmark's parent CPU time19.45 s excludes
child workers; per-window worker CPU times are recorded separately. Total
including-child CPU for the entire benchmark was not independently captured.

**F. Reporting design.** The new
[stdlib-only module](../infrastructure/append_only_scientific_log_v1.py) allocates
a fresh owned log outside Git/preserved inputs. Canonical NDJSON records have
strict envelopes, sequence numbers, prior hashes, source/config/run identity,
typed changes and immutable material/seed/arm associations. Histories append in
monotonic update order. Checkpoint, failure and censoring events are explicit.
Invalid multi-change transactions roll back; a short write poisons the writer,
preserves the partial file and forbids automatic retry/resume/finalization.

Summary replacement is atomic and bounded to4096 bytes; observed summaries are
319–336 bytes. A final record and separately selected manifest bind the journal,
summary, identity file and canonical report. Verification is read-only and
reconstructs every supplied field, number type, list order and negative status.
It never derives a scientific result or invokes the training code.

`flush` and `fsync` durability are explicit. Measured flush-mode gains do not
apply automatically to fsync mode. Newline/hash validation detects torn records;
the reader never repairs or resumes them. An explicitly selected unfinalized
valid prefix remains incomplete and may have a lagging summary. Valid-suffix
deletion or deliberate chain rewriting requires an external trusted final hash
to detect; a plain hash chain is not authentication. Interrupted finalization
fails closed rather than claiming a completed report.

**G. Equivalence and integrity tests.** Seventeen new focused tests passed,
zero failed/skipped, final run1.561 s. Complete synthetic D0/N0/N1 histories,
incomplete N0, numerical/global failures, checkpoint references, pre-update
labels, final evaluations and unstarted histories reconstruct exactly. Tests
reject malformed/truncated/duplicated/missing/reordered records, bad chains,
summary corruption, provenance mismatch, changed durability, bool/int identity
ambiguity, unsafe paths, collisions and unexpected resumes. Owned allocation
guards are exercised with temporary user-owned directories. Caller
mutation cannot modify written data; verification leaves artifacts unchanged.
Earlier suites, recovery histories and native diagnostics were not rerun.

**H. Whole-artifact costs and tradeoffs.** For three saves at censor scale, R0
writes71,300,574 logical bytes versus R1's8322 (about8568× fewer). These count
payload bytes, not physical filesystem journals or device traffic. The planned
scale writes142,164,063 versus9819 bytes in the measured save window.

R1 pays one-time costs: at planned scale bootstrap is about10 s,
finalization1.37 s, combined chain verification/reconstruction11.90 s, and
pretty-report materialization2.34 s. Verification and reconstruction were not
separately instrumented; an additional raw journal hash scan was. Bootstrap
seeds a prefix once, not all historical R0 rewrites. Across only three events,
full R1 lifecycle wall time is actually larger than R0; gains accrue over many
saves, not by hiding one-time work.

Planned-scale peak RSS is433,197,056 bytes for R0 and652,042,240 for R1; retained
artifacts are47,391,972 and77,101,522 bytes. R1 retains both journal and final
materialization, and this prototype holds reconstructed history in memory.
Thus it trades more storage/finalization memory for lower recurrent cost.
Scientific integration must recheck the original4GiB RSS/8GiB address limits,
including live models/optimizers and failure-time materialization.

**I. Conditional feasibility.** Late censored throughput was.185625 s/update.
Subtracting the current1.109708-second final-report proxy once per ten updates
leaves a5000-update residual about373 s. That residual includes science, Adam,
observers and bookkeeping; it is not an isolated scientific-compute measurement.
The proxy is about59.8% of the late ten-update interval. A deliberately explicit
±20% proxy-cost sensitivity gives262–484 s, not a confidence interval or changed
scientific tolerance. Exact historical savings remain unknown.

An independent observational fit's non-growing residual is382.18 s/5000.
Adding501 worst measured R1 saves costs about.648 s, suggesting a control history
near382.83 s if the model and integration assumptions hold. This is a plausible
route for all nine D0 and nine N0 controls within900 s, not an observed new run.

N1 is different. The accepted short-fixture estimate608.61 s used observed
forward/backward maxima plus a synthetic Adam timing; it was never a complete
history guarantee. Combining that historical estimate with the control model
would give roughly12390 s (3.44 h) for27 histories plus final reporting, below
28800 s. That is a **conditional projection**, not measured neural training.
Actual N1 optimization, finite-state checks, allocation behavior and sustained
throughput remain unmeasured. Do not extrapolate N0 as actual N1 runtime or
claim recovery feasibility/success from this estimate.

**J. Remaining blockers.** The prototype is deliberately not connected to the
frozen harness. A reviewed adapter must map every mutation, supplied trajectory
digest, snapshot update, event order and failure path without moving or adding
scientific forward/backward/Adam calls or changing timer boundaries. Unsupported
fields must block integration, not disappear. Consumers need explicit summary
or journal access during execution; repeatedly materializing the full report
would recreate the bottleneck. Durability policy, external final-hash storage,
live-memory behavior, source bindings and authorization rejection need review.

**K. Recommended next milestone.** Review this isolated prototype and the
[engineering draft](protocols/NEURAL_MATERIAL_REPORTING_OPTIMIZATION_V1_DRAFT.md).
Then, under separate approval, implement a reporting-only adapter with synthetic
full-lifecycle/reconstruction tests and obtain a bounded target-independent N1
field/optimizer/observer performance validation without recovery targets or
scientific rendering. Only after those gates should a new source inventory,
reviewed scientific protocol/freeze and fresh one-run authorization be proposed.
Keep the old freeze and incomplete Phase D evidence; never resume4860, splice
trajectories, reuse its output, skip controls or enlarge the scientific limits.

Reproduce only new offline tests with:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest tests.scientific.test_append_only_scientific_log_v1 -v
```

The versioned benchmark CLI accepts only the admitted saved report and a fresh
explicit output directory. Its documented draft caps and no-science import
guards apply; it is not a route to execute or resume Phase D. Stop here for
review. No revised scientific freeze or experiment is authorized by this audit.

For a separately requested repeat of this offline benchmark, select an absolute
new directory under an approved writable parent outside all checkouts and inputs:

```bash
: "${REPORTING_AUDIT_OUTPUT:?fresh absolute benchmark output required}"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 240s \
python3 -B -m experiments.benchmark_neural_material_reporting_v1 \
  --report /home/isavand/paper/evidence/3d2-recovery-v1-3e3001d85600/report.json \
  --output "$REPORTING_AUDIT_OUTPUT"
```
