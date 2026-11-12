# D.1 reporting optimization v1 — engineering draft, not an execution freeze

Scope: saved-data analysis, synthetic report fixtures and bounded CPU reporting
benchmarks only, from `de5277686653033d5655f71bcaa3d70627c12945`. The original
scientific freeze `730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`
and all Phase D evidence remain immutable. This document grants no recovery
execution, no revised freeze, no budget increase and no resumption of update4860.

Before analysis, verify the original report SHA-256
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`,
Phase D bounded summary `1f7f0d2cb9d41104eca447dc09a8f66016b7fd9e87354881763ba410526dac7e`,
all322 frozen sources,99 output files and eight control files. Inventory all
693 baseline tracked files for post-task preservation. Use only fresh owned
benchmark outputs outside Git, datasets, historical results and accepted runs.

Saved-data analysis reports every attempted optimizer history, recorded update
timestamps,100-update and early/middle/late windows, checkpoint-crossing
intervals, CPU/wall time, snapshot count, work and censoring. The timestamp at a
logged update precedes its persist/checkpoint; the next interval includes those
costs. Update1000 also adds the original prefix forward. No per-persist times or
historical report byte sizes were saved: do not manufacture either. Correlation
with preceding history count is observational and confounded by run order.

The isolated stdlib-only prototype has explicit protocol/run/source/config
identity, fresh owner/path/collision guards, typed JSON changes, associated
material/seed/arm, monotonic history events, ordered sequence/hash chaining,
newline record boundaries, explicit failure/censor events, bounded atomic
summary replacement and deterministic final materialization. Every JSON field,
number type and list order must survive; no deletions or automatic recovery
classification are introduced. Unsupported mutations must fail closed.

Durability is explicit. Benchmark `flush` uses unbuffered append plus atomic
summary replacement without `fsync`, matching the original process-crash
buffering guarantee; it does not promise power-loss durability. Separate `fsync`
mode synchronizes new files, records and directory entries and is unit-tested,
not included in flush-mode speed claims. A torn/malformed record fails strict
verification, with no repair, truncation or resume. A valid unfinalized prefix
requires explicit forensic selection and remains unfinalized; a stale summary
is identified by lag. Final verification requires the separately selected final
manifest SHA. An unkeyed chain alone cannot authenticate a deliberately rewritten
log or detect every removed valid suffix without that external anchor.
The final manifest also hashes the identity file, binding its durability
declaration; strict integer fields reject Python's bool/int equality ambiguity.

Predeclared benchmark: one thread, GPUs hidden;180 seconds internal wall cap,
external `timeout --signal=TERM --kill-after=10s 240s`,8GiB address backstop and
2GiB parent RSS check. These are engineering benchmark limits, not changes to
the900/28800-second scientific limits. Preserve failed/partial benchmark outputs.
No scientific shader/model/recovery optimizer imports are permitted. The only
Torch use is bounded inert-tensor measurement of exact detached observer ASTs,
with Adam construction and backward entry points blocked. No real network is
constructed, no backward occurs and no target is generated.

Measure the exact original final report copied through the original
`json.dumps(indent=2,allow_nan=False)`/seek/write/truncate/flush sequence three
times. Separately instrument encoding CPU/wall and writing-plus-flush CPU/wall;
the latter is cached filesystem traffic, not a physical disk throughput claim.

Compare identical representative synthetic payloads at four fixed scales:

| Case | Prefix scale |
| --- | --- |
| early | First D0 history through1000 |
| middle | Thirteen completed histories plus the next through1000 |
| censor_scale | Fifteen completed histories plus the next through4860 |
| planned_27_synthetic | Twenty-six complete histories plus the final through4970 |

Use three subsequent log-persistence events, ten nominal optimizer updates
apart, per case and method. These are payload labels, not executed updates.
The complete27-history scale and N1 history content are synthetic projections.
Use real saved JSON schema/identity shapes and N1 initial gradient-group and
84-tensor metadata, with synthetic losses/work/counts/steps. Record fixture file
hashes and actual sizes; quantify differences from the observed25.7MB report.
Do not claim the synthetic scale is an exact old persistence-call replay.

R0 rewrites the full report at each event. R1 appends deltas and atomically
updates a summary bounded to4096 bytes. Use separate worker processes for
independent peak RSS, identical fixture loads and identical payload changes.
Report all three timings, CPU, logical bytes, writes, steady-state speedup,
prefix bootstrap cost, finalization, chain verification/reconstruction,
one-time materialization and total retained artifact size. R1 combined append
validation/encoding/writing and combined verification/reconstruction are not
misrepresented as individually profiled components. Bootstrap is not a replay
of all historical R0 writes; do not extrapolate a three-event window's total
lifecycle speedup to training. Compare canonical reconstructed reports exactly.

Other observer costs use20 repeats per D0/N1 synthetic shape for detached
gradient summaries and parameter/moment hashing proxies,100 exact seven-vector
transition-digest observer calls, one uncompressed numeric checkpoint write per
shape and1000 resource queries. Identify synthetic proxies, static hypotheses
and unisolated costs. Never infer that N0's seven-coordinate wrapper trained a
large neural architecture. No neural training cost is directly measured here.

Future integration requires separate source review. Map each original mutation
and persist boundary to ordered events without inserting, removing or moving
scientific queries/gradients/updates, observations, checkpoint gates, timers or
RNG calls. Preserve every recorded loss, physical pre-update observation, work
counter, transition digest and all negative outcomes. Reconstruct original-style
reports for existing offline consumers; revalidate exact semantics, source
bindings, failure persistence and authorization rejection. Do not silently
recompute omitted scientific fields.

Only after equivalence/integration/performance review may a separately approved
new protocol/source inventory/freeze be proposed. Preserve original failed Phase
D and its freeze permanently. Any future scientific execution starts freshly
from the original initializations in a new directory; never resume4860, combine
partial trajectories, skip D0/N0 controls or loosen scientific/resource limits.
N1 full training throughput remains an explicit uncertainty. Recommend the
smallest target-independent validation for review if evidence cannot resolve it.
