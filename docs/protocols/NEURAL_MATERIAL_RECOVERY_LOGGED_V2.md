# Neural material recovery logged v2: final scientific protocol

Phase D.3 prepares a new freeze from
`7580126b8d30fd7ad6621dc673e473d49aebfe2d`. This protocol and its deterministic
`NEURAL_MATERIAL_RECOVERY_LOGGED_V2.freeze.json` do not authorize execution.
A separate researcher instruction and hash-selected, run-specific v2 review
are mandatory. Original v1, its freeze, failed execution, validation and D.1/D.2
source/results remain immutable. No recovery is executed during D.3.

## Question, scientific lineage and scope

Compare D0 direct coordinates, exact N0 module control and N1 original-A-derived
constant neural field on accepted known-plane/known-lighting radiance. This
compares architecture plus orientation encoding and redundant parameterization,
not architecture alone. Three selected materials/three starts do not establish
population reliability or native neural inverse-rendering generalization.
A complete trajectory can fail material recovery. Preserve all such outcomes.

G:79252178e65914ad1acbc14790b1514f276bfea0;
A:e1e859779301faced46d6476691e20b546c27ae0;
accepted3D-1:ccb03d0ec5adc814775538b9cc7427cc57480a98.
Known-plane execution does not run G. Accepted A2/atan2 is distinct from native A.
C.2:ffed2b075b997e7b6faf37840fdec7e1c489d8ef;
original incomplete D:de5277686653033d5655f71bcaa3d70627c12945;
D.1:40b7b301d02835d1b324ed54db6a6c2649f0e049; D.2 is the base above.

Preserve original freeze SHA256
`730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`
and failed original D report SHA256
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`.
The new freeze binds the exact original C/C.1 aggregate, original source
inventory, D.1/D.2 evidence and new code/test/protocol identities. Saved-array
aggregate admission independently rechecks original initial/replay/gauge/state
artifacts; it never rerenders or reruns old optimization. Original C failure
and original incomplete D remain failed. D.2 field saturation remains recorded.
The original aggregate equality helper embeds its C.2 repository path. In a
new worktree, the D.3 adapter permits ONLY `completion_versioned/path` relocation
after hashing both the historical and current completion JSON to the admitted
same SHA256. It reuses unchanged `verify_aggregate` for all saved numerical checks
and requires every other aggregate field to match exactly. The original
aggregate/equality helper and its recorded failed direct relocation check are
not edited or relabeled.

## Unchanged complete scientific contract

The normative numerical contract is the entire original freeze configuration,
settings, inputs, initial-model catalog and work counters, copied key-for-key.
Only a separate `reporting` contract is added. The original
[scientific protocol](NEURAL_MATERIAL_FIELD_RECOVERY_V1.md) remains normative
for equations, arithmetic, sampling, physical criteria and failure semantics;
its v1 authorization/output/minimum-storage instructions are superseded ONLY
by the v2 sections here. New freeze validation requires all original numerical
fields to remain identical. No threshold or hyperparameter tuning is allowed.

Physical order:mx,my,theta,kR,kG,kB,metallic.

| Material | Truth |
| --- | --- |
| M1 | [.28,.49,-.62,.36,.58,.24,.10] |
| M2 | [.17,.62,.91,.56,.31,.18,.65] |
| M3 | [.37,.47,.19,.24,.43,.70,.35] |

Exactly3 materials x seeds101,103,105 x D0,N0,N1 =27 histories. Weight
seeds10101,10103,10105, original fresh latent starts, no historical warm start.
N0 retains exact seven-coordinate D0 initialization/arithmetic. N1 retains
1485584 parameters/84 tensors, original eight-layer256-wide trunk, positional
encoding/skip, five original heads, raw origin/anchor, fixed pre-sigmoid
calibration, rho=.5 and actual normalized two-channel orientation. No decoder,
roughness, physical transformation, frame, gradient or calibration change.

Retain original A2/atan2 float32 CPU shader,512 diffuse/1024 specular samples,
original plane/cameras/lights,24 fitting/24 held-out/36 unseen-relighting
observations in their original order. Reuse admitted independent targets;
never regenerate references or identifiability gates. Training tapes3000–3003,
external1000–1003 and original draw order are unchanged. Loss is four-tape mean
linear RGB followed by original MSE, no clipping/regularizer/scheduler/LM.
Global scientific RNG must not be consumed or reseeded by infrastructure.

All27 initial identities/forwards/backwards and all nine gate sets precede
any optimizer update. Preserve exact D0/N0 sample/component/encoding/gradient
admission and bounded D0/N1 frame/Q/sample/RGB/arithmetic tolerances, warning
policy and invalid-encoding rejection from the original configuration.
Then all9 D0 controls with exact archived1000 prefix; all9 N0 controls with exact
logged/snapshot/final-state and every-update rolling trajectory identity;
then all9 N1. Incomplete D0/N0 stops before neural comparisons.

Exactly5000 continuous successful finite Adam returns per complete history:
lr.01, betas(.9,.999), eps1e-8, weight_decay0 and original accepted effective
flags. Snapshots250,1000,5000, no restarts/selection by held-out performance.
Reuse update251 for post250 objective; retain counted prefix1000 forward,
terminal5000 forward/backward and one84-view external evaluation. Do not add,
remove or reorder any scientific operation. Physical success simultaneously
requires relative Q<=.05, identifiable orientation<=5deg, max albedo<=.02,
metallic<=.03, held-out total RGB relativeL2<=.01, with original null-orientation
and gauge conventions. Preserve fit/held-out/relighting component errors,
compensation, gradients, encoding/radius and every unsuccessful outcome.

Per history:5000 Adam returns,5003 fit forwards,1 external,5002 backwards,
480624 shaded points,246079488 diffuse/492158976 specular directions; two
extra field probes and one separate N1 calibration. Complete totals remain
135000 Adam returns,135108 shader forwards,135054 backwards,12976848 points,
6644146176 diffuse/13288292352 specular directions. Freeze stores all attempted
and completed counter keys and exact27-history aggregate, not just these totals.

## Resources and charged operations

One CPU thread, GPUs hidden, accepted Python/NumPy/Torch CPU float32 runtime.
900 accumulated active seconds/history,10-second endpoint reserve,28800-second
whole process,4GiB RSS and8GiB AS. No budget increase. Original scientific
clocks, handlers and cleanup are reused. Active optimization, scheduled states
and endpoint evaluation retain original charged locations. Failure-only state
serialization retains the original teardown/whole-watchdog semantics; it adds
no optimizer time. Admission, fsync, source checks and setup share the original
whole/external budget. Outer watchdog remains exactly:

```bash
timeout --signal=TERM --kill-after=10s 28800s python3 ...
```

The grace is for preservation only. Blocking storage or SIGKILL can prevent
latest checkpoint, anchor or terminal seal; no claim assumes `finally` runs on
SIGKILL or that every failure is finalized within10seconds.

## Execution-time journal and shutdown trust boundary

The original scientific training code object is reused. Two dictionary
allocations are instrumented as in D.2. Original persist locations, row aliases,
mutation order, intermediate counters/digests, checkpoint/evaluation metadata,
status/failure/censoring and unstarted histories are retained. Every original
scientific global remains identical except output/review and the derived
execute/train bindings. New source hashes must pass before allocation.

Each persist appends bounded ordered v2 events, including a sequential boundary
with mutation count and preceding sequence. No full-report serialization,
whole-history scan or chain reconstruction occurs during progress. D.2's
transactional mutation validator and narrow empty-history initialization schema
are unchanged. Per-record flush is followed by **journal fsync then atomic
fsynced `commit.json`**, binding source/config/run identity, identity-manifest
hash, sequence/head, committed byte count, persistence ID and original status and the saved execution-review hash.
The first commit also fsyncs the journal identity file and journal directory
entries before committing its anchor; a durable data fd alone is insufficient.
Summary is advisory; it can be ahead of the durable anchor after interruption.
All referenced own checkpoint/archive files are closed and fsynced before
returning metadata to a later committed journal event.

Close writes only bounded atomic `execution-end.json`, linked to the exact
commit-anchor hash and exception type. No report digest, full-chain hash,
reconstruction or large report is required at shutdown. A successful seal
requires no pending mutations, no poisoned writer and a terminal committed
status. Close preserves the original exception and closes its owned descriptor.
Seal/anchor control files are at most4096bytes. No science is moved out of its
budget. D.2's expensive9-second materialization is removed from this path.

SIGTERM during optimization uses original scientific handlers and failure
callbacks. SIGTERM during a write may poison the writer or leave a torn tail;
no retries/repair are attempted. SIGTERM after handlers are restored or SIGKILL
can leave no terminal seal. Disk exhaustion can prevent the next commit.
Previously fsynced anchor/bytes remain the selected evidence if the filesystem
honors fsync; hardware/media/filesystem corruption remains detectable, not
repairable by this protocol. Missing latest updates are reported, never invented.

## Separate read-only post-execution finalization

Use `experiments.finalize_neural_material_journal_v2` after the execution process
has ended. Mount or otherwise expose the original run read-only. Select exact
new freeze path/hash, run root, commit-anchor SHA256, full journal SHA256 and
execution-end SHA256. Write ONLY to an explicit fresh, separate owned directory.
No model, optimizer, renderer, checkpoint restoration or scientific query runs.

The finalizer verifies canonical envelopes, every sequence/hash/source/config,
association, transactional mutation and persistence boundary, then all referenced
checkpoint/diagnostic metadata and numeric-file hashes, including the complete
Adam catalog, frozen moment shapes/dtypes and exact float32 counter identity. It writes a single
materialized report and a new `final.json` with input/output hashes and checks.
This file finalizes **postprocessing evidence**, not the original execution
records. Preserve its hash independently. Finalization never edits the input
journal, its summary, anchor, seal, checkpoint or accepted evidence.

Scientific completion additionally requires: complete27 exact matrix,5000
returns each, all work counters and snapshots, initial identity/target/tape
contracts, all initial gates, D0 prefix and exact N0 controls, all endpoints,
original runtime/RNG cleanup and a matching clean terminal seal plus its exact authorized review record. Physical
recovery success remains a separate outcome. Invalid complete claims fail.
Failed/censored/incomplete outcomes retain their original statuses.

For missing seals or an uncommitted/torn tail, an explicitly selected
`--prefix-only` diagnostic may verify exactly the anchor's byte prefix after
checking the caller's hash of the entire original file. It reports tail bytes
and can **never** admit scientific completion, even if a prefix says complete.
This is read-only diagnosis, not repair, implicit recovery, resume or promotion.
Without explicit prefix-only selection, any tail or missing seal blocks.
Missing/corrupt referenced artifacts and changed source/hash identities block.
Interrupted offline materialization leaves partial output with no valid final
manifest; never reuse it. A new explicit output may be selected for analysis,
without running scientific work or changing original inputs.

## Storage and memory readiness

The freeze contains the exact evidence-derived resource budget, including all
three scheduled snapshots and a failure-only state per history, full N1 weights/
moments/buffers, provenance JSON,28 conservatively overbounded triple-audit
failure archives, journal cap, report plus atomic temporary, controls and an
additional reserve equal to the subtotal. Measured checkpoint pairs are expanded
by their own measured size and explicit new provenance allowance. Numeric
shape/source accounting, measured D.2 full-journal/report sizes and32 additional
source entries bound the selected implementation. Four times measured full
journal/report bytes allow payload expansion, with28 maximum record allowances
for failures. These are explicitly enforced storage caps, not claims that
arbitrary exception text has a natural mathematical upper bound. Exceeding a cap
fails closed and preserves partial files. No truncation to fit a budget.

Pre-allocation check requires the frozen complete-run minimum free capacity,
not the old1GiB floor or seven-parameter-only estimate. Recheck ownership,
unaliased containment, capacity and freshness before allocation. Finalization
also checks capacity for its report/temporary. No deleting old files or
permission changes to create space. Numeric large arrays stay outside Git.

Memory accounting combines observed original D peak, D.2 full-report peak,
all27 parameter/gradient/two-moment/buffer bytes, checkpoint temporaries and
observed N1 field-only footprint, deliberately double-counting measured bases.
A bounded touched-array/two-report fixture adds source-sized working proxies,
with no real autograd. This is a conservative planning estimate, not a universal
allocator guarantee. Unmeasured simultaneous shader/autograd/OS behavior is
explicit; unchanged4GiB RSS/8GiB AS runtime guards remain mandatory.
N1 ten-step engineering saturation limits throughput inference. Freeze-results
analysis separates observed times, conditional projections and sensitivities.

## Fresh outputs and future authorization

Approved parent remains `/home/isavand/paper/evidence`. A future reviewed run
must be a nonexistent exact child with unique `3d2-recovery-v2-` run ID. Reject
existing even-empty roots, symlinks/traversal/other parents, implicit resume and
old IDs at their selected paths. No global ID service is introduced.

`experiments.neural_material_recovery_logged_v2_execution.execute_reviewed`
requires the exact schema: phase`D.logged-v2`, explicit researcher scientific
execution authorization, new freeze path/hash, source-inventory hash, exact
approved parent/root/run ID and watchdog28800/10. The old v1 freeze/Phase D
review, missing/incorrect fields, stale hashes and unreviewed paths fail before
allocation/model execution. Freeze always has execution_authorized=false;
readiness is not permission. The API has no default execution route and the CLI
is inspection-only. D.3 creates NO positive authorization or Phase D run.

A future separately authorized command must select its reviewed JSON and hash:

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${V2_REVIEW:?separately approved review required}"
: "${V2_REVIEW_SHA256:?independent exact review hash required}"
timeout --signal=TERM --kill-after=10s 28800s python3 -B - <<'PYTHON'
import os
from infrastructure.signatures import read_report
from experiments.neural_material_recovery_logged_v2_execution import execute_reviewed
review,_=read_report(os.environ['V2_REVIEW'],os.environ['V2_REVIEW_SHA256'])
execute_reviewed(review,review['approved_output_root'],review['approved_run_id'])
PYTHON
```

This is documentation, not authorization to execute it. An external researcher
controller must retain command/review hash, exit status, watchdog result and
independently selected artifact hashes in its own fresh evidence directory.
No current task starts that controller or writes an approved execution review.

## Deterministic freeze and stop

Sorted explicit byte SHA256 inventory includes original sources/validation,
D.1/D.2 lineage, new runtime/finalizer/resources/review/freezer/audit/tests and this
protocol. The readiness evidence is a separately hash-selected DAG input and
contains no freeze hash. Exclude freeze itself and freeze-results prose from
source inventory to avoid cycles. Caller selects the final freeze SHA256.
Source/config changes invalidate it; regenerate a candidate only with documented
review of the changed implementation. Never overwrite the original v1 freeze.
Stop after D.3 readiness; no27 histories, real shader, recovery updates, new
references, N1 recovery or automatic follow-on authorization.
