# D.2 reporting integration review draft

This is an engineering validation plan and a draft for future freeze review.
It is NOT a final recovery freeze or authorization. The public v2 execution
function rejects every request, including the original v1 freeze.

Baseline: `40b7b301d02835d1b324ed54db6a6c2649f0e049`.
Preserve the original runner, all 322 frozen source identities, original freeze
`730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`,
and original report
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`.

## Predeclared field-only performance fixture

One attempt, no retry or seed selection. Exactly one fresh original N1 model,
seed101, weight seed10101, unchanged calibration/architecture; admit its complete
initial model identity against original M1/101/N1 freeze. Exactly **10** queries
at the original `(1,3)` zero anchor, **10** field-only synthetic scalar objectives,
**10** backward attempts and at most **10** successful original Adam returns.
The objective is `mean((physical * [1,2,3,4,5,6,7])**2) + .1*sum(encoding)`.
This is an engineering objective, with no radiance target or scientific shader.
No A2 constructor/forward, fit/evaluation tapes, initial scientific admissions,
old state replay, or recovery optimization runs. No calibration tuning.

Original Adam configuration is reused unchanged. Record all ten timings,
losses, gradients, optimizer counters, parameter changes, source identities,
RNG observations and negative results. One original-format own numeric NPZ
checkpoint after step10; read/verify its arrays with `allow_pickle=False`,
without restoring/rerunning a model. Measure serialization and read verification.
No restoration timing was collected in D.2. The archived pre-execution plan
explicitly excluded restoration; an intervening draft sentence proposing it
was never implemented or executed and remains preserved in the field source epoch.
No tensor weights in JSON. An external 120-second timeout and internal timer,
hidden GPUs, one CPU thread, peak RSS at most2GiB, 8GiB address backstop.
Stop on mandatory failure and preserve partial evidence. No additional attempts.
This fixture cannot establish actual N1 shader-training feasibility by itself.

## Synthetic integration and reporting measurements

Run original and v2 execution orchestration with explicit inert stand-ins for
shader/model/optimizer/backward/targets/timers. The lifecycle test scales ONLY
its private test loop's 250/1000/5000 milestones to1/2/3. No actual optimizer,
shader or autograd is allowed. Production loop bytecode remains identical.
Compare exact reports and persistence-boundary hashes, including every negative
control, failure/censor, checkpoint and unstarted history. Fixture output cannot
be admitted as scientific completion or execution authorization.

Separately measure actual tracked v2 reporting at the full **synthetic** cadence:
27x5000 completed updates and15x5000 plus one4860-update N0 censor, remaining
histories unstarted. Up to600seconds per synthetic reporting case, <=2GiB RSS.
No model, optimizer, renderer, training targets or scientific updates. Capture
all per-update counters/digests and rich original-schema log/snapshot payloads.
Measure original full-report persistence on identical reports at preselected
history indices0,13,26 (completed),0,13,15 (censored). R0 full-run costs are
projections from those samples, not measured repeated full R0 trajectories.
Measure append, bounded summary, finalization/materialization, independent scan,
RSS and logical storage separately. Keep output journals outside Git.

## Future scientific protocol boundary

Retain original5000 updates,250/1000/5000 snapshots,27 initial admissions,
D0 historical1000 prefix and complete N0 transition/control checks, field/shader,
four-tape mean, MSE, Adam settings/order, physical criteria, failure taxonomy,
900active seconds/history,28800whole seconds,4GiB RSS and8GiB AS. Do not move
reporting outside the original active regions or change timer boundaries.
Flush durability means no power-loss guarantee; corrupt/torn evidence is rejected
without repair/resume. Require independently selected final-manifest hash and
exact final report equality. A finalized failed experiment remains failed.

Before future freeze: review all D.2 measured evidence and limitations, inventory
all new source hashes, complete output/checkpoint admission, capacity sizing,
watchdog-grace feasibility and explicit run-specific approval. Never reuse the
old authorization. Do not authorize or run recovery in D.2.
