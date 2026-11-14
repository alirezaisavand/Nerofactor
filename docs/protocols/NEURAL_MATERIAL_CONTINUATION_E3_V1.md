# E.3 v1: exact state-10 to state-100 neural material continuation

**Preparation only. Separate explicit researcher execution approval is required.**
Baseline: `fea5cfb2ef32c0f5b7c10c6c83b8d314d0c0dcac`, branch
`feat/neural-material-early-step-e2a`. This protocol changes only the endpoint
of the accepted [E.2b experiment](../NEURAL_MATERIAL_LR_E2B_RESULTS.md).
It does not authorize additional interventions or long-horizon recovery.

## Scientific contract and admitted inputs

Continue M1/101, M1/103, M2/103, M3/101 and M3/103, in that order, with
LR=0.01 then LR=0.001 per case. Each of the ten arms begins at its own exact
saved E.2b state 10 and ends at state 100: exactly 90 additional Adam returns.
Do not replay the first ten updates or initialize optimization from state 0.

The selected E.2b run is
`/home/isavand/paper/evidence/neural-e2b-d1f760d630964a2786ff45bde7438825`.
The E.3 freeze enumerates the exact checkpoint manifest, numeric archive and
terminal measurement hashes for all ten arms, plus the accepted report,
completion seal, external process receipt and tracked archival manifest.
Original files remain immutable. Numeric archives use `allow_pickle=False`;
no historical binary checkpoint/pickle loading is involved.

Keep the accepted N1 architecture, calibration, original A2/atan2 shader,
CPU float32 arithmetic, 24 fitting observations, original independent targets,
four fitting tapes and four evaluation tapes, 512 diffuse/1024 specular samples,
linear-RGB MSE and optimizer options unchanged. There is no scheduler,
regularization or gradient clipping. The frozen source/input/runtime manifests
are inherited verbatim from E.2a v2. Only the selected arm's LR differs between
paired arms, as in E.2b. No new seeds, sample draws or reference generation.

## Exact restoration and global parity gate

Read and verify every checkpoint array's hash, finite status, dtype and shape.
Restore all 84 parameter tensors, 12 calibration buffers, all Adam first and
second moments and float32 step scalars (exactly 10), CPU Torch RNG and NumPy
MT19937 state. Verify exact byte equality of the complete restored array
inventory, optimizer options and saved parameter identity before a query.

Reconstructing the original N1 architecture necessarily executes its original
scratch constructor and fixed calibration pass using the original latent and
seed. Those scratch parameters and buffers are replaced completely by the
saved state before any fitting query. This is not state-0 optimization or a
replay of early updates. Record the one original calibration pass per arm
separately from scientific renderer calls.

The ten saved E.2b global RNG states have been checked to be identical.
Install this common state before the legacy runtime context snapshots RNG;
restore the caller's original RNG on every exit. Recheck saved common RNG at
runtime exit. No infrastructure sampling or NumPy/Torch random draw is added.

First restore and measure **all ten arms before any optimizer update**. The
first fitting forward/backward at state 10 must reproduce that arm's saved
terminal loss, original gradient norm and physical values exactly, using the
unchanged one-thread CPU runtime. Persist actual and expected values even if
parity fails. Any restoration/parity failure stops the whole run with zero new
Adam returns. Keep each successful first pass's parameter gradients for
update 11; remove observation hooks and release graph references. Do not
repeat this pass or clear its gradients before the first continuation step.

The original harness's optimizer counters begin at 10 so its unchanged
continuous Adam counter check remains valid through 100. New-work counters
begin at zero. Renderer/backward counters begin at zero: the earlier E.2b
calls are not rerun or counted as new work.

## Measurements and bounded work

Measure pre-update fitting states 10 through 99, then the final fitting state
100. Each fitting measurement includes its original loss backward. At each
transition, call the original Adam step exactly once and observe actual
pre/post float32 parameter movement without altering gradients or weights.
After state 100, perform exactly one independent evaluation forward using
the accepted evaluation tapes and all original fitting, held-out and relighting
observations. No intermediate external evaluation or favorable-state selection.

| New work | Per arm | Ten arms |
| --- | ---: | ---: |
| Update fitting forwards/backwards (including reused parity pass) | 90 / 90 | 900 / 900 |
| Terminal fitting forwards/backwards | 1 / 1 | 10 / 10 |
| Independent external evaluation forwards | 1 | 10 |
| Total renderer forwards/backwards | 92 / 91 | 920 / 910 |
| Additional successful Adam returns | 90 | 900 |

Per arm, 9,072 point/tape combinations yield 4,644,864 diffuse and 9,289,728
specular directions. Full native counters must agree, including exactly one
calibration pass, 91 fitting forwards and Adam counter 100. There are no
new identifiability/reference gates or neural recovery histories outside this
explicit continuation.

Every pre-update record preserves fitting loss, physical values/recovery
errors, all head logits and sigmoid values and their actual loss gradients,
exact sigmoid-boundary channel counts, shader-input gradient summaries,
shared-feature gradients, ReLU activity/gradient summaries and original
parameter-gradient group summaries. Every update also records actual float32
movement by parameter group, full pre/post state identities, and eight fixed
evenly spaced scalar probes per tensor. These permit distinguishing exact
saturation, gradient attenuation and continued Adam displacement; zero current
gradient with movement is consistent with momentum, not alone proof of a
particular stored-moment mechanism.

## Evidence storage and limitations

Use a fresh explicit `neural-e3-<unique-id>` directory directly under
`/home/isavand/paper/evidence`. Never reuse a failed run or modify E.2b outputs.
The freeze itself does not authorize execution. An affirmative execution
review may only be created after researcher approval and binds the exact
clean reviewed commit/branch, E.3 freeze hash, run ID, output and watchdog.

- Each arm has an append-only, fsynced hash-chained `trajectory.jsonl`,
  exactly 91 records on success, at most 64 KiB per record.
- Complete numeric parameter/buffer/Adam/RNG snapshots at states 20, 50, 100;
  each archive is capped at 18 MiB and has a SHA-256 array manifest.
- Complete observed graph signal arrays at states 10, 20, 50, 100 only,
  at most 1 MiB per archive. Per-state compact summaries remain in JSONL.
- `state10-parity.json`, `arm-result.json`, original evaluation predictions,
  component/radiance/physical metrics and selected artifact identities.
- At most one whole-run failing arm's partial full state is captured, with
  explicit failed/censored status and counters. No automatic retry/resume.
- `execution-review.json`, immutable `result.json` and hash-selecting
  `finalization/seal.json`. External command/log/process receipt is separate.

Full per-update tensors, Adam moments and XOR deltas are deliberately not
retained. Do not claim full reconstruction of all 900 transitions from the
compact record. Actual full-tensor movement is measured, while selected full
snapshots/probes allow independent spot checks. Estimated capture is under
700 MiB; enforce a conservative 1 GiB cumulative output cap and require 2 GiB
free before execution. Neither the output allowance nor the thresholds may be
increased after observing results.

## Runtime, failure handling and terminal integrity

CPU only, GPUs hidden, one BLAS/Torch thread; exact E.2b Python/NumPy/Torch and
arithmetic runtime. No Docker, GPU work or package/environment changes.
Use a live foreground supervisor for the whole execution, with external
`timeout --signal=TERM --kill-after=10s 1800s`. Internal legacy backstops:
1,800 seconds whole context, 120 seconds for the entire ten-arm restoration/
parity stage, 120 active seconds for each arm's continuation, 4 GiB peak RSS,
8 GiB address-space soft cap (or an already stricter limit). These are
conservative planning bounds based on E.2b's approximately 50-second whole
execution and 0.83 GB measured peak RSS, not measured E.3 performance.

Before launching, check no conflicting scientific process remains active,
source/input/freeze hashes, CPU runtime, free storage and owned fresh output.
The supervisor must remain attached and record actual exit status. Do not
detach, retry, extend a limit, choose an alternative launch strategy or modify
scientific settings after failure.

Fail closed on identity, restoration, parity, finite-value, counter, storage,
runtime or scientific correctness failure. Preserve completed records and
partial arm status; unstarted arms stay explicitly unstarted. Hooks, timers,
signal handlers, address limit and caller RNG are restored by context cleanup.
Cleanup is acknowledged only after normal context exit. After cleanup,
independently recheck all frozen source/accepted-input identities, fixed
in-memory tape hashes, per-arm counters and aggregate totals. Only then can
scientific work completion be claimed. `result.json` remains an immutable
pre-finalization snapshot; a valid seal and successful external process exit
are also mandatory. A failure report/seal never implies scientific success.

## Prespecified offline analysis

Use saved values only; no additional model/shader/loss queries. Compare paired
state-10 and state-100 fitting losses, gauge-invariant Q errors, roughness,
orientation/albedo/metallic errors and original held-out/relighting/component
errors. Retain every arm, including failures. Use unchanged physical success
criteria; no retrospective threshold, checkpoint or initialization selection.

Plot trajectories by material/seed/LR; identify the first measured exact
sigmoid boundary, shader versus head/shared-network gradient attenuation,
zero-gradient versus nonzero displacement intervals and state-100 endpoint
tradeoffs. Distinguish observations from causal hypotheses. Ten selected arms
through 100 updates do not establish population-level generalization, prevention
of saturation at later times, or 5,000-step recovery.

Rank future separate trunk/head LR, output reparameterization, nonsaturating
mapping, gradient-preserving/trust-region optimization, physically informed
optimization and ambiguity-focused interventions by scientific impact, novelty,
feasibility and cost. None is authorized by this protocol.

## Preparation and future execution

Preparation CLI (no scientific imports):

```bash
python3 -m experiments.neural_material_continuation_e3 --describe
python3 -m pytest -q -p no:cacheprovider tests/scientific/test_neural_material_continuation_e3.py
python3 -m experiments.neural_material_continuation_e3 --prepare
python3 -m experiments.neural_material_continuation_e3 --verify-sha256 <E3-freeze-sha256>
```

The programmatic future entry is `execute_reviewed(review_path, review_sha,
freeze_sha)`. Preparation does not create an affirmative review or execute that
entry. After the one explicit execution approval, the foreground supervisor
must call it under the frozen watchdog with CPU-only environment variables;
the exact run/output/freeze identities are presented with the approval request.
