# E.3 recovery v1: four unfinished state-100 arms

**Preparation only. Explicit researcher execution approval is required.**
Baseline `6ce6afc22ce49ce4553bd95bc1c224a8d7cbbba7`, branch
`feat/neural-material-early-step-e2a`. Preserve the
[interrupted report](../NEURAL_MATERIAL_CONTINUATION_E3_RESULTS.md),
[original protocol](NEURAL_MATERIAL_CONTINUATION_E3_V1.md), their frozen code,
all previous evidence and the six completed endpoints unchanged.

## Selected evidence and restoration

Original E.3 run:
`/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae`.
The original failed report SHA-256 is
`79a876fffd11f3e5d44c637a2ca4b9f1aa05ebd6a0f2484744a7403ecbe2a490`.
The v1 E.3 freeze is selected by
`ae258c5f6b7d2b0015f071f01c4bbb53c164f19bc2df1e412bb5752c2b98c606`.

Rehash the accepted original inventory: all 125 raw files and four external
sidecars, including the failed seal/exit receipt, plus selected E.2b inputs.
These file checks are not repeated scientific experiments. The new freeze
enumerates all input identities and exact selected numeric checkpoint paths.
Do not discover latest/best files or deserialize pickle/Torch checkpoints.
All numerical NPZ reads use `allow_pickle=False` and the existing bounded
array-manifest reader.

| Material/seed | LR | Exact restore state | Final state | New Adam returns | Forwards | Backwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3/101 | .01 | 21 | 100 | 79 | 81 | 80 |
| M3/101 | .001 | 10 | 100 | 90 | 92 | 91 |
| M3/103 | .01 | 10 | 100 | 90 | 92 | 91 |
| M3/103 | .001 | 10 | 100 | 90 | 92 | 91 |
| Total | | | | **349** | **357** | **353** |

M3/101 LR .01 selects the original `m3-101-0.01/failure.npz.json`, manifest
SHA-256 `59995eb659d72fa44ed694d65ba94e821277843dc2fb82de82eb721cc04b7a51`.
Its saved parameters must equal the hash recorded after the original update
21. The other three select their exact E.2b final-state manifests, as already
bound by the E.3 freeze. No original update is repeated and no Adam state reset.

Restore all 84 parameter tensors, 12 buffers, Adam moments/float32 counters,
Torch CPU RNG and NumPy RNG with exact array-byte equality and matching options.
Reuse the original strict E.3 restoration function's **same code object** in a
private globals copy, binding only its expected `START` guard to 10 or 21.
The original module/global value remains 10; no frozen source is edited.
The original scratch N1 constructor/calibration runs once per restored arm;
replace its entire parameter/buffer state before the first fitting query.
Record four constructor-calibration passes separately from renderer work.

The selected saved RNG states must be identical. Install their common state
before entering the unchanged legacy resource/RNG context; restore caller RNG
on exit. Infrastructure adds no scientific random draw.

## Representation-aware diagnostic correction

The renderer's float32 lower bound is `float32(.04)`, represented after exact
promotion as `0.03999999910593033`. The legacy diagnostic validator and the
Q-tensor helper each compare with the float64 literal `.04`, rejecting this
representable rendering boundary. A positive sigmoid, for example `1e-9`,
can already map to that floor through float32 `.96*q + .04`; this does not
establish an exact sigmoid endpoint.

Only the new `material_diagnostics_e3r_v1` path changes admission:

1. Preserve the origin dtype. Accept only finite float32/float64 seven-vectors.
2. Compare roughness against the exact `.04` representation of that dtype,
   retain upper bound 1 and albedo/metallic bounds 0–1. Reject one ULP below
   the float32 floor, a float64 value below `.04`, NaN/Inf and invalid shapes.
3. Promote accepted values without clamping, rounding or changing any value.
4. Compile a closed list of hash-pinned legacy analytical metric definitions
   in a private namespace. Replace **only two lower-bound constants in the
   Q-tensor domain guard**. Retain its entire tensor formula and the complete
   gauge/physical-error formulas exactly. Origin-aware validation runs before
   the private tensor helper. No original module/global is monkeypatched.
5. Run the original evaluation function's identical code object in a private
   globals copy whose `previous` dependency delegates everything unchanged
   except `recovery_errors`. Keep shader queries, means, splits, component
   metrics and the original success predicate unchanged.

The original frozen diagnostics retain their old behavior. The new path has
its own protocol identity; no recovery tolerance, physical material transform,
BRDF, derivative, optimizer setting or model architecture changes. Original
shader/field bounds stay unchanged. Interior metric results must match the
legacy results exactly in focused fixtures.

## Global restored-state gate and measurements

Prepare all four restored arms **before any optimizer update**. For the three
state-10 arms, reproduce saved E.2b loss, original gradient norm and physical
values exactly. For state 21, recompute one fitting forward/backward and retain
its raw signals/physical values. This is a **new restored-state measurement**,
not historical parity: the old state-21 diagnostic values were not archived.
Require a roughness component at the exact float32 floor, valid remaining
physical-domain conditions, and successful corrected diagnostics. Otherwise
stop before any new update and report that the proposed mechanism was not
confirmed. Do not widen admission to make the experiment continue.

Reuse these first parameter gradients for update 22/11; no extra first query
or gradient clearing before that update. Continue once per state to 100, with
one original fitting forward/backward per update, one terminal fitting
forward/backward at 100, then one independent evaluation forward per arm.
No intermediate external evaluation or checkpoint selection.

Hold fixed N1/A2-atan2, original independent targets, 24 fitting observations,
tapes 3000–3003/1000–1003, 512 diffuse/1024 specular samples, CPU float32 MSE,
Adam options/each arm's LR, and original physical success thresholds. No
regularizer, scheduler, clipping, new seeds or reference generation.

Native optimizer counters start at 21/10 and must finish at 100. New work
counters start at zero. State-21 native totals are 81 field/shader forwards,
80 fitting forwards/backwards and one external forward; state-10 native totals
are 92/91/1. Point/tape combinations total 35,232, with 18,038,784 diffuse and
36,077,568 specular directions. Six completed arms are never constructed,
rendered or optimized again.

Combined original/recovery executions will contain **900 additional Adam
returns**, **924 forwards and 914 backwards**, plus 14 constructor-calibration
passes. The four extra fitting forward/backwards relative to single-launch E.3
are the repeated state-21 measurement and the three renewed state-10 parity
measurements. Do not conceal these calls or relabel the failed original run
successful. De-duplicate identical state-10 observations for plotting only,
retaining their separate provenance and work counts.

## Storage, runtime and stop rules

Fresh explicit `neural-e3r-<unique-id>` under `/home/isavand/paper/evidence`;
all originals remain read-only. Reuse the existing owned `Store`, hash-chained
JSONL, observer and exclusive report/seal patterns. Per arm: 80 or 91 records,
at most 64 KiB each; full 18-MiB-capped state snapshots only at 50 and 100;
1-MiB-capped raw signal captures at restored state, 50 and 100. Record all
per-state physical/logit/sigmoid/gradient summaries and actual float32 Adam
movement/probes as in E.3. Selected raw signals are saved **before** physical
diagnostics, so a future diagnostic rejection does not erase them. Save a
raw physical rejection record if that diagnostic raises. On later failure,
preserve one partial full state and counters. Do not claim full per-step
parameter/XOR reconstruction from compact evidence.

Keep existing ceilings: whole foreground watchdog **1800 seconds**, **10-second
kill grace**; internal 120-second global restoration stage and 120 seconds per
continuation arm; CPU one thread/hidden GPUs, 4 GiB RSS and 8 GiB address space;
1 GiB cumulative output cap and 2 GiB minimum free before launch. No Docker,
GPU work, package installation or environment/permission changes.

After the one explicit approval, commit this frozen preparation to satisfy the
clean-checkout gate. Bind that commit, branch, freeze, fresh output and watchdog
in an execution authorization. Recheck conflicting processes, source/input
identities, runtime, disk and ownership immediately before one live foreground
launch. Do not detach, retry, extend budgets or alter settings after failure.

Scientific completion requires all four counter sets, normal resource/RNG
context exit, terminal source/input/tape revalidation, immutable report, valid
completion seal and successful external process exit. A failed/censored report
preserves partial/unstarted outcomes and never claims full completion or
verified cleanup. No scientific execution is authorized by this freeze alone.

## After separately authorized completion

Offline analysis must use saved data only. Combine the six original endpoints
with four new endpoints and preserve execution identities and the original
failed status. Report all ten state-10/state-100 physical, held-out/relighting
and component outcomes, saturation/gradient routes and actual Adam motion.
Keep original success criteria and null orientation semantics. Update tables,
figures and a concise findings report; distinguish observations from hypotheses
and 100-step evidence from long-horizon recovery/generalization.

Rank trunk/head LR, material/Q parameterization, gradient-preserving bounded
maps, overshoot-aware optimizers and physically motivated constraints or
observations. Do not execute interventions. Commit/push final compact results
only after successful authorized continuation; retain all raw states outside
Git and report authentication limitations without retries.

Preparation commands (no N1 construction or shader execution):

```bash
python3 -m pytest -q -p no:cacheprovider tests/scientific/test_neural_material_recovery_e3r_v1.py
python3 -m experiments.neural_material_recovery_e3r_v1 --describe
python3 -m experiments.neural_material_recovery_e3r_v1 --prepare
python3 -m experiments.neural_material_recovery_e3r_v1 --verify-sha256 <recovery-freeze-sha256>
```

Future execution remains gated by `execute_reviewed(review_path, review_sha,
freeze_sha)`. No affirmative review or scientific run is created during this
preparation milestone.
