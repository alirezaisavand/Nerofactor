# E.12 — native smoke blocked before lazy network initialization

**BLOCKED: instrumentation lifecycle defect.** The one approved smoke exited2
with `AttributeError: 'StageTrainer' object has no attribute 'network'`.
It completed **zero optimizer returns** and produced no update records. No
renderer/model instance, training forward/backward, loss, gradient or optimizer
state was reached. This is not observed failure of corrected BRDF calculations,
the dataset, physical recovery or numerical gradients.

The original failed batch is preserved unchanged:
`/home/isavand/paper/evidence/material-core-e12-smoke-43dc4f8845b14a58b91674c4eaa842c0`.
Run ID: `bell-core-e12-smoke-43dc4f8845b14a58b91674c4eaa842c0`.
Source commit: `0e3706cf82d8eea0e37fcec0eedd2e7472923263`.
The [failure manifest](evidence/BELL_NATIVE_MATERIAL_E12_FAILURE_V1.json) records
all original report/receipt hashes and sizes. The
[small console copy](../results/bell_e12_smoke_failure/console.txt) is byte-identical
to the original; caches and raw reports remain outside Git.

## Identity and completed-work assessment

- Stage report SHA256:
  `ff1c14755884707d0b3bc6c965218b293cbf11793c68721c5cacd53cfa752abe`.
- Freeze SHA256:
  `faaffe48071184c495d135b1edae883756e244ed88c824ee0fb3ab45cb53fad4`.
- Execution authorization SHA256:
  `e8621f5059d962b5b11eec2f804ca2eb4e1fbdc00cb6e56eea93b13f7727ba63`.
- Approved dataset admission SHA256:
  `f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.
- Native `status=failed`, `phase=trainer_constructor`,
  `optimizer_calls_completed=0`, `updates=[]`. No success metadata is substituted.

The launcher/authorization/source/admission identities match. All64 smoke-frozen
files, both pinned G/A snapshots and the original E.11D/E/F/G source freezes
verify unchanged. All512 dataset records still match the admitted inputs;
generation/labels/admission identities remain unchanged. The original pending
candidate is not edited. Dataset admission as a weak-cue control remains valid.

Native runtime admission succeeded: Python3.8.10, Torch1.8.1+cu111, CUDA11.1,
ABI0, A10080GB PCIe, driver550.163.01, fork loader context and four Torch threads.
The admitted corrected extension loaded with its exact binary identity. This
establishes import/runtime access only; no BVH instance or geometry query from
model initialization occurred. Config/source/cache-view preparation and the
original Trainer constructor completed before the observer error.

No checkpoint or validation/NVS render files exist. The two-update observer was
never entered, so there are no recorded worker/writer cleanup events or explicit
`cleanup_errors` field. Source control flow shows the original run-created
DataLoader iterator/progress/TensorBoard writer were never created. Do not infer
a completed cleanup gate or peak training-memory measurement from this failure.
The Docker foreground process returned2 and `--rm` was requested; Codex cannot
independently inspect the engine because its socket remains inaccessible.

| Required native gate | Outcome |
| --- | --- |
| Dataset/source/runtime identity | Passed before failure |
| Original Trainer constructor and seeds | Reached |
| Original run/dataset/network initialization | Not entered |
| Anisotropic forward and RGB loss | Not executed |
| Backward and active gradients | Not executed |
| Adam moments, steps, parameter displacement | Not executed |
| Two successful returns and bounded stop | Not executed;0 returns |
| Owned loop-resource cleanup verification | Not executed |
| Full training, checkpointing, NVS, new two-light rendering | Not executed |

## Root cause and smallest repair proposal

Pinned A's `Trainer.__init__` starts at line94 and does not create `self.network`.
The unchanged `Trainer.run` starts at line105 and calls `_init_dataset`,
`_init_network`, `_init_logger` in that order. `_init_network` creates the selected
renderer, losses, optimizer and scheduler. This is the original lazy lifecycle.

In `infrastructure/corrected_material_smoke.py:59`, `run_bounded` calls
`probe.bind_calls(type(trainer.network), type(trainer.network.shader_network),...)`
**before** `TwoUpdateTrace.run()` enters that original lifecycle. Actual method
binding for the corrected subclass therefore accesses an attribute that does not
exist yet. The report's last phase remains `trainer_constructor` because no
observer event was reached. Metadata-only `--validate-only` could not catch this;
the preparation's toy tests exercised different paths, not this lazy binding.

The smallest safe correction is instrumentation-only: initialize fail-closed
forbidden-call guards before entering the original run, then bind the actual
selected renderer/shader methods once at the existing observer initialization
boundary, **after original `_init_network` returns and before the first forward**.
Do not construct the model early, call initialization twice, change seeding,
alter Trainer ordering, remove call guards, weaken two-return tracing or modify
scientific implementations. A focused CPU lazy-Trainer fixture should demonstrate
that initialization/method binding occurs once, that early forbidden calls still
fail and that genuine forward/backward/two-return boundaries remain enforced.

No repair was silently applied to the failed run's frozen files. A correction
must have a distinct reviewed source/freeze, preserve the old commit and all
failure evidence, and use a fresh run/evidence directory with one new explicit
execution authorization. The consumed authorization is not a retry permission.

The immediate next action is this narrow lifecycle repair, not a dataset change,
shader correction, lighting study or full recovery comparison. The separately
prepared [two-light proposal](protocols/BELL_TWO_LIGHT_INFORMATION_E12_V1.md)
remains unexecuted and is not required to repair native smoke instrumentation.

## Review checks

File-only checks passed for launch/approval/source/artifact identities,512 dataset
records, empty checkpoint/render outputs and AST-confirmed lazy initialization.
Markdown links/whitespace and the copied console hash are checked. No historical
test suite, checkpoint load, CUDA probe, native retry or new scientific computation
was launched during failure assessment. This report records an incomplete E.12
native objective; it does not relabel the smoke as successful.
