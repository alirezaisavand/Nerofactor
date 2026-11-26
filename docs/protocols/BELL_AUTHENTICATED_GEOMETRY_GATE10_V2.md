# Bell Gate 10 v2, explicit spawned dummy workers

This amendment preserves the full scientific definition and schedules of
BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.md. Its one execution failed before the
first optimizer return while original Trainer.run created the training iterator.
All eight workers reported segmentation faults. Original source, authorization,
freeze, console and outputs remain at the original evidence path.

The saved receipt has zero Adam returns, validations and checkpoint saves.
The host recorded exit1 and container absence. Native cleanup_verified is false,
so complete internal cleanup is unverified. Container removal is the available
owned-process cleanup evidence.

## Correction and scientific scope

Use an explicit multiprocessing_context='spawn' for both original G DataLoaders.
Retain eight workers, native dummy datasets, collator, RandomSampler, batch size,
shuffle policy, parent RNG draws, photometric ray-pool shuffle and original
Trainer.run ordering. A scoped constructor adapter adds only the process context
and a read-only worker initializer. The initializer verifies eight-worker identity,
absence of initialized CUDA and pinned G module/source provenance. It does not
seed RNG or load images, construct models, sample rays or update parameters.
The original DataLoader binding is restored even on exception.

Pinned Trainer initializes CUDA/network/native libraries before the first loader
iterator. Default Linux workers therefore inherit that initialized process state.
This is a hazardous fork boundary and the correction avoids inheriting those
runtimes. The saved Python traceback cannot identify the exact native instruction
that segfaulted, distinguish CUDA from OpenMP/library state, or prove that spawn
will resolve the admitted container runtime. The actual corrected native launch
remains unexecuted. [PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/2.15/notes/multiprocessing.html)

DummyDataset returns {} during training, and only {'index': index} during
validation. The actual photometric rays are selected in the parent renderer.
Worker scheduling cannot choose a different image/ray. The original
99,999,999-element dummy RandomSampler is retained, including its potentially
large parent-side randperm/list. There is no shortened sampler, new generator,
zero-worker substitution, optimizer change or model initialization call.

## Focused verification and failure evidence

CPU tests verify the constructor argument delta and restoration on failure.
A standalone synthetic PyTorch fixture compares eight spawned workers with and
without the admission initializer. It checks empty training payloads, ordered
validation indices, parent Torch RNG after loader construction and each iteration,
worker termination and no CUDA initialization. This is a synthetic lifecycle
test on the host PyTorch installation. It does not establish native CUDA admission
or full 300k timing. Retain the existing split/schedule/SDF CPU checks.

The separately versioned v2 freeze binds corrected adapter source, new tests,
this amendment, the unchanged v1 protocol and the existing acquisition identities.
The failure report records original receipt/log/freeze/authorization and all
regular output identities, without deserializing any checkpoint. Native runtime
and all original G/A source files remain unchanged.

## Fresh execution and stop rules

A new exact commit/freeze/output-bound researcher approval is required because
the original one-launch authorization was consumed by the failed native launch.
The v2 output path in the new freeze is fresh. No v2 authorization or run directory
is created during repair. Use the same foreground launcher, which now explicitly
selects the v2 freeze. Its GPU-idle/image/container checks remain mandatory.
No retry of the failed batch, resume or output reuse is permitted.

All original limits remain in force, including 300000 original updates,
first1000 lifecycle gate, preserved15k/30k/50k/100k/200k checkpoints,
original losses/schedules, 512 extraction, six-camera audit and all handoff
criteria. Eight spawn processes may have different initialization overhead.
Existing36h train/2h evaluation/1h first-gate limits and memory/output ceilings
are retained. Stop on any worker, numerical, identity, cleanup or resource failure.
The next approval may cover one complete fresh v2 batch and saved-results analysis.
