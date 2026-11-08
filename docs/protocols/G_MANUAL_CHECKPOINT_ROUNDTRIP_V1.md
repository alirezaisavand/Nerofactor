# G manual checkpoint roundtrip v1

Protocol ID: `bell-G-manual-two-update-roundtrip-v1`. Prepared for separate review,
not part of the first historical compatibility/validation experiment. This is
**manually triggered serialization**, not training-triggered checkpoint selection.

Use original G source/config/bell processing and fresh source-isolated process,
empty read-only thesis placeholder (no historical checkpoint), read-only datasets,
fresh writable outputs and explicit manual-roundtrip acknowledgement. Run the
original G Trainer loop with the already frozen two-update observer: exactly two
successful Adam returns at steps 0,1, original zero LR at 0 and warm-up LR at 1,
full configured total_step300000/intervals/seed6033/no clipping. Stop immediately
after the second optimizer return. No extra donor update, validation or natural
checkpoint may occur. Require populated Adam moments and no checkpoint files.

Create a fresh original G network with `training=False` and original optimizer
factory using the **continuing constructor RNG**, no reseeding or tensor edits
to make a test pass. Record that receiver weights initially differ. Invoke
original G `Trainer._save_model(donor, 2, 0.0, owned_fresh_path)` directly, then
original `Trainer._load_model(receiver)` with only the receiver's explicit owned
path attribute supplied. Original methods and four-field format remain unchanged;
default network loading is strict, optimizer state restoration is allowed only
here for this newly G-produced file. No post-load optimizer step is permitted.

Require exact network/buffer names/shapes/dtypes/byte hashes, optimizer group
settings and named moment/counter records, stored step2 and best0.0. Verify
checkpoint/donor immutability and receiver restoration; refuse path collision.
Record file SHA, original method source identities, complete runtime, actual
two-update probe, output hashes and failure diagnostics. No implicit historical
resume or scientific hyperparameter repair.

The original natural schedule would first validate at loop999, save step1000,
select the final validation set, replace best only on strict improvement and
separately save periodic `model.pth`. None of that selection/cadence/logging is
tested by manually serializing step2. Do not label this a natural 1,000-step test
or run 1,000 steps without separate approval. No RNG state exists in the original
checkpoint; roundtrip state equality cannot establish exact future continuation.

Budget: one selected GPU, four CPUs, 64 GiB/no extra swap, 128 PIDs, 2 GiB shm,
900 seconds + 30 seconds termination grace. Use the same immutable image and
isolation as [lifecycle instructions](../G_GEOMETRY_LIFECYCLE.md), with a new
empty thesis mount and `--action roundtrip --acknowledge-manual-roundtrip`.
Never mount historical thesis results for this action. Neither the donor nor
this checkpoint is research geometry; do not extract or hand it to material.
