# G historical checkpoint compatibility v1

Frozen protocol ID: `bell-G-checkpoint-compatibility-v1`. Prepared in 1D.6A;
native execution pending review. Source is exactly G
`79252178e65914ad1acbc14790b1514f276bfea0` under the immutable scientific manifest.
Use the independent [G camera-0 protocol](G_CAMERA0_VALIDATION_V1.md) only after
this protocol succeeds. A architecture/camera-0 success is not a substitute.

Select only `thesis:model/bell_shape/model_best.pth`, SHA-256
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
An explicit trusted-checkpoint acknowledgement is mandatory. Torch 1.8's
`torch.load(..., map_location='cpu')` deserializes the full pickle container,
including optimizer state; it is not safe/weights-only deserialization. Open one
unaliased regular file read-only; verify hash, inode/size/mtime and hash again
around loading. No alternate checkpoint, conversion or partial load is allowed.

The original G Trainer constructor performs its own shallow defaults merge and
seed operations. Construct the original G network with `training=False` (the
original extraction-mode constructor; no dataset initialization/forward), CUDA
transfer and original G optimizer factory. Verify exact stored step 150000 and
best metric 23.787122565451323, container keys/types and bounded primitive
provenance. Catalog **every** network key, shape and dtype, including missing,
unexpected and mismatched tensors. Require finite tensors. Do not assume the
historical A count of 137 is the expected G count. Only exact equality permits
`load_state_dict(..., strict=True)`; verify loaded network/buffer byte identities.
Record both catalogs, source/config/runtime identities, strict-load status and
all failures. There is no forward, optimizer restoration, update or resume.

`G-diagnostic.json` must be complete, action `compatibility`, exact G source,
checkpoint identity, `exact_structural_match` and `strict_network_loaded=true`.
Fresh-run collisions, wrong identities, mixed imports, nonfinite tensors or
incompatibilities fail closed before validation. Producing revision, training
split and G resume eligibility remain unresolved even on success.

## Separately controlled historical optimizer diagnostic

Independent protocol ID: `bell-G-historical-optimizer-diagnostic-v1`, action
`optimizer`. Requires explicit completed compatibility report **and its SHA-256**,
the identical implementation/protocol source hashes, a fresh output and the
additional optimizer-diagnostic acknowledgement. Never run it as part of default
compatibility or camera validation.

Before library loading, check group counts, positional parameter counts, unique
saved IDs, Adam field names, nonnegative finite counters, moment shapes/dtypes/
finiteness and group scalar validity. Record the positional ID-to-G-name map:
historical saved IDs do not prove producing parameter-name order. Checkpoint
moments are not reinterpreted/reordered/repaired. Call original optimizer
`load_state_dict` on this disposable object only. Record fresh and loaded settings,
scalar types and all changes: especially epsilon, AMSGrad, betas, weight decay,
and separately progress LR/counters. Library loading can override scientific
G defaults; success must not authorize continuation. Verify unchanged network
bytes; zero optimizer updates, no backward, forward, Trainer.run or save.

## Isolation and evidence

Use the [exact commands and budgets](../G_GEOMETRY_LIFECYCLE.md): reviewed immutable
image, explicit GPU, non-root, no network, read-only root/source/datasets/thesis
and prior report, fresh writable `/outputs`, four CPUs, 64 GiB/no extra swap,
128 PIDs, 2 GiB shm, 300-second timeout plus 30-second termination grace. A GPU
is exposed solely for unchanged G construction/CUDA transfer; this is not the
previous CPU metadata inspector. No native execution from Codex.

Save original YAML, effective G trainer/renderer/shader scopes, complete runtime
and imported-source identities, console/outer exit, before/loaded hashes and
failure phase. No tensor values or optimizer moment arrays are exported. No
checkpoint is written. Compatibility evidence must be reviewed before its
explicit consumption by validation; no automatic sequencing or trust upgrade.
