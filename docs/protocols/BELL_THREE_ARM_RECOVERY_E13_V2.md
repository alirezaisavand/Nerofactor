# E.13 v2: original-variant observer compatibility correction

This is an instrumentation-only amendment to
[the unchanged scientific protocol](BELL_THREE_ARM_RECOVERY_E13_V1.md).
New scientific execution is **not authorized** by this correction. A fresh
approval must bind the new commit, v2 freeze and output/run IDs.

## Preserved failure and root cause

The v1 batch from63c21d1200d174c36ee0a924643033cc365a6d14 failed in A_original
at the post-network `initialized` observer boundary. Network construction and
its geometry preprocessing occurred; no training forward, backward or original
Adam return was reached. The saved optimizer counter is0; B/C are unexecuted.
The recorded cleanup_errors list is empty, but cleanup_verified is false. Do not
promote that to independently verified full cleanup or claim zero native work.
See [the file-backed failure inventory](../evidence/BELL_THREE_ARM_E13_FAILURE_V1.json).

Original-A configuration semantics intentionally leave material_variant absent.
The existing resolver treats this as A_original without inserting the key.
The reused observer incorrectly indexed that optional key. It now reads
`cfg.get('material_variant','A_original')` and still requires exact equality with
both actual renderer and shader identities. Missing B/C declarations, explicit
null/foreign values and mismatched renderer/shader identities fail closed.
The observer does not mutate configuration, construct the model, consume RNG,
initialize the optimizer or alter scientific operations.

## Versioning and unchanged experiment

Preserve the v1 freeze, preparation report, failed batch, authorization, claim,
receipts, logs and outputs unchanged. Historical freezes validate their original
source commits; the shared observer has a new infrastructure identity in v2.
Do not rewrite E.12B/v1 hashes to fit the corrected source.

V2 binds the same667 oracle input records, image, admission, scientific snapshots,
variant code, legacy YAML, initialization/seed, A/B/C order,2000 updates per arm,
sampling, optimizer, losses, physical/photometric metrics, thresholds, endpoint
selection, RNG policy and resource limits. No prior run is resumed or reused.
Only source hashes, parent/failure identities and fresh output/run IDs differ.

The CLI/launcher select BELL_THREE_ARM_RECOVERY_E13_V2.freeze.json. The metadata
test reads the entry point's selected freeze rather than assuming v1. New CPU
fixtures execute the actual E.13 initialized observer with all three existing
configurations and tiny stand-in networks, zero training calls. They verify
configuration/RNG immutability, successful implicit original identity, explicit
derivative identities and rejection of missing/conflicting identities.

After fresh approval, use the same foreground launcher with its v2-bound external
authorization. Stop on any failure; no retry or budget extension. Codex's existing
Docker restriction remains in force. No native launch is performed by the fix.
