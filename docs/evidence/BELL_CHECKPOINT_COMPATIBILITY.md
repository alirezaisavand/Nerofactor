# Bell geometry checkpoint compatibility: 1D.4B-1

Current follow-up **1D.4B-3A**: the researcher-generated metadata, real maintained-A
reference and comparison have been independently inspected. The result is
`exact_structural_match` for 137 keys/shapes/dtypes, with original paths/hashes in
[bell-structural-match.json](bell-structural-match.json). This establishes
structural compatibility only. Producing revision/configuration/split remain
unresolved. The [inference adapters](../BELL_INFERENCE_REGRESSION.md) are prepared
for review without native execution. The earlier investigation below and its
JSON remain a historical blocked snapshot, not a replacement inspection result.

**Status: blocked by the execution environment; checkpoint compatibility remains
undetermined.** The exact checkpoint hash matches the approved admission evidence
and the researcher explicitly confirmed trust. No checkpoint was deserialized,
Torch imported, model constructed or weights loaded. The planned CPU-only
inspection stopped before deserialization because isolation could not be verified.

Follow-up 1D.4B-1A: the
[standalone metadata inspector](../CHECKPOINT_METADATA_INSPECTOR.md) is now
implemented with CPU fixture tests and an exact researcher-side Docker command.
It has not deserialized this historical checkpoint or been executed by Codex in
Docker. This document and its JSON preserve the earlier blocked investigation;
actual tensor compatibility remains undetermined pending returned evidence.

Review baseline: `940c2d83463ff9f506ac84bb6fab8b67ee163316`, clean
`paper/infrastructure` before edits. The
[machine-readable compatibility report](bell-checkpoint-compatibility.json)
records the hash/metadata, trust confirmation, failed isolation probe, exact
source hashes/AST comparisons and the gated inspection procedure. The previous
[admission report](BELL_CHECKPOINT_ADMISSION.md) remains historical evidence.

## Verified identity and authorization

Selected input:
`/home/isavand/paper/final_results/model/bell_shape/model_best.pth`.
SHA-256: `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
Size: **29,966,819 bytes**. Device/inode: **66310 / 109587444**.
Size, device, inode, modification and metadata-change times were stable across
opaque-byte hashing and agree with the approved 1D.4B-0 identity. The alternate
`model.pth` was not substituted, hashed or loaded in this milestone.

After seeing the exact path/hash and limited scope, the researcher replied:

> I confirm trust and authorize that limited inspection

That authorizes only CPU deserialization to inspect keys, tensor names/shapes/
dtypes, stored training step and available provenance metadata, with no model
execution or optimizer restoration. **The trust gate is satisfied for this exact
hash and scope**; it is not evidence of the producing branch or split. Do not
request the same confirmation again unless the selected bytes or scope change.

## Execution blocker

A minimal `bwrap` probe requested a separate user/network namespace, read-only
system-library binds, synthetic `/dev` and disposable `/tmp`, then `/usr/bin/true`.
It exited **1** before launching the payload:

```text
bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted
```

The full command is recorded in the JSON. No checkpoint was passed to this
probe. The required network/device-isolated CPU inspection environment could not
be established. No privilege escalation, socket/namespace workaround, package
installation, Docker operation or environment/permission change was attempted.
Restricted network access or an empty `CUDA_VISIBLE_DEVICES` value alone is not
an attestation that arbitrary deserialization runs without network or GPU access.

There is also a separate **reference-construction limitation**: unchanged
`NeROShapeRenderer.__init__` builds `AppShadingNetwork`, which calls
`utils/ref_utils.py:generate_ide_fn`; its lines 81–82 invoke `.cuda()`, including
when `training=False`. Merely disabling dataset initialization does not make this
a CPU-only constructor. No constructor patch, CUDA monkeypatch, fake model or
scientific correction was used to bypass that behavior.

## Source-only architecture comparison

No scientific module was imported. Git source and location-independent ASTs were
compared at the maintained baseline, A `anisotropic` commit
`e1e859779301faced46d6476691e20b546c27ae0`, and G `aniso_geometry_safe` commit
`79252178e65914ad1acbc14790b1514f276bfea0`.

| Inspected declarations | Maintained versus A | Maintained versus G |
| --- | --- | --- |
| `SDFNetwork`, `SingleVarianceNetwork`, `NeRFNetwork` full class ASTs | Exact match | Exact match |
| `Embedder`, `get_embedder`, `IdentityActivation`, `ExpActivation`, `make_predictor` ASTs | Exact match | Exact match |
| Four registered module constructor assignments: `sdf_network`, `deviation_network`, `outer_nerf`, `color_network` | Exact match | Exact match |
| Shape renderer defaults, shader constructor and shader defaults | Exact match | Exact match |
| Bell legacy YAML bytes | Exact match | Exact match |
| Shader diffuse/specular light-prediction method ASTs | Exact match | Differ |

The G difference is in `AppShadingNetwork.predict_diffuse_lights` and
`predict_specular_lights`: A supplies `eps=1e-6` to `F.normalize` in the
`sphere_direction` branches; G omits that argument. These methods do not add
registered network tensors. The inspected shader default is
`sphere_direction=False`; its historical effective value is not established for
these saved weights. Source/default equality does not prove historical overrides,
trained behavior or numerical equivalence.

**Inferred:** matching construction declarations suggest the same registered
layout for these inspected A/G configurations and runtime assumptions. This is
not a generated full reference tensor signature and not a checkpoint comparison.
No actual saved tensor key, shape, dtype, stored step, scalar metric or provenance
field has been observed. No actual tensor incompatibility has been found **or
excluded**. Matching layout later would not distinguish A from G or establish
training/evaluation split identity. Earlier revisions/local modifications remain
possible; none is attributed to this file by the present source comparison.

## Prepared gated CPU inspection

The JSON records a proposed procedure, **not an implemented or executed runner**:

1. Reverify the exact selected hash and existing trust confirmation; establish an
   approved OS-enforced no-network environment with no GPU devices, read-only
   checkpoint/source and only disposable outputs writable. Pin/record the actual
   Python/Torch runtime and inspection script. Set reviewed time/memory limits.
2. In a fresh CPU-only process, load that explicit trusted file once using
   `map_location='cpu'`. Do not enter training initialization/resume, restore
   optimizer/scheduler/RNG state or call any renderer/forward. No alternate file
   or deserialization fallback may bypass a failed gate.
3. Report top-level key names/types; every network tensor name, shape, dtype and
   CPU device; exact stored `step`/available scalar `best_para`; and bounded,
   allowlisted primitive provenance metadata. Other fields receive names/types
   only. Do not publish arbitrary object representations, credentials or tensor
   values, or claim optimizer payloads were omitted from container deserialization.
4. Compare all keys, including buffers and weight-normalization `weight_g`/
   `weight_v`, shapes and dtypes against **separately reviewed CPU-only reference
   catalogues** for A and G. Those catalogues are not generated here; a declaration-
   derived reference must not silently replace or patch scientific constructors.
   Record exact matches, missing/unexpected keys and shape/dtype mismatches.
   Do not call `load_state_dict` merely to inspect, strip/rename keys, cast/reshape
   tensors, partially load weights or relax strict compatibility.
5. Preserve script/runtime/input identities, process result, limits and partial
   failures alongside the new JSON. Recheck input identity. Structural agreement
   can support an architecture decision only; it cannot establish producing-code
   provenance, held-out splits or historical reproduction.

This preparation introduces no executable inspection CLI or scientific option.
A reviewed runner/reference catalogue and working isolation remain necessary.

## Decision and next action

**Current A architecture compatibility: undetermined. A inference: hold.
Historical G reproduction: hold.** Actual tensor/progress inspection is blocked,
not failed or passed. Producing commit/configuration/split/selection remain
unresolved independently of tensor compatibility. The previously verified A
validation ID `"58"` remains a G training ID under the inspected G split.

The smallest safe next action is to provide an approved existing CPU-only,
network-disabled environment and review the metadata-only inspection runner and
reference catalogue. The researcher's hash-bound trust confirmation remains in
force; another trust request is unnecessary for unchanged bytes/scope. No GPU
launch, model execution, image rebuild or scientific change is needed or
permitted to resolve this metadata-inspection boundary.

Checks passed: hash/metadata agreement, pinned Git source/AST evidence, JSON
structure, local Markdown links/anchors, whitespace and the exact three-file
Markdown/JSON-only scope. `docker/paper/static_checks.py` passed definition/version,
staged/pre-install, Compose and five shell-parse checks plus Python 3.8 grammar
for 61 files. Its existing full-build provenance/artifact warnings remain; none
of these checks establishes checkpoint compatibility. The infrastructure suite
was not rerun for these documentation/metadata-only changes.

Changed files: this review, `bell-checkpoint-compatibility.json` and `AGENTS.md`.
Original scientific code/configuration and artifacts remain unchanged. No unit
model construction, checkpoint/optimizer loading, GPU execution, Docker changes,
package installation, automatic commit or push was performed.
