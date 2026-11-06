# Strict bell architecture comparison: 1D.4B-2

Implementation and commands for review. **Current result: `blocked`** because the
real maintained-A reference has not been generated. Codex has not executed Docker,
GPU construction, checkpoint deserialization, inference, rendering or training.
No tensor signatures are inferred from ASTs or manually invented.

Review baseline: `56046e584da4f6ff01705273f2ac268f054d5a20`, clean
`paper/infrastructure` before edits. Scientific source and legacy YAML are unchanged.

## Admitted checkpoint metadata input

Exactly one matching-path report was found under
`/home/isavand/paper/evidence/bell-metadata.*/checkpoint-metadata.json`:

- Selected original: `/home/isavand/paper/evidence/bell-metadata.xIvZ9k0z/checkpoint-metadata.json`.
- Original report SHA-256: `6108d45e49f502e48b78f71a8dd329b8f37b95fd249a04fb2a386688d2eb6720`;
  27,711 bytes. It is read-only evidence, not rewritten or replaced.
- Checkpoint SHA-256: `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
- Verified schema/protocol and `metadata_extracted` / `complete` status; saved step
  **150000**, saved scalar `best_para` **23.787122565451323**.
- All **137** names are unique; every shape consists of valid nonnegative integer
  dimensions; every dtype is `torch.float32` and every recorded device is CPU.
  Scalars retain empty shapes. There are 37 `.weight_g` and 37 `.weight_v` entries;
  `color_network.FG_LUT` is present with shape `[1,256,256,2]`.
- The earlier inspector's source hash matches the reviewed script. Its primitive
  provenance projection is empty; this does not prove absence of every possible
  provenance field. The producing commit and split remain unknown.

[Metadata admission evidence](evidence/bell-metadata-admission.json) records every
discovered report identity, the explicit admitted path/hash, all validated 137
entries and the blocked comparison. No latest/glob-first selection occurs.
The production comparator pins this report hash, not merely its checkpoint-hash
field. If additional reports appear, inventory/identify them and review admission;
do not silently change the pinned constant or substitute another file. No
historical checkpoint was reopened or deserialized for this milestone.

## Utilities and actual reference construction

[generate_bell_reference.py](../generate_bell_reference.py) is a separate explicit
researcher-side execution entry point. Standard-library runtime guards and fresh
output allocation precede scientific imports. It reads **the actual**
`configs/shape/syn/bell.yaml` using `yaml.safe_load`, without overrides, then calls
unchanged `NeROShapeRenderer(cfg, training=False)`. It does not instantiate a
Trainer, configure an optimizer, load any checkpoint or call a forward method.
The built-in `training=False` constructor argument skips dataset/ray setup; it
preserves neural module construction and shader defaults. No datasets or meshes
are required or mounted. The original direction encoder's `.cuda()` allocations
are why the constructor needs one explicitly selected GPU.

The utility calls the constructed model's real `state_dict()`, `named_parameters()`
and `named_buffers()`. Every actual state entry receives its exact name, shape,
dtype, device and parameter/persistent-buffer role. Nonpersistent buffers are
excluded by `state_dict()` itself; unclassifiable state fails rather than being
invented. Weight-normalization `weight_g`/`weight_v` and registered `FG_LUT` are
included when actually present. No tensor values, hashes or reductions are read.
No historical weights are loaded, and there is no model `.cuda()` rewrite,
constructor patch, replacement network or AST-based shape synthesis.

`reference-signature.json` is the output manifest: protocol/status/phase,
constructor arguments, full state key inventory, actual tensor catalogue,
original YAML/hash/raw values, effective renderer and shader configuration,
observed source-file/BSDF/config hashes, runtime versions/GPU and caller-declared
image/commit/diff/tree identities. Caller declarations are not independently
attested. Renderer defaults are merged by the original constructor. Trainer
optimizer/scheduler defaults are not merged into this structural-only constructor.
No seeds are overridden, and random initial tensor values are never compared;
there is no training RNG or numeric-parity claim.

[compare_bell_signatures.py](../compare_bell_signatures.py) and the shared
[infrastructure/signatures.py](../infrastructure/signatures.py) are JSON-only,
stdlib-only code. They do not import Torch, deserialize weights or execute models.
JSON parsing rejects duplicate object keys, nonfinite values, duplicate tensor
names, invalid dimensions/dtypes/counts, unsupported status/protocol and missing
reference configuration/source/runtime evidence. Both input reports are bound to
explicit SHA-256 arguments; the checkpoint report must also match its admitted
hash. Reference state key inventory and weight-normalization roles/pairs are
validated. Report hashes bind bytes, not authenticity of the claimed execution.

## 1. Researcher-side GPU reference command (not executed by Codex)

Run after reviewing/recording this source snapshot, using ordinary Docker access.
No build/install is included. A changed image ID or runtime failure requires
review, not an automatic replacement image, package upgrade or permission change.
The expected ID is from the preserved smoke reports; local engine inspection
checks that tag-to-ID association before launch.

```bash
set -eu
ARCH_SOURCE=/home/isavand/paper/Nerofactor-paper
ARCH_IMAGE_ID=$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)
test "$ARCH_IMAGE_ID" = sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
ARCH_UID=$(id -u)
ARCH_GID=$(id -g)
test "$ARCH_UID" -ne 0
ARCH_GPU=0
ARCH_COMMIT=$(git -C "$ARCH_SOURCE" rev-parse HEAD)
ARCH_DIFF_SHA=$(git -C "$ARCH_SOURCE" diff HEAD --binary | sha256sum | cut -d ' ' -f 1)
if test -z "$(git -C "$ARCH_SOURCE" status --porcelain)"; then
  ARCH_TREE_STATE=clean
else
  ARCH_TREE_STATE=dirty
fi
ARCH_REFERENCE_ROOT=$(mktemp -d /home/isavand/paper/evidence/bell-reference.XXXXXXXX)

docker run --rm --pull=never --init \
  --user "$ARCH_UID:$ARCH_GID" --gpus "device=$ARCH_GPU" \
  --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true \
  --memory=8g --memory-swap=8g --cpus=2 --pids-limit=128 --shm-size=128m \
  --ulimit core=0 --env HOME=/outputs \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$ARCH_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$ARCH_REFERENCE_ROOT,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout \
  "$ARCH_IMAGE_ID" --signal=TERM --kill-after=30s 180 \
  /usr/bin/python3 -I -B /workspace/generate_bell_reference.py \
  --output-root /outputs --source-commit "$ARCH_COMMIT" \
  --source-diff-sha256 "$ARCH_DIFF_SHA" --source-tree-state "$ARCH_TREE_STATE" \
  --image-id "$ARCH_IMAGE_ID" --execute --confirm-isolated-runtime

printf 'Reference manifest: %s\n' "$ARCH_REFERENCE_ROOT/reference-signature.json"
```

Select `ARCH_GPU` explicitly for the intended device. Only read-only source,
configuration/assets and a fresh output directory are mounted: **no historical
checkpoint, datasets, meshes, Docker socket or unrelated experiment storage**.
HOME/library caches/JIT/temp files use `/outputs/runtime`; no namespace-permission
workaround is attempted. The writable output is required to be empty; do not
pre-create a log there. The manifest is created exclusively before construction
and retains failure/partial state. An OOM/timeout/incomplete manifest is not a
reference. No automatic retries or scientific downsizing are allowed.

The diff SHA covers `git diff HEAD --binary`, which excludes untracked files;
review/record those separately or commit the approved implementation before
execution. Observed utility/scientific file hashes supplement this declaration,
but neither establishes complete image/build provenance. Retain the exact command
and external exit status with the result. A reference requires exit 0 and
`status=reference_generated`, `phase=complete` under the stated protocol.

## 2. Subsequent CPU-only JSON comparison command

This is a different operation, **after a real reference exists**. It needs no
Docker, GPU, Torch, checkpoint file or scientific imports. Use the reference root
printed by the successful first command (the variable below persists in the same
terminal; otherwise explicitly set it to that actual run directory).

```bash
set -eu
ARCH_SOURCE=/home/isavand/paper/Nerofactor-paper
ARCH_CHECKPOINT_REPORT=/home/isavand/paper/evidence/bell-metadata.xIvZ9k0z/checkpoint-metadata.json
ARCH_REFERENCE_REPORT="$ARCH_REFERENCE_ROOT/reference-signature.json"
ARCH_REFERENCE_SHA=$(sha256sum "$ARCH_REFERENCE_REPORT" | cut -d ' ' -f 1)
ARCH_COMPARISON_ROOT=$(mktemp -d /home/isavand/paper/evidence/bell-comparison.XXXXXXXX)
python3 -B "$ARCH_SOURCE/compare_bell_signatures.py" \
  --checkpoint-report "$ARCH_CHECKPOINT_REPORT" \
  --checkpoint-report-sha256 6108d45e49f502e48b78f71a8dd329b8f37b95fd249a04fb2a386688d2eb6720 \
  --reference-signature "$ARCH_REFERENCE_REPORT" --reference-sha256 "$ARCH_REFERENCE_SHA" \
  --output-root "$ARCH_COMPARISON_ROOT"
```

Both reports remain untouched. Each comparison requires a fresh unaliased output
directory outside the repository and input directories. Collisions, escapes and
input/output overlap fail before overwriting anything. Invalid/missing/mismatched
input identities produce a `blocked` report when safe output allocation succeeds;
unsafe output allocation produces stderr/nonzero exit without writing a report.

## Results and admission criteria

`architecture-comparison.json` contains both report identities, total tensor
counts, exact key-set equality, every missing/unexpected key, every shape/dtype
mismatch, reference persistent-buffer coverage and weight-normalization pair
coverage. No prefixes are stripped, no names normalized, no dimensions reshaped
and no dtypes cast. Device placement is reported but is not a structural
key/shape/dtype criterion; no values or numerical tolerances are compared.

The only compatibility results are:

- `exact_structural_match` (exit 0): complete key sets, shapes and dtypes agree.
- `structural_mismatch` (exit 1): all valid inputs were compared, with explicit
  key/shape/dtype differences. Do not partially load weights to force agreement.
- `blocked` (exit 2): inputs/reference missing, invalid or identity-unverified;
  no compatibility verdict can be made.

Checkpoint parameter/buffer roles were not independently extracted by the old
inspector; their coverage is associated by exact reference names and declared as
such. A structural match permits a later reviewed architecture-admission decision,
not automatic inference or loading. It establishes none of producing revision,
final-version status, tensor-value equality, correct BRDF/geometry mathematics,
optimizer history, scientific/numerical reproducibility or thesis scores.
The actual saved step 150000 is not silently equated to the YAML's 300000-step
endpoint. A/G constructor layouts can coincide despite different light-prediction
behavior and split algorithms. In particular, A's validation `"58"` is a G
training ID under the inspected G split. A match does **not** prove A origin or
its three-way split, nor authorize substituting A's split for G reproduction.

Current metadata admission is verified, but the reference identity and all
comparison differences remain null/blocked in the checked-in evidence until a
real constructor result is supplied. No failed/blocked/synthetic result replaces
original smoke, checkpoint metadata or historical artifacts.

## CPU and static validation

- Full infrastructure suite: **139 discovered, 136 passed, 0 failed, 3 skipped**
  on Python 3.9.12. The 14 new tests all passed there, including a real disposable
  CPU Torch module with weight normalization and persistent/nonpersistent buffers.
  Existing native OpenCV resize/crop and Open3D export checks remain skipped.
- Focused Python **3.8.19** suite: **12 passed, 2 skipped, 0 failed**; this thin
  interpreter lacks PyYAML for the mocked generator-main fixture and Torch for
  the real CPU module fixture. All assertions remain enabled when dependencies
  are available. No package was installed to address the skips.
- **28 Docker-definition/prerequisite CPU tests passed.** Static checks passed
  staged-definition/pre-install policy, five Docker shell parses and Python 3.8
  grammar for 67 files. Existing full-build/base/artifact provenance warnings are
  separate from the reported working image and are not silently resolved here.
- Report SHA/status/count/unique-entry/step/metric validation, source-call/import
  audits, local Markdown links/anchors, whitespace, exact changed-file scope and
  shell syntax/isolation flags for both command blocks passed. Shell blocks were
  parsed only, not executed.

Unit fixtures cover complete matches, missing/unexpected keys, shape/dtype
mismatches, duplicates, invalid shapes/counts/protocols/provenance, report hashes,
JSON duplicate keys/nonfinite numbers, output escapes/aliases/collisions,
constructor arguments/config preservation, actual state-dictionary buffer and
weight-normalization coverage, native-gate failures and blocked missing-reference
reports. Synthetic signatures are labeled fixtures and never substitute for the
pending real reference. No historical checkpoint was loaded, no original report
or scientific configuration changed, and no commit/push was performed.
