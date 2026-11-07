# Bell checkpoint admission and provenance: 1D.4B-0

**Current follow-up 1D.4B-3B:** hash-bound metadata and structural compatibility
were subsequently admitted, and the researcher A inference regression passed.
See [successful evidence](BELL_INFERENCE_REGRESSION_SUCCESS.md) and
[historical attribution audit](BELL_EVALUATION_PROVENANCE.md). Producing source,
configuration and split remain unknown; no historical thesis/G reproduction is
established. The 1D.4B-0 status and observations below are preserved as a
**historical investigation snapshot**, not the current inference-admission state.

**Historical 1D.4B-0 decision: neither current-code A inference nor historical G reproduction is
admitted for execution yet.** Checkpoint identity, source-defined A/G splits and
existing smoke-report contents are verified below. Producing-code provenance,
checkpoint trust/architecture and stored progress remain unresolved. No checkpoint
has been deserialized, no model executed and no scientific behavior changed.

Review baseline: `5c791a77c160c32d038bdbe1a6f82f20d5800233` on
`paper/infrastructure`; working tree was clean before this investigation.
Observation timestamps are recorded in the linked machine-readable manifests.

Classification throughout:

- **Verified:** directly inspected file bytes, filesystem metadata, Git objects,
  or results of the bounded CPU split check. This does not authenticate an
  artifact's claimed origin or independently attest a historical GPU launch.
- **Inferred:** a plausible interpretation with its assumptions stated.
- **Unresolved:** absent, ambiguous, inaccessible or deliberately not inspected.

## 1. Checkpoint identity: verified bytes, unresolved contents

[Checkpoint identity manifest](bell-checkpoint-identity.json) records exact
filesystem identities. The selected file was streamed through SHA-256 as opaque
bytes; it was not loaded, unpickled, unpacked or copied. The alternate was only
statted and is never substituted.

| Field | Selected `model_best.pth` | Alternate `model.pth` |
| --- | --- | --- |
| Original directory | `/home/isavand/paper/final_results/model/bell_shape` | Same directory |
| Bytes | 29,966,819 | 29,966,819 |
| SHA-256 | `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc` | Not collected; metadata-only alternate |
| Modification time, UTC | `2026-11-03T08:17:35.891692+00:00` | `2026-11-03T08:17:35.855691+00:00` |
| `mtime_ns` | `1793693855891692219` | `1793693855855691516` |
| Device / inode | `66310 / 109587444` | `66310 / 109587443` |
| UID / GID / mode / link count | `1109 / 1109 / 0100644 / 1` | Same |
| Path aliases | No final-component symlink; resolved path equals original | Same |

Selected size, modification/metadata-change time, inode and device were stable
across hashing. Filesystem identity is local to this filesystem; it is not a
portable content identity. Equal sizes do not establish equal weights. The
November 2026 times coincide with workspace preparation and do not establish
training dates, producing code, checkpoint step or best-checkpoint selection.

**Unresolved:** trust declaration binding this hash to the original research run;
state-dict keys/shapes/dtypes, stage/architecture compatibility, saved step and
`best_para`. No conclusion about these is drawn from the filename or size.

## 2. Producing-code provenance investigation

[Provenance investigation manifest](bell-provenance-investigation.json) records
source/configuration hashes, inspected text-log summaries, event metadata,
validation filenames and seven relevant historical split revisions. No event,
pickle or checkpoint payload was deserialized. Validation JPGs were inspected
as filenames/sizes/times only; they were not decoded or scored.

### Verified historical evidence

- **A** is `anisotropic` at `e1e859779301faced46d6476691e20b546c27ae0`;
  **G** is `origin/aniso_geometry_safe` at
  `79252178e65914ad1acbc14790b1514f276bfea0`. Their Git merge base is
  `92b9e4ba8b66830dcda1025abb181cd75edc043c`, not either branch tip.
- Bell's YAML bytes match at A, G, `origin/ablation_anisotropic` and the review
  baseline. Renderer, field, loss, metrics and trainer file bytes differ between
  A and G. Identical YAML therefore cannot identify producing scientific code.
- `model/bell_shape/train.txt` has 15,000 scalar lines; `val.txt` has 300. Neither
  inspected text log contains explicit commit/branch/configuration/seed/split/
  resume/step provenance fields. Their hashes and first/last lines are recorded
  in the manifest; no score was recomputed or used to select a checkpoint.
- The last validation line reports PSNR `23.43791`, SSIM `0.89940`, preserving the
  earlier historical transcription. It is not an expected score for the selected
  file and does not identify the view or the best-checkpoint step.
- One event file exists:
  `events.out.tfevents.1760598355.1ccc15bb9069`, 5,811,089 bytes. The integer in
  its filename decodes to `2025-10-16T07:05:55+00:00`. **Only that label and file
  metadata were inspected.** The binary event payload, actual wall times, steps,
  text summaries and any embedded provenance remain unexamined by constraint.
- There are 300 JPGs in `train_vis/bell_shape-val`; filenames range by numeric
  label from `999-index-0.jpg` to `299999-index-0.jpg`, all with `index-0`.
  Index 0 denotes a validation iterator position, not a database image ID.
  Checkpoint/view/step linkage is unverified.
- No run-specific YAML or provenance manifest was found in the historical root's
  configuration/JSON inventory; its five JSON files are PBRNeRF reports for other
  comparison purposes. Source inspection of both legacy trainers shows saved
  payload keys `step`, `best_para`, `network_state_dict`,
  `optimizer_state_dict`; the save code does not explicitly bind a producing
  commit/configuration/split. This is a source fact, not inspection of this
  checkpoint's actual keys.

### Source history prevents a binary A-versus-G attribution

The history contains additional states, not merely the two current tips:

| Inspected commit | Verified validation helper behavior for numerically ordered IDs |
| --- | --- |
| `b0815ea2acf1a4fbc0d5c8e7a2a1884a539f51c7` | Two-way, unshuffled, `img_num=1`: first ID validation |
| `de0d3412689fabca52cf248241974a3990386139` | Same first-ID rule; this commit changes bell/angel YAML |
| `36dd0bb56936fab12bdc0055e802675c26dd692a` | Two-way, unshuffled, `img_num=8`: eighth ID validation |
| `11f0eac05f73447443cadd63e4d71ed96bac42d2` | Three-way seeded shuffle, five NVS IDs |
| `75644a6dc3b01978ba28683386df11e53892f538` | Returns to two-way, unshuffled, `img_num=8` |
| `3321944553b67e219d051b99c4e317d41604f4ba` | Three-way seeded shuffle, fifteen NVS IDs |
| `32e0d42e523be609f8629e3c4317510b98639eef` | Three-way seeded shuffle, five NVS IDs |

These are source observations, not recovered settings for the checkpoint. The
filename-derived event date precedes the November 2025 A/G tip author dates.
**Inferred:** if that label faithfully reflects this run's original start, an
earlier implementation is plausible. This neither attributes the checkpoint to
G nor rules out later continuation/local edits, copied event files or unreliable
clocks/dates. Git dates and directory names do not authenticate producing history.

| Origin hypothesis | Supporting/consistent evidence | Counterevidence or limitation | Conclusion |
| --- | --- | --- | --- |
| Exact A tip / A-like three-way run | Matching bell YAML and scalar field names; the new smoke demonstrates the current A path can initialize | No historical split/commit link; event date label is earlier than the tip under the assumptions above; smoke uses fresh weights | **Unresolved**, not established or definitively excluded |
| Exact G tip / G-like two-way run | Matching YAML; G's two-way renderer/validation interface is internally consistent; earlier first-ID source states exist | No checkpoint-linked ID lists; date label also predates G tip; shape/file-name similarity cannot distinguish source variants | **Unresolved**, not established or definitively excluded |
| Earlier ancestor, another historical revision or local edits | Verified split evolution includes first/eighth-ID and five/fifteen-NVS rules; earlier-run hypothesis is plausible | No run manifest/command/source diff tying any such state to this file | **Inferred possibility; exact origin unresolved** |

Needed to establish origin: a trusted hash-bound run/checkpoint chain, original
commit plus dirty patch, command and raw/effective configuration, authenticated
input ordering and actual ordered split lists (or attributable reconstruction
corroborated by run records), and selection records binding the saved file/step
to the validation view/metric. An explicit researcher declaration should be
preserved as supplied evidence, separately from independent corroboration.

## 3. Bell split verification: exact helpers, metadata-only inputs

[Split verification manifest](bell-split-verification.json) includes both exact
helper source texts/hashes, full module hashes, source commits, verification
runner source/hash, all ordered IDs, camera filename metadata and RNG-state
hashes. Both historical helper ASTs were executed **unchanged**, in an isolated
stdlib-only CPU process, against a facade exposing only `get_img_ids()`.

Verifier: `/home/isavand/anaconda3/envs/point_nerf/bin/python3.8`, **Python 3.8.19**,
with `-I -S -B` (isolated, no site packages, no bytecode). This is not the smoke
image's Python 3.8.10. No dataset constructor, camera decode, Torch, NumPy, native
extension or model was imported. The unused pickle-based split path is guarded
to fail if called. No scientific source or split algorithm was changed.

**Verified input ordering:** there are exactly 128 scene-root camera pickle
filenames `0-camera.pkl` through `127-camera.pkl`, no aliases, with corresponding
RGB/depth/mask paths present. Both source synthetic loaders define ID order as
`[str(k) for k in range(img_num)]` from the pickle count; they do not take glob's
filename traversal order as ID order. Thus the input IDs are strings `"0"`
through `"127"` in numeric order. Camera contents/actual calibration integrity
remain unverified because pickles were not opened.

| Property | A | G |
| --- | --- | --- |
| Source commit | `e1e859779301faced46d6476691e20b546c27ae0` | `79252178e65914ad1acbc14790b1514f276bfea0` |
| Algorithm | Reset seed 6033; shuffle copied IDs; NVS first five, validation next one, training remainder | Reset seed 6033; do not shuffle; validation first ID, training remainder |
| Returned lists | Three: training, validation, NVS | Two: training, validation; no NVS partition |
| Training count | 122 | 127 |
| Validation IDs | `["58"]` | `["0"]` |
| NVS IDs | `["11", "105", "114", "17", "76"]` | Absent, not an empty third list |
| RNG result | Matches seed 6033 followed by one shuffle | Matches seed 6033 with no shuffle draws; still resets global RNG |
| Repeated from two different initial RNG states | Same IDs and same final RNG state | Same IDs and same final RNG state |

The A ordered training/validation/NVS IDs also match the inspected existing
geometry smoke report, whose owning manifest records Python 3.8.10. This is
independent local 3.8.19 helper verification plus agreement with a preserved
3.8.10 report, not a new container/3.8.10 execution.

**A validation `"58"` is a training ID under the verified G helper.** Both current
source-defined splits are now verified for the metadata-derived order; neither
establishes the split used to produce the checkpoint. Historical reproduction
must not substitute A's three-way split for G's two-way split without producing-run
evidence. An A-split run of G weights cannot be labeled G held-out reproduction.

Ordered training IDs follow (JSON strings; validation and NVS above):

**A training order:**

```json
[
  "14", "98", "103", "117", "91", "63", "67", "104", "20", "39", "74", "35",
  "51", "18", "87", "2", "47", "100", "22", "32", "62", "72", "106", "19",
  "88", "59", "86", "54", "123", "99", "16", "3", "75", "122", "8", "84",
  "64", "23", "119", "36", "0", "60", "6", "10", "120", "121", "116", "70",
  "29", "102", "27", "55", "97", "25", "109", "82", "127", "124", "125", "57",
  "28", "52", "24", "41", "77", "26", "15", "126", "68", "95", "81", "107",
  "30", "12", "33", "94", "111", "80", "43", "31", "110", "21", "78", "37",
  "101", "44", "5", "40", "48", "89", "1", "49", "108", "79", "66", "71",
  "65", "34", "42", "61", "46", "83", "90", "113", "13", "9", "7", "56",
  "96", "85", "4", "112", "92", "50", "115", "118", "69", "45", "73", "53",
  "38", "93"
]
```

**G training order:**

```json
[
  "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
  "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
  "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
  "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48",
  "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
  "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72",
  "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84",
  "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96",
  "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108",
  "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
  "121", "122", "123", "124", "125", "126", "127"
]
```

## 4. Native smoke evidence preservation

**Verified:** the original directories still exist and contain complete native
pass reports, consistent phase logs and original/effective configuration records:

- Geometry: `/tmp/nerofactor-geometry-smoke.WjqeMs49`, run
  `geometry-smoke-20261105054231`.
- Material: `/tmp/nerofactor-material-smoke.dLqrhRPG`, run
  `material-smoke-20261105054527`.

**Archive complete at the researcher's subsequently approved destination:**
`/home/isavand/paper/evidence/1d3c/`. The originally requested
`/home/isavand/paper/paper_runs/evidence/1d3c/` was absent, and creating it failed
because `paper_runs` was not writable by this session. After the researcher
approved the alternate path, its write access was verified before archiving.
No ownership, permissions or Docker settings were changed; original datasets,
historical results and smoke outputs were never moved or overwritten.

Each run has seven byte-identical copies: `smoke.json`, `logs/console.log`,
`config.original.yaml`, `config.legacy.yaml`, `config.effective.yaml`,
`geometry.json` or `material.json`, and `.experiment.json`. Total original payload
copied: **14 files, 296,784 bytes**. No caches, generated datasets, build products,
images or model tensors were copied. Source/copy SHA-256 values and source metadata
were checked before/after copying; differing/existing destinations were not
silently overwritten.

The [versioned archive inventory](native-smoke-evidence.json) is byte-identical to
`/home/isavand/paper/evidence/1d3c/archive-manifest.json`. It records every original
and archived path/hash and evidence limitation. Manifest SHA-256:
`7197646af2db83cb2c69f2cd2e4b409511c64b4d5515afcac62af841d57f289b`.
Copies reside under each exact run ID within the archive root.

| Inspected original file | SHA-256 (matches archived copy) |
| --- | --- |
| Geometry `smoke.json` | `b78259996848735c39dd94cef79b0c3133c2ec599ee79f206e64bd2b13bd911f` |
| Geometry `logs/console.log` | `694c77d15f7d7112f0cb888fa25b0620dec4399a96282bc95e26fa26659f75d5` |
| Material `smoke.json` | `476e8be5df1227e7eb325d6552ab4f4581283e848488e57ef323ac6c1991a01d` |
| Material `logs/console.log` | `87d82de5d93b125f91bc95902f6ce94ba032335e15458862a3c3b68a972d83ac` |

### What these files establish, and what they do not

**Verified file contents:** both reports identify `native-training-forward-v1`,
`backend=native`, `status=passed`, `phase=complete`, one forward at step 0 and
finite CUDA float32 primary RGB of shape `[512,3]`. Both report
`NVIDIA A100 80GB PCIe`, Torch `1.8.1+cu111`, compiled CUDA `11.1`; their stage
manifests record Python `[3,8,10]`. All tensor finiteness flags are true, console
phase sequences reach complete, and original checkpoint directories are empty.
The policy fields specify fresh weights, no checkpoint I/O, optimizer/backward
or evaluation/NVS. This supports the **inference** that the reported native
smokes completed successfully; Codex did not witness or re-execute those launches.

Both reports contain the same **caller-declared immutable image ID**:
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`.
The supplied tag is `nerofactor-paper:quick-candidate-1`. A narrowly formatted
read-only `docker image inspect` failed with Docker socket permission denied.
The ID is therefore verified as a report field, **not independently corroborated
through Docker** or established as a registry manifest/historical base digest.
No bypass, privilege escalation, Docker lifecycle command or settings change was
attempted.

The reports' recorded source hashes all match Git blobs at both the 1D.3C commit
`b4379971b07cdbcb3edc9e589efd20ce38b63bd5` and this review baseline: 29 geometry
files and 34 material files, including the CLI/helper hashes. **Verified content
agreement for those listed files** is not a recovered full launch Git commit,
clean/dirty status or complete workspace identity. The reports do not record
those fields, nor the full invocation, external process exit status, GPU UUID or
driver version. Run-ID/mtime labels do not replace recorded start/end times.

**Researcher-reported only:** 109/109 infrastructure tests passed on Python 3.8.
No original suite log was supplied or found among these selected smoke/log files;
that suite was not rerun here. Smoke/report inspection does not reproduce trained
weights, metrics, checkpoint architecture compatibility or the historical image's
complete build provenance.

## 5. Admission decisions and remaining blockers

| Intended comparison | Decision | Evidence still required |
| --- | --- | --- |
| Current-code A-based inference regression | **HOLD / not admitted for execution**; selected file is a hashed candidate only | Trusted declaration for this exact hash; separately authorized safe checkpoint inspection establishing compatible keys/shapes/dtypes and stored step; original/effective settings for the intended comparison; input/preprocessing identity coverage and reviewed reference/tolerances |
| Historical G-based thesis geometry reproduction | **HOLD / not admitted** | Checkpoint-linked producing G commit/patch/configuration, authenticated historical input/order and two-way split evidence, checkpoint-selection/view/step/metric linkage and historically appropriate reference/environment; current A split or fresh-weight smoke is insufficient |

For a reviewer-authorized A routing regression, unknown producing branch can
remain an explicitly bounded limitation **only after** checkpoint trust and
current architecture/settings are established. This would test current routing,
not historical correctness. G attribution cannot be relaxed into a reproduction
claim merely because such an A regression eventually passes. Material smoke is
fresh-weight environment evidence; it supplies no missing trained bear material
checkpoint.

The smallest safe next action is to obtain a researcher-reviewed, hash-bound
ownership/origin declaration and archived run/configuration/split/selection
records for the selected bell checkpoint. Record absent records as absent. Then
review whether to authorize trusted CPU-only structure/compatibility inspection;
Torch 1.8 pickle loading is not safe merely because a file has a hash. No such
loading is authorized or performed by this report. If provenance remains
unrecoverable, decide explicitly whether a limited current-code A regression is
scientifically useful; historical G reproduction stays blocked.

Before any later inference, review the reference, input-content coverage,
effective defaults, protocol/tolerances and resource budget under
[checkpoint-inference v1](../protocols/CHECKPOINT_INFERENCE_V1.md). Historical
reproduction needs its own attributable G protocol. No view substitution,
partial weight loading, checkpoint repair/conversion, split rewrite, scientific
correction, training, rendering or GPU evaluation is authorized here.

## 6. Validation and preserved boundaries

This investigation runs only opaque-byte hashing/copying, read-only Git/text/
metadata inspection, the stdlib-only CPU split check, and static/documentation
checks. The source loader, split functions, YAML, renderers, losses, optimizers,
sampling, seeds, Docker definitions and original artifacts are unchanged.
No binary event/pickle/checkpoint deserialization, model construction, inference,
training, package/environment modification, commit or push was performed.

Checks passed:

- Both exact historical helpers executed on the metadata facade under Python
  3.8.19; repeated initial-state probes, ordered membership, return arity, input
  preservation and final RNG effects agreed with the recorded rules. Report A
  IDs exactly matched the existing Python 3.8.10 geometry smoke evidence.
- All 14 original/copy hashes and original report metadata agreed after copying;
  the external/archive inventory bytes matched, with 15 small archived files
  including that inventory. Checkpoint metadata stayed stable and all 206 prior
  historical/config inventory records retained their sizes/modification times.
- JSON parsing, exact pinned-helper/runner source hashes, report-versus-manifest
  ordered IDs, local Markdown links/anchors, whitespace, branch/review baseline
  and documentation/JSON-only changed-file scope passed.
- `python3 -B docker/paper/static_checks.py` passed definition/version structure,
  pre-install/staged-file gates, Compose policy, five Docker shell parses and
  Python 3.8 grammar checks for 61 files. Its full-build artifact/base provenance
  warnings remain separate from the preserved native smoke evidence. This
  static command does not test Docker engine, GPU behavior or Python 3.8 runtime;
  the separate stdlib helper check above is the only new 3.8 runtime execution.
- `git diff --check` passed. No infrastructure/unit suite was rerun; 109/109
  remains a researcher-reported earlier run with its original test log missing
  from the inspected evidence set.

Repository changes: this report, four small JSON evidence manifests, milestone
rules in `AGENTS.md` and follow-up pointers in the baseline/protocol documents.
The external archive is separate from Git and contains only the approved small
copies and its manifest. No automatic commit or push.
