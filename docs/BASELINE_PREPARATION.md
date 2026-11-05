# Baseline evidence and preparation: 1D.4A

This milestone prepares a comparison for review. It does not execute checkpoint
inference, GPU evaluation, training, mesh extraction or metric recomputation.
All historical inputs remain in place. The repository started clean on
`paper/infrastructure` at `b4379971b07cdbcb3edc9e589efd20ce38b63bd5`.

Follow-up: [1D.4B-0 checkpoint admission](evidence/BELL_CHECKPOINT_ADMISSION.md)
now records a selected checkpoint byte hash, exact CPU-verified A/G source splits
and copies of the original smoke reports at the researcher's approved alternate
archive destination. The initial 1D.4A inventory/evidence status below is retained
as an earlier observation. Producing checkpoint provenance and admission remain
unresolved; no inference or historical reproduction has been performed.

## Evidence status

| Evidence | Status and supported claim | Missing evidence / limits |
| --- | --- | --- |
| Geometry native model smoke in `nerofactor-paper:quick-candidate-1` | Researcher reports a pass after the 1D.3C implementation | No launch-specific `smoke.json`, console log, immutable image ID, command, source diff or input identity supplied/checked in here |
| Material native model smoke in the same image tag | Researcher reports a pass | Same missing launch evidence; a tag can change and does not identify image bytes |
| Python 3.8 infrastructure suite | Researcher reports **109 tests passed** | Full interpreter/package metadata and test log not supplied/checked in; this is separate from the earlier local 106 passes / 3 skips on Python 3.9 |
| 1D.3C harness and CPU/static checks | Versioned source/tests/documentation at the starting commit | Establishes harness behavior, not independently observed GPU output or historical numerical parity |
| Historical artifacts | Independently inspected names, counts, sizes and modification times in this milestone | No checkpoint deserialization, mesh/image decoding or binary artifact hashing; metadata cannot prove content integrity or compatibility |
| Scientific reproduction | **Not performed** | Successful fresh-weight initialization/forward does not reproduce trained weights, historical metrics or optimizer trajectories |

These reports supersede the previous *pending* model-smoke status, with their
attribution retained. Codex did not run or independently witness the native
launches. The 1D.3C implementation record remains an accurate account of what was
executed during that earlier milestone. Do not promote a reported runtime pass
to a complete, verified reconstruction of the historical Docker build.

The checked-in [metadata snapshot](evidence/1d4a-metadata.json) records this
inspection and the reported statuses. Its reported-status fields are an evidence
index, **not substitute launch reports**. No matching smoke reports were found in
the tracked repository evidence inventory; `/home/isavand/paper/paper_runs` is
present and empty. Disposable smoke outputs may exist elsewhere; their location
was not supplied and no global or `/tmp` search was made.

Before promoting a pass to archived runtime evidence, preserve the original
`smoke.json`, console output and original/effective configurations from each
launch, exact exit status, full command, source commit/diff, immutable image ID,
GPU/driver/runtime versions, input identities and timestamps. Hash these small
reports when supplied, record who supplied them and what was independently
checked, and retain incomplete/failed reports. Do not reconstruct missing logs.

## Read-only artifact inventory

Roots are the original host roots in [BASELINE_INVENTORY.md](BASELINE_INVENTORY.md).
The JSON snapshot uses root aliases, relative paths, byte sizes and `mtime_ns`.
It contains 206 individual file records: 26 checkpoints, 50 PLY artifacts,
37 text logs, 15 TensorBoard event files, five historical JSON reports and
73 repository configuration files (72 YAML plus one split pickle). Binary files
were identified by metadata only; no pickle, checkpoint or event file was loaded.

| Inspected root/category | Observation |
| --- | --- |
| Original datasets | 10,297 files, 9,344,508,790 bytes; three families and 20 scenes, matching the earlier family/scene counts |
| Historical results | 5,869 files, 2,283,625,659 bytes across the same 15 top-level directories |
| Checkpoints | 26 `.pth` files in 13 geometry-labeled directories, each with `model.pth` and `model_best.pth`; **no material checkpoint** found anywhere in this root |
| Meshes / point clouds | 50 PLY files: extracted 22, pretrained 18, ablation one, cropped one, aligned real references five, generic evaluation clouds three |
| Historical logs | 26 model text logs, 15 event files, eight main material NVS text logs, two NeRO-labeled and one ablation-labeled material log |
| Evaluation/rendered artifacts | 5,571 `train_vis` JPGs, 40 main NVS PNGs, 10 NeRO-labeled PNGs, five ablation-labeled PNGs, five PBRNeRF JSON reports and 50 associated JPGs; ablation separately has 40 PNGs and 20 JPGs |
| Repository configurations | 72 YAML files (70 legacy plus two schema-1 envelopes), and `synthetic_split_128.pkl`; no historical run-specific YAML/JSON configuration snapshot found under the historical root beyond the five PBRNeRF reports |
| Future outputs | `paper_runs` exists, empty; this differs from its absence during 1D.1 |

Inspection used sorted recursive `scandir` and non-following `stat`, with no
symlinks or access errors observed. Every scene and historical top-level directory
has count/size/type summaries; large image collections have group summaries rather
than per-image records. Sizes count regular entries, not unique inodes/content.
Only selected small YAML/source files, bell's validation text and bear's JSON
report were read as text. Four selected current YAML files have content SHA-256
in the snapshot; these identify present configurations, not producing history.
No dataset, historical mesh or checkpoint content hash was collected. The
inspection's UTC timestamp is in the JSON; modification times are filesystem
metadata, not reliable training dates. Host inputs are readable and writable by
their owner; this inventory neither changes permissions nor claims OS read-only
mount enforcement. Future execution must enforce it at the container boundary.

The original [1D.1 inventory](BASELINE_INVENTORY.md) retains all historical scores,
per-scene checkpoint sizes and branch roles. It is an earlier snapshot, not a
current runtime status report. Exact current refs are in the new JSON; no branch
was changed, fetched or merged.

## Ranked baseline candidates

### Two distinct geometry questions

**A** denotes `anisotropic` at `e1e859779301faced46d6476691e20b546c27ae0`,
the basis of the maintained implementation. **G** denotes the inspected
`origin/aniso_geometry_safe` snapshot at
`79252178e65914ad1acbc14790b1514f276bfea0`. These refs identify inspected code,
not the producing branch of a historical checkpoint.

| Objective | Implementation and split | Claim boundary |
| --- | --- | --- |
| Current-code A-based inference regression | Compare direct-renderer and isolated routing using the same pinned maintained A-based scientific implementation and A's seeded three-way split | Can test routing preservation with admitted weights; cannot establish a thesis split or reproduce G-based thesis geometry |
| Historical G-based thesis geometry reproduction | Recover the attributable producing G commit/configuration and preserve its two-way geometry split, preprocessing and evaluation behavior | Requires checkpoint-linked historical evidence and a separately reviewed protocol; not implemented by the proposed A-based v1 regression |

For `split_type='validation'`, source inspection of
`dataset/database.py:get_database_split` establishes:

- **A:** call `random.seed(6033)`, copy `database.get_img_ids()`, then
  `random.shuffle(img_ids)`. Return `train_ids=img_ids[6:]`, validation
  `test_ids=img_ids[5:6]`, and `nvs_ids=img_ids[:5]`, in that three-list order.
- **G:** call `random.seed(6033)`, copy `database.get_img_ids()`, set `img_num=1`;
  the shuffle is commented out. Return `train_ids=img_ids[1:]` and validation
  `test_ids=img_ids[0:1]`, in that two-list order. There is no NVS reservation or
  third returned list. G's geometry renderer and validation dataset consume two
  lists. The helper still resets global RNG state despite not shuffling, so it
  must not be characterized as RNG-neutral.

For the currently inventoried 128 bell cameras and loader-generated IDs
`"0"` through `"127"` in numeric order, G's source rule would give validation
`["0"]` and 127 training IDs `"1"` through `"127"`. A's source-derived proposal
gives validation `["58"]`, five NVS IDs and 122 training IDs. These are
conditional code-derived results, **not verified producing splits**. In
particular, `"58"` belongs to training under that G rule; substituting A's split
could label a trained-on view as held out. Both branches' `test` mode reads a
separate split pickle and is outside this validation-only proposal.

**Historical geometry reproduction must not substitute A's three-way split for
G's two-way split without checkpoint-linked evidence establishing the actual
producing split.** A regression pass, matching YAML, or a source-derived ID does
not supply that evidence. If G origin is established, preserve its two-way split;
an A-split evaluation of those weights would be a separately labeled cross-split
experiment, not G thesis reproduction. No split policy or source is changed here.

### 1. Bell geometry: first A-based inference-regression candidate, gated

Use `configs/experiments/geometry.example.yaml` referencing
`configs/shape/syn/bell.yaml`, scene `GlossySynthetic/bell` (`syn/bell`). Select
**`final_results/model/bell_shape/model_best.pth` explicitly before inference**;
`model.pth` is a distinct alternate, never an implicit fallback. Both files are
29,966,819 bytes; equality of size proves neither identical weights nor step.
`model_best` is a filename label; the actual selection rule/step must be verified.

Available scene metadata: 641 files including 128 camera pickles, 128 source
RGB PNGs, 128 depth PNGs, 128 training masks, 128 converted JPGs and an evaluation
cloud. No camera pickle was opened. **A-based** source-derived split proposal:
validation ID `"58"`, NVS IDs `"11", "105", "114", "17", "76"`, 122 ordered
training IDs (full lists in the JSON). This is derived on Python 3.9 from the
128 contiguous camera filenames and the inspected split helper's shuffle rule;
it is not a native Python 3.8 or recovered historical split observation. Verify
all ordered IDs in the target interpreter for A-regression admission; that check
would still not establish which split produced the checkpoint. Do not present
`"58"` as the historically verified bell validation ID or a G validation ID.

Other evidence: `meshes/bell_shape-300000.ply` (19,116,958 bytes),
`meshes-pretrained/syn/bell.ply` (4,729,497 bytes), model train/validation text and
one event file, and 300 JPGs in `train_vis/bell_shape-val`. Their checkpoint/mesh/
view correspondence is unknown. The 300-line `val.txt` ends with historical
reported `val-psnr 23.43791 val-ssim 0.89940`; this is a text transcription only,
not a new measurement or the expected result of `model_best.pth`. Do not infer
step or checkpoint identity from the line position or image filename.

Bell's current legacy YAML bytes match all three inspected historical branches.
That **does not** identify producing code or establish architectural/mathematical
equivalence. Checkpoint trust, content hash, producing architecture/config and
stored step are unresolved. This candidate has the most complete local evidence,
but is not yet an admitted, fully reproducible thesis baseline.

#### Bell checkpoint attribution evidence

Establish a chain tied to the exact `bell_shape/model_best.pth` file, rather than
to the directory name or a current branch tip:

1. A later, separately authorized content hash and trust/ownership declaration
   binding this checkpoint to an archived run ID or original checkpoint copy.
2. That run's producing Git commit and any dirty patch/local split edits, launch
   command/entry point, original YAML and effective overrides/defaults. A branch
   name alone is mutable; bell YAML is identical at the inspected A/G tips and
   cannot distinguish them. Current code or tensor shapes alone cannot prove
   the producing implementation.
3. The run's ordered database IDs and camera/filename mapping, exact ordered
   training/validation lists, presence or absence of NVS IDs, split-helper source,
   split mode, seed and call order. An archived split manifest is strongest;
   alternatively, attributable producing source/configuration plus authenticated
   dataset identity/order must allow the lists to be reconstructed and checked
   against run records. A or G tip behavior alone is insufficient.
4. Checkpoint-selection records linking the chosen file and stored step to the
   validation IDs, metric definition and best-checkpoint decision: attributable
   logs/events and saved GT/render outputs where available. Unlinked scalar logs,
   image index 0, filenames or a visually similar view do not establish this link.

Preserve researcher declarations as supplied evidence and record any independent
corroboration separately. Recovering trusted checkpoint fields later may help
corroborate a run, but missing provenance is not filled by deserialization or by
inventing split IDs. The producing branch/configuration/split remain unresolved;
G-based thesis reproduction is blocked until that chain is reviewed. No
checkpoint loading, hashing or new artifact inspection is performed by this
documentation follow-up.

### 2. Bear material: preferred Stage II target, blocked by missing weights

Use `configs/experiments/material.example.yaml` referencing
`configs/material/real/bear.yaml`, `real/bear/raw_1024` and the explicitly selected
`final_results/meshes-pretrained/real/bear.ply` (11,047,097 bytes). Keep the
extracted mesh (22,557,326 bytes), cropped mesh (14,916,374 bytes) and aligned
reference (70,429,804 bytes) distinct; none may silently substitute for it.

The dataset has 397 files: 97 original JPGs, 97 resized JPGs plus 97 PNGs in
`images_raw_1024`, 97 `train_masked` PNGs, one camera cache pickle, one object
point cloud and seven COLMAP-related files. Raw-directory presence does not
establish that every file is selected by the loader. Registered camera IDs/order
and cache/COLMAP equivalence remain unverified; no cache pickle was loaded.

There is no bear material checkpoint, main NeRO bear NVS log/render directory,
or `train_vis/bear_material-val` directory in the inspected historical root.
`pbrnerf/bear/metrics.json` and its ten JPGs are **a different historical method's
report**: NVS IDs `[40, 3, 69, 61, 24]` and average PSNR
`21.829033579600182`, SSIM `0.8420837104562414`. These values/IDs were read from
the small JSON, not recalculated or verified against the current loader. They
cannot serve as expected NeRO material scores, a checkpoint substitute or an
authoritative current split. No bear material inference comparison is possible
until compatible, trusted weights and their provenance are supplied.

The current bear YAML matches `anisotropic` and `ablation_anisotropic`, differs
from `aniso_geometry_safe`; source algorithms differ even where YAML matches.
Top-level energy/specular coefficients `0.01` are not forwarded into
`shader_cfg`; effective shader defaults remain `0.05` under the maintained
implementation. Preserve this scope, all sample counts and `n_lobes` scopes;
never infer that an old checkpoint used corrected or alternative settings.

Other scenes with geometry checkpoints can be considered later; they do not
remove bear's missing Stage II weights. No scene substitution is approved here.

## Review gates and smallest next step

The proposed [checkpoint-inference protocol v1](protocols/CHECKPOINT_INFERENCE_V1.md)
defines admission, split/preprocessing/metric identity, comparison arms, output
files and proposed tolerances. It is a design for review, not a runnable evaluator
or authorization to evaluate. Runtime smoke success removes a reported basic
execution blocker; it does not remove artifact provenance/compatibility blockers.

Before implementing an inference runner, review this protocol and obtain:

1. Archived smoke/test reports with immutable image/source identities.
2. Trusted ownership/origin declaration for the selected bell checkpoint, its
   producing configuration/architecture and known training/selection history;
   establish A/G origin and the checkpoint-linked split evidence above before
   making any historical geometry reproduction claim.
3. The bear material checkpoint and hash-bound origin declaration; preserve the
   original if recovered elsewhere, expose it only as an explicit read-only root.
4. Target-interpreter ordered split evidence, especially bear cache/COLMAP ID
   mapping; exact preprocessing inputs/products and source of reused caches.
5. Approval of the comparison reference, tentative tolerances and bounded GPU
   budget. Reference images/tensors are currently absent; historical montage JPGs
   and rounded metrics alone are inadequate parity targets.

Then separately authorize selected-artifact hashing and trusted CPU checkpoint
structure/shape review (Torch 1.8 pickle loading is not safe for untrusted input).
Only after those gates should a separately reviewed, isolated inference adapter
and CPU fixtures be implemented, followed by explicitly approved one-view GPU
comparisons. No training, full NVS, geometry extraction, scientific repairs,
Docker/image changes or automatic commit/push is part of this milestone.

## Preparation checks performed

- `git diff --check`, changed-file scope and local Markdown link/whitespace checks
  passed. Only Markdown and this milestone's metadata JSON are changed/added;
  scientific code, infrastructure code/tests, YAML and Docker definitions match
  the starting commit.
- All 206 snapshot records were checked against their live sizes/modification
  times; four current configuration hashes agree. Dataset/scene and historical
  directory totals, split membership and renderer defaults were cross-checked
  using metadata/AST only, without importing scientific modules.
- A second non-following metadata scan of datasets, historical results,
  `paper_runs` and configurations matched the initial paths/sizes/modification
  times exactly. No runtime/evaluation artifacts were created.
- `python3 -B docker/paper/static_checks.py` passed definition/version structure,
  complete staged-file/pre-install gates, Compose policy, five Docker shell parses
  and Python 3.8 grammar for 61 files. Its unresolved full-build provenance/artifact
  warnings remain separate from the reported working candidate image's smoke
  success; no engine/build/ABI/runtime/GPU execution is established by this check.

No unit suite was rerun for documentation-only edits. The 109 Python 3.8 passes
are the researcher's reported run, not a new local test result. No GPU or Docker
command, checkpoint deserialization, dataset-wide content hash, installation,
permission change, commit or push was performed in this milestone.
