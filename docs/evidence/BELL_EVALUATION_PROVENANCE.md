# Bell historical evaluation attribution: 1D.4B-3D follow-up

The maintained renderer now reproduces the checkpoint's exact stored best PSNR,
**23.787122565451323 dB**, on explicit **source camera `"0"`**. SSIM
**0.8772724776097482** rounds to historical logged **0.87727**. Steps 149999 and
150000 produce bitwise-identical RGB, and both new complete validation montages
are byte-identical to the historical montage. The verified
[camera-0 evidence](BELL_CAMERA0_DIAGNOSTIC_SUCCESS.md) accounts for the observed
**9.062709797804164 dB** discrepancy through camera selection under this
checkpoint and tested evaluation implementation.

The original [A image-58 regression](BELL_INFERENCE_REGRESSION_SUCCESS.md) remains
unchanged and valid as a current-code regression. Its validation 58 is **not the
historical geometry evaluation split**, nor a historically verified held-out
view. The unshuffled two-way historical training split (train 1–127, validation
0) is **strongly supported but unproven**; producing source/configuration and
actual training membership remain unresolved. Metric or montage agreement does
not supply a producing-run record. This follow-up only inspects saved evidence:
no new checkpoint/event/pickle deserialization, GPU inference, training or
original-artifact modification occurred.

## Sources and citation convention

Artifact paths below are relative to the read-only **thesis root**. `S-A` denotes
Git commit `e1e859779301faced46d6476691e20b546c27ae0`; `S-G` denotes
`79252178e65914ad1acbc14790b1514f276bfea0`. They are inspected snapshots, not
established producing revisions. Source citations give commit label, path and
1-based lines. Full SHA-256 identities for every cited source file, function
ranges/AST comparisons, artifact metadata and historical split revisions are in
[the machine-readable 1D.4B-3B audit](bell-historical-evaluation-audit.json).
That JSON is preserved as an as-of snapshot, including its then-unknown
metric-cause finding. The versioned
[1D.4B-3D success manifest](bell-camera0-diagnostic-success.json) supplies the
new findings and supersedes that uncertainty only within the tested evaluation
scope. It leaves the original image-58 evidence and source audit intact.

| ID | Historical artifact / source | SHA-256 / evidence pointer |
| --- | --- | --- |
| H1 | `model/bell_shape/val.txt`, 300 lines | `898de60200fe0cb8743ecf79e0c6c4d1fc4a310359acbe07d82fe308b63d7702` |
| H2 | `model/bell_shape/train.txt`, 15,000 lines | `7250b274e0d4d3b6626f3596d92327be26419ed228ff33c270b678eb133243f7` |
| H3 | `train_vis/bell_shape-val/149999-index-0.jpg`, 800 × 1000 montage | `2800003a665d47f757b1ec30b98d225bcfe19751b2608f1caa75587f8bcb3a44` |
| H4 | `model/bell_shape/events.out.tfevents.1760598355.1ccc15bb9069`, 5,811,089 bytes | `b2179a3e075cf6086f7e10e2d452af5c8a3b357c8064b0df7a85e1a488b548f5` |
| H5 | Hash-bound checkpoint metadata, step/best field | [Prior metadata admission](bell-metadata-admission.json), report SHA `6108d45e49f502e48b78f71a8dd329b8f37b95fd249a04fb2a386688d2eb6720` |
| H6 | Admitted current inference reports/arrays | [Success manifest](bell-inference-regression-success.json), four explicitly selected report hashes |
| H7 | Camera-0 primary/control/comparison and saved artifacts | [Camera-0 success manifest](bell-camera0-diagnostic-success.json); full paths/hashes in [summary](BELL_CAMERA0_DIAGNOSTIC_SUCCESS.md#original-evidence-and-identity-checks) |
| S-A / S-G | Source file bytes at the exact commits above | `sources.A.files_sha256` / `sources.G.files_sha256` in [audit JSON](bell-historical-evaluation-audit.json) |

## Historical metric and checkpoint-selection chain

**Verified (H1, H5):** validation row **150** is:

```text
val val-loss_rgb 0.06437 val-loss_eikonal 0.00006 val-psnr 23.78712 val-ssim 0.87727
```

It is the only maximum of the 300 logged PSNR values. Formatting the admitted
checkpoint's full-precision `best_para=23.787122565451323` to five decimals gives
`23.78712`. The last row, 300, instead reports PSNR `23.43791`, SSIM `0.89940`;
it is not the selected checkpoint's best metric. Nearby rows 149/151 report
PSNR `23.35360` / `23.62050`. Exact original lines are preserved as small excerpts
in the audit JSON; no log was changed or supplemented with inferred step fields.

**Verified source rules, not recovered run settings:**

- S-A `configs/shape/syn/bell.yaml:28–35` and the byte-identical S-G file declare
  `key_metric_name: psnr` (line 12), one `val` set, `total_step=300000`, validation/
  save intervals 1000 and train-log interval 20. Both leave `lr_cfg: {}` intact.
- S-A `train/trainer.py:162–185` and S-G `train/trainer.py:157–180` validate when
  `(step+1)` reaches the interval, pass **zero-based `step`** to the evaluator,
  select using the last validation set's key metric, require strict improvement
  `val_para > best_para`, and save best/checkpoint progress as **`step+1`**.
- S-A/S-G `train/train_valid.py:19–53` call the model's existing evaluation mode,
  set `eval=True` and the supplied step, aggregate the existing metrics and return
  the selected key metric. `network/metrics.py:134–135` in A (132–133 in G)
  returns mean PSNR. `train/train_tools.py:103–112` in both formats text values
  to **five decimals**, writes TensorBoard scalars with a step, but appends text
  lines **without step, view ID, commit or configuration identity**.
- S-A `train/trainer.py:201–220` and S-G `train/trainer.py:190–209` show implicit
  legacy `model.pth` resume and payload fields `step`, `best_para`, network and
  optimizer state. They do not save an explicit producing commit/dirty patch,
  split manifest or effective configuration. Resume and append logging prevent
  treating text row numbers as unconditional optimizer-step identities.

**Supported_but_not_proven (H1–H5 and source rules):** 300 val rows, 15,000 train
rows and 300 JPG labels `999,1999,…,299999`, all `index-0`, fit the declared
300000-step/1000-validation/20-log cadence. Under an uninterrupted run with this
configuration, row 150 would correspond to validation call step **149999**,
saved progress **150000**, and montage **149999-index-0.jpg**. The saved step and
rounded metric both fit that interpretation. There is no hash-bound log-to-
checkpoint run record, so this is corroboration, not identification of a run or
proof of its source, view, split, resume history or exact unrounded scalar.
`model.pth` is a separate alternate; its payload was not opened or substituted.

## A/G split and evaluation comparison

| Property | Inspected A | Inspected G | Classification / evidence |
| --- | --- | --- | --- |
| Input ordering | Synthetic loader creates numeric strings `"0"`…`"127"` from the camera count | Same ordering rule | verified source rule; S-A `dataset/database.py:248–273`, corresponding G class, hashes in audit; current order is in H6, historical input identity is not recovered |
| Split RNG | Global Python seed 6033, then one shuffle | Global seed 6033, shuffle commented out | verified; S-A `dataset/database.py:577–590`, S-G `:574–589` |
| Validation | Shuffled position 5: current `"58"` | First unshuffled ID: conditional `"0"` | verified rules, H6 and [prior split check](bell-split-verification.json); H7 explicitly evaluates 0 externally with A tables intact and reproduces the historical metric/montage; training membership still unproven |
| Training / NVS | 122 training, five reserved NVS IDs | 127 training, no NVS partition | verified rules; same sources. `"58"` is G training under the inspected input order |
| Geometry consumer | Historical A renderer still unpacks two values from a three-list helper | G helper and consumer both use two lists | verified; S-A `_init_dataset:278–302`, S-G `:263–287`; current isolated A uses the separately approved explicit shim |
| Geometry validation dispatch | `test_step(index, step)` | AST-identical method | verified; S-A renderer `:511–570`, S-G `:496–555`; this does not imply every renderer/shader calculation is identical |
| Downsampling/chunking defaults | Ratio 0.25, downsample enabled, rays 512 | Same declared defaults | verified source defaults; S-A renderer `:220–230`, S-G `:208–218`; producing effective settings unknown |
| Shape metrics | Full-image uint8 PSNR/SSIM, not a mask-only score | Same compute/shape-metric implementation | verified; S-A/S-G `network/metrics.py:12–18,55–84`; actual producing revision unknown |

Both inspected test-step implementations blur/resize RGB through
`imgs_info_downsample`, scale intrinsics, resize depth/mask with nearest-neighbor
and preserve masks as auxiliary outputs. S-A renderer `:173–186,511–570`, S-G
`:161–174,496–555` and S-A/S-G `utils/base_utils.py` define these operations.
Shape metrics multiply float RGB by 255, clip/truncate to uint8, compute float32
pixel MSE per channel then mean channels for PSNR, and SSIM with `win_size=11`, `channel_axis=2`,
`data_range=255`. Neither inspected shape metric multiplies RGB by the GT mask.
Matching code at these two tips does not establish the producing environment,
image decoding versions, scientific defaults or full algorithm.

**Verified artifact observation (H3, H6):** the historical montage and current R
montage are both 800 × 1000. Their decoded top-left 200 × 200 RGB tiles differ:
mean absolute uint8 difference **66.386**, maximum **206** (Pillow 9.0.1, no
resizing or new outputs). Under the inspected montage layout this tile is GT;
therefore a changed validation target/preprocessing was **supported_but_not_proven**
at 1D.4B-3B. H7 now records actual source-0 getter IDs and reproduces the entire
historical JPEG byte-for-byte, corroborating the researcher's visual camera-0
identification. Producing source/layout history remains unattributed; these
observations do not establish that G produced the checkpoint.

The matching montage dimensions are consistent with quarter-scale 800-pixel
inputs; dimensions alone were not proof of historical 200 × 200 evaluation.
H6 independently establishes **200 × 200**, A view `"58"`; H7 establishes the
same dimensions for explicit camera `"0"` and exact historical montage output.
The original run's effective settings and data versions are still not recovered.

The original image-58 protocol passed stored step **150000** in each arm. Under
the source selection policy the original best validation could have used
**149999**. H7 now tests both steps on camera 0: RGB is bitwise-identical, metrics
and auxiliary outputs agree, with no identity mismatches. This pair supplies no
step-dependent explanation for the gap. The same checkpoint, raw/effective
settings, scientific source, initial RNG and shared runtime/package identities
give the historical score on camera 0 and the higher score on camera 58. Thus
camera selection accounts for the observed discrepancy **in the tested
evaluation implementation**. Historical training membership is a separate
question: the claim that 58 scored better *because it was trained on* is still
unproven. Shared key shapes, YAML or even reproduced output cannot identify the
producing revision, dirty patch or complete training protocol.

H7's native GT-to-historical-JPEG-panel MAE is **3.7261083333333334** uint8 levels
(max 66), with no resizing or compensation. Host Pillow 9.0.1/libjpeg 9.0 decoding
differs from the saved native Pillow 10.4.0/libjpeg 6.2 panel: panel-to-panel MAE
**0.5248083333333333**, max 33; host-decoded GT-to-panel MAE **3.828666666666667**.
The native panel PNG/signed array independently reproduce its reported statistic;
the historical JPEG and both full new montages have the same SHA-256. Keep
decoded-pixel differences distinct from compressed-byte identity and do not
invent a JPEG acceptance threshold. Neither panel is used as the scoring GT.

The inspected training implementations also differ: S-A
`train/trainer.py:155–158` clips gradient norm at **0.7** before Adam's step;
S-G's corresponding backward/optimizer block has no clipping call. This is a
verified source difference, not evidence of which policy produced the checkpoint.
Any historical training reproduction must recover its actual policy; the current-A
bounded test below must preserve current A clipping without reconciling the tips.

## Attribution findings

| Finding / claim | Classification | Specific evidence and limit |
| --- | --- | --- |
| Current A R/R-repeat/C agree exactly | verified | H6 selected report/output hashes and saved-array audit; limited to this view/runtime |
| Stored `best_para` and step are 23.787122565451323 / 150000 | verified | H5 hash-bound metadata, current opaque-byte checkpoint hash recheck in H6; no new payload read |
| H1 row 150 uniquely maximizes the rounded logged PSNR | verified | H1:150; all 300 scalar records scanned |
| Selected checkpoint is the best checkpoint of the run that wrote H1 | supported_but_not_proven | H1:150, H3 label, H5 values and source save policy agree; no run/hash linkage |
| Historical best validation used zero-based step 149999 | supported_but_not_proven | S-A/S-G save/evaluation policy plus H5; actual producing policy and append/resume history unknown |
| Exact producing Git commit, branch, dirty patch or final-version status | unknown | No attributable original launch/run manifest in inspected roots; H4 metadata/time and directory names are insufficient |
| Original effective training YAML/overrides, data version and ordered training split | unknown | A/G YAMLs identical; H1/H2 omit producing identities; matching H7 output does not recover training membership |
| Current explicit camera 0 reproduces historical evaluation evidence | verified for tested checkpoint/renderer | H7 getter IDs, exact stored PSNR, rounded logged SSIM and byte-identical full montage H3 |
| Camera selection accounts for the observed score discrepancy | verified within tested evaluation scope | H6 camera 58 versus H7 camera 0; shared checkpoint/settings/source/runtime, initial RNG; H7 step control has zero difference |
| Historical unshuffled two-way training split, validation 0 / train 1–127 | strongly supported hypothesis, not proven | Researcher visual identification, H3/H7 montage equality and scores, inspected G split; no attributable original ordered training IDs or producing run |
| Historical target versus current A validation 58 | supported by matching camera-0 artifact; not the same target | H3/H7 full JPEG bytes equal; H6 camera-58 GT differs; iterator index 0 alone is still not database identity |
| Stored best metric is the current 32.84983236325549 dB view-58 result | contradictory | H5/H7 camera-0 metric equals 23.787122565451323; H6 view 58 differs by 9.062709797804164 dB |
| A and G have equivalent splits, or A view 58 is G held out | contradictory | Exact helpers and prior split verification; 58 belongs to G training under inspected ordering |
| Successful unmodified A-tip native geometry training through the inspected entry point | contradictory to the inspected interface | S-A two-value consumer versus three-list helper; this does not exclude earlier revisions/local shims or establish G origin |
| Original C completed successfully | contradictory | H6 interrupted report remains `running`, no completion/metric output; admitted successful C is another process/directory |
| SIGHUP caused original C's interruption | supported_but_not_proven | Researcher declaration plus truncated phase logs; no captured signal/exit status |
| Producing event-file scalar/view/step linkage | unknown | H4 opaque hash/metadata only; no scalar export found; TensorBoard/tensorboardX absent in CPU environment; no installation/decoder invention attempted |

Search scope was the known evidence root, repository source/history/accessible
run-record paths and historical results. Five JSON reports under historical
results are PBRNeRF reports, not bell geometry run manifests; no saved YAML,
scalar CSV/JSON export, dirty patch, launch command or hash-bound selection record
was found there. Current `run_all_*.sh` scripts launch **material** configurations;
they are not bell geometry launch evidence. Private home configuration, unrelated
repositories, source checkout staging and network services were not searched.
Earlier inspected split revisions also include first/eighth unshuffled validation
and five/fifteen shuffled NVS policies; their exact commits/file hashes/ranges are
in the audit JSON and [earlier provenance inventory](bell-provenance-investigation.json).
The origin question is not restricted to two branch tips.

## Next milestones and execution gates

1. **Historical evaluation attribution, still needed for thesis claims.** Obtain
   a researcher-attested original run record binding this checkpoint hash to a
   producing commit/dirty patch, original command and effective config, dataset/
   ordered split IDs, validation selection and any resume/continuation history.
   Recover an existing TensorBoard scalar export or propose separately reviewed,
   bounded CPU-only extraction from a read-only event copy. Use actual scalar
   steps/tags to test the row-150 hypothesis; an event scalar alone still does not
   establish the checkpoint/source/view chain. H7 already matches the historical
   montage using camera 0 under maintained A; it does not recover original
   training input versions/membership or settings. Do not substitute a fresh G
   rendering to fill provenance gaps. No new GPU run is necessary for this
   records step.
2. **Small current-A geometry training smoke/regression, separately reviewed.**
   Propose a fresh bounded **two optimizer-update** test, then independent
   R/R-repeat/C repetitions only if the initial native path is viable. Use the
   existing admitted image/GPU class, unique disposable outputs, read-only data/
   artifacts, independent config resolution and the approved A split shim. Start
   fresh; no historical checkpoint resume or optimizer restoration. Preserve the
   original YAML `total_step=300000`, worker/split/RNG order, full data preparation,
   ray/sample counts, loss formulas/coefficients, Adam parameters, clipping and
   hard-coded scheduler. A reviewed external stop boundary must count updates
   without rewriting scientific settings or pretending the shortened run reached
   its endpoint. A two-update smoke cannot test a later coefficient transition,
   convergence or checkpoint-selection behavior. Preserve 1000-step validation/
   save policy: no checkpoint is expected before its natural interval, and no
   forced save/early validation should masquerade as legacy policy. Record finite
   losses/gradients, actual LR, RNG/sampler progress, parameter changes, peak
   resources and partial failure evidence. Freeze numerical comparison policy
   before native execution; do not borrow the RGB tolerance for gradients or
   optimizer state without review. Any scientific/native defect stops the test.
3. **Full training reproduction, later.** Require an approved producing baseline
   or explicitly declared current-A baseline, full data/config/source/environment
   provenance, successful bounded training checks, natural validation/checkpoint
   interval checks and reviewed resource/disk budgets before a separate full-run
   approval. Historical G reproduction additionally requires its exact two-way
   split and producing evidence. Inference parity alone does not authorize or
   predict training equivalence or reproduce a thesis benchmark.
4. **Paper anisotropic-material experiments, separate scientific work.** Admit
   each geometry mesh/checkpoint and material architecture, resolve bear weights/
   dataset/camera/evaluation provenance, preserve effective shader scopes, and
   predeclare hypotheses, seeds, budgets, splits and metrics for controlled
   anisotropic comparisons. Bell's appearance regression neither validates the
   BRDF scientifically nor supplies a bear material baseline. No material/GPU
   experiment, full training or scientific repair is implemented here.

The smallest immediate action is review of the camera-0 success and recovery of
missing historical training/run/scalar records. The separately prepared bounded
current-A training test remains a distinct review item; this evidence
finalization authorizes no new native execution or historical training claim.

## Future paper split labels

| Purpose | Suggested explicit label | Reporting rule |
| --- | --- | --- |
| Historical bell checkpoint evaluation correspondence | `bell-historical-checkpoint-camera0-v1` | State checkpoint hash, source camera 0, evaluation protocol/step and reproduced score/montage. Historical ordered training split remains an unproven hypothesis; do not imply newly trained thesis reproduction or verified holdout status. |
| Maintained A infrastructure regression | `bell-current-A-threeway-seed6033-v1` | State 122 training IDs, validation 58, NVS `[11,105,114,17,76]`, full ordered ID manifest/hash and exact source revision. This is a current-code split, **not the historical geometry evaluation split**. Its labels do not prove holdout status for loaded historical weights. |
| Newly designed paper experiments | `<dataset>-<scene>-paper-split-vN` | Freeze ordered train/validation/test IDs, source image identities, algorithm/seed, preprocessing, metric/mask policy and a manifest SHA-256 before training. Apply the same declared policy across controlled comparisons; use validation for selection and a separately declared test set for reporting. |

Keep historical evaluation targets, historical training membership, current-code
compatibility partitions and newly designed paper partitions as separate fields.
Any two-way historical-style experiment must explicitly select an approved,
versioned protocol; it cannot obtain that status by relabeling current A's
three-way split. No split implementation or configuration is changed here.

Use “held out” only when producing-run evidence or a new run's saved ordered
training IDs establish exclusion. For this historical checkpoint, use “camera-0
historical evaluation correspondence” and “camera-58 current-code regression.”
The camera-0 diagnosis was selected to investigate an existing historical
artifact; it is not a newly predeclared unbiased paper benchmark. Never choose
evaluation IDs or alter split seeds after inspecting scores to improve a table.

CPU evidence verification and limitations are recorded in the
[success summary](BELL_CAMERA0_DIAGNOSTIC_SUCCESS.md#validation) and manifest;
original 1D.4B-3B machine evidence and all image-58 reports remain unchanged.
