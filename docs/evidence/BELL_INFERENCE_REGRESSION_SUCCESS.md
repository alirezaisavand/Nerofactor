# Bell inference regression evidence: 1D.4B-3B

**Verified from preserved files: `regression_passed`.** R versus R-repeat and R
versus the admitted C have identical saved float RGB, GT and mask arrays at
**200 × 200**. Maximum absolute RGB error, PSNR difference and SSIM difference
are all **0.0**; identity mismatches are empty. All arms report PSNR
**32.84983236325549 dB** and SSIM **0.946790605392639**.

This is **current-code A-based one-view routing regression**, not historical
thesis reproduction or a historically verified held-out/NVS benchmark. The
checkpoint's producing revision/configuration/split remain unknown; view `"58"`
is a training view under inspected G. See the separate
[historical evaluation audit](BELL_EVALUATION_PROVENANCE.md).

Review baseline: `a72db40145b814943f4ce04ad46b7ea0b632bf2b`; the branch was
`paper/infrastructure` and clean before documentation edits. No native experiment,
checkpoint deserialization or scientific/configuration change occurred here.

## Explicit evidence selection

The [compact machine-readable manifest](bell-inference-regression-success.json)
records original identities, source/input/runtime summaries, output hashes,
canonical identity hashes and all **21** discovered inference/comparison reports.
Paths below and in the manifest are **relative to the approved evidence root**;
logical dataset/thesis roots and container paths avoid publishing personal host
paths. Original files are not copied, rewritten, regenerated or relocated.
Selection uses the four supplied hashes, not directory recency or scores.

| Evidence | Original relative path | SHA-256 |
| --- | --- | --- |
| R, complete | `bell-regression.tjRWmKTp/r/inference.json` | `d5a7f4a08fbdca90cf2cf6d464236c2e0b571838538c35d7856d2b84dff8e832` |
| R-repeat, complete | `bell-regression.tjRWmKTp/r-repeat/inference.json` | `16d49548d52bddee244737fbaab155b2dd9d9784471744bb24ef01a6fe9299c5` |
| Original C, incomplete | `bell-regression.tjRWmKTp/c/inference.json` | `9049b2785cc5609ebef8daeb2f555a7916d3da2b3b6224b20a3781de4aec7594` |
| Fresh C, complete | `bell-c-resume.41v84l15/c/inference.json` | `ca7b8f09243b9c0df73161df520b92d88b0b5eacd504e44b7f6748afae5ea261` |
| Final comparison | `bell-c-resume.41v84l15/final/regression-comparison.json` | `5b716048066a790e5380696a16a354f80074d2da388aa49e98675388a2754c99` |
| Original repeatability gate | `bell-regression.tjRWmKTp/repeatability/regression-comparison.json` | `480ceb62ecc728ec286d08b55c21abd68d1b8482627381b84ea50a1d3418b222` |

The original R/R-repeat completed at 02:56:01 / 02:56:51 UTC on 2026-11-06.
Original C began at 02:56:53; its report remains `running` at
`dataset_model_initialization`, without completion, metrics or final output
manifest. Its console contains only the first two phase records. The researcher
reports external **SIGHUP**; that signal/cause is **supported_but_not_proven**
without a saved signal/exit record. The incomplete record is never a success.

Admitted fresh C ran 03:52:00–03:52:47 UTC with a different unique run ID and
output directory. Despite the directory label `c-resume`, this is fresh C-only
inference from the explicitly selected historical network weights, **not**
same-run continuation or optimizer/RNG-state restoration. The final comparison
binds this C to the unchanged original R/R-repeat identities and rechecks their
repeatability. Its terminal log says `regression_passed`; the JSON and arrays
provide stronger file-backed evidence than that line alone.

Five other complete C reports are present (four other `bell-c-resume.*` runs and
one `bell-regression.xgvNp0tt` run), plus other R/R-repeat attempts. The inventory
preserves their distinct hashes/states and verifies declared artifact hashes;
it does not substitute them into this admission. Reasons for additional completed
attempts remain unknown. The supplied-hash comparison is the admitted record;
this is not a claim about an uninterrupted first attempt or absence of failures.

## What was independently checked

- All selected report hashes/sizes, completed arm/schema/route/view-count/step
  identities, strict structural evidence and final arm linkage match. Original
  report hashes remained unchanged throughout inspection.
- Every declared output hash in all 21 discovered reports was checked, when a
  manifest existed. For the admitted three arms, seven output hashes are also
  identical across arms: float prediction/GT/mask NPYs, lossless prediction/GT
  PNGs, the diagnostic montage JPG and metrics JSON.
- Admitted NPYs load with `allow_pickle=False`: prediction/GT are finite float32
  `[200,200,3]`, mask is int32 `[200,200,1]`. Decoded PNGs are `[200,200,3]` and
  exactly match the original multiply/clip/truncate RGB conversion.
- Existing CPU pair-comparison calculations reproduce **all** final diagnostics,
  including identity checks and zero errors. Both saved signed-difference arrays
  are float64 zeros and both violation masks are boolean false everywhere;
  their hashes, dtypes, shapes and values were checked. The original repeatability
  report agrees with the final report's repeatability section.
- Checkpoint `thesis:model/bell_shape/model_best.pth` was opaque-byte hashed again:
  `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`,
  **29,966,819 bytes**. No checkpoint was deserialized. Stored step **150000**
  and `best_para` **23.787122565451323** come from the previously admitted,
  hash-bound metadata report, not newly read payloads.
- The 375 used camera/RGB/mask/depth files (**111,513,661 bytes**) match recorded
  content hashes; all 641 scene source metadata entries match size/mtime/ctime.
  Camera pickles were only hashed. This is not a full dataset-family hash audit.
- Full ordered A split equals the protocol's seeded three-way lists: 122 training
  IDs, validation `["58"]`, NVS `["11","105","114","17","76"]`. Complete
  lists remain in each original report; a canonical JSON hash is in the compact
  manifest. Pre/post inference RNG identities each match across the arms;
  inference does consume RNG, so before and after are not claimed identical.
- Raw YAML and C envelope hashes match the checkout and Git baseline. Effective
  trainer/renderer/shader settings and preprocessing identities match across all
  three arms. All nine recorded adapter/config/training-module hashes match Git
  at the declared inference commit, and the earlier reference's pinned source/
  config/asset hashes also match. Full launch clean-tree state is not attested.

No supplied numerical or identity discrepancy was found. The extra attempts,
incomplete statuses and limited signal evidence are retained as qualifications.

## Runtime and numerical limits

The reports declare source commit
`a72db40145b814943f4ce04ad46b7ea0b632bf2b` and immutable image ID
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`
(tag annotation `nerofactor-paper:quick-candidate-1`). Runtime fields agree:
Python 3.8.10, Torch 1.8.1+cu111, CUDA 11.1, A100 80GB PCIe, driver 550.163.01,
C++ ABI 0; native package/backend/thread settings are preserved in the manifest.
These are independently inspected report fields, not fresh Docker/GPU queries
or a complete build/launch attestation. Native-library/extension/JIT binary
closure and the image build chain remain incomplete.

The previously frozen thresholds remain unchanged: float error
`<= 1e-4 + 1e-4*abs(R)` elementwise, uint8 channel error <=1, absolute PSNR delta
<=0.01 dB, absolute SSIM delta <=1e-4. Both pairs pass with zero actual errors;
this does not prove general bitwise determinism or training/resume equivalence.

CPU verification used Python 3.9.12 / NumPy 1.21.5 / scikit-image 0.19.2 / Pillow
9.0.1. Recomputing the existing saved-array metrics produces the same numbers on
this host. This is an additional consistency check, **not** general cross-version
metric equivalence: the native reports use NumPy 1.24.4 / scikit-image 0.21.0.
The version-guarded native comparator was not relaunched or patched to bypass
that guard. No new view/model execution was performed.

## Current acceptance and next gate

The admitted A inference routing regression is complete for this checkpoint,
view, implementation and runtime. Historical producing branch/split attribution
and the reason for the stored/current metric difference remain unresolved.
**32.84983236325549 dB is not a historically verified thesis benchmark score.**

Review the [historical attribution findings and next milestones](BELL_EVALUATION_PROVENANCE.md#next-milestones-and-execution-gates).
Additional historical-record recovery and a separately reviewed bounded current-A
training test are distinct activities. Full training reproduction and paper
material experiments are not authorized by this evidence finalization.

## Validation performed for this documentation follow-up

- Read-only audit verified 21 report identities and every declared artifact hash;
  admitted output arrays/PNG dimensions/conversion, full zero-difference arrays,
  frozen thresholds, source/config/used-input identities and numerical report
  fields were independently checked. No supplied evidence discrepancy was found.
- Complete CPU infrastructure suite: **155 tests; 152 passed, 0 failed, 3 skipped**
  under Python 3.9.12. Existing skips are native OpenCV/transforms3d crop parity,
  native Open3D downsample/export parity and native OpenCV RAW resize parity.
  They are unavailable-backend limitations, not native passes. No historical
  checkpoint or scientific model was used by these disposable CPU fixtures.
- Existing Docker static checks passed: definition/isolation policy, five RUN
  shell parses and Python 3.8 grammar for 72 files. No Docker engine was invoked;
  the legacy artifact-readiness warning is not a new assessment of the working
  image. Full current-suite Python 3.8 runtime execution was not performed here.
- JSON and local Markdown links/anchors, documented Bash syntax, whitespace,
  source-line/hash references, preservation of earlier machine evidence and the
  documentation-only changed-file scope were checked. No commit/push was made.

The unchanged CPU/static checks can be rerun from the repository root:

```bash
MPLCONFIGDIR=/tmp/nerofactor-bell-evidence-matplotlib python3 -B -m unittest discover -s tests/infrastructure -v
python3 -B docker/paper/static_checks.py
```

A small independent identity check (no Torch, models or checkpoint input) is:

```bash
python3 -B - ../evidence <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
m = json.loads(Path('docs/evidence/bell-inference-regression-success.json').read_text())
def digest(path):
    with path.open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()
for item in [*m['reports'].values(), m['final_comparison'], m['original_repeatability']]:
    path = root / item['path']
    path.resolve().relative_to(root)
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    report = json.loads(path.read_text())
    for name, expected in report['output_hashes'].items():
        artifact = (path.parent / name).resolve()
        artifact.relative_to(path.parent.resolve())
        assert digest(artifact) == expected
print('PASS: selected report and declared output byte identities')
PY
```

This convenience identity snippet does not replace the stronger saved-array,
source and input verification summarized above. Every external
report/artifact remains an original read-only input to these checks.
