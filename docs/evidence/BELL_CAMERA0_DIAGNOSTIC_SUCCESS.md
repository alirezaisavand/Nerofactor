# Bell camera-0 historical evaluation evidence: 1D.4B-3D

The maintained renderer reproduces the admitted checkpoint's exact stored PSNR
on **source camera `"0"`**: **23.787122565451323 dB**. SSIM is
**0.8772724776097482**, which rounds to historical logged **0.87727**. Independent
steps 149999 and 150000 produce bitwise-identical RGB and identical metrics.
Both newly generated complete validation montages are also **byte-identical to
the historical `149999-index-0.jpg`** (58,376 bytes).

These are independently inspected files from the researcher's successful native
execution, not a new Codex GPU run. The [versioned success manifest](bell-camera0-diagnostic-success.json)
records identities, checks, original paths and limitations. The frozen
[diagnostic protocol](../protocols/BELL_CAMERA0_DIAGNOSTIC_V1.md) and original
reports remain unchanged. The producing revision/configuration and historical
training membership remain unresolved.

## Original evidence and identity checks

Paths are relative to `/home/isavand/paper/evidence`. There is one discovered
`bell-camera0.*/comparison/step-comparison.json`. Primary/control were selected
using that report's recorded hashes and sizes, not a search for favorable scores.
The comparison's own hash was calculated during this audit; no separately signed
or researcher-supplied comparison digest was available.

| Record | Original path | SHA-256 |
| --- | --- | --- |
| Primary, source 0 / step 149999 | `bell-camera0.Ri4zwVXG/primary/camera-diagnostic.json` | `8fec3209db18b997bae261ee81a8d0be8fa1ed6fcfbdd534d8351acdcfc17793` |
| Control, source 0 / step 150000 | `bell-camera0.Ri4zwVXG/step-control/camera-diagnostic.json` | `3f3215e7df64a489de1c3864ae91fb3ab427af2f1cbce6bf86525bb4dfee2c33` |
| Step comparison | `bell-camera0.Ri4zwVXG/comparison/step-comparison.json` | `007b56ff2a518074e4049cdb2f01fa61b82780e9b64d5d342908cff7cd89b673` |

The checkpoint remains `thesis:model/bell_shape/model_best.pth`, 29,966,819 bytes,
SHA-256 `d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`.
Opaque hashing confirmed it; no checkpoint or camera pickle was deserialized.
Stored progress 150000 and `best_para` come from the previously admitted metadata.

The audit verified all **24 declared output hashes**, lossless finite float32
RGB/GT `[200,200,3]`, int32 masks `[200,200,1]`, original PNG quantization,
metrics JSON and saved full difference arrays. Source getters explicitly record
`"0"` for image, pose, K, mask and depth; diagnostic iterator index 0 maps to
that source ID. The real A training/validation/NVS tables remain unchanged,
including validation `["58"]`. Reports record strict architecture/weight loading,
137 unchanged tensor fingerprints and RNG-neutral camera selection. These checks
were inspected as saved evidence, not rerun against a model.

All **376 used input files / 111,584,129 bytes** match recorded hashes, and
**641 scene metadata entries** match size/mtime/ctime. The four diagnostic source/
protocol hashes and 57 pinned reference source/configuration/asset hashes match
both the checkout and declared execution commit
`75b8d5dc9dc3441b163f0c3a73a961bdc12236ab`. The earlier nine image-58 implementation
hashes and six selected original report identities/output manifests remain valid.
No original report, output, cache or input was regenerated or modified.

Launch metadata, researcher launch script/log, outer terminal logs and all three
zero exit-code records are hash-inventoried. Reports declare immutable image
`sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137`,
Python 3.8.10, Torch 1.8.1+cu111, CUDA 11.1, A100 80GB PCIe, driver 550.163.01
and C++ ABI 0. These are file-backed declarations/observations; this audit did
not query Docker, attest a full clean launch tree or recover the complete native
binary/build closure. Original execution timestamps remain explicitly UTC.

## Numerical findings and interpretation

| Observation | Result | Interpretation |
| --- | --- | --- |
| Camera-0 PSNR minus stored best | **0.0 dB** | Exact equality to admitted full-precision checkpoint field |
| Camera-0 PSNR / SSIM displays | `23.78712` / `0.87727` | Matches historical log row 150 at its five-decimal precision |
| 149999 versus 150000 | Maximum/mean RGB difference **0.0**, no identity mismatches, auxiliary outputs identical | `within_frozen_step_comparison_bounds`; all original thresholds retained |
| Whole validation JPEGs versus historical JPEG | Both byte-identical; SHA `2800003a665d47f757b1ec30b98d225bcfe19751b2608f1caa75587f8bcb3a44` | Exact artifact correspondence, stronger than score proximity alone |
| Native GT versus decoded historical GT panel | MAE **3.7261083333333334** uint8 levels; max **66**, RMSE **5.6172806291538135** | Lossy JPEG comparison, not an acceptance threshold or scoring GT |
| Unchanged image-58 result | PSNR **32.84983236325549**, SSIM **0.946790605392639** | Separate current-A regression; PSNR exceeds camera 0 by **9.062709797804164 dB** |

**Camera selection accounts for the observed metric discrepancy under the tested
checkpoint and evaluation implementation.** Raw/effective configurations,
checkpoint, scientific source, initial RNG and shared reported runtime/package
identities agree. Camera 0 recovers the historical metric/artifact; camera 58
remains the higher-scoring different target. The step control rules out a step
149999/150000 RGB difference in this pair. Post-evaluation RNG differs between
cameras; no claim of identical view-dependent RNG consumption is made.

This does **not** prove that camera 58 was a historical training view or that
training membership caused its higher score. The unshuffled two-way hypothesis
(training IDs `1`–`127`, validation `0`) is now **strongly supported**, conditional
on the inspected ordering/algorithm, but producing-run attribution remains
unproven. Other implementations or split policies could evaluate camera 0 too.
No original unrounded historical SSIM was recovered. Neither score is promoted
to a new historically verified held-out/NVS benchmark.

## JPEG decoding and CPU consistency limits

The saved native decoded panel uses Pillow **10.4.0 / libjpeg 6.2**. Its PNG and
signed GT-difference array reproduce the native MAE **3.7261083333333334** exactly
on CPU. A fresh decode here with Pillow **9.0.1 / libjpeg 9.0**, without libjpeg-turbo,
differs from that panel (MAE **0.5248083333333333**, max **33**); comparison to this
host-decoded panel gives GT MAE **3.828666666666667**. Both measurements are
recorded separately. The original JPEG bytes match exactly in all three files.
No resizing, alignment, recompression, denoising, decoder substitution or tolerance
adjustment was used to make a comparison pass.

Saved-array PSNR/SSIM recomputation matches the native reports exactly with CPU
Python 3.9.12 / NumPy 1.21.5 / scikit-image 0.19.2. This is a consistency check
on these arrays, not general metric/decoder equivalence across versions. The
version-guarded native comparator was neither relaunched nor patched/bypassed;
an independent read-only CPU audit checked its recorded diagnostics.

## Future paper labels and remaining gates

Use distinct protocol/split labels; see the expanded recommendations in the
[attribution report](BELL_EVALUATION_PROVENANCE.md#future-paper-split-labels).
Historical camera-0 evaluation correspondence is established for this checkpoint;
historical training membership is still conditional. Current A's seeded three-way
split, validation 58, **must not be called the historical geometry evaluation
split**. A source-defined split cannot retrospectively establish that a loaded
historical checkpoint was trained without a given evaluation image.

The smallest next action is to obtain a researcher-attested producing-run record
binding checkpoint hash, original source/patch, effective training configuration,
ordered split IDs and resume history. Any paper training experiment must freeze
its own versioned train/validation/test manifest before training. This follow-up
does not authorize a new evaluation, training run or split change.

## Validation

The complete CPU infrastructure suite passed **194 tests, 0 failed, 3 skipped**
(197 total). Skips remain native OpenCV/transforms3d crop, Open3D point-cloud and
OpenCV RAW-resize parity. Static checks passed Python 3.8 grammar for **85 files**,
five Docker RUN shell parses, definition/isolation policy, JSON/documentation
links, documented Bash syntax, Git whitespace and documentation-only scope.
The existing static artifact-readiness warning is not a fresh assessment of the
working image. No Docker engine, GPU inference, training, installation or package/
image changes occurred. Nothing was committed or pushed.

An optional identity-only recheck reads original files without generating output:

```bash
python3 -B - ../evidence <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
m = json.loads(Path('docs/evidence/bell-camera0-diagnostic-success.json').read_text())
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()
for item in [*m['reports'].values(), m['comparison']]:
    path = root / item['path']
    assert path.resolve() == path
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    report = json.loads(path.read_text())
    for name, expected in report['output_hashes'].items():
        artifact = path.parent / name
        assert artifact.resolve() == artifact
        artifact.relative_to(path.parent)
        assert digest(artifact) == expected
print('PASS: selected camera-0 report and output identities')
PY
```

This snippet checks byte identities, not all source/input/array/metric/provenance
relationships audited above. Source and evidence files remain read-only inputs.
