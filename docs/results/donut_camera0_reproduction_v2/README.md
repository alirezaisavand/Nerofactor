# Donut camera0 results

See [scientific findings](../../DONUT_CAMERA0_REPRODUCTION_RESULTS_V1.md).
`camera0-comparison.png` is an8-bit contact sheet, not the metric input.
`comparison.json` contains the original full-precision metrics.
Receipts, authorization and console are exact small copies of native evidence.
`artifact-manifest.json` identifies originals, copied files, frozen inputs,
sources and the preserved failed v1 launch. Raw linear EXR,16-bit PNGs and
numerical archives remain in the external evidence root recorded there.

To reverify saved evidence and collect another small results package, select a
fresh destination under this checkout's `docs/results`. This command performs
only hashing, array/metric verification and file copying; it never starts Blender
or renders an image:

```bash
cd /home/isavand/paper/Nerofactor-donut-reproduction
python3 -B -m experiments.finalize_donut_camera0 \
  --evidence /home/isavand/paper/evidence/donut-camera0-reproduction-v2-cacfd58bc62d44aa84141984ae5e16ea \
  --destination "$PWD/docs/results/donut_camera0_reverification"
```

The destination must not exist. Neither this command nor the native launch
should be rerun to overwrite evidence. Figure generation and metric definitions
remain in `experiments/donut_camera0_reproduction.py`; its native entry point
requires separate authorization and a fresh run, and is not needed for file-only
verification. All same-render export checks passed without a second render.
