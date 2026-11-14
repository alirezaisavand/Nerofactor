# Interrupted E.3 state-100 continuation: saved-data results

**The single scientific run failed; E.3 is incomplete.** Six arms reached state
100, one stopped at state 21, and three stopped after state-10 parity only.
Do not treat absent endpoints as failed physical-recovery measurements or
compare the partial panel as five complete LR pairs.

See the [scientific report](../../NEURAL_MATERIAL_CONTINUATION_E3_RESULTS.md)
and [frozen protocol](../../protocols/NEURAL_MATERIAL_CONTINUATION_E3_V1.md).

- `comparison.json` and `endpoints.csv`: all ten planned outcomes, state-10
  versus state-100 metrics where observed, full original success checks and
  physical/component errors. Missing endpoints are null/blank, never imputed.
- `trajectories.csv`: 560 recorded fitting states, including three unused
  parity measurements. The failed state's last completed rendering/backward
  at state 21 was counted but its physical diagnostics were not archived.
- `gradient_routes.csv`: 2,800 saved head/state records; actual gradients and
  exact sigmoid endpoints, without additional model queries.
- `adam_displacements.csv`: 551 actual float32 transitions, group movement
  summaries and recorded current gradients. Full per-step weights/moments/XORs
  are not retained; do not claim reconstruction of every complete transition.
- `integrity.json`: offline hash-chain, numeric snapshot/probe and input/source
  checks. These do not replace the interrupted runtime's unverified cleanup or
  terminal checks.
- `evidence_manifest.json`: all 125 original raw files, absolute paths, sizes
  and SHA-256 hashes, plus authorization, launch, console and process receipt.
  Raw storage is 388,436,575 bytes, outside Git.
- `package_manifest.json`: original/copy equality for these compact files and
  the three figures in PNG and vector PDF formats; offline analysis identity
  and the preserved first offline analysis failure.

Original run:
`/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae`.
Completed offline analysis:
`/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae-analysis-v2`.
The first offline analysis failed because JSON key sorting changed the order
of a floating-point gradient-norm reduction. The corrected analyzer explicitly
reproduces the frozen model's original parameter-group summation order; no
scientific tolerance changed. Its failure record remains in `-analysis-v1`.
This was not another scientific launch.

To independently regenerate compact analysis and figures, with original read-only
evidence present, select a **fresh** external output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 \
  docs/results/neural_material_continuation_e3/analyze_saved_e3.py \
  --run /home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae \
  --output-dir /home/isavand/paper/evidence/neural-e3-offline-reproduction-UNIQUE
```

The analyzer pins the original report/seal/receipt/freeze identities, requires
the recorded failed status, and blocks all Torch/scientific experiment imports.
It reads numerical archives with `allow_pickle=False`. It never constructs N1,
calls a shader, computes a new radiance reference, changes weights or resumes
optimization. Plot-cache files remain in the external analysis output.
