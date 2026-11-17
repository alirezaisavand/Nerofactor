# Donut D1/D2 recovery v2: explicit HDR illumination preview

This is a display/export correction to the existing matched
[D1/D2 protocol](DONUT_KURT_RECOVERY_V1.md), not a new scientific experiment.
**No corrected native execution is authorized or performed by preparation.**

## Observed failure and preservation

The first training batch is preserved at
`/home/isavand/paper/evidence/donut-kurt-recovery-v1-1cd964f4026f4711bed5b5dd83f7f863`.
D1's terminal report records failure during the initial illumination PNG export,
zero Adam returns, and no completed evaluation endpoint. D2 never started;
no training log or checkpoint exists. D1 constructed the native shader, passed
the native/reference gate and performed 408 shader calls / 279,257,088 shadow
rays. Its cleanup flag is false; successful cleanup is not independently attested
for this failed run. No Docker inspection or cleanup command was attempted here.

The [154-file identity manifest](../evidence/donut_training_preview_v2/failed-batch-identities.json)
preserves exact identities of the saved evidence, including the old authorization,
freeze, console, initial-state identity, gate and numerical arrays. It does not
rewrite or reinterpret the failed terminal status.

The saved illumination map is finite, nonnegative float32, range 0.01–3.4031355,
with 13,952 channels above one. D1's map equals its analytic known-light map
exactly. Radiance above one is valid. The strict reference PNG encoder rejects
such inputs by design; the adapter used it directly for an HDR diagnostic.

## Minimal correction

Keep `illumination.npz`, all scientific linear radiance, spherical statistics,
illumination errors, BRDF inputs, losses and gradients unchanged. Before the PNG
encoder only, apply explicit `min(linear_RGB, 1)` and encode standard-sRGB uint8
at exposure one. Record upper-clipped channel/pixel counts and the display rule
in `metrics.json` under `illumination.preview`. Negative or nonfinite input still
fails; no automatic exposure, tone mapping or modification of the reference
encoder is introduced. The PNG is a clipped visualization, never an HDR target.

The separate training source freeze changes because the adapter changed. Its
`reference_freeze_sha256` explicitly binds the original admitted generation
freeze `1c0af2688f36bef6e4da7e15666e0245a76c9a11294a3f719bfe5601c53b7730`.
The original generation, dataset admission, source freezes and reports remain
unchanged. The reference/training hash comparison now checks this explicit
binding; legacy definitions without it retain exact same-freeze checking.
The launcher accepts only the additional literal `DONUT_KURT_RECOVERY_V2.freeze.json`.

All physical settings, splits, corrected-core and frozen A/G sources, seed,
initialization policy, priors, optimizer/scheduler, sampling, evaluation draws,
10000 updates per arm, intervals, compatibility bounds and resource/work ceilings
are identical to the previously approved definition. No new reference rendering,
model selection, training arms, historical checkpoint loading or resume.

## Focused validation and next launch

The synthetic CPU regression covers the actual illumination preview helper at
128x256, out-of-range HDR, byte-preserved raw NPZ, unchanged in-domain encoding,
pixel/channel ordering, invalid values/shapes, evaluation wiring, unchanged
physical settings/scientific source hashes, explicit reference binding and
mismatch rejection. Scientific imports are blocked. Saved JUnit/log/exit evidence:
`docs/evidence/donut_training_preview_v2/`; 15 passed, zero failures/errors/skips.
No old scientific experiments or historical test suites were repeated.

Companion freeze: `DONUT_KURT_RECOVERY_V2.freeze.json`. Fresh reserved batch:
`/home/isavand/paper/evidence/donut-kurt-recovery-v2-4c47339a0d204bc0abe8480be2325b63`.
This directory is not allocated by preparation. The consumed v1 authorization
does not authorize it. Obtain one new exact commit/freeze/dataset/output approval;
then use the existing foreground launcher with `DONUT_PHASE=training` and
`DONUT_FREEZE_NAME=DONUT_KURT_RECOVERY_V2.freeze.json`. Preserve failures and stop
without retry, tuning, extension or reuse of the failed output.

Dataset generation SHA-256 remains
`2052306bdf87a0beaa0a3f9b0bddeeb12d787f1d65357d520b5ce2fe0a33ee64`;
admission SHA-256 remains
`02e4f7ca909de6b7c2c9fa2a2e291850a9bff190b426e319daf82e3ecdb82c9e`.
Same corrected image:
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`.
