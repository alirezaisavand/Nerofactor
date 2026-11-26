# Gate 10.2 v2 — Serialization correction and saved-extraction continuation

Decision: **FIX AND PROCEED**, pending one new exact-source execution approval.

The authorized v1 run completed its single raw marching-cubes call and saved
671,550 float64 vertices, 1,338,864 uint64 faces, the exact transformed world
coordinates and the supported-cell mask. It then failed at the first NPZ write:
five callers incorrectly supplied keywords to the existing `save(path, arrays)`
helper. No native ray audit, checkpoint load or SDF consistency query occurred.
The host exit code is 1 and the owned container is absent. Native receipt cleanup
is **unverified** (`false`); the host removal receipt does not retroactively
change that field. All original files, authorization and freeze remain unchanged.

The [failure manifest](../results/publication_gate102/failure-v1/artifact-manifest.json)
pins every saved artifact. The numerical coordinate transformation of saved
voxel to world vertices replays exactly. The saved raw extraction is a reusable
intermediate, not admitted geometry or a successful quality evaluation.

## Narrow correction

Convert the five incorrect calls to `save(path, dict(...))`. Keep the existing
writer unchanged. Focused CPU tests evaluate all seven actual production save
expressions through that real writer and verify exact keys, values, dtypes and
refusal to overwrite. No scientific operation is needed for this regression.

Do not repeat marching cubes. Read the three hash-bound raw array files from
the failed v1 run through a read-only mount. Require their original shapes,
dtypes, finite coordinates, valid face indices and exact `voxel/511*2-1`
coordinate relation. Copy them to the fresh output as provenance-bearing inputs,
then continue with the same domain classification, cleanup and audit. A changed
or missing capture stops the run; there is no extraction fallback.

The only processing correction is the NPZ call interface. Reusing the completed
extraction replaces a repeated deterministic call; it changes no coordinates,
faces, winding, raw SDF values, domain mask or scientific arithmetic. Native
and offline verification require exactly zero new MC attempts/returns and exact
array agreement with the original capture.

## Unchanged science and limits

All domain handling, float32 exact-equality welding, exact-zero-only removal,
open-boundary/component preservation, normal conventions, quality thresholds,
camera rays and evaluation definitions remain those in the
[v1 protocol](BELL_DOMAIN_AWARE_EXTRACTION_GATE102_V1.md). Raw grid, domain mask,
accepted endpoint, original failed mesh, oracle archives, corrected image/BVH
and authoritative G source identities remain identical. Ground truth remains
evaluation-only. No failed or difficult observation is removed from coverage.

The v2 freeze binds the failed v1 receipt and three completed extraction arrays,
new source and fresh output. The only reduced work limit is **zero MC calls**.
Remaining ceilings are unchanged: 196,608 BVH rays, 384 independent CPU ray
checks, one isolated CPU load of the accepted step-300000 checkpoint
`27c8c2256d529a84680e1ab806081acfa796a04937a2681587a8bb6d1a3681cd`,
1,536 raw-SDF point evaluations and 512 first-derivative points. The load occurs
only for the original consistency audit. Zero shader calls or optimizer updates.

Use the same corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A100 GPU0, four CPU threads, 128 GiB RAM/no swap, 128 PIDs, 75% CUDA allocator,
8 GiB outputs, 1,800-second foreground watchdog/30-second grace and 1,740-second
inner limit. Estimated 1–10 minutes, still unmeasured for the remaining audit.
Read-only source/inputs/root, non-root, no network/build/install. Stop on
mandatory failure without retry, fallback, policy/threshold change or training.

One new approval covers the cached-extraction continuation and original audit,
single bounded checkpoint load, file-only analysis, figures and result commit/push.
The previous execution approval is consumed. No new scientific execution or
authorization file is created by this correction. After approval, use the same
foreground launcher with the v2 freeze and its fresh authorization path.

The verifier preserves all original quality gates and full-raster accounting.
Only a complete passing audit can produce a handoff candidate; researcher
geometry admission remains separate. Failure to meet geometry quality is
reported as a completed negative result, without repair or material training.
