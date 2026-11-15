# E.11C v2 — native backend evidence accepted

**PROCEED to a separately authorized Bell generation.** This finalization used
only existing researcher-produced GPU captures. No extension rebuild, CUDA
execution, new rendering, optimizer update or training was performed by this task.
The [versioned report](evidence/RAY_BACKEND_E11C_ACCEPTANCE_V2.json) records the
source/authorization/build/image/artifact chain and every permitted replay delta.
Earlier reports/receipts, authorizations, captures and failed analyses are retained.

## Replay correction, not a scientific tolerance change

The previous offline comparator required whole-row dictionary equality. Replaying
float32 normal/view dot products under the host NumPy arithmetic differs from
native-report assembly in5 original and4 corrected diagnostics by exactly
5.960464477539063e−8. All9 are solely `facing_dot`; scientific pass/fail flags,
hit/miss classifications, identities, counts, depths, errors and every other
compared field are exact matches.

The new file-only comparator permits **absolute2⁻²⁴**(float32 unit roundoff,
eps32/2) for finite `facing_dot` values only, at Bell/cube case-row paths. At
these observed magnitudes this is one binary32 ULP. There is no relative tolerance,
whole-row allclose, clamping, substituted reference value or native recomputation.
Missing/extra fields, different types, nonfinite values and differences elsewhere
fail. Native `compare()`, build(), probe(), scientific TOL and stored acceptance
flags are unchanged; AST/source regressions enforce that separation.

## Verified native outcomes

| Evidence | Original backend | Corrected predicate backend |
|---|---|---|
|Native status|pass: historical failure reproduced|pass: independent references matched|
|Independent case agreement|42/44; known errors preserved|44/44|
|Camera1/ray13621 depth|3.6032063961029053|2.751337766647339|
|Known-ray depth error|.8518686463544203|1.6898853871794017e−8|
|Known-ray normal component error|1.3590459617529222|2.0607917705639522e−8|
|Permitted facing replay deltas|5|4|
|Other replay mismatches|0|0|
|Returned trace calls/ray queries|27 /819244|27 /819244|
|Cleanup receipt|verified|verified|

Corrected known-ray facing is+.9585404992103577; position error2.9802322387695312e−8.
The primitive ID is inferred from independent closest position/depth and triangle
normal compatibility; the extension does not expose original PLY IDs. Normal and
visibility controls include legitimate inside-exit rays, without weakening the
outside-camera front-facing generation gate.

All native reports match their original completion hashes. Captured Bell/cube/
timing NPZs, all28 staged inputs, source Git objects at executed commit
`6af0474dc0167a64ad1dfedcc3b1f5f4dc049ca2`, authorization/freeze, build receipt,
compiler records, COPY-context source/patch/binary and base-layer ancestry were
inspected. No standalone failed-analysis log/report was found under the original
root; its strict-equality failure is independently demonstrated by the retained
nine value pairs. It is not rewritten or retroactively marked successful.

Recorded immutable corrected image:
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`.
Corrected binary SHA256:
`800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.
Historical binary SHA256 remains
`9cff02fdd0752c17a613874c0f71e47803a666234e5e5046cf194c5646102180`.
Image identity comes from researcher-captured immutable ID/layers and the inspected
COPY context; no Docker daemon reinspection was performed in this offline task.
Large binaries and raw captures remain under the original evidence root outside Git.

## Performance and limits

Across all20 predeclared timed batches(no outlier removal), synchronized API wall
median is .6527630612ms original versus .6877304986ms corrected: **1.05356835×,
about5.36% slower**. CUDA-event medians are .6369280219ms and .6712319851ms,
ratio1.05385846×. The predeclared descriptive2× penalty flag is false. Sequential
small-panel timing is not a significance test or an end-to-end generation runtime
prediction.44 cases and finite timing replays do not prove global watertightness.

## Integration and next action

`verified_ray_backend.py` verifies checked-in native admission and exact .so bytes
before explicitly loading the corrected sidecar, with no fallback. Bell generation
uses that loader before the pinned wrapper import. All three oracle A/B/C material
plans require the same corrected image/backend; dataset admission checks the
recorded tracing identity, and loading occurs before scientific Trainer imports.
Shader/YAML/network/loss/optimizer/sampling settings are unchanged. Historical
nonoracle/legacy execution retains its original image/backend, and E.11C's explicit
original-backend selection remains available in the derivative image. A_original
in the oracle comparison denotes original BRDF mathematics with common corrected
visibility, not bitwise legacy tracing.

[Generation v3](protocols/BELL_ORACLE_GENERATION_E11_V3.md) pins this dependency
and a new output directory. Full Bell generation still requires its own explicit
commit/freeze-bound authorization. No dataset or neural-training admission is
inferred from the ray-backend regression. Stop on any subsequent genuine camera,
geometry, energy/domain or reference-convergence failure without changing bounds.
