# Sphere surface-comparison launcher correction v2

The v1 attempt from `7937a1f10d8186af1514da2e2c83d20a426e3695` failed before
scientific execution. Its first arm printed `metadata_only` and exited0 because
the launcher omitted `--execute`. No native training or checkpoint loading was
reached in that entry-point branch. The launcher used exit0 as success and then
attempted to mount the absent control report for arm2. Docker rejected that
mount with exit125. Neither arm produced a `run` directory.

The failed batch remains untouched at
`/home/isavand/paper/evidence/sphere-surface-adb195af0a874453a73cb42ae697a9ae`.
Its original v1 freeze, authorization, console logs, container ID and exit-code
receipts are recorded in `docs/evidence/SPHERE_SURFACE_LAUNCH_FAILURE_V1.json`.
Do not reuse or relabel that attempt. The old freeze and prior validation reports
are preserved; the missing-flag bug was not covered by their unit fixtures.

## Exact correction

1. Pass `--execute` explicitly to the guarded Python entry point inside Docker.
2. After each zero exit, require an existing unredirected `surface-run.json` with
   status complete, matching arm/protocol/commit/freeze,8000 absolute returns,
   4000 new returns and both cleanup flags true. A zero exit alone is insufficient.
3. Use the same completion gate when admitting the control report in arm2.
4. Select the new `SPHERE_SURFACE_COMPARISON_V2.freeze.json`, with a fresh batch.

The metadata-only entry remains available and continues to do no native work.
No scientific implementation, initialization/fork policy, checkpoint, source
input, regularizer formula, optimizer setting, schedule, sampling or resource
limit changes. Both arms still fork exact state4000 weights/Adam with the common
declared RNG reset and execute4000 new updates, ending at8000. The entire
[v1 scientific protocol](protocols/SPHERE_SURFACE_COMPARISON_V1.md) remains in force.

## Focused CPU validation

The new tests run the real Bash launcher against a fake Docker executable, never
the Docker daemon. They verify explicit execution in both arm argument vectors,
normal two-arm advancement for synthetic complete receipts, and immediate stop
for zero-exit metadata-only/missing-report cases before any second-arm mount.
Separate tests reject wrong counts, identities, incomplete cleanup and redirected
reports. The existing cold-entry/no-implicit-authorization test is also run.
Full JUnit and process-exit evidence are in `docs/results/sphere_surface_launch_v2`.
These are routing/receipt tests, not native training validation.

Fresh source/freeze/output authorization is required before another native
launch. No retry, Docker invocation, model construction or real checkpoint
deserialization is performed by this correction task. After approval, use the
same foreground launcher with its new bound authorization; stop on any failure.
