# Publication Gate 12 preparation

**PROCEED with the separately authorized rendering study. No scientific rendering
or optimization has executed in this preparation.** Native Bell runtime and
integration convergence are untested for this new adapter, not reported passes.

## Admission and provenance

The researcher has admitted the authenticated O1 candidate for controlled
downstream evaluation as **G_background_opacity_oracle_v1**, explicitly assisted
by perfect synthetic masks. The [variant-aware handoff](O1-controlled-handoff.json)
is byte-identical to the new separate admission record:

`/home/isavand/paper/evidence/bell-background-opacity-gate11-a8d84c48f9a04864a659b10786bd23b9-controlled-admission-v1.json`

It binds producing source2d5a1ac0d2f04e701e0a75276d7ac67aee1ad9a1, the accepted
Gate11 freeze and completed training/evaluation receipts, step300k checkpoint
`b9a5fb4c461de77ec9f387466e05373f959df65baebd5151f02278d1648c46e7`,
and mesh `0becad75d5f5404b84f5a154870eaf66e71f00ba5b843776f50dd6b1989d7ac3`.
Only bytes were hashed; no checkpoint was deserialized. Original G remains
pinned and unchanged; O1 is a derivative. O0's failed admission and every prior
Gate11 receipt, verification and geometry artifact remain unchanged.

## Focused experiment

The [rendering protocol](../../protocols/BELL_GEOMETRY_RGB_GATE12_V1.md) and
[freeze](../../protocols/BELL_GEOMETRY_RGB_GATE12_V1.freeze.json) define one
six-camera comparison of oracle, admitted O1 and diagnostic-only O0. Reuse the
existing first hits, 128² calibration and original Bell v3 analytic material and
two-lobe light. This is not a switch to the sharper L2 or a heterogeneous dataset.

| Saved inventory | Oracle | O0 diagnostic | O1 derivative |
| --- | ---: | ---: | ---: |
| Defined material-rendering observations | 24,001 | 23,615 | 23,607 |
| Undefined reconstructed hits | 0 | 3,193 | 377 |
| Reconstruction-only excess | 0 | 2,807 | 21 |
| Missing oracle hits | 0 | 0 | 38 |

Counts are per-arm camera observations, not unique surface points. Undefined
includes excess and failed common-surface correspondences. All source arrays,
37 selected input-file identities, per-camera counts and overlap-aware cohort
counts appear in the [inventory](input-inventory.json). O0/O1 selected artifact
identities must match their already accepted manifests, not merely new hashes.

The primary evaluates the same original analytic Bell field at actual valid-hit
positions and normals. It is a consistent field across cameras, rather than
per-ray source-label copying. No oracle labels are manufactured for unmatched geometry. Full-raster known-support
scores must accompany undefined counts/energy; consequently the experiment may
not establish a complete all-raster ranking of O0 versus O1 radiance.

The adapter reuses the independent Kurt kernel and quadrature and the corrected
BVH as a visibility service. The same-direction oracle-visibility substitution
and unoccluded comparison separate the defined visibility and normal/frame
effects using the transported-source unoccluded counterfactual. Source-material
transport is camera-conditioned and is explicitly not the primary field. Analytic
coordinate and projection-versus-transport Q changes are saved separately.
Material-free visibility is also
measured at undefined hits without inventing their RGB. Open-edge and radial-band
associations do not by themselves establish a causal boundary mechanism.

Keep .003 absolute/.002 relative convergence, with prospectively bounded paired
256²/512² and optional1024²/2048² refinement. All component cases are checked.
Stop on numerical, identity, budget or resource failure, without retry or tuning.
The verifier prepares per-camera/split/tail/cohort metrics, source-Q versus
transport/resampling diagnostics and identical-scale RGB/diffuse/specular figures.
Fixed-material discrepancies are not attainable bounds for optimized fitting.

## Bounded authorization to obtain next

One approval must bind the final commit, exact freeze hash, corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
three immutable geometries and fresh output:

`/home/isavand/paper/evidence/bell-geometry-rgb-gate12-0ea32eb2081a435aa20ffef242773ce5`

| Maximum new work | Ceiling |
| --- | ---: |
| Incident visibility directions | 45,018,316,800 |
| Independent BRDF point-directions | 42,678,681,600 |
| Environment point-directions | 25,607,208,960 |
| Primary rays / SDF queries / checkpoint loads / models / optimizer returns | 0 |

A100 GPU0,4 CPU threads,64GiB RAM/no swap,128PIDs,25% CUDA allocator,2GiB
outputs. Scientific watchdog7200s/30s grace, inner7080s/20s grace; file-only
verification≤600s, overall foreground wrapper7800s/30s grace. Runtime estimate
20–110min is unmeasured. Non-root, no network/build/install, read-only source,
inputs and root; refuse a busy GPU and existing prior containers. The launcher
requires exact approved JSON and a clean committed source. No new rendering
authorization has been written or inferred from geometry admission.

## Checks and remaining limitations

[Eight focused synthetic CPU tests](focused-cpu-tests-final.junit.xml) pass:
undefined support, gauge/axis equivalence and tensor normal transport, actual
quadrature/kernel adapter plumbing with toy visibility, independent component
sums, absolute/pooled refinement and work-cap stopping, strict scientific flags
with scalar norm roundoff only, deterministic anchor-preserving ordering and
the global analytic field's independent projected-ambient-matrix identity.
The initial six-test record is also retained. No accepted training or geometry
test suites were repeated. Python3.8 syntax and shell checks are recorded in
[preparation evidence](preparation.json).

These checks do not prove native runtime, scene convergence, full-raster radiance
accuracy, inverse-rendering identifiability or realistic segmentation robustness.
The shading-label versus prior audit-normal difference is at most2.43e-8 and is
reported separately; it does not change any geometric eligibility mask.

## Realistic masks and next recovery

The fitting-calibration-only [SAM2 angular audit](sam2-ordering-v2.json) keeps all
122 fitting IDs and both original prompt anchors, with no held rows in the path
calculation. Adjacent angular median/p95/maximum change from86.37°/120.35°/122.91°
in numerical order to10.17°/13.36°/17.15°. Deterministic first-improvement2-opt
performs56 swaps. Both original prompt sets, four-pass consensus, raw score
capture, abstention and uncertainty margins remain unchanged in the
[prospective V2 addendum](../../protocols/BELL_SAM2_ANGULAR_ORDER_GATE12_V2.md).
No SAM2 import, checkpoint load, inference, prompt search or installation occurred.

After reviewing Gate12, the [minimum recovery proposal](../../protocols/BELL_O1_MATCHED_MATERIAL_RECOVERY_PROPOSAL_GATE12.md)
starts with a matched K1-style oracle/O1 known-light pair, preserving original
v3 observations and fixed10k endpoints. It retains a separate D2 unknown-light
pair and matched representation validation where justified. No material-truth
loss or training is authorized. Strongest-recipe selection is independent of
which representation was developed most recently.

The [updated backlog](../../RESEARCH_BACKLOG_GATE12.md) preserves same-mask R0/R1,
confidence-aware free-space SDF, anisotropic geometry appearance and adaptive
schedules. Bell remains development evidence; realistic-method confirmation
requires an additional dense scene or prospectively reserved acquisition.
