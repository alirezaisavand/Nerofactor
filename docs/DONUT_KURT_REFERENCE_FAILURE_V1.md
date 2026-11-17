# Donut reference v1: reporting correction and independent convergence blocker

Decision: **BLOCKED for dataset admission and training.** Camera0's saved image
passes the frozen image-convergence test. Its unoccluded-reflectance convergence
does not pass. Fixing JSON reporting cannot admit this dataset or authorize a
rerun. Cameras50/1/45/28/34 were not started; there were no material models,
optimizer updates or Blender calls.

## Reporting defect and focused fix

The reused `measures()` helper returns `(scalar_summary, Boolean_pixel_mask)`.
The generator incorrectly assigned the entire tuple to `highlight_diagnostics`.
After both camera0 integrations and its NPZ/previews were saved, printing the
view row raised `TypeError: Object of type ndarray is not JSON serializable`.
The same tuple then broke terminal JSON writing, leaving a truncated
`generation.json`. This was an implementation error in the new adapter.

The correction explicitly unpacks the helper. Only the unchanged scalar summary
enters JSON; the unchanged Boolean highlight mask is saved as a numerical NPZ
field. The report writer now serializes with strict `allow_nan=False` before
opening its exclusive output file, so a serialization error cannot create a
truncated terminal record. It still rejects arrays/NaNs in JSON and refuses to
overwrite existing evidence; there is no generic string conversion or relaxed
scientific validation.

Regression fixtures use the actual reused helper and full nested view/failed
terminal report, including both ordinary and HDR components, mask preservation,
bad-payload rejection and existing-file protection. The focused suite reports
**23 passed,0 failed,0 errors,0 skipped** (17 existing +6 new).

## Saved numerical evidence, rechecked offline

Original immutable batch:
`/home/isavand/paper/evidence/donut-kurt-reference-v1-5df0234193664e9b93cae546542ca7a2`.
Source commit8f22b544b1a9f35de387223bdc7352263ca13f34; original freeze SHA-256
064147fa7f2a966f523dcc104b1a88cd1618d9ec2d4649af1ebb7de4bd265718.
The host exit code is1. The invalid terminal JSON is preserved byte-for-byte,
not repaired or relabeled. Its fully parsed scalar prefix is explicitly labeled
as salvaged evidence in the new analysis; the original report remains invalid.
All eleven original files, copied freeze/authorization and the three original
input identities are hash-recorded/verified in
[the offline report](evidence/donut_reference_reporting_fix_v1/offline-analysis.json).

Camera0 saved4,539 geometric hits,4,536 valid one-sided observations and complete
256²/512² arrays. Counter evidence records16,384 primary rays,8 CPU checks and
1,486,356,480 shadow/BRDF point-directions each. No new rays, reference integration
or model query was used for this analysis. Cleanup success is not established by
the original terminal record (`cleanup_verified` is false); the native helper
contains finally cleanup, and the host launch exited, but no complete receipt
survives to upgrade that flag.

| Frozen check | Max absolute difference | Relative L2 | Decision |
| --- | ---: | ---: | --- |
| Image RGB | 0.001240183573 | 0.000759862773 | PASS |
| Unoccluded reflectance | 0.008422707391 | 0.000442193677 | FAIL |

Both tests require max difference≤0.003 **and** relative L2≤0.002. Five of4,536
points fail the reflectance absolute bound. Their source pixels(x,y) and n·wo are:

| Pixel | n·wo |
| --- | ---: |
| (91,98) | 0.00155036 |
| (69,31) | 0.02127431 |
| (37,98) | 0.00779723 |
| (89,37) | 0.00103165 |
| (59,31) | 0.01982033 |

The largest per-channel reflectance values observed on the fine grid are
[0.10769115,0.93650555,0.05927530]; none exceed1 in this view. This does not prove
global energy conservation. The discrepancy is convergence error between grids,
not evidence that reflectance exceeded1. All failing configurations are grazing;
this is consistent with difficult angular resolution of narrow reflected lobes,
but neither the absolute accuracy of the fine grid nor a unique numerical cause
has been established.

The preserved scalar highlight statistics are promising for this first view:
integrated specular fraction0.70699, frozen highlight coverage0.32055 and maximum
linear RGB0.45411. These are not orientation-identifiability measurements or a
six-view dataset admission.

## Minimum next action

Do not simply repeat the six-view command. First review a bounded higher-accuracy
unoccluded-reflectance check at the five saved material/view points. An independent
half-vector-coordinate quadrature is a concrete alternative to brute-force
hemisphere refinement near grazing. It can use the saved normals, directors and
views with **zero new geometry/visibility rays**, preserving the existing BRDF
and0.003/0.002 criteria. Any such new BRDF integrations require separate explicit
approval. Do not drop these pixels, change normals, clip energy or relax bounds.

If that resolves reference accuracy, a separately frozen continuation can propose
reusing the saved camera0 image/label arrays and generating only the unstarted
views. That is not authorized here. The original v1 freeze, authorization,
preparation evidence, failed outputs and frozen G/A/core sources remain unchanged.
No replacement scientific freeze or affirmative execution approval is created by
this reporting correction; the original freeze intentionally rejects changed
adapter bytes. D1/D2 training remains blocked.
