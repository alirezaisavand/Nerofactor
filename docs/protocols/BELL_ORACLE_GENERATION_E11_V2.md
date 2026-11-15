# E.11 corrected implementation freeze v2 — mask layout

This amendment changes **array layout only**, following the confirmed native v1
failure at geometry_gate. The scientific definition, thresholds and work limits
in [v1](BELL_ORACLE_GENERATION_E11_V1.md) remain unchanged. The generation protocol
identifier stays `bell-oracle-generation-e11-v1`; the corrected implementation
and its source closure have a new `.freeze.json`, selected explicitly by the
foreground launcher. A new commit/freeze-bound authorization is mandatory.

## Correction and focused validation

The original nearest-center800→128 mask sampling and Boolean threshold>127 are
unchanged. Pack the resulting128×128 mask with `ravel(order='C')`: pixel[y,x]
maps to ray y*128+x, matching the original primary-ray meshgrid/reshape. The
geometry gate requires hit/mask/depth shapes(16384,) and vector shapes(16384,3),
Boolean hit/mask arrays and a nonempty union before computing IoU. It rejects
mismatches explicitly instead of broadcasting. No resize, pixel, threshold,
normal, geometry, material, ray, reference integration or RNG change is made.

Focused synthetic CPU regressions exercise the actual mask sampling assignments
from the generator and the real geometry_gate, including asymmetric masks,
C-order versus F-order packing, exact IoUs1,.5,.75,.95,0, empty unions and
mismatched dimensions. Nearby flat RGB/components, saved image masks, label pixel
indices, depth and relighting indexing were inspected and checked synthetically;
no other shape defect was confirmed, and those operations were not changed.
An AST comparison verifies render() is unchanged except for mask packing.
Tests also check v1 freeze identity and all its nonedited source/config/tape
identities. No previous scientific experiments or historical test suites rerun.

## Immutable failed attempt

Preserve `/home/isavand/paper/evidence/bell-oracle-e11-432eb086711a493aa23cf6fa69fb75d2`
and its original authorization, console, failure report, calibration archive,
mesh and execution records. Their read-only size/hash/filesystem identities are
in [the preservation inventory](../evidence/BELL_E11_MASK_FIX_V2_PRESERVATION.json).
Original freeze SHA256 remains
`4682a6304be08d3f03d525dde10c27ecc90910e9f6581395f8c09bd83d1fb3af`.
Do not rewrite the failed report as successful. Its primary-ray counter is zero
because the increment follows geometry_gate; the traceback nevertheless
establishes a first-camera BVH return. It is not evidence of no GPU work. No
reference integration was reached. This correction does not change accounting.

## Fresh launch and authorization

New proposed batch (not allocated by preparation):
`/home/isavand/paper/evidence/bell-oracle-e11-v2-7a327024561e4bf3bf4a7f0450107bc3`.
Output package is its `benchmark` child. The old authorization cannot authorize
this source/freeze/batch. Prepare a separate approved authorization file only
after the researcher's explicit reply. The image, GPU0, CPU/memory/work ceilings,
7200s outer watchdog/30s grace,7140s inner timeout/20s grace, no-network/non-root/
read-only mounts, independent reference and failure-stop/no-retry rules remain
exactly v1's. Use `benchmarks/run_bell_e11.sh`, now selecting the **v2 freeze**.

The [versioned launch specification](../evidence/BELL_E11_MASK_FIX_V2.json)
binds the planned batch, inputs, image, freeze and preserved failure identities.
The actual execution authorization also binds the final corrected Git commit.
No scientific rendering is authorized by this amendment or a test pass.
After separate approval: one generation, original admission checks and offline
verification; stop honestly on any geometry/calibration/convergence failure.
Native material training still requires its own subsequent authorization.
