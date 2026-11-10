# 3A-2 scientific close-out: saved Bell r128 measurements

This is a read-only, offline interpretation of the completed native 3A-2 action,
prepared for milestone3B-1 at infrastructure base
`959db370cd18a09a20b2a0c9bb24a20a2cdba22a`. No SDF was queried, checkpoint
loaded, mesh extracted, ray traced or native experiment repeated. None of the
original failed/unresolved scientific statuses is promoted to pass.

## Exact evidence and reproducible analysis

Explicit source directory:
`/home/isavand/paper/evidence/bell-sdf-audit.leLmVWdK/run`.

| Artifact | SHA-256 |
| --- | --- |
| `sdf-mesh-audit.json` | `839f101d31d349e9b45bf6cfcf559681ea381cc013bb375f23930c6cdc8a8cb8` |
| `measurements.npz` | `522bf62ce7e02d2148866e35a6dec1fa86a16b73c681118809efdedc4309b66e` |

The report declares the exact base above, complete action, strict full G loading,
unchanged network/RNG, 19,649 point evaluations, 154 SDF calls and 18 first-gradient
calls. These are inspected saved records, not a repeat native observation.
The existing identities remain G `79252178e65914ad1acbc14790b1514f276bfea0`,
A `e1e859779301faced46d6476691e20b546c27ae0`, historical checkpoint
`d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc`,
and exported mesh `eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5`.
No new training/provenance attribution follows.

[Machine-readable close-out](GEOMETRY_SDF_MESH_CLOSEOUT_V1.json) includes every
failed sample's original face/vertex index or saved ray sample, point, radius,
area where meaningful, signed raw SDF, gradient length, distance proxy and four
saved gradient-direction probe values. Arrays load with allow_pickle=False;
explicit hashes, numerical types, finite data, sizes, unique probe bindings and
recomputed norms/proxies/counts are checked. Input bytes are rechecked afterward.
This analysis reuses no 3A-1/3A-2 execution functions.

```bash
cd /home/isavand/paper/Nerofactor-paper
PYTHONDONTWRITEBYTECODE=1 python3 -B -m audits.sdf_mesh_closeout \
  --report /home/isavand/paper/evidence/bell-sdf-audit.leLmVWdK/run/sdf-mesh-audit.json \
  --report-sha256 839f101d31d349e9b45bf6cfcf559681ea381cc013bb375f23930c6cdc8a8cb8 \
  --measurements /home/isavand/paper/evidence/bell-sdf-audit.leLmVWdK/run/measurements.npz \
  --measurements-sha256 522bf62ce7e02d2148866e35a6dec1fa86a16b73c681118809efdedc4309b66e \
  --output /tmp/bell-3a2-closeout-review.json
```

Output must be new; the CLI refuses replacement. This is an optional reproduction
of the **offline analysis**, never a command to repeat native3A-2.

## Frozen criterion and localization

h=2/127. Eligibility requires norm(gradient)>1e-6, original inside-sphere base
mask and both saved gradient probes at ±h/2 inside the original float32 sphere.
A failure is proxy=|SDF|/norm(gradient)>h/2 **or** no robust ±h/2 sign crossing
(magnitudes>1e-7). All thresholds remain exactly as frozen before native execution.

| Population | Count |
| --- | ---: |
| Total / eligible / excluded | 2,275 / 2,119 / 156 |
| Eligible failures / passes | 74 / 2,045 |
| Failed centroids / vertices / saved intersections | 65 / 8 / 1 |
| Proxy failures / half-grid sign failures / overlap | 73 / 74 / 73 |
| Failures within one h of radius1 | 73 |
| Failed faces with area≤1e-10 | 0 |

Failure radii min/median/max are 0.3010726668 / 0.9903656830 (rounded) /
0.9926444956; use JSON for full precision. Median absolute raw SDF is
0.0606538206 and median proxy/h is3.83700149, versus about0.00368453 for
eligible passes. Failed-face areas range1.82564e-5–2.11220e-4; median≈1.24005e-4
(about h²/2). Failures are not confined to the tiniest triangles. Sampling was
area/radius-stratified; these proportions are **not** unbiased whole-mesh rates.

73 boundary-localized failures have negative raw SDF, with both quarter- and
half-grid probes negative. The original extraction replaces raw values outside
norm(x)≥1 with+1. The observed negative near-boundary field is consistent with
mesh cap triangles created by interpolating that masked-field transition rather
than a raw-network zero. **Inference:** a strongly supported boundary mechanism;
exact generating grid cells/corner values were not saved, so triangle-by-triangle
causal proof is unavailable. Boundary-associated failures also occur among the
156 excluded points; exclusions cannot be relabeled successes. Do not remove
these faces, expand the sphere, tune h or repair normals to improve this report.
The failed saved intersection is sample2209, source58/ray557948, also the prior
CPU later-than-first-hit observation. This association does not establish the
cause of the BVH hit discrepancy.

The remaining interior failure is sample7, original face27265, radius0.3010726668,
area1.82564395e-5, raw SDF+6.79790974e-5, gradient norm1.13783065 and
proxy/h≈0.00379377. Saved gradient-direction probes are:

| Offset | Raw SDF |
| --- | ---: |
| +h/4 | +0.00119128823 |
| −h/4 | −0.00428751111 |
| +h/2 | −0.00351250172 |
| −h/2 | −0.00834047794 |

It passes the proxy bound but fails the frozen half-grid sign test. The quarter
crossing and subsequent positive-to-negative sampled change support local
nonmonotonicity/additional zero crossings along that segment (continuous MLP),
not a large residual at the mesh point. Thin structure is a hypothesis; there is
no independent thickness/ground-truth geometry measurement. A wider sign bracket
can cross more than one surface and does not monotonically improve this test.
Keep the original failure; a future adaptive/local crossing protocol would need
its own version and validation, not retroactive threshold changes.

## Normal and silhouette interpretation

All1,919 eligible centroid signed gradient/face cosines are negative. There is
no observed mixed **sign** population; the unresolved orientation status arose
because not all were within30° of perfectly opposite. Negated-face median error
is0.516947°, p95≈28.0581°, maximum≈51.8555°. No global outward proof follows.

Of89 geometrically eligible non-anomalous A controls, source14/ray266838 has
exterior-signed alignment0.7980639565 (<cos30°), radius0.2357719754. This interior
control failure cannot be explained solely by the sphere boundary. All99 saved
hit gradient-vs-final cosines are positive; the failing criterion concerns angular
agreement, not an observed globally flipped sign. A normals stay unchanged.

Anomalies retain prior classifications: source58/ray322750 ambiguous,
source58/ray557948 a CPU-supported later-than-first hit observation,
source11/ray390143 ambiguous. No new trace or causal attribution was made.
Existing FP counts14/58/11 are4,933/3,093/3,384; their >4-pixel counts are
138/141/17 respectively. Use the exact original status block in JSON for all
bands/FNs. Silhouette characterization is not independent3D accuracy.

## Close-out decision

The diagnostic action completed, but scientific zero-level and A-normal criteria
failed and orientation/anomaly conclusions remain unresolved. Close3A-2 as
**measured, localized limitations**, not geometry acceptance. Historical r128 is
still diagnostic only. Checkpoint origin, global normal correctness, scientific
geometry accuracy and ordinary research G→A handoff remain unverified/unadmitted.
For3B-1 use analytic unit normals/flat surfaces to isolate BRDF mathematics;
do not confound estimator corrections with changes to this historical mesh.
