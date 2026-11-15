# Bell camera1 anomaly: missed front intersection, not a flipped triangle

**FIX AND PROCEED with a separately validated ray-predicate derivative.** Keep
the original mesh, normals and strict admission gate. No generation, GPU work,
material training, image build or original-file repair was performed here.
[Numeric report](evidence/BELL_E11_TRIANGLE_DIAGNOSIS_V1.json) preserves input
hashes, both triangle one-rings, all25 neighboring rays and CPU comparisons.
All triangle/vertex indices below are zero-based original PLY indices, not
BVH-reordered primitive IDs.

| Item | Saved farther hit | Actual closest intersection |
|---|---|---|
|Triangle|158220|181142|
|Vertex IDs|79470,78633,79478|90064,90871,90065|
|Depth|3.6032063961|2.75133774975|
|Area|5.20198092546e−5|1.81417824617e−5|
|Longest edge/altitude|1.18253|3.41208|
|Final A normal|(.57615650,−.57919625,.57669350)|(.39438296,.48206510,−.78235243)|
|Final normal dot view|−.72679910|+.95854049|
|Ray barycentrics|( .309094675,.399272219,.291633106 )|( .00000540394,.065878449,.934116147 )|

Triangle158220 vertices are
(.1609443873,−.9607843161,.2235294133),
(.1607843190,−.9531342983,.2313725501),
(.1686274558,−.9531415701,.2235294133).
Its raw cross-product normal is the negative of the final A normal above.
The saved point is5.3163e−7 from its plane; projected point barycentrics are
(.309083109,.399234293,.291682597), consistent with float32 position rounding.
There is no evidence that this triangle itself needs repair.

## Diagnosis and extent

Independent float64 Möller–Trumbore, a linear-system solution and extended-
precision arithmetic agree on the closer front hit and farther exit. The depth
error is **.85186864635 scene units**. All24 other saved5×5 pixels agree with
CPU closest depths within3.95e−7 (the numeric report contains the measured values);
their facing dots are .9366–.9774. The abnormal normal belongs to a coherent
far-side surface, not the nearby front surface. Flipping that normal would leave
the large position/depth error intact.

Each edge of both selected faces has exactly two incident faces and opposite
shared-edge directions. Front-face neighbors181141,181143,179521 have normal
dots>.99995 with181142. Exit neighbors158219,158221,158218 have normal
dots .99999990,.58166436,.999999998: the lower dot is a real local bend, not
reversed winding. Both targets have nonzero ordinary-size areas, not slivers.
There are no boundary/nonmanifold edges or inconsistent two-face edge directions
in the mesh. No global self-intersection test was performed; topology alone
would not prove absence of overlaps.

Both targets lie in the same dominant closed component(124450 vertices,
248900 faces,signed volume−.85474821546). A separate microscopic6-vertex,
8-face closed component has opposite signed volume≈+1.04e−16; it is not this
ray's hit and was **not removed or repaired**. All saved camera centers lie
outside the mesh bounding box. Keep that secondary component as a documented
residual risk; do not infer globally correct normals from aggregate volume.

Testing the two one-rings against **existing camera0/1 rays only** finds one
closer-than-saved discrepancy, exactly ray13621. Front triangle181142 has a
front-facing in-frame projection in86 cameras; exit158220 in11. These are
potential footprints, not occlusion-tested visibility counts. The numerical
failure class applies to shared edges throughout the mesh and to shadow rays;
it is not safely confined to this one pixel. No new camera images were rendered.

## Pinned source and narrowly scoped candidate correction

The inspected upstream source is raytracing commit
`29329268a86921f1c938b05277937369531209a4`, verified by Git blob hashes; setup.py
matches the historical SHA256. Primary source:
[triangle predicate](https://github.com/ashawkey/raytracing/blob/29329268a86921f1c938b05277937369531209a4/include/raytracing/triangle.cuh),
[closest-hit traversal](https://github.com/ashawkey/raytracing/blob/29329268a86921f1c938b05277937369531209a4/src/bvh.cu).
The predicate computes cross/dot/barycentric quantities in float32 and applies
strict0≤u,v,u+v≤1 tests independently per face. A CPU fmaf evaluation order of
this exact formula makes front181142 give u+v=1.0000001192092896 and its shared-
edge neighbor181143 give v=−5.4791e−6. Both reject. Float64 accepts181142 and
rejects181143 correctly. Ordinary noncontracted float32 can accept181142:
**CPU float32 success does not certify CUDA contraction behavior.**

Verified: the saved BVH result violates the closest-hit contract. Supported
mechanism: a non-watertight float32 predicate creates a shared-edge hole. The
actual compiled CUDA instruction order, precise traversal and AABB contribution
remain unverified offline; no claim is made that the source-level simulation
captures the exact machine path.

The [candidate patch](../docker/paper/patches/raytracing-triangle-fp64-v1.patch)
promotes only intersection predicate operands **before subtraction**, and its
cross/dot/barycentric/depth intermediates, to float64. It retains strict bounds,
float32 inputs/output depth,1e6 miss sentinel and existing normal calculation;
Triangle.normal(), BVH traversal/AABBs and all scientific snapshots are unchanged.
It introduces no barycentric epsilon or shading/mesh correction. It is **not
wired into the Dockerfile, generator or native trainer**, and has not been
compiled. The patch and scalar fixtures pass focused offline tests. Float64 is
a narrow candidate, **not a proof of general watertightness** or a certified
fix for every possible BVH failure.

## Smallest safe continuation

1. Separately approve staging the exact upstream commit plus the hash-pinned
   predicate patch, retaining licences, in a **new** extension/image variant.
   Preserve the reviewed historical image and installed binary. Record the new
   source, build and binary/image identities; do not update the current image.
2. Before generation, authorize a tiny native intersection acceptance case using
   this exact saved ray, its entry/shared-edge/exit triangles and edge-side/miss
   controls. Require closest depth≈2.75133775, correct normal, unchanged ordinary
   controls, and no false misses. If promotion does not fix the native case,
   stop and isolate traversal/AABB; do not retry generation or loosen admission.
3. Only after that passes, prepare a new generation implementation freeze and
   fresh authorization/output directory. Pin the new dependency/image in
   `benchmarks/bell_e11.py`, the launcher and authorization; keep all cameras,
   mesh/tapes/field/BRDF/convergence bounds and work ceilings unchanged unless a
   separately reviewed performance constraint requires a budget change.
4. Use the same admitted tracing variant for generation and all three native
   material arms. Label A_original as the original **BRDF** comparator with common
   corrected visibility, not bitwise legacy tracing. Keep native material-smoke
   authorization separate. Do not claim GPU correctness from the CPU fixtures.

Do **not** lower the front-facing fraction: for this closed component and an
outside camera, the nearest opaque intersection should face the camera under
the accepted normal convention. A near-edge missing entry followed by an exit
is precisely what this gate should catch. Open/two-sided surfaces would need a
different material protocol; that is not a justification to exempt this hit.
