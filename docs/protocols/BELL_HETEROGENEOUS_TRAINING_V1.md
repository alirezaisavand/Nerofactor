# Matched heterogeneous Bell recovery v1

The researcher admitted the exact [verified Bell reference](../results/bell_heterogeneous_reference_v1/README.md).
Fit cameras 0, 4, 19 and 2, with 17,573 linear observations. Evaluate held-out
cameras 105 and 114, with 7,082 observations. Use the archived Bell mesh and
faceted negated BVH shading normals. All arms use learned direct illumination
with the same zero-indirect assumptions as the Donut comparisons. L2 supplies
the independent images and the known-light compatibility check.

| Arm | Starting field and training loss |
| --- | --- |
| D2 | Original corrected core, fresh accepted chart weights, original RGB and material priors |
| D6 | Same D2 recipe plus 0.05 transported world-Q smoothness |
| D8-C | Matched D8 0.25 isotropic start, chart predictor, tensor renderer and sampler, scalar/energy/specular/white priors plus transported world-Q |
| D8-T | Same D8 controls with ambient tensor prediction |

Keep seed 6033, the full spatial model, 512 training points, 512/1024 shader samples,
linear Charbonnier loss, original warm-up/cosine schedule, Adam/AMSGrad and
0.7 gradient clipping. Match the previously accepted fresh weights and buffer
hashes from metadata. Both D8 arms retain their common tapes and paired initial
Q, scalar, shared/light state and RGB gate. The 24 positions and 32 draws of the
existing known-light check are selected on camera 0 using the accepted rule with
the actual Bell labels and unchanged compatibility thresholds. Run those
checks inside the original setup before any optimizer returns. No standalone
smoke or historical weight loading is needed.

The fitting pool contains position, normal, outgoing direction and linear RGB.
Ground-truth materials, geometric region labels, boundary probes and held-out
observations are used only by no-gradient admission and evaluation paths.
The four runs retain 8-draw native evaluation and material exports at
0, 2,000, 4,000, 6,000, 8,000 and 10,000. Save and verify the own-run checkpoint
roundtrip at each of the five post-initialization states, retaining Adam
state and RNG in the output. The fixed primary endpoint is 10,000.

## Bell transport domain

The same D6 edge-connected distance and dihedral constraints retain 362,520
eligible pairs with normals equal to the negated PLY face normals. Reserve
1,024 evenly indexed pairs and sample 128 per update from the remaining
361,496 using the same independent seed 26061 and permutations. Domain
integer arrays and six count snapshots are frozen. Each arm hashes its
full numeric domain, and offline verification replays the numeric geometry
and requires matching archives across the three spatial arms.
There is no boundary mask or change to distance weighting. Very short
Bell edges can amplify the unchanged distance normalization, and the
analysis will report fitting gradient interactions and sampled coverage
to characterize this effect. The eligible training domain covers about
99.87% of mesh area. The fitting and held-out observations cover about
9.43% and 3.84%, respectively, under the observed-triangle area scheme.

## Evaluation

Report the existing relative world-Q Frobenius errors, width MAEs,
director-sign-invariant orientation, scalar and illumination errors, and
fitting/held-out linear RGB. Evaluate orientation at gap ≥0.1 in the
oracle and prediction and include the identical eligible subset across
methods. Report exact-isotropic, weak nonisotropic, smooth-interior,
boundary band and each side of the band, and the existing normal-aware
fitting and highlight-coverage cohorts. Report both observation-weighted
and observed triangle-area-weighted errors. Observed area allocates a
triangle's area across its observations within the split.

The strict adjacent-image diagnostic has zero Bell pairs. Preserve the
0.035-native-unit boundary-band metrics for comparison across objects.
The supplemental field-only diagnostic selects every triangle cut
transversely by exactly one region plane. Intersect the two edges and
take the midpoint of their crossing segment. Take one point halfway
toward the minimum and another halfway toward the maximum signed-distance
vertex. Keep the established `[1e-6, .02]` native distance span.
Order by plane and PLY index, and arrange each pair in region-0 minus
region-1 order. Both points lie on the same original face and use its
same actual negated normal. This fixes 2,474 pairs, 1,565 and 909 from the
two planes, with median separation 0.00479 native units. There is no
new primary ray, shadow ray or volume query for these probes. At each
saved state query both sides once using the full field predictor.
Report signed width jump, Q jump projection onto the analytic oracle
jump, and absolute errors. Projection ratios require oracle Q jump norm
above 1e−12. The pairs differ in support and separation from Donut
image pairs, so those jump ratios are object-specific. Use the common
boundary-band errors for the consolidated homogeneous Donut,
heterogeneous Donut and Bell comparison.

## Work and launch boundary

| Batch work | Maximum |
| --- | ---: |
| Original Adam returns | 40,000 |
| Native shader calls | 49,924 |
| Shadow-ray slots | 38,733,201,408 |
| Loss-component diagnostic reverse sweeps | 114 |
| Surface training pairs | 3,840,000 |
| Added boundary field calls / points | 48 / 118,752 |
| New independent references or primary rays | 0 |

Use the reviewed corrected image and A100 GPU 0, four CPU threads,
64 GiB host RAM without swap, 128 PIDs, 25% CUDA allocator cap and up
to 8 GiB owned output per arm. Use one sequential foreground launch
with a 14,760-second watchdog and 30-second kill grace. Each arm gets
3,600 seconds, with a 3,540-second inner limit and 20 seconds of kill
grace. Read-only source, mesh, reference, initialization metadata and
root, with non-root UID and no network. Stop the batch on mandatory
failure and preserve all records. No retries, tuning, prior arms,
additional reference rendering or new regularizers.

The freeze binds separate dataset admission and execution authorization.
The Bell adapter adds only acquisition, facet-domain and evaluation
routing to the shared trainer. The frozen scientific snapshots and
method definitions remain unchanged. Replaying a historical freeze
uses its recorded commit, including the earlier shared adapter bytes.
