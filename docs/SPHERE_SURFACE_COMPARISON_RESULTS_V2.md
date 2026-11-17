# Corrected sphere surface-regularizer comparison: results

**Execution complete; world-Q does not meet the frozen global seam criterion.**
The native pipeline completed both matched forks and the world-Q arm slightly
improves RGB PSNR, but does not prevent albedo darkening or reduce the primary
Q-discontinuity p95. Do not restart100000-step training on the assumption that
this candidate has solved those problems.

## Evidence and execution

- Source: `c1a06d4e685628974f8fcbd1f9791f8c6b6f8ab1`.
- V2 freeze SHA-256:
  `5b1660f4c7fed73676e08646dd41e3ec95460214384660092160a240acb3627e`.
- Authorization SHA-256:
  `ffabebc8541b71904b5a72fb1f5dd469d02a64d822b4f2b953fa854d876a036f`.
- Batch: `/home/isavand/paper/evidence/sphere-surface-v2-712adf6d038d49cc87807855ec360fd6`.
- File-only analysis: the same absolute path plus `-analysis-v1`.
- Checkpoint: exact selected state4000, SHA-256
  `610b42044d43d64fd3bf5c8d309fe8acd9f266b05e85acccd92acdd222f173ea`.
- Image: `sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`;
  corrected BVH binary SHA-256
  `800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.

Both exit-code receipts are0. Both native reports have status complete,4000 new
Adam returns,8000 absolute returns, strict checkpoint/Adam restoration and true
owned/runtime cleanup flags with no cleanup errors. Parameter displacement is
nonzero and Adam advances at all three recorded update probes. Each arm saves
camera36 validation/maps/checkpoints at6000 and8000; there is no NVS or full run.
The absence of cleanup.log is consistent with Docker --rm and an already removed
owned container; actual daemon-level removal is not independently inspected here.

All108 initial parameter hashes, initial buffers, Adam state/configuration,
sample tapes, RNG identities, splits, scientific/runtime configurations and
surface-pair descriptors match. Source implementation and original209 input
identities verify. The source uses Torch1.8.1+cu111 / Python3.8.10 / A100; both
runtimes report identical settings, including4 intra-op and16 default inter-op
threads. The host container CPU quota remains4. Historical RNG/sampler state
was unavailable: these are matched forks with a common reset, not exact
continuations of the historical sample trajectory.

The loss domain contains512 deterministic mesh pairs, of which192 have positive
weights under training camera8's visibility/image gate. Both arms use the same
arrays. Descriptor hashes name pre-conversion float64 arrays; the saved NPZ has
float32 arrays. Those byte streams are not interchangeable. The cache was not in
the native terminal output-hash inventory; its current hash and exact cross-arm
array equality are recorded separately. The validation archives/checkpoints and
images do match their original native terminal hashes.

The [machine-readable results](results/sphere_surface_comparison_v2/results.json)
record131 external artifact identities. Large checkpoints, numerical archives,
logs and original failed evidence remain outside Git. The previous metadata-only
v1 failure remains unchanged and contributes zero training updates.

## Matched endpoint results

All rows use the same340315 foreground pixels and678886 eligible projected
neighbor pairs. RGB PSNR/SSIM use the original full-image uint8/sRGB definitions;
they were independently recomputed from saved arrays and agree with native logs.
Material/illumination means use numeric foreground data, not PNG display values.

| Quantity | Initial4000 | Chart control8000 | World-Q8000 |
|---|---:|---:|---:|
|PSNR,dB|34.25008|35.68083|35.84899|
|SSIM|.952741|.957414|.957135|
|Transported-Q neighbor p95|.0395695|.0380015|.0389189|
|Mean red kd|.992424|.097396|.095790|
|Mean metallic|.846449|.845060|.848267|
|Mean red kd(1−metallic)|.152268|.015017|.014451|
|Effective red diffuse incident light|.468068|4.143822|4.302931|
|Mean major width|.999935|.999977|.999941|
|Mean minor width|.040158|.040461|.040537|
|Mean anisotropy gap|.996779|.996728|.996715|
|Mean Q trace|1.001483|1.001592|1.001528|

Frozen interpretation: require at least20% lower Q p95, no more than.2dB PSNR
loss and.005SSIM loss. World-Q gives **2.414% higher** Q p95,+.16816dB PSNR and
−.000279SSIM. Thus the primary smoothness condition fails, while both RGB bounds
pass. Mean gap and trace differ by less than.01%; neither the20% isotropization
nor shrinkage flag triggers. Widths remain close to their inherited limits.

A separate extreme-jump diagnostic falls from675 to349 pairs (48.3% fewer).
These pairs have raw orientation jumps>45° and Q differences≥.05 under the
previously specified diagnostic gates. This indicates a limited tail benefit,
but does not override the failed primary p95 criterion. Most remaining large raw
jumps also have a physical-Q discontinuity in this diagnostic; they cannot all
be dismissed as artificial angle-color seams. A screen-neighbor check is not
whole-mesh accuracy or a global smooth-field guarantee on a sphere.

## Albedo and gradient assessment

Both arms reproduce the rapid darkening. Diffuse allocation falls while sampled
effective incoming illumination increases; image reconstruction improves despite
the change. About93.34% of visible diffuse-light display pixels have all channels
clipped at8000. The images and numeric arrays corroborate compensation rather
than a display-only kd change. No independent material GT exists, so the absolute
physical correctness of either albedo estimate is unknown.

Energy is zero in all400 recorded training component rows per arm, all three
gradient probes per arm and both saved validation views per arm. This is not a
record of every one of the4000 minibatches. At the probes its neural gradient is
also exactly zero. Specular-prior gradient norms are approximately2e−6…1.3e−5
of the corresponding RGB norm. The saved evidence therefore gives no support
for energy or a dominant direct specular-prior gradient as the main darkening
driver in this panel. Shared-feature coupling, adaptive Adam scaling and unsaved
minibatches prevent an exhaustive causal conclusion.

At the zero-based7999 probe, the actual weighted components and full-network
gradient norms, before original gradient clipping, are:

|Term|Chart loss|Chart gradient norm|World-Q loss|World-Q gradient norm|
|---|---:|---:|---:|---:|
|RGB|.138944|.427573|.138108|.701937|
|Anisotropy spatial|9.319e−6|3.302e−4|3.218e−4|7.137e−3|
|Scalar spatial|8.075e−5|2.697e−3|8.226e−5|3.031e−3|
|Energy|0|0|0|0|
|Specular|1.754e−7|2.412e−6|1.687e−7|2.323e−6|
|Diffuse-light chromaticity|.001754|.005825|.001751|.005888|

World-Q's rotation/shared-feature gradients are active; direct kd/metallic/light
groups of that term are correctly zero. It is substantially stronger than the
tiny inherited chart orientation penalty, despite sharing the declared.05
coefficient. These are different objectives, not a pure coordinate relabeling.
Q-versus-RGB gradient cosine is+.258,−.023,+.114 across its three probes; the
interaction varies along the stochastic history. Scalar spatial, RGB and
illumination gradients also interact. Full per-group norms/cosines remain in
the machine-readable report. Three probes are insufficient for global causal
attribution. The original clipping threshold.7 is unchanged; actual combined
post-clipping gradient vectors were not separately archived.

## Work, cost and decision

Exactly8000 new original Adam returns/backwards across two arms,36 component
reverse sweeps,8192000 additional surface-predictor points,2048 visibility rays
and four validation-camera renderings. No new references, NVS, checkpoint
evaluation/model queries or GPU calls were performed by offline analysis.

Native wall times are551.98s /575.61s (world-Q+4.28%, one run each). Peak RSS is
27.73 /32.11GiB. Torch peak allocated CUDA memory is7.053GiB each; peak reserved
memory77.28 /76.80GiB. These are allocator measurements, not total process GPU
memory or a statistically established performance benchmark. All recorded
wall/output/RAM values remain within the approved limits.

**Native execution: PROCEED. Candidate promotion/full reconstruction: FIX AND
PROCEED only after a reviewed scientific change.** Preserve both completed
endpoints. Do not promote this sparse world-Q configuration as a verified global
seam correction, or infer physical recovery from the PSNR improvement.

The smallest next causal test for darkening is a short paired fork holding the
already learned state4000 illumination networks fixed in one arm, against joint
material/lighting optimization with identical material settings and budget.
That would test illumination co-adaptation without importing Bell known-light
settings; it would still not establish true albedo. It needs separate approval.
For smoothness, the observed192-pair single-camera coverage motivates a later
coverage-controlled extension rather than increasing the coefficient from these
results. No such intervention or100000-step restart is implemented/authorized.

## Artifacts and validation

[RGB/components](results/sphere_surface_comparison_v2/RGB-component-comparison.png),
[material/frame maps](results/sphere_surface_comparison_v2/material-comparison.png),
[progression](results/sphere_surface_comparison_v2/progression.png), and
[loss/gradient diagnostics](results/sphere_surface_comparison_v2/loss-gradient-diagnostics.png)
retain consistent display scales. Layout-v2 changes only canvas height; originals
and all numeric results remain unchanged. Endpoints/gradients CSVs, original-copy
hashes and regeneration commands are in the same tracked directory.

Three new offline tests pass,0failures/errors/skips, with Torch/native scientific
imports blocked. They test the frozen decision rules and an analytic RGB metric
case. Existing23 math/visual tests and5 launcher regressions are preserved and
were not unnecessarily repeated. All99 G and106 A snapshot files, original YAML,
corrected-core, v1/v2 freezes and original failed evidence verify unchanged.
The prior seven unresolved reflectance quadrature checks remain unresolved;
no new integration or energy correction is performed here.
