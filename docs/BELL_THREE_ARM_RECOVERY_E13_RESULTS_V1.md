# E.13: first native three-arm Bell material recovery

All three native histories completed2000 original Adam returns and their bounded
endpoint evaluations. **Native lifecycle validation passes; successful anisotropic
material recovery is not established.** Estimator/core corrections improve image
fit and several material errors over original A, but none of the19846 held-view
observations meets the frozen relative-Q target. Bell v3 remains a useful weak-cue
control, not the primary informative paper benchmark.

## Provenance and scope

Execution commit: `c372859dfa38789528517859a49e7f0f132cbdab`.
V2 freeze SHA-256: `4fd53bd58ed55d2900b4e959b1bb8c48ca86b7f26db90aff87aaec87596cf7cb`.
The [v1 scientific protocol](protocols/BELL_THREE_ARM_RECOVERY_E13_V1.md) and
[v2 observer amendment](protocols/BELL_THREE_ARM_RECOVERY_E13_V2.md) were approved
before execution. Fixed endpoint selection uses no held-view or material-truth
model selection. The earlier v1 startup failure and its authorization are retained.

Batch: `/home/isavand/paper/evidence/bell-material-e13-v2-be28c65f719543e09065d728bd2bad71`.
Each arm has its own named child directory, report, completion seal, logs,
checkpoint,128 material exports and12 image archives.

All arms use pinned A `e1e859779301faced46d6476691e20b546c27ae0`, the admitted
known-light Bell v3 oracle, seed6033 and identical initial parameter tensors.
A_original retains original BRDF/sampler/decoder bodies under the common known-
light wrapper; B is A_estimator_v1; C is A_corrected_core_v1. The energy-budget
variant is excluded. The corrected image is
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
with the same admitted FP64-predicate BVH backend for every arm.

Recorded runtime: Python3.8.10, Torch1.8.1+cu111, CUDA11.1, ABI0, A10080GB PCIe,
driver550.163.01, four Torch CPU threads. Original TF32 settings remain enabled.
Inputs/source/root were mounted read-only, non-root, without network access.
Image/binary identities are bound by the reviewed launcher and native receipts;
Codex did not independently inspect the Docker engine or repeat GPU execution.

The oracle mesh is benchmark-defined geometry, not independently measured truth
or an attributable new G reconstruction. Geometry, normals, known v3 illumination,
splits, shader defaults and priors were preserved. A/B declare shader regularizers
on, including effective energy/specular coefficients.05, but renderer reg_mat=false
gates them off. C declares them disabled. No prior contributes in any arm.

## Numerical and lifecycle evidence

|Evidence|Verified result|
|---|---|
|Native training forwards / backwards / Adam returns|6000 /6000 /6000;2000 per arm|
|Training direction slots|4718592000|
|Endpoint image evaluations|36 images;456 shader chunks;589824 primary rays|
|Evaluation direction slots|221193216, below the frozen905969664 panel maximum|
|Material exports|515698 observations/arm;1547094 total;197 predictor batches/arm|
|Checkpoint|One strict own-endpoint roundtrip/arm; no optimizer restoration|
|Initial parameter equality|All108 named parameter tensors match across arms|
|Variant identity|Distinct signature buffers; common non-signature buffer matches|
|First two returns|Zero LR preserves weights and advances Adam; second return changes weights|
|Final state|84 active Adam states at step2000;84/108 parameter-tensor hashes changed|
|Known-light inactivity|24 learned illumination tensors remain unchanged; no human-light model|
|Invalid material/frame counts|Zero across all515698 exported observations in every arm|
|Recorded cleanup|All three cleanup_verified=true; cleanup_errors empty|
|Native arm time|A405.14s; B403.53s; C403.62s; summed1212.29s|
|Torch peak allocated GPU memory|A413.82MiB; B382.48MiB; C314.83MiB|
|Parent peak RSS|9.48–9.49GiB; excludes separate worker peaks|

Torch allocator measurements exclude extension/driver allocations. Recorded cleanup
and checkpoint roundtrip are inspected native evidence, not an independent
Docker-engine cleanup audit or a second checkpoint load by Codex.

Offline verification checks all85 frozen source hashes,667 input identities,486
recorded output hashes, exit codes, authorizations and completion seals. It
independently recomputes physical tensors, domains, frames and orientation cosine
from saved parameter/label arrays, then recomputes all canonical per-view/split
statistics and36 image scores. Float64-versus-float32 differences are below
4.65e−7 in the checked physical quantities. Verification roundoff bounds do not
change scientific criteria; identities, counts and eligibility/validity decisions
remain exact. No checkpoints, camera pickles or TensorBoard events are deserialized
in offline analysis; no model, shader or optimizer is queried.

Native geometry-gate reports show zero origin/direction/intersection/depth/normal
differences at the evaluation views. Saved masks agree exactly with oracle masks.
The full native ray batches were not retained separately, so those native geometry
measurements cannot all be independently recomputed from additional captured rays.

## Image reconstruction versus physical recovery

Held views are11,105,114,17,76; validation58 is reported separately in the CSVs.
PSNR/SSIM use frozen full-frame native uint8 conventions. Linear relative L2 uses
the independent reference and fixed GT foreground mask, pooled over the five
held views. Roughness and parameter errors below are observation-weighted means.

|Arm|Source PSNR(dB)|Source SSIM|Source linear relative L2|Relight PSNR(dB)|Relight linear relative L2|
|---|---:|---:|---:|---:|---:|
|A original|51.6424|.998536|4.7493%|44.6230|9.2107%|
|B estimator|55.5648|.999223|2.0321%|55.6689|2.1275%|
|C core|55.6194|.999233|1.9947%|55.7349|2.0690%|

None meets the predeclared held-source linear-relative-L2 target≤1%. High full-
frame PSNR/SSIM should not be equated with physical success: only24.23% of the
held image pixels are foreground, and the scene has weak specular cues.

|Arm|Relative Q mean|Major width MAE|Minor width MAE|Eligible orientation MAE(°)|Albedo error†|Metallic MAE|
|---|---:|---:|---:|---:|---:|---:|
|A original|.507003|.035132|.078863|29.2873|.027900|.078151|
|B estimator|.361828|.062871|.048305|18.5369|.005561|.021730|
|C core|.364658|.063180|.046501|18.7066|.005752|.021964|

†Mean of each observation's maximum absolute linear-RGB albedo-channel error.
World/local Q means agree to numerical roundoff. Orientation is modulo sign and
axis interchange, evaluated only on17607 eligible observations (GT width gap≥.1).
All19846 held observations remain in Q/material denominators; no invalid frames
were repaired or excluded. These are pixel/view observations on one surface, not
19846 independent objects or unique surface locations. Held-view scoring is not
proof of spatial generalization to unseen surface regions.

|Frozen descriptive target|A passing fraction|B passing fraction|C passing fraction|
|---|---:|---:|---:|
|Relative Q≤.05|0/19846|0/19846|0/19846|
|Eligible orientation≤5°|9.84%|27.66%|18.60%|
|Max-channel albedo error≤.02|49.70%|100%|100%|
|Metallic error≤.03|1.05%|72.64%|72.16%|

Corrections do not improve every quantity: original A has smaller major-width MAE,
while B/C improve minor width, mean Q, albedo and metallic. C's slightly better
photometry does not establish superiority over B: B has lower mean Q error and a
larger eligible orientation pass fraction. One scene, one seed and one endpoint
cannot establish significance or a general method ranking.

## Loss trajectories and compensation

The original NeRFRenderLoss applies its existing cosine RGB-weight schedule
(begin1, peak5, steps0/5000/10000). At zero-based step1999 the multiplier is
2.3807710004578606. Therefore the rise in logged aggregate loss after the initial
fall is not evidence by itself of optimization divergence.

The original Charbonnier is sqrt(sum_RGB(error²)+.001), whose nonzero floor is
sqrt(.001)=.03162277660168379. No coefficient or loss was changed. Dividing each
logged row by its actual original weight gives:

|Arm|Final logged loss|Final unweighted Charbonnier|Mean raw loss, updates1001–1500|Mean raw loss, updates1801–2000|
|---|---:|---:|---:|---:|
|A|.07837436|.03291974|.03370611|.03272183|
|B|.07597348|.03191129|.03206035|.03195217|
|C|.07595715|.03190443|.03206315|.03195004|

B/C are close to the photometric floor but continue to make small photometric
progress. Finite material/shared gradients remain; the stored endpoint and all
milestone Adam counters advance correctly. This neither proves global convergence
nor establishes that more updates will recover physical parameters. No intermediate
material exports exist, so no physical-recovery trajectory is invented.

Held source-light diffuse/specular relative L2 errors are10.69%/32.18%(A),
2.82%/9.69%(B),2.80%/9.57%(C), larger than their total-RGB errors. A descriptive
saved-data diagnostic gives diffuse/specular error-vector cosines−.8497,−.6692,
−.6736: component errors partly cancel. This follows from
cos(e_d,e_s)=(||e_total||²−||e_d||²−||e_s||²)/(2||e_d||||e_s||), using the verified
component sums. It is an additional descriptive diagnostic, not a revised gate.
This supports observed compensation; it does not prove global non-identifiability
or uniquely attribute the failure to optimization versus weak observations.

Initial parameters, first sampled ray records, non-signature buffers, and training-
endpoint/evaluation-endpoint RNG identities match across arms. The frozen policy
continues native RNG without reseeding. Matching RNG identities strengthen this
controlled comparison but do not replace repeated seeds or estimate MC variance.

## Paper artifacts and next action

The [results package](../results/bell_e13/README.md) contains exact endpoint and
per-view tables,50-update training bins, three SVG figures and the file-only
regeneration script reference. All six prescribed views, under both source light
and unseen+90°Y relighting, appear in the montage without exposure adjustment.
The [evidence manifest](evidence/BELL_THREE_ARM_E13_RESULTS_V1.json) records raw
report/checkpoint locations, sizes and hashes; large outputs stay outside Git.

Use this panel as the first completed object-level native weak-cue control, with
physical failure reported alongside improved image fitting. The minimum next
scientific evidence should be the already proposed bounded two-light observability
study, retaining the same points/material controls and original noise-aware
criteria. No additional lighting search or neural optimization was executed here.
If duration must be isolated later, predeclare a separate duration comparison with
physical endpoint exports; these checkpoints alone do not implement exact RNG/
ray-pool resumption. Do not assume longer training resolves the observed ambiguity.

**Decision: PROCEED for the validated native comparison lifecycle. Physical
material-recovery success and readiness of v3 as the primary informative paper
benchmark remain unestablished.** No further execution is authorized by this report.
