# Neural material recovery v2 — authorized segmented completion

The seven remaining histories completed successfully in one separately authorized CPU invocation. The combined catalog now contains all27 prescribed update5000 endpoints, zero execution failures, zero censored histories and zero unstarted histories. Physical recovery remains unsuccessful in many completed histories: **D0 4/9, N0 4/9, N1 1/9**.

This is a completed catalog across two execution segments, not a claim that the original interrupted invocation completed. The [prior incomplete report](NEURAL_MATERIAL_RECOVERY_LOGGED_V2_RESULTS.md), its original JSON and all original artifacts remain unchanged. The researcher attributed the interruption to a token-usage limit and explicitly authorized replay from update1000 to finish all seven. That cause is researcher-reported; the original process exit and termination record remain unavailable.

## Admission, protocol and identities

- Parent results commit: `b953250fe5f6f172caefa1151fb2ff45dcac1def`; branch `feat/neural-material-logged-v2-continuation`.
- Original frozen scientific baseline: `678f3bf3230ea9e36aa97065491466a816c10310`; freeze SHA-256 `dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af`.
- [Pre-execution continuation protocol](protocols/NEURAL_MATERIAL_CONTINUATION_V2.md) and three executed source/protocol/test hashes were bound before the scientific invocation. All348 frozen original source hashes remain identical.
- Positive continuation authorization SHA-256: `a718c226356cd54085d642a5076165a631c8bebf0081821968834f288c82ac5f`.
- Run root: `/home/isavand/paper/evidence/3d2-continuation-v2-a19f831f80b8`.
- Controller root: `/home/isavand/paper/evidence/3d2-continuation-control-a19f831f80b8`.
- Separate offline finalization: `/home/isavand/paper/evidence/3d2-continuation-finalized-a19f831f80b8`.
- [Full versioned evidence](evidence/NEURAL_MATERIAL_RECOVERY_CONTINUATION_V2_RESULTS.json) SHA-256 `f60a3b8fa6db646efec916428c75a3bfb3196da89cd03a3ef3955f3609254135`.
- Offline selection manifest SHA-256: `38fc3a753697fc5f37fb7497286472db937baf0aac89b15f5a9d19a07d65b3e8`.
- Original source inventory identity remains `485c0bca449ebbe5629cc7de04f57b66ecda76d057f5dff7e6a86a3d2b1135fb`.

| Selected continuation artifact | SHA-256 |
|---|---|
| anchor | `cb36512ccf3bb6fc820d3bbdce81f12e0757a921e8afecb321477dc454fc4770` |
| journal | `051979f5e46095f154ce9ced034266170ad238862a578b4f3a4406c744c2f88d` |
| end | `a13488e86d217fc4f61f79ca8b52159759ab6ce9eb7fc3894f1a0e3a5e13a559` |
| process | `a6919582eedef87fc274b09ff0b3e503c455dfc7aea2497cf9ddcc8513240057` |

The review archived by the existing runtime uses pretty JSON, whereas the external selected authorization uses compact JSON. Their byte hashes are both recorded; exact parsed review equality and the archive hash bound by the anchor are verified independently. The archived review SHA-256 is `e31c9176f0e3b264bac38d8c0e910464cc6ec5ad3ab94db9587b618deeaae96c`. No review fields were changed.

## Restoration and scientific preservation

M1/105 has no full checkpoint at its former last logged update4270. The selected own update1000 manifest SHA-256 is `dd61017d34c00d0fbdde667a61a5754384d4b6d363551ad885fb1fd454bc1c18`. Full weights, buffers, Adam counters, first/second moments and effective options were strictly restored from its hash-selected numeric NPZ using allow_pickle=False. Exact compact-state equality was required before continuation. No optimizer reset, historical pickle loading or new initialization was substituted.

Updates1001–4270 were explicitly replayed, then4271–5000 added the remaining progress. All327 preserved replay log records1010,1020,…,4270 matched exactly for loss/pre-update label, physical/encoding diagnostics, gradient norms/groups and logical work counters, both during execution and independent offline verification. There is no historical full tensor state at4270, so full bitwise state comparison at that point is unavailable and is not claimed.

For resume only, the new adapter changes four original setup/range expressions and a private optimizer factory. Reverse normalization recovers the exact original AST; optimizer, rendering, loss, backward, clipping policy (none), terminal query and evaluation operations are unchanged. The six fresh histories use the exact original train_history code object. Their old successful initial admissions and all eighteen completed controls were reused without rerendering. Seven deterministic N1 constructor calibrations were reconstructed and separately recorded; all resulting initial parameters/buffers match the accepted catalog.

The inherited no-grad prefix1000 forward was counted once rather than repeated. Its counter carry was cross-checked using saved logs1000/1010. The exact historical checkpoint1000 clock was unavailable, so log1010 active time113.3752745s was conservatively carried as an upper bound. Resume active time includes this bound plus new segment work; its CPU figure excludes old prefix CPU. Discarded old replay work remains explicitly charged as duplicate physical work, not new logical progress.

All scientific settings remain frozen: A2/atan2 CPU float32; samples512/1024; original24 fitting/84 external observations and independent targets; training tapes3000–3003 and evaluation1000–1003; original seeds101/103/105 and weight seeds10101/10103/10105; Adam LR0.01, betas(0.9,0.999), eps1e-8, zero weight decay; no scheduler, clipping, regularization, calibration tuning or held-out checkpoint selection. No GPU, neural architecture change, reference regeneration or previous successful history rerun occurred.

## Verification and work accounting

- Continuation exited0 with a valid clean terminal seal. Source, authorization, run/configuration identity, all seven logical work counters/snapshot sets, original RNG/thread/memory-limit cleanup and endpoint records pass.
- Verified14056 chained journal records,3423 persistence boundaries,51066441 committed bytes and zero uncommitted tail.
- Verified complete parameter/buffer/Adam catalogs for all19 new checkpoints, including names/shapes/dtypes/counters/options and immutable buffers. Independently loaded and hashed all21 referenced numeric archives (19 new plus2 inherited M1 snapshots) with allow_pickle=False.
- Recomputed all seven new endpoint recovery, radiance and compensation metrics from saved predictions/accepted targets. The20 earlier endpoint records are hash-selected from their previously verified summary; they were not rerendered.
- All nine D0 historical-prefix reproductions and all nine exact N0 controls retain their prior verified status and trajectory/state equality.
- Exactly34000 new Adam returns:3270 authorized replay plus30730 previously unfinished logical updates. Across both invocations,138270 committed physical returns represent135000 logical returns plus3270 duplicate replay returns.
- Combined logical counters equal every original frozen total:135108 shader forwards,135054 backwards,12976848 queried points,6644146176 diffuse/13288292352 specular directions;27 initial admissions,27 prefix/terminal queries and27 evaluations.
- New physical work:34020 shader forwards,34007 backwards,3267600 points,1673011200 diffuse/3346022400 specular directions. No new initial scientific shader/backward admissions were performed; seven constructor calibrations are additional reconstruction overhead.

## All27 prescribed endpoints

Every row below is complete at5000 updates. Fit is the original terminal fitting MSE. Held/relight are independent total linear-RGB relative L2; Q is gauge-invariant relative tensor error; theta is identifiable major-axis error in degrees; albedo is maximum RGB absolute error; metallic is absolute error. A null theta is unidentifiable and fails the unchanged orientation criterion. Full precision, axis errors, components, physical parameters, gradients and trajectories are in the JSON.

| Material | Seed | Arm | Fit MSE | Held L2 | Relight L2 | Q | Theta ° | Albedo | Metallic | Physical pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | 101 | D0 | 5.238543e-07 | 0.006024446 | 0.00921806 | 0.3603701 | 0.3811345 | 0.01749305 | 0.05000208 | no |
| M1 | 101 | N0 | 5.238543e-07 | 0.006024446 | 0.00921806 | 0.3603701 | 0.3811345 | 0.01749305 | 0.05000208 | no |
| M1 | 101 | N1 | 0.03623095 | 0.8330647 | 0.8351463 | 4.729213 | — | 0.42 | 0.1 | no |
| M1 | 103 | D0 | 2.722453e-06 | 0.01224 | 0.0230099 | 1.284056 | 4.818952 | 0.07881163 | 0.186673 | no |
| M1 | 103 | N0 | 2.722453e-06 | 0.01224 | 0.0230099 | 1.284056 | 4.818952 | 0.07881163 | 0.186673 | no |
| M1 | 103 | N1 | 2.699163e-08 | 0.0009611806 | 0.001341107 | 0.01243002 | 0.008535505 | 0.0004180837 | 0.002057044 | yes |
| M1 | 105 | D0 | 4.721516e-07 | 0.005693171 | 0.00864336 | 0.3379996 | 0.4489993 | 0.01618646 | 0.04650896 | no |
| M1 | 105 | N0 | 4.721516e-07 | 0.005693171 | 0.00864336 | 0.3379996 | 0.4489993 | 0.01618646 | 0.04650896 | no |
| M1 | 105 | N1 | 0.02494076 | 0.6679201 | 0.6722497 | 4.729213 | — | 0.42 | 0.9 | no |
| M2 | 101 | D0 | 6.034363e-07 | 0.007213317 | 0.008599627 | 0.02737766 | 0.4059991 | 0.004338565 | 0.008450186 | yes |
| M2 | 101 | N0 | 6.034363e-07 | 0.007213317 | 0.008599627 | 0.02737766 | 0.4059991 | 0.004338565 | 0.008450186 | yes |
| M2 | 101 | N1 | 3.887089e-07 | 0.006150766 | 0.01578127 | 0.05463788 | 0.1217291 | 0.002969208 | 0.0006810069 | no |
| M2 | 103 | D0 | 6.359247e-07 | 0.008429507 | 0.01091216 | 0.04808776 | 0.3362371 | 0.008515778 | 0.01677946 | yes |
| M2 | 103 | N0 | 6.359247e-07 | 0.008429507 | 0.01091216 | 0.04808776 | 0.3362371 | 0.008515778 | 0.01677946 | yes |
| M2 | 103 | N1 | 0.009047536 | 0.5740023 | 0.5951798 | 2.98269 | — | 0.44 | 0.35 | no |
| M2 | 105 | D0 | 3.884423e-07 | 0.006127088 | 0.01561737 | 0.05302218 | 0.1264044 | 0.002740803 | 0.001287603 | no |
| M2 | 105 | N0 | 3.884423e-07 | 0.006127088 | 0.01561737 | 0.05302218 | 0.1264044 | 0.002740803 | 0.001287603 | no |
| M2 | 105 | N1 | 0.02218017 | 0.9301718 | 0.9316176 | 2.98269 | — | 0.69 | 0.35 | no |
| M3 | 101 | D0 | 3.460318e-07 | 0.002457146 | 0.004365107 | 0.01897318 | 1.108684 | 0.002286839 | 0.00660398 | yes |
| M3 | 101 | N0 | 3.460318e-07 | 0.002457146 | 0.004365107 | 0.01897318 | 1.108684 | 0.002286839 | 0.00660398 | yes |
| M3 | 101 | N1 | 3.148072e-05 | 0.07380316 | 0.1796812 | 0.9915249 | — | 0.07361395 | 0.1689026 | no |
| M3 | 103 | D0 | 3.305825e-07 | 0.002254933 | 0.004214841 | 0.02495425 | 1.069566 | 0.001737309 | 0.001532888 | yes |
| M3 | 103 | N0 | 3.305825e-07 | 0.002254933 | 0.004214841 | 0.02495425 | 1.069566 | 0.001737309 | 0.001532888 | yes |
| M3 | 103 | N1 | 6.762313e-07 | 0.003966025 | 0.006200907 | 0.09390875 | 11.61382 | 0.001104546 | 8.983016e-05 | no |
| M3 | 105 | D0 | 1.28828e-06 | 0.01093419 | 0.02014069 | 0.5158988 | 5.170629 | 0.06399368 | 0.1131328 | no |
| M3 | 105 | N0 | 1.28828e-06 | 0.01093419 | 0.02014069 | 0.5158988 | 5.170629 | 0.06399368 | 0.1131328 | no |
| M3 | 105 | N1 | 0.01266182 | 0.4333407 | 0.447133 | 4.474072 | — | 0.57 | 0.65 | no |

Success still requires all five: Q≤0.05, identifiable theta≤5°, maximum albedo error≤0.02, metallic error≤0.03, held-out RGB L2≤0.01. No tolerance was relaxed to admit a near miss.

| Arm | Physical recovery | Held-out RGB pass | M1 physical | M2 physical | M3 physical |
|---|---:|---:|---:|---:|---:|
| D0 | 4/9 | 7/9 | 0/3 | 2/3 | 2/3 |
| N0 | 4/9 | 7/9 | 0/3 | 2/3 | 2/3 |
| N1 | 1/9 | 3/9 | 1/3 | 0/3 | 0/3 |

## Scientific findings

Under this fixed small-panel protocol, N1 recovered physically correct material in only1/9 histories versus4/9 paired D0 histories. This does not demonstrate a reliable neural-representation advantage. D0 and N0 match exactly. All nine N1 outcomes are reported, including failures, with no favorable seed or endpoint selection.

**Complete saturation failures (five histories).** M1/101, M1/105, M2/103, M2/105 and M3/105 have boundary-valued physical outputs and zero gradients in all six material/shared-feature groups at every sampled record from update10 through5000 (500 records/history). Their optimization still returned5000 finite Adam steps. These are completed physical/photometric failures, not exceptions or resource censoring. The exact first unlogged saturation step and the causal mechanism are unresolved; no learning-rate/decoder/sampler correction was made.

**Successful N1 case.** M1/103 passes all criteria: Q0.01243001972, theta0.00853550519°, max albedo0.000418083668, metallic0.00205704421, held-out0.000961180586, relight0.00134110748. Its swapped roughness axes/quarter-turn orientation are evaluated through unchanged physical gauge equivalences. It is a single successful selected-panel history, not population reliability evidence.

**Photometric near misses remain physical failures.** M2/101 has held-out L2 0.00615076644 and good orientation/albedo/metallic, but Q0.05463787894 exceeds0.05. M3/103 has held-out0.00396602481 and relight0.00620090714, but Q0.09390875460 and theta11.61381538° fail. M3/103 raw orientation channels are near zero (last logged about5.57e-7/5.58e-8), with a nonzero centered radius1.41421, finite unit encoding and very small rotation gradient squared norm9.54e-22: this is near-boundary orientation behavior, not evidence of a zero centered-encoding radius.

**Additional M3 failure.** M3/101 approaches the roughness floor (about0.0400000/0.0400042), with Q0.991525, held-out0.0738032 and relight0.179681. All final groups retain finite nonzero entries, but roughness/rotation gradients are extremely small. This differs from the five exactly-zero-gradient collapses.

**Paired comparisons and component compensation.** N1 improves held-out RGB relative to D0 for M1/103 and M2/101 only (2/9); only M1/103 also improves Q (1/9). M2/101 therefore illustrates that a better RGB fit can coexist with worse roughness recovery. N1 M3/103 has held-out diffuse/specular errors0.00150214/0.0155240 despite total0.00396602, and fails physical orientation. All full component residual correlations and unseen-relighting seen/unseen-view errors are preserved. D0 also passes held-out RGB7/9 while physical recovery is4/9; neither representation can use RGB accuracy as a replacement for physical criteria.

This remains a known-plane/known-light, constant-anchor material-field benchmark. It does not establish spatial material recovery or native G-to-A scene reconstruction.

Initialization sensitivity is material-dependent: N1 seed103 succeeds on M1, collapses on M2, and reaches a photometric/physical mismatch on M3; seed101 collapses on M1, nearly passes M2, and approaches the M3 roughness floor; seed105 collapses on all three. No new seeds were introduced. Architecture, orientation encoding, calibration and redundant parameterization differ together; these results do not isolate depth or prove a defect in the BRDF equations.

## Runtime, storage and testing

Accepted environment: Python3.9.12, NumPy1.21.5, Torch2.4.0+cu121 used on CPU, one thread, GPUs hidden. Original900 active seconds/history,10-second endpoint reserve,4GiB RSS/8GiB AS and external timeout28800/10 were retained. A conservative8280s prior-window charge plus new execution remained below the whole-experiment limit.

Controller observed3808.46195 wall seconds (63.47 minutes),3799.36587 user CPU plus7.18425 system CPU seconds, peak RSS1229254656 bytes. The inner controlled runtime recorded3790.31753 wall/3788.32799 CPU seconds. These scopes differ by startup/preflight/teardown. Per-history logical active-time ratios N1/D0 span1.4820–1.5264; M1/105 includes its conservative inherited checkpoint-clock bound and is not an uninterrupted runtime measurement. Original whole-run CPU/peak/exit remain unavailable; no exact combined-process total is invented.

Continuation original outputs:45 files,394548645 logical bytes, including the51.07MB journal and19 new numeric checkpoint pairs. Original artifacts remain separately stored. Large archives, materialized reports and journals are excluded from Git. Offline finalization took24.1629s and executed zero scientific operations.

New focused tests:5 continuation fixtures passed (resume edit boundary, strict numeric Adam restoration/next-step equality, prefix counter carry, explicit authorization, synthetic suffix-only original-Adam execution);2 offline-verifier fixtures passed (no Torch import, hash gate before parsing). **7 passed,0 failed,0 skipped** in the final test executions. An initial test setup failed because its temporary OwnedOutput directory was absent; the disposable fixture was corrected before source freeze and scientific execution. No scientific test or experiment was repeated to tune results.

## Offline replay of verification

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B -m experiments.finalize_neural_material_continuation_v2 \
  --selection /home/isavand/paper/evidence/3d2-continuation-control-a19f831f80b8/finalization-selection.json \
  --selection-sha256 38fc3a753697fc5f37fb7497286472db937baf0aac89b15f5a9d19a07d65b3e8 \
  --output /home/isavand/paper/evidence/3d2-continuation-offline-new-review
```

Select a new nonexistent output directory; the verifier refuses collisions and writes outside original inputs. The [continuation runner](../experiments/continue_neural_material_logged_v2.py) requires the exact separately authorized selection; it must not be rerun against the existing output. The [offline verifier](../experiments/finalize_neural_material_continuation_v2.py) never restores a model/optimizer or calls a shader.

## Recommended next review

Close this bounded comparison with the negative N1 result intact. Next, inspect saved early optimizer/activation/gradient transitions to distinguish saturation mechanisms, and predeclare any subsequent one-factor optimizer or parameterization ablation for approval. Do not automatically lower LR, change initialization/encoding/calibration, add seeds or rerun recovery. No further experiment, GPU work or native neural training was initiated.
