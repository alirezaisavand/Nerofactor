# Bell mesh-discretization-robust distance weighting v1

Run B1 and B2 using the exact admitted heterogeneous Bell acquisition
and accepted D6 recipe. Preserve the coefficient 0.05, world-Q and normal
transport, the training domain, all distances and normals, the independent
seed-26061 surface sampler, original initial weights, photometric stream,
corrected renderer, learned direct light, legacy priors and 10,000-update
budget. Keep the 1,024 reserved diagnostic edges and the existing Bell
camera, surface and boundary evaluation paths.
[Accepted Bell result](../results/bell_heterogeneous_training_v1/README.md)

## Frozen geometry-only weights

For eligible training edge (e), define the original multiplier
(w_e=(0.02/d_e)^2), where (d_e) is the existing edge-midpoint
surface distance. The Bell median is 22.58139155448564, and the median
of the historical Donut training domain is 4.320295771278477. The two
Donut archives agree exactly on eligible IDs, distances and face pairs.
The medians exclude the already reserved diagnostic edges.

| Arm | Multiplier | Effective ceiling |
| --- | --- | ---: |
| B1 | `min(w, 4 * Bell median)` | 90.32556621794257 |
| B2 | `min(w, 4 * Bell median) * (Donut median / Bell median)` | 17.281183085113908 |

B2 uses scale 0.1913210601239621. Both retain every eligible edge.
Apply the scalar multiplier to the original squared transported
Frobenius residual and take the arithmetic batch mean as in D6.
There is no weight-sum normalization and no use of material or
held-out labels. Runtime uses the original native distance dtype and
the frozen scalar cap/scale.

Of 361,496 training edges, 21 (0.00581%) reach the cap. The 10,000-batch
CPU replay records 76 capped draws over 75 batches and verifies all
original sampler snapshots. The worst single-edge share of geometric
multipliers changes from 99.9472% under D6 to 3.2140% under either cap.
At update 1,696 the corresponding shares are 99.9435% and 2.9085%.
The shortest pair, ID 156225, also appears at updates 2,885, 5,961 and
9,993. B2 scales all capped weights uniformly and shares B1's geometric
weight proportions. The geometry archive contains every eligible
weight and every batch's maximum share, maximizing edge ID and cap
count. The tracked summary records the digest, source archives and
distribution quantiles.
[Weight replay evidence](../evidence/BELL_DISTANCE_ROBUST_V1_WEIGHTS.json)

## Execution and analysis

Run both arms fresh, sequentially, with the same native compatibility
gate, initial state/device and original Adam schedule as D6. Record the
strongest single edge's actual contribution to the added loss,
(0.05 w_e lVert Q_0-RQ_1R^mathsf{T}Vert_F^2/128), and its ID in
each batch. When sampled, record the shortest pair's contribution as
well. Log preclip joint gradient norm, sorted fitting-batch width
minima, means and maxima, and minor widths at or below 0.0401. The
diagnostics use detached values from the original forwards and
request no extra gradients.

Keep all D6 metrics and render checkpoints at 0, 2,000, 4,000, 6,000,
8,000 and 10,000, with the five original checkpoint roundtrips. Use
the fixed 10,000 endpoint. Replay the logged edge IDs against the
deterministic sampler, inspect collapse onset and all four shortest
pair events, and compare actual loss contributions and widths.
Compare against completed Bell D2 and D6 for physical Q, both widths,
eligibility and common-subset orientation, linear RGB, scalar
materials, illumination, low-support and boundary cohorts, and
reserved-pair spatial consistency. Report forward CUDA timing and
total wall time, which includes logging. Stability relative to D6
and improvement relative to D2 are separate outcomes.

The exact one-time batch uses the existing corrected image, A100
GPU 0, four CPU threads, 64 GiB RAM without swap, 128 PIDs, 25% CUDA
allocator cap, and 8 GiB output per arm. Use a 7,500-second
foreground watchdog with 30-second kill grace and 3,600 seconds
per arm, with a 3,540-second inner cap and 20-second kill grace.
Read-only input/source/root, non-root and no network. Stop at the
first mandatory failure and preserve the distinct partial batch.

| Batch limit | Count |
| --- | ---: |
| Original Adam returns | 20,000 |
| Shader calls | 24,960 |
| Shadow-ray slots | 19,366,526,976 |
| Diagnostic reverse sweeps | 60 |
| Training surface pairs | 2,560,000 |
| Evaluation-only boundary predictor calls / points | 24 / 59,376 |
| Detached edge-log rows | 20,000 |
| New primary rays and independent references | 0 |

This isolates distance clipping and then median scaling. Robust
tensor TV and area-consistent discretization remain subsequent
methodological options. The original results and their source-bound
replay remain at the accepted commits.
