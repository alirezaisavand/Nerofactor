# Publication Gate 1. Dense Bell seed replication

Keep K1-Dense as the strongest accepted reconstruction reference and the final tensor pair as the representation ablation. Replicate both comparisons at seeds 6034 and 6035 with six fresh histories in seed order, K1-Dense then Final-Chart then Final-Ambient within each seed. Use 10,000 Adam returns per history, 512 observations per update, 65,419 fitting observations from 16 cameras, and the same 7,082 held-out observations in cameras 105 and 114. The frozen endpoint remains 10,000, with intermediate states every 2,000. [Accepted results](../results/bell_final_tensor_pair_v1/README.md) · [Matched recipe and numerical gates](BELL_FINAL_TENSOR_PAIR_V1.md)

## Exact recipe changes

The seed controls fresh Python, NumPy, Torch and CUDA initialization and the existing NumPy fitting sampler. Record the exact 5,120,000 fitting indices per arm. All three histories in a seed must match the same precomputed index hash. Subsequent seeds have distinct index hashes. The paired world-Gaussian/specular and diffuse tape definitions, evaluation seeds and independent compatibility draws retain their accepted offsets. Lighting stays analytic L2 with zero indirect and no optimizable lighting parameters.

K1 uses the accepted corrected-core chart, original fresh initialization, native shading and objective. The final pair uses the accepted weak geometry-only D9 Q start, shared tensor renderer and K1-derived scalar-compatible objective without the surface-Q term. No oracle material supervision or held-out selection is used. K1 starts without a checkpoint or initial-state anchor. Its completed receipt binds the fresh common base-state digest, which each tensor arm independently reproduces before selecting its representation. Each tensor arm also constructs its reciprocal sibling from identical seed-specific base weights and must pass existing Q, RGB, PDF, gradient and scalar parity gates before its first update. The originals remain available byte-for-byte. The isolated source overlay replaces only the historical initial-state check with the fresh initialization check and sets a local copy of the policy seed. Offline verification compares the reciprocal captures and saved step-0 maps within each seed.

## Execution ceiling

Run one foreground batch after exact-source researcher approval. A failure stops the batch and preserves completed and partial histories with their exit codes. Use the existing corrected image, one A100 on GPU 0, four CPUs and threads, 64 GiB RAM with no swap, 128 PIDs, native CUDA allocator fraction .25 and the existing 8 GiB cooperative output cap per arm. The outer foreground watchdog is 22,200 seconds with 30 seconds of kill grace. Keep original per-arm timeouts at 3,600 seconds on host and 3,540 seconds in container with 30 and 20 seconds of grace, respectively. Source, inputs and root filesystem remain read-only. Run non-root without network access. No retries, extensions, reference generation, geometry training or recovery on reconstructed geometry.

| Work ceiling | K1 per seed | Each paired arm per seed | Full batch |
| --- | ---: | ---: | ---: |
| Adam returns | 10,000 | 10,000 | 60,000 |
| Shader calls | 17,184 | 17,186 | 103,112 |
| Shadow-ray slots | 13,210,853,376 | 13,210,927,104 | 79,265,415,168 |
| Diagnostic component reverse sweeps | 30 | 30 | 180 |
| Extra tensor-gate predictor calls and reverse sweeps | 0 | 2 each | 8 each |
| Boundary predictor calls / points | 12 / 29,688 | 12 / 29,688 | 72 / 178,128 |
| New primary rays / spatial training pairs | 0 | 0 | 0 |

## Decision endpoints

Report paired within-seed ambient-minus-chart and ambient-minus-K1 deltas for held-out Q median/p95, width MAEs and signed biases, orientation p95 on common eligible subsets, eligibility counts, fixed-boundary jump recovery, linear RGB and compute. Report the seeds separately and summarize directional consistency alongside the accepted 6033 results. Use saved camera and cohort metrics, including near-isotropy and sparse support. A repeated matched advantage supports the representation ablation. Promotion as the main method additionally requires a competitive absolute error profile against K1. With three paired seeds on one geometry and one acquisition, emphasize effect sizes and conflicts among metrics. Preserve the held-out set and report all results, including incomplete histories.
