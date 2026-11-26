# Publication Gate 7 — Controlled Donut comparison with NeRO

**PROCEED.** On the heterogeneous Donut oracle benchmark, both project methods reduced held-out RGB error relative to official NeRO Stage II under known L2 and under unseen rotated L2. K1 achieved the lowest pooled unseen-light error. Final-Ambient improved over its matched Final-Chart control and achieved the lowest error on camera 34. [Contribution inventory and source attribution](README.md)

## Execution and controls

The single NeRO fit completed 10,000 optimizer returns at seed 6033. The offline verifier confirmed launcher exit code 0, cleanup, the exact fitting-index stream, all 11 sealed outputs and unchanged endpoint weights during evaluation. The fit replayed 5,120,000 ray indices from 15,191 fitting observations. All scoring reused saved arrays and accepted independent references. [Frozen protocol](../../protocols/DONUT_NERO_GATE7_V1.md) · [Artifact and execution identities](results-v1/artifact-manifest.json)

All methods used oracle geometry, the same fitting and held-out observations, fixed analytic L2, zero indirect light, common visibility conventions and fixed linear scoring. NeRO retained its upstream isotropic GGX shader, optimizer and sRGB fitting loss. The project recipes use anisotropic Kurt/Beckmann and linear RGB loss. The training budgets match at 10,000 updates, 512 observations per update and 7.864 billion shadow slots. Native evaluation uses one deterministic NeRO pass and archived project estimates from eight L2 or 32 unseen-light draws. The unseen light is the Gate 6 rotation of L2 by 45° about normalized (1,1,1). [Upstream source identities](nero-source.json) · [Gate 6 relighting](../publication_gate6/README.md)

## Image accuracy

RGB errors below use the common foreground and unclipped linear RGB. Relative L2 is the norm of the pooled RGB residual divided by the norm of the pooled reference. PSNR uses a fixed unit linear peak. The held-out set contains 4,830 observations from camera 28 and 2,476 from camera 34. [All 672 metrics by light, method, camera, support and component](results-v1/image-metrics.csv)

| Method | L2 fit rel. L2 | L2 held-out rel. L2 / RMSE | Unseen held-out rel. L2 / RMSE | Unseen PSNR (dB) | Unseen rel. L2, camera 28 / 34 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NeRO Stage II | 0.0951 | 0.6975 / 0.03432 | 0.9890 / 0.04825 | 26.33 | 1.1519 / 0.5783 |
| K1 | 0.0497 | 0.1430 / 0.00704 | 0.1549 / 0.00756 | 42.43 | 0.1243 / 0.1995 |
| Final-Chart | 0.0748 | 0.2571 / 0.01265 | 0.2621 / 0.01279 | 37.86 | 0.2585 / 0.2686 |
| Final-Ambient | 0.0518 | 0.1649 / 0.00811 | 0.1567 / 0.00765 | 42.33 | 0.1695 / 0.1298 |

K1 and Final-Ambient reduced pooled unseen-light relative L2 by 84.3% and 84.2% versus NeRO. Within the matched pair, Final-Ambient reduced unseen-light error by 40.2%. The ordering persists over all 7,344 positive-facing foreground pixels and the 32,768-pixel full raster, which also includes 25,422 known misses and two below-horizon zeros. No raster support was undefined. The figure applies identical sRGB preview settings and a shared linear error scale. Eleven NeRO camera 28 channels exceed unit range and clip in the preview only. [Relighting comparison PDF](results-v1/held-out-native-relighting.pdf) · [Figure PNG](results-v1/held-out-native-relighting.png)

| Unseen-light cohort | Pixels | NeRO | K1 | Final-Chart | Final-Ambient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boundary band | 852 | 1.596 | 0.155 | 0.338 | 0.203 |
| Relighting highlight | 1,140 | 1.205 | 0.115 | 0.225 | 0.149 |
| Near-isotropic | 241 | 0.539 | 0.202 | 0.092 | 0.070 |
| No nearby observation in another fitting view | 765 | 0.493 | 0.322 | 0.197 | 0.183 |

## Material interpretation

NeRO assigned metallic above 0.98 to 6,951 of 7,306 held-out points. Its fitted mean luminance was 0.0490, close to the oracle's 0.0482, while its fitting diffuse luminance was 0.000264 versus the oracle's 0.02037. Specular contributed 99.5% of the fitted luminance versus 57.7% in the reference. This endpoint is consistent with a specular-dominated compensation for the observed RGB. The experiment does not isolate the role of isotropy, BRDF mismatch, objective or optimization in producing that endpoint. On unseen-light foreground, NeRO's diffuse and specular relative errors were 0.993 and 1.529. The corresponding errors were 0.119 and 0.222 for K1, and 0.119 and 0.225 for Final-Ambient. [Component and metallic diagnostics](results-v1/material-diagnostics.json)

Held-out max-channel base-color MAE and metallic MAE were 0.295 / 0.319 for NeRO, 0.028 / 0.041 for K1, 0.052 / 0.047 for Final-Chart, and 0.034 / 0.045 for Final-Ambient. Those scalar estimates depend on the respective reflectance models. NeRO's native isotropic roughness had held-out p05/median/p95 of 0.159/0.438/0.844. Beckmann widths, orientation and world Q have no direct NeRO counterpart. [Complete comparison and scope definitions](results-v1/comparison.json)

## Compute and publication claim

NeRO recorded 353.2 seconds on the A100, 1,086,986 trainable parameters, 10,062 shader calls, 7.910 billion shadow slots and 0.327 GiB peak allocated GPU memory. Archived K1, Final-Chart and Final-Ambient runs took 839.2, 826.9 and 825.1 seconds respectively, each with about 9.524 billion shadow slots. Those histories include intermediate evaluations. The separate Gate 6 unseen-light work and different evaluation draws limit direct runtime comparisons. [Native counters](results-v1/artifact-manifest.json) · [Archived Gate 4 histories](../publication_gate4/README.md) · [Archived Gate 5 history](../publication_gate5/RESULTS.md)

The evidence supports competitiveness against the official isotropic NeRO material stage on one known-geometry, known-light acquisition with two held-out cameras and one unseen light. The matched Ambient/Chart result supports a representation-level benefit in this acquisition. K1 and NeRO compare complete recipes against Kurt-generated targets. Claims against external anisotropic methods require an anisotropic external comparison. A separate matched A-original run is the smallest experiment to quantify the combined changes from the thesis implementation to K1.
