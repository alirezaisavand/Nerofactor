# Publication Gate 8 — Original thesis attribution on Donut

**PROCEED.** Under the same known-illumination Donut acquisition, K1 reduced original-light held-out RGB error by **28.4%**, unseen-light RGB error by **28.3%**, and held-out Q p95 by **21.1%** relative to the pinned thesis shader. Original A retained smaller sorted-width MAEs and less signed width bias. All 22,497 endpoint observations had valid physical exports. [Attribution inventory](README.md) · [Frozen protocol](../../protocols/DONUT_THESIS_ATTRIBUTION_GATE8_V1.md)

## Matched controls and renderer diagnostic

One A-Original-Oracle history ran for exactly 10,000 updates at seed 6033. It reused the original proposals, half-angle decoder, NDF/PDF, estimator reduction and legacy regularization code under the Gate 4 acquisition adapter. Verification matched K1's initial learned state and buffers apart from the required behavior tag, fitting indices and learning-rate stream. All raw material heads matched bit-for-bit at step zero. Maximum initial world-Q difference was 1.73 × 10⁻⁷ in Frobenius norm, and initial pooled held-out RGB relative L2 against the target was 0.8954 for original A and 0.8415 for K1.

At identical oracle materials, the 24-point, 32-draw diagnostic gave native versus independent RGB relative L2 of **0.1610** for original A versus **0.00873** for K1. Original A exceeded the unchanged agreement envelope on 37 of 72 RGB components, compared with zero for K1. Both draw ensembles met the precision check. The measured discrepancy is part of the frozen attribution design, which requires finite, nonnegative, additive native captures before fitting. This evidence supports an aggregate difference in the rendering computation even before material optimization. [Full comparison](results-v1/comparison.json)

## Fixed endpoint

Image rows below use relative L2 over the common foreground and unclipped linear RGB. Q errors are pointwise relative Frobenius errors. PSNR uses a unit linear peak. Orientation compares 7,065 commonly eligible held-out points out of 7,306.

| Metric | Original A | K1 |
| --- | ---: | ---: |
| L2 fitting RGB relative L2 | .07955 | .04972 |
| L2 held-out RGB relative L2 / RMSE | .19988 / .00983 | .14302 / .00704 |
| Held-out Q median / p95 | .26115 / 1.24745 | .24408 / .98388 |
| Held-out major / minor width MAE | .04026 / .01457 | .04654 / .01502 |
| Held-out major / minor signed width bias | +.00064 / −.00046 | −.02581 / −.00994 |
| Held-out common orientation median / p95 | 6.04° / 70.65° | 3.00° / 59.11° |
| Held-out max-channel albedo MAE / metallic MAE | .05028 / .04280 | .02838 / .04063 |
| Unseen-light RGB relative L2 / RMSE | .21601 / .01054 | .15490 / .00756 |
| Unseen-light PSNR | 39.54 dB | 42.43 dB |

Original A's held-out Q median reached .2504 at 4,000 updates, rose to .3075 at 8,000, and ended at .2611. Its Q p95 increased from 1.0478 at 2,000 to 1.2475 at 10,000 even as held-out L2 RGB error fell from .2324 to .1999. K1 improved L2 image error across the saved checkpoints and kept Q p95 below 1.0 from 2,000 through 10,000. K1's stronger negative width bias coexists with a smaller positive minor-width error tail (p95 .02708 versus .03840). [Trajectory figure](results-v1/held-out-trajectory.pdf)

At both held-out cameras, the inferred signed major-width jump was .0073–.0167 against a true step near .12. Both methods strongly attenuated the boundary. Q gains also varied by region. On camera 28, K1 worsened median Q in material region 0 (.2207 to .3105) and improved it in region 1 (.2486 to .1005). On the 241 near-isotropic points, median Q improved from 1.5527 to 1.1294 and remained high. [Camera, boundary and coverage cohorts](results-v1/comparison.json)

## Unseen light and numerical scale

On cameras 28 and 34, the native unseen-light errors were .18569 and .26295 for original A versus .12427 and .19947 for K1. K1 reduced the pooled diffuse/specular errors from .2822/.2684 to .1194/.2223. The gain persisted on the expanded 7,344-pixel foreground, including the 38 extra foreground pixels, and full 32,768-pixel raster. Full-raster RMSE fell from .00504 to .00357. In the 51-pixel grazing cohort, relative L2 fell from .5764 to .1744. On the 765 pixels with no nearby observation in another fitting view, it fell from .3760 to .3217. These cohorts use the original Gate 6 masks.

Per-camera native RGB standard-error RMS was .00038–.00043, and the independent reference's grid disagreement RMS was .000009–.000011. Both were smaller than the pooled unseen-light RMSE gap of .00298. These numerical checks describe the measured draws and grids. [Unseen-light comparison and common-scale error maps](results-v1/unseen-light-comparison.pdf)

## Evidence and publication scope

Execution completed with verified cleanup, 12,288 L2 and 480 unseen-light shader calls, 9,885,130,752 total shadow slots, and 30 diagnostic reverse sweeps. All 893 native artifacts verified. Original A took 817.3 seconds including relighting on A100, with 0.393 GiB peak allocated GPU memory. The 839.2-second archived K1 measurement covers its Gate 4 fit and L2 evaluations. Its Gate 6 relighting is separate. The first offline replay rejected a one-binary64-ULP difference in the derived pooled limit. Replay V2 permits that difference for this scalar alone and preserves every scientific decision, value elsewhere and the frozen sources. Eight focused replay tests passed. [Execution and replay verification](results-v1/verification.json) · [Artifact identities](results-v1/artifact-manifest.json)

Use this comparison to attribute the improvement to the **combined original-to-K1 correction package** under the fixed acquisition. The benchmark targets use the corrected law. The match controls acquisition and optimizer settings while the interventions change the renderer and the computed material-prior proxies together. The experiment measures neither an individual correction effect nor full historical thesis-pipeline performance. The matched Chart/Ambient results address the separate representation question. For a subsequent external comparison, the released ShinyNeRF method is a candidate for held-out anisotropic appearance and eligible tangent direction after camera, mask and radiometry checks. Preserve the NeRO full-schedule sensitivity as a distinct reviewer control. [External feasibility](README.md)
