# Publication Gate 9 — Unseen-light transfer after joint fitting

**PROCEED.** The independently rendered D8-T materials reduce unseen-light RGB error by **40.7%** relative to the matched D8-C control. D2 gives the lowest pooled error of the three learned-light recipes. These results extend the representation finding to material transfer after joint material and directional-light fitting. [Frozen protocol](../../protocols/DONUT_LEARNED_LIGHT_RELIGHTING_GATE9_V1.md)

## Unseen-light appearance

The renderer used only the saved step-10,000 material fields, oracle geometry and the specified unseen rotated L2. It used the same independent Kurt/Beckmann integration, incident visibility, exposure and Gate 6 target for all methods. The results below use 7,306 common foreground points, unclipped linear RGB and unit-linear-peak PSNR. Bold marks the best pooled result among the three joint fits. The analysis also covers all 7,344 positive-facing pixels, the 38 extras and the 32,768-pixel raster using the archived support. [All per-camera, component and cohort metrics](results-v1/image-metrics.csv)

| Training recipe | Camera 28 RGB L2 | Camera 34 RGB L2 | Pooled RGB L2 | Pooled RGB RMSE | PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| D2, joint | .15614 | .16797 | **.16040** | **.00783** | **42.13 dB** |
| D8-C, joint | .42652 | .32430 | .39365 | .01921 | 34.33 dB |
| D8-T, joint | .25122 | .19690 | .23358 | .01140 | 38.86 dB |
| K1, known light | .12378 | .19971 | .15476 | .00755 | 42.44 dB |
| Final-Chart, known light | .25842 | .26781 | .26176 | .01277 | 37.88 dB |
| Final-Ambient, known light | .16944 | .12999 | .15671 | .00765 | 42.33 dB |

D8-T improved over D8-C on both cameras. Its pooled specular relative L2 fell from .5500 to .3199. Diffuse error fell from .1561 to .1459. On relighting highlights, error fell from .4433 to .2359. On the low-coverage cohort with no nearby observation in another fitting view, it fell from .4091 to .2453. D8-T's boundary-band error remained .3132. Both ambient and chart predictions retain errors in the same acquisition, so the matched gain alone does not establish recovery at material boundaries. D2 and the known-light methods differ in recipe or fitting support. Their close pooled scores and differing per-camera rankings are descriptive comparisons.

Preview order is reference, D2, D8-C, D8-T, K1, Final-Chart and Final-Ambient. Every preview uses the same exposure and display conversion. [Camera 28](results-v1/camera-28-reference-D2-D8C-D8T-K1-chart-ambient.png) · [Camera 34](results-v1/camera-34-reference-D2-D8C-D8T-K1-chart-ambient.png)

## Learned fitting light and numerical checks

Solid-angle-weighted fitting-light relative L2 was .2110, .4024 and .2759 for D2, D8-C and D8-T. A single diagnostic gain reduced these only to .2019, .3959 and .2732. This leaves substantial mismatch after amplitude alignment. The gains were calculated offline and never used in the unseen-light renderer or its metrics. The experiment tests transfer of recovered materials under an independently specified light. It does not establish absolute illumination recovery or an end-to-end relighting result on reconstructed geometry.

Native runtime was 145.2 seconds on A100. Cleanup and all 16 capture identities verified. The run traced 2,437,939,200 shared shadow directions and evaluated 7,313,817,600 BRDF point-directions. It used no primary rays, checkpoint loads, optimizer updates, native shader calls or new oracle reference integration. Six camera-34 pixels used the frozen 1024²/2048² refinement. All convergence gates passed. Across the two cameras and three fields, the largest maximum grid RGB difference was .002332 (limit .003), and the largest pooled relative grid difference was .001752 (limit .002). [Comparison and convergence records](results-v1/comparison.json) · [Artifact manifest](results-v1/artifact-manifest.json)

The first offline replay stopped on three scalar Q reconstruction residuals from the two runtimes. The largest difference was 4.16 × 10⁻¹⁷. The versioned replay allows one binary64 epsilon at unit Q scale for that diagnostic only. It checks all support counts exactly and preserves the 10⁻⁶ scientific Q threshold and the frozen convergence decisions. The saved-capture replay and eight focused tests passed. [Verification record](results-v1/verification.json)

## Publication decision

Retain D2 as the strongest of the learned-light recipe references on this Donut transfer test. Retain the D8 chart/ambient comparison as representation evidence. For the next publication milestone, prepare the authenticated Bell G→A handoff and its split-preserving geometry admission. Use the dense Bell v3 acquisition for a separate ShinyNeRF appearance and tangent comparison once a project reconstruction can be evaluated on the identical split. An exact known-versus-learned Final-Ambient comparison remains the isolated follow-up for attributing illumination effects. [Publication plan](README.md)
