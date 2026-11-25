# Publication Gate 4. Matched Donut material recovery

Run K1-Oracle, K1-Reconstruction, Ambient-Oracle, and Ambient-Reconstruction in that order. Compare geometry within each recipe at the fixed 10,000-update endpoint. The accepted Gate 3 fixed-material discrepancy motivates a test of photometric compensation through material bias. The historical reconstructed mesh remains provenance-unverified. [Research proposal](DONUT_MATCHED_GEOMETRY_RECOVERY_PROPOSAL_V1.md) · [Gate 3 results](../results/publication_gate3/README.md)

## Exact inputs and scene adapter

The freeze binds the corrected image, both meshes, original camera calibration, admitted Donut v2 oracle RGB and material arrays, every Gate 2 array, and all Gate 3 captures. Use Gate 2 common hits with a valid transfer and positive facing on both meshes, preserving row-major pixels and camera order. Keep camera 1 pixel 8,239 in the common set. The geometry-specific scene adapter supplies the frozen first-hit position and shading normal, the same outgoing camera direction, and byte-matched original oracle RGB. Recompute direct visibility on each mesh. No primary rays or new references are generated. All excluded pixels, original masks, Gate 2 reasons, and the boundary/grazing/degenerate/depth cohorts remain in the saved coverage accounting.

| Camera | Split | Observations in every history |
| --- | --- | ---: |
| 0 | Fitting | 4,513 |
| 50 | Fitting | 2,605 |
| 1 | Fitting | 4,131 |
| 45 | Fitting | 3,942 |
| 28 | Held out | 4,830 |
| 34 | Held out | 2,476 |
| Total | 15,191 fitting / 7,306 held out | 22,497 |

Only fitting positions, normals, viewing directions and observed oracle RGB form the optimizer pool. The recorded index tape follows the accepted seed-6033 permutation, uses 512 observations per update and has the same frozen SHA-256 in all histories. Use the 24 evenly spaced common fitting pixels in camera 0 for 32 native draws per mesh. This no-gradient compatibility gate compares each geometry's independent Gate 3 RGB and grid error on those exact material points. Gate 3 RGB stays in a separate diagnostic array. Held-out RGB, boundary labels, correspondence weights and oracle materials do not enter the optimizer.

## Native recipes and publication endpoints

K1 inherits the original corrected-core chart network, native specular/diffuse proposals and the K1 legacy material and white objectives. Ambient inherits the accepted Final-Ambient tensor renderer and world-Gaussian proposal, geometry-only D9 projected-world-X weak start and scalar-compatible legacy plus energy/specular/white objective. Both use seed 6033, analytic L2, zero indirect, no point light, the accepted Charbonnier RGB objective, original Adam/AMSGrad, native warm-up/cosine schedule and gradient clipping. No learned-light groups receive updates. Each history freshly constructs its full predictor and must reproduce the accepted initial weight and buffer hash for its recipe. The native ambient admission check still instantiates its chart sibling inside the diagnostic, and verifies fixed-Q RGB, sampling, PDF and finite gradient compatibility before updates.

The source-bound adapter changes the scene observations, triangle inputs, compatibility target and evaluation metadata. It omits prior evaluation-only Bell boundary and reserved off-image smoothness queries. Boundary jumps now use the established Donut right/down neighboring pixels on the shared oracle-face adjacency, with the same .02 native-unit and 30° checks. These are scored offline from existing endpoint exports. No optimizer parameter, renderer, loss or training operation changes. Neither pair has the optional added surface-Q term. Historical sources and run artifacts remain byte-identical.

Record initialization and every 2,000-update checkpoint using the original eight draws. Compare recipe-matched arms by Q median/p95, width MAE and signed bias, common eligible orientation, albedo, metallicity, boundary jumps and RGB. Measure initialization shifts across geometry while checking identical weights. Use the archived physical target for each geometry and transport reconstructed-frame tensors to the same-ray oracle tangent plane for paired Q and boundary comparisons. Compare fitting and held-out photometric error against the original oracle RGB, with Gate 3 recomputed on exactly the same support. Report if lower photometric error accompanies worse physical recovery. Fixed-material RGB discrepancy describes its own material condition and does not bound the outcome of inverse rendering.

## Authorized work envelope

| Work | K1 per history | Ambient per history | Batch ceiling |
| --- | ---: | ---: | ---: |
| Adam returns | 10,000 | 10,000 | 40,000 |
| Native shader calls | 12,288 | 12,290 | 49,156 |
| Shadow-ray slots | 9,524,158,464 | 9,524,232,192 | 38,096,781,312 |
| Component reverse sweeps | 30 | 30 | 120 |
| Tensor admission predictor calls and reverse sweeps | 0 | 2 each | 4 each |
| Historical checkpoint loads, primary rays and new reference integrations | 0 | 0 | 0 |

Reuse A100 GPU 0, the reviewed corrected image, four CPU threads, 64 GiB RAM without swap, 128 PIDs, 25% CUDA allocator and 8 GiB owned outputs per history. Run one batch with the outer foreground 14,760-second watchdog and 30-second kill grace, 3,600-second host arm limit, and 3,540-second inner limit with 20-second grace. The trainer roundtrips only its own five newly saved checkpoints per history. The launcher runs non-root, without network and with read-only source/root/inputs. A failed gate stops the batch and preserves completed and partial evidence. One separate exact-source, freeze and unused-output-path approval is required. No retries or threshold/recipe changes.
