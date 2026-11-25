# Publication Gate 7 — Upstream NeRO Stage II on the Donut acquisition

## Method and scope

Use the official NeRO synthetic-scene branch at commit 89166a96ea81141dfeec5ba0fc541ccaec92aae4, verified against the public upstream ref and the local Git objects. The 44-file import and configuration closure is exported byte-for-byte to a separate read-only input mount. The project does not vendor upstream source in this commit. The external arm uses the native isotropic MCShadingNetwork and released synthetic helmet Stage II recipe as a predeclared parameter choice. The published full pipeline also estimates geometry and light. The result is a controlled, oracle-geometry, known-light Stage II recipe comparison. [Release source](https://github.com/liuyuan-pal/NeRO/tree/89166a96ea81141dfeec5ba0fc541ccaec92aae4) · [Upstream synthetic configuration](https://github.com/liuyuan-pal/NeRO/blob/89166a96ea81141dfeec5ba0fc541ccaec92aae4/configs/material/nerf/helmet.yaml) · [Attribution and interpretation](../results/publication_gate7/README.md)

## Matched inputs and declared differences

- Reuse all 15,191 Gate 4 fitting observations on cameras 0/50/1/45 and 7,306 held-out observations on 28/34 in the exact frozen order. Replay the 5,120,000 actual fitting indices. Provide only positions, normals, outgoing directions and observed RGB to fitting. Keep labeled physical parameters and held-out arrays outside the optimization inputs.
- Same oracle geometry, corrected predicate_fp64 BVH image and native near-surface offset, depth cutoff and black misses. Supply fixed analytic L2 at the upstream outer-light callback and zero inner light. Disable human light as in the synthetic YAML. Keep learned light parameters frozen and outside Adam. This acquisition change removes illumination fitting for every arm.
- Keep native NeRO isotropic GGX NDF, Schlick geometry and Fresnel, original perturbation/range and diffuse-white priors, original renderer Charbonnier, original Adam defaults (1e-8 epsilon, no AMSGrad, no clipping), native warm-up/cosine, and native initialization at seed 6033. The scheduled peak is .0005 after 1,000 warm-up steps on the stock 100,000-step horizon. Fit to the one fixed 10,000-update endpoint. An explicit runner supplies the saved observation pool and its index stream and uses the native loss functions and scheduler.
- Keep NeRO's sRGB fit domain. The same unscaled linear targets pass through upstream's unclipped sRGB transfer. K1 and the matched chart/ambient recipes use a linear fitting loss. The fitting domain, priors and optimizer are declared complete-recipe differences. Image error and exposure are defined in the same linear convention for all arms.
- Use 512 diffuse and 1,024 specular directions per shading point. The NeRO synthetic preset supplies 256 specular samples. The exposed count is increased to match training visibility slots. Its sampler, weights, clamps and BRDF are retained. Native evaluation uses one deterministic pass per light/camera. Accepted project endpoints use eight L2 or 32 unseen-light draws for their stochastic specular estimators. Report the different evaluation costs alongside the matched fitting budget.

## Readout and guards

Keep the upstream sRGB transfer and observe its exact pre-transfer total, diffuse and specular tensors during endpoint scoring. Check the six-call transfer inventory, exact additive total and consistency with the native display output. Save raw linear arrays and the native scalar NeRO roughness separately. Require intact upstream source namespaces, the exact backend, finite first forward/backward and material gradients, deterministic fitting-index identity, the exact work counts, unchanged endpoint weights during evaluation and cleanup.

Score fitting and held-out L2 on the Gate 4 common pool. Relight both held-out cameras under the same frozen E.14 rotation used in Gate 6. Compare with the existing independent reference and accepted native project endpoints. Report common foreground, all 7,344 positive-facing foreground pixels, the 38 extras and full raster including known black support. Preserve the Gate 6 cohort masks and radiometric metrics. Project independent tensor rendering is separate context because the Kurt integrator does not represent NeRO's native GGX. Report per-camera and pooled relative L2, RMSE, fixed unit-linear-peak PSNR, diffuse/specular components where available, and scalar material errors. Compare linear base color and metallic with a model-mismatch caveat. Keep NeRO roughness in its native units. Use identical sRGB previews, unit exposure and a common linear-error map scale.

Use the frozen files and a fresh output. Preparation runs only metadata, static and synthetic checks. A separate exact commit/freeze/source/image/output authorization is required before either training or evaluation. On first failure stop and preserve partial evidence. No reruns, pretrained checkpoints, geometry fitting, new reference rendering, or best-checkpoint selection.

| Work and resources | Ceiling |
| --- | --- |
| A100 GPU 0, host isolation | 4 threads/cores, 64 GiB RAM with no swap, 128 PIDs, .25 CUDA allocator, 512 MiB output |
| Watchdogs | 3,600 s foreground plus 30 s kill grace, 3,540 s inner plus 20 s kill grace |
| Fitting | 10,000 backwards and Adam returns, 512 pixels per return, 7,864,320,000 training shadow slots |
| Evaluation | 62 shader calls over 29,841 points, 45,835,776 evaluation shadow slots, 8 camera/light pairs |
| Total shader and visibility work | 10,062 calls and 7,910,155,776 shadow slots |
| Primary rays, reference renders, checkpoint loads | 0 / 0 / 0 |

The existing corrected image and read-only/no-network non-root runtime are mandatory. Time on this workload is unmeasured. Report actual wall time, GPU allocation, parameter counts and visibility slots. Matching supervision and update counts controls the fitting budget. This one known-geometry scene and one unseen light support a conditional recipe comparison. They do not establish a ranking against anisotropic external methods or causally separate the effects of BRDF family and training recipe.
