# Publication Gate 9 — Relighting after joint material and light fitting

## Scientific question and inputs

Can materials recovered jointly with illumination on oracle geometry render under an independently known, previously unseen light? Reuse exactly the completed heterogeneous Donut D2, D8-C and D8-T step-10,000 material arrays. D2 is an earlier corrected-core chart recipe with learned light. D8-C and D8-T are the historical matched predictor pair with common initialization policy, objective, tensor renderer and sampling. These are the validated earlier recipes. Final-Chart, Final-Ambient and K1 from Gates 4–6 provide known-light context. Their fitting supports and objective or initialization details differ. Differences across these groups cannot isolate the effect of learning illumination. [Learned-light histories](../results/donut_heterogeneous_training_v1/README.md) · [Gate 6 controls](../results/publication_gate6/README.md)

Use both held-out cameras 28 and 34, with 4,849 and 2,495 positive-facing observations. Use the exact Gate 6 oracle geometry, directions, normals, exposures and same unseen rotated L2. The 7,306-point Gate 6 common subset is primary for across-history comparisons. Also report all 7,344 positive-facing points, the 38 extras and the 32,768-pixel full raster with separate support accounting. The admitted Gate 6 independent renders and original masks supply targets and evaluation cohorts. The renderer reads spatial materials and geometry plus the explicitly specified unseen light. Learned light arrays and the unseen target images are mounted only for offline host analysis. Archived view containers are used for saved geometry, raster IDs and original-light parity. Their other labels do not drive rendering or refinement.

Verify receipt, seed, variant, endpoint export, fitting reference and support hashes. Reconstruct the two positive in-plane eigenwidths and sign-invariant director using the accepted Gate 6 world-Q audit. Require tangent, symmetric and reconstructible Q, scalar range and exact archived positions/normals. Use archived direct Q on the tensor arms and raw exported world Q for D2. No substitution or repair of invalid predictions. Archive all validation failures. No checkpoint deserialization, optimizer update, native material shader call or new reference image.

## Independent rendering and evaluation

Reuse the Gate 6 float64 Kurt/Beckmann independent kernel, analytic incident-light function, float32 corrected predicate BVH visibility, and quadrature weights. Evaluate each of the three recovered materials on the existing 256²/512² hemisphere grids. Share visibility across all three. Apply the exact existing Gate 6 deterministic refinement ordering and 1024²/2048² patches, with at most 64 pixels per view and 128 total. Require the unchanged RGB convergence limits of .003 maximum absolute and .002 pooled relative L2. Refine only on the prediction grid disagreement. The archived oracle and known-light estimates are not reintegrated. Preserve coarse/fine captures and replay every refinement and convergence decision file-only. Stop on ambiguity, failed numerical gates or exhausted resources. [Gate 6 integration contract](DONUT_RELIGHTING_GATE6_V1.md)

Report unclipped linear RGB RMSE, relative L2, unit-linear-peak PSNR and diffuse/specular component differences by camera and support. Keep existing near-isotropic, grazing, material-region, boundary, observation-coverage and highlight cohorts. Use fixed exposure and an identical transfer for preview figures only. Report grid error separately from image error. Interpret D8-T versus D8-C as a within-pair comparison; compare D2 as a recipe reference. This evaluation substitutes an unseen light to test the recovered materials in a shared kernel. It does not measure the native estimator on these endpoints or calibrate the recovered fitting light.

Use the saved fitting-light equirectangular exports for raw solid-angle-weighted RGB error, mean RGB and a single global least-squares gain diagnostic. Fit the gain only offline against the training-light reference, report it beside the unaligned result, and exclude it from every image render and score. This distinguishes global amplitude from directional and chromatic error while retaining the absolute radiometric result. No per-channel gain, light rotation, exposure fit, ground-truth material calibration, or held-out selection.

## Exact execution envelope

| Work | Base | Maximum including refinement |
| --- | ---: | ---: |
| Independent shadow directions | 2,406,481,920 | 3,077,570,560 |
| Independent BRDF point-directions | 7,219,445,760 | 9,232,711,680 |
| Primary rays, optimizer updates, native shader calls, checkpoint loads, new reference integrations | 0 | 0 |

One A100 GPU 0, frozen corrected image, four CPU cores and threads, 64 GiB host RAM without swap, 128 PIDs, 25% CUDA allocator fraction and 512 MiB owned outputs. Use the existing foreground 1,800-second watchdog with 30 seconds kill grace and inner 1,740-second limit with 20 seconds grace. Source/root/inputs read-only, non-root, network disabled, no rebuild or pull. Report elapsed time, peak allocated GPU memory, host RSS, capture hashes, work counts and cleanup. One launch after explicit authorization bound to the committed freeze, image and fresh output. No retry, tolerance change, expanded work or new training under this approval.

[Foreground launcher](../../experiments/run_donut_learned_light_relighting_gate9_v1.sh) · [Runner and file-only verifier](../../experiments/donut_learned_light_relighting_gate9_v1.py) · [Publication plan](../results/publication_gate9/README.md)
