# Publication Gate 6 v2 — Historical material-batch replay

## Failure evidence

The original launch exited 1 at the restored-field parity gate. Its receipt records one checkpoint load, 320 native shader calls and 238,338,048 native shadow slots, which identify K1 and camera 28. No output arrays or per-field residuals were sealed. Independent quadrature never started. The original failure receipt leaves cleanup unverified. The original authorization, failed output and freeze are preserved at the paths in the failure record. [V1 failure record](../results/publication_gate6/v1-failure.json)

V1 inserted the extra full-raster points into the batches used for historical field parity. This changed batch membership for 93 retained points in camera 28 and 22 in camera 34. The historical final-batch lengths of 222 and 428 changed to 241 and 447. GPU neural evaluation can depend numerically on batch layout. The failure archive cannot establish which field or pixel exceeded the tolerance, or how much error arose from this mismatch. Preserve the gate and measure those values if parity fails again.

## Narrow correction

Retain the full approved V1 scientific definition and artifact identities. For each restored checkpoint, evaluate the material field on the physical inputs from the exact historical pixel order and 512-row batch partition, including the original final partial batch on each camera. Evaluate each camera's 19 additional valid foreground points separately. K1 forms world Q from the corrected rotated tangent frame using the historical encoded axes and the original NumPy float32 export order. Chart and Ambient export their direct world Q. Record the maximum absolute difference and corresponding historical pixel for Q, base color and metallic before applying the unchanged 1e-6 parity tolerance. Both cameras for an arm must pass before relighting that arm.

Scatter the returned physical parameters into the original V1 expanded camera raster order. Pass those parameters into the unchanged native shade methods, retaining their exact V1 chunk sizes, seeds, sample budgets and estimators. Each independent field is derived from those same material parameters. The V2 adapter layers this change over the verified V1 checkpoint loader, quadrature, renderer work tracking and analysis. It adds no optimization, primary rays, reference-only work or supplemental light.

The original V1 analysis definitions remain unchanged. The V2 verifier additionally recomputes the field drift and its maximum-error pixels from the native archive and checks the documented historical batches for each camera. Preserve the physical comparison and report native/independent integration differences according to the accepted V1 protocol. [Preserved V1 protocol](DONUT_RELIGHTING_GATE6_V1.md)

## New execution envelope

A fresh launch needs explicit approval for the V2 commit, freeze hash, corrected image and output directory. Approval must confirm isolated CPU pickle trust for only the same three frozen checkpoint hashes. All other identities remain bound to V1. V1's approval was consumed.

The new material replay has exactly 51 material-only batches and 22,032 material points. Shader work remains capped at 1,440 calls, 1,082,916,864 native shadow slots, 3,077,570,560 independent shadow directions and 12,310,282,240 BRDF point-directions. No threshold, scientific convergence policy, support, camera, material, or reconstruction recipe changes. Resources remain one A100 GPU 0, four CPU cores and threads, 64 GiB host RAM/no swap, PID cap 128, 25% CUDA allocator fraction, 512 MiB outputs, 1,800 seconds foreground with 30 seconds kill grace, and 1,740 seconds inner with 20 seconds kill grace. The intact V1 renderer and analyzer remain unexecuted by the preparation checks. Native parity and independent convergence remain open until the separately authorized run. Preserve any failed evidence and stop at the first mandatory failure.
