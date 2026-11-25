# Publication Gate 2 readiness

**K1-Dense remains the best overall reconstruction reference.** The accepted three-seed match establishes Final-Ambient as the chart-independent candidate. Compared with Final-Chart, ambient consistently improves Q tails, minor-width error, orientation and held-out RGB. K1-Dense retains the lower Q p95 and errors in both widths at all three seeds. The method decision rests on a single known-light oracle acquisition. [Accepted three-seed results](../bell_publication_seeds_v1/README.md)

Gate 2 isolates geometry with the historical Donut reconstruction. The frozen six-camera audit traces both meshes at the admitted resolution, checks the oracle replay and independent FP64 witnesses, accounts for every pixel, and tests forward/reverse surface correspondence. It transfers the existing oracle field for evaluation and separates Q shifts from changing the sampled oracle point and from changing the normal frame. Numerical thresholds, work ceilings, and the selected reconstruction come from the reviewed geometry protocol. The historical PLY remains unattributed to a producing G checkpoint. [Reviewed protocol](../../protocols/DONUT_RECONSTRUCTED_GEOMETRY_VALIDATION_V1.md) · [Preliminary geometry evidence](../publication_gate1/donut-geometry-readiness-v1.json)

## Remaining publication checks

1. **Cross-geometry representation check.** Use a new same-recipe chart/ambient comparison on a second admitted geometry. Carry the orientation benefit and the width and boundary limitations forward as separate outcomes.
2. **Reconstructed-geometry material recovery.** First complete Gate 2. Conditional on its coverage and correspondence results, use the same fixed oracle material and L2 illumination on both meshes to measure the geometry-induced RGB floor. Recompute visibility on each mesh. Use a later matched recovery pair to measure how that floor changes physical recovery.
3. **Unseen-illumination relighting.** Evaluate fixed recovered materials with an independent unseen-light reference, shared linear radiometry, and explicit common-foreground and full-raster errors.
4. **External baseline.** Match observations, masks, and compute accounting. Document the baseline BRDF family and any comparability limits for material-space metrics.
