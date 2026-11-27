# Geometry research backlog

Updated 2026-11-27. Current Bell geometry admission and G→A handoff remain blocked
by the accepted Gate 10.2 silhouette criteria. This backlog integrates the
researcher's imperfect-segmentation clarification; it authorizes no native
evaluation, new geometry optimization or segmentation generation.

| Priority | Work item | State and completion condition |
| --- | --- | --- |
| 1 | Preserve original scientific rationale | Integrated: foreground-only mask, curvature and opacity losses intentionally mitigate anisotropic geometry degeneration under approximate geometry-stage reflectance. SAM2-unlabeled pixels are unknown. |
| 2 | Exact Bell v3 mask provenance | Complete file-only audit: 128/128 generated masks equal oracle primary-ray hits; 123 staged masks match producing freeze; 122 fitting, one validation, no held views staged. These are synthetic reference masks, not SAM2 masks. |
| 3 | Historical SAM2 confidence recovery | Scoped inspection complete: active script, prompt/threshold/propagation/export and predictor postprocessing recorded. Original score maps not recovered; binary PNGs cannot recover them. Exact per-scene producing recipes and any external archives remain unresolved. |
| 4 | Gate 10.3 checkpoint evolution and opacity | Continue existing preparation unchanged: 15k/30k/50k/300k, fixed 300k endpoint, existing mesh/volume panels and limits. Native execution remains pending its separate exact authorization. Do not repeat accepted Gate 10.2 or the original 300k optimization. |
| 5 | Select the first realistic background mechanism | Pending Gate 10.3 evidence. Stable high background opacity motivates trimap opacity; low opacity with erroneous mesh support motivates direct free-space SDF. Sampling flips, domain effects and generalization remain separate explanations. |
| 6 | Imperfect-mask acquisition and trimap design | Prepared conceptually. Future recipe must preserve original foreground, acquire scores/prompts reproducibly, admit conservative background with dual thresholds/margins/consensus and abstain on uncertainty. No GT or held-view tuning. |
| 7 | Multi-view confidence-aware free-space SDF | Prepared conceptually. Fitting cameras only, corroborated background, soft outside-positive constraint, supported domain, conflict abstention and separately frozen query/weight budget. No hard visual-hull carving. |
| 8 | Cost-effective matched optimization comparison | Future only: reuse O0; first compare same imperfect masks under R0 and one of R1/R2. Optional O1 perfect-mask binary diagnostic isolates objective under accurate masks. Other mechanism and extra seeds require a separate frozen follow-up. |
| 9 | Independent confirmation and other mechanisms | Freeze chosen method before a new dense scene or reserved acquisition. Keep directional anisotropic appearance/ambient tensors, background-NeRF interactions and geometry–material refinement as separate future studies. |

Candidate order is **existing foreground-only baseline → perfect-mask binary
diagnostic control → imperfect-mask confidence-aware trimap → multi-view
confidence-aware free-space SDF**. This is a hierarchy of controls and candidates,
not a claim that a binary background loss is the preferred realistic method.

The [Gate 10.3 preparation](results/publication_gate103/PREPARATION.md),
[scientific clarification and comparison matrix](results/publication_gate103/MASK_SUPERVISION_CLARIFICATION_V1.md),
[mask audit](results/publication_gate103/mask-provenance-v1.json) and
[SAM2 search scope](results/publication_gate103/sam2-artifact-search-v1.json)
contain the evidence and prospective definitions. Completed extraction,
checkpoint authentication, loss trajectories and Gate 10.2 analysis are reused.
