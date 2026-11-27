# Geometry research backlog after Gate 10.3

Updated 2026-11-27. This separate addendum preserves the original backlog bytes
bound by the completed Gate 10.3 freezes. It authorizes no new scientific execution.

1. **First isolated diagnostic: background opacity with perfect synthetic masks.**
   The 300k fitting panel has 313 mesh false-positive rays; all have `fg_acc>.5`
   in both draws, with zero stable-low-opacity mesh false positives. Full fitting
   rasters also fail precision. This motivates an opacity intervention before a
   direct SDF intervention. Use the existing authenticated original history as
   O0, not another baseline run. Prospectively prepare one new 300k O1 history,
   changing only the additive background term below. This is an oracle-mask
   diagnostic, not a realistic segmentation method or a demonstrated remedy.
2. **Realistic method: same-mask foreground baseline versus confidence-aware trimap.**
   Preserve the intentional foreground-only rationale. Freeze one image-only
   imperfect SAM2 acquisition, keep uncertain pixels unlabeled and compare R0/R1
   on identical foreground and confidence evidence. Do not use Bell oracle hits,
   GT geometry or held views to set prompts, thresholds, margins or losses.
   Binary historical masks do not recover the missing original confidence maps.
3. **Direct multi-view free-space SDF remains conditional.**
   No stable-low-opacity mesh false positive appears on the current fixed panel.
   This weakens the immediate case for starting with direct SDF supervision, but
   the panel is limited and cannot rule out such errors elsewhere. Revisit if
   volume/zero-level discrepancies persist after a matched opacity intervention.
   Keep confidence support matched when attributing opacity-versus-SDF effects.
4. **Preserve appearance and background mechanisms as separate studies.**
   Geometry-stage anisotropy approximation, background-NeRF compensation,
   curvature/opacity/occlusion interactions and geometry–material refinement
   remain hypotheses. Temporal coincidence with scheduled losses is not a
   component ablation. Do not modify them in O1.
5. **Independent confirmation precedes a robustness claim.**
   All current Bell cameras are development evidence. Freeze the intervention
   before another sufficiently dense scene or a prospectively reserved acquisition.
   No held-view checkpoint replacement, oracle-driven mesh repair or automatic
   G→A admission. The original 300k endpoint remains blocked.

## Prospectively specified O1 diagnostic (not implemented or launched)

Use the same Bell v3 acquisition and 122/1/5 split, seed 6033, exact original G
architecture/appearance, initialization policy, loader/RNG policy, all other
losses and coefficient schedules, learning rate, optimizer and 300,000 updates.
For original binary synthetic mask `m` and foreground accumulation `a`, use

```text
L_mask_O1(s) = w_mask(s) * [mean(abs(m - m*a)) + 1.0 * mean((1-m)*a)]
```

Retain the original full-ray mean, exact absolute foreground expression and its
roundoff/subgradient behavior. The original mask schedule .3→1 by loop index
30,000 multiplies the whole sum. Do not replace it by BCE, separately rebalance
foreground/background means, clip opacity or change entropy/curvature terms.
Only this diagnostic may use the complement of the perfect synthetic mask.

Keep the original 1k lifecycle check and fixed 15k/30k/50k/300k checkpoints;
use the accepted domain-aware extraction and unchanged numerical/scientific
criteria for evaluation. Primary inference is the fixed 300k O1−O0 difference
in full fitting support, sampled opacity and original validation/held-view
geometry, with all exclusions retained. Temporal curves are secondary and
cannot select a replacement endpoint. A future authorization must bind the
new variant/source, full original schedules and exact resource/work limits.

One history previously took about 11 GPU-hours; this is a cost estimate from the
original run, not a measured duration for O1. A realistic same-imperfect-mask
R0/R1 comparison would require two separate histories (roughly 22 GPU-hours
before segmentation/evaluation overhead), without a coefficient sweep. O1
cannot establish robustness to segmentation errors. Background-mask supervision
is established practice; any novelty claim must concern a separately validated
method, not the existence of a background penalty or this replay correction.

See [Gate 10.3 results](results/publication_gate103/RESULTS_V2.md) and the preserved
[imperfect-segmentation clarification](results/publication_gate103/MASK_SUPERVISION_CLARIFICATION_V1.md).
