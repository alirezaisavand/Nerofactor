# Research priorities after Gate 11 preparation

This versioned addendum preserves every backlog file bound by earlier freezes.
It authorizes no scientific execution. Bell is development evidence.

1. Complete the isolated O1 perfect-mask diagnostic against archived O0, with the
   original300k endpoint and all schedules intact. Determine whether precision
   improves without sacrificing foreground recall, depth or signed normals.
   O1 is established background supervision used as a diagnostic, not a new
   realistic segmentation method or proof of robustness to SAM2 omissions.
2. Admit a prospectively frozen, fitting-RGB-only SAM2 acquisition before a common
   imperfect-mask R0/R1 comparison. Preserve raw scores, prompts/order, conservative
   foreground and explicit uncertainty. No GT/held-view decisions or complement-
   as-background shortcut. Acquisition inference and both fits need separate
   approvals. Reuse available source/environment/weights; no installation now.
3. Keep multi-view confidence-aware free-space SDF as a separate conditional
   intervention. Current measured mesh excess has stable high foreground opacity;
   direct SDF is not yet the first supported target. Compare it on matched
   confidence support if needed; keep outside-positive sign/scene-unit margins
   explicit and avoid hard carving or copying opacity coefficients into SDF units.
4. Preserve anisotropy-aware directional appearance and ambient tensor integration
   as future methodological opportunities. Relevant original components are
   `geometry_g/network/field.py` (SDF and AppShadingNetwork appearance/LUT/light),
   `network/renderer.py` (normals, alpha/foreground/background integration), and
   `network/loss.py` (coefficient schedules). Do not implement them in O1 or infer
   that isotropic approximation caused a particular defect without intervention.
5. Keep background-NeRF interactions, geometry–material refinement and adaptive
   geometry schedules separate. Existing checkpoints establish temporal changes,
   not individual schedule causality. Any adaptation requires prospective controls
   and full-schedule confirmation, without choosing earlier held-view checkpoints.
6. Freeze a resulting method before independent confirmation on another dense
   scene or a newly reserved calibrated acquisition. Historical Bell views and
   prior Donut experiments are not pristine test evidence. Reusing an object
   requires disclosure and a prospectively reserved acquisition; prefer a fresh
   dense scene for cross-object geometry confirmation. Do not launch new data or
   geometry training as part of this preparation.

See the [O1 full-run protocol](protocols/BELL_BACKGROUND_OPACITY_GATE11_V1.md),
[realistic acquisition/R0-R1 protocol](protocols/BELL_IMPERFECT_MASK_ACQUISITION_GATE11_V1.md)
and [readiness report](results/publication_gate11/READINESS.md).
