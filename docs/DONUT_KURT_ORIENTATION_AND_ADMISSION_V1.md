# Donut rotation clarification and dataset admission

Blender's **Anisotropic Rotation was preserved**. The accepted source metadata
records an unlinked value of **0.25 turns (90 degrees)**. The packaged Cycles
Principled implementation applies `rotate_around_axis(T, N, rotation * 2*pi)`.
The exported `rotated_Principled_tangent_world` already contains this rotation;
the reference generator uses that array rather than the unrotated tangent.
Adding another 90-degree rotation would change the intended material.

The [file-only verification](results/donut_kurt_reference_v2/orientation-verification-v1.json)
reconstructed the rotated tangents from the original unrotated tangents and
normals, then reconstructed interpolated unoriented line projectors and world-Q
for every saved surface observation. This does not invoke Blender, a BRDF,
visibility queries, a neural model or optimization.

- All 36,864 exported corner tangents match the reconstructed rotation exactly.
- All 22,639 surface labels across six views retain the rotated director.
- Maximum line-projector difference: 6.661338147750939e-16.
- Maximum world-Q difference: 8.326672684688674e-17.
- All 285 frozen source identities and 31 scientific output identities verify.
- The original `.blend` SHA-256 remains
  `fcf8956740d6543782c190ec1c90e1bbef6f3ae36dcc9c3c0aa9b7fd7b742d57`.

The identity replay uses a 1e-12 absolute roundoff check; it changes no scientific
generation or convergence threshold. Director sign is immaterial to the line
projector and Q. This verifies saved source-derived references, rather than
claiming a newly executed Cycles shader measurement.

## Why the new images look different

These differences were explicitly intended in the controlled dataset definition:

| Historical Blender acquisition | New controlled Kurt oracle |
| --- | --- |
| Principled MULTI_GGX, slope widths approximately 0.387379/0.0387379 | Additive Kurt/Beckmann, newly specified widths 0.35/0.08 |
| Original HDR plus point light and indirect illumination | L2 distant environment, no point light, zero indirect |
| 1080x1080, AgX and denoising | 128x128, unclipped linear targets and standard-sRGB previews |

The material labels are not a conversion claiming GGX/Beckmann equivalence.
Preserving the world director does not make these different BRDFs and image
formation settings visually identical. No orientation repair or regeneration
is needed.

## Conditional researcher admission

The researcher approved admission conditional on the appearance changes being
intentional rather than an omitted rotation. The checks above establish that
condition. A separate admission record was written without changing the dataset,
generation report, prior verification, failed runs or frozen protocols:

`/home/isavand/paper/evidence/donut-kurt-reference-v2-262b3cfcf73340a0867829c48c0198f7-admission.json`

Admission SHA-256:
`02e4f7ca909de6b7c2c9fa2a2e291850a9bff190b426e319daf82e3ecdb82c9e`.
Generation SHA-256:
`2052306bdf87a0beaa0a3f9b0bddeeb12d787f1d65357d520b5ce2fe0a33ee64`.
Orientation verification SHA-256:
`7b28bbd7343c8fdbc57acd574ab8feaa99ea4a4091e033d91817dad0d5678da7`.

Admission is limited to this controlled known-geometry, L2/zero-indirect oracle.
It establishes neither historical Blender reproduction, unique identifiability
nor physical material recovery. Earlier reports' pending-admission status remains
an accurate historical record. **D1/D2 training still requires its separate
exact-source execution authorization.** No model construction or optimizer
updates occurred during this clarification.
