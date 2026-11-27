# Gate 10.3 preparation and intervention priorities

Decision: **PROCEED with the frozen evaluation-only study after separate approval.**
Geometry admission and downstream G→A remain BLOCKED by Gate 10.2's original
silhouette criterion. This preparation does not promote an earlier checkpoint.

The [protocol](../../protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V1.md) binds the four
authenticated checkpoint files and reuses the accepted 300k mesh/quality audit.
File hashes and producing receipt identities have been verified without tensor
deserialization. Intermediate tensor fingerprints and native compatibility remain
unverified until authorized execution. Only the three earlier dense extractions
are new. All 122 fitting mesh silhouettes are measured at every step; a fixed
16-camera, 4,055-ray-per-draw panel measures original neural accumulation using
two common training-policy sampling draws per checkpoint, with no optimizer.

## Scientific rationale and mask provenance

The thesis foreground-only mask loss was intentional. Together with curvature
and opacity regularization, it addressed degeneration of anisotropic geometry
under NeRO's approximate geometry-stage reflectance model. The researcher reports
that ordered multi-view images were segmented as a SAM2 video, with interior
positive prompts and a threshold favoring foreground precision over recall.
Unlabeled pixels therefore do not establish background; treating their complement
as a binary background target can remove genuine geometry.

The authenticated Bell v3 acquisition is a distinct synthetic control. The new
[file-only mask audit](mask-provenance-v1.json) verifies all 128 generated PNGs
against the saved oracle-mesh primary-ray hit arrays, and all 123 staged masks
against the producing training freeze. Only 122 fitting masks enter optimization;
camera 58 is validation and the five held views are absent from staging. These
are synthetic reference masks, not SAM2 output. The older 800×800 Bell masks were
used for reference-generation camera/mesh admission, not copied into training.
Their original producing procedure remains unresolved.

The [supervision clarification](MASK_SUPERVISION_CLARIFICATION_V1.md) governs future
intervention interpretation, including the legacy prospective binary-loss paragraph
in the unchanged evaluation protocol. It changes no checkpoint, camera, sampling,
threshold, extraction, resource limit or quality decision in Gate 10.3. Gate 10.2
evidence remains intact: its excess support is an observed quality failure, not
proof of a mask-loss bug or a controlled causal attribution.

## Source-backed mechanism assessment

Pinned G computes `L1(m, m*a)` for mask m and accumulated foreground opacity a.
For binary masks and 0<a<1, its derivative is −m/N. Background pixels have zero
direct derivative from this term even when they accumulate opaque foreground.
For imperfect masks, that absence intentionally avoids supervising unknown labels
as background. For Bell v3's synthetic reference masks it motivates an isolated
diagnostic control, without establishing the preferred realistic method.
Its scheduled weight changes .3→1 over 0–30k; increasing it does not create a
missing background derivative. The original full raster ray pool contains both
foreground and background, so this is not a camera/ray-admission omission.

The opacity regularizer is the entropy `−a log(a)−(1−a)log(1−a)` after the source
clamp [.001,.999], differentiated through both appearances of a. Its interior
derivative is log((1−a)/a); descent favors zero below .5 and one above .5 without
knowing the true mask. Clamp-saturated derivatives can vanish. This differs from
BCE against a fixed true foreground label. RGB remains a full-image objective
with original background NeRF; computed foreground/background RGB losses are
not selected in Bell's loss list. None of these facts isolates a cause of the
accepted 300k mesh's excess support.

Synthetic CPU checks use the actual original mask assignment and entropy method,
plus analytic/finite-difference checks of the proposed additive background term.
For example, with m=[1,1,0,0] and a=[.2,.8,.2,.8], original derivatives are
[−.25,−.25,0,0], while the lambda_bg=1 candidate yields [−.25,−.25,.25,.25].
These are toy derivatives, not measured neural parameter gradients or evidence
that the proposed loss will fix geometry. The candidate preserves the original
foreground term exactly, including its absolute-value behavior near a=1.

The synthetic lambda_bg=1 calculation is a perfect-mask diagnostic, not a
recommendation to label all SAM2-negative pixels background. Any future variant
preserves the original foreground expression, full-batch mean and .3→1 mask
schedule when isolating an added background contribution. Reuse the authenticated
original history as the perfect-mask control; no loss variant is implemented or
launched in Gate 10.3.

## Ranking after the diagnostic

| Candidate | Evidence and purpose | Cost | Attribution/publication status |
| --- | --- | --- | --- |
| Existing foreground-only mask loss | Intentional precision-first supervision; preserve as the matched baseline | Reuse authenticated perfect-mask history; imperfect-mask baseline requires a future separate history | Valid original method; unknown pixels remain unsupervised |
| Conventional binary loss with perfect masks | Isolate the effect of adding background opacity using the same synthetic reference masks | Optional one matched 300k diagnostic history | Explicit oracle-mask control; cannot establish robustness to imperfect masks |
| Confidence-aware trimap with imperfect masks | Add only corroborated background; ignore uncertain regions and retain original foreground and schedule | One future matched history plus its same-mask foreground-only baseline | Realistic method candidate if stable background-opacity errors are observed |
| Multi-view confidence-aware free-space SDF | Penalize occupied SDF in conservatively supported fitting-camera free space | Same-mask comparison and separately bounded SDF sampling | Realistic method candidate if low opacity conceals erroneous zero geometry; no hard carving |

Background-NeRF interaction, anisotropy-aware directional appearance including
ambient-tensor integration, and geometry–material refinement remain separate
research candidates. Keep them fixed during the first supervision comparison.
The [research backlog](../../RESEARCH_BACKLOG.md) records dependencies and the
cost-effective matched comparison; Gate 10.3 results will motivate which realistic
background mechanism to test first, without proving either one's effectiveness.

Relevant source components are `geometry_g/network/field.py` (SDF, appearance,
geometric-mean roughness, FG LUT and directional-light approximation),
`geometry_g/network/renderer.py` (normals/NeuS alpha, foreground/background
accumulation and mask construction), and `geometry_g/network/loss.py` (weights
and schedules). Future anisotropic appearance should condition on physical
direction/tangent tensors without changing the authenticated original reference.
No causal isotropic-appearance claim or new method implementation is made here.

## Reporting and independent confirmation

Report all four fixed checkpoints, including failures. Separate fitting versus
held-view errors, mesh versus volume disagreements, sampling flips, domain-band
effects and raw negative extension. Precision/recall and full exclusion counts
accompany IoU. Preserve the original 300k endpoint and do not choose a better
checkpoint from development metrics. Temporal schedule overlap is descriptive.

The existing Bell validation/held views are now diagnostic/development data for
geometry-method design. Independent confirmation requires another dense scene
or a new prospectively reserved acquisition not among the original 128 poses,
defined before observing the intervention's recovery outcomes. Freeze the method
before that confirmation and disclose any subsequent adaptation. Another Bell
view panel alone does not establish cross-object generalization. No such new
reference generation is part of this preparation or its evaluation authorization.

Native extraction, rendering, plugin-cache reuse and runtime cost remain untested
for this combined study. One bounded approval will cover all four checkpoint
loads, three extractions, full fitting mesh audit and fixed forward-only opacity
panel. There is no intermediate review cycle if its mandatory gates pass. Any
identity, numerical or resource failure stops without retry. Geometry-quality
failure is retained as a completed diagnostic result, without admission.

The separate `clarification-validation-v1.json` preserves the file snapshot at
the mask-supervision amendment. It is not the final execution identity. Final
preparation additionally corrects the unfrozen launcher's freeze filename and
limits checkpoint mounts to the four named inputs, preserves original CPU ray
construction, and adds focused interface checks. The final committed freeze and
preparation-validation record bind those reviewed bytes. No scientific execution
or earlier accepted definition was changed by these preparation checks.
