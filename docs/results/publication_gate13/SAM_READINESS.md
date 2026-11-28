# Limited SAM2/SAM3 readiness — 2026-11-28 UTC

Segmentation enables realistic geometry supervision; it is not the main material
method contribution and does not block Gate 13. No install, download, model
import, checkpoint deserialization or inference occurred in this preparation.

## Available assets and compatibility

Keep the accepted [angular ordering](../publication_gate12/sam2-ordering-v2.json)
and [same-mask trimap definition](../publication_gate11/realistic-mask-definition.json)
unchanged. The [SAM2 metadata audit](../publication_gate11/sam2-readiness.json)
records source `2b90b9f5ceec907a1c18123530e92e794ad901a4`, the existing
SAM2.1 Hiera large checkpoint SHA
`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`,
Python 3.10.18 and torch 2.6.0+cu124. Native loading remains untested; historical
SAM2 confidence maps were not recovered. Do not rescan historical archives.

Official SAM3 code is public. Its current README requires Python >=3.12,
PyTorch >=2.7 and CUDA >=12.6. The same repository now includes SAM3.1: pin the
requested SAM3 version explicitly rather than silently substituting that update.
These requirements do not match the existing SAM2 environment.
[Official repository](https://github.com/facebookresearch/sam3).

The official model card gates checkpoint access and distinguishes concept
segmentation from interactive image/video tracking. The visual-tracking path
supports the point-prompt comparison needed here; text-driven all-instance
segmentation is not automatically an equivalent control.
[Official SAM3 weights and interface documentation](https://huggingface.co/facebook/sam3).

A scoped filename search under `/home/isavand/paper` and `/home/isavand/sam2`
found no SAM3 installation/weight candidate. This does not prove system-wide
absence. Before a future authorized comparison, record explicit code/weight
hashes, access/license acceptance, raw-logit interface, coordinate mapping,
driver compatibility and selected numeric dtype. Do not modify the material GPU
image. A separate environment and sequential one-GPU inference are expected;
actual RAM/VRAM/runtime remain unmeasured, not promised by the published API.

## Small prospective comparison and stopping rule

Use the original fitting RGB only. Fix eight audit frames by evenly spaced
indices along the accepted 122-frame angular path, without inspecting masks or
material estimates: `(index,camera)` = (0,0), (17,15), (34,81), (51,46),
(69,6), (86,126), (103,77), (121,127). Keep validation58 and all held views out.
The reflective scene supplies the challenging image formation; do not select
frames after examining segmentation errors.

Start with the prepared SAM2 protocol: recorded P1/P2 point prompts at cameras
0 and 127, fresh forward/reverse states per prompt set, accepted angular frame
order and unchanged foreground/background consensus, thresholds and 4-pixel
Chebyshev uncertainty margin. Preserve pre-fill raw score maps. Neither the
complement of conservative foreground nor raw SAM scores are calibrated
background probabilities.

Create independent manual fitting-RGB audit labels on these eight frames, blind
to model outputs and oracle geometry; mark genuinely ambiguous boundaries as
unknown. No oracle mask, geometry or material enters prompts, thresholds,
margins or model choice. Oracle-mask scores may be reported afterward as a
separately labeled synthetic diagnostic, not used for method design.

Prospective engineering sufficiency criteria, not learned thresholds: pooled
confident foreground precision >=.99, confident background precision >=.995,
at least .70 useful foreground and .50 useful background coverage on manually
certain regions, no empty-object failures, and nominal-mask boundary F1 >=.90
within two pixels of manually certain boundary segments. Report all per-frame
values, uncertain-label fractions, worst frame, and annotation precision limits;
do not certify true probabilities from this small audit. Nominal masks and
conservative training trimaps have different purposes and must be scored
separately. These criteria are declared before inference, not material-recovery
acceptance thresholds.

If SAM2 is sufficient, select it and stop without running SAM3. Otherwise allow
one separately authorized SAM3 visual-tracking comparison using the same images,
angular ordering, point budget and forward/reverse protocol. Keep raw logits and
uncertainty; verify their units/interface before freezing its conservative rule.
Do not assume SAM2 numerical thresholds have the same meaning in SAM3, and do
not tune them on this audit. If the needed raw output is unavailable, report that
specific limitation. Never replace the comparison by extra text/box supervision
without explicitly declaring a different acquisition budget.

Plan sequential inference, no fine-tuning, at most four 122-frame propagations
per admitted model and the eight manual audit frames. A future inference freeze
must set measured-compatible memory/time caps before running; initial planning
allowance is one A100, four CPUs and 64 GiB host RAM, with a 30-minute wall cap
per model (estimate unmeasured). Preserve a failed capped inference; no automatic
retry or installation is authorized here. No threshold sweeps, ensembles or model
zoo. Consider SAM-V/another alternative only after both fail the declared need.

The realistic geometry experiment remains same-mask R0/R1: original intentional
foreground term versus the same term plus confident-background opacity support,
original schedules intact, uncertainty inactive. It is separate from O1's
perfect-mask diagnostic and from Gate 13 recovery. Direct multi-view free-space,
appearance-model changes and adaptive schedules remain separate future work.
