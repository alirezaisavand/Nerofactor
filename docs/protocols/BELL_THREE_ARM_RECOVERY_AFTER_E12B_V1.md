# First bounded native three-arm recovery comparison — review proposal

**Prepared, not execution-authorized or launch-frozen.** E.12B admits corrected-
core native construction and two updates only. This proposal uses the existing
admitted Bell v3 weak-cue control. It does not wait for another lighting search,
claim the control is a fully informative primary benchmark, or authorize the
separate two-light study. Preserve all previous evidence and failed launches.

## Fixed panel and scientific question

One fresh history each, in fixed A/B/C order, seed6033, no seed selection/tuning:

| Arm | Native selector | Existing schema1 config |
| --- | --- | --- |
|A |A_original |configs/experiments/material_a_bell_original_known_v1.yaml |
|B |A_estimator_v1 |configs/experiments/material_a_bell_estimator_known_v1.yaml |
|C |A_corrected_core_v1 |configs/experiments/material_a_bell_core_known_v1.yaml |

Question: how do estimator/core corrections affect actual native optimization,
photometric generalization and nominal/valid physical material recovery on the
same known-light oracle? Original known-light A retains original BRDF/decoder/
sampler method bodies; only explicit known-light/safety infrastructure is shared.
The energy-budget variant changes material semantics and is excluded.

Exactly **2000 original Adam returns per arm**, at most6000 total. Preserve
512 training rays,512 diffuse/1024 specular samples, native sRGB Charbonnier,
Adam/AMSGrad eps1e−4,betas(.9,.999),weight decay0,clipping.7,1000-step warm-up,
original100000 cosine semantics and all other existing config/defaults. Use
known_env_v1/zero_bounce_v1,human_lights=false,reg_mat=false and diffuse-light
whiteness off for all arms. B's declared legacy shader regularization flags stay
declared; the renderer reg_mat gate prevents their use. Do not silently rewrite
ignored legacy A fields or turn priors on for another arm.

Same admission manifest
`f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`,
same original v3 environment (not L2), oracle mesh, observations and label field.
Same122 ordered training IDs,validation58 and held IDs11/105/114/17/76. No
held RGB/relights/material truth enters training, hyperparameter selection or
arm selection. Record initial named-parameter hashes for each native model;
they should match across arms, with deliberately distinct variant signature
buffers. Stop on unexplained mismatch; never force-copy weights to manufacture
parity. Same seed is not a claim of identical finite samples after algorithmic
paths diverge; report actual RNG/sampling identities and differences.

## Admission within the first histories, not extra smoke reruns

A/B native correctness is not established by C's smoke. Include their native
construction, source/variant/initialization, first-two-update call/gradient/state
gates in the proposed first histories; do not repeat C's accepted smoke. Verify
that extended instrumentation only observes the original loop and does not
replay its first updates or change numerical work. Bind its final source and
interfaces before approval. A source/identity/gradient/numerical failure stops
the batch, preserves completed/partial arms and marks remaining arms unexecuted.
No automatic retries, recovery-state restoration or parameter repair.

## Endpoint, export and evaluation definition

Use the endpoint after2000 returns, not the checkpoint selected for best held
performance. Preserve original validation/checkpoint scheduling; native final
validation's zero-based step1999 versus stored step2000 must be explicit in any
separate inference call. Save source/config/variant-bound checkpoints and retain
all logs; strict loading only, no cross-variant/partial state substitution.

Before interpreting recovery, admit strict checkpoint tag/roundtrip, parameter
export and rendering/relighting interfaces within this bounded comparison.
These are **not** proved by E.12B, which wrote no checkpoint and ran no evaluation.
Use isolated external evaluation adapters; do not silently fix original evaluators.
If an interface cannot preserve declared science, stop and report its blocker.

Evaluate material parameters at the admitted GT label positions with their
original faceted normals/frames, not nearest mesh vertices with a different
normal convention. All128 label archives provide515,698 pixel/view observations,
not that many unique surface points. Keep train/validation/held aggregates
separate. One endpoint export per arm; no intermediate material-score selection.

Report relative/absolute world/local Q errors, sorted mx/my errors, orientation
modulo physical equivalences only where the predeclared width gap≥.1 is eligible,
linear albedo/metallic errors, and all eligibility/invalid counts. Retain the
existing component criteria (Q≤.05 relative,orientation≤5°,max RGB albedo≤.02,
metallic≤.03,held linear radiance relative L2≤.01) as descriptive legacy targets;
freeze their object-level aggregation and exclusions before execution. Do not
claim flat-material success thresholds are a calibrated object-level guarantee.

For original A, nominal decoded-parameter Q and actual rendering-frame validity
are separate: a wrong/nonorthonormal frame must be reported invalid, never repaired
by the evaluator or treated as correct physical recovery. Keep all invalid cases
and denominator counts. Near-isotropic orientations are unidentifiable, not
conveniently deleted physical failures or fabricated angular successes.

Preserve native validation/NVS metrics and also report independent fixed GT-mask
linear/sRGB errors and PSNR/SSIM for held views. Do not use model hit masks alone
to certify accuracy. Retain original source-light evaluation and unseen+90°Y
relighting at the six predeclared evaluation cameras; no new lighting or cameras.
RGB fit, diffuse/specular compensation and physical recovery are distinct outcomes.

## Bounded work and final approval requirements

Training ceiling6000*512*1536=4,718,592,000 shader direction slots, not a claimed
exact BVH-query count. Constructor geometry/preprocessing and evaluation work
must be reported separately. Proposed one-GPU0/A100 envelope:4 CPUs/threads,
64GiB host RAM/no swap,2100s per complete arm and7200s foreground whole-batch
watchdog with30s kill grace. These are **proposed limits**, not a validated runtime
estimate. The two smoke intervals include heavy parameter-copy/report overhead
and do not predict long-run throughput or stability reliably.

Before launch, freeze exact sources/configs/admission/image, fresh per-arm outputs,
material-export/query and six-camera source-light/relighting budgets, output-space
limits, metrics/aggregation and interface checks, then obtain ONE explicit bounded
execution approval. No command in this document authorizes training or evaluation.
If the approved budget is exhausted, retain partial results and stop; do not
extend, resume, add seeds or tune hyperparameters automatically. All three outcomes
are required for a complete panel, including failures/unexecuted status.
