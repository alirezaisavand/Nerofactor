# E.17 v1 — Observation-derived Q retention during native photometric refinement

Preparation only until one explicit approval binds the committed source, freeze,
inputs, image, work/resource ceilings and fresh output. Preserve frozen G/A,
corrected shader, E.11–E.16 data/protocols/evidence and unsuccessful outcomes.
Only two new C histories are authorized for preparation; do not repeat A/B,
distillation, teacher estimation, compatibility rendering or Bell generation.

## Hypothesis and exact controls

Retaining observation-derived physical Q may prevent the E.16 photometric phase
from erasing useful physical recovery. It may also propagate incorrect teachers
or hinder fitting. This single-coefficient experiment tests that trade-off; it
does not establish a general neural inverse-rendering method or causal certainty
about the E.16 degradation mechanism.

Reuse E.16's84 physical points,137 fitting /83 withheld-view correspondences,
original and stronger materials, exact geometry/normals/frames, known kd/metal,
L2 and rotated L2, corrected BVH and `A_corrected_core_v1` native shader. Both
lights are observed in fitting; no unseen-light or spatial-position holdout is
claimed. The field keeps1,485,584 material parameters /84 tensors, the original
positional encoding, shared features and all material heads. Every parameter
remains trainable and in Adam; known color/metal/light groups remain inactive.

For each material create C from the **exact E.16 B post-distillation full state**.
The accepted E.16 native report SHA256 is
`f14aaa8dde21afbb1dd6626c49568e3083fa8d548979f8033f749bdd8af37f2b`.

| Material | Post-distillation weights archive SHA256 | Full network-state SHA256 |
| --- | --- | --- |
| Original |`5659e55ca32c391e6a1b0dd02187dba7886413ff925515637db14bd4ca428a9f` |`3fe86a5db626ec80cea241daeeb0c6d690016b59fb9a57d908e163b1d1bfba20` |
| Stronger |`5d60885421d9d0969f0161e307de2ede34956fc8d28233fa142df39382cacc5f` |`cefc5cb21a12bbdc087cace3de757052a5ec0372bf0551c76b11695bbb06f120` |

Weights are primitive NPZ with allow_pickle=False. Verify exact archive hashes,
all key/shape/dtype identities and strict full state-dictionary loading, with no
casting, missing-key tolerance or partial load. Verify the loaded state hash and
bitwise parity of captured field outputs at all84 points. **Both restoration
gates must pass before either arm updates.** Construct only two native models;
do not restore an optimizer or rerun old scientific validation. The original
sample-direction attributes are reconstructed by the unchanged native constructor
and source; all registered parameters/buffers are strictly restored.

The E.16 radiance/gradient admission remains applicable: renderer, field,
sampling, geometry, inputs, effective shader configuration and BRDF are unchanged.
The new Q term calls the previously tested differentiable world-Q construction
and teacher-normalized loss; it does not alter the shader. No new32-draw renderer
compatibility or derivative panel is executed. Strict restoration and joint-loss
CPU tests address the changed boundary.

## One frozen coefficient, selected without truth/withheld data

For all84 points uniformly:

`L_Q = mean_i ||Q_i − Q_teacher_i||_F² / ||Q_teacher_i||_F²`

`L_total = L_RGB + 0.0001 * L_Q`.

`L_RGB` is the same mean linear-RGB squared residual over137 fitting
correspondences ×two lights ×RGB. Teachers are the unchanged E.15 fitting-selected
free-ratio/two-light/continuous estimates, including every unsuccessful/inaccurate
estimate. No confidence weighting, GT Q loss, oracle-ratio constraint, extra prior,
new lighting or material substitution. World Q preserves theta+pi and swapped-axis
equivalence. Native widths/angle encodings and tensor precision remain unchanged.

The single coefficient is justified mechanically by saved **fitting** loss and
**teacher-relative** Q drift, not physical truth or held-view quality:

| Material | E.16 last100 mean fitting loss | Post-distillation L_Q | Final B L_Q | Increase |
| --- | ---: | ---: | ---: | ---: |
| Original |1.5070737447e−6 |.0009652461 |.0114266610 |.0104614149 |
| Stronger |1.5852163972e−6 |.0000391060 |.0202483782 |.0202092722 |

Divide the pooled mean fitting loss by pooled mean L_Q increase: approximately
`1.546145071e−6 / .015335344 = 1.00822e−4`. The nearest power of ten is **1e−4**.
At the old unanchored endpoints, this would charge about1.14e−6 /2.02e−6,
comparable to the observed fitting objective. This is a scale argument, not proof
of balanced gradient magnitudes or an optimal weight. No coefficient search,
alternate value or post-outcome adjustment is permitted. The exact calculation
and its source hashes are frozen; tests block reads of GT/withheld fields.

## Training, gradients and work matching

Each C uses **fresh** Adam: LR.001,betas(.9,.999),eps1e−8,AMSGrad,weight_decay0,
global gradient clip.7, no scheduler; exactly1000 successful returns. These are
the E.16 paired settings, not legacy Trainer defaults. Preserve native512 diffuse
/1024 specular samples. Update j uses seed800000+j, j=0…999, identically to E.16 B.
Reuse the admitted E.16 field/render/trace/Adam/clipping/snapshot implementations
without modifying them or their globals. No A/B histories or distillation rerun.

Reuse the same field forward for RGB and Q. Take **one combined backward** and
one Adam step per update. Record total/RGB/Q/weighted-Q losses, teacher drift,
drift from the initial post-distillation Q, active group gradients and clipping.
The initial-state drift is diagnostic only, not a second retention term.

At updates0 and999 in each arm, inspect photometric and weighted-retention
component gradients using two autograd.grad calls on the existing graph. Preserve
.grad=None, all weights and RNG; do not step or render for these probes. Exactly
eight additional reverse sweeps are allowed. Record component group norms,
cosines and synchronized wall times separately. The normal joint backward still
follows. CPU fixtures verify the probes leave the eventual gradient unchanged.

Record the added objective/drift forward time, component-probe backward time,
reference-preparation time and endpoint evaluation/serialization time separately.
Training active time excludes probe time but includes normal instrumentation;
there is no claim of an exact all-update retention-only backward cost, equal
total compute or speedup. Compare updates primarily and report extra work.

Finite objectives/gradients, full active groups, correct Adam advancement,
strict scientific identities and unchanged known inputs/inactive weights are
mandatory. Stop and preserve partial records on failure or resource exhaustion.
No retry, resume, repair, tuning, additional arm or budget extension.

## Frozen evaluation and interpretation

Reuse the captured B post-distillation evaluation as the initial reference;
strict bitwise field-output restoration is checked with two field-only queries.
Do not rerender its eight-draw assessment. Evaluate each final C exactly as E.16:
8 common native draws with seeds900000–900007 across all220 correspondences and
both lights, plus independent256²/512² predictions using cached quadrature and
visibility. Evaluate only the final1000-return state; no best-state selection.
Own intermediate weights/moments are saved every250 updates for provenance,
without model execution to create new evaluation checkpoints.

Compare every C point with captured A1000, B1000 and B post-distillation. Report:

- World/local relative Q **median, mean, p95** (NumPy default linear percentile),
  Q≤.05 counts over all84, and per-point differences.
- Canonical major/minor width and log-mean/log-ratio errors; orientation modulo
  physical equivalences. Preserve truth major−minor≥.1 eligibility (75/80) for
  orientation≤5° counts; retain all84 in tables and near-isotropy diagnostics.
- Teacher-relative and initial-state-relative Q drift distributions and normalized
  squared losses, before and after refinement.
- Fitting/withheld native mean linear-RGB errors, single-draw variance, and
  independent fine/coarse errors at identical predicted materials. Native means
  are eight-draw estimates, not proof of systematic bias. Keep actual8-bit codes
  and component captures as in E.16; no image formation change.
- Teacher-error propagation: separately describe the fixed teacher-Q>.05 groups
  (16 original /17 stronger), only **after** training. Identify points where C
  stays closer to an inaccurate teacher while B improves physical Q/RGB, and
  report improvements/failures in both teacher groups. Never use these truth-based
  descriptive groups for supervision, selection, weighting or exclusions.

Use nominal paired Pareto comparisons rather than inventing a new post-outcome
trade-off tolerance. A strong first-component result requires lower mean/median
Q error, at least as many Q≤.05 points as unanchored B, and **no increased**
independent held-view relative error in both fields. Check the RGB ordering on
both grids; disagreements are numerically unresolved. Report nominal results,
grid error and native uncertainty even for very small differences. No new
practical degradation allowance is hidden in the decision.

Separately compare C with stopping at B post-distillation using the same physical
and photometric coordinates. If C does not offer a physical–photometric advantage,
say so even if it beats unanchored B. Do not declare global dominance from one
median or RGB value. Teacher-group associations do not prove a causal counterfactual.
One seed, known kd/metal/lights and sparse observed locations do not establish
full-object recovery, unseen-light/position generalization or a publication-ready
method. If supported, propose more seeds, spatial holdouts and material fields;
otherwise recommend just one discriminating next ablation. Execute none here.

## Execution identity, ceilings and evidence

One A100GPU0, existing image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`;
4 CPU threads,64GiB host RAM/no swap,pids128,25% CUDA allocator and2GiB outputs.
One foreground1800s watchdog/30s grace; inner1680s/20s. Estimate2–10 minutes,
unmeasured; E.16 took385s for more histories, gates and evaluations.

| Maximum new work | Count |
| --- | ---: |
| Adam returns /joint backwards |2,000 /2,000 |
| Additional component-gradient reverse sweeps |8 |
| Photometric /evaluation native shader calls |4,000 /32 |
| Field calls: updates /restoration /final evaluation |2,000 /2 /2 |
| Native shadow rays |852,541,440 |
| Independent endpoint BRDF point-directions |144,179,200 |
| Cached-geometry coefficients /environment point-directions |72,089,600 /55,050,240 |
| New primary/outgoing queries /ground-truth reference datasets |0 /0 |

Source, all eight input roots (including E.16 checkpoints), mesh, authorization
and container root are read-only. Fresh owned `/outputs/retention` only;
non-root, no network/build/install/permissions change. Hashes, runtime, counters,
gradient probes, intermediate own NPZ weights/Adam moments, both endpoint captures
and cleanup are recorded. Complete status requires normal cleanup and terminal
source/input identity verification. Failures/partial work remain preserved.

The preparation includes focused CPU stand-ins, source/Python3.8/static checks and
an exact freeze. It performs no real native construction, rendering or optimization.
Commit/push preparation for the clean-source gate, obtain one exact execution
approval, then provide the existing foreground researcher workflow if Codex Docker
access remains restricted. After that single execution, finish file-only analysis,
paper-facing tables/figures, evidence commit/push without an additional preparation
milestone. Do not bypass restrictions or initiate further experiments.
