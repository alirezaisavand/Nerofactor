# Phase E.2 draft — targeted causal tests, not execution authorization

This draft follows the [E.1 analysis](../NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.md)
from baseline `e380a5a464b18de24204e6ac4c8a789a0ab73a14`. No experiment in
this document was implemented or executed in E.1. Each requires separate
researcher approval, a new named implementation/configuration, tests and a
hash-selected pre-execution protocol/freeze. Preserve the original A/G sources,
A2/atan2 shader and all accepted evidence. Do not call a revised result baseline.

## Shared controls

Use the same known plane/lighting, independent targets, observation order,
float32 CPU implementation, original seed/weight-seed pairing, fixed fitting
tapes 3000–3003, evaluation tapes 1000–1003 and sample counts 512/1024. Keep
Adam betas (0.9,0.999), epsilon 1e-8, zero weight decay and all other effective
flags. No scheduler, clipping or regularizer. Any departure must be the one
declared intervention below. Do not change tolerances or choose a best held-out
checkpoint. Retain every start and failure.

The endpoint physical criteria remain Q≤0.05, identifiable orientation≤5°,
maximum RGB albedo error≤0.02, metallic error≤0.03 and held-out total RGB relative
L2≤0.01. Short diagnostics report these descriptively; they cannot establish
5,000-update recovery. Report fitting, held-out and unseen-relighting component
and total errors, compensation, exact-zero groups and physical boundaries.

## Priority 1 — instrumented baseline admission, then one-factor LR pair

**Hypothesis:** the early Adam step scale, interacting with the current output
mapping, is responsible for entry into saturation. Current records show an
association, not that LR=0.01 is excessive or that a lower LR fixes recovery.

**Cases:** M1/101 (complete collapse), M1/103 (successful control), M3/101
(roughness-floor failure), M3/103 (orientation/photometric near miss). These
cases are selected to discriminate observed mechanisms, not estimate a success
rate. Their selection is part of this draft and cannot change after execution.

**Instrumented comparator:** preserve the exact baseline for ten successful
updates and record every state, including update zero. Retain pre-sigmoid logits,
raw sigmoid values, physical outputs, shared features/ReLU activity, upstream
loss derivatives at physical outputs and logits, parameter/Adam states and actual
pre/post-update displacement. Retain these values during the existing forward/
backward operations; do not rerender to collect them or alter the gradient.
Require source/initialization/tape identity and exact saved baseline log parity
at labels 1 and 10 before interpreting the intervention for each case.

**Only scientific intervention:** LR=0.001 instead of 0.01. Use fresh paired
original initializations, never the already saturated checkpoint. All weights,
architecture, calibration, output transformations, orientation decoder, sampling,
loss and optimizer flags stay fixed. No LR search or alternative fallback rate.
The instrumented LR=0.01 comparator is necessary for the new capture protocol;
its repetition must be explicitly authorized, not silently counted as new data.

**Bounded first stage:** four cases × two arms × ten updates = 80 successful
Adam returns maximum; one initial and one endpoint query per arm maximum in
addition to the ten fitting forwards (at most 96 shader forwards, 88 backwards
including optional initial derivative admission). One CPU thread, GPUs hidden,
120 active seconds/arm, 1,200 seconds whole, original 4-GiB RSS/8-GiB AS caps,
no retry. Freeze exact work accounting and capture volume before approval.

**Primary diagnostic:** change in Q from state 0 to state 10 together with the
first observed boundary/zero-gradient state. Photometric endpoints and component
errors are secondary; success at ten updates is not a recovery claim. Captured
upstream/activation derivatives must determine whether the zero first appears
in the shader-to-material path, sigmoid, ReLU/shared features, or elsewhere.

**Interpretation:** removal/delay of saturation under the sole LR change supports
a causal step-scale contribution for these cases; it does not prove reliable
recovery or exclude the bounded mapping. If both arms saturate, or LR=0.001 merely
slows motion, preserve that negative finding. If the baseline capture loses
parity, stop before attributing differences to LR. No automatic extension.

**Later recovery stage:** only after review, separately authorize the same
predeclared paired cases to a common 5,000-update endpoint, with the original
900 active seconds/history and a separately frozen whole budget. A partial
ten-step comparison does not authorize this stage.

**Ranking:** highest causal information for modest implementation/CPU cost;
practical optimization diagnosis rather than a claimed novel BRDF contribution.
No GPU validation is required for the CPU diagnostic.

## Priority 2 — state-conditioned derivative/conditioning diagnostic

**Hypotheses:** H2/H3/H4: bounded-output derivatives or a badly scaled neural
Jacobian, rather than a weak physical renderer derivative, cause the observed
loss of optimization sensitivity.

**Intervention:** observation only at the same fixed states. Use initial states
and explicitly selected post-250 checkpoints for the four cases above; no
optimization. Capture material-output derivatives and compute physical-output
Jacobians with respect to weights using a separately reviewed AD/JVP procedure.
Compare singular values/rank of the observable material Jacobian with the
seven-coordinate control at the same physical material, accounting for the
orientation gauge. Do not compare different materials and call that conditioning.

**Held constant:** all scientific rendering/target/tape/settings above and the
selected weight/Adam identities; do not perturb a saturated checkpoint to make
AD nonzero. Direct-material sensitivity and the network parameter Jacobian are
different measurements. Coordinate conditioning depends on weight normalization
and gauge; use a stated metric rather than raw parameter count as proof.

**Budget proposal:** eight N1 states plus eight matched direct controls, at most
16 renderer forwards, seven field-output Jacobian sweeps plus one scalar-loss
derivative/state (128 derivative sweeps total), zero
Adam updates; one CPU thread, 1,800 whole seconds, 4-GiB RSS/8-GiB AS. Freeze
exact same-forward graph retention, sweep memory and attempted/completed counters
before approval. An inability to fit the budget is a blocker, not authorization
to silently switch devices or reduce the derivative catalog.

**Primary result:** distinguish nonzero physical sensitivity with zero logit or
weight sensitivity from weak physical sensitivity itself. Q/RGB values describe
the fixed states and must remain unchanged. No new recovery success is inferred.

**Ranking:** strong mechanistic information, more implementation/memory complexity,
no performance-improvement claim. CPU-only; GPU not a prerequisite.

## Priority 3 — orientation-head activation family (conditional proposal)

Only pursue if the earlier captures implicate the rotation head. Keep LR=0.01,
all non-rotation heads, architecture, loss and sampling fixed. Replace only the
rotation head's sigmoid with bounded softsign
`q=(1+z/(1+abs(z)))/2`, using its analytic inverse to match the same initial raw
channels and the same fixed pre-activation centering. This is one output-mapping
family intervention; the required inverse-calibration constants are part of that
intervention and cannot be represented as an unchanged calibration buffer.

Compare baseline versus this new, distinctly named rotation variant on M1/103,
M2/103 and M3/103 (successful baseline, full collapse, orientation near miss).
Admit exact/bounded initial physical/sample/component equivalence before updates;
initial derivative equality is not expected because the intervention changes it.
Freeze new equivalence tolerances before observing results; do not silently reuse
the old N1 initial-model hashes or claim identical optimizer coordinates.

First-stage budget: six arms × at most 250 updates = 1,500 returns, at most
1,512 shader forwards including initial/endpoint, CPU one thread, 120 active
seconds/arm and 1,200 whole seconds, original memory caps. Q/orientation are
primary; record albedo/metallic, held-out/relighting, raw channels, radius and
all derivative factors. This is not the canonical 5,000-update recovery panel.

If rotation gradients improve but other heads still collapse, that argues against
rotation being the sole cause. If radiance improves without Q/orientation recovery,
do not call it a physical success. Bounded softsign can also saturate in finite
precision; no guarantee is assumed. A new source/configuration/freeze is mandatory.

**Ranking:** potentially useful parameterization comparison, lower priority because
it changes the model and cannot isolate the LR question. No claim of publishable
novelty without comparisons to existing methods; GPU validation not required.

## A100 planning boundary

No CUDA development or experiment is authorized by this draft. A GPU path requires
its own engineering/numerical equivalence milestone and then a separate scientific
freeze. Validate identical explicit sample tapes, initial outputs, BRDF components,
gradients, optimizer prefix and physical/gauge metrics before using acceleration
for any ablation. CPU and GPU outcomes must remain distinct evidence families.
Do not replace this small causal diagnostic with another full 27-history study.
