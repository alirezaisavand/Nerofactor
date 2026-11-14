# E.7 v1: cached physical correction → neural-head projection

Preparation is authorized; scientific execution requires one separately bound
researcher approval. Accepted base: `e7ae53cf991f8fe597547d8757612be1cf089c0a`,
branch `feat/neural-material-early-step-e2a`. No E.7 scientific results exist at
freeze time. Preserve all original sources, protocols and evidence.

## Question and rationale

[E.3](../NEURAL_MATERIAL_E3_COMPLETE_RESULTS.md) found stable nonsaturated N1
gradients without full recovery. [E.4](../MATERIAL_PARAMETERIZATION_E4_RESULTS.md)
did not establish a useful alternative chart. [E.5](../MATERIAL_OBJECTIVE_E5_RESULTS.md)
found local physical optimization progress and parameter compensation.
[E.6](../MATERIAL_OPTIMIZER_E6_RESULTS.md) recovered M2 with LM and M3 with Adam;
LM had lower fitting loss in both, but worse M3 Q recovery. These observations
motivate transferring observation-derived corrections rather than selecting a
teacher using truth. Two cases cannot establish novelty or generalization.

Hypotheses: (1) a small neural-head update realizes accepted physical corrections;
(2) the corrections preserve their physical/radiance properties after projection;
(3) this improves on continued original N1 Adam under the declared work proxy.
The M3 diagnostic tests whether the LM/Adam objective ranking persists across
independent fitting tapes; it cannot establish global identifiability or causation.

## Immutable initialization and arms

Cases M2/103 and M3/101 use original N1 LR=.001 state 100: all 84 parameter
tensors, 12 buffers, Adam moments/counters/options and CPU Torch/NumPy RNG.
Each archive has 350 numerical arrays. Exact selected paths, full archive and
manifest hashes are bound in the generated freeze; no implicit checkpoint search.

| Case | Producing evidence run | State archive SHA-256 |
| --- | --- | --- |
| M2/103 | `neural-e3-b4d1389b174e4b20b2f892725050a0ae` | `0e0408716bbf5b6b56dd0a5bbd6a894df7ed5dab139cff9d3bb7e80549bacfc8` |
| M3/101 | `neural-e3r-39baf93f9caa4d89b3fc2ddc549eb205` | `51d28f6b3f8fe2ead6f5e26515b0888a54fcc5b55e8927522ae07d868952370d` |

All six original N1 constructions/restorations and exact physical/fitting-loss
parity gates pass before any projection or Adam update. The control's initial
forward/backward is reused for its first optimizer step; read-only probes must
not clear or alter these cached gradients. Constructor calibration is separate
work and is not an additional shader query. No new targets are generated.

Three predefined arms per case:

1. **control:** original neural Adam continuation, including original moments,
   LR=.001, betas(.9,.999), eps=1e-8, decay=0, no clipping/scheduler/regularizer.
   M2 makes exactly 115 new returns (absolute state 215); M3 makes 162 (262).
2. **hybrid_LM:** accepted E.6 final LM material, projected once into exact N1
   state 100, then original fitting-objective acceptance and fixed evaluation.
3. **hybrid_Adam:** accepted E.6 final direct-Adam material, same procedure.

Both teachers are retained for both cases regardless of truth-based success.
E.6's nine/thirteen LM steps and four-trial censoring are not extended. No new
physical optimization, neural post-projection Adam, teacher selection, retries,
threshold adjustment or best-held-out selection. Hybrid output is weights-only;
restored Adam moments become stale after projection and are not a resumable state.

## Closed-form head projection

Keep N1 architecture, original feature network, all weights, calibration buffers
and A2/atan2 rendering unchanged. At the constant anchor, each centered logit is
`a = target + (W h + b - center)`. Thus `da/db = I`. For teacher
`p=(mx,my,theta,kd_R,kd_G,kd_B,metal)`, desired sigmoid outputs, in head order,
are:

`q* = ((mx-.04)/.96, (my-.04)/.96, metal,
       (1+.5*cos(2theta))/2, (1+.5*sin(2theta))/2, kd_R,kd_G,kd_B)`.

Require finite strict interior probabilities; do not clip or repair. Desired
logits are `log(q*/(1-q*))`. Update only the final biases by
`delta_b = a* - a`, the minimum Euclidean bias displacement at fixed features
and calibration. Five tensors/eight coordinates change; no full-network inverse
or iterative projection. Rotation radius .5 is the original calibration gauge,
predeclared to keep its two sigmoids in [.25,.75]. This explicitly changes the
raw radial gauge, so subsequent conditioning improvement cannot be inferred.

Retain the teacher's native theta and axis order: no sorting, wrapping,
eigenframe division or near-isotropic angle prior. The ideal Q symmetry does
not guarantee actual finite-tape sampling invariance. Compare the realized
double-angle encoding as well as Q. The projection has no access to truth.

Actual neural output must have non-angle physical max error ≤2e-6, relative
Q error to **teacher** ≤1e-5, normalized double-angle max error ≤2e-5, and no
exact sigmoid 0/1. Then execute one original fitting forward/backward, retaining
actual shader/head/shared gradients. Require teacher-loss agreement within
`1e-10 + 1e-4*max(trial_loss, teacher_loss)` and fitting improvement exceeding
`16*eps32*max(old_loss,new_loss,1e-12)`. These roundoff/realization bounds are
not revised after results. Never accept on Q/truth/held-out/component scores.

Only then commit. Reject or exception restores exact old biases, including an
exception after the commit request. Verify every other parameter and buffer is
unchanged. Failed realization omits the trial query; no padding to consume the
budget. Retain rejected proposals and evaluate the actual restored endpoint;
trial gradients are not presented as gradients of that restored endpoint.

## Bounded M3 diagnostic and spatial limitation

M3 truth, E.6 LM and E.6 Adam endpoints each use 24 original fitting observations
with tape quartets 3000–3003, 4000–4003 and 5000–5003. Original targets are
unchanged. Report float32 fitting MSE plus independent diffuse/specular/total
errors and compensation statistics. Reuse accepted-tape LM predictions from
its bound last Jacobian archive only after verifying its physical identity;
this is one reused point and **eight new forward queries**, no derivatives or
optimization. Original-tape LM/Adam fitting objectives must match E.6 exactly.
These diagnostics cannot choose the teacher, endpoint or acceptance rule.

N1 deliberately ignores query positions and predicts at one constant anchor.
Measure this behavior on a nine-point xy grid {-0.1,0,0.1}² at z=0. Separately
query its original `_predict` at (.1,0,0), (0,.1,0), (-.1,-.1,0), retaining
raw widths, albedo, metallic, rotation sigmoid outputs and encodings before/after.
Those three-point predictor extensions are not N1's deployed forward behavior
and are not validated spatial recovery. Report their drift and any invalid
orientation encodings; never invent a fallback angle. There is no off-anchor
shader evaluation or ground truth. Shared final biases may change all locations.

## Cost accounting, fixed endpoints and resources

Reuse E.6 proxy `C=2*(fitting forwards+reverse sweeps)+7*external forwards`.
Inherited teacher costs include E.6 Jacobians, queried rejections, diagnostics
and evaluations, not merely successful steps. Projection work and extra field
queries are reported separately with timings; C is not FLOPs or wall time.

| Case | Inherited LM / Adam C | New hybrid C each | Total LM / Adam C | Control returns / C |
| --- | ---: | ---: | ---: | ---: |
| M2/103 | 459 / 458 | 13 | 472 / 471 | 115 / 471 |
| M3/101 | 649 / 646 | 13 | 662 / 659 | 162 / 659 |

Each successful hybrid has initial/trial fitting forwards, one reverse and
one external forward: C=13. A failed realization without a trial has C=9 and
is explicitly not work matched. Control C=4*n+11 includes its initial cached
pass, n returns, terminal derivative measurement and one external evaluation.
Control length mechanically matches the smaller inherited+new teacher cost,
not a score. LM differs by 1/3 C units. Shared baseline probes and M3 diagnosis
are separate. A cached-teacher marginal-cost benefit is not an end-to-end gain;
direct versus neural reverse calls have different computational costs.

Successful nominal caps, including all parity/terminal/external work:

| Operation | Maximum |
| --- | ---: |
| New physical-diagnostic shader forwards | 8 |
| New neural shader forwards | 293 |
| Renderer backwards | 283 |
| Neural Adam returns | 277 |
| Closed-form projection attempts | 4 |
| Extra N1 grid probes / original predictor-extension calls | 8 / 8 |
| Original N1 constructor-calibration passes | 6 |

Thus ≤301 new shader forwards, 301 N1 wrapper calls (293 render +8 probes),
eight underlying predictor extensions, no new physical optimizer/Jacobian or
reference calls. All attempted/completed counters and original native counters
must agree. Censored/rejected work is preserved; do not manufacture a full panel.

Use one CPU foreground execution, hidden GPUs and one Torch/BLAS thread;
original Python3.9.12/NumPy1.21.5/Torch2.4.0+cu121 float32 runtime. Watchdog
1800 seconds with 10-second kill grace, 120-second admission/diagnostic/arm
stages, 4 GiB RSS, 8 GiB address space, 1 GiB outputs, ≥2 GiB free. Fresh
explicit `material-e7-<uuid>` directory under `/home/isavand/paper/evidence`.
No detached supervision, automatic retries, extensions or alternate settings.
Cleanup succeeds only after both context exits. Terminal source/input/tape
hash checks and counters precede scientific-completion claims. Separate final
seal plus external process receipt establishes evidence finalization; immutable
result.json retains its pre-finalization status. Failure saves partial weights,
moments, RNG and diagnostics where possible, without modifying previous evidence.

## Evaluation and interpretation

Use accepted independent external tapes1000–1003 and all84 observations exactly
once per endpoint. Original thresholds remain relative Q≤.05, identifiable
orientation≤5°, maximum albedo error≤.02, metallic error≤.03 and held-out
relative RGB L2≤.01. Report sorted width errors, orientation modulo physical
equivalences, fitting/held-out/relighting diffuse/specular/total errors, final
physical values, gradient transport, saturation, projection residuals, drift,
costs and timings. Near-isotropic orientation uses original diagnostic rules.

Distinguish teacher failure, projection failure and preserved successful physical
correction. Report all six arms, including negatives. Good constant-anchor
transfer is not evidence of spatial-field consistency, compensation removal,
an attributable new method or superiority to a direct physical solver.

After one approved successful execution, offline code may derive tables/figures
and rank next experiments (head preconditioning, gauge-aware output trust
regions, spatial consistency, sensitivity-aware observations, geometry uncertainty,
larger benchmarks) by impact/novelty/feasibility/cost. Do not issue further model
queries. Recommend PROCEED / FIX AND PROCEED / BLOCKED with explicit limitations.

## Preparation and execution gate

Focused tests use only tiny synthetic modules/Adam states and block scientific
imports process-wide. Freeze new source/test/protocol hashes, exact evidence
selections, work limits and test JUnit/exit status before asking for approval.

```bash
python3 -m experiments.hybrid_material_e7_v1 --describe
python3 -m experiments.hybrid_material_e7_v1 --prepare
python3 -m experiments.hybrid_material_e7_v1 --verify-sha256 FROZEN_SHA256
```

These commands never construct N1 or execute the shader. No public execute CLI.
After explicit approval, a bound review selects freeze, clean committed source,
fresh output and external watchdog before `execute_reviewed` imports scientific
modules. Approval also covers the preparation commit required for the clean
gate, one launch, offline analysis/final commit and one ordinary push attempt.
