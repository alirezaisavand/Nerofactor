# E.8 v1: spatial physical correction and shared material-head projection

Preparation only until one explicit researcher scientific-execution approval.
Base `3cd6bb8e11a8972cd6a85536471f8c8b8a8a9af0`; branch
`feat/neural-material-early-step-e2a`. Preserve every accepted G/A, N1, A2,
E.3–E.7 source, protocol and evidence artifact. No E.8 results exist at freeze.

## Scientific basis and coordinate audit

[E.3](../NEURAL_MATERIAL_E3_COMPLETE_RESULTS.md) found nonsaturated N1 gradients
without accurate recovery; [E.4](../MATERIAL_PARAMETERIZATION_E4_RESULTS.md)
did not establish an improved chart. [E.5](../MATERIAL_OBJECTIVE_E5_RESULTS.md)
identified correlated physical directions and component compensation.
[E.6](../MATERIAL_OPTIMIZER_E6_RESULTS.md) and
[E.7](../HYBRID_MATERIAL_E7_RESULTS.md) found case-dependent physical solvers;
E.7 transferred all corrections successfully but did not recover both cases
with either predefined solver. Retain these negative findings, not an oracle
choice of solver based on material truth.

Pinned A `network/field.py` uses eight-frequency coordinate positional encoding
in MaterialFeatsNetwork, plus raw coordinates concatenated to its 256 features
at every material head. Its final linear layers use weight normalization.
The accepted N1 wrapper instead evaluates BOTH feature and raw-coordinate skip
at its fixed zero anchor; querying N1 at multiple positions is not spatial recovery.

E.8 subclasses the original N1 component wrapper in a new factory and evaluates
its admitted `_predict(self, positions)` at actual coordinates. It retains
the same 1,485,584 parameters,84 parameter tensors,12 buffers,head architecture,
original centering calibrated at zero,parameter ranges and atan2 diagnostics.
No frozen implementation changes. Both primary arms and the ablation use this
same spatial variant; the old constant-anchor N1 is context only.

## One spatial benchmark, two seeds

Known flat square z=0, x,y∈[-.1,.1], normal(0,0,1), original tangent(0,1,0),
bitangent(-1,0,0). Coordinates are arbitrary benchmark length units, without
mesh or unknown geometry. Illumination is the original position-independent
distant-light model, so moving a surface sample changes its material/field
input but not the independent directional-integral formula.

Training points: (-.1,-.1),(-.1,.1),(.1,-.1),(.1,.1). Spatial holdouts:
(-.04,.02),(.045,-.025). Define u=x/.1,v=y/.1 and smooth truth:

```
mx=.28+.035*u+.015*v          my=.49-.025*u+.020*v
theta=-.62+.18*u-.12*v
kd=(.36+.04*u, .58+.03*v, .24+.02*(u+v))
metallic=.25+.05*u-.04*v
```

Every truth component varies and stays physically interior throughout the square.
The entire six-location set is one shared field per seed/arm, never six fits.
Seeds101/103 use their original saved initialization-coordinate specifications
and original N1 component weight seeds10000+seed. These are fresh spatial
fields, not continuation of constant-anchor state100. All three arms have
identical full weights,buffers and actual initial predictions per seed.
No calibration against truth or tuning after reference/experiment results.

Reuse the original84 view/light observations per location:24 fit,24 held-out
views,36 relighting. New per-location four-tape fitting seeds6000+10*i through
6003+10*i; evaluation7000+10*i through7003+10*i. Holdout fitting tapes are
identity-declared but unused. Same private tapes for every arm,512 diffuse and
1024 specular samples,A2/atan2 and total-linear-RGB MSE. Only the four training
locations supply optimization losses or retention constraints.

## Mandatory new-data gates, before any update

After approval, generate six independent NumPy references using the original
two hemisphere grids256×512 /512×1024 and original2e-6+5e-5|reference| bounds.
Require component/total convergence at every location. At four training truths,
reuse the unchanged explicit-truth, two-step/two-grid finite-difference rank
gate (steps1e-4/5e-5,relative rank threshold1e-4,observed-discrepancy robustness).
No repeated historical targets, grid refinements or replacement material cases.
This certifies only local observation information, not global identifiability.

Exact independent work:6 full-reference calls plus4×28 perturbations =118
two-grid render calls,487,587,840 direction evaluations. Each location uses a
private budget under the original reference cap; a new aggregate counter limits
the total before counting each computed increment. Preserve perturbation gate
records and partial evidence. Any mandatory failure stops the pilot, without
changing thresholds or trying a different field.

All six actual models then pass coordinate dependence (>1e-6 width span over
six locations),matched full initialization,identical paired initial training RGB,
physical predictions and finite gradient checks before Adam/projection updates.
The control's initial derivatives are reused for its first update. Original
constructor calibration is counted separately.

## Three predefined arms per seed

**Adam:** same spatial N1 architecture, original photometric Adam algorithm,
LR=.001,betas(.9,.999),eps1e-8,decay0, no scheduler/clipping/regularizer. Accumulate
four per-location derivatives of loss_i/4, then one optimizer return. Serial
shader graphs bound memory; the declared objective is the equal joint mean.

**guided:**12 cyclic correction rounds selecting training index round%4.
At the selected current physical material, reuse the unchanged seven-column
two-reverse JVP construction, physically scaled by(.96,.96,pi,1,1,1,1).
Use one E.6 damped Gauss–Newton proposal with fixed damping.01,scaled step
radius.1 and original.99 bound fraction. No damping adaptation,line search,
solver switching or truth/held-out selection. Check the canonical-direct versus
actual-neural RGB bridge within1e-7+1e-5*max|direct RGB| before using its Jacobian.
This fixed-policy proposal is a correction candidate, not a separate admitted
LM optimization history. Nonpositive predicted improvement/unchanged proposal
is recorded without an unnecessary neural trial; no retry.

**no_retention:** identical policy and12 rounds, with the projection retention
weight zero instead of one. Keep the same actual joint-loss and physical-drift
guards. This isolates the soft projection penalty, not removal of every guard.

## Shared, physically weighted final-head projection

Cache each training location's actual penultimate head features during initial
shader queries. They remain valid because guided arms freeze trunk/hidden layers;
verify exact feature equality during subsequent trials. This avoids extra field
queries for each projection. Actual centered logits are passively observed.

For each head, center features across the four training points,divide by their
RMS deviation with fixed floor.001,and append an intercept. This is a solve
preconditioner, not a forward-network change. The teacher defines selected-site
logit targets using original width/albedo/metal transforms and double-angle
encoding with radial gauge.5. Keep its native axis order/theta sampling gauge;
do not sort axes or substitute a truth-preferred gauge.

Solve eight independent4×4 dual ridge systems minimizing weighted output-logit
error plus a preconditioned parameter penalty. Selected-site weight=1; other
three training-site weights=gamma(1 guided,0 ablation), target change0 there.
Channel sensitivity is sigmoid derivative in normalized physical coordinates;
rotation channels use2*q*(1-q)*anisotropy_gap/raw_radius. Thus there is no
isotropic angle prior or division by an eigenvalue gap; exact isotropy gives
zero rotation weight. This diagonal metric approximates physical sensitivity,
not a full rendering Hessian or exact quotient-space solve.

Ridge beta=.001*mean(row squared norm of centered/standardized design).
Convert coefficients back to effective last-layer weights/biases. Apply one
common factor to bound predicted training logit changes by.5. Map effective
weights into the original weight_v/weight_g representation without replacing
weight-normalization modules; reject zero/nonfinite target row norms. Only
15 final-layer parameter tensors may change; all other weights/buffers stay exact.

Evaluate the actual candidate jointly at all four training locations. Define
embedding E(p)=(Qxx,sqrt(2)Qxy,Qyy,kd_R,kd_G,kd_B,metal),
Q=R(theta)diag(mx²,my²)Rᵀ. Require: selected E-distance to teacher decreases;
each other-location E-drift≤.05; other-location training loss increase≤10%+1e-10;
joint mean training loss improves by more than16eps32*max(old,new,1e-12);
actual head probabilities remain strictly interior. No Q-to-truth,held-out,
component target or physical-success score enters this decision. Reject or
exception restores exact saved weight_g,weight_v and biases, even after a
commit request. No second projection attempt or threshold adjustment.

The radial-gauge reset and diagonal sensitivity are explicit approximations.
Large raw head-parameter changes can follow feature preconditioning; the trust
cap constrains observed logits, not every unobserved point. Retention constrains
training locations only. Held-out spatial behavior may be poor and must be reported.

## Fixed evaluation and computation comparison

Evaluate both guided arms after12 proposals. Save control evaluation at12
Adam returns (equal-update context, not equal work). After both guided arms,
mechanically choose control n=floor((min(correction_C)-26)/16),31≤n≤37.
Then evaluate its final endpoint. No score enters n. Final C is16n+100 for
control and correction_C+74 for each guided arm,including initialization,
terminal derivative measurement and all evaluation calls. The extra control
state12 evaluation is charged. Correction_C lies528–624; all queried rejected
trials,physical Jacobians and reverse sweeps count. C follows E.6's proxy:
2*(fitting shader forwards+reverse sweeps)+7*84-observation evaluation forwards.
Report actual C differences and runtime; update count is not efficiency.

| Maximum new work, successful full panel | Cap |
| --- | ---: |
| Shader forwards | 960:528 neural fit +48 evaluation +384 physical |
| Backward/reverse calls | 1008:336 neural +672 physical JVP sweeps |
| Original Adam returns | 74 |
| Physical Jacobians / proposals | 48 /48 |
| Projection attempts / small linear solves | 48 /384 |
| Extra field probes / original constructor calibrations | 14 /6 |
| Independent reference calls / direction evaluations | 118 /487,587,840 |

Extra field probes are6 initial,6 final,2 control state12; each contains all
six points. No extra shader queries for spatial probes. All attempted/completed
work,original backward/Adam counters,projection acceptance/rejection and costs
are retained. Terminal joint derivatives are diagnostics, not optimizer updates.
Six finite-point probes do not establish continuous smoothness.

## Measurements, failure policy and next research

Per training and held-out spatial location report original Q,sorted width,
identifiable orientation,albedo/metallic errors and original full-recovery
criteria (Q≤.05,angle≤5°,max albedo≤.02,metal≤.03,held-view relative RGB L2≤.01).
Report means,worst locations and success counts per seed,never only the best.
Record independent held-view/relighting diffuse/specular/total errors and
compensation,realized versus intended corrections,drift of other training
locations,held-out prediction changes and sampled spatial finite differences.
No truth-based stopping/checkpoint choice. Preserve all six arms and failures.

Shared fields receive jointly assessed corrections; this is not complete3D
inverse rendering or unknown-geometry recovery. Tests use tiny synthetic modules
and fake shaders only; real N1/reference/shader imports are blocked process-wide
during preparation. Freeze code/tests/protocol,input/tape identities and actual
JUnit/process evidence before requesting ONE execution approval.

Resources:CPU float32,one Torch/BLAS thread,hidden GPUs,accepted runtime;
one foreground1800-second watchdog with10-second kill grace,120 seconds per
reference/rank/admission/arm stage,4 GiB RSS,8 GiB address space,1 GiB outputs,
≥2 GiB free. Fresh `material-e8-<uuid>` under `/home/isavand/paper/evidence`.
Stop on identity,mandatory gate,nonfinite,invalid orientation or resource failure;
preserve partial states/Jacobians,no automatic retry/extension. Cleanup success
is claimed only after context exit; terminal source/input/tape/counter checks
precede completion. External receipt plus separate seal finalize evidence.

After one approved run, analyze saved arrays only,produce tables/figures and
rank gauge-aware head optimization,spatial regularization,sensitivity-aware
lighting,geometry uncertainty,broader benchmarks and trunk/head conditioning
by impact/novelty/cost/feasibility. No novelty claim from two seeds. Recommend
PROCEED/FIX AND PROCEED/BLOCKED based on physical AND spatial evidence.

```bash
python3 -m experiments.spatial_pilot_e8_v1 --describe
python3 -m experiments.spatial_pilot_e8_v1 --prepare
python3 -m experiments.spatial_pilot_e8_v1 --verify-sha256 FROZEN_SHA256
```

CLI is preparation only. A bound explicit approval and clean committed source
are required before `execute_reviewed` imports scientific modules. The single
approval will cover the preparation commit,one launch,offline analysis/final
commit and one ordinary push attempt; it does not permit another arm or retry.
