# E.4 AS1: anchored anisotropic material chart and matched pilot

**Preparation only; separate researcher approval is required for scientific
execution.** Base `f8cde24889fc44d0d2e25a08e625907662f08b87`, branch
`feat/neural-material-early-step-e2a`. Preserve all frozen sources and accepted
[E.1](../NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.md),
[E.2b](../NEURAL_MATERIAL_LR_E2B_RESULTS.md), and
[E.3](../NEURAL_MATERIAL_E3_COMPLETE_RESULTS.md) evidence.

## Scientific question and source interpretation

E.3 low LR avoided exact sigmoid endpoints in 5/5 cases, but physical recovery
was 0/10 across both LRs. Eight arms improved held-out RGB from state 10 to 100,
only two improved Q. Saturation, weak physical sensitivity, finite-tape
estimation error, parameter compensation and optimizer path remain distinct
explanations. Local identifiability gates near truth do not establish globally
unique inverse solutions. No parameterization can create missing information.

The accepted material source is A at `e1e859779301faced46d6476691e20b546c27ae0`,
with the separately accepted A2/atan2 estimator. Despite a sampler's historical
GGX name, its ideal NDF is Beckmann in slope space. In the original unrotated
tangent/bitangent coordinates, with slope s and half-vector cosine c_h,

\[
Q=R(\theta)\operatorname{diag}(m_x^2,m_y^2)R(\theta)^T,\quad
D(h)=\frac{\exp[-s^TQ^{-1}s]}{\pi\sqrt{\det Q}\,c_h^4}.
\]

Gaussian slope covariance is **Q/2**, not Q and not a GGX roughness substitution.
The accepted ideal specular kernel is F D/(4 v·h); do not add a Smith G term or
replace the actual finite estimator/clamps. Original diffuse and Fresnel use
`kd*(1-metallic)` and `.04*(1-metallic)+metallic*kd`. Consequently good total
RGB may hide component/parameter compensation.

The physical gauges are theta→theta+pi and
`(mx,my,theta)→(my,mx,theta+pi/2)`. Q is invariant under both. Its theta
derivative vanishes as mx→my; orientation is unidentifiable at isotropy. Keep
the original .1 anisotropy-gap scoring rule and null orientation semantics.

The [accepted symmetry audit](../MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2_RESULTS.md)
shows that identical physical materials can have different finite-tape
estimates under axis exchange. A Q-only decoder using eigenvectors would
introduce an isotropic eigenvector singularity and potentially change sample
coupling. **Do not canonicalize eigenframes or replace sampling here.** Choose
the simpler, gauge-compatible AS1 chart to preserve the actual A2 objective.
It is not a unique Q quotient or an observation-adaptive identifiability method.

## One alternative: AS1

Let d be the original predictor's pre-sigmoid output minus its original fixed
calibration center, and t its original target-logit buffer. Original roughness
uses q=σ(t+d), m=.04+.96q. AS1 changes only the anisotropic four-coordinate
block (two widths and two orientation channels):

\[
q_{AS1}(d;t)=\sigma(t+\operatorname{asinh}d),\qquad m=.04+.96q_{AS1}.
\]

At d=0, values and first derivatives equal the original map. Away from the
anchor, q' = q(1−q)/sqrt(1+d²), with algebraic rather than exponential tails
in raw displacement. The fixed compression scale is **1 logit unit**; no scale
search is authorized. Metal and RGB albedo retain their original sigmoid heads.

For original rotation target t_r, set q0=σ(t_r), c0=2q0−1, e0=c0/||c0|| and

\[
j_0=\frac{[-c_{0y}q_{0x}(1-q_{0x}),\ c_{0x}q_{0y}(1-q_{0y})]}{\|c_0\|^2},
\quad a=j_0d_r,\quad e=R(2a)e_0.
\]

Feed e directly to the **unchanged A2/atan2 shader**. This is unit-circle
orientation transport with the same initial angle Jacobian as the original
normalized double-sigmoid encoding. Keep both original neural rotation outputs;
their one-dimensional null direction is now fixed rather than state-dependent.
The original nonzero calibration anchor is required; there is no fallback for
zero/near-zero anchors. Recompute constants from the original fixed target
buffer; add no trainable weights or extra state buffers.

Axis swap exchanges width targets/displacements, negates rotation targets and
displacements, and yields e→−e with invariant Q. A pi angular increment yields
the same e. Equal widths yield Q=m²I with zero ideal Q sensitivity to orientation,
without fixing an artificial preferred angle or dividing by an eigenvalue gap.
The chart is periodic and many-to-one; it does not remove all gauge redundancy.

This keeps the same physical material family (interior and boundary limits),
1,485,584 trainable parameters, 84 tensors and 12 buffers. Every initial weight
and calibration byte comes from the accepted state-0 checkpoint. Changing the
global coordinate geometry is the treatment; equal parameter counts do not
establish equal spatial-field function classes. This pilot is constant-material.

Float32 still permits saturation/physical-floor rounding at extreme values.
The final shader's atan2 branch and finite-tape gauge sensitivity also remain.
Do not claim globally nonsaturating behavior, improved information content or
physical recovery from analytical gradient checks alone.

## Validation and remaining native gate

Use only generic tensor algebra, tiny fake backbones and fake-counter drivers
before approval. Tests cover exact origin values/local Jacobians, finite and
correct derivatives/domain, axis/period gauges, independent analytical
Beckmann/Q equality, isotropy, parameter inventory, observer cleanup and the
matched schedule. They never import real N1 or MaterialShaderV1.

Coordinate-only evaluation of the saved state-0 calibration buffers reproduces
all five captured shader inputs bitwise for all five cases. This is a saved-data
algebra check, not an executed neural forward or shader query. After approval,
all five real first fitting passes must pass **before any optimizer update**:
strict original parameter/buffer identity, bitwise shader-input/physical parity,
exact saved fitting loss, and initial group-gradient norm agreement within
`1e-8 + 5e-5*reference_norm`. Generic float32/float64 derivative fixtures validate
the local-Jacobian contract; the native tolerance is frozen before execution.
Reuse those fitting gradients for update 1. If native matching fails, preserve
actual captures and stop. The smallest additional control would be one original
state-0 admission query for each failing case, requiring separate approval;
do not automatically rerun whole baseline histories or change tolerances.

## Controlled comparison and exact budget

Primary control: accepted E.3 original N1 **LR=.001**, fixed state 100.
LR=.01 is historical context only. No original histories are rerun. AS1 uses
the same five cases, original state-0 weights/buffers/latent seed, optimizer,
24 fitting observations, independent targets, fit tapes 3000–3003, evaluation
tapes 1000–1003, 512 diffuse/1024 specular samples and CPU float32 MSE.
Adam LR=.001, betas (.9,.999), eps 1e-8, weight decay 0; no scheduler,
clipping, extra regularizer, initialization tuning or optimizer-state inheritance.

Match the baseline's actual query schedule, including its diagnostic overhead:
updates 1–10; fitting terminal measurement at 10; independent evaluation at 10;
an additional unused fitting parity pass for the two M3 cases; updates 11–100;
terminal fitting measurement and independent evaluation at 100. Initial state-0
measurement is cached from admission, not repeated. Intermediate diagnostics
do not select the endpoint or modify optimizer state. Evaluation uses separate
fresh **counters**, not fresh weights, so the frozen one-evaluation counter
helper remains unchanged. Sum all native counter containers in the report.

| Case | New updates | Shader forwards | Backwards | External evaluations |
| --- | ---: | ---: | ---: | ---: |
| M1/101 | 100 | 104 | 102 | 2 |
| M1/103 | 100 | 104 | 102 | 2 |
| M2/103 | 100 | 104 | 102 | 2 |
| M3/101 | 100 | 105 | 103 | 2 |
| M3/103 | 100 | 105 | 103 | 2 |
| Total | **500** | **522** | **512** | **10** |

These are the same total shader/backward counts as the historical primary
arms. The two duplicate M3 measurements add no updates. Five scratch N1
constructor-calibration passes are separately counted; historical interrupted
controls had twelve, so **do not make whole-run wall-time efficiency claims**.
Point/tape combinations: 512×96+10×336=52,512; diffuse/specular directions
26,886,144 / 53,772,288. No full 1,000/5,000-step, GPU or real-scene experiment.

## Frozen resources, evidence and stop rules

Fresh `material-e4-<unique-id>` under `/home/isavand/paper/evidence`; original
datasets/results/evidence remain unchanged. Reuse owned Store, strict hashes,
array manifests with `allow_pickle=False`, selected protocol/source identities
and foreground review gate. No new provenance framework or Docker changes.

Keep CPU one thread/hidden GPUs, exact accepted runtime, 4 GiB RSS/8 GiB AS,
120-second all-case admission stage, 120 seconds per history, 1800-second
foreground external watchdog with **10-second kill grace**, 1 GiB output cap
and 2 GiB free before launch. No retries, resource extensions or substitutions.

Record every fitting measurement, actual parameter-gradient groups, shader/
head/shared-feature gradients, raw and compressed displacement, unit-circle
transport, physical/Q errors and actual float32 parameter movement. Retain
signals at 0/10/100, selected full weights/buffers/Adam/RNG at 10/100, both
external endpoint metric sets and source/input/runtime identities. Repeated
state-10 captures have unique names. Signal/full-state caps remain 1/18 MiB;
JSONL rows are capped at 128 KiB. On failure retain counters and one partial
state where available. No full per-update parameter/XOR reconstruction claim.

Require all per-arm/aggregate/native counters, immutable tapes and source/input
hashes, normal resource/RNG cleanup, report+seal and successful external process
exit for completion. Failed/unstarted cases stay visible. A representation-aware
diagnostic adapter from E.3R handles float32 roughness-floor values without
clamping or changing success criteria. True numerical/scientific failures stop.

After explicit execution approval, commit this frozen package to satisfy the
clean-checkout gate, bind its commit/freeze to a fresh run and live supervisor,
check conflicting processes and make **one** bounded launch. No affirmative
execution review or real model/shader call is created during preparation.

## Analysis, novelty and next decisions

Compare all five paired state-100 physical/radiance outcomes, not favorable
checkpoints. Keep Q≤.05, identifiable orientation≤5°, max albedo error≤.02,
metallic error≤.03 and held-out relative L2≤.01; null orientation never passes.
Compare sorted width errors, diffuse/specular/relighting errors, saturation,
gradient transport, actual movement and local conditioning. Saved-data analysis
adds no renderer query. Improved RGB alone is not success. A joint chart pilot
cannot attribute effects separately to compression versus circle transport;
component ablations would need later authorization if evidence justifies them.

No novelty is claimed for matrices, asinh, circle encodings or activation
conditioning. Relevant primary work checked during design:

- [Olano and Baker, LEAN Mapping (2010)](https://userpages.cs.umbc.edu/olano/papers/lean/):
  slope-distribution moments and anisotropic shading precede this study.
- [Heitz et al., SGGX (2015)](https://research.nvidia.com/labs/rtr/publication/heitz2015sggx/):
  matrix-based anisotropic distributions are established; this is a different
  microflake family and is not substituted for the accepted Beckmann kernel.
- [Zhou et al., rotation representations (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html):
  representation continuity is prior art; SO(3) dimensional statements are not
  a novelty proof for this SO(2) chart.
- [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html):
  saturation and Jacobian conditioning are longstanding issues. Their results
  do not establish this variant's BRDF recovery performance.

Any methodological contribution must be supported by the controlled recovery
results, later component ablations and a broader literature review. Preserve
trunk/head LR, trust-region physical step control, alternative bounded maps,
ambiguity/lighting design, broader synthetic/real scenes and geometry-to-material
uncertainty as a ranked post-pilot backlog. Implement none in this pilot.

Preparation commands:

```bash
python3 -m pytest -q -p no:cacheprovider tests/scientific/test_material_parameterization_e4_v1.py
python3 -m experiments.material_parameterization_e4_v1 --describe
python3 -m experiments.material_parameterization_e4_v1 --prepare
python3 -m experiments.material_parameterization_e4_v1 --verify-sha256 <E4-freeze-sha256>
```
