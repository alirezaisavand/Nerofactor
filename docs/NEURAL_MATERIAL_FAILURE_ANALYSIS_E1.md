# Phase E.1 — Offline neural material optimization failure analysis

## 1. Executive summary

The independently rechecked 27-endpoint catalog gives physical recovery D0=4/9, N0=4/9, N1=1/9 and held-out RGB passes 7/9,7/9,3/9. The N1 failures are not one mechanism: five complete boundary/zero-gradient collapses, one successful physical recovery, two photometric near misses, and one nonzero-gradient roughness-floor approach. No neural model, shader or optimizer was constructed or executed in E.1.

The strongest new numerical observation is that all five collapsed networks have **zero changed parameter elements from checkpoint1000 to5000**, while their Adam counters and second moments continue changing. Between250 and1000 they retain only8–18 changed parameter elements, with net L2 displacement about9.65e-13–1.54e-12. The logs bracket the first observed collapse between completed states0 and9; they do not reveal its exact update or the upstream cause. A large LR, a bounded-output bottleneck, dead ReLUs, physical sensitivity and neural conditioning cannot be causally separated from these records alone.

Raw rotation channels near zero are **not** a near-zero centered vector: c=2q−1 is then near(−1,−1). All4509 saved N1 pre-update records have centered radius≥0.4999999702; maximum encoding-norm error is8.1994e-8. No observed normalization singularity explains M3/103.

First recommendation: a separately approved, short instrumented baseline diagnostic followed by a paired **sole LR change 0.01→0.001** on four predeclared cases. Do not execute a new full panel or assume reduced LR solves physical recovery. The [E.2 draft](protocols/NEURAL_MATERIAL_ABLATION_E2_DRAFT.md) is a proposal only.

## 2. Verified scientific provenance

Analysis branch `feat/neural-material-failure-analysis-e1` starts at `e380a5a464b18de24204e6ac4c8a789a0ab73a14`. It preserves original frozen baseline678f3bf3230ea9e36aa97065491466a816c10310 and original interrupted results commitb953250fe5f6f172caefa1151fb2ff45dcac1def. Pinned G79252178e65914ad1acbc14790b1514f276bfea0 and Ae1e859779301faced46d6476691e20b546c27ae0 remain untouched; G is not executed by this known-plane benchmark.

| Input | SHA-256 |
|---|---|
| freeze | `dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af` |
| original | `7a9ca7ae1e930c1a90143079f83e7376ebd00ed8d0bea514bf321686ea37084e` |
| complete | `f60a3b8fa6db646efec916428c75a3bfb3196da89cd03a3ef3955f3609254135` |

All348 frozen source hashes, continuation source/protocol/test identities, original/continuation journal identities, anchors/seals where available, reference identity, all131 original and45 continuation output files, and all81 referenced numeric checkpoints are verified before numerical analysis. Full parameter/buffer/Adam catalogs are bound to their manifests and numeric arrays are independently read with allow_pickle=False and matched by shape/dtype/content hash. The same evidence is rechecked after analysis. All27 endpoint metrics and frozen success decisions are recomputed from saved predictions and accepted independent targets; none are rerendered. D0/N0 endpoints remain exactly equal, while their accepted full trajectory/prefix verification is preserved under its original hashed evidence.

Twenty completed original histories and seven verified continuation histories are selected uniquely, without latest/best discovery. M1/105/N1 uses original250/1000 checkpoints and a continuation5000 checkpoint; its logs0–999 are inherited, states1000–4269 belong to the authorized replay, and subsequent states belong to new progress. Loss/material/gradient measurements describe PRE-update state u−1, whereas work counters and active time in the same log were captured POST optimizer return u. Admission gradients are separate original pre-optimization observations; first-training gradients are labeled independently.

The original invocation remains incomplete with unknown exit status and cause. Token-usage interruption is researcher-reported, not inferred from its missing seal. Logical work totals135000 returns; the two invocations contain138270 committed physical returns including3270 authorized duplicates. The continuation is not one uninterrupted original experiment. No old artifact or prior summary is rewritten.

The [machine report](evidence/NEURAL_MATERIAL_FAILURE_ANALYSIS_E1.json) is the complete quantitative analysis. Its source hash identifies the [NumPy analyzer](../experiments/analyze_neural_material_failures_e1.py). Full external analysis is `/home/isavand/paper/evidence/neural-material-e1-20261113-v3/analysis.json`, SHA-256 `5cb4011e170ab4312bb3f4f41c09b9440e3d5983e0e90c2deaa10c572b5e0c55`. The preliminary v1/v2 offline reports/plots remain separately preserved; final v3 adds an explicit orientation legend. Every numerical field in v2/v3 is identical; only analyzer/plot artifact identities differ. These are read-only analysis passes, not scientific experiments. Raw4509-record trajectory JSON and plotting cache stay outside Git.

## 3. Complete history classification

All27 histories are complete at5000 updates. [Catalog CSV](figures/neural-material-e1/catalog.csv) records source segment, initial/final fit, physical/gauge errors and pass decisions; JSON additionally contains every initial/final seven-component material, available records and checkpoint identities. Each trajectory has501 records (labels1,10,…,5000), three snapshots250/1000/5000, and a separately labeled terminal evaluation. No unrecorded intermediate measurements are fabricated.

| N1 history | Group | Q error | Theta error ° | Held-out L2 | Relight L2 | Physical pass |
|---|---|---:|---:|---:|---:|---|
| M1/101 | A | 4.729213 | null | 0.8330647 | 0.8351463 | no |
| M1/103 | B | 0.01243002 | 0.008535505 | 0.0009611806 | 0.001341107 | yes |
| M1/105 | A | 4.729213 | null | 0.6679201 | 0.6722497 | no |
| M2/101 | C | 0.05463788 | 0.1217291 | 0.006150766 | 0.01578127 | no |
| M2/103 | A | 2.98269 | null | 0.5740023 | 0.5951798 | no |
| M2/105 | A | 2.98269 | null | 0.9301718 | 0.9316176 | no |
| M3/101 | D | 0.9915249 | null | 0.07380316 | 0.1796812 | no |
| M3/103 | C | 0.09390875 | 11.61382 | 0.003966025 | 0.006200907 | no |
| M3/105 | A | 4.474072 | null | 0.4333407 | 0.447133 | no |

A=complete boundary saturation; B=physical recovery; C=photometric near miss; D=nonzero-gradient failure with an observed floor approach. These categories are derived from saved exact-zero groups and unchanged endpoint decisions, with continuous floor distances reported rather than a new near-boundary success threshold. GroupD is the one residual failure, M3/101: its final widths0.0400000438/0.0400041528 approach the float32 floor but remain distinct from an all-zero-gradient collapse. Null orientation is retained as unidentifiable, never converted to zero error. The original identifiability rule requires both truth/prediction anisotropy gap≥0.1.

## 4. Training trajectory analysis

Six standalone PDF figures show all nine N1 cases and group labels: [loss vs paired D0](figures/neural-material-e1/loss.pdf), [roughness](figures/neural-material-e1/roughness.pdf), [RGB/metallic](figures/neural-material-e1/color_metallic.pdf), [orientation/norm/radius](figures/neural-material-e1/orientation.pdf), [six gradient-group norms](figures/neural-material-e1/gradients.pdf), and [physical-boundary distances](figures/neural-material-e1/boundary_distance.pdf). Lines connect saved observations; they are not measurements of intervening updates. Terminal material/loss points are squares at state5000. Log losses preserve their actual values. Gradient/boundary axes use symlog with linear threshold1e-12 as a display convention only; exact gradient zeros have x markers at zero and are not replaced by positive floors.

M1/103 reduces terminal fitting MSE to2.69916e-8 and passes physical recovery; its trajectory has nonmonotone excursions and continued parameter movement, not an assumed monotonic optimizer. M2/101 and M3/103 fit radiance well but miss different physical criteria. M3/101 retains nonzero gradients while widths approach the lower bound; its tiny roughness/rotation gradients differ from the five exact complete-zero cases. Recorded active times and logical work counters remain available in external trajectories; the replay clock has its original conservative checkpoint bound and must not be treated as uninterrupted wall time.

## 5. Saturation onset analysis

For each collapsed history, state0 (log label1) is the last recorded nonsaturated/nonzero-gradient observation. State9 (log label10) is the first recorded fully bounded physical solution and all-six-groups-zero observation. Thus the transition occurred after one of updates1–9; calling it “at update10” would confuse a log label with its PRE-update measurement. All500 later saved records remain zero, with no observed recovery. Loss and gradients are measured at the same sampled states, so the ordering of loss change versus boundary crossing within the missing interval is unknown.

| Collapsed history | Loss at state0 | Loss at state9 | Observed increase? | Later nonzero records |
|---|---:|---:|---|---:|
| M1/101 | 0.02078046 | 0.03623095 | true | 0 |
| M1/105 | 0.02730261 | 0.02494076 | false | 0 |
| M2/103 | 0.02691354 | 0.009047536 | false | 0 |
| M2/105 | 0.02160882 | 0.02218017 | true | 0 |
| M3/105 | 0.0251034 | 0.01266182 | false | 0 |

Only two of these five cases have a larger sampled loss after the transition; three decrease despite becoming physically wrong and stuck. Loss increase is therefore not a necessary diagnostic of the observed collapse. [Classification CSV](figures/neural-material-e1/classification.csv) preserves both observed log/state labels and zero counts. Initial logits, per-update activations, material-upstream derivatives and full states1–9 were not captured. An exact causal onset cannot be reconstructed without new authorized measurements.

## 6. Source-level material parameterization

The [N1 field](../experiments/neural_material_field_v1.py) compiles only the admitted original-A definitions from [field.py](../scientific_sources/material_a/network/field.py). It queries a fixed origin anchor rather than a spatially varying object. The feature trunk has eight width256 Linear layers, an encoded-input skip and ReLUs except its last layer. Each of five heads has three width256 Linear/ReLU stages, a final Linear, and sigmoid output. Linear weights use weight normalization. Shared features plus the raw anchor form each259-dimensional head input. There are1485584 trainable coordinates in84 tensors, grouped into shared features and five heads.

For head h, let f_h(W) be its original pre-sigmoid output, c_h its detached fixed initial output, and t_h the fixed target logit. The implemented centered head is z_h=t_h+[f_h(W)−c_h], q_h=sigmoid(z_h). Roughness is mx=0.04+0.96q_x and my=0.04+0.96q_y; RGB kd and metallic use their sigmoid outputs directly. The lower/upper roughness bounds are0.04/1; no new clipping or regularizer is applied to them. Orientation target logits are logit((1+rho[cos(2theta0),sin(2theta0)])/2), rho=0.5. Hence initial centered radius is about0.5. Centers/targets/anchor are fixed buffers, not optimizable offsets.

The exact local factors in real arithmetic are dq/dz=q(1−q), dm/dz=0.96q(1−q), and dz/dW=df_h/dW. ReLU contributes1 on its positive branch and0 on its inactive branch (zero at the kink under the implemented autograd convention). Weight normalization uses w=g v/||v||: dw/dg=v/||v|| and dw/dv=(g/||v||)(I−vvᵀ/||v||²) on the nonzero-v branch. Raw g/v coordinate displacements are consequently not invariant changes of the represented function.

The [actual A2/atan2 shader](../experiments/material_shader_v1.py) consumes the actual normalized two-channel orientation, not a reconstructed angle re-encoded through cos/sin. Diagnostic theta=0.5 atan2(e_y,e_x) is gauge-dependent; the frozen Q/major-axis metric handles axis swap and quarter-turn equivalence. RGB loss is four-tape mean linear RGB followed by MSE, not a gamma-space objective. Gradients travel through Monte Carlo radiance/sampling/BRDF, physical outputs, bounded heads and shared features. No illumination/geometry model is optimized in this known-input benchmark.

Finite-precision sigmoid outputs can round to0/1, where the output-based q(1−q) factor vanishes. Affine roughness rounding can also yield a physical bound even if an unlogged q was not exactly at that bound. Boundary material values alone therefore do not prove which derivative became zero. Direct observations establish boundary outputs and complete zero network gradients; source equations establish possible local bottlenecks. The renderer’s upstream physical sensitivities, hidden logits and ReLU activity remain unobserved. No BRDF defect is established.

## 7. Adam and gradient diagnostics

Effective options are the saved Adam LR0.01, betas(0.9,0.999), eps1e-8, weight_decay0, amsgrad=false, maximize=false, capturable=false, differentiable=false, foreach=null and fused=null. No scheduler, clipping or regularizer was added. The inspected installed Torch2.4 optimizer source and its hash are recorded in JSON; its exact file hash was not captured at original execution, so present source inspection is not independent proof of the historical dispatch/binary. The actual effective options and archived moments are execution evidence.

Using saved m_t,v_t, E.1 derives the magnitude of d_t=[lr/(1−beta1^t)] m_t / [sqrt(v_t)/sqrt(1−beta2^t)+eps] in float64. This is a stored-moment algebraic diagnostic, not an optimizer step, next-gradient prediction or measured float32 displacement. No parameter is updated. Checkpoint differences are net displacement over750 or4000 updates, not cumulative path length or individual step magnitudes.

| N1 history | ||p250|| | Changed elements250→1000 | ||Δp250→1000|| | Changed elements1000→5000 | ||Δp1000→5000|| |
|---|---:|---:|---:|---:|---:|
| M1/101 | 78.76703 | 13 | 1.485969e-12 | 0 | 0 |
| M1/103 | 83.54151 | 359922 | 6.291759 | 357942 | 4.777403 |
| M1/105 | 78.00558 | 8 | 9.650834e-13 | 0 | 0 |
| M2/101 | 86.21644 | 312574 | 0.661502 | 314018 | 4.999969 |
| M2/103 | 77.86699 | 18 | 1.536761e-12 | 0 | 0 |
| M2/105 | 77.91016 | 14 | 1.457721e-12 | 0 | 0 |
| M3/101 | 84.89541 | 156208 | 0.02109083 | 348122 | 3.178195 |
| M3/103 | 84.84114 | 444678 | 4.124304 | 442725 | 4.533202 |
| M3/105 | 77.32181 | 9 | 1.200945e-12 | 0 | 0 |

All84 parameter tensors differ from their initial hashes by checkpoint250 in every N1 history, proving that the networks did update; initial full tensor values were not saved in the selected run artifacts, so their initial norms and the initial-to250 displacement magnitudes are unavailable. The first full checkpoint is already long after the onset interval.

In the five collapsed cases, first-moment L2 norms at250 are about1.77e-14–2.96e-14, reaching about5.64e-42–5.86e-42 at1000 and5000; second-moment norms continue decreasing. Counter/moment advancement is not network movement: every stored parameter element is identical between1000 and5000. The tiny250→1000 net movement is consistent with residual Adam momentum/finite-precision behavior, but the intermediate update sequence was not saved and no new recurrence was simulated.

First-training N1 gradient norms span0.01097–0.02593; the successful case0.01213 lies within the failed range. Initial group L2 max/min ratios span about683–10455, often dominated by RGB versus rotation, but groups have different sizes and units. Adam rescales coordinates; these ratios are not a proof of excessive LR or ill-conditioning. For fresh Adam in real arithmetic, |d_1,i|≤lr and ||d_1||≤lr sqrt(nonzero initial coordinates): the saved counts give N1 upper bounds8.79–8.87 versus at most0.02646 for seven direct coordinates. These are bounds, not observed updates, and say nothing by themselves about physical-output changes through an unknown network Jacobian.

## 8. Orientation encoding analysis

The implemented path is q=sigmoid(z), c=2q−1, e=c/max(||c||,1e-6), followed by validity checks, shader normalization and the atan2 decoder. On the observed regular branch r=||c||≥1e-6, de/dz=(2/r)(I−eeᵀ)diag(q(1−q)) and dtheta/dz=[−c_y,c_x]diag(q(1−q))/r². These are source-level analytical factors computed from saved raw channels; they are not measured loss Jacobians.

All observed radii remain at least0.4999999702, including all initial observations. M3/103’s last logged raw channels are about5.5693e-7/5.5769e-8, giving c near(−1,−1), r≈1.41421 and normalization operator norm≈0.7071. This is far from the zero-centered-vector configuration q=(0.5,0.5). Its sigmoid factors are tiny and its final rotation group squared norm is about9.54e-22 but nonzero. This supports near-boundary output/weak logit sensitivity as a diagnostic, not a demonstrated normalization blow-up.

M3 truth anisotropy gap≈0.23477 exceeds the frozen0.1 identifiability gate, and M3/103 remains identifiable: its11.6138° error cannot be excused as a null orientation. By contrast the collapsed mx=my=1 solutions and M3/101’s nearly equal floor widths have unidentifiable predicted orientation. The successful M1/103 solution uses a swapped-axis/quarter-turn equivalent representation; its small gauge-invariant error is retained. Weak physical angular sensitivity, neural Jacobian conditioning and sigmoid attenuation cannot be separated by network gradient norms alone.

## 9. Physical versus photometric recovery

[Paired endpoint CSV](figures/neural-material-e1/paired_endpoints.csv) reports every material/seed pair. N1 improves Q in1/9 pairs, held-out RGB in2/9 and relighting in1/9. It improves physical pass status only for M1/103, while losing direct physical passes on M2/101,M2/103,M3/101,M3/103. M2/101 is the clearest appearance/physics reversal: held-out L2 improves0.00721332→0.00615077, while Q worsens0.0273777→0.0546379 and crosses the unchanged failure boundary. M3/103 passes held-out RGB0.00396602 and albedo/metallic but fails Q0.0939088 and angle11.6138°.

Diffuse/specular residual correlation and signed inner product are retained for fit, held-out and relighting. Negative inner product is evidence of cancellation, not a new pass threshold. D0 M1/103 has held-out component errors0.0955131/0.688085 with correlation−0.990438 and total error0.0122400. N1’s same-case component errors fall to0.00154616/0.0104669 with total0.000961181. Other low total-RGB cases still fail physical recovery. M2/105/N1 has positive held-out residual correlation≈0.208833, so cancellation is not universal. No appearance metric substitutes for the original five-criterion physical decision.

## 10. Ranked hypotheses and evidence

Confidence labels describe this evidence, not posterior probabilities or confirmed causal attribution.

| Hypothesis | Confidence | Supporting evidence / mechanism | Counterevidence or missing evidence | Smallest discriminator |
|---|---|---|---|---|
| H2 bounded heads attenuate gradients | supported as a mechanism, not an exclusive root cause | Sigmoid factors, boundary outputs, zero groups; late weights cease changing | No hidden logits, ReLU activity or upstream physical gradients; successful/near-miss histories use identical mapping | Retain upstream physical/logit gradients in unchanged early baseline |
| H5 photometric agreement permits physically incorrect estimates | supported observationally | Seven direct RGB passes vs four physical; M2/101 and M3/103 near misses; component cancellation | Does not prove fundamental nonidentifiability or distinct exact global minima; accepted local-rank gates remain | Compare fixed-state physical and radiance sensitivities under the same data |
| H1 early Adam step scale drives saturation | plausible | Five transitions bracketed in first1–9 updates; large-dimensional first-step bound; two sampled loss increases | No early per-coordinate updates; three collapsed losses decrease; shared LR alone is not a causal explanation | Instrumented ten-step LR0.01/0.001 pair, all else fixed |
| H4 orientation boundary behavior contributes | weakly supported | M3/103 raw channels approach bounds, tiny rotation gradients, wrong angle despite good RGB | No small centered radius; exact normalization singularity unsupported; M1/103 succeeds with same encoding | Capture upstream angle/logit derivatives, then isolated rotation-family comparison only if implicated |
| H3 neural parameterization conditioning is unfavorable | weakly supported | High redundancy, disparate group scales, parameter movement with limited physical progress | No material-to-weight Jacobian spectrum, Hessian or metric-invariant function distance; parameter count is not proof | Matched-physical-state Jacobian/sensitivity diagnostic |
| H6 late finite-precision/momentum stagnation | supported as a late state observation | Zero changed weights1000→5000, counters/moments continue | Explains persistence, not the unobserved onset mechanism; no per-step recurrence captured | Early captured states and actual step displacements |
| H7 renderer has weak material sensitivity in failed states | plausible, unmeasured | Floor/near-isotropy can reduce angular information; tiny network gradients | Those gradients also include bounded-head/network factors; no physical-upstream derivative recorded | Same-forward physical-material gradient capture before any model change |

The strongest defensible explanation is an early transition into absorbing or nearly absorbing output states, followed by persistent vanishing network gradients. The evidence does not identify which link first failed, establish that LR is excessive, or indict the BRDF equations.

## 11. Unresolved identifiability and observability limits

No per-update weights, logits or hidden activations exist for states1–9. Initial full N1 weights are represented by hashes only; checkpoint250 is too late to measure an early overshoot. Group norms cannot recover individual gradient signs or correlations. Physical-upstream derivatives, Jacobian/Hessian spectra and parameter-function distances are missing. Sparse losses do not reveal when loss worsened relative to saturation. Local truth identifiability does not prove global uniqueness or optimizer accessibility. Two execution segments and authorized replay must remain explicit; the original interruption cause and process totals remain unknown independently of researcher testimony.

## 12. Proposed Phase E.2 experiments

The [draft matrix](protocols/NEURAL_MATERIAL_ABLATION_E2_DRAFT.md) ranks: (1) instrumented ten-step baseline/LR-only pair on four fixed cases, at most80 updates, to test early step-scale causality; (2) matched-state physical/weight-Jacobian diagnostics with zero updates; (3) a conditional rotation-head activation-family comparison, keeping LR/other heads fixed. It specifies comparators, data/seeds, held constants, physical/photometric diagnostics, work/time/memory caps, negative-result interpretation and required new versions/freezes. The short diagnostics are not5,000-update recovery claims. A later full endpoint study needs separate approval; no E.2 code or experiment was run.

## 13. A100 80GB feasibility planning

The feature MLP/heads and batched Monte Carlo direction/BRDF/tape computations could benefit from GPU execution, but no speedup is measured or guaranteed for these small constant-material batches. The present field cpu_float/default-device guards, shader _tensor/RandomInputs checks, explicit CPU random-input namespace, CPU tensor construction, .numpy() calls, scalar validation and original snapshot code prohibit a drop-in device switch. A100 memory capacity alone does not establish compatibility.

A future port should transfer fixed observations/targets/tapes once and retain parameters, moments and tensors on device; repeated per-step host/device transfers, bool/.item() checks and full checkpoint copies may force synchronization and dominate runtime. Preserve dtype, source operation order, random draw identity and stage-specific measurements. CPU/GPU reductions, matrix kernels, TF32, math libraries and autograd can differ even with identical seeds. Pin device/driver/CUDA/Torch/math flags and determinism policy under a new GPU identity; do not promise bitwise CPU equivalence.

Require separately predeclared component/sample/mask/frame/output/physical/gauge comparisons, initial-gradient and optimizer-prefix comparisons, float32 arithmetic envelopes, RNG/state/cleanup and checkpoint validation before GPU scientific use. Diagnose any branch or clamp-boundary differences without changing tolerances afterward. A GPU engineering milestone and a separately frozen scientific protocol are required; none was started or needed for E.1.

## 14. Research implications and recommended next milestone

The original-A-derived constant neural field is less reliable than direct coordinates on this selected panel under the frozen optimizer:1/9 versus4/9 physical recovery. The one excellent neural result proves capability in one start, not broad robustness. Architecture, orientation encoding and redundant parameterization are confounded; no depth-only or population conclusion follows.

Recommend the targeted E.2 step-scale experiment only after review of its baseline instrumentation and exact initial/log parity, retaining all four cases and both outcomes. First capture the missing links rather than simultaneously changing architecture, activation, sampling and LR. If the new evidence fails to discriminate causes, retain that uncertainty. Stop here: no shader execution, AD, training, optimizer update, target regeneration, ablation, GPU work or scientific source change occurred in E.1.

### Reproduction and validation

From this analysis checkout, run only the offline analyzer with a new output directory:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B -m experiments.analyze_neural_material_failures_e1 \
  --output /home/isavand/paper/evidence/neural-material-e1-new-review
```

It refuses existing/unapproved output roots, checks evidence before computation and again afterward, reads NPZ with allow_pickle=False, and writes analysis/plots/CSV/cache only under its new owned output. Numeric summaries are deterministic for the saved inputs; output path identities naturally differ for a newly selected destination. Matplotlib3.5.1 PDF metadata omits creation/modification timestamps. Scientific plotting is standalone, not image generation.

New tests use synthetic NumPy fixtures plus the newly generated analysis report: exact27 coverage/counts; segmented/replay labels; exact six-group zero classification; nullable orientation; PRE/POST labels; no fabricated intermediate measurements; float32 boundaries; analytic orientation factors; read-only stored-moment algebra; reproducible CSV bytes; mismatched JSON/unsafe object-NPZ rejection; and import without Torch. No historical scientific suite is repeated. Final focused test result: **15 passed, 0 failed, 0 skipped**. AST/document-link checks, all 348 frozen source hashes, all figure/table hashes and the analyzer-source identity passed. Only new E.1 files are included in the Git change.
