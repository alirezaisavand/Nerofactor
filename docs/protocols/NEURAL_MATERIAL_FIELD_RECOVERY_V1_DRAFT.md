# 3D-2: controlled constant neural-material recovery — proposed v1

**PHASE A DRAFT — NOT FROZEN, NOT IMPLEMENTED, NOT EXECUTED.** The original
request authorizes audit/design only. No freeze JSON, model checkpoint, new
reference target or recovery result is created. Decisions D1–D7 in the
[implementation plan](../NEURAL_MATERIAL_FIELD_IMPLEMENTATION_PLAN_V1.md) need
review before a final execution protocol can be admitted. The
[source/evidence audit](../A_NEURAL_MATERIAL_FIELD_AUDIT_V1.md) distinguishes
verified properties from proposals. All future commands must use `python3`.

## Question and scientific scope

Under the SAME known plane, known lighting, observations, independent targets,
fixed Monte Carlo draws, physical initialization and Adam settings, how do a
direct material, an identity module and an A-topology-derived constant-input
network differ in physical recovery and optimization behavior?

This is a synthetic constant-material representation experiment. It cannot
establish native A reconstruction, spatial field recovery, learned-geometry or
unknown-lighting performance. Native G/A stay separate and unchanged at
`79252178e65914ad1acbc14790b1514f276bfea0` /
`e1e859779301faced46d6476691e20b546c27ae0`.
The scientific reference is completed 3D-1 commit
`ccb03d0ec5adc814775538b9cc7427cc57480a98`.

## Fixed inputs and comparison matrix

Reuse, do not regenerate, the admitted 3D-1 independent NumPy references and
local-identifiability evidence. Exactly M1/M2/M3 and original observation order:

| Material | `[mx,my,theta,kR,kG,kB,metallic]` |
| --- | --- |
| M1 | `[.28,.49,-.62,.36,.58,.24,.10]` |
| M2 | `[.17,.62,.91,.56,.31,.18,.65]` |
| M3 | `[.37,.47,.19,.24,.43,.70,.35]` |

Same scene origin, normal +Z, base T=+Y/B=−X, 12 views, seven known lights,
24 fit/24 held-out/36 relight observations, original split memberships and RGB
definitions. No meshes/checkpoint loading, raytracing, learned geometry, variable
visibility, unknown illumination or spatially varying truth.

| Arm | Trainable representation | Initialization | Primary optimizer |
| --- | --- | --- | --- |
| D0 | One original seven-vector; six logits and raw theta | Exact original fresh seed 101/103/105 records | Continuous Adam, proposed 5000 updates |
| N0 | `nn.Parameter` of shape [7] exposed by a constant module | Bitwise identical to D0 | Same continuous Adam and update ceiling |
| N1-A8 | Original A `MaterialFeatsNetwork` and five `make_predictor` heads; constant input; fixed initialization calibration | Same six scalar logits, prescribed bounded orientation match to D0 | Same continuous Adam and update ceiling |
| N2 | **Excluded**; a zero-final-weight initialization ablation is scientifically possible but would need a separately approved protocol | Not selected after results | No runs authorized/proposed in the primary matrix |

Exactly 3 materials × 3 paired fresh initialization seeds × 3 arms = **27
primary histories**. These are three starts per material, not nine independent
starts: N0/N1 are paired representations. Repeat the same seed vectors/network
weight draws across materials as deliberate control, independent of their truth.
Do not choose a successful 3D-1 checkpoint as the starting vector. The nine
3D-1 direct Adam histories are reproducibility references, and 3D-1 P0 LM
endpoints are separately labeled archival optimizer context, not primary scores.

No additional P1-U/P1-C recovery matrix is added here. Retain gauge and signed
sample-coupling checks as admission fixtures, not extra optimizer arms.

## Preserve the rendering calculation

Every arm uses the SAME admitted `MaterialShaderV1('A2','atan2')`, float32,
512 diffuse/1024 specular samples, original Fresnel/clamps/PDF/masks and approved
Beckmann/Kurt-inspired anisotropic formulation. Preserve A2's already versioned
midpoint diffuse/total-count specular estimator; do not revert to native A0 or
replace it with GGX/Smith. No new decoder, roughness transform or BRDF correction.
Loss remains original four-tape mean RGB FIRST, then linear-RGB MSE against the
same float32 fit targets. No native Charbonnier, energy/material/light loss,
regularizer, clipping, scheduler or learned light component.

Training tapes remain 3000–3003; evaluation tapes remain 1000–1003. Original
private NumPy RandomState generation, draw order, endpoint conventions and
array hashes from 3D-1 must match. One fixed realization, shared across profiles
and representations, is not a sampling-population study. No redraw, dynamic
coupling, angle canonicalization, sampled-direction repair or seed search.
The shader's explicit `is_train=True` taped azimuth convention is retained for
the original external evaluation too; module `.eval()` must not change it.

D0/N0 use the original seven-vector decoding and original ObservationBatch
operation order. N1 needs a NEW explicit material-bundle adapter: broadcast its
six physical scalars and its actual two-channel normalized orientation directly
into the existing shader interface, preserving the same component means.
Do **not** convert N1's encoding into theta and back into cos/sin for rendering;
that would insert a new numerical/gradient path. Theta reconstructed with the
existing half-atan2 rule is for logging/metrics only. This adapter is separately
versioned; the old batch, shader and source files are not edited.

## Constant field and exact initialization contract

### D0 and N0

Use original fresh seeds 101,103,105; preserve private draws and exact float32
latent/physical initial values. N0 has one seven-vector parameter, NOT a trainable
`Linear(1,7)` with both weight and bias. Its forward returns the parameter itself
without detached copies or a changed broadcast/reduction path. Theta remains an
unbounded coordinate, not a seventh sigmoid output. D0/N0 must have bitwise
identical forward values, parameter gradients, Adam moments/counters and paired
trajectories in the admitted CPU runtime.

### N1-A8: reuse, constant input and calibration

Reuse exact admitted source definitions `Embedder`, `get_embedder`,
`MaterialFeatsNetwork`, `make_predictor` and their required activation helpers
through a reviewed CPU-only source extraction. Do not import the full native
GPU field module or construct `MCShadingNetwork`/renderer. Preserve original
widths, activation order, skip, weight normalization and head channels/order.
No smaller surrogate network, extra learned embedding or illumination networks.

The wrapper ignores all query positions and substitutes fixed `x_ref=(0,0,0)`
for BOTH the trunk's xyz and the raw xyz concatenated into each predictor.
Compute one material at the anchor and broadcast it across positions. Test
different query positions and batch sizes; replacing only the trunk input is
insufficient. This is intentionally a controlled constant-input restriction,
not a claim that native A is a constant field or batch-numerically identical.
Field-only fixtures use the nine plane points `(x,y,0)` with x/y in `[-1,0,1]`,
in lexicographic order; batch1 is the origin, batch96 repeats that grid and
truncates to96. Physical outputs and actual shader-input encodings must be
bitwise constant. Primary admission repeats the nine-point query twice.
These probes do not move the actual rendering point away from the original origin.

Proposed private network weight seeds are **10101,10103,10105** (`10000+seed`),
shared across materials. Build original components in original material order,
using a CPU-only local Torch RNG scope that restores global RNG exactly. Freeze
initial named-parameter order, values/shapes/dtypes/hashes, buffers and RNG state
after Phase B/C, before scientific execution. Never seed from material truth.

Recommended calibration keeps random nonzero final weights and preserves live
hidden-layer gradient paths. For each original head, let `f(w,x_ref)` be its
pre-sigmoid logits and store detached immutable `c=f(w_initial,x_ref)`. Apply
the original final sigmoid to `target_logit + (f(w,x_ref)-c)`. At initialization
the subtraction is exactly zero under deterministic replay, so six scalar
target logits are exactly the original D0 six logits. This is an explicit fixed
preactivation calibration adapter, algebraically a bias offset over the reals;
its floating-point operation order is part of the new design. It is NOT the
unchanged original A forward. Keep modules registered once, preserve all weight-
norm hooks, and do not recompute/detach `c` each update or detach the live path.

For rotation, use the proposed fixed radius **rho=.5**,
`e0=(cos(2theta0),sin(2theta0))`, `q0=(1+rho*e0)/2`, then finite logit(q0) for the
two original sigmoid outputs. Apply native `2*q-1` and `F.normalize(...,eps=1e-6)`
before supplying the normalized encoding to the unchanged experimental decoder.
Freeze exact arithmetic/dtypes in Phase B; no target-dependent optimization or
search to improve calibration. Preserve scalar width/metallic/color activations.

**Do not promise bitwise theta parity.** D0's unbounded theta versus two sigmoid
channels, centering and normalization cannot be assumed bitwise equivalent.
Recommended gate: six physical scalars bitwise equal; actual orientation encoding/
Q/frame/render correspondence within the original 3D-1 bounds below. Record all
initial discrepancies and actual array hashes. Exact seven-coordinate bitwise
matching would require a different representation/intervention and is an open
approval decision, not a silent exception. If the proposed matching fails,
stop without adjusting rho, seeds, final weights or tolerances.

N1 has eight raw output channels and about 1.49 million weight parameters for
seven physical degrees of freedom. The orientation radius and weight-normalized
parameterization affect optimization even with matching initial material.
This compares architecture PLUS output encoding, not architecture alone. Fixing
or regularizing the radius later would be another scientific intervention.
Report the centered pre-normalization radius and the actual shader-input norm
separately. Apply the existing validity threshold to the actual shader input;
do not invent a new lower-bound penalty/stop on the pre-normalization radius.

Zero-final-weight initialization is not the default: it blocks hidden-network
gradients on the first step. A separately approved N2 could measure that effect;
it is not an automatic fallback if calibration fails.

## Optimization and baseline reproduction

Proposed primary budget: **5000 continuous successful Adam updates per history**,
lr=.01, betas=(.9,.999), eps=1e-8, weight_decay=0, original admitted runtime
defaults, no schedule/clipping/regularizer. Train all N1 material weights; no
illumination/geometry weights exist. Do not restart moments at 250 or 1000.
Keep all 27 histories, including unsuccessful/censored/numerical outcomes.

The reason for proposing 5000 rather than 1000 is explicit: all nine archived
direct Adam1000 states fail Q. A 1000-only success comparison would have a known
zero-success direct control. Five thousand is a fixed candidate budget, NOT a
demonstrated sufficient duration, tuned result or promise of recovery. Its
approval and CPU feasibility are required before freezing. If every direct run
still fails, report that the physical-reliability question remains inconclusive;
equal zero-success counts do not validate neural recovery.

Before any primary history begins, all 27 initial admission checks must pass.
Then run the nine D0 histories first: require exact original 3D-1 prefix through
update1000 (states, original logged losses/gradients, snapshot250/1000 moments,
step counts, effective optimizer flags and saved objective). Store a counted
no-grad fit forward at1000 exactly as the old endpoint measurement; continue
the SAME optimizer for the remaining4000. Do not duplicate the first1000.
Only after all D0 controls reproduce and complete may N0/N1 runs begin. On a
reproduction mismatch, stop globally and preserve evidence; do not compare
neural scores. N0 must then match its entire paired D0 numerical history, not
only the initial loss. No favorable seed/start/checkpoint selection.

**No LM in the primary matrix.** Original LM expects seven variables and its
seven-direction Jacobian; it can apply to N0 but not N1's weights unchanged.
N1 dense float64 H alone would be ~17.66 TB and is rank-deficient in weight
space. Do not relabel output-space LM, freeze-the-trunk LM or a new low-rank/
matrix-free solver as original full-network LM. 3D-1 P0 Adam→LM outcomes remain
archival context with their own budgets, never the matched primary baseline.

## Admission, reproducibility and failure gates

| Gate | Required property | Failure action |
| --- | --- | --- |
| S: authority/evidence | G/A original blobs, all 3D-1 hashes, source extraction whitelist, final protocol/config identity, targets/splits/runtime admitted | Global stop before model execution |
| F: field | Phase C checks query batches1/9/96; primary admission checks two repeated nine-position queries plus the integrated batch broadcast. Same mode/dtype, no spatial/material-truth input, expected topology | Stop all recovery before optimization |
| I: initialization | D0/N0 exact latent/physical/encoding and gradients; N1 six scalars exact, declared orientation/Q match, correct initial hashes and private RNG restoration | Stop; no seed/calibration search |
| P: physical/frame | Widths [.04,1], colors/metallic [0,1], valid nonzero encoding norm>=1e-6, finite outputs; Q `2e-7+2e-6*abs(Q0)`; actual frame<=2e-6 | Stop before optimization |
| R: rendering | Same original batch/tapes. D0/N0 bitwise components/loss; N1 independently every-tape/mean specular+total `2e-6+2e-5*abs(D0)`, diffuse bitwise; exact mean/MSE replay | Global stop before optimization |
| V: sampling/arithmetic | h `2e-6+2e-5*abs(h0)`, wi `4e-6+2e-5*abs(wi0)`, angles<=2e-5 rad, mask XOR0, diffuse directions bitwise; inherited gamma75 assumptions/envelope, old scalar test warning only | Stop, preserving all negative diagnostics |
| G: gradients | Finite renderer backward; all seven physical directions locally reachable in field-only fixture; expected active head/trunk gradients distinguished from structural zeros | Stop on invalid graph; do not demand every weight be nonzero-gradient |
| B: baseline | D0 archived1000 prefix exact; N0 paired D0 trajectory exact; no unintended new RNG/target/shader path | Global stop before interpreting N1 |
| E: evaluation | Own material truth/targets, exact split and external tapes, correct decoder/metric branch, complete status/work/input identities | No admitted recovery result if invalid |

Gauge metamorphic fixtures retain Q and theta periodicity/axis-swap equivalence,
with the accepted signed quarter-turn sample coupling for paired representations.
They are separate Phase C validation probes, not training-time recoupling or
extra recovery arms. No new tolerances after results. Field-only probes may
check analytical reachability; do not form a full shader Jacobian over N1 weights.

Structural gradient exceptions must be predicted before the run: ReLU-inactive
units, constant-coordinate effective-weight columns and orientation radial gauge
directions. Raw weight_v columns need not have zero gradients when an effective
input column is zero, because weight normalization couples row coordinates.
Do not turn an observed missing whole head gradient into an expected exception.

Keep original Adam decoder behavior: log chart/cut crossings, but do not transplant
LM's proposal-censor policy into Adam or wrap/fix angles. Invalid zero/near-zero
N1 encoding fails the existing shader gate; no fallback direction/radial penalty.
Source/target/tape/reproduction violations stop globally. After admission,
isolated nonfinite numerical failures stop that arm, preserve its state and may
continue remaining arms only under a precisely frozen failure taxonomy. Time-
limited arms are censored, not replaced. Incomplete cases never pass the primary
completed5000-update reliability test; valid partial metrics are labeled descriptive.

## Evaluation and accounting

Keep the five simultaneous criteria UNCHANGED: Qrel<=.05; identifiable orientation
error<=5 degrees; maximum albedo absolute error<=.02; metallic absolute error<=.03;
held-out total RGB relativeL2<=.01. Null/nonidentifiable orientation is not zero
and does not pass. Keep sorted axes, gauge-invariant Q, physical/encoding
identities, per-component fit/held-out/relight metrics, seen/unseen-view relight
subsets, residual compensation, gradients, status and cost. Theta for reporting
comes from the actual shader decoder convention; do not silently sort axes or
wrap neural state during optimization.

Exactly one external84-observation evaluation at the final5000 state per arm;
no best checkpoint, held-out stopping, external evaluation at every update or
selection of a favorable representation. Save checkpoints at250/1000/5000 and
initial identity. Snapshot250 objective reuses update251's ordinary forward;
snapshot1000 has one explicit fit-only measurement; final5000 has one terminal
fit forward/backward without a step. Initial admission includes one fit forward/
backward. Clear admission/terminal gradients appropriately; optimizer counters
remain exactly5000. Intermediate logged physical values are PRE-update values
from existing forwards; avoid extra N1 queries to fabricate after-update curves.

Equal-update primary comparisons use5000 updates and equal shader sample counts;
they are NOT equal parameter/FLOP/backward compute. N1 evaluates its constant
MLP once per shader batch and broadcasts; log field calls separately from shaded
points. Compare physical curves at shared updates/renderer work and exact saved
states at predeclared active-time checkpoints; no interpolated orientation or
new held-out renders. Report wall/CPU, peak RSS, parameter count/bytes, field and
shader calls, reverse sweeps, optimizer returns and all failed attempts. No
architecture-efficiency claim from equal reverse-call counts alone.

Proposed CPU ceiling, pending Phase C target-independent feasibility checks:
one thread, hidden GPUs, **28800 seconds whole process**, **900 accumulated active
seconds/history**,4 GiB RSS/8 GiB address backstop. Retain at least10 active seconds
for terminal work without reducing the exact5000-update cap. A run that cannot
complete within its cap is incomplete/censored; no extension or GPU fallback.
3D-1 measured ~96–100 seconds/1000 direct Adam updates, but N1 runtime has NOT
been measured. GPU is unnecessary for this proposal; native A construction or
GPU execution would need a different admission/runtime study.

| Proposed maximum primary work | Per history | All27 |
| --- | ---: | ---: |
| Adam returns |5000|135000|
| Fit shader forwards (initial + updates +1000 +terminal) |5003|135081|
| External shader forwards |1|27|
| All shader forward attempts/completions |5004|135108|
| Renderer backward/reverse sweeps (initial +updates +terminal) |5002|135054|
| Fit shaded points |480288|12967776|
| External shaded points |336|9072|
| Total shaded points |480624|12976848|
| Diffuse directions |246079488|6644146176|
| Specular directions |492158976|13288292352|
| JVPs / LM solves / new quadrature |0|0|

Two field-only constant/replay probes per model add54 field queries, zero shader
calls, no backward. Nine N1 initialization-calibration passes add nine more
field-only computations, counted separately:63 extra field computations total.
Field reachability/gauge fixtures belong to a separately bounded Phase C
validation budget, not hidden scientific-run calls. Primary field calls: N0
5006 each (5004 integrated +2 probes), N1 5007 each (plus its calibration pass);
D0 has two field-decoding probes plus its original cheap physical transforms.
Numerical attempts and completed operations must be distinct. No resumption/
retry after an execution stop.

Store large N1 weights/moments/RNG/named-parameter identities as owned numeric
artifacts outside Git, with explicit JSON manifests/hashes and safe pickle-free
arrays. No historical checkpoint deserialization. Commit only bounded summary
evidence and references; do not put hundreds of MB of repeated neural weights in
the repository. Freeze checkpoint schema and storage/counter overhead in Phase B.

## Hypotheses, reporting and interpretation

H1 N0 is an exact differentiable/optimizer wrapper control; H2 calibrated N1 has
admitted constant, physically matched outputs and live gradients; H3 physical
recovery of N1 at the fixed update budget matches or differs from D0 across this
panel; H4 representation changes conditioning/optimization trajectories; H5
RGB/material ambiguity persists; H6 precision/encoding/radial/decoder issues
limit recovery. Classify confirmed, supported but limited, contradicted or
unresolved using measured gates/errors. Three paired starts/profile are a
descriptive benchmark, not a statistical noninferiority or population claim.
All-zero direct/neural success is inconclusive for physical reliability.

Required tables: (1) source/runtime/admission status and missing evidence;
(2) per material and arm completed/failed/censored counts, five-criterion passes,
median/range Q, orientation/color/metallicity and held-out/relight components;
(3) all27 initial/1000/final states and criterion failures; (4) D0 reproduction
and N0 exact paired parity; (5) N1 calibration/radius/frame/gradient diagnostics;
(6) update/work/time/parameter/storage cost; (7) clearly separate archival 3D-1
LM context. Preserve every unsuccessful start and all original evidence.

Expected future artifacts: final protocol/config/freeze after review; source/
target/runtime admission manifest; separate fresh owned run output; all27 traces,
numeric checkpoints and hashes; versioned primary/analysis JSON, figures and a
concise results report. No current document constitutes an executed gate or
result for 3D-2. Phase A stops here; implementation, validation and scientific
execution each require the later instruction described in the plan.
