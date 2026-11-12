# 3D-2 Phase A implementation plan and review decisions

**Status: design complete for review; implementation/execution not authorized.**
Read [the source/evidence audit](A_NEURAL_MATERIAL_FIELD_AUDIT_V1.md) and
[proposed protocol](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1_DRAFT.md).
Current HEAD is accepted 3D-1 `ccb03d0ec5adc814775538b9cc7427cc57480a98` in the
separate `feat/neural-material-field-design-v1` documentation worktree. No code,
old evidence, old protocol or AGENTS.md was edited, and no model/shader execution,
training, optimization, Docker, GPU, dependency or permission change occurred.
No protocol freeze or commit/push is produced by Phase A.

## Decisions that must precede a final freeze

| ID | Unresolved choice | Recommended choice and implications |
| --- | --- | --- |
| D1 | What constitutes an A-faithful N1? | Reuse exact eight-layer feature trunk and five weight-normalized heads, including eight raw output channels. Constant anchor substitutes both spatial paths. This tests architecture plus orientation encoding; do not claim an architecture-only causal effect or native A pipeline equivalence. |
| D2 | How to match initialization without disabling the trunk? | Fixed pre-sigmoid centering around original nonzero random weights, preserving original head sigmoids and material decoding. Six scalar logits/physical values exact; rho=.5 orientation calibration with bounded physical/render matching using original tolerances. This is a new explicit calibration adapter; live CPU parity is unverified. A requirement for bitwise all-seven theta equality is incompatible with promising faithful A8 behavior without further design. Reject on failed gates, do not tune the initializer. |
| D3 | What fixed primary update budget is scientifically useful? | Propose5000 continuous Adam updates for ALL D0/N0/N1 histories, retaining250/1000/5000 checkpoints. All nine archived direct Adam1000 states fail Q; use1000 as exact reproduction prefix and secondary checkpoint, not the only success endpoint. No evidence yet shows5000 suffices. If all direct/neural runs fail, the physical-reliability conclusion is inconclusive. This budget increase requires review before implementation freeze. |
| D4 | Should original LM optimize N1? | No. Use equal-update Adam as the primary comparison; keep 3D-1 direct Adam→LM as archival context only. Original LM is seven-dimensional; dense full-weight N1 H would need~17.66 TB. Output-space LM/last-layer LM/dual solvers require distinct scientific designs and would confound the primary comparison. |
| D5 | What resource and compute comparison is feasible? | Existing CPU runtime, one thread, hidden GPUs, proposed8-hour whole/900-second active-history cap,4 GiB RSS/8 GiB address cap. Validate target-independent N1 cost before freeze; actual N1 throughput is unknown. Match renderer work/updates, separately report field/parameter/backward/time costs. No automatic GPU fallback or budget extension. |
| D6 | How are decoder/gauge failures handled? | Preserve original Adam behavior and A2/atan2. Log chart crossings; do not silently apply LM's chart-censor controller to Adam. Reject invalid near-zero encoding without fallback/regularization. Freeze global integrity stops versus isolated numerical failures/censoring before execution, preserving every unsuccessful arm. |
| D7 | Where are weights/evidence written and when is the protocol frozen? | Explicit new owned run root, unique ID, validated writable destination, pickle-free own numeric checkpoints with named-parameter/state hashes; small versioned JSON/report references only. Select/probe the destination during a later authorized phase, without permission changes. Freeze source/tests/initial states/tapes/calibration/work/runtime only after D1–D6 and Phase C pass. |

The optional N2 zero-final-weight initialization experiment is deferred. It would
separate initialization-induced first-step hidden-gradient starvation, but adds
a scientific factor and is not a fallback arm. Do not expand the27-history
matrix after seeing performance or replace original A with a smaller surrogate.

## Proposed Phase B: implementation only, after a separate instruction

Create new versioned files; preserve every existing scientific/experimental file:

| Proposed new file | Responsibility |
| --- | --- |
| `experiments/neural_material_field_v1.py` | Admit exact A source definitions through an AST whitelist; CPU component construction, D0/N0/N1 material bundle, fixed spatial anchor, original decoding, declared calibration buffers and model identities. No native renderer construction or global source patching. |
| `experiments/neural_material_recovery_v1.py` | Closed bounded experiment entry point; explicit accepted inputs; original data/illumination composition; strict run directory; continuous Adam;27 initial admission records; all D0 reproduction gates before neural runs; snapshots/work/status/RNG/resource accounting. No historical runner invocation. |
| `experiments/analyze_neural_material_recovery_v1.py` | Saved-data-only physical/component/gauge metrics, paired comparisons, cost tables and figures. No new inference, targets, derivatives or optimization for analysis. |
| `tests/scientific/test_neural_material_field_v1.py` | Exact source/topology/decoder tests; multiple-position and batch constancy; no hidden raw-xyz path; initializer independence; calibration replay; nonzero encoding; expected gradient structure; N0 identity control; CPU/import/RNG behavior. |
| `tests/scientific/test_neural_material_recovery_v1.py` | Fresh state/target/tape admission; original operation-order parity; continuous optimizer/snapshot semantics; direct prefix and N0 full-history comparisons using saved fixtures; stop-before-optimization gates; sample/work/time caps; failure persistence; owned pickle-free state/storage. |
| `docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.md` | Final reviewed protocol, retaining this Phase A draft and decision history separately. Exact model/init/numerical settings and work/failure taxonomy; no unimplemented configuration options. |
| `docs/NEURAL_MATERIAL_FIELD_RECOVERY_V1.md` | User-facing runner/admission/storage/validation documentation; explicit distinction from native A training. |

Only if needed, update AGENTS.md's authorized milestone section in that later
phase. Do not edit `scientific_sources/`, any old protocol/freeze/evidence,
`material_shader_v1.py`, its estimator, metrics or the3D-1 implementation. The
new adapter can compose original `ObservationBatch` scene/tape/illumination data,
but N1 must pass its actual normalized encoding to `shader.shade` directly.
No theta→cos/sin round-trip inserted into the N1 rendering gradient path.

Do not add schema options, new storage frameworks or native training integration
unneeded by this closed analytical experiment. Keep field-only computations,
renderer calls and optimizer work separately countable. Store material truth
only in reference/evaluation association, never the field's inputs/initializer.

## Proposed Phase C: bounded admission and implementation validation

No full recovery or reference-target regeneration. Run only new focused tests
and explicitly approved tiny CPU parity/feasibility fixtures. Suggested separate
validation ceiling:300 seconds/4 GiB, at most64 full-batch shader forwards and64
renderer backward sweeps, plus at most200 field-only VJP probes. Exact fixture
counts must be specified before this validation, not treated as spare optimizer
starts. Synthetic scalar optimizer fixtures may verify state semantics; do not
run a complete neural material recovery as a test.

Verify original source bytes/ASTs and module dimensions, weight_norm g/v/b names
and live parameter count1485584. Test standalone original components against the
source-extracted components before calibration. Keep the original deprecated
weight_norm behavior; do not modernize it silently. Demonstrate deterministic
local construction and restoration of global RNG, constancy at multiple points
and batch sizes, calibration arithmetic and matching gates. Unit tests must
reject zero/near-zero rotation, incorrect head order, squared-width mistakes,
detached live paths, duplicated module registration and unapproved activations.

Verify D0/N0 exact outputs/gradients/Adam transitions on cheap fixtures. For N1,
check seven physical directions are reachable through the field, not a full
72×P shader Jacobian, and classify structural gradient zeros in advance.
Capture same-return sample/frame/component diagnostics for parity. No relaxed
mask/component test or repeated initializer to obtain a desired pass. Use
target-independent timing/parameter byte estimates to assess the proposed
execution cap; if infeasible, stop for a revised design BEFORE final freeze.

After all design decisions and fixtures pass, produce a reviewed
`NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json` containing final protocol/source/
test hashes, accepted3D-1 identities, fixed material/initialization/tape/model
state hashes, radius/calibration/anchor, optimizer/runtime/evaluation policy,
checkpoint schema, exact counters/caps and failure taxonomy. This freeze is
not authorized or created in Phase A. Passing fixtures do not authorize Phase D.

## Proposed Phase D: one later-authorized scientific execution

Re-admit every frozen source/input, ensure a fresh explicit output directory,
then construct all27 initial states and apply all admission gates before any
optimizer update. Perform all nine direct D0 histories first, reproducing each
archived1000 prefix within its continuous5000 updates. Do not re-execute an
old runner or duplicate a prefix. Stop globally if exact reproduction fails.
Run N0 with exact paired-history checks, then N1 with the admitted initializer,
under the fixed update/work/resource limits. Keep failures/censoring and never
replace seeds, materials, model weights or tapes. Do not select an intermediate
state by loss/held-out score. Evaluate only planned final states on the original
external tapes. Numeric checkpoints are owned outputs, not modifications to
historical checkpoints or a general resume feature. No retries/extensions.

## Proposed Phase E: offline analysis and review

Analyze saved states/predictions only. Required rows include every planned arm,
actual admission/completion status, initial/1000/final physical values and errors,
all five recovery tests, component and relighting errors, paired direct/neural
differences, gradient/encoding diagnostics, counters/CPU/time/storage and all
negative results. Physical Q and error metrics do not become training losses.
No claim of equality/noninferiority from a tiny count or joint failure. Compare
original3D-1 LM only as a separately labeled archival reference; preserve all
previous reports. Commit/push or launch further neural/GPU experiments only if
the later instruction authorizes it.

## Immediate recommended action

Review D1–D7 and approve a **Phase B implementation-only** task, initially limited
to the admitted A component extractor, constant material interfaces, initializer
and focused CPU fixtures. Keep the candidate5000-update/CPU budget and the
bounded-orientation matching rule explicit in that approval. Do not begin
Phase C shader validation, freeze a protocol or run recovery merely because
this Phase A design exists. If bitwise seven-coordinate matching or unchanged
full-native-A semantics are required, resolve that incompatibility first.
