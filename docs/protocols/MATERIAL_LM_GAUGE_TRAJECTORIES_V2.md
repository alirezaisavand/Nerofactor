# Controlled LM gauge trajectories v2 — frozen 3C-9c protocol

Base `c37337e195db57b9a331021d9a55147abf866bb1`; branch
`feat/material-lm-gauge-trajectories-v2`, separate worktree. ONE15-arm CPU
optimization experiment, no pilot/retry/replacement, new population or optimizer.
Create new v2 source/tests/protocol/freeze/evidence/results. All prior scientific
source/protocol/evidence, including failed3C-9 and failed3C-5, stays immutable.
Do not edit or execute the historical runners. Read-only admission helpers and
explicitly reused LM/branch/evaluation functions are permitted.

## Admitted chain and scientific computation

Admit complete3C-1–3C-9b through existing checked read-only helpers. Check all
55 chain-derived source/protocol/evidence identities, scenes, original target
arrays, frozen G/A and accepted shader/sampler source bytes before execution,
and recheck afterwards. Check3C-9b's source/test/protocol freeze and six complete
forward-only records, plus its unchanged RNG/tapes/runtime flags. In particular:

| Artifact | SHA-256 |
| --- | --- |
| 3C-9 failed raw report | `88b7e83ec7f79b3e9f476c9aa5ccd199e5394342ca83205394188f800cfb8f9a` |
| 3C-9b primary report | `4753e1e6d971b033d24d686e146ab0d22dfd4941eee3d03df9780dc0fce37c5b` |
| 3C-9b offline summary | `510775832a45f9ec1956f2cc6ab5fc27ce0a1f9d5a88a9ff53e8944a4dbf934c` |
| 3C-9b freeze | `87788c2dbae00cce036080f44f6b66151e0ad4cf192e03e54c55c98f99a78527` |
| 3C-9b results | `ec0b709230dd4348ee8d0db832e5c03dd39c6af5649a68ac79ed8d8af216132f` |

Exactly five complete3C-3 update1000 vectors, seeds101–105. Restore exact original
float32 seven-coordinate/physical records with shape/dtype/value/hash checks;
no Adam construction. For each seed allocate/run P0, P1-U, P1-C, in seed order.
P1 swaps original width logits and performs the original ONCE-float32 theta+pi/2
translation, other logits unchanged. P1-U and P1-C coordinates must match exactly.
Use original tapes for P0/U. Determine signed quarter-turn from actual decoded
96-point flat frames before any losses. Copy C tapes ONCE using accepted float64
v-s/4, modulo, float32 conversion, second periodic endpoint reduction; clone u
and diffuse azimuth unchanged, storage independent. Do not reselect sign/tapes.

Unchanged A2/atan2, original parameter transformations, flat geometry, lighting,
visibility,24fit/24held-out/36relight observations, targets, training tapes0–3,
evaluation tapes1000–1003,512 diffuse/1024 specular samples and float32
mean-before-MSE. No alternate encoding, shader, BRDF, loss, regularizer, target,
precision, optimizer, schedule or neural/GPU experiment.

## Initial forwards: all15, then all five gate sets

Before any optimization, allocate all15 arms and execute exactly one original
fitting forward per arm. Reuse v1 Arm/forward numerical operations. An observer
subclasses the original mask observer: call the original shade ONCE, copy its
returned h/wi/PDF/diffuse directions/frames/RGB and active argument encodings/
views/tape hashes, return the SAME output object. No second render, source patch
or profiling intervention. The training objective and initial autograd-enabled
forward path remain the same as historical P0. These are forward-only initial
checks; copying/detaching observations does not run AD or consume RNG.

Store initial scalar/mean RGB/mask/source-array identities, full per-tape RGB and
branch charts. Evaluate A–E for ALL five seeds before enabling LM. Record every
available failed gate; any failed mandatory check stops before any update.
Unsafe source/runtime/input construction failure may stop earlier. Never omit
a seed, adjust bounds, regenerate tapes/encodings or rerun. The old scalar loss
gate is now a reported warning only; failed3C-9 remains failed.

### A: exact identity

P0 initial latent AND physical descriptions, all initial predictions/hash, mask
and original MSE equal historical3C-4 exactly. Original fitting/evaluation tapes,
target float32 conversion, scene and observation order, shader/sampler hashes
must match accepted records. Numerical runtime exactly Python3.9.12/NumPy1.21.5/
Torch2.4.0+cu121 CPU/one thread. Runtime/source violations are fatal before rendering.

### B: physical gauge

Elementwise `abs(Q1-Q0)<=2e-7+2e-6*abs(Q0)`; albedo/metallic byte-identical;
P1-U/P1-C initial states identical. Actual RETURNED T/B frames must be uniform
throughout96 points. Verify the quarter-turn at inherited absolute2e-6 tolerance
and its sign must equal the pre-loss construction sign. No loss-based sign choice.

### C: samples and masks

Capture during existing P0/C initial forwards; no additional calls.

- World h components: `2e-6+2e-5*abs(h0)`.
- World reflected wi components: `4e-6+2e-5*abs(wi0)`.
- Both maximum angular differences <=2e-5 radians, using atan2(norm(cross),dot).
- Mask XOR exactly0, diffuse directions byte-identical, active original/copied
  tape-array identities unchanged.

Record component/vector/per-tape errors, angular errors and PDF discrepancies.
PDF is not an equivalence gate; report world abs(wo-dot-h)<=1e-5 separately as a
float64 diagnostic indicator, not the original clamped PDF local. Preserve the
known43 historical3C-6 exceptions without claiming this is their reproduction.

### D: independent RGB components

Separately compare per-tape AND mean diffuse/specular/total RGB. Diffuse must be
byte-identical. Specular and total must each satisfy elementwise
`abs(C-P0)<=2e-6+2e-5*abs(P0)`, independently for ALL four tapes and mean.
Record max absolute, RMSE, relative L2 (floor1e-30), per-channel maxima, per-tape
differences and exceeding-element counts. Recreate original independent
component-wise contiguous Torch four-tape means exactly. No total-only
compensation may hide a failing component.

### E: arithmetic consistency, conditional on A–D

Record all three original float32 MSEs and `old_scalar_gate_passed`, absolute
difference, `1e-12+5e-7*abs(L0)` bound and signed excess. A false old scalar gate
does not fail E. Preserve exact values and historical failed status.

Recompute original loss from saved initial RGB using original float32 subtraction,
square and Torch mean; require exact scalar agreement. Use3C-9b residual identity:
float32 r0/r1 promoted to float64, dr=r1-r0, compare mean(r1²)-mean(r0²) with
mean(2*r0*dr+dr²), with the same64*eps64 cancellation envelope and exact dyadic
Fraction identity. No new shader, Jacobian or objective arithmetic is introduced.

Before execution justify and test this bound, without seed102-derived constants:

```
n = 72, u = 2^-24, gamma75 = 75*u/(1-75*u)
epsilon_i = abs(C_i-P0_i) + u*(abs(P0_i-target_i)+abs(C_i-target_i))
Bresidual = mean(2*abs(r0_i)*epsilon_i + epsilon_i^2)
Breduction = gamma75*(mean(r0_i^2)+mean(r1_i^2))
Btotal = Bresidual + Breduction
require abs(L1-L0) <= Btotal
```

The implementation constructs epsilon, both terms and observed difference with
exact rational arithmetic on float32-representable operands; no float64
round-down can loosen/incorrectly tighten the comparison. Report each bound as
the next float64 value upward. This is an arithmetic check based on observed RGB,
NOT independent evidence for material/rendering equivalence. It cannot override
any A–D failure.

### Verified arithmetic assumptions and source proof

The installed Torch reports Git revision
`e4ee3be4063b7c430974252fdf7db42273388d86` (same as reviewed v2.4.0 sources).
Read-only source review (not execution/download of packages) used:

| Primary source | SHA-256 of reviewed UTF-8 source |
| --- | --- |
| [ReduceOps.cpp](https://github.com/pytorch/pytorch/blob/e4ee3be4063b7c430974252fdf7db42273388d86/aten/src/ATen/native/ReduceOps.cpp) | `9ac64ff7e7dead6f4ccac17fad6162d1d82233e8a7e7d119fa56d53e42fb7eb7` |
| [SumKernel.cpp](https://github.com/pytorch/pytorch/blob/e4ee3be4063b7c430974252fdf7db42273388d86/aten/src/ATen/native/cpu/SumKernel.cpp) | `6639e5b687c24fbda26f8e4e5d22fec6c54065a5ecbbc43273b12a00b509981e` |
| [PowKernel.cpp](https://github.com/pytorch/pytorch/blob/e4ee3be4063b7c430974252fdf7db42273388d86/aten/src/ATen/native/cpu/PowKernel.cpp) | `4b7f34db7eead1db381470e9fa603462a8ba65259d809cc7416dba8d457df82e` |

CPU float32 exponent2 uses one multiplication. CPU mean is a sum followed by
scalar scaling. The sum kernel loads each operand once into cascade/vector
partial sums and combines them by additions; no cancellation, subtraction or
nonlinear transform for nonnegative squares. Ignoring exact additions of zero,
it is a binary tree with72 leaves, so any input's path contains at most71
nontrivial additions regardless of SIMD width/association. Allow one additional
final-store cast and at most two mean-scaling roundings (rounded reciprocal plus
multiply conservatively covers a single division). With one square rounding,
the maximum path is71+1+1+2=75. Standard gamma75 therefore bounds the square/
sum/mean error relative to exact mean of the already rounded residual squares.
Subtraction errors are handled separately by epsilon. Normal/finite arithmetic
and the triangle/residual perturbation inequalities give Btotal above.

Before the run, independent small synthetic fixtures compare actual Torch MSE
to exact Fraction sums for multiple fixed magnitudes/orderings; verify the full
envelope on adjacent float32 inputs. This does not execute a scientific shader
or optimizer. Runtime additionally requires the reviewed Torch Git version,
default float32, FE_TONEAREST, and an independent ties-to-even scalar probe.
Gate E checks that every nonzero RGB/target/residual/squared residual and exact
subtraction is normal; zero is allowed. Sum(r²)<float32max/8 guarantees no sum
overflow, and nonzero means must be normal. Fail rather than invent an underflow
allowance or change the bound. Inputs not meeting assumptions stop the experiment.

This source/runtime/probe justification is not a formal binary build-identity
proof. Exact historical P0 replay remains mandatory. No network/source retrieval
is needed during scientific execution; reviewed source identities enter the
pre-execution freeze.

## Unchanged LM and branch policy

Reuse frozen v1.run_arm/guarded_loop: exact original3C-4 run_lm bytecode in its
private observational namespace, original normalized residual/JVP/solve/SVD,
candidate rounding/prediction/controller and evaluation helper. Do not modify
any historical source module globals or files.

Each arm starts fresh lambda=.01; bounds1e-8..1e8,80 outers/four trials,
H=J.T@J, damping diag(H.diag()+1e-8), original float32 shader/AD, float64 small
solve, candidate `(z.double()+delta).float()`. Original represented-step
prediction, acceptance ratio>.1 plus actual decrease/positive prediction,
lambda /3 if ratio>.75, *2 if ratio<.25 otherwise unchanged; rejection *10/cap.
Rejection already at1e8 saturates. No old terminal damping restored, no LR/
regularizer/clipping/new trust-region rule.

Freeze initial own and counterpart actual decoder cells exactly as v1: P0
counterpart once-float32 +pi/2, P1 inverse -pi/2. Cell inferred from actual
normalized double-angle atan2, independently checked analytically away from
cuts. Check BOTH cells on every represented proposal BEFORE rendering or
predicted-reduction rejection; first crossing terminates branch_crossing_blocked
without rendering, acceptance, rejection, damping change or retry. Preserve prior
accepted states and all blocked-proposal diagnostics. Coupled accepted states
must preserve the actual initial local frame sign. Such a stop is censored,
not an unrestricted failed recovery. No angle wrapping/canonicalization.

## Histories, derivatives and terminal evaluation

Preserve all original histories, proposed mathematical/represented steps,
predictions, masks, trial decisions, damping, state/clock/work snapshots and
negative/stagnant/saturated outcomes. At the common INITIAL state compare J/H,
latent gradient, mathematical/represented proposal, prediction and proposed Q.
Use inherited3C-7/3C-8 diagnostic bounds, not new winner-selection thresholds.

The inherited budget permits seven LATENT directional JVPs per linearization,
not seven extra physical ADs. Report physical sensitivity via analytical inverse
coordinate-transform diagonal at float32 sigmoid outputs, applied to the saved
initial latent normalized J. Label these gradients/J as CHAIN-DERIVED, not
independently executed physical-coordinate AD or bitwise native physical
derivatives. The native latent gradients/initial proposals remain unchanged.

Compare new P0 to archived3C-4 initial and shared outer starts, J hash/H/gradient,
all shared proposals/rounding/prediction/damping/decisions, masks and full endpoint
when uncensored. Ignore instrumentation/time. A deliberate branch stop permits
only shared-prefix comparison. Unexpected finite parity failure is retained as
a FAILED control; no correction/retry or clean-replication claim for that seed.

Align trajectories separately by outer-start index and accepted-update count,
recording actual underlying states. Map P1 back into P0 coordinates; Q is gauge
invariant. After divergence do not call matching indices identical physical
states. First practical divergence uses the final similarity criterion below.

Terminal state is the state at frozen stopping condition, never best-held-out
selection. One terminal fit forward/backward and original84-observation evaluation
with original tapes1000–1003. P1 physical widths swapped/theta-pi/2 ONCE in
float32 for common-gauge evaluation; record ideal/rounded mapping and Q roundoff.
Original success assessment unchanged: Qrel<=.05, orientation<=5deg, max albedo
error<=.02, metallic<=.03, held-out total RGB relative L2<=.01. Record all splits,
components, compensation, final parameters/Q, gradients, costs and termination.
Report censored/budget-limited endpoints descriptively and separately in counts.

Cross-arm practical similarity for TWO COMPLETED trajectories only: Qrel<=.005,
max albedo difference<=.002, metallic difference<=.002. Orientation/objective/
history/termination separate. Censored/incomplete pair verdict stays censored,
never forced pass/fail. No automatic coupling-winner claim.

## Budgets, failures and exact accounting

Exactly15 planned arms,80 outers/four trials,120 accumulated active wall seconds
each INCLUDING initial forward/capture/persistence, with10 reserved for terminal
work. Shared construction/gate arithmetic is charged to whole-run setup, not
one selectively penalized arm. All work:2400s whole,4 GiB peak RSS,8 GiB address
backstop, CPU one Torch/BLAS/OpenMP thread, GPUs hidden, no dependencies installed.

| Maximum work | Per arm | Whole15 |
| --- | ---: | ---: |
| Initial forwards | 1 | 15 |
| Jacobian constructions | 80 | 1200 |
| Directional JVP/primal forwards | 560 | 8400 |
| Reverse sweeps including terminal gradient | 1121 | 16815 |
| Linear solves/attempted proposals | 320 | 4800 |
| Candidate objective forwards | 320 | 4800 |
| Terminal fit forward/backward | 1/1 | 15/15 |
| Common-gauge evaluation forwards | 1 | 15 |
| Total full-batch forwards | 883 | 13245 |
| Fit sampled points | 84672 | 1270080 |
| Evaluation sampled points | 336 | 5040 |
| Total sampled points | 85008 | 1275120 |
| Diffuse directions | 43524096 | 652861440 |
| Specular directions | 87048192 | 1305722880 |

Reuse inherited caps and add explicit phase COMPLETION counters to existing
phase attempts. Evaluation attempts, terminal gradient completions, JVP/reverse/
Jacobian/solve attempts/completions are separate. Completed work cannot exceed
attempts; accepted+rejected+branch-blocked cannot exceed attempted solves.
Branch blocks count as solves, neither accepted nor rejected. Initial capture
does not add any full forward. No extra AD for physical diagnostics.

Arm budget exhaustion retains last completed state and attempts endpoint work
only within its original allowance, with explicit incomplete/censored status.
Whole budget, memory, identity, construction and unrecognized runtime failures
stop globally. Ordinary finite poor accuracy/consistency and censored outcomes
continue remaining predeclared arms. Two known nonfinite-output exceptions
(original shader prediction or directional derivative) may be isolated as an
arm failure ONLY with finite unchanged latent state and unchanged tape hashes;
other exceptions stay fatal. No retry or repaired state. Preserve all partial data.

## Freeze, tests, one execution and analysis

Freeze runner/tests/protocol SHA, canonical POLICY and CAPS hashes, reviewed
Torch source identities and focused test result BEFORE scientific execution.
Output uses exclusive create at `docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json`.
Strict JSON validates before replacing each owned partial snapshot. Restore
thread/RNG/memory/signal state; unexpected state mutation invalidates the report.

Only new synthetic/read-only tests: admitted chain/restoration/tapes/frames,
independent sample/mask/component gates, scalar-warning versus component failure,
arithmetic model/exact oracle/assumption rejection, original J/solve/controller,
historical first-solve metadata, synthetic original-loop prefix parity, natural
branch blocking/zero-step saturation without patching old policies, common-gauge
evaluation, recovery/caps, strict partial JSON and source preservation. No old
suite or scientific experiment reruns, no pilot optimization.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest tests.scientific.test_material_lm_gauge_trajectories_v2 -v

# Exactly ONE invocation after freeze; external timeout includes imports:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 2400s \
python3 -B -m experiments.material_lm_gauge_trajectories_v2 \
  --output "$PWD/docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V2.json"
```

Afterwards generate figures ONLY from saved evidence: loss versus outer index,
accepted updates, active wall and full-forward work (including JVP primals),
Q-error versus accepted updates, damping versus attempted outer. Endpoint points
use the last attempted outer index, never an implicit reset to zero. Mark
censored endpoints with x; preserve rejected/blocked trials in JSON. Normalize
new SVG whitespace before FIRST hashing, record figure identities separately.
No shaders/derivatives/solves in plotting or scientific-results analysis.

H1 independent initial equivalence, H2 initial useful proposals, H3 trajectories,
H4 recovery, H5 numerical/branch limits, H6 historical-group explanation receive
separate confirmed/supported-but-limited/contradicted/unresolved verdicts. H6
causation, global convergence, unbiased optimization or neural inverse-rendering
improvement cannot be inferred from this bounded flat-material study. Retain
all fifteen outcomes and negative results. Commit/push only this feature branch,
then stop without merge, additional diagnostic or next experiment.
