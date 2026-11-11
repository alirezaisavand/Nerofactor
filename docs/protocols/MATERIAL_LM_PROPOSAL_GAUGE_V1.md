# Offline LM proposal gauge-consistency audit v1

Protocol `material-lm-proposal-gauge-v1`, milestone **3C-8**. Freeze before one
scientific execution. Base `31dd7979256795abd8e0043099c649518aabad35`, branch
`feat/material-lm-proposal-gauge-audit-v1`, separate worktree. No renderer,
new target, differentiation, optimizer, update application, damping sweep,
historical rerun, new dependency, GPU or Docker execution. Fifteen hypothetical
next proposals only, preserving all previous code/evidence and user changes.

## Damping admission established before implementation

Read-only inspection of the accepted 3C-4 evidence and `run_lm`, `proposal`,
`accepted_lambda`, `bounded_lambda`, `acceptance`, and `Arm.forward` establishes:

- Initial lambda=.01, min=1e-8, max=1e8; at most 80 outers/four trials.
- Each attempt records its **used** damping. On acceptance, new<old and
  ratio>0.1; lambda divides by 3 if ratio>.75, doubles if ratio<.25, otherwise
  stays, with the original min/max clamp.
- A rejected trial at lambda<max multiplies lambda by 10 and clamps, including
  after trial 4. Rejection **at** max terminates `damping_saturated`.
- `lambda_before` is the outer linearization's input; `lambda_after` and
  outcome `damping` are the post-controller value attached to `z_after`.
  The trajectory records that same state/damping; endpoint measurement leaves
  the state unchanged. An accepted trial's used damping is not generally the
  terminal damping.

The exact terminal records observed before implementation are:

| Seed | Outcome index | Terminal outer | Terminal trial | Terminal lambda | Termination | Last accepted outer/trial |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 1 | 14 | 1 | 100000000 | damping_saturated | 11/1 |
| 102 | 3 | 17 | 3 | 100000000 | damping_saturated | 14/4 |
| 103 | 5 | 11 | 4 | 100000000 | damping_saturated | 9/3 |
| 104 | 7 | 9 | 2 | 100000000 | damping_saturated | 5/1 |
| 105 | 9 | 14 | 4 | 100000000 | damping_saturated | 11/3 |

All terminal trials were rejected for zero/negative/nonfinite predicted
reduction, with saved represented steps zero and predicted reduction -0.0.
The final logged state is unchanged by those terminal trials and matches the
exact 3C-7 endpoint. The new diagnostic must independently replay the **logged
controller decisions only**, from initial .01, checking every ordered outer/
trial, state/MSE continuity, acceptance decision/ratio, post-controller lambda,
terminal state hashes, final trajectory and endpoint. No past solve, render or
optimizer iteration is recomputed. Missing, unfinished, contradictory, nonfinite
or out-of-range provenance is fatal. Record exact zero-based JSON pointers and
one-based scientific outer/trial identifiers, including last accepted trial.

Use each terminal lambda identically for P0/P1-U/P1-C. All five are saturation
endpoints, so every proposed next step is explicitly **counterfactual**; the
original optimizer would not execute a next iteration. Do not replace lambda
with the last accepted trial value, reset it to .01, or select a lower value to
make the result more informative. Small/zero proposed motion is an admissible
scientific outcome, not permission for another run.

## Source, evidence and state identity

| Input | SHA-256 |
| --- | --- |
| `docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json` | 66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f |
| Original 3C-4 optimizer source | 4abdf70b1a8c3b309f32561e31bcdb605392d0858bbe2015bf98c1a8f065cb6e |
| `docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json` | 58a47eca9978e32385d94c1b3c1f087b2be0dff15fb095fe95b3e5601bc34070 |
| `docs/MATERIAL_GRADIENT_EQUIVARIANCE_V1_RESULTS.md` | f92e5bfadc1dd38e25dc05c26585cc5787550079b5554963026be0ec8f2d24c8 |
| 3C-7 frozen protocol | 352452eb1699a8203efa6a77a5e9c42101614b1a8104447844eb9dd5960204c3 |
| 3C-7 runner | b53bd46bf766d0e2203bb74cfc09bddac1680bc7b75a9c8c23972bccb556495c |
| Original target evidence | f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451 |
| Original parameter-transform source | b168dc1a0432a246dd6f2346d7b6150fa6d88b2f816ec062fb5c9674aaab2231 |
| Original Q-metric source | 316e79b5e06615c9fa46e714f35f515b204c1f175c6c56636a6c4d9235ccabc4 |

Verify these hashes and every inherited accepted source/protocol/evidence
identity before arithmetic. Hash old failed 3C-5 files without parsing their
truncated raw JSON. Verify both frozen G/A snapshot manifests; no source pin
changes. Parse only explicitly named regular JSON inputs after hash admission,
with duplicate/nonfinite rejection and a 32 MiB file cap (3C-7 is about 24 MB).
No implicit latest/best discovery, binary/checkpoint loading or deserialization.

Exactly saved final LM seeds 101–105, three arms each: P0, P1-U, P1-C. Truth is
not an experimental case. Use the archived exact unconstrained seven-vectors,
physical reconstruction records, original float32 dp/dz diagonals and complete
fitting component Jacobians. Check shapes, exact values and array hashes.
P1-U and P1-C must have identical coordinates/materials; only archived residuals/
derivatives differ. Require `D1[permutation]==D0` exactly.

The original `physical` function is extracted alone from admitted source AST
and executed without gradients on explicit CPU float32 tensors; do not import
its experiment module or shader. Use it for ten input reconstruction checks
(P0/P1 per seed) and fifteen proposed parameter arrays. These arrays are never
installed as parameters or evaluated by a renderer. Extract only the original
pure `roughness_tensor` function for float64 **analysis** of physical values.
No original scientific code is edited or dynamically patched.

## Normalization and exact proposal formulation

Order physical p as `[mx,my,theta,kR,kG,kB,metallic]`; z has the corresponding
width/albedo/metallic logits and unconstrained theta. Fit mean total RGB has
24 observations x3 channels. Original float32 targets and their conversion hash
are admitted from the original converged evidence, without generating targets.

For each archived fit arm:

```
error32 = float32(mean_RGB32 - target32)             # mean before subtraction
r32 = error32.reshape(72) / float32(sqrt(72))
r = float64(r32)
Jz = float64(Jphysical32) * float64(recorded_dp_dz32) # column scaling
J = Jz / sqrt(72)                                  # float64 chain analysis
H = J.T @ J
g = J.T @ r
d = diag(H) + 1e-8
A = H + historical_lambda * diag(d)
delta64 = solve_float64(A, -g)
candidate32 = float32(float64(z32) + delta64)
represented64 = float64(candidate32) - float64(z32)
predicted = -2 * dot(g, represented64) - dot(represented64, H@represented64)
```

Each archived latent component matrix must match `Jphysical*dp_dz` and its saved
values/hash exactly before H is built. Total matrices are 72x7. Require the
normalized residual norm squared to agree with archived MSE under the original
`1e-12+5e-7*abs(MSE)` convention. `g=J.T r` is the normal-equation vector;
the full MSE gradient is approximately `2g`, subject to the explicitly recorded
normalization/rounding difference. Save both `2g` and the archived MSE gradient,
never substitute one silently for the other.

This is **chain-derived archived latent-Jacobian analysis**: the original 3C-4
computed float32 latent directional derivatives through its normalized residual
and then promoted them to double. The new Jz was already derived in float64
analysis in 3C-7, and its analytic division here stays float64. Original residual
**values** retain their float32 division. The normal-equation formulation and
normalization are unchanged, but these Jacobians and the NumPy float64 solver
are not a bitwise historical Torch AD/solver replay. Report this limitation,
including small differences between `2g` and the archived full gradient.

Retain the original scalar-dot grouping for predicted reduction, including
signed -0.0. Also calculate the same model formula at the unrounded mathematical
delta to expose rounding effects; this is not another proposal or solve.
Zero/negative/nonfinite predicted reductions are outcomes, not retry triggers.
Never calculate actual reduction, render a candidate, run an acceptance policy,
or update damping. A positive prediction does not establish a successful step.

## Gauge proof and comparisons

Let P exchange columns 0 and 1. Under exact correspondence `r1=r0` and
`J1=J0 P.T`, we have `H1=P H0 P.T`, `g1=P g0` and
`diag(diag(H1)+eps)=P diag(diag(H0)+eps) P.T`. Equal lambda therefore gives
`A1=P A0 P.T`; uniqueness of the damped solve implies **delta1=P delta0**.
The constant angle offset has zero derivative. This proof is independently
tested on deterministic general/diagonal full-rank systems, unequal coordinate
sensitivities, near-stationary residuals and small/large damping.

Map **both** mathematical and represented P1 steps by `P.T` before comparison.
Never subtract raw candidate z values across the angle translation. For an
affine candidate-coordinate comparison remove the *admitted actual* float32
base theta offset. Record that offset separately from ideal pi/2. For physical
parameter comparison exchange widths and subtract ideal pi/2; leave the original
float32 phase discrepancy visible. Compare Q directly on each raw physical
material in the same original tangent coordinates. No normal/angle repair,
axis canonicalization or hidden alignment is allowed.

Freeze these elementwise absolute-plus-relative bounds before the run:

| Comparison | Absolute | Relative | Relative metric floor |
| --- | --- | --- | --- |
| Mathematical delta | 1e-16 | 2e-4 | 1e-16 |
| Represented delta / affine candidate coordinates | 2e-7 | 2e-5 | 1e-16 |
| Mapped physical material / Q | 2e-7 | 2e-6 | 1e-12 |
| Predicted reduction | 1e-30 | 2e-4 | 1e-30 |

Mathematical-step bounds expose relative sensitivity even at these tiny scales;
they do not imply physical relevance below float32 spacing. Represented/physical
bounds cover original float32 coordinate and angle quantization. Report every
actual norm/error/column and denominator flag, never only a binary pass.
Cosine is null when either norm is at/below the relevant floor. Zero represented
steps are explicitly stagnation at the selected precision, not convergence.

For model reduction, record mathematical and represented values/signs, signed
zero, absolute/relative differences, and whether rounding changes them.
The interpretation floor `1e-12+5e-7*abs(archived_MSE)` is the old operational
objective precision bound, **not** a new acceptance rule. Mark smaller reductions
as too small for a robust improvement claim. Nonfinite values serialize as null
with a classification/reason, never JSON NaN/Infinity.

Save r/J/H/g/d/A, input/proposed z and physical values/Q, all system singular
values/condition/numerical rank, exact raw/represented deltas and per-parameter
comparisons. SVD numerical rank uses `max(1e-300,1e-12*sigma_max)`. Condition is
null for a zero smallest singular value. Verify each solve using only the
explicit matvec residual: `norm(A delta+g)/(norm(A)_F norm(delta)+norm(g))`,
denominator floor 1e-300, mandatory bound 1e-11. No inverse, pseudoinverse,
least-squares fallback, alternative factorization or second solve.

## Work, resources, failure and execution

At most **15** `numpy.linalg.solve` calls, one per seed/arm, counted before
execution and after success. No verification solves. Compute 45 SVD summaries
(J, H, A per case), which are decompositions, not normal-equation solves.
Exactly 15 candidate arrays/physical transforms and ten input reconstructions
on full completion. No shader, target, AD or optimizer calls. Retain the original
input arrays read-only and verify their hashes after arithmetic.

One CPU process, one Torch/BLAS/OpenMP thread, GPUs hidden; original Python/
NumPy/Torch versions match 3C-7. Torch is used only for the exact original small
parameter transform and owned thread/RNG-state inspection; no shader module is
imported. External timeout and internal alarm **300 seconds**, no automatic retry.

Peak resident memory allowance **2 GiB**, checked cooperatively before/after
admission/arithmetic. Also cap process RLIMIT_DATA at 2 GiB (or stricter inherited
limit), restoring it afterward with inherited hard limit unchanged. This bounds
data allocations; RSS checks are cooperative, not a cgroup hard RSS limit.
Read-only CPU library inspection observed about 3.1 GiB *virtual* mappings but
only about 347 MiB resident. Thus an address-space cap of 2 GiB would obstruct
the required original Torch transform despite low resident use; no such cap or
host/container permission change is introduced. Actual measured RSS is reported.

Restore owned signals/threads/data limits and verify global NumPy/Torch RNG
states unchanged. Missing/contradictory provenance, changed hashes, invalid
inputs/normalization/coordinates, nonfinite system/solution, failed solve
verification, budget/resource violation, or shader/AD/optimizer execution is
fatal. Preserve partial system/case evidence; no repair/retry. Finite scientific
discrepancies continue through the fixed 15 cases without changed tolerances or
damping. Nonpositive model reductions are retained without candidate evaluation.

New focused tests only, in `tests/scientific/test_material_lm_proposal_gauge_v1.py`:
provenance/contradictions, original equation/normalization, archived chain hashes,
independent gauge proofs, original pure proposal parity on synthetic systems,
float32 rounding/signed zero, singular/ill-conditioned handling, budget guards,
strict serialization/failure persistence, exact source transform and no forbidden
execution. No old suite or scientific experiment is rerun. Use python3 throughout.

Implementation `experiments/material_lm_proposal_gauge_v1.py`; exclusive-create
`docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json`, serialize full strict JSON
before replacing a snapshot; refuse collisions. Post-run analysis is
`docs/MATERIAL_LM_PROPOSAL_GAUGE_V1_RESULTS.md`. Preserve all earlier artifacts.

From `/home/isavand/paper/Nerofactor-3c8-lm-proposal`, execute **once**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 300s \
python3 -B -m experiments.material_lm_proposal_gauge_v1 \
  --output /home/isavand/paper/Nerofactor-3c8-lm-proposal/docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json
```

Report H1 coupled mathematical/represented consistency, H2 uncoupled differences,
H3 near-stationary sensitivity/rounding/saturation, H4 physical candidate Q and
H5 unresolved historical implications, classifying each confirmed/supported but
limited/contradicted/unresolved. Saturated counterfactuals do not establish
convergence, recovery improvement, acceptance probability or optimizer-group
causation. Recommend one substantive next experiment without implementing it.
Commit/push this feature branch only; do not merge or begin another experiment.

## Frozen implementation identities

Recorded after the final new focused tests and before the single scientific
execution. Preserve these bytes afterward.

| File | SHA-256 |
| --- | --- |
| `experiments/material_lm_proposal_gauge_v1.py` | `0a4cfae0c5fc561773b691076bc8e78650e02540f148473ad575eaf73521e94d` |
| `tests/scientific/test_material_lm_proposal_gauge_v1.py` | `f6ab71b57aca8ce0eecabf44859151ed4c07c9857a601a226cef66e685f6ce31` |
