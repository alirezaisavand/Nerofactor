# Controlled LM gauge trajectories v1 — frozen 3C-9 protocol

Authorized base: `ebdfa2cdbf4bbd5d7b6ad32c1e558349c2c16153`; branch
`feat/material-lm-gauge-trajectories-v1`, separate worktree. This is ONE new
CPU optimization experiment, not a replay or continuation of terminal LM states.
Freeze this protocol, runner and new tests before execution in
`MATERIAL_LM_GAUGE_TRAJECTORIES_V1.freeze.json`. The output is created exclusively
at `docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json`. No collision, retry,
replacement arm, tuning, threshold revision or additional experiment is allowed.

## Identities and mandatory gates

Admit the complete original 3C-1–3C-8 hash chain through the existing read-only
admission helpers. This includes sources, protocols, targets, samplers, pinned
G/A snapshots and both failed 3C-5 artifacts. Additionally admit the exact 3C-8
report SHA `54920234b87edf17eeba688e0ffab83cdbd71f9246a4ca7654fb68f12bd5663e`.
Never execute an old runner, quadrature or target generator. Preserve the original
user worktree and its uncommitted scientific-audit document.

Restore exactly seeds 101–105 from the complete update-1000 records in
`MATERIAL_RECOVERY_CONTINUATION_V1.json` (SHA
`224cb9a0cb27bbaacc61b8d575efbcd2c37854b716fc51bc78b439020239f24f`).
Validate seven float32 coordinates, physical values, shapes/dtypes and hashes.
Do not construct Adam or restore its old moments for these new LM runs. Admit
historical 3C-4 initial coordinates independently. Per seed, allocate arms in
order P0, P1-U, P1-C; run seeds in order 101–105.

P1 swaps the two unconstrained width logits and performs ONE float32
`theta + pi/2`; other logits are byte-identical. There is no wrapping or inverse
sigmoid. Check actual rounded physical Q elementwise against P0 using
`2e-7 + 2e-6*abs(Q0)`; albedo and metallic must be identical. Determine the actual
decoded-frame sign using the accepted 3C-6 frame helper at all 96 fitting points.
Copy P1-C training tapes once with the accepted helper: float64 shift
`v-s/4`, modulo 1, float32 conversion, second modulo for rounded endpoint 1;
u and diffuse azimuth cloned unchanged, all storage independent. Record arrays'
hashes, bounds, rounding and endpoint reductions. Never change the sign or copy
tapes again during optimization. P0/P1-U share original unmodified training tapes.

All 15 initial forwards occur before ANY optimization. P0 predictions, prediction
hash, mask and MSE must exactly equal its archived 3C-4 initial records. Require
`abs(LC-L0) <= 1e-12 + 5e-7*abs(L0)`. Record uncoupled discrepancy without
rejecting it. Identity, restoration, physical-equivalence or coupled initial
gate failures stop the entire experiment before updates; preserve partial JSON.

## Frozen computation

Reuse unchanged `MaterialShaderV1('A2','atan2')`, original known flat geometry,
lighting/visibility, seven-coordinate transformation, original target arrays,
24 fit/24 held-out/36 relight observations, training tapes 0–3, evaluation tapes
1000–1003, 512 diffuse and 1024 specular samples. The original float32 mean of
four tape predictions precedes linear RGB MSE; no extra loss or regularization.
No reference generation, neural training, GPU, Docker or installation.

The exact original 3C-4 `run_lm` code object executes in a PRIVATE namespace.
The original module's globals are never mutated. Its original residual/JVP,
float64 solve, SVD, prediction, rounding and controller functions execute once
per original call, in their original order. Wrappers only account, record and
enforce the new branch policy before candidate rendering. Initial lambda .01,
clamped to [1e-8,1e8], damping `diag(J.T@J)+1e-8`, four trials per outer, 80
outers. Residual is original float32 flattened RGB error divided by sqrt(72);
seven unit-direction double-reverse JVPs form the original float64 72x7 matrix.
The candidate is `(z.double()+delta).float()`. Predicted reduction uses this
represented delta, preserving original scalar-dot operation grouping.
Accept only original positive prediction, decreased MSE and ratio > .1; accepted
lambda /3 if ratio > .75, *2 if ratio < .25, otherwise unchanged. Rejections
multiply lambda by 10, capped. A rejected attempt already at 1e8 saturates.
Zero/nonfinite predicted reduction retains original rejection semantics.

## Frozen branch-local chart and stop

The sign relates TWO actual decoded frames. Define an arm's chart as its own
decoder cell plus the relevant gauge counterpart's decoder cell: P0 counterpart
is the once-float32 `+pi/2` swapped representation; P1 counterpart is the
once-float32 `-pi/2` inverse representation. Freezing both cells is necessary
because a counterpart cut can change the coupling sign while the arm's own
decoder remains continuous. Apply this identical two-frame chart policy to ALL
three arms. Record both initial cells; this is additional deliberate censoring
relative to unrestricted historical LM, not a modification of its controller.

Determine cells from actual float32 normalized `(cos(2theta), sin(2theta))` and
`principal=.5*atan2(...)`, then `round((theta-principal)/pi)`. Independently
verify reconstruction within `8*eps32*max(1,abs(theta))` and agreement with
analytical `floor((theta+pi/2)/pi)` away from a 2e-6-radian neighborhood of the
cut. Actual decoder chooses the cell at float32 boundary ambiguities. Record
theta, encoding, decoded angle, reconstruction error and branch margin.

For EVERY finite represented proposal, inspect both chart cells after the
original solve but BEFORE predicted-reduction validation or candidate forward.
At the FIRST departure from either initial cell, do not render, accept, reject,
retry, increase damping or repair anything. Record original/proposed coordinates,
delta/represented delta, predicted reduction, trial, damping, margins and
transition keys. Terminate `branch_crossing_blocked`; keep previous state.
This is a valid censored outcome. Nonfinite proposals retain original rejection
semantics. P1-C accepted states must also reproduce the initial actual frame
sign (frame tolerance 2e-6) using their own inverse-gauge counterpart. A sign
or chart failure at an accepted state is an integrity failure, stopping execution.

## Budgets and counting

CPU runtime must exactly match historical Python 3.9.12 / NumPy 1.21.5 /
Torch 2.4.0+cu121, CPU device and one Torch thread. Hide GPUs; set OpenBLAS,
MKL and OpenMP threads to 1. Each arm: 120 s accumulated active wall time,
including initial forward and logging; optimization checks reserve 10 s for
endpoint gradient/evaluation. Other arms' waiting time is excluded. Whole
process including admission/persistence: 2400 s; peak RSS 4 GiB, address space
soft limit 8 GiB as a backstop, checked RSS before/after operations. Signals stop
without retry. Report wall and CPU times, restore signal/thread/memory state.

Per arm / whole-run maxima:

| Work | Per arm | All 15 |
| --- | ---: | ---: |
| Initial full-batch forwards | 1 | 15 |
| Jacobian constructions | 80 | 1200 |
| Directional JVPs / JVP primal forwards | 560 | 8400 |
| Reverse sweeps including terminal fit gradient | 1121 | 16815 |
| Linear solves / attempted proposals | 320 | 4800 |
| Candidate objective forwards | 320 | 4800 |
| Terminal fit forward / backward | 1 / 1 | 15 / 15 |
| Common-gauge evaluation forwards | 1 | 15 |
| All full-batch forwards | 883 | 13245 |
| Fit sampled points (96 per batch) | 84672 | 1270080 |
| Evaluation sampled points (336 per batch) | 336 | 5040 |
| Total sampled points | 85008 | 1275120 |
| Diffuse directions (512 per point) | 43524096 | 652861440 |
| Specular directions (1024 per point) | 87048192 | 1305722880 |

Count attempts and completions separately in inherited counters. Candidate,
initial, terminal and evaluation calls have separate phase counters; Jacobian
primals are accounted in JVP/forward counters. Blocked proposals count as solves
and branch blocks, neither rejected nor accepted. Accepted + rejected + blocked
cannot exceed solves (interrupted solve/trial may remain explicitly incomplete).
No raw-arm evaluation, extra gradient, fallback derivative or extra solve.

Time exhaustion keeps last completed state, logs incomplete work, attempts only
the reserved endpoint work and marks budget/censor status. Whole-run, memory,
identity, construction or runtime violations stop all remaining arms. Retain
finite scientific failures and continue remaining predeclared arms. No restarts.

## Endpoints, controls and comparisons

Keep final state at predeclared termination; never choose a best checkpoint.
One endpoint fitting forward plus original backward gives final gradient. One
84-observation evaluation uses ORIGINAL evaluation tapes and physical common
gauge: for P1 swap widths and subtract pi/2 in float32; no gauge choice based on
scores. Record ideal float64 mapping, actual float32 difference and Q roundoff.
Evaluate even censored endpoints if time permits, but distinguish those from
completed trajectories. Original `evaluate_start`/recovery/metric helpers remain
unchanged. Save full RGB/component predictions and errors, compensation inner
products/correlations, Q, roughness, material values and scalar identities.

Original simultaneous recovery thresholds: Q relative <= .05, identifiable
major-axis orientation <=5 degrees, maximum albedo error <=.02, metallic error
<=.03, held-out total relative L2 <=.01. No rounding into success. Record both
unmapped physical errors and rounded common-gauge errors. Keep historical 3C-4
endpoints separate.

Historical P0 control: exact initial vectors/predictions/MSE, each common outer
start, original Jacobian hash/H/gradient, trial damping, mathematical/represented
step, prediction, acceptance/status, objective, ratio and mask. Ignore clocks
and instrumentation. Compare final coordinates/predictions/MSE/termination if
not censored; otherwise shared prefix including the blocked proposal. Unexpected
finite mismatch marks control FAILED, preserves mismatches and does not trigger
tuning/retry. Such seeds cannot establish causal effects by historical parity.

At equivalent INITIAL states only compare full normalized J, H, proposals,
represented proposals, predictions and proposed Q. Inherited derivative bounds
J: 2e-6+2e-4*abs(reference), norm floor1e-8; H descriptive bound
2e-9+2e-4*abs(reference), floor1e-12. Inherited mathematical step bound
1e-16+2e-4*abs(reference), floor1e-16; represented bound
2e-7+2e-5*abs(reference). Do not turn loose represented bounds into claims of
mathematical equality. Preserve all actual differences.

Align later trajectories by explicitly numbered outer starts and separately by
accepted-update counts, recording both underlying states. They are NOT the same
physical states after divergence. Report MSE versus outer, active wall and
forward work; Q error versus accepted updates, albedo/metallic changes, damping,
masks, proposal lengths and branch events. Initial state is index0. First
practical divergence uses the same physical criteria below, not a post hoc bound.

Final cross-arm practical consistency (distinct from recovery): relative Q
<=.005, max albedo difference <=.002, metallic difference <=.002. Orientation,
objective, histories and termination remain separate. Branch-censored or
incomplete pairs get censored/incomplete verdict, never forced pass/fail.

H1 moving proposals, H2 trajectories, H3 recovery, H4 numerical/branch limits,
H5 historical solution-group explanation must receive separate qualified
verdicts. H5 causality is not established by a changed endpoint. No further
experiment without separate authorization.

## Execution and reporting

Only NEW focused tests with disposable/synthetic numerical fixtures and
read-only metadata admission. No old suites or old experiment reruns.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest tests.scientific.test_material_lm_gauge_trajectories_v1 -v

# Exactly one invocation after creating the reviewed pre-execution hash freeze:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m experiments.material_lm_gauge_trajectories_v1 \
  --output "$PWD/docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json"
```

Strict JSON encoding is validated before replacing each owned partial snapshot.
Save all trials, negative/censored outcomes and resource counts. Afterwards,
offline plots read only this new JSON, with no rendering/AD/solve. Publish results,
four SVG figures (MSE versus outer/wall/forwards, Q versus accepted updates),
frozen hashes and exact test results. Verify all old tracked files unchanged
except authorized AGENTS milestone update. Commit/push only the feature branch;
no merge or next experiment.
