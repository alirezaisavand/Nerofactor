# Multi-material recovery v1 — preregistered benchmark (3D-1)

One CPU execution from base `34548db58e7e0d3adb4cc2be1d2a4f1ed45443c7`,
branch `feat/multi-material-recovery-v1`, in a separate worktree. No pilot,
retry, alternate material/tape/start, grid refinement or post-result tuning.
Freeze this protocol, implementation, tests, panel, initial vectors, original
and evaluation tape hashes, numerical policies and budgets before computing any
new scientific target. New targets are generated only in the authorized run.

## Panel, scope and unchanged scene

Parameter order is `[mx, my, theta, kR, kG, kB, metallic]`, float64 truths:

| Material | mx | my | theta | kR | kG | kB | metallic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.28 | 0.49 | -0.62 | 0.36 | 0.58 | 0.24 | 0.10 |
| M2 | 0.17 | 0.62 | 0.91 | 0.56 | 0.31 | 0.18 | 0.65 |
| M3 | 0.37 | 0.47 | 0.19 | 0.24 | 0.43 | 0.70 | 0.35 |

M1 is moderately anisotropic/mostly dielectric; M2 strongly anisotropic/high
metallicity; M3 less anisotropic/intermediate metallicity. These descriptions
do not predict measured difficulty. Do not replace difficult or failed cases.
Three materials and three starts each do not support population-level or neural
inverse-rendering generalization. Known flat geometry and lighting remain fixed.

Unchanged `known_material_reference_v1.design()`: origin, normal +Z, base tangent
+Y, bitangent −X, 12 views, seven lights, original view/light order, 24 fitting,
24 held-out and 36 relighting observations. No camera, lighting, visibility,
background, observation redesign, learned geometry or image-based preprocessing.
The original `[.20,.55,.31,.30,.45,.60,.25]` case and 3C-10 are archival context;
do not rerun them or combine their outcomes with the new success denominator.

## Historical admission and source preservation

Read-only admission reuses the complete accepted chain through 3C-10, including
original 3C-1, continuous continuation, optimizer comparison, sample coupling,
gradient/proposal audits, both trajectory versions, numerical analysis, original
failed attempts, source/test/protocol freezes, reports and figure hashes.
Verify 83 inherited source/evidence identities before and after execution.
Require completed 3C-10 evidence, matching runtime, fixed-tape/RNG/state integrity
and its offline analysis with zero extra render/AD/solve calls. No old runner or
scientific experiment executes during admission.

All historical files, both pinned G/A snapshots, A2 shader, sampler, BRDF/decoder,
previous targets, protocols, evidence and user changes remain untouched. Only new
versioned files and the authorized milestone paragraph in AGENTS.md change.
The separate preservation inventory hashes every original tracked file and the
existing user-edited scientific audit document; exclude only AGENTS.md from the
unchanged-file assertion. Original datasets/final_results remain read-only and
are not needed by this analytical CPU experiment.

## Phase A: independent reference and identifiability

Use `known_material_reference_v1.render(explicit_parameters, ...)` unmodified,
including original NumPy Cartesian quadrature, cached-view kernels, affine
Fresnel treatment and diffuse/specular/total definitions. The new wrapper accepts
explicit float64 parameters; never change or monkeypatch the old global TRUTH.
Exactly the original grids `(256,512)` and `(512,1024)`, made once and reused.
Record grid direction/weight/lighting hashes, all 84 coarse/fine RGB component
arrays, target hashes, absolute deltas, limits, per-component maximum bound
ratios, truth identities and counted direction evaluations for each material.

Unchanged elementwise convergence is
`abs(fine-coarse) <= 2e-6 + 5e-5*abs(fine)` for all components. Total convergence
also requires both component gates. Generate ALL THREE targets, save every
result, then stop globally before identifiability/optimization if any target
fails. No refinement, selected observation, tolerance change or replacement.

For each admitted truth, the separately versioned explicit-parameter wrapper
reproduces the original finite-difference design: seven physical parameters;
scales `[.96,.96,pi,1,1,1,1]`; central normalized steps `1e-4` and `5e-5`; signs
−1/+1; two original grids and 24 original fitting observations. Exactly 28
perturbed parameter evaluations per material. Retain every perturbation's truth,
coarse/fine component arrays, hashes and convergence gate. Complete all 28 to
retain failures; no additional perturbations or alternative differences.

Normalized Jacobians have shape `[2 steps,2 grids,72 total-RGB coordinates,7]`;
each column is `(plus-minus)/(2*step)`, preserving the original scale convention.
Use original `jacobian_gate`: fine-grid smallest/largest singular ratio >=1e-4
for BOTH steps; maximum cross-step/cross-grid spectral discrepancy strictly
smaller than the minimum fine-grid smallest singular value. Require all 28
perturbation convergence gates too. Record complete matrices, singular values,
rank, right singular vectors, condition numbers for every step/grid, spectral
drifts and smallest-value stability margin. Undefined/infinite condition numbers
are explicit JSON null plus a nonfinite flag, never a successful gate.

Compute all three identifiability analyses; stop before any Adam/shader
construction if any fails. Local full rank does not establish global uniqueness.
Save all target/reference identities and mandatory outcomes before optimization.

Reference accounting is unchanged cached views, not observations/channels:
`(256*512+512*1024)*(12+28*6) = 117964800` directions per full material;
three materials total **353894400**, with **23592960** target and **330301440**
perturbation directions. Each render prechecks remaining cap and verifies its
exact counter increment. Grid directions total 655360 and cached scalar light
values 4587520, recorded separately from kernel direction evaluations. No extra
reference targets, checks with a third grid or post-result quadrature.

## Frozen tapes and fresh starts

Training seeds ONLY 3000–3003, original `make_tapes`, private NumPy RandomState,
24 observations per seed, seed-major order, original draws/float32 conversion,
512 diffuse and 1024 specular samples. The same draws across materials are
deliberate common-random-number control, not independent realizations per material.
Original external evaluation seeds 1000–1003 use 84 observations per seed.
Training seeds do not overlap original 0–3, 3C-10 2000–2003 or evaluation seeds.
Freeze each per-seed combined/individual-array hash and full owned u/v/azimuth
array shape/dtype/hash before target computation. Runtime must match the entire
manifest and original evaluation identities. Repeated deterministic identity
generation in fixtures is not another scientific realization. No global RNG use.

Fresh seeds ONLY 101,103,105 from the unchanged original `initializations()`
constructor/private draws (select these records; do not optimize 102/104).
Freeze their exact float32 latent and reconstructed physical records. The same
three initial vectors are reused across materials as intentional control; they
do not depend on the new truths. No restoration from old Adam or LM endpoints.

## Phase B: nine continuous Adam histories

For each material/start run exactly 1000 successful original Adam steps in one
continuous history, lr=.01, betas=(.9,.999), eps=1e-8, weight_decay=0; original
runtime defaults, no scheduler/clipping/regularizer. Same A2/atan2 shader, physical
transform, sample counts/distributions and float32 four-tape mean-before-MSE.
No truth-guided initialization, early stopping, extra start or held-out selection.

Record typed/hashed latent, reconstructed physical, Adam step/moments and effective
options at updates 250 and 1000. Snapshot 250's exact post-update objective reuses
the NORMAL update251 forward (before its gradient/update). Snapshot 1000 uses
one counted terminal forward with no backward/update. No update251 restart,
repeat of the first 250 steps, or endpoint optimization. All snapshot objectives
label their update count/source. History stores first/every-tenth update loss at
the PRE-update coordinates, gradient norm and post-update coordinates, complete
work counts/time; snapshots retain both exact optimizer states. Preserve latest
partial state on failure. Any failure or active cap prevents admission of all
nine endpoints and stops the benchmark; no replacement.

Only exact new material-associated completed update1000 states enter LM. Target
identity verifies material name, float64 truth, full fine target hash and float32
fit-target hash each time the target is supplied. No inference from metadata names.

## Phase C: paired LM and nine independent initial gate sets

For each of nine new Adam endpoints: P0 is exact latent state; U/C share exact
axis-swapped state `[z1,z0,float32(z2+pi/2),z3:]`. U uses original new training
tapes; C preserves u/diffuse arrays and uses accepted signed `(v-s/4) mod 1`,
float64 shift, float32 conversion and periodic endpoint reduction, disjoint
owned storage. Determine sign from actual decoded uniform frames, freeze for
the arm and never dynamically recouple. No theta wrap/canonicalization.

Construct all 27 arms and perform all 27 original initial forwards before
checking all nine A–E gate sets; capture samples/components from those returns,
not additional renders. A requires exact NEW Adam state/count, target/tape/source/
runtime/observation identity, same U/C coordinates and exact original float32
mean/RGB/MSE replay for all three arms. No historical original-material RGB parity.
The reused gate helper's self-comparison A is discarded; the explicit new A
provides the actual saved-state, target and replay checks. B–E remain unchanged:

- B: Q `2e-7+2e-6*abs(Q0)`, bitwise albedo/metallic, uniform actual frames and
  actual signed quarter-turn within 2e-6, identical U/C states.
- C: h `2e-6+2e-5*abs(h0)`, wi `4e-6+2e-5*abs(wi0)`, angular error<=2e-5 radians,
  reflection-mask XOR0, bitwise diffuse directions, correct active tape hashes.
  PDF/grazing diagnostics are reported separately without a PDF equality gate.
- D: separately every tape and mean diffuse/specular/total RGB; diffuse byte
  identity; specular AND total `2e-6+2e-5*abs(P0)`. Cancellation cannot hide failures.
- E: exact original Torch float32 loss replay and unchanged source-verified
  gamma75 envelope under finite/normal/no-overflow/round-to-nearest assumptions.
  Conditional arithmetic consistency cannot override B–D. Original scalar
  `1e-12+5e-7*abs(L0)` comparison is retained as warning only. No retuning.

Save ALL nine gate outcomes. A mandatory failure stops before ALL LM updates,
retaining every initial forward and completed Adam history. No removed material,
replacement start or gate retry.

Use exact accepted original LM code object/controller/AD/solve through the frozen
guarded runner: fresh lambda=.01, bounds1e-8..1e8, 80 outers, four trials;
`r=vec(mean_RGB-target)/sqrt(72)`, J=dr/dz, H=J.T@J, D=diag(H.diag()+1e-8),
float64 solve `(H+lambda*D)*delta=-J.T@r`, original float32 candidate conversion,
represented-step prediction, acceptance/ratio/rejection/damping/saturation.
No prior LM state or Adam moments restored into LM.

Inherited fixed own/counterpart chart and signed-frame policy blocks first
candidate crossing BEFORE rendering/acceptance/rejection/retry, preserves current
accepted state/blocked proposal, labels censored. No chart repair, alternate
encoding, step retry, recoupling or restart. Retain complete trials, gradients,
Jacobians/singular values, mathematical/represented proposals, masks, damping,
accepted/rejected states and counts. Known nonfinite-prediction/JVP failures use
the unchanged accepted safe-isolation policy only with finite latent state and
unchanged tapes. Other integrity/resource failures stop globally.

## Explicit-truth evaluation and comparisons

The original endpoint runner's private namespace binds only an explicit-truth
evaluator, preserving its original optimization/branch/terminal-gradient code
object. No historical module/global is modified. Every valid terminal receives
one original fit forward/backward and one 84-observation external evaluation.
Map P1 physical parameters once-float32 into that start's original P0 gauge;
record float64 ideal/float32 roundoff and Q mapping error. Original split masks,
radiance metrics and component-compensation definitions are unchanged.

New recovery adapter uses ORIGINAL `gauge_metrics`/roughness Q with EXPLICIT
material truth. Never use historical truth-dependent `recovery_errors` unchanged.
Report raw and common-gauge recovery, sorted axis errors, Q, orientation,
albedo/metallic, predictions and every fit/held/relight component/subset metric.
Near-isotropic/nonidentifiable orientation is null with reason, not artificial
zero. Simultaneous success retains Qrel<=.05, identifiable orientation<=5 degrees,
max RGB albedo error<=.02, metallic<=.03, held-out total relativeL2<=.01.

Offline trajectory Q scoring binds explicit truth locally; comparison helper's
sole fixed seed enumeration becomes 101/103/105 in a private namespace. All other
comparison formulas/thresholds remain unchanged. Pair orientation may be null
near isotropy; practical similarity still depends only on Qrel<=.005,
max albedo<=.002, metallic<=.002. No historical global or frozen source mutation.
Initial J/H/gradients/math/represented proposals are matched-physical-state
comparisons; physical sensitivities are chain-derived, no extra AD. Later
same-index distinct states are trajectory comparisons, not equal-state proposal
proofs. Record first divergence, endpoints, accepted-index and outer alignment,
controller histories and common-gauge rendering differences.

No best intermediate checkpoint. Retain poor, censored, incomplete and failed
outcomes with explicit denominators. Compare conditioning/difficulty descriptively
only: three selected profiles cannot isolate causal effects of anisotropy or
metallicity. Good total RGB is not material recovery; component anticorrelation
and errors are reported separately. Archive original-material/3C-10 context
without combining denominators or claiming a common experimental pipeline.

## Budget and partial evidence

CPU only; GPUs hidden; one Torch/BLAS/OpenMP thread; no dependencies, Docker,
GPU/neural/geometry training or environment changes. Original numerical runtime
must match accepted 3C-10. Maximum 7200 process-wall seconds (external timeout
includes imports), 4 GiB RSS, 8 GiB address backstop, 180 accumulated active
seconds per Adam history, 120 per LM arm with original 10-second endpoint reserve.
No cap extensions. Reference checks use the whole-process deadline and RSS cap.

| Maximum work | Each Adam / 9 total | Each LM / 27 total |
| --- | --- | --- |
| Forward attempts/completions |1001 / 9009|883 / 23841|
| Successful optimizer returns |1000 / 9000|80 accepted LM / 2160|
| JVPs/primal forwards |0|560 / 15120|
| Jacobian constructions |0|80 / 2160|
| Reverse sweeps |1000 / 9000|1121 / 30267|
| Small LM solves |0|320 / 8640|
| Initial LM forwards |0|1 / 27|
| Candidate forwards |0|320 / 8640|
| Terminal fit forwards/gradients |1/0 / 9/0|1/1 / 27/27|
| External evaluation forwards |0|1 / 27|
| Fit sampled points |96096 / 864864|84672 / 2286144|
| Evaluation sampled points |0|336 / 9072|
| Diffuse directions |49201152 / 442810368|43524096 / 1175150592|
| Specular directions |98402304 / 885620736|87048192 / 2350301184|

Combined caps: 32850 forward attempts/completions, 15120 JVPs, 39267 reverse
sweeps, 8640 LM solves, 1617960960 diffuse and 3235921920 specular directions.
Reference directions are separate. Count attempts/completions separately;
blocked trials do not count as accepted/rejected, zero represented proposals
do not cause a candidate render. Per-run and aggregate counts are enforced.
Initial capture adds no forward, physical AD, reference work or extra solve.

Output exclusive-create `docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json`;
strict JSON validation precedes replacing partial snapshots. Save all available
reference/optimizer/gate evidence before stopping, preserving failure phase and
latest Adam/LM state. Undefined condition/orientation is explicit null, not NaN.
Restore global RNG/thread/signal/memory state and recheck inherited hashes.
No automatic retry even after an implementation/runtime failure.

## Tests, execution, offline analysis and hypotheses

Only new focused synthetic/read-only tests, no prior suite or full scientific
shader/quadrature/optimizer experiment as a test. Synthetic Adam verifies one
continuous optimizer, exact 1000 returns and both snapshot objective semantics.
Test explicit truth/reference/Jacobians, rejected historical-TRUTH reuse,
convergence/rank/undefined values, original tapes/fresh starts/RNG, target
association, coupling/gates, original LM/chart policy, common evaluation,
components, resource caps, strict partial persistence and prior preservation.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m unittest tests.scientific.test_multi_material_recovery_v1 -v

# Exactly ONE invocation after full protocol/code/test/panel/tape freeze:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=10s 7200s \
python3 -B -m experiments.multi_material_recovery_v1 \
 --output "$PWD/docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json"
```

Afterwards, analyze saved JSON only: per-material recovery/conditioning, every
Adam and all 27 planned LM outcomes, actual/censored/failed denominators,
initial proposal and physical trajectory consistency, component compensation,
held-out/relight errors and work-normalized curves. No extra rendering, AD,
quadrature or optimization for figures or analysis. If an earlier gate stops,
mark unexecuted phases explicitly, never fabricate the planned optimizer rows.

H1 reference convergence/local rank for all three profiles; H2 physical recovery
beyond the original material; H3 coupled proposal/trajectory consistency;
H4 measured material-dependent difficulty; H5 low-RGB/physical-error ambiguity;
H6 precision/mask/damping/chart limitations. Classify each confirmed, supported
but limited, contradicted or unresolved, retaining negative and mixed evidence.
No broad statistical or neural-geometry claim from this small panel.
Recommend a next substantive neural inverse-rendering milestone for review;
do not execute it. Commit/push only this feature branch; no merge or follow-up.
