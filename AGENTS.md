# Research and development rules

## Purpose and methodology

This repository develops NeRO-based geometry reconstruction and anisotropic
material estimation from calibrated multiview images. Geometry training learns
an implicit surface; mesh extraction supplies geometry to material training.
Rendering, novel-view synthesis (NVS), and geometry evaluation assess different
parts of the method and must retain their own protocols and provenance.

The research workflow is: state a hypothesis, identify a reference implementation
and its exact configuration, define controlled changes and evaluation splits,
record all runs, and distinguish historical reports from independent evidence.
An infrastructure refactor must preserve the computation being studied.

## Workspace and repository conventions

The host is `marineris`. The maintained development branch is
`paper/infrastructure`, initialized at the `anisotropic` baseline.

**Authoritative scientific selection (1D.5-RESET):** Stage I geometry uses
`aniso_geometry_safe` at `79252178e65914ad1acbc14790b1514f276bfea0`; Stage II
material uses `anisotropic` at `e1e859779301faced46d6476691e20b546c27ae0`.
Infrastructure ancestry is not scientific-stage authority. Use the exact separate
snapshots and fresh-process namespace selection, never an A geometry/G material
substitution. This researcher decision does not identify old producing revisions.

| Location | Role |
| --- | --- |
| `/home/isavand/paper/Nerofactor-paper` | Versioned code, configurations, documentation, and small reviewed fixtures |
| `/home/isavand/paper/data` | Original input datasets; preserve without modification |
| `/home/isavand/paper/final_results` | Historical thesis results, meshes, checkpoints, logs, and comparison artifacts; preserve without modification |
| `/home/isavand/paper/paper_runs` | Future experiment outputs; not yet created in milestone 1D.1 |

Keep code separate from datasets and generated storage. Do not introduce a
writable repository `data` symlink that combines original inputs with outputs.
Proposed container paths are `/inputs/datasets` and `/inputs/thesis` read-only,
and `/outputs` for the selected experiment's writable storage. These mounts are
planned, not implemented by this document.

Use existing Python/PyTorch/NumPy style, `snake_case` names, and focused diffs.
Avoid unrelated formatting, dependency upgrades, or configuration rewrites.
Keep host-specific settings and secrets outside committed files. Ignore rules
reduce accidental Git additions; they are not filesystem protection.

## Historical branches

The inspected baseline refs and exact hashes are recorded in
[BASELINE_INVENTORY.md](docs/BASELINE_INVENTORY.md).

| Branch | Historical role and limitation |
| --- | --- |
| `anisotropic` | Starting implementation for infrastructure development; contains ablation-labeled settings whose effective scope needs verification and is not automatically a canonical full-method preset |
| `aniso_geometry_safe` | Alternative material parameterization, geometry/tangent experiments, optimization settings, and dataset/evaluation interfaces; the name does not establish mathematical correctness |
| `ablation_anisotropic` | Single-angle ablation plus other configuration and numerical differences; not automatically a controlled one-variable comparison |

Only `anisotropic` has a local historical branch at inspection; the other two
are available as `origin/...` remote-tracking refs. Inspect historical files with
`git show` and compare them without changing branches or merging. Do not treat
any pair of these implementations, checkpoints, or reported metrics as
equivalent. See [BRANCH_COMPARISON.md](docs/BRANCH_COMPARISON.md).

## Scientific integrity

- Never invent measurements, claim an unexecuted validation passed, or label an
  old log as an independently reproduced result.
- Identify historical reported results, newly measured results, and hypotheses
  separately. Preserve original precision, source paths, and uncertainty.
- Record failed, interrupted, and excluded runs and the reasons for exclusions.
  Do not select seeds, views, checkpoints, masks, or metrics after inspecting
  test outcomes to improve a reported comparison.
- Separate validation-based checkpoint selection from final reporting. Record
  exact image IDs, masking, color space, resolution, coordinate frames, scale,
  point sampling, metric definitions, and aggregation rules.
- Do not infer training steps from mesh filenames or producing commits from
  directory names. Mark missing provenance as unknown.
- Treat cache generation, preprocessing, random-number consumption, numerical
  tolerances, and dependency versions as potentially scientifically relevant.

## Prohibited destructive operations

- Do not move, delete, overwrite, regenerate in place, or change permissions on
  original datasets or anything in `final_results`.
- Do not run cleanup commands against research storage, force-add large or
  ignored artifacts, rewrite historical Git history, or merge historical
  branches as part of infrastructure cleanup.
- Do not overwrite another experiment, implicitly resume by scene name, or write
  generated caches into an input dataset.
- Do not start training, expensive evaluation, or Docker containers unless the
  current task explicitly authorizes that execution.
- Do not change Docker socket permissions, group membership, system permissions,
  drivers, or security settings, or bypass restricted access. Report limitations.
- Do not expose credentials in code, configuration, logs, documentation, or tool
  output. Inspect only the task-relevant environment and files.

## Changes to numerical behavior

Infrastructure work must not silently change algorithms, optimizer defaults,
schedules, losses, coefficients, sampling, formulas, tensor precision, epsilon
values, clipping, or seeds. Preserve the selected legacy behavior before
introducing alternatives. A suspicious implementation is evidence to document,
not permission to fix it during cleanup.

For a proposed numerical change, identify the exact legacy commit and execution
path, explain the mathematical or experimental rationale, isolate the change
from storage refactoring, obtain authorization within the task scope, and supply
focused regression evidence with stated tolerances. Give corrected behavior a
distinct configuration/protocol version. Do not retrofit corrected measurements
into historical result tables or silently convert old checkpoints.

## Reproducibility requirements for future experiments

Each experiment must have a unique ID and separate persistent output directory.
Resuming must explicitly name the experiment and checkpoint and verify compatible
configuration and inputs. A material run must identify its exact geometry mesh,
source run/checkpoint when known, coordinate convention, and content hash.

Record Git commit and any diff, original and resolved configuration, command,
dataset identity and split IDs, preprocessing/cache parameters, seed policy,
optimizer parameters, scheduler semantics, all loss weights, sampling settings,
environment/dependency versions, image digest when applicable, GPU/driver details,
start/end times, exit status, and evaluation protocol. Future checkpoints should
include training progress, optimizer state, and relevant RNG/sampler state.
Distinguish seeded execution from demonstrated deterministic execution.

Milestone 1D.1 uses lightweight names/counts/sizes and small-file inspection.
Do not hash entire large datasets here. Later provenance work must hash selected
meshes/checkpoints and define a dataset content-manifest policy; record the scope
and any missing hashes rather than implying full integrity verification.

## Review, testing, and milestone boundaries

For every change, report its purpose, changed files, evidence, limitations, and
whether numerical behavior could change. Verify branch and working-tree state
before editing, preserve user changes, and keep reviews small enough to assess.

For documentation-only work, validate source/ref/path claims, Markdown links,
whitespace, ignore behavior, and the exact changed-file scope. Do not run training
to validate documentation. For later implementation, add focused tests for
configuration semantics, path containment, write isolation, split IDs,
optimizer/scheduler parity, and explicit resume. Report CPU checks separately
from GPU tests; blocked GPU checks are not passes. Do not import training entry
points merely to inspect metadata, because existing loaders can write inputs.

Current authorized task: **Milestone 3C-9b, gauge-roundoff diagnostic**,
from `f9043aaecb709cebde597eab970458c72ad0f2c5` on
`feat/material-gauge-roundoff-v1` in a separate worktree. Freeze the
[roundoff protocol](docs/protocols/MATERIAL_GAUGE_ROUNDOFF_V1.md), runner and new
tests before ONE CPU execution: offline saved-array analysis first, four exact
baseline forwards (101/102 x P0/P1-C), then two P1-C-E forwards replacing only
the orientation encoding with -E0. No other variants, new states/targets/seeds,
optimizer, updates, derivatives/backward, reference quadrature, shader/sampling
changes, GPU/neural training, Docker or installation. Preserve the failed 3C-9
gate and ALL older evidence/source/protocols. Reuse original ObservationBatch,
A2/atan2 and coupled tapes; read-only profiling may copy actual return locals,
never alter numerical operations. Require exact baseline RGB/MSE/mask/tapes;
stop on mandatory identity/equivalence/resource failure with no retry. At most
six calls/576 points,294912 diffuse/589824 specular directions,600 seconds,
4 GiB RSS/8 GiB address cap, one CPU thread, GPUs hidden. Only new focused
synthetic/read-only tests; python3 commands. Separate measured stage differences,
roundoff amplification and one-factor causal evidence. A future criterion or
new trajectory run is a recommendation only; neither is implemented/authorized.
Commit/push this branch after analysis; no merge or further experiment.

The ONE 3C-9b execution is complete: exactly6 forwards/576 points in3.94s,
zero derivatives/updates. All four baseline RGB/MSE/mask identities match3C-9;
seed102's original failed gate is reproduced and remains failed. Offline final
mean64 accumulation alone would pass that stored-prediction comparison. Encoding
-E0 has no downstream effect for101; for102 it changes the objective gap from
6 to5 float32 loss ULPs, passing that DIAGNOSTIC gate, without removing remaining
sample/RGB differences or uniformly improving RGB error. Preserve frozen
code/tests/protocol, [evidence](docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json),
offline [summary](docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1_OFFLINE_SUMMARY.json)
and [results](docs/MATERIAL_GAUGE_ROUNDOFF_V1_RESULTS.md). The25 new focused tests
passed. No criterion, shader or failed evidence was changed. No extra rendering,
retry, optimizer work or new trajectory attempt without separate approval.

Earlier authorized task: **Milestone 3C-9, controlled LM gauge trajectories**,
from `ebdfa2cdbf4bbd5d7b6ad32c1e558349c2c16153` on
`feat/material-lm-gauge-trajectories-v1` in a separate worktree. Freeze the
[trajectory protocol](docs/protocols/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.md), runner
and new focused tests before ONE CPU optimization experiment. Exactly five
admitted update1000 states x P0/P1-U/P1-C, fresh lambda=.01 for every arm; reuse
original 3C-4 LM bytecode/AD/solve/controller, A2/atan2, transforms, targets,
observations, samples and tapes. Construct signed coupled tapes once; no dynamic
sign/tape reselection. Admit all 15 initial identities/objectives before updates.
Freeze both actual decoder cells defining each local gauge chart; first proposed
departure stops before rendering/rejection/retry, recorded as censored. Check
accepted coupled frames/sign. Preserve complete histories and compare historical
P0 shared prefixes exactly; retain failed controls without tuning. Common-gauge
terminal evaluation only, original recovery gates and separate cross-arm criteria.
At most15 arms/80 outers/four trials,120 active seconds per arm with10 reserved,
2400 whole seconds,4 GiB RSS/8 GiB address cap, one thread, GPUs hidden. No retry,
budget extension, new optimizer/shader/reference/initialization, prior evidence
or source changes, old suite/experiment reruns, GPU/neural training, Docker,
installs or original artifact edits. Only new synthetic/read-only tests. Use
python3. Commit/push this feature branch after analysis; no merge or next
experiment without separate authorization.

The ONE 3C-9 attempt stopped at the mandatory initial objective gate for seed102:
coupled absolute MSE difference5.4569682106375694e-12 exceeds frozen bound
5.447887931746663e-12. Seed101 passed; six initial forwards total, ZERO JVPs,
solves or updates. All15 states were restored; seeds103–105 were not rendered.
Preserve frozen code/tests/protocol, the failed raw
[report](docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1.json), offline-only
[analysis](docs/evidence/MATERIAL_LM_GAUGE_TRAJECTORIES_V1_FAILURE_ANALYSIS.json)
and [results](docs/MATERIAL_LM_GAUGE_TRAJECTORIES_V1_RESULTS.md). The24 new tests
passed. No trajectory/recovery verdict, retry, threshold change or further
rendering/optimization is authorized. Publish this failed attempt, then stop.

Earlier authorized task: **Milestone 3C-8, offline LM proposal gauge audit**,
from `31dd7979256795abd8e0043099c649518aabad35` on
`feat/material-lm-proposal-gauge-audit-v1` in a separate worktree. Historical
damping/state provenance was inspected before implementation: all five endpoints
terminated damping_saturated at lambda=1e8, with exact 3C-7 state matches. Replay
logged controller decisions only; never invent/reset/sweep damping. Freeze the
[proposal protocol](docs/protocols/MATERIAL_LM_PROPOSAL_GAUGE_V1.md), code and tests
before one offline CPU execution, at most 15 solves/300 seconds/2 GiB peak RSS
with owned data-allocation cap. Exactly LM101–105 x P0/P1-U/P1-C, no truth case.
Use hash-admitted chain-derived archived latent Jacobians; no bitwise historical
latent-AD replay claim. Keep mathematical versus represented float32 steps and
normal-equation g versus full MSE gradient separate. All proposals are
counterfactual because original runs already saturated. No renderer/target/AD,
optimizer loop, applied update, acceptance test, new state/seed beyond proposal
arrays, shader/source change, GPU/neural training, Docker or installation.
Only new synthetic/read-only tests and static checks; preserve every old source,
protocol/evidence file and original user changes. Stop on integrity/construction/
resource failures; retain finite scientific failures without tuning/retry.
Commit/push this feature branch, then stop without merging or implementing the
next experiment.
The one offline execution is complete: 15 solves in 1.52 s, all inputs admitted
at original terminal lambda=1e8. All five coupled and uncoupled mathematical
step comparisons fail the frozen bounds; every represented float32 step is
zero. LM102's coupled mathematical discrepancy is 77.17% relatively but only
6.11e-15 in L2. Zero represented steps/Q agreement must not be labeled useful
proposal consistency or convergence. Preserve frozen code/tests/protocol and
the [evidence](docs/evidence/MATERIAL_LM_PROPOSAL_GAUGE_V1.json) and
[results](docs/MATERIAL_LM_PROPOSAL_GAUGE_V1_RESULTS.md); the 19 new tests passed.
No extra solves, damping changes, renderer/AD/optimizer work or proposed paired
trajectory experiment without separate approval.

Earlier authorized task: **Milestone 3C-7, gradient/Jacobian gauge-equivariance**,
from `f5a3d714fd57bfef02baf6ca47b15ab69927e7ea` on
`feat/anisotropy-gradient-equivariance-v1` in a separate worktree. Implement,
test and freeze the [derivative protocol](docs/protocols/MATERIAL_GRADIENT_EQUIVARIANCE_V1.md)
before one approved CPU-only execution. Exactly the six accepted states,
fixed original/coupled tapes and unchanged A2/atan2; AD through the original
physical shader, with exact forward identity gates. Record physical versus
original optimizer coordinates separately; never substitute truth's inexact
inverse roundtrip. At most 330 forwards/54,720 points, 210 directional ADs,
420 double-reverse plus 18 direct-gradient sweeps, and at most 12 tiny coordinate
transform VJPs without rendering. Fixed FD states truth/LM101/LM104, mx/my/theta,
steps 1e-4 and 5e-5, no adaptive signs or changes to masks/decoder/sampling.
One process/thread, hidden GPUs, 1,800 seconds, 4 GiB RSS/8 GiB address cap;
no retry/extension. Fail on integrity/construction errors but retain finite
scientific comparison failures and continue the frozen plan. Only new focused
tests/static checks; no old suite/experiment reruns. Preserve all old sources,
protocols and evidence, including failed 3C-5 and 43 grazing PDF exceptions.
No optimizer, parameter update, estimator correction, new target/reference/seed,
GPU/neural training, Docker or installation. Commit/push this feature branch,
then stop; do not merge or implement the next experiment without approval.
The single run is complete: 330 forwards/210 JVPs in 196.67 s, all mandatory
gates passed. Coupled J/g meet frozen absolute-plus-relative bounds; uncoupled
specular/total derivatives fail. Preserve the qualification that near-zero
LM102 gradients differ by 46.6% relatively despite absolute-bound acceptance.
Truth inverse-roundtrip coordinates are inexact and excluded from latent
derivative comparisons. Twenty-four of 36 FD secants change masks; none crosses
a decoder branch. The 25 new tests passed. Preserve the frozen source/tests/
protocol, [evidence](docs/evidence/MATERIAL_GRADIENT_EQUIVARIANCE_V1.json) and
[results](docs/MATERIAL_GRADIENT_EQUIVARIANCE_V1_RESULTS.md). No retry, proposal
calculation, optimizer execution or further scientific experiment is authorized.

Earlier authorized task: **Milestone 3C-6, common-world-sample coupling**,
from `1d16e9e147f6cdf270fbc0ed19e7add36ad17281` on
`feat/anisotropy-sample-coupling-v1` in a separate worktree. The researcher
explicitly authorizes one bounded CPU forward-only diagnostic under the frozen
[coupling protocol](docs/protocols/MATERIAL_SAMPLE_COUPLING_V1.md). Derive the
signed quarter-turn from actual A2/atan2 frames; copy only diagnostic tapes,
preserving radial/diffuse draws and original storage. Admit all accepted source,
protocol/evidence/tape identities; reuse archived P1-U and recapture P0 exactly
before each P1-C. At most 24 calls/5,184 points, one thread, hidden GPUs,
1,800 seconds, 4 GiB RSS/8 GiB address cap, no retry or threshold adjustment.
Only new focused fixtures/static checks and one scientific execution; preserve
the failed 3C-5 files and every old artifact. No optimizer, parameter update,
backward, new target/integral/seed/count, accepted shader/sampler/source changes,
old experiment/suite rerun, GPU/neural training, Docker or installation. Record
all states, actual PDFs/clamps, masks, per-tape RGB and objectives; separate
sample correspondence from causal claims about LM groups. Commit/push this
feature branch, then stop. No further experiment or merge without approval.
The one approved execution is complete: 24 forwards in 8.12 s; all recaptured
P0 identities match, coupled h/wi/masks and six objectives pass their frozen
bounds. The source-PDF bound fails for 43 grazing pairs; retain that failed
subcheck, with no tolerance adjustment. Preserve the new
[evidence](docs/evidence/MATERIAL_SAMPLE_COUPLING_V1.json) and
[results](docs/MATERIAL_SAMPLE_COUPLING_V1_RESULTS.md), the frozen protocol/code
and all earlier files. The 19 new focused tests passed. No further scientific
execution or proposed derivative diagnostic without separate approval.

Earlier authorized task: **approved 3C-5 reporting repair and one fresh attempt2**,
from `13fe645af355f66400d435f9e25f5fcc63870912` on the existing
`feat/anisotropy-symmetry-audit-v1` branch. The researcher approved the proposed
native-Boolean reporting correction, strict-JSON regression coverage and ONE
new versioned attempt. Preserve the raw failed output, valid failure manifest,
original frozen protocol and failed-attempt report. Keep the original scientific
specification, shader, states/transforms, samples/tapes/targets, bounds and
CPU/memory budgets unchanged. Freeze
[attempt2](docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.md) before execution;
require its new output path and refuse collisions. Validate serialization before
touching the prior snapshot. Only the current focused audit/reporting tests and
static checks; no old experiment/suite reruns or further science. No optimization,
autograd, new reference, GPU/neural training, installs, Docker or original artifact
edits. Commit/push this feature branch, then stop. No automatic third attempt,
policy/tolerance adjustment or next experiment without separate approval.
The approved attempt2 is complete: 36 forwards in 11.79 s, physical gates passed,
no optimization. P1 changes all six fixed-tape objectives despite invariant Q/
ideal kernel; P2 is within frozen roundoff bounds. Preserve the successful
[evidence](docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2.json) and
[results](docs/MATERIAL_SYMMETRY_AUDIT_V1_ATTEMPT2_RESULTS.md) separately from the
first failed attempt. Causation of the LM solution groups remains unresolved.
Do not run a third attempt, alter science or implement the proposed coupling
diagnostic without separate approval.

Earlier authorized task: **Milestone3C-5, symmetry and sampling audit**,
from `f4f5f4d57fc9f57c4d6387663ac9ef19b40144c0` on
`feat/anisotropy-symmetry-audit-v1`. Preserve all accepted source/protocol/evidence
and original user edits in a separate worktree. Admit exactly truth plus the five
final LM states. Freeze [the symmetry protocol](docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1.md)
before ONE bounded CPU forward-only diagnostic: P0, width swap+pi/2 and theta+pi,
unchanged A2/atan2, samples/tapes/geometry/lights/targets/precision. Nominal and
rounded Q/kernel/eigen/orientation checks precede shader calls; failed physical
equivalence must not become a sampling claim. Count all36 original-batch forwards,
record per-tape/mean RGB, objectives, frames/directions/PDF/mask comparisons and
strict saved-mean reuse gates. Whole1800s, one thread, GPUs hidden, explicit memory
limits, no retries/extensions. No optimizer creation, parameter updates, autograd,
new targets/integrals/seeds/sample counts, old-suite/experiment reruns, decoder/
shader/scientific changes, GPU/neural training, installs, Docker, original artifact
edits or merges. Only NEW tiny CPU fixtures/static checks; python3 in commands.
Separate physical symmetry, finite estimator/objective preferences and unresolved
causal relationship to LM groups. Commit/push only this feature branch, then stop;
do not implement or execute the recommended next experiment without approval.
The one authorized3C-5 attempt failed before any shader call: a NumPy Boolean
in the first physical-check record is not JSON serializable. Preserve the
frozen implementation/protocol/tests, raw truncated
[output](docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json), separate valid
[failure manifest](docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json) and
[failure report](docs/MATERIAL_SYMMETRY_AUDIT_V1_RESULTS.md). The21 new fixtures
passed but missed successful analytical-record JSON serialization. No H1–H4
result is established; no automatic fix/retry or further numerical execution.
Publish this failed attempt as authorized, then stop. A reporting-only repair
and any new versioned attempt require separate researcher approval; never reuse
or overwrite the failed evidence path.

Earlier authorized task: **Milestone3C-4, controlled optimizer comparison**,
from `a64e7e77503fb3ed1a9fb26969d2bf2ad330fcea` on
`feat/material-recovery-optimizer-comparison-v1`. Admit the five complete
update1000 states and preserve all previous science/source/protocol/evidence.
Freeze [the optimizer protocol](docs/protocols/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.md)
before ONE CPU comparison: original-state Adam at most750 more returns versus
new damped GN/LM at most80 outers/four trials, seven verified directional AD
JVPs per Jacobian. Each arm/seed120 active wall seconds, whole run1800 seconds,
one thread/hidden GPUs; no retries/extensions. Demand identical initial vectors,
physical values/predictions/objectives before any update. Preserve A2/atan2,
original samples/tapes/targets/transforms/objective and success criteria; no
held-out selection, old-run/reference/test-suite replay, tuning, other numerical
changes, GPU/neural training, dependencies, Docker, historical artifact edits or
merges. Run only NEW focused fixtures/static checks; python3 for every command.
Retain all ten outcomes, partial failures, attempts, time/work counts and honest
H1–H4 interpretation. Use a separate worktree to preserve the original user edit.
Commit/push this feature branch only; no further experiment without approval.
The single comparison is complete:344.79s, all ten endpoints evaluated, no
time stops; LM3/5 successes versus Adam0/5. Preserve the frozen protocol,
[evidence](docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json),
[results](docs/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1_RESULTS.md) and all
failures/attempts. The remaining LM starts fail Q only. Do not rerun, tune the
policy, relax thresholds, implement the proposed gauge diagnostic or start any
additional experiment without separate approval.

Earlier authorized task: **Milestone3C-3, controlled optimizer continuation**,
from `15d0a07e65e5d3f18f72d68819b62e386886c492` on
`feat/material-recovery-continuation-v1`. Implement and test exact JSON restoration
of all five admitted update250 seven-scalar parameter/Adam states. Freeze
[the continuation protocol](docs/protocols/MATERIAL_RECOVERY_CONTINUATION_V1.md)
before ONE CPU execution (one thread,1,800-second wall cap, no retry/extension).
Change only update budget:750 additional Adam returns per seed101..105 to1000.
Preserve original A2/atan2, optimizer options/state, transforms, fit/evaluation
tapes, samples, targets and success criteria. Reuse accepted reference/Jacobian
gates; no prefix replay, new starts, hyperparameter/decoder/sampling/regularizer
changes, prior evidence/source edits, GPU/neural training, old suites, Docker,
installs or historical artifacts. Stop and preserve partial evidence on a failed
mandatory check. Only NEW focused tests; python3 for every Python command.
Use the new continuation worktree; preserve the original user document edit.
Commit/push only the requested feature branch; no merge or further experiment.
The one authorized continuation is complete:3,750 additional Adam returns,
275.22s, all five endpoints at1000,0/5 recovery successes. Preserve its
[evidence](docs/evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json) and
[results](docs/MATERIAL_RECOVERY_CONTINUATION_V1_RESULTS.md). Do not repeat it,
extend optimization, change settings or implement the recommended comparison
without separate researcher approval.

Earlier authorized task: **Milestone3C-2, diagnose failed material recovery**,
from `72629688bee84c0b06f4fa3c19b207917d2dd450` on
`feat/material-recovery-diagnostics-v1`. Preserve the accepted 3C-1 evidence,
protocols, A2 shader and frozen G/A. First analyze saved trajectories without
shader execution. Freeze [the new diagnostic protocol](docs/protocols/MATERIAL_RECOVERY_DIAGNOSTICS_V1.md)
before ONE bounded CPU diagnostic (1,800-second wall cap, one thread, no retry).
Use admitted original fit/evaluation tapes; reuse true-parameter targets/Jacobians,
and generate new two-grid references only at five terminals/required perturbations.
Retain original FD steps/scales and convergence bounds; stop on failed reference
convergence or exhausted budget. Explicitly separate finite objective gradients,
continuous gradients, mask/branch artifacts, compensation and causal uncertainty.
Only NEW focused CPU fixtures/static checks; python3 in every Python command.
No optimizer construction/steps, new initializations, loss/LR/regularizer changes,
prior suite/experiment reruns, accepted shader/source/evidence edits, GPU/neural
training, datasets/checkpoints, Docker, installs or permissions changes.
Commit/push only this feature branch; no merge or proposed next experiment.
Use the separate diagnostics worktree; preserve the existing uncommitted edit in
the original workspace's ANISOTROPIC_BRDF_SCIENTIFIC_AUDIT_V1.md unchanged.
The one authorized diagnostic is complete (81.49s, zero optimizer calls).
Preserve its two new evidence files and
[results](docs/MATERIAL_RECOVERY_DIAGNOSTICS_V1_RESULTS.md). A longer controlled
continuation is recommended only; no extra execution or correction is authorized.

Earlier authorized task: **Milestone3C-1, first known-material recovery**,
from `179e36eb01e976d216a45fd323cc764308b49bd0` on
`feat/known-material-recovery-v1`. The researcher approves implementation and
ONE bounded CPU experiment under the unchanged
[v1 protocol](docs/protocols/KNOWN_MATERIAL_RECOVERY_V1.md).
Use A2/atan2, known flat geometry/lighting and independent float64 targets.
Mandatory reference convergence and independent seven-parameter Jacobian gates
must pass before any optimizer construction. Preserve all five declared starts,
failed gates/starts, fixed fit/evaluation tapes and criteria. At most250 Adam
returns per start,1,250 total;1,800-second process cap, one CPU thread. No
threshold/hyperparameter tuning or rerun. Only NEW focused tests and essential
static checks are allowed. Use python3; no old experiments/suites, frozen-source/
prior-evidence changes, GPU/neural-network training, native data/checkpoints,
Docker, installs or extra scientific variants. Commit/push only this feature
branch after analysis, without merging or starting another experiment.
The single authorized run is now finished: both gates passed, all1,250 updates
completed, but0/5 starts met the recovery criteria. Preserve its JSON and
[results](docs/KNOWN_MATERIAL_RECOVERY_V1_RESULTS.md). No additional execution
or setting change is authorized by this completed milestone.

Earlier authorized task: **Milestone3C-0, experimental shader and known-material
recovery preparation**, from `1067d3d5370eaaeabac7eedd650d4d913f87f820`
on `feat/material-recovery-preparation-v1`. Implement a separate differentiable
CPU shader: A0/original, A1/B3 and A2/midpoint cosine diffuse heights, with
original versus atan2 decoding as an independent explicit factor. Preserve the
frozen G/A snapshots, earlier operators, protocols and accepted evidence. Reject
invalid corrected encodings; do not silently choose a zero-vector fallback.
Analyze energy/regularizer semantics without modifying actual training losses.
Design the known-material benchmark, independent references and gauge-invariant
metrics, but do not execute recovery or generate observations. Run only NEW tiny
CPU fixtures and essential static checks; use python3. No old experiments/suites,
GPU queries, native models, checkpoint loading, datasets, Docker, installs,
optimization/training or additional scientific changes. CPU autograd/finite
differences on analytical inputs are authorized. Commit/push only this feature
branch when complete; no merge or production/training integration.

Earlier authorized task: **Milestone3B-2, controlled anisotropic estimator
correction and ablation**, from `759a1253bf1db207221d6989ec4ca6f9c3ae3225`
on `feat/anisotropic-estimator-ablation-v1`.
Freeze [the experiment protocol](docs/ANISOTROPIC_ESTIMATOR_ABLATION_V1_PROTOCOL.md)
before execution. Implement B0/original, B1/total-count specular normalization,
B2/cosine diffuse radial mapping and B3/both in separate experimental modules.
Hash-admit the original A pure methods; preserve frozen G/A, old evidence,
sample counts, paired draw ordering, a=0 BRDF, decoder and regularization.
Only NEW focused CPU fixtures, essential static checks and exactly ONE bounded
fixed-grid 3B-2 CPU experiment are authorized. Use python3. Do not repeat old
tests/diagnostics, query CUDA, construct networks, load checkpoints, run training,
datasets, ray tracing, Docker, installs or additional model changes. Keep the
two fixed quadrature grids and predeclared tolerances; unresolved references
remain unresolved without retries. Separate mathematical expectation, finite
sampling error and physical validity; no reconstruction improvement claims.
Record errors, budget and runtime, preserve the one-run evidence, and publish
the completed feature branch by commit/push only. Do not merge or start GPU work.

Earlier authorized task: **Milestone3B-1, scientific anisotropic BRDF audit**,
from `959db370cd18a09a20b2a0c9bb24a20a2cdba22a` on
`feat/anisotropic-brdf-scientific-audit`.
Read [ANISOTROPIC_BRDF_SCIENTIFIC_AUDIT_V1.md](docs/ANISOTROPIC_BRDF_SCIENTIFIC_AUDIT_V1.md)
and its equation map/3A-2 close-out. Independently analyze existing hash-selected
3A-2 JSON/NPZ offline; do not repeat native experiments or promote failed geometry.
Inspect frozen A and researcher-provided thesis/Kurt/NeRO PDFs read-only. Derive
actual equations, sampling/PDF/estimator weights, roughness/theta/frame settings
and effective regularization scopes. Only new analytical CPU tests and essential
static checks are authorized; use python3 in every Python command. Pure, explicitly
selected/hash-verified source method bodies may be exercised on CPU fixtures;
no scientific imports/constructors, checkpoints, GPU, training or optimization.
Keep defects, numerical fragility, approximations and proposed improvements
distinct. All proposed changes require separate approval; reference mathematics
is not a production implementation. Preserve frozen sources/YAML and accepted
evidence. No installs, original artifact changes, old-suite/diagnostic repetitions
or infrastructure expansion. The researcher explicitly authorizes committing
and pushing this reviewed feature branch; do not merge it or start the next
controlled experiment. Earlier no-commit rules do not override this authorization.

Earlier authorized task: **Milestone3A-2, bounded scientific SDF/mesh/A-normal
audit**, on `feat/geometry-sdf-mesh-audit-v1` from accepted base
`3fe197bbdc37957c187c8589fffdb123aaac692d`.
Read [GEOMETRY_SDF_MESH_AUDIT_V1.md](docs/GEOMETRY_SDF_MESH_AUDIT_V1.md).
Preserve accepted 3A-1 evidence and frozen G/A snapshots. Consume only explicitly
selected/hash-admitted extraction/mesh/ray/NPZ/checkpoint/3A-1 artifacts. Prepare
original G eval-mode strict full-state loading and bounded float32 SDF queries
with first input derivatives (create_graph=False), only in the reviewed isolated
researcher runtime with explicit pickle trust. No optimizer, dataset loading,
rendering, reinitialization after load or checkpoint writes. Deterministic samples
and all three anomalous rays, at most 20,000 SDF point evaluations; no tolerance
tuning, mesh repair/transform, normal flip, rescaling or scientific substitutions.
Separate raw network SDF from outside-sphere masked extraction field; signed
normal comparisons and local measured exterior polarity never prove producing
provenance or research handoff. Both remain unresolved. Codex may implement and
run only NEW analytical CPU tests and essential syntax/static checks; do not
repeat 3A-1, full/historical suites, lifecycle or smokes. No native checkpoint
loading/inference, GPU/Docker, extraction, training, installs, original evidence/
scientific source/config/data edits, permission changes, automatic commit/push/
merge or later milestones. Prepare bounded read-only-input researcher commands
and stop for review. Do not merge into paper/infrastructure.

Earlier authorized task: **Milestone3A-1, independent CPU geometry evidence and
coordinate audit**, on a new feature branch from accepted base
`aec29d2063b9c1a88113380018239722dd2682b6`; `d3f3bec4` is not the accepted base.
Read [GEOMETRY_SCIENTIFIC_AUDIT_V1.md](docs/GEOMETRY_SCIENTIFIC_AUDIT_V1.md).
Consume only explicitly selected/hash-admitted existing extraction JSON/PLY and
saved A ray NPZ evidence. Verify pinned G/A source identity without importing
models, loading original camera/checkpoint pickle, tracing rays or changing any
input. Use independent NumPy/CPU camera/coordinate/intersection/normal/silhouette
calculations, not the existing summarize() helper. Numerical loading must use
allow_pickle=False, with finite/shape/dtype/resource checks. Freeze bounds before
original-evidence execution; no tolerance override, repair, transform, normal flip,
implicit file discovery or retries with changed criteria. Write only a fresh
explicit output directory and versioned report with per-check pass/fail/unresolved
and blocker reasons. Separate original producing provenance, prior architecture
compatibility, artifact identity, coordinate consistency, scientific3D accuracy,
3A-2 SDF orientation and research handoff; never promote one from another.
Historical r128 remains diagnostic and ordinary handoff unchanged. Codex may
implement/test synthetic CPU fixtures and static checks, inspect read-only metadata
and hashes, and prepare exact researcher CPU commands. Do not repeat infrastructure
smokes or run training/model inference/native validation, GPU/Docker, SDF queries,
installs, original evidence/source/config/data/checkpoint changes, permissions,
automatic commit/push/merge or further milestones. Stop after bounded audit review.

Earlier authorized task: **Milestone2, bounded original G/A training/checkpoint/
evaluation lifecycle preparation**, reviewed from
`d72b2b84c2e7d704ee2f6642eca9b2ab0366350b`.
Read [TRAINING_CHECKPOINT_EVALUATION_LIFECYCLE.md](docs/TRAINING_CHECKPOINT_EVALUATION_LIFECYCLE.md).
Milestone1 native A bell ray/two-update evidence is accepted only in its narrow
diagnostic scope; preserve its original reports and every pinned snapshot. Prepare
separate save (two original optimizer returns + explicitly early original save),
cold-process restore (original strict model/Adam load + one original index2
update), and original held-out evaluation actions for pinned G/A. Trust/hash/
source/configuration/mesh/split/optimizer compatibility gates precede restoration.
No partial loading, implicit old-checkpoint selection or hyperparameter repair.
Preserve original serialized fields, scheduler floating-point operation order,
G no-clipping/A clipping and all losses/rendering/sampling/defaults. Original
checkpoints omit RNG/loader/ray cursor; record actual resumed sampling/reset and
never claim bitwise uninterrupted continuation without proof. No continuation
sidecar is implemented in this milestone. Evaluate G source0/current G split;
A source58 then11/current A split, full original metric/background/channel
semantics. Record black misses, encoded normals, absent rotation/unfilled alpha
without correcting them. Require weights/optimizer unchanged by evaluation.
Keep r128 mesh historical/diagnostic and ordinary research handoff unchanged.
Codex may inspect small JSON/file identities, implement wrappers/manifests and
CPU toy/static tests only. No native models/camera/checkpoint loading, Docker/GPU
execution, installations, scientific/config/input/original-evidence modifications, image/
permission changes, commit or push. Provide bounded isolated researcher commands
and stop for review. Milestone2 is not complete until separately reviewed native
save/restore/evaluation evidence passes for both stages; no convergence claims.

Earlier authorized task: **historical G bell mesh → pinned A bell material
integration diagnostic preparation**, reviewed from
`e6d2262b908e7125e440c64d298d12b8ef5fd105`.
Read [BELL_MATERIAL_INTEGRATION_DIAGNOSTIC.md](docs/BELL_MATERIAL_INTEGRATION_DIAGNOSTIC.md).
Only the exact SHA-pinned completed G r128 extraction report and original PLY
may receive a separate diagnostic-only exception. Ordinary research handoff and
all existing evidence/source snapshots stay unchanged. Preserve mesh winding,
original A trace normal negation, camera conventions, original bell YAML, full
material resolution/masks, optimizer/scheduler/RNG and shader-setting scopes.
Prepare independent bounded ray-only and separately gated two-update actions.
Observe raw BVH/final A normals without repair; frozen camera/normal/silhouette
gates must pass before explicit two-update execution can be admitted. Reuse the
existing MaterialProbe/TwoUpdateTrace and owned cleanup; no shader/model/optimizer
construction in ray-only mode. Historical checkpoint provenance remains unresolved;
r128 is diagnostic, not canonical512 or attributable new G training/research mesh.
Codex may implement metadata/I/O wrappers and CPU toy/static tests only; no native
camera/checkpoint loading, native GPU/Docker/ray tracing/training, original input
or source/YAML modifications, installs, image/permission changes, commit or push.
Provide fresh-output, read-only-input, no-network tmux/Docker commands with timeout
and stop for review before native execution or full G→A research experiments.

Earlier authorized task: **G bell historical-best checkpoint mesh extraction
diagnostic preparation**, reviewed from `d4e80bde47fbcb0d2a30fe4bf462d7a28e8fb9d1`.
Read [G_MESH_EXTRACTION_DIAGNOSTIC.md](docs/G_MESH_EXTRACTION_DIAGNOSTIC.md).
Prepare a separate explicit best-checkpoint/resolution128 protocol using exact
pinned G extraction/SDF functions and original Trimesh processing/export defaults.
Original G CLI still selects model.pth/default512; no silent substitution.
Preserve bbox±1, zero level, outside-sphere+1, SDF scale/coordinates, winding,
normal handling, scientific snapshots, YAML and all existing evidence. Inspect
and report finite/nonempty/index-valid/nondegenerate raw, processed and serialized
geometry without repair. Keep selected historical producing provenance unresolved;
no new G training attribution or ordinary research handoff admission. The
researcher's completed G/A smoke and strict G compatibility are prerequisites,
not permission to alter old evidence or claim full training reproduction.
Codex may implement wrappers, metadata and CPU/static fixtures only. No native
checkpoint/camera loading, native models, GPU/Docker/extraction execution, full
material/G-to-A runs, installs, source/artifact/permission changes, commit or push.
Prepare bounded researcher tmux/Docker instructions and stop for review.

Earlier authorized task: **bounded native A anisotropic material smoke
preparation**, reviewed from `71257544d9038929197d45b3dafb2434ed2d731f`.
Read [MATERIAL_TRAINING_SMOKE.md](docs/MATERIAL_TRAINING_SMOKE.md).
Prepare exactly two original A optimizer returns using `TwoUpdateTrace`; preserve
the pinned A scientific snapshot and original bear YAML, including effective
shader regularization scopes, warm-up, clipping, sampling and all mathematics.
The explicitly selected/hash-pinned historical bear mesh is approved only for
this isolated diagnostic. Ordinary research runs still require an attributable
G-to-A mesh handoff. No checkpoint initialization, resume, validation, saving,
final NVS, material extraction or full training is authorized by this task.
Record actual anisotropic/RGB/regularizer calls and per-group gradients, effective
settings, Adam state, sampling and source identities with the existing helpers.
CPU toy autograd/explicit stand-ins and static checks only from Codex. Native
Docker/GPU execution is researcher-side after patch review; no scientific edits,
original input modifications, installations, image/permission changes, commit
or push. Short diagnostic success is not publication/full-training acceptance.

Earlier authorized milestone: **1D.6A, G geometry validation and checkpoint
compatibility**, reviewed from `596c070664ffd9cfff50805939026cff9ce21883`.
Read [G lifecycle audit and commands](docs/G_GEOMETRY_LIFECYCLE.md) and
[G training acceptance](docs/evidence/G_GEOMETRY_TRAINING_ACCEPTANCE.md).
The verified native G R/R-repeat/C evidence admits only the fresh two-update
prefix. Use evidence policy v2; preserve v1 and all original reports/protocols.

Prepare independent hash/source-gated G compatibility, optional historical
optimizer-load diagnostic (no update), original G camera-0 validation, and
manual G-produced save/load roundtrip protocols. Use only the exact G snapshot;
no scientific method replacement or A scientific import. Historical pickle
loading remains explicitly trusted and researcher-side only in an isolated
runtime. Strict network keys/shapes/dtypes and loading must precede validation.
An optimizer library load can override G settings and is not G resume admission.
Manual serialization after two G updates is not the natural 1,000-step schedule.
Extraction is source audit/design only; original `model.pth` selection cannot
be silently replaced by `model_best.pth`. Random/two-update diagnostic geometry
must never be accepted as research geometry or a material handoff.

Codex may inspect read-only bytes/metadata, implement isolated adapters and CPU
toy/explicit stand-in tests, and run CPU/static checks. No native model,
historical checkpoint/camera deserialization, GPU/Docker execution, extraction,
1,000-step/full training, scientific edits, installs, environment/permission
changes, commit or push. Stop for review before researcher native execution.
Earlier source-preservation and scientific-integrity rules continue to apply.

Earlier authorized milestone: **1D.5-RESET, authoritative dual scientific
baselines**, reviewed from `bed325fe511ec52a61f5793d0ee0acd91dad26b9`.
Read [DUAL_SCIENTIFIC_BASELINES.md](docs/DUAL_SCIENTIFIC_BASELINES.md), the pinned
[source manifest](scientific_sources/manifest-v1.json),
[evidence supersession v1](docs/evidence/dual-baseline-supersession-v1.json) and
[G two-update protocol](docs/protocols/GEOMETRY_G_TWO_UPDATE_V1.md).

Authorize source audit, exact immutable G/A Git-blob copies, stage-specific
source/import selection, filesystem-only I/O adaptation, metadata/provenance,
CPU fixtures and static checks. Root legacy scientific code/configuration,
original branch histories, artifacts/reports/checkpoints/meshes/logs and existing
A protocols remain unchanged. Source snapshots must match every pinned byte;
never fix suspicious science inside them. Any required scientific change is a
blocker for separate review and a newly versioned scientific source/protocol.

Selected execution uses `run_stage_experiment.py`: G's actual geometry trainer,
renderer/dependencies, two-way unshuffled splits, original losses, Adam defaults,
configured cosine schedule and **no clipping**; A's actual anisotropic material
implementation, defaults/regularizer scopes, clipping, optimizer/scheduler and
validation/NVS behavior. One scientific namespace per cold process; reject
ambiguous sources, module fallbacks, custom overrides or unadmitted checkpoint
initialization/resume before native execution. Record infrastructure and
scientific commits/hashes separately. The old experiment CLIs reject --execute;
legacy A diagnostic code is historical-only, not selected G acceptance.

Preserve and supersede prior evidence by versioned records, not rewriting it.
A two-update R is invalid for G acceptance; prepared A R-repeat/C remain
unexecuted/superseded. Exact camera-0 PSNR and montage identity remain valid A
evaluation observations. The original unshuffled training split is strongly
supported but unproven producing-run attribution, and the historical checkpoint
is not automatically G-produced or eligible for G resume. Future G-to-A meshes
require explicit read-only file/hash, producing G run/source evidence and
coordinate/scale/normal/orientation/preprocessing conventions; no substitution.

Stop after implementation/CPU/static checks for review. New G R, independent
R-repeat and C must be executed afresh by the researcher under the frozen G
protocol, reviewed for compatibility/repeatability before research progression.
No Codex Docker/GPU/native model/inference/training, original camera/checkpoint
deserialization, installs, builds, permission/environment changes or automatic
commit/push. CPU toy autograd/Adam and original source bodies with explicit
stand-ins are allowed; they are not native scientific acceptance. Earlier A
preservation rules below describe historical paths and do not override the
new selected G geometry authority.

Earlier authorized milestone: **1D.4B-3D, finalize historical bell camera-0
evaluation evidence**. Inspect the existing primary/control/comparison reports
and their recorded hashes without regeneration. Write documentation and a small
versioned success manifest, preserve original evidence and all image-58 code/
reports, and run CPU/static validation only. No scientific source/configuration,
checkpoint/dataset, Docker/image/environment/permission changes or native GPU/
training execution; no automatic commit or push.

The exact camera-0 PSNR, five-decimal SSIM match and step-pair equality may be
recorded as verified within the inspected files and tested implementation. Camera
selection accounts for the observed metric discrepancy in that scope; it does
not establish producing-run training membership. The unshuffled two-way training
split remains a strongly supported hypothesis. Current A validation 58 must not
be labeled the historical geometry evaluation split. Preserve native versus host
JPEG-decoder measurements separately and never fit a tolerance or rewrite an
original report. See [camera-0 success](docs/evidence/BELL_CAMERA0_DIAGNOSTIC_SUCCESS.md)
and the updated [attribution report](docs/evidence/BELL_EVALUATION_PROVENANCE.md).

Earlier authorized milestone: **1D.4B-3C, historical bell camera-0 diagnostic
preparation**, on the existing `paper/infrastructure` worktree. Preserve the
uncommitted 1D.5A implementation/protocol and all original image-58 code/evidence;
shared AGENTS/static-check changes may add this scope only. No commit/push.

Prepare a separate explicit source-camera "0" adapter using the pinned admitted
bell checkpoint, original maintained A renderer/mathematics and existing strict
hash/state checks. Keep A train/test/NVS tables unchanged; do not substitute G or
infer source ID from validation iterator index. An external selection facade
may supply camera-0 information to original forward/test_step methods without
editing renderer internals or weights. Preserve preprocessing, resolution,
backgrounds, sampling, SDF gradients, conversion and metric definitions. Compare
fresh evaluation steps 149999 and 150000 with stored checkpoint step still 150000.

Record the researcher's visual historical-0/current-58 identification as supplied
evidence. The unshuffled two-way producing split and explanation of the metric
gap remain hypotheses. Freeze the [camera-0 protocol](docs/protocols/BELL_CAMERA0_DIAGNOSTIC_V1.md):
reuse unchanged inference RGB bounds for the step pair; historical score/JPEG
comparisons are descriptive with no invented acceptance threshold, resizing,
alignment or compression compensation. Closeness is never producing-run proof.
See [BELL_CAMERA0_DIAGNOSTIC.md](docs/BELL_CAMERA0_DIAGNOSTIC.md).

Codex may prepare the isolated runner, CPU fixtures/static checks and documentation
only. **No historical checkpoint/camera-pickle deserialization, native model
construction, Docker/GPU execution, native inference/training, optimization,
package/image/permission changes or original-artifact modification from Codex.**
Researcher commands must use explicit trust/runtime gates, existing admitted image,
read-only source/datasets/thesis, no network, one selected GPU, non-root/fresh
outputs and resource limits. Stop for review before native execution. Document
any scientific-semantic blocker rather than repairing it or fitting the result.

Earlier preparation retained for review: **1D.5A, bounded bell geometry training smoke
implementation and CPU/static validation for review**, based on
`ef7e6b270137c1315c1ef01f303af910ee1dcbaf`. Read the admitted inference evidence,
historical attribution audit and existing geometry/training implementation first.
Implement only separate researcher-side entry points, a scoped observer/stop,
evidence/comparison helpers, disposable CPU fixtures and documentation. Keep the
actual Trainer.run body, renderer, datasets/loaders, losses, optimizer/scheduler,
scientific source and legacy configurations unchanged. No loop replacement,
operator simplification, sample reduction or numerical repair is authorized.

The fixed budget is exactly **two successful Adam returns at steps 0 and 1**.
Preserve full total_step=300000, intervals, seed 6033, actual A splits/loader/RNG
order, clipping 0.7 and native Adam construction. Step 0 LR is zero: moment/counter
state advances without parameter changes; step 1 LR is about 1e-7 and parameters
must change. Do not add a third update or modify warm-up to get two nonzero-LR
updates. The proposed external CPython observer must stop immediately after
Adam return 2, reject incompatible code/traces, retain failures and close owned
workers/writer/progress. If preserving these operations is impossible, document
the blocker and stop; never silently modify the scientific loop.

Use independent legacy R/R-repeat and schema-1 C configuration routes, fresh
unique run directories and no checkpoint/optimizer restoration. Prepare read-only
source/dataset mounts and an **empty read-only thesis placeholder**, so historical
weights/results are not exposed. Record finite loss/gradient/parameter/Adam-state,
actual aggregate/clipping/LR, parameter changes, sampling/RNG/runtime identities,
source/config/input hashes and failure/cleanup evidence. No checkpoint/validation
is expected before the preserved natural 1000-step intervals. Do not force them.

Freeze the [two-update protocol](docs/protocols/GEOMETRY_TWO_UPDATE_V1.md) before
native execution. Require R-repeat repeatability before C interpretation; keep
strict tensor/state hashes and training-unit scalar bounds distinct from RGB
inference tolerances. Hash/norm agreement for this short instrumented prefix
cannot establish full training, historical thesis or material/BRDF equivalence.
See [GEOMETRY_TRAINING_SMOKE.md](docs/GEOMETRY_TRAINING_SMOKE.md) for the design,
report schema, expected commands, scientific limits and native validation gaps.

Codex may execute **CPU toy autograd/Adam fixtures and static/configuration
checks only**, with temporary inputs and explicit renderer/loader/I/O stand-ins.
Never construct the native scientific model, decode original camera/checkpoint
payloads or run native training from Codex. **No Docker/GPU commands, native
training/inference/rendering, package/image/environment/permission changes, mesh
extraction or historical artifact modifications.** Prepared researcher commands
are for code review, not execution approval. Preserve earlier real inference and
blocked/provenance evidence; do not invent measurements or solve producing-run
uncertainty through this smoke. Stop after implementation/CPU/static checks;
do not commit or push automatically or start full/material training.

The researcher approved archiving the original small smoke reports/configuration
and manifest files at `/home/isavand/paper/evidence/1d3c` after the originally
requested `paper_runs/evidence/1d3c` destination was inaccessible. Verify the
approved destination is writable, preserve original bytes and source paths/
hashes, refuse overwrites, and exclude caches/generated datasets. Do not change
ownership/permissions or Docker settings. That earlier milestone additionally
authorized only the dual-baseline reset implementation and CPU/static work above;
earlier A evidence and preparations remain preserved under supersession policy.
No scientific code/YAML,
image/build, package/environment or original artifact changes are allowed.

Use **verified**, **inferred** and **unresolved** classifications in
[BELL_CHECKPOINT_ADMISSION.md](docs/evidence/BELL_CHECKPOINT_ADMISSION.md).
Distinguish file-backed smoke evidence, independently corroborated runtime
identity and researcher-reported 109/109 Python 3.8 test results. An image ID in
a report is caller-declared until independently checked; source-file hash matches
do not establish the launch's full Git commit or clean status. Report Docker
socket limitations separately without bypassing them.

Current A-based inference regression and historical G-based thesis reproduction
have separate admission decisions. Preserve the exact A three-way seeded split
and G two-way unshuffled split/RNG effects; verified present source-defined IDs
are not verified producing-run IDs. Checkpoint hashing, matching YAML, same-shaped
weights or smoke success cannot establish origin/compatibility/reproducibility.
No historical checkpoint loading is allowed from Codex in this implementation
milestone. Saved step/metric are now observed in the admitted JSON report, while
architecture compatibility is structurally verified against the real reference.
Producing commit/configuration/split/selection remain unresolved. Inference
adapters are implemented and the admitted researcher execution passed. The
separate two-update training preparation above is authorized; no scientific
changes or new native execution are authorized in this follow-up.
Do not commit or push automatically. The earlier
[baseline preparation](docs/BASELINE_PREPARATION.md) and
[inference protocol](docs/protocols/CHECKPOINT_INFERENCE_V1.md) remain proposals,
not execution approval.

Keep installed evidence, publisher checksums and byte-verified artifacts distinct.
Registry presence/tag agreement does not establish historical base identity;
do not mark an incomplete resolution reviewed or change the accepted Docker image.
Preserve the accepted Docker architecture, full pre-install validation gate,
dependency pins and historical runtime. Use filtered read-only diagnostics,
registry metadata queries, dependency/source assessment and CPU/static fixtures.
Preserve infrastructure test-harness behavior and every existing assertion; do
not introduce unrelated fixes or stage originals in this documentation milestone.
Never run setup to inspect it, copy a working tree wholesale, invent missing pins,
or promote publisher artifact metadata to verified downloaded/historical identity.
The earlier 1D.3C scope was the separate smoke CLI/helper, CPU tests,
the AST compatibility fix and supporting documentation; it is not extended here.
Use the recovered system Python 3.8.10 / CUDA 11.1 / Torch
1.8.1+cu111 / ABI 0 reference; no Micromamba or scientific upgrades. Pin nvdiffrast
and raytracing to the exact recovered revisions/setup hashes. Record unknown
package/system versions, hashes and base digest as unresolved; never invent them
or label an incomplete manifest a complete lock. Do not copy original checkouts
into Git or change their contents. CPU infrastructure fixtures must use temporary
datasets and explicit model/device fakes; never run native training or rendering.
Do not install packages, build images, start/stop containers, run GPU code or
download package artifacts. Read-only access may fail; do not bypass restrictions.
Preserve the existing image/container and all scientific/infrastructure behavior.
Do not change host configuration/permissions or bypass access controls. Separate
reference declarations, local observations, Codex metadata and proposals.
Inspect only task-relevant fields;
never publish credentials, private environment values, tokens or unfiltered
container/build configuration. If access fails, report it and supply filtered
read-only commands for the already-authorized regular terminal.

The following integration rules remain in force for later authorized work.
Geometry behavior must remain unchanged, including its split adaptation and
final-NVS deferral.
Preserve the original split helper, call count, train/validation/NVS IDs and RNG effects.
For geometry, record final NVS as deferred to a separately versioned evaluation
protocol; do not introduce cameras, renders, or metrics for that protocol. Keep the legacy
`run_training.py` path and legacy defaults, YAML, training-loop computations,
network definitions, evaluation protocols, losses, optimizers, scheduler,
sampling, seeds, precision, and arithmetic unchanged. Do not migrate legacy
configurations or repair scientific defects.

The new envelopes are accepted by separate `run_experiment.py` geometry and
`run_material_experiment.py` material entry points; existing entry points remain
legacy. `run_native_smoke.py` is a separate explicit preflight/one-forward harness;
see [NATIVE_GPU_SMOKE.md](docs/NATIVE_GPU_SMOKE.md) for mount gates and limits.
Preserve shallow legacy merges and the scope of shader fields; top-level
shader-like fields must not be silently forwarded into `shader_cfg`. Diagnostics
are evidence, not authorization to correct behavior. Run allocation and the
shared trainer loop are tested only in disposable CPU fixtures with explicit
model/device fakes; do not allocate a real experiment or execute native training
for this task. Configuration validation must not import scientific modules,
construct models, load tensor checkpoints, or consume scientific RNG state.
Fresh runs must refuse collisions; same-run continuation requires an explicitly
named checkpoint and matching identity. Historical initialization is explicitly
weights-only in a new run. Neither continuation nor old checkpoints establish
exact resumption without RNG/sampler state.
Dataset construction without `preprocessing` remains legacy compatibility mode.
New mode must write derived products only inside the selected experiment cache,
preserving existing source-side products as read-only inputs. Cache identities
must distinguish dataset, scene/mode/parameters, and implementation versions.
Schema-1 loads require an explicitly approved configuration root; references must
not escape it through traversal or symlinks. Record native-backend skips and
pre-existing custom-loader/evaluator defects instead of silently fixing them.
The split-unpack adaptation requires the explicit renderer keyword
`isolated_geometry=True`, supplied by the experiment trainer. Preprocessing-cache
opt-in alone must not select it. The legacy split-unpack and final NVS dispatch
behavior remains unchanged. Isolated execution records the NVS deferral in
`geometry.json` and the console log. See
[GEOMETRY_TRAINING.md](docs/GEOMETRY_TRAINING.md). CPU routing/interface tests do
not prove native geometry training is runnable or numerically equivalent on GPU.

Material runs must select `inputs.mesh` through an explicitly declared input
root. Keep mesh bytes and coordinate conventions unchanged; computed normals
remain in-memory legacy behavior. Never forward top-level regularization,
anisotropy, lobe, or other shader-like fields into `shader_cfg`. Record actual
renderer and shader settings separately. Material NVS retains its original split,
rendering, metric, image composition and append/overwrite behavior within the
explicitly selected run. Debug point export is routed when called, not enabled
as a new training option.

Before material checkpoint state loading, verify stage/algorithm declaration and
network keys/shapes/dtypes. Untagged historical initialization requires a reviewed
manifest under the approved config root binding checkpoint SHA-256 to an approved
source commit. Same tensor shapes do not establish branch/mathematical equivalence;
reject foreign origins and conflicting tags. Declarations are evidence supplied
by the researcher, not independently recovered producing history. A future
scientific variant must receive a distinct algorithm identity. See
[MATERIAL_TRAINING.md](docs/MATERIAL_TRAINING.md) for manifest requirements,
writer inventory, preserved behavior, and remaining provenance/native limits.

The environment evidence and proposed recovery commands are in
[GPU_ENVIRONMENT_INVENTORY.md](docs/GPU_ENVIRONMENT_INVENTORY.md) and
[GPU_ENVIRONMENT_DIAGNOSTICS.md](docs/GPU_ENVIRONMENT_DIAGNOSTICS.md).
See [docker/paper/README.md](docker/paper/README.md) for the proposed definition,
unresolved build inputs and Python 3.8 assessment. Test-harness compatibility only
is authorized; no interpreter upgrade or scientific compatibility fix is allowed.
See [docker/paper/PREREQUISITES.md](docker/paper/PREREQUISITES.md) for recovery,
verified source staging and remaining gates, and
[docker/paper/RUNTIME_DEPENDENCIES.md](docker/paper/RUNTIME_DEPENDENCIES.md) for
the recovered pins and unverified CUDA base candidate. Before any future build, review
the full exact dependency/source/artifact resolution and use a distinct candidate
image tag. Preserve source licences, compiler flags and scientific settings.
Do not begin first builds/environment installation, native GPU smoke tests,
extraction/evaluation implementation or work beyond current 1D.5A preparation,
or scientific repairs without separate user approval. Do not run Docker builds/
runs, GPU tests, real training or expensive experiments in this milestone.
Do not commit or push automatically. A later explicit user instruction
updates this boundary. See [CONFIGURATION_AND_PATHS.md](docs/CONFIGURATION_AND_PATHS.md)
for implemented APIs and limitations, and [INFRASTRUCTURE_PLAN.md](docs/INFRASTRUCTURE_PLAN.md)
for later milestones and acceptance criteria. The writer inventory, opt-in API,
preservation assessment, and limitations are in
[DATASET_CACHE_ISOLATION.md](docs/DATASET_CACHE_ISOLATION.md).
