# Controlled material recovery continuation v1 — results

**Completed: zero of five starts recovered the material under the unchanged
criteria.** All five training objectives, held-out total radiance errors and
unseen-relighting total errors improved after 750 additional Adam updates each.
Physical Q error nevertheless worsened in four starts. Additional duration
helps fit radiance, but extending 250 to 1,000 updates is insufficient to establish
successful parameter recovery or that duration was the primary cause of failure.

This is one seven-scalar CPU experiment with the unchanged accepted A2/atan2
shader. It is not neural training, native reconstruction, historical reproduction,
or a change to the frozen scientific implementations.

## Evidence and execution

| Item | Identity/result |
| --- | --- |
| Accepted base | `15d0a07e65e5d3f18f72d68819b62e386886c492` |
| Feature branch | `feat/material-recovery-continuation-v1` |
| [Frozen protocol](protocols/MATERIAL_RECOVERY_CONTINUATION_V1.md) SHA-256 | `e6bbde21188d7178669369ea551cfeb663633b60e0c09ce775154d3e473b0693` |
| [Implementation](../experiments/material_recovery_continuation_v1.py) SHA-256 | `de737d49fd04078c89c6b429e82bbbea7c7c66f6fc8c72f1a936dca012a7b6f0` |
| [Original evidence](evidence/KNOWN_MATERIAL_RECOVERY_V1.json) SHA-256 | `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451` |
| [Reused diagnostic](evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json) SHA-256 | `6106278d6f3bb41eed0933b668627626c462faaf20433c13c34d1c5d7685539f` |
| [New machine-readable result](evidence/MATERIAL_RECOVERY_CONTINUATION_V1.json) SHA-256 | `224cb9a0cb27bbaacc61b8d575efbcd2c37854b716fc51bc78b439020239f24f` |
| Result size | 596,885 bytes |
| Execution count | One; no retry, prefix replay, new initialization or reference integration |
| Wall / process CPU time | 275.21672060899436 / 275.17775427600003 seconds |
| Runtime | Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121; CPU, one thread, GPUs hidden |
| Optimizer returns | Exactly 3,750: 750 for each seed101–105, update250 to1000 |
| Fitting work | 3,750 update forwards + five endpoint gradient measurements; 360,480 points |
| Fit directions | 369,131,520 specular + 184,565,760 diffuse |
| Final evaluation | Five calls, 1,680 points, 1,720,320 specular + 860,160 diffuse directions |
| New independent reference directions | Zero; original convergence/identifiability gates reused |
| Integrity/cleanup | Both global CPU RNGs and fixed tapes unchanged; owned thread settings/signal handlers restored |

The literal command is frozen in the protocol and saved in JSON `argv`. The
experiment ran from a fresh separate worktree, with Git HEAD at the accepted
base and new implementation/protocol files recorded by hash. It was not run from
a previously committed 3C-3 implementation. Published source hashes allow
verification; the eventual publication commit was necessarily created afterward.
Torch's CUDA-linked package version does not imply CUDA execution. Installed
runtime version agreement is verified; original library/binary byte identity
was not recorded and cannot be established retrospectively.

## Exact state restoration and unchanged computation

All five original `last_unconstrained` vectors, scalar step250 counters, first
moments and second moments were restored **before any forward or update**.
Restoration records include complete values, CPU device, float32 dtype, shapes
and byte hashes. Every JSON number round-tripped exactly as float32, physical
values matched the unchanged transform bitwise, and optimizer flags matched.
All final Adam counters are1000; no optimizer was restarted or prefix repeated.

Original JSON values did not carry dtype tags. Representation is established
from the original admitted float32 runner, original Adam flags/default dtype
and matching Torch version. We distinguish that source-supported reconstruction
from nonexistent original dtype metadata. Synthetic uninterrupted-versus-JSON-
restored Adam fixtures match parameters and all moment/counter bytes at every
tested continuation update. As an additional observed check, the **first
pre-update loss of every real continuation is exactly the saved 3C-2 update250
objective**; all five comparisons pass without a tolerance.

The only changed experimental setting was the endpoint1000. Learning rate .01,
betas(.9,.999), eps1e-8, weight decay0, optimizer flags, parameter transformations,
24 fit observations, targets, tape hashes, 512/1024 sample counts and A2/atan2
shader identities match the accepted run. No regularizer, schedule, clipping,
noise or held-out-based selection was introduced. No references or scientific
gates were regenerated. Every state was evaluated at its fixed final update.

## Parameter recovery: update250 versus1000

Relative Q errors are dimensionless. Orientation uses the original modulo-pi
major-axis/axis-swap gauge and is identifiable in all five final states. Albedo
and metallic errors are absolute in[0,1]. Arrows show250→1000, not selected best
intermediate checkpoints.

| Seed | Relative Q error | Orientation error, degrees | Maximum RGB albedo error | Metallic error | Full success |
| --- | --- | --- | --- | --- | --- |
| 101 | .496486 → .600267 | 7.357124 → .394162 | .087870 → .082089 | .278234 → .185369 | fail |
| 102 | .576588 → .718296 | 4.369551 → .315399 | .103085 → .095310 | .257403 → .206847 | fail |
| 103 | .645957 → .697309 | 1.179568 → .666052 | .108617 → .086853 | .243960 → .190470 | fail |
| 104 | .275276 → .395386 | 2.012829 → .045624 | .034969 → .030046 | .102556 → .068238 | fail |
| 105 | 1.008917 → .904181 | .291733 → 2.197009 | .109202 → .115128 | .207647 → .234401 | fail |

Every start passes orientation at1000. Every start fails **each** of Q<=.05,
maximum albedo<=.02, metallic<=.03 and held-out relativeL2<=.01. There is no
partial-pass reinterpretation. Median Q error increased .576588→.697309.
The improved total-radiance fit did not imply physical material recovery.

Final physical parameters (truth: `.20,.55,.31,.30,.45,.60,.25`):

| Seed | mx | my | theta, radians | kR | kG | kB | metallic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | .658253074 | .410114020 | -1.267675757 | .345334709 | .513767004 | .682088733 | .435369372 |
| 102 | .691506863 | .413559169 | -1.255291581 | .352339685 | .523657918 | .695309997 | .456847399 |
| 103 | .694220126 | .392726690 | 1.892421126 | .347786814 | .517484248 | .686853170 | .440469593 |
| 104 | .260609448 | .647974193 | .309203714 | .316853166 | .473281085 | .630046308 | .318238050 |
| 105 | .430880249 | .732280135 | .271654963 | .363626838 | .539280593 | .715127528 | .484400749 |

Sorted minor/major roughness-axis errors, accounting for axis exchange:

| Seed | Absolute axis errors250 | Absolute axis errors1000 | Relative axis errors1000 |
| --- | --- | --- | --- |
| 101 | .200354,.088597 | .210114,.108253 | 1.050570,.196824 |
| 102 | .263197,.011960 | .213559,.141507 | 1.067796,.257285 |
| 103 | .248525,.094784 | .192727,.144220 | .963633,.262218 |
| 104 | .150049,.010642 | .060609,.097974 | .303047,.178135 |
| 105 | .339911,.142843 | .230880,.182280 | 1.154401,.331418 |

The JSON retains full precision, all old/new recovery metrics, complete seven-
coordinate terminal/Adam states and pairwise Q distances. Raw theta differences
or the gauge-dependent scalar parameter median must not replace Q comparisons.

## Generalization and radiance components

All entries below are **percent relativeL2 errors**,250→1000, using the original
independent evaluation tapes1000..1003. The fit column uses those evaluation
tapes on fitting observations; it is not the training-tape MSE.

| Seed | Fit total | Held-out total | Unseen-light total | Relight seen views | Relight unseen views |
| --- | --- | --- | --- | --- | --- |
| 101 | 4.1168 → 1.2423 | 6.6260 → 2.0530 | 8.0019 → 3.5461 | 7.0598 → 3.6969 | 8.9752 → 3.3620 |
| 102 | 2.4082 → 1.3037 | 3.8005 → 2.2073 | 5.1582 → 3.7370 | 4.5411 → 3.8705 | 5.7947 → 3.5753 |
| 103 | 1.6504 → 1.2335 | 2.7002 → 2.1162 | 4.2254 → 3.5339 | 4.0294 → 3.6980 | 4.4429 → 3.3327 |
| 104 | 2.2242 → .5434 | 2.5168 → 1.0633 | 3.3125 → 1.7285 | 3.2433 → 1.9305 | 3.3913 → 1.4585 |
| 105 | 3.5661 → 1.3307 | 3.6543 → 2.4519 | 5.8066 → 4.1702 | 6.0216 → 4.3023 | 5.5459 → 4.0108 |

The smallest held-out error is **.010632800229688513**, still above .01. It
must not be rounded into a pass. There is no all-start average replacing failures.

Component relativeL2 errors at1000, also percent:

| Seed | Fit diffuse | Fit specular | Held-out diffuse | Held-out specular | Relight diffuse | Relight specular |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 14.1576 | 51.5369 | 14.1579 | 51.6519 | 14.1577 | 51.4466 |
| 102 | 15.8153 | 57.5417 | 15.8156 | 58.3869 | 15.8154 | 57.7693 |
| 103 | 14.3309 | 52.2304 | 14.3312 | 53.0197 | 14.3310 | 52.5418 |
| 104 | 4.4266 | 17.0979 | 4.4269 | 16.8667 | 4.4267 | 16.8392 |
| 105 | 17.7375 | 65.5895 | 17.7378 | 66.5740 | 17.7376 | 65.3515 |

Full RMSE, MAE, relativeL2 and maximum absolute errors for every component and
all five splits, plus all84 RGB component predictions per state and their hashes,
are retained in JSON. Seed105 illustrates compensation: held-out diffuse error
increased13.8110%→17.7378% and specular56.2543%→66.5740%, while total error fell
3.6543%→2.4519%. Its metallic and maximum albedo errors also worsened. On fitting
observations with evaluation tapes, final diffuse/specular residual cosines range
from−.996075 to−.991486; cancellation fractions range .924721–.952953. These are
offline calculations on saved arrays, not additional shader calls. Compensation
is observed; exact global non-identifiability is not established.

## Optimization trajectories and numerical behavior

Training-tape MSE below is evaluated at the actual endpoints. Update250 values
are reused from3C-2, not its update249 pre-step log. The1000 endpoint gradient
requires one extra forward/backward per start and **no extra optimizer return**.

| Seed | MSE250 | MSE1000 | Reduction | Latent gradient norm250→1000 | Last20 logged loss reduction |
| --- | --- | --- | --- | --- | --- |
| 101 | 8.79327636e-5 | 8.29917371e-6 | 90.5619% | 2.19178902e-4 → 1.18835942e-5 | 1.2121% |
| 102 | 2.91898905e-5 | 8.89577586e-6 | 69.5245% | 8.89322312e-5 → 1.09020393e-5 | .7061% |
| 103 | 1.34295124e-5 | 7.89723526e-6 | 41.1949% | 3.18764638e-5 → 1.32000314e-5 | 1.1746% |
| 104 | 3.11546755e-5 | 1.36357301e-6 | 95.6232% | 5.46983421e-4 → 1.06896496e-5 | 1.4668% |
| 105 | 7.13120244e-5 | 1.04801375e-5 | 85.3038% | 8.47415495e-4 → 1.46960747e-5 | .4055% |

Each start has76 logged states: update251 and every tenth return260..1000.
Logged losses correspond to250,259..999. The last20 column compares losses979
and999, not980 and1000. All sampled log transitions decreased (zero increases
among75 adjacent records per seed), but unlogged steps were not measured for
monotonicity. Parameters still changed over990→1000. Gradients and states remained
finite; no numerical failure was observed. Neither finite gradients nor slowing
progress proves stationarity or a local minimum. No convergence threshold was
introduced afterward.

The accepted3C-2 **latent-roundtrip** truth state had the same finite objective
`3.0707849418831756e-7` under this parameter transform. Every1000 endpoint remains
above that known feasible value (approximately27.026,28.969,25.717,4.440,34.129
times, seed order). This reused observation requires no new reference or shader
evaluation. It excludes claiming a global finite-objective minimum at these
endpoints. It does not prove that the true parameters minimize the finite-tape
objective, nor that another optimizer will recover them.

## Scientific interpretation and next controlled experiment

**Established:** exact restored Adam continuation can substantially reduce the
same finite-tape objective and improve unseen radiance without achieving the
material criteria. Four Q estimates worsened and compensation persisted. The
duration-only intervention250→1000 produced no recovery success. It therefore
does not establish that insufficient duration was the primary cause of0/5.

**Still unresolved:** proximity to a local stationary point; slow/curved
compensation valleys versus local basins; contribution of finite-pattern
forward/gradient and branch artifacts from3C-2 at these new endpoints; global
uniqueness; attainable parameter errors at a fully minimized finite objective.
No new continuous integrals/gradients were computed, so3C-2's local findings at
update250 cannot be silently promoted to verified claims at1000. Local Jacobian
rank never established global uniqueness. No native-scene improvement is shown.

**Recommended next single comparison, requiring separate approval:** compare a
predeclared curvature-aware seven-scalar optimizer (for example bounded damped
Gauss–Newton/Levenberg–Marquardt) against continued original Adam, starting from
all five saved1000 states. Keep A2/atan2, the same fit tapes/targets/objective,
physical transforms, independent evaluation and recovery bounds fixed; change
only optimizer family. Freeze damping/acceptance rules, equal wall-work budgets
and fixed endpoints before execution, preserve unsuccessful steps/states and
never use held-out performance for selection. This would test whether a slow
compensating optimization valley explains the remaining failures, rather than
assuming still more identical Adam steps suffice. If both minimize the fit
objective yet recover poorly, a later independently approved estimator-objective
comparison becomes better motivated. Neither comparison is implemented or run
here, and no success is promised.

## Validation and preservation

- **15 new focused CPU tests passed, zero failures/skips**, final test-body
  time1.390s. Fixtures cover bitwise uninterrupted restoration, original flags,
  exact750 returns, endpoint no-step measurement, counters/shapes/nonfinite
  rejection, no restoration RNG/update, budget/time interruption, saved partial
  state, collision refusal and thread/signal cleanup.
- An earlier development test invocation had one temporary-path mocking error
  in its collision fixture; the fixture was corrected before the successful
  final tests, protocol freeze and single scientific run. No failed/retried
  scientific execution occurred.
- New Python files pass Python3.8 AST parsing. Runtime tests and this experiment
  used Python3.9.12; this is not a claim of Python3.8 runtime validation.
- Offline arithmetic independently checked all restoration values/hashes and
  counters, all75 split/component metric records (300 numerical metrics), Q,
  roughness axes, orientation, thresholds and prediction hashes. First restored
  losses exactly match all five prior terminal objectives.
- All32 old evidence files, the new frozen protocol, both frozen scientific
  snapshots, the accepted shader and previous runners/protocols remain unchanged.
  The original workspace's uncommitted audit-document edit remains byte-identical.
  Only this milestone's six files are included. No old suite, experiment, GPU,
  Docker, native artifact, package installation or further optimization was run.

New focused test command (already completed; no prior suites were run):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p test_material_recovery_continuation_v1.py -v
```
