# 3D-2 Phase D: controlled execution, incomplete after resource censoring

**The single authorized execution stopped before N1 optimization. Phase D did
not complete scientifically.** All nine D0 histories and six N0 histories
completed. N0 M3/101 was censored after 4,860 optimizer returns; the required
control-completion guard stopped the process. Two N0 and all nine N1 optimizer
histories remain unstarted. Nothing was resumed, retried, repaired or retuned.

**A. Experiment identity and frozen configuration.** Scientific execution used
the clean `feat/neural-material-field-execution-freeze-v1` worktree at
`ffed2b075b997e7b6faf37840fdec7e1c489d8ef`. Reporting uses the separate
`feat/neural-material-field-recovery-v1` worktree/branch. The
[final protocol](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.md), original G/A
snapshots, A2/atan2 shader and every frozen source remain unchanged. Exact final
freeze SHA-256:
`730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`.
Aggregate C/C.1 SHA-256:
`f9ef3477a4189958a6e0e16067b35dade62069a11797e90958515996b6cb9a2e`.
G remains `79252178e65914ad1acbc14790b1514f276bfea0`, A remains
`e1e859779301faced46d6476691e20b546c27ae0`. Known-plane geometry does not run G.

Keep M1/M2/M3 ×101/103/105 ×D0/N0/N1, original targets/observations/tapes,
float32 CPU runtime, 512/1024 samples, continuous Adam .01/(.9,.999)/1e-8/zero
weight decay, 5,000-update endpoints, 250/1000/5000 snapshots, fixed calibration
and all original thresholds. No scheduler, clipping, regularizer, LM or target
generation occurred. The machine-readable report retains the full contract.

**B. Execution status and work.** Run ID `3d2-recovery-v1-3e3001d85600`, output:
`/home/isavand/paper/evidence/3d2-recovery-v1-3e3001d85600`.
Control files, authorization, exact command, console and process outcome:
`/home/isavand/paper/evidence/3d2-phase-d-control-3e3001d85600`.
Hash-selected authorization SHA-256:
`770d65fc218a97da932eb62f45b9d5112de0bbe5c499bb5bd9b6fcb7c89aac33`.
One scientific invocation, exit 1; raw report status `failed`, global category
`baseline_reproduction`. The underlying event is resource censoring of a
required control, **not an observed numerical reproduction mismatch**.

| Arm | Planned | Complete valid endpoints | Physical pass / eligible complete | Censored | Optimizer-unstarted |
| --- | ---: | ---: | --- | ---: | ---: |
| D0 | 9 | 9 | 4/9 | 0 | 0 |
| N0 | 9 | 6 | 2/6 | 1 | 2 |
| N1 | 9 | 0 | unavailable; no denominator | 0 | 9 |

Of 15 valid endpoints, six satisfy all five recovery criteria and nine fail
physical recovery. There are zero numerical optimization failures. All 27
models were constructed and initially admitted: optimizer-unstarted does NOT
mean no shader work. Nine N1 calibration calls and nine N1 initial shader
forwards/backwards occurred within the prescribed admission; N1 optimizer
returns and final evaluations are both zero.

| Counter | Actual attempted/completed | Planned complete experiment |
| --- | ---: | ---: |
| Adam | 79860/79860 | 135000 |
| Shader forwards | 79933/79933 | 135108 |
| Renderer backwards | 79902/79902 | 135054 |
| Field queries | 79987/79987 | 135162 |
| Calibration forwards | 9 | 9 |
| Shaded points | 7677168/7677168 | 12976848 |
| Diffuse directions | 3930710016 | 6644146176 |
| Specular directions | 7861420032 | 13288292352 |
| Initial / prefix / terminal / external forwards | 27/16/15/15 | 27/27/27/27 |

**C. Source and evidence integrity.** The read-only preflight independently
admitted C/C.1, the accepted 3D-1 reference, all 322 frozen source hashes, all 27
initial identities, targets, tapes and the exact Python 3.9.12/NumPy 1.21.5/
Torch 2.4.0+cu121 CPU runtime. Model/shader/Adam factories were blocked during
preflight; no output root or scientific work was created by preflight.
Authorization was separately recorded; the original freeze remains unauthorized.

The original 25,744,940-byte report remains unchanged, SHA-256
`4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49`.
The [bounded evidence summary](evidence/NEURAL_MATERIAL_FIELD_RECOVERY_V1.json),
SHA-256 `1f7f0d2cb9d41104eca447dc09a8f66016b7fd9e87354881763ba410526dac7e`,
records all 27 statuses,15 evaluations, exact work, source/input identities,
control-file hashes and every output-file hash. The original failed C, successful
C.1, final freeze, historical3D-1 evidence and earlier research artifacts remain
separate and preserved.

Offline verification checked 48 checkpoint manifests/NPZ archives, hashes before
`allow_pickle=False`, names/shapes/dtypes/finite values, Adam moments/counters,
RNG provenance and run/freeze/config/source binding. No checkpoint was restored
into a model, no historical pickle/checkpoint was loaded, and no extra scientific
shader, derivative, optimizer or reference call occurred during analysis.

**D. D0 historical reproduction.** All nine histories passed the actual frozen
1,000-update prefix gates and continued the SAME Adam through 5,000. Saved-data
verification separately matched 101 recorded loss/gradient pairs per history,
250/1000 parameters/moments/options/physical/objective snapshots and input
identities. It did not replay updates or independently reconstruct unarchived
per-update poststates. Original runtime guards checked those states in-process.

**E. Exact N0 controls.** Six completed controls match their D0 partners in all
501 logged records, 250/1000/5000 compact snapshots, final parameters/Adam state,
full rolling transition hash and external evaluation. The rolling digest binds
every transition at runtime; offline verification compares its saved endpoint,
not nonexistent full transition tensor archives. M3/101's 487 logged pairs and
250/1000 snapshots match; its full 5,000 trajectory/endpoint is unverified. M3/103
and M3/105 have no optimizer trajectory. No approximate parity was accepted.

**F. N1 recovery.** All nine fresh N1 constructions matched their frozen
identities and passed initial field/shader/sample/gradient gates. None received
an optimizer update or endpoint evaluation. Neural recovery accuracy,
conditioning, cost and reliability remain unresolved. Do not present this as
zero physical successes out of nine evaluated neural runs.

**G. Per-material/per-seed endpoints.** Values below are D0 at fixed update 5,000;
the completed M1/M2 N0 endpoints are exactly identical. Q is relative Frobenius
error, theta is gauge-aware degrees, k is maximum RGB albedo absolute error,
metal is absolute error, and held/relight are total RGB relative L2. Full
precision, sorted-axis errors, physical values, diffuse/specular metrics,
relighting partitions, gradients and residual compensation are in the summary.

| Material/seed | Q | theta | k | metal | held | relight | All five |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M1/101 | .360370 | .381135 | .017493 | .050002 | .006024 | .009218 | fail |
| M1/103 | 1.284056 | 4.818952 | .078812 | .186673 | .012240 | .023010 | fail |
| M1/105 | .338000 | .448999 | .016186 | .046509 | .005693 | .008643 | fail |
| M2/101 | .027378 | .405999 | .004339 | .008450 | .007213 | .008600 | pass |
| M2/103 | .048088 | .336237 | .008516 | .016779 | .008430 | .010912 | pass |
| M2/105 | .053022 | .126404 | .002741 | .001288 | .006127 | .015617 | fail |
| M3/101 | .018973 | 1.108684 | .002287 | .006604 | .002457 | .004365 | pass |
| M3/103 | .024954 | 1.069566 | .001737 | .001533 | .002255 | .004215 | pass |
| M3/105 | .515899 | 5.170629 | .063994 | .113133 | .010934 | .020141 | fail |

**H. Physical versus radiometric outcomes.** D0 passes physical recovery in
M1 0/3, M2 2/3 and M3 2/3. Seven of nine D0 endpoints pass held-out total RGB,
versus only four physical passes. N0 has five held-out passes among six eligible
endpoints but two physical passes; its M3 outcomes are missing. M2/105's
Q = .053022 exceeds the unchanged .05 threshold. Relighting is a separate outcome,
not an invented sixth success criterion. For example M1/101 fits held-out total
RGB at .006024 while Q error = .360370 and metallic error = .050002 fail; component
compensation must not be described as correct BRDF recovery.

**I. Runtime and storage.** External watchdog command was
`timeout --signal=TERM --kill-after=10s 28800s python3 -B ...`;
its exact hash-selected argv/environment are preserved in `process-start.json`.
It did not expire. External wall time 10427.458013s (2h53m47.458s), CPU 10426.711823s;
the internal protected runtime reports 10402.655301s and CPU 10401.831446s.
Peak RSS 1029165056 bytes, below 4 GiB; the 8 GiB address limit and original thread/
memory settings were restored. Global Torch/NumPy RNG remained unchanged.

Completed D0 active times 386.978–660.802s; completed N0 times 699.846–884.037s.
M3/101 N0 stopped at 890.830474s: 900 seconds minus the frozen 10-second endpoint
reserve is the pre-update stopping boundary. It saved a failure-only state at
4860 (finite parameter/moment values, Adam counter 4860), with .002548s recorded
failure-state persistence. It did not take update 4861 or run endpoint evaluation.
No grace period was used for extra optimization.

The output root contains 99 files /28,364,602 bytes, including 48 numeric archives
totaling 437,856 bytes. No N1 optimizer checkpoint exists. Neural weights were
not committed; the raw 25.7 MB report remains outside Git. Numeric archives,
partial states and all unsuccessful endpoints remain in their original owned
run directory; no deletion or cleanup occurred.

**J. Failures, validation and limitations.** The source guard behaved as frozen:
an incomplete required N0 control prevented all N1 optimization. Increasing
per-history active times coincide with growing cumulative report history.
Source inspection shows full-report JSON persistence every ten updates inside
the active clock; it includes previous histories. Reporting overhead is therefore
a plausible contributor, not a separately profiled causal measurement. These
order-confounded D0/N0 timings do not establish representation-only compute cost.
The short C.1 feasibility estimates did not guarantee this complete workload.

Six NEW offline-only regression tests passed; 0 failures/0 skips. Tests cover
hash-before-load, strict JSON/path containment, pickle-free numeric rejection,
unchanged simultaneous thresholds, incomplete denominators and import guards
blocking Torch/scientific execution. Earlier scientific suites/experiments were
not rerun. Additional static checks verify ASTs, links, whitespace, exact file
scope, freeze/source preservation and immutable original evidence.

**K. Supported scientific conclusions.** H1 is supported for six complete paired
controls and their observed exact trajectories; it remains incomplete for M3.
H2's initial neural admission passed, but does not establish recovery. H3's
D0-versus-N1 recovery question is unanswered. H4's distinction between physical
and RGB recovery is directly visible in D0. H5's complete representation-cost
comparison is blocked and timings are confounded by execution order/reporting.
Do not infer neural failure/success, architecture-only effects, full-object
reconstruction quality, population reliability or statistical noninferiority.

**L. Smallest safe next step.** Review the resource/reporting bottleneck using
saved data and a separately versioned reporting-only proposal, for example
append-only histories with bounded summary updates while retaining every record,
state hash, work counter, failure rule and scientific setting. Demonstrate its
integrity/cost without shader work before considering a new freeze and ONE
fresh separately authorized controlled experiment. Do not enlarge the budget,
resume 4860, skip controls, retry seeds or launch another experiment now.

Read-only verification can be repeated from this results worktree into a NEW
summary filename; it cannot execute the scientific experiment:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m experiments.analyze_neural_material_recovery_v1 \
  --report /home/isavand/paper/evidence/3d2-recovery-v1-3e3001d85600/report.json \
  --report-sha256 4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49 \
  --control /home/isavand/paper/evidence/3d2-phase-d-control-3e3001d85600 \
  --output /tmp/3d2-phase-d-offline-review.json
```
