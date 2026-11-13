# Neural material recovery logged v2 — Phase D.4 incomplete results

One authorized fresh scientific invocation was attempted from the exact frozen v2 source. The execution session became unavailable without a controller exit report or terminal seal. The frozen offline finalizer admitted only a **valid committed prefix**: `scientifically_complete=false`. No scientific retry, resume, replacement initialization, threshold change or source modification was performed. The interruption cause and actual process exit status remain unresolved.

All nine D0 and nine N0 controls completed. Two N1 endpoints completed: M1/101 failed physical recovery; M1/103 passed all five frozen criteria. M1/105 is interrupted at the last committed update 4270. All six M2/M3 N1 histories were initially admitted but their optimization was unstarted. These missing endpoints prevent the full three-material neural-versus-direct comparison.

## Source, authorization and evidence

- Frozen branch/commit: `feat/neural-material-logged-v2-freeze` / `678f3bf3230ea9e36aa97065491466a816c10310` (clean before launch and after investigation).
- Results branch: `feat/neural-material-logged-v2-results`, based directly on that commit.
- [Frozen protocol](protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.md); [freeze](protocols/NEURAL_MATERIAL_RECOVERY_LOGGED_V2.freeze.json) SHA-256 `dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af`.
- Source inventory identity: `485c0bca449ebbe5629cc7de04f57b66ecda76d057f5dff7e6a86a3d2b1135fb`; all 348 frozen source identities were rechecked.
- Configuration identity: `759b096b0d146d29bbaa606d6ad09485d13a5373b635bea6bb7664bf2f4b804c`.
- Readiness evidence identity: `e579bd36a1cbf450604d2ce6db1e609cf95541b3f1c6861663aba9f2e7d29f85`.
- Separate explicit authorization SHA-256: `50cb1a3dc8662c61f9a23c19d04ec86033ca8b8081265d2c52240eba3320e274`. The frozen `execution_authorized=false` field remains unchanged.
- Run: `3d2-recovery-v2-09026db80e59`.
- Original run root: `/home/isavand/paper/evidence/3d2-recovery-v2-09026db80e59`.
- Controller: `/home/isavand/paper/evidence/3d2-v2-control-09026db80e59` (authorization, zero-scientific-work preflights, exclusive invocation claim, actual argv/environment, console, source scripts and interruption observation).
- Separate frozen finalization output: `/home/isavand/paper/evidence/3d2-v2-finalized-09026db80e59-prefix`.
- Separate offline analysis evidence: `/home/isavand/paper/evidence/3d2-v2-results-analysis-09026db80e59`.
- [Versioned results JSON](evidence/NEURAL_MATERIAL_RECOVERY_LOGGED_V2_RESULTS.json) SHA-256 `7a9ca7ae1e930c1a90143079f83e7376ebd00ed8d0bea514bf321686ea37084e` binds all 27 statuses, complete endpoint metrics, sampled trajectories, checkpoint catalogs and original-file inventory.

Original frozen v1, C/C.1, D.1/D.2/D.3 evidence and pinned G/A snapshots were admitted before launch and remain preserved. In particular, original v1 freeze SHA-256 `730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad` and incomplete v1 report SHA-256 `4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49` are archival evidence, not results from this invocation. No historical scientific validation or reference targets were regenerated.

## Execution and interruption

The recorded scientific argv was `timeout --signal=TERM --kill-after=10s 28800s python3 -B -c <reviewed API program>`, with the exact program and authorization selection in the JSON process record. It called only `experiments.neural_material_recovery_logged_v2_execution.execute_reviewed`. An exclusive controller claim preceded the single launch. No extra invocation was launched when the session was later unavailable.

Accepted CPU runtime: Python 3.9.12, NumPy 1.21.5, Torch 2.4.0+cu121, one CPU thread, GPUs hidden. `CUDA_VISIBLE_DEVICES=""`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. A CUDA-tagged Torch installation does not imply GPU execution. All 27 native initial admissions and nine admission groups preceded optimization. All D0 histories preceded all N0 controls; N1 began only after the control gates passed.

All settings remained frozen: A2/atan2; float32; diffuse/specular samples 512/1024; 24 fitting and 84 external observations; tapes 3000–3003 for fitting and 1000–1003 for evaluation; Adam LR 0.01, betas (0.9,0.999), epsilon 1e-8, weight decay 0; no scheduler, clipping, regularization or warm start. Completed histories each returned exactly 5000 updates with snapshots at 250/1000/5000.

The process started at 2026-11-12 17:58:07.536 Mountain Standard Time (2026-11-13 00:58:07.536 UTC). The last anchor was written at 2026-11-12 20:16:03.190 MST (2026-11-13 03:16:03.190 UTC): 8275.654 seconds from launch to the last committed filesystem timestamp. This is an observed evidence window, **not a measured process termination time or duration**. The original session ID later returned `Unknown process id`. There is no `process-outcome.json`, no `execution-end.json`, no shutdown traceback, and no evidence identifying an external signal, watchdog expiry, runtime crash or platform interruption as the cause. No native status was rewritten to failed or censored.

## Read-only finalization and independent verification

- Anchor SHA-256: `44489f0aded38463aaf76e360754249be05f53e78048df0eefc4cd348b3daf06`.
- Complete existing journal SHA-256: `c2527b50c89487866878c0091343262b9706dbc6ca477e4e91de4cfef3e3678f`.
- Seal: absent, represented by null rather than fabricated evidence.
- Frozen finalizer `--prefix-only` returned zero in 68.571 seconds, classifying `valid_committed_prefix_only`.
- Finalization JSON SHA-256: `36e95cb46a7ae47ad4e1b9121c5b194a6f0da04f80927bb67499756438504413`.
- Materialized read-only report SHA-256: `bded2f341b90d3480c2c1c6107bb712d46d5a863cd1b9a8a674753c295858eb2`.
- Verified 42156 chained journal records, 10499 persistence boundaries, 157452541 committed bytes, zero uncommitted tail bytes.
- Verified all 62 selected numeric checkpoint archives and 3124 arrays using `allow_pickle=False`: tensor shapes/dtypes/content hashes, finite values, Adam counters/nonnegative second moments and RNG array identities. No model was constructed and no optimizer state was restored.
- Recomputed all 20 endpoint metric sets, success decisions and diffuse/specular compensation from saved predictions and accepted independent targets.
- Independently compared all nine D0 recorded historical prefixes and 250/1000 snapshots; all nine N0 complete states, snapshots, logged trajectories, evaluations and full runtime trajectory digests exactly match D0. A digest binds the runtime transitions; offline analysis did not rerun them.
- Matrix, original model identities, settings, targets, authorization, initial gates and exact controls pass. Whole-matrix completion, all endpoint records, all snapshots/work and shutdown-cleanup evidence do not pass because this is an incomplete prefix.

The original run files were hash-inventoried before and after offline analysis and stayed identical. Finalization/reporting wrote only to separate outputs. A preliminary unpublished reporting summary was preserved before excluding the analysis process's own in-flight log from stable input inventory; the reviewed summary adds the analyzer source hash, and every scientific result is identical to that preliminary summary. This involved no scientific execution or original-evidence changes.

## All planned histories

All errors below are update-5000 endpoint measurements. Fit is the terminal frozen fitting MSE; held/relight are external total linear-RGB relative L2; Q is relative Frobenius error; theta is identifiable major-axis error in degrees modulo physical equivalences; albedo is maximum RGB absolute error; metallic is absolute error. `—` is unavailable, never a zero or an inferred pass. Full precision, all components, axis errors, final physical parameters and pass checks are in the JSON.

| Material | Seed | Arm | Outcome | Updates | Fit MSE | Held L2 | Relight L2 | Q | Theta ° | Albedo | Metallic | Physical pass | Active s |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| M1 | 101 | D0 | complete | 5000 | 5.23854e-07 | 0.00602445 | 0.00921806 | 0.36037 | 0.381135 | 0.0174931 | 0.0500021 | no | 370.179 |
| M1 | 101 | N0 | complete | 5000 | 5.23854e-07 | 0.00602445 | 0.00921806 | 0.36037 | 0.381135 | 0.0174931 | 0.0500021 | no | 369.482 |
| M1 | 101 | N1 | complete | 5000 | 0.036231 | 0.833065 | 0.835146 | 4.72921 | — | 0.42 | 0.1 | no | 558.941 |
| M1 | 103 | D0 | complete | 5000 | 2.72245e-06 | 0.01224 | 0.0230099 | 1.28406 | 4.81895 | 0.0788116 | 0.186673 | no | 370.966 |
| M1 | 103 | N0 | complete | 5000 | 2.72245e-06 | 0.01224 | 0.0230099 | 1.28406 | 4.81895 | 0.0788116 | 0.186673 | no | 371.243 |
| M1 | 103 | N1 | complete | 5000 | 2.69916e-08 | 0.000961181 | 0.00134111 | 0.01243 | 0.00853551 | 0.000418084 | 0.00205704 | yes | 553.09 |
| M1 | 105 | D0 | complete | 5000 | 4.72152e-07 | 0.00569317 | 0.00864336 | 0.338 | 0.448999 | 0.0161865 | 0.046509 | no | 369.742 |
| M1 | 105 | N0 | complete | 5000 | 4.72152e-07 | 0.00569317 | 0.00864336 | 0.338 | 0.448999 | 0.0161865 | 0.046509 | no | 370.604 |
| M1 | 105 | N1 | interrupted prefix | 4270 | — | — | — | — | — | — | — | — | — |
| M2 | 101 | D0 | complete | 5000 | 6.03436e-07 | 0.00721332 | 0.00859963 | 0.0273777 | 0.405999 | 0.00433856 | 0.00845019 | yes | 369.524 |
| M2 | 101 | N0 | complete | 5000 | 6.03436e-07 | 0.00721332 | 0.00859963 | 0.0273777 | 0.405999 | 0.00433856 | 0.00845019 | yes | 369.495 |
| M2 | 101 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |
| M2 | 103 | D0 | complete | 5000 | 6.35925e-07 | 0.00842951 | 0.0109122 | 0.0480878 | 0.336237 | 0.00851578 | 0.0167795 | yes | 370.509 |
| M2 | 103 | N0 | complete | 5000 | 6.35925e-07 | 0.00842951 | 0.0109122 | 0.0480878 | 0.336237 | 0.00851578 | 0.0167795 | yes | 369.174 |
| M2 | 103 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |
| M2 | 105 | D0 | complete | 5000 | 3.88442e-07 | 0.00612709 | 0.0156174 | 0.0530222 | 0.126404 | 0.0027408 | 0.0012876 | no | 367.89 |
| M2 | 105 | N0 | complete | 5000 | 3.88442e-07 | 0.00612709 | 0.0156174 | 0.0530222 | 0.126404 | 0.0027408 | 0.0012876 | no | 368.983 |
| M2 | 105 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |
| M3 | 101 | D0 | complete | 5000 | 3.46032e-07 | 0.00245715 | 0.00436511 | 0.0189732 | 1.10868 | 0.00228684 | 0.00660398 | yes | 369.009 |
| M3 | 101 | N0 | complete | 5000 | 3.46032e-07 | 0.00245715 | 0.00436511 | 0.0189732 | 1.10868 | 0.00228684 | 0.00660398 | yes | 368.676 |
| M3 | 101 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |
| M3 | 103 | D0 | complete | 5000 | 3.30582e-07 | 0.00225493 | 0.00421484 | 0.0249543 | 1.06957 | 0.00173731 | 0.00153289 | yes | 368.686 |
| M3 | 103 | N0 | complete | 5000 | 3.30582e-07 | 0.00225493 | 0.00421484 | 0.0249543 | 1.06957 | 0.00173731 | 0.00153289 | yes | 368.397 |
| M3 | 103 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |
| M3 | 105 | D0 | complete | 5000 | 1.28828e-06 | 0.0109342 | 0.0201407 | 0.515899 | 5.17063 | 0.0639937 | 0.113133 | no | 369.439 |
| M3 | 105 | N0 | complete | 5000 | 1.28828e-06 | 0.0109342 | 0.0201407 | 0.515899 | 5.17063 | 0.0639937 | 0.113133 | no | 369.219 |
| M3 | 105 | N1 | unstarted | 0 | — | — | — | — | — | — | — | — | — |

The only interrupted history is M1/105/N1: raw status remains `optimizing`, with 4270 committed successful optimizer returns. Its last sampled active time is 480.236 seconds, not a final runtime. There are zero recorded native failed or censored outcomes; the interruption is separate from those categories. Unstarted M2/M3 N1 histories still incurred their approved initial admission forwards/backwards. No endpoint metrics or physical decision are imputed to any of these seven missing endpoints.

Frozen success requires all of Q≤0.05, identifiable orientation≤5°, maximum albedo error≤0.02, metallic error≤0.03, held-out total relative L2≤0.01. An isotropic recovered solution with unidentifiable orientation does not pass the identifiable-orientation requirement. No held-out-best checkpoint selection was used.

## Scientific findings from the available evidence

**Control representation.** D0 and N0 are exactly paired numerically in all nine completed histories. Each passes physical recovery 4/9 and held-out RGB 7/9. M1 passes physical recovery 0/3; M2 2/3; M3 2/3. D0 M2/105 narrowly fails Q (0.0530222>0.05) despite acceptable held-out RGB; M3/105 fails the recovery criteria. The seed-specific outcomes are all retained. Low RGB loss alone therefore remains insufficient evidence of physical recovery.

**N1 M1/101: observed optimization collapse.** The initial loss 0.020780459 rises to 0.036230952. By the first later sampled record (update 10), the physical output is `[1,1,1.178097248,0,1,0,0]`, raw rotation `[0,1]`, with all six material/shared-feature gradient groups present and finite but identically zero. All 500 logged records from update 10 through 5000 retain zero gradient; no claim is made about the exact first unlogged saturation update. The endpoint Q error is 4.72921, held-out relative L2 is 0.833065 and relighting 0.835146. This is a completed optimization history with failed physical/photometric recovery, not a nonfinite exception or censoring. Boundary saturation is observed; its causal mechanism has not been experimentally isolated.

**N1 M1/103: successful bounded recovery.** Q error 0.01243001972, orientation 0.00853550519°, maximum albedo error 0.000418083668, metallic error 0.00205704421, held-out L2 0.000961180586, relight L2 0.00134110748. The final physical vector is `[0.491379708,0.274895757,0.950647354,0.359728187,0.579581916,0.239808038,0.097942956]`. Its roughness axes are swapped relative to the truth and theta differs by approximately a quarter turn: the unchanged gauge-invariant evaluation handles this physical equivalence. All final gradient groups are present, finite and nonzero. Centered orientation radius remains nonzero (last logged 0.662187); the normalized encoding is finite.

**N1 M1/105: partial failure evidence, no endpoint claim.** Its first later sampled record (update 10) has boundary output `[1,1,-1.178097248,0,1,0,1]` and all logged gradient groups zero. All 427 later records through update 4270 remain at loss 0.024940765 with zero gradients. Raw rotation `[0,0]` still has nonzero centered radius 1.41421 and finite normalized orientation; this is not evidence of the centered-encoding singularity. The missing final evaluation prevents an endpoint physical/held-out decision. Nothing about M2/M3 neural recovery is established.

**Available paired comparisons and compensation.** For seed 101, N1 is substantially worse than paired D0 (held-out L2 0.833065 versus 0.00602445); D0 nevertheless fails Q and metallic. For seed 103, N1 is better than paired D0 (Q 0.0124300 versus 1.28406; held-out L2 0.000961181 versus 0.0122400). D0 seed 103 has held-out diffuse/specular relative errors 0.0955131/0.688085 while their residual correlation is −0.990438: strong cancellation makes total RGB look substantially better than individual components. N1 seed 103 reduces these component errors to 0.00154616/0.0104669; compensation remains present (correlation −0.731931) but at much smaller absolute residual energy. No relative-component threshold was introduced after observing these values.

These two paired neural endpoints give opposite outcomes, with the third only partial and all other materials missing. Reporting 1/2 observed endpoints is not a 1/9 success rate and not an estimate of population reliability. D0 and N1 differ in architecture, orientation encoding and redundant parameterization; none of these results isolates network depth as the cause.

## Work and resources

- Committed successful Adam returns: 104270 (20×5000 + 4270), attempts equal returns.
- Committed shader forwards: 104358; backwards: 104317; field calls: 104412; initial forwards: 27; external evaluations: 20. Attempted/completed counters match within the selected prefix.
- Committed queried points: 10023168; diffuse directions 5131862016; specular directions 10263724032. These are recorded operation counts, not wall-time measurements.
- Sum of the 20 completed histories: 7763.246 active wall seconds and 7757.416 CPU seconds. These exclude unrecorded terminal totals and do not include a measured CPU total for the interrupted history.
- Last logged active time for the partial history: 480.236 seconds. Full process CPU, exit status, duration and peak RSS are null/unavailable; configured 4-GiB RSS and 8-GiB address limits are not peak measurements.
- Completed paired N1 histories took 558.941/553.090 active seconds versus D0 370.179/370.966: ratios 1.50992/1.49094, both below the unchanged 900-second history limit.
- Original output inventory: 131 files, 305204488 logical bytes, 305438720 allocated bytes. The journal is 157452541 bytes. No caches, checkpoints or journals are committed to Git.

Zero uncommitted journal tail bytes does not rule out scientific operations after the last persist and before interruption. The recorded work counters and last-update claim refer strictly to the verified committed prefix. Shutdown RNG/thread/memory-limit restoration is not verified without the terminal record. No process resource cap or watchdog expiry is inferred merely from missing evidence.

## Validation and offline usage

Only six new focused CPU reporting tests were run: lost-session classification and endpoint refusal, completed physical failure, hash/path admission, zero-gradient/pre-update log preservation, rejection of object NPZ payloads with `allow_pickle=False`, and importing the offline analyzer without Torch. Result: **6 passed, 0 failed, 0 skipped**. No prior scientific experiment or historical test suite was repeated. Static source/JSON/link/diff checks are separate from scientific completion.

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B -m experiments.analyze_neural_material_recovery_logged_v2_results \
  --selection /home/isavand/paper/evidence/3d2-v2-control-09026db80e59/post-execution-selection.json \
  --selection-sha256 5fc33370e75fe9c19879713046f2f4e6d5efada0de9760b5faeb6de8f14b99e2 \
  --final /home/isavand/paper/evidence/3d2-v2-finalized-09026db80e59-prefix/final.json \
  --final-sha256 36e95cb46a7ae47ad4e1b9121c5b194a6f0da04f80927bb67499756438504413 \
  --output /tmp/neural-v2-results-new-offline-review.json
```

Run from the results checkout and select a fresh output filename; collisions are refused. This command reads saved data only, performs no model/optimizer restoration, and never resumes training. The [analyzer](../experiments/analyze_neural_material_recovery_logged_v2_results.py) reuses unchanged saved-data metric/control helpers, validates all frozen source identities and numeric archives, and keeps the original status separate from observed interruption. The [focused tests](../tests/infrastructure/test_neural_material_logged_v2_results.py) use only disposable synthetic fixtures.

## Recommended next review

First investigate why the execution and controller disappeared without terminal evidence, and audit the early boundary-saturation transitions offline using the accepted saved states. Preserve the negative and partial results. A future complete neural panel would require separately approved execution planning and any revised scientific protocol; do not automatically retry, resume, extend the budget or change learning rate/calibration/architecture. No next experiment, GPU work, neural retraining, reference regeneration or protocol alteration was initiated.
