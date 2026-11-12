# 3D-2 Phase C.1: validation completion

**Status: PASSED.** The original27 initial shader/backward fixtures were
independently reverified from saved arrays, not repeated. All eight missing
replays, nine gauge forwards and three synthetic optimizer/checkpoint fixtures
completed. Combined Phase C/C.1 numerical admission and the predeclared bounded
feasibility test pass. This does not demonstrate material recovery or authorize
Phase D. No final protocol freeze was created.

The [C.1 evidence](evidence/NEURAL_MATERIAL_FIELD_VALIDATION_COMPLETION_V1.json)
separates old/reverified, previously captured replay, newly executed and startup
failure evidence. The [original failed report](evidence/NEURAL_MATERIAL_FIELD_VALIDATION_V1.json)
and [original results](NEURAL_MATERIAL_FIELD_VALIDATION_V1_RESULTS.md) remain
byte-for-byte unchanged and retain status `failed`.

## Reporting correction and changed files

Worktree `Nerofactor-3d2-validation-completion`, branch
`feat/neural-material-field-validation-completion-v1`, base
`42c8f4b08a0db11a47413f1007fb504f6e995378`.

The failed `dict(material=m,...,**info)` collided with `info['material']`.
`replay_row()` now stores identity as `material_name` and preserves the original
diagnostics under `material`. It rejects any other metadata/diagnostic collision.
A regression exercises the complete path: actual synthetic capture metadata,
saved numeric archive, row construction and strict report serialization.

AST admission proves that only this helper and one call site changed in
`experiments/validate_neural_material_recovery_v1.py`. All other definitions,
including every numerical calculation and resource policy, are unchanged.
Three similar `**info` constructions were reviewed; remaining explicit keys
are disjoint from captured diagnostics. No frozen G/A, shader, Phase B harness,
previous tests, protocols, historical evidence or AGENTS.md change occurred.

Files added: the bounded completion runner, focused regression tests, pre-execution
[completion protocol](protocols/NEURAL_MATERIAL_FIELD_VALIDATION_COMPLETION_V1.md),
versioned JSON evidence and this results document. The only existing file changed
is the reporting helper/call site above.

## Previous evidence independently admitted

All677 other pre-existing repository/user files and all34 old owned-evidence
files retain their preservation hashes. The old runner remains available with
its exact original bytes in Git and the original Phase C worktree.

Pinned inputs:

- Old versioned JSON SHA-256:
  `03bb86b7c3300839cebad5313a22b50e7169e5e7b81720eef7de0446b52705bf`.
- Original owned report SHA-256:
  `a5dff2d22167c00ba1cd144c81b9dbce49f6903f26c9197854d6b118b877d9e5`.
- Previously captured M1/101/D0 replay SHA-256:
  `f3b7d45f2fb61445ad5c74b682dde419513fc025645a24692f1314393b073603`.
- Material A `e1e859779301faced46d6476691e20b546c27ae0`; geometry G
  `79252178e65914ad1acbc14790b1514f276bfea0`; accepted3D-1
  `ccb03d0ec5adc814775538b9cc7427cc57480a98`.

The old source/test/protocol Git blobs, original pre-execution/process manifests,
console and106 inherited identities were verified. All29 old numeric archives
(inputs,27 initial captures and one replay) passed whole-file SHA-256 checks
before `np.load(...,allow_pickle=False)`, catalogue, shape, dtype and finite checks.
Counters exactly match the original28 forwards/backwards,21 field derivatives
and zero Adam state updates before the reporting failure.

Reverification recomputed four-tape means and original float32 MSE, camera/normal
and target identities, sampling masks/counts, actual tangent-frame orthonormality,
material bounds, source-backed parameter/gradient shapes, whole-group finite/
nonzero gradients, neural head normalization/calibration and shader encoding.
It recalculated all nine exact D0/N0 and bounded D0/N1 sample/component/Q/
gamma75 decisions and compared complete diagnostics with the old report.
Rank7 was recomputed from each saved7×8 reachability matrix, without new VJPs.
The old unregistered replay again matched every initial array bitwise.
Success flags alone were never sufficient. No initial model forward/backward,
source-audit numerical probe or target regeneration was repeated.

## Exact new and aggregate work

| Operation | New attempted/completed | Combined attempted/completed |
| --- | ---: | ---: |
| Scientific shader forwards |17 /17|45 /45|
| Renderer backwards |8 /8|36 /36|
| Field-only derivative probes |3 /3|24 /24|
| Synthetic Adam steps |3 /3|3 /3|
| Explicit field queries |20 /20|63 /63|
| D0 / N0 / N1 constructions |12 /4 /4|25 /13 /16|
| N1 calibration |4 /4|16 /16|
| Recovery optimizer updates |0|0|

All original additive work caps are met exactly. New points/diffuse/specular
directions are1632/835584/1671168; combined4320/2211840/4423680. No numeric
fixture was retried or replaced; no warmup calls, LM, external recovery
evaluation, reference generation, GPU, Docker or long training occurred.

One startup was rejected BEFORE allocation or scientific work because its
proposed ID exceeded the existing64-character limit. Its console and zero-work
failure manifest are preserved. A shorter fresh ID was validated using the
existing guard; sources/settings were unchanged. There was then exactly one
C.1 scientific invocation. This input rejection is not omitted or counted as
a scientific forward/derivative attempt.

## Replay and gauge results

All eight NEW replay fixtures match their original initial arrays, loss,
shader encoding/parameter gradients and initial model identities exactly.
Together with the separately admitted M1/D0 replay, replay passes9/9. Replay
source identities and fixed tape hashes stayed unchanged.

All nine gauge fixtures pass their applicable unchanged gates:

| Seed | Periodic theta+pi | Uncoupled swap | Signed coupled swap | Coupling sign |
| --- | --- | --- | --- | ---: |
|101|pass|Q invariant; RGB differs diagnostically|pass|+1|
|103|pass|Q invariant; RGB differs diagnostically|pass|−1|
|105|pass|Q invariant; RGB differs diagnostically|pass|+1|

Worst coupled half-vector difference5.364418029785156e-7, outgoing direction
8.195638656616211e-7 and angular difference1.0008710880386617e-6 rad remain
inside the original vector/angular bounds. Coupled mean specular/total RGB
differences are at most2.9802322387695312e-8/5.960464477539063e-8. Diffuse
directions/RGB and validity masks remain exact. Periodic101/103 samples/RGB
are exact;105 retains bounded rounding differences. Full worst indices,
relative differences, gamma75 and component checks are in JSON/raw archives.

Uncoupled mean-component maximum absolute differences are0.005541205406188965,
0.003347255289554596 and0.0011940784752368927 for101/103/105. These retained
fixed-Monte-Carlo discrepancies are not violations of physical gauge invariance.
Signed sample coupling restores bounded correspondence. Coupled PDF absolute
differences reach0.0697021484375; they remain diagnostics under the inherited
policy, not a newly introduced equality gate. No estimator or decoder changed.

## Optimizer, checkpoint and feasibility measurements

Each arm performed exactly ONE target-independent synthetic Adam update from
seed101, using the original optimizer options and squared physical-output loss.
These are disposable state fixtures, not recovery starts. All three states
restored exactly after clearing their own parameters/optimizer state, using
explicit manifest hashes and pickle-free NPZ. Nothing historical was loaded.

| Quantity | D0 | N0 | N1 |
| --- | ---: | ---: | ---: |
| Parameter bytes |28|28|5942336|
| Adam state bytes |60|60|11885008|
| Archive bytes |9122|9122|17919730|
| Synthetic Adam seconds |0.000666393|0.000325248|0.021789357|
| Serialization seconds |0.001766488|0.001523476|0.081177177|
| Hash/restore seconds |0.002639880|0.002452876|0.142602695|
| Estimated5000-update active seconds |416.3988|547.0431|608.6132|

N1 real forward median/max0.0491641/0.0571764s; backward median/max
0.0403267/0.0426882s, using12 initial/replay observations. The predeclared
maximum-observed forward+backward+synthetic-Adam estimate, plus checkpoint/
endpoint costs, is608.6131639629602s versus the unchanged900-second cap.
All three arms pass this bounded feasibility criterion. Minimum numeric
checkpoint storage for27 histories and three snapshots is484325298 bytes,
excluding growing logs/reports and additional evidence.

This supports feasibility, not guaranteed completion. Only one synthetic Adam
sample per arm was authorized. Full repeated reporting, prefix/trajectory
checks, allocator/OS variability and long-run costs were not measured. Do not
infer successful material recovery or relax a timeout from this estimate.

New internal validation elapsed8.82310259155929s; external process including
imports10.51826703734696s; peak RSS618209280 bytes. Original plus completion
scientific process time was20.896912695840002s, below the shared300-second
ceiling; the startup rejection adds1.522629488259554s with zero scientific work.
New external cap289 seconds was stricter than a fresh300-second allowance.
CPU one thread, hidden GPUs,4GiB RSS/8GiB address limits and all scientific
settings were retained. Global RNG stayed unchanged; owned runtime state was
restored. All20 new NPZ archives (17 captures/three state archives) were checked
offline by hashes, shapes/dtypes, finite values and safe loading. No extra
scientific calls were made for offline reporting.

## New evidence, tests and next decision

Owned run: `neural-field-c1-be4f67bc7d3640c3b23b62dd903cd169` under
`/home/isavand/paper/evidence/`. Original C.1 report SHA-256:
`9289801a3f7fc2120525354dcb09ee6760df18b4877fe24c63d03b81f1ac3901`.
It remains unmodified. Its pre-execution manifest binds runner/test/protocol
hashes; the new versioned summary includes full process/console identity,
startup rejection, old revalidation, new results and all archive references.
Large numeric arrays stay outside Git.

New focused tests: **9 passed,0 failed,0 skipped**,3.564s. Tests cover full
replay-row assembly/persistence, collision failure, metadata-only AST scope,
rejection of a changed scientific threshold, exact remainder/caps, independent
saved-array failure detection, original parameter topology and hash-before-load.
Only synthetic shader fixtures ran in tests; older suites were not rerun.
Static/source/JSON/documentation and preservation checks are reported separately.

All prescribed Phase C numerical gates NOW pass across the immutable old
evidence and this new completion. Recommend **preparing the final protocol for
review**, retaining the candidate5000 updates/900 seconds and their uncertainty.
Before freezing, explicitly review admission of this hash-bound C.1 aggregate:
the Phase B execution-review helper currently expects the original Phase C
protocol string, so it must not silently substitute the old failed report.
Retain the previously documented initial-admission-helper/negative-evidence
limitations in that integration review. These helpers were not changed here.

No1000-step D0 prefix or5000-step N0 paired trajectory/recovery run was executed;
those remain actual-execution gates under the future protocol. No final freeze,
Phase D authorization, experiment launch or merge is part of this milestone.
