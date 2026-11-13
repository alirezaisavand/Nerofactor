# E.2a preparation readiness

Prepared on `feat/neural-material-early-step-e2a` from
`06f837af623df76e194f60d475a1deb94fa4045a` in a separate worktree. No scientific
execution is authorized. Researcher review of this commit is the next action.

## Prepared design

The [protocol](protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A.md) freezes M1/101,
M1/103, M2/103, M3/101 and M3/103, each with independently reconstructed original
N1 initial states and LR .01/.001. The sole intervention is LR. Exactly ten
optimizer returns/arm are proposed: total 100, with 120 renderer forwards and
110 backwards including terminal and external measurements. No long-horizon
recovery or LR improvement has been demonstrated.

Passive hooks capture actual calibrated Sigmoid inputs/outputs, 22 ReLUs,
shared features and actual shader inputs/gradients. Rotation encoding gradients
are distinguished from disconnected diagnostic theta/radius. Actual float32
parameter displacement is auditable using initial parameters plus bitwise XOR
deltas; moment-derived proposals are not treated as observed changes. Record
pre/post state identities, Adam options/moments/counters, physical/Q diagnostics,
per-group gradient statistics and first observed boundary/zero-gradient states.

Baseline parity gates use historical label 1 = state 0 and label 10 = state 9.
Their source/initial/target/tape identities and saved associations are verified;
real shader loss/gradient parity is **pending E.2b**, not a passed E.2a result.
No fallback, threshold tuning, replacement cases or continuation is provided.

## Validation and preservation

[Machine-readable validation](evidence/NEURAL_MATERIAL_E2A_VALIDATION.json)
SHA-256: `9303e05691b6febc679f819ffc2f0ee8033830c76cf8561ec775944ce4e59eaf`.

The preserved complete JUnit report records **31 passed, 0 failed, 0 errors,
0 skipped**, in 6.597 seconds including collection. All reported test names match
the current test source and all selected sources predate that report. The process
session expired across the usage-limit interruption, so its exit code is not
claimed; the complete original JUnit XML/hash is embedded in the versioned
validation artifact. An earlier toy RGB-sum assertion was corrected to account
for its three output channels; no scientific computation was changed.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -p no:cacheprovider -q tests/scientific/test_neural_material_early_step_e2a.py --junitxml=/tmp/neural-material-e2a-reviewed-tests.xml
```

Tests use unrelated tiny engineering networks, toy Adam steps, saved metadata
and mocked resources. Session and subprocess scientific-import blockers verify
that original N1/shader/recovery modules are not imported during validation.
The metadata freezer also completes without importing Torch. **Zero scientific
model calls, renderer calls, optimizer updates or reference generation occurred.**
No positive E.2b review was created and the scientific runner was not called.

Verified all 348 files in the old source freeze, the expanded 357-file inherited
inventory, E.1 report/analysis identities, original/continuation report identities,
all 131 original and 45 continuation artifact hashes, checkpoint catalogs and
five selected saved N1 histories. Original G/A, A2/atan2, recovery harness,
reference inputs, old E.2 draft, evidence and figures remain unchanged. Only new
E.2a files are included. No training/checkpoint pickle deserialization was used.

## Resource planning

Candidate limits remain CPU/one thread/hidden GPUs, 120 active wall seconds/arm,
1800 whole seconds, ten-second pre-update reserve, 4 GiB RSS and 8 GiB address
space, with an external 1800-second watchdog and ten-second termination grace.
No retry/extension is allowed. Guards remove hooks and preserve bounded partial
evidence on interruption; hard KILL/full-disk cases cannot guarantee final JSON.

Shape-derived capture reserve is 241 MiB/arm, 2414 MiB total with root metadata,
and minimum free disk 5062524928 bytes. Read-only preparation inspection found
2662651469824 bytes free under the approved owned/writable evidence parent;
no scientific output directory was allocated. Admission rechecks capacity later.
Archive, JSON inventory and total-byte caps are enforced. Retained signal/gradient
payload is 98528 bytes/pass. Planning peak 2687902944 bytes is based on twice
the accepted prior peak plus conservative transient allowances, not a new native
memory benchmark. Prior continuation timing projects roughly 1.12 compute
seconds/ten updates; hook/checkpoint/fsync overhead remains unmeasured.

## Immutable preparation freeze

[Freeze](protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A.freeze.json) SHA-256:

`69af5785e37a0843e9a4d854ffd441ff390f5a51b98fc78044621f0f83183ed6`

The freeze binds all five new implementation/test/protocol files, inherited
identities, initial states, saved parity rows, full runtime/settings, resources,
capture/failure schemas and separately hash-selected validation. It excludes
itself and this human summary to avoid circular hashes. Exclusive creation and
external hash selection prevent silent re-freezing. Independent verification:

```bash
sha256sum docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A.freeze.json
python3 -m experiments.freeze_neural_material_early_step_e2a --verify-sha256 69af5785e37a0843e9a4d854ffd441ff390f5a51b98fc78044621f0f83183ed6
python3 -m experiments.neural_material_early_step_e2a --describe
```

Remaining gates require separately reviewed E.2b authorization binding this exact
freeze, clean commit/branch, explicit fresh output root/run ID and watchdog:
native passive-hook equality, exact historical loss/gradient parity, actual
runtime/RSS/I/O feasibility, measured onset and paired LR outcomes. The review
must check the external invocation; a JSON watchdog declaration is not an OS
attestation. No E.2b execution or 5000-update recovery is authorized by readiness.
