# 3D-2 neural material field recovery v1: final scientific protocol

This is the final protocol prepared in **Phase C.2**, based on
`d331eda7837c620666e3ea5a9c7b7cfd27ad2dd8`. Its separate
`NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json` binds exact sources, tests,
configuration, initial identities, targets and aggregate validation. Neither
file authorizes execution. A separate explicit researcher Phase D instruction,
hash-selected review and approved output selection are mandatory. No recovery
updates, shader calls, reference generation or optimizer steps occur in C.2.
The [Phase A draft](NEURAL_MATERIAL_FIELD_RECOVERY_V1_DRAFT.md) stays unchanged.

## Research question and hypotheses

On one known plane with known lighting and independent accepted radiance
references, does an original-A-derived constant neural material field recover
seven physical material parameters as reliably as matched direct Adam controls?
H1: N0 is an exact identity/optimizer control for D0. H2: calibrated N1 has
constant, admitted initial material/radiance and connected original networks.
H3: at a fixed5000-update budget, representation/encoding changes conditioning
and physical recovery. H4: low total RGB error may coexist with component or
physical errors. H5: equal-update compute cost differs materially by arm.
Report supported, contradicted and unresolved findings, preserving failures.

This compares architecture PLUS orientation encoding and redundant weight
parameterization, not architecture alone. Three selected materials and three
paired starts cannot establish population reliability, statistical
noninferiority, full neural inverse-rendering generalization or native-A parity.
If all arms fail physical recovery, joint failure does not establish reliability.

## Authority, validation and immutable evidence

Geometry G is `79252178e65914ad1acbc14790b1514f276bfea0`; material A is
`e1e859779301faced46d6476691e20b546c27ae0`. Their distinct snapshot manifests
and actual bytes remain admitted. Known-plane geometry does not execute G.
Accepted3D-1 is `ccb03d0ec5adc814775538b9cc7427cc57480a98`; Phase A/B are
`a1f15d1030cb954deddd18fb408de39a452fd441` /
`a11efd615d18ac69fa1c26a3692af511f1c78302`.

The original Phase C report at `42c8f4b08a0db11a47413f1007fb504f6e995378`
remains **failed** with the reporting-key TypeError. It completed27 initial
forwards/backwards and one replay. C.1 independently reverified its29 numeric
archives and completed eight replays, nine gauge forwards and three synthetic
state fixtures. The two separately hash-selected versioned reports, both owned
reports, source epochs and all49 numeric artifacts are admitted through
`NEURAL_MATERIAL_FIELD_VALIDATION_AGGREGATE_V1.json`, not arbitrary success flags.
The aggregate recalculates sample/component/arithmetic/replay/gauge/state
checks and exact45 forwards/36 renderer backwards/24 field derivatives/three
synthetic steps. No validation recovery updates occurred. A zero-work C.1
startup ID rejection is retained separately, not a retried numerical fixture.

C.2 preserves those reports and protocols. Its narrowly scoped harness changes
add complete exact D0/N0 admission, same-return failure evidence, missing-prefix
rejection, output/authorization guards, checkpoint provenance and precise
control/failure reporting. Field/shader, loss, derivative equations, optimizer
transitions, calibration, sampling and accepted thresholds remain unchanged.
The source inventory binds the strengthened harness separately from its
historical validation epoch. Final readiness tests use only saved data and
synthetic fixtures; no new real-shader admission is claimed for these guards.

## Exactly27 histories and original known scene

Order `[mx,my,theta,kR,kG,kB,metallic]`:

| Material | Physical truth |
| --- | --- |
| M1 | `[.28,.49,-.62,.36,.58,.24,.10]` |
| M2 | `[.17,.62,.91,.56,.31,.18,.65]` |
| M3 | `[.37,.47,.19,.24,.43,.70,.35]` |

Exactly3 materials × seeds101,103,105 × D0,N0,N1 =27 histories. Reuse original
fresh latent draws, not successful LM endpoints. D0 uses seven leaf coordinates;
N0 has one identical seven-element Parameter. Six logits plus unbounded scalar
theta retain the original physical transform and operation order.

N1 retains the original eight-layer256-wide weight-normalized trunk,51-channel
positional encoding,307-wide skip and five259-input heads, construction order
mx/my/metallic/rotation/kd. Exactly1485584 parameters in84 tensors, eight raw
outputs/seven physical DOFs. Both trunk coordinates and raw head coordinates
are `(0,0,0)`. No geometry, lighting or learned position parameters.
Private CPU weight seeds10101,10103,10105 are reused across materials. Fixed
pre-sigmoid centering keeps original random nonzero weights/gradients; six scalar
logits match D0; orientation uses rho=.5, original two-channel sigmoid, `2q-1`
and `normalize(eps=1e-6)`. Actual encoding is passed directly into the shader.
Diagnostic half-atan2 theta is never recoded for N1 rendering. No calibration
tuning, zero-weight fallback, radius penalty or material-truth input.

Keep original known origin/+Z normal, tangent+Y/bitangent−X,12 views/seven
lights,24 fitting/24 held-out/36 relighting observations and their order.
Targets are the accepted3D-1 saved arrays with explicit own-truth association;
do not regenerate quadrature or local identifiability checks. Training tapes
3000–3003 and external tapes1000–1003 retain original draw order, dtype, sample
order and hashes. Global RNG must not be consumed; its unspecified absolute
initial state is not a scientific seed. No redraw or dynamic sample coupling.

## Numerical contract and initial gates

Every arm uses `MaterialShaderV1('A2','atan2')`, accepted CPU float32 runtime,
512 diffuse/1024 specular samples, original Fresnel/PDF/masks/clamps/tangent
conventions and `is_train=True` taped azimuth behavior even for external
evaluation. A2's approved estimator/decoder differs from frozen native A; no
new BRDF or GGX/Smith substitution is introduced. Loss is four-tape mean linear
RGB FIRST, then original fitting MSE. No sRGB loss or native regularizers.

Construct/admit ALL27 frozen initial identities before any update. Check two
nine-position field probes per arm and each integrated fit forward/backward.
D0/N0: exact latent/material/encoding, full samples/PDFs/masks/frames,
every-tape/mean components, loss, latent and actual-encoding gradients.
N1: exact six scalars and frozen calibration buffers; Q
`2e-7+2e-6*abs(Q0)`; actual frame absolute2e-6; h
`2e-6+2e-5*abs(h0)`; outgoing directions `4e-6+2e-5*abs(wi0)`;
angles<=2e-5 rad; mask XOR0; diffuse directions/RGB exact; every-tape AND mean
specular/total RGB `2e-6+2e-5*abs(D0)`; exact original mean/MSE replay and
unchanged gamma75 arithmetic envelope/assumptions. Old scalar objective bound
is warning-only. PDF changes remain diagnostic except exact D0/N0 identity
and finite values. Whole head/trunk and actual shader-encoding gradients must
be finite/connected/nonzero at initialization; individual structural zeros are
allowed. Invalid zero/near-zero encoding fails with no repair.

## Continuous Adam and control reproduction

Exactly5000 continuous successful finite-state Adam returns per complete history:
lr=.01, betas=(.9,.999), eps=1e-8, weight_decay=0 and accepted effective runtime
defaults. No restart at250/1000, scheduler, gradient clipping, regularizer, LM,
warm start, extra start or held-out optimizer/checkpoint selection.

Run all nine D0 histories first. Compare archived first/every-tenth PRE-update
loss/native gradient norm and POST-update state, snapshots250/1000 including
moments/options/counters and post1000 fitting objective exactly. Missing or
mismatched records stop globally before neural runs. Continue the SAME Adam
from1001 to5000. No1000-step reproduction is executed during C.2.

All N0 controls then match their paired D0 logged values/snapshots and final
state exactly. A rolling SHA-256 includes EVERY update's loss, gradient,
parameters and Adam state; terminal equality detects divergence even between
logged checkpoints or if a state later reconverges. Incomplete N0 controls
stop before N1. Only after all exact controls complete do the nine N1 runs begin.
No5000-step trajectory comparison is claimed from preparation tests.

Post250 objective reuses update251's normal forward. Post1000 adds one counted
no-grad fitting measurement. Post5000 adds one terminal fitting forward/
backward without update, then one84-observation external evaluation. Log physical
quantities from existing forwards as PRE-update values; no extra neural queries.

## Evaluation and interpretation

At the fixed final5000 state, require simultaneously: Q-relative<=.05;
identifiable orientation<=5 degrees; max RGB albedo absolute<=.02;
metallic absolute<=.03; held-out total RGB relative L2<=.01. Nonidentifiable
orientation is null/reason, not a pass or artificial zero. Preserve sorted
axes, physical/encoding/radius and gauge diagnostics, fit/held-out/relighting
diffuse/specular/total errors, seen/unseen-view relighting, residual compensation,
gradient/optimizer trajectories and all unsuccessful outcomes. Low image/RGB
error is not proof of physical BRDF recovery. No endpoint selection by held-out
performance; archived3D-1 Adam→LM results are separate optimizer context.

## Work, resources and negative evidence

Per history:5000 optimizer attempts/returns,5003 fitting shader forwards,
one external forward,5002 renderer backwards,480624 shaded points,
246079488 diffuse/492158976 specular directions. Two field probes add two
queries; N1 construction adds one separately counted calibration. Complete
totals:135000 Adam returns,135108 shader forwards,135054 backwards,
12976848 points,6644146176 diffuse/13288292352 specular directions.
Attempts and valid pipeline completions are separate; all failed attempts count.

CPU one thread, GPUs hidden;28800 whole-process seconds;900 accumulated active
seconds/history with10-second endpoint reserve;4GiB RSS/8GiB address backstop.
Admission/setup share whole-run time; optimizer work, checkpoint I/O and endpoint
evaluation share each history's active budget. The internal timer starts after
imports/admission/allocation, so an EXTERNAL watchdog is mandatory, enclosing
the entire Python process including startup, setup and teardown. The10-second
kill grace is only for failure persistence, never extra optimizer time.
Measured416.4/547.0/608.6-second estimates are not observed5000-step runtimes.
No budget extension, retry, GPU fallback or replacement after failure/censoring.

| Failure/outcome | Policy |
| --- | --- |
| Source/evidence integrity | Reject before allocation when possible; otherwise global stop |
| Initialization or renderer/sample admission | Global stop before updates; save same-call arrays/parameters/gradients |
| D0 reproduction or exact N0 control | Global stop; no admitted neural comparison |
| N1 numerical optimization failure | Stop arm, save failure-only numeric state, continue remaining admitted N1 arms |
| Active resource limit | Censored arm with latest state; incomplete controls stop neural work |
| Whole-process resource limit | Globally censored; retain partial state under watchdog grace |
| Numerical endpoint evaluation | Failed evaluation; retain same-call returns/state; no retry |
| Unknown evaluation/integrity exception | Global failed result with diagnostic/state |
| Incomplete27-history matrix | `incomplete`/failed/censored, never complete recovery |
| Complete valid endpoint | `complete_pass` or `complete_fail` under all five criteria |

Failure diagnostics detach already-produced tensors, never rerender. Nonfinite
arrays may be preserved with finite flags in failure-only NPZ; strict JSON stays
finite. Checkpoint callbacks preserve partial state on censoring or failure,
including before the first scheduled snapshot. Hard kill, disk exhaustion or
process failure can prevent final persistence; existing owned partial artifacts
remain evidence and must never be cleaned up or relabeled complete.
Failure-only serialization is teardown under the whole/external watchdog,
recorded separately as persistence time; it never grants more optimizer time
after an active cap. Successful snapshots remain charged to the active clock.

## Outputs, checkpoint schema and authorization

The researcher approved separate `3d2-recovery-*` experiments directly under
`/home/isavand/paper/evidence`. Bind the exact parent in the freeze; do not use
the currently unwritable `paper_runs`, change permissions or mix a Phase D run
with diagnostic evidence. C.2 only inspects parent ownership/free capacity,
allocating NO Phase D directory. Each later authorization must select the exact
owned, unaliased, empty/new child and unique run ID with that prefix. Reject
traversal, symlinks, collisions, arbitrary parents and implicit resume.

Check actual free capacity before allocation against the measured minimum
484325298 numeric checkpoint bytes; logs/failure artifacts require additional
space. Recheck ownership/path/capacity at execution. No historical checkpoint
deserialization. Own states are numeric NPZ plus strict JSON, per-array/whole-file
SHA-256, names/shapes/dtypes, immutable buffers, Adam moments/counters, RNG
provenance and explicit run/freeze/config/source identities. Verify hashes
before `allow_pickle=False` loading. Large weights remain outside Git.
Failure-only states are not resumable; the experiment has no resume route.

The review API requires phase D, explicit scientific authorization, exact
freeze path/hash, watchdog contract and exact approved root/run ID. The freeze
always records `execution_authorized=false`. No CLI execution bypass exists.
Example for a LATER separately authorized invocation; C.2 never runs it:

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${PHASE_D_REVIEW:?explicit researcher review JSON required}"
: "${PHASE_D_REVIEW_SHA256:?exact independently reviewed hash required}"
timeout --signal=TERM --kill-after=10s 28800s python3 -B - <<'PY'
import os
from infrastructure.signatures import read_report
from experiments.neural_material_recovery_v1 import execute_reviewed
review,_=read_report(os.environ['PHASE_D_REVIEW'],os.environ['PHASE_D_REVIEW_SHA256'])
execute_reviewed(review,review['approved_output_root'],review['approved_run_id'])
PY
```

## Deterministic freeze and required analysis

Hash sorted explicit source/test/protocol paths plus original snapshot bytes.
The aggregate is a separately hash-selected DAG dependency, recomputed from
both immutable validation epochs. Exclude the final freeze itself, aggregate
JSON and readiness report from the source inventory; the freeze separately
binds the aggregate hash and a caller binds the final freeze hash. No circular
self/commit hash. A new scientific/configuration/source change requires review
and a new explicit freeze; never silently regenerate this one.

Future analysis uses saved data only: every27 status/criterion row, paired
representation comparisons, initial/1000/final physical values, Q/orientation/
albedo/metallicity and held-out/relighting components, compensation, gradients,
radius/encoding/conditioning, exact control reproduction and cost/storage tables.
Retain all unsuccessful, censored, unstarted and interrupted cases with reasons.
No optimistic denominator, interpolated checkpoint selection or population claim.
Execution readiness is preparation only; stop after C.2 for separate review.
