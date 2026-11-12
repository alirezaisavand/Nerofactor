# 3D-2 Phase C.2: final protocol and execution readiness

**Decision: READY for separate Phase D review/authorization. Execution remains
closed.** The final protocol and reproducible freeze were created and verified.
No Phase D directory, model, shader, optimizer, derivative, reference render or
recovery history was executed during actual freeze preparation.

## A. Repository and scientific source identity

Dedicated worktree `Nerofactor-3d2-freeze`, branch
`feat/neural-material-field-execution-freeze-v1`, from approved C.1
`d331eda7837c620666e3ea5a9c7b7cfd27ad2dd8`. Other worktrees/user changes remain
untouched. All682 prior repository/user files except the explicitly corrected
experimental harness retain their preservation hashes; all62 previously
inventoried owned-evidence files remain unchanged. AGENTS.md remains unchanged;
the researcher's explicit C.2 task supersedes its historical authorization text.

G remains `79252178e65914ad1acbc14790b1514f276bfea0`; A remains
`e1e859779301faced46d6476691e20b546c27ae0`. The original snapshot manifests and
bytes, experimental A2/atan2 shader, field/calibration implementation, native
code/configurations and historical3D-1 targets/results are preserved. The
freeze records322 source/test/protocol/evidence/snapshot SHA-256 entries.

## B. Explicit C/C.1 aggregate admission

The new [aggregate](evidence/NEURAL_MATERIAL_FIELD_VALIDATION_AGGREGATE_V1.json)
verifies both immutable versioned reports, original owned reports, all49 numeric
artifacts (29 earlier captures/inputs/replay,17 completion captures, three own
synthetic state archives), metadata/console manifests and the zero-work startup
rejection. SHA-256 precedes safe numeric loading; shape/dtype/finiteness,
parameter/moment/counter catalogues and diagnostics are checked independently.
The9 initial exact/bounded comparisons, saved mean/MSE/gradient checks,9 replays,
9 gauge cases and3 archived state-fixture outcomes complete the original work
matrix:45 forwards,36 renderer backwards,24 field derivatives,3 synthetic steps,
ZERO recovery updates. No missing/duplicate/substituted fixture is accepted.

The original C report remains FAILED with its TypeError and original SHA
`a5dff2d22167c00ba1cd144c81b9dbce49f6903f26c9197854d6b118b877d9e5`.
C.1 owned report SHA remains
`9289801a3f7fc2120525354dcb09ee6760df18b4877fe24c63d03b81f1ac3901`.
Round-trip assertions are admitted from the hash-pinned C.1 report/executed
source and their referenced state bytes; they are not rerun as model restoration
in C.2. Old/new harness epochs are explicit, not claimed byte-equivalent.

The aggregate reuses the admitted C.1 numeric verifier in a private function
namespace, adapting ONLY two source-admission predicates to verify historical
Git blobs separately from the current C.2 harness. Exact AST selectors and
source hashes guard that adaptation. No scientific comparison, global module
state, source digest or original report is silently substituted.

## C–E. Final matrix, configuration and representation identities

Exactly M1/M2/M3 ×101/103/105 ×D0/N0/N1 =27 histories. Native geometry/lighting
are not learned. Keep the original plane/lights/24 fitting/84 external
observations, own saved targets, training tapes3000–3003 and evaluation1000–1003.
D0/N0 have exactly seven coordinates; N1 retains1485584 original-A-derived
parameters, constant anchor, eight-channel/seven-DOF material encoding, private
weight seeds10101/10103/10105 and fixed pre-sigmoid calibration/rho=.5.

Frozen: A2/atan2, accepted CPU float32,512 diffuse/1024 specular samples,
four-tape mean RGB then MSE, continuous Adam lr=.01/betas(.9,.999)/eps1e-8/
weight_decay0, all verified effective optimizer flags,5000 returns and
snapshots250/1000/5000. No scheduler/clipping/regularizer/LM/warm start or
held-out selection. Five simultaneous original recovery criteria remain intact.

The freeze includes all27 exact initial model/parameter/buffer identity records,
own target hashes, archived prefix identities and original source lineage.
Actual freshly constructed models must match those records at execution; C.2
does not construct or shade them. Initial identity catalogue SHA-256:
`87eba04043638f0d6781bb0a928f2b6b09b8bed26689106cf428510488db6723`.
Configuration SHA-256:
`d2851b97c9855ae4879c11629bc133a318573f5e17ffd4a4fed0ab5b5bb8f83e`.

## F. Strengthened admission and control gates

The experimental harness now enforces exact D0/N0 latent/material/encoding,
samples/PDFs/masks/frames, per-tape and mean components, losses and actual
encoding/parameter gradients. Original N1 bounded Q/frame/sample/component/
gamma75 gates are unchanged. Retaining nonleaf encoding gradients is
observational; synthetic tests demonstrate unchanged values/parameter gradients.

Missing archived D0 history entries now fail closed instead of being skipped.
Original1000-prefix states/options/moments/objective must match before neural
runs; SAME Adam continues through5000. N0 snapshots/logs/final states and rolling
EVERY-transition hashes must match D0, detecting unlogged divergence even after
reconvergence. All N0 controls must complete before N1. No1000/5000-step
reproduction or training was executed in C.2.

## G–H. Resources, outputs and failure policy

One CPU thread/GPUs hidden;28800 whole seconds,900 active/history with10 endpoint
reserve;4GiB RSS/8GiB address limit. C.1's416.4/547.0/608.6-second estimates
support feasibility, not observed full runtimes or guaranteed completion.
Repeated reporting, prefix/trajectory overhead, allocation and OS variation
remain uncertainty; no budget extension or automatic retry is allowed.

The researcher explicitly approved separate `3d2-recovery-*` children of
`/home/isavand/paper/evidence`. Read-only capacity inspection found
2665867550720 bytes available versus484325298 minimum numeric checkpoint bytes.
Logs/failure artifacts require extra space. Capacity, ownership, empty/fresh
selection and exact approval are rechecked before execution. The unwritable
`paper_runs` is not used and no permission change occurred. No Phase D child
was created in C.2.

Failed admission saves already-produced arrays, actual input tapes, parameters,
buffers and available gradients as failure-only numeric NPZ plus finite JSON.
No extra shader call is made for logging. Censor/failure callbacks retain state
even before the first scheduled checkpoint; inspecting an uninitialized Adam
state no longer mutates its mapping. Snapshot manifests bind run/freeze/config/
source identity, with schema/counter/RNG checks before loading. No historical
pickle/checkpoint deserialization or experiment resume is introduced.

Source/admission/control failures stop globally; N1 numerical/active-limit
failures preserve unsuccessful arms; endpoint numerical failures and unknown
exceptions remain distinct; whole-process limits censor globally. Complete
endpoints are labeled `complete_pass`/`complete_fail` under all five criteria;
incomplete matrices cannot be reported complete. Successful endpoint/whole-run
work totals must exactly equal frozen counters. Teardown persistence is separate
from optimizer time and never permits more updates after a cap.

The internal timer starts after imports/admission/allocation. The final protocol
provides a concrete external28800-second `timeout` command enclosing the entire
Python process, with10-second kill grace for failure persistence. Future approval
must bind that watchdog and the exact selected root/run ID. The metadata CLIs
cannot launch training; the freeze explicitly says execution is unauthorized.

## I–J. Tests and scientific impact

**23 new focused tests passed;0 failed;0 skipped**,12.923s. Coverage includes
real saved-evidence verification with model/shader/Adam factories blocked,
missing/corrupt/fabricated/duplicate evidence, exact initialization/catalogue
guards, sample/PDF and cancelling per-tape failures, gradient observational
parity, nonfinite failure persistence, D0 prefix and snapshot mismatches,
between-log N0 divergence, immutable optimizer/loss/decoding helper ASTs,
resource/work limits, partial state/provenance, output/capacity and authorization.
Only synthetic seven-vector/autograd fixtures are used in functional unit tests;
no real shader or optimizer step runs. Earlier suites/experiments were not rerun.

Development tests caught fixture bookkeeping and a pre-first-step serializer
edge case, corrected before freezing. The live preparation then ran with model,
scientific-shader, Adam and execution entry factories blocked: all remained
uncalled. Preparation took34.73990814387798s, consumed no global Torch/NumPy RNG,
and made zero model/shader/optimizer/derivative/reference/recovery calls.
Protocol/source/JSON/Markdown/whitespace and preservation checks are separate
from these unit results. No scientific numerical formula or tolerance changed.

## K–L. Freeze identities, remaining execution checks and decision

- [Final protocol](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.md).
- [Final freeze](protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json), SHA-256
  `730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad`.
- Aggregate SHA-256
  `f9ef3477a4189958a6e0e16067b35dade62069a11797e90958515996b6cb9a2e`.

Sorted byte-hash inventory, separate aggregate selection and external final
freeze selection form a deterministic DAG. Freeze/aggregate/readiness outputs
are excluded from the source inventory; the aggregate is separately bound by
its exact hash and reverified. No circular self/commit hash is claimed.

**READY for review; NOT authorized to execute.** Remaining intentional run-time
gates: separate Phase D instruction/hash-bound review; source/artifact/runtime
revalidation; actual output ownership/capacity; all27 fresh model/admission gates;
exact1000-step D0 reproduction; complete N0 trajectory parity; finite-state
5000-update completion and final evaluation. These are not unexecuted successes.
Physical material recovery remains unknown. No new LM, GPU, long Adam, reference
generation, merge or Phase D work was started.
