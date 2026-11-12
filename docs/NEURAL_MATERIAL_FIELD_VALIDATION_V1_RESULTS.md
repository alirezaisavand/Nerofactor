# 3D-2 Phase C validation: partial numerical success, reporting failure

**Overall status: FAILED / INCOMPLETE. Not ready for final freezing or Phase D.**
There was one scientific invocation and no retry. A new validation-reporting
defect stopped execution after all27 initial shader/backward fixtures and the
first deterministic replay. No recovery, synthetic Adam step, gauge rendering,
reference generation, GPU operation or final freeze occurred.

The [frozen pre-execution validation plan](protocols/NEURAL_MATERIAL_FIELD_VALIDATION_V1.md)
and [machine-readable evidence](evidence/NEURAL_MATERIAL_FIELD_VALIDATION_V1.json)
retain planned/actual counts, hashes, every initial comparison and the failure.
Do not rerun the current command merely to obtain a passing result.

## Source and execution provenance

Dedicated branch: `feat/neural-material-field-validation-v1`, from Phase B
`a11efd615d18ac69fa1c26a3692af511f1c78302`. The original workspace/worktrees,
all673 pre-existing file hashes including the user-edited audit document,
all Phase A/B source/tests/docs, accepted3D-1 evidence and both scientific
snapshots are preserved. AGENTS.md is unchanged; the researcher's explicit
Phase C instruction supersedes its historical Phase B authorization paragraph.

- Material A: `e1e859779301faced46d6476691e20b546c27ae0`.
- Geometry G: `79252178e65914ad1acbc14790b1514f276bfea0`.
- Accepted3D-1: `ccb03d0ec5adc814775538b9cc7427cc57480a98`.
- Original A field SHA-256:
  `b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe`.
- Phase C runner SHA-256:
  `ba6871bfd4e6bd1b4b86e189a6403f7e721d301104e415a4d3e9e7e0b0412cde`.
- Executed test SHA-256:
  `ce599bf95b63e19edfe46277e6bd7615012c2385b1ecceb9c4a15e5a374aa99f`.
- Pre-execution protocol SHA-256:
  `00c0b99e0d79218e0cdbbb5f32c59e744d9248e1390b9c82af36ed9a0b981e10`.

Read-only admission verified106 inherited identities: the accepted102-file
chain plus both Phase B implementations and tests, including comparison against
their exact Git blobs. No old runner or historical optimization was executed.
Runtime matched accepted Python3.9.12 / NumPy1.21.5 / Torch2.4.0+cu121 on CPU,
one thread, GPUs hidden. Original A2/atan2, known plane/lights,24 fitting
observations, four tapes3000–3003 and512/1024 samples were retained.

## Actual work and evidence location

Run ID: `neural-field-validation-v1-59db4091cb324d72a4e77cbba82a5fc9`.
Owned output root:
`/home/isavand/paper/evidence/neural-field-validation-v1-59db4091cb324d72a4e77cbba82a5fc9`.
No previous output directory was reused.

| Operation | Planned | Attempted | Completed |
| --- | ---: | ---: | ---: |
| Full shader forwards |45|28|28|
| Renderer backwards |36|28|28|
| Field-only derivative probes |24|21|21|
| Synthetic Adam returns |3|0|0|
| Explicit field queries |63|43|43|
| D0 / N0 / N1 construction |25 /13 /16|13 /9 /12|13 /9 /12|
| N1 calibration |16|12|12|
| Independent original component construction/forward checks |3 /3|3 /3|3 /3|

Actual shaded points2688; diffuse directions1376256; specular directions2752512.
No attempted operation was retried. Initial coverage is all27 planned arms;
the additional forward/backward is M1/101/D0 replay only. Remaining shader
fixtures: eight replays and nine gauge probes. There are zero recovery updates,
LM calls, external recovery evaluations or new reference/identifiability renders.

Internal validation elapsed8.757739813998342s; external process including imports
elapsed10.378645658493042s and exited2. Peak RSS559816704 bytes, well below4GiB.
The process did not reach its300-second timeout. Global Torch/NumPy RNG stayed
unchanged; thread/memory limits and signal/timer ownership were restored.

Original unmodified `report.json` SHA-256:
`a5dff2d22167c00ba1cd144c81b9dbce49f6903f26c9197854d6b118b877d9e5`.
Console SHA-256:
`d4ffebf0fdb9947917667d3a639cfc8149deeb9309c68d6f228d25eca76a7a7a`.
Pre-execution and process-outcome manifests and a verified console copy are
preserved beside that report. Original report bytes were never edited.

Offline inspection verified29 numeric archives (inputs,27 initial returns and
one replay), whole-file SHA-256 before loading, catalogue hashes, numeric
shapes/dtypes and finiteness using `allow_pickle=False`. Their total size is
151309494 bytes, outside Git. The versioned JSON references these exact files,
keeps bounded numeric comparisons and hashes the full original catalogues.
Offline inspection made zero shader/derivative/optimizer calls.

## Source, architecture, initialization and gradients

All three independent source/field audits passed: exact original definition
ASTs and component parameter values;51-channel positional encoding; eight-layer
trunk/307-wide skip; five259-input heads, original activations/weight normalization;
28 linear modules,84 trainable tensors and exactly1485584 parameters. No learned
geometry, illumination or positional embedding was introduced.

Observed trunk inputs and EVERY head's raw-coordinate inputs were `(0,0,0)`.
All predeclared batch1/9/96 and repeated nine-position outputs were constant.
Six scalar initial values were bitwise equal. The original rho=.5 target
arithmetic, detached immutable centering buffers and nonzero trainable original
weights matched. N1 identities matched across all three truth associations for
each seed: truth did not select the initialization. Scalar theta differences
were5.960464477539063e-8 radians at101 and0 at103/105; bitwise theta parity is
not required. Q stayed inside its unchanged bound.

All27 initial losses/predictions/parameter gradients were finite. All six N1
groups (shared trunk and five heads) received nonzero finite gradients with no
missing parameters, while expected individual structural zeros remained.
The actual shader-input encoding received finite nonzero gradients. Independent
7×8 final-bias Jacobians had rank7 for all three seeds; no dense shader Jacobian
over the1.49M weights was constructed. Invalid zero/near-zero encoding rejection
passed using the existing decoder contract.

## Initial numerical comparisons

All nine D0/N0 comparisons were bitwise exact for samples/PDFs, every-tape and
mean components, loss, latent gradients and shader-input encoding gradients.
All nine N1 physical/frame/sample/mask/component/arithmetic comparisons passed.
All original mean/MSE replays and gamma75 assumptions/envelopes passed. The old
scalar warning test passed9/9; initial D0/N1 losses were bitwise equal in this run.

| Material | Seed | D0 and N1 initial MSE |
| --- | ---: | ---: |
| M1 |101|0.02078045904636383|
| M1 |103|0.0064084408804774284|
| M1 |105|0.027302611619234085|
| M2 |101|0.01579143851995468|
| M2 |103|0.026913540437817574|
| M2 |105|0.02160882204771042|
| M3 |101|0.007212584838271141|
| M3 |103|0.006835490930825472|
| M3 |105|0.02510339580476284|

Worst N1 versus D0 measurements below occur at seed101, identically across
materials because their initial material/light inputs are shared. The JSON
retains association, exact indices, actual values and component/relative errors.

| Quantity | Maximum absolute difference | Maximum angle |
| --- | ---: | ---: |
| Q |1.7309891853223291e-9|—|
| Actual double-angle encoding |8.940696716308594e-8|—|
| Tangent/bitangent |5.960464477539063e-8|—|
| Specular half-vector |1.1920928955078125e-7|1.4673391039186107e-7 rad|
| Specular outgoing direction |3.2782554626464844e-7|3.4743103603500235e-7 rad|
| Per-tape/mean specular RGB |2.9802322387695312e-8|—|
| Per-tape/mean total RGB |5.960464477539063e-8|—|
| Source PDF (diagnostic) |0.00371551513671875|—|

Normals, diffuse directions/RGB, masks and valid-sample counts were exact.
All actual frames were orthonormal within the existing limit. Half-vector and
outgoing component-relative maxima were0.0088235294 and0.0175438596 near small
reference components; their absolute-plus-relative AND angular gates still
passed. PDF maximum relative difference was7.038559641808491e-5. No PDF equality
threshold was introduced: inherited3D-1 explicitly reports it diagnostically.
Seed103/105 sample/PDF differences were zero. Metric equality alone does not
replace these frame/sample/component checks or demonstrate physical recovery.

## Reporting defect, replay and unexecuted work

At [runner line448](../experiments/validate_neural_material_recovery_v1.py#L448),
`dict(material=m, ..., **info)` receives a second `material` key from
`capture()`'s material diagnostics. Python raises
`TypeError: dict() got multiple values for keyword argument 'material'`.
This is a new metadata-assembly defect, not a BRDF, calibration or optimizer
failure. The15 new preflight tests passed but did not cover complete replay-row
assembly. This missing test coverage is retained as a limitation.

The M1/101/D0 replay archive was saved before that row construction failed.
Offline hash-pinned inspection confirms EVERY saved sample/component/loss/
gradient array bitwise matches its initial capture. It is recorded separately
as recovered unregistered evidence; no rendering was repeated or report row
fabricated. This supports ONLY one of nine replay fixtures, not complete replay.

No theta-periodicity/signed-swap rendering probe was reached. Those gates are
**unresolved**, even though Q/encoding/initial frame checks and invalid-encoding
rejection passed. No synthetic Adam/checkpoint storage fixture was reached.
No optimizer-state memory, serialization, restoration or900-second history
feasibility pass is claimed.

Measured N1 initial forward max0.050338272005319595s, backward max
0.042688172310590744s, construction max0.11628973484039307s (nine real initial
forward/backward measurements). D0 maxima were0.04991212487220764/
0.032683657482266426s and N0 maxima0.04399256221950054/
0.03289753198623657s. These include validation instrumentation and cannot
establish5000-update feasibility without Adam/storage/other overhead. The
candidate budget remains5000 updates and900 active seconds, unchanged.

## Read-only Phase B execution-gate review

| Function | Assessment and remaining limitation |
| --- | --- |
| `MaterialBatch.render` | D0/N0 preserve original `ObservationBatch`; N1 uses its actual encoding directly, same component order/means. Real initial parity supports this narrow path. |
| `initial_admission` | Enforces N1 Q/frame/sample/mask/component/envelope and D0/N0 mean/loss/latent-gradient checks. D0/N0 per-tape/sample/PDF exactness is not explicitly gated there; Phase C checks those additionally. Propose making the complete reviewed contract explicit before Phase D. Negative returned arrays are also not persisted by this helper on its own failure. |
| `prefix_check` | Source review matches archived pre-update loss/native gradient norm and post-update vectors; moments/options at250/1000 and post1000 objective checked. NO1000-update reproduction was executed here. |
| `train_history` | Continuous original Adam, counted snapshots, D0-prefix checks, N0 snapshot/log and rolling EVERY-transition digest; no LM/clipping/scheduler. NO5000-update trajectory/parity was executed here. |
| `evaluate` | Explicit own truth and original external tapes, five simultaneous Q/orientation/albedo/metallic/held-out criteria and component diagnostics retained; nonidentifiable orientation does not pass. Not executed in this validation. |
| `execute_reviewed` | All initial gates before updates; all D0 controls before neural arms; fresh owned outputs. Numerical failures versus integrity stops and partial states are recorded. External-evaluation exceptions stop globally; descriptive partial-result handling needs final protocol review. |
| `validate_execution_review` | Requires explicit Phase D selection, final protocol/freeze hashes and actual passed Phase C source/runtime evidence before allocation. This FAILED report cannot admit execution. |
| Resource/storage guards | Work/active/whole caps and owned numeric SHA-verified no-pickle format implemented. Main runtime context begins after some admission/import/allocation work; future external whole-process timeout remains necessary. Storage performance/strict N1 round trip remain unmeasured. |

These are inspection findings, not authorization to change Phase B. No original
scientific or experimental file was edited to address them.

## Decision and smallest next task

Source/architecture and the complete INITIAL numerical/gradient matrix passed.
Overall scientific numerical admission remains FAILED/INCOMPLETE because the
full prescribed replay, gauge and feasibility matrix did not complete.
Do not freeze a final protocol or authorize Phase D from this partial result.

Propose a separately reviewed **Phase C reporting correction and bounded
completion**: rename the replay row's identifier to `material_name` (or otherwise
separate identity from diagnostics) and add a synthetic row-assembly regression.
Preserve this runner/test/protocol SHA and failed evidence; give any corrected
runner a new identity. Reuse hash-verified completed initial evidence and the
M1/D0 replay. Predeclare only the remaining eight replays, nine gauge forwards
and three synthetic optimizer/storage fixtures:17 scientific forwards/eight
backwards/three field-only derivative probes, with unchanged thresholds and
the same or stricter ceiling. No repeat of the27 initial fixtures, no calibration
search or new recovery run. This correction/remainder was NOT implemented or
executed after the failed invocation.

New focused preflight result: **15 passed,0 failed,0 skipped**,10.161s, all real
shader calls replaced by synthetic algebraic fixtures in those tests. Static
AST/JSON/documentation/source-preservation checks are reported separately from
the failed scientific invocation. No previous test suite was rerun.
