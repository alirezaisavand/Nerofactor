# Material symmetry audit v1 — stopped attempt

**The single authorized diagnostic failed before its first shader call.**
It did not produce a completed physical-symmetry, radiance, fitting-objective or
sampling comparison. H1–H4 remain unassessed by this milestone. No symmetry
violation, objective preference or explanation of the LM groups can be claimed.

The failure is a report-serialization defect in the new audit, not an observed
BRDF/sampling defect. A NumPy Boolean in the first analytical check cannot be
encoded by the standard JSON writer. The failure was missed by the focused
fixtures, which checked the analytical values but did not JSON-serialize a
successful physical-equivalence record.

The diagnostic was **not retried**. The runner, tests and pre-execution protocol
remain byte-identical to their frozen versions; no reporting fix or additional
numerical execution was implemented after failure. The raw failed artifact is
preserved exactly, even though it is incomplete JSON.

## Artifact identities and scope

Base `f4f5f4d57fc9f57c4d6387663ac9ef19b40144c0`; feature branch
`feat/anisotropy-symmetry-audit-v1`. The accepted artifacts/source/protocol chain
passed read-only admission before the run. All previous evidence, scientific
sources, shader and user edits remain unchanged.

| Artifact | SHA-256 | Status |
| --- | --- | --- |
| [Frozen protocol](protocols/MATERIAL_SYMMETRY_AUDIT_V1.md) | `8e510d566d443ed44f3e0428c429837ccd70b6bcfba0951efd5ce6bcf1d4572a` | Preserved pre-execution protocol |
| [Frozen implementation](../experiments/material_symmetry_audit_v1.py) | `65a5b9e134df648ac6633336c3d59ef8a65fa9015d4deefd8c30b490f1fc68fc` | Contains the identified serialization defect |
| [Original partial output](evidence/MATERIAL_SYMMETRY_AUDIT_V1.json) | `055aa0a2c13457dc198bb888c45ff15062fec90856de31220e09ada196568a55` | **Invalid/truncated JSON; preserved raw bytes** |
| [Separate failure record](evidence/MATERIAL_SYMMETRY_AUDIT_V1_FAILURE.json) | `47e879568455b593117304a89707a4216620fe6ca0e0a8985e4c98686f579b46` | Valid machine-readable failure manifest |
| Accepted3C-4 evidence | `66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f` | Unchanged; not regenerated |
| Accepted3C-4 results document | `699fe50ebdfe2cd087147814787e8ee23b1267e3753c9ad92d5961382579d5f2` | Unchanged |

The raw output is13,840 bytes. Its JSON parser error is:
`Expecting value: line419 column31 (char13840)`.
The separate failure record reports this defect explicitly and references the
original bytes; it does not replace, sanitize or complete the failed output.
Consumers must not treat the partial `.json` file as valid experimental evidence.

## What executed and why it stopped

The intended single diagnostic was launched using the literal command in the
frozen protocol, with GPUs hidden and one Torch/BLAS/OpenMP thread. CPU runtime
versions matched the accepted Python3.9.12/NumPy1.21.5/Torch2.4.0+cu121 environment.
Only read-only metadata, source checks and the beginning of analytical physical
validation were reached. No original dataset, checkpoint or native model was
used or loaded.

Observed traceback:

```text
execute(): row['representations'].append(rep); persist()
write_report(): json.dump(value, stream, indent=2, allow_nan=False)
TypeError: Object of type bool_ is not JSON serializable
```

The first truth/P0 nominal and rounded physical calculations reached report
serialization. In `physical_equivalence`, the value for the `orientation` check
is formed by `angle is None or angle <= ANGLE_ATOL`. For an identifiable case,
`angle` is a NumPy scalar, so the comparison returns `numpy.bool_`, rather than a
native Python Boolean. Other checks explicitly use `bool(...)`, but this check
was missed. This source-level diagnosis does not require another numerical run.

The shared writer seeks to the beginning before dumping. When dumping fails,
the original report is left incomplete. Subsequent failure/finally persistence
attempts encounter the same Boolean, so they also fail. The last partial output
contains root `status="failed"`, `physical_gate="not_started"` and a complete
work-counter object, then ends at `checks.orientation` without a value. The
control-flow phase is identified from the traceback/source; the root phase field
itself was not reached by serialization. The process exited with code1.

The traceback was inspected from the Codex execution-tool output. No original
console-log file was configured; a fabricated byte-identical console archive
is not claimed. Actual elapsed/CPU time, peak-RSS, final RNG and cleanup fields
were not durably serialized, so no file-backed measurements of those fields
are claimed. The traceback reaches final persistence after process-limit/signal
cleanup statements, but that is control-flow evidence, not a complete saved
cleanup report. Process termination also releases its owned resource limits.

## Preserved counters and unavailable comparisons

The complete counter object in the raw JSON prefix records:

| Quantity | Recorded count |
| --- | --- |
| Shader calls attempted/completed | **0 / 0** |
| Shader points | **0** |
| Specular/diffuse sampled directions | **0 / 0** |
| Independent pointwise kernel scalar values |90 |
| Independent pointwise PDF scalar values |30 |

These are the recorded counters from the incomplete output. Only the first
truth/P0 nominal/rounded analytical records were being serialized. No dense
hemisphere integration, new targets, optimizer, parameter update or GPU/neural
training occurred. No transformed comparison completed the physical gate.

| State | P0 | P1: swapped widths+pi/2 | P2: theta+pi | Objectives/radiance/sampling |
| --- | --- | --- | --- | --- |
| Truth | Partial analytical self-check; report serialization failed | Not reached | Not reached | Unavailable |
| LM101 | Not reached | Not reached | Not reached | Unavailable |
| LM102 | Not reached | Not reached | Not reached | Unavailable |
| LM103 | Not reached | Not reached | Not reached | Unavailable |
| LM104 | Not reached | Not reached | Not reached | Unavailable |
| LM105 | Not reached | Not reached | Not reached | Unavailable |

The partial truth/P0 zero differences are self-comparisons, not verification of
P1/P2 symmetry. No new held-out or relighting metrics exist. No representation
was preferred, substituted or selected. Accepted3C-4 results remain valid in
their original scope and are not promoted to evidence for this audit.

| Hypothesis | This attempt's assessment |
| --- | --- |
| H1: physical BRDF/Q invariance | **Unresolved:** no complete transformed physical comparison |
| H2: finite-estimator non-invariance | **Unresolved:** no shader forward |
| H3: fixed-tape objective preference | **Unresolved:** no fitting-objective measurement |
| H4: contribution to LM solution groups | **Unresolved:** no new scientific comparison or causal experiment |

## Tests and static validation

**21 new focused tests passed, zero failures/skips**, test-body time1.348s,
before the single attempt. They exercise nominal tensor/kernel/isotropic gauges,
float32 encoding, a tiny A2 capture, common tapes/order, component consistency,
per-tape versus averaged differences, mask accounting, original MSE, cache hashes,
no optimizer/backward/RNG use, budgets, failure/collision cleanup and source
preservation. They are not a successful full audit. Their missing coverage is
now explicit: successful physical-check objects were not round-tripped through
strict JSON serialization. The failure/admission fixtures only serialized simpler
primitive-valued records.

The new Python files pass Python3.8 AST checks; actual tests/runtime are3.9.12,
not a claim of Python3.8 runtime compatibility. Read-only admission verified18
bound identities. All268 protected previous source/protocol/evidence files,
including34 old evidence artifacts, remain byte-identical, as do the frozen new
runner/protocol/tests and the original workspace's uncommitted audit-document
edit. No old suite or scientific experiment was repeated.

The separate failure manifest passes strict JSON parsing; the raw output is
**expected to fail** parsing and retains its original trailing whitespace and
unfinished final line. Source/document whitespace and Markdown links can be
checked while explicitly excluding that forensic raw artifact. A blanket
all-JSON/all-whitespace pass is not claimed. No source, sampling, threshold or
protocol was changed in response to this failed attempt.

## Smallest safe next action — requires separate approval

Review a reporting-only correction converting `checks.orientation` to a native
Python Boolean, with a focused strict-JSON roundtrip regression covering valid
anisotropic/isotropic P0/P1/P2 and rounded analytical records. Consider validating
serialization before modifying the output stream so future errors retain the
last valid status report. Neither correction is implemented here.

After that review, separately authorize **at most one fresh, versioned attempt**
of the same bounded scientific audit. Retain this failed file and manifest;
never delete, overwrite or reuse the consumed output path, and do not relax any
numerical bound, change shader/sample settings or introduce new optimization.
No further scientific experiment can be recommended from new symmetry results
because none were obtained. The current milestone stops here as requested.
