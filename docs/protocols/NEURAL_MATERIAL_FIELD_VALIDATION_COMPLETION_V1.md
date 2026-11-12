# 3D-2 Phase C.1: reporting repair and bounded completion v1

Base `42c8f4b08a0db11a47413f1007fb504f6e995378`; dedicated
`feat/neural-material-field-validation-completion-v1` branch. This is completion
of the failed Phase C validation, not a recovery experiment or final freeze.
The original report, versioned evidence, arrays, protocol, tests and failure
status stay immutable. Only the reporting helper/call site in the old runner
changes: `material_name` identifies M1/M2/M3; `material` retains diagnostics.
An AST comparison admits precisely that metadata-only change. No shader,
source, calibration, threshold, optimizer or training-harness change is allowed.

## Evidence admission before new scientific calls

Pin the old versioned evidence SHA-256 to
`03bb86b7c3300839cebad5313a22b50e7169e5e7b81720eef7de0446b52705bf` and original
owned report to `a5dff2d22167c00ba1cd144c81b9dbce49f6903f26c9197854d6b118b877d9e5`.
Require the original TypeError/failure status, exact28/28 forward/backward,
21 field derivatives, zero Adam and all remaining recorded counters. Verify
the original source/test/protocol Git blobs, Phase B and106 admitted identities,
pre-execution manifest, process outcome and console hashes. No implicit discovery.

Verify inputs and all27 initial NPZ archives against whole-file hashes BEFORE
loading, catalogue hashes, shapes, dtypes and finite numeric values using
`allow_pickle=False`. Independently recompute means/MSE, bounds, Q/frame/masks,
calibration/actual encoding, parameter-gradient topology/finite/nonzero groups
and all original D0/N0 exact and D0/N1 bounded sample/component/arithmetic gates
from saved arrays. Recompute rank7 from the saved seven-by-eight Jacobians;
do not rerun field VJPs. Check recorded model metadata against original topology
and fixed source AST identities. Source/weight hashes are provenance identities,
not a new execution of the old networks.

Verify the unregistered saved M1/101/D0 replay with pinned SHA
`f3b7d45f2fb61445ad5c74b682dde419513fc025645a24692f1314393b073603` and require
every array's shape/dtype/bytes to match its admitted initial capture. Any
missing, corrupted or inconsistent artifact stops before new shader/model work.
Do not reconstruct evidence. Record reverified results separately from new work.

## One exact remainder, same order and numerical contract

1. Replay seed101 in original material/arm order, SKIPPING M1/D0: M1/N0,N1;
   M2/D0,N0,N1; M3/D0,N0,N1. Eight forwards and eight original RGB-MSE backwards.
   Fresh models must match saved initial identities. Exact array/gradient replay.
2. Original nine M1 gauge forwards: each seed101,103,105 x periodic theta+pi,
   uncoupled axes/theta+pi/2, inherited signed-quarter-turn coupled swap. Reuse
   saved initial M1/D0 arrays. Zero backwards. Uncoupled radiance differences
   remain diagnostic; periodic/coupled gates are mandatory and unchanged.
3. Original three synthetic fixtures, seed101 x D0,N0,N1: one field-only
   squared-physical-output derivative and one Adam step each. No fit targets or
   shader. Verify the existing hash-selected pickle-free state round trip after
   clearing only the disposable model's own state. No recovery updates.

| Additional operation | Attempts/completions if complete |
| --- | ---: |
| Shader forwards / renderer backwards |17 /8|
| Field-only derivative probes / synthetic Adam steps |3 /3|
| Explicit field queries |20|
| D0 / N0 / N1 construction |12 /4 /4|
| N1 calibration |4|
| Original component or source-audit numerical probes |0|
| Shaded points / diffuse / specular directions |1632 /835584 /1671168|

Additive old+new counters must equal the original45/36/24/3 plan, not a new
experiment matrix. No duplicate operation labels, hidden warmup, extra seed,
retry, reference regeneration or partial recovery run. Stop/preserve on the
first mandatory failure; do not continue until success by changing settings.

Use CPU one thread, GPUs hidden,4GiB RSS/8GiB address backstop. Maximum new
external wall budget is289 seconds, so the prior10.378645658493042-second
process plus this invocation remain below300 seconds. Shorten the owned timer
accordingly. This is stricter than a fresh300-second allowance, not an extension.
All source/evidence admission, new fixtures and storage share that new ceiling.

Reuse every tolerance and arithmetic gate from
[Phase C](NEURAL_MATERIAL_FIELD_VALIDATION_V1.md). PDF differences remain
diagnostic except exact D0/N0/replay identity and finite values. No scalar-angle
round trip for neural shading, angle repairs, new clamps or calibration tuning.

## Feasibility, provenance and final decision

Reuse saved initial timings and combine them with new replay timings and the
single synthetic Adam overhead per arm. Keep the existing max-forward +
max-backward + Adam5000-step estimate, three serialization costs and endpoint
overhead. Report state bytes, archive bytes, serialization/hash/restore times,
RSS and uncertainty about logging, prefix checks and OS variability. The
900-second history budget remains unchanged. A bounded estimate supports
feasibility but cannot prove full-run completion. Stop for budget review if it
appears infeasible; no GPU fallback or extra benchmark repetitions.

Freeze runner/test/protocol hashes for THIS validation before its single
invocation. New owned outputs and versioned C.1 JSON/results refer to old
reports by immutable hashes. Never rewrite the failed Phase C report as passed.
Source and evidence preservation are rechecked afterward. A complete passed
matrix may support recommending final protocol preparation; it does not create
a final Phase D freeze or authorize scientific recovery execution. Retain the
Phase B admission-helper limitations and unexecuted1000/5000-step gates.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m experiments.complete_neural_material_validation_v1 --verify-only

# One separately owned fresh output, after tests and evidence admission:
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5s 289s \
python3 -B -m experiments.complete_neural_material_validation_v1 \
  --execute-once --output-root "$ROOT" --run-id "$RUN_ID"
```
