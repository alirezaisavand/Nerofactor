# E.8R v1: endpoint-corrected recovery of the frozen spatial pilot

Preparation only until explicit scientific execution approval. Accepted base:
`51eac252cc15060612c808286ca06a0bf4774ece`, branch
`feat/neural-material-early-step-e2a`. Preserve the original
[E.8 protocol](SPATIAL_MATERIAL_E8_V1.md),
[freeze](SPATIAL_MATERIAL_E8_V1.freeze.json),
[failed attempt](../SPATIAL_MATERIAL_E8_ATTEMPT1_REPORT.md) and every prior
scientific source/evidence artifact unchanged.

## Scope and unchanged scientific design

The [original E.8 scientific design](SPATIAL_MATERIAL_E8_V1.md) remains binding:
four training corners/two held-out coordinates, the same seven-parameter affine
material field, known flat geometry/distant lighting, seeds101/103, identical
initialization of one shared coordinate-dependent field per seed/arm.

All six arms remain: spatial N1 photometric Adam, physically guided shared-head
projection, and no-soft-retention ablation. Preserve A2/atan2,512/1024 samples,
24 fitting observations/84 external observations, all original targets,
LR=.001/Adam settings,12 cyclic proposal rounds,damping.01/scaled radius.1,
head-projection ridge/sensitivity/trust/retention settings,actual joint training
acceptance,all rejection rules and all original physical-recovery thresholds.
No added seeds,solver selection,hyperparameter search,checkpoint selection or
method changes. Both equal-update(state12) and mechanical31–37-return baseline
comparisons are preserved, including every query in the original cost proxy.

The new runner copies E.8's scientific Core and both initialization/optimization
loops exactly. Focused AST comparisons verify their identity and the unchanged
work-matching function. The original spatial-network/projection module is reused
without edits. Only tape conversion/preflight, reference reuse, run identity,
zero new-reference accounting and setup-stage labeling differ.

## Explicit finite-precision uniform policy

Policy ID: `float64-uniform-to-float32-open-upper-endpoint-v1`.

For every source sample, first require original dtype float64,finite,and[0,1).
Perform the original float32 cast. If and only if a valid source rounds to
exact float32(1),replace that rounded endpoint by:

```python
np.nextafter(np.float32(1.0), np.float32(0.0))
```

This is0.9999999403953552 (float32 bits0x3f7fffff). Never repair invalid source
samples,redraw,change seeds or relax renderer validation. All other float32 bits,
including signed zero,remain exact. This is an explicitly authorized finite-
precision endpoint mapping, not a claim of an unchanged quantized distribution.

Seed/draw order/shapes/sample counts remain original. Apply the same policy to
all36 arrays covering six locations' fitting/evaluation tapes,including declared
unused held-out fitting tapes. Exactly one of5,311,008 values changes:
location5,evaluation seed7050,array v,index[42,982],original float64
.9999999982641566,old float32 1.0,new float32 .9999999403953552.
Thirty-five array identities remain unchanged; location5/evaluation/v receives
a new identity. Freeze both original and corrected identities and the complete
correction inventory. Any unexpected correction fails closed.

## Complete preflight before neural/scientific work

Preparation and the guarded execution path independently generate all declared
tapes with private RandomState instances,verify original hashes against E.8,
and check finite float64 source values and float32 shapes/dtypes/strict domain.
Compare old/new uint32 views: only the documented rounded-one bit changes are
permitted. Freeze corrected hashes and require deterministic replay of the full
preflight summary. No global RNG consumption or sample repair beyond this policy.

Validate every frozen source/input/evidence identity and the scientific-design
parity with original E.8. Complete tape preflight before importing Torch or the
scientific harness. Only then may an explicitly approved runner construct
scientific objects. The original strict RandomInputs validator is untouched.
CLI --describe/--prepare/--verify-sha256 never executes scientific rendering,
N1 construction,physical optimization or independent reference generation.

## Reuse admitted scientific references, zero new reference work

Explicit root:
`/home/isavand/paper/evidence/material-e8-565980ba2b1245a39b221288ea2b7fac`.
Original failed report hash:
`42abc3d3d4a672cc8faa4f1733701ca488b52642690a9eda632321a475fc9ecb`.
Reference-producing freeze:
`425956188c550f3139996f70e06f70b19a36c9a574af88a2fddc8df862d68aca`.

The accepted E.8 evidence manifest binds all six independent NumPy reference
archives/catalogs,four rank reports,failed report/seal and launch sidecars by
exact sizes/hashes. Require the same truth,coordinates,view/light observations,
independent grid identities,reference source/BRDF and convergence/rank settings.
Validate stored reference shapes/dtypes,coarse/fine differences/convergence
bounds and cached rank-admission metadata. This reads accepted measurements;
it does not integrate radiance or repeat finite-difference rank measurements.

The corrected Monte Carlo tape endpoint does not enter the independent NumPy
quadrature or its truth-perturbation rank calculations. Therefore those targets
and rank evidence remain valid. Any incompatibility stops; no regeneration.
References in the E.8R report explicitly point to their original root and have
role `reused_successful_E8_evidence_not_new_measurements`. Do not interpret their
relative selectors against the new run. Original failure/cleanup/completion
flags remain unchanged; a new successful run cannot rewrite E.8 as successful.

## Execution budget and resources

| Maximum new E.8R work | Cap |
| --- | ---: |
| Shader forwards | 960:528 neural fitting +48 evaluation +384 physical |
| Backwards/reverse calls | 1008:336 neural +672 physical JVP sweeps |
| Adam returns | 74 |
| Physical Jacobians / proposals / projection attempts | 48 /48 /48 |
| Small linear solves | 384 |
| Extra field probes / constructor calibrations | 14 /6 |
| **New reference integrations / directions / rank measurements** | **0 /0 /0** |

All optimization/query caps remain E.8's approved values. Only new-reference
caps drop to zero;118 integrations/487,587,840 directions/four rank measurements
are historical reused work and must be reported separately. Exact new counts
may be below caps under original skipped-proposal/work-matching rules; retain
queried rejected trials and all six arm outcomes without favorable selection.

Resources are unchanged:CPU float32,one Torch/BLAS thread,hidden GPUs,accepted
Python3.9.12/NumPy1.21.5/Torch2.4.0+cu121 runtime. One foreground launch under
timeout1800s/10s kill grace,120-second admission/setup/arm stages,4 GiB RSS,
8 GiB address space,1 GiB output cap,≥2 GiB free. Fresh explicitly authorized
`material-e8r-<uuid>` under `/home/isavand/paper/evidence`.

Any identity,preflight,source-parity,scientific-domain/resource failure stops
and preserves partial records; no retry,extension or alternate settings. Actual
scientific gradients/weights,source/input/tape identities,original backward/
Adam counters and cleanup gates follow the frozen E.8 runner. Successful
cleanup is recorded only after context exit; completion requires terminal
checks plus a separate seal and successful external receipt. No GPU/Docker.

## Reporting and authorization boundary

Report all six arms,per-location physical/Q/axis/identifiable angle/albedo/metal
errors,mean/worst train and spatial holdout outcomes,original full-recovery
criteria,held-view/relighting/component errors,projection residuals,other-location
drift,spatial variation,saturation/gradient transport,acceptance/rejection,costs
and runtime. Preserve both positive and negative results; truth/held-out scores
never enter optimization or endpoint selection. Known geometry/lighting and
six sparse sites do not establish real-object generalization or novelty.

Preserve raw data outside Git. After one successful authorized execution,
offline analysis may generate tables/figures/report and rank gauge-aware
optimization,sensitivity-aware observations and geometry uncertainty directions.
No extra scientific queries or automatic extension. Conclude PROCEED /
FIX AND PROCEED /BLOCKED with evidence and limitations.

```bash
python3 -m experiments.spatial_pilot_e8r_v1 --describe
python3 -m experiments.spatial_pilot_e8r_v1 --prepare
python3 -m experiments.spatial_pilot_e8r_v1 --verify-sha256 FROZEN_SHA256
```

Commit the frozen preparation so the single execution request can identify its
exact clean execution commit,as required by E.8R. Scientific execution remains
unauthorized until explicit researcher approval of the new freeze/commit,fresh
output and bounded budget. Approval covers one launch,offline finalization,
final commit and one ordinary push attempt; never another launch or new arm.
