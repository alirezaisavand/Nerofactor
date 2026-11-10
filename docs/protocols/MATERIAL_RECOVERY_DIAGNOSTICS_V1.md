# 3C-2 frozen diagnostic protocol v1

Status: written **before** the single CPU shader/reference diagnostic. Its hash
is recorded by execution. Do not edit it afterward. Results belong in a separate
report; no thresholds, probes or budgets will be tuned to those results.
Base `72629688bee84c0b06f4fa3c19b207917d2dd450`;
branch `feat/material-recovery-diagnostics-v1`.

## Evidence admission and workspace

Explicit input is the accepted `docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json`,
SHA-256 `f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451`.
Require completion, both original gates and all five250-update starts. Check all
six recorded numerical source hashes, the old protocol/execution specification
and the old results document against the approved Git base. No checkpoint,
pickle, camera, mesh or original dataset is loaded. The A2 shader, original
sources and old evidence are immutable. No new optimizer is constructed.

Use `/home/isavand/paper/Nerofactor-3c2-diagnostics`, a separate worktree from
that base. The original worktree has a researcher edit to
ANISOTROPIC_BRDF_SCIENTIFIC_AUDIT_V1.md; leave it untouched and out of this commit.
All new evidence is separately named with exclusive creation; no overwrite,
implicit latest discovery, continuation or retry.

The completed **offline, no-shader** analysis is
`docs/evidence/MATERIAL_RECOVERY_TRAJECTORIES_V1.json`, SHA-256
`4f69828f70d0f8d99063d2ffd10169aa991c312b8e930dbb558d7366d2c2693e`.
It preserves27 recorded parameter snapshots per start, late loss/gradient/parameter
changes, descriptive correlations, gauge-invariant Q comparisons, saved residual
compensation, all recorded branch brackets and explicit selected replay states.
Correlations across dependent trajectory points are not causal/significance tests.
Pre-/post-update logs retain their one-update offset; do not correlate them as
though the exact intermediate pre-update parameter vectors were saved.

Offline preparation initially hit a provisional40-state replay guard. Inspecting
the saved angle brackets showed44 selected snapshots. Before any shader/reference
diagnostic execution, the metadata-only guard was set to45 and the analysis
completed. No full diagnostic, reference generation or optimization was repeated.
The actual44-state selection below is now frozen, not adaptively extended.

## Fixed states and observations

Known flat p=0, n=+Z, original base T=+Y/B=-X. Preserve original24 fitting
observations and all lighting definitions/order; use saved84 targets for the
separately labeled evaluation-tape comparison. Material coordinates are
(mx,my,theta,kR,kG,kB,m). Exact requested truth is (.20,.55,.31,.30,.45,.60,.25).
Source shader inputs/math remain float32. Retain represented physical values
and roundoff; never substitute a shader-generated target for the independent one.

Primary states: truth and saved finals for seeds101,102,103,104,105, exactly as
recorded in admitted evidence. The five actual saved final unconstrained vectors
must decode bitwise to their saved float32 physical parameters. Compute actual
autograd gradients with respect to those seven unconstrained coordinates and the
intermediate physical vector, and compare chain conversion where dp/dz>1e-8.
The diagonal physical/logit derivative is original width factor .96*q*(1-q),
identity theta derivative and k*(1-k),m*(1-m).

At truth, evaluate the explicitly requested physical float32 vector directly.
Record physical autograd and chain-converted coordinate gradients there. Also
make one separately labeled latent-roundtrip evaluation with inverse-sigmoid
coordinates: report its actual decoded physical offsets and genuine coordinate
autograd gradient rather than pretending float32 inversion is exact.

Use exactly fit tapes0..3, regenerated in original u/v/azimuth order and required
to match every saved array/combined hash before shader execution. Evaluation tapes
1000..1003 must similarly match the original84-observation identities. No random
inputs are resampled, reordered or changed. Average source float32 outputs over
four tapes before the original float32 MSE; report residual analysis against
saved float64 fine targets separately, including target-rounding effects. Per-
tape variation/SE with only four fixed tapes is not population bias or uncertainty
in all deterministic radial quadrature errors. Label mean residuals as observed
finite-pattern error, not proof of systematic bias in expectation.

For all primary states, save per-tape/component/observation RGB predictions,
source mean, objective, physical/coordinate gradients, counts and mask hashes.
For truth also evaluate the original evaluation tapes on84 observations once.
Reuse saved terminal evaluation-tape predictions from3C-1 in offline comparisons;
do not redundantly regenerate them.

## Local finite-objective derivative probes

For each of six primary physical states, perturb each of seven physical
coordinates by +/-h*scale, using BOTH original h=1e-4,5e-5 and scales
(.96,.96,pi,1,1,1,1). Exactly28 common-tape probes per state. No parameter update,
line search or candidate optimization. Record requested/represented endpoints,
actual finite objective, valid counts and mask XOR/direction changes from center.
Compute central physical-objective differences and normalized radiance Jacobians
using the original requested denominators. Float32 endpoint rounding is recorded,
not silently corrected with a different step definition.

Compare autograd and finite-difference objective vectors at both steps. Report
norms, directions, absolute difference, cosine and relative discrepancy, without
calling any single noisy finite-difference mismatch a proven code defect. Cosine
and relative-to-reference metrics are null when the corresponding gradient norm
is <=1e-12; absolute values remain. This display floor is not an accuracy or
convergence criterion. Check mask changes and cross-step variation alongside
those metrics. A correct pathwise gradient need not equal the continuous-integral
gradient; no density/sampling/clamp/decoder change is authorized.

## Independent continuous objectives and gradients

At truth, reuse the original saved fine/coarse predictions and2x2x72x7
scale-normalized Jacobians. **Do not regenerate truth integrals or Jacobians.**

Only at five terminal states, compute new baseline and required28 perturbed
independent references per terminal. Use the unchanged Cartesian Beckmann/Kurt
a=0 reference with the original256x512 and512x1024 grids. Recreated grid/weight/
lighting hashes must equal the original saved identities. Reuse only exactly
identical per-view/channel/Fresnel factors, as in the admitted reference code.
Each old render call has its own local work ledger; a separate3C-2 aggregate
ledger enforces this diagnostic's budget, without altering old source or rerunning
its experiment.

All components/RGB/totals must pass unchanged
abs(fine-coarse)<=2e-6+5e-5*abs(fine), with total requiring converged components.
On any new reference/perturbation failure, preserve values and stop the diagnostic;
remaining comparisons are unresolved, not retried at a different resolution.
The two-step physical Jacobian is J_normalized/scale. Independent objective is
MSE(R_continuous(p)-saved_fine_truth); gradient is 2*J^T*residual/72. Preserve
coarse/fine and both-step gradients, raw matrices, spectra/right vectors, near-null
directions at the unchanged1e-4 singular-value ratio and observed step/grid drift.
Do not require a terminal to be full rank merely to report its conditioning;
low rank is a possible diagnostic finding, not grounds to hide the state.

Compare finite and continuous physical and coordinate gradients, with dot product
of the continuous coordinate gradient and negative finite gradient as a local
**directional derivative**, not an executed step. At truth the fine continuous
objective/gradient is zero by construction; cosine against it is undefined.
Full rank is local and does not prove global uniqueness or recovery.

## Saved-trajectory mask replay

Replay only the following recorded physical snapshots, with original fit tapes:

| Seed | Completed update snapshots |
| --- | --- |
|101|0,10,100,110,120,130,200,230,240,250|
|102|0,10,50,60,100,140,150,200,230,240,250|
|103|0,10,60,70,100,200,230,240,250|
|104|0,10,100,200,230,240,250|
|105|0,10,100,200,230,240,250|

These44 states are initial +10/100/200/230/240/250 plus all endpoints of observed
angle-branch brackets. Reuse the five terminal baseline results rather than
re-evaluating them. Save counts per tape/observation, mask bit hashes, valid-to-
rejected/rejected-to-valid/XOR counts and adjacent selected-state radiance changes.
This brackets changes between selected snapshots; it cannot recover unseen
per-step masks or locate every individual transition within ten updates.
Do not infer a jump's cause from counts alone.

## Branch-cut probes

For each of two fixed material backgrounds (truth, saved seed103 final), replace
only theta by center+offset. Centers are -pi/2,+pi/2; offsets are
-h1*pi,-h2*pi,0,+h2*pi,+h1*pi, using the same original h values. Exactly20
additional fit-tape shader evaluations. Keep winding, normal, tangent construction,
reflection rules, counts and decoder unchanged. The represented sin/cos encoding
at a float32 center is not assumed to be the literal pair(-1,0).

Compare across-cut mean-radiance jumps, masks and Q changes at both scales.
Independently inspect the physical kernel at incoming unit directions
normalize(.2,.4,1), normalize(-.4,.1,1), normalize(.6,-.3,1), outgoing
normalize(.3,-.1,1), F0=.1. Report pointwise kernel differences and pi-periodicity;
these are analytical checks, **not new hemisphere references or image scores**.
Finite samples can jump when a half-angle frame flips even though its distribution
is physically equivalent. Report measured behavior without treating three
pointwise checks as an integral error bound or proof of original-run causality.

## Budgets, stop behavior and exact command

Expected shader work: six baselines + one truth latent roundtrip +168 local
probes +39 uncached snapshot replays +20 branch probes =234 fit batch calls;
one truth evaluation batch. Each fit call has96 seed/observation points; the
evaluation has336. Expected22,800 points /35,020,800 sample directions.
Hard caps: fit23,040 points (240 batch calls), evaluation336 points (one call),
combined35,905,536 diffuse+specular directions. Always1024 specular/512 diffuse
samples per point. No sweeps, extra seed or new start.

Independent work: five*(1+28)*6*(256*512+512*1024) = **570,163,200** new per-view
kernel directions, hard aggregate cap. Grid/light construction is separately
counted. Zero new true-parameter integral/Jacobian evaluations. All calls are
CPU only, one Torch/BLAS/OpenMP thread, GPUs hidden. No optimizer construction,
Adam updates, neural models, historical inputs, installs or Docker.

Internal monotonic checks and SIGALRM plus external timeout enforce1,800 seconds
including all numerical stages/report writes (external cap also includes startup).
No automatic extension or rerun. Preserve partial JSON on failure, identify phase
and unresolved states, restore owned signal/thread state and verify global CPU
RNG/tapes unchanged. Code/protocol are frozen before the single execution.

Run only the NEW focused fixtures:

```bash
cd /home/isavand/paper/Nerofactor-3c2-diagnostics
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests/scientific \
  -p test_material_recovery_diagnostics_v1.py -v
```

Then execute once:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 1800s \
  python3 -B -m experiments.material_recovery_diagnostics_v1 \
  --input /home/isavand/paper/Nerofactor-3c2-diagnostics/docs/evidence/KNOWN_MATERIAL_RECOVERY_V1.json \
  --sha256 f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451 \
  --offline /home/isavand/paper/Nerofactor-3c2-diagnostics/docs/evidence/MATERIAL_RECOVERY_TRAJECTORIES_V1.json \
  --output /home/isavand/paper/Nerofactor-3c2-diagnostics/docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json
```

Classify findings as observations, strongly supported hypotheses or unresolved.
Recommend one next controlled experiment with a single isolated change, but do
not implement/execute it. Commit/push this feature branch only; do not merge.
