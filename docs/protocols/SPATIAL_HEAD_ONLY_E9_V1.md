# E.9 v1: head-only photometric Adam spatial control

Preparation only until explicit scientific execution approval. Base
`e21d3dda9da35c24c162bdc0cc2194089e3ef52b`, branch
`feat/neural-material-early-step-e2a`. Preserve all accepted sources, protocols,
E.8/E.8R raw evidence and results. No new architecture, physical solver, damping,
sampling strategy, observations, loss, threshold or hyperparameter search.

## Question and two new arms

[E.8R](../SPATIAL_MATERIAL_E8R_RESULTS.md) completed six spatial arms. Guidance
helped Q at seed103 and worsened it at101, with worse RGB for both and0/36 full
recoveries. Guidance changed only final heads; the Adam comparator changed the
full network. This control isolates active-parameter restriction, without
claiming to isolate every optimizer-geometry/update-policy difference.

Exactly two new arms: seed101 and seed103 **head_only_Adam**. Use the same
spatial N1,1,485,584 total parameters/84 tensors/12 buffers and genuine input
coordinates as [E.8R protocol](SPATIAL_MATERIAL_E8R_V1.md). One jointly trained
field per seed: four training corners and two evaluation-only spatial holdouts.
Reuse its smooth affine material truths, six independent reference archives,
four cached rank reports,24 fit/84 external observations,corrected fixed tapes,
known flat geometry/distant illumination,A2/atan2,512 diffuse/1024 specular samples
and total linear-RGB MSE. No old scientific arm,reference or rank measurement
is rerun. No oracle teacher/seed/checkpoint selection.

## Exact saved initialization, not regeneration as a substitute

E.8R run:
`/home/isavand/paper/evidence/material-e8r-2392ca1944dd4ba7a8111200e13598f3`.
Report SHA-256:
`34d25fc9820bba74f29e50c87723d61664c0fee47e8466207ffb543b386fcad1`.

Its no-retention arms rejected all12 updates per seed and their final parameter
identities equal the recorded shared initial identities. Use these admitted
98-array snapshots as the exact saved initial weights/buffers/RNG; do not
rerun the ablation. Snapshot catalog SHA-256 values:

| Seed | Source selector | SHA-256 |
| --- | --- | --- |
|101|`101-no_retention/final-state.npz.json`|`8749121ef143cbf811ffebac8fe04643c131aca9bea97ab588b81f3c5e1df41d`|
|103|`103-no_retention/final-state.npz.json`|`812f80f6fa9aacb3d5cd83ca65802a7114208d2cd8de956a1db0844fd7cf7acf`|

Bind full archives/catalogs and their role by the accepted E.8R evidence manifest.
Construct the identical component architecture twice using original seeds,then
strictly copy every saved parameter/buffer and restore saved CPU Torch/NumPy
RNG. The common saved RNG is verified across seeds and restored to the caller
after execution. No checkpoint pickle loading and no old optimizer restoration:
these initial snapshots have no Adam payload. Verify exact numeric round-trip,
registration inventory,shapes,dtypes,values and complete parameter identity.
Fresh head-only Adam starts at step0.

Both new arms must pass exact initial six-point physical parity, four-site
physical/RGB/loss parity,and saved initial head/shader signal/gradient parity
against E.8R **before either starts updating**. The all-rejected ablation's
first saved round contains the initial training RGB; original Adam captures
provide the actual initial graph observations. No extra model queries for these
checks: use the admitted initial probe and cached first forward/backward.

## Trainable set and frozen-gradient handling

Only the same15 tensors permitted by guided E.8R may change:
`{mx,my,metallic,rotation,kd}_predictor.predictor.6.{bias,weight_g,weight_v}`.
This is2064 scalar parameters. Freeze the other69 tensors(1,483,520 scalars)
and all12 calibration buffers; preserve all module/weight-normalization objects,
values and forward behavior. Filter Adam parameters in original registration
order. Check frozen gradients remain None, values/buffers exact, and frozen
parameters have no optimizer state after every update.

Use original Adam LR=.001,betas(.9,.999),eps1e-8,decay0,other accepted Torch
defaults; no clipping,scheduler or regularization. Apply the original backward
and step routines through an explicit trainable-parameter view so their
missing-gradient/state checks cover the correct subset. This does not alter
loss derivatives or Adam equations.

The original full-network graph observer expects trainable trunk activations.
E.9 passively observes connected final logits/sigmoids and actual shader inputs
instead. Frozen features/hidden inputs must lack gradients; this is expected,
not a scientific missing-gradient failure. Retain and check actual gradients
for all trainable heads/shader nodes. No renderer,shader or frozen source edit.

Accumulate four derivatives of loss_i/4 in fixed original location order,then
one Adam return. Reuse the initial accumulated derivative for update1. Preserve
all states/diagnostics,including unsuccessful recovery and zero/saturated channels.

## Fixed budget, evaluation and fair comparison

Exactly37 successful Adam returns per seed; no extension/early favorable
selection. Evaluate at states12 and37 using the same E.8R independent tapes
and all six locations. Perform the same terminal four-site gradient measurement
without another update. Initial/state12/final six-point probes are the only
extra field queries. Initial-state restoration and reference verification do
not execute the neural field.

| Total new work across two arms | Exact cap |
| --- | ---: |
| Shader forwards | 328:304 fit +24 external |
| Renderer backwards | 304 |
| Adam returns | 74 |
| Extra field probes / original constructor-calibration passes | 6 /2 |
| Physical forwards/Jacobians/proposals/projections/solves | 0 |
| Reference integrations/directions/rank measurements | 0 |

Per arm C=692,using E.8R's proxy2*(fit forwards+reverse calls)+7*external
forwards. State12 C=234,with its evaluation charged; fixed final C includes
initial/terminal derivatives and both evaluation sets. Thus compare head-only
and saved full-network Adam at equal37 returns/C692,and saved guidance at
C698(.86% more). No different endpoint is needed. Frozen backward graphs cost
less than full-network graphs,so equal updates/C are not equal FLOPs or wall
time. Measure new per-arm and whole-run wall/CPU time; do not fabricate missing
historical CPU timings. Report constructor/restore/probe costs separately.

Fresh snapshots include all84 parameters/12 buffers/RNG plus Adam states only
for15 trainable tensors(143 numeric arrays). Label them head-only continuation,
not interchangeable with full-network Adam checkpoints.

## Unchanged data, gates and resources

Before scientific imports,run the accepted E.8R NumPy preflight on all36 tapes;
require identical corrected hashes and its single declared float32 endpoint
mapping. Bind all saved sources/evidence and cached references by hash. Any
invalid/missing artifact or parity mismatch stops without alternate samples,
reinitialization,looser tolerance or retries. No extra reference/rank budget.

Retain accepted CPU float32/Python3.9.12/NumPy1.21.5/Torch2.4.0+cu121,one
Torch/BLAS thread,hidden GPUs. One foreground launch,watchdog1800s/10s kill
grace,120-second setup/admission/arm stages,4 GiB RSS,8 GiB address space,
1 GiB outputs and≥2 GiB free. Fresh `material-e9-<uuid>` under
`/home/isavand/paper/evidence`. Keep scientific completion,successful cleanup,
terminal identity/counter checks,seal/external exit and failures distinct.
No model or renderer work during preparation; tests use small synthetic modules.

## Analysis and publication decision

Reuse all E.8R comparison endpoints; do not rerender them. Report every seed/site
Q,sorted widths,identifiable orientation,RGB albedo/metallic errors,original full
recovery checks(Q≤.05,angle≤5°,max albedo≤.02,metal≤.03,held-view relative RGB
L2≤.01),mean/worst train and spatial-holdout outcomes,fit/held-view/relight RGB,
component compensation,gradients/saturation,spatial variation and measured cost.
Include state12 context,objective-versus-C plots and all fixed final endpoints.

If guidance beats this matched-active-set control,identify the mechanism and
smallest controlled next step. If head-only Adam matches/beats guidance,do not
promote the current hybrid; prioritize correction quality/sensitivity/damping.
If all fail,separate finite duration,wrong physical directions and spatial
generalization without automatically extending training. Two seeds/one synthetic
field do not establish causal completeness,novelty or real-object performance.

The final report will provide a **design-only** publication roadmap for accepted
G→A:ground-truth synthetic anisotropic objects,geometry/normal controls,fair
inverse-rendering baselines,material/relighting metrics,essential ablations
and real-object requirements. Rank future work by impact/novelty/cost/feasibility.
No additional experiment or G→A deployment is authorized by this control.

```bash
python3 -m experiments.spatial_head_only_e9_v1 --describe
python3 -m experiments.spatial_head_only_e9_v1 --prepare
python3 -m experiments.spatial_head_only_e9_v1 --verify-sha256 FROZEN_SHA256
```

Freeze the new sources/tests/protocol,exact input identities and actual JUnit/
process evidence. Commit preparation for an exact clean execution identity,
then request ONE explicit scientific authorization for the two arms. Approval
covers one bounded launch,offline analysis/figures,final commit and one ordinary
push attempt. No automatic retries,other arms,hyperparameter tuning or expansion.
