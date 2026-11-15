# E.12B — corrected-core native two-update smoke: PROCEED

The one approved E.12B launch completed exactly **two original Adam returns**
through the pinned Trainer.run. Lazy method binding occurred once, after original
network initialization. Required anisotropic/RGB forward, backward, active
gradients, Adam state, displacement and owned cleanup gates passed.
**PROCEED applies only to native corrected-core two-update execution**, not
physical recovery, long-run stability or the other two comparison arms.

Source commit: `8247fe3a5107577660c7a211294b639deb897280`.
[Frozen smoke protocol](protocols/BELL_MATERIAL_SMOKE_E12B_V1.md), SHA256:
`54e53195734bfef17275d366f44e894a6c42ab60a8aa4b96dbb71adcddfa8d58`.
Native stage report SHA256:
`09ff4545f0e4b1ac973003c393fe34897bd7a048a907802739a5fb6173eddbe2`.
Execution authorization SHA256:
`caefbee61d23136e4afc88fa63bcae33ef179d437d2ae60fed797a9fd1292a3f`.
Admitted v3 manifest SHA256:
`f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.

The original run remains at
`/home/isavand/paper/evidence/material-core-e12b-smoke-94fdfd743da34558a6d8be81c8e78aa2`.
File-only acceptance/CSVs:
`/home/isavand/paper/evidence/material-core-e12b-smoke-94fdfd743da34558a6d8be81c8e78aa2-analysis-v1`.
The [evidence manifest](evidence/BELL_NATIVE_MATERIAL_E12B_RESULTS_V1.json)
records raw report/receipt/output identities; raw caches and tensor metadata
archives stay outside Git. All failed E.12 evidence and its old observer archive
remain unchanged. No native launch, checkpoint loading or scientific evaluation
was repeated during analysis.

## Actual native construction and numerical evidence

The renderer and shader were constructed by original A `_init_network` using
the explicit A_corrected_core_v1 derivative, base
`e1e859779301faced46d6476691e20b546c27ae0`. The observer records guards before
run and exactly one method binding. Both returns have one entered/returned call
for the shader forward, anisotropic forward, mixed shading, anisotropic sampling,
tangent frame, radiance, renderer RGB loss and NeRFRenderLoss. The original
Charbonnier-on-sRGB loss is unchanged; no linear-MSE objective was substituted.

| Zero-based step | Learning rate | RGB loss before update | Gradient L2 | Adam step,84 tensors | Changed elements |
| --- | ---: | ---: | ---: | ---: | ---: |
|0 |0 |.09208256006240845 |.0732278189643773 |1 |0 |
|1 |5e−7 |.09209322929382324 |.07285904366957856 |2 |493,861 |

All108 initialized parameter tensors are finite. Fresh Adam initially has no
state; the84 active tensors acquire/advance first moments, second moments and
AMSGrad maxima at both returns. Betas(.9,.999),eps1e−4,weight decay0, AMSGrad,
seed6033 and original warm-up/cosine are preserved. Gradients are below the
unchanged .7 clipping limit; recorded before/after norms are equal. Ray cursors
advance512→1024 with512 selected hit rays per update and unchanged512/1024
diffuse/specular sample settings.

Step0 parameter snapshots are exactly identical. On step1, all six active groups
change; aggregate displacement L2 is3.4180685981643005e−5 and maximum absolute
float32 element displacement4.991888999938965e−7.

| Group | Gradient L2,step0 | Gradient L2,step1 | Step1 changed elements |
| --- | ---: | ---: | ---: |
|mx |.00438795 |.00418797 |57,907 |
|my |.00384745 |.00357274 |56,979 |
|rotation |.000977320 |.000138490 |24,212 |
|metallic |.0176724 |.0177694 |95,917 |
|kd |.0707765 |.0704037 |110,272 |
|shared features |.00238133 |.00238270 |148,574 |

Each material head has12 gradient-bearing parameter tensors; shared features
have24. Known-light outer/inner networks each have12 parameter tensors but no
gradients, state or displacement, as intended by known_env_v1/zero_bounce_v1.
Human-light parameters are absent because human_lights=false. These are expected
inactive groups, not unexplained missing gradients. The configured reg_mat=false
prevents material-prior computation; the registered mat_reg loss wrapper returns
without adding a loss. No prior was silently enabled to obtain finite gradients.

Both losses above are observed **before** their optimizer return. Because step0
has LR0, both forward passes use the initial weights, with different ray batches
and sampling draws. Their small difference is not evidence of improvement or
regression after optimization. The smoke stops immediately after the first
nonzero-LR weight change and makes no post-update rendering query.

## Identity, cleanup and resource limits

All69 new frozen files, actual imported pinned scientific modules, both scientific
snapshots,512 native dataset records, oracle mesh/generation/labels/admission
and six recorded native output hashes verify. Parameter snapshot continuity,
gradient finite/norm/group metadata, per-parameter Adam step/moment shapes and
weight-displacement counts were checked from the saved report. These are
verification of recorded detached statistics and hashes, **not independent
recomputation of raw tensor values or backpropagation**. No raw tensor/pickle or
TensorBoard event was deserialized.

Exit0; `status=complete`; phase `bounded_stop_after_second_optimizer_return`;
`cleanup_errors=[]`. The reviewed TwoUpdateTrace closes owned iterator/workers,
progress and TensorBoard writer before returning; the stage reaches complete
only after runtime-context exit and terminal source/input checks. No checkpoints
or validation/NVS images exist. Docker `--rm`/foreground exit are recorded;
independent engine/process inspection remains unavailable inside Codex. The
generic CLI's final G R/R-repeat/C reminder does not convert this oracle material
smoke into G training evidence or invalidate its specific two-update result.

Recorded stage duration17.165555s, including construction/observer/metadata work.
Observed step intervals .559781s/.764434s include CPU snapshot/report overhead;
they are not clean full-training throughput benchmarks. Peak Torch CUDA allocated
306.674MiB/reserved338MiB, excluding non-Torch allocations. The launcher enforces
64GiB host RAM,4 CPUs and900s; host peak RSS was not separately recorded and is
not inferred from its limit. Runtime: Python3.8.10,Torch1.8.1+cu111,CUDA11.1,
ABI0,A10080GB PCIe,driver550.163.01,fork workers,4 Torch threads. Existing TF32/
cuDNN settings remain recorded, not changed or represented as strict determinism.

Corrected image:
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`;
binary SHA256 `800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`.
The console's original normal-negation warning is preserved; no normal flip,
mesh repair or convention change was made. Geometry remains the admitted
benchmark-defined oracle, not measured truth or attributable G reconstruction.

## Next bounded comparison, prepared only

The [three-arm proposal](protocols/BELL_THREE_ARM_RECOVERY_AFTER_E12B_V1.md)
uses the already admitted weak-cue v3 control, existing paired configs and one
seed6033. It requests no additional smoke cycle: A/B initialization/gradient
checks should be inside their proposed first histories. All arms' physical
recovery, checkpoint/export/evaluation/relighting interfaces and2000-step behavior
remain unvalidated. No full training, checkpoint evaluation, relighting or
two-light rendering was run or authorized by smoke success.

Eight new tiny metadata-reporting tests pass,0 failures/errors/skips, exit0.
They check zero-LR and second-return evidence, six corruption cases and an import
blocker. The ten accepted lifecycle fixture cases were not rerun. Python3.8
grammar, Markdown links, source identities and exact small-file copies pass.

File-only regeneration, with a fresh output directory:

```bash
python3 -B -m experiments.verify_bell_material_smoke_e12b \
  --batch /home/isavand/paper/evidence/material-core-e12b-smoke-94fdfd743da34558a6d8be81c8e78aa2 \
  --output /absolute/new/verification-directory
```
