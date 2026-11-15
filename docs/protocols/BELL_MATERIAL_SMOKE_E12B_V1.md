# E.12B — corrected lazy observer, same two-update native material experiment

Base `0ea9630eb7cad08a961370f87366282cd812f1f6`. Only the observer lifecycle
changes; no native renderer/shader, frozen G/A, scientific variant, optimizer,
loss, dataset, YAML or RNG behavior changes. The prior failed run, authorization,
E.12 freeze and reports remain unchanged. No retry is permitted by its consumed
authorization. This fresh protocol requires one new explicit launch approval.

## Narrow implementation correction

`CorrectedProbe.prepare_calls` uses the already selected renderer registry class
to forbid validation, NVS, material extraction, checkpoint and evaluator bodies
before the original Trainer.run starts. It uses the existing inspect.unwrap
guard policy and initially empty training-call map; it does not access a network
instance, allocate a model or run initialization.

At the existing initialized observer boundary, after the original `_init_network`
and before `_init_logger`, bind the actual selected network/shader method identities.
Require their presence and selected renderer type, exactly one binding, and no
loss of early forbidden guards. All later training boundaries require that bind.
The existing numeric initialize, call-count, RGB, gradient, Adam-state, displacement,
two-return and resource-cleanup checks remain in place. The observer never calls
an original initialization method. Missing/repeated boundaries fail explicitly.

One shared CPU toy fixture exercises the actual pinned Trainer.run body and
unchanged TwoUpdateTrace, with no real BVH, shader or CUDA model execution. Tests
cover absent network after construction, original single initialization, no RNG
use during guard preparation, five early forbidden operations, missing objects/
boundary, real toy forward/backward/Adam checks, exact two-return stop, repeat-bind
rejection and cleanup after a pending-backward failure. A frozen torch.cross
deprecation warning is recorded, not repaired.

## Preserve prior source identities

The old observer SHA256 is
`9e54e8232f938e81ff41ce6bd16f4bd42cbb97487d8348074780b8c5abda9818`.
Its bytes are preserved at commit `0e3706cf82d8eea0e37fcec0eedd2e7472923263`
and separately under
`/home/isavand/paper/evidence/material-core-e12-smoke-43dc4f8845b14a58b91674c4eaa842c0-source-archive/`.
Historical E.11D–G/E.12 manifests are not rewritten. Their observer hash refers
to the old source revision/archive, not the corrected current working tree.
All their other source hashes and both complete scientific snapshots remain
unchanged. Replaying a historical full source check requires its original
checkout, not silently accepting the new observer under its old hash.

## Same admitted data, science and resources

Retain admitted v3 weak-cue oracle control, manifest SHA256
`f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`,
original dataset/mesh/labels/environment/splits and corrected BVH. No renewed
dataset admission or generation is necessary. Original A source:
`e1e859779301faced46d6476691e20b546c27ae0`; variant A_corrected_core_v1.
Use the unchanged `configs/experiments/material_a_bell_core_known_v1.yaml` and
`run_stage_experiment.py --material-smoke`. This uses v3's known environment,
not L2. The separate two-light information proposal remains unexecuted.

Exactly two successful original Adam returns at steps0/1, LR0/5e−7; seed6033,
512 training rays,512 diffuse/1024 specular samples, Adam/AMSGrad eps1e−4,
betas(.9,.999), weight decay0, clipping.7, original warm-up/cosine and worker_num8.
Keep configured disabled priors and known-light inactive outer/inner/human groups.
Require finite active material/shared gradients, moment/step advancement at both
returns, zero first-step and nonzero second-step weight displacement and owned
cleanup. No validation, checkpoint writing/resume, NVS or extra shader probes.

Use the existing corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
GPU0/A10080GB PCIe,4 CPUs/threads,64GiB host RAM/no swap,pids128, no new GPU
allocator cap. One foreground900s watchdog/30s kill grace, inner840s/20s.
Read-only source/datasets/geometry/admission/historical inputs/root filesystem,
non-root/no network; only fresh owned outputs/cache may be written. No build,
installation, permission change or security bypass. If Codex Docker remains
restricted, provide the identical approved foreground tmux command.

Fresh batch:
`/home/isavand/paper/evidence/material-core-e12b-smoke-94fdfd743da34558a6d8be81c8e78aa2`.
Run ID: `bell-core-e12b-smoke-94fdfd743da34558a6d8be81c8e78aa2`.
The failed E.12 batch is never reused or edited. Source/freeze/admission/image,
run ID and resources must be bound by the new approval. Approval covers one
launch and file-only analysis/report/commit/push, no automatic retry or tuning.

On success mark only the native two-update scope PROCEED and prepare the first
matched A_original/A_estimator_v1/A_corrected_core_v1 recovery protocol using the
already admitted control. Preserve all physical Q/width/eligible-orientation/
albedo/metallic and held-view metrics, distinct from photometric fit. Do not
execute full training, a2000-update comparison, checkpoint evaluation, relighting
or the two-light study. On failure preserve exact phase/work and stop.
