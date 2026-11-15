# E.12 — admitted v3 Bell, two original Adam returns

The researcher approved the existing v3 package as a **known-light weak-cue oracle
control**. This does not admit it as the sufficiently informative primary paper
benchmark. Generation/reference/labels/splits remain unchanged; E.11D–G and the
original pending candidate remain immutable. There is no dataset regeneration.

The separate admission is
`/home/isavand/paper/evidence/bell-oracle-e12-admission-v1/oracle-admission.json`,
SHA256 `f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605`.
It retains the exact512 dataset records, generation hash
`5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7`, labels hash
`2baab9c6b49f1d6074b3ada01c94c39c31f0b71b9fd317c5d12a3eb48d6a9cdd`, oracle
mesh hash `4ea98341ca915cf55f2b37581556583308cc50a8e2d2d8d48140d49224866e01`
and explicit researcher approval. The candidate hash remains
`80c47aae7cd1d9e5b8757527b3b929583f140916884e8c7d39fa3da416f61af6`.
Only the new manifest sets researcher_reviewed_generation=true. A dedicated
read-only admission mount avoids writing into the original geometry/dataset.

## One separately approved native execution

Use existing `run_stage_experiment.py --stage material --arm C
--material-variant A_corrected_core_v1 --material-smoke`, schema1
`configs/experiments/material_a_bell_core_known_v1.yaml`, exact original A source
`e1e859779301faced46d6476691e20b546c27ae0` and unchanged E.10 native derivative.
The known environment is **v3's original environment**, not L2. No setting is
copied from the later observability studies. No original or derivative science,
YAML, optimizer, renderer or training loop is edited for this smoke.

The existing original Trainer and TwoUpdateTrace own the stop after the second
successful original optimizer return. Steps0/1 use learning rates0/5e−7;
fresh Adam eps1e−4, betas(.9,.999), AMSGrad, weight decay0, clipping.7 and original
warm-up/cosine settings remain. Preserve seed6033,512 training rays,512 diffuse
and1024 specular samples, original sampling/RNG, worker_num8, all neural heads
and the configured disabled material priors. No hidden regularization is enabled.
Known-light outer/inner/human networks are explicitly inactive, not missing
material gradients. Do not change declared budget2000 or schedules to implement
the bound; the external reviewed observer stops the unchanged loop.

Require construction, original RGB loss and corrected anisotropic forward/backward,
finite active mx/my/rotation/metallic/kd/shared gradients before/after clipping,
Adam moment/step advancement after each successful return, zero-rate first-step
behavior and nonzero second-step displacement. Preserve all failures. The run
must stop before validation, checkpoint writes or final NVS. No initial checkpoint
or implicit resume, extraction, relighting, third update or extra shader probe.
Two-update success is native execution admission, not recovery/evaluation proof.

The existing observer closes its owned DataLoader iterator, progress and writer;
runtime context exit and terminal source/dataset/admission rechecks precede a
complete stage report. Inspect actual cleanup records, finite diagnostics and
counter/state identities rather than inferring success from exit code alone.

## Environment, outputs and exact authorization

Corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
BVH binary `800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5`;
one A10080GB PCIe GPU0,4 CPU threads/CPUs,64GiB host RAM/no swap,pids128.
One foreground900-second watchdog/30-second grace, inner840s/20s. GPU memory
use is observed; this existing native path has no imposed allocator-fraction cap.
Resource ceilings do not alter worker_num8 or the original scientific config.
No network, non-root, read-only source, benchmark/datasets, geometry, admission,
historical inputs and root filesystem; writable outputs/cache exclusively owned
by the fresh run. No image rebuild, install, Docker permission changes or bypass.

Proposed fresh batch:
`/home/isavand/paper/evidence/material-core-e12-smoke-43dc4f8845b14a58b91674c4eaa842c0`.
Run ID: `bell-core-e12-smoke-43dc4f8845b14a58b91674c4eaa842c0`.
The existing mounted RunStore uses its new empty `run/` child. Preserve console,
launch/exit/container receipts and stage-run.json, including partial failure state.
Do not copy caches, dataset views or model-size payloads into Git.

ONE explicit launch approval must bind the final commit, this freeze/admission,
image, run/output IDs and limits. Admission approval alone does not authorize it.
After launch approval complete this one smoke and file-only review without an
intermediate engineering milestone. If Docker remains restricted in Codex,
provide the identical authorized researcher-side foreground tmux command.
No retry, tuning, full training or2000-update comparison follows automatically.

## First three-arm experiment after native admission

Keep A_original, A_estimator_v1 and A_corrected_core_v1 on the same admitted
v3 control, oracle mesh, known light, observations, source family and initialization
policy. The energy-budget family stays separate. Before authorizing a bounded
three-arm comparison, explicitly freeze seeds/work, parameter and checkpoint
variant identities, held-view/relighting protocol and physical metrics:
gauge-invariant Q error, ordered width errors, orientation modulo physical
equivalences only on declared eligible truth, linear albedo/metallic errors,
held-view/relighting linear RGB and native sRGB errors. Record all arms and failed
runs. A smoke proves neither checkpoint roundtrip nor full material export and
relighting correctness; admit those interfaces within the next bounded comparison
before using them for conclusions. Native gradients can be valid even when weak
observation cues prevent physical recovery. See the separately prepared
[two-light diagnostic](BELL_TWO_LIGHT_INFORMATION_E12_V1.md).
