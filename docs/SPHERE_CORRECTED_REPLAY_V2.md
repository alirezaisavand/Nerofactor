# Sphere replay v2: outgoing-direction interface correction

The first authorized run at source `6661e363ec49e0d339625cf34c461a415f08a727`
failed on its first training forward with `Unit views required; no normal repair`.
Exit1; **zero Adam returns**, no checkpoints or reconstruction maps. The saved
report records72.108s native wall time, peak RSS34426680KiB, cleanup_errors=[],
and cleanup_verified=false. Do not upgrade that last flag to successful full
cleanup. Original run, authorization and freeze remain unchanged under
`/home/isavand/paper/evidence/sphere-core-8c2c1feb8b6449678ac18f7460c9b261`.
See [the failure identities](evidence/SPHERE_CORRECTED_REPLAY_FAILURE_V1.json).

## Confirmed cause and bounded correction

Pinned A's `_construct_nerf_ray_batch` builds directions
`[(u-cx)/fx, -(v-cy)/fy, -1]`, rotates them by the camera-to-world rotation and
stores them without normalization. `train_step` and `test_step` pass their
negations to `shade`. Neither original anisotropic_forward nor its shader
dispatch normalizes those outgoing views. Corrected core explicitly requires
unit outgoing vectors before reflection, angular factors and PDFs.

An offline calculation of the original1080² camera-coordinate grid gives ray
lengths1.0…1.122140884;1166263/1166400 pixels deviate from1 by more than1e-5.
These are derived grid values, **not captured first-training-batch measurements**.
The first batch's actual direction tensors were not saved. The traceback and
source path establish the interface mismatch; this error does not establish a
surface-normal orientation defect.

V2 normalizes **only the outgoing argument passed by the sphere adapter's
`shade` to corrected core**, for both training and all validation/NVS calls.
Use an out-of-place division by vector length; reject nonfinite/zero vectors
without an epsilon fallback. Preserve tensor dtype/device, direction and sign;
do not modify raw camera rays, camera calibration, stored batches, BVH inputs,
depth/position semantics, normals, masks, splits or pixel ordering.

This is an explicit numerical interface correction relative to historical A's
raw shader inputs, not numerical parity with historical A and not an observer-only
change. It supplies the physical unit-direction representation already required
by `A_corrected_core_v1`; its frozen implementation, unit tolerance, BRDF,
sampling, loss, optimizer, scheduler and regularization remain unchanged.
No normal flipping/filtering, invalid-view deletion, relaxed validation, new
random draws, new model probes or shader modification is introduced.

All [v1 replay settings and visualization conventions](SPHERE_CORRECTED_REPLAY_V1.md)
remain, including original learned lighting, full network,100000 intended Adam
returns,1080² images, validation ID36 every2000 and five original NVS views
every10000. The adapter records its interface identity and initial raw view-norm
range separately for training/evaluation. The failure of the first launch was
preventable; its focused fixtures did not previously supply nonunit views to
the actual corrected unit-domain guard. New CPU regressions exercise that exact
guard through a fake shade boundary in both modes, preserving arguments/RNG and
rejecting zero/NaN/Inf/wrong-shaped views before any shader call.

## Fresh execution boundary

Preserve v1 freeze and validation reports. Their source identities refer to the
historical v1 Git commit; the corrected working tree must not claim to validate
as that old source. The launcher now requires the separately versioned
`docs/protocols/SPHERE_CORRECTED_REPLAY_V2.freeze.json`, revision2 and interface
`sphere-unit-outgoing-at-shade-v1`. A fresh bound approval and output directory
are mandatory; the first authorization is consumed. No native retry is made
while preparing/testing this correction.

Runtime/work/resource ceilings are unchanged: oneGPU0/A100,4 CPU threads,
original8 loader workers,128GiB RAM/no swap,256PIDs,64GiB cooperative output
guard,86400s foreground watchdog/30s grace and85800s inner timeout. No new GPU
allocator cap. Stop on any further numerical/domain/resource failure and preserve
the partial result; no automatic repair, retry, extension or implicit resume.
Unit normalization alone does not establish valid hemisphere-facing normals,
native convergence or reconstruction quality. Those remain real execution gates.
