# Gate 10.3 v2 — Calibrated camera prefix and completed-extraction reuse

Status: prepared; one new exact commit/freeze/output approval is required. The
failed v1 authorization is consumed. No new native work is authorized by this
correction. Preserve v1's freeze, implementation, output, authorization and all
Gate 1–10.2 evidence. This version changes only the audit camera-prefix execution
device, adds a bounded precision witness, and reuses already completed work.

## Observed failure and limits of attribution

V1 stopped at camera 14's neural/mesh ray gate, before its first neural render.
It completed one 15k CPU checkpoint load, one 512³ field, one extraction, the six
camera geometry audit (196,608 native rays, 384 independent CPU witnesses), and
1,536 raw SDF / 512 gradient consistency witnesses. It additionally completed
16,384 fitting camera rays and four CPU witnesses. No optimizer or material code
ran. The six-view numerical witnesses and SDF gate passed. Intermediate geometry
quality scores remain diagnostic, not admission of an earlier endpoint.

The host exit code is 1 and the container-removal receipt says `absent`. The
native successful-cleanup flag is false because the execution did not reach its
success boundary; do not relabel it a verified complete run. All 44 captured
files are hash-bound in
[the failed artifact manifest](../results/publication_gate103/failure-v1/artifact-manifest.json).

The failed run records CUDA matmul TF32 enabled. V1 did not save the offending
native ray values or maximum errors, so TF32 is a plausible mechanism, not an
established causal finding. CPU replay of the original camera methods against
all 16 real camera archives produces zero maximum origin/direction difference in
the current CPU fixture runtime. This supports a calibrated CPU camera boundary;
it does not establish GPU arithmetic behavior in the pinned PyTorch 1.8 runtime.

## Narrow numerical interface correction

Use original `_construct_ray_batch` **and** `_process_ray_batch` on CPU float32,
including original pixel centers, calibration inversion, rotation, normalization,
near/far and human poses. Select exactly the frozen C-order pixel panel. Save
the resulting rays and original archived rays before checking the unchanged
maximum absolute error `2e-6`. Stop on failure with the witness preserved.
Only then transfer the five returned tensors to CUDA for the original render.

This changes camera-prefix arithmetic placement relative to v1 and historical
GPU training. It preserves calibration and exact source methods, but must not be
described as bitwise replay of historical training rays or sampling states.
Do not change frozen G or disable TF32 globally. Original shader/SDF arithmetic,
sampling policy, checkpoint loop indices, two draw seeds and acceptance criteria
remain unchanged. The two original source camera helpers consume no random tape.

At the first camera only, additionally call original CUDA `_process_ray_batch`
twice on identical selected CPU directions/poses copied to CUDA: first with its
entry matmul TF32 setting, then with matmul TF32 disabled. Save both origin and
direction arrays, CPU/archived arrays, errors and policy settings. Restore the
entry setting in `finally`, including failures, **before any neural render**.
These two diagnostic outputs never select or replace the admitted CPU rays.
This is tensor camera algebra, with zero new SDF, mesh or visibility queries.
All later camera prefixes use CPU without further precision probes.

## Reuse and work envelope

Hash-check the entire failed batch, its exact failure boundary, step and checkpoint
identity, then copy only the completed 15k raw field/extraction/domain/cleanup/
topology/PLY/six-camera rasters/SDF witnesses into the new owned 15k directory.
Preserve bytes and record provenance. Load the 15k checkpoint once for its neural
panel and require its network fingerprint to match the one recorded before the
failed run's extraction. Do not rerun 15k extraction, its six-view audit or SDF
witness queries. Recompute the fitting panel in a fresh directory; the single
partially completed fitting view is not treated as a complete old history.

Retain the accepted Gate 10.2 300k reuse exactly as before. Only 30k and 50k need
new extraction and six-view/SDF evaluation. New work ceilings are therefore:

| Work | V2 maximum |
| --- | --- |
| CPU checkpoint loads | Four, once each, the same four authenticated files |
| New dense grids / tiles / SDF nodes | 2 / 1,024 / 268,435,456 |
| New marching-cubes calls | 2 |
| New six-camera native rays | 393,216 |
| Fitting native rays | 7,995,392, all 122 cameras at all four steps |
| New independent CPU witnesses | 1,024 |
| New consistency SDF / gradient points | 3,072 / 1,024 |
| Additional CUDA camera-prefix diagnostic calls | 2, first camera only |
| Neural renders / rays | 128 / planned 32,440, ceiling 32,768 |
| Neural SDF forward / gradient slots | 35,127,296 / 10,485,760 |
| Appearance calls / point slots | 128 / 5,242,880 |
| Original occlusion origins | 196,608, original 64+16 probes |
| Optimization / checkpoint writes / Material A calls | 0 / 0 / 0 |

All remaining limits, 512³ resolution, fully-inside-cell domain, exact float32
cleanup, signed normals, original quality criteria and reporting definitions
remain those of the
[v1 protocol](BELL_GEOMETRY_EVOLUTION_GATE103_V1.md). The same four checkpoints,
122 fitting views, fixed 16-camera/two-draw panel and 300k publication endpoint
remain fixed. The imperfect-segmentation rationale and candidate hierarchy in
[the clarification](../results/publication_gate103/MASK_SUPERVISION_CLARIFICATION_V1.md)
remain authoritative. No geometry-loss intervention is implemented here.

Use the same corrected image
`sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331`,
A100 GPU0, four CPUs, 128 GiB RAM/no swap, 128 PIDs, 75% CUDA allocator,
8 GiB outputs, foreground 3,600s watchdog/30s grace, inner 3,540s. Inputs/source/
root read-only, non-root, no network or installs. Same accepted original JIT
cache validation policy. Copying reusable arrays is included in output/time
limits. No retries, training, threshold changes or geometry admission.

## Verification and interpretation

Before approval, run only focused CPU tests, file hashes and source/shell syntax
checks. No checkpoint payload load, CUDA, new SDF query or extraction.

After the approved one-time run, verify all new and reused identities, exact
reuse maps/topology, raw/SDF/raster replay and camera-prefix numerical witnesses.
Report default-GPU versus TF32-disabled differences without claiming that a
single backend witness diagnoses all numerical behavior. Preserve every geometry
quality failure and excluded observation. Earlier geometry scores remain
development evidence; no intermediate checkpoint replaces 300k.

Stop on identity, numerical or resource failure without retry. If complete,
continue the originally frozen checkpoint-evolution, volume/mesh and mask-loss
interpretation, with reused versus new work reported separately. No G→A handoff
or material training follows automatically.
