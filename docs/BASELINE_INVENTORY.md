# Baseline inventory

Recorded on 2026-11-03 on `marineris` for milestone 1D.1. This is a lightweight
inventory, not a full content-integrity manifest or reproduction report. Names,
counts, file sizes, small text/JSON files, and Git objects were inspected without
moving or modifying original artifacts. Large datasets were not hashed;
checkpoints were not deserialized; no training or evaluation was run.

## Repository baseline

Repository: `/home/isavand/paper/Nerofactor-paper`.
Active branch before edits: `paper/infrastructure`.
Working tree before edits: clean, including no untracked files.
Baseline commit subject: `reduce total steps`.

| Branch or ref | Exact commit at inspection | Availability and role |
| --- | --- | --- |
| `paper/infrastructure` | `e1e859779301faced46d6476691e20b546c27ae0` | Active development branch, initialized at anisotropic baseline |
| `anisotropic` | `e1e859779301faced46d6476691e20b546c27ae0` | Local historical branch |
| `origin/anisotropic` | `e1e859779301faced46d6476691e20b546c27ae0` | Remote-tracking snapshot matching local anisotropic |
| `origin/aniso_geometry_safe` | `79252178e65914ad1acbc14790b1514f276bfea0` | Alternative historical implementation; no local branch at inspection |
| `origin/ablation_anisotropic` | `2519ab9b480f414ba2ae96dd860d8b868c584054` | Historical ablation implementation; no local branch at inspection |

The reflog records checkout from `anisotropic` to `paper/infrastructure` at this
commit. No refs were changed or fetched by this milestone. The hashes identify
the inspected snapshots, not the current state of an independently queried
upstream repository. See [BRANCH_COMPARISON.md](BRANCH_COMPARISON.md) for their
substantive differences.

## Storage roots and accessibility

| Host root | Observed status | Preservation policy |
| --- | --- | --- |
| `/home/isavand/paper/data` | Accessible; three dataset families | Original inputs; no writes or relocation |
| `/home/isavand/paper/final_results` | Accessible; 15 top-level directories | Preserve all historical artifacts in place |
| `/home/isavand/paper/paper_runs` | Does not exist at inspection | Reserved for future isolated experiment outputs |
| `/home/isavand/paper/Nerofactor-paper/data` | Does not exist; not a symlink | Relative code paths currently do not resolve to host inputs |

Readability sampling in the preceding read-only inspection covered images and
camera metadata in each dataset family. All 40 PLY headers in `meshes` and
`meshes-pretrained` were readable. Sample checkpoint bytes were readable without
loading tensors. These checks do not establish complete file integrity,
geometry validity, checkpoint compatibility, or the correctness of a metric.

## Original datasets

Paths below are relative to `/home/isavand/paper/data`. File counts describe
regular-file entries found beneath each scene, not numbers of views or unique
content identities. They are metadata snapshots and can include cached products
already present in the original workspace.

| Family | Scene names with file counts | Family file count |
| --- | --- | --- |
| `GlossyReal` | bear (397), bunny (396), coral (387), maneki (393), vase (393) | 1,966 |
| `GlossySynthetic` | angel (641), bell (641), cat (769), horse (641), luyu (641), potion (769), tbell (641), teapot (769) | 5,512 |
| `nerf_synthetic` | cone (402), cube (403), donut (403), icosphere (303), smoothcylinder (502), sphere (402), tetra (402) | 2,817 |

GlossyReal contains photographs, COLMAP data, object point clouds, and existing
`cache.pkl` files. GlossySynthetic contains image/depth/camera files and existing
`eval_pts.ply` files. NeRF scenes have JSON camera transforms and image/mask
directories; depth coverage is not assumed identical across scenes.

The preceding inspection parsed each NeRF `transforms_train.json`: each has 100
frames, and all referenced training images and corresponding training-mask paths
exist. The first recorded frame path is `train\0001.png`. This establishes
path existence, not full image decoding or semantic validation.

The following NeRF scene names referenced by current configs are absent under
the supplied `nerf_synthetic` root: ball, car, chair, coffee, cylinder,
cylinder_rect, drums, ficus, helmet, hotdog, lego, materials, mic, ship,
simplecylinder, teapot, toaster. GlossySynthetic's `teapot` is a different family
entry and must not be silently substituted. Cone and icosphere exist but have no
dedicated current shape/material YAML files under those exact names.

## Historical geometry artifacts

All following paths are relative to `/home/isavand/paper/final_results`.
Filenames provide labels only: `300000` is not independently verified training
progress, and `NeRO`, `pretrained`, or `gt` does not prove producing code,
alignment correctness, or scientific provenance.

| Path pattern | Scene names or files | Count |
| --- | --- | --- |
| `meshes/<scene>_shape-300000.ply` | angel, bear, bell, bunny, cat, coral, cube, donut, horse, luyu, maneki, potion, smoothcylinder, sphere, tbell, teapot, tetra, vase | 18 |
| `meshes/NeRO/<scene>_shape-300000.ply` | cube, donut, smoothcylinder, sphere | 4 |
| `meshes-pretrained/real/<scene>.ply` | bear, bunny, coral, maneki, vase | 5 |
| `meshes-pretrained/syn/<scene>.ply` | angel, bell, cat, horse, luyu, potion, tbell, teapot | 8 |
| `meshes-pretrained/nerf/<scene>.ply` | cube, donut, smoothcylinder, sphere, tetra | 5 |
| `ablation/smoothcylinder_shape-100000.ply` | One ablation-labeled geometry artifact | 1 |
| `cropped-meshes/bear_shape-300000.ply` | One cropped bear mesh | 1 |
| `glossy-real-meshes-gt/<scene>-align.ply` | bear, bunny, coral, maneki, vase | 5 |
| `eval_pts/` | `eval.ply`, `eval_gt.ply`, `eval_pts.ply` | 3 |

Total: 50 PLY files under the historical results root. No mesh contents were
changed, sampled, transformed, or regenerated. Meshes with matching sizes or
scene names have not been proven identical.

## Historical checkpoints

There are 26 `.pth` files under `final_results/model`, in 13 directories. Each
directory below contains both `model.pth` and `model_best.pth`; the byte sizes are
listed separately. All observed checkpoint directories are geometry-labeled.

| Directory relative to `final_results/model` | `model.pth` bytes | `model_best.pth` bytes |
| --- | ---: | ---: |
| `NeRO/cube_shape` | 27,122,925 | 27,122,925 |
| `NeRO/sphere_shape` | 27,122,641 | 27,122,385 |
| `angel_shape` | 29,966,820 | 29,966,820 |
| `bell_shape` | 29,966,819 | 29,966,819 |
| `cat_shape` | 29,966,830 | 29,966,830 |
| `coral_shape` | 31,656,621 | 31,656,065 |
| `horse_shape` | 29,966,823 | 29,966,823 |
| `luyu_shape` | 29,966,743 | 29,966,807 |
| `potion_shape` | 29,966,822 | 29,966,822 |
| `smoothcylinder_shape` | 29,967,111 | 29,966,855 |
| `sphere_shape` | 29,967,118 | 29,967,118 |
| `teapot_shape` | 29,966,843 | 29,966,843 |
| `vase_shape` | 31,656,274 | 31,656,274 |

No material `.pth` checkpoints were found anywhere under `final_results`, despite
the presence of material renders and scalar metrics. Geometry checkpoint coverage
also does not cover every scene with an extracted mesh. Saved training steps,
state-dict architecture, optimizer settings, seeds, producing commits, and
checkpoint-to-mesh correspondence remain unverified because checkpoints were
not loaded. Equal byte sizes do not establish equal weights.

## Original results directories and purposes

Counts include all files recursively beneath each listed directory. Purposes
are inferred from names, file types, inspected small files, and matching code
paths; they do not independently attribute an artifact to a historical branch.

| Directory under `final_results` | Files | Observed contents / intended role |
| --- | ---: | --- |
| `ablation` | 61 | One 100k-labeled mesh, 40 NVS PNGs, and 20 validation JPGs |
| `cropped-meshes` | 1 | Cropped bear geometry for historical evaluation |
| `eval_pts` | 3 | Predicted/GT/evaluation-labeled point-cloud artifacts; exact scene mapping unknown |
| `glossy-real-meshes-gt` | 5 | Aligned reference meshes for real scenes |
| `logs` | 8 | Material NVS scalar text logs |
| `logs_NeRO` | 2 | NeRO-labeled material NVS logs for maneki and tbell |
| `logs_ablation` | 1 | Ablation-labeled smoothcylinder material NVS log |
| `meshes` | 22 | Extracted geometry, including four `NeRO` entries |
| `meshes-pretrained` | 18 | Real/synthetic/NeRF meshes used as material inputs in current configs |
| `model` | 67 | 26 checkpoints, 26 text logs, and 15 TensorBoard event files |
| `nvs_imgs` | 40 | Material NVS PNGs for eight scenes |
| `nvs_imgs_NeRO` | 10 | NeRO-labeled material NVS PNGs for maneki and tbell |
| `nvs_imgs_ablation` | 5 | Ablation-labeled material NVS PNGs for smoothcylinder |
| `pbrnerf` | 55 | Five real-scene `metrics.json` files and 50 masked GT/rendered JPGs |
| `train_vis` | 5,571 | Validation JPGs, including `NeRO`, `temp`, cone, and icosphere entries |

## Historical reported results: not independently reproduced

The tables in this section transcribe existing reports. No result here was
generated or independently recomputed during 1D.1. Do not pool the three sources
below or assert matching split/masking/selection protocols.

### Preserved thesis-workspace text logs

Source root: `/home/isavand/paper/final_results`. Each file below has five
per-view lines and one reported average line. Values are copied from the average
line at its original five-decimal precision, not recomputed from rounded views.
The source files remain unchanged.

| Source file | Reported average PSNR | Reported average SSIM |
| --- | ---: | ---: |
| `logs/cube_material/log.txt` | 35.66100 | 0.97995 |
| `logs/donut_material/log.txt` | 34.14081 | 0.95606 |
| `logs/horse_material/log.txt` | 25.55774 | 0.91237 |
| `logs/luyu_material/log.txt` | 28.74506 | 0.94361 |
| `logs/potion_material/log.txt` | 31.79829 | 0.93880 |
| `logs/smoothcylinder_material/log.txt` | 37.11176 | 0.97979 |
| `logs/sphere_material/log.txt` | 35.08154 | 0.95358 |
| `logs/tetra_material/log.txt` | 39.17865 | 0.98317 |
| `logs_NeRO/maneki_material/log.txt` | 30.74613 | 0.97285 |
| `logs_NeRO/tbell_material/log.txt` | 30.65684 | 0.96554 |
| `logs_ablation/smoothcylinder_material/log.txt` | 35.65316 | 0.97221 |

### Preserved PBRNeRF-labeled JSON reports

Source: `final_results/pbrnerf/<scene>/metrics.json`. Each JSON records
`split_type: validation`, seed 6033, NVS IDs, per-image values, and an average.
The existing numeric average fields are transcribed below without recomputation.
Exact producing scripts, renderer settings, and input hashes are unknown.

| Scene | Reported `psnr_masked` | Reported `ssim_masked` |
| --- | ---: | ---: |
| bear | 21.829033579600182 | 0.8420837104562414 |
| bunny | 27.79816040040551 | 0.9675338152944306 |
| coral | 24.742832992042832 | 0.9218072276314867 |
| maneki | 26.25541628586533 | 0.9445801257156571 |
| vase | 24.3041482159984 | 0.9202293334830894 |

### Inherited repository README reports

Source: [README.md](../README.md), `Results (on Blender Datasets)`, at commit
`e1e859779301faced46d6476691e20b546c27ae0`. These are inherited repository
claims, not identified thesis-run measurements or results reproduced on this
workspace. All listed scene names are absent from the supplied NeRF dataset
root. Their relationship to the thesis experiments remains unverified.

| README scene | Reported shape PSNR | Reported material PSNR |
| --- | ---: | ---: |
| chair | 27.73918 | 28.74847 |
| drums | 21.06286 | 24.88227 |
| ficus | 22.51317 | 28.38085 |
| hotdog | 29.33451 | 32.13475 |
| lego | 23.47746 | 25.66081 |
| materials | 24.32323 | 24.8514 |
| mic | 24.54512 | 28.63963 |
| ship | 22.91336 | 26.54597 |
| ball | 39.62966 | 33.66338 |
| car | 26.09878 | 26.9762 |
| coffee | 30.61401 | 33.76237 |
| helmet | 29.56573 | 29.59044 |
| teapot | 35.41234 | 40.28731 |
| toaster | 25.23647 | 27.30664 |

### Independently reproduced results

**None.** No training, checkpoint inference, metric recomputation, or geometry
evaluation was performed in 1D.1. Future reproduced results belong under a new
`paper_runs/<experiment-id>` directory with a manifest and evaluation protocol.
They must not replace the reports above or overwrite `final_results`.

## Known gaps and deferred investigations

- Artifact-to-commit/config/seed mapping is unknown; no complete per-run
  provenance manifest was identified. Folder names alone are insufficient.
- Material checkpoints are absent; extracted mesh coverage exceeds saved
  geometry-checkpoint coverage. Cone and icosphere have validation images but no
  matching main mesh/checkpoint entries in this inventory.
- Generic evaluation point-cloud names lack scene, coordinate-system, and
  sampling provenance. Reference-mesh alignment/cropping procedures and exact
  checkpoint-to-mesh extraction settings need reconstruction.
- Current configs parse as YAML (70 files), but 26 use `your_dataset_dir` and
  eight opacity flags are strings (`True,`) rather than Booleans. Missing example
  datasets and old PointNeRF absolute paths are documented, not repaired.
- Existing loaders write dataset-local caches/images. Split generation resets
  global RNG state. Checkpoints saved by the inspected trainer omit RNG state;
  legacy resume is implicit by model name. These are future design inputs.
- The current synthetic shape evaluator imports a helper absent at the A tip;
  the masked evaluator prints a JSON destination with its write commented out.
  These issues require Step 2 review, not automatic corrections.
- There is no Docker/environment lock specification in the inspected branch
  tips. `requirements.txt` pins only Open3D 0.16.0 and PyMCubes 0.1.4; PyTorch,
  CUDA, extension revisions, and compiler compatibility are not established.
- Docker access previously failed inside Codex. No Docker command, container, or
  permission change was attempted for this milestone. A preceding read-only
  NVIDIA-SMI query could not communicate with the driver; no GPU-runtime
  validation is claimed and no further GPU probe was made in 1D.1.

Future manifests must hash the selected geometry, checkpoints, resolved config,
split definitions, and environment specifications. Define a dataset content-hash
manifest covering the actually used inputs before claiming full reproducibility;
schedule any large full-dataset scan explicitly. Hash absence remains explicit
in this lightweight inventory. See [INFRASTRUCTURE_PLAN.md](INFRASTRUCTURE_PLAN.md).
