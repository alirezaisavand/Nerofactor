# Dataset and cache isolation: 1D.2B-1

Follow-up: 1D.2B-2A connects this opt-in API to the separate geometry entry point.
1D.2B-2B adds the material renderer/validation integration; see
[MATERIAL_TRAINING.md](MATERIAL_TRAINING.md) for the material writer inventory.
See [GEOMETRY_TRAINING.md](GEOMETRY_TRAINING.md) for the current writer inventory,
CPU evidence, and native-execution blockers. The scope and test counts below
describe the historical 1D.2B-1 milestone.

Scope: opt-in dataset loading and derived-artifact destinations, from approved
commit `2f40ca311950a6323727508ac6470f3a21beb768`. Original inputs and historical
results are preserved. Training entry points, renderers, losses, optimizers,
sampling, legacy YAML, and evaluation protocols are unchanged. No real experiment,
Docker container, GPU work, or expensive evaluation was started.

## Writer inventory

Legacy destinations describe existing behavior, not authorization to write the
original workspace. Distinguish dataset-construction writers from standalone
preprocessing/export tools and from general training/evaluation outputs.

| Owner / operation | Legacy writer and destination | Isolated-mode destination / status |
| --- | --- | --- |
| `GlossyRealDatabase._parse_colmap` | `save_pickle` writes `<scene>/cache.pkl`: poses, intrinsics, image names, IDs | Same payload in scene-specific experiment cache; existing source product is read-only |
| `CustomDatabase._parse_colmap` | Same camera-cache write | Same routing; existing abstract-class defect is preserved |
| Real/custom raw-resolution constructors | `mkdir`/`imsave`: `<scene>/images_raw_<size>/<image>` | Experiment cache with unchanged product names and encoding suffix |
| Real/custom `_crop` | `mkdir`/`imsave`: `<scene>/images_<size>/<image>`; `save_pickle`: `meta_info.pkl` with cropped poses/intrinsics | Experiment cache; existing source-side products remain read-only cache hits |
| Real/custom `get_image` | Reads processed images under scene | Reads matching experiment products first, otherwise existing source products |
| Real `get_mask` | Reads `<scene>/images_<size>/masks_train/<name>.png` | Remains an original input; no copying, resizing, or mask-generation change |
| `GlossySyntheticDatabase` | Reads images, camera pickles, masks, depth; no active construction-time writer | Same reads; new identity metadata is outside the dataset, preserving `*.pkl` view counting |
| `NeRFSyntheticDatabase` | Reads train transforms, images, masks, depth; no construction-time writer | Same selection/calculations with contained input paths |
| `get_database_eval_points_nerf` | Open3D writes `save_path`, default `data/eval_gt.ply` | Scene cache `evaluations/<parameter-digest>/<save_path>`; point calculations/protocol unchanged |
| `DummyDataset` | Delegates validation database construction; no own writes | Optional preprocessing storage forwarded; legacy call remains distinct |
| `run_colmap.run_sfm` | Project directory, SQLite `database.db`, `sparse/`, `dense/`, fused `points.ply`, delegated COLMAP products | Already accepts separate `image_dir`/`project_dir`; CLI still co-locates them. Not invoked or newly integrated here |
| `colmap.database` | SQLite table creation/inserts/commit at caller filename | Generic writer; no new invocation |
| `colmap.read_write_model` | `write_cameras_*`, `write_images_*`, `write_points3D_text`, `write_points3d_binary`, `write_model` emit models | Production loaders use its reader only; utility writers remain caller-selected |
| `colmap.read_write_dense` / `read_write_fused_vis` | `write_array`, `write_fused*` emit arrays, PLY, visibility binaries | Standalone conversion/export utilities; unchanged |
| `colmap.plyfile.PlyData.write` | Caller-selected PLY file/stream | Production loaders use its reader; fixtures write only temporary inputs |
| COLMAP converters/exporters | `bundler_to_ply.py`, `nvm_to_ply.py`, `export_to_bundler.py`, `export_to_visualsfm.py`, `export_inlier_pairs.py`, `export_inlier_matches.py`: PLY, keypoints, lists, matches, scripts, compressed exports | Standalone tools; no new isolated-mode entry point or execution |
| Other vendored COLMAP tools | Download/build tools, camera-spec generation, source formatting can create downloads/build/generated files or overwrite selected source | Not loader calls; unchanged and not executed |
| `utils.base_utils` | Generic `save_pickle`, `save_depth`, `make_dir`: pickle/PNG/directories | Dataset pickle call sites redirected; other callers remain legacy |
| `utils.draw_utils` / `utils.geom` | Point text/PLY, plot `savefig`, light-visibility/XYZ/normal/alpha array/visualization exports | Caller-directed utilities outside loader construction; unchanged |

Adjacent writers remain outside this milestone: evaluator `data/geometry*.log`
appends, `eval_nerf_shape.py`'s `data/eval.ply`, masked evaluator
`save_dir/masked_rendered` images (JSON write still commented), renderer debug
`cloud.ply`, NVS/validation images, checkpoints/logs, extraction, and relighting.
Allocating a dataset cache does not redirect these pipelines; later reviewed
integration must route them before claiming end-to-end input immutability.

Third-party import/runtime caches are separate from dataset products. Importing
image I/O in this session caused Matplotlib to select a temporary directory when
its home config path was unavailable. This API does not globally rewrite library
environment variables; runtime-cache environment setup remains later integration
work. No system permissions were changed.

## Explicit mode selection

`parse_database_name(name, dataset_dir)` or constructors without `preprocessing`
select **legacy compatibility mode**, including original write locations. Existing
legacy training/extraction/evaluation entry points still use it. Do not run them against
immutable inputs expecting this opt-in change to redirect the entire pipeline.

New mode uses an already selected experiment and an explicit configuration root:

```python
from pathlib import Path
from infrastructure.config import load_config
from infrastructure.datasets import load_database

cfg = load_config("configs/experiments/geometry.example.yaml",
                  config_root=Path("configs").resolve())
# resolver: PathResolver with protected datasets/thesis roots
# run: RunPaths from explicit RunStore allocation/resume
database = load_database(cfg, resolver, run)
```

The factory resolves selected inputs and passes a `PreprocessingCache` to the
existing constructor. It leaves `cfg.legacy` and `database.root` unchanged.
Direct callers may construct `PreprocessingCache(resolver, run, settings)` and
pass keyword-only `preprocessing`. Validation `DummyDataset` accepts the same
argument. At the 1D.2B-1 boundary the trainer did not supply it; the separate
geometry mode now supplies the same cache to validation and the shape renderer;
the separate material mode supplies it to validation and the material renderer.

Schema 1 supports the implemented infrastructure field below, also the default
when omitted. It is a relative destination inside the selected experiment.

```yaml
preprocessing:
  cache: cache/datasets
```

Absolute paths, traversal, writable symlinks, and overlapping input/output roots
are rejected. No numerical option was added. Future mounts remain read-only
`/inputs/datasets`, read-only `/inputs/thesis`, and writable `/outputs` for one run.

## Approved configuration root

Schema 1 now requires an explicit absolute existing `config_root`. The envelope
and canonical `legacy_config` target must both lie beneath it. Sibling references
such as `../shape/syn/bell.yaml` work when approving repository `configs`.
Traversal and symlink escapes fail before opening the referenced file, including
in `strict=False` inspection mode. There is no inferred fallback root.

Unversioned YAML remains loadable without a root as the separate legacy API;
supplying a root confines that source too. This intentionally tightens the 1D.2A
schema-1 API. Original YAML semantics and shallow overrides are preserved.

## Identity, source products, and writes

```text
<experiment>/<preprocessing.cache>/<family>/<scene>/<identity-digest>/
  identity.json
  cache.pkl                    # when newly generated
  images_<mode>/...            # when newly generated
  evaluations/<parameter-digest>/...
```

Identity records schema, full database name/mode, canonical dataset/scene paths,
passed preprocessing parameters, hashes of relevant implementation source files,
Python/package/loaded backend versions, and source names/targets/sizes/nanosecond
mtime/ctime. NeRF records `testskip` without changing train-only loading. Evaluation
exports additionally distinguish voxel size and scene scale.

Only this identity document and small implementation files are hashed, not full
datasets. Ordinary input/parameter/code/dependency changes select a different
namespace. Missing, incomplete, or mismatched identity metadata fails closed;
unmarked directories are not adopted. Encountered source changes/removals after
binding are rejected. New mode rejects directory symlinks and source-file
symlinks escaping the scene.

Existing scene `cache.pkl`, processed images, and crop metadata remain legacy
cache hits, read without writes. Their names/metadata are in the identity and the
preservation policy is recorded. No stale historical product is repaired or
assumed consistent with raw COLMAP/images. Missing products are generated only
inside the selected experiment cache. This preserves legacy cache-hit semantics.

New pickle payloads retain existing structures and serialization calls. Images
keep writer arguments and suffixes. Unique temporary files retain the same
encoding suffix and replace only cache destinations after success. Failed
temporaries are not hits. No source product is overwritten, no seeds are set,
and cache handling introduces no Python/NumPy RNG draws.

Metadata fingerprints are not full content provenance, even with ctime checks
for edits preserving size/mtime. Different canonical mount locations select new
namespaces. Full dataset hashes remain future work. Concurrent/incomplete
namespace initialization fails clearly rather than adopting partial state;
multi-process ownership/locking remains later integration work. Storage is
assumed managed by a trusted process, not adversarially replaced concurrently.

## Numerical preservation and CPU tests

Preprocessing arithmetic, camera normalization, resolution calculations,
array/tensor precision, train-only metadata selection, split membership/global
RNG reset, depth decoding, pose conventions, and evaluation selection are not
corrected. Split/pure-helper ASTs are identical to the approved baseline. Tests
compare preprocessing arithmetic expressions, array dtypes, pickle bytes, and
PNG/JPEG serialization where exercised.

```sh
python3 -B -m unittest discover -s tests/infrastructure -v
```

Recorded result: **65 tests discovered, 62 passed, 3 native-backend tests skipped**
on Python 3.9.12 (NumPy 1.21.5, scikit-image 0.19.2, ImageIO 2.9.0, PyYAML 6.0).
Whitespace, syntax, documentation links, and changed-file scope checks also pass.

Tiny temporary fixtures exercise actual NumPy, COLMAP binary/PLY reads, image
I/O, and pickle serialization. Bytes/mtimes are snapshotted and guards reject
attempted input writes without chmod or destructive probes. Differential tests
load dataset definitions from the baseline above and current source, checking
cold/warm caches, existing products/masks, splits/RNG states, factory dispatch,
incompatibility, unsafe paths, failed writes, and approved-root confinement.

OpenCV, Open3D, and transforms3d are missing in this session. Fixture tests use
explicit stand-ins at those boundaries and verify storage/control-flow parity,
not native warp/resize/downsampling equivalence. Dedicated native CPU tests are
skipped, not passed. No dependency installation or GPU/container probe was made.

`CustomDatabase` already lacks abstract `get_img_ids`; its public constructor
remains non-instantiable. A test-only concrete subclass exercises raw/crop/cache
methods without repairing production behavior. Known evaluator/split defects
also remain unchanged.

## Proposed 1D.2B-2, requiring separate approval

1. Thread approved config roots, experiment selection, and preprocessing storage
   through training/validation/rendering owners without changing numerical
   mappings or random-number behavior.
2. Route remaining logs/checkpoints/renders/extraction/evaluation outputs and
   review optional COLMAP/debug tools before claiming full write isolation.
   Configure library runtime caches before imports where needed.
3. Define scientific checkpoint compatibility and explicit warm-start/resume
   behavior separately from historical input selection and missing RNG state.
4. Execute skipped native CPU checks in an authorized dependency-complete
   environment. Review the custom-loader abstract-class defect and evaluator
   issues separately; no implicit scientific repair is approved here.

At the 1D.2B-1 boundary no training integration was implemented. Geometry routing
is now implemented separately in 1D.2B-2A and material routing in 1D.2B-2B;
standalone tools remain deferred. No real run allocation, native training, Docker
setup, or expensive evaluation was executed in these milestones.
