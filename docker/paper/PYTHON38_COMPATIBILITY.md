# Python 3.8 compatibility assessment

Reference: **system CPython 3.8.10**, CUDA 11.1, Torch 1.8.1+cu111, ABI 0. Python
3.9.12 was the earlier CPU fixture environment, not the target for reconstruction.
No Micromamba environment, interpreter upgrade or scientific code adaptation is
introduced. The executable `python3.8` is unavailable in Codex; no interpreter or
package was installed and the reference container was not executed here.

## Static findings

`python3 -B docker/paper/static_checks.py` passes:

- Definition/verified-version/source-pin consistency and Python 3.8 grammar for
  definition scripts.
- Proposed Compose read-only mounts, explicit user/GPU/image/input selections and
  runtime isolation fields; **not** actual Docker Compose engine semantics.
- Bash syntax parsing of all **5** offline Docker RUN instructions (after 1D.3B-1A).
- Python 3.8 grammar parsing of **44** pipeline/infrastructure/helper/definition
  files, including the two experiment CLIs and used COLMAP reader helpers.

1D.3B-1A adds a stdlib-only host preflight, complete staged-definition inventory
checks and 14 static artifact/shell regression tests. They do not assume Python
in the historical base or install a bootstrap interpreter. The regression suite
runs on the available host interpreter, not the untested reference Python 3.8.

The infrastructure uses stdlib `dataclasses`, `importlib.metadata`, `pathlib`,
typing aliases and subprocess APIs present in Python 3.8. No obvious post-3.8
runtime API was found in those modules during this focused read. This is a static
assessment, not execution under 3.8, import success or proof of scientific parity.
Grammar parsing on Python 3.9 cannot prove every Python 3.8 runtime behavior,
annotation evaluation, dependency import or native ABI interaction.

## Definite existing test-harness incompatibility

Python 3.8 has no `ast.unparse`. Three existing call sites occur in:

- `tests/infrastructure/test_geometry_training.py` (two source-comparison sites).
- `tests/infrastructure/test_material_training.py` (one source-comparison site).

Those tests would fail when reaching these calls under Python 3.8. No backport,
test rewrite, monkey patch or Python upgrade is applied here. Keep the earlier
Python 3.9 CPU evidence labelled as such. A narrowly reviewed test-only
compatibility change or explicitly separate characterization interpreter must be
agreed before claiming a complete reference-container test-suite pass. It must
preserve the tests' source/numerical comparisons rather than skip them silently.

## Dependency/native limitations

The verified live versions are retained in the reference declarations; unverified
Pillow/PyYAML/h5py/plyfile/Matplotlib/build-tool/transitive versions remain missing.
Python 3.8 wheel availability and dependency metadata must be checked for every
recovered artifact, rather than inferred from package names. The staged full
hashed wheel set, `pip check`, explicit Python patch/ABI and future native checks
are acceptance gates. None was executed or resolved by installing anything here.

The unused `utils.ref_utils.get_lat_long` helper passes `indexing='ij'` to
`torch.meshgrid`; this is a potential Torch 1.8 API-compatibility concern if that
helper is enabled. The source search found no caller in the repository. It was
not changed or evaluated on GPU. Main training ray construction and renderer
mathematics are left exactly as reviewed. Static Python compatibility does not
authorize unrelated Torch API or evaluator corrections.

nvdiffrast's exact source supports a Python/source distribution with lazy native
builds; first plugin use needs the retained toolkit and writable experiment cache.
It clears `TORCH_CUDA_ARCH_LIST`; an image ENV declaration cannot guarantee that
revision's eventual plugin architecture flags. Raytracing source/flags, GCC/Eigen,
Torch C++ ABI and CUDA/Python extension linkage need future build/native evidence.
Matching source commits/setup hashes do not establish matching historical binary
hashes. The reference A100/driver facts do not validate the proposed image.

No scientific source, legacy entry point, YAML, dataset, mesh, checkpoint or result
was modified, and no CPU training loop, package import smoke, Docker build/run,
native extension compilation or CUDA operation was performed for these checks.
