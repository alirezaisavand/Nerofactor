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
- Python 3.8 grammar parsing of **58** pipeline/infrastructure/helper/definition
  and infrastructure test files, including the two experiment CLIs and used
  COLMAP reader helpers (expanded in 1D.3B-2A).

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

## Test-harness incompatibility corrected in 1D.3B-2A

Python 3.8 has no `ast.unparse`. Three existing branch selectors used it in:

- `tests/infrastructure/test_geometry_training.py` (two source-comparison sites).
- `tests/infrastructure/test_material_training.py` (one source-comparison site).

The newly authorized test-only fix compares each condition's `ast.dump` with the
parsed expected expression's `ast.dump`. It preserves all original source/math
assertions, branch selection, baselines and test counts; no unparser backport,
scientific code change or interpreter upgrade is introduced. Static checks reject
any remaining `ast.unparse` calls in infrastructure tests. On available Python
3.9.12, all 100 infrastructure tests run: 97 pass and the same 3 native-backend
checks skip. A second run removes the unavailable-on-3.8 API before discovery to
exercise this specific compatibility boundary; this is **not** a Python 3.8 run.

The Docker/prerequisite suite has 28 passing tests after 1D.3B-2B, including source
export, local bundle verification, sanitized dependency-closure fixtures and the
recovered manifest's consistency/artifact boundary. No actual
Python 3.8 interpreter is available in Codex, so the full reference-runtime suite,
annotation/import behavior and dependency/native compatibility remain unverified.
See [PREREQUISITES.md](PREREQUISITES.md) for exact scope and recovery commands.

## Dependency/native limitations

The verified live versions are retained in the reference declarations.
1D.3B-2B's supplied inventory adds exact Pillow/PyYAML/h5py/plyfile/Matplotlib,
build-tool and transitive installed versions in
[runtime.reference.json](runtime.reference.json). All 82 declared Python-version
constraints accept 3.8.10; twenty distributions omit `Requires-Python`. This is
metadata consistency, not execution or verification of acquired wheel tags/bytes.
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
