# Configuration and path foundation (1D.2A)

The standalone APIs in `infrastructure/` load configuration, resolve declared
paths, and allocate isolated experiment directories. They import no training,
rendering, dataset, NumPy, or GPU modules. The only non-stdlib runtime dependency
is the already-required PyYAML. Python 3.9 syntax is supported.

**Existing training/extraction/evaluation entry points are unchanged. Do not pass
the new envelopes to `run_training.py`.** Connecting these APIs and redirecting
writers belongs to 1D.2B after review. This milestone creates no real experiment,
copies no inputs, loads no tensor checkpoints, and changes no numerical behavior.

## Versioned envelope and examples

Examples are [geometry.example.yaml](../configs/experiments/geometry.example.yaml)
and [material.example.yaml](../configs/experiments/material.example.yaml).
Both reference existing YAML without changing its scientific fields:

```yaml
schema_version: 1
stage: material
legacy_config: ../material/real/bear.yaml
inputs:
  dataset: GlossyReal
  mesh:
    root: thesis
    path: meshes-pretrained/real/bear.ply
overrides: {}
```

The fields have implemented infrastructure meanings only:

| Field | Meaning |
| --- | --- |
| `schema_version` | Only integer `1` is supported for new envelopes; absent means unversioned legacy YAML |
| `stage` | `geometry` or `material`; must agree with effective legacy `network: shape` or `material` |
| `legacy_config` | Relative reference anchored at the envelope's directory; may contain `..` to reference sibling config directories; must select unversioned YAML |
| `inputs.dataset` | Directory relative to the runtime `datasets` root; normally a dataset family matching legacy `dataset_dir` semantics, not a scene name |
| `inputs.mesh` | Required for material, rejected for geometry; `{root, path}` selects an existing input file |
| `inputs.checkpoint` | Optional `{root, path}` selecting an existing checkpoint as an input artifact; does not request resume or load weights |
| `overrides` | Explicit shallow replacement of legacy top-level fields; no deep merge or forwarding into shader settings |

No variant/algorithm selector, automatic configuration migration, training
launcher, implicit run ID, or environment-variable expansion is implemented.
Host roots are runtime arguments, never committed in default envelopes.

```python
from infrastructure.config import load_config

cfg = load_config("configs/experiments/material.example.yaml")
for diagnostic in cfg.diagnostics:
    print(diagnostic.level, diagnostic.code, diagnostic.field, diagnostic.message)

# Original path strings and scientific settings remain in cfg.legacy.
# cfg.original is a snapshot of the document that was loaded.
# cfg.legacy_source identifies the referenced legacy YAML.
```

Unversioned YAML uses the same PyYAML `FullLoader` scalar/alias/merge semantics
as `utils.base_utils.load_cfg`. Compatibility mode retains invalid or unverified
fields and returns diagnostics; it does not silently coerce, remove, or reinterpret
them. Explicit duplicate YAML keys retain the last value and emit an error
diagnostic. Ordinary YAML merge-key overrides are not duplicate-key errors.

Strict mode is the default for schema 1. Legacy files default to compatibility
mode; `strict=True` opts into rejection of error diagnostics. `strict=False`
explicitly permits diagnostic inspection even for a new envelope, retaining
unsupported envelope fields only in `original` and reporting them as errors.
Do not treat that inspection mode as a validated executable experiment.

Strict checks cover unknown envelope/input/ref fields, required selections,
stage/network agreement, portable relative input paths, and selected existing
legacy types: Boolean flags, steps/intervals, seed/worker counts, mapping/list
shapes, finite coefficients, and learning-rate values. Unsupported optimizer or
scheduler declarations are diagnosed against the current baseline. Structural
errors, unsupported schema versions, and malformed YAML always raise `ConfigError`.

This is deliberately a bounded legacy audit, not a complete schema for every
scientific field. Unchecked top-level fields receive `unvalidated-field`
diagnostics; the payload also carries a `legacy-payload` diagnostic describing
that limit. Known unconsumed/not-forwarded fields remain warnings even in strict
mode, because their original scope is preserved. No claim of training support or
scientific validity follows from merely retaining a legacy field.

## Defaults, overrides, and numerical preservation

`legacy_merge(defaults, overrides)` implements exactly `{**defaults, **overrides}`,
including nested-reference sharing. `cfg.with_defaults(owner_defaults)` uses the
same rule. There is no new global numerical-default table and no import of
scientific classes to obtain defaults. Loading snapshots caller-provided override
values so subsequent caller mutations do not change the loaded configuration.

For example, overriding `lr_cfg` with `{end_warm: 1000}` replaces the entire old
`lr_cfg`. It does not retain an old `end_iter` through recursive merging. A later
legacy scheduler merge supplies its own missing defaults, exactly as before.
`cfg.legacy` is not a fully resolved set of all renderer/shader/loss defaults;
those defaults still belong to their existing owners.

Characterization tests preserve these observed baseline behaviors:

- All 70 original YAML mappings match direct `yaml.FullLoader` loading, including
  list ordering, unknown fields, empty mappings, and `True,` string values.
- Trainer seed defaults remain 6033. No seeds are set and no Python RNG state is
  consumed by loading or UUID-based run-ID generation.
- Adam construction still supplies `lr=1e-3`, `eps=1e-4`, and `amsgrad=True`.
  Gradient clipping remains `0.7`; no optimizer or scientific source is edited.
- Warm-up and scheduler defaults remain unchanged. In particular, the actual
  cosine schedule still clamps at 100,000 with alpha `0.2`, independently of a
  configured `end_iter`. Tests characterize warm-up and endpoint boundaries.
- Opacity coefficients/schedule, loss list order, and shader sampling settings
  remain in their original scopes with their original values.
- Top-level material shader-like declarations are not moved into `shader_cfg`.
  A fresh baseline shader can retain energy/specular defaults `0.05` even when
  top-level declarations are zero. This is diagnosed, not corrected.
- Dataset split code, global RNG side effects, BRDFs, distributions, PDF formulas,
  and sampling implementations are untouched. Their mathematical correctness
  and GPU trajectory equivalence are not established by CPU infrastructure tests.

## Runtime roots and explicit input resolution

```python
import os
from pathlib import Path
from infrastructure.paths import PathResolver

roots = PathResolver(
    {"datasets": Path(os.environ["NERO_DATASETS_ROOT"]),
     "thesis": Path(os.environ["NERO_THESIS_ROOT"])},
    output_root=Path(os.environ["NERO_OUTPUT_ROOT"]),
)
selected = roots.resolve_inputs(cfg.inputs)
# selected.dataset, selected.mesh, selected.checkpoint are canonical Paths.
```

Supply absolute roots from the environment/caller. Input roots must already be
directories. Construction and resolution create no directories. Missing inputs
fail rather than creating empty replacements. Legacy strings are never remapped
automatically: `cfg.inputs` is `None` for unversioned YAML, which requires an
explicit envelope/selection before path resolution.

`PathResolver.container()` declares `/inputs/datasets`, `/inputs/thesis`, and
`/outputs` and fails if the input directories are missing. It starts no container
and mounts nothing. Read-only enforcement will require actual runtime mounts in
1D.3; Python path selection does not change filesystem permissions.

Additional named input roots can expose geometry from a prior run without copying
it into `final_results`. Supply that directory as, for example, runtime root
`geometry` and select `{root: geometry, path: meshes/surface.ply}`. Such roots
must be disjoint from writable storage. If a prior run lies within a host run
collection, select a separate output directory using mounted layout so the
declared writable root does not also contain the prior input run.

Input paths cannot use absolute paths, `..`, empty/dot segments, backslashes,
drive/URI syntax, control characters, or shell-style expansion. Input symlinks
are allowed only when their canonical targets stay inside their declared root.
Input and output roots cannot overlap in either direction. Writable children
reject all symlinks, including aliases pointing inside the root, to prevent
cross-run redirection. Parent traversal is rejected before normalization.

## Run allocation and explicit resume

```python
from infrastructure.runs import RunStore, generate_run_id

store = RunStore(roots)  # collection layout: output root contains run directories
run_id = generate_run_id("bear-material-s6033")
run = store.create(run_id)
# Destinations only; this API never trains or writes a model/log/cache product.
checkpoint_destination = run.output("checkpoints/model.pth")
log_directory = run.logs
preprocessing_cache = run.cache

# Later, only after that exact file exists inside this run:
# resumed = store.resume(run_id, checkpoint="model.pth")
```

IDs contain 1-64 lowercase letters/digits/underscores/hyphens, starting with a
letter or digit. `generate_run_id(label)` appends a UUID4 suffix; labels are at
most 31 characters. Allocation still checks collisions, even for generated IDs.

Collection allocation exclusively creates `<output_root>/<id>` and rejects every
pre-existing directory/file, including an empty directory. The run contains
`checkpoints`, `logs`, `cache`, `meshes`, `materials`, `renders`, and `evaluations`.
`.experiment.json` records only allocation schema/version and ID. It is not a
full provenance manifest, seed record, or validation of checkpoint contents.

For a single selected experiment mounted at `/outputs`, use
`RunStore(roots, layout="mounted")`. It exclusively initializes an empty root
without adding an ID subdirectory; a nonempty root is never adopted or cleared.
The same mode also supports a host-selected run directory. Concurrent allocators
use exclusive directory/marker creation; exactly one can claim the same run.
Failures leave partial allocations for inspection and reject automatic reuse.

`resume(id, checkpoint=...)` requires an existing matching allocation marker,
all expected directories, and an explicitly selected existing file relative to
that run's `checkpoints/`. Absolute historical checkpoint paths, symlinks, and
escapes to another run are rejected. It neither deserializes nor writes the
checkpoint. Historical checkpoints are selected separately as input artifacts,
not as resumed writable history. Run identity/location validation is implemented;
architecture/config/optimizer/progress/RNG compatibility remains deferred.

Path checks cover existing symlinks and recheck output roots between calls.
They assume a trusted local process owns the declared output storage; they are
not a sandbox against hostile concurrent filesystem-parent replacement or
arbitrary direct filesystem writes by training code.

## CPU validation

Run only the infrastructure suite from the repository root:

```sh
python3 -B -m unittest discover -s tests/infrastructure -v
```

Recorded validation on 2026-11-03: **38 tests passed**, using Python 3.9.12 and
PyYAML 6.0. The working tree started clean on `paper/infrastructure` at
`74e029d632feea1508d8639cfccb4a1b0765effe`; this milestone is left uncommitted.

Tests use disposable temporary directories, text checkpoint fixtures, and no
permission changes. Legacy characterization parses source AST and executes only
isolated scheduler/setup methods with scalar/recording stand-ins, without
importing Trainer, renderers, NumPy, or PyTorch. Tests cover original YAML parity,
defaults/overrides, diagnostics, path resolution, missing/invalid paths, root
overlap, symlink containment, collisions including concurrency, both storage
layouts, partial allocations, and explicit resume. GPU execution, actual mounts,
dataset write redirection, scientific checkpoint compatibility, and training
parity remain unvalidated.

## Proposed 1D.2B integration after review

1. Review the wrapper/schema and diagnostics against intended legacy experiments.
   Choose a specific supported reference before adding entry-point integration;
   retain unversioned loading as a distinct legacy path.
2. Thread declared dataset, mesh, checkpoint-input, output, log, and cache paths
   into existing owners. Keep original path strings/provenance and numerical
   mappings separately visible; do not deep-merge or forward misplaced settings.
3. Redirect preprocessing and evaluation writers with fixture tests that verify
   identical derived content and no original-input writes. Allocation alone
   cannot protect datasets while old writers remain unchanged.
4. Connect explicit new-run/resume requests to checkpoint loading only after
   defining architecture/config/input/progress compatibility checks. Define
   warm-start versus resume separately; old checkpoints lack full RNG history.
5. Characterize actual tensor results in an authorized working environment with
   declared tolerances and resource budgets. Keep mathematical fixes, dependency
   changes, Docker execution, and full provenance work in separately reviewed
   scope. No 1D.2B integration or later milestone is implemented here.
