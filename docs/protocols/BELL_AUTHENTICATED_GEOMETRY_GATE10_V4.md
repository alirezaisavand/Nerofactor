# Bell Gate 10 v4, executable owned Torch extension cache

**FIX AND PROCEED to fresh execution approval.** The v3 launch reached G's first
forward call and stopped before any Adam return at nvdiffrast texture lookup.
The stored receipt records zero updates, zero validation and no checkpoints.
Exit1 and container absence are recorded. Native cleanup_verified remains false.
Preserve all v1–v3 executions, freezes, authorizations, sources and failed outputs.

## Confirmed mount incompatibility and correction

V3 mounts /tmp as rw,nosuid,nodev,noexec and sets HOME/XDG_CACHE_HOME there.
PyTorch resolves its default extension directory to /tmp/torch_extensions.
nvdiffrast builds/loads its existing plugin lazily on the first texture call.
The native loader then reports failed to map segment from shared object for
/tmp/torch_extensions/nvdiffrast_plugin/nvdiffrast_plugin.so. Executable shared
library mapping is incompatible with the noexec mount.

V4 adds one explicit launcher variable,
TORCH_EXTENSIONS_DIR=/outputs/torch_extensions.
The existing owned evidence bind mount permits executable library mappings.
The parent evidence filesystem reports ext4 with rw,nosuid,nodev,relatime.
A focused CPU test compiles and loads a trivial shared library on this same host
evidence filesystem, returning its known constant. This tests executable mapping
on the host path. It does not execute nvdiffrast or establish native CUDA success.

The native entry checks that the selected cache equals the frozen path inside
its own /outputs mount before activating G. The original Torch/nvdiffrast JIT
build uses its existing source, flags and compiler behavior. Existing shader
calls trigger the build at their original boundary. No image rebuild, alternate
plugin implementation, synthetic CUDA warm-up or extra shader query is added.
PyTorch1.8.1 implements this directory override in _get_build_directory.
[Original PyTorch1.8.1 source](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/utils/cpp_extension.py)

The cache persists with this run. Native terminal receipts record byte sizes
and SHA-256 of its regular files, including compiled objects/library and build
metadata, on success or failure. V3's ephemeral /tmp plugin is unavailable after
container removal, so its binary hash cannot be retrospectively recovered.

## Scientific and execution invariants

Only cache placement and its evidence recording change. Retain v3's eight
spawned workers and closed G bootstrap, exact scientific source/configuration,
architecture, initial weights policy, seed, 122/1/5 split, sampling, losses,
coefficients, schedules,300k endpoint, intermediate saves, extraction and all
numeric/quality gates. HOME, XDG_CACHE_HOME and the /tmp noexec flag retain their
prior values. Original G/A and earlier accepted evidence remain byte-identical.

Runtime cache storage contributes to the existing16GiB owned-output guard per
phase. Runtime, GPU0/A100, four CPUs,128GiB RAM/no swap, eight loader workers,
.75 allocator fraction,36h training/2h evaluation, first-gate1h and30s termination
grace are unchanged. There is no additional scientific work budget.
No modified launch or scientific execution occurs during preparation.

## Validation and fresh approval

Focused tests verify the exact cache environment/mount wiring, ordering of its
native guard before scientific activation, terminal cache evidence recording,
and CPU shared-library loading on the owned evidence filesystem. The existing
schedule/split/SDF and worker regressions are retained. Full JUnit, exit code,
inventory and hashes are recorded in preparation-v4.json.

The adjacent new freeze binds corrected source, unchanged inputs and new output.
The shared launcher explicitly selects v4. V3's single native launch is consumed,
so execution requires one fresh commit/freeze/output-bound researcher approval.
Do not reuse a previous output, retry automatically, resume, change thresholds
or alter the full training horizon. The same approval can cover training,
its internal1000 lifecycle gate, endpoint extraction/audit and file-only results
analysis. Native CUDA execution remains unverified.
