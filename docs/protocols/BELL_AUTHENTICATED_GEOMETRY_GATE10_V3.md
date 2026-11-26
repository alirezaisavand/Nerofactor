# Bell Gate 10 v3, closed-source bootstrap in spawned workers

**FIX AND PROCEED to fresh execution approval.** Preserve the v1 and v2
executions, authorizations, freezes and original G/A scientific sources. The v2
receipt records zero original Adam returns, zero validation and no checkpoints.
The host records exit1 and container absence. Native cleanup_verified remains
false. No geometry or material result was produced.

## Confirmed provenance defect and narrow repair

The parent uses stage_sources.activate to install StageFinder and select pinned
geometry G. Python spawn carries sys.path into the child, but it does not copy
that parent sys.meta_path finder. The selected G packages and repository-level
packages are namespace packages without __init__.py. During dataset unpickling,
normal namespace discovery merges their directories. The strict imported_sources
check consequently raises Mixed namespace path. V2's synthetic payload test did
not load a scientific namespace and missed this defect.

V3's sole execution correction calls the existing geometry activate function in
the entry script's __mp_main__ branch, immediately after establishing the
repository path and before any experiment imports or Process payload unpickling.
This verifies the pinned manifest, reinstalls the same existing closed import
finder and selects the same G source path. No scientific model is constructed
during bootstrap. No RNG is seeded or consumed there.

The original worker initializer still checks eight workers, uninitialized CUDA,
manifest hashes and strict imported-source provenance. The parent validator is
unchanged. Already imported mixed scientific namespaces fail activation, and
unavailable G modules cannot fall back to repository or Material A files.
No namespace is silently cleared, rewritten or accepted after a mixed import.

The spawn transport, original dummy datasets and collator, sampler, RNG policy,
worker count, shader, losses, optimizer, schedules, exact300k endpoint, checkpoint
cadence, resource/work limits, extraction and quality gates remain as v2.
The main entry's __name__=='__main__' guard prevents a spawned child from
executing the scientific CLI.

## Focused CPU evidence

Regressions reproduce the merged namespace under ordinary path-based discovery,
demonstrate that the bootstrap closes dataset.__path__ to the pinned G directory,
reject a deliberately foreign fallback submodule, and reject activation after a
mixed namespace was already imported. The main-entry AST check locks bootstrap
ordering before benchmark/experiment imports.

The real eight-worker CPU fixture now mirrors __mp_main__ bootstrap and imports
the actual pinned dataset namespace in each child before the worker initializer.
It checks exact namespace paths and strict provenance as well as dummy payloads,
validation order, parent RNG parity and worker termination. It constructs no
original neural field, CUDA context, optimizer or scientific renderer. This
supersedes v2's insufficient namespace coverage. It remains host CPU evidence;
the corrected container runtime is unexecuted.

The original split/schedule/SDF fixtures are retained. Full JUnit, inventory,
process exit and exact artifact identities are recorded in preparation-v3.json.
Original failed runtime artifacts are hashed in failure-v2.json.

## Authorization and unchanged scientific definition

The adjacent v3 freeze binds the corrected sources, new protocol, tests,
unchanged acquisition/source identities, image and a new output root. It extends
the v1 full-run definition and v2 spawn amendment. Both earlier freezes and their
reports remain historical evidence.

V2's one native launch is consumed. A new approval must bind the correction
commit, v3 freeze and fresh output. The launcher now explicitly selects v3.
No v3 authorization, run directory, native launch or retry is created by this
repair. Approval may cover exactly one complete fresh300k run, the internal1000
lifecycle gate, prescribed extraction/evaluation and subsequent file-only
analysis. Retain GPU0/A100, four CPUs, eight workers,128GiB host RAM/no swap,
.75 CUDA allocator,16GiB outputs/phase,36h train/2h audit, first-gate1h and30s
termination grace. Stop at mandatory failure with no automatic retry.
