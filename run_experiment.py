"""Explicit isolated geometry entry point; legacy run_training.py is unchanged."""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    # Do this before local/scientific imports; worker processes inherit the
    # runtime environment policy when execution is explicitly requested.
    sys.dont_write_bytecode = True
    from infrastructure.config import ConfigError, load_config
    from infrastructure.geometry import KNOWN_LIMITS, check_geometry_selection, prepare_geometry, validate_geometry
    from infrastructure.paths import PathError, PathResolver
    from infrastructure.runs import RunError, validate_run_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--thesis-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layout", choices=["collection", "mounted"], default="collection")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--resume-checkpoint", help="Explicit filename relative to this run's checkpoints/")
    selection.add_argument("--initialize-from-input", action="store_true", help="Load only weights from inputs.checkpoint into a new run")
    args = parser.parse_args(argv)
    try:
        validate_run_id(args.run_id)
        cfg = load_config(args.cfg, config_root=args.config_root)
        for diagnostic in cfg.diagnostics:
            print("{} [{}] {}: {}".format(diagnostic.level, diagnostic.code,
                                          diagnostic.field, diagnostic.message))
        resolver = PathResolver({"datasets": args.datasets_root, "thesis": args.thesis_root}, args.output_root)
        plan = validate_geometry(cfg, resolver)
        mode = "resume" if args.resume_checkpoint else "initialize" if args.initialize_from_input else "new"
        if mode == "new" and plan.selected.checkpoint is not None:
            raise RunError("Use --initialize-from-input for the declared historical checkpoint")
        if mode == "initialize" and plan.selected.checkpoint is None:
            raise RunError("--initialize-from-input requires inputs.checkpoint")
        for message in KNOWN_LIMITS:
            print("Compatibility limit:", message)
        if args.validate_only:
            check_geometry_selection(plan, args.run_id, mode, args.resume_checkpoint, args.layout)
            # No run allocation, scientific imports, checkpoint decoding, or RNG
            # seeding. Tensor-state compatibility is checked only at execution.
            print("Geometry configuration/path structure validated; no run allocated or model executed.")
            return 0
        experiment = prepare_geometry(plan, args.run_id, mode, args.resume_checkpoint, args.layout)
        from infrastructure.geometry_execution import execute_geometry
        execute_geometry(experiment)
    except (ConfigError, PathError, RunError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
