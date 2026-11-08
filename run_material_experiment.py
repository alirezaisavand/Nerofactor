"""Explicit isolated material entry point; legacy and geometry CLIs are unchanged."""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    sys.dont_write_bytecode = True
    from infrastructure.config import ConfigError, load_config
    from infrastructure.material import KNOWN_LIMITS, check_material_selection, prepare_material, validate_material
    from infrastructure.paths import PathError, PathResolver
    from infrastructure.runs import RunError

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cfg", "config-root", "datasets-root", "thesis-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layout", choices=["collection", "mounted"], default="collection")
    parser.add_argument("--input-root", action="append", default=[], metavar="NAME=/ABSOLUTE/PATH",
                        help="Additional read-only input root, e.g. geometry=/inputs/geometry")
    parser.add_argument("--initialization-manifest", type=Path,
                        help="Reviewed JSON within config-root binding an untagged historical checkpoint to baseline origin")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--resume-checkpoint")
    selection.add_argument("--initialize-from-input", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("1D.5-RESET: use run_stage_experiment.py with an explicit pinned scientific stage; this earlier entry point is validation-only")
    try:
        cfg = load_config(args.cfg, config_root=args.config_root)
        for d in cfg.diagnostics:
            print("{} [{}] {}: {}".format(d.level, d.code, d.field, d.message))
        roots = {"datasets": args.datasets_root, "thesis": args.thesis_root}
        for item in args.input_root:
            name, separator, path = item.partition("=")
            if not separator or name in roots:
                raise PathError("Additional input roots require distinct NAME=/ABSOLUTE/PATH entries")
            roots[name] = Path(path)
        resolver = PathResolver(roots, args.output_root)
        plan = validate_material(cfg, resolver, args.initialization_manifest)
        mode = "resume" if args.resume_checkpoint is not None else "initialize" if args.initialize_from_input else "new"
        for message in KNOWN_LIMITS:
            print("Compatibility limit:", message)
        if args.validate_only:
            check_material_selection(plan, args.run_id, mode, args.resume_checkpoint, args.layout)
            print("Material configuration/path structure validated; no run allocated or model executed.")
            return 0
        experiment = prepare_material(plan, args.run_id, mode, args.resume_checkpoint, args.layout)
        from infrastructure.material_execution import execute_material
        execute_material(experiment)
    except (ConfigError, PathError, RunError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
