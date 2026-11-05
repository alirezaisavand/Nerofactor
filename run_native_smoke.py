"""Explicit native smoke preflight/one-forward entry point; never starts training."""

import argparse
from pathlib import Path
import re
import sys


def main(argv=None):
    sys.dont_write_bytecode = True
    from infrastructure.config import ConfigError, load_config
    from infrastructure.native_smoke import execute_smoke, validate_smoke
    from infrastructure.paths import PathError, PathResolver
    from infrastructure.runs import RunError

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cfg', 'config-root', 'datasets-root', 'thesis-root', 'output-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--layout', choices=['collection', 'mounted'], default='collection')
    parser.add_argument('--input-root', action='append', default=[], metavar='NAME=/ABSOLUTE/PATH')
    parser.add_argument('--image-id', help='Caller-declared immutable sha256 image ID, recorded as such')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--validate-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.image_id is not None and not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image_id):
            raise ConfigError('--image-id must be an immutable sha256 image ID')
        cfg = load_config(args.cfg, config_root=args.config_root)
        for diagnostic in cfg.diagnostics:
            print('{} [{}] {}: {}'.format(diagnostic.level, diagnostic.code,
                                          diagnostic.field, diagnostic.message))
        roots = {'datasets': args.datasets_root, 'thesis': args.thesis_root}
        for item in args.input_root:
            name, separator, path = item.partition('=')
            if not separator or name in roots:
                raise PathError('Additional roots require distinct NAME=/ABSOLUTE/PATH entries')
            roots[name] = Path(path)
        resolver = PathResolver(roots, args.output_root)
        plan = validate_smoke(cfg, resolver, args.run_id, args.layout)
        if args.validate_only:
            print('Smoke structure validated; no run/model/GPU allocated, no mount or native compatibility claimed.')
            return 0
        run, report = execute_smoke(plan, args.run_id, args.layout, args.image_id)
        print('Smoke {}: {}'.format(report['status'], run.output('smoke.json')))
    except (ConfigError, PathError, RunError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
