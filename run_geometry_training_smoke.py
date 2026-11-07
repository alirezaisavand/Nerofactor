"""Researcher-side fresh bell geometry training: exactly two Adam calls, no resume."""

import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('R', 'R-repeat', 'C'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--confirm-isolated-runtime', action='store_true', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image_id) or not re.fullmatch(r'[a-f0-9]{40}', args.source_commit):
        parser.error('Invalid image/source declaration')
    try:
        from infrastructure.geometry_training_smoke import execute_smoke
        execute_smoke(args.arm, args.run_id, Path('/inputs/datasets'), Path('/inputs/thesis'),
                      Path('/outputs'), args.image_id, args.source_commit)
        print('bounded_training_smoke_passed: two optimizer calls; no full-training equivalence claim')
        return 0
    except BaseException as exc:
        print('Training smoke failed (' + type(exc).__name__ + '): ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
