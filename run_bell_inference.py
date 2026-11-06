"""Explicit researcher-side one-view bell inference; Codex runs fixtures only."""

import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    from infrastructure.bell_inference import execute_arm
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['R', 'R-repeat', 'C'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--acknowledge-trusted-checkpoint', action='store_true', required=True)
    parser.add_argument('--confirm-isolated-runtime', action='store_true', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image_id) or not re.fullmatch(r'[a-f0-9]{40}', args.source_commit):
        parser.error('Invalid source/image identity')
    try:
        execute_arm(args.arm, args.run_id, Path('/inputs/datasets'), Path('/inputs/thesis'), Path('/outputs'),
                    Path('/inputs/metadata/checkpoint-metadata.json'), Path('/inputs/reference/reference-signature.json'),
                    Path('/inputs/comparison/architecture-comparison.json'), args.image_id, args.source_commit)
        print('One A-based validation view completed; not a thesis held-out score.')
        return 0
    except Exception as exc:
        print('Bell inference failed (' + type(exc).__name__ + '): ' + str(exc) +
              '; preserve partial evidence; no fallback.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
