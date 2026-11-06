"""CPU-only R/R-repeat/C report and lossless-array comparison; no Torch imports."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure.signatures import reserve_output, write_report


def main(argv=None):
    from infrastructure.bell_comparison import compare_sequence, load_arm
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'repeat'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.candidate is None) != (args.candidate_sha256 is None):
        parser.error('Candidate path/hash must be explicitly selected together')
    output = None
    result = None
    try:
        inputs = [args.reference, args.repeat] + ([args.candidate] if args.candidate is not None else [])
        output = reserve_output(args.output_root, 'regression-comparison.json', inputs)
        r = load_arm(args.reference, args.reference_sha256, 'R')
        repeat = load_arm(args.repeat, args.repeat_sha256, 'R-repeat')
        # Repeatability gates candidate loading and interpretation, not only the final pass flag.
        candidate = (lambda: load_arm(args.candidate, args.candidate_sha256, 'C')) if args.candidate is not None else None
        def record(partial):
            nonlocal result
            result = partial
            write_report(output, partial)
        result = compare_sequence(r, repeat, candidate, args.output_root, record=record)
        write_report(output, result)
        print(result['result'])
        return 0 if result['result'] in ('repeatability_passed', 'regression_passed') else 1
    except Exception as exc:
        if output is not None:
            failed = result if result is not None else {'schema_version': 1}
            failed.update(result='blocked', error_type=type(exc).__name__, error=str(exc))
            write_report(output, failed)
        print('Comparison blocked (' + type(exc).__name__ + ')', file=sys.stderr)
        return 2
    finally:
        if output is not None:
            output.close()


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
