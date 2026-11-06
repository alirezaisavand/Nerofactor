"""JSON-only strict comparison. Never imports Torch or deserializes a checkpoint."""

import argparse
import datetime
import hashlib
from pathlib import Path
import sys

from infrastructure.signatures import (ADMITTED_REPORT_SHA256, SignatureError, compare_catalogs,
                                       read_report, require, reserve_output, validate_checkpoint,
                                       validate_reference, write_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-report', type=Path, required=True)
    parser.add_argument('--checkpoint-report-sha256', required=True)
    parser.add_argument('--reference-signature', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    output = None
    report = {'schema_version': 1, 'protocol': 'bell-strict-architecture-comparison-v1', 'result': 'blocked',
              'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checkpoint_identity': None, 'reference_identity': None,
              'comparison_source_sha256': {str(p.relative_to(Path(__file__).resolve().parent)):
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__).resolve(), Path(__file__).resolve().parent / 'infrastructure/signatures.py')}}
    try:
        output = reserve_output(args.output_root, 'architecture-comparison.json',
                                [args.checkpoint_report, args.reference_signature])
        require(args.checkpoint_report_sha256 == ADMITTED_REPORT_SHA256, 'Select the explicitly admitted checkpoint report')
        checkpoint, identity = read_report(args.checkpoint_report, args.checkpoint_report_sha256)
        report['checkpoint_identity'] = identity
        catalog = validate_checkpoint(checkpoint)
        reference, identity = read_report(args.reference_signature, args.reference_sha256)
        report['reference_identity'] = identity
        expected = validate_reference(reference)
        report.update(compare_catalogs(catalog, expected))
        write_report(output, report)
        print(report['result'])
        return 0 if report['result'] == 'exact_structural_match' else 1
    except (SignatureError, OSError) as exc:
        report['error'] = str(exc) if isinstance(exc, SignatureError) else type(exc).__name__
        if output is not None:
            write_report(output, report)
        print('blocked: ' + report['error'], file=sys.stderr)
        return 2
    finally:
        if output is not None:
            output.close()


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
