"""Read-only CPU coordinate audit of explicitly hashed saved bell evidence."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path,help='Explicit version-1 artifact-selection JSON')
    parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-dir',required=True,type=Path,help='New directory; must not already exist')
    args=parser.parse_args(argv)
    try:
        from infrastructure.geometry_scientific_audit import audit
        report=audit(args.inputs,args.inputs_sha256,args.output_dir)
        print('Bounded coordinate checks: '+report['status']+'; report: '+str(args.output_dir/'audit.json'))
        print('Historical origin, scientific accuracy, 3A-2 SDF orientation and research admission remain unresolved.')
        return 0 if report['status']=='pass' else 2
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
