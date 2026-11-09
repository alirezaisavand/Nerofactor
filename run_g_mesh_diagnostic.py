"""Explicit historical-best bell checkpoint -> diagnostic G mesh, resolution128 only."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scientific-commit',required=True)
    for name in ['datasets-root','thesis-root','output-root','checkpoint','compatibility-report']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--compatibility-sha256',required=True)
    parser.add_argument('--resolution',type=int,choices=[128],required=True)
    parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-trusted-checkpoint',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.g_extraction import prepare,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        if args.execute and (not args.acknowledge_trusted_checkpoint or not args.confirm_isolated_runtime or
            not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or '')):
            parser.error('Execution requires explicit trusted-pickle and isolated image/infrastructure acknowledgements')
        p=prepare(args.scientific_commit,args.datasets_root,args.thesis_root,args.output_root,args.checkpoint,
                  args.checkpoint_sha256,args.compatibility_report,args.compatibility_sha256,args.resolution)
        if args.validate_only:
            print('G best-checkpoint mesh diagnostic paths/config/hash/evidence validated; no allocation, model, pickle or extraction.');return 0
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,trusted=args.acknowledge_trusted_checkpoint)
        print('Diagnostic mesh complete; no new G training provenance or research G-to-A handoff admitted.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
