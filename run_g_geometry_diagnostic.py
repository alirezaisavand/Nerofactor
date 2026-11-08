"""Researcher-side G checkpoint compatibility, validation and manual roundtrip.

Historical torch.load is trusted pickle, never safe deserialization. No native
work occurs without explicit execution/runtime/trust acknowledgement flags.
"""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['compatibility','optimizer','validation','roundtrip'],required=True)
    parser.add_argument('--scientific-commit',required=True)
    parser.add_argument('--datasets-root',type=Path,default=Path('/inputs/datasets'))
    parser.add_argument('--thesis-root',type=Path,default=Path('/inputs/thesis'))
    parser.add_argument('--output-root',type=Path,default=Path('/outputs'))
    parser.add_argument('--compatibility-report',type=Path)
    parser.add_argument('--compatibility-sha256')
    parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-trusted-checkpoint',action='store_true')
    parser.add_argument('--acknowledge-optimizer-diagnostic',action='store_true')
    parser.add_argument('--acknowledge-manual-roundtrip',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.g_lifecycle import prepare,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        if args.action=='optimizer' and not args.acknowledge_optimizer_diagnostic:
            parser.error('Optimizer loading is a separate diagnostic requiring explicit acknowledgement')
        if args.action!='optimizer' and args.acknowledge_optimizer_diagnostic:parser.error('Unused optimizer acknowledgement')
        if args.action=='roundtrip' and args.acknowledge_trusted_checkpoint:parser.error('Roundtrip cannot consume a historical checkpoint')
        if args.action!='roundtrip' and args.acknowledge_manual_roundtrip:parser.error('Checkpoint writes are not allowed in this action')
        if args.execute:
            if not args.confirm_isolated_runtime or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
                parser.error('Execution requires isolated-runtime confirmation and explicit image/infrastructure identities')
            if (args.action=='roundtrip' and not args.acknowledge_manual_roundtrip) or (args.action!='roundtrip' and not args.acknowledge_trusted_checkpoint):
                parser.error('Explicit trusted-checkpoint or manual-roundtrip acknowledgement required')
        p=prepare(args.action,args.scientific_commit,args.datasets_root,args.thesis_root,args.output_root,
                  args.compatibility_report,args.compatibility_sha256)
        if args.validate_only:
            print('G diagnostic paths/config/evidence validated; no allocation, model, checkpoint pickle or optimizer load.');return 0
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,args.acknowledge_trusted_checkpoint,
                args.acknowledge_manual_roundtrip,args.acknowledge_optimizer_diagnostic)
        print('G diagnostic complete; producing origin, resume, natural schedule and research geometry are not admitted.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
