"""Separate bounded G/A save, fresh-process restore/update and held-out evaluation."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['geometry','material'],required=True)
    parser.add_argument('--action',choices=['save','restore','evaluate'],required=True)
    parser.add_argument('--scientific-commit',required=True)
    for name in ['datasets-root','thesis-root','output-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['geometry-root','mesh','extraction-report','accepted-material-report','accepted-ray-report',
                 'checkpoint-root','checkpoint','producer-report']:
        parser.add_argument('--'+name,type=Path)
    for name in ['mesh-sha256','extraction-sha256','checkpoint-sha256','producer-sha256']:
        parser.add_argument('--'+name)
    parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-trusted-state-and-diagnostic-save',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.lifecycle_execution import plan,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        p=plan(args.stage,args.action,args.scientific_commit,args.datasets_root,args.thesis_root,args.output_root,
               geometry=args.geometry_root,mesh=args.mesh,mesh_sha=args.mesh_sha256,extraction=args.extraction_report,
               extraction_sha=args.extraction_sha256,accepted_material=args.accepted_material_report,accepted_ray=args.accepted_ray_report,
               checkpoint_root=args.checkpoint_root,checkpoint=args.checkpoint,checkpoint_sha=args.checkpoint_sha256,
               producer=args.producer_report,producer_sha=args.producer_sha256)
        if args.validate_only:
            print('Lifecycle metadata admitted; no allocation, native imports, pickle loading or model execution.');return 0
        if not args.confirm_isolated_runtime or not args.acknowledge_trusted_state_and_diagnostic_save or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
            parser.error('Execution requires explicit isolated runtime, image/source and camera/checkpoint trust/manual-save acknowledgement')
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,trusted=True)
        print('Bounded lifecycle action complete; inspect native evidence before progression.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
