"""Historical G bell mesh to pinned A: read-only rays, then gated two updates."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['rays','two-update'],required=True)
    parser.add_argument('--scientific-commit',required=True)
    for name in ['datasets-root','thesis-root','geometry-root','output-root','mesh','extraction-report','ray-report']:
        parser.add_argument('--'+name,type=Path,required=name!='ray-report')
    for name in ['mesh-sha256','extraction-sha256','run-id']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--ray-sha256')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-diagnostic-mesh-and-trusted-cameras',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.bell_material_integration import plan,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        p=plan(args.scientific_commit,args.datasets_root,args.thesis_root,args.geometry_root,args.output_root,
               args.mesh,args.mesh_sha256,args.extraction_report,args.extraction_sha256,args.action,args.ray_report,args.ray_sha256)
        if args.validate_only:
            print('Bell diagnostic inputs/configuration/admission verified; no native imports, allocation or camera loading.');return 0
        if not args.acknowledge_diagnostic_mesh_and_trusted_cameras or not args.confirm_isolated_runtime or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
            parser.error('Execution requires trusted original cameras, diagnostic-only acknowledgement and isolated image/source identities')
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,acknowledged=True)
        print('Bell diagnostic complete; review native evidence before progression.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
