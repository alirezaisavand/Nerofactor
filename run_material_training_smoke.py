"""Diagnostic-only A anisotropic bear material training: exactly two Adam returns."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scientific-commit',required=True)
    for name in ['datasets-root','thesis-root','output-root','mesh']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mesh-sha256',required=True)
    parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-historical-diagnostic-mesh',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.stage_sources import ROOT
        from infrastructure.stage_execution import plan,execute
        from infrastructure.material_smoke import CONFIG,MESH_SHA256,admit_mesh
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        if args.mesh_sha256!=MESH_SHA256:raise ValueError('Explicit approved historical mesh SHA-256 required')
        p=plan('material',args.scientific_commit,'C',args.datasets_root,args.thesis_root,args.output_root,ROOT/CONFIG,ROOT)
        if not args.mesh.is_absolute() or args.mesh.resolve()!=args.mesh or args.mesh!=p['selected'].mesh:
            raise ValueError('Explicit unaliased approved bear mesh selection required')
        admit_mesh(p)
        if args.validate_only:
            print('A bear diagnostic paths/YAML/mesh verified; no allocation, model imports or training.');return 0
        if not args.acknowledge_historical_diagnostic_mesh or not args.confirm_isolated_runtime or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
            parser.error('Execution requires diagnostic-mesh acknowledgement and isolated image/infrastructure identities')
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,bounded_material=True)
        print('A material two-update diagnostic complete; native evidence requires review.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
