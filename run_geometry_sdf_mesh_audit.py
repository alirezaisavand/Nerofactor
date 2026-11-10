"""One-shot, hash-admitted native G SDF/mesh/normal audit; no training/rendering."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--geometry-commit',required=True);parser.add_argument('--material-commit',required=True)
    parser.add_argument('--run-id',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-trusted-checkpoint',action='store_true')
    parser.add_argument('--confirm-isolated-runtime',action='store_true')
    parser.add_argument('--image-id');parser.add_argument('--infrastructure-commit')
    args=parser.parse_args(argv)
    try:
        from infrastructure.geometry_sdf_mesh_audit import prepare,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        p=prepare(args.inputs,args.inputs_sha256,args.output_root,args.geometry_commit,args.material_commit)
        if args.validate_only:
            print('SDF audit metadata admitted; no numerical archive loading, model construction or checkpoint deserialization.');return 0
        if not args.acknowledge_trusted_checkpoint or not args.confirm_isolated_runtime or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
            parser.error('Native execution requires explicit checkpoint pickle trust, isolated runtime and image/infrastructure identities')
        execute(p,args.run_id,args.image_id,args.infrastructure_commit,trusted=True)
        print('Bounded SDF audit complete; inspect all scientific statuses, exclusions and unresolved claims.');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
