"""Authoritative G geometry / A material execution; one stage per fresh process."""
import argparse
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['geometry','material'],required=True)
    p.add_argument('--scientific-commit',required=True)
    p.add_argument('--material-variant',choices=['A_original','A_estimator_v1','A_corrected_core_v1','A_energy_budget_v1'],default='A_original')
    p.add_argument('--material-smoke',action='store_true',help='Corrected material only: stop after two original Adam returns, before validation/checkpoint/NVS')
    p.add_argument('--arm',choices=['R','R-repeat','C'],required=True)
    p.add_argument('--cfg',type=Path);p.add_argument('--config-root',type=Path)
    for name in ['datasets-root','thesis-root','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--input-root',action='append',default=[],metavar='NAME=/ABSOLUTE/PATH')
    p.add_argument('--mesh-handoff',type=Path)
    p.add_argument('--oracle-benchmark',type=Path,help='Explicit generated-benchmark admission manifest; separate from G research handoff')
    p.add_argument('--run-id',required=True)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--validate-only',action='store_true');a.add_argument('--execute',action='store_true')
    p.add_argument('--confirm-isolated-runtime',action='store_true')
    p.add_argument('--image-id');p.add_argument('--infrastructure-commit')
    args=p.parse_args(argv)
    try:
        from infrastructure.stage_execution import plan,execute
        from infrastructure.runs import validate_run_id
        validate_run_id(args.run_id)
        roots={}
        for entry in args.input_root:
            name,sep,path=entry.partition('=')
            if not sep or name in roots:raise ValueError('Distinct NAME=/ABSOLUTE/PATH input roots required')
            roots[name]=Path(path)
        selected=plan(args.stage,args.scientific_commit,args.arm,args.datasets_root,args.thesis_root,args.output_root,
                      args.cfg,args.config_root,roots,material_variant=args.material_variant,oracle_benchmark=args.oracle_benchmark is not None)
        if args.material_smoke and (args.stage!='material' or args.material_variant=='A_original'):
            raise ValueError('Corrected material smoke requires an explicit corrected material variant')
        if args.stage=='geometry' and args.mesh_handoff is not None:raise ValueError('Geometry cannot accept a material mesh handoff')
        if args.stage=='material':
            if args.oracle_benchmark is not None:
                if args.mesh_handoff is not None:raise ValueError('Oracle benchmark and G handoff are distinct geometry policies')
                from infrastructure.material_variants import admit_oracle
                admit_oracle(selected,args.oracle_benchmark)
            else:
                from infrastructure.stage_workspace import handoff
                handoff(args.mesh_handoff,selected['selected'].mesh,selected['resolver'],selected['configuration']['database_name'])
        if args.validate_only:
            print('Pinned '+args.stage+' scientific configuration validated; no allocation/import/model execution.');return 0
        if not args.confirm_isolated_runtime or not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id or '') or not re.fullmatch(r'[a-f0-9]{40}',args.infrastructure_commit or ''):
            p.error('Execution requires isolated-runtime confirmation and explicit image/infrastructure identities')
        execute(selected,args.run_id,args.image_id,args.infrastructure_commit,args.mesh_handoff,corrected_material_smoke=args.material_smoke,oracle_manifest=args.oracle_benchmark)
        print('stage_execution_complete; native acceptance still requires reviewed G R/R-repeat/C evidence');return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
