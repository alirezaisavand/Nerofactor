"""Separate explicit camera-0 checkpoint diagnostic and CPU saved-output comparison."""

import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    evaluate=sub.add_parser('evaluate')
    evaluate.add_argument('--source-id',choices=['0'],required=True)
    evaluate.add_argument('--eval-step',type=int,choices=[149999,150000],required=True)
    evaluate.add_argument('--run-id',required=True)
    evaluate.add_argument('--image-id',required=True)
    evaluate.add_argument('--source-commit',required=True)
    for name in ('execute','acknowledge-trusted-checkpoint','confirm-isolated-runtime'):
        evaluate.add_argument('--'+name,action='store_true',required=True)
    compare=sub.add_parser('compare')
    for name in ('primary','control'):
        compare.add_argument('--'+name,type=Path,required=True);compare.add_argument('--'+name+'-sha256',required=True)
    compare.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=='evaluate':
        if not re.fullmatch(r'sha256:[a-f0-9]{64}',args.image_id) or not re.fullmatch(r'[a-f0-9]{40}',args.source_commit):
            parser.error('Invalid source/image declaration')
        try:
            from infrastructure.bell_camera_diagnostic import execute_diagnostic
            execute_diagnostic(args.run_id,args.source_id,args.eval_step,Path('/inputs/datasets'),Path('/inputs/thesis'),Path('/outputs'),
                Path('/inputs/metadata/checkpoint-metadata.json'),Path('/inputs/reference/reference-signature.json'),
                Path('/inputs/comparison/architecture-comparison.json'),args.image_id,args.source_commit)
            print('camera0_diagnostic_complete: no producing-run attribution implied');return 0
        except BaseException as exc:
            print('Camera-0 diagnostic failed ('+type(exc).__name__+'): '+str(exc),file=sys.stderr);return 1
    output=None
    try:
        from infrastructure.signatures import reserve_output,write_report
        from infrastructure.bell_camera_comparison import PROTOCOL,load_diagnostic,compare_steps
        output=reserve_output(args.output_root,'step-comparison.json',[args.primary,args.control])
        primary=load_diagnostic(args.primary,args.primary_sha256,149999)
        control=load_diagnostic(args.control,args.control_sha256,150000)
        result=compare_steps(primary,control,args.output_root);write_report(output,result);print(result['result'])
        return 0 if result['result']=='within_frozen_step_comparison_bounds' else 1
    except Exception as exc:
        if output is not None:
            write_report(output,{'schema_version':1,'protocol':PROTOCOL,'result':'blocked','error_type':type(exc).__name__,'error':str(exc)})
        print('Step comparison blocked ('+type(exc).__name__+'): '+str(exc),file=sys.stderr);return 2
    finally:
        if output is not None:output.close()


if __name__=='__main__':raise SystemExit(main())
