"""CPU JSON-only repeatability/candidate comparison for two-update geometry smoke."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure.signatures import reserve_output, write_report


def main(argv=None):
    from infrastructure.training_comparison import PROTOCOL, load_training_arm, compare_training_sequence
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','repeat'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args(argv)
    if (args.candidate is None)!=(args.candidate_sha256 is None):parser.error('Select candidate report and hash together')
    output=None; result=None
    try:
        inputs=[args.reference,args.repeat]+([args.candidate] if args.candidate is not None else [])
        output=reserve_output(args.output_root,'training-comparison.json',inputs)
        reference=load_training_arm(args.reference,args.reference_sha256,'R')
        repeat=load_training_arm(args.repeat,args.repeat_sha256,'R-repeat')
        candidate=(lambda:load_training_arm(args.candidate,args.candidate_sha256,'C')) if args.candidate is not None else None
        def record(value):
            nonlocal result
            result=value; write_report(output,value)
        result=compare_training_sequence(reference,repeat,candidate,record=record)
        write_report(output,result);print(result['result'])
        return 0 if result['result'] in ('repeatability_passed','bounded_training_regression_passed') else 1
    except Exception as exc:
        if output is not None:
            failed=result if result is not None else {'schema_version':1,'protocol':PROTOCOL,'candidate_interpreted':False}
            failed.update(result='blocked',error_type=type(exc).__name__,error=str(exc));write_report(output,failed)
        print('Training comparison blocked ('+type(exc).__name__+'): '+str(exc),file=sys.stderr)
        return 2
    finally:
        if output is not None:output.close()


if __name__=='__main__':
    raise SystemExit(main())
