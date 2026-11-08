"""CPU-only comparison of explicit SHA-bound G training reports; no native work."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['reference','repeat']:
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-sha256',required=True)
    p.add_argument('--candidate',type=Path);p.add_argument('--candidate-sha256')
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args(argv)
    from infrastructure.signatures import reserve_output,write_report
    out=None
    try:
        from infrastructure.g_comparison import load,sequence
        if bool(a.candidate)!=bool(a.candidate_sha256):raise ValueError('Explicit candidate and SHA-256 required together')
        out=reserve_output(a.output_root,'G-training-comparison.json',[a.reference,a.repeat]+([a.candidate] if a.candidate else []))
        r,ri=load(a.reference,a.reference_sha256,'R');repeat,qi=load(a.repeat,a.repeat_sha256,'R-repeat')
        result=sequence(r,repeat)
        if result['repeatability']['passed'] and a.candidate:
            c,ci=load(a.candidate,a.candidate_sha256,'C');result=sequence(r,repeat,c);result['candidate_identity']=ci
        result.update(reference_identity=ri,repeat_identity=qi);write_report(out,result);print(result['result'])
        return 0 if result['result'] in ['G_repeatability_passed','G_regression_passed'] else 1
    except Exception as exc:
        if out is not None:write_report(out,{'result':'blocked','error_type':type(exc).__name__,'error':str(exc)})
        print(str(exc),file=sys.stderr);return 2
    finally:
        if out is not None:out.close()


if __name__=='__main__':raise SystemExit(main())
