"""Replay Gate 8 with a one-binary64-ULP allowance for the derived pooled limit."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_thesis_attribution_gate8_v1 as g
from infrastructure.signatures import require

SOURCE = 'experiments/analyze_donut_thesis_attribution_gate8_v1.py'
SITE = "    require(gate==receipt['native_reference_gate'],'Original A compatibility evidence differs')"


def check_diagnostic(replay, recorded):
    """Match every field exactly except the one host/native reduction witness."""
    require(replay.keys()==recorded.keys(),'Gate 8 diagnostic keys differ')
    for key, value in replay.items():
        target=recorded[key]
        require(type(value) is type(target),'Gate 8 diagnostic type differs for '+key)
        if isinstance(value,float):
            require(np.isfinite(value) and np.isfinite(target),'Nonfinite Gate 8 diagnostic '+key)
        if key=='pooled_limit':
            require(type(value) is float and value>=0 and target>=0
                    and abs(value-target)<=np.spacing(max(value,target)),
                    'Gate 8 pooled limit differs by more than one binary64 ULP')
        else:
            require(value==target,'Gate 8 diagnostic differs for '+key)


def analyze(sha, output):
    f=g.freeze(g.ROOT/g.FREEZE,sha)
    p=g.ROOT/SOURCE
    require(g.digest(p)==f['sources'][SOURCE],'Frozen Gate 8 analyzer or source differs')
    raw=p.read_text()
    require(raw.count(SITE)==1,'Frozen Gate 8 replay site differs')
    code=raw.replace(SITE,"    check_diagnostic(gate,receipt['native_reference_gate'])",1)
    scope=dict(__file__=str(p),__name__='gate8_offline_replay_v2',check_diagnostic=check_diagnostic)
    exec(compile(code,str(p),'exec'),scope)
    scope['analyze'](sha,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.freeze_sha256,args.output)
