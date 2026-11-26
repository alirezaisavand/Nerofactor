"""Replay frozen Gate 9 with a binary64 allowance for Q-residual diagnostics."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_learned_light_relighting_gate9_v1 as g
from infrastructure.signatures import require

SOURCE = 'experiments/donut_learned_light_relighting_gate9_v1.py'
SITE = "    require(r['endpoint_parity']==parity,'Saved material-field admission differs')"
RESIDUAL = 'Q_axis_reconstruction_max_abs'
# Residuals subtract two dimensionless Q entries (unit-scale bound), so
# near-zero residuals inherit absolute floating-point error, not relative error.
RESIDUAL_ROUNDOFF = np.finfo(np.float64).eps


def check_endpoint_parity(recorded, replay):
    expected = {str(camera): count for camera, count in zip(g.CAMERAS, g.g6.COUNTS)}
    require(recorded.keys()==replay.keys()==expected.keys(), 'Gate 9 endpoint camera inventory differs')
    for camera, count in expected.items():
        require(recorded[camera].keys()==replay[camera].keys()==set(g.ARMS),
                'Gate 9 endpoint method inventory differs')
        for arm in g.ARMS:
            before, after = (values[camera][arm] for values in (recorded,replay))
            require(before.keys()==after.keys()=={'pixels',RESIDUAL} and
                    all(type(row['pixels']) is int and row['pixels']==count for row in (before,after)),
                    'Gate 9 endpoint pixel count or diagnostic inventory differs')
            a,b=before[RESIDUAL],after[RESIDUAL]
            require(all(type(x) is float and np.isfinite(x) and 0<=x<=g.g6.MATERIAL_ATOL for x in (a,b))
                    and abs(a-b)<=RESIDUAL_ROUNDOFF, 'Gate 9 endpoint Q residuals differ')


def analyze(sha, output):
    f=g.freeze(sha)
    path=g.ROOT/SOURCE
    require(g.digest(path)==f['sources'][SOURCE],'Frozen Gate 9 analysis source differs')
    raw=path.read_text()
    require(raw.count(SITE)==1,'Expected one frozen Gate 9 replay check')
    code=raw.replace(SITE,"    check_endpoint_parity(r['endpoint_parity'],parity)",1)
    scope=dict(__name__='gate9_offline_v2',__file__=str(path),check_endpoint_parity=check_endpoint_parity)
    exec(compile(code,str(path),'exec'),scope)
    scope['analyze'](sha,output)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    analyze(args.freeze_sha256,args.output)
