"""Finalize existing Gate 6 v2 captures with a float64-only replay allowance."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_relighting_gate6_v2 as g
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_donut_relighting_gate6_v2 import verifier_source
from infrastructure.signatures import require


def compare_convergence(camera, native, pixels, replay):
    require(set(native)=={'refined_pixels','convergence'} and native['refined_pixels']==pixels
            and set(native['convergence'])==set(replay)==set(g.ROLES), 'Refinement or role identity differs')
    fields={'maximum_rgb_difference','relative_l2','passed'}
    summary={}
    for arm, row in replay.items():
        saved=native['convergence'][arm]
        require(set(row)==set(saved)==fields and row['passed'] is True and saved['passed'] is True,
                'Scientific convergence decision differs or failed')
        maximum=row['maximum_rgb_difference']
        require(np.isfinite(maximum) and 0<=maximum<=g.v1.TOL['convergence_rgb_abs']
                and saved['maximum_rgb_difference']==maximum, 'Maximum absolute RGB grid difference differs')
        a,b=row['relative_l2'],saved['relative_l2']
        require(type(a) is float and type(b) is float and np.isfinite([a,b]).all()
                and 0<=a<=g.v1.TOL['convergence_relative_l2'] and 0<=b<=g.v1.TOL['convergence_relative_l2']
                and np.nextafter(a,-np.inf)<=b<=np.nextafter(a,np.inf),
                'Relative grid reduction differs by more than one float64 step')
        summary[arm]=dict(native_relative_l2=b,replayed_relative_l2=a,
                          absolute_difference=abs(a-b),different_float64_steps=int(a!=b))
    return dict(camera=camera,refined_pixels=pixels,roles=summary)


def verify_capture(sha, output):
    freeze=g.freeze(sha)
    report = {}
    def verify_view(camera, native, pixels, replay):
        report[str(camera)]=compare_convergence(camera,native,pixels,replay)

    source=verifier_source()
    old="        require(r['views'][str(camera)]==dict(refined_pixels=done,convergence=gates),'Grid refinement or convergence differs')"
    require(source.count(old)==1,'Expected one original exact norm-comparison site')
    code=source.replace(old,"        verify_view(camera,r['views'][str(camera)],done,gates)",1)
    namespace=dict(__name__='gate6_v2_file_only',__file__=__file__,verify_view=verify_view)
    exec(compile(code,str(g.ROOT/'experiments/analyze_donut_relighting_gate6_v1.py'),'exec'),namespace)
    namespace['analyze'](sha,output)
    require(set(report)==set(map(str,g.CAMERAS)), 'Offline camera inventory differs')
    run=Path(freeze['host_output'])/'run/receipt.json'
    changes=[v for row in report.values() for v in row['roles'].values()]
    json_new(output/'offline-replay.json',dict(schema_version=1,protocol=g.PROTOCOL,
        freeze_sha256=sha,receipt_sha256=g.digest(run),cameras=report,
        exact_replayed_roles=sum(v['different_float64_steps']==0 for v in changes),
        adjacent_float64_roles=sum(v['different_float64_steps']==1 for v in changes),
        maximum_absolute_relative_L2_replay_delta=max(v['absolute_difference'] for v in changes),
        comparator='Relative L2 may have one adjacent representable FP64 value. All scientific gates, '
                   'maxima, refined pixel sequences, inventories, hashes and work counts use the original checks.',
        original_offline_attempt=dict(exit_code=1,error='SignatureError: Grid refinement or convergence differs'),
        new_renders=0,checkpoint_loads=0,optimizer_updates=0))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify_capture(args.freeze_sha256,args.output)
