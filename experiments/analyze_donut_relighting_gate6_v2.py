"""Run the committed Gate 6 analysis with replay evidence checked per field and pixel."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks import donut_relighting_gate6_v2 as g


V1='experiments/analyze_donut_relighting_gate6_v1.py'


def verifier_source():
    return g.replay_source(V1,(
        ('from benchmarks import donut_relighting_gate6_v1 as g','from benchmarks import donut_relighting_gate6_v2 as g'),
        ("            require(restore['strict'] and restore['state_unchanged'] and not restore['optimizer_restored']",
         "            replay=g.field_parity(arm,v,d,old)\n"
         "            replay['Q_axis_reconstruction_max_abs']=restore['field_parity'][str(camera)]['Q_axis_reconstruction_max_abs']\n"
         "            require(replay==restore['field_parity'][str(camera)]==r['field_admission'][arm][str(camera)]\n"
         "                    and replay['passed'] and 0<=replay['Q_axis_reconstruction_max_abs']<=g.MATERIAL_ATOL,\n"
         "                    'Historical batch replay or diagnostic receipt differs')\n"
         "            require(restore['strict'] and restore['state_unchanged'] and not restore['optimizer_restored']"),
    ))


if __name__=='__main__':
    exec(compile(verifier_source(),str(g.ROOT/V1),'exec'),dict(__name__='__main__',__file__=__file__))
