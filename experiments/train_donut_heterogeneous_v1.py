"""Cold-process entry for one authorized heterogeneous recovery arm."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import donut_heterogeneous_training_v1 as h
from benchmarks.donut_recovery_v1 import inputs
from infrastructure.signatures import require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=h.ARMS,required=True)
    for k in ('freeze','oracle','cameras','benchmark','admission','verification'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('initial','arm-initial','output','authorization'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--execute',action='store_true')
    for k in ('authorization-sha256','commit','image-id'):
        p.add_argument('--'+k)
    a=p.parse_args()
    f=h.freeze(a.freeze,a.freeze_sha256)
    if not a.execute:
        g,_=h.dataset(f,a.benchmark,a.admission,a.verification)
        inputs(a.oracle,a.cameras)
        require(not any(n.split('.')[0] in ('torch','network','raytracing') for n in sys.modules),
                'Metadata branch must stay file-only')
        print(f'Admitted dataset verified for {a.phase}: {g["splits"]}; no scientific execution')
        return
    require(all((a.initial,a.arm_initial,a.output,a.authorization,a.authorization_sha256,a.commit,a.image_id)),
            'Explicit heterogeneous execution identity required')
    a.initial_sha256=f['base_initial']['sha256']
    a.benchmark_sha256=h.GENERATION
    from experiments.donut_material_recovery_v1 import train
    train(a,heterogeneous=True)


if __name__=='__main__':
    main()
