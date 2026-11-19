"""Cold-process dispatch for one authorized robust tensor-Q arm."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import robust_tensor_q_training_v1 as h
from infrastructure.signatures import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=h.ARMS,required=True)
    for key in ('freeze','oracle','benchmark','admission','verification'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('cameras','initial','arm-initial','output','authorization'):
        p.add_argument('--'+key,type=Path)
    for key in ('freeze-sha256','authorization-sha256','commit','image-id'):
        p.add_argument('--'+key,required=key=='freeze-sha256')
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    spec = h.Acquisition(a.phase)
    f = spec.freeze(a.freeze,a.freeze_sha256)
    require(spec.FACETED or a.cameras is not None,'Explicit Donut calibrations required')
    if not a.execute:
        spec.host_inputs(f)
        g,views = spec.dataset(f,a.benchmark,a.admission,a.verification)
        require(not any(n.split('.')[0] in ('torch','network','raytracing') for n in sys.modules),
                'Robust metadata route must remain file-only')
        print(f'{a.phase} admitted panel {g["splits"]}, {len(spec.fit_pool(views)["position"])} fitting observations')
        return
    require(all((a.initial,a.arm_initial,a.output,a.authorization,a.authorization_sha256,a.commit,a.image_id)),
            'Explicit robust execution identity required')
    a.initial_sha256 = f['base_initial']['sha256']
    a.benchmark_sha256 = spec.GENERATION
    from experiments.donut_material_recovery_v1 import train
    train(a,acquisition=spec)


if __name__=='__main__':
    main()
