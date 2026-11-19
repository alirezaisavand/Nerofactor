"""Cold-process entry for one admitted Bell recovery arm."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_heterogeneous_training_v1 as h
from infrastructure.signatures import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=h.ARMS,required=True)
    for key in ('freeze','oracle','benchmark','admission','verification'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('initial','arm-initial','output','authorization'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--execute',action='store_true')
    for key in ('authorization-sha256','commit','image-id'):
        p.add_argument('--'+key)
    a = p.parse_args()
    f = h.freeze(a.freeze,a.freeze_sha256)
    if not a.execute:
        g,views = h.dataset(f,a.benchmark,a.admission,a.verification)
        geo,_,_ = h.geometry(f,a.oracle)
        domain,_,probes = h.diagnostics(geo,f)
        require(len(probes['face_ids'])==f['boundary_probe_count'] and len(domain['face_pairs'])==h.SPATIAL_POLICY['expected_pairs']
                and not any(n.split('.')[0] in ('torch','network','raytracing') for n in sys.modules),
                'Bell metadata route must stay file-only')
        print(f'Admitted {a.phase} panel {g["splits"]}, {len(h.fit_pool(views)["position"])} fitting observations')
        return
    require(all((a.initial,a.arm_initial,a.output,a.authorization,a.authorization_sha256,a.commit,a.image_id)),
            'Explicit Bell execution identity required')
    a.initial_sha256 = f['base_initial']['sha256']
    a.benchmark_sha256 = h.GENERATION
    from experiments.donut_material_recovery_v1 import train
    train(a,acquisition=h)


if __name__=='__main__':
    main()
