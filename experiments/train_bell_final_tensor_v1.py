"""Cold-process entry for one arm of the approved final Dense Bell pair."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_final_tensor_v1 as final
from experiments.bell_final_tensor_admission_v1 import tensor_admission
from experiments.bell_final_tensor_trainer_v1 import run
from experiments.train_bell_dense_known_l2_v1 import Acquisition
from infrastructure.signatures import require


class FinalAcquisition(Acquisition):
    def __getattr__(self, name):
        return getattr(final,name)

    tensor_admission = staticmethod(tensor_admission)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase',choices=final.ARMS,required=True)
    cli.add_argument('--freeze-sha256',required=True)
    for arg in ('freeze','oracle','benchmark','admission','verification'):
        cli.add_argument('--'+arg,required=True,type=Path)
    for arg in ('initial','arm-initial','output','authorization'):
        cli.add_argument('--'+arg,type=Path)
    for arg in ('authorization-sha256','commit','image-id'):
        cli.add_argument('--'+arg)
    cli.add_argument('--execute',action='store_true')
    a = cli.parse_args()
    f = final.freeze(a.freeze,a.freeze_sha256)
    if not a.execute:
        _,views = final.dataset(f,a.benchmark,a.admission,a.verification)
        require(final.index_hash(len(final.fit_pool(views)['position']))==f['photometric_index_sha256'],
                'Admitted fitting indices differ')
        print(f'{a.phase}: admitted dense observations and prevalidated index stream')
        return
    require(all((a.initial,a.arm_initial,a.output,a.authorization,a.authorization_sha256,a.commit,a.image_id)),
            'Explicit final pair runtime identity required')
    a.initial_sha256 = f['base_initial']['sha256']
    a.benchmark_sha256 = final.GENERATION
    run(a,FinalAcquisition(a.output,f['photometric_index_sha256']),f['trainer_overlay_sha256'])


if __name__=='__main__':
    main()
