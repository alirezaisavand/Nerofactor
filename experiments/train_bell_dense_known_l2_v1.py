"""Run the unchanged native trainer on the admitted ordered dense Bell pool."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_dense_known_l2_v1 as dense
from infrastructure.signatures import require


class RecordedPositions:
    """Observe the trainer's actual indices and return the ordinary NumPy selection."""
    def __init__(self, values, output):
        self.values = values
        self.path = output/dense.STREAM
        with self.path.open('xb'):
            pass
        self.expected = dense.index_batches(len(values),dense.known.POLICY['seed'],10000,512)
        self.steps = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):
        expected = next(self.expected,None)
        require(expected is not None and np.array_equal(indices,expected),'Native fitting indices differ from frozen shuffle')
        with self.path.open('ab') as stream:
            stream.write(indices.astype(dense.INDEX_DTYPE,copy=False).tobytes())
        self.steps += 1
        return self.values[indices]

    def verify(self, expected_sha):
        require(self.steps==10000 and self.path.stat().st_size==10000*512*dense.INDEX_DTYPE.itemsize
                and dense.b.digest(self.path)==expected_sha,'Incomplete or different native photometric stream')


class Acquisition:
    def __init__(self, output, expected_sha):
        self.output, self.expected_sha = output, expected_sha
        self.record = None

    def __getattr__(self, name):
        return getattr(dense,name)

    def dataset(self, *args):
        result = dense.dataset(*args)
        if self.record is not None:
            self.record.verify(self.expected_sha)
        return result

    def fit_pool(self, views):
        require(self.record is None,'An arm may build its ordered fitting pool only once')
        pool = dense.fit_pool(views)
        self.record = RecordedPositions(pool['position'],self.output)
        return dict(pool,position=self.record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=dense.ARMS,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    for arg in ('freeze','oracle','benchmark','admission','verification'):
        parser.add_argument('--'+arg,required=True,type=Path)
    for arg in ('initial','arm-initial','output','authorization'):
        parser.add_argument('--'+arg,type=Path)
    for arg in ('authorization-sha256','commit','image-id'):
        parser.add_argument('--'+arg)
    parser.add_argument('--execute',action='store_true')
    a = parser.parse_args()
    f = dense.freeze(a.freeze,a.freeze_sha256)
    if not a.execute:
        _,views = dense.dataset(f,a.benchmark,a.admission,a.verification)
        print(f'{a.phase}: {len(dense.fit_pool(views)["position"])} admitted fitting observations')
        return
    require(all((a.initial,a.arm_initial,a.output,a.authorization,a.authorization_sha256,a.commit,a.image_id)),
            'Explicit dense training runtime identity required')
    a.initial_sha256 = f['base_initial']['sha256']
    a.benchmark_sha256 = dense.GENERATION
    from experiments.donut_material_recovery_v1 import train
    train(a,acquisition=Acquisition(a.output,f['photometric_index_sha256']))


if __name__=='__main__':
    main()
