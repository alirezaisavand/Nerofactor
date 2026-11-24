"""Apply only seed and fresh initialization admission to each accepted Bell trainer."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_publication_seeds_v1 as spec
from experiments import bell_final_tensor_trainer_v1 as final_trainer
from experiments.bell_final_tensor_admission_v1 import tensor_admission
from experiments.train_bell_dense_known_l2_v1 import Acquisition, RecordedPositions
from infrastructure.signatures import require


def trainer_source(method):
    if method == 'K1-Dense':
        raw = (spec.ROOT/final_trainer.SOURCE).read_bytes()
        require(sha256(raw).hexdigest() == final_trainer.SOURCE_SHA, 'Accepted K1 trainer differs')
        source = raw.decode()
    else:
        source = final_trainer.transformed_source()
    start = '    expected_initial=None\n'
    end = '    if log_tensor:\n        initial_controls='
    require(source.count(start) == source.count(end) == 1, 'Unexpected original initialization block')
    left, original = source.split(start)
    _, right = original.split(end)
    source = left+'    expected_initial=acquisition.base_initial\n'+end+right
    old = ("        if heterogeneous:\n            require(initial==json.loads(args.arm_initial.read_text()),\n"
           "                    'Fresh arm weights or buffers differ from its accepted historical initialization')\n"
           "            report['historical_arm_initial_parity_verified']=True")
    new = ("        acquisition.verify_initial(args.phase, initial)\n"
           "        report['fresh_seed_initialization_verified']=True\n"
           "        report['seed_matched_base_initial_sha256']=args.initial_sha256")
    report = "report.update(indirect_illumination='identically zero', initialization_anchor_sha256=args.initial_sha256)"
    require(source.count(old) == source.count(report) == 1, 'Unexpected original initialization reporting')
    source = source.replace(old, new).replace(report, "report.update(indirect_illumination='identically zero')")
    return source, sha256(source.encode()).hexdigest()


class SeededPositions(RecordedPositions):
    def __init__(self, values, output, seed):
        super().__init__(values, output)
        self.expected = spec.dense.index_batches(len(values), seed, 10000, 512)


class SeedAcquisition(Acquisition):
    def __init__(self, args, f):
        self.f = f
        self.seed, self.method = spec.seed_method(args.phase)
        self.base_initial = None
        super().__init__(args.output, f['photometric_index_sha256'][str(self.seed)])

    def __getattr__(self, name):
        return getattr(spec, name)

    tensor_admission = staticmethod(tensor_admission)

    def fit_pool(self, views):
        require(self.record is None, 'An arm may build its fitting pool only once')
        pool = spec.fit_pool(views)
        self.record = SeededPositions(pool['position'], self.output, self.seed)
        return dict(pool, position=self.record)

    def runtime(self, args):
        f = self.f
        auth = spec.b.read(args.authorization)
        require(args.image_id == spec.IMAGE and spec.b.digest(args.authorization) == args.authorization_sha256
                and auth == spec.authorization(f, args.commit, args.freeze_sha256), 'Exact seed execution approval required')
        require(os.geteuid() != 0 and args.output == Path('/outputs')/args.phase and args.output.resolve() == args.output
                and not args.output.exists() and not os.statvfs('/outputs').f_flag & os.ST_RDONLY,
                'Fresh non-root seed output required')
        inputs = (Path('/'), spec.ROOT, args.freeze, args.oracle, args.authorization, args.benchmark,
                  args.admission, args.verification)
        require(all(os.statvfs(p).f_flag & os.ST_RDONLY for p in inputs), 'Read-only seed inputs required')
        require({ln.split(':')[0].strip() for ln in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
                'Networkless execution required')
        if self.method == 'K1-Dense':
            require(args.initial is None and args.initial_sha256 is None, 'K1 must initialize from scratch')
        else:
            require(args.initial == Path('/inputs/seed-initial.json') and args.initial_sha256 is not None
                    and os.statvfs(args.initial).f_flag & os.ST_RDONLY
                    and spec.b.digest(args.initial) == args.initial_sha256, 'Immutable preceding seed anchor required')
            self.base_initial = spec.b.read(args.initial)
            spec.validate_initial(self.base_initial, f'K1-Dense-{self.seed}')
        method = f['arms'][args.phase]
        plan = dict(method, benchmark_generation_sha256=spec.GENERATION, protocol=spec.PROTOCOL)
        active = dict(f, arm_training=plan)
        if self.method != 'K1-Dense':
            active['tensor_training'] = dict(plan, arms=list(spec.d9.ARMS), initialization_policy=spec.INITIALIZATION,
                                             surface_q_enabled=False)
        return active, auth

    def verify_initial(self, arm, initial):
        spec.validate_initial(initial, arm, self.base_initial)
        require(initial['state_sha256'] != self.f['historical_initial_hashes'][self.method],
                'Fresh seed unexpectedly produced the historical state')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase', choices=spec.ARMS, required=True)
    for name in ('freeze', 'oracle', 'benchmark', 'admission', 'verification', 'output', 'authorization'):
        cli.add_argument('--'+name, type=Path, required=True)
    for name in ('freeze-sha256', 'authorization-sha256', 'commit', 'image-id'):
        cli.add_argument('--'+name, required=True)
    cli.add_argument('--initial', type=Path)
    cli.add_argument('--initial-sha256')
    cli.add_argument('--execute', action='store_true', required=True)
    args = cli.parse_args()
    f = spec.freeze(args.freeze, args.freeze_sha256)
    args.benchmark_sha256 = spec.GENERATION
    acquisition = SeedAcquisition(args, f)
    source, digest = trainer_source(acquisition.method)
    require(digest == f['trainer_overlay_sha256'][acquisition.method], 'Seed trainer source differs')
    namespace = dict(__name__=__name__, __file__=str(spec.ROOT/final_trainer.SOURCE))
    exec(compile(source, f'bell-publication-{acquisition.method}', 'exec'), namespace)
    require(namespace['POLICY'] == spec.POLICY, 'Original policy differs')
    namespace['POLICY'] = spec.policy(args.phase)
    namespace['train'](args, acquisition=acquisition)


if __name__ == '__main__':
    main()
