"""Run only the frozen Final-Chart-Oracle history using the unchanged Gate 4 trainer."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import donut_representation_gate5_v1 as g
from experiments import train_donut_matched_geometry_v1 as gate4
from infrastructure.signatures import require


class Acquisition(gate4.Acquisition):
    freeze = staticmethod(g.freeze)

    def __init__(self, frozen, paths, args):
        self.frozen = frozen
        parent = g.parent()
        # Admit the original inputs and support with the unchanged Gate 4 code.
        super().__init__(dict(parent,arms=frozen['arms']),paths,args)

    def runtime(self, args):
        auth = g.read(args.authorization)
        require(auth==g.authorization(self.frozen,args.commit,args.freeze_sha256)
                and g.digest(args.authorization)==args.authorization_sha256, 'Exact Gate 5 approval required')
        require(args.image_id==g.m.IMAGE and args.benchmark_sha256==g.m.GENERATION
                and args.initial_sha256==self.f['base_initial']['sha256'], 'Acquisition identity differs')
        recipe = dict(self.recipe,initial_sha256=args.initial_sha256,
                      benchmark_generation_sha256=g.m.GENERATION,protocol=g.PROTOCOL)
        return dict(self.f,arm_training=recipe,tensor_training=dict(recipe,arms=list(g.m.bell.d9.ARMS),
                    initialization_policy=g.m.bell.INITIALIZATION,surface_q_enabled=False)),auth

    def tensor_admission(self, shader, sibling, points, normals, outgoing, output, budget, torch):
        report = super().tensor_admission(shader,sibling,points,normals,outgoing,output,budget,torch)
        report['gate4_ambient_replay'] = g.replay_captures(self.frozen,self.p,output)
        return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ('freeze-sha256','commit','authorization-sha256'):
        cli.add_argument('--'+flag,required=True)
    cli.add_argument('--execute',action='store_true',required=True)
    args = cli.parse_args()
    args.freeze = g.ROOT/g.FREEZE
    frozen = g.freeze(args.freeze,args.freeze_sha256)
    paths = g.paths(frozen,native=True)
    args.phase = g.ARM
    args.output = Path('/outputs')/g.ARM
    args.authorization = Path('/authorization.json')
    require(os.geteuid()!=0 and args.output.resolve()==args.output and not args.output.exists()
            and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
            and all(os.statvfs(path).f_flag&os.ST_RDONLY for path in (Path('/'),g.ROOT,args.authorization,*paths.values()))
            and {ln.split(':')[0].strip() for ln in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Fresh isolated non-root Gate 5 runtime required')
    g.check_inputs(frozen,paths)
    args.image_id = g.m.IMAGE
    args.initial,args.oracle,args.benchmark = (paths[k] for k in ('initial','oracle','reference'))
    args.admission,args.verification = paths['admission'],paths['verification']
    args.initial_sha256 = g.parent()['base_initial']['sha256']
    args.benchmark_sha256 = g.m.GENERATION
    args.arm_initial = paths['chart_initial']
    code = gate4.trainer_source(True)
    require(sha256(code.encode()).hexdigest()==frozen['arms'][g.ARM]['trainer_sha256'],
            'Gate 4 trainer overlay differs')
    acquisition = Acquisition(frozen,paths,args)
    namespace = dict(__name__='donut_gate5_native',__file__=str(g.ROOT/gate4.native.SOURCE))
    exec(compile(code,'donut-representation-gate5-v1','exec'),namespace)
    namespace['train'](args,acquisition=acquisition)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,gate4.interrupted)
    main()
